"""Verify and promote one hash-bound release into the runtime package."""

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


APPLICATION_ROOT = Path(__file__).resolve().parents[1]
SELECTION_PATH = APPLICATION_ROOT / "eval" / "results" / "selected-release.json"
LEADERBOARD_PATH = APPLICATION_ROOT / "eval" / "leaderboard.json"
RUNTIME_ROOT = APPLICATION_ROOT / "src"
DATA_SCRIPTS = APPLICATION_ROOT / "data" / "scripts"
sys.path.insert(0, str(DATA_SCRIPTS))

from _artifacts import (
    ArtifactError,
    artifact_bytes,
    inside,
    resolve_path,
    sha256_path,
    verify_reference,
    verify_references,
)
from _contract import (
    ContractError,
    load_contract,
    require_approved,
    validate_schema,
)
from _dataset import verify_dataset_manifest
from _evaluation import verify_evaluation_manifest
from _training import verify_training_manifest


class PromotionError(ValueError):
    """Raised when release-selection evidence is incomplete."""


def _load_json(path: Path, label: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PromotionError(f"Create {label}: {path}.") from exc
    except json.JSONDecodeError as exc:
        raise PromotionError(
            f"Write valid JSON to {path}: line {exc.lineno}, column {exc.colno}."
        ) from exc
    if not isinstance(payload, dict):
        raise PromotionError(f"Set {label} to a JSON object.")
    return payload


def _load_selection() -> Dict[str, Any]:
    payload = _load_json(SELECTION_PATH, "eval/results/selected-release.json")
    required = {
        "schema_version",
        "status",
        "package",
        "evaluation_manifest",
        "release_manifest",
        "artifacts",
    }
    if set(payload) != required:
        raise PromotionError(
            "Set selected-release fields to: " + ", ".join(sorted(required)) + "."
        )
    if payload["schema_version"] != "miniagent.release-selection.v1":
        raise PromotionError(
            "Set selected-release schema_version to miniagent.release-selection.v1."
        )
    if payload["status"] != "selected":
        raise PromotionError("Set selected-release status to selected.")
    package = payload["package"]
    if not isinstance(package, str) or re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", package
    ) is None:
        raise PromotionError("Set selected-release package to the runtime package name.")
    if not isinstance(payload["artifacts"], list) or not payload["artifacts"]:
        raise PromotionError("Add every runtime bundle artifact to the selection.")
    return payload


def _record_key(record: Mapping[str, Any]) -> Tuple[str, str]:
    return str(record["path"]), str(record["sha256"])


def _release_hashes(release: Dict[str, Any]) -> set[str]:
    hashes = {
        release["evaluation"]["manifest"]["sha256"],
        release["provenance"]["application"]["sha256"],
        release["provenance"]["dataset"]["sha256"],
        release["provenance"]["training"]["sha256"],
        release["policy"]["action_parser"]["artifact"]["sha256"],
        release["policy"]["runtime"]["artifact"]["sha256"],
    }
    hashes.update(item["artifact"]["sha256"] for item in release["policy"]["weights"])
    hashes.update(item["artifact"]["sha256"] for item in release["artifacts"])
    hashes.update(item["sha256"] for item in release["provenance"]["workflows"])
    hashes.update(item["sha256"] for item in release["provenance"]["tools"])
    hashes.update(item["evidence"]["sha256"] for item in release["verification"])
    return hashes


def _allowed_bundle_hashes(release: Dict[str, Any]) -> Dict[str, set[str]]:
    provenance = release["provenance"]
    artifacts: Dict[str, set[str]] = {}
    for item in release["artifacts"]:
        artifacts.setdefault(item["role"], set()).add(item["artifact"]["sha256"])
    return {
        "application_contract": {
            provenance["application"]["sha256"],
            *artifacts.get("application_contract", set()),
        },
        "policy": {
            item["artifact"]["sha256"] for item in release["policy"]["weights"]
        },
        "workflow": {
            *(item["sha256"] for item in provenance["workflows"]),
            *artifacts.get("workflow", set()),
        },
        "tool_contract": {
            *(item["sha256"] for item in provenance["tools"]),
            *artifacts.get("tool_contract", set()),
        },
        "knowledge_base": artifacts.get("knowledge_base", set()),
        "configuration": artifacts.get("configuration", set()),
        "runtime_asset": {
            release["policy"]["action_parser"]["artifact"]["sha256"],
            release["policy"]["runtime"]["artifact"]["sha256"],
            *artifacts.get("runtime_asset", set()),
        },
        "runtime_code": set(),
        "runtime_contract": set(),
        "dependency": set(),
        "test": set(),
    }


def _verify_release_artifacts(release: Dict[str, Any]) -> None:
    verify_references(
        [item["artifact"] for item in release["policy"]["weights"]],
        "release policy weights",
    )
    verify_reference(release["policy"]["action_parser"]["artifact"], "release action parser")
    verify_reference(
        release["policy"]["runtime"]["artifact"], "release policy runtime adapter"
    )
    weight_roles = [item["role"] for item in release["policy"]["weights"]]
    if len(weight_roles) != len(set(weight_roles)):
        raise PromotionError("Assign each policy weight role once.")
    if "merged" not in weight_roles and not {"base", "adapter"}.issubset(weight_roles):
        raise PromotionError("Release one merged policy or a base policy with its adapter.")
    verify_references(
        [item["artifact"] for item in release["artifacts"]], "release artifacts"
    )
    artifact_keys = [
        (item["role"], *_record_key(item["artifact"]))
        for item in release["artifacts"]
    ]
    if len(artifact_keys) != len(set(artifact_keys)):
        raise PromotionError("List each released artifact once.")
    verify_reference(release["evaluation"]["manifest"], "release evaluation")
    provenance = release["provenance"]
    verify_reference(provenance["application"], "release application contract")
    verify_references(provenance["workflows"], "release workflows")
    verify_references(provenance["tools"], "release tools")
    for name in ("workflows", "tools"):
        keys = [_record_key(item) for item in provenance[name]]
        if len(keys) != len(set(keys)):
            raise PromotionError(f"List each release provenance {name} artifact once.")
    verify_reference(provenance["dataset"], "release dataset")
    verify_reference(provenance["training"], "release training run")
    verify_references(
        [item["evidence"] for item in release["verification"]],
        "release verification evidence",
    )


def _preflight_artifacts(
    items: Iterable[Any], source_root: Path, release: Dict[str, Any]
) -> List[Tuple[Path, Path, str]]:
    verified = []
    destinations = set()
    release_hashes = _release_hashes(release)
    allowed_by_role = _allowed_bundle_hashes(release)
    bundle = {item["path"]: item for item in release["runtime"]["bundle"]}
    if len(bundle) != len(release["runtime"]["bundle"]):
        raise PromotionError("Assign each runtime bundle path once.")
    for position, value in enumerate(items):
        if not isinstance(value, dict) or set(value) != {"source", "destination", "sha256"}:
            raise PromotionError(
                f"Set artifacts[{position}] to source, destination, and sha256."
            )
        source, observed = verify_reference(
            {"path": value["source"], "sha256": value["sha256"]},
            f"artifacts[{position}]",
        )
        destination_value = value["destination"]
        if not isinstance(destination_value, str) or not destination_value:
            raise PromotionError(f"Set artifacts[{position}].destination to a path.")
        destination = source_root / destination_value
        if not inside(destination, source_root) or destination == source_root:
            raise PromotionError(
                f"Keep artifacts[{position}].destination inside src/."
            )
        resolved_destination = destination.resolve()
        if resolved_destination in destinations:
            raise PromotionError(f"Assign each destination once: {destination_value}.")
        destinations.add(resolved_destination)
        bundle_record = bundle.get(destination_value)
        if bundle_record is None:
            raise PromotionError(f"Bind artifacts[{position}].destination into runtime.bundle.")
        if bundle_record["sha256"] != observed:
            raise PromotionError(f"Match artifacts[{position}] and runtime.bundle hashes.")
        if bundle_record["bytes"] != artifact_bytes(source):
            raise PromotionError(
                f"Match artifacts[{position}] and runtime.bundle byte counts."
            )
        if observed not in release_hashes:
            raise PromotionError(f"Bind artifacts[{position}].sha256 into the release manifest.")
        if observed not in allowed_by_role[bundle_record["role"]]:
            raise PromotionError(
                f"Match artifacts[{position}] to its runtime.bundle role."
            )
        verified.append((source, destination, observed))
    selected_paths = {
        destination.relative_to(source_root).as_posix()
        for _, destination, _ in verified
    }
    if not selected_paths.issubset(bundle):
        raise PromotionError("Bind every selected artifact into runtime.bundle.")
    return verified


def _replace_artifact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as handle:
            staged = Path(handle.name)
        try:
            shutil.copy2(source, staged)
            staged.replace(destination)
        finally:
            staged.unlink(missing_ok=True)
        return
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=f".{destination.name}."
    ) as temporary:
        staged = Path(temporary) / "artifact"
        shutil.copytree(source, staged)
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        staged.replace(destination)


def _verify_runtime_inventory(
    release: Dict[str, Any], source_root: Path, package: str
) -> None:
    bundle = release["runtime"]["bundle"]
    by_path = {item["path"]: item for item in bundle}
    if len(by_path) != len(bundle):
        raise PromotionError("Assign each runtime bundle path once.")
    required_paths = {
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        f"{package}/agent.py",
        f"{package}/api.py",
        f"{package}/cli.py",
        f"{package}/release_verifier.py",
        f"{package}/business_logic/application.json",
    }
    if not required_paths.issubset(by_path):
        raise PromotionError("Inventory the runtime code, lock, and application contract.")
    ignored_parts = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
    }
    symlinks = [
        path
        for path in source_root.rglob("*")
        if path.is_symlink()
        and not any(part in ignored_parts for part in path.relative_to(source_root).parts)
    ]
    if symlinks:
        raise PromotionError("Build the runtime source tree from regular files.")
    manifest_relative = f"{package}/deployment_manifest.json"
    observed = {
        path.relative_to(source_root).as_posix(): path
        for path in source_root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.name != ".env"
        and path.relative_to(source_root).as_posix() != manifest_relative
        and not any(part in ignored_parts for part in path.relative_to(source_root).parts)
    }
    if set(by_path) != set(observed):
        missing = sorted(set(observed).difference(by_path))
        extra = sorted(set(by_path).difference(observed))
        raise PromotionError(
            f"Runtime bundle must inventory the complete src tree; missing={missing}, extra={extra}."
        )
    for relative, path in observed.items():
        record = by_path[relative]
        if sha256_path(path) != record["sha256"]:
            raise PromotionError(f"Verify the runtime source hash for {relative}.")
        if artifact_bytes(path) != record["bytes"]:
            raise PromotionError(f"Verify the runtime source byte count for {relative}.")


def _under(record: Dict[str, Any], root: Path, label: str) -> Path:
    path, _ = verify_reference(record, label)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PromotionError(f"Keep {label} under {root.relative_to(APPLICATION_ROOT)}.") from exc
    return path


def _validate_dataset_lineage(
    release: Dict[str, Any], evaluation: Dict[str, Any], contract: Dict[str, Any]
) -> Dict[str, Any]:
    dataset_root = resolve_path(
        contract["artifacts"]["dataset_manifests"], "artifacts.dataset_manifests"
    )
    dataset_path = _under(release["provenance"]["dataset"], dataset_root, "release dataset")
    if dataset_path.parent != dataset_root:
        raise PromotionError("Select a direct immutable dataset manifest.")
    dataset = verify_dataset_manifest(dataset_path, contract)
    splits = [item for item in dataset["splits"] if item["purpose"] == "evaluation"]
    if len(splits) != 1:
        raise PromotionError("Bind one sealed evaluation split into the dataset manifest.")
    split = splits[0]
    evaluated = evaluation["dataset"]
    if (
        evaluated["dataset_id"] != dataset["dataset_id"]
        or _record_key(evaluated) != _record_key(split)
        or evaluated["cases"] != split["records"]
        or evaluated["sealed"] is not True
    ):
        raise PromotionError("Bind the evaluated sealed split to release dataset provenance.")
    return dataset


def _validate_contract_lineage(
    release: Dict[str, Any], dataset: Dict[str, Any]
) -> None:
    by_role: Dict[str, list[Dict[str, Any]]] = {}
    for item in dataset["source_contracts"]:
        by_role.setdefault(item["role"], []).append(item)
        source = resolve_path(item["source_path"], "dataset source contract")
        snapshot, _ = verify_reference(item["artifact"], "dataset contract snapshot")
        if sha256_path(source) != sha256_path(snapshot):
            raise PromotionError("Match current approved contracts to dataset snapshots.")
    provenance = release["provenance"]
    if len(by_role.get("application", [])) != 1 or provenance["application"] != by_role["application"][0]["artifact"]:
        raise PromotionError("Bind the dataset application snapshot into release provenance.")
    expected_workflows = {_record_key(item["artifact"]) for item in by_role.get("workflow", [])}
    expected_tools = {_record_key(item["artifact"]) for item in by_role.get("tool", [])}
    if {_record_key(item) for item in provenance["workflows"]} != expected_workflows:
        raise PromotionError("Bind every dataset workflow snapshot into release provenance.")
    if {_record_key(item) for item in provenance["tools"]} != expected_tools:
        raise PromotionError("Bind every dataset tool snapshot into release provenance.")


def _validate_training_lineage(
    release: Dict[str, Any], dataset_record: Dict[str, Any], contract: Dict[str, Any]
) -> Dict[str, Any]:
    training_root = resolve_path(
        contract["artifacts"]["training_runs"], "artifacts.training_runs"
    )
    training_path = _under(
        release["provenance"]["training"], training_root, "release training run"
    )
    if training_path.name != "training-manifest.json" or training_path.parent.parent != training_root:
        raise PromotionError("Select a direct immutable training manifest.")
    training, _ = verify_training_manifest(training_path, contract)
    if _record_key(training["dataset"]) != _record_key(dataset_record):
        raise PromotionError("Bind the released training run to the released dataset.")
    return training


def _verify_leaderboard_entry(release: Dict[str, Any]) -> None:
    leaderboard = _load_json(LEADERBOARD_PATH, "eval/leaderboard.json")
    if leaderboard.get("schema_version") != "miniagent.leaderboard.v1" or not isinstance(
        leaderboard.get("entries"), list
    ):
        raise PromotionError("Use the MiniAgent leaderboard contract.")
    selected = release["evaluation"]["leaderboard_entry"]
    matches = [
        item
        for item in leaderboard["entries"]
        if item.get("evaluation_id") == selected["evaluation_id"]
    ]
    if matches != [selected]:
        raise PromotionError("Bind the exact indexed leaderboard entry into the release.")
    if selected["manifest"] != release["evaluation"]["manifest"]:
        raise PromotionError("Bind the leaderboard entry to the released evaluation manifest.")


def main() -> int:
    try:
        selection = _load_selection()
        contract = load_contract()
        require_approved(contract)
        if selection["package"] != contract["package"]:
            raise PromotionError("Match the selected package to the application contract.")

        evaluation_root = resolve_path(
            contract["artifacts"]["evaluation_results"], "artifacts.evaluation_results"
        )
        evaluation_path = _under(
            selection["evaluation_manifest"], evaluation_root, "evaluation_manifest"
        )
        if evaluation_path.name != "evaluation-manifest.json" or evaluation_path.parent.parent != evaluation_root:
            raise PromotionError("Select a direct immutable evaluation manifest.")
        evaluation_sha = selection["evaluation_manifest"]["sha256"]
        evaluation = _load_json(evaluation_path, "evaluation manifest")
        validate_schema(evaluation, "evaluation-manifest.schema.json", "Evaluation manifest")
        if evaluation["summary"]["status"] != "passed":
            raise PromotionError("Select an evaluation with passed status.")

        release_root = resolve_path(contract["artifacts"]["releases"], "artifacts.releases")
        release_path = _under(selection["release_manifest"], release_root, "release_manifest")
        if release_path.name != "release-manifest.json" or release_path.parent.parent != release_root:
            raise PromotionError("Select a direct immutable release manifest.")
        release_sha = selection["release_manifest"]["sha256"]
        release = _load_json(release_path, "release manifest")
        validate_schema(release, "release-manifest.schema.json", "Release manifest")
        if release["release_id"] != release_path.parent.name:
            raise PromotionError("Match release_id to its immutable release directory.")
        _verify_release_artifacts(release)

        package = selection["package"]
        if release["runtime"]["package"] != package:
            raise PromotionError("Match the selected package to release.runtime.package.")
        if release["application_id"] != contract["application_id"] or evaluation["application_id"] != contract["application_id"]:
            raise PromotionError("Match release and evaluation application identities.")
        if release["evaluation"]["evaluation_id"] != evaluation["evaluation_id"]:
            raise PromotionError("Match the release and evaluation identities.")
        if (
            release["evaluation"]["manifest"]["sha256"] != evaluation_sha
            or release["evaluation"]["manifest"]["path"]
            != selection["evaluation_manifest"]["path"]
        ):
            raise PromotionError("Bind the verified evaluation into the release manifest.")
        _verify_leaderboard_entry(release)

        dataset = _validate_dataset_lineage(release, evaluation, contract)
        evaluation = verify_evaluation_manifest(evaluation_path, contract, dataset)
        if evaluation["summary"]["status"] != "passed":
            raise PromotionError("Select an evaluation with passed status.")
        _validate_contract_lineage(release, dataset)
        training = _validate_training_lineage(
            release, release["provenance"]["dataset"], contract
        )
        evaluated_policy = {_record_key(item) for item in evaluation["candidate"]["artifacts"]}
        released_policy = {
            _record_key(item["artifact"]) for item in release["policy"]["weights"]
        }
        released_policy.add(_record_key(release["policy"]["action_parser"]["artifact"]))
        released_policy.add(_record_key(release["policy"]["runtime"]["artifact"]))
        if evaluated_policy != released_policy:
            raise PromotionError("Promote the exact policy artifacts used by evaluation.")
        if evaluation["candidate"]["candidate_id"] != release["policy"]["model_id"]:
            raise PromotionError("Match the evaluated and released model identities.")
        if evaluation["candidate"]["version"] != release["policy"]["model_revision"]:
            raise PromotionError("Match the evaluated and released model revisions.")
        if evaluation["protocol"]["action_parser"] != release["policy"]["action_parser"]["name"]:
            raise PromotionError("Match the evaluated and released action parsers.")
        if evaluation["candidate"]["runtime_entrypoint"] != release["policy"]["runtime"]["entrypoint"]:
            raise PromotionError("Match the evaluated and released policy runtime entrypoints.")
        base_hashes: Dict[str, set[str]] = {}
        for item in training["base_model"]["artifacts"]:
            base_hashes.setdefault(item["role"], set()).add(
                item["artifact"]["sha256"]
            )
        output_hashes: Dict[str, set[str]] = {}
        for item in training["outputs"]:
            output_hashes.setdefault(item["role"], set()).add(
                item["artifact"]["sha256"]
            )
        for weight in release["policy"]["weights"]:
            role = weight["role"]
            allowed = (
                base_hashes.get(role, set()) | output_hashes.get(role, set())
                if role in {"tokenizer", "chat_template"}
                else base_hashes.get("base", set())
                if role == "base"
                else output_hashes.get(role, set())
            )
            if weight["artifact"]["sha256"] not in allowed:
                raise PromotionError(
                    "Bind every released policy weight to its matching training role."
                )
        if release["policy"]["action_parser"]["artifact"]["sha256"] not in output_hashes.get(
            "action_parser", set()
        ):
            raise PromotionError("Bind the action parser to its training output role.")
        if release["policy"]["runtime"]["artifact"]["sha256"] not in output_hashes.get(
            "runtime_adapter", set()
        ):
            raise PromotionError("Bind the policy runtime adapter to its training output role.")

        provenance = release["provenance"]
        role_hashes: Dict[str, set[str]] = {}
        for item in release["artifacts"]:
            role_hashes.setdefault(item["role"], set()).add(item["artifact"]["sha256"])
        if role_hashes.get("application_contract") != {provenance["application"]["sha256"]}:
            raise PromotionError("Include the application contract as a release artifact.")
        if role_hashes.get("workflow") != {item["sha256"] for item in provenance["workflows"]}:
            raise PromotionError("Include every workflow as a release artifact.")
        if role_hashes.get("tool_contract") != {item["sha256"] for item in provenance["tools"]}:
            raise PromotionError("Include every tool contract as a release artifact.")

        package_root = RUNTIME_ROOT / package
        if contract["artifacts"]["runtime_source"] != f"src/{package}":
            raise PromotionError("Match artifacts.runtime_source to the selected package.")
        if not package_root.is_dir():
            raise PromotionError(f"Create the runtime package: {package_root}.")
        artifacts = _preflight_artifacts(selection["artifacts"], RUNTIME_ROOT, release)
        contract_hash = provenance["application"]["sha256"]
        if not any(
            observed == contract_hash
            and destination.relative_to(RUNTIME_ROOT).as_posix()
            == f"{package}/business_logic/application.json"
            for _, destination, observed in artifacts
        ):
            raise PromotionError(
                "Promote the approved application contract to business_logic/application.json."
            )

        for source, destination, _ in artifacts:
            _replace_artifact(source, destination)
        _verify_runtime_inventory(release, RUNTIME_ROOT, package)
        _replace_artifact(release_path, package_root / "deployment_manifest.json")
        if sha256_path(package_root / "deployment_manifest.json") != release_sha:
            raise PromotionError("Verify the promoted deployment manifest hash.")
    except (OSError, ArtifactError, ContractError, PromotionError) as exc:
        print(f"Promotion action: {exc}")
        return 2

    print(f"Promoted verified release {release['release_id']} into {package_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
