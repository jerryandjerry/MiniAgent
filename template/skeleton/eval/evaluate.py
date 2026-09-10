"""Launch, verify, and immediately index one immutable evaluation."""

import hashlib
import json
from pathlib import Path
import re
import shutil
import shlex
import subprocess
import sys
import tempfile
from typing import Any, Dict, Tuple


APPLICATION_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = APPLICATION_ROOT / "eval" / "config.json"
LEADERBOARD_PATH = APPLICATION_ROOT / "eval" / "leaderboard.json"
DATA_SCRIPTS = APPLICATION_ROOT / "data" / "scripts"
sys.path.insert(0, str(DATA_SCRIPTS))

from _artifacts import (
    ArtifactError,
    artifact_bytes,
    resolve_path,
    sha256_path,
    verify_reference,
)
from _contract import (
    ContractError,
    load_contract,
    require_approved,
)
from _dataset import verify_dataset_manifest
from _evaluation import verify_evaluation_manifest


IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:[._-][A-Za-z0-9]+)*$")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _load_config() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version",
        "status",
        "command",
        "dataset_manifest",
        "sealed_dataset",
        "evaluator_configuration",
        "output_manifest",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError(
            "Set eval/config.json fields to: "
            + ", ".join(sorted(expected_fields))
            + "."
        )
    if payload["schema_version"] != "miniagent.evaluation-run.v1":
        raise ValueError(
            "Set eval config schema_version to miniagent.evaluation-run.v1."
        )
    if payload["status"] != "approved":
        raise ValueError(
            "Select the sealed dataset, evaluator, and immutable evaluation ID, then "
            "approve eval/config.json."
        )
    command = payload["command"]
    if not isinstance(command, list) or not command or not all(
        isinstance(part, str) and part for part in command
    ):
        raise ValueError("Set eval/config.json command to a non-empty argument list.")

    contract = load_contract()
    require_approved(contract)
    dataset_root = resolve_path(
        contract["artifacts"]["dataset_manifests"], "artifacts.dataset_manifests"
    )
    evaluation_root = resolve_path(
        contract["artifacts"]["evaluation_results"], "artifacts.evaluation_results"
    )
    dataset_manifest_path = resolve_path(
        payload["dataset_manifest"], "eval.dataset_manifest"
    )
    if dataset_manifest_path.parent != dataset_root or not dataset_manifest_path.is_file():
        raise ValueError("Select an immutable manifest from artifacts.dataset_manifests.")
    output_path = resolve_path(payload["output_manifest"], "eval.output_manifest")
    evaluation_directory = output_path.parent
    if (
        evaluation_directory.parent != evaluation_root
        or output_path.name != "evaluation-manifest.json"
        or IDENTIFIER.fullmatch(evaluation_directory.name) is None
    ):
        raise ValueError(
            "Write output_manifest as artifacts.evaluation_results/<evaluation_id>/"
            "evaluation-manifest.json."
        )
    if evaluation_directory.exists():
        raise ValueError(
            f"Select a new evaluation_id; immutable evaluation {evaluation_directory.name} exists."
        )
    evaluator_configuration = resolve_path(
        payload["evaluator_configuration"], "eval.evaluator_configuration"
    )
    configuration_root = APPLICATION_ROOT / "eval" / "configurations"
    if not _inside(evaluator_configuration, configuration_root) or not evaluator_configuration.exists():
        raise ValueError("Create the evaluator configuration under eval/configurations/.")
    sealed_path = resolve_path(payload["sealed_dataset"], "eval.sealed_dataset")
    if not sealed_path.is_file():
        raise ValueError("Create the configured sealed evaluation split.")

    dataset_manifest = verify_dataset_manifest(dataset_manifest_path, contract)
    evaluation_splits = [
        item for item in dataset_manifest["splits"] if item["purpose"] == "evaluation"
    ]
    if len(evaluation_splits) != 1:
        raise ValueError("Declare one evaluation-purpose split in the dataset manifest.")
    evaluation_split = evaluation_splits[0]
    if evaluation_split["sealed"] is not True:
        raise ValueError("Seal the evaluation-purpose split.")
    split_path, _ = verify_reference(evaluation_split, "sealed evaluation split")
    if split_path.resolve() != sealed_path.resolve():
        raise ValueError("Match sealed_dataset to the dataset manifest evaluation split.")
    return payload, contract, dataset_manifest, evaluation_split


def _copy_snapshot(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def _launcher_payload(config: Dict[str, Any], argv: list[str]) -> Dict[str, Any]:
    return {
        "schema_version": "miniagent.effective-invocation.v1",
        "configuration": config,
        "argv": argv,
    }


def _prepare_evaluation(
    config: Dict[str, Any], argv: list[str]
) -> Tuple[Path, Path, Path]:
    output_path = resolve_path(config["output_manifest"], "eval.output_manifest")
    evaluation_directory = output_path.parent
    evaluation_directory.mkdir(parents=True, exist_ok=False)
    launcher_snapshot = evaluation_directory / "provenance" / "launcher-config.json"
    evaluator_snapshot = evaluation_directory / "provenance" / "evaluator-configuration"
    launcher_snapshot.parent.mkdir(parents=True, exist_ok=True)
    launcher_snapshot.write_text(
        json.dumps(_launcher_payload(config, argv), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    _copy_snapshot(
        resolve_path(config["evaluator_configuration"], "eval.evaluator_configuration"),
        evaluator_snapshot,
    )
    return evaluation_directory, launcher_snapshot, evaluator_snapshot


def _validate_bound_snapshot(record: Dict[str, Any], expected: Path, label: str) -> None:
    observed, _ = verify_reference(record, label)
    if observed.resolve() != expected.resolve():
        raise ContractError(f"Bind {label} to its immutable evaluation snapshot.")


def _validate_evaluation_manifest(
    config: Dict[str, Any],
    contract: Dict[str, Any],
    dataset_manifest: Dict[str, Any],
    evaluation_split: Dict[str, Any],
    evaluation_directory: Path,
    launcher_snapshot: Path,
    evaluator_snapshot: Path,
    argv: list[str],
) -> Dict[str, Any]:
    manifest_path = resolve_path(config["output_manifest"], "eval.output_manifest")
    manifest = verify_evaluation_manifest(
        manifest_path, contract, dataset_manifest, evaluation_split
    )
    expected_dataset = resolve_path(config["sealed_dataset"], "eval.sealed_dataset")
    dataset_path, _ = verify_reference(manifest["dataset"], "evaluation dataset")
    if dataset_path.resolve() != expected_dataset.resolve():
        raise ContractError("Bind the configured sealed dataset into the evaluation.")
    evaluator = manifest["evaluator"]
    _validate_bound_snapshot(evaluator["config"], evaluator_snapshot, "evaluator configuration")
    _validate_bound_snapshot(
        evaluator["launcher_configuration"],
        launcher_snapshot,
        "evaluation launcher configuration",
    )
    current_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if current_config != config:
        raise ContractError("Keep eval/config.json unchanged while evaluation executes.")
    if json.loads(launcher_snapshot.read_text(encoding="utf-8")) != _launcher_payload(
        config, argv
    ):
        raise ContractError("Keep the effective evaluation invocation unchanged.")
    if manifest["reproducibility"]["argv"] != argv:
        raise ContractError("Bind the effective evaluation argv into the result manifest.")
    if manifest["reproducibility"]["command"] != shlex.join(argv):
        raise ContractError("Derive the evaluation command from its effective argv.")
    source_evaluator_config = resolve_path(
        config["evaluator_configuration"], "eval.evaluator_configuration"
    )
    if sha256_path(source_evaluator_config) != sha256_path(evaluator_snapshot):
        raise ContractError("Keep evaluator configuration unchanged while evaluation executes.")
    return manifest


def _comparability_key(manifest: Dict[str, Any]) -> str:
    identity = {
        "application_id": manifest["application_id"],
        "dataset": {
            "dataset_id": manifest["dataset"]["dataset_id"],
            "sha256": manifest["dataset"]["sha256"],
        },
        "evaluator": {
            "name": manifest["evaluator"]["name"],
            "version": manifest["evaluator"]["version"],
            "source_sha256": manifest["evaluator"]["source"]["sha256"],
            "config_sha256": manifest["evaluator"]["config"]["sha256"],
        },
        "protocol": manifest["protocol"],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _append_leaderboard(manifest: Dict[str, Any], manifest_path: Path) -> None:
    leaderboard = json.loads(LEADERBOARD_PATH.read_text(encoding="utf-8"))
    if not isinstance(leaderboard, dict) or set(leaderboard) != {"schema_version", "entries"}:
        raise ContractError("Restore eval/leaderboard.json to the leaderboard contract.")
    if leaderboard["schema_version"] != "miniagent.leaderboard.v1" or not isinstance(
        leaderboard["entries"], list
    ):
        raise ContractError("Restore eval/leaderboard.json to miniagent.leaderboard.v1.")
    manifest_record = {
        "path": manifest_path.relative_to(APPLICATION_ROOT).as_posix(),
        "sha256": sha256_path(manifest_path),
        "bytes": artifact_bytes(manifest_path),
    }
    entry = {
        "evaluation_id": manifest["evaluation_id"],
        "created_at": manifest["created_at"],
        "candidate_id": manifest["candidate"]["candidate_id"],
        "candidate_version": manifest["candidate"]["version"],
        "comparability_key": _comparability_key(manifest),
        "status": manifest["summary"]["status"],
        "metrics": manifest["metrics"],
        "manifest": manifest_record,
    }
    for existing in leaderboard["entries"]:
        if existing.get("evaluation_id") == manifest["evaluation_id"]:
            if existing != entry:
                raise ContractError("Keep each leaderboard evaluation_id immutable.")
            return
    leaderboard["entries"].append(entry)
    rendered = json.dumps(leaderboard, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=LEADERBOARD_PATH.parent,
        prefix=".leaderboard.",
        delete=False,
    ) as handle:
        staged = Path(handle.name)
        handle.write(rendered)
    try:
        staged.replace(LEADERBOARD_PATH)
    finally:
        staged.unlink(missing_ok=True)


def main() -> int:
    passthrough = list(sys.argv[1:])
    try:
        config, contract, dataset_manifest, evaluation_split = _load_config()
        dataset_manifest_path = resolve_path(
            config["dataset_manifest"], "eval.dataset_manifest"
        )
        dataset_manifest_sha = sha256_path(dataset_manifest_path)
        command = config["command"] + passthrough
        evaluation_directory, launcher_snapshot, evaluator_snapshot = _prepare_evaluation(
            config, command
        )
    except (ArtifactError, ContractError, json.JSONDecodeError, OSError, ValueError) as exc:
        print(f"Evaluation action: {exc}")
        return 2

    print("Evaluation command:", " ".join(command))
    return_code = subprocess.run(command, cwd=APPLICATION_ROOT, check=False).returncode
    if return_code:
        return return_code

    manifest_path = resolve_path(config["output_manifest"], "eval.output_manifest")
    try:
        verify_dataset_manifest(dataset_manifest_path, contract)
        if sha256_path(dataset_manifest_path) != dataset_manifest_sha:
            raise ContractError(
                "Keep the verified dataset manifest unchanged during evaluation."
            )
        manifest = _validate_evaluation_manifest(
            config,
            contract,
            dataset_manifest,
            evaluation_split,
            evaluation_directory,
            launcher_snapshot,
            evaluator_snapshot,
            command,
        )
        _append_leaderboard(manifest, manifest_path)
    except (ArtifactError, ContractError, json.JSONDecodeError, OSError, ValueError) as exc:
        print(f"Evaluation action: {exc}")
        return 2
    print(f"Validated and indexed evaluation manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
