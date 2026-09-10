"""Build one immutable, provenance-bound application dataset."""

import importlib.util
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Dict, Iterable, Tuple

from jsonschema.exceptions import SchemaError

from _artifacts import ArtifactError, artifact_bytes, resolve_path, sha256_path
from _contract import (
    APPLICATION_ROOT,
    CONTRACT_PATH,
    ContractError,
    declared_json_documents,
    load_contract,
    require_approved,
    validate_schema,
)
from _dataset import validate_dataset_content


PIPELINE_PATH = APPLICATION_ROOT / "data" / "pipeline.py"
CONVERSATION_VALIDATOR_PATH = APPLICATION_ROOT / "data" / "conversation_validator.py"
CONFIG_PATH = APPLICATION_ROOT / "data" / "config.json"
IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:[._-][A-Za-z0-9]+)*$")


def _load_config(contract: Dict[str, Any]) -> Dict[str, Any]:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "status",
        "dataset_id",
        "output_directory",
        "output_manifest",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ContractError("Set data/config.json to the documented build fields.")
    if payload["schema_version"] != "miniagent.data-build.v1":
        raise ContractError("Set data config schema_version to miniagent.data-build.v1.")
    if payload["status"] != "approved":
        raise ContractError("Approve data/config.json with a new immutable dataset ID.")
    dataset_id = payload["dataset_id"]
    if not isinstance(dataset_id, str) or IDENTIFIER.fullmatch(dataset_id) is None:
        raise ContractError("Set data.config.dataset_id to a stable identifier.")

    output_directory = resolve_path(payload["output_directory"], "data.output_directory")
    output_manifest = resolve_path(payload["output_manifest"], "data.output_manifest")
    manifest_root = resolve_path(
        contract["artifacts"]["dataset_manifests"],
        "artifacts.dataset_manifests",
    )
    payload_root = manifest_root.parent
    if output_manifest.parent != manifest_root or output_manifest.name != f"{dataset_id}.json":
        raise ContractError(
            "Write output_manifest as artifacts.dataset_manifests/<dataset_id>.json."
        )
    if output_directory.parent != payload_root or output_directory.name != dataset_id:
        raise ContractError("Write dataset payloads as data/artifacts/<dataset_id>/.")
    if output_manifest.exists() or output_directory.exists():
        raise ContractError(
            f"Select a new dataset_id; immutable dataset {dataset_id} already exists."
        )
    return payload


def _load_pipeline() -> Any:
    spec = importlib.util.spec_from_file_location("application_data_pipeline", PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise ContractError("Provide an importable data/pipeline.py module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _artifact_record(path: Path) -> Dict[str, Any]:
    return {
        "path": path.relative_to(APPLICATION_ROOT).as_posix(),
        "sha256": sha256_path(path),
        "bytes": artifact_bytes(path),
    }


def _snapshot(
    source: Path,
    staging_root: Path,
    published_root: Path,
    relative: Path,
) -> Dict[str, Any]:
    staged = staging_root / relative
    if staged.exists():
        raise ContractError(f"Reserve {relative.as_posix()} for provenance snapshots.")
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, staged)
    record = _artifact_record(staged)
    record["path"] = (published_root / relative).relative_to(APPLICATION_ROOT).as_posix()
    return record


def _snapshot_contracts(
    contract: Dict[str, Any], staging_root: Path, published_root: Path
) -> list[Dict[str, Any]]:
    sources: list[Tuple[str, Path]] = [("application", CONTRACT_PATH)]
    sources.extend(
        ("workflow", path) for path in declared_json_documents(contract, "workflows")
    )
    sources.extend(("tool", path) for path in declared_json_documents(contract, "tools"))
    records = []
    for role, source in sources:
        source_path = source.relative_to(APPLICATION_ROOT)
        artifact = _snapshot(
            source,
            staging_root,
            published_root,
            Path("provenance") / "contracts" / source_path,
        )
        records.append(
            {
                "role": role,
                "source_path": source_path.as_posix(),
                "artifact": artifact,
            }
        )
    return records


def _verify_staged_reference(
    value: Any,
    label: str,
    staging_root: Path,
    published_root: Path,
) -> Path:
    if not isinstance(value, dict):
        raise ContractError(f"Set {label} to an artifact record.")
    published = resolve_path(value.get("path"), f"{label}.path")
    try:
        relative = published.relative_to(published_root)
    except ValueError as exc:
        raise ContractError(
            f"Keep {label} inside this dataset's immutable payload directory."
        ) from exc
    staged = staging_root / relative
    observed = sha256_path(staged)
    if value.get("sha256") != observed:
        raise ContractError(f"{label} hash differs from its staged artifact.")
    if value.get("bytes") is not None and value["bytes"] != artifact_bytes(staged):
        raise ContractError(f"{label} byte count differs from its staged artifact.")
    return staged


def _generated_records(result: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    yield "generator.source", result["generator"]["source"]
    yield "generator.configuration", result["generator"]["configuration"]
    yield (
        "generator.conversation_validator",
        result["generator"]["conversation_validator"],
    )
    for position, item in enumerate(result["source_contracts"]):
        yield f"source_contracts[{position}].artifact", item["artifact"]
    yield "world.specification", result["world"]["specification"]
    yield "world.simulator", result["world"]["simulator"]
    yield "world.determinism_check.evidence", result["world"]["determinism_check"]["evidence"]
    yield "record_schema", result["record_schema"]
    for position, split in enumerate(result["splits"]):
        yield f"splits[{position}]", split
    yield "grouping.leakage_check.evidence", result["grouping"]["leakage_check"]["evidence"]


def _validate_result(
    result: Dict[str, Any],
    contract: Dict[str, Any],
    config: Dict[str, Any],
    staging_root: Path,
    published_root: Path,
) -> None:
    validate_schema(result, "dataset-manifest.schema.json", "Generated dataset manifest")
    if result["application_id"] != contract["application_id"]:
        raise ContractError(
            "Set the dataset manifest application_id to the compiled application_id."
        )
    if result["dataset_id"] != config["dataset_id"]:
        raise ContractError("Match the generated dataset_id to data/config.json.")

    declared_artifacts = [
        _verify_staged_reference(record, label, staging_root, published_root)
        for label, record in _generated_records(result)
    ]
    resolved_artifacts = [path.resolve() for path in declared_artifacts]
    if len(resolved_artifacts) != len(set(resolved_artifacts)):
        raise ContractError("Bind each dataset artifact path once.")
    if any(
        left in right.parents
        for left in resolved_artifacts
        for right in resolved_artifacts
        if left != right
    ):
        raise ContractError("Keep dataset artifact records non-overlapping.")
    symlinks = [path for path in staging_root.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ContractError("Build dataset payloads from regular files and directories.")
    unbound = []
    for path in (item for item in staging_root.rglob("*") if item.is_file()):
        if not any(
            path.resolve() == declared
            or (declared.is_dir() and declared in path.resolve().parents)
            for declared in resolved_artifacts
        ):
            unbound.append(path.relative_to(staging_root).as_posix())
    if unbound:
        raise ContractError(
            "Bind every generated dataset file into the manifest: "
            + ", ".join(sorted(unbound))
            + "."
        )

    expected_sources = {
        CONTRACT_PATH.relative_to(APPLICATION_ROOT).as_posix(),
        *(
            path.relative_to(APPLICATION_ROOT).as_posix()
            for path in declared_json_documents(contract, "workflows")
        ),
        *(
            path.relative_to(APPLICATION_ROOT).as_posix()
            for path in declared_json_documents(contract, "tools")
        ),
    }
    observed_sources = {item["source_path"] for item in result["source_contracts"]}
    if observed_sources != expected_sources:
        raise ContractError(
            "Snapshot the exact approved application, workflow, and tool contracts."
        )
    source_roles = [item["role"] for item in result["source_contracts"]]
    if source_roles.count("application") != 1:
        raise ContractError("Snapshot the application contract exactly once.")
    for item in result["source_contracts"]:
        source = resolve_path(item["source_path"], "source_contracts.source_path")
        staged = _verify_staged_reference(
            item["artifact"], "source contract snapshot", staging_root, published_root
        )
        if sha256_path(source) != sha256_path(staged):
            raise ContractError("Match every source contract snapshot to its approved source.")

    def staged_resolver(record: Dict[str, Any], label: str) -> Path:
        return _verify_staged_reference(
            record, label, staging_root, published_root
        )

    observed_workflow_ids, observed_route_ids, _ = validate_dataset_content(
        result, contract, staged_resolver
    )
    for label, record in _generated_records(result):
        _verify_staged_reference(record, label, staging_root, published_root)
    if any(path.is_symlink() for path in staging_root.rglob("*")):
        raise ContractError("Conversation validation created a dataset symlink.")
    post_validation_files = [
        item for item in staging_root.rglob("*") if item.is_file()
    ]
    post_validation_unbound = [
        path.relative_to(staging_root).as_posix()
        for path in post_validation_files
        if not any(
            path.resolve() == declared
            or (declared.is_dir() and declared in path.resolve().parents)
            for declared in resolved_artifacts
        )
    ]
    if post_validation_unbound:
        raise ContractError(
            "Conversation validation created unbound dataset files: "
            + ", ".join(sorted(post_validation_unbound))
            + "."
        )
    splits = result["splits"]
    split_names = [item["name"] for item in splits]
    split_paths = [item["path"] for item in splits]
    if len(split_names) != len(set(split_names)):
        raise ContractError("Assign a unique name to every dataset split.")
    if len(split_paths) != len(set(split_paths)):
        raise ContractError("Assign a unique path to every dataset split.")
    if not any(item["purpose"] == "train" for item in splits):
        raise ContractError("Create a training-purpose dataset split.")
    evaluation_splits = [item for item in splits if item["purpose"] == "evaluation"]
    if len(evaluation_splits) != 1 or evaluation_splits[0]["sealed"] is not True:
        raise ContractError("Create one sealed evaluation-purpose dataset split.")
    if any(item["groups"] > item["records"] for item in splits):
        raise ContractError("Keep each split group count within its record count.")

    coverage_by_dimension = {}
    for coverage in result["coverage"]:
        dimension = coverage["dimension"]
        if dimension in coverage_by_dimension:
            raise ContractError(f"Assign one coverage record for {dimension}.")
        expected = set(coverage["expected"])
        observed = set(coverage["observed"])
        missing = set(coverage["missing"])
        if missing != expected.difference(observed):
            raise ContractError(f"Compute coverage.missing for {dimension}.")
        if missing:
            raise ContractError(f"Complete the missing {dimension} coverage.")
        coverage_by_dimension[dimension] = coverage
    route_ids = set()
    workflow_ids = set()
    for path in declared_json_documents(contract, "workflows"):
        workflow = json.loads(path.read_text(encoding="utf-8"))
        workflow_ids.add(workflow["workflow_id"])
        route_ids.update(route["id"] for route in workflow["routes"])
    workflow_coverage = coverage_by_dimension.get("workflow")
    if workflow_coverage is None:
        raise ContractError("Add workflow coverage to the dataset manifest.")
    if set(workflow_coverage["expected"]) != workflow_ids:
        raise ContractError("Derive expected workflow coverage from approved workflows.")
    if set(workflow_coverage["observed"]) != observed_workflow_ids:
        raise ContractError("Derive observed workflow coverage from dataset records.")
    route_coverage = coverage_by_dimension.get("route")
    if route_coverage is None:
        raise ContractError("Add route coverage to the dataset manifest.")
    if set(route_coverage["expected"]) != route_ids:
        raise ContractError("Derive expected route coverage from the approved workflows.")
    if set(route_coverage["observed"]) != observed_route_ids:
        raise ContractError("Derive observed route coverage from dataset records.")
    if observed_route_ids != route_ids:
        raise ContractError("Cover every approved route in the dataset records.")


def main() -> int:
    try:
        contract = load_contract()
        require_approved(contract)
        config = _load_config(contract)
        manifest_path = resolve_path(config["output_manifest"], "data.output_manifest")
        published_root = resolve_path(config["output_directory"], "data.output_directory")
        payload_root = published_root.parent
        payload_root.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        pipeline = _load_pipeline()
        if pipeline.PIPELINE_READY is not True:
            raise ContractError(
                "Implement data/pipeline.py and set PIPELINE_READY = True after its "
                "application tests pass."
            )

        with tempfile.TemporaryDirectory(
            dir=payload_root, prefix=f".{config['dataset_id']}."
        ) as temporary:
            staging_root = Path(temporary)
            result = pipeline.build(contract, staging_root, published_root)
            if not isinstance(result, dict):
                raise ContractError(
                    "Return a dataset-manifest object from data.pipeline.build()."
                )
            result["source_contracts"] = _snapshot_contracts(
                contract, staging_root, published_root
            )
            generator = result.get("generator")
            if not isinstance(generator, dict):
                raise ContractError("Return generator metadata from data.pipeline.build().")
            generator["source"] = _snapshot(
                PIPELINE_PATH,
                staging_root,
                published_root,
                Path("provenance/generator/pipeline.py"),
            )
            generator["configuration"] = _snapshot(
                CONFIG_PATH,
                staging_root,
                published_root,
                Path("provenance/generator/build-config.json"),
            )
            generator["conversation_validator"] = _snapshot(
                CONVERSATION_VALIDATOR_PATH,
                staging_root,
                published_root,
                Path("provenance/generator/conversation-validator.py"),
            )
            _validate_result(result, contract, config, staging_root, published_root)
            rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=manifest_path.parent,
                prefix=f".{manifest_path.name}.",
                delete=False,
            ) as handle:
                manifest_staging = Path(handle.name)
                handle.write(rendered)
            try:
                staging_root.replace(published_root)
                manifest_staging.replace(manifest_path)
            finally:
                manifest_staging.unlink(missing_ok=True)
    except (
        AttributeError,
        ArtifactError,
        ContractError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        SchemaError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Data-build action: {exc}")
        return 2

    print(f"Wrote immutable dataset manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
