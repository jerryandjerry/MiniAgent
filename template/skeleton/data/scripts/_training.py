"""Verification for immutable MiniAgent training runs."""

import json
from pathlib import Path
import shlex
from typing import Any, Dict, Tuple

from _artifacts import resolve_path, verify_reference, verify_references
from _contract import ContractError, validate_schema
from _dataset import verify_dataset_manifest


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _key(record: Dict[str, Any]) -> Tuple[str, str]:
    return str(record["path"]), str(record["sha256"])


def verify_training_manifest(
    manifest_path: Path, contract: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_schema(manifest, "training-manifest.schema.json", "Training manifest")
    if manifest["application_id"] != contract["application_id"]:
        raise ContractError("Match the training and application contract identities.")
    training_root = resolve_path(
        contract["artifacts"]["training_runs"], "artifacts.training_runs"
    )
    run_directory = manifest_path.parent
    if (
        run_directory.parent.resolve() != training_root.resolve()
        or manifest_path.name != "training-manifest.json"
        or manifest["run_id"] != run_directory.name
    ):
        raise ContractError("Use the canonical immutable training run path and ID.")

    dataset_path, _ = verify_reference(manifest["dataset"], "training dataset")
    dataset = verify_dataset_manifest(dataset_path, contract)
    if manifest["dataset"]["sha256"] != verify_reference(
        manifest["dataset"], "training dataset"
    )[1]:
        raise ContractError("Bind the verified dataset hash into the training run.")
    matching_splits = [
        item
        for item in dataset["splits"]
        if item["name"] == manifest["training_split"]["name"]
        and item["purpose"] == "train"
    ]
    if len(matching_splits) != 1 or manifest["training_split"] != matching_splits[0]:
        raise ContractError("Bind one exact training-purpose split into the training run.")
    verify_reference(manifest["training_split"], "training split")

    base_records = [item["artifact"] for item in manifest["base_model"]["artifacts"]]
    verify_references(base_records, "base model artifacts")
    base_keys = [(_key(item["artifact"]), item["role"]) for item in manifest["base_model"]["artifacts"]]
    if len(base_keys) != len(set(base_keys)):
        raise ContractError("List each typed base-model artifact once.")

    declared = [
        manifest["launcher_configuration"],
        manifest["configuration"],
        manifest["trainer"]["source"],
        manifest["trainer"]["environment"],
        *(item["artifact"] for item in manifest["outputs"]),
    ]
    verify_references(declared, "training run artifacts")
    launcher_path, _ = verify_reference(
        manifest["launcher_configuration"], "training launcher configuration"
    )
    launcher = json.loads(launcher_path.read_text(encoding="utf-8"))
    expected_launcher_fields = {"schema_version", "configuration", "argv"}
    if (
        not isinstance(launcher, dict)
        or set(launcher) != expected_launcher_fields
        or launcher["schema_version"] != "miniagent.effective-invocation.v1"
        or launcher["argv"] != manifest["reproducibility"]["argv"]
        or manifest["reproducibility"]["command"]
        != shlex.join(manifest["reproducibility"]["argv"])
    ):
        raise ContractError("Bind the effective training invocation into the run.")
    declared_paths = [
        resolve_path(item["path"], "training run artifact").resolve()
        for item in declared
    ]
    if len(declared_paths) != len(set(declared_paths)):
        raise ContractError("Bind each training run artifact path once.")
    if any(
        left in right.parents
        for left in declared_paths
        for right in declared_paths
        if left != right
    ):
        raise ContractError("Keep training run artifact records non-overlapping.")
    if any(not _inside(path, run_directory) for path in declared_paths):
        raise ContractError("Keep trainer snapshots and outputs inside the immutable run.")
    if manifest_path.resolve() in declared_paths:
        raise ContractError("Keep the training manifest outside its hashed artifacts.")
    if any(path.is_symlink() for path in run_directory.rglob("*")):
        raise ContractError("Build training runs from regular files and directories.")
    unbound = []
    for path in (item for item in run_directory.rglob("*") if item.is_file()):
        if path.resolve() == manifest_path.resolve():
            continue
        if not any(
            path.resolve() == declared
            or (declared.is_dir() and declared in path.resolve().parents)
            for declared in declared_paths
        ):
            unbound.append(path.relative_to(run_directory).as_posix())
    if unbound:
        raise ContractError(
            "Bind every training run file into the manifest: "
            + ", ".join(sorted(unbound))
            + "."
        )

    output_roles: Dict[str, list[Dict[str, Any]]] = {}
    for output in manifest["outputs"]:
        output_roles.setdefault(output["role"], []).append(output["artifact"])
    for role in ("adapter", "merged", "tokenizer", "chat_template", "action_parser", "runtime_adapter"):
        if len(output_roles.get(role, [])) > 1:
            raise ContractError(f"Write at most one final {role} training output.")
    deployable = [
        item
        for role in ("adapter", "merged")
        for item in output_roles.get(role, [])
    ]
    if not deployable:
        raise ContractError("Write an adapter or merged policy as a training output.")
    if manifest["reproducibility"]["final_sha256"] not in {
        item["sha256"] for item in deployable
    }:
        raise ContractError("Set final_sha256 to the deployable policy output.")
    return manifest, dataset
