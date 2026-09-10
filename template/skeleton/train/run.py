"""Launch and verify one immutable application-owned training run."""

import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, Tuple


APPLICATION_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = APPLICATION_ROOT / "train" / "config.json"
DATA_SCRIPTS = APPLICATION_ROOT / "data" / "scripts"
sys.path.insert(0, str(DATA_SCRIPTS))

from _artifacts import (
    ArtifactError,
    resolve_path,
    sha256_path,
    verify_reference,
)
from _contract import ContractError, load_contract, require_approved
from _dataset import verify_dataset_manifest
from _training import verify_training_manifest


IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:[._-][A-Za-z0-9]+)*$")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _load_config() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version",
        "status",
        "command",
        "input_manifest",
        "input_split",
        "run_configuration",
        "output_manifest",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError(
            "Set train/config.json fields to: "
            + ", ".join(sorted(expected_fields))
            + "."
        )
    if payload["schema_version"] != "miniagent.training-run.v1":
        raise ValueError("Set train config schema_version to miniagent.training-run.v1.")
    if payload["status"] != "approved":
        raise ValueError(
            "Select the training stack and immutable run ID, then approve train/config.json."
        )
    command = payload["command"]
    if not isinstance(command, list) or not command or not all(
        isinstance(part, str) and part for part in command
    ):
        raise ValueError("Set train/config.json command to a non-empty argument list.")

    contract = load_contract()
    require_approved(contract)
    dataset_root = resolve_path(
        contract["artifacts"]["dataset_manifests"], "artifacts.dataset_manifests"
    )
    training_root = resolve_path(
        contract["artifacts"]["training_runs"], "artifacts.training_runs"
    )
    input_path = resolve_path(payload["input_manifest"], "train.input_manifest")
    if input_path.parent != dataset_root or not input_path.is_file():
        raise ValueError("Select an immutable dataset manifest from artifacts.dataset_manifests.")
    output_path = resolve_path(payload["output_manifest"], "train.output_manifest")
    run_directory = output_path.parent
    if (
        run_directory.parent != training_root
        or output_path.name != "training-manifest.json"
        or IDENTIFIER.fullmatch(run_directory.name) is None
    ):
        raise ValueError(
            "Write output_manifest as artifacts.training_runs/<run_id>/training-manifest.json."
        )
    if output_path.exists():
        raise ValueError(f"Select a new run_id; immutable run {run_directory.name} exists.")
    run_configuration = resolve_path(
        payload["run_configuration"], "train.run_configuration"
    )
    configuration_root = APPLICATION_ROOT / "train" / "configurations"
    if not _inside(run_configuration, configuration_root) or not run_configuration.exists():
        raise ValueError("Create the trainer configuration under train/configurations/.")

    dataset = verify_dataset_manifest(input_path, contract)
    training_splits = [
        item
        for item in dataset["splits"]
        if item["name"] == payload["input_split"] and item["purpose"] == "train"
    ]
    if len(training_splits) != 1:
        raise ValueError("Select one training-purpose dataset split by name.")
    return payload, contract, dataset


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


def _write_launcher(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _prepare_run(
    config: Dict[str, Any], argv: list[str]
) -> Tuple[Path, Path, Path]:
    output_path = resolve_path(config["output_manifest"], "train.output_manifest")
    run_directory = output_path.parent
    launcher_snapshot = run_directory / "provenance" / "launcher-config.json"
    trainer_snapshot = run_directory / "provenance" / "trainer-configuration"
    expected_launcher = _launcher_payload(config, argv)
    if run_directory.exists():
        if not launcher_snapshot.exists() or not trainer_snapshot.exists():
            raise ValueError(
                "Resume an in-progress run only with its launcher and trainer snapshots."
            )
        observed_launcher = json.loads(launcher_snapshot.read_text(encoding="utf-8"))
        if observed_launcher != expected_launcher:
            raise ValueError("Resume the run with its original effective invocation.")
        current_trainer = resolve_path(
            config["run_configuration"], "train.run_configuration"
        )
        if sha256_path(current_trainer) != sha256_path(trainer_snapshot):
            raise ValueError("Resume the run with its original trainer configuration.")
        return run_directory, launcher_snapshot, trainer_snapshot
    run_directory.mkdir(parents=True, exist_ok=False)
    _write_launcher(launcher_snapshot, expected_launcher)
    _copy_snapshot(
        resolve_path(config["run_configuration"], "train.run_configuration"),
        trainer_snapshot,
    )
    return run_directory, launcher_snapshot, trainer_snapshot


def _validate_bound_snapshot(record: Dict[str, Any], expected: Path, label: str) -> None:
    observed, _ = verify_reference(record, label)
    if observed.resolve() != expected.resolve():
        raise ValueError(f"Bind {label} to its immutable run snapshot.")


def _validate_training_manifest(
    config: Dict[str, Any],
    contract: Dict[str, Any],
    dataset: Dict[str, Any],
    run_directory: Path,
    launcher_snapshot: Path,
    trainer_snapshot: Path,
    argv: list[str],
) -> None:
    output_path = resolve_path(config["output_manifest"], "train.output_manifest")
    manifest, verified_dataset = verify_training_manifest(output_path, contract)
    if verified_dataset != dataset:
        raise ValueError("Keep the verified dataset identity fixed for this training run.")
    dataset_path, _ = verify_reference(manifest["dataset"], "training dataset")
    expected_dataset = resolve_path(config["input_manifest"], "train.input_manifest")
    if dataset_path.resolve() != expected_dataset.resolve():
        raise ValueError("Bind the configured dataset manifest into the training run.")
    training_splits = [
        item
        for item in dataset["splits"]
        if item["name"] == config["input_split"] and item["purpose"] == "train"
    ]
    if len(training_splits) != 1 or manifest["training_split"] != training_splits[0]:
        raise ValueError("Bind the configured training split into the training run.")

    _validate_bound_snapshot(
        manifest["launcher_configuration"], launcher_snapshot, "training launcher configuration"
    )
    _validate_bound_snapshot(
        manifest["configuration"], trainer_snapshot, "trainer configuration"
    )
    current_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if current_config != config:
        raise ValueError("Keep train/config.json unchanged while the run executes.")
    if json.loads(launcher_snapshot.read_text(encoding="utf-8")) != _launcher_payload(
        config, argv
    ):
        raise ValueError("Keep the effective training invocation unchanged.")
    if manifest["reproducibility"]["argv"] != argv:
        raise ValueError("Bind the effective training argv into the run manifest.")
    source_configuration = resolve_path(
        config["run_configuration"], "train.run_configuration"
    )
    if sha256_path(source_configuration) != sha256_path(trainer_snapshot):
        raise ValueError("Keep the trainer configuration unchanged while the run executes.")


def main() -> int:
    passthrough = list(sys.argv[1:])
    try:
        config, contract, dataset = _load_config()
        dataset_path = resolve_path(config["input_manifest"], "train.input_manifest")
        dataset_manifest_sha = sha256_path(dataset_path)
        command = config["command"] + passthrough
        run_directory, launcher_snapshot, trainer_snapshot = _prepare_run(
            config, command
        )
    except (ArtifactError, ContractError, json.JSONDecodeError, OSError, ValueError) as exc:
        print(f"Training action: {exc}")
        return 2

    print("Training command:", " ".join(command))
    return_code = subprocess.run(command, cwd=APPLICATION_ROOT, check=False).returncode
    if return_code:
        return return_code
    try:
        verify_dataset_manifest(dataset_path, contract)
        if sha256_path(dataset_path) != dataset_manifest_sha:
            raise ValueError("Keep the verified dataset manifest unchanged during training.")
        _validate_training_manifest(
            config,
            contract,
            dataset,
            run_directory,
            launcher_snapshot,
            trainer_snapshot,
            command,
        )
    except (ArtifactError, ContractError, json.JSONDecodeError, OSError, ValueError) as exc:
        print(f"Training action: {exc}")
        return 2
    print(f"Validated immutable training manifest: {APPLICATION_ROOT / config['output_manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
