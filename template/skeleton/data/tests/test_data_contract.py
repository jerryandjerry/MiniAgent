import json
import importlib
from pathlib import Path
import sys

from jsonschema import Draft202012Validator
import pytest


APPLICATION_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APPLICATION_ROOT / "data" / "scripts"))
artifacts_module = importlib.import_module("_artifacts")


def test_compiled_contract_has_stable_top_level_shape() -> None:
    contract = json.loads(
        (APPLICATION_ROOT / "data/specification/application.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(contract) == {
        "schema_version",
        "status",
        "application_id",
        "package",
        "name",
        "version",
        "business",
        "interfaces",
        "artifacts",
        "acceptance",
    }
    assert contract["schema_version"] == "miniagent.application.v1"
    assert contract["application_id"]
    assert set(contract["artifacts"]) == {
        "workflows",
        "tools",
        "dataset_manifests",
        "training_runs",
        "evaluation_results",
        "releases",
        "runtime_source",
    }


def test_data_pipeline_exposes_build_entrypoint() -> None:
    source = (APPLICATION_ROOT / "data/pipeline.py").read_text(encoding="utf-8")
    compile(source, "data/pipeline.py", "exec")
    assert "def build(" in source


def test_data_build_config_declares_immutable_outputs() -> None:
    config = json.loads((APPLICATION_ROOT / "data/config.json").read_text(encoding="utf-8"))
    assert set(config) == {
        "schema_version",
        "status",
        "dataset_id",
        "output_directory",
        "output_manifest",
    }
    assert config["schema_version"] == "miniagent.data-build.v1"


def test_vendored_contract_schemas_are_valid() -> None:
    schema_root = APPLICATION_ROOT / "data/specification/schemas"
    schemas = sorted(schema_root.glob("*.schema.json"))
    assert len(schemas) == 7
    for path in schemas:
        Draft202012Validator.check_schema(json.loads(path.read_text(encoding="utf-8")))


def test_artifact_paths_reject_symlinked_ancestry(tmp_path, monkeypatch) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "artifact.json").write_text("{}\n", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(artifacts_module, "APPLICATION_ROOT", tmp_path)
    with pytest.raises(artifacts_module.ArtifactError, match="ancestry"):
        artifacts_module.resolve_path("alias/artifact.json", "artifact")
