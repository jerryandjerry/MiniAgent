import json
from pathlib import Path


APPLICATION_ROOT = Path(__file__).resolve().parents[2]


def test_training_config_declares_stable_launcher_fields() -> None:
    config = json.loads(
        (APPLICATION_ROOT / "train/config.json").read_text(encoding="utf-8")
    )
    assert set(config) == {
        "schema_version",
        "status",
        "command",
        "input_manifest",
        "input_split",
        "run_configuration",
        "output_manifest",
    }
    assert config["schema_version"] == "miniagent.training-run.v1"


def test_training_launcher_is_valid_python() -> None:
    source = (APPLICATION_ROOT / "train/run.py").read_text(encoding="utf-8")
    compile(source, "train/run.py", "exec")
