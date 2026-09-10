import importlib.util
from pathlib import Path
import sys

import pytest


APPLICATION_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = APPLICATION_ROOT / "eval" / "promote.py"
SPEC = importlib.util.spec_from_file_location("application_promote", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PROMOTE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROMOTE
SPEC.loader.exec_module(PROMOTE)


def _release(artifact_hash: str, destination: str) -> dict:
    artifact = {"path": "source", "sha256": artifact_hash, "bytes": 4}
    return {
        "policy": {
            "weights": [{"role": "merged", "artifact": artifact}],
            "action_parser": {"artifact": artifact},
            "runtime": {"artifact": artifact},
        },
        "artifacts": [],
        "evaluation": {"manifest": artifact},
        "provenance": {
            "application": artifact,
            "dataset": artifact,
            "training": artifact,
            "workflows": [artifact],
            "tools": [artifact],
        },
        "verification": [{"evidence": artifact}],
        "runtime": {
            "bundle": [
                {
                    "role": "policy",
                    "path": destination,
                    "sha256": artifact_hash,
                    "bytes": 4,
                }
            ]
        },
    }


def test_release_verification_checks_provenance(monkeypatch) -> None:
    artifact_hash = "1" * 64
    release = _release(artifact_hash, "model/policy")
    monkeypatch.setattr(PROMOTE, "verify_references", lambda values, label: None)

    def reject_dataset(value, label):
        if label == "release dataset":
            raise PROMOTE.ArtifactError("dataset mismatch")

    monkeypatch.setattr(PROMOTE, "verify_reference", reject_dataset)
    with pytest.raises(PROMOTE.ArtifactError, match="dataset mismatch"):
        PROMOTE._verify_release_artifacts(release)


def test_preflight_requires_the_runtime_bundle(tmp_path, monkeypatch) -> None:
    package_root = tmp_path / "src" / "example"
    package_root.mkdir(parents=True)
    artifact_hash = "2" * 64
    source = tmp_path / "source"
    source.write_bytes(b"data")
    monkeypatch.setattr(PROMOTE, "verify_reference", lambda value, label: (source, artifact_hash))
    release = _release(artifact_hash, "model/policy")
    selected = [
        {
            "source": "source",
            "destination": "model/policy",
            "sha256": artifact_hash,
        }
    ]
    verified = PROMOTE._preflight_artifacts(selected, package_root, release)
    assert verified[0][1] == package_root / "model" / "policy"

    release["runtime"]["bundle"][0]["path"] = "model/another-policy"
    with pytest.raises(PROMOTE.PromotionError, match="runtime.bundle"):
        PROMOTE._preflight_artifacts(selected, package_root, release)
