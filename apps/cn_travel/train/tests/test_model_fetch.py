"""Integrity checks for fetching the sealed training model artifacts."""
from __future__ import annotations

import importlib.util
import shutil
import sys

import pytest

from cn_travel.artifact_integrity import build_directory_manifest, verify_directory_manifest
from project_paths import TRAIN_ROOT


def _load(name: str):
    module_path = TRAIN_ROOT / "fetch_models.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree(tmp_path):
    root = tmp_path / "artifact"
    (root / "nested").mkdir(parents=True)
    (root / "tokenizer.json").write_bytes(b"tokenizer-v1")
    (root / "nested" / "config.json").write_bytes(b'{"rank":32}')
    return root


def test_model_fetch_installs_a_clean_snapshot(tmp_path):
    module = _load("cn_travel_fetch_models_test")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    snapshot.joinpath("config.json").write_bytes(b"new")
    snapshot.joinpath(".cache").mkdir()
    snapshot.joinpath(".cache", "metadata.json").write_bytes(b"metadata")
    destination = tmp_path / "installed"
    destination.mkdir()
    destination.joinpath("old.json").write_bytes(b"old")

    module._install_snapshot(snapshot, destination)
    assert destination.joinpath("config.json").read_bytes() == b"new"
    assert not destination.joinpath("old.json").exists()
    assert not destination.joinpath(".cache").exists()


def test_model_fetch_installs_only_the_pinned_inventory(tmp_path):
    module = _load("cn_travel_fetch_inventory_test")
    snapshot = _tree(tmp_path)
    snapshot.joinpath(".gitattributes").write_bytes(b"upstream-only")
    expected = build_directory_manifest(snapshot)
    expected_files = tuple(
        entry["path"] for entry in expected["files"] if entry["path"] != ".gitattributes"
    )
    clean = tmp_path / "clean"
    shutil.copytree(snapshot, clean)
    clean.joinpath(".gitattributes").unlink()
    expected = build_directory_manifest(clean)

    destination = tmp_path / "installed"
    module._install_snapshot(
        snapshot,
        destination,
        file_inventory=expected_files,
        expected_manifest=expected,
    )
    assert build_directory_manifest(destination) == expected
    assert not destination.joinpath(".gitattributes").exists()


def test_model_fetch_preserves_destination_when_a_required_file_is_missing(tmp_path):
    module = _load("cn_travel_fetch_missing_test")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    destination = tmp_path / "installed"
    destination.mkdir()
    destination.joinpath("existing.json").write_bytes(b"preserved")

    with pytest.raises(FileNotFoundError, match="config.json"):
        module._install_snapshot(snapshot, destination, file_inventory=("config.json",))
    assert destination.joinpath("existing.json").read_bytes() == b"preserved"


def test_every_pinned_model_inventory_matches_the_preserved_tree():
    module = _load("cn_travel_fetch_seals_test")
    for name, (_env, _repo, destination) in module.MODEL_SOURCES.items():
        expected = module._deployment_directory_manifest(name)
        assert expected is not None
        assert verify_directory_manifest(destination, expected, name) == expected
        assert module._default_file_inventory(name) == tuple(
            entry["path"] for entry in expected["files"]
        )


def test_model_source_override_keeps_the_sealed_inventory(monkeypatch, tmp_path):
    module = _load("cn_travel_fetch_override_test")
    captured = {}

    def snapshot_download(**kwargs):
        captured["download"] = kwargs
        return str(tmp_path)

    def install_snapshot(snapshot, destination, **kwargs):
        captured["install"] = (snapshot, destination, kwargs)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(module, "_install_snapshot", install_snapshot)
    monkeypatch.setattr(sys, "argv", ["fetch_models.py", "qwen"])
    monkeypatch.setenv("CN_TRAVEL_QWEN_REPO", "mirror/Qwen3.5-0.8B")
    monkeypatch.setenv("CN_TRAVEL_QWEN_REVISION", "mirror-revision")

    assert module.main() == 0
    expected = module._deployment_directory_manifest("qwen")
    assert captured["download"] == {
        "repo_id": "mirror/Qwen3.5-0.8B",
        "revision": "mirror-revision",
        "allow_patterns": [entry["path"] for entry in expected["files"]],
    }
    assert captured["install"][2] == {
        "file_inventory": tuple(entry["path"] for entry in expected["files"]),
        "expected_manifest": expected,
    }
