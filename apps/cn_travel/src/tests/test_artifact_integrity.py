"""Integrity checks for complete promoted model directories."""
from __future__ import annotations

import copy
import os
import shutil

import pytest

from cn_travel.artifact_integrity import (
    ArtifactIntegrityError,
    build_directory_manifest,
    directory_manifest_errors,
    validate_directory_manifest,
    verify_directory_manifest,
)


def _tree(tmp_path):
    root = tmp_path / "artifact"
    (root / "nested").mkdir(parents=True)
    (root / "tokenizer.json").write_bytes(b"tokenizer-v1")
    (root / "nested" / "config.json").write_bytes(b'{"rank":32}')
    return root


def test_manifest_binds_every_file_and_is_deterministic(tmp_path):
    root = _tree(tmp_path)
    expected = build_directory_manifest(root)
    assert expected == build_directory_manifest(root)
    assert expected["file_count"] == 2
    assert expected["total_bytes"] == sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    assert verify_directory_manifest(root, expected, "artifact") == expected


@pytest.mark.parametrize("change", ["missing", "changed", "unexpected"])
def test_directory_manifest_rejects_every_tree_change(tmp_path, change):
    root = _tree(tmp_path)
    expected = build_directory_manifest(root)
    if change == "missing":
        (root / "tokenizer.json").unlink()
    elif change == "changed":
        (root / "tokenizer.json").write_bytes(b"Tokenizer-v1")
    else:
        (root / "trainer_state.json").write_bytes(b"{}")

    errors, actual = directory_manifest_errors(root, expected, "artifact")
    assert errors and actual is not None
    with pytest.raises(ArtifactIntegrityError):
        verify_directory_manifest(root, expected, "artifact")


def test_directory_manifest_rejects_symlinks(tmp_path):
    root = _tree(tmp_path)
    (root / "linked-config.json").symlink_to(root / "nested" / "config.json")
    with pytest.raises(ArtifactIntegrityError, match="symlink"):
        build_directory_manifest(root)


def test_directory_manifest_rejects_non_regular_nodes(tmp_path):
    root = _tree(tmp_path)
    fifo = root / "artifact.pipe"
    try:
        os.mkfifo(fifo)
    except (AttributeError, NotImplementedError, OSError) as exc:
        pytest.skip(f"FIFO creation is unavailable: {exc}")
    with pytest.raises(ArtifactIntegrityError, match="non-regular"):
        build_directory_manifest(root)


@pytest.mark.parametrize("portable_path", ["nested\\config.json", "cafe\u0301.json"])
def test_manifest_rejects_non_portable_paths(tmp_path, portable_path):
    manifest = build_directory_manifest(_tree(tmp_path))
    malformed = copy.deepcopy(manifest)
    malformed["files"][0]["path"] = portable_path
    assert any(
        "invalid path" in error for error in validate_directory_manifest(malformed)
    )


def test_hash_disabled_verification_still_requires_exact_inventory_and_sizes(tmp_path):
    root = _tree(tmp_path)
    expected = build_directory_manifest(root)
    assert verify_directory_manifest(
        root, expected, "artifact", verify_hashes=False
    )["file_count"] == 2

    (root / "unexpected.json").write_bytes(b"{}")
    errors, _ = directory_manifest_errors(
        root, expected, "artifact", verify_hashes=False
    )
    assert any("unexpected" in error for error in errors)


def test_identical_source_and_target_corruption_still_fails_the_seal(tmp_path):
    source = _tree(tmp_path)
    expected = build_directory_manifest(source)
    target = tmp_path / "target"
    shutil.copytree(source, target)
    source.joinpath("tokenizer.json").write_bytes(b"Tokenizer-v1")
    target.joinpath("tokenizer.json").write_bytes(b"Tokenizer-v1")
    source_errors, _ = directory_manifest_errors(source, expected, "source")
    target_errors, _ = directory_manifest_errors(target, expected, "target")
    assert any("SHA-256 differs" in error for error in source_errors)
    assert any("SHA-256 differs" in error for error in target_errors)


def test_manifest_schema_is_self_consistent(tmp_path):
    manifest = build_directory_manifest(_tree(tmp_path))
    assert validate_directory_manifest(manifest) == []

    malformed = copy.deepcopy(manifest)
    malformed["files"][0]["path"] = "../escape"
    malformed["file_count"] += 1
    malformed["total_bytes"] += 1
    errors = validate_directory_manifest(malformed)
    assert any("invalid path" in error for error in errors)
    assert any("file_count" in error for error in errors)
    assert any("total_bytes" in error for error in errors)

    wrong_tree = copy.deepcopy(manifest)
    wrong_tree["tree_sha256"] = "0" * 64
    assert any(
        "tree_sha256" in error for error in validate_directory_manifest(wrong_tree)
    )
