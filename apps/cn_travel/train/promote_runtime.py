#!/usr/bin/env python3
"""Promote the selected policy and retrieval artifacts into the runtime package."""
from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from cn_travel.artifact_integrity import directory_manifest_errors
from training.paths import APP_ROOT, DATA_ROOT, PACKAGE_ROOT, TRAIN_ROOT


DIRECTORIES = {
    TRAIN_ROOT / "base" / "Qwen3.5-0.8B": PACKAGE_ROOT / "model" / "policy" / "base",
    TRAIN_ROOT / "qwen3_5_0_8b_lora" / "GRPO_TRL" / "adapter": (
        PACKAGE_ROOT / "model" / "policy" / "adapter"
    ),
    TRAIN_ROOT / "embedding" / "embeddinggemma-300m": (
        PACKAGE_ROOT
        / "tool"
        / "guide"
        / "knowledge_base"
        / "embedding"
        / "embeddinggemma-300m"
    ),
}
FILES = {
    DATA_ROOT / "knowledge_base" / "milvus.db": (
        PACKAGE_ROOT / "tool" / "guide" / "knowledge_base" / "milvus.db"
    ),
    DATA_ROOT / "knowledge_base" / ".embedding_model.json": (
        PACKAGE_ROOT / "tool" / "guide" / "knowledge_base" / ".embedding_model.json"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_files(path: Path) -> dict[Path, Path]:
    return {
        item.relative_to(path): item
        for item in path.rglob("*")
        if item.is_file()
    }


def _directories_equal(source: Path, target: Path) -> bool:
    source_files = _directory_files(source)
    target_files = _directory_files(target)
    return source_files.keys() == target_files.keys() and all(
        filecmp.cmp(source_files[name], target_files[name], shallow=False)
        for name in source_files
    )


def _verify_manifest_sources() -> None:
    manifest_path = PACKAGE_ROOT / "deployment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy = manifest["policy"]
    retrieval = manifest["retrieval"]
    checks = {
        DATA_ROOT / "knowledge_base" / "milvus.db": retrieval["index_sha256"],
        DATA_ROOT / "knowledge_base" / ".embedding_model.json": (
            retrieval["fingerprint_sha256"]
        ),
    }
    errors = []
    if manifest.get("schema_version") != "cn_travel.deployment.v2":
        errors.append("unsupported deployment manifest schema")
    sealed_directories = (
        (
            TRAIN_ROOT / "base" / "Qwen3.5-0.8B",
            policy.get("base_directory"),
            "source policy base",
        ),
        (
            TRAIN_ROOT / "qwen3_5_0_8b_lora" / "GRPO_TRL" / "adapter",
            policy.get("adapter_directory"),
            "source policy adapter",
        ),
        (
            TRAIN_ROOT / "embedding" / "embeddinggemma-300m",
            retrieval.get("embedding_directory"),
            "source embedding model",
        ),
    )
    for directory, expected, label in sealed_directories:
        directory_errors, _ = directory_manifest_errors(directory, expected, label)
        errors.extend(directory_errors)
    for path, expected in checks.items():
        if not path.is_file():
            errors.append(f"missing source artifact: {path.relative_to(APP_ROOT)}")
        elif _sha256(path) != expected:
            errors.append(f"source artifact hash differs: {path.relative_to(APP_ROOT)}")
    if errors:
        raise RuntimeError("\n".join(errors))


def _replace_directory(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.", dir=target.parent) as tmp:
        staged = Path(tmp) / target.name
        shutil.copytree(source, staged)
        backup = Path(tmp) / "previous"
        if target.exists():
            target.replace(backup)
        staged.replace(target)


def _replace_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{target.name}.", dir=target.parent,
                                     delete=False) as handle:
        staged = Path(handle.name)
    try:
        shutil.copy2(source, staged)
        staged.replace(target)
    finally:
        staged.unlink(missing_ok=True)


def verify_promoted() -> None:
    _verify_manifest_sources()
    manifest = json.loads(
        (PACKAGE_ROOT / "deployment_manifest.json").read_text(encoding="utf-8")
    )
    policy = manifest["policy"]
    retrieval = manifest["retrieval"]
    target_manifests = {
        PACKAGE_ROOT / "model" / "policy" / "base": policy["base_directory"],
        PACKAGE_ROOT / "model" / "policy" / "adapter": policy["adapter_directory"],
        PACKAGE_ROOT
        / "tool"
        / "guide"
        / "knowledge_base"
        / "embedding"
        / "embeddinggemma-300m": (
            retrieval["embedding_directory"]
        ),
    }
    errors = []
    for source, target in DIRECTORIES.items():
        directory_errors, _ = directory_manifest_errors(
            target, target_manifests[target], f"runtime {target.relative_to(PACKAGE_ROOT)}"
        )
        errors.extend(directory_errors)
        if not target.is_dir() or not _directories_equal(source, target):
            errors.append(
                f"runtime directory differs: {target.relative_to(APP_ROOT)}"
            )
    for source, target in FILES.items():
        if not target.is_file() or not filecmp.cmp(source, target, shallow=False):
            errors.append(f"runtime file differs: {target.relative_to(APP_ROOT)}")
    if errors:
        raise RuntimeError("\n".join(errors))


def promote() -> None:
    _verify_manifest_sources()
    for source, target in DIRECTORIES.items():
        _replace_directory(source, target)
    for source, target in FILES.items():
        _replace_file(source, target)
    verify_promoted()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="copy the selected artifacts before verifying the runtime bundle",
    )
    args = parser.parse_args(argv)
    if args.write:
        promote()
    else:
        verify_promoted()
    print("runtime promotion verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
