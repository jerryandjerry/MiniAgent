"""Deterministic integrity manifests for packaged artifact directories."""
from __future__ import annotations

import hashlib
import json
import stat
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any


ALGORITHM = "sha256-tree-v1"


class ArtifactIntegrityError(ValueError):
    """Raised when an artifact directory differs from its sealed manifest."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(files: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        {
            "algorithm": ALGORITHM,
            "file_count": len(files),
            "total_bytes": sum(entry["size"] for entry in files),
            "files": files,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _scan_directory(directory: Path, hash_files: bool) -> list[dict[str, Any]]:
    directory = Path(directory)
    if not directory.is_dir() or directory.is_symlink():
        raise ArtifactIntegrityError(f"artifact directory is missing: {directory}")

    entries: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*"), key=lambda item: item.relative_to(directory).as_posix()):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink():
            raise ArtifactIntegrityError(f"artifact tree contains a symlink: {relative}")
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ArtifactIntegrityError(
                f"artifact tree contains a non-regular node: {relative}"
            )
        entry: dict[str, Any] = {"path": relative, "size": path.stat().st_size}
        if hash_files:
            entry["sha256"] = sha256_file(path)
        entries.append(entry)
    return entries


def build_directory_manifest(directory: Path) -> dict[str, Any]:
    """Hash every regular file beneath ``directory`` in canonical path order."""
    entries = _scan_directory(directory, hash_files=True)

    return {
        "algorithm": ALGORITHM,
        "tree_sha256": _tree_hash(entries),
        "file_count": len(entries),
        "total_bytes": sum(entry["size"] for entry in entries),
        "files": entries,
    }


def validate_directory_manifest(manifest: Any) -> list[str]:
    """Return schema and self-consistency errors for one directory manifest."""
    if not isinstance(manifest, dict):
        return ["manifest is not an object"]
    errors: list[str] = []
    if manifest.get("algorithm") != ALGORITHM:
        errors.append(f"algorithm must be {ALGORITHM}")
    files = manifest.get("files")
    if not isinstance(files, list):
        return errors + ["files must be an array"]

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            errors.append(f"files[{index}] is not an object")
            continue
        path = entry.get("path")
        size = entry.get("size")
        digest = entry.get("sha256")
        pure = PurePosixPath(path) if isinstance(path, str) else None
        if (
            pure is None
            or pure.is_absolute()
            or not pure.parts
            or "\\" in path
            or unicodedata.normalize("NFC", path) != path
            or any(part in {"", ".", ".."} for part in pure.parts)
            or pure.as_posix() != path
        ):
            errors.append(f"files[{index}] has an invalid path")
            continue
        if path in seen:
            errors.append(f"duplicate path: {path}")
        seen.add(path)
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            errors.append(f"{path}: invalid size")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            errors.append(f"{path}: invalid SHA-256")
        if isinstance(size, int) and not isinstance(size, bool) and size >= 0 and isinstance(digest, str):
            normalized.append({"path": path, "size": size, "sha256": digest})

    paths = [entry["path"] for entry in normalized]
    if paths != sorted(paths):
        errors.append("files are not in canonical path order")
    if manifest.get("file_count") != len(files):
        errors.append("file_count differs from files")
    expected_bytes = sum(entry["size"] for entry in normalized)
    if manifest.get("total_bytes") != expected_bytes:
        errors.append("total_bytes differs from files")
    if len(normalized) == len(files) and manifest.get("tree_sha256") != _tree_hash(normalized):
        errors.append("tree_sha256 differs from files")
    return errors


def directory_manifest_errors(
    directory: Path,
    expected: Any,
    label: str,
    verify_hashes: bool = True,
) -> tuple[list[str], dict[str, Any] | None]:
    """Compare a directory with its sealed manifest and return useful evidence."""
    schema_errors = validate_directory_manifest(expected)
    if schema_errors:
        return ([f"{label}: {error}" for error in schema_errors], None)
    try:
        if verify_hashes:
            actual = build_directory_manifest(directory)
        else:
            entries = _scan_directory(directory, hash_files=False)
            actual = {
                "algorithm": ALGORITHM,
                "tree_sha256": None,
                "file_count": len(entries),
                "total_bytes": sum(entry["size"] for entry in entries),
                "files": entries,
            }
    except ArtifactIntegrityError as exc:
        return ([f"{label}: {exc}"], None)

    expected_files = {entry["path"]: entry for entry in expected["files"]}
    actual_files = {entry["path"]: entry for entry in actual["files"]}
    errors: list[str] = []
    for path in sorted(expected_files.keys() - actual_files.keys()):
        errors.append(f"{label}: missing {path}")
    for path in sorted(actual_files.keys() - expected_files.keys()):
        errors.append(f"{label}: unexpected {path}")
    for path in sorted(expected_files.keys() & actual_files.keys()):
        wanted = expected_files[path]
        found = actual_files[path]
        if wanted["size"] != found["size"]:
            errors.append(f"{label}: size differs for {path}")
        elif verify_hashes and wanted["sha256"] != found["sha256"]:
            errors.append(f"{label}: SHA-256 differs for {path}")
    if verify_hashes and not errors and actual["tree_sha256"] != expected["tree_sha256"]:
        errors.append(f"{label}: directory tree SHA-256 differs")
    return errors, actual


def verify_directory_manifest(
    directory: Path,
    expected: Any,
    label: str,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Return the actual manifest or raise for any mismatch."""
    errors, actual = directory_manifest_errors(
        directory, expected, label, verify_hashes=verify_hashes
    )
    if errors:
        raise ArtifactIntegrityError("\n".join(errors))
    assert actual is not None
    return actual
