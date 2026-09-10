"""Hash and verify application-owned files and directory bundles."""

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


APPLICATION_ROOT = Path(__file__).resolve().parents[2]
SHA256 = re.compile(r"^[a-f0-9]{64}$")


class ArtifactError(ValueError):
    """Raised when an artifact path, size, or digest differs from its record."""


def inside(path: Path, root: Path = APPLICATION_ROOT) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def require_regular_ancestry(path: Path, root: Path = APPLICATION_ROOT) -> None:
    root_path = root.absolute()
    path_value = path.absolute()
    try:
        relative = path_value.relative_to(root_path)
    except ValueError as exc:
        raise ArtifactError(f"Keep artifact paths inside {root_path}.") from exc
    cursor = root_path
    if cursor.is_symlink():
        raise ArtifactError(f"Artifact path ancestry cannot contain symlinks: {cursor}.")
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ArtifactError(
                f"Artifact path ancestry cannot contain symlinks: {cursor}."
            )


def resolve_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ArtifactError(f"Set {label} to an application-relative path.")
    path = APPLICATION_ROOT / value
    if not inside(path, APPLICATION_ROOT):
        raise ArtifactError(f"Keep {label} inside {APPLICATION_ROOT}.")
    require_regular_ancestry(path, APPLICATION_ROOT)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_files(path: Path) -> list[Path]:
    entries = sorted(path.rglob("*"))
    symlinks = [item for item in entries if item.is_symlink()]
    if symlinks:
        raise ArtifactError(f"Artifact bundles contain regular files only: {symlinks[0]}.")
    files = [item for item in entries if item.is_file()]
    if any(not inside(item) for item in files):
        raise ArtifactError(f"Keep every artifact file inside {APPLICATION_ROOT}.")
    return files


def sha256_path(path: Path) -> str:
    if path.is_symlink():
        raise ArtifactError(f"Artifact paths cannot be symbolic links: {path}.")
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        digest = hashlib.sha256()
        files = _artifact_files(path)
        if not files:
            raise ArtifactError(f"Artifact directory contains no files: {path}.")
        for item in files:
            relative = item.relative_to(path).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(bytes.fromhex(sha256_file(item)))
        return digest.hexdigest()
    raise ArtifactError(f"Create the declared artifact: {path}.")


def artifact_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in _artifact_files(path))
    raise ArtifactError(f"Create the declared artifact: {path}.")


def verify_reference(value: Any, label: str) -> Tuple[Path, str]:
    if not isinstance(value, dict):
        raise ArtifactError(f"Set {label} to an object containing path and sha256.")
    path = resolve_path(value.get("path"), f"{label}.path")
    expected = value.get("sha256")
    if not isinstance(expected, str) or SHA256.fullmatch(expected) is None:
        raise ArtifactError(f"Set {label}.sha256 to a lowercase SHA-256 value.")
    actual = sha256_path(path)
    if actual != expected:
        raise ArtifactError(
            f"{label} hash mismatch: expected {expected}, observed {actual}."
        )
    expected_bytes = value.get("bytes")
    if expected_bytes is not None and artifact_bytes(path) != expected_bytes:
        raise ArtifactError(f"{label} byte count differs from its manifest.")
    return path, actual


def verify_references(values: Iterable[Dict[str, Any]], label: str) -> None:
    for position, value in enumerate(values):
        verify_reference(value, f"{label}[{position}]")
