#!/usr/bin/env python3
"""Fetch base and embedding model assets into the training workspace."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import tempfile

TRAIN_ROOT = pathlib.Path(__file__).resolve().parent
BASE_MODELS_DIR = TRAIN_ROOT / "base"
EMBEDDING_MODELS_DIR = TRAIN_ROOT / "embedding"


MODEL_SOURCES = {
    "qwen": (
        "CN_TRAVEL_QWEN_REPO",
        "Qwen/Qwen3.5-0.8B",
        BASE_MODELS_DIR / "Qwen3.5-0.8B",
    ),
    "lfm": (
        "CN_TRAVEL_LFM_REPO",
        "LiquidAI/LFM2.5-350M",
        BASE_MODELS_DIR / "LFM2.5-350M",
    ),
    "embedding": (
        "CN_TRAVEL_EMBEDDING_REPO",
        "google/embeddinggemma-300m",
        EMBEDDING_MODELS_DIR / "embeddinggemma-300m",
    ),
}

PINNED_REVISIONS = {
    "qwen": "2fc06364715b967f1860aea9cf38778875588b17",
    "lfm": "9e6c6ccf47cd318696e137d381a7ded8fe4df09f",
    "embedding": "57c266a740f537b4dc058e1b0cda161fd15afa75",
}


LFM_DIRECTORY_MANIFEST = {
    "algorithm": "sha256-tree-v1",
    "tree_sha256": "36843b8b1dae9b5fb51b2435fa4f9593e1db5c537cfc1865838122554983347f",
    "file_count": 9,
    "total_bytes": 713751951,
    "files": [
        {
            "path": ".gitattributes",
            "size": 1519,
            "sha256": "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
        },
        {
            "path": "LICENSE",
            "size": 10574,
            "sha256": "4d28ca14dedc0b3d0fcc2b3339f0e79931faa33874f3d24f522183a8fc70068c",
        },
        {
            "path": "README.md",
            "size": 14508,
            "sha256": "a2bb9f40aca96e21d0e2c497af58f459a4b1f92c9d459dc0ece67ec63f7480b5",
        },
        {
            "path": "chat_template.jinja",
            "size": 5487,
            "sha256": "ba551d58630afa3190b1be3602e28301f3d2e9bbac978dfc49d6d825171648b6",
        },
        {
            "path": "config.json",
            "size": 1281,
            "sha256": "720b43d6ddc2ed25be23eed355aefcf342434a176dedad23dbe0a5e3ac24bbb8",
        },
        {
            "path": "generation_config.json",
            "size": 134,
            "sha256": "0ec7d228c4de7e3453e98bf6910c7eb7841e87fede63ea11c0f75a5c64e1c3c3",
        },
        {
            "path": "model.safetensors",
            "size": 708984464,
            "sha256": "1c9c77a4471a7f590f85240f74ed1fc26df7fbde88c3006724e2f93ca993ea4e",
        },
        {
            "path": "tokenizer.json",
            "size": 4733389,
            "sha256": "df1d8d5ec5d091b460562ffd545e4a5e91d17d4a0db7ebe733be34ed374377bd",
        },
        {
            "path": "tokenizer_config.json",
            "size": 595,
            "sha256": "3701f370c70034d06e28947ff51c1063983393319f429e3ce58f06a45fae67cc",
        },
    ],
}


def _deployment_directory_manifest(name: str) -> dict | None:
    manifest_path = TRAIN_ROOT.parent / "src" / "cn_travel" / "deployment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if name == "qwen":
        return manifest["policy"]["base_directory"]
    if name == "embedding":
        return manifest["retrieval"]["embedding_directory"]
    if name == "lfm":
        return LFM_DIRECTORY_MANIFEST
    return None


def _default_file_inventory(name: str) -> tuple[str, ...]:
    manifest = _deployment_directory_manifest(name)
    if manifest is not None:
        return tuple(entry["path"] for entry in manifest["files"])
    raise ValueError(f"unsupported model inventory: {name}")


def _install_snapshot(
    snapshot: pathlib.Path,
    destination: pathlib.Path,
    *,
    file_inventory: tuple[str, ...] | None = None,
    expected_manifest: dict | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as temporary:
        temporary_root = pathlib.Path(temporary)
        staged = temporary_root / destination.name
        if file_inventory is None:
            shutil.copytree(
                snapshot,
                staged,
                symlinks=False,
                ignore=shutil.ignore_patterns(".cache"),
            )
        else:
            staged.mkdir()
            for relative_name in sorted(set(file_inventory)):
                relative = pathlib.PurePosixPath(relative_name)
                if (
                    relative.is_absolute()
                    or not relative.parts
                    or any(part in {"", ".", ".."} for part in relative.parts)
                    or relative.as_posix() != relative_name
                ):
                    raise ValueError(f"invalid model inventory path: {relative_name}")
                source = snapshot.joinpath(*relative.parts)
                if not source.is_file():
                    raise FileNotFoundError(
                        f"model snapshot is missing required file: {relative_name}"
                    )
                target = staged.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

        if expected_manifest is not None:
            from cn_travel.artifact_integrity import verify_directory_manifest

            verify_directory_manifest(
                staged,
                expected_manifest,
                f"downloaded model {destination.name}",
            )
        previous = temporary_root / "previous"
        if destination.exists():
            destination.replace(previous)
        try:
            staged.replace(destination)
        except Exception:
            if previous.exists() and not destination.exists():
                previous.replace(destination)
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models",
        nargs="*",
        choices=tuple(MODEL_SOURCES),
        default=list(MODEL_SOURCES),
        help="model assets to fetch; the default fetches all three",
    )
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    for name in args.models:
        env_name, default_repo, destination = MODEL_SOURCES[name]
        repository_override = os.environ.get(env_name)
        repository = repository_override or default_repo
        revision_override = os.environ.get(f"CN_TRAVEL_{name.upper()}_REVISION")
        revision = revision_override or PINNED_REVISIONS.get(name)
        file_inventory = _default_file_inventory(name)
        expected_manifest = _deployment_directory_manifest(name)
        identity = f"@{revision}" if revision else ""
        print(f"fetching {repository}{identity} -> {destination}", flush=True)
        snapshot = pathlib.Path(snapshot_download(
            repo_id=repository,
            revision=revision,
            allow_patterns=list(file_inventory) if file_inventory else None,
        ))
        _install_snapshot(
            snapshot,
            destination,
            file_inventory=file_inventory,
            expected_manifest=expected_manifest,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
