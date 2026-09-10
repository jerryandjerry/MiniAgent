"""Filesystem locations owned or consumed by CN Travel training."""
from __future__ import annotations

import sys
from pathlib import Path


TRAIN_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = TRAIN_ROOT.parent
SRC_ROOT = APP_ROOT / "src"
DATA_ROOT = APP_ROOT / "data"
PACKAGE_ROOT = SRC_ROOT / "cn_travel"
BUSINESS_LOGIC_ROOT = PACKAGE_ROOT / "business_logic"
TOOL_SCHEMAS = BUSINESS_LOGIC_ROOT / "tool_schemas.json"
BASE_MODELS_DIR = TRAIN_ROOT / "base"
EMBEDDING_MODELS_DIR = TRAIN_ROOT / "embedding"


def install_import_paths() -> None:
    """Make the training helpers and installed-layout package importable."""

    for path in (APP_ROOT, SRC_ROOT, TRAIN_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def resolve_app_path(value: str | Path) -> Path:
    """Resolve an active configuration path from the application root."""

    path = Path(value)
    return path if path.is_absolute() else APP_ROOT / path
