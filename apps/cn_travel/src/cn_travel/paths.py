#!/usr/bin/env python3
"""Filesystem resources contained in the CN Travel deployment bundle."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
DEPLOY_ROOT = PACKAGE_ROOT.parent
APP_ROOT = PACKAGE_ROOT
ENV_FILE = Path(os.getenv("CN_TRAVEL_ENV_FILE", DEPLOY_ROOT / ".env")).expanduser()
MODELS_DIR = PACKAGE_ROOT / "model"
BASE_MODELS_DIR = MODELS_DIR / "policy" / "base"
GUIDE_TOOL_DIR = PACKAGE_ROOT / "tool" / "guide"
GUIDE_KNOWLEDGE_BASE_DIR = GUIDE_TOOL_DIR / "knowledge_base"
EMBEDDING_MODELS_DIR = GUIDE_KNOWLEDGE_BASE_DIR / "embedding"


@dataclass(frozen=True)
class AppPaths:
    """Read-only paths used by the installed online application."""

    name: str = "cn_travel"

    @property
    def root(self) -> Path:
        return PACKAGE_ROOT

    @property
    def deploy_root(self) -> Path:
        return DEPLOY_ROOT

    @property
    def env_file(self) -> Path:
        return ENV_FILE

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def deployment_manifest(self) -> Path:
        return self.root / "deployment_manifest.json"

    @property
    def business_logic(self) -> Path:
        return self.root / "business_logic"

    @property
    def system_prompt(self) -> Path:
        return self.business_logic / "system_prompt.md"

    @property
    def user_info(self) -> Path:
        return self.business_logic / "user_info.md"

    @property
    def tool_schemas(self) -> Path:
        return self.business_logic / "tool_schemas.json"

    @property
    def knowledge_base(self) -> Path:
        return GUIDE_KNOWLEDGE_BASE_DIR

    @property
    def milvus_db(self) -> Path:
        return self.knowledge_base / "milvus.db"

    @property
    def guides(self) -> Path:
        return self.knowledge_base / "travel_guides"

    @property
    def models(self) -> Path:
        return MODELS_DIR

    @property
    def policy_model(self) -> Path:
        return self.models / "policy"

    @property
    def base_models(self) -> Path:
        return BASE_MODELS_DIR

    @property
    def policy_adapter(self) -> Path:
        return self.policy_model / "adapter"

    @property
    def embedding_models(self) -> Path:
        return EMBEDDING_MODELS_DIR

    @property
    def runtime(self) -> Path:
        return self.deploy_root / ".runtime"


CN_TRAVEL = AppPaths()


def load_app_env() -> bool:
    """Load the deployment's optional environment file without overriding values."""
    from dotenv import load_dotenv

    return bool(load_dotenv(CN_TRAVEL.env_file, override=False))


def app(name: str = "cn_travel") -> AppPaths:
    """Return the CN Travel runtime resource contract."""
    if name != CN_TRAVEL.name:
        raise ValueError(f"unsupported app {name!r}; expected {CN_TRAVEL.name!r}")
    return CN_TRAVEL
