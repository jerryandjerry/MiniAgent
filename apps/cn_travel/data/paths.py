"""Filesystem paths owned by the CN Travel data pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parent
APPLICATION_ROOT = DATA_ROOT.parent
SOURCE_ROOT = APPLICATION_ROOT / "src"
RUNTIME_PACKAGE_ROOT = SOURCE_ROOT / "cn_travel"


@dataclass(frozen=True)
class DataPaths:
    """Inputs, outputs, and runtime contracts used to build CN Travel data."""

    @property
    def root(self) -> Path:
        return DATA_ROOT

    @property
    def application_root(self) -> Path:
        return APPLICATION_ROOT

    @property
    def source_root(self) -> Path:
        return SOURCE_ROOT

    @property
    def runtime_package(self) -> Path:
        return RUNTIME_PACKAGE_ROOT

    @property
    def env_file(self) -> Path:
        return SOURCE_ROOT / ".env"

    @property
    def governing_doc(self) -> Path:
        return APPLICATION_ROOT / "README_CN TRAVEL.md"

    @property
    def business_logic(self) -> Path:
        return RUNTIME_PACKAGE_ROOT / "business_logic"

    @property
    def system_prompt(self) -> Path:
        return self.business_logic / "system_prompt.md"

    @property
    def tool_schemas(self) -> Path:
        return self.business_logic / "tool_schemas.json"

    @property
    def service(self) -> Path:
        return RUNTIME_PACKAGE_ROOT / "service"

    @property
    def tool(self) -> Path:
        return RUNTIME_PACKAGE_ROOT / "tool"

    @property
    def knowledge_base(self) -> Path:
        return DATA_ROOT / "knowledge_base"

    @property
    def milvus_db(self) -> Path:
        return self.knowledge_base / "milvus.db"

    @property
    def guides(self) -> Path:
        return self.knowledge_base / "travel_guides"


PATHS = DataPaths()


def load_data_env() -> bool:
    """Load the application environment used by data-generation commands."""
    from dotenv import load_dotenv

    return bool(load_dotenv(PATHS.env_file, override=False))
