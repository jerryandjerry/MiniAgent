"""Filesystem paths owned by the CN Travel evaluator."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


EVAL_ROOT = Path(__file__).resolve().parent
APPLICATION_ROOT = EVAL_ROOT.parent
DATA_ROOT = APPLICATION_ROOT / "data"
RUNTIME_PACKAGE_ROOT = APPLICATION_ROOT / "src" / "cn_travel"


@dataclass(frozen=True)
class EvaluationPaths:
    """Sealed inputs, evaluator outputs, and runtime contracts."""

    @property
    def root(self) -> Path:
        return EVAL_ROOT

    @property
    def application_root(self) -> Path:
        return APPLICATION_ROOT

    @property
    def data(self) -> Path:
        return DATA_ROOT

    @property
    def eval_set(self) -> Path:
        return DATA_ROOT / "6_multiturn" / "leaderboard_eval.json"

    @property
    def runs(self) -> Path:
        return EVAL_ROOT / "runs"

    @property
    def tool_schemas(self) -> Path:
        return RUNTIME_PACKAGE_ROOT / "business_logic" / "tool_schemas.json"

    @property
    def materializer(self) -> Path:
        return DATA_ROOT / "scripts" / "2_materialize.py"

    @property
    def execution_cache(self) -> Path:
        return DATA_ROOT / "2_materialized" / "exec_cache"


PATHS = EvaluationPaths()
