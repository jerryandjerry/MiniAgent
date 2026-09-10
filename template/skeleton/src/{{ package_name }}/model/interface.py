"""Application policy interface and promoted-policy loader."""

import importlib
import json
from pathlib import Path
from typing import Protocol

from ..business_logic import AgentRequest, AgentResponse


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class Policy(Protocol):
    def respond(self, request: AgentRequest) -> AgentResponse:
        """Choose the application's next response or action."""


class ScaffoldPolicy:
    def respond(self, request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            status="policy_compilation_ready",
            message=(
                "The runtime scaffold is ready. Compile BUSINESS_BRIEF.md into "
                "the application workflows, tools, policy, and selected release."
            ),
        )


def load_policy() -> Policy:
    manifest_path = PACKAGE_ROOT / "deployment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "ready":
        return ScaffoldPolicy()
    entrypoint = manifest["policy"]["runtime"]["entrypoint"]
    module_name, attribute_name = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name)
    policy = factory()
    if isinstance(policy, ScaffoldPolicy) or not callable(getattr(policy, "respond", None)):
        raise RuntimeError("The promoted runtime entrypoint must construct the selected policy.")
    return policy
