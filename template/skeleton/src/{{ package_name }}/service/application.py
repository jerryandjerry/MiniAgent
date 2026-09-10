"""Application service facade for policy and tool execution."""

from typing import Any, Mapping, Optional, Tuple

from ..business_logic import AgentRequest, AgentResponse
from ..model import Policy, load_policy
from .tools import ToolResult, ToolService


class ApplicationService:
    def __init__(
        self,
        policy: Optional[Policy] = None,
        tools: Optional[ToolService] = None,
    ) -> None:
        self._policy = policy
        self._tools = tools or ToolService()

    @property
    def tool_names(self) -> Tuple[str, ...]:
        return self._tools.names

    def execute_tool(
        self, name: str, arguments: Mapping[str, Any]
    ) -> ToolResult:
        return self._tools.dispatch(name, arguments)

    def respond(self, request: AgentRequest) -> AgentResponse:
        if self._policy is None:
            self._policy = load_policy()
        return self._policy.respond(request)
