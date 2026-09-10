"""Typed messages exchanged by the application runtime."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AgentRequest:
    message: str
    session_id: Optional[str] = None


@dataclass(frozen=True)
class AgentResponse:
    status: str
    message: str
    actions: List[Dict[str, Any]] = field(default_factory=list)
