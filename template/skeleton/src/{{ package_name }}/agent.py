"""Application-owned agent orchestration."""

from typing import Optional

from .business_logic import AgentRequest, AgentResponse
from .service import ApplicationService


class Agent:
    def __init__(
        self,
        service: Optional[ApplicationService] = None,
    ) -> None:
        self._service = service or ApplicationService()

    def respond(self, message: str, session_id: Optional[str] = None) -> AgentResponse:
        if not message.strip():
            raise ValueError("Message must contain text.")
        request = AgentRequest(message=message, session_id=session_id)
        return self._service.respond(request)
