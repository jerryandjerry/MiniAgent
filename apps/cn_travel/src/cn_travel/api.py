"""FastAPI transport for CN Travel application services."""
from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from cn_travel.agent import TravelAssistantFuncCall
from cn_travel.deployment import deployment_status, policy_endpoint_status
from cn_travel.service.agent import AgentError
from cn_travel.service.guide import get_guide_service


class UserContext(BaseModel):
    name: str = "旅行者"
    city_id: str = "101010100"
    city_name: str = "北京"
    travel_date_range: str = ""
    start_coordinates: str = "116.481028,39.989643"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None
    user: UserContext | None = None
    reset: bool = False


class ChatResponse(BaseModel):
    session_id: str
    response: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=100)
    search_type: str = "hybrid"
    vector_weight: float = 1.0
    keyword_weight: float = 1.5


class LocationSearchRequest(BaseModel):
    province: str | None = None
    city: str | None = None
    limit: int = Field(default=10, ge=1, le=100)


@dataclass
class _Session:
    agent: Any
    user: UserContext
    lock: threading.RLock = field(default_factory=threading.RLock)


class SessionStore:
    """Process-local conversation sessions with per-session serialization."""

    def __init__(self, agent_factory: Callable[..., Any]):
        self._agent_factory = agent_factory
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    def acquire(
        self,
        session_id: str | None,
        user: UserContext | None,
        *,
        reset: bool,
    ) -> tuple[str, _Session]:
        key = session_id or uuid.uuid4().hex
        with self._lock:
            if reset:
                self._sessions.pop(key, None)
            session = self._sessions.get(key)
            if session is None:
                user = user or UserContext()
                session = _Session(
                    agent=self._agent_factory(
                        user_name=user.name,
                        user_city_id=user.city_id,
                        user_city_name=user.city_name,
                        travel_date_range=user.travel_date_range,
                        start_coordinates=user.start_coordinates,
                        verbose=False,
                    ),
                    user=user,
                )
                self._sessions[key] = session
            elif user is not None and session.user != user:
                raise ValueError("user context differs from the existing session")
        return key, session


def create_app(
    *,
    agent_factory: Callable[..., Any] = TravelAssistantFuncCall,
    guide_service_factory: Callable[[], Any] = get_guide_service,
    policy_probe: Callable[[], dict[str, Any]] = policy_endpoint_status,
) -> FastAPI:
    application = FastAPI(title="CN Travel", version="0.1.0")
    application.state.sessions = SessionStore(agent_factory)
    application.state.guide_service_factory = guide_service_factory
    application.state.policy_probe = policy_probe

    @application.post("/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest) -> ChatResponse:
        try:
            session_id, session = application.state.sessions.acquire(
                payload.session_id,
                payload.user,
                reset=payload.reset,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        try:
            with session.lock:
                response = session.agent.process_user_input(payload.message)
        except AgentError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return ChatResponse(session_id=session_id, response=response)

    @application.get("/health")
    def health() -> dict[str, Any]:
        try:
            service = application.state.guide_service_factory()
            count = service.collection_size
            return {
                "status": "healthy",
                "service": "CN Travel API",
                "collection_size": count,
            }
        except Exception as exc:
            return JSONResponse(
                status_code=500,
                content={"status": "unhealthy", "error": str(exc)},
            )

    @application.get("/ready")
    def ready() -> JSONResponse:
        verify = os.getenv("CN_TRAVEL_VERIFY_DEPLOYMENT_HASHES", "1") != "0"
        status = deployment_status(verify_hashes=verify)
        policy = application.state.policy_probe()
        status["policy_endpoint"] = policy
        if not policy.get("ready"):
            status["errors"].append(
                f"policy endpoint: {policy.get('error', 'configured model unavailable')}"
            )
            status["ready"] = False
        try:
            service = application.state.guide_service_factory()
            status["collection_size"] = service.collection_size
            if status["collection_size"] < 1:
                status["errors"].append("knowledge index is empty")
                status["ready"] = False
            expected = status.get("expected_collection_size")
            if expected is not None and status["collection_size"] != expected:
                status["errors"].append(
                    "knowledge index contains "
                    f"{status['collection_size']} entities; expected {expected}"
                )
                status["ready"] = False
        except Exception as exc:
            status["errors"].append(f"knowledge index: {exc}")
            status["ready"] = False
        return JSONResponse(status_code=200 if status["ready"] else 503, content=status)

    @application.post("/search")
    def search(payload: SearchRequest) -> dict[str, Any]:
        service = application.state.guide_service_factory()
        results = service.search(
            payload.query,
            payload.limit,
            payload.search_type,
            payload.vector_weight,
            payload.keyword_weight,
        )
        first = results[0] if results else {}
        return {
            "status": "success",
            "query": payload.query,
            "search_type": payload.search_type,
            "search_strategy": first.get("search_strategy", "hybrid"),
            "matched_locations": first.get(
                "matched_locations", {"provinces": [], "cities": []}
            ),
            "weights": {
                "vector_weight": payload.vector_weight,
                "keyword_weight": payload.keyword_weight,
            },
            "limit": payload.limit,
            "results_count": len(results),
            "results": results,
        }

    @application.post("/search_by_location")
    def search_by_location(payload: LocationSearchRequest) -> dict[str, Any]:
        if not payload.province and not payload.city:
            raise HTTPException(status_code=400, detail="province or city is required")
        service = application.state.guide_service_factory()
        results = service.search_by_location(
            payload.province,
            payload.city,
            payload.limit,
        )
        return {
            "status": "success",
            "filters": {"province": payload.province, "city": payload.city},
            "limit": payload.limit,
            "results_count": len(results),
            "results": results,
        }

    @application.get("/stats")
    def stats() -> dict[str, Any]:
        service = application.state.guide_service_factory()
        return {"status": "success", **service.statistics()}

    @application.exception_handler(404)
    async def not_found(_: Request, __: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={
                "error": "endpoint does not exist",
                "status": "error",
                "available_endpoints": [
                    "POST /chat",
                    "GET /health",
                    "GET /ready",
                    "POST /search",
                    "POST /search_by_location",
                    "GET /stats",
                ],
            },
        )

    return application


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "cn_travel.api:app",
        host=os.getenv("CN_TRAVEL_HOST", "0.0.0.0"),
        port=int(os.getenv("CN_TRAVEL_PORT", "8010")),
    )
