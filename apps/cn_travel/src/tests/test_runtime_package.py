"""Tests for the independently deployable CN Travel runtime."""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cn_travel import api
from cn_travel import deployment
from cn_travel import policy_server
from cn_travel.agent import TravelAssistantFuncCall
from cn_travel.paths import CN_TRAVEL
from cn_travel.service import agent as agent_service


class _FakeAgent:
    def __init__(self, **kwargs):
        self.turns: list[str] = []

    def process_user_input(self, message: str) -> str:
        self.turns.append(message)
        return f"{len(self.turns)}:{message}"


class _FakeGuideService:
    collection_size = 2

    def search(self, query, limit, search_type, vector_weight, keyword_weight):
        return [
            {
                "city_name": "嘉兴",
                "province_name": "浙江",
                "content": "guide",
                "search_strategy": "location_priority",
                "matched_locations": {"provinces": [], "cities": ["嘉兴"]},
            }
        ][:limit]

    def search_by_location(self, province, city, limit):
        return [{"province_name": province or "浙江", "city_name": city or "嘉兴"}]

    def statistics(self):
        return {
            "collection_name": "travel_guides",
            "total_travel_guides": self.collection_size,
        }


def _client(monkeypatch) -> TestClient:
    monkeypatch.setattr(
        api,
        "deployment_status",
        lambda verify_hashes=True: {"ready": True, "errors": [], "artifacts": {}},
    )
    return TestClient(
        api.create_app(
            agent_factory=_FakeAgent,
            guide_service_factory=_FakeGuideService,
            policy_probe=lambda: {
                "ready": True,
                "model": "cn-travel-policy-cee7c1bfa881",
                "served_models": ["cn-travel-policy-cee7c1bfa881"],
            },
        )
    )


def test_package_owns_runtime_resources():
    assert CN_TRAVEL.config.is_file()
    assert CN_TRAVEL.system_prompt.is_file()
    assert CN_TRAVEL.user_info.is_file()
    assert CN_TRAVEL.tool_schemas.is_file()
    assert CN_TRAVEL.deployment_manifest.is_file()
    assert CN_TRAVEL.milvus_db.is_file()
    assert CN_TRAVEL.policy_model.joinpath("base", "config.json").is_file()
    assert CN_TRAVEL.policy_adapter.joinpath("adapter_config.json").is_file()
    assert CN_TRAVEL.embedding_models.joinpath(
        "embeddinggemma-300m", "modules.json"
    ).is_file()


def test_default_agent_targets_the_manifest_policy(monkeypatch):
    sentinel = object()
    seen = []

    def make_client(model_name=None):
        seen.append(model_name)
        return sentinel, "cn-travel-policy-cee7c1bfa881"

    monkeypatch.setattr(agent_service, "make_client", make_client)
    assistant = TravelAssistantFuncCall(verbose=False)
    assert seen == [None]
    assert assistant.client is sentinel
    assert assistant.model_name == "cn-travel-policy-cee7c1bfa881"


def test_policy_launcher_exposes_environment_binaries(monkeypatch):
    seen = {}
    monkeypatch.setattr(policy_server.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(
        policy_server,
        "policy_server_command",
        lambda _extra_args=(): [sys.executable, "-m", "vllm"],
    )

    def call(command, *, env):
        seen["command"] = command
        seen["env"] = env
        return 0

    monkeypatch.setattr(policy_server.subprocess, "call", call)
    with pytest.raises(SystemExit) as stopped:
        policy_server.main()

    assert stopped.value.code == 0
    assert seen["command"] == [sys.executable, "-m", "vllm"]
    assert seen["env"]["PATH"].split(os.pathsep)[0] == str(
        Path(sys.executable).parent
    )


def _model_response(root: str) -> io.BytesIO:
    payload = {
        "data": [
            {
                "id": "cn-travel-policy-cee7c1bfa881",
                "parent": "cn-travel-base",
                "root": root,
            }
        ]
    }
    return io.BytesIO(json.dumps(payload).encode())


def test_remote_policy_identity_uses_manifest_model_id(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://policy.example/v1")
    monkeypatch.setenv("LLM_MODEL", "cn-travel-policy-cee7c1bfa881")
    monkeypatch.setattr(
        deployment,
        "urlopen",
        lambda _request, timeout: _model_response("/remote/runtime/policy/adapter"),
    )

    status = deployment.policy_endpoint_status()

    assert status["ready"] is True
    assert status["identity_mode"] == "manifest_model_id"


def test_local_policy_identity_rejects_a_different_adapter(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("LLM_MODEL", "cn-travel-policy-cee7c1bfa881")
    monkeypatch.setattr(
        deployment,
        "urlopen",
        lambda _request, timeout: _model_response(str(tmp_path)),
    )

    status = deployment.policy_endpoint_status()

    assert status["ready"] is False
    assert status["identity_mode"] is None


def test_chat_preserves_follow_up_state(monkeypatch):
    client = _client(monkeypatch)
    first = client.post("/chat", json={"message": "first"})
    assert first.status_code == 200
    session_id = first.json()["session_id"]
    assert first.json()["response"] == "1:first"

    second = client.post(
        "/chat",
        json={"session_id": session_id, "message": "second"},
    )
    assert second.status_code == 200
    assert second.json() == {"session_id": session_id, "response": "2:second"}


def test_session_user_context_is_stable(monkeypatch):
    client = _client(monkeypatch)
    session_id = client.post("/chat", json={"message": "first"}).json()["session_id"]
    response = client.post(
        "/chat",
        json={
            "session_id": session_id,
            "message": "second",
            "user": {"city_name": "上海"},
        },
    )
    assert response.status_code == 409


def test_follow_up_can_reuse_custom_user_context(monkeypatch):
    client = _client(monkeypatch)
    first = client.post(
        "/chat",
        json={"message": "first", "user": {"city_name": "上海"}},
    )
    session_id = first.json()["session_id"]
    response = client.post(
        "/chat",
        json={"session_id": session_id, "message": "second"},
    )
    assert response.status_code == 200
    assert response.json()["response"] == "2:second"


def test_health_readiness_and_rag_compatibility(monkeypatch):
    client = _client(monkeypatch)
    assert client.get("/health").json()["collection_size"] == 2
    readiness = client.get("/ready")
    assert readiness.status_code == 200
    assert (
        readiness.json()["policy_endpoint"]["model"]
        == "cn-travel-policy-cee7c1bfa881"
    )

    search = client.post("/search", json={"query": "嘉兴", "limit": 3})
    assert search.status_code == 200
    assert search.json()["results"][0]["city_name"] == "嘉兴"

    by_city = client.post("/search_by_location", json={"city": "嘉兴"})
    assert by_city.status_code == 200
    assert by_city.json()["results"][0]["city_name"] == "嘉兴"

    stats = client.get("/stats")
    assert stats.status_code == 200
    assert stats.json()["total_travel_guides"] == 2

    missing = client.get("/missing")
    assert missing.status_code == 404
    assert "available_endpoints" in missing.json()
