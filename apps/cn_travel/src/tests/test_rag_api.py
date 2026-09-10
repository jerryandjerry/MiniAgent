"""Tests for the packaged RAG service and its FastAPI transport."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from cn_travel.api import create_app
from cn_travel.paths import CN_TRAVEL

pytestmark = pytest.mark.rag

IN_CORPUS = ["北京", "嘉兴", "南京", "青岛", "哈尔滨"]
OUT_OF_CORPUS = ["平壤", "东京", "首尔"]


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(
        create_app(
            policy_probe=lambda: {
                "ready": True,
                "model": "cn-travel-policy-cee7c1bfa881",
                "served_models": ["cn-travel-policy-cee7c1bfa881"],
            }
        )
    )


def search(client, query, mode="hybrid", limit=3, **kwargs):
    body = {"query": query, "search_type": mode, "limit": limit, **kwargs}
    response = client.post("/search", json=body)
    response.raise_for_status()
    return response.json()


def test_health(client):
    payload = client.get("/health").json()
    assert payload["status"] == "healthy"
    assert payload["collection_size"] > 0


def test_stats(client):
    payload = client.get("/stats").json()
    assert payload["status"] == "success"
    assert payload["collection_name"] == "travel_guides"
    assert payload["total_travel_guides"] > 0


def test_unknown_route_returns_404_with_help(client):
    response = client.get("/nope")
    assert response.status_code == 404
    assert "available_endpoints" in response.json()


def test_search_by_location(client):
    payload = client.post(
        "/search_by_location",
        json={"city": "北京", "limit": 3},
    ).json()
    assert payload["status"] == "success"
    assert any(item["city_name"] == "北京" for item in payload["results"])


@pytest.mark.parametrize("city", IN_CORPUS)
def test_hybrid_returns_the_right_city(client, city):
    payload = search(client, city, "hybrid")
    assert payload["results_count"] >= 1
    assert payload["results"][0]["city_name"] == city


@pytest.mark.parametrize("city", IN_CORPUS)
def test_vector_returns_the_right_city(client, city):
    payload = search(client, city, "vector")
    assert payload["results_count"] >= 1
    assert payload["results"][0]["city_name"] == city


def test_guide_content_is_substantive(client):
    payload = search(client, "嘉兴", "hybrid")
    assert len(payload["results"][0]["content"]) > 500


def test_hybrid_accepts_weight_overrides(client):
    payload = search(
        client,
        "北京",
        "hybrid",
        vector_weight=1.5,
        keyword_weight=0.7,
    )
    assert payload["status"] == "success"


def test_out_of_corpus_city_returns_no_guide(client):
    payload = search(client, OUT_OF_CORPUS[0], "hybrid")
    assert payload["results_count"] == 0


@pytest.mark.parametrize("city", OUT_OF_CORPUS)
def test_out_of_corpus_never_returns_a_different_city(client, city):
    payload = search(client, city, "hybrid")
    assert all(item["city_name"] == city for item in payload["results"])


def test_keyword_search_ranks_the_best_match_first(client):
    payload = search(client, "北京", "keyword", limit=5)
    assert payload["results"]
    assert payload["results"][0]["city_name"] == "北京"


def test_index_matches_the_deployment_manifest(client, monkeypatch):
    monkeypatch.setenv("CN_TRAVEL_VERIFY_DEPLOYMENT_HASHES", "0")
    manifest = json.loads(CN_TRAVEL.deployment_manifest.read_text(encoding="utf-8"))
    response = client.get("/ready")
    assert response.status_code == 200
    payload = response.json()
    assert payload["collection_size"] == manifest["retrieval"]["entity_count"]
