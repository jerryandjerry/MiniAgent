#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared fixtures and capability detection for all application test suites.

Tests are grouped by what external resource they need, so the suite degrades
gracefully instead of erroring when a dependency is absent:

    no marker  -> pure, offline, always runs
    rag        -> needs the packaged vector index and embedding model
    amap       -> needs network access to the AMap API
    live       -> needs a working DASHSCOPE_API_KEY (costs money)
    reviews    -> needs explicit live hotel-review integration opt-in
    model      -> needs torch + the complete local Qwen3.5-0.8B checkpoint
    policy     -> needs the configured policy inference endpoint
"""
import json
import os
import re
from pathlib import Path

import pytest

from cn_travel.paths import CN_TRAVEL

APPLICATION_ROOT = Path(__file__).resolve().parent
DATA_ROOT = APPLICATION_ROOT / "data"
TRAIN_ROOT = APPLICATION_ROOT / "train"

ROOT = APPLICATION_ROOT
DATA_MODEL_ROOT = TRAIN_ROOT / "base"

# --------------------------------------------------------------------------
# capability probes (each runs once per session)
# --------------------------------------------------------------------------
def _has_key() -> bool:
    try:
        from dotenv import load_dotenv

        load_dotenv(CN_TRAVEL.env_file)
    except Exception:
        pass
    return bool(os.getenv("DASHSCOPE_API_KEY"))


def _has_rag() -> bool:
    try:
        import pymilvus  # noqa: F401
        import sentence_transformers  # noqa: F401
    except Exception:
        return False
    return CN_TRAVEL.milvus_db.is_file() and CN_TRAVEL.embedding_models.is_dir()


def _has_net() -> bool:
    try:
        from dotenv import load_dotenv

        load_dotenv(CN_TRAVEL.env_file, override=False)
    except Exception:
        pass
    key = os.getenv("AMAP_KEY", "").strip()
    if not key:
        return False
    try:
        import requests

        response = requests.get(
            "https://restapi.amap.com/v3/geocode/geo",
            params={"address": "北京", "city": "北京", "key": key},
            timeout=5,
        )
        payload = response.json()
        return response.status_code == 200 and payload.get("status") == "1"
    except Exception:
        return False


def _has_model() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return False

    model = DATA_MODEL_ROOT / "Qwen3.5-0.8B"
    required = (model / "config.json", model / "tokenizer_config.json")
    if not all(path.is_file() for path in required):
        return False

    index_path = model / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shards = set(index["weight_map"].values())
        except (KeyError, TypeError, ValueError):
            return False
        return bool(shards) and all((model / shard).is_file() for shard in shards)

    return any(model.glob("*.safetensors")) or any(model.glob("pytorch_model*.bin"))


def _has_policy() -> bool:
    try:
        from cn_travel.deployment import policy_endpoint_status
    except Exception:
        return False
    return bool(policy_endpoint_status(timeout=3).get("ready"))


def _has_reviews() -> bool:
    return os.getenv("CN_TRAVEL_TEST_REVIEWS", "").lower() in {"1", "true", "yes"}


CANNED_HOTELS = {"假日酒店", "商务精选酒店", "经济型连锁酒店"}


def _hotel_backend_is_real() -> bool:
    """Return whether recommendations are query-dependent, non-placeholder listings."""
    try:
        from cn_travel.tool.get_hotel import get_hotel_recommendations

        a = get_hotel_recommendations("北京", "五星级", limit=3)
        b = get_hotel_recommendations("嘉兴", "经济型", limit=3)
        if a.get("status") != "ok" or b.get("status") != "ok":
            return False
        names_a = {h["name"] for h in a["hotels"]}
        names_b = {h["name"] for h in b["hotels"]}
        if names_a & CANNED_HOTELS or names_b & CANNED_HOTELS:
            return False
        return bool(names_a) and names_a != names_b
    except Exception:
        return False


CAPS = {}


def _require_capabilities() -> None:
    requested = {
        name.strip()
        for name in os.getenv("CN_TRAVEL_REQUIRED_CAPABILITIES", "").split(",")
        if name.strip()
    }
    unknown = requested - CAPS.keys()
    if unknown:
        raise pytest.UsageError(
            "unknown required capabilities: " + ", ".join(sorted(unknown))
        )
    missing = sorted(name for name in requested if not CAPS.get(name))
    if missing:
        raise pytest.UsageError(
            "required capabilities unavailable: " + ", ".join(missing)
        )


def _offline_selection(config) -> bool:
    """Return whether pytest selected the app's fully offline capability set.

    The canonical ``make test-offline`` expression excludes every capability
    whose discovery can touch a backend or external network.  Detect that
    selection before probing capabilities so collection itself stays offline.
    """
    markexpr = config.getoption("markexpr") or ""
    excluded = {
        marker
        for marker in ("live", "model", "policy", "reviews", "rag", "amap")
        if re.search(rf"\bnot\s+{marker}\b", markexpr)
    }
    return excluded == {"live", "model", "policy", "reviews", "rag", "amap"}


def pytest_configure(config):
    config.addinivalue_line("markers", "rag: loads the packaged vector index and embeddings")
    config.addinivalue_line("markers", "amap: needs network access to AMap")
    config.addinivalue_line("markers", "live: needs DASHSCOPE_API_KEY (costs money)")
    config.addinivalue_line(
        "markers", "reviews: explicitly enables the live hotel-review integration"
    )
    config.addinivalue_line(
        "markers", "model: needs torch and the complete local Qwen checkpoint"
    )
    config.addinivalue_line(
        "markers", "policy: needs the configured policy inference endpoint"
    )
    config.addinivalue_line("markers", "slow: takes >10s")
    if _offline_selection(config):
        CAPS.update(
            live=False,
            rag=False,
            amap=False,
            model=False,
            policy=False,
            reviews=False,
            hotel_backend_real=False,
            offline=True,
        )
        _require_capabilities()
        return

    CAPS["offline"] = False
    CAPS["live"] = _has_key()
    CAPS["rag"] = _has_rag()
    CAPS["amap"] = _has_net()
    CAPS["model"] = _has_model()
    CAPS["policy"] = _has_policy()
    CAPS["reviews"] = _has_reviews()
    CAPS["hotel_backend_real"] = _hotel_backend_is_real()
    _require_capabilities()


STRICT = os.getenv("STRICT_DEFECTS", "").lower() in {"1", "true", "yes"}


def pytest_collection_modifyitems(config, items):
    reasons = {
        "live": "DASHSCOPE_API_KEY not set",
        "rag": "packaged RAG dependencies or artifacts unavailable",
        "amap": "no network access to AMap",
        "model": "torch or complete local Qwen checkpoint unavailable",
        "policy": "configured policy inference endpoint unavailable",
        "reviews": "live hotel-review integration is not enabled",
    }
    for item in items:
        for cap, reason in reasons.items():
            if cap in item.keywords and not CAPS.get(cap):
                item.add_marker(pytest.mark.skip(reason=reason))

        # STRICT_DEFECTS=1 removes xfail markers so every selected assertion is
        # reported directly as a pass or failure.
        if STRICT:
            item.own_markers = [m for m in item.own_markers if m.name != "xfail"]


def pytest_report_header(config):
    lines = ["capabilities: " + ", ".join(
        f"{k}={'yes' if v else 'NO'}" for k, v in sorted(CAPS.items())
    )]
    return lines


@pytest.fixture(scope="session")
def hotel_backend_real() -> bool:
    return CAPS.get("hotel_backend_real", False)


@pytest.fixture(scope="session")
def canned_hotels() -> set:
    return set(CANNED_HOTELS)


# --------------------------------------------------------------------------
# data fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def all_tools():
    return json.loads(CN_TRAVEL.tool_schemas.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def storyboard():
    return json.loads(
        (DATA_ROOT / "1_storyboard.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="session")
def train_conversations():
    return json.loads(
        (DATA_ROOT / "5_merged" / "train_validation.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture(scope="session")
def test_conversations():
    return json.loads(
        (DATA_ROOT / "5_merged" / "leaderboard_eval.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture(scope="session")
def multiturn_v2():
    return json.loads(
        (DATA_ROOT / "6_multiturn" / "train_validation.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture(scope="session")
def world_policy_episodes():
    path = DATA_ROOT / "7_world_policy" / "world_policy_episodes.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


@pytest.fixture(scope="session")
def synthetic_china_world():
    return json.loads(
        (DATA_ROOT / "7_world_policy" / "synthetic_china_world_v1.json").read_text(
            encoding="utf-8"
        )
    )


# --------------------------------------------------------------------------
# agent driver
# --------------------------------------------------------------------------
@pytest.fixture
def make_agent():
    """Build a TravelAssistantFuncCall that records every tool call it makes."""
    from cn_travel.agent import TravelAssistantFuncCall

    created = []

    def _make(city_id="101010100", coords="116.481028,39.989643",
              dates="2026-08-10~2026-08-15"):
        a = TravelAssistantFuncCall(
            user_name="测试用户", user_city_id=city_id,
            travel_date_range=dates, start_coordinates=coords,
        )
        a.calls = []          # list of (name, args, result)
        original = a.call_function

        def spy(name, args):
            result = original(name, args)
            a.calls.append((name, args, result))
            return result

        a.call_function = spy
        a.tool_names = lambda: [c[0] for c in a.calls]
        created.append(a)
        return a

    return _make
