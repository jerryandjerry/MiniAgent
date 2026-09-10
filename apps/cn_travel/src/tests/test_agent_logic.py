#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline unit tests for the agent's own logic — no network, no API key.

Covers the pieces that decide *which* tools run, which is what the SFT data
ultimately encodes: the tool-chain heuristic, date parsing, the dispatch
table, and the tool schemas themselves.
"""
import json
import os

import pytest

from cn_travel.paths import CN_TRAVEL


@pytest.fixture(scope="module")
def all_tools():
    return json.loads(CN_TRAVEL.tool_schemas.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def agent():
    """A real agent instance; a dummy key lets us construct one offline."""
    os.environ.setdefault("DASHSCOPE_API_KEY", "sk-dummy-for-offline-tests")
    if not os.environ.get("DASHSCOPE_API_KEY"):
        os.environ["DASHSCOPE_API_KEY"] = "sk-dummy-for-offline-tests"
    from cn_travel.agent import TravelAssistantFuncCall

    return TravelAssistantFuncCall(
        user_name="测试", user_city_id="101010100",
        travel_date_range="2026-08-10~2026-08-15",
        start_coordinates="116.481028,39.989643",
    )


# ---------------------------------------------------------------- schemas --
def test_exposes_exactly_five_tools(agent):
    assert len(agent.tools) == 5


def test_tool_names_match_the_training_schema_file(agent, all_tools):
    assert {t["function"]["name"] for t in agent.tools} == {
        t["function"]["name"] for t in all_tools
    }


def test_every_tool_declares_required_params(agent):
    for t in agent.tools:
        fn = t["function"]
        assert t["type"] == "function"
        assert fn["description"]
        params = fn["parameters"]
        for req in params.get("required", []):
            assert req in params["properties"], f"{fn['name']}: {req} missing"


@pytest.mark.parametrize(
    "name,required",
    [
        ("search_travel_guide", ["location"]),
        ("get_weather_info", ["location", "start_date"]),
        ("query_route", ["start_location", "end_location", "city"]),
        ("recommend_hotels", ["location"]),
        ("get_hotel_reviews", ["hotel_name"]),
    ],
)
def test_tool_required_args_are_stable(agent, name, required):
    """The student was trained against these exact signatures."""
    fn = next(t["function"] for t in agent.tools if t["function"]["name"] == name)
    assert fn["parameters"]["required"] == required


# ------------------------------------------------- tool-chain conditionals --
def _payload(status):
    return json.dumps({"status": status, "source": "test"}, ensure_ascii=False)


# Workflow 1: search_travel_guide -> get_weather_info only when the guide came back ok
def test_chain_continues_after_a_substantive_guide(agent):
    assert agent._should_continue_tool_chain("search_travel_guide", _payload("ok")) is True


@pytest.mark.parametrize("status", ["empty", "error"])
def test_chain_stops_when_guide_is_empty_or_failed(agent, status):
    """Weather is queried only after a substantive guide result."""
    assert agent._should_continue_tool_chain("search_travel_guide", _payload(status)) is False


# Workflow 3: recommend_hotels -> get_hotel_reviews only when hotels came back
def test_chain_continues_after_hotels_are_found(agent):
    assert agent._should_continue_tool_chain("recommend_hotels", _payload("ok")) is True


@pytest.mark.parametrize("status", ["empty", "error"])
def test_chain_stops_when_no_hotels(agent, status):
    assert agent._should_continue_tool_chain("recommend_hotels", _payload(status)) is False


@pytest.mark.parametrize(
    "name", ["query_route", "get_weather_info", "get_hotel_reviews", "nonexistent_tool"],
)
def test_terminal_tools_never_chain(agent, name):
    assert agent._should_continue_tool_chain(name, _payload("ok")) is False


def test_chain_guard_is_not_fooled_by_prose(agent):
    """Tool chaining depends on typed results rather than prose substrings."""
    assert agent._should_continue_tool_chain("search_travel_guide", "攻略内容很长" * 20) is False


# --------------------------------------------------------- date handling --
def test_parses_a_date_range(agent):
    assert agent.user_info["travel_start_date"] == "2026-08-10"
    assert agent.user_info["travel_end_date"] == "2026-08-15"


def test_parses_a_single_date_as_both_ends():
    os.environ.setdefault("DASHSCOPE_API_KEY", "sk-dummy-for-offline-tests")
    from cn_travel.agent import TravelAssistantFuncCall

    a = TravelAssistantFuncCall(travel_date_range="2026-08-10")
    assert a.user_info["travel_start_date"] == "2026-08-10"
    assert a.user_info["travel_end_date"] == "2026-08-10"


# ------------------------------------------------------------- dispatch ---
def test_unknown_function_is_reported_not_raised(agent):
    assert "未知函数" in agent.call_function("no_such_tool", {})


@pytest.mark.parametrize(
    "name,args",
    [
        ("search_travel_guide", {"location": "北京"}),
        ("get_weather_info", {"location": "101010100", "start_date": "2026-08-10"}),
        ("query_route", {"start_location": "116.4,39.9", "end_location": "天坛"}),
        ("recommend_hotels", {"requirements": "北京"}),
        ("get_hotel_reviews", {"hotel_name": "假日酒店"}),
    ],
)
def test_every_declared_tool_is_dispatchable(agent, name, args, monkeypatch):
    """Each schema name must map to a real method — caught by returning a str."""
    sentinel = f"<{name} ok>"
    monkeypatch.setattr(agent, name if name != "get_hotel_reviews" else "get_hotel_reviews_func",
                        lambda *a, **k: sentinel)
    assert agent.call_function(name, args) == sentinel


# ------------------------------------------------------------- history ----
def test_history_is_capped_at_twenty_messages(agent):
    agent.conversation_history = []
    for i in range(30):
        agent.add_to_history("user", f"m{i}")
    assert len(agent.conversation_history) == 20
    assert agent.conversation_history[-1]["content"] == "m29"
