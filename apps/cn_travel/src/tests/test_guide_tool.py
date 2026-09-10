#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime boundary checks for travel-guide retrieval.

In-corpus queries return guides for the requested city. Out-of-corpus queries
return ``empty`` so the agent reports that coverage is unavailable.
"""
import pytest

pytestmark = pytest.mark.rag


IN_CORPUS = ["北京", "上海", "嘉兴", "哈尔滨", "南京", "青岛"]
OUT_OF_CORPUS = ["平壤", "东京", "首尔", "曼谷", "纽约", "新加坡"]


# ------------------------------------------------------------------ Tool layer
@pytest.mark.parametrize("city", IN_CORPUS)
def test_in_corpus_city_returns_its_own_guide(city):
    from cn_travel.tool.guide import search_travel_guide

    r = search_travel_guide(city)
    assert r["status"] == "ok", f"{city} 在语料里，却返回 {r['status']}"
    assert r["guides"], "status=ok 必须带内容"
    got = [g.get("city") for g in r["guides"]]
    assert any(city in (g or "") for g in got), f"查 {city} 拿回来的是 {got}"


@pytest.mark.parametrize("city", OUT_OF_CORPUS)
def test_out_of_corpus_city_returns_empty_not_another_city(city):
    from cn_travel.tool.guide import search_travel_guide

    r = search_travel_guide(city)
    assert r["status"] == "empty", (
        f"{city} 不在语料里，应当 empty，实际 {r['status']}，"
        f"内容来自 {[g.get('city') for g in (r.get('guides') or [])]}")
    assert not r["guides"], "非 ok 的载荷必须是空的"


# ------------------------------------------------------------------ Agent layer
@pytest.mark.policy
@pytest.mark.parametrize("city", OUT_OF_CORPUS[:2])
def test_agent_declines_instead_of_inventing_an_itinerary(city):
    from cn_travel.agent import TravelAssistantFuncCall

    a = TravelAssistantFuncCall(user_name="测试", user_city_name="北京", verbose=False)
    reply = a.process_user_input(f"我想去{city}旅游，帮我制定旅行计划")

    admits = any(k in reply for k in
                 ("没有", "未找到", "查不到", "暂无", "无法", "抱歉", "不在", "没能"))
    assert admits, f"{city} 无攻略，智能体却没说查不到:\n{reply[:400]}"

    # Treat a structured day-by-day itinerary as fabrication
    fabricated = sum(k in reply for k in ("第1天", "第一天", "Day 1", "行程安排", "上午", "下午"))
    assert fabricated < 2, f"{city} 无攻略，智能体却编出了行程:\n{reply[:400]}"


@pytest.mark.policy
@pytest.mark.amap
def test_agent_still_plans_normally_for_a_city_it_has(caplog):
    from cn_travel.agent import TravelAssistantFuncCall

    a = TravelAssistantFuncCall(user_name="测试", user_city_name="北京", verbose=False)
    reply = a.process_user_input("我想去嘉兴旅游，帮我制定旅行计划")
    assert "嘉兴" in reply
    assert len(reply) > 200, f"嘉兴有攻略，回复却过短:\n{reply}"


# ------------------------------------------------------ Colloquial, non-administrative names
@pytest.mark.amap
@pytest.mark.parametrize("query,expect_city", [
    ("无锡新区", "无锡"),
    ("上海浦东", "上海"),
    ("北京市中心", "北京"),
])
def test_colloquial_area_names_resolve_to_their_city(query, expect_city):
    from cn_travel.tool import _amap as amap

    r = amap.resolve_city(query)
    assert r is not None, f"{query} 未能解析"
    assert expect_city in r["name"], f"{query} 解析成了 {r['name']}"


@pytest.mark.amap
def test_geocode_fallback_refuses_to_guess_a_city():
    from cn_travel.tool import _amap as amap

    assert amap.resolve_city("zzz这个地方不存在") is None
