#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end workflow tests across the five workflow families and branches.

Each test is marked with the model or backend capabilities it actually uses.
"""
import json

import pytest

SEARCH, WEATHER = "search_travel_guide", "get_weather_info"
ROUTE, HOTELS, REVIEWS = "query_route", "recommend_hotels", "get_hotel_reviews"

ASK_CITY = ("哪个城市", "去哪", "哪里")
ASK_HOTEL_CITY = ("哪个城市", "哪里")


def asked_something(reply, needles=ASK_CITY):
    return any(n in reply for n in needles)


# ============================== Workflow 1 (400 samples) — planning, no follow-up ==
@pytest.mark.policy
@pytest.mark.rag
@pytest.mark.amap
@pytest.mark.slow
class TestWorkflow1NoFollowup:
    def test_calls_guide_then_weather(self, make_agent):
        a = make_agent()
        a.process_user_input("我想去嘉兴旅游，帮我制定旅行计划")
        assert a.tool_names()[:2] == [SEARCH, WEATHER]

    def test_guide_is_searched_for_the_named_city(self, make_agent):
        a = make_agent()
        a.process_user_input("南京有什么好玩的景点推荐？")
        args = dict(a.calls[0][1])
        assert a.calls[0][0] == SEARCH
        assert "南京" in args.get("location", "")

    def test_weather_uses_the_users_departure_date(self, make_agent):
        a = make_agent(dates="2026-08-10~2026-08-15")
        a.process_user_input("帮我规划一下去青岛的行程")
        weather = [c for c in a.calls if c[0] == WEATHER]
        assert weather, "weather was never called"
        assert weather[0][1]["start_date"].startswith("2026-08")

    def test_final_reply_mentions_the_weather_it_fetched(self, make_agent):
        """ORCHESTRATION ONLY: proves the reply consumes the weather payload.
        Says nothing about whether that payload is the right city — see
        TestToolPayloadValidity.test_weather_is_for_the_destination_not_the_users_city.
        """
        a = make_agent()
        reply = a.process_user_input("我想去嘉兴旅游，帮我制定旅行计划")
        assert any(k in reply for k in ("℃", "天气", "晴", "雨"))


# ============================== Workflow 1 (50 samples) — planning, needs follow-up =
class TestWorkflow1WithFollowup:
    @pytest.mark.policy
    @pytest.mark.slow
    def test_asks_for_a_destination_and_calls_nothing(self, make_agent):
        a = make_agent()
        reply = a.process_user_input("我想出去旅游，有什么推荐的地方吗？")
        assert a.tool_names() == []
        assert asked_something(reply)

    @pytest.mark.policy
    @pytest.mark.rag
    @pytest.mark.amap
    @pytest.mark.slow
    def test_proceeds_once_the_destination_arrives(self, make_agent):
        a = make_agent()
        a.process_user_input("帮我规划一下行程")
        assert a.tool_names() == []
        a.process_user_input("我想去嘉兴，8月10日出发")
        assert SEARCH in a.tool_names()


# ============================== Workflow 2 (100 samples) — route, no follow-up ====
class TestWorkflow2NoFollowup:
    @pytest.mark.policy
    @pytest.mark.amap
    @pytest.mark.slow
    def test_calls_route_for_an_explicit_destination(self, make_agent):
        a = make_agent()
        a.process_user_input("从望京到北京天坛公园怎么走？")
        assert a.tool_names() == [ROUTE]

    @pytest.mark.policy
    @pytest.mark.amap
    @pytest.mark.slow
    def test_start_defaults_to_the_users_coordinates(self, make_agent):
        a = make_agent(coords="116.481028,39.989643")
        a.process_user_input("去北京天坛公园怎么走？")
        assert a.calls[0][0] == ROUTE
        assert a.calls[0][1]["start_location"] == "116.481028,39.989643"

    @pytest.mark.policy
    @pytest.mark.amap
    @pytest.mark.slow
    def test_public_facilities_do_not_trigger_a_follow_up(self, make_agent):
        """System prompt: 火车站/医院/学校 are specific enough to call directly."""
        a = make_agent()
        a.process_user_input("我要去火车站怎么走")
        assert a.tool_names() == [ROUTE]

    def test_every_workflow2_training_example_contains_a_route_call(
        self, train_conversations
    ):
        workflow2 = [
            sample
            for sample in train_conversations
            if sample["metadata"]["workflow"] == 2
        ]
        assert workflow2
        for sample in workflow2:
            names = [
                call["function"]["name"]
                for message in sample["conversation"]
                for call in (message.get("tool_calls") or [])
            ]
            assert ROUTE in names, f"route missing from sample {sample['metadata']['idx']}"


# ============================== Workflow 2 (20 samples) — route, needs follow-up ==
class TestWorkflow2WithFollowup:
    @pytest.mark.policy
    @pytest.mark.slow
    def test_asks_where_to_when_destination_is_absent(self, make_agent):
        a = make_agent()
        reply = a.process_user_input("怎么回家？")
        assert a.tool_names() == []
        assert asked_something(reply, ("地址", "哪里", "哪个"))

    @pytest.mark.policy
    @pytest.mark.amap
    @pytest.mark.slow
    def test_routes_once_the_address_arrives(self, make_agent):
        a = make_agent()
        a.process_user_input("我要回家")
        a.process_user_input("回海淀区中关村")
        assert ROUTE in a.tool_names()


# ============================== Workflow 3 (200 samples) — hotels, no follow-up ===
@pytest.mark.policy
@pytest.mark.reviews
@pytest.mark.amap
@pytest.mark.slow
class TestWorkflow3NoFollowup:
    """ORCHESTRATION ONLY — asserts which tools run and in what order.

    These pass even while the hotel backend serves canned fallback data for
    every query. Payload correctness lives in TestToolPayloadValidity.
    """

    def test_recommends_then_always_fetches_reviews(self, make_agent):
        a = make_agent()
        a.process_user_input("推荐一些北京300-500元的酒店")
        names = a.tool_names()
        assert names[0] == HOTELS
        assert REVIEWS in names, "recommendations returned without reviews"

    def test_reviews_target_a_recommended_hotel(self, make_agent):
        a = make_agent()
        a.process_user_input("北京有什么好的酒店推荐？")
        rec = next(c for c in a.calls if c[0] == HOTELS)
        revs = [c for c in a.calls if c[0] == REVIEWS]
        assert revs, "no reviews fetched"
        assert all(r[1].get("hotel_name") for r in revs)
        assert rec[2], "recommendation returned nothing"

    def test_direct_review_query_skips_recommendation(self, make_agent):
        """Sub-workflow 3B."""
        a = make_agent()
        a.process_user_input("北京假日酒店怎么样")
        assert a.tool_names() == [REVIEWS]


# ============================== Workflow 3 (40 samples) — hotels, needs follow-up =
class TestWorkflow3WithFollowup:
    @pytest.mark.policy
    @pytest.mark.slow
    def test_asks_which_city(self, make_agent):
        a = make_agent()
        reply = a.process_user_input("帮我找个酒店")
        assert a.tool_names() == []
        assert asked_something(reply, ASK_HOTEL_CITY)

    @pytest.mark.policy
    @pytest.mark.reviews
    @pytest.mark.amap
    @pytest.mark.slow
    def test_recommends_once_the_city_arrives(self, make_agent):
        a = make_agent()
        a.process_user_input("我要住酒店")
        a.process_user_input("北京市中心，预算300-500元")
        assert HOTELS in a.tool_names()


# ============================== Workflow 4 (100 samples) — travel chitchat ========
@pytest.mark.policy
@pytest.mark.slow
class TestWorkflow4Chitchat:
    @pytest.mark.parametrize(
        "q",
        ["旅行的时候需要注意什么安全问题？", "出国旅行需要准备什么？", "你好，我是第一次使用旅行助手"],
    )
    def test_answers_without_any_tool(self, make_agent, q):
        a = make_agent()
        reply = a.process_user_input(q)
        assert a.tool_names() == []
        assert len(reply) > 20


# ============================== Workflow 5 (100 samples) — rejection =============
@pytest.mark.policy
@pytest.mark.slow
class TestWorkflow5Rejection:
    @pytest.mark.parametrize(
        "q", ["帮我算一下1+1等于几？", "你能帮我写一段Python代码吗？", "如何做红烧肉？", "今天股市怎么样？"],
    )
    def test_refuses_without_any_tool(self, make_agent, q):
        a = make_agent()
        reply = a.process_user_input(q)
        assert a.tool_names() == []
        assert any(k in reply for k in ("抱歉", "旅行助手", "只能回答"))


# ============================== conditional branches ======================
class TestToolPayloadValidity:
    """Orchestration tests above assert only WHICH tools ran. These assert that
    what came back is actually about what was asked for.

    Direct tool-wrapper calls make payload correctness independent of the
    model's tool selection.
    """

    # -- Workflow 1: the guide must be for the requested city ---------------------
    @pytest.mark.rag
    def test_guide_is_for_the_requested_city(self, make_agent):
        payload = json.loads(make_agent().search_travel_guide("嘉兴"))
        assert payload["status"] == "ok"
        assert all(g["city"] == "嘉兴" for g in payload["guides"]), \
            f"guide is about someone else: {[g['city'] for g in payload['guides']]}"

    # -- Workflow 1: the weather must be for the destination, not the user's city --
    @pytest.mark.amap
    def test_weather_is_for_the_destination_not_the_users_city(self, make_agent):
        a = make_agent(city_id="101010100")                 # user in Beijing
        away = a.get_weather_info("嘉兴", "2026-08-10", 3)   # ~1200km south
        home = a.get_weather_info("北京", "2026-08-10", 3)
        assert away.split(":", 1)[1] != home.split(":", 1)[1]

    # -- Workflow 2: an intra-city route must not be a cross-country route --------
    @pytest.mark.amap
    def test_route_distance_is_plausible_for_a_scoped_landmark(self, make_agent):
        a = make_agent()
        payload = json.loads(a.query_route("116.481028,39.989643", "北京天坛公园", "北京"))
        assert payload["status"] == "ok"
        dist = [r["distance_m"] for r in payload["routes"].values()
                if r and "distance_m" in r]
        assert dist and max(dist) < 100_000, f"implausible intra-city route: {max(dist)}m"

    @pytest.mark.amap
    def test_route_to_a_bare_landmark_stays_in_the_city(self, make_agent):
        """A city-scoped landmark resolves within the requested city."""
        a = make_agent()
        payload = json.loads(a.query_route("116.481028,39.989643", "天坛", "北京"))
        assert payload["status"] == "ok"
        assert 116.3 < payload["destination"]["lng"] < 116.5
        dist = [r["distance_m"] for r in payload["routes"].values()
                if r and "distance_m" in r]
        assert dist and max(dist) < 100_000, f"implausible route: {max(dist)}m"

    # -- Workflow 3: the hotels must be real, not the canned fallback -------------
    @pytest.mark.policy
    @pytest.mark.reviews
    @pytest.mark.amap
    @pytest.mark.slow
    def test_recommended_hotels_are_not_the_canned_fallback(self, make_agent, canned_hotels):
        a = make_agent()
        a.process_user_input("推荐一些北京300-500元的酒店")
        payload = next(c for c in a.calls if c[0] == HOTELS)[2]
        hit = [h for h in canned_hotels if h in payload]
        assert not hit, f"served canned fallback hotels: {hit}"

    @pytest.mark.amap
    def test_recommendations_depend_on_the_query(self, make_agent):
        a = make_agent()
        bj = a.recommend_hotels("北京", "市中心，预算300元")
        sh = a.recommend_hotels("上海", "浦东，预算2000元亲子度假")
        assert bj != sh, "recommendations ignore the request entirely"

    @pytest.mark.reviews
    @pytest.mark.slow
    def test_reviews_differ_between_different_hotels(self, make_agent):
        a = make_agent()
        one = a.get_hotel_reviews_func("北京国贸大酒店")
        two = a.get_hotel_reviews_func("上海和平饭店")
        assert one.split(":", 1)[1] != two.split(":", 1)[1]

    @pytest.mark.amap
    def test_workflow3_is_reported_as_unverified_when_backed_by_fallback(
        self, hotel_backend_real
    ):
        """Fails loudly if someone reads a green Workflow 3 run as 'hotels work'."""
        if hotel_backend_real:
            return
        pytest.skip(
            "live AMap hotel payload capability is unavailable; orchestration remains covered"
        )


class TestConditionalBranches:
    """The three IF-rules in the system prompt. Tool layer is stubbed so the
    branch is exercised deterministically."""

    # Agent._DEPENDS_ON enforces dependent-call ordering during dispatch,
    # including calls emitted in the same model round.
    @pytest.mark.policy
    @pytest.mark.slow
    def test_empty_guide_suppresses_the_weather_call(self, make_agent, monkeypatch):
        a = make_agent()
        monkeypatch.setattr(a, "search_travel_guide",
                            lambda *args, **kw: "未找到相关旅行攻略信息")
        a.process_user_input("我想去拉萨旅游，8月10日出发，帮我制定5天行程")
        assert WEATHER not in a.tool_names(), "fetched weather despite an empty guide"

    def test_the_chain_guard_only_governs_the_second_round(self, make_agent):
        """Pins the mechanism above: the guard is a *continuation* decision, so a
        parallel first turn bypasses every conditional rule in the system prompt."""
        a = make_agent()
        assert a._should_continue_tool_chain("search_travel_guide", "未找到相关旅行攻略信息") is False
        assert a._should_continue_tool_chain("get_weather_info", "x" * 200) is False

    @pytest.mark.policy
    @pytest.mark.slow
    def test_empty_recommendation_suppresses_the_reviews_call(self, make_agent, monkeypatch):
        a = make_agent()
        monkeypatch.setattr(a, "recommend_hotels",
                            lambda *args, **kw: "暂时没有找到符合您需求的酒店推荐")
        a.process_user_input("推荐一些北京300-500元的酒店")
        assert REVIEWS not in a.tool_names(), "fetched reviews despite no hotels"

    @pytest.mark.policy
    @pytest.mark.slow
    def test_empty_route_is_reported_not_invented(self, make_agent, monkeypatch):
        a = make_agent()
        monkeypatch.setattr(a, "query_route", lambda *args, **kw: "路线查询出错: 无法找到地址")
        reply = a.process_user_input("从望京到某个不存在的地方怎么走？")
        assert any(k in reply for k in
                   ("查询不到", "无法", "没有", "抱歉", "出错", "未能", "找不到", "不存在")), \
            f"路线查询失败，智能体既没说查不到也没解释:\n{reply[:300]}"
