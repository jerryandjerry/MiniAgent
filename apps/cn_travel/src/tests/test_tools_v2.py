#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contract and payload tests for the five tool modules.

Contract checks validate backend-independent result shapes. Payload checks
validate that each result corresponds to the requested entity.
"""
import pytest

from cn_travel.business_logic.contracts import CONTRACTS, TOOL_NAMES, validate


# ------------------------------------------------------------- contract ----
def test_all_five_tools_are_declared():
    assert set(TOOL_NAMES) == {
        "search_travel_guide", "get_weather_info", "query_route",
        "recommend_hotels", "get_hotel_reviews",
    }


# tool name -> module that defines it (grouped by domain, as the repo has it)
TOOL_MODULE = {
    # Model-issued tool name -> (module, implementing function name)
    "search_travel_guide": ("cn_travel.tool.guide", "search_travel_guide"),
    "get_weather_info":    ("cn_travel.tool.get_weather", "get_weather_by_date"),
    "query_route":         ("cn_travel.tool.get_route", "query_routes"),
    "recommend_hotels":    ("cn_travel.tool.get_hotel", "get_hotel_recommendations"),
    "get_hotel_reviews":   ("cn_travel.tool.get_hotel", "get_hotel_reviews"),
}


def test_every_tool_is_exposed_by_its_module():
    """A tool is a function with its contract's name."""
    import importlib

    for tool, (module, fn) in TOOL_MODULE.items():
        mod = importlib.import_module(module)
        assert callable(getattr(mod, fn, None)), f"{module} must define {fn}() for tool {tool}"


def test_no_tool_calls_another_tool():
    """Tools are standalone: none may invoke another tool to do its job."""
    import importlib
    import inspect

    impl = {fn for _, fn in TOOL_MODULE.values()}
    for tool, (module, fn) in TOOL_MODULE.items():
        src = inspect.getsource(getattr(importlib.import_module(module), fn))
        for other in impl - {fn}:
            assert f"{other}(" not in src, f"{fn}() calls {other}()"


def test_validator_rejects_a_malformed_payload():
    with pytest.raises(AssertionError):
        validate("get_weather_info", {"status": "ok"})
    with pytest.raises(AssertionError):
        validate("recommend_hotels",
                 {"status": "ok", "source": "x", "location": "北京", "hotels": []})


# -------------------------------------------------------- get_weather_info -
@pytest.mark.amap
class TestWeather:
    def test_contract(self):
        from cn_travel.tool.get_weather import get_weather_by_date

        out = get_weather_by_date("嘉兴", "2026-08-10", 3)
        validate("get_weather_info", out)
        assert len(out["days"]) == 3

    def test_resolves_the_destination_not_a_default(self):
        """Weather resolution preserves the requested destination."""
        from cn_travel.tool.get_weather import get_weather_by_date

        jx = get_weather_by_date("嘉兴", "2026-08-10", 3)
        bj = get_weather_by_date("北京", "2026-08-10", 3)
        assert jx["resolved"]["adcode"] == "330400"
        assert bj["resolved"]["adcode"] == "110000"
        assert [d["temp_max_c"] for d in jx["days"]] != [d["temp_max_c"] for d in bj["days"]]

    def test_unresolvable_location_errors_instead_of_guessing(self):
        from cn_travel.tool.get_weather import get_weather_by_date

        out = get_weather_by_date("没有这个地方xyz", "2026-08-10", 2)
        validate("get_weather_info", out)
        assert out["status"] == "error"
        assert out["resolved"] is None and out["days"] == []

    def test_bad_date_errors(self):
        from cn_travel.tool.get_weather import get_weather_by_date

        out = get_weather_by_date("北京", "not-a-date", 2)
        validate("get_weather_info", out)
        assert out["status"] == "error"


# ------------------------------------------------------------ query_route --
@pytest.mark.amap
class TestRoute:
    def test_contract(self):
        from cn_travel.tool.get_route import query_routes

        out = query_routes("116.481028,39.989643", "天坛", city="北京")
        validate("query_route", out)

    def test_bare_landmark_stays_in_the_requested_city(self):
        """A city-scoped landmark resolves within the requested city."""
        from cn_travel.tool.get_route import query_routes

        out = query_routes("116.481028,39.989643", "天坛", city="北京")
        assert out["status"] == "ok"
        assert 116.3 < out["destination"]["lng"] < 116.5
        assert 39.8 < out["destination"]["lat"] < 40.0
        assert out["routes"]["driving"]["distance_m"] < 100_000

    def test_generic_facility_resolves_inside_the_city(self):
        from cn_travel.tool.get_route import query_routes

        out = query_routes("120.755,30.747", "火车站", city="嘉兴")
        assert out["status"] == "ok"
        assert "嘉兴" in out["destination"]["name"] or 120.5 < out["destination"]["lng"] < 121.1

    def test_unresolvable_destination_errors(self):
        from cn_travel.tool.get_route import query_routes

        out = query_routes("116.481028,39.989643", "zzz不存在的地方zzz", city="北京")
        validate("query_route", out)
        assert out["status"] == "error"


# ------------------------------------------------------- recommend_hotels --
@pytest.mark.amap
class TestHotels:
    def test_contract(self):
        from cn_travel.tool.get_hotel import get_hotel_recommendations

        out = get_hotel_recommendations("北京", "五星级", limit=2)
        validate("recommend_hotels", out)
        assert len(out["hotels"]) <= 2

    def test_hotels_exclude_canned_placeholders(self):
        from cn_travel.tool.get_hotel import get_hotel_recommendations

        names = {h["name"] for h in get_hotel_recommendations("北京", "", limit=5)["hotels"]}
        assert not (names & {"假日酒店", "商务精选酒店", "经济型连锁酒店"})
        assert all(n.strip() for n in names)

    def test_results_differ_between_cities(self):
        from cn_travel.tool.get_hotel import get_hotel_recommendations

        bj = {h["name"] for h in get_hotel_recommendations("北京", "", limit=5)["hotels"]}
        jx = {h["name"] for h in get_hotel_recommendations("嘉兴", "", limit=5)["hotels"]}
        assert bj != jx and not (bj & jx)

    def test_covers_small_cities(self):
        """94% of the RAG corpus is small cities; coverage there is the whole point."""
        from cn_travel.tool.get_hotel import get_hotel_recommendations

        out = get_hotel_recommendations("建阳", "经济型", limit=2)
        assert out["status"] == "ok" and out["hotels"]
        assert out["hotels"][0]["address"]

    def test_tier_keyword_is_honoured(self):
        from cn_travel.tool.get_hotel import get_hotel_recommendations

        lux = get_hotel_recommendations("北京", "五星级豪华", limit=5)["hotels"]
        assert any(h["tier"] == "豪华型" for h in lux)

    def test_unresolvable_city_errors(self):
        from cn_travel.tool.get_hotel import get_hotel_recommendations

        out = get_hotel_recommendations("没有这个地方xyz", "", limit=2)
        validate("recommend_hotels", out)
        assert out["status"] == "error"


# ---------------------------------------------------- search_travel_guide --
@pytest.mark.rag
class TestGuide:
    def test_contract_and_hit(self):
        from cn_travel.tool.guide import search_travel_guide

        out = search_travel_guide("嘉兴")
        validate("search_travel_guide", out)
        assert out["status"] == "ok"
        assert out["guides"][0]["city"] == "嘉兴"

    def test_fails_closed_on_out_of_corpus_city(self):
        """An out-of-corpus city cannot return another city's guide."""
        from cn_travel.tool.guide import search_travel_guide

        for city in ("平壤", "东京"):
            out = search_travel_guide(city)
            validate("search_travel_guide", out)
            assert out["status"] == "empty", f"{city} returned {out['guides'][:1]}"


# ------------------------------------------------------ get_hotel_reviews --
@pytest.mark.reviews
@pytest.mark.slow
class TestReviews:
    def test_contract(self):
        from cn_travel.tool.get_hotel import get_hotel_reviews

        out = get_hotel_reviews("北京国贸大酒店")
        validate("get_hotel_reviews", out)

    def test_returns_grounded_content_for_a_well_known_hotel(self):
        from cn_travel.tool.get_hotel import get_hotel_reviews

        out = get_hotel_reviews("北京国贸大酒店")
        assert out["status"] == "ok"
        assert out["rating"] and 1 <= out["rating"] <= 5
        assert out["reviews"] and len(out["reviews"][0]["text"]) > 15

    def test_reviews_exclude_canned_placeholder_text(self):
        from cn_travel.tool.get_hotel import get_hotel_reviews

        out = get_hotel_reviews("嘉兴道前宾馆", location="嘉兴")
        blob = out["summary"] + " ".join(r["text"] for r in out["reviews"])
        assert "商务设施齐全，会议室很专业" not in blob
        assert "酒店位置很好，房间干净整洁" not in blob


def test_missing_hotel_name_errors_without_backend_access(monkeypatch):
    from cn_travel.business_logic.contracts import validate
    from cn_travel.tool import get_hotel

    monkeypatch.setattr(
        get_hotel,
        "_reviews_via_search",
        lambda *_args, **_kwargs: pytest.fail("review backend was called"),
    )
    out = get_hotel.get_hotel_reviews("")
    validate("get_hotel_reviews", out)
    assert out["status"] == "error"


def test_unsupported_review_backend_fails_closed(monkeypatch):
    from cn_travel.business_logic.contracts import validate
    from cn_travel.tool import get_hotel

    monkeypatch.setenv("REVIEWS_BACKEND", "unsupported")
    monkeypatch.setattr(
        get_hotel,
        "_reviews_via_search",
        lambda *_args, **_kwargs: pytest.fail("review backend was called"),
    )
    out = get_hotel.get_hotel_reviews("Example Hotel")
    validate("get_hotel_reviews", out)
    assert out["status"] == "error"
    assert "unsupported reviews backend" in out["source"]
