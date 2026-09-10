#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functional tests for the weather, route, and hotel backends."""
import json
import os

import pytest
import requests

BEIJING = "116.481028,39.989643"   # Wangjing SOHO, the default start coordinate
AMAP_TIANTAN = (116.35, 116.50, 39.80, 39.95)   # Box around the Temple of Heaven, Beijing


# =========================================================== get_weather ===
@pytest.mark.amap
class TestWeather:
    def test_returns_one_row_per_requested_day(self):
        from cn_travel.tool.get_weather import get_weather_by_date_range

        rows = get_weather_by_date_range("101010100", "2026-08-10", 3)
        assert len(rows) == 3

    def test_rows_start_on_the_requested_date_and_are_consecutive(self):
        from cn_travel.tool.get_weather import get_weather_by_date_range

        rows = get_weather_by_date_range("101010100", "2026-08-10", 3)
        assert [r["日期"] for r in rows] == ["2026-08-10", "2026-08-11", "2026-08-12"]

    def test_every_row_has_the_fields_the_prompt_formats(self):
        from cn_travel.tool.get_weather import get_weather_by_date_range

        for r in get_weather_by_date_range("101010100", "2026-08-10", 2):
            for f in ("日期", "白天天气", "夜间天气", "最高温", "最低温"):
                assert r.get(f), f"missing {f}"
            assert r["最高温"].endswith("℃") and r["最低温"].endswith("℃")

    def test_accepts_raw_coordinates_as_well_as_a_city_id(self):
        from cn_travel.tool.get_weather import get_weather_by_date_range

        assert len(get_weather_by_date_range(BEIJING, "2026-08-10", 2)) == 2

    def test_unresolvable_location_returns_empty_not_an_exception(self):
        from cn_travel.tool.get_weather import get_weather_by_date_range

        assert get_weather_by_date_range("not-a-place", "2026-08-10", 2) == []

    def test_unresolvable_location_returns_none_not_a_default(self):
        """The Beijing fallback is gone: unresolvable must mean unresolvable."""
        from cn_travel.tool.get_weather import _get_coordinates_from_location

        assert _get_coordinates_from_location("zzz-not-a-place-zzz") is None

    def test_ids_outside_the_table_resolve_instead_of_falling_back(self):
        """IDs the hardcoded table never knew are now resolved, not defaulted."""
        from cn_travel.tool.get_weather import _get_coordinates_from_location

        for city in ["嘉兴", "建阳", "佳木斯"]:
            coords = _get_coordinates_from_location(city)
            assert coords is not None, f"{city} did not resolve"
            assert coords != (116.4074, 39.9042), f"{city} silently became Beijing"


@pytest.mark.amap
class TestWeatherDestinationRouting:
    """The agent must fetch weather for the DESTINATION, not the user's city."""

    @staticmethod
    def _agent(city_id="101010100"):
        os.environ.setdefault("DASHSCOPE_API_KEY", "sk-dummy-for-offline-tests")
        from cn_travel.agent import TravelAssistantFuncCall

        return TravelAssistantFuncCall(user_city_id=city_id,
                                       travel_date_range="2026-08-10~2026-08-15")

    @pytest.mark.parametrize("city", ["北京", "上海", "广州", "深圳", "杭州", "南京"])
    def test_the_six_mapped_cities_resolve(self, city):
        a = self._agent()
        assert a.get_weather_info(city, "2026-08-10", 2) and "error" not in a.get_weather_info(city, "2026-08-10", 2)

    def test_unmapped_destination_differs_from_the_users_own_city(self):
        a = self._agent(city_id="101010100")          # user in Beijing
        away = a.get_weather_info("嘉兴", "2026-08-10", 3)   # ~1200km south
        home = a.get_weather_info("北京", "2026-08-10", 3)
        assert away != home, "两地天气不该完全相同——说明又回退到同一个坐标了"


# ============================================================= get_route ===
@pytest.mark.amap
class TestRoute:
    def test_geocode_resolves_an_unambiguous_landmark(self):
        from cn_travel.tool.get_route import geocode

        lng, lat = map(float, geocode("北京天坛公园").split(","))
        assert AMAP_TIANTAN[0] < lng < AMAP_TIANTAN[1]
        assert AMAP_TIANTAN[2] < lat < AMAP_TIANTAN[3]

    def test_geocode_raises_on_an_unresolvable_address(self):
        from cn_travel.tool.get_route import geocode

        with pytest.raises(ValueError):
            geocode("zzzz-not-a-real-place-zzzz")

    def test_bare_landmark_resolves_within_the_given_city(self):
        """A city-scoped landmark resolves within the requested city."""
        from cn_travel.tool.get_route import geocode

        lng, lat = map(float, geocode("天坛", city="110000").split(","))
        assert AMAP_TIANTAN[0] < lng < AMAP_TIANTAN[1]
        assert AMAP_TIANTAN[2] < lat < AMAP_TIANTAN[3]

    def test_generic_facility_resolves_via_poi_when_scoped(self):
        from cn_travel.tool.get_route import geocode

        lng, lat = map(float, geocode("火车站", city="330400").split(","))
        assert 120.5 < lng < 121.1 and 30.5 < lat < 31.0

    def test_query_routes_returns_all_three_transport_modes(self):
        from cn_travel.tool.get_route import query_routes

        r = query_routes(BEIJING, "北京天坛公园", "北京")["routes"]
        assert set(r) == {"walking", "transit", "driving"}

    def test_an_intracity_route_is_plausible(self):
        """望京 -> 天坛 is ~15km; a sane result must be well under 100km."""
        from cn_travel.tool.get_route import query_routes

        r = query_routes(BEIJING, "北京天坛公园", "北京")["routes"]
        assert r["driving"] is not None
        assert r["driving"]["distance_m"] < 100_000

    def test_city_scoped_geocode_produces_intracity_distance(self):
        """City-scoped geocoding produces a plausible intracity route."""
        from cn_travel.tool.get_route import AMAP_KEY, get_driving

        resp = requests.get(
            f"https://restapi.amap.com/v3/geocode/geo?address=天坛&city=110000&key={AMAP_KEY}",
            timeout=10,
        ).json()
        dest = resp["geocodes"][0]["location"]
        assert get_driving(BEIJING, dest)["总距离(米)"] < 100_000


# ============================================================ get_hotel ====
class TestHotelFallback:
    """Fallback generators return their documented payload shapes."""

    def test_recommendation_fallback_shape(self):
        from cn_travel.tool.get_hotel import _generate_fallback_hotels

        hotels = _generate_fallback_hotels("北京")
        assert len(hotels) == 3
        for h in hotels:
            assert h["hotel_name"] and h["location"] and h["price_range"]

    def test_review_fallback_shape(self):
        from cn_travel.tool.get_hotel import _generate_fallback_reviews

        reviews = _generate_fallback_reviews("假日酒店")
        assert len(reviews) == 2
        for r in reviews:
            assert 1 <= r["rating"] <= 5 and r["review_content"]



@pytest.mark.slow
class TestHotelLive:
    """Credentialed hotel integrations satisfy their contracts."""

    @pytest.mark.reviews
    def test_reviews_are_real_prose(self):
        from cn_travel.business_logic.contracts import validate
        from cn_travel.tool.get_hotel import get_hotel_reviews

        out = get_hotel_reviews("北京国贸大酒店")
        validate("get_hotel_reviews", out)
        assert out["status"] == "ok"
        assert out["reviews"] and len(out["reviews"][0]["text"]) > 15
