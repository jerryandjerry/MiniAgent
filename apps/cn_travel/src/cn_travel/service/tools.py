"""Service facade for the five model-facing travel tools."""
from __future__ import annotations

from typing import Any


def search_travel_guide(location: str, search_mode: str = "hybrid") -> dict[str, Any]:
    from cn_travel.tool.guide import search_travel_guide as tool

    return tool(location, search_mode)


def get_weather_info(
    location: str,
    start_date: str,
    num_days: int = 1,
) -> dict[str, Any]:
    from cn_travel.tool.get_weather import get_weather_by_date

    return get_weather_by_date(location, start_date, num_days)


def query_route(
    start_location: str,
    end_location: str,
    city: str = "",
) -> dict[str, Any]:
    from cn_travel.tool.get_route import query_routes

    return query_routes(start_location, end_location, city)


def recommend_hotels(location: str = "", requirements: str = "") -> dict[str, Any]:
    from cn_travel.tool.get_hotel import get_hotel_recommendations

    return get_hotel_recommendations(location, requirements)


def get_hotel_reviews(hotel_name: str, location: str = "") -> dict[str, Any]:
    from cn_travel.tool.get_hotel import get_hotel_reviews as tool

    return tool(hotel_name, location)


__all__ = [
    "get_hotel_reviews",
    "get_weather_info",
    "query_route",
    "recommend_hotels",
    "search_travel_guide",
]
