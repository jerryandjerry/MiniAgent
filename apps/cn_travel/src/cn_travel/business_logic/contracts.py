#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Input and result contracts for the five CN Travel tools.

Provider-independent result rules live in :mod:`cn_travel.service.result`.
"""
from typing import Any, Dict

from cn_travel.service import result as _spec
from cn_travel.service.result import BASE, EMPTY, ERROR, OK  # noqa: F401  Re-export for tools

CONTRACTS: Dict[str, Dict[str, Any]] = {
    "search_travel_guide": {
        "args": {"location": str, "search_mode": str},
        "returns": {
            **BASE,
            "location": str,
            "guides": [{"city": str, "province": str, "content": str, "score": float}],
        },
        "skeleton": {"location": "", "guides": []},
    },
    "get_weather_info": {
        "args": {"location": str, "start_date": str, "num_days": int},
        "returns": {
            **BASE,
            "location": str,
            "resolved": {"name": str, "adcode": str, "lat": float, "lng": float},
            "days": [{"date": str, "day_weather": str, "night_weather": str,
                      "temp_min_c": int, "temp_max_c": int}],
        },
        "skeleton": {"location": "", "resolved": None, "days": []},
    },
    "query_route": {
        "args": {"start_location": str, "end_location": str, "city": str},
        "returns": {
            **BASE,
            "origin": {"query": str, "lat": float, "lng": float},
            "destination": {"query": str, "name": str, "lat": float, "lng": float},
            "routes": {  # any mode may be None when unavailable
                "walking": {"duration_min": int, "distance_m": int},
                "transit": {"duration_min": int, "fare_cny": float, "segments": [str]},
                "driving": {"duration_min": int, "distance_m": int, "tolls_cny": float},
            },
        },
        "skeleton": {"origin": None, "destination": None, "routes": {}},
    },
    "recommend_hotels": {
        "args": {"location": str, "requirements": str, "limit": int},
        "returns": {
            **BASE,
            "location": str,
            "hotels": [{"name": str, "address": str, "district": str,
                        "rating": float, "tier": str, "tel": str,
                        "lat": float, "lng": float,
                        # None without a price backend; never estimate
                        "price_cny": int}],
        },
        "skeleton": {"location": "", "hotels": []},
    },
    "get_hotel_reviews": {
        "args": {"hotel_name": str, "location": str},
        "returns": {
            **BASE,
            "hotel_name": str,
            "rating": float,        # None when unknown
            "price_hint": str,      # None when unknown - never invent a number
            "reviews": [{"text": str}],
            "summary": str,
        },
        "skeleton": {"hotel_name": "", "rating": None, "price_hint": None,
                     "reviews": [], "summary": ""},
    },
}

TOOL_NAMES = tuple(CONTRACTS)


def validate(tool_name: str, payload: Dict[str, Any]) -> None:
    """Validate one result against this application's contracts."""
    _spec.validate(CONTRACTS, tool_name, payload)


def err(tool: str, message: str, **extra) -> Dict[str, Any]:
    """Build this application's standard error payload."""
    return _spec.err(CONTRACTS, tool, message, **extra)


def empty(tool: str, message: str, **extra) -> Dict[str, Any]:
    """Build this application's standard empty-result payload."""
    return _spec.empty(CONTRACTS, tool, message, **extra)
