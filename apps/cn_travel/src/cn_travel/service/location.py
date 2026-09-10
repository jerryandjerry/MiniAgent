"""Application service facade for location lookup use cases."""
from __future__ import annotations

from typing import Any


def lookup_district(keywords: str, subdistrict: int = 0) -> dict[str, Any]:
    """Return the AMap district response for an administrative name or code."""
    from cn_travel.tool._amap import _get

    return _get(
        "config/district",
        keywords=keywords,
        subdistrict=subdistrict,
    )


def resolve_city(name: str) -> dict[str, Any] | None:
    """Resolve a city name without exposing the provider client to callers."""
    from cn_travel.tool._amap import resolve_city as _resolve_city

    return _resolve_city(name)


__all__ = ["lookup_district", "resolve_city"]
