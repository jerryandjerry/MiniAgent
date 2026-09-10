#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Private AMap client. Infrastructure, not a tool — nothing here is exposed to
the model, and no tool imports another tool through it.

Swap this file to change provider; the tool contracts stay identical.
"""
import os
import re
from typing import Any, Dict, List, Optional

import requests

# Read only from the environment; a hard-coded key would ship with the repository.
KEY = os.getenv("AMAP_KEY", "")
BASE = "https://restapi.amap.com/v3"
TIMEOUT = 15


def _get(path: str, **params) -> Dict[str, Any]:
    if not KEY:
        raise RuntimeError("未设置 AMAP_KEY，请在 .env 中配置（见 .env.example）")
    params["key"] = KEY
    # Rate limits such as CUQPS_HAS_EXCEEDED_THE_LIMIT are infrastructure failures,
    # not missing results. Without retries, valid cities can be frozen as unresolved.
    import time
    data = {}
    for wait in (0, 1, 2, 4, 8):
        if wait:
            time.sleep(wait)
        r = requests.get(f"{BASE}/{path}", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        info = str(data.get("info", ""))
        if data.get("status") == "1" or not any(k in info for k in ("QPS", "LIMIT", "TOO_FREQUENT", "OVER_QUOTA")):
            return data
    return data


def resolve_city(name: str) -> Optional[Dict[str, Any]]:
    """City name (or adcode) -> {name, adcode, lat, lng}.

    Resolution returns the requested place or ``None`` without substituting a
    default location.
    """
    if not name:
        return None
    data = _get("config/district", keywords=name, subdistrict=0)
    if data.get("status") != "1":
        return None
    for d in data.get("districts") or []:
        # Street-level results are towns or streets, not cities. They can hijack
        # generic facilities such as train stations, so resolve_city rejects them.
        if d.get("level") == "street":
            continue
        center = d.get("center") or ""
        if "," not in center:
            continue
        lng, lat = center.split(",")
        return {"name": d.get("name") or name,
                "adcode": d.get("adcode") or "",
                "lat": float(lat), "lng": float(lng)}
    return _resolve_city_via_geocode(name)


def _resolve_city_via_geocode(name: str) -> Optional[Dict[str, Any]]:
    """Resolve non-administrative city phrases through validated geocoding.

    Geocoding may guess an unrelated place, so the returned city must occur in
    the original query.
    """
    data = _get("geocode/geo", address=name)
    if data.get("status") != "1":
        return None
    for g in data.get("geocodes") or []:
        city = (g.get("city") or "").strip()
        if not city or not isinstance(city, str):
            continue
        # Compare without the city suffix so a city name can match its district query
        stem = city.rstrip("市省")
        if stem and stem not in name:
            continue                      # Reject guessed matches
        loc = g.get("location") or ""
        if "," not in loc:
            continue
        lng, lat = loc.split(",")
        return {"name": city, "adcode": (g.get("adcode") or "")[:4] + "00",
                "lat": float(lat), "lng": float(lng)}
    return None


def search_poi(keywords: str, adcode: str = "", types: str = "",
               limit: int = 10) -> List[Dict[str, Any]]:
    """POI text search scoped to a city. Used for landmarks and hotels.

    Note: this is deliberately used instead of /geocode/geo. Geocoding resolves
    addresses and returns nationwide matches for bare landmarks ('天坛' ->
    Sichuan); POI search with citylimit stays inside the city.
    """
    params = dict(keywords=keywords, offset=max(1, min(limit, 25)), page=1,
                  extensions="all")
    if adcode:
        params["city"] = adcode
        params["citylimit"] = "true"
    if types:
        params["types"] = types
    data = _get("place/text", **params)
    if data.get("status") != "1":
        return []
    return data.get("pois") or []


_STRIP = re.compile(r"[^\w一-鿿]+")


def relevant(query: str, *candidates: str, threshold: float = 0.8) -> bool:
    """Is `query` actually about one of `candidates`?

    Both AMap endpoints fuzzy-match and always return something:
    '没有这个地方xyz' resolves to a restaurant, 'zzzz-not-real' to a print shop.
    Accepting those recreates the wrong-location bug this module exists to fix.

    Exact substring is too strict the other way — '北京天坛公园' resolves to
    address '北京市东城区天坛公园', which does not contain that literal string.
    So this compares character coverage instead: what fraction of the query's
    characters appear in the candidate.
    """
    q = set(_STRIP.sub("", query or ""))
    if not q:
        return False
    return any(len(q & set(c or "")) / len(q) >= threshold for c in candidates)


def search_place(keywords: str, adcode: str = "") -> Optional[Dict[str, Any]]:
    """Resolve a place name to a single POI, or None.

    POI search is fuzzy and always returns *something*: 'zzzz-not-a-real-place'
    matches a print shop. Accepting that would recreate the wrong-location bug
    this module exists to fix, so a hit is only accepted when the query really
    appears in the POI's name or its category.

    That covers both shapes we care about:
      specific place  '天坛'   -> name '天坛公园'                (query in name)
      generic facility '火车站' -> type '交通设施服务;火车站;火车站'  (query in type)
    """
    if not keywords:
        return None
    for poi in search_poi(keywords, adcode=adcode, limit=5):
        if relevant(keywords, poi.get("name") or "", poi.get("type") or ""):
            return poi
    return None


def search_nearest(keywords: str, origin: str, adcode: str = "",
                   radius_m: int = 50000) -> Optional[Dict[str, Any]]:
    """Nearest POI matching `keywords` to `origin` ("lng,lat").

    Generic destinations like '火车站' or '医院' are ambiguous by name but not
    by distance. Answering with the nearest one is both correct and what the
    user meant, and it removes the model's reason to ask "which station?".
    """
    if not keywords or not origin:
        return None
    data = _get("place/around", keywords=keywords, location=origin,
                radius=radius_m, sortrule="distance", offset=10, page=1,
                extensions="all", **({"city": adcode} if adcode else {}))
    if data.get("status") != "1":
        return None
    for poi in data.get("pois") or []:
        if relevant(keywords, poi.get("name") or "", poi.get("type") or ""):
            return poi
    return None


def poi_lat_lng(poi: Dict[str, Any]) -> Optional[Dict[str, float]]:
    loc = poi.get("location") or ""
    if "," not in loc:
        return None
    lng, lat = loc.split(",")
    return {"lat": float(lat), "lng": float(lng)}


def walking(origin: str, destination: str) -> Optional[Dict[str, Any]]:
    d = _get("direction/walking", origin=origin, destination=destination)
    paths = ((d.get("route") or {}).get("paths")) or []
    if d.get("status") != "1" or not paths:
        return None
    p = paths[0]
    return {"duration_min": int(p["duration"]) // 60, "distance_m": int(p["distance"])}


def driving(origin: str, destination: str) -> Optional[Dict[str, Any]]:
    d = _get("direction/driving", origin=origin, destination=destination, extensions="all")
    paths = ((d.get("route") or {}).get("paths")) or []
    if d.get("status") != "1" or not paths:
        return None
    p = paths[0]
    return {"duration_min": int(p["duration"]) // 60,
            "distance_m": int(p["distance"]),
            "tolls_cny": float(p.get("tolls") or 0)}


def transit(origin: str, destination: str, adcode: str) -> Optional[Dict[str, Any]]:
    d = _get("direction/transit/integrated", origin=origin, destination=destination,
             city=adcode or "")
    transits = ((d.get("route") or {}).get("transits")) or []
    if d.get("status") != "1" or not transits:
        return None
    t = transits[0]
    segments: List[str] = []
    for seg in t.get("segments") or []:
        bus = (seg.get("bus") or {}).get("buslines") or []
        if bus:
            b = bus[0]
            segments.append(
                f"乘坐 {b['name']} ({b['departure_stop']['name']} → {b['arrival_stop']['name']})")
        walk = seg.get("walking")
        if isinstance(walk, dict) and walk.get("distance"):
            segments.append(f"步行 {walk['distance']} 米")
    return {"duration_min": int(t["duration"]) // 60,
            "fare_cny": float(t.get("cost") or 0),
            "segments": segments}
