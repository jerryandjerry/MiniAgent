#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build verified out-of-scope place lists for Step 1 boundary routes."""
from __future__ import annotations

import json

from data.paths import PATHS, load_data_env

# Verify every candidate place name as unresolvable through a real tool call.
UNRESOLVABLE_CANDIDATES = ["曾母暗沙", "三沙", "澎湖", "珠峰大本营", "漠河北极村",
                           "南极科考站", "月球背面", "马里亚纳海沟", "钓鱼岛海域",
                           "南沙群岛", "西沙永兴岛", "可可西里无人区"]


def _load_env() -> None:
    load_data_env()


def out_of_corpus_cities() -> list[str]:
    """Return listed cities whose normalized guide search is empty."""
    from cn_travel.service.tools import search_travel_guide

    listed = set()
    root = json.loads((PATHS.root / "china_cities_list.json").read_text(encoding="utf-8"))
    for kinds in root["中国城市列表"].values():
        for v in kinds.values():
            listed |= set(v) if isinstance(v, list) else {v}

    have = {p.name.split("_")[1] for p in PATHS.guides.glob("*_travel_guide.txt")}
    return [c for c in sorted({c.rstrip("市") for c in listed} - have)
            if search_travel_guide(c)["status"] == "empty"]


def unresolvable_places() -> tuple[list[str], list[str]]:
    """Return independently verified route and hotel resolution failures."""
    from cn_travel.service.tools import query_route, recommend_hotels

    routes, hotels = [], []
    for d in UNRESOLVABLE_CANDIDATES:
        try:
            if query_route("116.481028,39.989643", d, "北京")["status"] == "error":
                routes.append(d)
        except Exception:
            pass
        try:
            if recommend_hotels(d)["status"] == "error":
                hotels.append(d)
        except Exception:
            pass
    return routes, hotels


def main() -> None:
    _load_env()
    cities = out_of_corpus_cities()
    routes, hotels = unresolvable_places()
    if not cities:
        raise SystemExit("语料已经覆盖整份城市名单，W1 的 empty 分支没有素材了")
    if not routes or not hotels:
        raise SystemExit("没有地名会让 query_routes / get_hotel_recommendations 解析失败，"
                         "W2 或 W3 的 error 分支没有素材了")

    out = PATHS.root / "china_cities_list_out_of_scope.json"
    out.write_text(json.dumps(
        {"out_of_corpus_cities": cities,
         "unresolvable_for_routes": routes,
         "unresolvable_for_hotels": hotels},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(cities)} cities without a guide -> {out}")
    print(f"  no guide:            {cities}")
    print(f"  routes cannot solve: {routes}")
    print(f"  hotels cannot solve: {hotels}")


if __name__ == "__main__":
    main()
