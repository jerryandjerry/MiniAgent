#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add verified administrative codes for cities missing guide coverage.

Codes come from the AMap district service. Unresolved cities are reported and
excluded rather than assigned synthetic codes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from data.paths import PATHS, load_data_env

load_data_env()

from cn_travel.service import location

MAPPING = PATHS.knowledge_base / "city_code_mapping.json"
CITY_LIST = PATHS.root / "china_cities_list.json"


def all_cities() -> list[tuple[str, str]]:
    """Return ``(city, province)`` pairs from the canonical city list."""
    data = json.loads(CITY_LIST.read_text(encoding="utf-8"))["中国城市列表"]
    out = []
    for province, kinds in data.items():
        for kind, value in kinds.items():
            if isinstance(value, list):          # A provincial capital is a string, not a list
                out += [(c, province) for c in value]
    return sorted(set(out))


def already_have() -> set[str]:
    return {p.name.split("_")[1] for p in PATHS.guides.glob("*_travel_guide.txt")}


def resolve(name: str, province: str, retries: int = 4):
    """Resolve a city with backoff so rate limits do not mimic missing data."""
    for attempt in range(retries):
        try:
            r = location.resolve_city(name)
            if r and r.get("adcode"):
                return r["adcode"], {"name": name, "level": "地级市", "province": province}
        except Exception:
            pass
        time.sleep(0.6 * (attempt + 1))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="写回 city_code_mapping.json")
    ap.add_argument("--workers", type=int, default=3)   # Stay within Amap's free-tier capacity
    args = ap.parse_args()

    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    codes = mapping["城市编码映射"]
    have = already_have()
    known = {v["name"] for v in codes.values()} | {v["name"].rstrip("市") for v in codes.values()}

    todo = [(n, p) for n, p in all_cities()
            if n.rstrip("市") not in have and n not in have and n not in known]
    print(f"已有攻略 {len(have)} 个城市，映射里 {len(codes)} 条，待补 {len(todo)} 个")
    if not todo:
        return 0

    added, failed = {}, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(resolve, n, p): (n, p) for n, p in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            name, province = futures[fut]
            got = fut.result()
            if got:
                added[got[0]] = got[1]
            else:
                failed.append(name)
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}  已解析 {len(added)}，失败 {len(failed)}")

    print(f"\n解析成功 {len(added)}，失败 {len(failed)}")
    if failed:
        print(f"  跳过（高德查不到，不编造编码）: {failed[:15]}{' …' if len(failed) > 15 else ''}")

    if not args.write:
        print("\n（预演，未写文件。加 --write 生效）")
        return 0

    codes.update(added)
    mapping["统计信息"] = {**mapping.get("统计信息", {}), "城市总数": len(codes)}
    MAPPING.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {MAPPING}，映射现有 {len(codes)} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
