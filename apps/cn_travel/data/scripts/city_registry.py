#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the canonical city registry used by the corpus.

Corpus filename adcodes disambiguate same-named places. Geographic records are
resolved by adcode, with unique exact-name matches as the fallback. Weather IDs
come only from the online-verified entries in ``city_weather_id.json``.
"""
from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo

from data.paths import PATHS, load_data_env

load_data_env()

OUT = PATHS.root / "city_registry.json"
_pace = threading.Semaphore(4)


def lookup(adcode: str):
    """Resolve an adcode, retrying responses affected by rate limits."""
    import time
    from cn_travel.service.location import lookup_district
    for wait in (0, 1, 3, 8):
        if wait:
            time.sleep(wait)
        with _pace:
            d = lookup_district(adcode, subdistrict=0)
        if d.get("status") == "1":
            for x in d.get("districts") or []:
                if x.get("adcode") == adcode and "," in (x.get("center") or ""):
                    lng, lat = x["center"].split(",")
                    return {"official_name": x.get("name"), "level": x.get("level"),
                            "lat": float(lat), "lng": float(lng)}
            return None                     # The service responded, but this code does not exist.
    return None


def main() -> int:
    weather = json.loads((PATHS.root / "city_weather_id.json").read_text(encoding="utf-8"))
    wmap = {c: v for c, v in weather["mapping"].items() if v.get("verified_at")}

    pairs = {}
    for p in PATHS.guides.glob("*_travel_guide.txt"):
        code, city = p.name.split("_")[0], p.name.split("_")[1]
        pairs[city] = code

    reg, dead = {}, []
    lock = threading.Lock()

    def stem_match(city, name):
        import re as _re
        return city == _re.sub(r"(市|县|区|州|盟|地区|林区|新区)$", "", name or "")

    def by_name(city):
        """Resolve an unusable filename code by an unambiguous exact name."""
        import time
        from cn_travel.service.location import lookup_district
        for wait in (0, 1, 3):
            if wait:
                time.sleep(wait)
            with _pace:
                d = lookup_district(city, subdistrict=0)
            if d.get("status") == "1":
                cands = [x for x in d.get("districts") or []
                         if stem_match(city, x.get("name")) and "," in (x.get("center") or "")
                         and x.get("level") in ("city", "district")]
                cands.sort(key=lambda x: 0 if x["level"] == "city" else 1)
                # Exclude ambiguous same-level names, such as the two Tongzhou districts.
                if cands and sum(1 for c in cands
                                 if c["level"] == cands[0]["level"]) > 1:
                    return None
                if cands:
                    x = cands[0]
                    lng, lat = x["center"].split(",")
                    return {"official_name": x["name"], "level": x["level"],
                            "adcode": x["adcode"], "lat": float(lat), "lng": float(lng)}
                return None
        return None

    def work(city, code):
        got = lookup(code)
        source = "corpus_filename"
        if got is not None and not stem_match(city, got["official_name"]):
            # Reject a filename code that points to another city.
            got = None
        if got is None:
            fallback = by_name(city)
            if fallback is not None:
                got = {k: fallback[k] for k in ("official_name", "level", "lat", "lng")}
                code = fallback["adcode"]
                source = "name_reresolved"
        with lock:
            if got is None:
                dead.append({"city": city, "adcode": code})
            else:
                reg[city] = {"adcode": code, **got, "adcode_source": source,
                             "weather_id": wmap.get(city, {}).get("weather_id"),
                             "weather_verified": city in wmap}

    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(work, c, a) for c, a in pairs.items()]
        for i, fu in enumerate(as_completed(futs), 1):
            fu.result()
            if i % 100 == 0:
                print(f"  {i}/{len(pairs)}", flush=True)

    OUT.write_text(json.dumps({
        "_provenance": {
            "adcode_source": "corpus guide filenames {adcode}_{city}_travel_guide.txt",
            "geometry_source": "amap config/district queried BY ADCODE (unique hit)",
            "weather_source": "city_weather_id.json (online-verified entries only)",
            "built_at": datetime.now(ZoneInfo("America/Chicago")).isoformat(),
        },
        "cities": reg,
        "unresolvable_adcodes": dead,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    with_w = sum(1 for v in reg.values() if v["weather_verified"])
    print(f"{len(reg)} cities registered ({with_w} with verified weather id), "
          f"{len(dead)} adcodes not resolvable -> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
