#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify candidate weather IDs online before adding them to the city pool.

The official current-conditions endpoint is authoritative; the public mirror
is the fallback. IDs whose returned city does not match are removed and logged.
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from json import loads as json_loads

from data.paths import PATHS

DATA = PATHS.root
F = DATA / "city_weather_id.json"
TZ = ZoneInfo("America/Chicago")
_pace = threading.Semaphore(4)


def confirm(wid: str) -> str | None:
    """Return the city name confirmed by either weather endpoint."""
    with _pace:
        for url, pick in (
            (f"http://www.weather.com.cn/data/sk/{wid}.html",
             lambda j: j.get("weatherinfo", {}).get("city")),
            (f"http://t.weather.itboy.net/api/weather/city/{wid}",
             lambda j: j.get("cityInfo", {}).get("city")),
        ):
            for _ in range(2):
                try:
                    r = requests.get(url, timeout=12,
                                     headers={"Referer": "http://www.weather.com.cn/"})
                    if r.status_code == 200:
                        r.encoding = "utf-8"       # The official API omits charset, so requests may infer
                        name = pick(json_loads(r.text))  # Latin-1 and corrupt city names.
                        if name:
                            return name
                except Exception:
                    time.sleep(1.5)
        time.sleep(0.2)                      # Avoid flooding the API across all 700 cities.
    return None


def matches(city: str, got: str) -> bool:
    stem = re.sub(r"(市|县|区|州|盟|地区)$", "", got)
    return city == got or city == stem or stem in city or city in got


def main() -> int:
    d = json.loads(F.read_text(encoding="utf-8"))
    todo = {c: v for c, v in d["mapping"].items() if "verified_at" not in v}
    print(f"{len(d['mapping'])} mapped, {len(todo)} to verify online")
    demoted = []
    lock = threading.Lock()

    def work(city, entry):
        got = confirm(entry["weather_id"])
        with lock:
            if got and matches(city, got):
                entry["verified_city"] = got
                entry["verified_at"] = datetime.now(TZ).isoformat()
            else:
                demoted.append({"city": city, "weather_id": entry["weather_id"],
                                "service_said": got})
                del d["mapping"][city]

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(work, c, v): c for c, v in todo.items()}
        for i, fu in enumerate(as_completed(futs), 1):
            fu.result()
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}", flush=True)

    d["unmapped"] = sorted(set(d.get("unmapped", [])) | {x["city"] for x in demoted})
    d["_provenance"]["online_verification"] = {
        "endpoints": ["weather.com.cn/data/sk", "t.weather.itboy.net"],
        "verified_at": datetime.now(TZ).isoformat(),
        "demoted": demoted,
    }
    F.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for v in d["mapping"].values() if "verified_at" in v)
    print(f"verified {ok}, demoted {len(demoted)} -> {F.name}")
    for x in demoted[:12]:
        print("  demoted:", x)
    return 0


if __name__ == "__main__":
    sys.exit(main())
