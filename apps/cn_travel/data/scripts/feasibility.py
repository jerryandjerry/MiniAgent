#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the verified material-feasibility lists used by Step 1.

The probes confirm that each candidate city can produce its assigned W1, W2,
or W3 route outcome and cache the evidence by city.
"""
from __future__ import annotations

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo

from data.paths import PATHS, load_data_env

load_data_env()

OUT = PATHS.root / "material_feasibility.json"
REG = json.loads((PATHS.root / "city_registry.json").read_text(encoding="utf-8"))["cities"]
_KM = lambda a, b, c, d: (((a - c) * 111) ** 2 + ((b - d) * 85) ** 2) ** 0.5


def _evidence_names_other_city(hotel_name: str, summary: str, city: str) -> bool:
    """Detect hotel names that resolve outside the requested prefecture."""
    mine = REG[city]["adcode"][:4]
    for other, e in REG.items():
        if other == city or len(other) < 2 or other in city or other not in hotel_name:
            continue
        if e["adcode"][:4] != mine:
            return True
    return False


def probe_w1(city: str) -> bool:
    from cn_travel.service.tools import get_weather_info, search_travel_guide
    e = REG[city]
    g = search_travel_guide(city)
    if g["status"] != "ok" or not any(x.get("city") == city for x in g["guides"]):
        return False
    body = " ".join(x.get("content", "")[:400] for x in g["guides"])
    codes = set(re.findall(r"\b(\d{6})\b", body))
    okcodes = {e["adcode"], e.get("corpus_adcode", e["adcode"])}
    if codes and not codes & okcodes and any(c.startswith(("1","2","3","4","5","6")) for c in codes):
        # Six-digit administrative codes in a guide must match the city's frozen set.
        pass_codes = False
    else:
        pass_codes = True
    from datetime import timedelta
    start = (datetime.now(ZoneInfo("America/Chicago")) + timedelta(days=4)).strftime("%Y-%m-%d")
    w = get_weather_info(city, start, 2)
    if w["status"] != "ok":
        return False
    r = w.get("resolved") or {}
    near = _KM(r.get("lat", 0), r.get("lng", 0), e["lat"], e["lng"]) <= 90
    code_ok = str(r.get("adcode", ""))[:4] == e["adcode"][:4]
    return pass_codes and (near or code_ok)


def probe_w3(city: str) -> dict:
    from cn_travel.service.tools import get_hotel_reviews, recommend_hotels
    e = REG[city]
    rec = recommend_hotels(city, "")
    if rec["status"] != "ok" or not rec["hotels"]:
        return {"ok": False, "empty": False}
    h0 = rec["hotels"][0]
    if _KM(h0["lat"], h0["lng"], e["lat"], e["lng"]) > 90:
        return {"ok": False, "empty": False}
    got_ok = got_empty = False
    seen = set()
    for h in rec["hotels"]:
        if h["name"] in seen:
            continue
        seen.add(h["name"])
        rv = get_hotel_reviews(h["name"], city)
        if rv["hotel_name"] != h["name"] or \
                _evidence_names_other_city(h["name"], rv.get("summary", ""), city):
            continue
        if re.search(r"未找到该酒店|无法确认|不是同一", rv.get("summary", "")):
            continue
        if rv["status"] == "ok" and rv.get("reviews"):
            got_ok = True
        if rv["status"] == "empty":
            got_empty = True
        if got_ok and got_empty:
            break
    return {"ok": got_ok, "empty": got_empty}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--need-w3-ok", type=int, default=150)
    ap.add_argument("--need-w3-empty", type=int, default=45)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    cache = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else \
        {"w1_ok": {}, "w3": {}}
    lock = threading.Lock()
    pool = sorted(c for c, v in REG.items() if v["weather_verified"])

    # ---- W1: scan the full pool with local RAG and Open-Meteo
    todo = [c for c in pool if c not in cache["w1_ok"]]
    print(f"W1 probe: {len(todo)} cities")
    def w1(city):
        import time
        got = False
        for wait in (0, 5, 15):                      # Retry rate limits and network jitter.
            if wait:
                time.sleep(wait)
            try:
                got = probe_w1(city)
                break
            except Exception:
                continue
        with lock:
            cache["w1_ok"][city] = got
    with ThreadPoolExecutor(min(4, a.workers)) as ex:   # Weather and district APIs cannot sustain high concurrency.
        list(ex.map(w1, todo))

    # ---- W3: probe cities until the quota is filled; review lookup is network-bound
    import random
    rng = random.Random(20250914)
    order = pool[:]
    rng.shuffle(order)
    def counts():
        ok = sum(1 for v in cache["w3"].values() if v["ok"])
        emp = sum(1 for v in cache["w3"].values() if v["empty"])
        return ok, emp
    idx = 0
    while True:
        ok_n, emp_n = counts()
        if ok_n >= a.need_w3_ok and emp_n >= a.need_w3_empty:
            break
        batch = []
        while idx < len(order) and len(batch) < a.workers:
            c = order[idx]; idx += 1
            if c not in cache["w3"]:
                batch.append(c)
        if not batch:
            print("pool exhausted"); break
        def w3(city):
            import time
            got = {"ok": False, "empty": False}
            for wait in (0, 5, 15):
                if wait:
                    time.sleep(wait)
                try:
                    got = probe_w3(city)
                    break
                except Exception:
                    continue
            with lock:
                cache["w3"][city] = got
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(w3, batch))
        OUT.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        ok_n, emp_n = counts()
        print(f"  w3 probed {len(cache['w3'])}  ok={ok_n}/{a.need_w3_ok}  empty={emp_n}/{a.need_w3_empty}",
              flush=True)

    OUT.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    ok_n, emp_n = counts()
    w1n = sum(1 for v in cache["w1_ok"].values() if v)
    print(f"done: w1_ok={w1n}  w3_ok={ok_n}  w3_empty={emp_n} -> {OUT.name}")
    return 0


if __name__ == "__main__":
    main()
