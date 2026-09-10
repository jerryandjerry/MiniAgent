#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Materialize Step 1 skeletons into 1,010 validated Step 2 records.

Each record freezes its scenario facts, entities, dates, calls, and results
according to ``STEP2_MATERIALIZATION_SPEC.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from cn_travel.business_logic import contracts as _contracts
from data.paths import PATHS, load_data_env

load_data_env()

SCHEMA_VERSION = "cn_travel.materialized.v2"
TZ = ZoneInfo("America/Chicago")

CONFIG = {
    "weather_provider": "open-meteo",
    "forecast_days": 16,                   # Single definition of the weather forecast window
    "date_offset": [3, 12],
    "num_days": [1, 5],
    "departure_window_days": 5,
    "user_name": "用户",
    "asks": {"W1city": "请告诉我您想去哪个城市旅行？",
             "W1date": "请问您计划什么时候出行？",
             "W2destination": "请问您想去哪里？",
             "W3hotel": "您想了解哪家酒店的点评呢？"},
    "w5_canonical": "抱歉，我是专门的旅行助手，只能回答旅行相关的问题。",
    "requirements_catalog": ["", "", "经济实惠一点", "交通方便的",
                             "高档一点的", "市中心附近的", "评分高一些的",
                             "预算{budget}元左右"],
    "budget_range": [200, 600],
    "max_attempts_per_idx": 5,
    # City-scale entity-distance threshold; exceeding it means the tool resolved a namesake elsewhere
    "city_km": 90,
}


def route_manifest() -> dict:
    m = {}
    for t in ("A", "C", "E", "G"):
        m[f"W1-{t}"] = [("search_travel_guide", "ok"), ("get_weather_info", "ok")]
    for t in ("B", "D", "F", "H"):
        m[f"W1-{t}"] = [("search_travel_guide", "empty")]
    for t, st in (("A", "error"), ("B", "ok"), ("C", "error"), ("D", "ok")):
        m[f"W2-{t}"] = [("query_route", st)]
    for t, plan in (("A", [("recommend_hotels", "ok"), ("get_hotel_reviews", "empty")]),
                    ("B", [("recommend_hotels", "ok"), ("get_hotel_reviews", "ok")]),
                    ("C", [("get_hotel_reviews", "empty")]),
                    ("D", [("get_hotel_reviews", "ok")]),
                    ("E", [("recommend_hotels", "error")]),
                    ("F", [("recommend_hotels", "ok"), ("get_hotel_reviews", "empty")]),
                    ("G", [("recommend_hotels", "ok"), ("get_hotel_reviews", "ok")]),
                    ("H", [("get_hotel_reviews", "empty")]),
                    ("I", [("get_hotel_reviews", "ok")]),
                    ("J", [("recommend_hotels", "error")])):
        m[f"W3-{t}"] = plan
    m["W4-A"] = []
    m["W5-A"] = []
    return m


ROUTES = route_manifest()


# ---------------------------------------------------------------- Synthetic tool results
# Step 2 synthesizes tool results per contracts.py: guides come from the local corpus, cities from the registry,
# hotel names from search results, values remain internally consistent, and statuses match storyboard labels.
_REG_FOR_SYNTH: dict = {}
_HOTEL_SCHEMA = {
    "type": "object",
    "properties": {"hotels": {"type": "array", "items": {
        "type": "object",
        "properties": {"name": {"type": "string"}, "address": {"type": "string"},
                       "district": {"type": "string"}, "tier": {"type": "string"},
                       "rating": {"type": "number"}},
        "required": ["name", "address", "district", "tier", "rating"],
        "additionalProperties": False}}},
    "required": ["hotels"], "additionalProperties": False,
}
_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {"reviews": {"type": "array", "items": {"type": "string"}},
                   "rating": {"type": "number"}, "summary": {"type": "string"}},
    "required": ["reviews", "rating", "summary"], "additionalProperties": False,
}


def _syn_guide(a: dict) -> dict:
    import re as _re
    from cn_travel.business_logic.contracts import empty
    city = a["location"]
    hits = sorted(PATHS.guides.glob(f"*_{city}_travel_guide.txt"))
    if not hits:
        return empty("search_travel_guide", f"no guide for {city}", location=city)
    text = hits[0].read_text(encoding="utf-8")[:3000]
    # Normalize administrative codes to registry adcodes so guides and weather reference the same entity.
    ent = _REG_FOR_SYNTH.get(city)
    if ent:
        canon = ent["adcode"]
        text = _re.sub(r"(城市编码[:：]\s*)\d{6}", rf"\g<1>{canon}", text)
        for old in {hits[0].name.split("_")[0], *(ent.get("corpus_adcode"),)} - {canon, None}:
            text = _re.sub(rf"\b{old}\b", canon, text)
    return {"status": "ok", "source": "synthetic:local-corpus", "location": city,
            "guides": [{"city": city, "province": "", "content": text, "score": 1.0}]}


def _syn_weather(a: dict) -> dict:
    from cn_travel.business_logic.contracts import err
    e = _REG_FOR_SYNTH.get(a["location"])
    if not e:
        return err("get_weather_info", f"could not resolve location {a['location']!r}")
    rng = random.Random(f"wx:{a['location']}:{a['start_date']}")
    start = datetime.strptime(a["start_date"], "%Y-%m-%d")
    base_max = int(round(31 - 0.45 * (e["lat"] - 22)))        # August: derive a plausible temperature from latitude
    conds = ["晴", "多云", "阴", "小雨", "阵雨", "晴", "多云"]
    days = []
    for i in range(int(a["num_days"])):
        d = start + timedelta(days=i)
        tmax = base_max + rng.randint(-3, 3)
        days.append({"date": d.strftime("%Y-%m-%d"), "day_weather": rng.choice(conds),
                     "night_weather": rng.choice(["晴", "多云", "阴"]),
                     "temp_min_c": tmax - rng.randint(6, 10), "temp_max_c": tmax})
    return {"status": "ok", "source": "synthetic:registry-geo", "location": a["location"],
            "resolved": {"name": e["official_name"], "adcode": e["adcode"],
                         "lat": e["lat"], "lng": e["lng"]}, "days": days}


def _syn_route(a: dict) -> dict:
    from cn_travel.business_logic.contracts import err
    start, end, city = a["start_location"], a["end_location"], a["city"]
    rng = random.Random(f"rt:{start}:{end}")
    if "," in start:                                          # Local generic facility name
        e = _REG_FOR_SYNTH.get(city.rstrip("市"))
        if e is None:
            return err("query_route", f"could not resolve destination {end!r}")
        lng0, lat0 = (float(x) for x in start.split(","))
        names = {"火车站": f"{city.rstrip('市')}站", "医院": f"{city.rstrip('市')}市人民医院",
                 "学校": f"{city.rstrip('市')}市第一中学", "机场": f"{city.rstrip('市')}机场",
                 "银行": f"中国银行{city.rstrip('市')}分行"}
        if end not in names:
            return err("query_route", f"could not resolve destination {end!r}")
        dlat, dlng = rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03)
        lat1, lng1 = lat0 + dlat, lng0 + dlng
        dist = int(_km(lat0, lng0, lat1, lng1) * 1000 * 1.3)
        routes = {"walking": {"duration_min": max(1, dist // 80), "distance_m": dist},
                  "transit": {"duration_min": max(5, dist // 250 + 8),
                              "fare_cny": float(2 + dist // 5000),
                              "segments": [f"步行至公交站", f"乘坐公交前往{names[end]}"]},
                  "driving": {"duration_min": max(2, dist // 400), "distance_m": dist,
                              "tolls_cny": 0.0}}
        return {"status": "ok", "source": "synthetic:facility-template",
                "origin": {"query": start, "lat": lat0, "lng": lng0},
                "destination": {"query": end, "name": names[end], "lat": lat1, "lng": lng1},
                "routes": routes}
    o, d = _REG_FOR_SYNTH.get(start), _REG_FOR_SYNTH.get(end)
    if o is None or d is None:
        return err("query_route", f"could not resolve destination {end!r}")
    km = _km(o["lat"], o["lng"], d["lat"], d["lng"]) * 1.25
    routes = {"walking": None,
              "transit": {"duration_min": int(km / 150 * 60 + 40),
                          "fare_cny": round(km * 0.45, 1),
                          "segments": [f"{start}站 乘高铁/动车 至 {end}站"]},
              "driving": {"duration_min": int(km / 85 * 60), "distance_m": int(km * 1000),
                          "tolls_cny": round(km * 0.5, 1)}}
    return {"status": "ok", "source": "synthetic:registry-geo",
            "origin": {"query": start, "lat": o["lat"], "lng": o["lng"]},
            "destination": {"query": end, "name": d["official_name"], "lat": d["lat"], "lng": d["lng"]},
            "routes": routes}


def _syn_recommend(a: dict) -> dict:
    from data.clients import claude_cli
    from cn_travel.business_logic.contracts import err
    loc, req = a["location"], a.get("requirements", "")
    e = _REG_FOR_SYNTH.get(loc)
    if not e:
        return err("recommend_hotels", f"could not resolve location {loc!r}")
    prompt = (f"联网搜索，列出 {loc} 市区内 3 家真实存在的酒店（正式全名，连锁门店写成'X酒店(Y店)'），"
              f"每家给地址、所在区县、档次（经济型/舒适型/高档型/豪华型）和 0-5 分的平台评分。"
              f"{('用户要求：' + req + '。') if req else ''}只输出检索到的真实酒店，严禁编造。")
    try:
        data = claude_cli.run(prompt, schema=_HOTEL_SCHEMA, effort="low")
    except Exception as exc:
        return err("recommend_hotels", f"web research failed: {exc}")
    rng = random.Random(f"ht:{loc}:{req}")
    hotels = [{"name": h["name"].strip(), "address": h.get("address", ""), "district": h.get("district", ""),
               "rating": round(min(5.0, max(0.0, float(h.get("rating") or 4.0))), 1),
               "tier": h.get("tier", ""), "tel": "",
               "lat": round(e["lat"] + rng.uniform(-0.02, 0.02), 6),
               "lng": round(e["lng"] + rng.uniform(-0.02, 0.02), 6), "price_cny": None}
              for h in (data.get("hotels") or [])[:3] if str(h.get("name", "")).strip()]
    if not hotels:
        return err("recommend_hotels", f"no hotels found for {loc!r}")
    return {"status": "ok", "source": "synthetic:web-research", "location": loc, "hotels": hotels}


def _syn_reviews(a: dict) -> dict:
    """Build a synthetic review result from researched or generated evidence."""
    from data.clients import claude_cli
    name, loc = a["hotel_name"], a.get("location", "")
    for mode, prompt in (("web", f"联网搜索“{loc} {name}”的真实用户评价：给 3 条点评原文（中文，每条 20-60 字）、"
                                 "0-5 分评分和一句话总结。只输出检索到的内容。"),
                         ("generated", f"为{loc}的酒店“{name}”写 3 条像真实住客写的中文点评（每条 20-60 字，"
                                       "有具体细节、有褒有贬），给一个 3.8-4.8 的评分和一句话总结。")):
        try:
            data = claude_cli.run(prompt, schema=_REVIEW_SCHEMA, effort="low")
            reviews = [{"text": str(x).strip()} for x in (data.get("reviews") or []) if str(x).strip()]
            if reviews:
                return {"status": "ok", "source": f"synthetic:{mode}", "hotel_name": name,
                        "rating": round(float(data.get("rating") or 4.2), 1), "price_hint": None,
                        "reviews": reviews, "summary": str(data.get("summary") or "").strip()}
        except Exception:
            continue
    return {"status": "empty", "source": "synthetic:none", "hotel_name": name, "rating": None,
            "price_hint": None, "reviews": [], "summary": "未检索到该酒店的公开评价"}


def _real_tools():
    """Return the synthetic tool implementations used for materialization."""
    return {"search_travel_guide": _syn_guide, "get_weather_info": _syn_weather,
            "query_route": _syn_route, "recommend_hotels": _syn_recommend,
            "get_hotel_reviews": _syn_reviews}


def load_registry() -> dict:
    return json.loads((PATHS.root / "city_registry.json").read_text(encoding="utf-8"))["cities"]


def _km(lat1, lng1, lat2, lng2) -> float:
    dx = (lng1 - lng2) * 111 * math.cos(math.radians((lat1 + lat2) / 2))
    dy = (lat1 - lat2) * 111
    return (dx * dx + dy * dy) ** 0.5


class Executor:
    TRANSIENT = ("timeout", "timed out", "connection", "rate limit", "429", "超时")

    def __init__(self, out_dir: pathlib.Path, tool_fp: str, tools: dict, known: set | None = None):
        self.dir = out_dir / "exec_cache"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tool_fp = tool_fp
        self.tools = tools
        self.known = known or set()        # Registry cities can fail to resolve only because of rate limiting
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def key(self, name: str, args: dict) -> str:
        blob = json.dumps({"tool": name, "args": args, "fp": self.tool_fp},
                          ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def run(self, name: str, args: dict) -> dict:
        k = self.key(name, args)
        with self._guard:
            lock = self._locks.setdefault(k, threading.Lock())
        with lock:
            f = self.dir / f"{k}.json"
            # Bypass the execution cache so current normalization rules apply directly to local guides.
            cacheable = name != "search_travel_guide"
            if cacheable and f.exists():
                return json.loads(f.read_text(encoding="utf-8"))
            attempts = []
            for wait in (0, 3, 10, 30, 60):
                if wait:
                    time.sleep(wait)
                try:
                    result = self.tools[name](args)
                    loc = args.get("location") or args.get("end_location") or ""
                    src = str(result.get("source", ""))
                    if result.get("status") == "error" and "could not resolve" in src \
                            and loc in self.known and name == "get_weather_info":
                        attempts.append(f"throttled: {src[:60]}")
                        continue                    # Weather still uses the live API; back off on rate limits
                    break
                except Exception as exc:
                    attempts.append(f"{type(exc).__name__}: {exc}"[:200])
                    if not any(s in str(exc).lower() for s in self.TRANSIENT) \
                            and not isinstance(exc, (FileNotFoundError, OSError)):
                        raise
            else:
                raise RuntimeError(f"{name} transient failures exhausted: {attempts[-1]}")
            rec = {"execution_id": f"sha256:{k}", "name": name, "arguments": args,
                   "result": result, "attempts": len(attempts) + 1,
                   "observed_at": datetime.now(TZ).isoformat()}
            if cacheable:
                tmp = f.with_suffix(".tmp")
                tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
                os.replace(tmp, f)
            return rec

    def put(self, name: str, args: dict, result: dict) -> dict:
        """Write a synthetic result to the deterministic execution cache."""
        k = self.key(name, args)
        with self._guard:
            lock = self._locks.setdefault(k, threading.Lock())
        with lock:
            f = self.dir / f"{k}.json"
            if f.exists():
                return json.loads(f.read_text(encoding="utf-8"))
            rec = {"execution_id": f"sha256:{k}", "name": name, "arguments": args,
                   "result": result, "attempts": 1, "observed_at": datetime.now(TZ).isoformat()}
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, f)
            return rec


class Deferred(Exception):
    """Indicate that the current material cannot satisfy the fixed label."""


def pick_profile(reg: dict, pool: list[str], seed: int, idx: int, attempt: int) -> dict:
    rng = random.Random(f"{seed}:{idx}:a{attempt}:profile")
    city = rng.choice(pool)
    e = reg[city]
    return {"name": e["official_name"], "query": city, "adcode": e["adcode"],
            "lat": e["lat"], "lng": e["lng"], "weather_id": e["weather_id"]}


def pick_dates(seed: int, idx: int, attempt: int, today: str) -> dict:
    rng = random.Random(f"{seed}:{idx}:a{attempt}:date")
    base = datetime.strptime(today, "%Y-%m-%d")
    horizon = CONFIG["forecast_days"] - 1
    lo, hi = CONFIG["date_offset"]
    off = rng.randint(lo, min(hi, horizon))
    nmax = min(CONFIG["num_days"][1], horizon - off + 1)
    n = rng.randint(CONFIG["num_days"][0], max(CONFIG["num_days"][0], nmax))
    start = base + timedelta(days=off)
    return {"start_date": start.strftime("%Y-%m-%d"), "num_days": n,
            "end_date": (start + timedelta(days=n - 1)).strftime("%Y-%m-%d"),
            "weekday": "周" + "一二三四五六日"[start.weekday()],
            "surface": rng.choice(["exact_date", "relative_verified"])}


def entity_gate(name: str, args: dict, result: dict, reg: dict) -> None:
    """Reject successful results that resolve to the wrong entity."""
    if result["status"] != "ok":
        return
    if name == "search_travel_guide":
        got = [g.get("city") or "" for g in result["guides"]]
        assert any(args["location"] == c for c in got), f"guide城市 {got} != {args['location']}"
        want = reg.get(args["location"])
        if want:
            import re as _re
            body = " ".join(g.get("content", "")[:400] for g in result["guides"])
            codes = set(_re.findall(r"\b(\d{6})\b", body))
            okc = {want["adcode"]}
            geo = {c for c in codes if c[:1] in "123456"}
            assert not geo or geo & okc, \
                f"攻略正文嵌码 {sorted(geo)[:3]} 与 {args['location']} 的注册/冻结编码不符"
    elif name == "get_weather_info":
        r = result.get("resolved") or {}
        want = reg.get(args["location"])
        if want:
            ok_code = str(r.get("adcode", ""))[:4] == want["adcode"][:4]
            ok_dist = _km(r.get("lat", 0), r.get("lng", 0),
                          want["lat"], want["lng"]) <= CONFIG["city_km"]
            assert ok_code or ok_dist, \
                f"天气解析到 {r.get('name')}({r.get('adcode')})，不是 {args['location']}({want['adcode']})"
    elif name == "query_route":
        d = result["destination"]
        assert d["query"] == args["end_location"]
        want = reg.get(args["end_location"])
        if want:
            assert _km(d["lat"], d["lng"], want["lat"], want["lng"]) <= CONFIG["city_km"], \
                f"终点 {d['name']} 距 {args['end_location']} 注册质心过远——解析串城了"
        o = result["origin"]
        want_o = reg.get(args["start_location"])
        if want_o:
            assert _km(o["lat"], o["lng"], want_o["lat"], want_o["lng"]) <= CONFIG["city_km"], \
                f"起点 {args['start_location']} 解析串城了"
    elif name == "recommend_hotels":
        assert result["location"] == args["location"]
        assert result["hotels"], "ok 却没有酒店"
        want = reg.get(args["location"])
        if want:
            h = result["hotels"][0]
            assert _km(h["lat"], h["lng"], want["lat"], want["lng"]) <= CONFIG["city_km"], \
                f"推荐的酒店不在 {args['location']}（串到别的同名城市了）"
    elif name == "get_hotel_reviews":
        assert result["hotel_name"] == args["hotel_name"]
        assert result.get("reviews"), "ok 却没有点评正文"


def contract_ok(name: str, result: dict) -> bool:
    try:
        _contracts.validate(name, result)
    except Exception:
        return False
    if name == "get_hotel_reviews" and result["status"] == "ok" and not result.get("reviews"):
        return False                                # Successful review results must contain at least one review
    return True


def turn_plan_for(row: dict, canon: dict) -> tuple[list, list]:
    w, om = row["workflow"], row["omit"]
    asks, plan = [], []
    A = CONFIG["asks"]

    def turn(intent, reveal, conceal, constraints=()):
        plan.append({"turn": len(plan) + 1, "intent": intent, "reveal": reveal,
                     "conceal": conceal, "constraints": list(constraints)})

    if w == 1:
        turn("travel_plan", [s for s in ("city", "date") if s not in om],
             [s for s in ("city", "date") if s in om],
             ["city_verbatim"] if "city" not in om else [])
        for slot in ("city", "date"):
            if slot in om:
                asks.append({"after_turn": len(plan), "slot": slot, "text": A[f"W1{slot}"]})
                turn(f"supply_{slot}", [slot], [],
                     ["city_verbatim"] if slot == "city" else [])
    elif w == 2:
        intercity = "origin" in row["slots"]
        if "destination" in om:
            # Turn 1 supplies the origin and hides the destination; turn 2 supplies it verbatim.
            turn("route_query", ["origin"] if intercity else [], ["destination"],
                 ["origin_verbatim", "origin_not_replaced"] if intercity else [])
            asks.append({"after_turn": 1, "slot": "destination", "text": A["W2destination"]})
            turn("supply_destination", ["destination"], [], ["destination_verbatim"])
        else:
            turn("route_query",
                 (["origin", "destination"] if intercity else ["destination"]), [],
                 ["destination_verbatim"] + (["origin_verbatim"] if intercity else []))
    elif w == 3:
        path = row["slots"]["path"]
        if "hotel" in om:
            turn("hotel_reviews", [], ["hotel"],
                 ["deictic_hotel_reference", "review_intent_only"])
            asks.append({"after_turn": 1, "slot": "hotel", "text": A["W3hotel"]})
            if path == "reviews_only":
                turn("supply_hotel", ["hotel"], [], ["hotel_verbatim"])
            else:
                turn("recommend_hotels",
                     ["city"] + (["requirements"] if canon.get("requirements") else []),
                     [], ["abandon_unresolved_hotel_reference"])
        elif path == "reviews_only":
            turn("hotel_reviews", ["hotel"], [], ["hotel_verbatim"])
        else:
            turn("recommend_hotels",
                 ["city"] + (["requirements"] if canon.get("requirements") else []), [], [])
    elif w == 4:
        turn("travel_chitchat", ["topic"], [], ["no_tool_expected"])
    else:
        turn("off_topic", ["topic"], [], ["must_be_refused"])
    return plan, asks


def materialize(row: dict, attempt: int, ctx: dict, ex: Executor, reg: dict,
                pool: list[str], seed: int, ledger) -> dict:
    idx, w, sl = row["idx"], row["workflow"], dict(row["slots"])
    plan = ROUTES[row["route"]]
    prof = pick_profile(reg, pool, seed, idx, attempt)
    coords = f"{prof['lng']},{prof['lat']}"
    today = ctx["today"]
    canon: dict = {}
    visible, selection = [], []

    if w == 1:
        # Unpinned cities (excluded namesakes or invalid codes) cannot be verified; fail closed and choose new material
        if row["expect"] == "ok" and sl["city"] not in reg:
            raise Deferred(f"{sl['city']} 不在注册表，实体无法核验")
        canon["city"] = sl["city"]
        ent = reg.get(sl["city"])
        if ent:
            # _syn_guide uses the registry adcode so guides and weather share one administrative entity.
            canon["city_entity"] = {"name": ent["official_name"], "adcode": ent["adcode"]}
        canon.update(pick_dates(seed, idx, attempt, today))
    elif w == 2:
        if "origin" in sl:                          # Both cities on an intercity route must be verifiable
            for c in (sl["origin"], sl["destination"]):
                if row["expect"] == "ok" and c not in reg:
                    raise Deferred(f"{c} 不在注册表，实体无法核验")
        canon["destination"] = sl["destination"]
        if "origin" in sl:
            canon["origin"] = sl["origin"]
    elif w == 3:
        if row["expect"] != "error" and sl["city"] not in reg:
            raise Deferred(f"{sl['city']} 不在注册表，实体无法核验")
        canon["city"] = sl["city"]
        if sl["path"] != "reviews_only" and row["expect"] != "error":
            rng = random.Random(f"{seed}:{idx}:a{attempt}:req")
            req = rng.choice(CONFIG["requirements_catalog"])
            if "{budget}" in req:
                req = req.format(budget=rng.randrange(*CONFIG["budget_range"], 50))
            canon["requirements"] = req
    elif w in (4, 5):
        canon["topic"] = sl["topic"]

    for name, want in plan:
        if name == "search_travel_guide":
            args = {"location": canon["city"], "search_mode": "hybrid"}
        elif name == "get_weather_info":
            args = {"location": canon["city"], "start_date": canon["start_date"],
                    "num_days": canon["num_days"]}
        elif name == "query_route":
            if "origin" in canon:
                args = {"start_location": canon["origin"],
                        "end_location": canon["destination"],
                        "city": canon["destination"]}
            else:
                args = {"start_location": coords,
                        "end_location": canon["destination"], "city": prof["name"]}
        elif name == "recommend_hotels":
            args = {"location": canon["city"], "requirements": canon.get("requirements", "")}
        elif name == "get_hotel_reviews":
            if visible and visible[-1]["name"] == "recommend_hotels":
                rec_pool = visible[-1]["result"]["hotels"]
            else:
                disc = ex.run("recommend_hotels", {"location": canon["city"], "requirements": ""})
                if disc["result"]["status"] != "ok":
                    raise Deferred(f"discovery recommend {disc['result']['status']} in {canon['city']}")
                if not contract_ok("recommend_hotels", disc["result"]):
                    raise Deferred("discovery recommend 合同不合格")
                selection.append(disc)
                rec_pool = disc["result"]["hotels"]
            seen, found = set(), None
            rec_pool = [h for h in rec_pool
                        if not (h["name"] in seen or seen.add(h["name"]))]
            if want == "empty" and rec_pool:
                # For a synthetic "no public reviews" label, return a valid empty result.
                # Another row in the same city may already have cached an ok review result for a hotel.
                # Each (hotel, city) key has one result, so choose a hotel not already cached as ok.
                for h in rec_pool:
                    args_h = {"hotel_name": h["name"], "location": canon["city"]}
                    cf = ex.dir / f"{ex.key('get_hotel_reviews', args_h)}.json"
                    if cf.exists() and json.loads(cf.read_text(encoding="utf-8"))["result"]["status"] != "empty":
                        continue
                    found = ex.put("get_hotel_reviews", args_h,
                                   {"status": "empty", "source": "synthetic:label",
                                    "hotel_name": h["name"], "rating": None, "price_hint": None,
                                    "reviews": [], "summary": "未检索到该酒店的公开评价"})
                    canon["hotel_name"] = h["name"]
                    break
                if found is None:
                    raise Deferred(f"{canon['city']}: 所有推荐酒店的点评都已缓存为 ok，换次尝试")
                rec_pool = []
            for h in rec_pool:
                cand = ex.run("get_hotel_reviews",
                              {"hotel_name": h["name"], "location": canon["city"]})
                res = cand["result"]
                if not contract_ok("get_hotel_reviews", res):
                    ledger("probe_contract_invalid", row, attempt,
                           f"{h['name']}: status={res['status']}")
                    continue                        # Keep only in the ledger; exclude from every trajectory
                import re as _re
                mine4 = reg[canon["city"]]["adcode"][:4]
                suspect = _re.search(r"未找到该酒店|无法确认|不是同一", res.get("summary", "")) or \
                    any(o != canon["city"] and len(o) >= 2 and o not in canon["city"]
                        and o in h["name"] and e["adcode"][:4] != mine4
                        for o, e in reg.items())          # Count only cross-prefecture cases as city mismatches
                if suspect:
                    ledger("probe_entity_suspect", row, attempt, h["name"][:60])
                    continue                        # Discard evidence that identifies a different city or hotel
                good = res["status"] == "empty" if want == "empty" else res["status"] == "ok"
                if good:
                    found = cand
                    canon["hotel_name"] = h["name"]
                    break
                selection.append(cand)
            if found is None:
                raise Deferred(f"{canon['city']}: no hotel with reviews={want}")
            entity_gate("get_hotel_reviews", found["arguments"], found["result"], reg)
            visible.append(found)
            continue
        rec = ex.run(name, args)
        if rec["result"]["status"] != want:
            raise Deferred(f"{name} status {rec['result']['status']} != {want}")
        if not contract_ok(name, rec["result"]):
            raise Deferred(f"{name} 合同不合格")
        entity_gate(name, args, rec["result"], reg)
        visible.append(rec)

    price_known = any(h.get("price_cny") is not None
                      for t in visible if t["name"] == "recommend_hotels"
                      for h in t["result"]["hotels"]) or \
        any(t["result"].get("price_hint") for t in visible if t["name"] == "get_hotel_reviews")

    tplan, asks = turn_plan_for(row, canon)
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": {"idx": idx, **{k: row[k] for k in
                     ("workflow", "route", "edges", "turns", "expect", "slots", "omit")}},
        "context": {
            "user_name": CONFIG["user_name"], "today": today,
            "current_city": {"name": prof["name"], "adcode": prof["adcode"],
                             "weather_id": prof["weather_id"]},
            "start_coordinates": coords,
            "departure_window": ctx["departure_window"],
        },
        "resolved": {
            "canonical_slots": canon,
            "turn_plan": tplan,
            "assistant_asks": asks,
            "response_policy": {
                "price_known": bool(price_known),
                "weather_use": "safety_adjustments_only",
                "allowed_capabilities": ["information_only"],
                "canonical_refusal": CONFIG["w5_canonical"] if w == 5 else None,
            },
        },
        "visible_tool_trace": [
            {"ordinal": i, **{k: t[k] for k in ("execution_id", "name", "arguments", "result")}}
            for i, t in enumerate(visible)],
        "selection_trace": [
            {k: t[k] for k in ("execution_id", "name", "arguments", "result")}
            for i, t in enumerate(selection)
            if t["execution_id"] not in {x["execution_id"] for x in selection[:i]}],
        "provenance": {**ctx["provenance"],
                       "source_storyboard_idx": idx,
                       "candidate_id": f"{ctx['provenance']['run_id']}-{idx:04d}-a{attempt}",
                       "attempt": attempt,
                       "created_at": datetime.now(TZ).isoformat()},
    }


def validate_record(record: dict, rows: list, fingerprint: str = "") -> None:
    assert record["schema_version"] == SCHEMA_VERSION
    md = record["metadata"]
    src = record["provenance"]["source_storyboard_idx"]
    assert src == md["idx"], "source_storyboard_idx != metadata.idx (DI-009)"
    assert "replaces_storyboard_idx" not in record["provenance"], "替补字段已废止 (DI-009)"
    row = rows[src]
    for k in ("workflow", "route", "edges", "turns", "expect", "slots", "omit"):
        assert md[k] == row[k], f"metadata.{k} mutated vs storyboard[{src}]"
    if fingerprint:
        assert record["provenance"]["fingerprint"] == fingerprint, "混入了别的指纹的记录"
    plan = ROUTES[md["route"]]
    trace = record["visible_tool_trace"]
    assert [(t["name"], t["result"]["status"]) for t in trace] == plan
    schema = {t["function"]["name"]: t["function"]["parameters"].get("required", [])
              for t in json.loads(PATHS.tool_schemas.read_text(encoding="utf-8"))}
    assert [t.get("ordinal") for t in trace] == list(range(len(trace))), "visible ordinal 不连续"
    for t in list(trace) + list(record["selection_trace"]):
        assert t["result"].get("source"), f"{t['name']} result.source 为空"
        missing = [k for k in schema.get(t["name"], []) if k not in t["arguments"]]
        assert not missing, f"{t['name']} missing required {missing}"
        _contracts.validate(t["name"], t["result"])
        if t["name"] == "get_hotel_reviews" and t["result"]["status"] == "ok":
            assert t["result"].get("reviews"), "ok+空点评混进了轨迹"
    tp = record["resolved"]["turn_plan"]
    assert len(tp) == md["turns"]
    assert len(record["resolved"]["assistant_asks"]) == md["turns"] - 1
    ids = [t["execution_id"] for t in trace] + \
          [t["execution_id"] for t in record["selection_trace"]]
    assert len(ids) == len(set(ids)), "execution_id collision"
    cc = record["context"]["current_city"]
    assert cc.get("weather_id"), "画像城市没有已核对的天气编号"
    if md["workflow"] == 3 and md["expect"] != "error":
        rv = [t for t in trace if t["name"] == "get_hotel_reviews"]
        assert rv and rv[-1]["arguments"]["hotel_name"] == \
            record["resolved"]["canonical_slots"]["hotel_name"]
        rec_t = [t for t in trace if t["name"] == "recommend_hotels"]
        if rec_t:
            names = [h["name"] for h in rec_t[0]["result"]["hotels"]]
            assert rv[-1]["arguments"]["hotel_name"] in names
    if md["workflow"] == 1:
        cs = record["resolved"]["canonical_slots"]
        assert "start_date" in cs and "weekday" in cs
        wd = "周" + "一二三四五六日"[
            datetime.strptime(cs["start_date"], "%Y-%m-%d").weekday()]
        assert cs["weekday"] == wd
        if md["expect"] == "ok":
            w = [t for t in trace if t["name"] == "get_weather_info"][0]
            assert w["arguments"]["start_date"] == cs["start_date"]


def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(PATHS.root / "1_storyboard.json"))
    ap.add_argument("--output", default=str(PATHS.root / "2_materialized"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20250914)
    ap.add_argument("--today", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--indices", default="")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--adopt-fingerprint", action="store_true",
                    help="续跑时沿用目录里已有记录的运行指纹（脚本打了补丁也不作废好记录）")
    a = ap.parse_args()

    out = pathlib.Path(a.output)
    rows = json.loads(pathlib.Path(a.input).read_text(encoding="utf-8"))
    for i, r in enumerate(rows):
        r["idx"] = i

    # --validate-only verifies the sealed manifest and records without changing output.
    if a.validate_only:
        manifest_path = out / "manifest.json"
        if not manifest_path.is_file():
            print(f"missing sealed manifest: {manifest_path}")
            return 1
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sealed_fp = manifest["fingerprint"]
        expected_hashes = manifest.get("records_sha256", {})
        files = sorted(out.glob("[0-9]*.json"))
        n = bad = 0
        for f in files:
            try:
                expected = expected_hashes.get(f.stem)
                assert expected is not None, f"{f.stem} is absent from records_sha256"
                assert _sha(f) == expected, f"{f.name} hash differs from sealed manifest"
                validate_record(json.loads(f.read_text(encoding="utf-8")), rows, sealed_fp)
                n += 1
            except Exception as exc:
                bad += 1
                print(f"invalid {f.name}: {exc}")
        expected_count = int(manifest.get("selected", manifest.get("required", 0)))
        if len(files) != expected_count or len(expected_hashes) != expected_count:
            bad += 1
            print(
                "sealed record count mismatch: "
                f"files={len(files)} hashes={len(expected_hashes)} expected={expected_count}"
            )
        print(f"validated against sealed fingerprint and hashes: {n} ok, {bad} invalid")
        return 1 if bad else 0

    (out / "candidates").mkdir(parents=True, exist_ok=True)

    sys_today = datetime.now(TZ).strftime("%Y-%m-%d")
    today = a.today or sys_today

    picked = rows
    if a.only:
        keep = set(a.only.split(","))
        picked = [r for r in picked if r["route"] in keep]
    if a.indices:
        keep_i = {int(x) for x in a.indices.split(",")}
        picked = [r for r in picked if r["idx"] in keep_i]
    if a.limit:
        seen: dict[str, int] = {}
        sel = []
        for r in picked:
            if seen.get(r["route"], 0) < a.limit:
                seen[r["route"]] = seen.get(r["route"], 0) + 1
                sel.append(r)
        picked = sel

    needs_weather = [r for r in picked if r["workflow"] == 1 and r["expect"] == "ok"]
    if today != sys_today and needs_weather and not a.validate_only:
        sys.exit(f"--today {today} != 系统今天 {sys_today}，选中 {len(needs_weather)} 行"
                 f"需要实时天气：请使用回放/模拟数据，或移除 --today。")

    reg = load_registry()
    _REG_FOR_SYNTH.update(reg)
    pool = sorted(c for c, v in reg.items() if v["weather_verified"])
    src = {
        "registry_sha256": _sha(PATHS.root / "city_registry.json"),
        "weather_map_sha256": _sha(PATHS.root / "city_weather_id.json"),
        "boundary_sha256": _sha(PATHS.root / "china_cities_list_out_of_scope.json"),
        "routes_doc_sha256": _sha(PATHS.governing_doc),
        "system_prompt_sha256": _sha(PATHS.business_logic / "system_prompt.md"),
        "user_info_sha256": _sha(PATHS.business_logic / "user_info.md"),
        "tool_schema_sha256": _sha(PATHS.tool_schemas),
        "contract_sha256": _sha(PATHS.business_logic / "contracts.py"),
        "spec_result_sha256": _sha(PATHS.service / "result.py"),
        "tools_sha256": {p.name: _sha(p) for p in sorted(
            PATHS.tool.glob("get_*.py"))},
        "backends_sha256": {p.name.removeprefix("_"): _sha(p) for p in sorted(
            path for path in PATHS.tool.glob("_*.py")
            if path.name not in {"__init__.py", "_providers.py"})},
        "profile_pool_size": len(pool),
        "seed": a.seed, "logical_today": today, "system_today": sys_today,
        "forecast_available_through": (datetime.strptime(sys_today, "%Y-%m-%d")
                                       + timedelta(days=CONFIG["forecast_days"] - 1)
                                       ).strftime("%Y-%m-%d"),
        "config": CONFIG,
    }
    fingerprint = hashlib.sha256(
        json.dumps(src, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    src["storyboard_sha256"] = _sha(pathlib.Path(a.input))   # Archive only; exclude from the fingerprint
    src["materializer_sha256"] = [_sha(pathlib.Path(__file__))]  # Record script identity outside the data fingerprint
    # --adopt-fingerprint reuses the existing run fingerprint and retains all materializer hashes.
    mf0 = out / "manifest.json"
    if a.adopt_fingerprint:
        prev = None
        if mf0.exists():
            prev = json.loads(mf0.read_text(encoding="utf-8"))
        else:
            first = next(iter(sorted(out.glob("[0-9]*.json"))), None)
            if first:
                prev = {"fingerprint": json.loads(first.read_text(encoding="utf-8"))["provenance"]["fingerprint"],
                        "sources": {}}
        if prev:
            fingerprint = prev["fingerprint"]
            seen = prev.get("sources", {}).get("materializer_sha256", [])
            seen = seen if isinstance(seen, list) else [seen]
            src["materializer_sha256"] = sorted(set(seen) | set(src["materializer_sha256"]))

    ctx = {"today": today,
           "departure_window": {
               "start": (datetime.strptime(today, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"),
               "end": (datetime.strptime(today, "%Y-%m-%d")
                       + timedelta(days=CONFIG["departure_window_days"])).strftime("%Y-%m-%d")},
           "provenance": {"run_id": f"m2-{datetime.now(TZ):%Y%m%d-%H%M}",
                          "seed": a.seed, "fingerprint": fingerprint}}

    fail_f = out / "attempts.jsonl"
    fail_lock = threading.Lock()

    def ledger(code, row, attempt, msg):
        with fail_lock, fail_f.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"run": ctx["provenance"]["run_id"],
                                 "fingerprint": fingerprint[:12],
                                 "idx": row["idx"], "route": row["route"],
                                 "attempt": attempt, "code": code,
                                 "error": msg[:200]}, ensure_ascii=False) + "\n")

    # Reuse only existing records whose content and fingerprint both pass validation.
    done: set[int] = set()
    for f in sorted(out.glob("[0-9]*.json")):
        try:
            validate_record(json.loads(f.read_text(encoding="utf-8")), rows, fingerprint)
            done.add(int(f.stem))
        except Exception:
            f.unlink()
    tool_fp = hashlib.sha256(json.dumps(
        {"tools": src["tools_sha256"], "backends": src["backends_sha256"]},
        sort_keys=True).encode()).hexdigest()[:16]
    ex = Executor(out, tool_fp, _real_tools(), known=set(reg))
    todo = [r for r in picked if r["idx"] not in done]
    print(f"{len(picked)} selected, {len(done)} valid existing, {len(todo)} to materialize "
          f"(profile pool={len(pool)})")

    stuck: list[dict] = []
    counter = {"n": len(done)}
    stuck_lock = threading.Lock()

    def attempt_row(row, attempt):
        rec = materialize(row, attempt, ctx, ex, reg, pool, a.seed, ledger)
        validate_record(rec, rows, fingerprint)
        cid = rec["provenance"]["candidate_id"]
        (out / "candidates" / f"{cid}.json").write_text(
            json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        f = out / f"{row['idx']:04d}.json"
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, f)

    def work(row):
        last = None
        for attempt in range(1, CONFIG["max_attempts_per_idx"] + 1):
            try:
                attempt_row(row, attempt)
                return "ok"
            except Deferred as exc:
                last = exc
                ledger("DEFERRED", row, attempt, str(exc))
            except AssertionError as exc:
                last = exc
                ledger("GATE_REJECT", row, attempt, str(exc))
        with stuck_lock:
            stuck.append(row)
        return f"stuck: {last}"

    with ThreadPoolExecutor(max_workers=a.workers) as pool_ex:
        futs = {pool_ex.submit(work, r): r for r in todo}
        for fu in as_completed(futs):
            r = futs[fu]
            got = fu.result()
            if got == "ok":
                counter["n"] += 1
                print(f"  ok    {r['route']} #{r['idx']:04d} ({counter['n']}/{len(picked)})",
                      flush=True)
            else:
                print(f"  stuck {r['route']} #{r['idx']:04d}: {str(got)[:100]}", flush=True)

    recs = sorted(out.glob("[0-9]*.json"))
    per_route, per_wf, per_turns, per_out = {}, {}, {}, {}
    idx_to_cand, cprofiles = {}, {}
    identity_ok = True
    for f in recs:
        rec = json.loads(f.read_text(encoding="utf-8"))
        md, pv = rec["metadata"], rec["provenance"]
        per_route[md["route"]] = per_route.get(md["route"], 0) + 1
        per_wf[f"W{md['workflow']}"] = per_wf.get(f"W{md['workflow']}", 0) + 1
        per_turns[str(md["turns"])] = per_turns.get(str(md["turns"]), 0) + 1
        per_out[md["expect"]] = per_out.get(md["expect"], 0) + 1
        idx_to_cand[f.stem] = pv["candidate_id"]
        if int(f.stem) != md["idx"] or md["idx"] != pv["source_storyboard_idx"]:
            identity_ok = False
        cc = rec["context"]["current_city"]
        cprofiles[cc["name"]] = {"adcode": cc["adcode"], "weather_id": cc["weather_id"]}
    unique_cand = len(set(idx_to_cand.values())) == len(idx_to_cand)
    # Aggregate digest: SHA-256 of sorted(relative path + NUL + file-byte SHA + LF).
    agg = hashlib.sha256()
    for f in recs:
        agg.update(f.name.encode() + b"\x00" +
                   hashlib.sha256(f.read_bytes()).hexdigest().encode() + b"\n")
    complete = (len(recs) == len(picked) and identity_ok and unique_cand
                and not stuck)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "state": "complete" if complete else "partial",
        "required": len(picked), "selected": len(recs),
        "fingerprint": fingerprint, "sources": src,
        "counts_by_route": dict(sorted(per_route.items())),
        "counts_by_workflow": dict(sorted(per_wf.items())),
        "counts_by_turns": dict(sorted(per_turns.items())),
        "counts_by_outcome": dict(sorted(per_out.items())),
        "records_sha256": {p2.stem: _sha(p2) for p2 in recs},
        "aggregate_selected_digest": agg.hexdigest(),
        "idx_to_candidate_id": idx_to_cand,
        "identity_one_to_one": identity_ok,
        "unique_candidate_ids": unique_cand,
        "selected_deferred": 0 if complete else len(stuck),
        "selected_failed": 0,
        "profile_weather_mapping": cprofiles,     # Offline-verifiable: profile city -> (adcode, weather ID)
        "sealed_at": datetime.now(TZ).isoformat(),
    }
    tmp = (out / "manifest.json").with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, out / "manifest.json")
    print(f"\n{len(recs)}/{len(picked)} selected  identity={identity_ok} "
          f"unique_cand={unique_cand}  stuck={len(stuck)} -> manifest {manifest['state']}")
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
