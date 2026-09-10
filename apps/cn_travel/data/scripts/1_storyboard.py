#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build route-balanced Step 1 scenario skeletons from the route table.

The output fixes route, hidden slots, and scenario values without generating
conversation text. Hotel identities are selected during materialization from
tool evidence. Workflow quotas are 450/120/240/100/100 with a 4:1 ratio of
successful to empty-result routes.
"""
from __future__ import annotations

import json
import random
import re

from data.paths import PATHS

DOC = PATHS.governing_doc
TOTALS = {1: (450, 400, 50), 2: (120, 100, 20), 3: (240, 200, 40), 4: (100, 100, 0), 5: (100, 100, 0)}
OK_SHARE = 0.8

# W2 resolves generic facility names nearest to the user's coordinates.
FACILITIES = ["火车站", "医院", "学校", "机场", "银行"]

# W4 travel-conversation topics.
W4_TOPICS = ["greetings and introductions", "general travel questions",
             "travel advice requests", "non-specific travel topics"]
# W5 non-travel topics, including the movie topic used in the example.
W5_TOPICS = ["math", "coding", "stocks", "recipes", "health", "movies"]


def in_corpus_cities() -> list[str]:
    """Return cities represented in the guide corpus."""
    return sorted({p.name.split("_")[1] for p in PATHS.guides.glob("*_travel_guide.txt")})


def feasibility() -> dict:
    """Load cities verified to satisfy each route outcome."""
    f = PATHS.root / "material_feasibility.json"
    if not f.exists():
        raise SystemExit(f"缺少 {f.name}，先跑 scripts/feasibility.py")
    d = json.loads(f.read_text(encoding="utf-8"))
    return {"w1_ok": sorted(c for c, v in d["w1_ok"].items() if v),
            "w3_ok": sorted(c for c, v in d["w3"].items() if v["ok"]),
            "w3_empty": sorted(c for c, v in d["w3"].items() if v["empty"])}


def registry_cities() -> list[str]:
    f = PATHS.root / "city_registry.json"
    d = json.loads(f.read_text(encoding="utf-8"))
    return sorted(c for c, v in d["cities"].items() if v.get("weather_verified"))


def out_of_scope() -> tuple[list[str], list[str], list[str]]:
    """Load the verified boundary-case place lists."""
    f = PATHS.root / "china_cities_list_out_of_scope.json"
    if not f.exists():
        raise SystemExit(f"缺少 {f.name}，先跑 scripts/out_of_scope.py")
    d = json.loads(f.read_text(encoding="utf-8"))
    return (d["out_of_corpus_cities"], d["unresolvable_for_routes"],
            d["unresolvable_for_hotels"])


def read_routes() -> dict:
    """Read the route table and edge labels from the governing specification."""
    doc = DOC.read_text(encoding="utf-8")
    out = {}
    for w in range(1, 6):
        a = doc.index(f"##### **Workflow {w}:")
        b = doc.index("##### **Workflow", a + 10) if w < 5 else doc.index("\n#### ", a)
        sec = doc[a:b]
        lab = dict(re.findall(r'-->\|"?(e\d+)([^"|]*)"?\|', sec))
        rs = []
        for tag, edges, turns in re.findall(
                r"^\|\s*`?(W\d-\w+)`?\s*\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|", sec, re.M):
            e = re.findall(r"e\d+", edges)
            # The final edge label determines the route outcome: empty, error, or ok.
            tail = lab.get(e[-1], "")
            fail = next((k for k in ("empty", "error") if k in tail), None)
            rs.append({"tag": tag, "edges": e, "turns": int(turns), "fail": fail})
        out[w] = rs
    return out


def allocate(routes: list[dict], total: int, no_ask: int, ask: int) -> dict:
    """Allocate route counts by follow-up group and outcome ratio."""
    counts = {}
    for group, budget in (([r for r in routes if r["turns"] == 1], no_ask),
                          ([r for r in routes if r["turns"] > 1], ask)):
        if not group or not budget:
            continue
        mixed = any(r["fail"] for r in group) and not all(r["fail"] for r in group)
        for flag, frac in ((False, OK_SHARE), (True, 1 - OK_SHARE)):
            rs = [r for r in group if bool(r["fail"]) is flag]
            if not rs:
                continue
            n = round(budget * frac) if mixed else budget
            base, rem = divmod(n, len(rs))
            for i, r in enumerate(rs):
                counts[r["tag"]] = base + (1 if i < rem else 0)
    drift = total - sum(counts.values())
    if drift:
        counts[max(counts, key=counts.get)] += drift
    return counts


def row(w: int, r: dict, rng: random.Random, cities: list[str]) -> dict:
    """Build one text-free scenario skeleton with slots and omissions."""
    e = set(r["edges"])
    fail = r["fail"]
    slots, omit = {}, []

    if w == 1:
        slots["city"] = rng.choice(OUT_OF_CORPUS if fail else FEAS["w1_ok"])
        slots["in_corpus"] = not fail
        if "e1" in e:
            omit.append("city")
        if "e4" in e:
            omit.append("date")
    elif w == 2:
        if fail:
            slots["destination"] = rng.choice(NO_ROUTE)
        elif rng.random() < 0.5:
            # Both endpoints of an intercity route are unambiguous registry cities.
            a, b = rng.sample(REG_CITIES, 2)
            slots["origin"], slots["destination"] = a, b
        else:
            # Resolve generic facility names nearest to the user's coordinates.
            slots["destination"] = rng.choice(FACILITIES)
        slots["resolvable"] = not fail
        if "e1" in e:
            omit.append("destination")
    elif w == 3:
        # Step 1 cannot choose the hotel name. It must be a real hotel in the city
        # with retrievable reviews, as selected by get_hotel_recommendations().
        slots["path"] = "reviews_only" if e & {"e3", "e5", "e6"} else "recommend"
        if fail == "error":
            # Like query_routes, get_hotel_recommendations returns error when the
            # city cannot be resolved. Real cities always have hotels, so only an
            # unresolvable place name can exercise this branch.
            slots["city"] = rng.choice(NO_HOTEL)
            slots["resolvable"] = False
        else:
            slots["city"] = rng.choice(FEAS["w3_empty"] if fail == "empty" else FEAS["w3_ok"])
            if fail == "empty":
                # Reviews are empty, not hotels. Step 2 must choose a hotel without
                # public reviews; freezing a name here would fabricate an entity.
                slots["reviews"] = "empty"
        if "e1" in e:
            omit.append("hotel")

    return {"workflow": w, "route": r["tag"], "edges": r["edges"], "turns": r["turns"],
            "expect": fail or "ok", "slots": slots, "omit": omit}


def main() -> None:
    rng = random.Random(20250914)          # Fixed seed for reproducible storyboards.
    cities = in_corpus_cities()
    global OUT_OF_CORPUS, NO_ROUTE, NO_HOTEL, FEAS, REG_CITIES
    OUT_OF_CORPUS, NO_ROUTE, NO_HOTEL = out_of_scope()
    FEAS = feasibility()
    REG_CITIES = registry_cities()
    routes = read_routes()
    rows = []
    for w, rs in routes.items():
        counts = allocate(rs, *TOTALS[w])
        for r in rs:
            for i in range(counts.get(r["tag"], 0)):
                item = row(w, r, rng, cities)
                # The topic is the refusal/chat sample's story; rotation covers every class.
                if w == 4:
                    item["slots"]["topic"] = W4_TOPICS[i % len(W4_TOPICS)]
                elif w == 5:
                    item["slots"]["topic"] = W5_TOPICS[i % len(W5_TOPICS)]
                rows.append(item)
    rng.shuffle(rows)

    out = PATHS.root / "1_storyboard.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    per = {}
    for r in rows:
        per[r["route"]] = per.get(r["route"], 0) + 1
    print(f"{len(rows)} rows -> {out}")
    for w in range(1, 6):
        tags = {k: v for k, v in sorted(per.items()) if k.startswith(f"W{w}-")}
        print(f"  W{w} {sum(tags.values()):>4}  " + "  ".join(f"{k}={v}" for k, v in tags.items()))


if __name__ == "__main__":
    main()
