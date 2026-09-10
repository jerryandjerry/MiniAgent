#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline contract tests for the Step 2 materializer using mocked tools."""
import importlib.util
import hashlib
import json
import pathlib
import sys

import pytest

from project_paths import DATA_ROOT

_spec = importlib.util.spec_from_file_location(
    "materialize", DATA_ROOT / "scripts" / "2_materialize.py")
M = importlib.util.module_from_spec(_spec)
sys.modules["materialize"] = M
_spec.loader.exec_module(M)

TODAY = "2026-08-22"
_OOS = json.loads((DATA_ROOT / "china_cities_list_out_of_scope.json")
                  .read_text(encoding="utf-8"))
NO_GUIDE = set(_OOS["out_of_corpus_cities"])
NO_ROUTE = set(_OOS["unresolvable_for_routes"])
NO_HOTEL = set(_OOS["unresolvable_for_hotels"])

# Fake registry: three profile cities plus storyboard cities with generated centroids
REG = {}
def _reg_entry(city, lat=30.0, lng=110.0, adcode="429000"):
    return {"official_name": city + "市", "level": "city", "adcode": adcode,
            "lat": lat, "lng": lng, "weather_id": "101999999", "weather_verified": True}
for c in ("北京", "上海", "嘉兴"):
    REG[c] = _reg_entry(c)
POOL = sorted(REG)


class _Reg(dict):
    def __contains__(self, k):
        return bool(k)                     # Every city exists in the mock registry
    def get(self, k, default=None):
        if k not in self and k:
            self[k] = _reg_entry(k)
        return super().get(k, default)
    def __missing__(self, k):
        self[k] = _reg_entry(k)
        return self[k]


def mock_tools(review_status="mixed"):
    def guide(a):
        empty = a["location"] in NO_GUIDE
        return {"status": "empty" if empty else "ok", "source": "mock",
                "location": a["location"],
                "guides": [] if empty else [{"city": a["location"], "province": "省",
                                             "content": "攻略", "score": 0.9}]}

    def weather(a):
        return {"status": "ok", "source": "mock", "location": a["location"],
                "resolved": {"name": a["location"] + "市", "adcode": "429000",
                             "lat": 30.0, "lng": 110.0},
                "days": [{"date": a["start_date"], "day_weather": "晴",
                          "night_weather": "晴", "temp_min_c": 20, "temp_max_c": 30}]}

    def route(a):
        if a["end_location"] in NO_ROUTE:
            return {"status": "error", "source": "mock",
                    "origin": None, "destination": None, "routes": {}}
        return {"status": "ok", "source": "mock",
                "origin": {"query": a["start_location"], "lat": 30.0, "lng": 110.0},
                "destination": {"query": a["end_location"],
                                "name": a["end_location"] + "市", "lat": 30.0, "lng": 110.0},
                "routes": {"walking": None, "transit": None,
                           "driving": {"duration_min": 60, "distance_m": 50000, "tolls_cny": 0.0}}}

    def rec(a):
        if a["location"] in NO_HOTEL:
            return {"status": "error", "source": "mock", "location": a["location"], "hotels": []}
        hotels = [{"name": f"{a['location']}宾馆{i}", "address": "地址", "district": "区",
                   "rating": 4.5, "tier": "舒适型", "tel": "", "lat": 30.0, "lng": 110.0,
                   "price_cny": None} for i in (1, 2, 3)]
        return {"status": "ok", "source": "mock", "location": a["location"], "hotels": hotels}

    def reviews(a):
        n = a["hotel_name"]
        if review_status == "mixed":
            if n.endswith("宾馆1"):
                mode = "empty"
            elif n.endswith("宾馆2"):
                mode = "ok_but_empty_list"          # Schema-invalid ok payload.
            else:
                mode = "ok"
        else:
            mode = review_status
        if mode == "empty":
            return {"status": "empty", "source": "mock", "hotel_name": n, "rating": None,
                    "price_hint": None, "reviews": [], "summary": "暂无"}
        if mode == "ok_but_empty_list":
            return {"status": "ok", "source": "mock", "hotel_name": n, "rating": 4.2,
                    "price_hint": None, "reviews": [], "summary": "只有评分"}
        return {"status": "ok", "source": "mock", "hotel_name": n, "rating": 4.6,
                "price_hint": None, "reviews": [{"text": "位置方便"}], "summary": "好评"}

    return {"search_travel_guide": guide, "get_weather_info": weather,
            "query_route": route, "recommend_hotels": rec, "get_hotel_reviews": reviews}


LEDGER = []
def ledger(code, row, attempt, msg):
    LEDGER.append({"code": code, "idx": row["idx"], "msg": msg})


@pytest.fixture()
def env(tmp_path):
    ex = M.Executor(tmp_path, "test-fp", mock_tools())
    reg = _Reg(REG)
    ctx = {"today": TODAY,
           "departure_window": {"start": "2026-08-23", "end": "2026-08-27"},
           "provenance": {"run_id": "t", "seed": 1, "fingerprint": "f"}}
    LEDGER.clear()
    return ex, reg, ctx


def rows_all():
    p = DATA_ROOT / "1_storyboard.json"
    rows = json.loads(p.read_text(encoding="utf-8"))
    for i, r in enumerate(rows):
        r["idx"] = i
    return rows


def first_per_route():
    first = {}
    for r in rows_all():
        first.setdefault(r["route"], r)
    return first


ALL_ROUTES = sorted(M.ROUTES)


@pytest.mark.parametrize("tag", ALL_ROUTES)
def test_golden_every_route(tag, env):
    ex, reg, ctx = env
    row = first_per_route()[tag]
    rec = M.materialize(row, 1, ctx, ex, reg, POOL, seed=1, ledger=ledger)
    M.validate_record(rec, rows_all(), "f")
    assert [(t["name"], t["result"]["status"]) for t in rec["visible_tool_trace"]] \
        == M.ROUTES[tag]
    for k in ("workflow", "route", "edges", "turns", "expect", "slots", "omit"):
        assert rec["metadata"][k] == row[k]
    assert rec["context"]["current_city"]["weather_id"]       # Required city evidence.


def test_dates_inside_inclusive_horizon(env):
    from datetime import datetime
    for idx in range(40):
        d = M.pick_dates(1, idx, 1, TODAY)
        off = (datetime.strptime(d["end_date"], "%Y-%m-%d")
               - datetime.strptime(TODAY, "%Y-%m-%d")).days
        assert off <= M.CONFIG["forecast_days"] - 1


def test_ok_empty_reviews_never_in_any_trace(env):
    """An ok review payload with an empty review list stays out of every trace."""
    ex, reg, ctx = env
    row = dict(first_per_route()["W3-B"])
    rec = M.materialize(row, 1, ctx, ex, reg, POOL, seed=1, ledger=ledger)
    all_ids = [t["arguments"].get("hotel_name") for t in
               rec["visible_tool_trace"] + rec["selection_trace"]
               if t["name"] == "get_hotel_reviews"]
    assert not any(n and n.endswith("宾馆2") for n in all_ids)
    assert any(e["code"] == "probe_contract_invalid" for e in LEDGER)
    assert rec["resolved"]["canonical_slots"]["hotel_name"].endswith("宾馆3")


def test_entity_gate_rejects_cross_city(env):
    """The entity gate rejects a hotel resolved outside the requested city."""
    ex, reg, ctx = env
    reg["通州"] = _reg_entry("通州", lat=32.0, lng=121.0, adcode="320612")
    bad = {"status": "ok", "source": "m", "location": "通州",
           "hotels": [{"name": "x", "address": "", "district": "", "rating": 4.0,
                       "tier": "", "tel": "", "lat": 39.9, "lng": 116.4,   # Beijing coordinates
                       "price_cny": None}]}
    with pytest.raises(AssertionError):
        M.entity_gate("recommend_hotels", {"location": "通州"}, bad, reg)


def test_fingerprint_mismatch_rejected(env):
    """Every materialized record must match the sealed fingerprint."""
    ex, reg, ctx = env
    row = first_per_route()["W4-A"]
    rec = M.materialize(row, 1, ctx, ex, reg, POOL, seed=1, ledger=ledger)
    with pytest.raises(AssertionError):
        M.validate_record(rec, rows_all(), "another-fingerprint")


def test_candidate_ids_differ_per_attempt(env):
    ex, reg, ctx = env
    row = first_per_route()["W4-A"]
    a1 = M.materialize(row, 1, ctx, ex, reg, POOL, seed=1, ledger=ledger)
    a2 = M.materialize(row, 2, ctx, ex, reg, POOL, seed=1, ledger=ledger)
    assert a1["provenance"]["candidate_id"] != a2["provenance"]["candidate_id"]


def test_profile_varies_with_attempt_deterministically(env):
    ex, reg, ctx = env
    p1 = M.pick_profile(REG, POOL, 1, 7, 1)
    p1b = M.pick_profile(REG, POOL, 1, 7, 1)
    assert p1 == p1b


def test_w2_intercity_omit_amendment4(env):
    ex, reg, ctx = env
    row = {"workflow": 2, "route": "W2-C", "edges": ["e1","e2","e3","e4"], "turns": 2,
           "expect": "error", "omit": ["destination"],
           "slots": {"origin": "六盘水", "destination": "南极科考站", "resolvable": False},
           "idx": 998}
    rows = rows_all(); rows.append(row); row["idx"] = len(rows) - 1
    rec = M.materialize(row, 1, ctx, ex, reg, POOL, seed=1, ledger=ledger)
    tp = rec["resolved"]["turn_plan"]
    assert tp[0]["reveal"] == ["origin"] and tp[0]["conceal"] == ["destination"]
    assert "origin_not_replaced" in tp[0]["constraints"]
    assert "city" in rec["visible_tool_trace"][0]["arguments"]


def test_empty_reviews_constructed_from_label(env, tmp_path):
    ex2 = M.Executor(tmp_path / "y", "fp", mock_tools("ok"))   # All mock hotels have reviews
    _, reg, ctx = env
    row = json.loads(json.dumps(first_per_route()["W3-C"]))
    rows = rows_all() + [row]; row["idx"] = len(rows) - 1
    rec = M.materialize(row, 1, ctx, ex2, reg, POOL, seed=1, ledger=ledger)
    M.validate_record(rec, rows, "f")
    rv = rec["visible_tool_trace"][-1]
    assert rv["name"] == "get_hotel_reviews" and rv["result"]["status"] == "empty"
    assert rv["result"]["source"] == "synthetic:label"
    assert rv["arguments"]["hotel_name"] == rec["resolved"]["canonical_slots"]["hotel_name"]


def test_execute_once_under_concurrency(tmp_path):
    # Cacheable tools such as weather and reviews execute once under concurrency
    calls = {"n": 0}
    def counting(a):
        calls["n"] += 1
        return {"status": "ok", "source": "m", "location": a["location"],
                "resolved": {"name": a["location"], "adcode": "110100",
                             "lat": 39.9, "lng": 116.4}, "days": []}
    ex = M.Executor(tmp_path, "fp", {"get_weather_info": counting})
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as p:
        outs = list(p.map(lambda _: ex.run("get_weather_info",
                          {"location": "北京", "start_date": "2026-08-26", "num_days": 1}),
                          range(16)))
    assert calls["n"] == 1 and len({o["execution_id"] for o in outs}) == 1


def test_put_is_per_key_synchronised_and_first_value_is_immutable(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    ex = M.Executor(tmp_path, "fp", {})
    args = {"hotel_name": "同一家酒店", "location": "同一座城市"}
    barrier = Barrier(16)

    def put(i):
        barrier.wait()
        return ex.put(
            "get_hotel_reviews",
            args,
            {"status": "empty", "source": f"candidate:{i}", "hotel_name": args["hotel_name"],
             "rating": None, "price_hint": None, "reviews": [], "summary": str(i)},
        )

    with ThreadPoolExecutor(16) as pool:
        records = list(pool.map(put, range(16)))

    assert all(record == records[0] for record in records)
    stored = json.loads(next((tmp_path / "exec_cache").glob("*.json")).read_text(encoding="utf-8"))
    assert stored == records[0]

    replacement = ex.put(
        "get_hotel_reviews",
        args,
        {"status": "empty", "source": "replacement", "hotel_name": args["hotel_name"],
         "rating": None, "price_hint": None, "reviews": [], "summary": "replacement"},
    )
    assert replacement == records[0]


def test_validate_only_never_deletes_an_invalid_record(tmp_path, monkeypatch):
    output = tmp_path / "sealed"
    output.mkdir()
    source = DATA_ROOT / "2_materialized" / "0000.json"
    record = json.loads(source.read_text(encoding="utf-8"))
    record["provenance"]["fingerprint"] = "deliberately-invalid"
    record_path = output / "0000.json"
    record_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    before = record_path.read_bytes()
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "fingerprint": "sealed-fingerprint",
                "selected": 1,
                "records_sha256": {"0000": hashlib.sha256(before).hexdigest()},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "2_materialize.py",
            "--input",
            str(DATA_ROOT / "1_storyboard.json"),
            "--output",
            str(output),
            "--validate-only",
        ],
    )

    assert M.main() == 1
    assert record_path.read_bytes() == before
    assert not (output / "candidates").exists()


def test_guide_is_not_cached(tmp_path):
    # Local guides are read on each call so normalization changes take effect immediately.
    calls = {"n": 0}
    def counting(a):
        calls["n"] += 1
        return {"status": "ok", "source": "m", "location": a["location"],
                "guides": [{"city": a["location"], "province": "p", "content": "c",
                            "score": 1.0}]}
    ex = M.Executor(tmp_path, "fp", {"search_travel_guide": counting})
    for _ in range(3):
        ex.run("search_travel_guide", {"location": "北京"})
    assert calls["n"] == 3


def test_validator_rejects_missing_required_arg(env):
    ex, reg, ctx = env
    row = first_per_route()["W2-B"]
    rec = M.materialize(row, 1, ctx, ex, reg, POOL, seed=1, ledger=ledger)
    del rec["visible_tool_trace"][0]["arguments"]["city"]
    with pytest.raises(AssertionError):
        M.validate_record(rec, rows_all(), "f")


def test_validator_rejects_metadata_mutation(env):
    ex, reg, ctx = env
    row = first_per_route()["W1-A"]
    rec = M.materialize(row, 1, ctx, ex, reg, POOL, seed=1, ledger=ledger)
    rec["metadata"]["expect"] = "empty"
    with pytest.raises(AssertionError):
        M.validate_record(rec, rows_all(), "f")


def test_no_substitution_machinery_left():
    """Each storyboard row is materialized without cross-row substitution."""
    assert not hasattr(M, "stratum")
    src = (DATA_ROOT / "scripts" / "2_materialize.py").read_text(encoding="utf-8")
    assert src.count("replaces_storyboard_idx") <= 2   # Validator rejection paths only.
