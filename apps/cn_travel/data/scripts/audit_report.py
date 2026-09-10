#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only validation of sealed Step 2 and Step 3 artifacts."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import re
import sys
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo

from data.paths import PATHS

TZ = ZoneInfo("America/Chicago")
QUOTA = {"W1-A": 320, "W1-B": 80, "W1-C": 14, "W1-D": 4, "W1-E": 13, "W1-F": 3, "W1-G": 13, "W1-H": 3,
         "W2-A": 20, "W2-B": 80, "W2-C": 4, "W2-D": 16,
         "W3-A": 14, "W3-B": 80, "W3-C": 13, "W3-D": 80, "W3-E": 13, "W3-F": 3, "W3-G": 16,
         "W3-H": 3, "W3-I": 16, "W3-J": 2, "W4-A": 100, "W5-A": 100}
AGG = {"workflow": {"W1": 450, "W2": 120, "W3": 240, "W4": 100, "W5": 100},
       "turns": {"1": 900, "2": 94, "3": 16},
       "outcome": {"ok": 848, "empty": 123, "error": 39}}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def tree_digest(d: pathlib.Path) -> tuple[int, int, str]:
    files = sorted(p for p in d.rglob("*") if p.is_file())
    h = hashlib.sha256(); size = 0
    for p in files:
        size += p.stat().st_size
        h.update(str(p.relative_to(d)).encode() + b"\x00" + sha(p).encode() + b"\n")
    return len(files), size, h.hexdigest()


def main() -> int:
    M = _load("m2", PATHS.root / "scripts" / "2_materialize.py")
    C = _load("c3", PATHS.root / "scripts" / "3_conversation.py")
    rows = json.loads((PATHS.root / "1_storyboard.json").read_text(encoding="utf-8"))
    for i, r in enumerate(rows):
        r["idx"] = i
    d2, d3 = PATHS.root / "2_materialized", PATHS.root / "3_conversations"
    m2 = json.loads((d2 / "manifest.json").read_text(encoding="utf-8"))
    m3 = json.loads((d3 / "manifest.json").read_text(encoding="utf-8"))
    gov = PATHS.governing_doc
    req: list[tuple[str, str, str]] = []          # (requirement, PASS/FAIL, evidence)
    fails: dict[str, list[str]] = {}

    def check(name, ok_set, bad, evidence=""):
        req.append((name, "PASS" if not bad else "FAIL",
                    evidence or (f"{ok_set}/1010" if not bad else f"{len(bad)} affected")))
        if bad:
            fails[name] = sorted(bad)

    # ---- Snapshot
    n2, b2, t2 = tree_digest(d2)
    n3, b3, t3 = tree_digest(d3)
    recs2 = sorted(d2.glob("[0-9]*.json")); recs3 = sorted(d3.glob("[0-9]*.json"))
    stems = [f"{i:04d}" for i in range(1010)]
    check("Exactly 0000–1009 (Step 2)", 1010, [s for s in stems if not (d2 / f"{s}.json").exists()]
          + [p.stem for p in recs2 if p.stem not in stems])
    check("Exactly 0000–1009 (Step 3)", 1010, [s for s in stems if not (d3 / f"{s}.json").exists()]
          + [p.stem for p in recs3 if p.stem not in stems])

    # ---- Manifest hash reproduction
    bad = [s for s, h in m2["records_sha256"].items() if not (d2 / f"{s}.json").exists() or sha(d2 / f"{s}.json") != h]
    check("Step 2 manifest hashes reproduce bytes", 1010, bad)
    bad = [s for s, h in m3["records_sha256"].items() if not (d3 / f"{s}.json").exists() or sha(d3 / f"{s}.json") != h]
    check("Step 3 manifest hashes reproduce bytes", 1010, bad)

    # ---- Step 2 records: schema, validator, lineage, and candidate uniqueness
    bad_v, bad_id, cands, per = [], [], Counter(), Counter()
    tuples: dict = {}
    for p in recs2:
        rec = json.loads(p.read_text(encoding="utf-8"))
        try:
            M.validate_record(rec, rows, m2["fingerprint"])
        except Exception as e:
            bad_v.append(p.stem)
        md, pv = rec["metadata"], rec["provenance"]
        if int(p.stem) != md["idx"] or md["idx"] != pv["source_storyboard_idx"]:
            bad_id.append(p.stem)
        cands[pv["candidate_id"]] += 1
        per[md["route"]] += 1
        cc = rec["context"]["current_city"]
        tuples.setdefault(cc["name"], set()).add((cc["adcode"], cc["weather_id"]))
    check("Step 2 offline validator (schema, trace, contracts, entity gates)", 1010, bad_v)
    check("One-to-one lineage: filename = metadata.idx = source idx (DI-009)", 1010, bad_id)
    dup = [c for c, n in cands.items() if n > 1]
    check("Unique selected candidate IDs", 1010, dup)
    check("Fixed 24-route quota (Step 2)", 1010, [r for r, n in QUOTA.items() if per.get(r, 0) != n])
    bad_t = [c for c, s in tuples.items() if len(s) > 1]
    bj = [c for c, s in tuples.items() if c.rstrip("市") != "北京" and any(w == "101010100" for _, w in s)]
    check("Profile tuples internally consistent; no Beijing fallback", len(tuples), bad_t + bj,
          f"{len(tuples)} distinct profile tuples")
    mp = m2.get("profile_weather_mapping") or {}
    check("Manifest carries profile/weather mapping payload", len(mp), [] if mp else ["missing"])
    check("Manifest idx→candidate map present & one-to-one", 1010,
          [] if m2.get("identity_one_to_one") and m2.get("unique_candidate_ids") and
          len(m2.get("idx_to_candidate_id", {})) == 1010 else ["manifest"])
    check("Manifest asserts zero selected deferred/failed", 1010,
          [] if m2.get("selected_deferred") == 0 and m2.get("selected_failed") == 0 else ["manifest"])

    # ---- Step 3 records
    bad_c, bad_meta, per3 = [], [], Counter()
    for p in recs3:
        conv = json.loads(p.read_text(encoding="utf-8"))
        rec = json.loads((d2 / p.name).read_text(encoding="utf-8"))
        if conv["metadata"] != rec["metadata"] or int(p.stem) != conv["metadata"]["idx"]:
            bad_meta.append(p.stem)
        try:
            msgs = conv["conversation"]
            turns = [m["content"] for m in msgs if m["role"] == "user"]
            C.check(rec, turns, msgs[-1]["content"])
            cc = rec["context"]["current_city"]
            assert f"当前城市ID: {cc['weather_id']}" in msgs[0]["content"]
            assert "城际出行" in msgs[0]["content"], "system policy lacks local/intercity rule"
        except Exception:
            bad_c.append(p.stem)
        per3[conv["metadata"]["route"]] += 1
    check("Step 3 metadata value-identical & same index as Step 2", 1010, bad_meta)
    check("Step 3 content validator (turn plan, dates, entities, W1 daily weather, W2/W3/W4 rules, policy)",
          1010, bad_c)
    check("Fixed 24-route quota (Step 3)", 1010, [r for r, n in QUOTA.items() if per3.get(r, 0) != n])
    check("Step 3 manifest links Step 2 seal", 1, [] if m3.get("step2_fingerprint") == m2["fingerprint"]
          and m3.get("step2_manifest_sha256") == sha(d2 / "manifest.json") else ["link"])
    check("Both manifests state complete", 2, [k for k, m in (("step2", m2), ("step3", m3))
                                               if m.get("state") != "complete"])

    verdict = "ACCEPT" if all(r[1] == "PASS" for r in req) else "REJECT"
    today = datetime.now(TZ)
    print(f"SELF-CHECK {verdict}  ({sum(1 for r in req if r[1]=='PASS')}/{len(req)} requirements)")
    if fails:
        for k, v in fails.items():
            print(f"  {k}: {' '.join(v[:40])}{' …' if len(v) > 40 else ''}")
    for n, r, e in req:
        print(f"  {r:4} {n} — {e}")
    return 0 if verdict == "ACCEPT" else 1


if __name__ == "__main__":
    sys.exit(main())
