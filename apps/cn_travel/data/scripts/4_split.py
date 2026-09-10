#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create the evidence-grouped 9:1 Step 4 split.

The 909 training/validation and 101 leaderboard files remain byte-identical to
their Step 3 sources. Shared tool-result evidence stays on one side, all 24
routes appear on both sides, and workflow and outcome quotas match README §4.2.
"""
import hashlib
import json
import shutil
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from data.paths import PATHS

S2 = PATHS.root / "2_materialized"
S3 = PATHS.root / "3_conversations"
OUT = PATHS.root / "4_split"
N = 1010

# Per-route leaderboard targets defined in README §4.2; W2 uses A1/B8/C2/D1.
LB_TARGET = {
    "W1-A": 32, "W1-B": 6, "W1-C": 1, "W1-D": 2, "W1-E": 1, "W1-F": 1, "W1-G": 1, "W1-H": 1,
    "W2-A": 1, "W2-B": 8, "W2-C": 2, "W2-D": 1,
    "W3-A": 1, "W3-B": 8, "W3-C": 1, "W3-D": 8, "W3-E": 1, "W3-F": 1, "W3-G": 1, "W3-H": 1,
    "W3-I": 1, "W3-J": 1, "W4-A": 10, "W5-A": 10,
}
ROUTES = list(LB_TARGET)


def _atom(t: dict) -> str:
    # Evidence is the model-visible result; arguments, execution_id, and ordinal are excluded.
    return hashlib.sha256(json.dumps(t["result"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def evidence_groups() -> tuple[list[list[int]], dict]:
    """Group indices sharing tool-result evidence and return their routes."""
    route, atom_to_idxs = {}, defaultdict(set)
    for i in range(N):
        rec = json.loads((S2 / f"{i:04d}.json").read_text(encoding="utf-8"))
        route[i] = rec["metadata"]["route"]
        for t in rec["visible_tool_trace"]:
            atom_to_idxs[_atom(t)].add(i)
    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for idxs in atom_to_idxs.values():
        idxs = sorted(idxs)
        for j in idxs[1:]:
            ra, rb = find(idxs[0]), find(j)
            if ra != rb:
                parent[ra] = rb
    groups = defaultdict(list)
    for i in range(N):
        groups[find(i)].append(i)
    return sorted(sorted(v) for v in groups.values()), route


def choose_leaderboard(glist: list[list[int]], route: dict) -> set[int]:
    ridx = {r: k for k, r in enumerate(ROUTES)}
    A = np.zeros((len(ROUTES), len(glist)))
    for gi, mem in enumerate(glist):
        for m in mem:
            A[ridx[route[m]], gi] += 1
    b = np.array([LB_TARGET[r] for r in ROUTES], float)
    res = milp(c=np.zeros(len(glist)),
               constraints=[LinearConstraint(A, b, b)],
               integrality=np.ones(len(glist)), bounds=Bounds(0, 1))
    if res.x is None:
        raise SystemExit(f"证据组配额不可行：{res.message}")
    picked = np.round(res.x).astype(int)
    return {m for gi, mem in enumerate(glist) if picked[gi] == 1 for m in mem}


def verify(route: dict, lb: set[int], glist: list[list[int]]) -> None:
    tv = set(range(N)) - lb
    tv_files = sorted((OUT / "train_validation").glob("[0-9]*.json"))
    lb_files = sorted((OUT / "leaderboard_eval").glob("[0-9]*.json"))
    assert len(tv_files) == 909 and len(lb_files) == 101, "两侧文件数不是 909/101"
    assert {int(f.stem) for f in tv_files} == tv and {int(f.stem) for f in lb_files} == lb
    assert tv.isdisjoint(lb) and (tv | lb) == set(range(N)), "两侧不构成 0..1009 划分"
    for side, members in (("train_validation", tv), ("leaderboard_eval", lb)):
        for i in members:
            assert (OUT / side / f"{i:04d}.json").read_bytes() == (S3 / f"{i:04d}.json").read_bytes(), \
                f"{side}/{i:04d}.json 与 Step 3 不逐字节相同"
    # Enforce each route target and representation on both sides.
    lbc, tvc = Counter(route[i] for i in lb), Counter(route[i] for i in tv)
    for r in ROUTES:
        assert lbc[r] == LB_TARGET[r], f"{r} leaderboard {lbc[r]} != {LB_TARGET[r]}"
        assert tvc[r] >= 1 and lbc[r] >= 1, f"{r} 未在两侧都出现"
    # Evidence groups cannot cross the split: no result appears on both sides.
    for mem in glist:
        assert len({i in lb for i in mem}) == 1, f"证据组 {mem[:3]}... 被拆到两侧"
    # Aggregates preserve the 9:1 workflow split, fixed outcome counts, and 86/13/2 turn counts.
    def agg(members, key):
        c = Counter()
        for i in members:
            c[key(json.loads((S3 / f"{i:04d}.json").read_text(encoding="utf-8"))["metadata"])] += 1
        return c
    assert agg(lb, lambda m: f"W{m['workflow']}") == Counter({"W1": 45, "W2": 12, "W3": 24, "W4": 10, "W5": 10})
    assert agg(tv, lambda m: f"W{m['workflow']}") == Counter({"W1": 405, "W2": 108, "W3": 216, "W4": 90, "W5": 90})
    assert agg(lb, lambda m: m["expect"]) == Counter({"ok": 82, "empty": 14, "error": 5})
    assert agg(tv, lambda m: m["expect"]) == Counter({"ok": 766, "empty": 109, "error": 34})
    assert agg(lb, lambda m: m["turns"]) == Counter({1: 86, 2: 13, 3: 2})
    assert agg(tv, lambda m: m["turns"]) == Counter({1: 814, 2: 81, 3: 14})


def main() -> int:
    glist, route = evidence_groups()
    lb = choose_leaderboard(glist, route)
    tv = set(range(N)) - lb
    for side in ("train_validation", "leaderboard_eval"):
        d = OUT / side
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    for side, members in (("train_validation", tv), ("leaderboard_eval", lb)):
        for i in members:
            shutil.copyfile(S3 / f"{i:04d}.json", OUT / side / f"{i:04d}.json")
    verify(route, lb, glist)
    print(f"Step 4 split OK: train_validation {len(tv)}, leaderboard_eval {len(lb)} "
          f"({len(glist)} evidence groups by result; 0 evidence crosses sides; "
          f"workflow 9:1 + all-24-routes-both-sides + byte-identity verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
