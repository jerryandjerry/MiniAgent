#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic scoring and leaderboard publication for the sealed evaluation.

Each run contains ``run.json`` and 101 transcripts. Scoring emits per-task
records plus summary slices by workflow, route, and expected outcome.

Arguments use strict JSON equality after the registered location and hotel-class
canonicalization. Trace match requires exact call order, assistant-turn structure,
and a valid endpoint for every user segment. Tool-call partial credit is the
order-sensitive longest common subsequence. Complete evidence and the sealed
six-round protocol are required for leaderboard publication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import sys

try:
    from .paths import PATHS
    from .world import normalize_tool_args, project_args
except ImportError:  # Direct ``python eval/score.py`` invocation.
    from paths import PATHS
    from world import normalize_tool_args, project_args

EVAL_SET = PATHS.eval_set
EVAL_DIR = PATHS.root
DATASET_NAME = "apps/cn_travel/data/6_multiturn/leaderboard_eval.json"
SEALED_CASES = 101
SEALED_MAX_ROUNDS = 6

# ---------------------------------------------------------------- golden --
def golden_segments(conversation: list[dict]) -> list[dict]:
    """Split a sealed conversation on user turns; within a segment, keep the call-bearing assistant turn structure."""
    segs, cur = [], None
    for msg in conversation:
        role = msg.get("role")
        if role == "user":
            if cur is not None:
                segs.append(cur)
            cur = {"user": msg["content"], "rounds": [], "calls": []}
        elif role == "assistant" and cur is not None:
            calls = [{"tool": tc["function"]["name"],
                      "args": json.loads(tc["function"]["arguments"])}
                     for tc in msg.get("tool_calls") or []]
            if calls:
                cur["rounds"].append(calls)
                cur["calls"].extend(calls)
    if cur is not None:
        segs.append(cur)
    return segs


def load_eval_set() -> dict[int, dict]:
    rows = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    out = {}
    for row in rows:
        m = row["metadata"]
        segs = golden_segments(row["conversation"])
        for s in segs:                       # golden arguments must be readable by the world - golden-set discipline
            for c in s["calls"]:
                assert project_args(c["tool"], c["args"]) is not None, \
                    f"idx {m['idx']}: golden call {c['tool']} has world-invalid args"
        out[m["idx"]] = {"workflow": f"W{m['workflow']}", "route": m["route"],
                         "expect": m["expect"], "turns": m["turns"], "segments": segs}
    return out


def dataset_sha256() -> str:
    return hashlib.sha256(EVAL_SET.read_bytes()).hexdigest()


# --------------------------------------------------------------- matching --
def _norm_args(args: dict, tool: str | None = None) -> dict:
    """Apply the shared closed canonicalization; all remaining JSON stays exact."""
    return normalize_tool_args(tool or "", args)


def _json_eq(a, b) -> bool:
    """Strict JSON-typed equality: bool and number are different types (true != 1); recurses into containers."""
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_eq(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_eq(x, y) for x, y in zip(a, b))
    if a != b:
        return False
    # equal values must also share a JSON type; int and float are both JSON number, so they match
    return type(a) is type(b) or (isinstance(a, (int, float)) and isinstance(b, (int, float)))


def _compare(golden_args: dict, made_args, tool: str | None = None) -> str | None:
    """None = equal; otherwise the reason for failure. The whole object is compared with strict JSON typing (key set included)."""
    if not isinstance(made_args, dict):
        return "unparseable_args"
    ng, nm = _norm_args(golden_args, tool), _norm_args(made_args, tool)
    if _json_eq(ng, nm):
        return None
    for key in sorted(set(ng) | set(nm)):    # deterministic: the first differing field in lexicographic order
        if key not in ng or key not in nm or not _json_eq(ng[key], nm[key]):
            return f"wrong_args:{key}"
    return "wrong_args:?"


def _call_eq(g: dict, mc: dict) -> bool:
    return (mc.get("tool") == g["tool"]
            and _compare(g["args"], mc.get("arguments"), g["tool"]) is None)


def _lcs_pairs(G: list[dict], M: list[dict]) -> list[tuple[int, int]]:
    """Longest-common-subsequence pairing of golden against actual (order-sensitive, leftmost-first, deterministic)."""
    n, m = len(G), len(M)
    L = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            L[i][j] = (L[i + 1][j + 1] + 1 if _call_eq(G[i], M[j])
                       else max(L[i + 1][j], L[i][j + 1]))
    pairs, i, j = [], 0, 0
    while i < n and j < m:
        if _call_eq(G[i], M[j]) and L[i][j] == L[i + 1][j + 1] + 1:
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif L[i + 1][j] >= L[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def _made_rounds(made_calls: list[dict]) -> list[list[dict]]:
    """Group the actual calls by model turn as recorded in the transcript (call-bearing turns only, in order of appearance)."""
    rounds, cur_key = [], object()
    for mc in made_calls:
        key = mc.get("round", 1)
        if key != cur_key:
            rounds.append([])
            cur_key = key
        rounds[-1].append(mc)
    return rounds


def match_segment(golden_rounds: list[list[dict]], made_calls: list[dict],
                  seg: int) -> tuple[list[dict], int, int, bool]:
    """Returns (call_detail, calls_correct, calls_extra, whether the segment structure matches golden exactly)."""
    G = [c for rnd in golden_rounds for c in rnd]
    pairs = _lcs_pairs(G, made_calls)
    g_paired = {i: j for i, j in pairs}
    m_used = {j for _, j in pairs}

    # structural check: the call-bearing turn sequence must equal golden turn for turn and position for position (order, grouping and arguments all identical)
    mr = _made_rounds(made_calls)
    structure_ok = (len(mr) == len(golden_rounds) and all(
        len(mr[r]) == len(golden_rounds[r]) and
        all(_call_eq(g, mc) for g, mc in zip(golden_rounds[r], mr[r]))
        for r in range(len(mr))))

    detail: list[dict] = []
    correct = len(pairs)
    for i, g in enumerate(G):
        req = {"tool": g["tool"], "args_truth": g["args"]}
        if i in g_paired:
            mc = made_calls[g_paired[i]]
            detail.append({"segment": seg, "required": req,
                           "made": {"tool": mc["tool"], "args": mc.get("arguments")},
                           "correct": True})
            continue
        reason, made = "missing", None
        for j, mc in enumerate(made_calls):          # give a specific reason for an unpaired call on the same tool
            if j in m_used or mc.get("tool") != g["tool"]:
                continue
            m_used.add(j)
            made = {"tool": mc["tool"], "args": mc.get("arguments")}
            reason = _compare(g["args"], mc.get("arguments"), g["tool"]) or "out_of_order"
            break
        detail.append({"segment": seg, "required": req, "made": made,
                       "correct": False, "reason": reason})
    extra = 0
    for j, mc in enumerate(made_calls):
        if j not in m_used:
            extra += 1
            detail.append({"segment": seg, "required": None,
                           "made": {"tool": mc.get("tool"), "args": mc.get("arguments")},
                           "correct": False, "reason": "extra_call"})
    return detail, correct, extra, structure_ok


def _segment_endpoint_valid(s: dict) -> bool:
    """A segment endpoint must be one of the two mutually exclusive outcomes actually
        reachable under the §6.2 protocol:
          * answer endpoint: final_assistant non-empty, cap not triggered, 1 <= rounds <= 6,
            and the call-bearing turns are exactly 1..rounds-1 (the last turn is the call-free
            reply, and the reply itself occupies a turn, so a rounds=0 "reply" is unreachable);
          * cap endpoint: cap_hit with final_assistant empty, rounds exactly 6, and call-bearing
            turns exactly 1..6 (rollout only records a cap when all 6 consecutive model turns are
            tool calls, so a rounds=0 "cap hit" is unreachable).
        Unreachable, premature, out-of-range or self-contradictory endpoint claims never count
        as protocol completion.
    """
    rounds = s.get("rounds")
    fin, cap = s.get("final_assistant"), bool(s.get("cap_hit"))
    rset = [c.get("round") for c in s.get("calls", [])]
    if not isinstance(rounds, int) or isinstance(rounds, bool) \
            or not all(isinstance(r, int) and not isinstance(r, bool) for r in rset):
        return False
    if fin is not None and not cap:
        return 1 <= rounds <= SEALED_MAX_ROUNDS and sorted(set(rset)) == list(range(1, rounds))
    if cap and fin is None:
        return rounds == SEALED_MAX_ROUNDS and sorted(set(rset)) == list(range(1, SEALED_MAX_ROUNDS + 1))
    return False


def score_task(golden: dict, transcript: dict | None) -> dict:
    n_req = sum(len(s["calls"]) for s in golden["segments"])
    rec = {"idx": None, "workflow": golden["workflow"], "route": golden["route"],
           "expect": golden["expect"], "turns": golden["turns"],
           "trace_match": False, "calls_required": n_req, "calls_correct": 0,
           "calls_extra": 0, "call_detail": [], "latency_ms": None, "rounds": 0,
           "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "transcript": None,
           "error": None, "cap_hit": False, "complete": False, "rollout_complete": False}
    made_segs = (transcript or {}).get("segments", [])
    correct = extra = 0
    all_ok = True
    for k in range(max(len(golden["segments"]), len(made_segs))):
        g_rounds = golden["segments"][k]["rounds"] if k < len(golden["segments"]) else []
        mc = made_segs[k].get("calls", []) if k < len(made_segs) else []
        d, c, ex, ok = match_segment(g_rounds, mc, k)
        rec["call_detail"].extend(d)
        correct += c
        extra += ex
        all_ok = all_ok and ok
    rec["calls_correct"], rec["calls_extra"] = correct, extra
    if transcript is None:
        rec["error"] = "transcript missing"
        return rec
    # Errored tasks retain call and latency measurements while trace match stays false.
    rec["error"] = transcript.get("error")
    # Completeness has answer-level and protocol-level evidence. Segment endpoints
    # pass the structural check first.
    # (_segment_endpoint_valid: an endpoint must be genuinely reachable under the §6.2 six-round protocol):
    #   complete         -- answer layer: every golden user turn has a matching transcript
    #                       segment, and the segment is a valid answer endpoint (closed by
    #                       final_assistant). Necessary for trace match; an empty transcript
    #                       or a rounds=0 "reply" does not qualify.
    #   rollout_complete -- protocol layer: every scripted user turn ran to a reachable
    #                       protocol endpoint (a valid answer endpoint, or a genuine 6-round
    #                       cap endpoint) with no terminal error. This is the evidence standard
    #                       for listing: hitting the cap is real weak-model behavior; empty or
    #                       missing segments, errors and unreachable claims are not.
    seg_n_ok = len(made_segs) == len(golden["segments"])
    valid = seg_n_ok and all(_segment_endpoint_valid(s) for s in made_segs)
    rec["complete"] = valid and all(s.get("final_assistant") is not None for s in made_segs)
    rec["rollout_complete"] = valid and rec["error"] is None
    rec["trace_match"] = all_ok and rec["complete"] and rec["error"] is None
    rec["latency_ms"] = transcript.get("latency_ms")
    rec["rounds"] = sum(s.get("rounds", 0) for s in made_segs)
    rec["tokens_in"] = transcript.get("tokens_in", 0)
    rec["tokens_out"] = transcript.get("tokens_out", 0)
    rec["cap_hit"] = any(s.get("cap_hit") for s in made_segs)
    return rec


# -------------------------------------------------------------- summarize --
def _rank(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank percentile (deterministic)."""
    if not sorted_vals:
        return 0.0
    return sorted_vals[max(0, math.ceil(q * len(sorted_vals)) - 1)]


def _agg(tasks: list[dict]) -> dict:
    req = sum(t["calls_required"] for t in tasks)
    cor = sum(t["calls_correct"] for t in tasks)
    lat = sorted(t["latency_ms"] for t in tasks if t["latency_ms"] is not None)
    return {"n": len(tasks),
            "trace_match": round(sum(t["trace_match"] for t in tasks) / len(tasks), 4) if tasks else 0.0,
            "toolcall_match": round(cor / req, 4) if req else 1.0,
            # No latency samples are represented by null.
            "p50_latency_ms": round(_rank(lat, 0.50), 1) if lat else None,
            "p95_latency_ms": round(_rank(lat, 0.95), 1) if lat else None,
            # Each slice links to its task-level evidence.
            "tasks": [f"tasks/{t['idx']:04d}.json" for t in tasks]}


def summarize(tasks: list[dict], run_meta: dict, run_dir: pathlib.Path) -> dict:
    overall = {
        **_agg(tasks),
        "calls_required": sum(t["calls_required"] for t in tasks),
        "calls_correct": sum(t["calls_correct"] for t in tasks),
        "calls_extra": sum(t["calls_extra"] for t in tasks),
        "tokens_in": sum(t["tokens_in"] for t in tasks),
        "tokens_out": sum(t["tokens_out"] for t in tasks),
        "cost_usd_total": round(sum(t["cost_usd"] for t in tasks), 6),
        "cost_usd_task": round(sum(t["cost_usd"] for t in tasks) / len(tasks), 6) if tasks else 0.0,
        "errors": sum(1 for t in tasks if t["error"]),
        "cap_hits": sum(1 for t in tasks if t["cap_hit"]),
        # Leaderboard publication requires complete transcripts, latencies, and rollouts.
        "transcripts": sum(1 for t in tasks if t["transcript"]),
        "latencies_recorded": sum(1 for t in tasks if t["latency_ms"] is not None),
        "rollouts_complete": sum(1 for t in tasks if t["rollout_complete"]),
    }
    def group(key):
        vals = sorted({t[key] for t in tasks})
        return {v: _agg([t for t in tasks if t[key] == v]) for v in vals}
    return {"system": run_meta.get("system"), "model": run_meta.get("model"),
            "note": run_meta.get("note", ""),
            "run_at": run_meta.get("finished_at"), "dataset": DATASET_NAME,
            "dataset_sha256": dataset_sha256(), "cases": len(tasks),
            "run_dir": str(run_dir.resolve()),
            "protocol": {"cases_run": run_meta.get("cases"),
                         "max_rounds": run_meta.get("max_rounds")},
            "overall": overall, "by_workflow": group("workflow"),
            "by_route": group("route"), "by_expect": group("expect")}


def score_run(run_dir: pathlib.Path) -> dict:
    """Score one submission: write tasks/ and summary.json, and return the summary."""
    run_meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    price_in = float(run_meta.get("price_in_per_mtok", 0.0))
    price_out = float(run_meta.get("price_out_per_mtok", 0.0))
    eval_rows = load_eval_set()
    tdir = run_dir / "tasks"
    tdir.mkdir(exist_ok=True)
    tasks = []
    for idx in sorted(eval_rows):
        tf = run_dir / "transcripts" / f"{idx:04d}.json"
        transcript = json.loads(tf.read_text(encoding="utf-8")) if tf.exists() else None
        rec = score_task(eval_rows[idx], transcript)
        rec["idx"] = idx
        rec["transcript"] = f"transcripts/{idx:04d}.json" if tf.exists() else None
        rec["cost_usd"] = round(rec["tokens_in"] * price_in / 1e6 +
                                rec["tokens_out"] * price_out / 1e6, 6)
        (tdir / f"{idx:04d}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        tasks.append(rec)
    summary = summarize(tasks, run_meta, run_dir)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# ------------------------------------------------------------ leaderboard --
class DatasetMismatch(ValueError):
    pass


class ProtocolMismatch(DatasetMismatch):
    """A run that is not the sealed protocol (101 tasks / 6 model rounds per user turn) may not be listed."""


def append_leaderboard(summary: dict, path: pathlib.Path) -> list[dict]:
    """Record one row per run in leaderboard.json and sync the static HTML.

        Re-scoring the same run overwrites its row and preserves its id. Runs must
        satisfy the sealed protocol and complete-evidence contract.
    """
    proto = summary.get("protocol") or {}
    if proto.get("cases_run") != SEALED_CASES or proto.get("max_rounds") != SEALED_MAX_ROUNDS:
        raise ProtocolMismatch(
            f"run protocol is {proto.get('cases_run')} tasks / max_rounds {proto.get('max_rounds')}, "
            f"not the sealed {SEALED_CASES}-task / {SEALED_MAX_ROUNDS}-round protocol")
    # Require 101 transcripts, 101 wall-clock latencies, and 101 rollouts that
    # reached a valid protocol endpoint.
    ov = summary["overall"]
    if (ov.get("transcripts") != SEALED_CASES
            or ov.get("latencies_recorded") != SEALED_CASES
            or ov.get("rollouts_complete") != SEALED_CASES):
        raise ProtocolMismatch(
            f"run evidence incomplete: {ov.get('transcripts')} transcripts, "
            f"{ov.get('latencies_recorded')} recorded latencies, "
            f"{ov.get('rollouts_complete')} protocol-complete rollouts; "
            f"{SEALED_CASES} of each required")
    entries: list[dict] = []
    if path.exists():
        entries = json.loads(path.read_text(encoding="utf-8"))
    prev = next((e for e in entries if e.get("dataset")), None)
    if prev and (prev["dataset"] != summary["dataset"]
                 or prev.get("dataset_sha256") != summary["dataset_sha256"]):
        raise DatasetMismatch(
            f"leaderboard {path} scores '{prev['dataset']}' ({prev.get('dataset_sha256', '')[:12]}); "
            f"this run scored '{summary['dataset']}' ({summary['dataset_sha256'][:12]})")
    run_dir_rel = os.path.relpath(summary["run_dir"], path.resolve().parent)
    o = summary["overall"]
    row = {
        "id": None, "name": summary["system"], "version": summary["model"],
        "run_at": summary["run_at"], "dataset": summary["dataset"],
        "dataset_sha256": summary["dataset_sha256"], "cases": summary["cases"],
        "run_dir": run_dir_rel, "note": summary.get("note", ""),
        "trace_match": o["trace_match"], "toolcall_match": o["toolcall_match"],
        "p50_latency_ms": o["p50_latency_ms"], "p95_latency_ms": o["p95_latency_ms"],
        "cost_usd_task": o["cost_usd_task"],
    }
    for w in sorted(summary.get("by_workflow") or {}):     # per-workflow trace match column
        row[f"trace_{w.lower()}"] = summary["by_workflow"][w]["trace_match"]
    existing = next((k for k, e in enumerate(entries) if e.get("run_dir") == run_dir_rel), None)
    if existing is not None:
        row["id"] = entries[existing].get("id")
        entries[existing] = row
    else:
        row["id"] = max((e.get("id", 0) for e in entries), default=0) + 1
        entries.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    _sync_html(path, entries)
    return entries


def _sync_html(leaderboard_json: pathlib.Path, entries: list[dict]) -> None:
    html_path = leaderboard_json.with_suffix(".html")
    if not html_path.exists():
        return
    marker = '<script type="application/json" id="leaderboard-data">\n'
    html = html_path.read_text(encoding="utf-8")
    start = html.find(marker)
    if start < 0:
        return
    data_start = start + len(marker)
    data_end = html.find("\n  </script>", data_start)
    if data_end < 0:
        return
    payload = json.dumps(entries, ensure_ascii=False, indent=2)
    html_path.write_text(html[:data_start] + payload + html[data_end:], encoding="utf-8")


def summary_line(summary: dict) -> str:
    o = summary["overall"]
    ms = lambda v: f"{v}ms" if v is not None else "n/a"  # noqa: E731
    return (f"{summary['system']} ({summary['model']}) on {summary['cases']} cases: "
            f"trace_match {o['trace_match']:.4f}, toolcall_match {o['toolcall_match']:.4f} "
            f"({o['calls_correct']}/{o['calls_required']} calls, {o['calls_extra']} extra), "
            f"p50 {ms(o['p50_latency_ms'])} p95 {ms(o['p95_latency_ms'])}, ${o['cost_usd_task']}/task")


def main() -> int:
    ap = argparse.ArgumentParser(description="Score a submission run dir (README §6.2)")
    ap.add_argument("run_dir", type=pathlib.Path)
    ap.add_argument("--leaderboard", type=pathlib.Path, default=EVAL_DIR / "leaderboard.json")
    ap.add_argument("--no-leaderboard", action="store_true", help="score only, do not add a leaderboard row")
    a = ap.parse_args()
    summary = score_run(a.run_dir)
    print(summary_line(summary))
    if not a.no_leaderboard:
        try:
            append_leaderboard(summary, a.leaderboard)
            print(f"leaderboard updated: {a.leaderboard} (+ .html)")
        except DatasetMismatch as exc:
            # Preserve scored artifacts while refusing publication to a mismatched board.
            print(f"warning: not recorded on the leaderboard. {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
