#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rollout runner: drive the system under test through all 101 eval tasks and emit one submission (README §6.2 Protocol).

The system under test is any OpenAI-compatible endpoint (local vLLM, DashScope, claude_shim):
    make eval EVAL_ARGS='--system Qwen3.5-0.8B --model qwen3.5-0.8b \
        --base-url http://localhost:8000/v1'

Protocol (word-for-word from §6.2):
  * per task: the sealed system message plus scripted user turns; temperature 0;
  * every turn's tool_calls are answered by FrozenWorld (no network, no live LLM);
  * at most 6 model rounds per user turn; going over records cap_hit and moves on;
  * record only, never rescue: unparseable arguments are left to the world to reject deterministically.

Produces a run directory (the input to score.py):
    runs/<YYYYMMDDHHMM Chicago>_<system>/run.json + transcripts/{idx:04d}.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
from datetime import datetime
from zoneinfo import ZoneInfo

try:
    from .paths import PATHS
    from .world import FrozenWorld
except ImportError:  # Direct ``python eval/rollout.py`` invocation.
    from paths import PATHS
    from world import FrozenWorld

CHICAGO = ZoneInfo("America/Chicago")
EVAL_SET = PATHS.eval_set
RUNS = PATHS.runs


def user_turns(conversation: list[dict]) -> list[str]:
    return [m["content"] for m in conversation if m.get("role") == "user"]


def run_task(client, model: str, row: dict, world: FrozenWorld, tools: list,
             max_rounds: int, timeout: float) -> dict:
    conv = row["conversation"]
    meta = row["metadata"]
    msgs: list[dict] = [dict(conv[0])]                      # the sealed system message
    out = {"idx": meta["idx"], "workflow": f"W{meta['workflow']}", "route": meta["route"],
           "expect": meta["expect"], "turns": meta["turns"], "segments": [],
           "latency_ms": 0.0, "tokens_in": 0, "tokens_out": 0, "requests": 0,
           "world_fallbacks": [], "error": None, "messages": msgs}
    fb0 = len(world.fallbacks)
    task_t0 = time.monotonic()      # §6.2: latency is whole-task wall clock (tool execution, retries and failures included),
    try:                            # Preserve observations from errored tasks.
        for user in user_turns(conv):
            msgs.append({"role": "user", "content": user})
            seg = {"user": user, "calls": [], "final_assistant": None,
                   "rounds": 0, "cap_hit": False}
            out["segments"].append(seg)
            while True:
                if seg["rounds"] >= max_rounds:
                    seg["cap_hit"] = True
                    break
                resp = None
                for attempt in range(3):                    # retry transport-layer errors only
                    try:
                        resp = client.chat.completions.create(
                            model=model, messages=msgs, tools=tools,
                            temperature=0, timeout=timeout)
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(2 + 3 * attempt)
                out["requests"] += 1
                u = getattr(resp, "usage", None)
                if u is not None:
                    out["tokens_in"] += getattr(u, "prompt_tokens", 0) or 0
                    out["tokens_out"] += getattr(u, "completion_tokens", 0) or 0
                m = resp.choices[0].message
                tool_calls = list(m.tool_calls or [])
                seg["rounds"] += 1
                if not tool_calls:
                    seg["final_assistant"] = m.content or ""
                    msgs.append({"role": "assistant", "content": m.content or ""})
                    break
                msgs.append({"role": "assistant", "content": m.content or "",
                             "tool_calls": [{"id": tc.id, "type": "function",
                                             "function": {"name": tc.function.name,
                                                          "arguments": tc.function.arguments}}
                                            for tc in tool_calls]})
                for tc in tool_calls:
                    raw = tc.function.arguments
                    try:
                        parsed = json.loads(raw)
                        if not isinstance(parsed, dict):
                            parsed = None
                    except (TypeError, json.JSONDecodeError):
                        parsed = None
                    seg["calls"].append({"round": seg["rounds"], "id": tc.id,
                                         "tool": tc.function.name,
                                         "arguments_raw": raw, "arguments": parsed})
                    result = world.run(tc.function.name, parsed if parsed is not None else {})
                    msgs.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(result, ensure_ascii=False)})
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"[:500]
    out["latency_ms"] = round((time.monotonic() - task_t0) * 1000, 1)
    out["world_fallbacks"] = world.fallbacks[fb0:]
    return out


def build_parser(required: bool = True,
                 parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    ap = parser or argparse.ArgumentParser(
        description="Roll out a system over the eval set (README §6.2)")
    ap.add_argument("--system", required=required, help="system name recorded in the run directory, e.g. Qwen3.5-0.8B")
    ap.add_argument("--model", required=required, help="model name on the endpoint")
    ap.add_argument("--base-url", required=required, help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1")
    ap.add_argument("--api-key-env", default="LLM_API_KEY", help="name of the environment variable holding the API key")
    ap.add_argument("--price-in-per-mtok", type=float, default=0.0, help="USD / 1M input tokens")
    ap.add_argument("--price-out-per-mtok", type=float, default=0.0, help="USD / 1M output tokens")
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--limit", type=int, default=0, help="debugging: run only the first N tasks")
    ap.add_argument("--note", default="", help="hardware/channel note (e.g. GPU model), carried onto the leaderboard row")
    return ap


def rollout(a: argparse.Namespace) -> pathlib.Path:
    """Run the whole eval set and return the submission (run directory)."""
    from openai import OpenAI
    client = OpenAI(base_url=a.base_url, api_key=os.getenv(a.api_key_env) or "EMPTY")
    tools = json.loads(PATHS.tool_schemas.read_text(encoding="utf-8"))
    rows = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    rows.sort(key=lambda r: r["metadata"]["idx"])
    if a.limit:
        rows = rows[:a.limit]

    run_dir = RUNS / f"{datetime.now(CHICAGO).strftime('%Y%m%d%H%M')}_{a.system}"
    (run_dir / "transcripts").mkdir(parents=True)
    run_meta = {"system": a.system, "model": a.model, "base_url": a.base_url,
                "note": a.note,
                "price_in_per_mtok": a.price_in_per_mtok,
                "price_out_per_mtok": a.price_out_per_mtok,
                "max_rounds": a.max_rounds, "temperature": 0, "cases": len(rows),
                "started_at": datetime.now(CHICAGO).isoformat(timespec="seconds"),
                "finished_at": None}
    (run_dir / "run.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    world = FrozenWorld()
    for n, row in enumerate(rows, 1):
        t = run_task(client, a.model, row, world, tools, a.max_rounds, a.timeout)
        (run_dir / "transcripts" / f"{t['idx']:04d}.json").write_text(
            json.dumps(t, ensure_ascii=False, indent=2), encoding="utf-8")
        flag = f"  ERROR: {t['error']}" if t["error"] else ""
        if n % 10 == 0 or t["error"] or n == len(rows):
            print(f"[{n}/{len(rows)}] idx {t['idx']:04d} {t['route']}: "
                  f"{sum(len(s['calls']) for s in t['segments'])} calls, "
                  f"{t['latency_ms']}ms{flag}", flush=True)
    run_meta["finished_at"] = datetime.now(CHICAGO).isoformat(timespec="seconds")
    (run_dir / "run.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(f"submission ready: {run_dir}")
    return run_dir


def main() -> int:
    rollout(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
