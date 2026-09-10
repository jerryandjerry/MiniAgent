#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe the current policy and select informative §5.3 training segments.

GRPO uses within-group advantages, so groups with identical rewards contribute
zero policy-gradient signal.

Output:
  * solved   pass rate = 1        (reliably correct; a --anchor_frac sample is kept as
                                   anti-forgetting anchors)
  * mixed    0 < pass rate < 1    (kept in full)
  * unsolved pass rate = 0        (excluded from this round)

It reads only §5.3's training segments (env.build_episodes), samples through an
OpenAI-compatible endpoint, and has tools answered by the same frozen world - the same
loop as training.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

_DIR = pathlib.Path(__file__).resolve().parent
TRAIN_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = TRAIN_ROOT.parent / "src"
for _path in (TRAIN_ROOT, SRC_ROOT, _DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from env import TravelEnv, build_episodes  # noqa: E402
from training.paths import TOOL_SCHEMAS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe per-segment pass rate (README §5.3)")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=pathlib.Path, default=_DIR / "task_filter.json")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--anchor-frac", type=float, default=0.15,
                    help="fraction of solved segments kept as anchors (anti-forgetting; stratified by route, fixed seed)")
    ap.add_argument("--min-per-route", type=int, default=1,
                    help="minimum anchors kept for every route that has solved segments (coverage floor)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-answer-tokens", type=int, default=64,
                    help="§5.4: once every golden call has matched, the terminating turn only needs to show whether a tool_call appears, "
                         "so cap that turn's max_tokens (0 = no cap)")
    ap.add_argument("--no-early-exit", action="store_true",
                    help="§5.4: disable the 'abandon once a perfect score is impossible' early exit (for cross-checking)")
    ap.add_argument("--from-records", type=pathlib.Path, default=None,
                    help="reuse the pass records in an existing filter file and only redo anchor sampling (no re-probe)")
    ap.add_argument("--w_call", type=float, default=1.0)
    ap.add_argument("--w_struct", type=float, default=0.5)
    ap.add_argument("--w_extra", type=float, default=0.2)
    ap.add_argument("--clip_min", type=float, default=-1.0)
    ap.add_argument("--clip_max", type=float, default=1.5)
    a = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=a.base_url, api_key="EMPTY")
    tools = json.loads(TOOL_SCHEMAS.read_text(encoding="utf-8"))
    TravelEnv.W = {"w_call": a.w_call, "w_struct": a.w_struct, "w_extra": a.w_extra,
                   "clip_min": a.clip_min, "clip_max": a.clip_max}
    ceiling = a.w_call + a.w_struct          # perfect = all matched with no extra calls

    episodes = build_episodes()
    if a.limit:
        episodes = episodes[: a.limit]

    records = []
    if a.from_records:
        records = json.loads(a.from_records.read_text(encoding="utf-8"))["records"]
        print(f"reusing {len(records)} sealed pass records from {a.from_records}")
        episodes = records            # kept the same length only so the statistics below line up
    for n, ep in enumerate([] if a.from_records else episodes, 1):
        passes = 0
        for _ in range(a.k):
            env = TravelEnv()
            env.reset(**ep)
            msgs = [dict(m) for m in ep["prompt"]]
            for _round in range(a.max_rounds):
                # §5.4 cap: once the golden calls have matched position for position, this turn only needs
                # to reveal whether another call is coming, not the reply body. Earlier turns are never
                # capped - truncating a tool call would fabricate a failure.
                cap = (a.max_answer_tokens
                       if (a.max_answer_tokens > 0
                           and env._calls_made() == env._calls_required()
                           and env._can_still_pass())
                       else None)
                kw = {"max_tokens": cap} if cap else {}
                r = client.chat.completions.create(
                    model=a.model, messages=msgs, tools=tools,
                    temperature=a.temperature, timeout=a.timeout, **kw)
                m = r.choices[0].message
                calls = list(m.tool_calls or [])
                if not calls:
                    break                     # no call means this is the terminating turn (including a reply cut short by the cap)
                msgs.append({"role": "assistant", "content": m.content or "",
                             "tool_calls": [{"id": tc.id, "type": "function",
                                             "function": {"name": tc.function.name,
                                                          "arguments": tc.function.arguments}}
                                            for tc in calls]})
                for tc in calls:
                    try:
                        parsed = json.loads(tc.function.arguments)
                        if not isinstance(parsed, dict):
                            parsed = {}
                    except (TypeError, json.JSONDecodeError):
                        parsed = {}
                    out = env._run(tc.function.name, parsed)   # the same frozen world
                    msgs.append({"role": "tool", "tool_call_id": tc.id, "content": out})
                if not a.no_early_exit and not env._can_still_pass():
                    break                     # §5.4: a perfect score is already impossible, so this rollout is scored as a loss
            if env.get_reward() >= ceiling - 1e-9:
                passes += 1
        records.append({"idx": ep["idx"], "seg_no": ep["seg_no"], "route": ep["route"],
                        "passes": passes, "k": a.k})
        if n % 50 == 0 or n == len(episodes):
            print(f"[{n}/{len(episodes)}] "
                  f"solved={sum(1 for r in records if r['passes'] == r['k'])} "
                  f"mixed={sum(1 for r in records if 0 < r['passes'] < r['k'])} "
                  f"unsolved={sum(1 for r in records if r['passes'] == 0)}", flush=True)

    solved = [r for r in records if r["passes"] == r["k"]]
    mixed = [r for r in records if 0 < r["passes"] < r["k"]]
    unsolved = [r for r in records if r["passes"] == 0]
    # Route-stratified anchors give every solved route an anti-forgetting floor.
    # Remaining slots are allocated in proportion to each route's solved count.
    rng = random.Random(a.seed)
    by_route: dict[str, list] = {}
    for r in solved:
        by_route.setdefault(r["route"], []).append(r)
    for rs in by_route.values():
        rs.sort(key=lambda r: (r["idx"], r["seg_no"]))      # deterministic
    budget = round(len(solved) * a.anchor_frac)
    quota = {rt: min(len(rs), a.min_per_route) for rt, rs in by_route.items()}
    left = max(0, budget - sum(quota.values()))
    if left:
        # allocate the remainder proportionally (largest-remainder method, deterministic)
        share = {rt: len(rs) / len(solved) * left for rt, rs in by_route.items()}
        base = {rt: int(v) for rt, v in share.items()}
        rem = sorted(by_route, key=lambda rt: (-(share[rt] - base[rt]), rt))
        for rt in rem[: left - sum(base.values())]:
            base[rt] += 1
        for rt in by_route:
            quota[rt] = min(len(by_route[rt]), quota[rt] + base[rt])
    anchors = [x for rt in sorted(by_route)
               for x in rng.sample(by_route[rt], quota[rt])]
    train_on = [{"idx": r["idx"], "seg_no": r["seg_no"],
                 "band": "mixed" if 0 < r["passes"] < r["k"] else "anchor"}
                for r in mixed + anchors]

    a.out.write_text(json.dumps(
        {"meta": {"model": a.model, "k": a.k, "temperature": a.temperature,
                  "seed": a.seed, "anchor_frac": a.anchor_frac,
                  "anchor_sampling": "stratified_by_route",
                  "early_exit": not a.no_early_exit,
                  "max_answer_tokens": a.max_answer_tokens,
                  "min_per_route": a.min_per_route,
                  "episodes_total": len(episodes), "solved": len(solved),
                  "mixed": len(mixed), "unsolved": len(unsolved),
                  "anchors_kept": len(anchors), "train_on": len(train_on)},
         "records": records, "train_on": train_on},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"sealed: {a.out}  train_on={len(train_on)} "
          f"(mixed {len(mixed)} + anchors {len(anchors)}); "
          f"solved {len(solved)}, unsolved {len(unsolved)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
