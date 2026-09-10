#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build one whole-conversation VERL row per sealed training episode.

Each row carries the model-visible initial prompt and the environment-side user
segments:

  prompt   = system + the first user turn (sealed, verbatim)
  segments = per user turn, {verbatim user text, that turn's golden call sequence};

Golden assistant actions and tool results remain environment-side rollout state.
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
APP_ROOT = pathlib.Path(__file__).resolve().parents[3]
TRAIN_ROOT = APP_ROOT / "train"
for _p in (str(TRAIN_ROOT), str(_HERE), str(_HERE.parent / "GRPO_TRL")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from env import GOVERNED_PREFIXES, TRAIN_FILE  # noqa: E402
from training.tool_matching import golden_segments  # noqa: E402


def build_conversations() -> list[dict]:
    rows = json.loads(TRAIN_FILE.read_text(encoding="utf-8"))
    if len(rows) != GOVERNED_PREFIXES:
        raise SystemExit(f"train_validation.json has {len(rows)} rows != the governed population {GOVERNED_PREFIXES}")
    by_idx: dict[int, list] = collections.defaultdict(list)
    for r in rows:
        by_idx[r["metadata"]["idx"]].append(r)
    # the longest cumulative slice is the full parent conversation (same rule as §5.3 build_episodes)
    convs = {i: max(v, key=lambda r: len(r["conversation"]))["conversation"]
             for i, v in by_idx.items()}

    out = []
    for idx in sorted(convs):
        conv = convs[idx]
        segs = golden_segments(conv)
        cut = 0
        while cut < len(conv) and conv[cut].get("role") != "user":
            cut += 1
        prompt = [dict(m) for m in conv[:cut + 1]]      # system through the first user turn
        route = next((r["metadata"]["route"] for r in by_idx[idx]), "")
        out.append({
            "data_source": "cn_travel",
            "agent_name": "cn_travel",
            "prompt": prompt,
            "extra_info": {
                "index": idx,
                "route": route,
                "segments": json.dumps(
                    [{"user": s["user"], "golden": s["calls"]} for s in segs],
                    ensure_ascii=False),
            },
        })
    return out


def main() -> None:
    import pandas as pd

    convs = build_conversations()
    n_seg = sum(len(json.loads(c["extra_info"]["segments"])) for c in convs)
    # the config decides where this lands (swap models by swapping config); with no config, write here
    cfg_path = os.environ.get("CN_TRAVEL_VERL_CONFIG")
    if cfg_path:
        cfg = json.loads(pathlib.Path(cfg_path).read_text(encoding="utf-8"))
        out = pathlib.Path(cfg["train_parquet"])
        if not out.is_absolute():
            out = APP_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        out = _HERE / "verl_train.parquet"
    pd.DataFrame(convs).to_parquet(out, index=False)
    print(f"{len(convs)} conversations / {n_seg} user segments -> {out}")
    print("multi-turn share:",
          sum(1 for c in convs if len(json.loads(c['extra_info']['segments'])) > 1), "/", len(convs))


if __name__ == "__main__":
    main()
