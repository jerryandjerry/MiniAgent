#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run rollout, deterministic scoring, and leaderboard publication (README §6.2).

A dataset mismatch preserves the scored run artifacts and blocks publication.

    # evaluate a live system (rollout + score + leaderboard row)
    make eval EVAL_ARGS='--system Qwen3.5-0.8B --model qwen3.5-0.8b \
        --base-url http://localhost:8000/v1'

    # score an existing submission and record it (skips rollout)
    make eval EVAL_ARGS='--submission eval/runs/<dir>'

    # evaluate without recording
    ... --leaderboard none
"""
from __future__ import annotations

import argparse
import pathlib
import sys

try:
    from . import rollout as _rollout
    from . import score as _score
except ImportError:  # Direct ``python eval/evaluate.py`` invocation.
    import rollout as _rollout
    import score as _score


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Evaluate a system end-to-end: rollout -> score -> leaderboard (README §6.2)")
    _rollout.build_parser(required=False, parser=ap)
    ap.add_argument("--submission", type=pathlib.Path,
                    help="an existing run directory: skip rollout and score it directly")
    ap.add_argument("--leaderboard", default=str(_score.EVAL_DIR / "leaderboard.json"),
                    help='path to the leaderboard JSON; pass "none" to evaluate without listing')
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.submission:
        run_dir = a.submission
        if not (run_dir / "run.json").exists():
            print(f"error: {run_dir} is not a submission (run.json missing)", file=sys.stderr)
            return 2
    else:
        missing = [f for f in ("system", "model", "base_url") if not getattr(a, f)]
        if missing:
            print(f"error: missing --{' --'.join(m.replace('_', '-') for m in missing)}"
                  f" (or use --submission <run_dir> instead)", file=sys.stderr)
            return 2
        run_dir = _rollout.rollout(a)

    summary = _score.score_run(run_dir)
    print(_score.summary_line(summary))

    if str(a.leaderboard).strip().lower() != "none":
        try:
            _score.append_leaderboard(summary, pathlib.Path(a.leaderboard))
            print(f"leaderboard updated: {a.leaderboard} (+ .html)")
        except _score.DatasetMismatch as exc:
            # the evaluation itself stands — it is only refused a row on this board
            print(f"warning: not recorded on the leaderboard. {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
