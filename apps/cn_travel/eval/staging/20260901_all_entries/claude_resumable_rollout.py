#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resume one Claude leaderboard rollout across CLI quota windows.

This is a narrow wrapper around the frozen evaluator.  It deliberately imports
and calls ``rollout.run_task`` instead of reproducing any rollout behavior.
Each invocation walks the same 101 rows in sorted ``metadata.idx`` order,
skips only protocol-complete error-free transcripts, and stops after the first
task for which the canonical runner returns an error.  A failed task is retained
only under ``attempts/``; it is never installed under ``transcripts/``.

The Claude proxy must already be listening on http://127.0.0.1:8765/v1.
Example (the same command resumes after the next quota reset):

    uv run python apps/cn_travel/eval/staging/20260901_all_entries/claude_resumable_rollout.py \
      --model claude-opus-5 \
      --run-dir apps/cn_travel/eval/runs/202609010200_claude-opus-5

Once all 101 tasks are present, the wrapper runs the ordinary deterministic
scorer without a leaderboard write.  The resulting directory can then be
checked independently with:

    uv run python apps/cn_travel/eval/evaluate.py \
      --submission <run-dir> --leaderboard none
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
from datetime import datetime


HERE = pathlib.Path(__file__).resolve().parent
EVAL_DIR = HERE.parents[1]
REPO = EVAL_DIR.parents[2]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import rollout as _rollout  # noqa: E402
import score as _score  # noqa: E402


MODELS = ("claude-opus-5", "claude-sonnet-5")
BASE_URL = "http://127.0.0.1:8765/v1"
NOTE = "Anthropic cloud · claude CLI channel"
CASES = 101
MAX_ROUNDS = 6
TIMEOUT = 120.0
TEMPERATURE = 0
CLAUDE_CODE_VERSION = "2.1.252 (Claude Code)"

CHANNEL_IDENTITY = {
    "kind": "claude_cli_json_schema_proxy",
    "claude_code": "2.1.252",
    "claude_executable_sha256": "b661c6a094fcc32656bf7c0071c5b45bf900b34d4f0a1ab3d78fd59aeba2c2c7",
    "proxy_sha256": "7fc41aba0582baafeae0a70343a89a5e979269dc5904276b0285df040e3f4b85",
    "shim_sha256": "1c72deff41e64c10fac5aa5474b4c325e2003c583713ccdc58fbd5f9363280a1",
    "transport_sha256": "a926b7c8476a95df194352a206d0f4de70fa2a7b0ddbfd634216a5f0b598d557",
}

EVALUATOR_IDENTITY = {
    "dataset_sha256": "f569366318e02df9f99785733038b53667c8958115e4935e39c8e922537abe4a",
    "tool_schemas_sha256": "61e46aa71e72b9a5b1f0de360490077b785fa0b408a9c1e21c4573a00f5ea78f",
    "evaluate_sha256": "1520db93a689a7835146f5193707d42128e3c408fd4c0db33f2270727bf2baba",
    "rollout_sha256": "619485b12fab184bb85dee6b1d68e1b355c3a9fb3413c20196a92aad366c17ae",
    "score_sha256": "037f35bed276a04b8967530c289db99b4b026e657626cd2f4c906a2259ee6586",
    "world_sha256": "3087b6b1b2324e4bdda24b0e1dcc1a196b534a68fb7393192dc63b4ad8dd5e11",
}


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(_rollout.CHICAGO).isoformat(timespec="seconds")


def _atomic_json(path: pathlib.Path, value: object) -> None:
    """Install one complete JSON file with a same-directory atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as dst:
            json.dump(value, dst, ensure_ascii=False, indent=2)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _verify_frozen_inputs() -> None:
    if os.getenv("DISABLE_AUTOUPDATER") != "1":
        raise RuntimeError("DISABLE_AUTOUPDATER=1 is required for a frozen Claude channel")
    claude = shutil.which("claude")
    if not claude:
        raise RuntimeError("claude executable not found")
    version = subprocess.run(
        [claude, "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if version != CLAUDE_CODE_VERSION:
        raise RuntimeError(
            f"claude version drift: expected {CLAUDE_CODE_VERSION!r}, got {version!r}"
        )

    fixed = {
        pathlib.Path(claude): CHANNEL_IDENTITY["claude_executable_sha256"],
        EVAL_DIR / "claude_proxy.py": CHANNEL_IDENTITY["proxy_sha256"],
        REPO / "src/miniagent/llm/claude_shim.py": CHANNEL_IDENTITY["shim_sha256"],
        REPO / "src/miniagent/llm/claude_cli.py": CHANNEL_IDENTITY["transport_sha256"],
        _rollout.EVAL_SET: EVALUATOR_IDENTITY["dataset_sha256"],
        REPO / "apps/cn_travel/business_logic/tool_schemas.json":
            EVALUATOR_IDENTITY["tool_schemas_sha256"],
        EVAL_DIR / "evaluate.py": EVALUATOR_IDENTITY["evaluate_sha256"],
        EVAL_DIR / "rollout.py": EVALUATOR_IDENTITY["rollout_sha256"],
        EVAL_DIR / "score.py": EVALUATOR_IDENTITY["score_sha256"],
        EVAL_DIR / "world.py": EVALUATOR_IDENTITY["world_sha256"],
    }
    mismatches = []
    for path, expected in fixed.items():
        actual = _sha256(path)
        if actual != expected:
            mismatches.append(f"{path}: expected {expected}, got {actual}")
    if mismatches:
        raise RuntimeError("frozen input drift:\n" + "\n".join(mismatches))


def _channel_paths() -> tuple[pathlib.Path, ...]:
    claude = shutil.which("claude")
    if not claude:
        raise RuntimeError("claude executable not found")
    return (
        pathlib.Path(__file__).resolve(),
        pathlib.Path(claude),
        EVAL_DIR / "claude_proxy.py",
        REPO / "src/miniagent/llm/claude_shim.py",
        REPO / "src/miniagent/llm/claude_cli.py",
        EVAL_DIR / "evaluate.py",
        EVAL_DIR / "rollout.py",
        EVAL_DIR / "score.py",
        EVAL_DIR / "world.py",
        _rollout.EVAL_SET,
        REPO / "apps/cn_travel/business_logic/tool_schemas.json",
    )


def _stat_snapshot() -> dict[str, tuple[int, int, int, int]]:
    """Cheap per-task guard; full hashes are verified at invocation boundaries."""
    return {
        str(path): (
            path.stat().st_dev,
            path.stat().st_ino,
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in _channel_paths()
    }


def _fixed_run_meta(model: str) -> dict:
    return {
        "system": model,
        "model": model,
        "base_url": BASE_URL,
        "note": NOTE,
        "price_in_per_mtok": 0.0,
        "price_out_per_mtok": 0.0,
        "max_rounds": MAX_ROUNDS,
        "temperature": TEMPERATURE,
        "cases": CASES,
        "request_timeout_seconds": TIMEOUT,
        "channel_identity": CHANNEL_IDENTITY,
        "evaluator_identity": EVALUATOR_IDENTITY,
        "orchestrator_sha256": _sha256(pathlib.Path(__file__)),
    }


def _load_or_create_run(run_dir: pathlib.Path, model: str) -> dict:
    run_dir = run_dir.resolve()
    runs_root = _rollout.RUNS.resolve()
    if not run_dir.is_relative_to(runs_root) or run_dir == runs_root:
        raise RuntimeError(f"run directory must be a child of {runs_root}")

    meta_path = run_dir / "run.json"
    fixed = _fixed_run_meta(model)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for key, expected in fixed.items():
            if meta.get(key) != expected:
                raise RuntimeError(
                    f"run identity drift at {key!r}: expected {expected!r}, "
                    f"got {meta.get(key)!r}"
                )
        if not isinstance(meta.get("started_at"), str):
            raise RuntimeError("existing run has no valid started_at")
        return meta

    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty run directory without run.json: {run_dir}")
    (run_dir / "transcripts").mkdir(parents=True, exist_ok=True)
    meta = {**fixed, "started_at": _now(), "finished_at": None}
    _atomic_json(meta_path, meta)
    return meta


def _load_json(path: pathlib.Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _transcript_complete(row: dict, transcript: dict | None, golden: dict) -> bool:
    """Use the frozen scorer's endpoint predicate; do not filter model quality."""
    if transcript is None or "error" not in transcript or transcript["error"] is not None:
        return False
    meta = row["metadata"]
    expected_identity = {
        "idx": meta["idx"],
        "workflow": f"W{meta['workflow']}",
        "route": meta["route"],
        "expect": meta["expect"],
        "turns": meta["turns"],
    }
    if any(transcript.get(k) != v for k, v in expected_identity.items()):
        return False
    messages = transcript.get("messages")
    if not isinstance(messages, list) or not messages or messages[0] != dict(row["conversation"][0]):
        return False
    latency = transcript.get("latency_ms")
    if (not isinstance(latency, (int, float)) or isinstance(latency, bool)
            or latency < 0):
        return False
    scored = _score.score_task(golden, transcript)
    return bool(scored["rollout_complete"])


def _attempt_path(run_dir: pathlib.Path, idx: int, kind: str) -> pathlib.Path:
    stamp = datetime.now(_rollout.CHICAGO).strftime("%Y%m%dT%H%M%S.%f%z")
    return HERE / "attempts" / run_dir.name / kind / f"{idx:04d}_{stamp}.json"


def _quarantine(path: pathlib.Path, run_dir: pathlib.Path, idx: int) -> pathlib.Path:
    target = _attempt_path(run_dir, idx, "rejected_existing")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(path, target)
    return target


def _load_rows() -> list[dict]:
    rows = json.loads(_rollout.EVAL_SET.read_text(encoding="utf-8"))
    rows.sort(key=lambda row: row["metadata"]["idx"])
    indices = [row["metadata"]["idx"] for row in rows]
    if len(rows) != CASES or len(set(indices)) != CASES:
        raise RuntimeError(
            f"sealed population drift: expected {CASES} unique rows, got "
            f"{len(rows)} rows / {len(set(indices))} unique indices"
        )
    return rows


def _validate_all(run_dir: pathlib.Path, rows: list[dict], golden: dict[int, dict]) -> bool:
    for row in rows:
        idx = row["metadata"]["idx"]
        transcript = _load_json(run_dir / "transcripts" / f"{idx:04d}.json")
        if not _transcript_complete(row, transcript, golden[idx]):
            return False
    return True


def run(model: str, run_dir: pathlib.Path) -> int:
    """Run missing tasks until complete or until canonical run_task returns an error."""
    from openai import OpenAI

    _verify_frozen_inputs()
    channel_snapshot = _stat_snapshot()
    rows = _load_rows()
    tools = json.loads((REPO / "apps/cn_travel/business_logic/tool_schemas.json").read_text(
        encoding="utf-8"))
    golden = _score.load_eval_set()
    run_dir = run_dir.resolve()
    run_meta = _load_or_create_run(run_dir, model)
    if run_meta.get("finished_at") is not None and not _validate_all(run_dir, rows, golden):
        raise RuntimeError("run is marked finished but its transcript set is incomplete")

    client = OpenAI(base_url=BASE_URL, api_key=os.getenv("LLM_API_KEY") or "EMPTY")
    world = _rollout.FrozenWorld()
    skipped = written = 0
    for ordinal, row in enumerate(rows, 1):
        idx = row["metadata"]["idx"]
        transcript_path = run_dir / "transcripts" / f"{idx:04d}.json"
        prior = _load_json(transcript_path) if transcript_path.exists() else None
        if _transcript_complete(row, prior, golden[idx]):
            skipped += 1
            continue
        if transcript_path.exists():
            quarantined = _quarantine(transcript_path, run_dir, idx)
            print(f"[{ordinal}/{CASES}] idx {idx:04d}: quarantined {quarantined}", flush=True)

        transcript = _rollout.run_task(
            client, model, row, world, tools, MAX_ROUNDS, TIMEOUT
        )
        if _stat_snapshot() != channel_snapshot:
            failure = _attempt_path(run_dir, idx, "channel_drift")
            _atomic_json(failure, transcript)
            print(
                f"[{ordinal}/{CASES}] idx {idx:04d}: STOP frozen channel files changed "
                f"during this invocation; evidence: {failure}",
                file=sys.stderr,
                flush=True,
            )
            return 74
        if transcript.get("error") is not None:
            failure = _attempt_path(run_dir, idx, "failed")
            _atomic_json(failure, transcript)
            print(
                f"[{ordinal}/{CASES}] idx {idx:04d}: STOP {transcript['error']}\n"
                f"failed task retained outside official transcripts: {failure}\n"
                f"resume with the identical command after quota recovery",
                file=sys.stderr,
                flush=True,
            )
            return 75
        if not _transcript_complete(row, transcript, golden[idx]):
            failure = _attempt_path(run_dir, idx, "invalid_endpoint")
            _atomic_json(failure, transcript)
            print(
                f"[{ordinal}/{CASES}] idx {idx:04d}: STOP canonical runner returned "
                f"an invalid protocol endpoint; evidence: {failure}",
                file=sys.stderr,
                flush=True,
            )
            return 70

        _atomic_json(transcript_path, transcript)
        written += 1
        calls = sum(len(segment["calls"]) for segment in transcript["segments"])
        print(
            f"[{ordinal}/{CASES}] idx {idx:04d} {transcript['route']}: "
            f"{calls} calls, {transcript['latency_ms']}ms",
            flush=True,
        )

    if not _validate_all(run_dir, rows, golden):
        raise RuntimeError("internal error: scan finished with an incomplete transcript set")

    if _stat_snapshot() != channel_snapshot:
        raise RuntimeError("frozen channel files changed before finalization")
    _verify_frozen_inputs()
    if run_meta.get("finished_at") is None:
        run_meta["finished_at"] = _now()
        _atomic_json(run_dir / "run.json", run_meta)
    summary = _score.score_run(run_dir)
    overall = summary["overall"]
    if not (
        summary["cases"] == CASES
        and overall["transcripts"] == CASES
        and overall["latencies_recorded"] == CASES
        and overall["rollouts_complete"] == CASES
        and overall["errors"] == 0
    ):
        raise RuntimeError(f"completed run failed scorer evidence checks: {overall}")
    print(
        f"submission ready: {run_dir} "
        f"({skipped} skipped, {written} written this invocation)",
        flush=True,
    )
    print(
        "independent check: uv run python apps/cn_travel/eval/evaluate.py "
        f"--submission {run_dir} --leaderboard none",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resume one frozen Claude rollout across CLI quota windows"
    )
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--run-dir", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        return run(args.model, args.run_dir)
    except (OSError, RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
