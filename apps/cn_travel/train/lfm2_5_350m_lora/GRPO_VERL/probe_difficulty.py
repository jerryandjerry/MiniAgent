#!/usr/bin/env python3
"""Adaptive RUN #2 probe over all 909 whole episodes.

Each episode gets four temperature-0.9 SFT rollouts.  An episode whose four
initial rollouts all fail the goal gets four additional temperature-0.8
rollouts.  Every attempt is scored with the same terminal-success-gated reward
used by VERL training.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import sys
import tempfile
import threading
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Sequence

HERE = pathlib.Path(__file__).resolve().parent
TRAIN_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = TRAIN_ROOT.parent / "src"
DEFAULT_CONFIG = HERE / "verl_traj_config.json"
for _path in (HERE, TRAIN_ROOT, SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from training.paths import TOOL_SCHEMAS, resolve_app_path  # noqa: E402
from world_policy import EpisodeRepository  # noqa: E402
from verl_data import (  # noqa: E402
    CATEGORIES,
    PROBE_SCHEMA,
    _category_for,
    _episode_map,
    _probe_behavior_hashes,
    _reward_policy,
    _sample_anchors,
    _strict_equal,
    _validate_filter_contract,
)


def _resolve(value: str | os.PathLike[str]) -> pathlib.Path:
    return resolve_app_path(value)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid EpisodeSpec JSON at {path}:{lineno}: {exc}") from exc
        if not isinstance(row, dict):
            raise SystemExit(f"EpisodeSpec at {path}:{lineno} is not an object")
        rows.append(row)
    return rows


def _json_document(path: pathlib.Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"missing {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {label} {path}: {exc}") from exc


def _validated_reused_records(
    source: Any,
    *,
    specs: Sequence[dict[str, Any]],
    required_identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Fail closed unless a replay source is a complete valid RUN #2 probe."""
    if not isinstance(source, dict) or source.get("schema_version") != PROBE_SCHEMA:
        raise SystemExit("reused probe has an unsupported or missing schema_version")
    meta = source.get("meta")
    if not isinstance(meta, dict):
        raise SystemExit("reused probe must contain a meta object")
    mismatched = {
        key: (meta.get(key), expected)
        for key, expected in required_identity.items()
        if not _strict_equal(meta.get(key), expected)
    }
    if mismatched:
        raise SystemExit(f"cannot reuse probe records with different identity: {mismatched}")
    expected_count = required_identity["expected_episode_count"]
    episodes = _episode_map(specs, expected_count)
    _validate_filter_contract(source, episodes, expected_count)
    records = source.get("records")
    assert isinstance(records, list)  # established by contract validation
    records_by_id = {record["episode_id"]: record for record in records}
    return [records_by_id[spec["episode_id"]] for spec in specs]


def _plain(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"cannot serialize runtime result {type(value).__name__}")


def _wire_tool_calls(tool_calls: Sequence[Any] | None) -> list[dict[str, Any]]:
    return [
        {
            "id": str(call.id),
            "type": "function",
            "function": {
                "name": str(call.function.name),
                "arguments": str(call.function.arguments),
            },
        }
        for call in tool_calls or ()
    ]


def _truncate(runtime: Any, reason: str) -> None:
    method = getattr(runtime, "truncate", None)
    if not callable(method):
        raise RuntimeError("world-policy runtime lacks truncate(reason)")
    method(reason)


def _one_rollout(
    *,
    client: Any,
    model: str,
    tools: list[dict[str, Any]],
    repository: EpisodeRepository,
    spec: dict[str, Any],
    runtime_config: dict[str, Any],
    attempt: int,
    phase: str,
    temperature: float,
    max_tokens: int,
    max_assistant_turns: int,
    timeout: float,
    seed: int,
) -> dict[str, Any]:
    episode_id = spec["episode_id"]
    runtime = repository.new_runtime(episode_id, config=runtime_config)
    runtime.reset()
    messages = [dict(message) for message in spec["initial_messages"]]

    for turn in range(max_assistant_turns):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            seed=seed + turn,
        )
        choice = response.choices[0]
        message = choice.message
        calls = _wire_tool_calls(message.tool_calls)
        content = message.content or ""
        if choice.finish_reason == "length":
            messages.append({"role": "assistant", "content": content})
            _truncate(runtime, "model_output_length")
            break
        transition = runtime.interpret_and_step(content, calls if calls else None)
        # Native LFM markers may arrive in content rather than tool_calls.
        if not calls and transition.tool_results:
            tool_messages = [item for item in transition.messages if item.get("role") == "tool"]
            if len(tool_messages) != len(transition.tool_results):
                raise RuntimeError("runtime tool executions/messages disagree")
            calls = [
                {
                    "id": str(tool_message["tool_call_id"]),
                    "type": "function",
                    "function": {
                        "name": execution.tool,
                        "arguments": json.dumps(
                            dict(execution.arguments),
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    },
                }
                for execution, tool_message in zip(transition.tool_results, tool_messages)
            ]
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": "" if calls else content,
        }
        if calls:
            assistant_message["tool_calls"] = calls
        messages.append(assistant_message)
        messages.extend(dict(item) for item in transition.messages)
        if transition.terminal:
            break
    else:
        _truncate(runtime, "assistant_turn_limit")

    reward_data = _plain(runtime.reward())
    outcome = _plain(runtime.outcome())
    return {
        "attempt": attempt,
        "phase": phase,
        "temperature": temperature,
        "seed": seed,
        "reward": float(reward_data["total"]),
        "goal_reached": bool(reward_data["goal_reached"]),
        "clean_trace": bool(reward_data["clean_trace"]),
        "U": int(reward_data["violation_count"]),
        "outcome": outcome,
    }


def _adaptive_attempts(
    run_one: Any,
    *,
    initial_count: int,
    initial_temperature: float,
    retry_temperature: float,
    retry_count: int,
) -> list[dict[str, Any]]:
    """Run K initial attempts, then all retries iff every initial goal fails."""
    attempts = [
        run_one(attempt_no, "initial", initial_temperature)
        for attempt_no in range(initial_count)
    ]
    if all(not attempt["goal_reached"] for attempt in attempts):
        attempts.extend(
            run_one(retry_no, "retry", retry_temperature)
            for retry_no in range(initial_count, initial_count + retry_count)
        )
    return attempts


def _write_filter(
    *,
    records: list[dict[str, Any]],
    out: pathlib.Path,
    meta: dict[str, Any],
    anchor_fraction: float,
    initial_count: int,
    seed: int,
) -> dict[str, Any]:
    by_category = {
        category: [record for record in records if record["category"] == category]
        for category in CATEGORIES
    }
    maximum = float(meta["reward_policy"]["maximum"])
    initial_by_category = {
        category: [
            record
            for record in records
            if _category_for(record["attempts"][:initial_count], maximum) == category
        ]
        for category in CATEGORIES
    }
    anchors = _sample_anchors(
        by_category["solved"],
        fraction=anchor_fraction,
        seed=seed,
    )
    anchor_ids = {record["episode_id"] for record in anchors}
    train_on = [
        {
            "episode_id": record["episode_id"],
            "source_idx": record["source_idx"],
            "route": record["route"],
            "selection": (
                "anchor" if record["episode_id"] in anchor_ids else record["category"]
            ),
        }
        for record in records
        if record["episode_id"] in anchor_ids
        or record["category"] == "mixed"
    ]
    document = {
        "schema_version": PROBE_SCHEMA,
        "meta": {
            **meta,
            "anchor_fraction": anchor_fraction,
            "anchor_sampling": "route_stratified_exact_target",
            "episodes_total": len(records),
            "complete_population": len(records) == meta["expected_episode_count"],
            **{
                f"initial_{category}": len(values)
                for category, values in initial_by_category.items()
            },
            **{category: len(values) for category, values in by_category.items()},
            "attempts_total": sum(len(record["attempts"]) for record in records),
            "anchors_kept": len(anchors),
            "train_on": len(train_on),
        },
        "records": records,
        "train_on": train_on,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".tmp", dir=out.parent)
    os.close(fd)
    temp = pathlib.Path(temp_name)
    try:
        temp.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, out)
    finally:
        temp.unlink(missing_ok=True)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description="Adaptive RUN #2 whole-episode probe")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=pathlib.Path)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--from-records", type=pathlib.Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.model != config["probe_model"]:
        raise SystemExit(
            f"probe model name must be {config['probe_model']!r}; got {args.model!r}"
        )
    initial_count = int(config["probe_initial_count"])
    initial_temperature = float(config["probe_initial_temperature"])
    retry_temperature = float(config["probe_retry_temperature"])
    retry_count = int(config["probe_retry_count"])
    anchor_fraction = float(config["probe_anchor_fraction"])
    seed = int(config["seed"])
    max_tokens = int(config["probe_max_tokens"])
    if (
        initial_count,
        initial_temperature,
        retry_count,
        retry_temperature,
        anchor_fraction,
        seed,
    ) != (4, 0.9, 4, 0.8, 0.15, 42):
        raise SystemExit("probe sampling/filter values differ from governed RUN #2")
    if max_tokens < 1024:
        raise SystemExit("probe max_tokens must be at least 1024")
    reward_policy = _reward_policy(config)

    episodes_path = _resolve(config["episodes_file"])
    world_path = _resolve(config["world_file"])
    out = (args.out or _resolve(config["task_filter"])).resolve()
    expected_count = int(config.get("expected_episode_count", 909))
    episodes_digest = _sha256(episodes_path)
    world_digest = _sha256(world_path)
    if episodes_digest != config["episodes_sha256"]:
        raise SystemExit("EpisodeSpec SHA-256 mismatch")
    if world_digest != config["world_sha256"]:
        raise SystemExit("Synthetic China SHA-256 mismatch")
    specs = _jsonl(episodes_path)
    _episode_map(specs, expected_count)
    if args.from_records and args.limit:
        raise SystemExit("--from-records requires the complete population")
    work_specs = specs[: args.limit] if args.limit else specs

    meta = {
        "model": args.model,
        "base_url": args.base_url,
        "initial_count": initial_count,
        "initial_temperature": initial_temperature,
        "retry_temperature": retry_temperature,
        "retry_count": retry_count,
        "max_tokens": max_tokens,
        "seed": seed,
        "expected_episode_count": expected_count,
        "episodes_sha256": episodes_digest,
        "world_sha256": world_digest,
        "sft_adapter_sha256": config["sft_adapter_sha256"],
        "sft_reference_weights_sha256": config["sft_reference_weights_sha256"],
        "reward_policy": reward_policy,
        "behavior_files_sha256": _probe_behavior_hashes(),
    }
    required_identity = {
        key: value for key, value in meta.items() if key != "base_url"
    }
    required_identity["anchor_fraction"] = anchor_fraction

    if args.from_records:
        source = _json_document(args.from_records, "reused probe")
        records = _validated_reused_records(
            source,
            specs=specs,
            required_identity=required_identity,
        )
    else:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit("the OpenAI Python client is required for probing") from exc
        tools = json.loads(TOOL_SCHEMAS.read_text(encoding="utf-8"))
        repository = EpisodeRepository.load(
            episodes_path,
            world_path,
            expected_count=expected_count,
        )
        if repository.world.world_id != config["world_id"]:
            raise SystemExit("Synthetic China world_id mismatch")
        runtime_config = {
            "reward": reward_policy,
            "max_rounds_per_user_turn": config["max_assistant_rounds_per_user_turn"],
            "max_assistant_rounds": config["max_assistant_turns"],
            "max_person_turns": config["max_user_turns"],
        }
        local = threading.local()

        def probe(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
            position, spec = item
            if not hasattr(local, "client"):
                local.client = OpenAI(base_url=args.base_url, api_key="EMPTY")
            initial_seed = seed + position * 10_000
            def run_one(attempt_no: int, phase: str, temperature: float) -> dict[str, Any]:
                return _one_rollout(
                    client=local.client,
                    model=args.model,
                    tools=tools,
                    repository=repository,
                    spec=spec,
                    runtime_config=runtime_config,
                    attempt=attempt_no,
                    phase=phase,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_assistant_turns=config["max_assistant_turns"],
                    timeout=args.timeout,
                    seed=initial_seed + attempt_no * 100,
                )
            attempts = _adaptive_attempts(
                run_one,
                initial_count=initial_count,
                initial_temperature=initial_temperature,
                retry_temperature=retry_temperature,
                retry_count=retry_count,
            )
            return {
                "episode_id": spec["episode_id"],
                "source_idx": spec["source_idx"],
                "route": spec["metadata"]["route"],
                "category": _category_for(attempts, float(reward_policy["maximum"])),
                "attempts": attempts,
            }

        records_by_id: dict[str, dict[str, Any]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(probe, item) for item in enumerate(work_specs)]
            for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
                record = future.result()  # external/parser/runtime failures abort the probe
                records_by_id[record["episode_id"]] = record
                if done % 25 == 0 or done == len(futures):
                    counts = defaultdict(int)
                    attempts_done = 0
                    for value in records_by_id.values():
                        counts[value["category"]] += 1
                        attempts_done += len(value["attempts"])
                    print(
                        f"[{done}/{len(futures)}] solved={counts['solved']} "
                        f"mixed={counts['mixed']} unsolved={counts['unsolved']} "
                        f"attempts={attempts_done}",
                        flush=True,
                    )
        records = [records_by_id[spec["episode_id"]] for spec in work_specs]

    document = _write_filter(
        records=records,
        out=out,
        meta=meta,
        anchor_fraction=anchor_fraction,
        initial_count=initial_count,
        seed=seed,
    )
    if not args.limit:
        _validate_filter_contract(document, _episode_map(specs, expected_count), expected_count)
    print(f"sealed adaptive whole-episode filter: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
