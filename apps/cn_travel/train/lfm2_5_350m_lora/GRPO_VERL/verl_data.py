#!/usr/bin/env python3
"""Pack whole CN Travel episodes into VERL parquet rows.

Each row contains only the model-visible initial prompt plus rollout-control
metadata. Private person state, future user replies, tool results, and reward
diagnostics remain in the sealed Step-7 files and are loaded by the agent loop
at rollout.

RUN #2 consumes its probe-filtered population. RUN #3 bypasses the probe and
packs every sealed EpisodeSpec, adding only a workflow sampler label to
``extra_info``; those labels are never rendered into the model prompt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import random
import tempfile
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


HERE = pathlib.Path(__file__).resolve().parent
APP_ROOT = pathlib.Path(__file__).resolve().parents[3]
TRAIN_ROOT = APP_ROOT / "train"
DEFAULT_CONFIG = HERE / "verl_traj_config.json"
RUN2_SOURCE_SNAPSHOT = HERE / "run_02_source"
PROBE_SCHEMA = "cn_travel.world_policy_probe.v3"
REWARD_POLICY = "terminal_success_gated_v1"
CATEGORIES = (
    "solved",
    "mixed",
    "unsolved",
)
RECOVERABLE_VIOLATIONS = frozenset({
    "wrong_question",
    "repeated_question",
    "multiple_questions",
    "extra_or_wrong_tool",
    "premature_tool",
    "repeated_tool",
    "malformed_output",
})
PROBE_BEHAVIOR_FILES = (
    "train/lfm2_5_350m_lora/GRPO_VERL/probe_difficulty.py",
    "train/lfm2_5_350m_lora/GRPO_VERL/verl_data.py",
    "train/world_policy/__init__.py",
    "train/world_policy/actions.py",
    "train/world_policy/interpreter.py",
    "train/world_policy/runtime.py",
    "train/world_policy/world.py",
    "src/cn_travel/business_logic/contracts.py",
    "src/cn_travel/business_logic/tool_schemas.json",
    "data/city_registry.json",
    "src/cn_travel/service/result.py",
)


def _resolve(path: str | os.PathLike[str], *, base: pathlib.Path = APP_ROOT) -> pathlib.Path:
    value = pathlib.Path(path)
    return value if value.is_absolute() else base / value


def _load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise SystemExit(f"missing Step-7 EpisodeSpecs: {path}") from exc
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid JSONL at {path}:{lineno}: {exc}") from exc
        if not isinstance(row, dict):
            raise SystemExit(f"EpisodeSpec at {path}:{lineno} is not an object")
        rows.append(row)
    return rows


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_behavior_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in PROBE_BEHAVIOR_FILES:
        path = APP_ROOT / relative
        if not path.is_file():
            raise SystemExit(f"probe behavior-critical source is missing: {path}")
        hashes[relative] = _sha256(path)
    return hashes


def _validate_probe_behavior_identity(recorded: Any) -> None:
    """Validate the recorded probe/runtime source identity.

    Recorded hashes may match the active behavior files or the immutable Run #2
    source snapshot. Snapshot paths stay beneath the snapshot root and are
    verified byte-for-byte.
    """
    if _strict_equal(recorded, _probe_behavior_hashes()):
        return
    if not isinstance(recorded, Mapping) or not recorded:
        raise SystemExit("task filter lacks probe/runtime source identity")
    snapshot_hashes: dict[str, str] = {}
    for relative, expected_digest in recorded.items():
        if not isinstance(relative, str) or not isinstance(expected_digest, str):
            raise SystemExit("task filter contains an invalid probe/runtime source identity")
        path = RUN2_SOURCE_SNAPSHOT / relative
        try:
            path.resolve().relative_to(RUN2_SOURCE_SNAPSHOT.resolve())
        except ValueError as exc:
            raise SystemExit(f"task filter source path escapes bundled snapshot: {relative}") from exc
        if not path.is_file():
            raise SystemExit(f"bundled Run #2 source is missing: {relative}")
        snapshot_hashes[relative] = _sha256(path)
    if not _strict_equal(recorded, snapshot_hashes):
        raise SystemExit("task filter was produced by unbundled probe/runtime source bytes")


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _strict_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):  # noqa: E721 - identity files bind JSON types
        return False
    if isinstance(left, Mapping):
        return left.keys() == right.keys() and all(
            _strict_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _reward_policy(config: Mapping[str, Any]) -> dict[str, Any]:
    reward = config.get("reward")
    expected_keys = {"policy", "maximum", "success_floor", "penalty_per_violation"}
    if not isinstance(reward, Mapping) or set(reward) != expected_keys:
        raise SystemExit(f"RUN #2 reward must contain exactly {sorted(expected_keys)}")
    if reward.get("policy") != REWARD_POLICY:
        raise SystemExit("unsupported RUN #2 reward policy")
    for key in ("maximum", "success_floor", "penalty_per_violation"):
        if not _is_finite_number(reward.get(key)):
            raise SystemExit(f"RUN #2 reward.{key} must be finite")
    if float(reward["maximum"]) != 1.5 \
            or float(reward["success_floor"]) != 0.5 \
            or float(reward["penalty_per_violation"]) != 0.2:
        raise SystemExit("RUN #2 reward constants differ from the governed specification")
    return dict(reward)


def _sample_anchors(
    maximum_records: Iterable[dict[str, Any]],
    *,
    fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Select exactly ``round(fraction * N)`` route-stratified anchors."""
    records = list(maximum_records)
    if not 0 <= fraction <= 1:
        raise ValueError("anchor fraction must be in [0, 1]")
    if not records:
        return []
    by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_route[record["route"]].append(record)
    for route_records in by_route.values():
        route_records.sort(key=lambda item: (item["source_idx"], item["episode_id"]))

    target = min(len(records), round(len(records) * fraction))
    ideals = {
        route: target * len(route_records) / len(records)
        for route, route_records in by_route.items()
    }
    quotas = {route: math.floor(value) for route, value in ideals.items()}
    remaining = target - sum(quotas.values())
    for route in sorted(
        by_route,
        key=lambda name: (-(ideals[name] - quotas[name]), name),
    )[:remaining]:
        quotas[route] += 1

    rng = random.Random(seed)
    selected = [
        record
        for route in sorted(by_route)
        for record in rng.sample(by_route[route], quotas[route])
    ]
    if len(selected) != target:
        raise RuntimeError("route-stratified anchor allocation missed its exact target")
    return selected


def _episode_map(rows: Iterable[dict[str, Any]], expected_count: int) -> dict[str, dict[str, Any]]:
    episodes: dict[str, dict[str, Any]] = {}
    source_indices: set[int] = set()
    for row in rows:
        episode_id = row.get("episode_id")
        source_idx = row.get("source_idx")
        if not isinstance(episode_id, str) or not episode_id:
            raise SystemExit("every EpisodeSpec must have a non-empty string episode_id")
        if not isinstance(source_idx, int) or isinstance(source_idx, bool):
            raise SystemExit(f"{episode_id}: source_idx must be an integer")
        metadata = row.get("metadata")
        route = metadata.get("route") if isinstance(metadata, Mapping) else None
        if not isinstance(route, str) or not route:
            raise SystemExit(f"{episode_id}: metadata.route must be a non-empty string")
        if episode_id in episodes or source_idx in source_indices:
            raise SystemExit(f"duplicate episode identity: {episode_id}/{source_idx}")
        episodes[episode_id] = row
        source_indices.add(source_idx)
    if len(episodes) != expected_count:
        raise SystemExit(
            f"Step-7 population has {len(episodes)} episodes; expected {expected_count}"
        )
    return episodes


def _expected_attempt_reward(goal: bool, u: int, reward: Mapping[str, Any]) -> float:
    if not goal:
        return 0.0
    return max(
        float(reward["success_floor"]),
        float(reward["maximum"]) - float(reward["penalty_per_violation"]) * u,
    )


def _validate_attempt(
    episode_id: str,
    attempt: Any,
    *,
    number: int,
    phase: str,
    temperature: float,
    seed: int,
    reward_policy: Mapping[str, Any],
) -> None:
    required = {
        "attempt", "phase", "temperature", "seed", "reward",
        "goal_reached", "clean_trace", "U", "outcome",
    }
    if not isinstance(attempt, Mapping) or not required <= set(attempt):
        raise SystemExit(f"{episode_id}: attempt {number} is incomplete")
    if attempt["attempt"] != number or attempt["phase"] != phase:
        raise SystemExit(f"{episode_id}: attempt {number} has wrong phase/index")
    if not _is_finite_number(attempt["temperature"]) \
            or float(attempt["temperature"]) != temperature \
            or not isinstance(attempt["seed"], int) \
            or isinstance(attempt["seed"], bool) \
            or attempt["seed"] != seed:
        raise SystemExit(f"{episode_id}: attempt {number} has wrong sampling identity")
    goal = attempt["goal_reached"]
    clean = attempt["clean_trace"]
    u = attempt["U"]
    value = attempt["reward"]
    if not isinstance(goal, bool) or not isinstance(clean, bool):
        raise SystemExit(f"{episode_id}: attempt {number} has non-boolean outcome flags")
    if not isinstance(u, int) or isinstance(u, bool) or u < 0:
        raise SystemExit(f"{episode_id}: attempt {number} has invalid U")
    if clean != (goal and u == 0):
        raise SystemExit(f"{episode_id}: attempt {number} has inconsistent clean_trace")
    expected_reward = _expected_attempt_reward(goal, u, reward_policy)
    if not _is_finite_number(value) or not math.isclose(
        float(value), expected_reward, rel_tol=0.0, abs_tol=1e-9
    ):
        raise SystemExit(f"{episode_id}: attempt {number} violates the reward formula")

    outcome = attempt["outcome"]
    if not isinstance(outcome, Mapping) or outcome.get("terminal") is not True:
        raise SystemExit(f"{episode_id}: attempt {number} lacks a terminal outcome")
    if outcome.get("goal_reached") is not goal or outcome.get("clean_trace") is not clean:
        raise SystemExit(f"{episode_id}: attempt {number} disagrees with its outcome")
    violations = outcome.get("violations")
    if not isinstance(violations, Sequence) or isinstance(violations, (str, bytes)):
        raise SystemExit(f"{episode_id}: attempt {number} lacks violation diagnostics")
    computed_u = sum(item in RECOVERABLE_VIOLATIONS for item in violations)
    if computed_u != u:
        raise SystemExit(f"{episode_id}: attempt {number} U disagrees with violations")
    breakdown = outcome.get("reward")
    if not isinstance(breakdown, Mapping) \
            or breakdown.get("violation_count") != u \
            or not _is_finite_number(breakdown.get("total")) \
            or not math.isclose(
                float(breakdown["total"]), float(value), rel_tol=0.0, abs_tol=1e-9
            ):
        raise SystemExit(f"{episode_id}: attempt {number} reward diagnostics disagree")


def _category_for(attempts: Sequence[Mapping[str, Any]], maximum: float) -> str:
    """Classify a probe group using every applicable trajectory outcome."""
    if not attempts:
        raise ValueError("cannot classify an empty probe group")
    if all(
        math.isclose(float(attempt["reward"]), maximum, rel_tol=0.0, abs_tol=1e-9)
        for attempt in attempts
    ):
        return "solved"
    if all(not attempt["goal_reached"] for attempt in attempts):
        return "unsolved"
    return "mixed"


def _validate_filter_contract(
    filter_doc: dict[str, Any],
    episodes: dict[str, dict[str, Any]],
    expected_count: int,
) -> None:
    meta = filter_doc.get("meta")
    records = filter_doc.get("records")
    if filter_doc.get("schema_version") != PROBE_SCHEMA:
        raise SystemExit("unsupported task-filter schema; RUN #2 requires v3")
    if not isinstance(meta, dict) or not isinstance(records, list):
        raise SystemExit("task filter lacks meta/records")

    expected_meta = {
        "initial_count": 4,
        "initial_temperature": 0.9,
        "retry_temperature": 0.8,
        "retry_count": 4,
        "anchor_fraction": 0.15,
        "anchor_sampling": "route_stratified_exact_target",
        "seed": 42,
    }
    for key, expected in expected_meta.items():
        if not _strict_equal(meta.get(key), expected):
            raise SystemExit(f"task filter meta.{key} differs from RUN #2: {meta.get(key)!r}")
    reward_policy = meta.get("reward_policy")
    if not isinstance(reward_policy, Mapping):
        raise SystemExit("task filter lacks reward_policy identity")
    _reward_policy({"reward": reward_policy})
    _validate_probe_behavior_identity(meta.get("behavior_files_sha256"))

    by_id: dict[str, dict[str, Any]] = {}
    counts = {category: 0 for category in CATEGORIES}
    initial_counts = {category: 0 for category in CATEGORIES}
    attempts_total = 0
    positions = {episode_id: pos for pos, episode_id in enumerate(episodes)}
    for pos, record in enumerate(records):
        if not isinstance(record, dict):
            raise SystemExit(f"probe record {pos} is not an object")
        episode_id = record.get("episode_id")
        if not isinstance(episode_id, str) or episode_id not in episodes:
            raise SystemExit(f"probe record {pos} has unknown episode_id")
        if episode_id in by_id:
            raise SystemExit(f"duplicate probe record for {episode_id}")
        source = episodes[episode_id]
        if record.get("source_idx") != source["source_idx"] \
                or record.get("route") != source["metadata"]["route"]:
            raise SystemExit(f"{episode_id}: probe labels differ from its EpisodeSpec")
        attempts = record.get("attempts")
        initial_count = int(meta["initial_count"])
        if not isinstance(attempts, list) or len(attempts) < initial_count:
            raise SystemExit(
                f"{episode_id}: probe must contain all {initial_count} initial attempts"
            )
        initial_seed = int(meta["seed"]) + positions[episode_id] * 10_000
        for attempt_no, attempt in enumerate(attempts[:initial_count]):
            _validate_attempt(
                episode_id,
                attempt,
                number=attempt_no,
                phase="initial",
                temperature=float(meta["initial_temperature"]),
                seed=initial_seed + attempt_no * 100,
                reward_policy=reward_policy,
            )
        initial_category = _category_for(
            attempts[:initial_count], float(reward_policy["maximum"])
        )
        expected_attempts = initial_count + (
            int(meta["retry_count"]) if initial_category == "unsolved" else 0
        )
        if len(attempts) != expected_attempts:
            raise SystemExit(
                f"{episode_id}: expected {expected_attempts} adaptive attempts; got {len(attempts)}"
            )
        for retry_no, attempt in enumerate(attempts[initial_count:], initial_count):
            _validate_attempt(
                episode_id,
                attempt,
                number=retry_no,
                phase="retry",
                temperature=float(meta["retry_temperature"]),
                seed=initial_seed + retry_no * 100,
                reward_policy=reward_policy,
            )
        category = _category_for(attempts, float(reward_policy["maximum"]))
        if record.get("category") != category:
            raise SystemExit(f"{episode_id}: category does not match adaptive attempts")
        by_id[episode_id] = record
        counts[category] += 1
        initial_counts[initial_category] += 1
        attempts_total += len(attempts)

    if len(by_id) != expected_count or set(by_id) != set(episodes):
        raise SystemExit("task filter must contain one record per EpisodeSpec")
    if meta.get("episodes_total") != expected_count or meta.get("complete_population") is not True:
        raise SystemExit("task filter does not certify the complete population")
    for category, count in counts.items():
        if meta.get(category) != count:
            raise SystemExit(f"task filter meta.{category} does not match records")
    for category, count in initial_counts.items():
        key = f"initial_{category}"
        if meta.get(key) != count:
            raise SystemExit(f"task filter meta.{key} does not match records")
    if meta.get("attempts_total") != attempts_total:
        raise SystemExit("task filter meta.attempts_total does not match records")
    expected_total = int(meta["initial_count"]) * expected_count \
        + int(meta["retry_count"]) * initial_counts["unsolved"]
    if attempts_total != expected_total:
        raise SystemExit("task filter total does not match the adaptive probe formula")

    maximum_records = [
        record for record in by_id.values() if record["category"] == "solved"
    ]
    expected_anchors = {
        record["episode_id"]
        for record in _sample_anchors(
            maximum_records,
            fraction=float(meta["anchor_fraction"]),
            seed=int(meta["seed"]),
        )
    }
    always_retained = {
        episode_id
        for episode_id, record in by_id.items()
        if record["category"] == "mixed"
    }
    expected_retained = always_retained | expected_anchors
    train_on = filter_doc.get("train_on")
    if not isinstance(train_on, list):
        raise SystemExit("task filter train_on must be a list")
    retained: dict[str, str] = {}
    for item in train_on:
        if not isinstance(item, dict) or not isinstance(item.get("episode_id"), str):
            raise SystemExit("task filter contains an invalid retained item")
        episode_id = item["episode_id"]
        if episode_id in retained or episode_id not in by_id:
            raise SystemExit(f"invalid or duplicate retained episode: {episode_id}")
        record = by_id[episode_id]
        if item.get("source_idx") != record["source_idx"] \
                or item.get("route") != record["route"]:
            raise SystemExit(f"{episode_id}: retained labels differ from probe record")
        expected_selection = (
            "anchor" if episode_id in expected_anchors else record["category"]
        )
        if item.get("selection") != expected_selection:
            raise SystemExit(f"{episode_id}: invalid retention selection")
        retained[episode_id] = expected_selection
    if set(retained) != expected_retained:
        raise SystemExit("task filter retained set differs from RUN #2 policy")
    if meta.get("anchors_kept") != len(expected_anchors):
        raise SystemExit("task filter meta.anchors_kept does not match anchors")
    if meta.get("train_on") != len(expected_retained):
        raise SystemExit("task filter meta.train_on does not match retained episodes")


def _retained_ids(filter_doc: dict[str, Any]) -> list[str]:
    train_on = filter_doc.get("train_on")
    if not isinstance(train_on, list) or not train_on:
        raise SystemExit("task filter must contain at least one retained episode")
    return [str(item["episode_id"]) for item in train_on]


def _prompt_for(spec: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = spec.get("initial_messages")
    if not isinstance(prompt, list) or len(prompt) < 2:
        raise SystemExit(f"{spec.get('episode_id')}: initial_messages must contain system + user")
    if prompt[0].get("role") != "system" or prompt[-1].get("role") != "user":
        raise SystemExit(f"{spec.get('episode_id')}: initial prompt must start system and end user")
    if any(message.get("role") not in {"system", "user"} for message in prompt):
        raise SystemExit(f"{spec.get('episode_id')}: initial prompt leaks future messages")
    return [dict(message) for message in prompt]


def _workflow_for(spec: Mapping[str, Any]) -> str:
    episode_id = spec.get("episode_id")
    metadata = spec.get("metadata")
    if not isinstance(metadata, Mapping):
        raise SystemExit(f"{episode_id}: metadata must be an object")
    route = metadata.get("route")
    if not isinstance(route, str) or not route.startswith("W") or "-" not in route:
        raise SystemExit(f"{episode_id}: metadata.route is not a workflow route")
    workflow = route.split("-", 1)[0]
    declared = metadata.get("workflow")
    expected_declared = int(workflow[1:]) if workflow[1:].isdigit() else workflow
    if declared not in (workflow, expected_declared):
        raise SystemExit(
            f"{episode_id}: metadata.workflow disagrees with route {route}"
        )
    return workflow


def _row_for_episode(
    spec: dict[str, Any],
    *,
    include_sampler_labels: bool,
) -> dict[str, Any]:
    extra_info: dict[str, Any] = {"episode_id": spec["episode_id"]}
    if include_sampler_labels:
        extra_info["workflow"] = _workflow_for(spec)
    return {
        "data_source": "cn_travel_world_policy",
        "agent_name": "cn_travel",
        "prompt": _prompt_for(spec),
        "extra_info": extra_info,
    }


def _validate_probe_identity(
    meta: dict[str, Any],
    config: dict[str, Any],
    *,
    episodes_sha256: str,
    world_sha256: str,
) -> None:
    expected = {
        "model": config["probe_model"],
        "initial_count": config["probe_initial_count"],
        "initial_temperature": config["probe_initial_temperature"],
        "retry_temperature": config["probe_retry_temperature"],
        "retry_count": config["probe_retry_count"],
        "anchor_fraction": config["probe_anchor_fraction"],
        "max_tokens": config["probe_max_tokens"],
        "seed": config["seed"],
        "expected_episode_count": config["expected_episode_count"],
        "episodes_sha256": episodes_sha256,
        "world_sha256": world_sha256,
        "sft_adapter_sha256": config["sft_adapter_sha256"],
        "sft_reference_weights_sha256": config["sft_reference_weights_sha256"],
        "reward_policy": _reward_policy(config),
    }
    mismatched = {
        key: (meta.get(key), value)
        for key, value in expected.items()
        if not _strict_equal(meta.get(key), value)
    }
    if mismatched:
        raise SystemExit(f"task filter has a different probe identity: {mismatched}")
    _validate_probe_behavior_identity(meta.get("behavior_files_sha256"))


def build_rows(
    episodes_path: pathlib.Path,
    filter_path: pathlib.Path,
    *,
    expected_count: int = 909,
) -> list[dict[str, Any]]:
    episodes = _episode_map(_load_jsonl(episodes_path), expected_count)
    filter_doc = _load_json(filter_path)
    if not isinstance(filter_doc, dict):
        raise SystemExit(f"task filter is not an object: {filter_path}")
    meta = filter_doc.get("meta")
    if not isinstance(meta, dict) or meta.get("episodes_sha256") != _sha256(episodes_path):
        raise SystemExit("task filter was probed against different EpisodeSpec bytes")
    _validate_filter_contract(filter_doc, episodes, expected_count)
    return [
        _row_for_episode(episodes[episode_id], include_sampler_labels=False)
        for episode_id in _retained_ids(filter_doc)
    ]


def build_all_rows(
    episodes_path: pathlib.Path,
    *,
    expected_count: int = 909,
) -> list[dict[str, Any]]:
    """Pack all sealed EpisodeSpecs once, ordered by their stable source index."""
    episodes = _episode_map(_load_jsonl(episodes_path), expected_count)
    ordered = sorted(
        episodes.values(),
        key=lambda item: (item["source_idx"], item["episode_id"]),
    )
    rows = [
        _row_for_episode(spec, include_sampler_labels=True)
        for spec in ordered
    ]
    if len({row["extra_info"]["episode_id"] for row in rows}) != expected_count:
        raise SystemExit("all-episode packing did not preserve one row per EpisodeSpec")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Pack whole episodes for CN Travel VERL")
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=pathlib.Path)
    parser.add_argument("--filter", dest="filter_path", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = _load_json(config_path)
    if not isinstance(config, dict):
        raise SystemExit(f"config is not an object: {config_path}")
    episodes_path = (args.episodes or _resolve(config["episodes_file"])).resolve()
    world_path = _resolve(config["world_file"]).resolve()
    output_path = (args.out or _resolve(config["train_parquet"])).resolve()
    expected_count = int(config.get("expected_episode_count", 909))
    episodes_digest = _sha256(episodes_path)
    world_digest = _sha256(world_path)
    if episodes_digest != config["episodes_sha256"]:
        raise SystemExit("EpisodeSpec bytes do not match the sealed config hash")
    if world_digest != config["world_sha256"]:
        raise SystemExit("Synthetic China bytes do not match the sealed config hash")
    population = config.get("training_population", "probe_filtered")
    if population == "all_episodes":
        forbidden = sorted(
            key for key in config
            if key == "task_filter" or key.startswith("probe_")
        )
        if forbidden:
            raise SystemExit(
                "all-episode training must not carry probe/filter settings: "
                + ", ".join(forbidden)
            )
        if args.filter_path is not None:
            raise SystemExit("--filter is invalid for all-episode training")
        rows = build_all_rows(episodes_path, expected_count=expected_count)
    elif population == "probe_filtered":
        filter_value = args.filter_path or _resolve(config["task_filter"])
        filter_path = pathlib.Path(filter_value).resolve()
        filter_doc = _load_json(filter_path)
        if not isinstance(filter_doc, dict) or not isinstance(filter_doc.get("meta"), dict):
            raise SystemExit("task filter lacks a meta object")
        _validate_probe_identity(
            filter_doc["meta"],
            config,
            episodes_sha256=episodes_digest,
            world_sha256=world_digest,
        )
        rows = build_rows(episodes_path, filter_path, expected_count=expected_count)
    else:
        raise SystemExit(f"unsupported training_population: {population!r}")

    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas plus a parquet engine are required to write VERL data") from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(fd)
    tmp_path = pathlib.Path(tmp_name)
    try:
        pd.DataFrame(rows).to_parquet(tmp_path, index=False)
        os.replace(tmp_path, output_path)
        output_path.chmod(0o644)
    finally:
        tmp_path.unlink(missing_ok=True)
    print(f"packed {len(rows)} whole episodes -> {output_path}")
    labels = "episode_id + workflow" if population == "all_episodes" else "episode_id"
    print(f"row contract: initial prompt + {labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
