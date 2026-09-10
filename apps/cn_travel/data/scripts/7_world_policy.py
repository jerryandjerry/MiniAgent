#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 7 -- compile whole-trajectory world-policy episodes.

Inputs are the 909 frozen training conversations from Step 5 and their matching
Step-2 materialized records.  This compiler is deliberately offline: it never
calls a tool, an LLM, or the network.

Outputs:

* ``7_world_policy/world_policy_episodes.jsonl`` -- one private causal episode
  specification per complete conversation.
* ``7_world_policy/synthetic_china_world_v1.json`` -- one order-independent,
  immutable mapping from normalized tool calls to canonical frozen results.

When multiple Step-2 records map the same normalized tool call to different
results, the world selects the minimum canonical-result SHA-256 independent of
input order and records every candidate and source occurrence. Source metadata
remains unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pathlib
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Iterable

from data.paths import PATHS

DATA = PATHS.root
DEFAULT_MERGED = DATA / "5_merged" / "train_validation.json"
DEFAULT_MATERIALIZED = DATA / "2_materialized"
DEFAULT_OUTPUT = DATA / "7_world_policy"

EPISODE_SCHEMA = "cn_travel.world_policy.v1"
WORLD_SCHEMA = "cn_travel.synthetic_china_world.v1"
WORLD_VERSION = "synthetic-china-world-v1"
EXPECTED_EPISODES = 909
VALID_STATUSES = ("ok", "empty", "error")

TOOL_STATE_NAMES = {
    "search_travel_guide": "guide",
    "get_weather_info": "weather",
    "query_route": "route",
    "recommend_hotels": "hotels",
    "get_hotel_reviews": "reviews",
}

WORKFLOW_GOALS = {
    1: "travel_plan",
    2: "route_query",
    4: "travel_chitchat",
    5: "off_topic_refusal",
}


def normalize_json(value: Any) -> Any:
    """Normalize a JSON value for frozen-world identity.

    Object keys are sorted during serialization and all strings (including
    keys) are Unicode NFC.  Values and JSON types are otherwise preserved: no
    whitespace trimming, optional-default insertion, numeric coercion, or city
    alias rewriting is performed.
    """

    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [normalize_json(item) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str):
                raise TypeError(f"JSON object key is not a string: {raw_key!r}")
            key = unicodedata.normalize("NFC", raw_key)
            if key in normalized:
                raise ValueError(f"keys collide after NFC normalization: {key!r}")
            normalized[key] = normalize_json(raw_value)
        return normalized
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise TypeError(f"not a JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return the canonical UTF-8 JSON serialization used by this step."""

    return json.dumps(
        normalize_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def world_key(tool: str, arguments: dict[str, Any]) -> str:
    """Canonical frozen-world key; exported for the runtime and tests."""

    if not isinstance(tool, str) or not tool:
        raise ValueError("tool name must be a non-empty string")
    if not isinstance(arguments, dict):
        raise TypeError("tool arguments must be a JSON object")
    return canonical_json({"tool": tool, "arguments": arguments})


def _load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc


def _parse_json_object(text: str, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must decode to a JSON object")
    return value


def _conversation_tool_trace(conversation: list[dict[str, Any]], idx: int) -> list[dict[str, Any]]:
    """Extract the exact ordered assistant-call/tool-result trace."""

    results: dict[str, dict[str, Any]] = {}
    result_order: list[str] = []
    for pos, message in enumerate(conversation):
        if message.get("role") != "tool":
            continue
        call_id = message.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError(f"idx {idx}: tool message {pos} has no tool_call_id")
        if call_id in results:
            raise ValueError(f"idx {idx}: duplicate tool result for {call_id}")
        results[call_id] = _parse_json_object(
            message.get("content"), label=f"idx {idx} tool result {call_id}"
        )
        result_order.append(call_id)

    trace: list[dict[str, Any]] = []
    call_ids: list[str] = []
    for pos, message in enumerate(conversation):
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            raise ValueError(f"idx {idx}: assistant message {pos} tool_calls is not a list")
        for call in tool_calls:
            try:
                call_id = call["id"]
                function = call["function"]
                name = function["name"]
                raw_arguments = function["arguments"]
            except (KeyError, TypeError) as exc:
                raise ValueError(f"idx {idx}: malformed tool call at message {pos}") from exc
            if call.get("type") != "function":
                raise ValueError(f"idx {idx}: tool call {call_id} is not type=function")
            if not isinstance(call_id, str) or not isinstance(name, str):
                raise ValueError(f"idx {idx}: malformed tool-call identity at message {pos}")
            if call_id in call_ids:
                raise ValueError(f"idx {idx}: duplicate tool call id {call_id}")
            if call_id not in results:
                raise ValueError(f"idx {idx}: tool call {call_id} has no result")
            arguments = _parse_json_object(
                raw_arguments, label=f"idx {idx} arguments for {call_id}"
            )
            call_ids.append(call_id)
            trace.append(
                {
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                    "result": results[call_id],
                }
            )

    if set(call_ids) != set(results):
        orphaned = sorted(set(results) - set(call_ids))
        raise ValueError(f"idx {idx}: orphaned tool results: {orphaned}")
    if call_ids != result_order:
        raise ValueError(f"idx {idx}: tool results are not ordered like their calls")
    return trace


def validate_joined_row(
    merged_row: dict[str, Any], materialized: dict[str, Any]
) -> list[dict[str, Any]]:
    """Validate one Step-5/Step-2 join and return its visible trace."""

    if set(merged_row) != {"conversation", "metadata"}:
        raise ValueError("merged row must contain exactly conversation and metadata")
    metadata = merged_row["metadata"]
    if not isinstance(metadata, dict) or not isinstance(metadata.get("idx"), int):
        raise ValueError("merged row has no integer metadata.idx")
    idx = metadata["idx"]
    if materialized.get("metadata") != metadata:
        raise ValueError(f"idx {idx}: Step-5 metadata differs from Step-2 metadata")

    conversation = merged_row["conversation"]
    if not isinstance(conversation, list) or len(conversation) < 3:
        raise ValueError(f"idx {idx}: conversation is incomplete")
    if [message.get("role") for message in conversation[:2]] != ["system", "user"]:
        raise ValueError(f"idx {idx}: conversation must start with system then user")
    if conversation[-1].get("role") != "assistant" or conversation[-1].get("tool_calls"):
        raise ValueError(f"idx {idx}: conversation must end with a plain assistant response")
    if sum(message.get("role") == "user" for message in conversation) != metadata.get("turns"):
        raise ValueError(f"idx {idx}: metadata.turns disagrees with conversation")

    source_trace = _conversation_tool_trace(conversation, idx)
    visible = materialized.get("visible_tool_trace")
    if not isinstance(visible, list) or len(visible) != len(source_trace):
        raise ValueError(f"idx {idx}: visible tool-trace length differs")
    for ordinal, (source, frozen) in enumerate(zip(source_trace, visible)):
        if frozen.get("ordinal") != ordinal:
            raise ValueError(f"idx {idx}: visible trace ordinal {ordinal} is invalid")
        for field in ("name", "arguments", "result"):
            if source[field] != frozen.get(field):
                raise ValueError(
                    f"idx {idx}: Step-5 {field} differs from Step-2 visible trace at {ordinal}"
                )

    resolved = materialized.get("resolved")
    if not isinstance(resolved, dict):
        raise ValueError(f"idx {idx}: missing resolved state")
    for field in ("canonical_slots", "turn_plan", "assistant_asks", "response_policy"):
        if field not in resolved:
            raise ValueError(f"idx {idx}: missing resolved.{field}")
    if len(resolved["turn_plan"]) != metadata["turns"]:
        raise ValueError(f"idx {idx}: turn_plan length differs from metadata.turns")
    if len(resolved["assistant_asks"]) != metadata["turns"] - 1:
        raise ValueError(f"idx {idx}: assistant_asks length differs from metadata.turns")

    observed_asks = [
        message["content"]
        for pos, message in enumerate(conversation[2:-1], start=2)
        if message.get("role") == "assistant" and not message.get("tool_calls")
    ]
    expected_asks = [ask.get("text") for ask in resolved["assistant_asks"]]
    if observed_asks != expected_asks:
        raise ValueError(f"idx {idx}: Step-5 questions differ from resolved assistant_asks")
    if metadata.get("workflow") == 5:
        canonical_refusal = resolved["response_policy"].get("canonical_refusal")
        if conversation[-1].get("content") != canonical_refusal:
            raise ValueError(f"idx {idx}: W5 final differs from canonical refusal")

    if not isinstance(materialized.get("context"), dict):
        raise ValueError(f"idx {idx}: missing structured context")
    for trace_name in ("visible_tool_trace", "selection_trace"):
        trace = materialized.get(trace_name)
        if not isinstance(trace, list):
            raise ValueError(f"idx {idx}: {trace_name} is not a list")
        for ordinal, record in enumerate(trace):
            for field in ("execution_id", "name", "arguments", "result"):
                if field not in record:
                    raise ValueError(f"idx {idx}: {trace_name}[{ordinal}] missing {field}")
            world_key(record["name"], record["arguments"])
            if not isinstance(record["result"], dict):
                raise ValueError(f"idx {idx}: {trace_name}[{ordinal}].result is not an object")
            if record["result"].get("status") not in VALID_STATUSES:
                raise ValueError(f"idx {idx}: {trace_name}[{ordinal}] has invalid status")

    return visible


def load_sources(
    merged_path: pathlib.Path = DEFAULT_MERGED,
    materialized_dir: pathlib.Path = DEFAULT_MATERIALIZED,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Load and validate exactly the 909 joined training records."""

    merged = _load_json(merged_path)
    if not isinstance(merged, list) or len(merged) != EXPECTED_EPISODES:
        size = len(merged) if isinstance(merged, list) else type(merged).__name__
        raise ValueError(f"{merged_path}: expected {EXPECTED_EPISODES} rows, got {size}")

    materialized_by_idx: dict[int, dict[str, Any]] = {}
    seen: set[int] = set()
    for row in merged:
        if not isinstance(row, dict) or not isinstance(row.get("metadata"), dict):
            raise ValueError("merged dataset contains a malformed row")
        idx = row["metadata"].get("idx")
        if not isinstance(idx, int):
            raise ValueError("merged dataset contains a non-integer metadata.idx")
        if idx in seen:
            raise ValueError(f"duplicate metadata.idx {idx}")
        seen.add(idx)
        path = materialized_dir / f"{idx:04d}.json"
        if not path.is_file():
            raise ValueError(f"missing matching materialized record: {path}")
        materialized = _load_json(path)
        if not isinstance(materialized, dict):
            raise ValueError(f"{path}: materialized record is not an object")
        validate_joined_row(row, materialized)
        materialized_by_idx[idx] = materialized

    return merged, materialized_by_idx


def _occurrence(idx: int, trace_name: str, ordinal: int, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_idx": idx,
        "trace": "visible" if trace_name == "visible_tool_trace" else "selection",
        "ordinal": ordinal,
        "execution_id": record["execution_id"],
    }


def build_world(materialized_by_idx: dict[int, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Build the single canonical Synthetic China snapshot.

    The returned lookup is keyed by :func:`world_key` and points to each chosen
    world entry.  Iteration order of ``materialized_by_idx`` cannot affect the
    result.
    """

    candidates: dict[str, dict[str, Any]] = {}
    for idx, materialized in materialized_by_idx.items():
        for trace_name in ("visible_tool_trace", "selection_trace"):
            for ordinal, record in enumerate(materialized[trace_name]):
                tool = normalize_json(record["name"])
                arguments = normalize_json(record["arguments"])
                key = world_key(tool, arguments)
                bucket = candidates.setdefault(
                    key,
                    {
                        "tool": tool,
                        "arguments": arguments,
                        "results": defaultdict(list),
                    },
                )
                result_hash = sha256_json(record["result"])
                bucket["results"][result_hash].append(
                    _occurrence(idx, trace_name, ordinal, record)
                )

    entries: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    lookup: dict[str, dict[str, Any]] = {}
    for key in sorted(candidates):
        bucket = candidates[key]
        candidate_results: list[dict[str, Any]] = []
        for result_hash in sorted(bucket["results"]):
            occurrences = sorted(
                bucket["results"][result_hash],
                key=lambda item: (
                    item["source_idx"], item["trace"], item["ordinal"], item["execution_id"]
                ),
            )
            first = occurrences[0]
            source_record = materialized_by_idx[first["source_idx"]][
                "visible_tool_trace" if first["trace"] == "visible" else "selection_trace"
            ][first["ordinal"]]
            candidate_results.append(
                {
                    "result_sha256": result_hash,
                    "result": copy.deepcopy(source_record["result"]),
                    "occurrences": occurrences,
                }
            )

        selected = candidate_results[0]
        key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
        entry = {
            "key_sha256": key_hash,
            "execution_id": f"sha256:{key_hash}",
            "tool": bucket["tool"],
            "arguments": bucket["arguments"],
            "result": copy.deepcopy(selected["result"]),
            "result_sha256": selected["result_sha256"],
        }
        entries.append(entry)
        lookup[key] = entry
        if len(candidate_results) > 1:
            conflicts.append(
                {
                    "key_sha256": key_hash,
                    "tool": bucket["tool"],
                    "arguments": bucket["arguments"],
                    "candidate_results": candidate_results,
                    "resolution": {
                        "rule": "minimum_canonical_result_sha256",
                        "selected_result_sha256": selected["result_sha256"],
                    },
                }
            )

    payload = {
        "schema_version": WORLD_SCHEMA,
        "world_version": WORLD_VERSION,
        "normalization": {
            "algorithm": "json-nfc-sort-keys-v1",
            "string_normalization": "NFC",
            "object_key_order": "lexicographic",
            "coercions": "none",
        },
        "entry_count": len(entries),
        "conflict_count": len(conflicts),
        "entries": entries,
        "conflicts": conflicts,
    }
    content_sha256 = sha256_json(payload)
    world = {
        **payload,
        "content_sha256": content_sha256,
        "world_id": f"{WORLD_VERSION}:{content_sha256}",
    }
    return world, lookup


def _slot_name(workflow: int, slot: str) -> str:
    mappings = {
        1: {"city": "destination", "date": "travel_date", "start_date": "travel_date",
            "city_entity": "destination_entity"},
        2: {"origin": "route_origin", "destination": "route_destination"},
        3: {"city": "hotel_location", "hotel": "hotel_name"},
    }
    return mappings.get(workflow, {}).get(slot, slot)


def _private_slots(workflow: int, canonical_slots: dict[str, Any]) -> dict[str, Any]:
    private: dict[str, Any] = {}
    for key, value in canonical_slots.items():
        target = _slot_name(workflow, key)
        if target in private and private[target] != value:
            raise ValueError(f"private-slot normalization collision at {target}")
        private[target] = copy.deepcopy(value)
    return private


def _goal_name(metadata: dict[str, Any]) -> str:
    workflow = metadata["workflow"]
    if workflow != 3:
        return WORKFLOW_GOALS[workflow]
    return (
        "hotel_reviews"
        if metadata.get("slots", {}).get("path") == "reviews_only"
        else "hotel_recommendation"
    )


def _tool_state_name(tool: str) -> str:
    return TOOL_STATE_NAMES.get(tool, tool.replace("_", "-"))


def _state_id(kind: str, ordinal: int, label: str = "") -> str:
    suffix = f"_{label}" if label else ""
    return f"{kind}_{ordinal:02d}{suffix}"


def _conversation_actions(
    merged_row: dict[str, Any], materialized: dict[str, Any]
) -> list[dict[str, Any]]:
    """Decode the frozen conversation into question/tool/terminal actions."""

    idx = merged_row["metadata"]["idx"]
    workflow = merged_row["metadata"]["workflow"]
    conversation = merged_row["conversation"]
    asks = materialized["resolved"]["assistant_asks"]
    turn_plan = materialized["resolved"]["turn_plan"]
    visible = materialized["visible_tool_trace"]
    actions: list[dict[str, Any]] = []
    ask_pos = 0
    tool_pos = 0
    pos = 2
    while pos < len(conversation):
        message = conversation[pos]
        if message.get("role") != "assistant":
            raise ValueError(f"idx {idx}: expected assistant message at position {pos}")
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            calls: list[dict[str, Any]] = []
            source_results: list[dict[str, Any]] = []
            for offset, call in enumerate(tool_calls):
                if pos + 1 + offset >= len(conversation):
                    raise ValueError(f"idx {idx}: incomplete tool round at position {pos}")
                tool_message = conversation[pos + 1 + offset]
                if tool_message.get("role") != "tool" or tool_message.get("tool_call_id") != call["id"]:
                    raise ValueError(f"idx {idx}: tool result order mismatch at position {pos}")
                frozen = visible[tool_pos]
                arguments = _parse_json_object(
                    call["function"]["arguments"],
                    label=f"idx {idx} arguments for {call['id']}",
                )
                calls.append({"name": call["function"]["name"], "arguments": arguments})
                source_results.append(copy.deepcopy(frozen["result"]))
                tool_pos += 1
            actions.append({"kind": "tool", "calls": calls, "source_results": source_results})
            pos += 1 + len(tool_calls)
            continue

        if pos == len(conversation) - 1:
            actions.append(
                {
                    "kind": "refusal" if workflow == 5 else "final",
                    "reference_response": message["content"],
                }
            )
            pos += 1
            continue

        if ask_pos >= len(asks) or pos + 1 >= len(conversation):
            raise ValueError(f"idx {idx}: unexpected plain assistant message at position {pos}")
        user_message = conversation[pos + 1]
        ask = asks[ask_pos]
        if message["content"] != ask["text"] or user_message.get("role") != "user":
            raise ValueError(f"idx {idx}: question/user-response pair differs from Step 2")
        after_turn = ask.get("after_turn")
        if not isinstance(after_turn, int) or after_turn >= len(turn_plan):
            raise ValueError(f"idx {idx}: assistant ask has invalid after_turn")
        response_turn = turn_plan[after_turn]
        if response_turn.get("turn") != after_turn + 1:
            raise ValueError(f"idx {idx}: response turn does not follow assistant ask")
        actions.append(
            {
                "kind": "ask",
                "slot": _slot_name(workflow, ask["slot"]),
                "user_response": user_message["content"],
                "revealed_slots": [
                    _slot_name(workflow, slot) for slot in response_turn.get("reveal", [])
                ],
                "concealed_slots": [
                    _slot_name(workflow, slot) for slot in response_turn.get("conceal", [])
                ],
                "response_constraints": copy.deepcopy(response_turn.get("constraints", [])),
            }
        )
        ask_pos += 1
        pos += 2

    if ask_pos != len(asks) or tool_pos != len(visible):
        raise ValueError(f"idx {idx}: conversation did not consume all structured asks/tools")
    if not actions or actions[-1]["kind"] not in {"final", "refusal"}:
        raise ValueError(f"idx {idx}: causal trace has no terminal action")
    return actions


def _compile_state_graph(
    actions: list[dict[str, Any]], world_lookup: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Compile a causal graph and its one-credit-per-ID reward milestones."""

    state_ids: list[str] = []
    for ordinal, action in enumerate(actions, start=1):
        if action["kind"] == "ask":
            state_ids.append(_state_id("ask", ordinal, action["slot"]))
        elif action["kind"] == "tool":
            labels = [_tool_state_name(call["name"]) for call in action["calls"]]
            label = labels[0] if len(labels) == 1 else "batch"
            state_ids.append(_state_id("tool", ordinal, label))
        else:
            state_ids.append(_state_id(action["kind"], ordinal))

    if len(state_ids) != len(set(state_ids)):
        raise ValueError("state IDs are not unique")
    terminal_state_id = state_ids[-1]
    graph: list[dict[str, Any]] = []
    for position, (state_id, action) in enumerate(zip(state_ids, actions)):
        next_state = state_ids[position + 1] if position + 1 < len(state_ids) else "done"
        ordinal = position + 1
        if action["kind"] == "ask":
            graph.append(
                {
                    "state_id": state_id,
                    "expected_action": {"type": "Ask", "slot": action["slot"]},
                    "user_response": action["user_response"],
                    "revealed_slots": copy.deepcopy(action["revealed_slots"]),
                    "concealed_slots": copy.deepcopy(action["concealed_slots"]),
                    "response_constraints": copy.deepcopy(action["response_constraints"]),
                    "next_state": next_state,
                    "milestone_ids": {"ask": state_id},
                }
            )
            continue

        if action["kind"] == "tool":
            if len(action["calls"]) != 1:
                raise ValueError("the frozen 909 episodes require one tool call per tool round")
            call = action["calls"][0]
            entry = world_lookup[world_key(call["name"], call["arguments"])]
            expected_status = entry["result"].get("status")
            if expected_status not in VALID_STATUSES:
                raise ValueError(f"canonical result for {call['name']} has invalid status")
            label = _tool_state_name(call["name"])
            call_id = f"call_{ordinal:02d}_{label}_01"
            round_id = f"round_{ordinal:02d}_{label}"
            branch_ids = {
                status: f"branch_{ordinal:02d}_{label}_{status}" for status in VALID_STATUSES
            }
            next_is_tool = actions[position + 1]["kind"] == "tool"
            result_branches = {
                status: (next_state if (not next_is_tool or status == "ok") else terminal_state_id)
                for status in VALID_STATUSES
            }
            graph.append(
                {
                    "state_id": state_id,
                    "expected_action": {"type": "ToolCalls", "calls": copy.deepcopy(action["calls"])},
                    "source_result_status": action["source_results"][0].get("status"),
                    "expected_result_status": expected_status,
                    "result_branches": result_branches,
                    "milestone_ids": {
                        "calls": [call_id],
                        "round": round_id,
                        "branches": branch_ids,
                    },
                }
            )
            continue

        expected_type = "Refusal" if action["kind"] == "refusal" else "Final"
        completion_id = f"valid_{action['kind']}_{ordinal:02d}"
        graph.append(
            {
                "state_id": state_id,
                "expected_action": {"type": expected_type},
                "next_state": "done",
                "milestone_ids": {"completion": completion_id},
            }
        )

    milestones: list[str] = []
    state_by_id = {node["state_id"]: node for node in graph}
    cursor = state_ids[0]
    visited: set[str] = set()
    while cursor != "done":
        if cursor in visited:
            raise ValueError(f"cycle in canonical state path at {cursor}")
        visited.add(cursor)
        node = state_by_id[cursor]
        kind = node["expected_action"]["type"]
        ids = node["milestone_ids"]
        if kind == "Ask":
            milestones.append(ids["ask"])
            cursor = node["next_state"]
        elif kind == "ToolCalls":
            milestones.extend(ids["calls"])
            milestones.append(ids["round"])
            status = node["expected_result_status"]
            milestones.append(ids["branches"][status])
            cursor = node["result_branches"][status]
        else:
            milestones.append(ids["completion"])
            cursor = node["next_state"]
    if len(milestones) != len(set(milestones)):
        raise ValueError("reward milestone IDs are not unique")
    return graph, milestones


def compile_episode(
    merged_row: dict[str, Any],
    materialized: dict[str, Any],
    world: dict[str, Any],
    world_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compile one joined source record into a private episode spec."""

    validate_joined_row(merged_row, materialized)
    metadata = merged_row["metadata"]
    idx = metadata["idx"]
    workflow = metadata["workflow"]
    actions = _conversation_actions(merged_row, materialized)
    state_graph, reward_milestones = _compile_state_graph(actions, world_lookup)
    canonical_slots = materialized["resolved"]["canonical_slots"]
    first_turn = materialized["resolved"]["turn_plan"][0]
    initially_hidden = [_slot_name(workflow, slot) for slot in metadata["omit"]]
    revealed = [_slot_name(workflow, slot) for slot in first_turn["reveal"]]

    canonical_tool_statuses = [
        node["expected_result_status"]
        for node in state_graph
        if node["expected_action"]["type"] == "ToolCalls"
    ]
    world_outcome = canonical_tool_statuses[-1] if canonical_tool_statuses else metadata["expect"]
    return {
        "schema_version": EPISODE_SCHEMA,
        "episode_id": f"cn-travel-{idx:04d}",
        "source_idx": idx,
        "metadata": copy.deepcopy(metadata),
        "world_id": world["world_id"],
        "initial_messages": copy.deepcopy(merged_row["conversation"][:2]),
        "person_state": {
            "public_context": copy.deepcopy(materialized["context"]),
            "private_slots": _private_slots(workflow, canonical_slots),
            "revealed_slots": revealed,
            "initially_hidden": initially_hidden,
            "goal": {
                "workflow": _goal_name(metadata),
                "route": metadata["route"],
                "expected_outcome": metadata["expect"],
                "world_expected_outcome": world_outcome,
            },
        },
        "response_policy": copy.deepcopy(materialized["resolved"]["response_policy"]),
        "initial_state_id": state_graph[0]["state_id"],
        "state_graph": state_graph,
        "reward_milestones": reward_milestones,
        "source": {
            "conversation": f"5_merged/train_validation.json#idx={idx}",
            "materialized": f"2_materialized/{idx:04d}.json",
        },
    }


def compile_dataset(
    merged_path: pathlib.Path = DEFAULT_MERGED,
    materialized_dir: pathlib.Path = DEFAULT_MATERIALIZED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate both sources and compile all episodes plus their shared world."""

    merged, materialized_by_idx = load_sources(merged_path, materialized_dir)
    world, lookup = build_world(materialized_by_idx)
    episodes = [
        compile_episode(row, materialized_by_idx[row["metadata"]["idx"]], world, lookup)
        for row in sorted(merged, key=lambda item: item["metadata"]["idx"])
    ]
    if len(episodes) != EXPECTED_EPISODES:
        raise AssertionError(f"compiled {len(episodes)} episodes, expected {EXPECTED_EPISODES}")
    if len({episode["episode_id"] for episode in episodes}) != EXPECTED_EPISODES:
        raise AssertionError("compiled episode IDs are not unique")
    return episodes, world


def _atomic_write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_outputs(
    episodes: Iterable[dict[str, Any]], world: dict[str, Any], output_dir: pathlib.Path
) -> dict[str, Any]:
    """Write deterministic artifacts and return a compact verification report."""

    episodes = list(episodes)
    episodes_path = output_dir / "world_policy_episodes.jsonl"
    world_path = output_dir / "synthetic_china_world_v1.json"
    episode_text = "".join(canonical_json(episode) + "\n" for episode in episodes)
    world_text = json.dumps(world, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    _atomic_write(episodes_path, episode_text)
    _atomic_write(world_path, world_text)

    routes = Counter(episode["metadata"]["route"] for episode in episodes)
    return {
        "episodes": len(episodes),
        "routes": len(routes),
        "world_entries": world["entry_count"],
        "conflicts": world["conflict_count"],
        "world_id": world["world_id"],
        "world_policy_episodes_sha256": sha256_file(episodes_path),
        "synthetic_china_world_v1_sha256": sha256_file(world_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", type=pathlib.Path, default=DEFAULT_MERGED)
    parser.add_argument("--materialized", type=pathlib.Path, default=DEFAULT_MATERIALIZED)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    episodes, world = compile_dataset(args.merged, args.materialized)
    report = write_outputs(episodes, world, args.output)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
