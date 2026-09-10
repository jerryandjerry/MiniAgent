#!/usr/bin/env python3
"""Focused offline tests for CN Travel Step-7 world-policy compilation."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import pathlib
import sys

import pytest

from project_paths import DATA_ROOT

SCRIPT = DATA_ROOT / "scripts" / "7_world_policy.py"
MERGED = DATA_ROOT / "5_merged" / "train_validation.json"
MATERIALIZED = DATA_ROOT / "2_materialized"

_spec = importlib.util.spec_from_file_location("world_policy_compiler", SCRIPT)
C = importlib.util.module_from_spec(_spec)
sys.modules["world_policy_compiler"] = C
_spec.loader.exec_module(C)


@pytest.fixture(scope="module")
def compiled():
    episodes, world = C.compile_dataset(MERGED, MATERIALIZED)
    return episodes, world


def _episode_map(episodes):
    return {episode["source_idx"]: episode for episode in episodes}


def _world_entry(world, tool, arguments):
    matches = [
        entry
        for entry in world["entries"]
        if entry["tool"] == tool and entry["arguments"] == arguments
    ]
    assert len(matches) == 1
    return matches[0]


def test_compiles_exact_train_population_and_preserves_metadata(compiled):
    episodes, world = compiled
    source = json.loads(MERGED.read_text(encoding="utf-8"))
    source_by_idx = {row["metadata"]["idx"]: row for row in source}

    assert len(episodes) == 909
    assert len({episode["episode_id"] for episode in episodes}) == 909
    assert len({episode["metadata"]["route"] for episode in episodes}) == 24
    assert world["entry_count"] == len(world["entries"]) == 925

    for episode in episodes:
        row = source_by_idx[episode["source_idx"]]
        assert episode["metadata"] == row["metadata"]
        assert episode["initial_messages"] == row["conversation"][:2]
        assert episode["world_id"] == world["world_id"]
        assert len(episode["reward_milestones"]) == len(set(episode["reward_milestones"]))


def test_join_validator_rejects_metadata_call_and_result_changes():
    row = next(
        item
        for item in json.loads(MERGED.read_text(encoding="utf-8"))
        if item["metadata"]["idx"] == 53
    )
    materialized = json.loads((MATERIALIZED / "0053.json").read_text(encoding="utf-8"))
    C.validate_joined_row(row, materialized)

    bad_metadata = copy.deepcopy(materialized)
    bad_metadata["metadata"]["route"] = "W1-A"
    with pytest.raises(ValueError, match="metadata differs"):
        C.validate_joined_row(row, bad_metadata)

    bad_call = copy.deepcopy(row)
    call = next(message for message in bad_call["conversation"] if message.get("tool_calls"))
    arguments = json.loads(call["tool_calls"][0]["function"]["arguments"])
    arguments["location"] = "别处"
    call["tool_calls"][0]["function"]["arguments"] = json.dumps(arguments, ensure_ascii=False)
    with pytest.raises(ValueError, match="arguments differs"):
        C.validate_joined_row(bad_call, materialized)

    bad_result = copy.deepcopy(row)
    tool_message = next(message for message in bad_result["conversation"] if message["role"] == "tool")
    result = json.loads(tool_message["content"])
    result["status"] = "empty"
    tool_message["content"] = json.dumps(result, ensure_ascii=False)
    with pytest.raises(ValueError, match="result differs"):
        C.validate_joined_row(bad_result, materialized)


def test_world_keys_are_unique_normalized_and_content_hash_is_valid(compiled):
    _, world = compiled
    keys = [C.world_key(entry["tool"], entry["arguments"]) for entry in world["entries"]]
    assert len(keys) == len(set(keys))
    assert keys == sorted(keys)
    for key, entry in zip(keys, world["entries"]):
        assert entry["key_sha256"] == hashlib.sha256(key.encode("utf-8")).hexdigest()
        assert entry["execution_id"] == f"sha256:{entry['key_sha256']}"
        assert entry["result_sha256"] == C.sha256_json(entry["result"])

    payload = {
        key: value
        for key, value in world.items()
        if key not in {"content_sha256", "world_id"}
    }
    assert world["content_sha256"] == C.sha256_json(payload)
    assert world["world_id"] == f"{C.WORLD_VERSION}:{world['content_sha256']}"


def test_known_hotel_review_conflict_is_order_independent_and_provenanced(compiled):
    episodes, world = compiled
    arguments = {"hotel_name": "哈尔滨伯爵摩赫酒店", "location": "双城"}
    entry = _world_entry(world, "get_hotel_reviews", arguments)

    assert world["conflict_count"] == len(world["conflicts"]) == 1
    conflict = world["conflicts"][0]
    assert conflict["tool"] == "get_hotel_reviews"
    assert conflict["arguments"] == arguments
    assert conflict["resolution"]["rule"] == "minimum_canonical_result_sha256"
    candidates = conflict["candidate_results"]
    assert [candidate["result_sha256"] for candidate in candidates] == sorted(
        candidate["result_sha256"] for candidate in candidates
    )
    assert {candidate["result"]["status"] for candidate in candidates} == {"ok", "empty"}
    assert entry["result"]["status"] == "ok"
    assert conflict["resolution"]["selected_result_sha256"] == entry["result_sha256"]
    occurrences = {
        occurrence["source_idx"]
        for candidate in candidates
        for occurrence in candidate["occurrences"]
    }
    assert occurrences == {139, 214, 994}

    by_idx = _episode_map(episodes)
    for idx in occurrences:
        review_node = next(
            node
            for node in by_idx[idx]["state_graph"]
            if node["expected_action"]["type"] == "ToolCalls"
            and node["expected_action"]["calls"][0]["name"] == "get_hotel_reviews"
        )
        assert review_node["expected_result_status"] == "ok"
        assert review_node["result_branches"]["ok"].startswith("final_")
    assert by_idx[214]["metadata"]["expect"] == "empty"  # source label is unchanged
    assert by_idx[214]["person_state"]["goal"]["world_expected_outcome"] == "ok"


def test_world_build_does_not_depend_on_input_order():
    rows, materialized = C.load_sources(MERGED, MATERIALIZED)
    assert len(rows) == 909
    reversed_materialized = dict(reversed(list(materialized.items())))
    forward_world, _ = C.build_world(materialized)
    reverse_world, _ = C.build_world(reversed_materialized)
    assert C.canonical_json(forward_world) == C.canonical_json(reverse_world)


def test_episode_53_is_one_complete_causal_trajectory(compiled):
    episodes, _ = compiled
    episode = _episode_map(episodes)[53]
    assert episode["initial_state_id"] == "ask_01_destination"
    assert episode["person_state"]["initially_hidden"] == ["destination", "travel_date"]
    assert episode["person_state"]["private_slots"]["destination"] == "信阳"
    assert episode["person_state"]["private_slots"]["travel_date"] == "2026-08-25"

    nodes = episode["state_graph"]
    assert [node["expected_action"]["type"] for node in nodes] == [
        "Ask", "Ask", "ToolCalls", "ToolCalls", "Final"
    ]
    assert nodes[0]["user_response"] == "目的地是信阳"
    assert nodes[1]["user_response"] == "就定8月25号那天，一天就行"
    assert nodes[2]["result_branches"] == {
        "ok": "tool_04_weather",
        "empty": "final_05",
        "error": "final_05",
    }

    milestones = episode["reward_milestones"]
    assert milestones == [
        "ask_01_destination",
        "ask_02_travel_date",
        "call_03_guide_01",
        "round_03_guide",
        "branch_03_guide_ok",
        "call_04_weather_01",
        "round_04_weather",
        "branch_04_weather_ok",
        "valid_final_05",
    ]


def test_w5_requires_refusal_not_ordinary_final(compiled):
    episodes, _ = compiled
    w5 = next(episode for episode in episodes if episode["metadata"]["workflow"] == 5)
    assert w5["state_graph"] == [
        {
            "state_id": "refusal_01",
            "expected_action": {"type": "Refusal"},
            "next_state": "done",
            "milestone_ids": {"completion": "valid_refusal_01"},
        }
    ]
    assert w5["reward_milestones"] == ["valid_refusal_01"]


def test_w3_pivot_response_reveals_city_not_missing_hotel(compiled):
    episodes, _ = compiled
    pivot = next(episode for episode in episodes if episode["metadata"]["route"] == "W3-G")
    ask = pivot["state_graph"][0]
    assert ask["expected_action"] == {"type": "Ask", "slot": "hotel_name"}
    assert ask["revealed_slots"][0] == "hotel_location"
    assert "hotel_name" not in ask["revealed_slots"]
    assert "abandon_unresolved_hotel_reference" in ask["response_constraints"]


def test_artifacts_are_byte_deterministic(compiled, tmp_path):
    episodes, world = compiled
    first = C.write_outputs(episodes, world, tmp_path / "first")
    second = C.write_outputs(episodes, world, tmp_path / "second")
    assert first["world_policy_episodes_sha256"] == second["world_policy_episodes_sha256"]
    assert first["synthetic_china_world_v1_sha256"] == second["synthetic_china_world_v1_sha256"]
    assert (tmp_path / "first/world_policy_episodes.jsonl").read_bytes() == (
        tmp_path / "second/world_policy_episodes.jsonl"
    ).read_bytes()
    assert (tmp_path / "first/synthetic_china_world_v1.json").read_bytes() == (
        tmp_path / "second/synthetic_china_world_v1.json"
    ).read_bytes()
