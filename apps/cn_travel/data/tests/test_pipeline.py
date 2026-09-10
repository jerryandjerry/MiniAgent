#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integrity checks for the sealed CN Travel Stage 1–7 data artifacts."""
from __future__ import annotations

import collections
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

from project_paths import DATA_ROOT


N_EPISODES = 1_010
TRAIN_EPISODES = 909
EVAL_EPISODES = 101
MULTITURN_SAMPLES = 3_632
WORLD_ENTRIES = 925
ROUTES = {
    "W1-A", "W1-B", "W1-C", "W1-D", "W1-E", "W1-F", "W1-G", "W1-H",
    "W2-A", "W2-B", "W2-C", "W2-D",
    "W3-A", "W3-B", "W3-C", "W3-D", "W3-E", "W3-F", "W3-G", "W3-H",
    "W3-I", "W3-J", "W4-A", "W5-A",
}
TOOL_NAMES = {
    "search_travel_guide",
    "get_weather_info",
    "query_route",
    "recommend_hotels",
    "get_hotel_reviews",
}
ALL_ROUTE_COUNTS = {
    "W1-A": 320, "W1-B": 80, "W1-C": 14, "W1-D": 4,
    "W1-E": 13, "W1-F": 3, "W1-G": 13, "W1-H": 3,
    "W2-A": 20, "W2-B": 80, "W2-C": 4, "W2-D": 16,
    "W3-A": 14, "W3-B": 80, "W3-C": 13, "W3-D": 80,
    "W3-E": 13, "W3-F": 3, "W3-G": 16, "W3-H": 3,
    "W3-I": 16, "W3-J": 2, "W4-A": 100, "W5-A": 100,
}
LEADERBOARD_ROUTE_COUNTS = {
    "W1-A": 32, "W1-B": 6, "W1-C": 1, "W1-D": 2,
    "W1-E": 1, "W1-F": 1, "W1-G": 1, "W1-H": 1,
    "W2-A": 1, "W2-B": 8, "W2-C": 2, "W2-D": 1,
    "W3-A": 1, "W3-B": 8, "W3-C": 1, "W3-D": 8,
    "W3-E": 1, "W3-F": 1, "W3-G": 1, "W3-H": 1,
    "W3-I": 1, "W3-J": 1, "W4-A": 10, "W5-A": 10,
}


def _numbered_files(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.glob("*.json") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot_sha256(path: Path) -> str:
    return hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest()


def _aggregate_digest(paths: list[Path]) -> str:
    aggregate = hashlib.sha256()
    for path in paths:
        aggregate.update(
            path.name.encode() + b"\x00" + _sha256(path).encode() + b"\n"
        )
    return aggregate.hexdigest()


def _conversation_trace(conversation: list[dict]) -> list[dict]:
    results = {
        message["tool_call_id"]: json.loads(message["content"])
        for message in conversation
        if message.get("role") == "tool"
    }
    return [
        {
            "name": call["function"]["name"],
            "arguments": json.loads(call["function"]["arguments"]),
            "result": results[call["id"]],
        }
        for message in conversation
        for call in (message.get("tool_calls") or [])
    ]


def _expand_one(row: dict) -> list[dict]:
    conversation = row["conversation"]
    metadata = row["metadata"]
    if not any(message.get("role") == "tool" for message in conversation):
        return [{"conversation": conversation, "metadata": metadata}]

    expanded = []
    assistant_positions = [
        index
        for index, message in enumerate(conversation)
        if message.get("role") == "assistant"
    ]
    for position, end in enumerate(assistant_positions):
        sample = {"conversation": conversation[:end + 1], "metadata": metadata}
        expanded.append(sample)
        if position == len(assistant_positions) - 1:
            expanded.append(copy.deepcopy(sample))
            expanded.append(copy.deepcopy(sample))
    return expanded


# ---------------------------------------------------------------- Stage 1
def test_storyboard_has_the_frozen_population(storyboard):
    assert len(storyboard) == N_EPISODES
    assert collections.Counter(row["workflow"] for row in storyboard) == {
        1: 450,
        2: 120,
        3: 240,
        4: 100,
        5: 100,
    }
    assert collections.Counter(row["route"] for row in storyboard) == ALL_ROUTE_COUNTS
    assert all(
        {"workflow", "route", "edges", "turns", "expect", "slots", "omit"}
        <= row.keys()
        for row in storyboard
    )


# ------------------------------------------------------------ Stages 2-3
def test_materialized_and_conversation_stages_cover_every_episode(storyboard):
    names = [f"{index:04d}.json" for index in range(N_EPISODES)]
    materialized = _numbered_files(DATA_ROOT / "2_materialized")
    conversations = _numbered_files(DATA_ROOT / "3_conversations")

    assert [path.name for path in materialized] == names
    assert [path.name for path in conversations] == names
    for index, (materialized_path, conversation_path) in enumerate(
        zip(materialized, conversations, strict=True)
    ):
        expected_metadata = {"idx": index, **storyboard[index]}
        assert _load(materialized_path)["metadata"] == expected_metadata
        assert _load(conversation_path)["metadata"] == expected_metadata


def test_stage_2_and_3_manifests_seal_every_record():
    stage2 = DATA_ROOT / "2_materialized"
    stage3 = DATA_ROOT / "3_conversations"
    manifest2 = _load(stage2 / "manifest.json")
    manifest3 = _load(stage3 / "manifest.json")

    for directory, manifest in ((stage2, manifest2), (stage3, manifest3)):
        files = _numbered_files(directory)
        assert manifest["state"] == "complete"
        assert len(files) == N_EPISODES
        assert set(manifest["records_sha256"]) == {path.stem for path in files}
        assert all(
            manifest["records_sha256"][path.stem] == _sha256(path)
            for path in files
        )
        assert manifest["aggregate_selected_digest"] == _aggregate_digest(files)

    assert manifest2["required"] == manifest2["selected"] == N_EPISODES
    assert manifest2["identity_one_to_one"] is True
    assert manifest2["unique_candidate_ids"] is True
    assert manifest2["selected_deferred"] == manifest2["selected_failed"] == 0
    assert manifest2["sources"]["storyboard_sha256"] == _sha256(
        DATA_ROOT / "1_storyboard.json"
    )
    frozen_sources = DATA_ROOT / "reproducibility" / "pipeline_sources"
    frozen_materializers = {
        _snapshot_sha256(path)
        for path in frozen_sources.glob("2_materialize.*.py.gz")
    }
    assert frozen_materializers == set(manifest2["sources"]["materializer_sha256"])
    assert _sha256(
        frozen_sources
        / "README_CN_TRAVEL.69d840fe499ca9b0829700abc969f5a3c65deea045998723fcd58795555f5ca8.md"
    ) == manifest2["sources"]["routes_doc_sha256"]

    stage2_snapshots = {
        "get_guide.py": frozen_sources
        / "get_guide.820f65e5e1a45f47f77b563e9240b20cfb51561cdaf0124ba3e4b7e35ae21dd8.py.gz",
        "get_hotel.py": frozen_sources
        / "get_hotel.b4d2ce8ecc624080477b77413073038942700c181d2ac5c2391f4f0527255982.py.gz",
        "get_route.py": frozen_sources
        / "get_route.4f193c268f35efe7b0a250a1f700333de9141b0de9f4a97ee8a048d56ad04dd2.py.gz",
    }
    assert {
        name: _snapshot_sha256(path) for name, path in stage2_snapshots.items()
    } == {
        name: manifest2["sources"]["tools_sha256"][name]
        for name in stage2_snapshots
    }
    backend_snapshots = {
        "hotel_price.py": frozen_sources
        / "hotel_price.ad23cc3b9047fda8f13adbb151ab00f40d254088f2860d756dbc7418d6de954f.py.gz",
        "llm_tool.py": frozen_sources
        / "llm_tool.f02d789edbab38606ce8e046336ad71528a7af6a93369c9e4920e5b778e3f80b.py.gz",
    }
    assert {
        name: _snapshot_sha256(path) for name, path in backend_snapshots.items()
    } == {
        name: manifest2["sources"]["backends_sha256"][name]
        for name in backend_snapshots
    }

    assert manifest3["required"] == manifest3["written"] == N_EPISODES
    assert manifest3["identity_one_to_one"] is True
    assert manifest3["selected_failures"] == 0
    assert manifest3["step2_fingerprint"] == manifest2["fingerprint"]
    assert manifest3["step2_manifest_sha256"] == _sha256(stage2 / "manifest.json")
    frozen_conversation = frozen_sources / "3_conversation.c39f3ba7ee7866c1.py.gz"
    assert manifest3["generator_sha256"] == _snapshot_sha256(frozen_conversation)


def test_conversations_are_valid_openai_tool_traces():
    for path in _numbered_files(DATA_ROOT / "3_conversations"):
        messages = _load(path)["conversation"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[-1]["role"] == "assistant"

        issued = set()
        answered = set()
        for message in messages:
            calls = message.get("tool_calls") or []
            if calls:
                assert message["role"] == "assistant"
                assert message.get("content") == ""
            for call in calls:
                assert call["type"] == "function"
                assert call["function"]["name"] in TOOL_NAMES
                json.loads(call["function"]["arguments"])
                issued.add(call["id"])
            if message["role"] == "tool":
                answered.add(message["tool_call_id"])
        assert answered == issued


def test_conversations_replay_materialized_traces_value_for_value():
    for materialized_path in _numbered_files(DATA_ROOT / "2_materialized"):
        materialized = _load(materialized_path)
        conversation = _load(
            DATA_ROOT / "3_conversations" / materialized_path.name
        )["conversation"]
        expected = [
            {
                "name": step["name"],
                "arguments": step["arguments"],
                "result": step["result"],
            }
            for step in materialized["visible_tool_trace"]
        ]
        assert _conversation_trace(conversation) == expected


# ---------------------------------------------------------------- Stage 4
def test_split_is_complete_disjoint_and_byte_identical_to_stage_3():
    stage3 = DATA_ROOT / "3_conversations"
    train = _numbered_files(DATA_ROOT / "4_split" / "train_validation")
    evaluation = _numbered_files(DATA_ROOT / "4_split" / "leaderboard_eval")
    train_ids = {int(path.stem) for path in train}
    evaluation_ids = {int(path.stem) for path in evaluation}

    assert len(train) == TRAIN_EPISODES
    assert len(evaluation) == EVAL_EPISODES
    assert train_ids.isdisjoint(evaluation_ids)
    assert train_ids | evaluation_ids == set(range(N_EPISODES))
    assert all(path.read_bytes() == (stage3 / path.name).read_bytes()
               for path in train + evaluation)


def test_split_preserves_routes_workflows_outcomes_and_turns():
    expected = {
        "train_validation": {
            "workflows": {1: 405, 2: 108, 3: 216, 4: 90, 5: 90},
            "outcomes": {"ok": 766, "empty": 109, "error": 34},
            "turns": {1: 814, 2: 81, 3: 14},
            "routes": {
                route: ALL_ROUTE_COUNTS[route] - LEADERBOARD_ROUTE_COUNTS[route]
                for route in ROUTES
            },
        },
        "leaderboard_eval": {
            "workflows": {1: 45, 2: 12, 3: 24, 4: 10, 5: 10},
            "outcomes": {"ok": 82, "empty": 14, "error": 5},
            "turns": {1: 86, 2: 13, 3: 2},
            "routes": LEADERBOARD_ROUTE_COUNTS,
        },
    }
    for side, targets in expected.items():
        metadata = [
            _load(path)["metadata"]
            for path in _numbered_files(DATA_ROOT / "4_split" / side)
        ]
        assert {row["route"] for row in metadata} == ROUTES
        assert collections.Counter(row["workflow"] for row in metadata) == \
            targets["workflows"]
        assert collections.Counter(row["expect"] for row in metadata) == \
            targets["outcomes"]
        assert collections.Counter(row["turns"] for row in metadata) == targets["turns"]
        assert collections.Counter(row["route"] for row in metadata) == targets["routes"]


def test_equal_tool_results_never_cross_the_split_boundary():
    evaluation_ids = {
        int(path.stem)
        for path in _numbered_files(DATA_ROOT / "4_split" / "leaderboard_eval")
    }
    result_sides: dict[str, set[str]] = collections.defaultdict(set)
    for path in _numbered_files(DATA_ROOT / "2_materialized"):
        side = "leaderboard_eval" if int(path.stem) in evaluation_ids \
            else "train_validation"
        for step in _load(path)["visible_tool_trace"]:
            digest = hashlib.sha256(json.dumps(
                step["result"], ensure_ascii=False, sort_keys=True
            ).encode()).hexdigest()
            result_sides[digest].add(side)

    assert result_sides
    assert all(len(sides) == 1 for sides in result_sides.values())


# ---------------------------------------------------------------- Stage 5
def test_merged_files_equal_the_sorted_stage_4_files(
    train_conversations, test_conversations
):
    assert len(train_conversations) == TRAIN_EPISODES
    assert len(test_conversations) == EVAL_EPISODES
    for side, merged in {
        "train_validation": train_conversations,
        "leaderboard_eval": test_conversations,
    }.items():
        source = [
            _load(path)
            for path in _numbered_files(DATA_ROOT / "4_split" / side)
        ]
        assert merged == source
        assert [row["metadata"]["idx"] for row in merged] == sorted(
            row["metadata"]["idx"] for row in merged
        )


# ---------------------------------------------------------------- Stage 6
def test_multiturn_stage_exactly_expands_the_training_partition(
    train_conversations, multiturn_v2
):
    expected = []
    for row in train_conversations:
        expected.extend(_expand_one(row))

    assert multiturn_v2 == expected
    assert len(multiturn_v2) == MULTITURN_SAMPLES
    assert collections.Counter(row["metadata"]["workflow"] for row in multiturn_v2) == {
        1: 2_002,
        2: 449,
        3: 1_001,
        4: 90,
        5: 90,
    }
    assert all(row["conversation"][-1]["role"] == "assistant" for row in multiturn_v2)


def test_multiturn_stage_preserves_the_leaderboard_bytes():
    merged = DATA_ROOT / "5_merged" / "leaderboard_eval.json"
    expanded = DATA_ROOT / "6_multiturn" / "leaderboard_eval.json"
    assert expanded.read_bytes() == merged.read_bytes()
    assert len(_load(expanded)) == EVAL_EPISODES


# ---------------------------------------------------------------- Stage 7
def test_world_policy_has_one_episode_per_training_conversation(
    train_conversations, world_policy_episodes, synthetic_china_world
):
    assert len(world_policy_episodes) == TRAIN_EPISODES
    assert len({episode["episode_id"] for episode in world_policy_episodes}) == \
        TRAIN_EPISODES

    training_metadata = {
        row["metadata"]["idx"]: row["metadata"] for row in train_conversations
    }
    assert {episode["source_idx"] for episode in world_policy_episodes} == \
        set(training_metadata)
    for episode in world_policy_episodes:
        assert episode["schema_version"] == "cn_travel.world_policy.v1"
        assert episode["metadata"] == training_metadata[episode["source_idx"]]
        assert episode["world_id"] == synthetic_china_world["world_id"]

    assert synthetic_china_world["schema_version"] == \
        "cn_travel.synthetic_china_world.v1"
    assert synthetic_china_world["world_version"] == "synthetic-china-world-v1"
    assert synthetic_china_world["entry_count"] == WORLD_ENTRIES
    assert len(synthetic_china_world["entries"]) == WORLD_ENTRIES
    assert synthetic_china_world["conflict_count"] == 1


def test_stage_7_compiler_reproduces_the_frozen_artifacts(
    world_policy_episodes, synthetic_china_world
):
    script = DATA_ROOT / "scripts" / "7_world_policy.py"
    spec = importlib.util.spec_from_file_location("cn_travel_stage7_pipeline_test", script)
    assert spec and spec.loader
    compiler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compiler)

    episodes, world = compiler.compile_dataset()
    assert episodes == world_policy_episodes
    assert world == synthetic_china_world
