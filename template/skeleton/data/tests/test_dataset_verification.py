import importlib
import json
from pathlib import Path
import sys

import pytest


APPLICATION_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APPLICATION_ROOT / "data" / "scripts"))
dataset_module = importlib.import_module("_dataset")


def _route(route_id: str):
    return {
        "workflow_id": "workflow.test",
        "initial_slots": [],
        "transition_ids": [f"{route_id}.done"],
        "actions": [{"state_id": "respond", "kind": "respond"}],
        "terminal_state": "done",
        "goal_reached": True,
        "evidence": [],
        "slots": {},
    }


def _record(episode_id: str, route_id: str):
    route = _route(route_id)
    return {
        "episode_id": episode_id,
        "workflow_id": "workflow.test",
        "route_id": route_id,
        "group_id": episode_id,
        "initial_context": {},
        "private_state": {},
        "trajectory": {
            key: route[key]
            for key in (
                "initial_slots",
                "transition_ids",
                "actions",
                "terminal_state",
                "goal_reached",
            )
        },
        "tool_evidence": [],
        "conversation": [
            {"role": "system", "content": "Follow the test workflow."},
            {"role": "user", "content": "request"},
            {"role": "assistant", "content": "response"},
        ],
        "contract_version": "1.0.0",
        "generator_version": "1.0.0",
    }


def test_record_requires_exact_route_trajectory() -> None:
    record = _record("episode-1", "route.one")
    record["trajectory"]["transition_ids"] = ["different"]
    with pytest.raises(dataset_module.ContractError, match="declared route"):
        dataset_module._validate_record(
            record,
            _route("route.one"),
            {},
            {"version": "1.0.0"},
            "1.0.0",
            {},
            lambda record, route: True,
            "record",
        )


def test_record_replays_the_actual_training_conversation() -> None:
    record = _record("episode-1", "route.one")
    record["conversation"][-1] = {"role": "user", "content": "unrelated"}
    with pytest.raises(dataset_module.ContractError, match="assistant message"):
        dataset_module._validate_record(
            record,
            _route("route.one"),
            {},
            {"version": "1.0.0"},
            "1.0.0",
            {},
            lambda record, route: True,
            "record",
        )


def test_record_requires_application_semantic_acceptance() -> None:
    record = _record("episode-1", "route.one")
    with pytest.raises(dataset_module.ContractError, match="application conversation"):
        dataset_module._validate_record(
            record,
            _route("route.one"),
            {},
            {"version": "1.0.0"},
            "1.0.0",
            {},
            lambda record, route: False,
            "record",
        )


def test_conversation_tool_call_must_match_frozen_evidence() -> None:
    route = {
        "actions": [
            {"state_id": "lookup", "kind": "tool", "tools": ["catalog.lookup"]},
            {"state_id": "respond", "kind": "respond"},
        ]
    }
    evidence = [
        {
            "state_id": "lookup",
            "tool_id": "catalog.lookup",
            "arguments": {"query": "alpha"},
            "normalized_arguments": {"query": "alpha"},
            "result": {"status": "ok", "data": {"item": "alpha"}},
        }
    ]
    conversation = [
        {"role": "system", "content": "Use the catalog."},
        {"role": "user", "content": "alpha"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "catalog.other",
                        "arguments": '{"query": "alpha"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"status": "ok", "data": {"item": "alpha"}}',
            "tool_call_id": "call-1",
        },
        {"role": "assistant", "content": "Alpha is available."},
    ]
    with pytest.raises(dataset_module.ContractError, match="tool name"):
        dataset_module._validate_conversation_replay(
            conversation, route, evidence, "record"
        )


@pytest.mark.parametrize(
    "status,result",
    [
        ("ok", {"status": "ok", "data": {"item": "alpha"}, "error": None}),
        ("empty", {"status": "empty", "data": None, "error": None}),
        (
            "error",
            {
                "status": "error",
                "data": None,
                "error": {
                    "code": "failed",
                    "message": "failed",
                    "retryable": False,
                },
            },
        ),
    ],
)
def test_tool_results_use_mutually_exclusive_envelopes(status, result) -> None:
    tool = {
        "outcomes": {
            "ok": {"data_schema": {"type": "object"}},
            "empty": {"reasons": ["missing"]},
            "error": {
                "codes": [
                    {
                        "code": "failed",
                        "description": "Failure.",
                        "retryable": False,
                    }
                ]
            },
        }
    }
    with pytest.raises(dataset_module.ContractError, match="exact .* envelope"):
        dataset_module._validate_tool_result(result, tool, status, "result")


def test_normalized_arguments_are_recomputed() -> None:
    assert dataset_module._normalize_value(
        "  Mixed Case  ", ["trim", "lowercase"], "query"
    ) == "mixed case"
    assert dataset_module._normalize_value(
        ["b", "a", "b"], ["sort_unique"], "values"
    ) == ["a", "b"]


def test_determinism_counts_are_derived_from_exact_replays(tmp_path) -> None:
    arguments = {"query": "alpha"}
    result = {"status": "ok", "data": {"item": "alpha"}}
    key = json.dumps(
        ["catalog.lookup", arguments],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    evidence = tmp_path / "determinism.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": "miniagent.determinism-evidence.v1",
                "queries": [
                    {
                        "tool_id": "catalog.lookup",
                        "normalized_arguments": arguments,
                        "results": [result, result],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    simulator = tmp_path / "simulator.py"
    simulator.write_text(
        "SIMULATOR_READY = True\n\n"
        "def execute(tool_id, normalized_arguments):\n"
        "    return {'status': 'ok', 'data': {'item': normalized_arguments['query']}}\n",
        encoding="utf-8",
    )
    manifest = {
        "world": {
            "simulator": {"path": str(simulator)},
            "determinism_check": {
                "normalized_queries": 2,
                "replayed_queries": 2,
                "conflicts": 0,
                "evidence": {"path": str(evidence)},
            }
        }
    }
    with pytest.raises(dataset_module.ContractError, match="Derive determinism counts"):
        dataset_module._validate_determinism_evidence(
            manifest,
            lambda record, label: Path(record["path"]),
            {
                key: json.dumps(
                    result,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            },
        )


@pytest.mark.parametrize(
    ("implementation", "error"),
    [
        (
            "def execute(tool_id, normalized_arguments):\n"
            "    return {'status': 'ok', 'data': {'item': 'different'}}\n",
            "differ from determinism evidence",
        ),
        (
            "counter = 0\n\n"
            "def execute(tool_id, normalized_arguments):\n"
            "    global counter\n"
            "    counter += 1\n"
            "    item = 'alpha' if counter == 1 else 'different'\n"
            "    return {'status': 'ok', 'data': {'item': item}}\n",
            "conflicting synthetic-world simulator replays",
        ),
    ],
)
def test_determinism_executes_the_hash_bound_world_simulator(
    tmp_path, implementation, error
) -> None:
    arguments = {"query": "alpha"}
    result = {"status": "ok", "data": {"item": "alpha"}}
    key = json.dumps(
        ["catalog.lookup", arguments],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    simulator = tmp_path / "simulator.py"
    simulator.write_text(
        "SIMULATOR_READY = True\n\n" + implementation,
        encoding="utf-8",
    )
    evidence = tmp_path / "determinism.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": "miniagent.determinism-evidence.v1",
                "queries": [
                    {
                        "tool_id": "catalog.lookup",
                        "normalized_arguments": arguments,
                        "results": [result, result],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "world": {
            "simulator": {"path": str(simulator)},
            "determinism_check": {
                "normalized_queries": 1,
                "replayed_queries": 2,
                "conflicts": 0,
                "evidence": {"path": str(evidence)},
            },
        }
    }
    with pytest.raises(dataset_module.ContractError, match=error):
        dataset_module._validate_determinism_evidence(
            manifest,
            lambda record, label: Path(record["path"]),
            {
                key: json.dumps(
                    result,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            },
        )


def test_tool_arguments_are_derived_from_record_slots() -> None:
    route = {
        "workflow_id": "workflow.test",
        "initial_slots": [],
        "transition_ids": ["start.lookup", "lookup.ok", "respond.done"],
        "actions": [
            {"state_id": "lookup", "kind": "tool", "tools": ["catalog.lookup"]},
            {"state_id": "respond", "kind": "respond"},
        ],
        "terminal_state": "done",
        "goal_reached": True,
        "evidence": [
            {
                "state_id": "lookup",
                "tool_id": "catalog.lookup",
                "status": "ok",
                "argument_bindings": {"query": "slot.request"},
            }
        ],
        "required_initial_context": ["request"],
        "slots": {
            "request": {
                "name": "request",
                "source": "context",
                "schema": {"type": "string"},
            }
        },
    }
    result = {"status": "ok", "data": {"item": "alpha"}}
    record = {
        "episode_id": "episode-1",
        "workflow_id": "workflow.test",
        "route_id": "route.test",
        "initial_context": {"request": "beta"},
        "private_state": {},
        "trajectory": {
            key: route[key]
            for key in (
                "initial_slots",
                "transition_ids",
                "actions",
                "terminal_state",
                "goal_reached",
            )
        },
        "tool_evidence": [
            {
                "state_id": "lookup",
                "tool_id": "catalog.lookup",
                "arguments": {"query": "alpha"},
                "normalized_arguments": {"query": "alpha"},
                "result": result,
            }
        ],
        "conversation": [],
        "contract_version": "1.0.0",
        "generator_version": "1.0.0",
    }
    tools = {
        "catalog.lookup": {
            "arguments": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            "normalization": {"query": ["trim", "lowercase"]},
            "outcomes": {
                "ok": {
                    "data_schema": {
                        "type": "object",
                        "properties": {"item": {"type": "string"}},
                        "required": ["item"],
                        "additionalProperties": False,
                    }
                },
                "empty": {"reasons": []},
                "error": {"codes": []},
            },
        }
    }
    with pytest.raises(dataset_module.ContractError, match="workflow bindings"):
        dataset_module._validate_record(
            record,
            route,
            tools,
            {"version": "1.0.0"},
            "1.0.0",
            {},
            lambda record, route: True,
            "record",
        )


def test_evaluation_partition_requires_every_route(tmp_path, monkeypatch) -> None:
    schema = tmp_path / "record.schema.json"
    schema.write_text(
        json.dumps({"type": "object", "additionalProperties": True}) + "\n",
        encoding="utf-8",
    )
    train = tmp_path / "train.json"
    train.write_text(
        json.dumps(
            [_record("train-1", "route.one"), _record("train-2", "route.two")]
        ),
        encoding="utf-8",
    )
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps([_record("eval-1", "route.one")]),
        encoding="utf-8",
    )
    conversation_validator = tmp_path / "conversation_validator.py"
    conversation_validator.write_text(
        "VALIDATOR_READY = True\n\ndef validate(record, route):\n    return True\n",
        encoding="utf-8",
    )
    manifest = {
        "generator": {
            "version": "1.0.0",
            "conversation_validator": {"path": str(conversation_validator)},
        },
        "record_schema": {"path": str(schema)},
        "splits": [
            {
                "name": "train",
                "purpose": "train",
                "path": str(train),
                "format": "json",
                "records": 2,
                "groups": 2,
            },
            {
                "name": "evaluation",
                "purpose": "evaluation",
                "path": str(evaluation),
                "format": "json",
                "records": 1,
                "groups": 1,
            },
        ],
        "grouping": {"keys": ["group_id"]},
    }
    routes = {"route.one": _route("route.one"), "route.two": _route("route.two")}
    monkeypatch.setattr(dataset_module, "_route_catalog", lambda contract: routes)
    monkeypatch.setattr(dataset_module, "_tool_catalog", lambda contract: {})

    def resolver(record, label):
        return Path(record["path"])

    with pytest.raises(dataset_module.ContractError, match="evaluation partition"):
        dataset_module.validate_dataset_content(
            manifest,
            {"version": "1.0.0"},
            resolver,
        )
