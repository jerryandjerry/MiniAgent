import importlib
from pathlib import Path
import sys

import pytest


APPLICATION_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APPLICATION_ROOT / "data" / "scripts"))
contract_module = importlib.import_module("_contract")


def _tool(tool_id, compatible=(), prerequisites=()):
    return {
        "tool_id": tool_id,
        "arguments": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        "normalization": {"value": ["trim"]},
        "outcomes": {
            "ok": {
                "data_schema": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                }
            },
            "error": {"codes": [{"code": "failed"}]},
        },
        "execution": {"prerequisites": list(prerequisites)},
        "batching": {
            "allowed": bool(compatible),
            "compatible_tools": list(compatible),
        },
        "idempotency": {
            "strategy": "normalized_arguments",
            "key_arguments": ["value"],
        },
    }


def test_acceptance_status_is_derived_from_metrics() -> None:
    contract = {
        "acceptance": {
            "metrics": [
                {
                    "name": "goal_completion",
                    "operator": ">=",
                    "threshold": 0.9,
                    "unit": "ratio",
                }
            ]
        }
    }
    evaluation = {
        "summary": {"attempted": 10, "completed": 10, "errors": 0, "status": "passed"},
        "metrics": [
            {
                "name": "goal_completion",
                "value": 0.95,
                "unit": "ratio",
                "direction": "maximize",
                "threshold": 0.9,
                "passed": True,
            }
        ],
    }
    contract_module.validate_acceptance_metrics(contract, evaluation)
    evaluation["metrics"][0]["value"] = 0.5
    evaluation["metrics"][0]["passed"] = False
    with pytest.raises(contract_module.ContractError, match="summary status"):
        contract_module.validate_acceptance_metrics(contract, evaluation)


def test_discontinuous_route_is_rejected() -> None:
    workflow = {
        "workflow_id": "workflow.test",
        "slots": [],
        "initial_state": "start",
        "states": [
            {"id": "start", "kind": "start"},
            {"id": "middle", "kind": "decision", "decision": "continue"},
            {"id": "done", "kind": "terminal", "outcome": "completed", "goal_reached": True},
        ],
        "transitions": [
            {"id": "first", "from": "start", "to": "middle", "when": {"event": "always"}},
            {
                "id": "last",
                "from": "middle",
                "to": "done",
                "when": {"event": "condition", "expression": "ready"},
            },
        ],
        "routes": [
            {
                "id": "route.test",
                "initial_slots": [],
                "transition_ids": ["last", "first"],
                "terminal_state": "done",
            }
        ],
    }
    with pytest.raises(contract_module.ContractError, match="discontinuous"):
        contract_module._validate_workflow_semantics(workflow, {})


def test_tool_normalization_covers_every_argument() -> None:
    tool = _tool("lookup")
    tool["normalization"] = {}
    with pytest.raises(contract_module.ContractError, match="cover exactly"):
        contract_module._validate_tool_references([tool])


def test_tool_normalization_matches_argument_types() -> None:
    tool = _tool("lookup")
    tool["arguments"]["properties"]["value"] = {"type": "integer"}
    with pytest.raises(contract_module.ContractError, match="non-string schema"):
        contract_module._validate_tool_references([tool])


def test_tool_error_codes_are_unique() -> None:
    tool = _tool("lookup")
    tool["outcomes"]["error"]["codes"].append(
        {"code": "failed", "description": "A second meaning for the same code."}
    )
    with pytest.raises(contract_module.ContractError, match="unique error codes"):
        contract_module._validate_tool_references([tool])


def test_tool_result_bindings_name_declared_fields() -> None:
    tool = _tool("lookup")
    tool["outcomes"]["ok"] = {
        "data_schema": {
            "type": "object",
            "properties": {"known": {"type": "string"}},
        }
    }
    tools = contract_module._validate_tool_references([tool])
    with pytest.raises(contract_module.ContractError, match="unknown result field"):
        contract_module._binding_dependencies(
            {"value": "tool.lookup.unknown"},
            "workflow.test",
            "call",
            {"value"},
            tools,
        )


def test_binding_schemas_must_be_assignable_to_tool_arguments() -> None:
    assert contract_module._schemas_are_assignable(
        {"type": "integer"}, {"type": "number"}
    )
    assert not contract_module._schemas_are_assignable(
        {"type": "string"}, {"type": "integer"}
    )


def test_route_can_declare_initial_user_slots() -> None:
    tool = _tool("lookup", prerequisites=({"kind": "slot", "id": "value"},))
    tools = contract_module._validate_tool_references([tool])
    workflow = {
        "workflow_id": "workflow.initial-user-slot",
        "slots": [
            {
                "name": "value",
                "source": "user",
                "schema": {"type": "string"},
            }
        ],
        "initial_state": "start",
        "states": [
            {"id": "start", "kind": "start"},
            {"id": "has-value", "kind": "decision"},
            {"id": "collect", "kind": "collect", "slot": "value"},
            {
                "id": "lookup",
                "kind": "tool",
                "tool": "lookup",
                "argument_bindings": {"value": "slot.value"},
            },
            {"id": "respond-ok", "kind": "respond"},
            {"id": "respond-empty", "kind": "respond"},
            {"id": "respond-error", "kind": "respond"},
            {"id": "done-ok", "kind": "terminal"},
            {"id": "done-empty", "kind": "terminal"},
            {"id": "done-error", "kind": "terminal"},
        ],
        "transitions": [
            {
                "id": "start.decide",
                "from": "start",
                "to": "has-value",
                "when": {"event": "always"},
            },
            {
                "id": "decide.direct",
                "from": "has-value",
                "to": "lookup",
                "when": {"event": "condition", "expression": "value present"},
            },
            {
                "id": "decide.collect",
                "from": "has-value",
                "to": "collect",
                "when": {"event": "condition", "expression": "value absent"},
            },
            {
                "id": "collect.lookup",
                "from": "collect",
                "to": "lookup",
                "when": {"event": "user_response", "slots_present": ["value"]},
            },
            *[
                {
                    "id": f"lookup.{status}",
                    "from": "lookup",
                    "to": f"respond-{status}",
                    "when": {
                        "event": "tool_result",
                        "tool": "lookup",
                        "status": status,
                    },
                }
                for status in ("ok", "empty", "error")
            ],
            *[
                {
                    "id": f"respond-{status}.done",
                    "from": f"respond-{status}",
                    "to": f"done-{status}",
                    "when": {"event": "always"},
                }
                for status in ("ok", "empty", "error")
            ],
        ],
        "routes": [
            {
                "id": f"route.{path}.{status}",
                "initial_slots": ["value"] if path == "direct" else [],
                "transition_ids": [
                    "start.decide",
                    f"decide.{path}",
                    *([] if path == "direct" else ["collect.lookup"]),
                    f"lookup.{status}",
                    f"respond-{status}.done",
                ],
                "terminal_state": f"done-{status}",
            }
            for path in ("direct", "collect")
            for status in ("ok", "empty", "error")
        ],
    }
    contract_module._validate_workflow_semantics(workflow, tools)


def test_tool_batching_compatibility_is_symmetric() -> None:
    first = _tool("first", compatible=("second",))
    second = _tool("second")
    with pytest.raises(contract_module.ContractError, match="symmetrically"):
        contract_module._validate_tool_references([first, second])


def test_dependent_tools_cannot_share_a_batch() -> None:
    first = _tool("first", compatible=("second",))
    second = _tool(
        "second",
        compatible=("first",),
        prerequisites=({"kind": "tool_result", "id": "first"},),
    )
    tools = contract_module._validate_tool_references([first, second])
    workflow = {
        "workflow_id": "workflow.batch",
        "slots": [
            {
                "name": "value",
                "source": "context",
                "schema": {"type": "string"},
            }
        ],
        "initial_state": "start",
        "states": [
            {"id": "start", "kind": "start"},
            {
                "id": "batch",
                "kind": "tool_batch",
                "calls": [
                    {"tool": "first", "argument_bindings": {"value": "slot.value"}},
                    {
                        "tool": "second",
                        "argument_bindings": {"value": "tool.first.value"},
                    },
                ],
            },
            {"id": "done", "kind": "terminal"},
        ],
        "transitions": [],
        "routes": [],
    }
    with pytest.raises(contract_module.ContractError, match="batches dependent tools"):
        contract_module._validate_workflow_semantics(workflow, tools)
