#!/usr/bin/env python3
"""Prepare and inspect the successful path of a rendered MiniAgent application."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import textwrap
from typing import Any, Dict, Iterable


APPLICATION_ID = "recipe_smoke"
PACKAGE = "recipe_smoke"
DATASET_ID = "smoke-dataset"
RUN_ID = "smoke-run"
EVALUATION_ID = "smoke-evaluation"
RELEASE_ID = "smoke-release"
CREATED_AT = "2026-01-01T00:00:00Z"


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(application: Path, path: Path) -> Dict[str, Any]:
    return {
        "path": path.relative_to(application).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _application_contract() -> Dict[str, Any]:
    return {
        "schema_version": "miniagent.application.v1",
        "status": "approved",
        "application_id": APPLICATION_ID,
        "package": PACKAGE,
        "name": "Recipe Smoke",
        "version": "1.0.0",
        "business": {
            "objective": "Return one deterministic catalog result for a supported request.",
            "users": ["catalog user"],
            "supported_requests": [
                {
                    "id": "request.lookup",
                    "description": "Look up one item in the catalog.",
                }
            ],
            "rules": [
                {
                    "id": "rule.lookup",
                    "statement": "Use the catalog result before answering.",
                }
            ],
            "context_fields": [
                {
                    "name": "request",
                    "description": "The normalized catalog query.",
                    "required": True,
                    "schema": {"type": "string", "minLength": 1},
                }
            ],
            "systems": [
                {
                    "id": "catalog",
                    "purpose": "Provide deterministic item records.",
                    "access": "read",
                }
            ],
            "outcomes": {
                "completion": "Return the catalog result.",
                "empty": "Report that the catalog has no matching item.",
                "error": "Report that the catalog lookup failed.",
                "refusal": "Explain the supported catalog scope.",
            },
        },
        "interfaces": ["api", "cli"],
        "artifacts": {
            "workflows": ["data/specification/workflows"],
            "tools": ["data/specification/tools"],
            "dataset_manifests": "data/artifacts/manifests",
            "training_runs": "train/runs",
            "evaluation_results": "eval/results/runs",
            "releases": "eval/releases",
            "runtime_source": f"src/{PACKAGE}",
        },
        "acceptance": {
            "criteria": ["Complete every sealed catalog case."],
            "metrics": [
                {
                    "name": "goal_completion",
                    "operator": ">=",
                    "threshold": 1.0,
                    "unit": "ratio",
                }
            ],
        },
    }


def _tool_contract() -> Dict[str, Any]:
    return {
        "schema_version": "miniagent.tool.v1",
        "application_id": APPLICATION_ID,
        "tool_id": "catalog.lookup",
        "name": "Catalog lookup",
        "description": "Return one deterministic catalog record.",
        "side_effect": "read",
        "determinism": "deterministic",
        "arguments": {
            "type": "object",
            "properties": {"query": {"type": "string", "minLength": 1}},
            "required": ["query"],
            "additionalProperties": False,
        },
        "normalization": {"query": ["trim", "lowercase"]},
        "outcomes": {
            "ok": {
                "description": "A catalog item matched.",
                "data_schema": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string"},
                        "available": {"type": "boolean"},
                    },
                    "required": ["item", "available"],
                    "additionalProperties": False,
                },
            },
            "empty": {
                "description": "No catalog item matched.",
                "reasons": ["not-found"],
            },
            "error": {
                "description": "The catalog lookup failed.",
                "codes": [
                    {
                        "code": "catalog-unavailable",
                        "description": "The catalog is unavailable.",
                        "retryable": True,
                    }
                ],
            },
        },
        "execution": {
            "timeout_seconds": 2,
            "max_attempts": 1,
            "prerequisites": [{"kind": "slot", "id": "request"}],
        },
        "batching": {"allowed": False, "compatible_tools": []},
        "implementations": {
            "synthetic": {"entrypoint": f"{PACKAGE}.tool.smoke:lookup"},
            "production": {"entrypoint": f"{PACKAGE}.tool.smoke:lookup"},
        },
        "authorization": {
            "required": False,
            "policy": "Read-only catalog access is available to every application user.",
        },
        "idempotency": {
            "strategy": "normalized_arguments",
            "key_arguments": ["query"],
        },
    }


def _workflow_contract() -> Dict[str, Any]:
    return {
        "schema_version": "miniagent.workflow.v1",
        "application_id": APPLICATION_ID,
        "workflow_id": "workflow.lookup",
        "name": "Catalog lookup",
        "goal": "Answer a supported request from deterministic catalog evidence.",
        "supported_request_ids": ["request.lookup"],
        "rule_ids": ["rule.lookup"],
        "slots": [
            {
                "name": "request",
                "description": "The catalog query supplied in request context.",
                "source": "context",
                "schema": {"type": "string", "minLength": 1},
            }
        ],
        "route_policy": {"max_transition_visits": 1},
        "initial_state": "start",
        "states": [
            {"id": "start", "kind": "start"},
            {
                "id": "lookup",
                "kind": "tool",
                "tool": "catalog.lookup",
                "argument_bindings": {"query": "slot.request"},
            },
            {
                "id": "respond.ok",
                "kind": "respond",
                "content_requirements": ["Use the matched catalog item."],
            },
            {
                "id": "respond.empty",
                "kind": "respond",
                "content_requirements": ["State that no item matched."],
            },
            {
                "id": "respond.error",
                "kind": "respond",
                "content_requirements": ["State that lookup execution failed."],
            },
            {
                "id": "done.ok",
                "kind": "terminal",
                "outcome": "completed",
                "goal_reached": True,
            },
            {
                "id": "done.empty",
                "kind": "terminal",
                "outcome": "completed",
                "goal_reached": True,
            },
            {
                "id": "done.error",
                "kind": "terminal",
                "outcome": "failed",
                "goal_reached": False,
            },
        ],
        "transitions": [
            {
                "id": "start.lookup",
                "from": "start",
                "to": "lookup",
                "when": {"event": "always"},
            },
            {
                "id": "lookup.ok",
                "from": "lookup",
                "to": "respond.ok",
                "when": {
                    "event": "tool_result",
                    "tool": "catalog.lookup",
                    "status": "ok",
                },
            },
            {
                "id": "lookup.empty",
                "from": "lookup",
                "to": "respond.empty",
                "when": {
                    "event": "tool_result",
                    "tool": "catalog.lookup",
                    "status": "empty",
                },
            },
            {
                "id": "lookup.error",
                "from": "lookup",
                "to": "respond.error",
                "when": {
                    "event": "tool_result",
                    "tool": "catalog.lookup",
                    "status": "error",
                },
            },
            {
                "id": "respond.ok.done",
                "from": "respond.ok",
                "to": "done.ok",
                "when": {"event": "always"},
            },
            {
                "id": "respond.empty.done",
                "from": "respond.empty",
                "to": "done.empty",
                "when": {"event": "always"},
            },
            {
                "id": "respond.error.done",
                "from": "respond.error",
                "to": "done.error",
                "when": {"event": "always"},
            },
        ],
        "routes": [
            {
                "id": "route.lookup.ok",
                "description": "The catalog returns a matching item.",
                "initial_slots": [],
                "transition_ids": [
                    "start.lookup",
                    "lookup.ok",
                    "respond.ok.done",
                ],
                "terminal_state": "done.ok",
            },
            {
                "id": "route.lookup.empty",
                "description": "The catalog has no matching item.",
                "initial_slots": [],
                "transition_ids": [
                    "start.lookup",
                    "lookup.empty",
                    "respond.empty.done",
                ],
                "terminal_state": "done.empty",
            },
            {
                "id": "route.lookup.error",
                "description": "The catalog lookup fails.",
                "initial_slots": [],
                "transition_ids": [
                    "start.lookup",
                    "lookup.error",
                    "respond.error.done",
                ],
                "terminal_state": "done.error",
            },
        ],
    }


PIPELINE_SOURCE = r'''
"""Deterministic dataset worker for the recipe smoke fixture."""

import hashlib
import json
from pathlib import Path


PIPELINE_READY = True
APPLICATION_ROOT = Path(__file__).resolve().parents[1]
CREATED_AT = "2026-01-01T00:00:00Z"


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _record(staged, published, include_bytes=True):
    content = staged.read_bytes()
    result = {
        "path": published.relative_to(APPLICATION_ROOT).as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    if include_bytes:
        result["bytes"] = len(content)
    return result


def build(contract, staging_directory, published_directory):
    world = staging_directory / "world.json"
    simulator = staging_directory / "simulator.py"
    determinism = staging_directory / "checks" / "determinism.json"
    record_schema = staging_directory / "record.schema.json"
    train = staging_directory / "train.jsonl"
    evaluation = staging_directory / "evaluation.jsonl"
    leakage = staging_directory / "checks" / "leakage.json"

    _write(world, json.dumps({"world_id": "catalog-world", "items": {"alpha": True}}, sort_keys=True) + "\n")
    _write(
        simulator,
        "SIMULATOR_READY = True\n\n"
        "def execute(tool_id, normalized_arguments):\n"
        "    if tool_id != 'catalog.lookup':\n"
        "        raise ValueError(f'Unsupported tool: {tool_id}')\n"
        "    query = normalized_arguments['query']\n"
        "    if query == 'alpha':\n"
        "        return {'status': 'ok', 'data': {'item': 'alpha', 'available': True}}\n"
        "    if query == 'missing':\n"
        "        return {'status': 'empty', 'data': None, 'reason': 'not-found'}\n"
        "    return {\n"
        "        'status': 'error',\n"
        "        'error': {\n"
        "            'code': 'catalog-unavailable',\n"
        "            'message': 'The synthetic catalog is unavailable.',\n"
        "            'retryable': True,\n"
        "        },\n"
        "    }\n",
    )
    _write(record_schema, json.dumps({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": [
            "episode_id", "workflow_id", "route_id", "group_id",
            "initial_context", "private_state", "trajectory", "tool_evidence",
            "conversation", "contract_version", "generator_version"
        ],
        "properties": {
            "episode_id": {"type": "string"},
            "workflow_id": {"type": "string"},
            "route_id": {"type": "string"},
            "group_id": {"type": "string"},
            "initial_context": {"type": "object"},
            "private_state": {"type": "object"},
            "trajectory": {"type": "object"},
            "tool_evidence": {"type": "array"},
            "conversation": {"type": "array"},
            "contract_version": {"type": "string"},
            "generator_version": {"type": "string"},
        },
        "additionalProperties": False,
    }, sort_keys=True) + "\n")
    routes = ["route.lookup.ok", "route.lookup.empty", "route.lookup.error"]

    def dataset_row(partition, index, route):
        status = route.rsplit(".", 1)[1]
        query = {"ok": "alpha", "empty": "missing", "error": "error"}[status]
        if status == "ok":
            result = {"status": "ok", "data": {"item": "alpha", "available": True}}
        elif status == "empty":
            result = {"status": "empty", "data": None, "reason": "not-found"}
        else:
            result = {
                "status": "error",
                "error": {
                    "code": "catalog-unavailable",
                    "message": "The synthetic catalog is unavailable.",
                    "retryable": True,
                },
            }
        response_state = f"respond.{status}"
        terminal_state = f"done.{status}"
        transitions = ["start.lookup", f"lookup.{status}", f"respond.{status}.done"]
        episode_id = f"{partition}-{index}"
        return {
            "episode_id": episode_id,
            "workflow_id": "workflow.lookup",
            "route_id": route,
            "group_id": f"{partition}-group-{index}",
            "initial_context": {"request": query},
            "private_state": {"episode_id": episode_id},
            "trajectory": {
                "initial_slots": [],
                "transition_ids": transitions,
                "actions": [
                    {"state_id": "lookup", "kind": "tool", "tools": ["catalog.lookup"]},
                    {
                        "state_id": response_state,
                        "kind": "respond",
                        "content_requirements": [{
                            "ok": "Use the matched catalog item.",
                            "empty": "State that no item matched.",
                            "error": "State that lookup execution failed.",
                        }[status]],
                    },
                ],
                "terminal_state": terminal_state,
                "goal_reached": status != "error",
            },
            "tool_evidence": [
                {
                    "state_id": "lookup",
                    "tool_id": "catalog.lookup",
                    "arguments": {"query": query},
                    "normalized_arguments": {"query": query},
                    "result": result,
                }
            ],
            "conversation": [
                {"role": "system", "content": "Use the catalog workflow."},
                {"role": "user", "content": query},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call-{episode_id}",
                            "type": "function",
                            "function": {
                                "name": "catalog.lookup",
                                "arguments": json.dumps({"query": query}, sort_keys=True),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": json.dumps(result, sort_keys=True),
                    "tool_call_id": f"call-{episode_id}",
                },
                {"role": "assistant", "content": f"Catalog outcome: {status}."},
            ],
            "contract_version": "1.0.0",
            "generator_version": "1.0.0",
        }

    train_rows = [dataset_row("train", index, route) for index, route in enumerate(routes, start=1)]
    eval_rows = [dataset_row("eval", index, route) for index, route in enumerate(routes, start=1)]
    _write(determinism, json.dumps({
        "schema_version": "miniagent.determinism-evidence.v1",
        "queries": [
            {
                "tool_id": row["tool_evidence"][0]["tool_id"],
                "normalized_arguments": row["tool_evidence"][0]["normalized_arguments"],
                "results": [
                    row["tool_evidence"][0]["result"],
                    row["tool_evidence"][0]["result"],
                ],
            }
            for row in train_rows
        ],
    }, sort_keys=True) + "\n")
    _write(train, "".join(json.dumps(row, sort_keys=True) + "\n" for row in train_rows))
    _write(evaluation, "".join(json.dumps(row, sort_keys=True) + "\n" for row in eval_rows))
    _write(leakage, json.dumps({
        "train_groups": [row["group_id"] for row in train_rows],
        "evaluation_groups": [row["group_id"] for row in eval_rows],
        "overlap": [],
    }, sort_keys=True) + "\n")

    def published(relative):
        return published_directory / relative

    return {
        "schema_version": "miniagent.dataset-manifest.v1",
        "dataset_id": "smoke-dataset",
        "application_id": contract["application_id"],
        "created_at": CREATED_AT,
        "generator": {
            "name": "recipe-smoke-generator",
            "version": "1.0.0",
            "source_revision": "fixture1",
            "seed": 7,
            "command": "make data-build",
            "source": {"path": "replaced", "sha256": "0" * 64},
            "configuration": {"path": "replaced", "sha256": "0" * 64},
        },
        "source_contracts": [],
        "world": {
            "world_id": "catalog-world",
            "version": "1.0.0",
            "specification": _record(world, published("world.json")),
            "simulator": _record(simulator, published("simulator.py")),
            "determinism_check": {
                "normalized_queries": 3,
                "replayed_queries": 6,
                "conflicts": 0,
                "evidence": _record(determinism, published("checks/determinism.json")),
            },
        },
        "record_schema": _record(record_schema, published("record.schema.json")),
        "splits": [
            {
                "name": "train",
                "purpose": "train",
                **_record(train, published("train.jsonl"), include_bytes=False),
                "format": "jsonl",
                "records": 3,
                "groups": 3,
                "sealed": False,
            },
            {
                "name": "evaluation",
                "purpose": "evaluation",
                **_record(evaluation, published("evaluation.jsonl"), include_bytes=False),
                "format": "jsonl",
                "records": 3,
                "groups": 3,
                "sealed": True,
            },
        ],
        "grouping": {
            "strategy": "grouped",
            "keys": ["group_id"],
            "leakage_check": {
                "status": "passed",
                "checked_at": CREATED_AT,
                "evidence": _record(leakage, published("checks/leakage.json")),
            },
        },
        "coverage": [
            {
                "dimension": "workflow",
                "expected": ["workflow.lookup"],
                "observed": ["workflow.lookup"],
                "missing": [],
            },
            {
                "dimension": "route",
                "expected": routes,
                "observed": routes,
                "missing": [],
            }
        ],
    }
'''


CONVERSATION_VALIDATOR_SOURCE = r'''
"""Semantic conversation checks for the recipe smoke application."""


VALIDATOR_READY = True


def validate(record, route):
    status = record["route_id"].rsplit(".", 1)[1]
    messages = record["conversation"]
    return (
        messages[1]["content"] == record["initial_context"]["request"]
        and messages[-1]["content"] == f"Catalog outcome: {status}."
        and route["actions"][-1]["kind"] == "respond"
    )
'''


TRAINER_SOURCE = r'''
"""Tiny deterministic trainer used by the generated-template smoke test."""

import hashlib
import json
from pathlib import Path
import shutil
import shlex


APPLICATION_ROOT = Path(__file__).resolve().parents[1]


def artifact(path):
    content = path.read_bytes()
    return {
        "path": path.relative_to(APPLICATION_ROOT).as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }


config = json.loads((APPLICATION_ROOT / "train/config.json").read_text(encoding="utf-8"))
dataset_manifest = json.loads(
    (APPLICATION_ROOT / config["input_manifest"]).read_text(encoding="utf-8")
)
training_split = next(
    item
    for item in dataset_manifest["splits"]
    if item["name"] == config["input_split"] and item["purpose"] == "train"
)
manifest_path = APPLICATION_ROOT / config["output_manifest"]
run_directory = manifest_path.parent
environment = run_directory / "provenance/environment.json"
source = run_directory / "provenance/trainer.py"
adapter = run_directory / "artifacts/adapter.safetensors"
parser = run_directory / "artifacts/action-parser.json"
runtime_adapter = run_directory / "artifacts/selected_policy.py"
environment.parent.mkdir(parents=True, exist_ok=True)
adapter.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(__file__, source)
environment.write_text(json.dumps({"python": "smoke", "dependencies": []}) + "\n", encoding="utf-8")
adapter.write_bytes(b"recipe-smoke-adapter-v1")
parser.write_text(json.dumps({"format": "json", "version": 1}, sort_keys=True) + "\n", encoding="utf-8")
runtime_adapter.write_text(
    "from ..business_logic import AgentResponse\n\n"
    "class SelectedPolicy:\n"
    "    def respond(self, request):\n"
    "        return AgentResponse(status='completed', message=f'Catalog result for {request.message}.')\n",
    encoding="utf-8",
)
dataset = APPLICATION_ROOT / config["input_manifest"]
base = APPLICATION_ROOT / "train/inputs/base-model.safetensors"
launcher = run_directory / "provenance/launcher-config.json"
trainer_config = run_directory / "provenance/trainer-configuration"
effective_argv = json.loads(launcher.read_text(encoding="utf-8"))["argv"]
adapter_record = artifact(adapter)
manifest = {
    "schema_version": "miniagent.training-manifest.v1",
    "run_id": run_directory.name,
    "application_id": "recipe_smoke",
    "created_at": "2026-01-01T00:00:00Z",
    "trainer": {
        "name": "recipe-smoke-trainer",
        "version": "1.0.0",
        "source_revision": "fixture1",
        "source": artifact(source),
        "environment": artifact(environment),
    },
    "dataset": artifact(dataset),
    "training_split": training_split,
    "base_model": {
        "model_id": "recipe-smoke-base",
        "revision": "base-v1",
        "artifacts": [{"role": "base", "artifact": artifact(base)}],
    },
    "method": "sft",
    "launcher_configuration": artifact(launcher),
    "configuration": artifact(trainer_config),
    "seed": 7,
    "summary": {
        "epochs": 1,
        "optimizer_steps": 1,
        "rollouts": 0,
        "wall_time_seconds": 0.01,
    },
    "outputs": [
        {"role": "adapter", "artifact": adapter_record},
        {"role": "action_parser", "artifact": artifact(parser)},
        {"role": "runtime_adapter", "artifact": artifact(runtime_adapter)},
    ],
    "metrics": [{"name": "loss", "value": 0.0, "step": 1}],
    "reproducibility": {
        "command": shlex.join(effective_argv),
        "argv": effective_argv,
        "checkpoint_policy": "one immutable final output",
        "final_sha256": adapter_record["sha256"],
    },
    "status": "completed",
}
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
'''


EVALUATOR_SOURCE = r'''
"""Tiny sealed evaluator used by the generated-template smoke test."""

import hashlib
import json
from pathlib import Path
import shutil
import shlex


APPLICATION_ROOT = Path(__file__).resolve().parents[1]


def artifact(path, kind=None, data_format=None, cases=None):
    content = path.read_bytes()
    result = {
        "path": path.relative_to(APPLICATION_ROOT).as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }
    if kind is not None:
        result["kind"] = kind
    if data_format is not None:
        result["format"] = data_format
    if cases is not None:
        result["cases"] = cases
    return result


config = json.loads((APPLICATION_ROOT / "eval/config.json").read_text(encoding="utf-8"))
manifest_path = APPLICATION_ROOT / config["output_manifest"]
evaluation_directory = manifest_path.parent
source = evaluation_directory / "provenance/evaluator.py"
environment = evaluation_directory / "provenance/environment.json"
transcripts = evaluation_directory / "evidence/transcripts.jsonl"
scores = evaluation_directory / "evidence/scores.jsonl"
source.parent.mkdir(parents=True, exist_ok=True)
transcripts.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(__file__, source)
environment.write_text(json.dumps({"python": "smoke", "dependencies": []}) + "\n", encoding="utf-8")
case_ids = ["eval-1", "eval-2", "eval-3"]
transcript_rows = [
    {"case_id": case_id, "goal_reached": True}
    for case_id in case_ids
]
transcripts.write_text(
    "".join(json.dumps(row) + "\n" for row in transcript_rows),
    encoding="utf-8",
)
score_rows = [
    {
        "case_id": row["case_id"],
        "status": "completed",
        "transcript_sha256": hashlib.sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "metrics": {"goal_completion": 1.0},
        "slices": ["workflow.lookup"],
    }
    for row in transcript_rows
]
scores.write_text(
    "".join(json.dumps(row) + "\n" for row in score_rows),
    encoding="utf-8",
)
sealed = APPLICATION_ROOT / config["sealed_dataset"]
base = APPLICATION_ROOT / "train/inputs/base-model.safetensors"
adapter = APPLICATION_ROOT / "train/runs/smoke-run/artifacts/adapter.safetensors"
parser = APPLICATION_ROOT / "train/runs/smoke-run/artifacts/action-parser.json"
runtime_adapter = APPLICATION_ROOT / "train/runs/smoke-run/artifacts/selected_policy.py"
launcher = evaluation_directory / "provenance/launcher-config.json"
evaluator_config = evaluation_directory / "provenance/evaluator-configuration"
effective_argv = json.loads(launcher.read_text(encoding="utf-8"))["argv"]
manifest = {
    "schema_version": "miniagent.evaluation-manifest.v1",
    "evaluation_id": evaluation_directory.name,
    "application_id": "recipe_smoke",
    "created_at": "2026-01-01T00:00:00Z",
    "evaluator": {
        "name": "recipe-smoke-evaluator",
        "version": "1.0.0",
        "source": artifact(source),
        "config": artifact(evaluator_config),
        "launcher_configuration": artifact(launcher),
    },
    "dataset": {
        "dataset_id": "smoke-dataset",
        "path": sealed.relative_to(APPLICATION_ROOT).as_posix(),
        "sha256": hashlib.sha256(sealed.read_bytes()).hexdigest(),
        "cases": 3,
        "sealed": True,
    },
    "candidate": {
        "candidate_id": "recipe-smoke-policy",
        "version": "smoke-v1",
        "runtime_entrypoint": "recipe_smoke.model.selected_policy:SelectedPolicy",
        "artifacts": [
            artifact(base),
            artifact(adapter),
            artifact(parser),
            artifact(runtime_adapter),
        ],
    },
    "protocol": {
        "interface": "library",
        "action_parser": "native-json",
        "parameters": {"temperature": 0.0},
    },
    "summary": {
        "attempted": 3,
        "completed": 3,
        "errors": 0,
        "status": "passed",
    },
    "metrics": [
        {
            "name": "goal_completion",
            "value": 1.0,
            "aggregation": "mean",
            "direction": "maximize",
            "unit": "ratio",
            "threshold": 1.0,
            "passed": True,
        }
    ],
    "evidence": [
        artifact(transcripts, "transcripts", "jsonl", 3),
        artifact(scores, "scores", "jsonl", 3),
    ],
    "reproducibility": {
        "command": shlex.join(effective_argv),
        "argv": effective_argv,
        "source_revision": "fixture1",
        "environment": artifact(environment),
    },
}
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
'''


TOOL_SOURCE = r'''
"""Catalog tool implementation used by the recipe smoke fixture."""

from typing import Any, Mapping

from ..service import ToolResult


def lookup(arguments: Mapping[str, Any]) -> ToolResult:
    query = str(arguments["query"]).strip().lower()
    if not query:
        return ToolResult.empty()
    return ToolResult.ok({"item": query, "available": True})
'''


def configure(application: Path) -> None:
    _write_text(
        application / "src/.env",
        (application / "src/.env.example").read_text(encoding="utf-8"),
    )
    _write_json(
        application / "data/specification/application.json",
        _application_contract(),
    )
    _write_json(
        application / "data/specification/tools/catalog.lookup.json",
        _tool_contract(),
    )
    _write_json(
        application / "data/specification/workflows/workflow.lookup.json",
        _workflow_contract(),
    )
    _write_text(application / "data/pipeline.py", PIPELINE_SOURCE)
    _write_text(
        application / "data/conversation_validator.py",
        CONVERSATION_VALIDATOR_SOURCE,
    )
    _write_text(application / "train/smoke_trainer.py", TRAINER_SOURCE)
    _write_text(application / "eval/smoke_evaluator.py", EVALUATOR_SOURCE)
    _write_text(application / f"src/{PACKAGE}/tool/smoke.py", TOOL_SOURCE)
    (application / "train/inputs").mkdir(parents=True, exist_ok=True)
    (application / "train/inputs/base-model.safetensors").write_bytes(
        b"recipe-smoke-base-v1"
    )
    _write_json(
        application / "data/config.json",
        {
            "schema_version": "miniagent.data-build.v1",
            "status": "approved",
            "dataset_id": DATASET_ID,
            "output_directory": f"data/artifacts/{DATASET_ID}",
            "output_manifest": f"data/artifacts/manifests/{DATASET_ID}.json",
        },
    )
    _write_json(
        application / "train/configurations/smoke.json",
        {"method": "sft", "epochs": 1, "seed": 7},
    )
    _write_json(
        application / "train/config.json",
        {
            "schema_version": "miniagent.training-run.v1",
            "status": "approved",
            "command": ["python", "train/smoke_trainer.py"],
            "input_manifest": f"data/artifacts/manifests/{DATASET_ID}.json",
            "input_split": "train",
            "run_configuration": "train/configurations/smoke.json",
            "output_manifest": f"train/runs/{RUN_ID}/training-manifest.json",
        },
    )
    _write_json(
        application / "eval/configurations/smoke.json",
        {"protocol": "sealed", "temperature": 0.0},
    )
    _write_json(
        application / "eval/config.json",
        {
            "schema_version": "miniagent.evaluation-run.v1",
            "status": "approved",
            "command": ["python", "eval/smoke_evaluator.py"],
            "dataset_manifest": f"data/artifacts/manifests/{DATASET_ID}.json",
            "sealed_dataset": f"data/artifacts/{DATASET_ID}/evaluation.jsonl",
            "evaluator_configuration": "eval/configurations/smoke.json",
            "output_manifest": (
                f"eval/results/runs/{EVALUATION_ID}/evaluation-manifest.json"
            ),
        },
    )


def _records_by_role(records: Iterable[Dict[str, Any]]) -> Dict[str, list[Dict[str, Any]]]:
    result: Dict[str, list[Dict[str, Any]]] = {}
    for record in records:
        result.setdefault(record["role"], []).append(record["artifact"])
    return result


def prepare_release(application: Path) -> None:
    dataset_path = application / f"data/artifacts/manifests/{DATASET_ID}.json"
    training_path = application / f"train/runs/{RUN_ID}/training-manifest.json"
    evaluation_path = (
        application
        / f"eval/results/runs/{EVALUATION_ID}/evaluation-manifest.json"
    )
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    training = json.loads(training_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    sources = _records_by_role(dataset["source_contracts"])
    application_record = sources["application"][0]
    workflows = sources["workflow"]
    tools = sources["tool"]
    outputs = {item["role"]: item["artifact"] for item in training["outputs"]}
    base = training["base_model"]["artifacts"][0]["artifact"]
    adapter = outputs["adapter"]
    parser = outputs["action_parser"]
    runtime_adapter = outputs["runtime_adapter"]
    leaderboard = json.loads(
        (application / "eval/leaderboard.json").read_text(encoding="utf-8")
    )
    leaderboard_entry = next(
        item
        for item in leaderboard["entries"]
        if item["evaluation_id"] == EVALUATION_ID
    )

    release_directory = application / f"eval/releases/{RELEASE_ID}"
    release_directory.mkdir(parents=True, exist_ok=False)
    verification_path = release_directory / "verification.json"
    _write_json(
        verification_path,
        {
            "evaluation_id": EVALUATION_ID,
            "training_run": RUN_ID,
            "checks": ["lineage", "hashes", "runtime-bundle"],
            "status": "passed",
        },
    )
    verification = _artifact(application, verification_path)
    dataset_record = _artifact(application, dataset_path)
    training_record = _artifact(application, training_path)
    evaluation_record = _artifact(application, evaluation_path)

    artifacts = [
        {"role": "application_contract", "artifact": application_record},
        *({"role": "workflow", "artifact": item} for item in workflows),
        *({"role": "tool_contract", "artifact": item} for item in tools),
        {"role": "runtime_asset", "artifact": parser},
        {"role": "runtime_asset", "artifact": runtime_adapter},
    ]
    destinations = [
        (
            "application_contract",
            application_record,
            f"{PACKAGE}/business_logic/application.json",
        ),
        *(
            (
                "workflow",
                item,
                f"{PACKAGE}/business_logic/workflows/workflow-{index}.json",
            )
            for index, item in enumerate(workflows, start=1)
        ),
        *(
            (
                "tool_contract",
                item,
                f"{PACKAGE}/business_logic/tools/tool-{index}.json",
            )
            for index, item in enumerate(tools, start=1)
        ),
        ("policy", base, f"{PACKAGE}/model/base.safetensors"),
        ("policy", adapter, f"{PACKAGE}/model/adapter.safetensors"),
        ("runtime_asset", parser, f"{PACKAGE}/model/action-parser.json"),
        (
            "runtime_asset",
            runtime_adapter,
            f"{PACKAGE}/model/selected_policy.py",
        ),
    ]
    promoted_bundle = [
        {
            "role": role,
            "path": destination,
            "sha256": record["sha256"],
            "bytes": record["bytes"],
        }
        for role, record, destination in destinations
    ]
    promoted_paths = {item["path"] for item in promoted_bundle}
    source_root = application / "src"
    ignored_parts = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
    }

    def source_role(relative: str) -> str:
        if relative.startswith("tests/"):
            return "test"
        if relative in {"pyproject.toml", "uv.lock", ".python-version"}:
            return "dependency"
        if relative.endswith(".schema.json"):
            return "runtime_contract"
        if relative.endswith(".py"):
            return "runtime_code"
        return "configuration"

    static_bundle = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(source_root).as_posix()
        if relative == ".env":
            continue
        if relative == f"{PACKAGE}/deployment_manifest.json":
            continue
        if any(part in ignored_parts for part in path.relative_to(source_root).parts):
            continue
        if relative in promoted_paths:
            continue
        static_bundle.append(
            {
                "role": source_role(relative),
                "path": relative,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    bundle = [*static_bundle, *promoted_bundle]
    release = {
        "schema_version": "miniagent.release-manifest.v1",
        "release_id": RELEASE_ID,
        "application_id": APPLICATION_ID,
        "version": "1.0.0",
        "created_at": CREATED_AT,
        "source_revision": "fixture1",
        "policy": {
            "model_id": "recipe-smoke-policy",
            "model_revision": evaluation["candidate"]["version"],
            "format": "safetensors",
            "action_parser": {"name": "native-json", "artifact": parser},
            "runtime": {
                "entrypoint": evaluation["candidate"]["runtime_entrypoint"],
                "artifact": runtime_adapter,
            },
            "weights": [
                {"role": "base", "artifact": base},
                {"role": "adapter", "artifact": adapter},
            ],
        },
        "runtime": {
            "package": PACKAGE,
            "python": ">=3.11",
            "entrypoints": {
                "api": f"{PACKAGE}.api:app",
                "cli": f"{PACKAGE}.cli:main",
            },
            "healthcheck": "/ready",
            "environment": [],
            "bundle": bundle,
        },
        "artifacts": artifacts,
        "evaluation": {
            "evaluation_id": EVALUATION_ID,
            "manifest": evaluation_record,
            "leaderboard_entry": leaderboard_entry,
            "status": "passed",
        },
        "provenance": {
            "application": application_record,
            "workflows": workflows,
            "tools": tools,
            "dataset": dataset_record,
            "training": training_record,
        },
        "verification": [
            {"name": "release-checks", "status": "passed", "evidence": verification}
        ],
        "status": "ready",
    }
    release_path = release_directory / "release-manifest.json"
    _write_json(release_path, release)
    selection = {
        "schema_version": "miniagent.release-selection.v1",
        "status": "selected",
        "package": PACKAGE,
        "evaluation_manifest": evaluation_record,
        "release_manifest": _artifact(application, release_path),
        "artifacts": [
            {
                "source": record["path"],
                "destination": destination,
                "sha256": record["sha256"],
            }
            for _, record, destination in destinations
        ],
    }
    _write_json(application / "eval/results/selected-release.json", selection)


def assert_evaluation(application: Path) -> None:
    leaderboard_path = application / "eval/leaderboard.json"
    leaderboard = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    entries = leaderboard.get("entries")
    if not isinstance(entries, list) or len(entries) != 1:
        raise RuntimeError("The successful evaluation must be indexed immediately once.")
    entry = entries[0]
    if entry.get("evaluation_id") != EVALUATION_ID or entry.get("status") != "passed":
        raise RuntimeError("The leaderboard entry must identify the passed evaluation.")
    manifest_path = application / entry["manifest"]["path"]
    if _sha256(manifest_path) != entry["manifest"]["sha256"]:
        raise RuntimeError("The leaderboard must bind the immutable evaluation manifest.")


def assert_release(application: Path) -> None:
    assert_evaluation(application)
    package_root = application / "src" / PACKAGE
    deployment = json.loads(
        (package_root / "deployment_manifest.json").read_text(encoding="utf-8")
    )
    if deployment.get("release_id") != RELEASE_ID or deployment.get("status") != "ready":
        raise RuntimeError("The promoted runtime must contain the ready selected release.")
    packaged_application = json.loads(
        (package_root / "business_logic/application.json").read_text(encoding="utf-8")
    )
    if packaged_application.get("status") != "approved":
        raise RuntimeError("Promotion must package the approved application contract.")
    for record in deployment["runtime"]["bundle"]:
        path = application / "src" / record["path"]
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"The runtime bundle differs at {record['path']}.")


def tamper_runtime(application: Path) -> None:
    agent_path = application / "src" / PACKAGE / "agent.py"
    agent_path.write_text(
        agent_path.read_text(encoding="utf-8") + "\nRUNTIME_TAMPER = True\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "configure",
            "prepare-release",
            "assert-evaluation",
            "assert-release",
            "tamper-runtime",
        ),
    )
    parser.add_argument("application", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    application = args.application.resolve()
    if args.action == "configure":
        configure(application)
    elif args.action == "prepare-release":
        prepare_release(application)
    elif args.action == "assert-evaluation":
        assert_evaluation(application)
    elif args.action == "assert-release":
        assert_release(application)
    else:
        tamper_runtime(application)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
