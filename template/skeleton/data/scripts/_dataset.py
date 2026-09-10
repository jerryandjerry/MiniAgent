"""Verification for immutable MiniAgent datasets and normalized trajectories."""

import copy
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Tuple
import unicodedata

from jsonschema import Draft202012Validator, FormatChecker

from _artifacts import (
    artifact_bytes,
    resolve_path,
    sha256_path,
    verify_reference,
)
from _contract import (
    APPLICATION_ROOT,
    CONTRACT_PATH,
    ContractError,
    declared_json_documents,
    validate_schema,
)


ArtifactResolver = Callable[[Dict[str, Any], str], Path]
ACTION_KINDS = {"collect", "tool", "tool_batch", "respond", "refuse"}


def artifact_records(manifest: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    yield "generator.source", manifest["generator"]["source"]
    yield "generator.configuration", manifest["generator"]["configuration"]
    yield (
        "generator.conversation_validator",
        manifest["generator"]["conversation_validator"],
    )
    for position, item in enumerate(manifest["source_contracts"]):
        yield f"source_contracts[{position}].artifact", item["artifact"]
    yield "world.specification", manifest["world"]["specification"]
    yield "world.simulator", manifest["world"]["simulator"]
    yield (
        "world.determinism_check.evidence",
        manifest["world"]["determinism_check"]["evidence"],
    )
    yield "record_schema", manifest["record_schema"]
    for position, split in enumerate(manifest["splits"]):
        yield f"splits[{position}]", split
    yield (
        "grouping.leakage_check.evidence",
        manifest["grouping"]["leakage_check"]["evidence"],
    )


def load_split_records(path: Path, data_format: str) -> list[Dict[str, Any]]:
    if data_format == "jsonl":
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif data_format == "json":
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, list):
            raise ContractError("Store JSON dataset splits as arrays of records.")
    elif data_format == "parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise ContractError(
                "Install pyarrow in the data environment to validate Parquet splits."
            ) from exc
        values = parquet.read_table(path).to_pylist()
    else:
        raise ContractError(f"Unsupported dataset split format: {data_format}.")
    if not all(isinstance(item, dict) for item in values):
        raise ContractError("Write each dataset split as objects.")
    return values


def _record_value(record: Dict[str, Any], key: str) -> Any:
    value: Any = record
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ContractError(f"Every dataset record must define grouping key {key}.")
        value = value[part]
    return value


def _load_contract_documents(
    contract: Dict[str, Any], artifact_name: str
) -> list[Dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in declared_json_documents(contract, artifact_name)
    ]


def _route_catalog(contract: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}
    for workflow in _load_contract_documents(contract, "workflows"):
        workflow_id = str(workflow["workflow_id"])
        slot_records = {
            str(item["name"]): item for item in workflow["slots"]
        }
        context_slots = {
            str(item["name"])
            for item in workflow["slots"]
            if item["source"] == "context"
        }
        states = {str(item["id"]): item for item in workflow["states"]}
        transitions = {
            str(item["id"]): item for item in workflow["transitions"]
        }
        for route in workflow["routes"]:
            route_id = str(route["id"])
            current = str(workflow["initial_state"])
            actions = []
            evidence = []
            for transition_id in route["transition_ids"]:
                transition = transitions[str(transition_id)]
                state = states[current]
                kind = state["kind"]
                if kind in ACTION_KINDS:
                    action: Dict[str, Any] = {
                        "state_id": current,
                        "kind": kind,
                    }
                    if kind == "collect":
                        action["slot"] = state["slot"]
                        action["prompt"] = state["prompt"]
                    elif kind == "tool":
                        action["tools"] = [state["tool"]]
                    elif kind == "tool_batch":
                        action["tools"] = [call["tool"] for call in state["calls"]]
                    elif kind == "respond":
                        action["content_requirements"] = list(
                            state["content_requirements"]
                        )
                    elif kind == "refuse":
                        action["policy"] = state["policy"]
                    actions.append(action)
                if kind == "tool":
                    evidence.append(
                        {
                            "state_id": current,
                            "tool_id": state["tool"],
                            "status": transition["when"]["status"],
                            "argument_bindings": state["argument_bindings"],
                        }
                    )
                elif kind == "tool_batch":
                    statuses = transition["when"]["statuses"]
                    evidence.extend(
                        {
                            "state_id": current,
                            "tool_id": call["tool"],
                            "status": statuses[call["tool"]],
                            "argument_bindings": call["argument_bindings"],
                        }
                        for call in state["calls"]
                    )
                current = str(transition["to"])
            terminal = states[current]
            catalog[route_id] = {
                "workflow_id": workflow_id,
                "initial_slots": list(route["initial_slots"]),
                "transition_ids": list(route["transition_ids"]),
                "actions": actions,
                "terminal_state": route["terminal_state"],
                "goal_reached": terminal["goal_reached"],
                "evidence": evidence,
                "required_initial_context": sorted(
                    context_slots
                ),
                "slots": slot_records,
            }
    return catalog


def _tool_catalog(contract: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(tool["tool_id"]): tool
        for tool in _load_contract_documents(contract, "tools")
    }


def _validate_tool_result(
    result: Any, tool: Dict[str, Any], expected_status: str, label: str
) -> None:
    if not isinstance(result, dict) or result.get("status") != expected_status:
        raise ContractError(f"{label} must carry the route-selected tool status.")
    if expected_status == "ok":
        if set(result) != {"status", "data"}:
            raise ContractError(f"{label} ok result must use the exact ok envelope.")
        if "data" not in result:
            raise ContractError(f"{label} ok result requires data.")
        validator = Draft202012Validator(
            tool["outcomes"]["ok"]["data_schema"],
            format_checker=FormatChecker(),
        )
        errors = list(validator.iter_errors(result["data"]))
        if errors:
            raise ContractError(f"{label} ok data violates its tool result schema.")
        return
    if expected_status == "empty":
        if set(result) not in ({"status", "data"}, {"status", "data", "reason"}):
            raise ContractError(f"{label} empty result must use the exact empty envelope.")
        if result.get("data") is not None:
            raise ContractError(f"{label} empty result requires null data.")
        reason = result.get("reason")
        if reason is not None and reason not in tool["outcomes"]["empty"]["reasons"]:
            raise ContractError(f"{label} empty reason is absent from the tool contract.")
        return
    if set(result) != {"status", "error"}:
        raise ContractError(f"{label} error result must use the exact error envelope.")
    error = result.get("error")
    if not isinstance(error, dict) or set(error) != {"code", "message", "retryable"}:
        raise ContractError(f"{label} error result requires an error object.")
    definitions = {
        item["code"]: item for item in tool["outcomes"]["error"]["codes"]
    }
    definition = definitions.get(error.get("code"))
    if definition is None:
        raise ContractError(f"{label} error code is absent from the tool contract.")
    if not isinstance(error.get("message"), str) or not error["message"]:
        raise ContractError(f"{label} error result requires a message.")
    if error.get("retryable") is not definition["retryable"]:
        raise ContractError(f"{label} retryable value differs from the tool contract.")


def _message_content(message: Dict[str, Any], label: str) -> str:
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ContractError(f"{label} requires non-empty text content.")
    return content


def _json_object(value: Any, label: str) -> Dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ContractError(f"{label} must contain valid JSON.") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must contain one JSON object.")
    return value


def _validate_conversation_replay(
    conversation: Any,
    route: Dict[str, Any],
    evidence: list[Dict[str, Any]],
    label: str,
) -> None:
    if not isinstance(conversation, list) or len(conversation) < 3:
        raise ContractError(f"{label} requires a complete function-calling conversation.")
    if not isinstance(conversation[0], dict) or conversation[0].get("role") != "system":
        raise ContractError(f"{label} conversation must begin with a system message.")
    _message_content(conversation[0], f"{label} system message")
    if not isinstance(conversation[1], dict) or conversation[1].get("role") != "user":
        raise ContractError(f"{label} system message must be followed by the user request.")
    _message_content(conversation[1], f"{label} initial user message")

    message_position = 2
    evidence_position = 0
    call_ids: set[str] = set()
    for action_position, action in enumerate(route["actions"]):
        action_label = f"{label} action {action_position}"
        if message_position >= len(conversation):
            raise ContractError(f"{action_label} is missing from the conversation.")
        assistant = conversation[message_position]
        if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
            raise ContractError(f"{action_label} requires an assistant message.")

        if action["kind"] == "collect":
            _message_content(assistant, f"{action_label} question")
            if assistant.get("tool_calls"):
                raise ContractError(f"{action_label} question cannot contain tool calls.")
            message_position += 1
            if message_position >= len(conversation):
                raise ContractError(f"{action_label} requires the user's answer.")
            user = conversation[message_position]
            if not isinstance(user, dict) or user.get("role") != "user":
                raise ContractError(f"{action_label} question must be followed by the user.")
            _message_content(user, f"{action_label} user answer")
            message_position += 1
            continue

        if action["kind"] in {"respond", "refuse"}:
            _message_content(assistant, f"{action_label} response")
            if assistant.get("tool_calls"):
                raise ContractError(f"{action_label} response cannot contain tool calls.")
            message_position += 1
            continue

        calls = assistant.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != len(action["tools"]):
            raise ContractError(f"{action_label} must contain its exact tool-call batch.")
        if assistant.get("content") not in (None, ""):
            raise ContractError(f"{action_label} tool emission must use empty content.")
        batch = evidence[evidence_position : evidence_position + len(calls)]
        for call_position, (call, tool_id, expected) in enumerate(
            zip(calls, action["tools"], batch)
        ):
            call_label = f"{action_label} tool_calls[{call_position}]"
            if (
                not isinstance(call, dict)
                or set(call) != {"id", "type", "function"}
                or call.get("type") != "function"
                or not isinstance(call.get("id"), str)
                or not call["id"]
                or not isinstance(call.get("function"), dict)
                or set(call["function"]) != {"name", "arguments"}
            ):
                raise ContractError(f"{call_label} must use the function-call envelope.")
            if call["id"] in call_ids:
                raise ContractError(f"{call_label} requires a unique call id.")
            call_ids.add(call["id"])
            if call["function"]["name"] != tool_id:
                raise ContractError(f"{call_label} tool name differs from the route.")
            arguments = _json_object(
                call["function"]["arguments"], f"{call_label} arguments"
            )
            if arguments != expected["arguments"]:
                raise ContractError(f"{call_label} arguments differ from frozen evidence.")
        message_position += 1
        for call_position, (call, expected) in enumerate(zip(calls, batch)):
            result_label = f"{action_label} tool results[{call_position}]"
            if message_position >= len(conversation):
                raise ContractError(f"{result_label} is missing from the conversation.")
            result_message = conversation[message_position]
            if (
                not isinstance(result_message, dict)
                or set(result_message) != {"role", "content", "tool_call_id"}
                or result_message.get("role") != "tool"
                or result_message.get("tool_call_id") != call["id"]
            ):
                raise ContractError(f"{result_label} must match its tool call id.")
            result = _json_object(result_message["content"], f"{result_label} content")
            if result != expected["result"]:
                raise ContractError(f"{result_label} differs from frozen evidence.")
            message_position += 1
        evidence_position += len(calls)

    if evidence_position != len(evidence):
        raise ContractError(f"{label} conversation omits frozen tool evidence.")
    if message_position != len(conversation):
        raise ContractError(f"{label} conversation contains actions outside its route.")


def _load_conversation_validator(path: Path) -> Callable[[Dict[str, Any], Dict[str, Any]], bool]:
    namespace: Dict[str, Any] = {
        "__file__": str(path),
        "__name__": f"miniagent_conversation_validator_{sha256_path(path)[:16]}",
    }
    source = path.read_text(encoding="utf-8")
    exec(compile(source, str(path), "exec"), namespace)
    validator = namespace.get("validate")
    if namespace.get("VALIDATOR_READY") is not True or not callable(validator):
        raise ContractError(
            "Implement the application conversation validator and set VALIDATOR_READY = True."
        )
    return validator


def _load_world_simulator(path: Path) -> Callable[[str, Dict[str, Any]], Any]:
    if not path.is_file():
        raise ContractError("Store the synthetic-world simulator as one Python file.")
    namespace: Dict[str, Any] = {
        "__file__": str(path),
        "__name__": f"miniagent_world_simulator_{sha256_path(path)[:16]}",
    }
    try:
        source = path.read_text(encoding="utf-8")
        exec(compile(source, str(path), "exec"), namespace)
    except Exception as exc:
        raise ContractError("Load the hash-bound synthetic-world simulator.") from exc
    execute = namespace.get("execute")
    if namespace.get("SIMULATOR_READY") is not True or not callable(execute):
        raise ContractError(
            "Implement execute(tool_id, normalized_arguments) and set "
            "SIMULATOR_READY = True."
        )
    return execute


def _slot_values(
    record: Dict[str, Any], route: Dict[str, Any], label: str
) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for slot_id, slot in route["slots"].items():
        source = record["initial_context"] if slot["source"] == "context" else record["private_state"]
        if slot_id not in source:
            raise ContractError(
                f"{label} is missing {slot['source']} slot value {slot_id}."
            )
        value = source[slot_id]
        validator = Draft202012Validator(
            slot["schema"], format_checker=FormatChecker()
        )
        if list(validator.iter_errors(value)):
            raise ContractError(f"{label} slot {slot_id} violates its workflow schema.")
        values[slot_id] = value
    return values


def _resolve_binding(
    expression: str,
    slots: Mapping[str, Any],
    prior_evidence: list[Dict[str, Any]],
    label: str,
) -> Any:
    if expression.startswith("slot."):
        return slots[expression[5:]]
    matches = [
        item
        for item in reversed(prior_evidence)
        if expression.startswith(f"tool.{item['tool_id']}.")
    ]
    if not matches:
        raise ContractError(f"{label} requires an unavailable prior tool result.")
    producer = matches[0]
    if producer["result"]["status"] != "ok":
        raise ContractError(f"{label} requires data from a non-ok prior tool result.")
    path = expression[len(f"tool.{producer['tool_id']}.") :].split(".")
    value: Any = producer["result"]["data"]
    for field in path:
        if not isinstance(value, dict) or field not in value:
            raise ContractError(f"{label} cannot resolve binding {expression}.")
        value = value[field]
    return value


def _normalize_value(value: Any, operations: list[str], label: str) -> Any:
    normalized = value
    for operation in operations:
        if operation in {"unicode_nfkc", "trim", "lowercase", "uppercase"}:
            if not isinstance(normalized, str):
                raise ContractError(f"{label} string normalization requires a string value.")
            if operation == "unicode_nfkc":
                normalized = unicodedata.normalize("NFKC", normalized)
            elif operation == "trim":
                normalized = normalized.strip()
            elif operation == "lowercase":
                normalized = normalized.lower()
            else:
                normalized = normalized.upper()
        elif operation == "sort_unique":
            if not isinstance(normalized, list):
                raise ContractError(f"{label} sort_unique normalization requires a list.")
            keyed = {
                json.dumps(
                    item,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ): item
                for item in normalized
            }
            normalized = [keyed[key] for key in sorted(keyed)]
    return normalized


def _validate_determinism_evidence(
    manifest: Dict[str, Any],
    resolve_artifact: ArtifactResolver,
    result_by_query: Mapping[str, str],
) -> None:
    determinism = manifest["world"]["determinism_check"]
    simulator_path = resolve_artifact(
        manifest["world"]["simulator"], "world.simulator"
    )
    execute = _load_world_simulator(simulator_path)
    path = resolve_artifact(
        determinism["evidence"], "world.determinism_check.evidence"
    )
    if not path.is_file():
        raise ContractError("Store determinism evidence as one JSON file.")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(evidence, dict)
        or set(evidence) != {"schema_version", "queries"}
        or evidence.get("schema_version") != "miniagent.determinism-evidence.v1"
        or not isinstance(evidence.get("queries"), list)
    ):
        raise ContractError("Use the stable synthetic-world determinism evidence envelope.")

    observed: Dict[str, list[str]] = {}
    replayed = 0
    for position, query in enumerate(evidence["queries"]):
        label = f"Determinism query {position}"
        if (
            not isinstance(query, dict)
            or set(query) != {"tool_id", "normalized_arguments", "results"}
            or not isinstance(query.get("tool_id"), str)
            or not query["tool_id"]
            or not isinstance(query.get("normalized_arguments"), dict)
            or not isinstance(query.get("results"), list)
            or len(query["results"]) < 2
        ):
            raise ContractError(f"{label} must record at least two exact replays.")
        key = json.dumps(
            [query["tool_id"], query["normalized_arguments"]],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if key in observed:
            raise ContractError("Record each normalized determinism query once.")
        results = [
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for value in query["results"]
        ]
        if len(set(results)) != 1:
            raise ContractError(f"{label} produced conflicting replay results.")
        actual_results = []
        for _ in results:
            try:
                actual = execute(
                    query["tool_id"], copy.deepcopy(query["normalized_arguments"])
                )
                actual_results.append(
                    json.dumps(
                        actual,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
            except Exception as exc:
                raise ContractError(
                    f"{label} could not be replayed by the synthetic-world simulator."
                ) from exc
        if len(set(actual_results)) != 1:
            raise ContractError(
                f"{label} produced conflicting synthetic-world simulator replays."
            )
        if actual_results != results:
            raise ContractError(
                f"{label} simulator replays differ from determinism evidence."
            )
        observed[key] = results
        replayed += len(results)

    if set(observed) != set(result_by_query):
        raise ContractError("Replay every normalized query present in the dataset.")
    for key, results in observed.items():
        if results[0] != result_by_query[key]:
            raise ContractError("Match determinism replays to frozen dataset results.")
    if (
        determinism["normalized_queries"] != len(observed)
        or determinism["replayed_queries"] != replayed
        or determinism["conflicts"] != 0
    ):
        raise ContractError("Derive determinism counts from exact replay evidence.")


def _validate_record(
    record: Dict[str, Any],
    route: Dict[str, Any],
    tools: Mapping[str, Dict[str, Any]],
    contract: Dict[str, Any],
    generator_version: str,
    result_by_query: Dict[str, str],
    conversation_validator: Callable[[Dict[str, Any], Dict[str, Any]], bool],
    label: str,
) -> None:
    required = {
        "episode_id",
        "workflow_id",
        "route_id",
        "initial_context",
        "private_state",
        "trajectory",
        "tool_evidence",
        "conversation",
        "contract_version",
        "generator_version",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise ContractError(f"{label} is missing stable fields: {', '.join(missing)}.")
    if str(record["workflow_id"]) != route["workflow_id"]:
        raise ContractError(f"{label} route belongs to a different workflow.")
    if record["contract_version"] != contract["version"]:
        raise ContractError(f"{label} contract_version differs from the application.")
    if record["generator_version"] != generator_version:
        raise ContractError(f"{label} generator_version differs from the manifest.")
    if not isinstance(record["initial_context"], dict) or not isinstance(
        record["private_state"], dict
    ):
        raise ContractError(f"{label} context and private state must be objects.")
    slot_values = _slot_values(record, route, label)

    trajectory = record["trajectory"]
    expected_trajectory = {
        "initial_slots": route["initial_slots"],
        "transition_ids": route["transition_ids"],
        "actions": route["actions"],
        "terminal_state": route["terminal_state"],
        "goal_reached": route["goal_reached"],
    }
    if trajectory != expected_trajectory:
        raise ContractError(f"{label} trajectory differs from its declared route.")

    observed_evidence = record["tool_evidence"]
    if not isinstance(observed_evidence, list) or len(observed_evidence) != len(
        route["evidence"]
    ):
        raise ContractError(f"{label} tool evidence differs from its route actions.")
    prior_evidence: list[Dict[str, Any]] = []
    for position, (observed, expected) in enumerate(
        zip(observed_evidence, route["evidence"])
    ):
        evidence_label = f"{label} tool_evidence[{position}]"
        if not isinstance(observed, dict) or set(observed) != {
            "state_id",
            "tool_id",
            "arguments",
            "normalized_arguments",
            "result",
        }:
            raise ContractError(f"{evidence_label} must use the stable evidence envelope.")
        if (
            observed["state_id"] != expected["state_id"]
            or observed["tool_id"] != expected["tool_id"]
        ):
            raise ContractError(f"{evidence_label} differs from the ordered route tools.")
        tool = tools[expected["tool_id"]]
        expected_arguments = {
            argument: _resolve_binding(
                expression, slot_values, prior_evidence, evidence_label
            )
            for argument, expression in expected["argument_bindings"].items()
        }
        if observed["arguments"] != expected_arguments:
            raise ContractError(
                f"{evidence_label}.arguments differ from workflow bindings."
            )
        expected_normalized = {
            argument: _normalize_value(
                value,
                tool["normalization"][argument],
                f"{evidence_label}.{argument}",
            )
            for argument, value in expected_arguments.items()
        }
        if observed["normalized_arguments"] != expected_normalized:
            raise ContractError(
                f"{evidence_label}.normalized_arguments differ from tool normalization."
            )
        for field in ("arguments", "normalized_arguments"):
            validator = Draft202012Validator(
                tool["arguments"], format_checker=FormatChecker()
            )
            if list(validator.iter_errors(observed[field])):
                raise ContractError(
                    f"{evidence_label}.{field} violates the tool argument schema."
                )
        _validate_tool_result(
            observed["result"], tool, expected["status"], evidence_label
        )
        query_key = json.dumps(
            [observed["tool_id"], observed["normalized_arguments"]],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        result_value = json.dumps(
            observed["result"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        prior = result_by_query.setdefault(query_key, result_value)
        if prior != result_value:
            raise ContractError(
                f"{evidence_label} conflicts with the frozen normalized tool result."
            )
        prior_evidence.append(observed)
    _validate_conversation_replay(
        record["conversation"], route, observed_evidence, label
    )
    try:
        accepted = conversation_validator(record, route)
    except Exception as exc:
        raise ContractError(f"{label} failed application conversation validation: {exc}") from exc
    if accepted is not True:
        raise ContractError(f"{label} failed application conversation validation.")


def validate_dataset_content(
    manifest: Dict[str, Any],
    contract: Dict[str, Any],
    resolve_artifact: ArtifactResolver,
) -> Tuple[set[str], set[str], Dict[str, str]]:
    schema_path = resolve_artifact(manifest["record_schema"], "record_schema")
    if not schema_path.is_file():
        raise ContractError("Set record_schema to one JSON Schema file.")
    record_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(record_schema)
    validator = Draft202012Validator(record_schema, format_checker=FormatChecker())
    route_catalog = _route_catalog(contract)
    tool_catalog = _tool_catalog(contract)
    expected_routes = set(route_catalog)
    expected_workflows = {item["workflow_id"] for item in route_catalog.values()}
    grouping_keys = manifest["grouping"]["keys"]
    groups_by_split: Dict[str, set[str]] = {}
    routes_by_purpose: Dict[str, set[str]] = {}
    workflows_by_purpose: Dict[str, set[str]] = {}
    episode_ids: set[str] = set()
    route_ids: set[str] = set()
    workflow_ids: set[str] = set()
    result_by_query: Dict[str, str] = {}
    conversation_validator_path = resolve_artifact(
        manifest["generator"]["conversation_validator"],
        "generator.conversation_validator",
    )
    if not conversation_validator_path.is_file():
        raise ContractError("Set generator.conversation_validator to one Python file.")
    conversation_validator = _load_conversation_validator(
        conversation_validator_path
    )

    for split_position, split in enumerate(manifest["splits"]):
        split_path = resolve_artifact(split, f"splits[{split_position}]")
        if not split_path.is_file():
            raise ContractError("Set each dataset split path to one data file.")
        records = load_split_records(split_path, split["format"])
        if len(records) != split["records"]:
            raise ContractError(f"Match split {split['name']} record count to its file.")
        groups = set()
        split_routes = set()
        split_workflows = set()
        for record_position, record in enumerate(records):
            errors = sorted(
                validator.iter_errors(record),
                key=lambda error: tuple(str(part) for part in error.path),
            )
            if errors:
                location = ".".join(str(part) for part in errors[0].absolute_path) or "root"
                raise ContractError(
                    f"Split {split['name']} record {record_position} violates record_schema "
                    f"at {location}: {errors[0].message}"
                )
            label = f"Split {split['name']} record {record_position}"
            episode_id = str(record.get("episode_id", ""))
            route_id = str(record.get("route_id", ""))
            workflow_id = str(record.get("workflow_id", ""))
            if not episode_id or not route_id or not workflow_id:
                raise ContractError(f"{label} requires stable episode, workflow, and route IDs.")
            if episode_id in episode_ids:
                raise ContractError(f"Assign unique dataset episode_id values: {episode_id}.")
            route = route_catalog.get(route_id)
            if route is None:
                raise ContractError(f"{label} names unknown route {route_id}.")
            _validate_record(
                record,
                route,
                tool_catalog,
                contract,
                manifest["generator"]["version"],
                result_by_query,
                conversation_validator,
                label,
            )
            episode_ids.add(episode_id)
            workflow_ids.add(workflow_id)
            route_ids.add(route_id)
            split_workflows.add(workflow_id)
            split_routes.add(route_id)
            group = json.dumps(
                [_record_value(record, key) for key in grouping_keys],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            groups.add(group)
        if len(groups) != split["groups"]:
            raise ContractError(f"Match split {split['name']} group count to its records.")
        groups_by_split[split["name"]] = groups
        routes_by_purpose.setdefault(split["purpose"], set()).update(split_routes)
        workflows_by_purpose.setdefault(split["purpose"], set()).update(split_workflows)

    split_names = sorted(groups_by_split)
    for position, left_name in enumerate(split_names):
        for right_name in split_names[position + 1 :]:
            if groups_by_split[left_name].intersection(groups_by_split[right_name]):
                raise ContractError(
                    f"Keep grouped evidence disjoint between {left_name} and {right_name}."
                )
    for purpose in ("train", "evaluation"):
        if routes_by_purpose.get(purpose) != expected_routes:
            raise ContractError(f"Cover every approved route in the {purpose} partition.")
        if workflows_by_purpose.get(purpose) != expected_workflows:
            raise ContractError(f"Cover every approved workflow in the {purpose} partition.")
    _validate_determinism_evidence(manifest, resolve_artifact, result_by_query)
    return workflow_ids, route_ids, result_by_query


def verify_dataset_manifest(
    manifest_path: Path, contract: Dict[str, Any]
) -> Dict[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"Write valid JSON to dataset manifest {manifest_path}.") from exc
    validate_schema(manifest, "dataset-manifest.schema.json", "Dataset manifest")
    if manifest["application_id"] != contract["application_id"]:
        raise ContractError("Match the dataset and application contract identities.")
    manifest_root = resolve_path(
        contract["artifacts"]["dataset_manifests"], "artifacts.dataset_manifests"
    )
    if (
        manifest_path.parent.resolve() != manifest_root.resolve()
        or manifest_path.name != f"{manifest['dataset_id']}.json"
    ):
        raise ContractError("Use the canonical immutable dataset manifest path.")
    payload_root = manifest_root.parent / manifest["dataset_id"]

    resolved = []
    for label, record in artifact_records(manifest):
        path, _ = verify_reference(record, label)
        try:
            path.resolve().relative_to(payload_root.resolve())
        except ValueError as exc:
            raise ContractError(f"Keep {label} inside the immutable dataset payload.") from exc
        resolved.append(path.resolve())
    if len(resolved) != len(set(resolved)):
        raise ContractError("Bind each dataset artifact path once.")
    if any(
        left in right.parents
        for left in resolved
        for right in resolved
        if left != right
    ):
        raise ContractError("Keep dataset artifact records non-overlapping.")
    if any(path.is_symlink() for path in payload_root.rglob("*")):
        raise ContractError("Build dataset payloads from regular files and directories.")
    unbound = []
    for path in (item for item in payload_root.rglob("*") if item.is_file()):
        if not any(
            path.resolve() == declared
            or (declared.is_dir() and declared in path.resolve().parents)
            for declared in resolved
        ):
            unbound.append(path.relative_to(payload_root).as_posix())
    if unbound:
        raise ContractError(
            "Bind every dataset payload file into the manifest: "
            + ", ".join(sorted(unbound))
            + "."
        )

    expected_sources = {
        CONTRACT_PATH.relative_to(APPLICATION_ROOT).as_posix(),
        *(
            path.relative_to(APPLICATION_ROOT).as_posix()
            for path in declared_json_documents(contract, "workflows")
        ),
        *(
            path.relative_to(APPLICATION_ROOT).as_posix()
            for path in declared_json_documents(contract, "tools")
        ),
    }
    if {item["source_path"] for item in manifest["source_contracts"]} != expected_sources:
        raise ContractError("Bind the exact approved source contracts into the dataset.")
    for item in manifest["source_contracts"]:
        source = resolve_path(item["source_path"], "source_contracts.source_path")
        snapshot, _ = verify_reference(item["artifact"], "source contract snapshot")
        if sha256_path(source) != sha256_path(snapshot) or artifact_bytes(source) != artifact_bytes(snapshot):
            raise ContractError("Match current approved contracts to dataset snapshots.")

    def resolver(record: Dict[str, Any], label: str) -> Path:
        return verify_reference(record, label)[0]

    workflow_ids, route_ids, _ = validate_dataset_content(manifest, contract, resolver)
    for label, record in artifact_records(manifest):
        verify_reference(record, label)
    if any(path.is_symlink() for path in payload_root.rglob("*")):
        raise ContractError("Conversation validation created a dataset symlink.")
    post_validation_unbound = []
    for path in (item for item in payload_root.rglob("*") if item.is_file()):
        if not any(
            path.resolve() == declared
            or (declared.is_dir() and declared in path.resolve().parents)
            for declared in resolved
        ):
            post_validation_unbound.append(path.relative_to(payload_root).as_posix())
    if post_validation_unbound:
        raise ContractError(
            "Conversation validation created unbound dataset files: "
            + ", ".join(sorted(post_validation_unbound))
            + "."
        )
    coverage = {item["dimension"]: item for item in manifest["coverage"]}
    for dimension, observed in (("workflow", workflow_ids), ("route", route_ids)):
        item = coverage.get(dimension)
        if item is None or set(item["observed"]) != observed:
            raise ContractError(f"Derive {dimension} coverage from verified dataset records.")
    return manifest
