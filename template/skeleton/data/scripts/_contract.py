"""Application-contract loading and validation."""

import json
from itertools import product
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Set, Tuple

from jsonschema import Draft202012Validator, FormatChecker


APPLICATION_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = APPLICATION_ROOT / "data" / "specification" / "application.json"
SCHEMA_ROOT = APPLICATION_ROOT / "data" / "specification" / "schemas"
IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:[._-][A-Za-z0-9]+)*$")


class ContractError(ValueError):
    """Raised when the compiled application contract violates its shape."""


def validate_schema(payload: Any, schema_name: str, label: str) -> None:
    schema_path = SCHEMA_ROOT / schema_name
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(f"Restore the application schema: {schema_path}.") from exc
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(payload),
        key=lambda error: tuple(str(part) for part in error.path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "root"
        raise ContractError(f"{label} violates {schema_name} at {location}: {error.message}")


def _object(value: Any, path: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"Set {path} to an object.")
    return value


def _non_empty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"Set {path} to a non-empty string.")
    return value


def _identifier(value: Any, path: str) -> str:
    text = _non_empty_string(value, path)
    if IDENTIFIER.fullmatch(text) is None:
        raise ContractError(f"Set {path} to a stable identifier.")
    return text


def _list(value: Any, path: str) -> List[Any]:
    if not isinstance(value, list):
        raise ContractError(f"Set {path} to a list.")
    return value


def _exact_keys(value: Dict[str, Any], expected: Iterable[str], path: str) -> None:
    expected_set = set(expected)
    absent = sorted(expected_set.difference(value))
    extra = sorted(set(value).difference(expected_set))
    if absent:
        raise ContractError(f"Add required {path} fields: " + ", ".join(absent))
    if extra:
        raise ContractError(f"Remove unsupported {path} fields: " + ", ".join(extra))


def _relative_path(value: Any, path: str) -> Path:
    text = _non_empty_string(value, path)
    candidate = Path(text)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ContractError(f"Keep {path} inside the application directory.")
    return candidate


def _validate_named_records(
    value: Any, path: str, required_fields: Iterable[str]
) -> List[Dict[str, Any]]:
    records = _list(value, path)
    identifiers = []
    result = []
    for position, item in enumerate(records):
        record = _object(item, f"{path}[{position}]")
        for field in required_fields:
            _non_empty_string(record.get(field), f"{path}[{position}].{field}")
        record_id = _identifier(record.get("id"), f"{path}[{position}].id")
        identifiers.append(record_id)
        result.append(record)
    if len(identifiers) != len(set(identifiers)):
        raise ContractError(f"Assign a unique id to every {path} entry.")
    return result


def _validate_business(value: Any) -> Dict[str, Any]:
    business = _object(value, "business")
    _exact_keys(
        business,
        {
            "objective",
            "users",
            "supported_requests",
            "rules",
            "context_fields",
            "systems",
            "outcomes",
        },
        "business",
    )
    _non_empty_string(business["objective"], "business.objective")
    users = _list(business["users"], "business.users")
    for position, user in enumerate(users):
        _non_empty_string(user, f"business.users[{position}]")
    _validate_named_records(
        business["supported_requests"],
        "business.supported_requests",
        {"id", "description"},
    )
    _validate_named_records(business["rules"], "business.rules", {"id", "statement"})
    context_names = []
    for position, value in enumerate(
        _list(business["context_fields"], "business.context_fields")
    ):
        record = _object(value, f"business.context_fields[{position}]")
        context_names.append(
            _identifier(record.get("name"), f"business.context_fields[{position}].name")
        )
    if len(context_names) != len(set(context_names)):
        raise ContractError("Assign a unique name to every business context field.")
    system_ids = []
    for position, value in enumerate(_list(business["systems"], "business.systems")):
        record = _object(value, f"business.systems[{position}]")
        system_ids.append(
            _identifier(record.get("id"), f"business.systems[{position}].id")
        )
    if len(system_ids) != len(set(system_ids)):
        raise ContractError("Assign a unique id to every business system.")
    outcomes = _object(business["outcomes"], "business.outcomes")
    _exact_keys(outcomes, {"completion", "empty", "error", "refusal"}, "outcome")
    for outcome in ("completion", "empty", "error", "refusal"):
        _non_empty_string(outcomes[outcome], f"business.outcomes.{outcome}")
    return business


def _validate_artifacts(value: Any, package: str) -> Dict[str, Any]:
    artifacts = _object(value, "artifacts")
    _exact_keys(
        artifacts,
        {
            "workflows",
            "tools",
            "dataset_manifests",
            "training_runs",
            "evaluation_results",
            "releases",
            "runtime_source",
        },
        "artifact",
    )
    for collection in ("workflows", "tools"):
        paths = _list(artifacts[collection], f"artifacts.{collection}")
        if not paths:
            raise ContractError(f"Add at least one path to artifacts.{collection}.")
        for position, item in enumerate(paths):
            _relative_path(item, f"artifacts.{collection}[{position}]")
    for name in (
        "dataset_manifests",
        "training_runs",
        "evaluation_results",
        "releases",
        "runtime_source",
    ):
        _relative_path(artifacts[name], f"artifacts.{name}")
    boundaries = {
        "dataset_manifests": Path("data/artifacts"),
        "training_runs": Path("train"),
        "evaluation_results": Path("eval"),
        "releases": Path("eval"),
    }
    for name, boundary in boundaries.items():
        path = Path(artifacts[name])
        try:
            path.relative_to(boundary)
        except ValueError as exc:
            raise ContractError(
                f"Keep artifacts.{name} under {boundary.as_posix()}/."
            ) from exc
    specification_root = Path("data/specification")
    for name in ("workflows", "tools"):
        for value in artifacts[name]:
            try:
                Path(value).relative_to(specification_root)
            except ValueError as exc:
                raise ContractError(
                    f"Keep artifacts.{name} under data/specification/."
                ) from exc
    if artifacts["runtime_source"] != f"src/{package}":
        raise ContractError("Set artifacts.runtime_source to the application package.")
    if artifacts["evaluation_results"] == artifacts["releases"]:
        raise ContractError("Use separate immutable evaluation and release directories.")
    return artifacts


def _validate_acceptance(value: Any) -> Dict[str, Any]:
    acceptance = _object(value, "acceptance")
    _exact_keys(acceptance, {"criteria", "metrics"}, "acceptance")
    criteria = _list(acceptance["criteria"], "acceptance.criteria")
    for position, criterion in enumerate(criteria):
        _non_empty_string(criterion, f"acceptance.criteria[{position}]")
    metrics = _list(acceptance["metrics"], "acceptance.metrics")
    metric_names = []
    for position, item in enumerate(metrics):
        metric = _object(item, f"acceptance.metrics[{position}]")
        name = _identifier(metric.get("name"), f"acceptance.metrics[{position}].name")
        if metric.get("operator") not in {">", ">=", "<", "<=", "=="}:
            raise ContractError(
                f"Set acceptance.metrics[{position}].operator to a supported comparator."
            )
        if not isinstance(metric.get("threshold"), (int, float)) or not math.isfinite(
            metric["threshold"]
        ):
            raise ContractError(
                f"Set acceptance.metrics[{position}].threshold to a number."
            )
        metric_names.append(name)
    if len(metric_names) != len(set(metric_names)):
        raise ContractError("Assign a unique name to every acceptance metric.")
    return acceptance


def load_contract() -> Dict[str, Any]:
    try:
        payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(
            "Create data/specification/application.json from BUSINESS_BRIEF.md."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ContractError(
            "Write valid JSON to data/specification/application.json: "
            f"line {exc.lineno}, column {exc.colno}."
        ) from exc

    contract = _object(payload, "application contract")
    validate_schema(contract, "application.schema.json", "Application contract")
    _exact_keys(
        contract,
        {
            "schema_version",
            "status",
            "application_id",
            "package",
            "name",
            "version",
            "business",
            "interfaces",
            "artifacts",
            "acceptance",
        },
        "application contract",
    )
    if contract["schema_version"] != "miniagent.application.v1":
        raise ContractError(
            "Set schema_version to miniagent.application.v1."
        )
    if contract["status"] not in {"draft", "approved"}:
        raise ContractError("Set status to draft or approved.")
    _identifier(contract["application_id"], "application_id")
    package = _non_empty_string(contract["package"], "package")
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", package) is None:
        raise ContractError("Set package to a Python import package name.")
    _non_empty_string(contract["name"], "name")
    _non_empty_string(contract["version"], "version")
    _validate_business(contract["business"])
    interfaces = _list(contract["interfaces"], "interfaces")
    if not interfaces or any(
        interface not in {"api", "cli", "event", "library"}
        for interface in interfaces
    ):
        raise ContractError("Set interfaces to supported application interfaces.")
    _validate_artifacts(contract["artifacts"], package)
    _validate_acceptance(contract["acceptance"])
    return contract


def _json_documents(relative_paths: Iterable[str], artifact_name: str) -> List[Path]:
    documents: List[Path] = []
    for relative in relative_paths:
        path = APPLICATION_ROOT / _relative_path(relative, artifact_name)
        try:
            path.resolve().relative_to(APPLICATION_ROOT.resolve())
        except ValueError as exc:
            raise ContractError(
                f"Keep declared {artifact_name} inside the application directory."
            ) from exc
        if path.is_symlink():
            raise ContractError(f"Use regular application-owned {artifact_name} paths.")
        if path.is_file() and path.suffix == ".json":
            documents.append(path)
        elif path.is_dir():
            candidates = sorted(path.glob("*.json"))
            if any(candidate.is_symlink() for candidate in candidates):
                raise ContractError(
                    f"Use regular application-owned files for {artifact_name}."
                )
            documents.extend(candidates)
        else:
            raise ContractError(f"Create the declared {artifact_name} path: {relative}.")
    if not documents:
        raise ContractError(f"Add at least one JSON document for {artifact_name}.")
    return documents


def declared_json_documents(payload: Dict[str, Any], artifact_name: str) -> List[Path]:
    return _json_documents(payload["artifacts"][artifact_name], artifact_name)


def _load_documents(
    relative_paths: Iterable[str], artifact_name: str, schema_name: str
) -> List[Tuple[Path, Dict[str, Any]]]:
    result = []
    for document in _json_documents(relative_paths, artifact_name):
        try:
            payload = json.loads(document.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ContractError(
                f"Write valid JSON to {document.relative_to(APPLICATION_ROOT)}: "
                f"line {exc.lineno}, column {exc.colno}."
            ) from exc
        item = _object(payload, str(document.relative_to(APPLICATION_ROOT)))
        validate_schema(item, schema_name, str(document.relative_to(APPLICATION_ROOT)))
        result.append((document, item))
    return result


def _unique_ids(records: Iterable[Dict[str, Any]], field: str, label: str) -> set[str]:
    values = [str(record[field]) for record in records]
    if len(values) != len(set(values)):
        raise ContractError(f"Assign unique {field} values in {label}.")
    return set(values)


def _validate_tool_references(
    tools: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    tool_ids = _unique_ids(tools, "tool_id", "tool contracts")
    tools_by_id = {str(tool["tool_id"]): tool for tool in tools}
    for tool in tools:
        tool_id = str(tool["tool_id"])
        arguments = tool["arguments"]
        properties = set(arguments["properties"])
        required = set(arguments.get("required", []))
        unknown_required = sorted(required.difference(properties))
        if unknown_required:
            raise ContractError(
                f"Tool {tool_id} requires undeclared arguments: "
                + ", ".join(unknown_required)
                + "."
            )
        normalized = set(tool["normalization"])
        if normalized != properties:
            missing = sorted(properties.difference(normalized))
            extra = sorted(normalized.difference(properties))
            raise ContractError(
                f"Tool {tool_id} normalization must cover exactly its arguments; "
                f"missing={missing}, extra={extra}."
            )
        for argument, operations in tool["normalization"].items():
            argument_schema = arguments["properties"][argument]
            argument_types = _schema_types(argument_schema)
            string_operations = {
                "unicode_nfkc",
                "trim",
                "lowercase",
                "uppercase",
            }.intersection(operations)
            if string_operations and "string" not in argument_types:
                raise ContractError(
                    f"Tool {tool_id} argument {argument} uses string normalization "
                    "with a non-string schema."
                )
            if "sort_unique" in operations and "array" not in argument_types:
                raise ContractError(
                    f"Tool {tool_id} argument {argument} uses sort_unique with a "
                    "non-array schema."
                )
            if {"lowercase", "uppercase"}.issubset(operations):
                raise ContractError(
                    f"Tool {tool_id} argument {argument} chooses one case normalization."
                )
        error_codes = [item["code"] for item in tool["outcomes"]["error"]["codes"]]
        if len(error_codes) != len(set(error_codes)):
            raise ContractError(f"Assign unique error codes for tool {tool_id}.")
        for prerequisite in tool["execution"]["prerequisites"]:
            if prerequisite["kind"] == "tool_result" and prerequisite["id"] not in tool_ids:
                raise ContractError(
                    f"Tool {tool_id} requires unknown tool result {prerequisite['id']}."
                )
            if prerequisite["kind"] == "tool_result" and prerequisite["id"] == tool_id:
                raise ContractError(f"Tool {tool_id} cannot require its own result.")
        compatible = set(tool["batching"]["compatible_tools"])
        unknown = sorted(compatible.difference(tool_ids))
        if unknown:
            raise ContractError(
                f"Tool {tool_id} names unknown compatible tools: {', '.join(unknown)}."
            )
        if tool_id in compatible:
            raise ContractError(f"Tool {tool_id} cannot batch with itself.")
        if tool["batching"]["allowed"] is not bool(compatible):
            raise ContractError(
                f"Tool {tool_id} batching.allowed must match whether compatible_tools "
                "contains tools."
            )
        strategy = tool["idempotency"]["strategy"]
        key_arguments = set(tool["idempotency"]["key_arguments"])
        if not key_arguments.issubset(properties):
            raise ContractError(
                f"Tool {tool_id} idempotency keys must be declared arguments."
            )
        if strategy == "none" and key_arguments:
            raise ContractError(
                f"Tool {tool_id} with idempotency strategy none has no key arguments."
            )
        if strategy == "normalized_arguments" and key_arguments != properties:
            raise ContractError(
                f"Tool {tool_id} normalized_arguments idempotency must use every argument."
            )
        if strategy == "explicit_key" and (
            not key_arguments or not key_arguments.issubset(required)
        ):
            raise ContractError(
                f"Tool {tool_id} explicit idempotency keys must be required arguments."
            )
    for tool_id, tool in tools_by_id.items():
        for peer_id in tool["batching"]["compatible_tools"]:
            peer = tools_by_id[peer_id]
            if not peer["batching"]["allowed"] or tool_id not in peer["batching"]["compatible_tools"]:
                raise ContractError(
                    f"Declare batching compatibility symmetrically for {tool_id} and {peer_id}."
                )
    return tools_by_id


def _binding_dependencies(
    bindings: Mapping[str, str],
    workflow_id: str,
    state_id: str,
    slots: Set[str],
    tools: Mapping[str, Dict[str, Any]],
) -> Tuple[Set[str], Set[str]]:
    slot_dependencies: Set[str] = set()
    tool_dependencies: Set[str] = set()
    for expression in bindings.values():
        if expression.startswith("slot."):
            slot_id = expression[5:]
            if slot_id not in slots:
                raise ContractError(
                    f"Workflow {workflow_id} state {state_id} binds unknown slot {slot_id}."
                )
            slot_dependencies.add(slot_id)
            continue
        matches = [
            tool_id
            for tool_id in tools
            if expression.startswith(f"tool.{tool_id}.")
        ]
        if not matches:
            raise ContractError(
                f"Workflow {workflow_id} state {state_id} has invalid binding {expression}."
            )
        producer_id = max(matches, key=len)
        field_path = expression[len(f"tool.{producer_id}.") :].split(".")
        schema = tools[producer_id]["outcomes"]["ok"]["data_schema"]
        for field in field_path:
            properties = schema.get("properties", {})
            if field not in properties:
                raise ContractError(
                    f"Workflow {workflow_id} state {state_id} binds unknown result field "
                    f"{expression}."
                )
            schema = properties[field]
        tool_dependencies.add(producer_id)
    return slot_dependencies, tool_dependencies


def _binding_schema(
    expression: str,
    slots: Mapping[str, Dict[str, Any]],
    tools: Mapping[str, Dict[str, Any]],
) -> Dict[str, Any]:
    if expression.startswith("slot."):
        return slots[expression[5:]]["schema"]
    producer_id = max(
        (
            tool_id
            for tool_id in tools
            if expression.startswith(f"tool.{tool_id}.")
        ),
        key=len,
    )
    schema = tools[producer_id]["outcomes"]["ok"]["data_schema"]
    for field in expression[len(f"tool.{producer_id}.") :].split("."):
        schema = schema["properties"][field]
    return schema


def _schema_types(schema: Mapping[str, Any]) -> Set[str]:
    value = schema.get("type")
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return {str(item) for item in value}
    return {"null", "boolean", "object", "array", "number", "integer", "string"}


def _schemas_are_assignable(
    source: Mapping[str, Any], target: Mapping[str, Any]
) -> bool:
    source_types = _schema_types(source)
    target_types = _schema_types(target)
    if "integer" in source_types and "number" in target_types:
        source_types = (source_types - {"integer"}) | {"number"}
    if not source_types.issubset(target_types):
        return False
    if "format" in target and source.get("format") != target["format"]:
        return False
    if "enum" in target:
        source_values = (
            set(source["enum"])
            if "enum" in source
            else {source["const"]}
            if "const" in source
            else None
        )
        if source_values is None or not source_values.issubset(set(target["enum"])):
            return False
    return True


def _validate_call(
    tool_id: str,
    bindings: Mapping[str, str],
    workflow_id: str,
    state_id: str,
    slots: Set[str],
    tools: Mapping[str, Dict[str, Any]],
    slot_records: Mapping[str, Dict[str, Any]] | None = None,
) -> Tuple[Set[str], Set[str]]:
    tool = tools.get(tool_id)
    if tool is None:
        raise ContractError(
            f"Workflow {workflow_id} state {state_id} calls unknown tool {tool_id}."
        )
    properties = set(tool["arguments"]["properties"])
    required = set(tool["arguments"].get("required", []))
    supplied = set(bindings)
    if not required.issubset(supplied) or not supplied.issubset(properties):
        raise ContractError(
            f"Workflow {workflow_id} state {state_id} must bind required arguments and "
            f"only declared arguments for tool {tool_id}."
        )
    slot_dependencies, tool_dependencies = _binding_dependencies(
        bindings, workflow_id, state_id, slots, tools
    )
    if slot_records is not None:
        for argument, expression in bindings.items():
            source_schema = _binding_schema(expression, slot_records, tools)
            target_schema = tool["arguments"]["properties"][argument]
            if not _schemas_are_assignable(source_schema, target_schema):
                raise ContractError(
                    f"Workflow {workflow_id} state {state_id} binds an incompatible "
                    f"value to {tool_id}.{argument}."
                )
    prerequisites = tool["execution"]["prerequisites"]
    required_slots = {item["id"] for item in prerequisites if item["kind"] == "slot"}
    required_tools = {
        item["id"] for item in prerequisites if item["kind"] == "tool_result"
    }
    unknown_slots = sorted(required_slots.difference(slots))
    if unknown_slots:
        raise ContractError(
            f"Workflow {workflow_id} tool {tool_id} requires unknown slots: "
            + ", ".join(unknown_slots)
            + "."
        )
    if not required_slots.issubset(slot_dependencies) or not required_tools.issubset(
        tool_dependencies
    ):
        raise ContractError(
            f"Workflow {workflow_id} state {state_id} must bind every prerequisite for "
            f"tool {tool_id}."
        )
    return slot_dependencies, tool_dependencies


def _validate_workflow_semantics(
    workflow: Dict[str, Any], available_tools: Mapping[str, Dict[str, Any]]
) -> None:
    workflow_id = str(workflow["workflow_id"])
    slots = _unique_ids(workflow["slots"], "name", f"workflow {workflow_id} slots")
    slot_records = {str(slot["name"]): slot for slot in workflow["slots"]}
    states = {str(state["id"]): state for state in workflow["states"]}
    if len(states) != len(workflow["states"]):
        raise ContractError(f"Assign unique state ids in workflow {workflow_id}.")
    initial = str(workflow["initial_state"])
    if initial not in states or states[initial]["kind"] != "start":
        raise ContractError(f"Workflow {workflow_id} initial_state must name a start state.")

    for state_id, state in states.items():
        if state["kind"] == "collect" and state["slot"] not in slots:
            raise ContractError(
                f"Workflow {workflow_id} state {state_id} collects unknown slot {state['slot']}."
            )
        if state["kind"] == "tool":
            _validate_call(
                state["tool"],
                state["argument_bindings"],
                workflow_id,
                state_id,
                slots,
                available_tools,
                slot_records,
            )
        if state["kind"] == "tool_batch":
            call_ids = [str(call["tool"]) for call in state["calls"]]
            if len(call_ids) != len(set(call_ids)):
                raise ContractError(
                    f"Workflow {workflow_id} state {state_id} batches each tool once."
                )
            for call in state["calls"]:
                _validate_call(
                    call["tool"],
                    call["argument_bindings"],
                    workflow_id,
                    state_id,
                    slots,
                    available_tools,
                    slot_records,
                )
            for tool_id in call_ids:
                tool = available_tools.get(tool_id)
                if tool is None:
                    continue
                peers = set(call_ids).difference({tool_id})
                compatible = set(tool["batching"]["compatible_tools"])
                if not tool["batching"]["allowed"] or not peers.issubset(compatible):
                    raise ContractError(
                        f"Workflow {workflow_id} state {state_id} contains an incompatible "
                        f"tool batch."
                    )
                prerequisites = {
                    item["id"]
                    for item in tool["execution"]["prerequisites"]
                    if item["kind"] == "tool_result"
                }
                if prerequisites.intersection(call_ids):
                    raise ContractError(
                        f"Workflow {workflow_id} state {state_id} batches dependent tools."
                    )

    transitions = {
        str(transition["id"]): transition for transition in workflow["transitions"]
    }
    if len(transitions) != len(workflow["transitions"]):
        raise ContractError(f"Assign unique transition ids in workflow {workflow_id}.")
    outgoing: Dict[str, List[str]] = {state_id: [] for state_id in states}
    outgoing_transitions: Dict[str, List[Dict[str, Any]]] = {
        state_id: [] for state_id in states
    }
    for transition_id, transition in transitions.items():
        source = str(transition["from"])
        target = str(transition["to"])
        if source not in states or target not in states:
            raise ContractError(
                f"Workflow {workflow_id} transition {transition_id} references an unknown state."
            )
        outgoing[source].append(target)
        outgoing_transitions[source].append(transition)
        condition = transition["when"]
        if condition["event"] == "tool_result" and condition["tool"] not in available_tools:
            raise ContractError(
                f"Workflow {workflow_id} transition {transition_id} references unknown tool "
                f"{condition['tool']}."
            )
        if condition["event"] == "user_response":
            unknown_slots = sorted(set(condition["slots_present"]).difference(slots))
            if unknown_slots:
                raise ContractError(
                    f"Workflow {workflow_id} transition {transition_id} references unknown "
                    f"slots: {', '.join(unknown_slots)}."
                )

    for state_id, state in states.items():
        branches = outgoing_transitions[state_id]
        kind = state["kind"]
        if kind == "terminal":
            continue
        if not branches:
            raise ContractError(
                f"Workflow {workflow_id} non-terminal state {state_id} has no transition."
            )
        events = [branch["when"]["event"] for branch in branches]
        expected_event = {
            "start": "always",
            "collect": "user_response",
            "tool": "tool_result",
            "tool_batch": "tool_batch_result",
            "decision": "condition",
            "respond": "always",
            "refuse": "always",
        }[kind]
        if any(event != expected_event for event in events):
            raise ContractError(
                f"Workflow {workflow_id} state {state_id} must branch on {expected_event}."
            )
        if kind in {"start", "respond", "refuse"} and len(branches) != 1:
            raise ContractError(
                f"Workflow {workflow_id} state {state_id} must have one unconditional transition."
            )
        if kind == "collect":
            if any(state["slot"] not in branch["when"]["slots_present"] for branch in branches):
                raise ContractError(
                    f"Workflow {workflow_id} state {state_id} must collect {state['slot']}."
                )
        if kind == "tool":
            branch_keys = [
                (branch["when"]["tool"], branch["when"]["status"])
                for branch in branches
            ]
            expected_keys = {(state["tool"], status) for status in ("ok", "empty", "error")}
            if set(branch_keys) != expected_keys or len(branch_keys) != len(expected_keys):
                raise ContractError(
                    f"Workflow {workflow_id} state {state_id} must branch once on each "
                    "ok, empty, and error tool result."
                )
        if kind == "tool_batch":
            call_ids = tuple(str(call["tool"]) for call in state["calls"])
            observed = []
            for branch in branches:
                statuses = branch["when"]["statuses"]
                if set(statuses) != set(call_ids):
                    raise ContractError(
                        f"Workflow {workflow_id} state {state_id} batch results must name "
                        "every called tool."
                    )
                observed.append(tuple(statuses[tool_id] for tool_id in call_ids))
            expected = set(product(("ok", "empty", "error"), repeat=len(call_ids)))
            if set(observed) != expected or len(observed) != len(expected):
                raise ContractError(
                    f"Workflow {workflow_id} state {state_id} must branch once on every "
                    "batch result combination."
                )
        if kind == "decision":
            expressions = [branch["when"]["expression"] for branch in branches]
            if len(expressions) != len(set(expressions)):
                raise ContractError(
                    f"Workflow {workflow_id} state {state_id} has duplicate conditions."
                )
        if kind == "refuse":
            target = states[str(branches[0]["to"])]
            if target["kind"] != "terminal" or target["outcome"] != "refused":
                raise ContractError(
                    f"Workflow {workflow_id} refusal state {state_id} must end in refusal."
                )
        if kind == "respond" and states[str(branches[0]["to"])]["kind"] != "terminal":
            raise ContractError(
                f"Workflow {workflow_id} response state {state_id} must end the workflow."
            )

    reachable = {initial}
    frontier = [initial]
    while frontier:
        source = frontier.pop()
        for target in outgoing[source]:
            if target not in reachable:
                reachable.add(target)
                frontier.append(target)
    unreachable = sorted(set(states).difference(reachable))
    if unreachable:
        raise ContractError(
            f"Workflow {workflow_id} has unreachable states: {', '.join(unreachable)}."
        )
    terminal_with_edges = sorted(
        state_id
        for state_id, state in states.items()
        if state["kind"] == "terminal" and outgoing[state_id]
    )
    if terminal_with_edges:
        raise ContractError(
            f"Workflow {workflow_id} terminal states have outgoing transitions: "
            + ", ".join(terminal_with_edges)
            + "."
        )

    route_ids = set()
    covered = set()
    for route in workflow["routes"]:
        route_id = str(route["id"])
        if route_id in route_ids:
            raise ContractError(f"Assign unique route ids in workflow {workflow_id}.")
        route_ids.add(route_id)
        current = initial
        initial_slots = set(route["initial_slots"])
        unknown_initial_slots = sorted(initial_slots.difference(slots))
        invalid_initial_slots = sorted(
            slot_id
            for slot_id in initial_slots.intersection(slots)
            if slot_records[slot_id]["source"] != "user"
        )
        if unknown_initial_slots or invalid_initial_slots:
            raise ContractError(
                f"Workflow {workflow_id} route {route_id} initial_slots must name "
                f"declared user slots; unknown={unknown_initial_slots}, "
                f"invalid={invalid_initial_slots}."
            )
        collected_user_slots: Set[str] = set(initial_slots)
        completed_tools: Set[str] = set()
        for transition_id in route["transition_ids"]:
            transition = transitions.get(str(transition_id))
            if transition is None:
                raise ContractError(
                    f"Workflow {workflow_id} route {route_id} names unknown transition "
                    f"{transition_id}."
                )
            if transition["from"] != current:
                raise ContractError(
                    f"Workflow {workflow_id} route {route_id} is discontinuous at "
                    f"{transition_id}."
                )
            state = states[current]
            if state["kind"] == "collect":
                if str(state["slot"]) in collected_user_slots:
                    raise ContractError(
                        f"Workflow {workflow_id} route {route_id} collects already "
                        f"available slot {state['slot']}."
                    )
                collected_user_slots.add(str(state["slot"]))
            calls = []
            if state["kind"] == "tool":
                calls = [state]
            elif state["kind"] == "tool_batch":
                calls = state["calls"]
            for call in calls:
                slot_dependencies, tool_dependencies = _binding_dependencies(
                    call["argument_bindings"],
                    workflow_id,
                    current,
                    slots,
                    available_tools,
                )
                unavailable_user_slots = {
                    slot_id
                    for slot_id in slot_dependencies
                    if slot_records[slot_id]["source"] == "user"
                    and slot_id not in collected_user_slots
                }
                if unavailable_user_slots or not tool_dependencies.issubset(completed_tools):
                    raise ContractError(
                        f"Workflow {workflow_id} route {route_id} calls {call['tool']} before "
                        "its bound values are available."
                    )
            completed_tools.update(str(call["tool"]) for call in calls)
            current = str(transition["to"])
            covered.add(str(transition_id))
        terminal = str(route["terminal_state"])
        if current != terminal or states[current]["kind"] != "terminal":
            raise ContractError(
                f"Workflow {workflow_id} route {route_id} must end at its terminal state."
            )
    uncovered = sorted(set(transitions).difference(covered))
    if uncovered:
        raise ContractError(
            f"Workflow {workflow_id} has transitions absent from its routes: "
            + ", ".join(uncovered)
            + "."
        )

    expected_routes = set()

    def enumerate_paths(state_id: str, used: set[str], path: List[str]) -> None:
        if states[state_id]["kind"] == "terminal":
            expected_routes.add(tuple(path))
            return
        available = [
            transition
            for transition in outgoing_transitions[state_id]
            if str(transition["id"]) not in used
        ]
        if not available:
            raise ContractError(
                f"Workflow {workflow_id} can stop at non-terminal state {state_id} "
                "under its transition-visit bound."
            )
        for transition in available:
            transition_id = str(transition["id"])
            enumerate_paths(
                str(transition["to"]),
                {*used, transition_id},
                [*path, transition_id],
            )

    enumerate_paths(initial, set(), [])
    declared_routes = {
        tuple(str(item) for item in route["transition_ids"])
        for route in workflow["routes"]
    }
    if declared_routes != expected_routes:
        missing = len(expected_routes.difference(declared_routes))
        extra = len(declared_routes.difference(expected_routes))
        raise ContractError(
            f"Workflow {workflow_id} route table differs from the graph: "
            f"{missing} missing and {extra} extra route(s)."
        )


def require_approved(payload: Dict[str, Any]) -> None:
    if payload["status"] != "approved":
        raise ContractError(
            "Review the compiled policy and set the application contract status to approved."
        )

    business = payload["business"]
    for field in ("users", "supported_requests", "rules"):
        if not business[field]:
            raise ContractError(f"Add at least one business.{field} entry before approval.")
    for field in ("criteria", "metrics"):
        if not payload["acceptance"][field]:
            raise ContractError(f"Add acceptance.{field} before approval.")

    artifacts = payload["artifacts"]
    tool_documents = _load_documents(
        artifacts["tools"], "tools", "tool.schema.json"
    )
    workflow_documents = _load_documents(
        artifacts["workflows"], "workflows", "workflow.schema.json"
    )
    application_id = payload["application_id"]
    for document, item in [*tool_documents, *workflow_documents]:
        if item["application_id"] != application_id:
            raise ContractError(
                f"Set {document.relative_to(APPLICATION_ROOT)} application_id to "
                f"{application_id}."
            )
    tools = [item for _, item in tool_documents]
    tools_by_id = _validate_tool_references(tools)
    workflows = [item for _, item in workflow_documents]
    _unique_ids(workflows, "workflow_id", "workflow contracts")
    route_ids = [str(route["id"]) for workflow in workflows for route in workflow["routes"]]
    if len(route_ids) != len(set(route_ids)):
        raise ContractError("Assign globally unique route ids across workflow contracts.")
    supported_request_ids = {
        str(item["id"]) for item in business["supported_requests"]
    }
    rule_ids = {str(item["id"]) for item in business["rules"]}
    covered_requests: Set[str] = set()
    covered_rules: Set[str] = set()
    for workflow in workflows:
        workflow_requests = set(workflow["supported_request_ids"])
        workflow_rules = set(workflow["rule_ids"])
        unknown_requests = sorted(workflow_requests.difference(supported_request_ids))
        unknown_rules = sorted(workflow_rules.difference(rule_ids))
        if unknown_requests or unknown_rules:
            raise ContractError(
                f"Workflow {workflow['workflow_id']} references unknown business ids: "
                f"requests={unknown_requests}, rules={unknown_rules}."
            )
        covered_requests.update(workflow_requests)
        covered_rules.update(workflow_rules)
        _validate_workflow_semantics(workflow, tools_by_id)
    if covered_requests != supported_request_ids:
        raise ContractError("Cover every supported business request with a workflow.")
    if covered_rules != rule_ids:
        raise ContractError("Bind every business rule to at least one workflow.")


def validate_acceptance_metrics(
    contract: Dict[str, Any], evaluation: Dict[str, Any]
) -> None:
    observed = {}
    for metric in evaluation["metrics"]:
        name = metric["name"]
        if name in observed:
            raise ContractError(f"Evaluation metric names must be unique: {name}.")
        if not math.isfinite(metric["value"]):
            raise ContractError(f"Evaluation metric {name} must be finite.")
        observed[name] = metric

    required_results = []
    comparisons = {
        ">": lambda value, threshold: value > threshold,
        ">=": lambda value, threshold: value >= threshold,
        "<": lambda value, threshold: value < threshold,
        "<=": lambda value, threshold: value <= threshold,
        "==": lambda value, threshold: value == threshold,
    }
    for requirement in contract["acceptance"]["metrics"]:
        name = requirement["name"]
        metric = observed.get(name)
        if metric is None:
            raise ContractError(f"Evaluation is missing required metric {name}.")
        expected_unit = requirement.get("unit")
        if expected_unit is not None and metric["unit"] != expected_unit:
            raise ContractError(f"Evaluation metric {name} uses a different unit.")
        threshold = requirement["threshold"]
        if not math.isfinite(threshold):
            raise ContractError(f"Acceptance metric {name} threshold must be finite.")
        passed = comparisons[requirement["operator"]](metric["value"], threshold)
        if metric.get("threshold") != threshold:
            raise ContractError(f"Evaluation metric {name} records a different threshold.")
        if metric.get("passed") is not passed:
            raise ContractError(f"Evaluation metric {name} records an incorrect pass result.")
        expected_direction = (
            "maximize"
            if requirement["operator"] in {">", ">="}
            else "minimize"
            if requirement["operator"] in {"<", "<="}
            else metric["direction"]
        )
        if metric["direction"] != expected_direction:
            raise ContractError(f"Evaluation metric {name} records an incorrect direction.")
        required_results.append(passed)

    summary = evaluation["summary"]
    complete = (
        summary["completed"] == summary["attempted"] and summary["errors"] == 0
    )
    expected_status = "passed" if complete and all(required_results) else "failed"
    if summary["status"] != expected_status:
        raise ContractError(
            f"Set evaluation summary status to {expected_status} from verified results."
        )
