"""Causal SyntheticPerson + SyntheticChina complete-episode runtime."""
from __future__ import annotations

import copy
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .actions import Ask, AssistantAction, Final, Invalid, Refusal, ToolCall, ToolCalls
from .interpreter import AssistantActionInterpreter
from .world import SyntheticChinaWorld, WorldExecution, WorldSnapshotError, normalize_json
from training.paths import DATA_ROOT


class EpisodeSpecError(ValueError):
    pass


class EpisodeTerminatedError(RuntimeError):
    pass


def _action_from_dict(raw: Mapping[str, Any]) -> AssistantAction:
    kind = str(raw.get("type") or "").lower()
    if kind == "ask":
        slot = raw.get("slot")
        if not isinstance(slot, str) or not slot:
            raise EpisodeSpecError("Ask action requires a slot")
        return Ask(slot=slot, text=str(raw.get("text") or ""))
    if kind in {"toolcalls", "tool_calls", "tools"}:
        items = raw.get("calls")
        if not isinstance(items, list) or not items:
            raise EpisodeSpecError("ToolCalls action requires a non-empty calls list")
        calls: list[ToolCall] = []
        for item in items:
            if not isinstance(item, Mapping):
                raise EpisodeSpecError("expected tool call must be an object")
            name = item.get("name", item.get("tool"))
            args = item.get("arguments", item.get("args"))
            if not isinstance(name, str) or not isinstance(args, Mapping):
                raise EpisodeSpecError("expected tool call requires name and arguments")
            calls.append(ToolCall(name, normalize_json(dict(args))))
        return ToolCalls(tuple(calls))
    if kind == "final":
        return Final(str(raw.get("text") or ""))
    if kind == "refusal":
        return Refusal(str(raw.get("text") or ""))
    raise EpisodeSpecError(f"unknown expected action type {raw.get('type')!r}")


@dataclass(frozen=True)
class StateNode:
    state_id: str
    expected_action: AssistantAction
    next_state: str | None = None
    user_response: str | None = None
    result_branches: Mapping[str, str] = field(default_factory=dict)
    expected_result_status: str | None = None
    milestone_ids: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StateNode":
        state_id = raw.get("state_id")
        expected = raw.get("expected_action")
        if not isinstance(state_id, str) or not state_id:
            raise EpisodeSpecError("state node lacks state_id")
        if not isinstance(expected, Mapping):
            raise EpisodeSpecError(f"state {state_id}: expected_action must be an object")
        branches = raw.get("result_branches") or {}
        if not isinstance(branches, Mapping) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in branches.items()
        ):
            raise EpisodeSpecError(f"state {state_id}: invalid result_branches")
        milestone_ids = raw.get("milestone_ids") or {}
        if not isinstance(milestone_ids, Mapping):
            raise EpisodeSpecError(f"state {state_id}: milestone_ids must be an object")
        nxt = raw.get("next_state")
        return cls(
            state_id=state_id,
            expected_action=_action_from_dict(expected),
            next_state=str(nxt) if nxt is not None else None,
            user_response=(str(raw["user_response"])
                           if raw.get("user_response") is not None else None),
            result_branches=dict(branches),
            expected_result_status=(str(raw["expected_result_status"])
                                    if raw.get("expected_result_status") is not None else None),
            milestone_ids=copy.deepcopy(dict(milestone_ids)),
            raw=copy.deepcopy(dict(raw)),
        )


@dataclass(frozen=True)
class EpisodeSpec:
    schema_version: str
    episode_id: str
    source_idx: int | None
    metadata: Mapping[str, Any]
    world_id: str
    initial_messages: tuple[Mapping[str, Any], ...]
    person_state: Mapping[str, Any]
    response_policy: Mapping[str, Any]
    initial_state_id: str
    state_graph: tuple[StateNode, ...]
    reward_milestones: tuple[str, ...]
    source: Mapping[str, Any]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EpisodeSpec":
        episode_id = raw.get("episode_id")
        initial_state = raw.get("initial_state_id")
        messages = raw.get("initial_messages")
        graph = raw.get("state_graph")
        if not isinstance(episode_id, (str, int)):
            raise EpisodeSpecError("episode_id is required")
        if not isinstance(initial_state, str) or not initial_state:
            raise EpisodeSpecError(f"episode {episode_id}: initial_state_id is required")
        if not isinstance(messages, list) or not messages:
            raise EpisodeSpecError(f"episode {episode_id}: initial_messages must be non-empty")
        if not all(isinstance(item, Mapping) for item in messages):
            raise EpisodeSpecError(f"episode {episode_id}: invalid initial message")
        if isinstance(graph, Mapping):
            graph = [dict(node, state_id=node.get("state_id", key))
                     for key, node in graph.items() if isinstance(node, Mapping)]
        if not isinstance(graph, list) or not graph:
            raise EpisodeSpecError(f"episode {episode_id}: state_graph must be non-empty")
        nodes = tuple(StateNode.from_dict(item) for item in graph)
        node_ids = [node.state_id for node in nodes]
        if len(node_ids) != len(set(node_ids)):
            raise EpisodeSpecError(f"episode {episode_id}: duplicate state_id")
        if initial_state not in set(node_ids):
            raise EpisodeSpecError(f"episode {episode_id}: unknown initial state {initial_state!r}")
        source_idx = raw.get("source_idx")
        if source_idx is None and isinstance(raw.get("metadata"), Mapping):
            source_idx = raw["metadata"].get("idx")
        return cls(
            schema_version=str(raw.get("schema_version") or "cn_travel.world_policy.v1"),
            episode_id=str(episode_id),
            source_idx=int(source_idx) if source_idx is not None else None,
            metadata=copy.deepcopy(dict(raw.get("metadata") or {})),
            world_id=str(raw.get("world_id") or ""),
            initial_messages=tuple(copy.deepcopy(dict(item)) for item in messages),
            person_state=copy.deepcopy(dict(raw.get("person_state") or {})),
            response_policy=copy.deepcopy(dict(raw.get("response_policy") or {})),
            initial_state_id=initial_state,
            state_graph=nodes,
            reward_milestones=tuple(str(x) for x in raw.get("reward_milestones") or ()),
            source=copy.deepcopy(dict(raw.get("source") or {})),
        )

    @property
    def nodes(self) -> dict[str, StateNode]:
        return {node.state_id: node for node in self.state_graph}


@dataclass(frozen=True)
class RewardConfig:
    """Terminal-success-gated trajectory reward configuration."""

    policy: str = "terminal_success_gated_v1"
    maximum: float = 1.5
    success_floor: float = 0.5
    penalty_per_violation: float = 0.2

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "RewardConfig":
        if not values:
            return cls()
        fields = cls.__dataclass_fields__
        unknown = sorted(set(values) - set(fields))
        if unknown:
            raise ValueError(f"unknown RUN #2 reward keys: {unknown}")
        picked: dict[str, Any] = {}
        for key, value in values.items():
            picked[key] = str(value) if key == "policy" else float(value)
        config = cls(**picked)
        if config.policy != "terminal_success_gated_v1":
            raise ValueError(f"unsupported reward.policy: {config.policy!r}")
        if not all(math.isfinite(value) for value in (
            config.maximum,
            config.success_floor,
            config.penalty_per_violation,
        )):
            raise ValueError("RUN #2 reward values must be finite")
        if config.maximum <= 0:
            raise ValueError("reward.maximum must be positive")
        if not 0 <= config.success_floor <= config.maximum:
            raise ValueError("reward.success_floor must be in [0, maximum]")
        if config.penalty_per_violation < 0:
            raise ValueError("reward.penalty_per_violation must be non-negative")
        return config


@dataclass(frozen=True)
class RuntimeConfig:
    reward: RewardConfig = field(default_factory=RewardConfig)
    max_rounds_per_user_turn: int = 6
    max_assistant_rounds: int = 24
    max_person_turns: int = 8

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "RuntimeConfig":
        if not values:
            return cls()
        reward_values = values.get("reward") if isinstance(values.get("reward"), Mapping) else None
        return cls(
            reward=RewardConfig.from_mapping(reward_values),
            max_rounds_per_user_turn=int(values.get("max_rounds_per_user_turn", 6)),
            max_assistant_rounds=int(values.get("max_assistant_rounds",
                                                values.get("max_assistant_turns", 24))),
            max_person_turns=int(values.get("max_person_turns",
                                            values.get("max_user_turns", 8))),
        )


@dataclass
class PersonState:
    episode_id: str
    public_context: dict[str, Any]
    private_slots: dict[str, Any]
    initially_hidden: set[str]
    revealed_slots: set[str]
    goal: dict[str, Any]
    current_state_id: str
    expected_slot: str | None = None
    expected_response: str | None = None
    node_policy: dict[str, Any] = field(default_factory=dict)
    response_history: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_episode(cls, spec: EpisodeSpec) -> "PersonState":
        raw = spec.person_state
        return cls(
            episode_id=spec.episode_id,
            public_context=copy.deepcopy(dict(raw.get("public_context") or {})),
            private_slots=copy.deepcopy(dict(raw.get("private_slots") or {})),
            initially_hidden={str(x) for x in raw.get("initially_hidden") or ()},
            revealed_slots={str(x) for x in raw.get("revealed_slots") or ()},
            goal=copy.deepcopy(dict(raw.get("goal") or {})),
            current_state_id=spec.initial_state_id,
        )


@dataclass(frozen=True)
class PersonReply:
    content: str
    kind: str
    revealed_slots: tuple[str, ...] = ()


_SLOT_ALIASES: dict[str, frozenset[str]] = {
    "destination": frozenset({"destination", "trip_city", "trip_destination", "city",
                               "route_destination"}),
    "route_destination": frozenset({"destination", "route_destination"}),
    "travel_date": frozenset({"travel_date", "date", "start_date"}),
    "hotel_name": frozenset({"hotel_name", "hotel", "intended_hotel"}),
    "hotel_city": frozenset({"hotel_city", "hotel_location", "location", "city"}),
}


def slots_equivalent(expected: str, made: str) -> bool:
    if expected == made:
        return True
    return made in _SLOT_ALIASES.get(expected, frozenset({expected}))


class SyntheticPerson:
    """One generic deterministic user policy shared by every PersonState."""

    _CORRECTIONS = {
        "destination": "先告诉我您想去的目的地吧。",
        "route_destination": "先确定您想去哪里吧。",
        "travel_date": "先确认一下您的出行日期吧。",
        "hotel_name": "请先告诉我具体是哪家酒店。",
        "hotel_city": "请先告诉我酒店所在的城市。",
    }

    def respond(self, action: AssistantAction, state: PersonState) -> PersonReply:
        expected = state.expected_slot
        policy = state.node_policy
        if isinstance(action, Ask) and expected and slots_equivalent(expected, action.slot):
            content = state.expected_response or self._render_slot(expected, state)
            # W3-F/G intentionally pivots after Ask(hotel_name): the response
            # reveals recommendation location/preferences while still
            # concealing hotel_name.  The compiler therefore owns this list.
            if "revealed_slots" in policy:
                revealed = tuple(str(item) for item in policy.get("revealed_slots") or ())
            else:
                revealed = (expected,)
            concealed = {str(item) for item in policy.get("concealed_slots") or ()}
            revealed = tuple(item for item in revealed if item not in concealed)
            return PersonReply(content=content, kind="correct", revealed_slots=revealed)

        if isinstance(action, Ask) and self._is_revealed(action.slot, state):
            content = state.response_history.get(action.slot) or self._render_slot(action.slot, state)
            repeated = policy.get("repeated_response")
            return PersonReply(content=str(repeated or content), kind="repeated")

        if isinstance(action, Invalid) and action.reason == "multiple_questions":
            content = policy.get("multiple_question_response") or "请一次只问一个必要问题。"
            return PersonReply(str(content), "multiple")

        corrections = policy.get("correction_responses") or policy.get("wrong_responses") or {}
        if isinstance(corrections, Mapping) and isinstance(action, Ask):
            custom = corrections.get(action.slot)
            if custom:
                return PersonReply(str(custom), "wrong")
        content = policy.get("correction_response") or self._CORRECTIONS.get(
            expected or "", "这个问题现在不需要，请继续当前的旅行需求。"
        )
        return PersonReply(str(content), "wrong")

    @staticmethod
    def _is_revealed(slot: str, state: PersonState) -> bool:
        return any(slots_equivalent(revealed, slot) or slots_equivalent(slot, revealed)
                   for revealed in state.revealed_slots)

    @staticmethod
    def _render_slot(slot: str, state: PersonState) -> str:
        aliases = _SLOT_ALIASES.get(slot, frozenset({slot}))
        key = next((name for name in (slot, *sorted(aliases))
                    if name in state.private_slots), slot)
        value = state.private_slots.get(key, "")
        if slot in {"destination", "route_destination"}:
            return f"目的地是{value}。"
        if slot == "travel_date":
            return f"出行日期是{value}。"
        if slot == "hotel_name":
            return f"酒店是{value}。"
        if slot == "hotel_city":
            return f"酒店在{value}。"
        return f"{slot}是{value}。"


def _load_registry_keys() -> frozenset[str]:
    path = DATA_ROOT / "city_registry.json"
    try:
        return frozenset(json.loads(path.read_text(encoding="utf-8"))["cities"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return frozenset()


_REGISTRY_CITIES = _load_registry_keys()
_LOCATION_FIELDS = frozenset({"location", "city", "start_location", "end_location"})
_RECOVERABLE_VIOLATIONS = frozenset({
    "wrong_question",
    "repeated_question",
    "multiple_questions",
    "extra_or_wrong_tool",
    "premature_tool",
    "repeated_tool",
    "malformed_output",
})


def _reward_normalize_args(args: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in args.items():
        if (key in _LOCATION_FIELDS and isinstance(value, str) and value.endswith("市")
                and value[:-1] in _REGISTRY_CITIES):
            out[key] = value[:-1]
        else:
            out[key] = value
    return out


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_json_equal(left[key], right[key])
                                                   for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_json_equal(a, b)
                                                for a, b in zip(left, right))
    if left != right:
        return False
    return type(left) is type(right) or (
        isinstance(left, (int, float)) and not isinstance(left, bool)
        and isinstance(right, (int, float)) and not isinstance(right, bool)
    )


def strict_call_equal(expected: ToolCall, made: ToolCall) -> bool:
    if expected.name != made.name or not isinstance(made.arguments, Mapping):
        return False
    return _json_equal(_reward_normalize_args(expected.arguments),
                       _reward_normalize_args(made.arguments))


def _lcs_pairs(expected: Sequence[ToolCall], made: Sequence[ToolCall]) -> tuple[tuple[int, int], ...]:
    n, m = len(expected), len(made)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            table[i][j] = (table[i + 1][j + 1] + 1
                           if strict_call_equal(expected[i], made[j])
                           else max(table[i + 1][j], table[i][j + 1]))
    pairs: list[tuple[int, int]] = []
    i = j = 0
    while i < n and j < m:
        if (strict_call_equal(expected[i], made[j])
                and table[i][j] == table[i + 1][j + 1] + 1):
            pairs.append((i, j))
            i += 1
            j += 1
        elif table[i + 1][j] >= table[i][j + 1]:
            i += 1
        else:
            j += 1
    return tuple(pairs)


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    unclipped_total: float
    ask_score: float
    call_score: float
    structure_score: float
    goal_reached: int
    clean_trace: int
    violation_count: int
    penalties: Mapping[str, float]
    penalty_total: float
    credited_milestones: tuple[str, ...]
    required_milestones: tuple[str, ...]
    calls_correct: int
    calls_required: int

    @property
    def is_clean_pass(self) -> bool:
        return bool(self.goal_reached and self.clean_trace)


@dataclass(frozen=True)
class Transition:
    previous_state_id: str
    state_id: str
    messages: tuple[Mapping[str, Any], ...]
    tool_results: tuple[WorldExecution, ...]
    terminal: bool
    success: bool
    goal_reached: bool
    clean_trace: bool
    new_violations: tuple[str, ...]
    violations: tuple[str, ...]
    environment_loss_mask: int = 0


@dataclass(frozen=True)
class EpisodeOutcome:
    episode_id: str
    state_id: str
    terminal: bool
    success: bool
    goal_reached: bool
    clean_trace: bool
    truncated: bool
    truncation_reason: str | None
    assistant_rounds: int
    person_turns: int
    violations: tuple[str, ...]
    reward: RewardBreakdown


class EpisodeRuntime:
    """Mutable state for one rollout; the world and EpisodeSpec stay immutable."""

    def __init__(
        self,
        spec: EpisodeSpec,
        world: SyntheticChinaWorld,
        config: RuntimeConfig | Mapping[str, Any] | None = None,
        *,
        interpreter: AssistantActionInterpreter | None = None,
        person: SyntheticPerson | None = None,
    ) -> None:
        self.spec = spec
        self.world = world
        self.config = (config if isinstance(config, RuntimeConfig)
                       else RuntimeConfig.from_mapping(config))
        self.interpreter = interpreter or AssistantActionInterpreter()
        self.person = person or SyntheticPerson()
        if spec.world_id and world.world_id and spec.world_id != world.world_id:
            raise WorldSnapshotError(
                f"episode {spec.episode_id} targets {spec.world_id}, loaded {world.world_id}"
            )
        self._nodes = spec.nodes
        (self._required_path, self._required_calls, self._required_call_ids,
         self._required_asks, self._required_structure,
         self._required_completion) = self._compile_expected_path()
        self.reset()

    @property
    def initial_messages(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(dict(message)) for message in self.spec.initial_messages]

    @property
    def state_id(self) -> str:
        return self.person_state.current_state_id

    @property
    def current_node(self) -> StateNode:
        try:
            return self._nodes[self.state_id]
        except KeyError as exc:
            raise EpisodeSpecError(f"unknown runtime state {self.state_id!r}") from exc

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def required_milestones(self) -> tuple[str, ...]:
        # _compile_expected_path has already proved exact equality between this
        # declaration and the canonical world-dependent path.
        return self.spec.reward_milestones

    def reset(self) -> "EpisodeRuntime":
        self.person_state = PersonState.from_episode(self.spec)
        self._terminal = False
        self._goal_reached = False
        self._truncated = False
        self._truncation_reason: str | None = None
        self._violations: list[str] = []
        self._made_calls: list[ToolCall] = []
        self._credited_asks: set[str] = set()
        self._credited_structure: set[str] = set()
        self._credited_completion: set[str] = set()
        self._successful_tool_keys: set[str] = set()
        self.assistant_rounds = 0
        self.rounds_this_user_turn = 0
        self.person_turns = 0
        self._sync_person_expectation()
        return self

    def clone_reset(self) -> "EpisodeRuntime":
        return EpisodeRuntime(self.spec, self.world, self.config,
                              interpreter=self.interpreter, person=self.person)

    def interpret_and_step(
        self,
        emission: str | Mapping[str, Any],
        tool_calls: Sequence[Mapping[str, Any]] | None = None,
    ) -> Transition:
        policy = self.spec.response_policy
        variants = policy.get("approved_refusals", policy.get("refusal_variants", ()))
        if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
            variants = ()
        action = self.interpreter.parse(
            emission,
            tool_calls,
            canonical_refusal=(str(policy["canonical_refusal"])
                               if policy.get("canonical_refusal") else None),
            approved_refusals=tuple(str(x) for x in variants),
            ordinary_text_is_final=isinstance(self.current_node.expected_action, Final),
        )
        return self.step(action)

    def step(self, action: AssistantAction) -> Transition:
        if self._terminal:
            raise EpisodeTerminatedError(f"episode {self.spec.episode_id} is terminal")
        previous = self.state_id
        self.assistant_rounds += 1
        self.rounds_this_user_turn += 1
        messages: list[Mapping[str, Any]] = []
        executions: list[WorldExecution] = []
        new_violations: list[str] = []
        node = self.current_node
        expected = node.expected_action

        if isinstance(action, Invalid):
            code = "multiple_questions" if action.reason == "multiple_questions" \
                else "malformed_output"
            self._violate(code, new_violations)
            # The rollout cannot ask LFM for another assistant turn without a
            # new environment message.  SyntheticPerson supplies a frozen
            # correction for both multi-question and malformed emissions.
            reply = self.person.respond(action, self.person_state)
            messages.append({"role": "user", "content": reply.content})
            self._person_replied()

        elif isinstance(action, Ask):
            if isinstance(expected, Ask) and slots_equivalent(expected.slot, action.slot):
                reply = self.person.respond(action, self.person_state)
                messages.append({"role": "user", "content": reply.content})
                self._credited_asks.add(str(node.milestone_ids["ask"]))
                for slot in reply.revealed_slots:
                    self.person_state.revealed_slots.add(slot)
                    self.person_state.response_history[slot] = reply.content
                self._advance(node.next_state)
                self._person_replied()
            else:
                reply = self.person.respond(action, self.person_state)
                code = "repeated_question" if reply.kind == "repeated" else "wrong_question"
                self._violate(code, new_violations)
                messages.append({"role": "user", "content": reply.content})
                self._person_replied()

        elif isinstance(action, ToolCalls):
            if not action.calls:
                self._violate("malformed_output", new_violations)
            else:
                self._made_calls.extend(action.calls)
                exact_round = (isinstance(expected, ToolCalls)
                               and len(action.calls) == len(expected.calls)
                               and all(strict_call_equal(want, got)
                                       for want, got in zip(expected.calls, action.calls)))
                for ordinal, call in enumerate(action.calls):
                    # Reward equality permits one registry-backed alias:
                    # a trailing city suffix on location fields. When a whole round is
                    # reward-equivalent, execute the compiler's canonical
                    # arguments so the alias cannot escape the frozen map and
                    # silently change an empty/error result into a fallback.
                    world_args = (expected.calls[ordinal].arguments
                                  if exact_round and isinstance(expected, ToolCalls)
                                  else call.arguments)
                    execution = self.world.execute(call.name, world_args)
                    executions.append(execution)
                    call_id = call.call_id or (
                        f"call_{self.spec.episode_id}_{self.assistant_rounds}_{ordinal}"
                    )
                    messages.append({
                        "role": "tool",
                        "content": json.dumps(execution.result, ensure_ascii=False,
                                              sort_keys=True, separators=(",", ":")),
                        "tool_call_id": call_id,
                    })

                if exact_round:
                    self._credited_structure.add(str(node.milestone_ids["round"]))
                    status = self._result_status(executions)
                    expected_status = node.expected_result_status
                    if expected_status and status != expected_status:
                        raise WorldSnapshotError(
                            f"episode {self.spec.episode_id} state {node.state_id}: "
                            f"world status {status!r} != compiled {expected_status!r}"
                        )
                    if node.result_branches:
                        nxt = node.result_branches.get(status)
                        if nxt is None:
                            raise EpisodeSpecError(
                                f"state {node.state_id}: no branch for world status {status!r}"
                            )
                        branch_ids = node.milestone_ids["branches"]
                        self._credited_structure.add(str(branch_ids[status]))
                    else:
                        nxt = node.next_state
                    for execution in executions:
                        self._successful_tool_keys.add(execution.key)
                    self._advance(nxt)
                else:
                    current_keys = {execution.key for execution in executions}
                    if current_keys and current_keys <= self._successful_tool_keys:
                        code = "repeated_tool"
                    elif not isinstance(expected, ToolCalls):
                        code = "premature_tool"
                    else:
                        code = "extra_or_wrong_tool"
                    self._violate(code, new_violations)

        elif isinstance(action, Final):
            if isinstance(expected, Final) and self._valid_final_text(action.text):
                self._finish_success(node)
            elif not isinstance(expected, (Final, Refusal)):
                self._violate("premature_final", new_violations)
                self._terminal = True
            elif not action.text.strip():
                self._violate("malformed_output", new_violations)
            else:
                self._violate("invalid_final", new_violations)
                self._terminal = True

        elif isinstance(action, Refusal):
            if isinstance(expected, Refusal) and self._approved_refusal(action.text):
                self._finish_success(node)
            elif not isinstance(expected, (Final, Refusal)):
                self._violate("premature_final", new_violations)
                self._terminal = True
            else:
                self._violate("invalid_final", new_violations)
                self._terminal = True
        else:  # defensive for callers bypassing the type checker
            self._violate("malformed_output", new_violations)

        self._apply_caps(new_violations)
        return self._transition(previous, messages, executions, new_violations)

    def truncate(self, reason: str = "external_limit") -> Transition:
        if self._terminal:
            raise EpisodeTerminatedError(f"episode {self.spec.episode_id} is terminal")
        previous = self.state_id
        new: list[str] = []
        self._truncate(reason, new)
        return self._transition(previous, [], [], new)

    def reward(self) -> RewardBreakdown:
        cfg = self.config.reward
        call_pairs = _lcs_pairs(self._required_calls, self._made_calls)
        calls_correct = len(call_pairs)
        ask_score = self._ratio(len(self._credited_asks), len(self._required_asks))
        call_score = self._ratio(calls_correct, len(self._required_calls))
        structure_score = self._ratio(len(self._credited_structure),
                                      len(self._required_structure))
        goal = int(self._goal_reached)
        counts = Counter(self._violations)
        credited = self._credited_milestone_names(call_pairs)
        # U counts recoverable illegal model emissions only.  Terminal failures
        # (premature/invalid final) and environment caps still remain in the
        # outcome diagnostics, but terminal gating already fixes their reward
        # at zero and they are not part of U.
        violation_count = sum(counts[name] for name in _RECOVERABLE_VIOLATIONS)
        clean = int(self._goal_reached and violation_count == 0)
        penalties = {
            name: round(count * cfg.penalty_per_violation, 12)
            for name, count in sorted(counts.items())
            if count and name in _RECOVERABLE_VIOLATIONS
        }
        penalty_total = round(violation_count * cfg.penalty_per_violation, 12)

        # Reward completed outcomes; diagnostic prefix coverage does not earn reward.
        if goal:
            raw = cfg.maximum - penalty_total
            total = max(cfg.success_floor, min(cfg.maximum, raw))
        else:
            raw = 0.0
            total = 0.0
        return RewardBreakdown(
            total=total,
            unclipped_total=raw,
            ask_score=ask_score,
            call_score=call_score,
            structure_score=structure_score,
            goal_reached=goal,
            clean_trace=clean,
            violation_count=violation_count,
            penalties=penalties,
            penalty_total=penalty_total,
            credited_milestones=credited,
            required_milestones=self.required_milestones,
            calls_correct=calls_correct,
            calls_required=len(self._required_calls),
        )

    def outcome(self) -> EpisodeOutcome:
        return EpisodeOutcome(
            episode_id=self.spec.episode_id,
            state_id=self.state_id,
            terminal=self._terminal,
            success=self._goal_reached,
            goal_reached=self._goal_reached,
            clean_trace=self._goal_reached and self._recoverable_violation_count() == 0,
            truncated=self._truncated,
            truncation_reason=self._truncation_reason,
            assistant_rounds=self.assistant_rounds,
            person_turns=self.person_turns,
            violations=tuple(self._violations),
            reward=self.reward(),
        )

    def _compile_expected_path(
        self,
    ) -> tuple[tuple[str, ...], tuple[ToolCall, ...], tuple[str, ...],
               tuple[str, ...], tuple[str, ...], str]:
        state = self.spec.initial_state_id
        seen: set[str] = set()
        path: list[str] = []
        calls: list[ToolCall] = []
        call_ids: list[str] = []
        asks: list[str] = []
        structure: list[str] = []
        canonical_milestones: list[str] = []
        completion = ""
        while state != "done":
            if state in seen:
                raise EpisodeSpecError(f"expected path contains a cycle at {state!r}")
            seen.add(state)
            node = self._nodes.get(state)
            if node is None:
                raise EpisodeSpecError(f"expected path references unknown state {state!r}")
            path.append(state)
            action = node.expected_action
            if isinstance(action, Ask):
                ask_id = self._milestone_string(node, "ask")
                asks.append(ask_id)
                canonical_milestones.append(ask_id)
                state = self._required_next(node.next_state, state)
            elif isinstance(action, ToolCalls):
                calls.extend(action.calls)
                raw_call_ids = node.milestone_ids.get("calls")
                if (not isinstance(raw_call_ids, list)
                        or len(raw_call_ids) != len(action.calls)
                        or not all(isinstance(item, str) and item for item in raw_call_ids)):
                    raise EpisodeSpecError(
                        f"state {state}: milestone_ids.calls must align with expected calls"
                    )
                call_ids.extend(raw_call_ids)
                canonical_milestones.extend(raw_call_ids)
                round_id = self._milestone_string(node, "round")
                structure.append(round_id)
                canonical_milestones.append(round_id)
                executions = [self.world.execute(call.name, call.arguments)
                              for call in action.calls]
                status = self._result_status(executions)
                if node.expected_result_status and status != node.expected_result_status:
                    raise WorldSnapshotError(
                        f"episode {self.spec.episode_id} state {state}: "
                        f"world status {status!r} != compiled {node.expected_result_status!r}"
                    )
                if node.result_branches:
                    branch_ids = node.milestone_ids.get("branches")
                    if not isinstance(branch_ids, Mapping):
                        raise EpisodeSpecError(f"state {state}: milestone_ids.branches is required")
                    branch_id = branch_ids.get(status)
                    if not isinstance(branch_id, str) or not branch_id:
                        raise EpisodeSpecError(
                            f"state {state}: no milestone ID for branch {status!r}"
                        )
                    structure.append(branch_id)
                    canonical_milestones.append(branch_id)
                    state = self._required_next(node.result_branches.get(status), state)
                else:
                    state = self._required_next(node.next_state, state)
            elif isinstance(action, (Final, Refusal)):
                completion = self._milestone_string(node, "completion")
                canonical_milestones.append(completion)
                state = node.next_state or "done"
            else:
                raise EpisodeSpecError(f"state {state}: invalid expected action")
        if not completion:
            raise EpisodeSpecError("expected path does not contain Final or Refusal")
        declared = self.spec.reward_milestones
        if tuple(canonical_milestones) != declared:
            raise EpisodeSpecError(
                f"episode {self.spec.episode_id}: reward_milestones drift: "
                f"declared={list(declared)!r}, canonical={canonical_milestones!r}"
            )
        if len(declared) != len(set(declared)):
            raise EpisodeSpecError(
                f"episode {self.spec.episode_id}: reward milestone IDs are not unique"
            )
        return (tuple(path), tuple(calls), tuple(call_ids), tuple(asks),
                tuple(structure), completion)

    @staticmethod
    def _milestone_string(node: StateNode, key: str) -> str:
        value = node.milestone_ids.get(key)
        if not isinstance(value, str) or not value:
            raise EpisodeSpecError(
                f"state {node.state_id}: milestone_ids.{key} must be a non-empty string"
            )
        return value

    @staticmethod
    def _required_next(value: str | None, state: str) -> str:
        if not value:
            raise EpisodeSpecError(f"state {state}: expected path lacks next state")
        return value

    @staticmethod
    def _result_status(executions: Sequence[WorldExecution]) -> str:
        statuses = [str(item.result.get("status") or "error") for item in executions]
        if "error" in statuses:
            return "error"
        if "empty" in statuses:
            return "empty"
        return "ok"

    def _sync_person_expectation(self) -> None:
        if self._terminal or self.state_id == "done":
            self.person_state.expected_slot = None
            self.person_state.expected_response = None
            self.person_state.node_policy = {}
            return
        node = self.current_node
        self.person_state.current_state_id = node.state_id
        self.person_state.expected_slot = (node.expected_action.slot
                                           if isinstance(node.expected_action, Ask) else None)
        self.person_state.expected_response = node.user_response
        self.person_state.node_policy = copy.deepcopy(dict(node.raw))

    def _advance(self, next_state: str | None) -> None:
        if not next_state:
            raise EpisodeSpecError(f"state {self.state_id}: legal transition lacks next state")
        if next_state == "done":
            self.person_state.current_state_id = "done"
        elif next_state not in self._nodes:
            raise EpisodeSpecError(f"state {self.state_id}: unknown next state {next_state!r}")
        else:
            self.person_state.current_state_id = next_state
        self._sync_person_expectation()

    def _finish_success(self, node: StateNode) -> None:
        self._credited_completion.add(self._required_completion)
        self._goal_reached = True
        self._terminal = True
        self.person_state.current_state_id = node.next_state or "done"
        self._sync_person_expectation()

    def _valid_final_text(self, text: str) -> bool:
        """Require a meaningful, goal-grounded answer without scoring payload prose.

        Tool-result wording remains deliberately ungraded.  This check only
        prevents a one-token/irrelevant completion from collecting the goal and
        clean bonuses after the action sequence: normal Chinese prose must be
        substantial, and workflows with a concrete destination/hotel must name
        that episode entity.
        """

        compact = "".join(str(text).split())
        cjk = re.findall(r"[\u3400-\u9fff]", compact)
        if len(compact) < 12 or len(cjk) < 6 or len(set(cjk)) < 4:
            return False

        goal = str(self.person_state.goal.get("workflow") or "")
        slots = self.person_state.private_slots
        entity_keys = {
            "travel_plan": ("destination",),
            "navigation": ("route_destination",),
            "hotel_reviews": ("hotel_name", "hotel_location"),
            "hotel_recommendation": ("hotel_name", "hotel_location"),
        }.get(goal, ())
        entities = []
        for key in entity_keys:
            value = slots.get(key)
            if value is None:
                continue
            entity = "".join(str(value).split())
            if entity.endswith("市") and len(entity) > 1:
                entity = entity[:-1]
            if entity:
                entities.append(entity)
        return not entities or any(entity in compact for entity in entities)

    @staticmethod
    def _normalize_refusal(text: str) -> str:
        return "".join(text.split()).rstrip("。.!！")

    def _approved_refusal(self, text: str) -> bool:
        policy = self.spec.response_policy
        values: list[str] = []
        if policy.get("canonical_refusal"):
            values.append(str(policy["canonical_refusal"]))
        variants = policy.get("approved_refusals", policy.get("refusal_variants", ()))
        if isinstance(variants, Sequence) and not isinstance(variants, (str, bytes)):
            values.extend(str(item) for item in variants)
        if not values:
            expected = self.current_node.expected_action
            if isinstance(expected, Refusal) and expected.text:
                values.append(expected.text)
        if not values:
            return False
        normalized = self._normalize_refusal(text)
        return any(normalized == self._normalize_refusal(value) for value in values)

    def _person_replied(self) -> None:
        self.person_turns += 1
        self.rounds_this_user_turn = 0

    def _violate(self, code: str, new: list[str]) -> None:
        self._violations.append(code)
        new.append(code)

    def _truncate(self, reason: str, new: list[str]) -> None:
        self._violate("truncation", new)
        self._truncated = True
        self._truncation_reason = reason
        self._terminal = True

    def _apply_caps(self, new: list[str]) -> None:
        if self._terminal:
            return
        if self.assistant_rounds >= self.config.max_assistant_rounds:
            self._truncate("max_assistant_rounds", new)
        elif self.rounds_this_user_turn >= self.config.max_rounds_per_user_turn:
            self._truncate("max_rounds_per_user_turn", new)
        elif self.person_turns >= self.config.max_person_turns:
            self._truncate("max_person_turns", new)

    def _transition(
        self,
        previous: str,
        messages: Sequence[Mapping[str, Any]],
        executions: Sequence[WorldExecution],
        new: Sequence[str],
    ) -> Transition:
        return Transition(
            previous_state_id=previous,
            state_id=self.state_id,
            messages=tuple(copy.deepcopy(dict(message)) for message in messages),
            tool_results=tuple(executions),
            terminal=self._terminal,
            success=self._goal_reached,
            goal_reached=self._goal_reached,
            clean_trace=self._goal_reached and self._recoverable_violation_count() == 0,
            new_violations=tuple(new),
            violations=tuple(self._violations),
        )

    @staticmethod
    def _ratio(credited: int, required: int) -> float:
        return 1.0 if required == 0 else credited / required

    def _recoverable_violation_count(self) -> int:
        return sum(code in _RECOVERABLE_VIOLATIONS for code in self._violations)

    def _credited_milestone_names(
        self, call_pairs: Sequence[tuple[int, int]]
    ) -> tuple[str, ...]:
        credited = (self._credited_asks
                    | {self._required_call_ids[i] for i, _ in call_pairs}
                    | self._credited_structure
                    | self._credited_completion)
        # Report in the compiler-declared order, never set/hash order.
        return tuple(item for item in self.spec.reward_milestones if item in credited)


class EpisodeRepository:
    """Read-only index of Step-7 EpisodeSpecs and their one shared world."""

    def __init__(self, episodes: Sequence[EpisodeSpec], world: SyntheticChinaWorld) -> None:
        self.world = world
        self._episodes: dict[str, EpisodeSpec] = {}
        self._by_source_idx: dict[int, EpisodeSpec] = {}
        for episode in episodes:
            if episode.episode_id in self._episodes:
                raise EpisodeSpecError(f"duplicate episode_id {episode.episode_id!r}")
            if episode.world_id and world.world_id and episode.world_id != world.world_id:
                raise WorldSnapshotError(
                    f"episode {episode.episode_id} targets {episode.world_id}, "
                    f"loaded {world.world_id}"
                )
            self._episodes[episode.episode_id] = episode
            if episode.source_idx is not None:
                if episode.source_idx in self._by_source_idx:
                    raise EpisodeSpecError(f"duplicate source_idx {episode.source_idx}")
                self._by_source_idx[episode.source_idx] = episode

    @classmethod
    def load(
        cls,
        episodes_path: str | Path,
        world_path: str | Path,
        *,
        expected_count: int | None = None,
    ) -> "EpisodeRepository":
        episodes: list[EpisodeSpec] = []
        with Path(episodes_path).open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EpisodeSpecError(f"invalid episode JSON on line {line_no}") from exc
                if not isinstance(raw, Mapping):
                    raise EpisodeSpecError(f"episode line {line_no} is not an object")
                episodes.append(EpisodeSpec.from_dict(raw))
        if expected_count is not None and len(episodes) != expected_count:
            raise EpisodeSpecError(
                f"episode count {len(episodes)} != expected {expected_count}"
            )
        return cls(episodes, SyntheticChinaWorld.load(world_path))

    def __len__(self) -> int:
        return len(self._episodes)

    def episode(self, episode_id: str | int) -> EpisodeSpec:
        key = str(episode_id)
        if key in self._episodes:
            return self._episodes[key]
        if isinstance(episode_id, int) and episode_id in self._by_source_idx:
            return self._by_source_idx[episode_id]
        raise KeyError(f"unknown episode {episode_id!r}")

    def new_runtime(
        self,
        episode_id: str | int,
        config: RuntimeConfig | Mapping[str, Any] | None = None,
    ) -> EpisodeRuntime:
        return EpisodeRuntime(self.episode(episode_id), self.world, config)

    def reset_group(
        self,
        episode_id: str | int,
        group_size: int = 4,
        config: RuntimeConfig | Mapping[str, Any] | None = None,
    ) -> tuple[EpisodeRuntime, ...]:
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        return tuple(self.new_runtime(episode_id, config) for _ in range(group_size))


__all__ = [
    "EpisodeOutcome",
    "EpisodeRepository",
    "EpisodeRuntime",
    "EpisodeSpec",
    "EpisodeSpecError",
    "EpisodeTerminatedError",
    "PersonReply",
    "PersonState",
    "RewardBreakdown",
    "RewardConfig",
    "RuntimeConfig",
    "StateNode",
    "SyntheticPerson",
    "Transition",
    "slots_equivalent",
    "strict_call_equal",
]
