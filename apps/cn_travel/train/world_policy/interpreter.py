"""Deterministic interpreter for ordinary LFM assistant emissions."""
from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .actions import Ask, AssistantAction, Final, Invalid, Refusal, ToolCall, ToolCalls

_LFM_OPEN = "<|tool_call_start|>"
_LFM_CLOSE = "<|tool_call_end|>"
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")

# Rules intentionally require question-like language as well as a slot cue.
# This avoids treating a final answer that merely mentions a date/hotel/city as
# another question. Rule names are part of the interpreter contract.
_QUESTION_RULES: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    (
        "travel_date",
        "ask.travel_date.v1",
        re.compile(
            r"(?:请问|请告诉|能否|可以|方便|想问|计划)?[^。！!]{0,24}"
            r"(?:什么时候|哪天|几号|什么日期|出发时间|出行时间|旅行日期)"
            r"|(?:什么时候|哪天|几号)[^。！!]{0,16}(?:出发|去|旅行|出行)",
            re.IGNORECASE,
        ),
    ),
    (
        "hotel_name",
        "ask.hotel_name.v1",
        re.compile(
            r"(?:哪家|什么|哪个)[^。！!]{0,8}(?:酒店|宾馆|旅馆)"
            r"|(?:酒店|宾馆|旅馆)[^。！!]{0,8}(?:名字|名称|哪家)"
            r"|想了解哪家酒店",
            re.IGNORECASE,
        ),
    ),
    (
        "hotel_city",
        "ask.hotel_city.v1",
        re.compile(
            r"(?:哪个|哪座|什么)[^。！!]{0,5}(?:城市|地方)[^。！!]{0,12}(?:酒店|住宿)"
            r"|(?:酒店|住宿)[^。！!]{0,12}(?:哪个|哪座|什么)(?:城市|地方)",
            re.IGNORECASE,
        ),
    ),
    (
        "route_destination",
        "ask.route_destination.v1",
        re.compile(
            r"(?:路线|怎么走|导航)[^。！!]{0,18}(?:去哪|去哪里|到哪|目的地|终点)"
            r"|(?:终点|路线目的地)[^。！!]{0,8}(?:哪里|哪儿|什么)",
            re.IGNORECASE,
        ),
    ),
    (
        "destination",
        "ask.destination.v1",
        re.compile(
            r"(?:想|打算|计划|准备)?(?:去|到)(?:哪里|哪儿|哪个城市|什么地方)"
            r"|(?:目的地|旅行城市|旅游城市)[^。！!]{0,10}(?:哪里|哪儿|哪个|什么)"
            r"|请告诉我[^。！!]{0,8}(?:哪个城市|目的地)",
            re.IGNORECASE,
        ),
    ),
)

_QUESTION_MARKER = re.compile(r"[?？]|(?:请问|请告诉|能否|可以告诉|方便告诉|想问)")
_REFUSAL_RE = re.compile(
    r"(?:抱歉|不好意思)[^。！!]{0,40}(?:旅行|旅游)[^。！!]{0,20}"
    r"(?:只能|仅能|只回答|无法回答)[^。！!]{0,24}(?:相关|问题|内容)",
    re.IGNORECASE,
)


def _plain_text(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


def _parse_arguments(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        return None
    try:
        got = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return got if isinstance(got, dict) else None


def _structured_calls(items: Any) -> tuple[ToolCall, ...] | None:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
        return None
    calls: list[ToolCall] = []
    for item in items:
        if not isinstance(item, Mapping):
            return None
        fn = item.get("function") if isinstance(item.get("function"), Mapping) else item
        name = fn.get("name") or fn.get("tool")
        args = fn.get("arguments", fn.get("args"))
        parsed = _parse_arguments(args)
        if not isinstance(name, str) or not name or parsed is None:
            return None
        call_id = item.get("id") or item.get("call_id")
        calls.append(ToolCall(name=name, arguments=parsed,
                              call_id=str(call_id) if call_id is not None else None))
    return tuple(calls)


def _literal(node: ast.AST) -> Any:
    """Parse an LFM argument without executing model-produced Python."""

    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError) as exc:
        raise ValueError("non-literal LFM tool argument") from exc


def _lfm_calls(text: str) -> tuple[ToolCall, ...]:
    if text.count(_LFM_OPEN) != 1 or text.count(_LFM_CLOSE) != 1:
        raise ValueError("malformed LFM tool-call markers")
    before, rest = text.split(_LFM_OPEN, 1)
    body, after = rest.split(_LFM_CLOSE, 1)
    # Thinking is allowed before a call.  Ordinary prose mixed with a tool call
    # is rejected because it is ambiguous whether the model meant to finish.
    if _plain_text(before) or _plain_text(after):
        raise ValueError("text mixed with LFM tool calls")
    try:
        parsed = ast.parse(body.strip(), mode="eval").body
    except SyntaxError as exc:
        raise ValueError("invalid LFM tool-call syntax") from exc
    nodes = parsed.elts if isinstance(parsed, (ast.List, ast.Tuple)) else [parsed]
    if not nodes:
        raise ValueError("empty LFM tool-call batch")
    calls: list[ToolCall] = []
    for node in nodes:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise ValueError("LFM tool batch contains a non-call")
        if node.args:
            raise ValueError("LFM tool calls must use named arguments")
        args: dict[str, Any] = {}
        for kw in node.keywords:
            if kw.arg is None or kw.arg in args:
                raise ValueError("duplicate or expanded LFM tool argument")
            args[kw.arg] = _literal(kw.value)
        calls.append(ToolCall(name=node.func.id, arguments=args))
    return tuple(calls)


class AssistantActionInterpreter:
    """Map one model emission to one typed action.

    The optional ``tool_calls`` argument is the preferred path when the LFM
    tokenizer has already parsed its response.  Raw LFM markers are supported
    as a deterministic fallback and are parsed with :mod:`ast`, never ``eval``.
    """

    version = "cn-travel.assistant-interpreter.v1"

    def parse(
        self,
        emission: str | Mapping[str, Any],
        tool_calls: Sequence[Mapping[str, Any]] | None = None,
        *,
        canonical_refusal: str | None = None,
        approved_refusals: Sequence[str] = (),
        ordinary_text_is_final: bool = False,
    ) -> AssistantAction:
        if isinstance(emission, Mapping):
            text = str(emission.get("content") or "")
            if tool_calls is None:
                tool_calls = emission.get("tool_calls")  # type: ignore[assignment]
        else:
            text = str(emission or "")

        if tool_calls is not None:
            parsed = _structured_calls(tool_calls)
            if parsed is None or not parsed:
                return Invalid("malformed_tool_calls", text)
            # Keep probe/OpenAI transport semantics identical to VERL's raw
            # LFM path: thinking may precede a call, ordinary prose may not be
            # mixed into the same assistant emission.
            if _plain_text(text):
                return Invalid("text mixed with structured tool calls", text)
            return ToolCalls(parsed, raw_text=text)

        has_open = _LFM_OPEN in text
        has_close = _LFM_CLOSE in text
        if has_open or has_close:
            try:
                return ToolCalls(_lfm_calls(text), raw_text=text)
            except ValueError as exc:
                return Invalid(str(exc), text)

        plain = _plain_text(text)
        if not plain:
            return Invalid("empty_emission", text)

        approved = [x for x in (canonical_refusal, *approved_refusals) if x]
        norm = lambda x: _SPACE_RE.sub("", x).rstrip("。.!！")
        if any(norm(plain) == norm(str(x)) for x in approved) or _REFUSAL_RE.search(plain):
            return Refusal(plain)

        # TEXT has no globally fixed dialogue act.  When the episode state is
        # ready for completion, all other nonempty ordinary prose is a Final
        # candidate and the runtime's grounded-final predicate decides whether
        # it is valid.  This prevents question-like wording inside a complete
        # answer from being mistaken for another Ask, while REFUSAL remains a
        # distinct syntax label that cannot satisfy an ordinary Final state.
        if ordinary_text_is_final:
            return Final(plain)

        matches = [(slot, rule) for slot, rule, pattern in _QUESTION_RULES
                   if pattern.search(plain)]
        # A route-specific phrase also contains a generic destination phrase;
        # keep the more specific route label instead of calling that ambiguity.
        slots = {slot for slot, _ in matches}
        if "route_destination" in slots and "destination" in slots:
            matches = [(s, r) for s, r in matches if s != "destination"]
            slots.remove("destination")
        if len(slots) > 1:
            return Invalid("multiple_questions", plain)
        if len(slots) == 1:
            slot = next(iter(slots))
            rule = next(r for s, r in matches if s == slot)
            return Ask(slot=slot, text=plain, matched_rule=rule)
        if _QUESTION_MARKER.search(plain):
            return Ask(slot="unknown", text=plain, matched_rule="ask.unknown.v1")
        return Final(plain)


__all__ = ["AssistantActionInterpreter"]
