"""Typed assistant actions used by the CN Travel world-policy runtime.

The model never emits these class names.  :mod:`.interpreter` converts normal
LFM text or already parsed tool calls into this small, deterministic action
vocabulary before the episode state machine sees an emission.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TypeAlias


@dataclass(frozen=True)
class ToolCall:
    """One parsed tool call.

    ``call_id`` is transport metadata only.  It is deliberately excluded from
    call correctness and world lookup.
    """

    name: str
    arguments: Mapping[str, Any]
    call_id: str | None = None


@dataclass(frozen=True)
class ToolCalls:
    calls: tuple[ToolCall, ...]
    raw_text: str = ""


@dataclass(frozen=True)
class Ask:
    slot: str
    text: str
    matched_rule: str | None = None


@dataclass(frozen=True)
class Final:
    text: str


@dataclass(frozen=True)
class Refusal:
    text: str


@dataclass(frozen=True)
class Invalid:
    reason: str
    raw_text: str = ""


AssistantAction: TypeAlias = ToolCalls | Ask | Final | Refusal | Invalid


__all__ = [
    "Ask",
    "AssistantAction",
    "Final",
    "Invalid",
    "Refusal",
    "ToolCall",
    "ToolCalls",
]
