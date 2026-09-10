#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provider-independent tool-result envelopes.

Every result carries ``status`` and ``source``. Empty results remain distinct
from errors, and successful list-valued results contain at least one item.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

OK, EMPTY, ERROR = "ok", "empty", "error"
Status = Literal["ok", "empty", "error"]

# Every response includes these two fields.
BASE = {"status": str, "source": str}


class ToolResult(BaseModel):
    """Base envelope that permits provider-specific result fields."""

    model_config = ConfigDict(extra="allow")

    status: Status
    source: str = Field(min_length=1)


class Ok(ToolResult):
    status: Literal["ok"] = "ok"


class Empty(ToolResult):
    """A successful query with no matching result."""

    status: Literal["empty"] = "empty"


class Error(ToolResult):
    """A failed query, including resolution, API, and retry failures."""

    status: Literal["error"] = "error"
    message: str = ""


def validate(contracts: Dict[str, Dict[str, Any]], tool_name: str,
             payload: Dict[str, Any]) -> None:
    """Validate the shared envelope and application-declared top-level fields."""
    assert tool_name in contracts, f"unknown tool {tool_name!r}"
    assert isinstance(payload, dict), f"{tool_name}: expected dict, got {type(payload).__name__}"

    status = payload.get("status")
    assert status in (OK, EMPTY, ERROR), f"{tool_name}: bad status {status!r}"
    assert isinstance(payload.get("source"), str) and payload["source"], \
        f"{tool_name}: missing source"

    spec = contracts[tool_name]["returns"]
    for key in spec:
        assert key in payload, f"{tool_name}: missing key {key!r}"

    # List fields must be nonempty for ok and empty otherwise. Rule 3.
    for key, expected in spec.items():
        if not isinstance(expected, list):
            continue
        items = payload.get(key)
        assert isinstance(items, list), f"{tool_name}: {key} must be a list"
        if status == OK:
            assert items, f"{tool_name}: status=ok but {key} is empty"
        else:
            assert not items, f"{tool_name}: status={status} but {key} is populated"


def _payload(contracts: Dict[str, Dict[str, Any]], tool: str, status: str,
             message: str, extra: Dict[str, Any]) -> Dict[str, Any]:
    """Combine the explicit tool skeleton, status, source, and extra fields."""
    return {**contracts[tool]["skeleton"], **extra, "status": status, "source": message}


def err(contracts: Dict[str, Dict[str, Any]], tool: str, message: str,
        **extra) -> Dict[str, Any]:
    """Build a standard error payload."""
    return _payload(contracts, tool, ERROR, message, extra)


def empty(contracts: Dict[str, Dict[str, Any]], tool: str, message: str,
          **extra) -> Dict[str, Any]:
    """Build a standard empty-result payload."""
    return _payload(contracts, tool, EMPTY, message, extra)
