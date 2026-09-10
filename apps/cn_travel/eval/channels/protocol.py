#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared schema and rendering protocol for evaluator CLI channels.

Claude and Codex return schema-constrained ``reply`` and ``tool_calls`` fields.
The evaluator adapters translate that representation into the subset of the
OpenAI chat-completions response shape consumed by the rollout driver.
"""
import json
import uuid
from typing import Any, Dict, List, Optional

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {
            "type": "string",
            "description": "给用户的最终回复。若本轮需要调用工具，留空字符串。",
        },
        "tool_calls": {
            "type": "array",
            "description": "本轮要调用的工具；不需要调用工具时为空数组。",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "工具名称"},
                    "arguments_json": {
                        "type": "string",
                        "description": "该工具的参数，JSON 对象的字符串形式",
                    },
                },
                "required": ["name", "arguments_json"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["reply", "tool_calls"],
    "additionalProperties": False,
}


class _Function:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, name: str, arguments: str):
        self.id = f"call_{uuid.uuid4().hex[:8]}"
        self.type = "function"
        self.function = _Function(name, arguments)


class _Message:
    def __init__(self, content: Optional[str], tool_calls: Optional[List[_ToolCall]]):
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, message: _Message):
        self.message = message
        self.finish_reason = "tool_calls" if message.tool_calls else "stop"


class _Response:
    def __init__(self, message: _Message, model: str):
        self.choices = [_Choice(message)]
        self.model = model


def render_tools(tools: Optional[List[Dict[str, Any]]]) -> str:
    if not tools:
        return "本轮没有可用工具，直接回复用户。"
    lines = ["你可以调用下列工具。需要调用时，把调用放进 tool_calls；"
             "不需要时 tool_calls 为空数组并在 reply 里给出最终回复。", ""]
    for t in tools:
        fn = t.get("function", t)
        params = fn.get("parameters", {})
        required = params.get("required", [])
        lines.append(f"### {fn.get('name')}")
        lines.append(fn.get("description", ""))
        for pname, pspec in (params.get("properties") or {}).items():
            flag = "必填" if pname in required else "可选"
            lines.append(f"  - {pname} ({flag}): {pspec.get('description', '')}")
        lines.append("")
    return "\n".join(lines)


def render_messages(messages: List[Dict[str, Any]]) -> str:
    """Flatten OpenAI-format messages into the evaluator's text protocol."""
    parts: List[str] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            parts.append(f"<系统指令>\n{m.get('content', '')}\n</系统指令>")
        elif role == "user":
            parts.append(f"<用户>\n{m.get('content', '')}\n</用户>")
        elif role == "assistant":
            calls = m.get("tool_calls") or []
            if calls:
                rendered = "\n".join(
                    f"  - {c['function']['name']}({c['function']['arguments']})"
                    for c in calls)
                parts.append(f"<你已发起的工具调用>\n{rendered}\n</你已发起的工具调用>")
            if m.get("content"):
                parts.append(f"<你的回复>\n{m['content']}\n</你的回复>")
        elif role == "tool":
            parts.append(f"<工具返回>\n{m.get('content', '')}\n</工具返回>")
    return "\n\n".join(parts)


def response_from_data(data: Dict[str, Any], model: str) -> _Response:
    """Translate one schema-constrained CLI response into OpenAI-like objects."""
    calls: List[_ToolCall] = []
    for call in data.get("tool_calls") or []:
        name = call.get("name")
        if not name:
            continue
        arguments = call.get("arguments_json") or "{}"
        try:
            json.loads(arguments)
        except json.JSONDecodeError:
            arguments = "{}"
        calls.append(_ToolCall(name, arguments))

    reply = data.get("reply") or ""
    return _Response(_Message(reply or None, calls or None), model)
