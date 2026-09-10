#!/usr/bin/env python3
"""Claude Code CLI channel used by the sealed evaluator proxy."""
from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Dict, List, Optional

from eval.channels.protocol import (
    RESPONSE_SCHEMA,
    render_messages,
    render_tools,
    response_from_data,
)

DEFAULT_MODEL = os.getenv("CLAUDE_CLI_MODEL", "claude-opus-5")
DEFAULT_EFFORT = os.getenv("CLAUDE_CLI_EFFORT", "medium")
TIMEOUT = int(os.getenv("CLAUDE_CLI_TIMEOUT", "300"))


class ClaudeCLIError(RuntimeError):
    """The evaluator could not obtain a valid Claude CLI response."""


def run(
    prompt: str,
    schema: Optional[Dict[str, Any]] = None,
    model: str = "",
    effort: str = "",
) -> Any:
    """Run ``claude --print`` using the evaluator operator's local login."""
    command = [
        "claude",
        "--print",
        "--model",
        model or DEFAULT_MODEL,
        "--effort",
        effort or DEFAULT_EFFORT,
    ]
    if schema is not None:
        command += ["--json-schema", json.dumps(schema, ensure_ascii=False)]

    environment = dict(os.environ)
    environment.pop("ANTHROPIC_API_KEY", None)
    try:
        process = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        raise ClaudeCLIError(f"claude --print timed out after {TIMEOUT}s") from exc

    if process.returncode != 0:
        detail = (process.stderr or process.stdout)[:300]
        raise ClaudeCLIError(
            f"claude --print exited with code {process.returncode}: {detail}"
        )

    output = (process.stdout or "").strip()
    if schema is None:
        return output
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise ClaudeCLIError(
            f"could not parse structured output: {output[:300]}"
        ) from exc


class _Completions:
    def create(
        self,
        model: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = None,
        **_ignored: Any,
    ):
        del tool_choice
        prompt = "\n\n".join(
            [
                render_tools(tools),
                "以下是当前对话，请按系统指令继续。工具返回的内容已经给你，"
                "不要重复调用已经拿到结果的工具。",
                render_messages(messages or []),
            ]
        )
        data = run(prompt, schema=RESPONSE_SCHEMA, model=model)
        return response_from_data(data, model)


class _Chat:
    def __init__(self) -> None:
        self.completions = _Completions()


class ClaudeCLIClient:
    """Minimal chat-completions client implemented by the Claude CLI channel."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.chat = _Chat()
