#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Claude Code CLI transport for synthetic-data generation.

    claude --print --model <id> --effort <level> --json-schema '<schema>'

Prompts are passed through stdin. Authentication comes from the local Claude
Code login, and optional JSON schemas constrain structured output.
"""
import json
import os
import subprocess
from typing import Any, Dict, Optional

DEFAULT_MODEL = os.getenv("CLAUDE_CLI_MODEL", "claude-opus-5")
DEFAULT_EFFORT = os.getenv("CLAUDE_CLI_EFFORT", "medium")
TIMEOUT = int(os.getenv("CLAUDE_CLI_TIMEOUT", "300"))


class ClaudeCLIError(RuntimeError):
    pass


def run(prompt: str, schema: Optional[Dict[str, Any]] = None,
        model: str = "", effort: str = "") -> Any:
    """Run ``claude --print`` and return text or schema-decoded JSON."""
    cmd = ["claude", "--print",
           "--model", model or DEFAULT_MODEL,
           "--effort", effort or DEFAULT_EFFORT]
    if schema is not None:
        cmd += ["--json-schema", json.dumps(schema, ensure_ascii=False)]

    env = dict(os.environ)
    # Remove explicitly to use the local login instead of a stale environment key
    env.pop("ANTHROPIC_API_KEY", None)

    try:
        proc = subprocess.run(cmd, input=prompt, capture_output=True,
                              text=True, timeout=TIMEOUT, env=env)
    except subprocess.TimeoutExpired as exc:
        raise ClaudeCLIError(f"claude --print timed out after {TIMEOUT}s") from exc

    if proc.returncode != 0:
        raise ClaudeCLIError(f"claude --print exited with code {proc.returncode}: "
                             f"{(proc.stderr or proc.stdout)[:300]}")

    out = (proc.stdout or "").strip()
    if schema is None:
        return out
    try:
        return json.loads(out)
    except json.JSONDecodeError as exc:
        raise ClaudeCLIError(f"could not parse structured output: {out[:300]}") from exc


def available() -> bool:
    """Return whether the CLI is installed and authenticated."""
    try:
        return run("回复 ok 两个字", schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        }).get("ok") is not None
    except Exception:
        return False
