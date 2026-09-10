#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenAI-compatible evaluation proxy for the logged-in Codex CLI.

The proxy translates ``/v1/chat/completions`` requests into
    codex exec --ephemeral --skip-git-repo-check --sandbox read-only \
        --model <m> -c model_reasoning_effort="medium" \
        --output-schema <schema> -o <out> -
The rendered prompt arrives on stdin. Schema-constrained responses are converted
into OpenAI-format assistant messages and tool calls.

    make codex-proxy PROXY_PORT=8766
    make eval EVAL_ARGS='--system gpt-5.6-sol --model gpt-5.6-sol \
        --base-url http://127.0.0.1:8766/v1 --note "OpenAI cloud · codex CLI channel"'

"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from eval.channels.protocol import RESPONSE_SCHEMA, render_messages, render_tools

TIMEOUT = 300
EFFORT = "medium"


def _run_codex(model: str, prompt: str) -> dict:
    with tempfile.TemporaryDirectory() as td:
        schema = pathlib.Path(td) / "schema.json"
        out = pathlib.Path(td) / "out.json"
        schema.write_text(json.dumps(RESPONSE_SCHEMA, ensure_ascii=False), encoding="utf-8")
        cmd = ["codex", "exec", "--ephemeral", "--skip-git-repo-check",
               "--sandbox", "read-only", "--model", model,
               "-c", f'model_reasoning_effort="{EFFORT}"',
               "--output-schema", str(schema), "-o", str(out), "-"]
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                              timeout=TIMEOUT)
        if proc.returncode != 0:
            raise RuntimeError(f"codex exec exit code {proc.returncode}: "
                               f"{(proc.stderr or proc.stdout)[:300]}")
        text = out.read_text(encoding="utf-8") if out.exists() else proc.stdout
        return json.loads(text)


def _completion_json(body: dict) -> dict:
    model = body.get("model", "")
    prompt = "\n\n".join([
        render_tools(body.get("tools")),
        "以下是当前对话，请按系统指令继续。工具返回的内容已经给你，"
        "不要重复调用已经拿到结果的工具。",
        render_messages(body.get("messages") or []),
    ])
    data = _run_codex(model, prompt)
    calls = []
    for c in data.get("tool_calls") or []:
        name = c.get("name")
        if not name:
            continue
        args = c.get("arguments_json") or "{}"
        try:
            json.loads(args)
        except json.JSONDecodeError:
            args = "{}"
        calls.append({"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                      "function": {"name": name, "arguments": args}})
    reply = data.get("reply") or ""
    message: dict = {"role": "assistant", "content": reply or None}
    if calls:
        message["tool_calls"] = calls
    return {"id": f"chatcmpl-codex-{int(time.time() * 1000)}",
            "object": "chat.completion", "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if calls else "stop"}]}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [
                {"id": "gpt-5.6-sol", "object": "model"},
                {"id": "gpt-5.6-terra", "object": "model"}]})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):  # noqa: N802
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": {"message": "not found"}})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            t0 = time.monotonic()
            out = _completion_json(body)
            dt = time.monotonic() - t0
            calls = len(out["choices"][0]["message"].get("tool_calls") or [])
            print(f"[codex-proxy] {dt:6.1f}s  tool_calls={calls}", file=sys.stderr, flush=True)
            self._send(200, out)
        except Exception as exc:  # CLI failure/timeout -> 500; rollout's transport layer retries
            print(f"[codex-proxy] ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            self._send(500, {"error": {"message": f"{type(exc).__name__}: {exc}",
                                       "type": "proxy_upstream_error"}})

    def log_message(self, *args):
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="codex CLI proxy (OpenAI chat.completions shape)")
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)   # parallel eval: one thread per request, each spawning its own codex subprocess
    print(f"codex proxy listening on http://127.0.0.1:{a.port}/v1 "
          f"(codex exec, local login, effort={EFFORT})", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
