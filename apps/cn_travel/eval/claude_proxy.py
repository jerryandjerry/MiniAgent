#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenAI-compatible evaluation proxy for the logged-in Claude CLI.

The proxy sends the rendered conversation and tool schemas through Claude's
JSON-schema transport, then converts the structured response into OpenAI-format
assistant messages and tool calls.

    make claude-proxy PROXY_PORT=8765
    # then
    make eval EVAL_ARGS='--system claude-opus-5 --model claude-opus-5 \
        --base-url http://127.0.0.1:8765/v1'

The evaluator records this transport in ``run.json``. Claude CLI omits token
usage, so token and cost fields are zero for this channel.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from eval.channels.claude_cli import ClaudeCLIClient

_CLIENT = ClaudeCLIClient()


def _completion_json(body: dict) -> dict:
    resp = _CLIENT.chat.completions.create(
        model=body.get("model", ""), messages=body.get("messages") or [],
        tools=body.get("tools"), tool_choice=body.get("tool_choice"))
    m = resp.choices[0].message
    message: dict = {"role": "assistant", "content": m.content}
    if m.tool_calls:
        message["tool_calls"] = [
            {"id": c.id, "type": c.type,
             "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in m.tool_calls]
    return {"id": f"chatcmpl-proxy-{int(time.time() * 1000)}",
            "object": "chat.completion", "created": int(time.time()),
            "model": resp.model,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": resp.choices[0].finish_reason}]}


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
                {"id": "claude-opus-5", "object": "model"}]})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):  # noqa: N802
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": {"message": "not found"}})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            t0 = time.monotonic()
            out = self._completion(body)
            dt = time.monotonic() - t0
            calls = len(out["choices"][0]["message"].get("tool_calls") or [])
            print(f"[proxy] {dt:6.1f}s  tool_calls={calls}", file=sys.stderr, flush=True)
            self._send(200, out)
        except Exception as exc:  # CLI failure/timeout -> 500; rollout's transport layer retries
            print(f"[proxy] ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            self._send(500, {"error": {"message": f"{type(exc).__name__}: {exc}",
                                       "type": "proxy_upstream_error"}})

    def _completion(self, body: dict) -> dict:
        return _completion_json(body)

    def log_message(self, *args):  # silence the default access log (stderr already carries a one-line summary)
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="claude CLI proxy (OpenAI chat.completions shape)")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    srv = HTTPServer(("127.0.0.1", a.port), Handler)
    print(f"claude proxy listening on http://127.0.0.1:{a.port}/v1 "
          f"(claude --print, local login)", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
