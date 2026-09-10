#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run CN Travel's application-local tool-calling and dependency loop."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from cn_travel.paths import CN_TRAVEL
from cn_travel.service.result import OK


class AgentError(RuntimeError):
    """A request failed because of an API error, rate limit, or timeout."""


@dataclass(frozen=True)
class Dependency:
    """Declare a prerequisite for a dependent tool.

    ``prerequisite`` must succeed before the dependent tool runs. When
    ``args_from_prereq`` is true, the dependent call must occur in a later
    model round so its arguments can use the prerequisite result.
    """

    prerequisite: str
    reason: str
    args_from_prereq: bool = False


def make_client(model_name: Optional[str] = None):
    """Build the configured chat-completions client and served model name."""
    from dotenv import load_dotenv

    from cn_travel.service.config import AppConfig

    load_dotenv(CN_TRAVEL.env_file, override=False)
    config = AppConfig.load(CN_TRAVEL.config).llm
    transport = os.getenv("LLM_TRANSPORT", config.provider).lower()

    if transport not in {"local", "openai", "openai_compat"}:
        raise ValueError(f"unsupported LLM transport: {transport}")

    from openai import OpenAI

    base_url = os.getenv("LLM_BASE_URL", config.base_url or "").rstrip("/")
    if not base_url:
        raise ValueError("LLM_BASE_URL or llm.base_url is required")
    api_key = os.getenv("LLM_API_KEY", "local")
    served_model = os.getenv("LLM_MODEL", model_name or config.model)
    return OpenAI(api_key=api_key, base_url=base_url), served_model


class Agent:
    """Run the tool-calling loop assembled by the application."""

    def __init__(
        self,
        *,
        system_prompt: str,
        tools: List[Dict[str, Any]],
        dispatch: Dict[str, Callable[[Dict[str, Any]], str]],
        depends_on: Optional[Dict[str, Dependency]] = None,
        model_name: Optional[str] = None,
        client: Any = None,
        history_limit: int = 20,
        verbose: bool = True,
    ):
        self.system_prompt = system_prompt
        self.tools = tools
        self.dispatch = dispatch
        self.depends_on = depends_on or {}
        self.history_limit = history_limit
        self.verbose = verbose
        self.conversation_history: List[Dict[str, str]] = []

        if client is None:
            client, model_name = make_client(model_name)
        self.client = client
        self.model_name = model_name

    # ------------------------------------------------------------------
    # History and serialization
    # ------------------------------------------------------------------
    def add_to_history(self, role: str, content: str) -> None:
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > self.history_limit:
            self.conversation_history = self.conversation_history[-self.history_limit:]

    @staticmethod
    def _serialise(payload: Dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, default=str)

    def _log(self, *args) -> None:
        if self.verbose:
            print(*args)

    # ------------------------------------------------------------------
    # Conditional dependencies must be enforced during dispatch
    # ------------------------------------------------------------------
    def _dependency_blocked(self, function_name: str, batch_status: dict) -> Optional[str]:
        """Return the reason a same-round dependent call must be blocked."""
        dep = self.depends_on.get(function_name)
        if not dep or dep.prerequisite not in batch_status:
            return None
        if batch_status[dep.prerequisite] != OK:
            return dep.reason
        if dep.args_from_prereq:
            return f"参数必须取自 {dep.prerequisite} 返回的结果"
        return None

    @property
    def _gating_tools(self) -> frozenset:
        """Return tools whose success can open another model round."""
        return frozenset(d.prerequisite for d in self.depends_on.values())

    def _should_continue_tool_chain(self, function_name: str, result: str) -> bool:
        """Continue only after a successful gating-tool result."""
        try:
            status = json.loads(result).get("status")
        except (json.JSONDecodeError, AttributeError):
            return False
        return status == OK and function_name in self._gating_tools

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def call_function(self, function_name: str, arguments: dict) -> str:
        """Dispatch one tool call and return its serialized result."""
        fn = self.dispatch.get(function_name)
        if fn is None:
            return f"未知函数: {function_name}"
        return fn(arguments)

    def _run_tool_calls(self, tool_calls, *, enforce_dependencies: bool):
        """Execute one model round and annotate each call result."""
        out, batch_status = [], {}
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            blocked = self._dependency_blocked(function_name, batch_status) if enforce_dependencies else None
            if blocked:
                self._log(f"⛔ 跳过工具: {function_name}（{blocked}）")
                result = json.dumps(
                    {
                        "status": "skipped",
                        "source": f"dependency not satisfied: {blocked}",
                        "next_action": (
                            "请先查看同一轮中前置工具的返回结果，"
                            "再用其中的真实值重新调用本工具"
                        ),
                    },
                    ensure_ascii=False,
                )
            else:
                self._log(f"🔧 调用工具: {function_name}")
                self._log(f"参数: {arguments}")
                result = self.call_function(function_name, arguments)
                self._log(f"🔧 工具结果: {result[:200]}{'...' if len(result) > 200 else ''}")

            try:
                batch_status[function_name] = json.loads(result).get("status")
            except (json.JSONDecodeError, AttributeError):
                batch_status[function_name] = None

            out.append(
                {
                    "call": tool_call,
                    "result": result,
                    "should_continue": self._should_continue_tool_chain(function_name, result),
                }
            )
        return out

    @staticmethod
    def _assistant_message(message) -> dict:
        return {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ],
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def process_user_input(self, user_input: str) -> str:
        self.add_to_history("user", user_input)
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name, messages=messages, tools=self.tools, tool_choice="auto"
            )
            message = response.choices[0].message

            if not message.tool_calls:
                content = message.content
                self.add_to_history("assistant", content)
                return content

            infos = self._run_tool_calls(message.tool_calls, enforce_dependencies=True)
            messages.append(self._assistant_message(message))
            for info in infos:
                messages.append(
                    {"role": "tool", "tool_call_id": info["call"].id, "content": info["result"]}
                )

            if any(i["should_continue"] for i in infos):
                second = self.client.chat.completions.create(
                    model=self.model_name, messages=messages, tools=self.tools, tool_choice="auto"
                ).choices[0].message
                if second.tool_calls:
                    # Later call arguments can see the first result, so allow them.
                    second_infos = self._run_tool_calls(second.tool_calls, enforce_dependencies=False)
                    messages.append(self._assistant_message(second))
                    for info in second_infos:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": info["call"].id,
                                "content": info["result"],
                            }
                        )

            final = self.client.chat.completions.create(model=self.model_name, messages=messages)
            final_content = final.choices[0].message.content
            self.add_to_history("assistant", final_content)
            return final_content

        except AgentError:
            raise
        except Exception as exc:
            raise AgentError(f"处理请求时出错: {exc}") from exc
