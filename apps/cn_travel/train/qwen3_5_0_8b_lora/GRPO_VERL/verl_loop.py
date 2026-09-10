#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trajectory-level VERL agent loop with one whole conversation per rollout.

The loop injects subsequent user turns and tool results from the sealed environment.
Those environment tokens carry ``response_mask=0``; optimization covers model-emitted
tokens. The shared environment supplies tool execution, parsing, and reward scoring.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from uuid import uuid4
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

_HERE = pathlib.Path(__file__).resolve().parent
# Import the shared frozen-world and reward implementation.
for _p in (str(_HERE), str(_HERE.parent / "GRPO_TRL")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import env as travel_env          # noqa: E402  frozen world + the 5 tool bodies + segment-level reward
from reward import parse_calls    # noqa: E402  Qwen3.5's <tool_call> XML emission parser

# Preserve business-schema order across training and serving.
_TOOL_SCHEMAS = json.loads(travel_env.TOOL_SCHEMAS.read_text(encoding="utf-8"))


@register("cn_travel")
class TravelAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mt = self.rollout_config.multi_turn
        self.max_assistant_turns = mt.max_assistant_turns or 24
        self.max_user_turns = mt.max_user_turns or 8
        self.response_length = self.rollout_config.response_length
        # reward weights come from the config file, with no implicit defaults (same convention as §5.2/§5.3)
        cfg_path = os.environ["CN_TRAVEL_VERL_CONFIG"]
        cfg = json.loads(pathlib.Path(cfg_path).read_text(encoding="utf-8"))
        travel_env.TravelEnv.W = {k: cfg[k] for k in
                                  ("w_call", "w_struct", "w_extra", "clip_min", "clip_max")}
        self.parse_fn = self._emission_parser(self.config.actor_rollout_ref.model.path)

    def _emission_parser(self, model_path: str):
        """Emission format varies by model family, so the parser is chosen from the model's own
        config.json (model_type) — the same criterion GRPO_TRL/train.py uses. No extra config
        switch, and no guessing.

        Qwen3.5：<tool_call><function=..><parameter=..>（reward.parse_calls）
        LFM2.5 uses <|tool_call_start|>[f(a='v')]<|tool_call_end|> (the tokenizer's response
        template). verl-env has no TRL, so try transformers' own parse_response first;
        if neither path works, fail loudly rather than silently falling back to the Qwen parser
        and generating a pile of bogus rewards.
        """
        fam = json.loads((pathlib.Path(model_path) / "config.json")
                         .read_text(encoding="utf-8")).get("model_type") or ""
        if fam != "lfm2":
            return parse_calls

        tok = self.tokenizer
        if not getattr(tok, "response_template", None):
            try:
                import trl.chat_template_utils as _ctu
                tok.response_template = _ctu.lfm2_2_5_template
            except Exception as e:
                raise RuntimeError(
                    f"LFM2 emission parser unavailable: tokenizer has no response_template and TRL cannot be imported ({e}). "
                    "Running LFM through Qwen's XML parser would make every reward wrong, so this fails hard.") from e

        def parse_lfm(text: str):
            try:
                parsed = tok.parse_response(text, prefix="")
            except Exception:
                return [], True
            calls = [{"tool": c["function"]["name"], "args": c["function"]["arguments"]}
                     for c in (parsed.get("tool_calls") or [])]
            return calls, ("<|tool_call_start|>" in text and not calls)

        return parse_lfm

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        extra = kwargs.get("extra_info", {}) or {}
        segments = extra["segments"]
        if isinstance(segments, str):
            segments = json.loads(segments)

        metrics: dict[str, Any] = {}
        request_id = uuid4().hex
        env = travel_env.TravelEnv()

        prompt_ids = await self.apply_chat_template(messages, tools=_TOOL_SCHEMAS)
        init_len = len(prompt_ids)
        response_mask: list[int] = []
        seg_rewards: list[float] = []
        assistant_turns = user_turns = 0

        seg_i = 0
        env._calls, env._golden = [], list(segments[0]["golden"])

        while True:
            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                )
            prompt_ids += output.token_ids
            response_mask += [1] * len(output.token_ids)
            assistant_turns += 1

            text = self.tokenizer.decode(output.token_ids)
            calls, _ = self.parse_fn(text)

            if len(response_mask) >= self.response_length or assistant_turns >= self.max_assistant_turns:
                seg_rewards.append(env.get_reward())
                break

            if calls:
                # tool results are environment output, mask 0; the world is the same frozen world as §6.2.
                # world lookups are synchronous: run them in the executor so the async loop is not blocked.
                with simple_timer("tool_calls", metrics):
                    results = await self.loop.run_in_executor(
                        None, lambda: [env._run(c["tool"], c["args"]) for c in calls])
                add = [{"role": "tool", "content": r} for r in results]
                delta = await self.apply_chat_template(add, remove_system_prompt=True)
                delta = self.turn_separator + delta
                if len(response_mask) + len(delta) >= self.response_length:
                    seg_rewards.append(env.get_reward())
                    break
                prompt_ids += delta
                response_mask += [0] * len(delta)
                continue

            # no call = this segment is answered: score it, then decide whether the person speaks again
            seg_rewards.append(env.get_reward())
            seg_i += 1
            if seg_i >= len(segments) or user_turns >= self.max_user_turns:
                break

            nxt = segments[seg_i]
            delta = await self.apply_chat_template(
                [{"role": "user", "content": nxt["user"]}], remove_system_prompt=True)
            delta = self.turn_separator + delta
            if len(response_mask) + len(delta) >= self.response_length:
                break
            prompt_ids += delta
            response_mask += [0] * len(delta)   # injected user turns stay out of the loss too
            user_turns += 1
            env._calls, env._golden = [], list(nxt["golden"])

        response_ids = prompt_ids[init_len:]
        reward = sum(seg_rewards) / max(1, len(seg_rewards))   # trajectory reward = mean over segments

        return AgentLoopOutput(
            prompt_ids=prompt_ids[:init_len],
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            reward_score=reward,
            num_turns=assistant_turns + user_turns + 1,
            metrics=metrics,
            extra_fields={"seg_rewards": seg_rewards,
                          "segments_done": len(seg_rewards),
                          "segments_total": len(segments)},
        )
