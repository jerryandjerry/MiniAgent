#!/usr/bin/env python3
"""VERL agent loop for one complete causal CN Travel episode per rollout.

The loop owns token generation and masking only.  The world-policy runtime owns
interpretation, person replies, tool execution, state transitions, violations,
terminal state, and the single complete-episode reward.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import sys
from typing import Any, Mapping
from uuid import uuid4

TRAIN_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = TRAIN_ROOT.parent / "src"
for _path in (TRAIN_ROOT, SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

from training.paths import TOOL_SCHEMAS, resolve_app_path
from world_policy import EpisodeRepository


HERE = pathlib.Path(__file__).resolve().parent
TOOLS = json.loads(TOOL_SCHEMAS.read_text(encoding="utf-8"))


def _resolve(value: str | os.PathLike[str]) -> pathlib.Path:
    return resolve_app_path(value)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return value


def _semantic_emission(decoded: str, tokenizer: Any) -> str:
    """Remove only the terminal transport EOS from text sent to the parser.

    The EOS token remains in ``response_ids`` and its policy-loss mask remains
    one.  It is transport framing, however, and must not turn an otherwise
    valid native LFM tool-call emission into mixed prose plus tool syntax.
    """
    eos = getattr(tokenizer, "eos_token", None)
    if isinstance(eos, str) and eos and decoded.endswith(eos):
        return decoded[:-len(eos)]
    return decoded


@register("cn_travel")
class TravelAgentLoop(AgentLoopBase):
    """Generate model turns and hand every emission to ``EpisodeRuntime.step``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config_path = pathlib.Path(os.environ["CN_TRAVEL_VERL_CONFIG"])
        self.task_config = json.loads(config_path.read_text(encoding="utf-8"))
        episodes_path = _resolve(self.task_config["episodes_file"])
        world_path = _resolve(self.task_config["world_file"])
        if _sha256(episodes_path) != self.task_config["episodes_sha256"]:
            raise RuntimeError("EpisodeSpec bytes differ from the sealed §5.4 config")
        if _sha256(world_path) != self.task_config["world_sha256"]:
            raise RuntimeError("Synthetic China bytes differ from the sealed §5.4 config")
        self.repository = EpisodeRepository.load(
            episodes_path,
            world_path,
            expected_count=int(self.task_config.get("expected_episode_count", 909)),
        )
        if self.repository.world.world_id != self.task_config["world_id"]:
            raise RuntimeError("Synthetic China world_id differs from the sealed §5.4 config")
        multi_turn = self.rollout_config.multi_turn
        self.max_assistant_turns = (
            multi_turn.max_assistant_turns or self.task_config["max_assistant_turns"]
        )
        self.response_length = self.rollout_config.response_length

    def _runtime(self, episode_id: str):
        return self.repository.new_runtime(
            episode_id,
            config={
                "reward": self.task_config["reward"],
                "max_rounds_per_user_turn": self.task_config[
                    "max_assistant_rounds_per_user_turn"
                ],
                "max_assistant_rounds": self.max_assistant_turns,
                "max_person_turns": self.task_config["max_user_turns"],
            },
        )

    @staticmethod
    def _mark_truncated(runtime: Any, reason: str) -> None:
        truncate = getattr(runtime, "truncate", None)
        if not callable(truncate):
            raise RuntimeError(
                "world-policy runtime lacks truncate(reason); truncation cannot be scored fail-closed"
            )
        truncate(reason)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra = kwargs.get("extra_info", {}) or {}
        episode_id = extra.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise RuntimeError("VERL row has no episode_id")

        runtime = self._runtime(episode_id)
        runtime.reset()
        raw_prompt = [dict(message) for message in kwargs["raw_prompt"]]
        expected_prompt = runtime.initial_messages
        if raw_prompt != expected_prompt:
            raise RuntimeError(f"{episode_id}: parquet prompt differs from sealed EpisodeSpec")

        metrics: dict[str, Any] = {}
        request_id = uuid4().hex
        prompt_ids = await self.apply_chat_template(raw_prompt, tools=TOOLS)
        initial_length = len(prompt_ids)
        response_mask: list[int] = []
        assistant_turns = 0
        environment_turns = 0
        terminal = False

        while not terminal:
            remaining = self.response_length - len(response_mask)
            if remaining <= 0:
                self._mark_truncated(runtime, "response_length")
                break
            turn_sampling = dict(sampling_params)
            configured_max = turn_sampling.get("max_tokens")
            turn_sampling["max_tokens"] = (
                min(int(configured_max), remaining)
                if configured_max is not None else remaining
            )
            generation_cap = int(turn_sampling["max_tokens"])
            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=turn_sampling,
                )
            emitted_ids = list(output.token_ids)
            accepted_ids = emitted_ids[:remaining]
            prompt_ids.extend(accepted_ids)
            response_mask.extend([1] * len(accepted_ids))
            assistant_turns += 1

            # Never award a milestone from an emission that VERL cannot retain
            # in response_ids.  Hitting the cap is a complete-episode
            # truncation, even if the backend returned more tokens than asked.
            if len(emitted_ids) >= generation_cap:
                reason = ("response_length" if generation_cap == remaining
                          else "model_output_length")
                self._mark_truncated(runtime, reason)
                break

            decoded = self.tokenizer.decode(accepted_ids, skip_special_tokens=False)
            emission = _semantic_emission(decoded, self.tokenizer)
            transition = runtime.interpret_and_step(emission)
            terminal = bool(transition.terminal)

            if terminal:
                break
            if len(response_mask) >= self.response_length:
                self._mark_truncated(runtime, "response_length")
                break
            if assistant_turns >= self.max_assistant_turns:
                self._mark_truncated(runtime, "assistant_turn_limit")
                break

            environment_messages = [dict(message) for message in transition.messages]
            if not environment_messages:
                raise RuntimeError(
                    f"{episode_id}: non-terminal transition produced no environment message"
                )
            with simple_timer("environment", metrics):
                delta = await self.apply_chat_template(
                    environment_messages,
                    remove_system_prompt=True,
                )
            delta = self.turn_separator + delta
            if len(response_mask) + len(delta) > self.response_length:
                self._mark_truncated(runtime, "environment_would_exceed_response_length")
                break
            prompt_ids.extend(delta)
            response_mask.extend([0] * len(delta))
            environment_turns += len(environment_messages)

        response_ids = prompt_ids[initial_length:]
        breakdown = runtime.reward()
        reward = float(breakdown.total)
        outcome = _plain(runtime.outcome())
        breakdown_data = _plain(breakdown)

        return AgentLoopOutput(
            prompt_ids=prompt_ids[:initial_length],
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            reward_score=reward,
            num_turns=assistant_turns + environment_turns + 1,
            metrics=metrics,
            extra_fields={
                "episode_id": episode_id,
                "assistant_turns": assistant_turns,
                "environment_turns": environment_turns,
                "reward_breakdown": breakdown_data,
                "outcome": outcome,
            },
        )


__all__ = ["TravelAgentLoop", "_semantic_emission"]
