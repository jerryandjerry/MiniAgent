"""Chat templating and assistant-only loss masking for CN Travel SFT."""

from __future__ import annotations

import inspect
import json
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer


def _flat_token_ids(value: Any) -> List[int]:
    """Normalize Transformers 4.x/5.x chat-template output to one token list."""
    if hasattr(value, "keys") and "input_ids" in value:
        value = value["input_ids"]
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)


def _supports_tools_kw(tokenizer: AutoTokenizer) -> bool:
    try:
        return any(
            parameter.name == "tools"
            for parameter in inspect.signature(tokenizer.apply_chat_template).parameters.values()
        )
    except Exception:
        return False


class JsonlConversations:
    """Load JSON/JSONL conversations and supervise assistant spans only."""

    def __init__(
        self,
        path: str,
        tokenizer: AutoTokenizer,
        max_seq_length: int,
        only_last_assistant: bool = False,
        default_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples: List[Dict[str, Any]] = []
        self._default_tools = default_tools if default_tools else None
        self.only_last_assistant = only_last_assistant

        def normalize(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            if not isinstance(raw, dict):
                return None
            messages = raw.get("messages")
            if not isinstance(messages, list):
                messages = raw.get("conversation")
            if not isinstance(messages, list):
                return None
            messages = [dict(message) for message in messages]
            for message in messages:
                calls = []
                for call in message.get("tool_calls") or []:
                    call = dict(call)
                    function = dict(call.get("function") or {})
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            function["arguments"] = json.loads(arguments)
                        except ValueError:
                            pass
                    call["function"] = function
                    calls.append(call)
                if calls:
                    message["tool_calls"] = calls
            normalized: Dict[str, Any] = {"messages": messages}
            tools = raw.get("tools")
            if isinstance(tools, list):
                normalized["tools"] = tools
            elif self._default_tools is not None:
                normalized["tools"] = self._default_tools
            return normalized

        content = open(path, encoding="utf-8").read()
        parsed = False
        if content.lstrip().startswith("["):
            try:
                values = json.loads(content)
                if isinstance(values, list):
                    self.examples.extend(
                        item for raw in values if (item := normalize(raw)) is not None
                    )
                    parsed = bool(self.examples)
            except Exception:
                parsed = False
        if not parsed:
            for line in content.splitlines():
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except Exception:
                    continue
                item = normalize(raw)
                if item is not None:
                    self.examples.append(item)
        if not self.examples:
            raise ValueError(
                "No valid samples found; expected JSON/JSONL objects containing "
                "a messages or conversation list"
            )
        self._use_tools_kw = _supports_tools_kw(tokenizer)

    def __len__(self) -> int:
        return len(self.examples)

    def _apply_template(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> List[int]:
        kwargs: Dict[str, Any] = {"tokenize": True, "add_generation_prompt": False}
        if tools and self._use_tools_kw:
            kwargs["tools"] = tools
        return _flat_token_ids(self.tokenizer.apply_chat_template(messages, **kwargs))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        example = self.examples[index]
        messages = example["messages"]
        tools = example.get("tools")
        full_ids = self._apply_template(messages, tools)
        spans: List[Tuple[int, int]] = []
        previous_length = 0
        for position, message in enumerate(messages):
            try:
                current_length = len(self._apply_template(messages[: position + 1], tools))
            except Exception:
                if message.get("role") == "assistant":
                    raise
                current_length = previous_length
            if message.get("role") == "assistant" and current_length > previous_length:
                spans.append((previous_length, current_length))
            previous_length = current_length

        assistant_mask = torch.zeros(len(full_ids), dtype=torch.bool)
        selected = spans[-1:] if self.only_last_assistant else spans
        for start, end in selected:
            assistant_mask[start:end] = True

        if len(full_ids) > self.max_seq_length:
            overflow = len(full_ids) - self.max_seq_length
            full_ids = full_ids[overflow:]
            assistant_mask = assistant_mask[overflow:]

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[~assistant_mask] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels,
        }


class DataCollatorForCausal:
    """Pad input, attention, and label tensors for causal-language-model SFT."""

    def __init__(
        self, tokenizer: AutoTokenizer, pad_to_multiple_of: Optional[int] = None
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        def pad(values: List[torch.Tensor], fill: int) -> torch.Tensor:
            result = torch.nn.utils.rnn.pad_sequence(
                values, batch_first=True, padding_value=fill
            )
            if self.pad_to_multiple_of:
                missing = (-result.size(1)) % self.pad_to_multiple_of
                if missing:
                    extra = torch.full(
                        (result.size(0), missing),
                        fill,
                        dtype=result.dtype,
                        device=result.device,
                    )
                    result = torch.cat((result, extra), dim=1)
            return result

        return {
            "input_ids": pad(
                [feature["input_ids"] for feature in features],
                self.tokenizer.pad_token_id,
            ),
            "attention_mask": pad(
                [feature["attention_mask"] for feature in features], 0
            ),
            "labels": pad([feature["labels"] for feature in features], -100),
        }


def _contiguous_true_spans(mask: torch.Tensor) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for position, value in enumerate(mask.tolist()):
        if value and start is None:
            start = position
        elif not value and start is not None:
            spans.append((start, position))
            start = None
    if start is not None:
        spans.append((start, mask.numel()))
    return spans
