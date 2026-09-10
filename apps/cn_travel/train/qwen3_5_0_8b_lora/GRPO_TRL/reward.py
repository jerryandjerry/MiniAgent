#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§5.3 GRPO verifiable reward: deterministic, a pure function of sealed data + emitted text.

Adjudication happens entirely on the "template-rendered text" side: the golden assistant
turn and the student completion both go through the same XML parser to extract
<tool_call><function=..><parameter=..> calls (argument values are always strings), and are
then compared position by position with §6.2's _compare (strict equality + registry-endorsed
trailing-"市" removal). Both sides share one domain, so there is no type mismatch.

  R = w_call   * calls_correct / max(1, calls_required)     exact per-call match
    + w_struct * structure_ok                               one turn, ordered, fully identical
    - w_extra  * calls_extra                                calls beyond golden
    - w_format * unparseable                                contains an unparseable <tool_call>

Zero-call golden (W4/W5/clarification turns): emitting no call makes the first term 1.0 with
structure_ok=1 - silence scores full marks exactly where the contract asks for it. Weights come
from train_config.json (config decides). The reward is used on the training side only; evaluation
still runs through §6.2's stock vLLM channel, and neither crosses into the other.
"""
from __future__ import annotations

import pathlib
import re
import sys

TRAIN_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

from training.tool_matching import compare_args as _compare  # noqa: E402

_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_LOOSE_TAG = re.compile(r"<\s*/?\s*tool_call")
_FUNCTION = re.compile(r"<function=([^>\s]+)>\s*(.*?)\s*</function>", re.S)
_PARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


def parse_calls(text: str) -> tuple[list[dict], bool]:
    """Extract the call sequence from rendered text. Returns (calls, unparseable).

        unparseable = a <tool_call> block appeared but no valid <function=...> could be pulled
        out of it (malformed). Argument values stay strings - golden is parsed the same way, so
        both sides compare in one domain.
    """
    calls: list[dict] = []
    unparseable = False
    blocks = _TOOL_CALL.findall(text)
    # Every call attempt must parse completely and unambiguously:
    #   * deformed, truncated or stray wrapper tags: "< tool_call>", "<\ntool_call>", "</ tool_call>",
    #     "<tool_call" and so on - the loose count (allowing whitespace after '<' and beside '/')
    #     must exactly equal the open+close tag count of the parseable blocks (2 x block count);
    #   * multiple sibling <function> nodes inside one wrapper: the wrapper structure is bad
    #     (match credit is kept, structure/format credit refused);
    #   * duplicate parameter names: ambiguous content, equally malformed;
    #   * residue inside a block or a function body: after stripping matched fragments only whitespace may remain.
    loose = len(_LOOSE_TAG.findall(text))
    if loose != 2 * len(blocks):
        unparseable = True
    for block in blocks:
        fns = _FUNCTION.findall(block)
        block_residue = _FUNCTION.sub("", block).strip()
        if not fns or block_residue or len(fns) != 1:
            unparseable = True
        for name, body in fns:
            pairs = _PARAM.findall(body)
            body_residue = _PARAM.sub("", body).strip()
            if body_residue or len(pairs) != len({k for k, _ in pairs}):
                unparseable = True
            calls.append({"tool": name, "args": {k: v for k, v in pairs}})
    return calls, unparseable


def score_emission(golden_calls: list[dict], completion: str, w: dict, parse=None) -> float:
    """The §5.3 reward formula. golden_calls also comes from the same parse(golden rendered
        text), so both sides share a domain.

        parse defaults to this file's XML parser (Qwen3.5's emission format). Other model families
        emit differently (LFM2.5 uses <|tool_call_start|>[f(a=1)]<|tool_call_end|>), so the caller
        passes that model's own parser; the scoring logic (positional pairing + _compare + penalties
        for extra calls and bad format) is unchanged.
    """
    made, unparseable = (parse or parse_calls)(completion)
    required = len(golden_calls)

    correct = 0
    used = [False] * len(made)
    for i, g in enumerate(golden_calls):        # ordered, position by position (within a single turn LCS degenerates to a positional
        if i < len(made):                        # prefix comparison, consistent with §6.2's single-turn structural check)
            m = made[i]
            if m["tool"] == g["tool"] and _compare(g["args"], m["args"]) is None:
                correct += 1
                used[i] = True
    extra = max(0, len(made) - required)   # "beyond golden" is counted by quantity: a wrong-argument call is not penalized twice
    # Structure and zero-call credit both require a fully parseable emission.
    structure_ok = (not unparseable) and len(made) == required and correct == required
    silence = (not made) and (not unparseable)

    r = (w["w_call"] * (correct / max(1, required) if required else (1.0 if silence else 0.0))
         + w["w_struct"] * (1.0 if structure_ok else 0.0)
         - w["w_extra"] * extra
         - w["w_format"] * (1.0 if unparseable else 0.0))
    return max(w["clip_min"], min(w["clip_max"], r))   # clip bounds must come from config; no implicit defaults
