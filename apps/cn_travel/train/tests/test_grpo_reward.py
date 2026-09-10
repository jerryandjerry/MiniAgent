#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline tests for the §5.3 verifiable GRPO reward."""
import json
import sys

import pytest

from cn_travel.paths import CN_TRAVEL
from project_paths import DATA_ROOT, EVAL_ROOT, TRAIN_ROOT

BASE_MODELS_DIR = TRAIN_ROOT / "base"
sys.path.insert(0, str(TRAIN_ROOT / "qwen3_5_0_8b_lora" / "GRPO_TRL"))
sys.path.insert(0, str(EVAL_ROOT))
import reward as rw  # noqa: E402
from train import normalize_tool_args  # noqa: E402

W = {"w_call": 1.0, "w_struct": 0.5, "w_extra": 0.2, "w_format": 1.0,
     "clip_min": -1.0, "clip_max": 1.5}

CALL = ("<tool_call>\n<function=get_weather_info>\n"
        "<parameter=location>\n嘉兴\n</parameter>\n"
        "<parameter=start_date>\n2026-08-25\n</parameter>\n"
        "<parameter=num_days>\n3\n</parameter>\n</function>\n</tool_call>")
GOLDEN, BAD = rw.parse_calls(CALL)
assert not BAD and GOLDEN == [{"tool": "get_weather_info",
                               "args": {"location": "嘉兴", "start_date": "2026-08-25",
                                        "num_days": "3"}}]


def test_exact_match_gets_full_reward():
    assert rw.score_emission(GOLDEN, f"<think>\n\n</think>\n\n{CALL}", W) == 1.5


def test_wrong_arg_loses_call_and_structure():
    wrong = CALL.replace("嘉兴", "嘉善")
    assert rw.score_emission(GOLDEN, wrong, W) == 0.0


def test_extra_call_is_penalized():
    extra = CALL + "\n" + CALL.replace("get_weather_info", "search_travel_guide")
    r = rw.score_emission(GOLDEN, extra, W)
    assert r == 1.0 - 0.2                      # Correct plus one extra call: no structure score, extra penalty


def test_missing_call_scores_partial():
    two_golden = GOLDEN + [{"tool": "search_travel_guide", "args": {"location": "嘉兴"}}]
    assert rw.score_emission(two_golden, CALL, W) == 0.5


def test_zero_call_golden_rewards_silence_and_punishes_calls():
    assert rw.score_emission([], "好的，已经帮您安排。", W) == 1.5
    assert rw.score_emission([], CALL, W) == -0.2


def test_unparseable_block_hits_format_penalty():
    broken = "<tool_call>\nget_weather_info(嘉兴)\n</tool_call>"
    assert rw.score_emission(GOLDEN, broken, W) == -1.0


def test_unclosed_tool_call_is_format_failure_not_silence():
    """Unclosed tool-call markup is a format failure, including zero-call targets."""
    assert rw.score_emission([], "<tool_call><function=x>", W) == -1.0
    assert rw.score_emission([], "好的。<tool_call>", W) == -1.0
    assert rw.score_emission(GOLDEN, CALL + "\n<tool_call><function=y>", W) < 1.5


def test_truncated_and_deformed_tags_are_format_failures():
    """Truncated and deformed tool-call delimiters are format failures."""
    assert rw.score_emission([], "<tool_call", W) == -1.0
    assert rw.score_emission([], "<tool_call ><function=x>好的</function></tool_call>", W) == -1.0
    assert rw.score_emission([], "好的</tool_call>", W) == -1.0


def test_whitespace_distorted_wrappers_are_format_failures():
    """Whitespace-distorted tool-call wrappers are format failures."""
    assert rw.score_emission([], "< tool_call>好的</ tool_call>", W) == -1.0
    assert rw.score_emission([], "<\ntool_call>好的</\ntool_call>", W) == -1.0
    assert rw.score_emission([], "</ tool_call>", W) == -1.0
    assert rw.score_emission(GOLDEN, CALL + "\n< tool_call>x", W) < 1.5


def test_multiple_sibling_functions_in_one_wrapper_are_format_failure():
    """Each tool-call wrapper contains exactly one function."""
    two = ("<tool_call>\n<function=x>\n<parameter=a>\n1\n</parameter>\n</function>\n"
           "<function=y>\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_call>")
    golden = [{"tool": "x", "args": {"a": "1"}}, {"tool": "y", "args": {"b": "2"}}]
    r = rw.score_emission(golden, two, W)
    assert r < 1.5 and r == 1.0 - 1.0            # Keep match score; reject structure and format


def test_duplicate_parameter_names_are_format_failure():
    """Duplicate parameter names are ambiguous and fail format validation."""
    dup = ("<tool_call>\n<function=x>\n<parameter=a>\nbad\n</parameter>\n"
           "<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>")
    golden = [{"tool": "x", "args": {"a": "1"}}]
    r = rw.score_emission(golden, dup, W)
    assert r < 1.5


def test_malformed_content_inside_recognized_block_is_format_failure():
    """Recognized wrappers reject unclosed or residual parameter content."""
    bad = ("<tool_call><function=x><parameter=a>\n1\n</parameter>"
           "<parameter=junk></function></tool_call>")
    golden = [{"tool": "x", "args": {"a": "1"}}]
    r = rw.score_emission(golden, bad, W)
    assert r < 1.5 and r == 1.0 - 1.0        # Keep match score; no structure score, format penalty


def test_shi_strip_equivalence_still_holds():
    para = CALL.replace("嘉兴", "嘉兴市")
    assert rw.score_emission(GOLDEN, para, W) == 1.5


@pytest.mark.model
def test_golden_roundtrip_through_real_template():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(BASE_MODELS_DIR / "Qwen3.5-0.8B"),
                                        local_files_only=True)
    tools = json.loads(CN_TRAVEL.tool_schemas.read_text(encoding="utf-8"))
    data = json.loads((DATA_ROOT / "6_multiturn" / "train_validation.json")
                      .read_text(encoding="utf-8"))
    checked = 0
    for s in data[:50]:
        conv = s["conversation"]
        golden_msg = conv[-1]
        if not golden_msg.get("tool_calls"):
            continue
        ctx = normalize_tool_args(conv[:-1])
        msg = {"role": "assistant", "content": golden_msg.get("content") or "",
               "tool_calls": [{"id": "t", "type": "function",
                               "function": {"name": tc["function"]["name"],
                                            "arguments": json.loads(tc["function"]["arguments"])}}
                              for tc in golden_msg["tool_calls"]]}
        full = tok.apply_chat_template(ctx + [msg], tools=tools, tokenize=False,
                                       add_generation_prompt=False)
        prefix = tok.apply_chat_template(ctx, tools=tools, tokenize=False,
                                         add_generation_prompt=False)
        golden_text = full[len(prefix):]
        calls, bad = rw.parse_calls(golden_text)
        assert not bad and len(calls) == len(golden_msg["tool_calls"])
        assert rw.score_emission(calls, golden_text, W) == 1.5, "金标必须得满分"
        checked += 1
    assert checked >= 5
