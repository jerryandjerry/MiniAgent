#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Qwen3.5 chat templating, masking, and the shipped SFT adapter."""
import json

import pytest

from project_paths import DATA_ROOT, TRAIN_ROOT

BASE_MODEL = TRAIN_ROOT / "base" / "Qwen3.5-0.8B"
SFT_DIR = TRAIN_ROOT / "qwen3_5_0_8b_lora" / "SFT"
TRAIN_DATA = DATA_ROOT / "6_multiturn" / "train_validation.json"
CONVERSATIONS = DATA_ROOT / "5_merged" / "train_validation.json"


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        str(BASE_MODEL), local_files_only=True, trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture
def dataset(tokenizer, all_tools):
    from training.masking import JsonlConversations

    return JsonlConversations(
        str(TRAIN_DATA),
        tokenizer, 20000, True,
        all_tools,
    )


# ------------------------------------------------------------- templating --
@pytest.mark.model
def test_template_accepts_a_tools_kwarg(tokenizer):
    from training.masking import _supports_tools_kw

    assert _supports_tools_kw(tokenizer)


@pytest.mark.model
def test_tool_schemas_land_in_the_prompt(tokenizer, all_tools):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}], tools=all_tools,
        tokenize=False, add_generation_prompt=True,
    )
    for t in all_tools:
        assert t["function"]["name"] in text


@pytest.mark.model
def test_tool_calls_render_as_tool_call_blocks(tokenizer):
    msgs = [
        {"role": "user", "content": "去嘉兴"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "search_travel_guide",
                          "arguments": {"location": "嘉兴"}}}]},
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    assert "<tool_call>" in text and "search_travel_guide" in text


# ----------------------------------------------------------------- masking --
@pytest.mark.model
def test_dataset_size(dataset):
    assert len(dataset) == 3632


@pytest.mark.model
def test_only_one_contiguous_loss_span_per_sample(dataset):
    from training.masking import _contiguous_true_spans

    for i in (0, 1, 2, 5, 100, 1000, 3631):
        spans = _contiguous_true_spans(dataset[i]["labels"].ne(-100))
        assert len(spans) == 1, f"sample {i}: {len(spans)} spans, expected 1"


@pytest.mark.model
def test_loss_span_is_the_final_assistant_turn(dataset, tokenizer):
    for i in (0, 1, 2, 100):
        s = dataset[i]
        span = s["labels"][s["labels"].ne(-100)]
        text = tokenizer.decode(span, skip_special_tokens=False)
        assert text.startswith("<|im_start|>assistant")
        assert "<|im_end|>" in text


@pytest.mark.model
def test_tool_call_turns_are_supervised(dataset, tokenizer, multiturn_v2):
    idx = next(i for i, s in enumerate(multiturn_v2)
               if s["conversation"][-1].get("tool_calls"))
    s = dataset[idx]
    text = tokenizer.decode(s["labels"][s["labels"].ne(-100)], skip_special_tokens=False)
    assert "<tool_call>" in text


@pytest.mark.model
def test_context_is_fully_masked_out(dataset):
    for i in (0, 5, 100):
        s = dataset[i]
        n_loss = int(s["labels"].ne(-100).sum())
        assert 0 < n_loss < s["input_ids"].numel() * 0.5


@pytest.mark.model
def test_shapes_line_up(dataset):
    s = dataset[0]
    n = s["input_ids"].numel()
    assert s["attention_mask"].numel() == n and s["labels"].numel() == n


@pytest.mark.model
def test_full_supervision_mode_covers_more_turns(tokenizer, all_tools):
    """only_last_assistant=False must supervise every assistant turn."""
    from training.masking import JsonlConversations, _contiguous_true_spans

    ds = JsonlConversations(str(CONVERSATIONS),
                            tokenizer, 20000, False, all_tools)
    multi = next(i for i in range(50)
                 if len(_contiguous_true_spans(ds[i]["labels"].ne(-100))) > 1)
    assert multi is not None


@pytest.mark.model
def test_collator_pads_with_the_right_sentinels(dataset, tokenizer):
    from training.masking import DataCollatorForCausal

    batch = DataCollatorForCausal(tokenizer)([dataset[0], dataset[1]])
    assert batch["input_ids"].shape == batch["labels"].shape
    assert (batch["labels"] == -100).any()
    shorter = min(dataset[0]["input_ids"].numel(), dataset[1]["input_ids"].numel())
    assert batch["attention_mask"][:, shorter:].min().item() in (0, 1)


def test_every_assistant_turn_contains_at_most_one_tool_call(
    train_conversations, test_conversations, multiturn_v2
):
    for split_name, samples in (
        ("train", train_conversations),
        ("evaluation", test_conversations),
        ("multiturn", multiturn_v2),
    ):
        for sample_no, sample in enumerate(samples):
            for turn_no, message in enumerate(sample["conversation"]):
                if message.get("role") != "assistant":
                    continue
                calls = message.get("tool_calls") or []
                assert len(calls) <= 1, (
                    f"{split_name} sample {sample_no}, turn {turn_no} contains "
                    f"{len(calls)} tool calls"
                )


# --------------------------------------------------------------- adapters --
@pytest.mark.parametrize(
    "variant,target_pattern",
    [
        (
            "adapter",
            r"^model\.layers\.\d+\."
            r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
            r"mlp\.(gate_proj|up_proj|down_proj))$",
        ),
        (
            "adapter_vllm",
            r"^model\.language_model\.layers\.\d+\."
            r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
            r"mlp\.(gate_proj|up_proj|down_proj))$",
        ),
    ],
)
def test_lora_adapter_config_is_intact(variant, target_pattern):
    d = SFT_DIR / variant
    cfg = json.loads((d / "adapter_config.json").read_text(encoding="utf-8"))
    assert cfg["peft_type"] == "LORA" and cfg["task_type"] == "CAUSAL_LM"
    assert cfg["r"] == 32 and cfg["lora_alpha"] == 64
    assert cfg["target_modules"] == target_pattern
    assert (d / "adapter_model.safetensors").is_file()


@pytest.mark.parametrize("variant", ["adapter", "adapter_vllm"])
def test_sft_trained_on_the_full_multiturn_file(variant):
    """454 optimizer steps == ceil(3,632 / (batch 1 * grad-accum 8))."""
    s = json.loads(
        (SFT_DIR / variant / "trainer_state.json").read_text(encoding="utf-8")
    )
    assert s["global_step"] == 454 and s["num_train_epochs"] == 1


def test_launcher_uses_the_sealed_training_config():
    config = json.loads((SFT_DIR / "train_config.json").read_text(encoding="utf-8"))
    launcher = (SFT_DIR / "run.sh").read_text(encoding="utf-8")
    assert config["train_file"] == "data/6_multiturn/train_validation.json"
    assert config["model_name_or_path"] == "train/base/Qwen3.5-0.8B"
    assert config["learning_rate"] == 2e-5
    assert config["num_train_epochs"] == 1
    assert "train_config.json" in launcher
