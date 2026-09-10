#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§5.3 verifiable-reward GRPO initialized from the §5.1 SFT LoRA.

  * Data: the 3,632 contexts in the sealed train_validation.json (with the final assistant
    turn dropped), pre-rendered here into text prompts with the base template + tools
    (train == serve; the trainer never touches the template). The golden assistant turn is
    rendered the same way and passed through reward.parse_calls to produce the comparison
    baseline (same domain on both sides).
  * Policy: base + SFT adapter (trainable), so the output is still "original base + a single
    adapter" and the serving path is identical to §5.1. beta=0: GRPO needs no reference model.
  * Reward: reward.score_emission (the §5.3 formula, weights from train_config.json).
  * The 101-row eval set never takes part. Checkpointing + resume. Every hyperparameter comes
    from the config file.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))
from reward import parse_calls, score_emission  # noqa: E402


def normalize_tool_args(messages: list[dict]) -> list[dict]:
    """Qwen3.5's per-argument template rendering requires arguments to be a mapping; wire-format strings from earlier turns are parsed before rendering."""
    import copy as _copy
    import json as _json
    msgs = _copy.deepcopy(messages)
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or {}
            if isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = _json.loads(fn["arguments"])
                except ValueError:
                    pass
    return msgs


def golden_calls_of(msg: dict) -> list[dict]:
    return [{"tool": tc["function"]["name"],
             "args": json.loads(tc["function"]["arguments"])
             if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]}
            for tc in msg.get("tool_calls") or []]


def to_template_message(content: str, tool_calls: list[dict]) -> dict:
    """Build a template-renderable assistant message (arguments as a mapping, as Qwen3.5's template requires)."""
    msg: dict = {"role": "assistant", "content": content or ""}
    if tool_calls:
        msg["tool_calls"] = [{"id": c.get("id") or f"mined_{i}", "type": "function",
                              "function": {"name": c["tool"], "arguments": c["args"]}}
                             for i, c in enumerate(tool_calls)]
    return msg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO with verifiable reward (README §5.3)")
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--sft_adapter", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--tools_file", type=str, required=True)
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--kl_beta", type=float, default=0.0)
    p.add_argument("--learning_rate", type=float, default=2e-6)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--max_prompt_length", type=int, default=7168)
    p.add_argument("--max_completion_length", type=int, default=1024)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)
    # Size the colocated KV cache for the task context instead of the model maximum.
    p.add_argument("--vllm_max_model_length", type=int, default=8192)
    # Optionally cap large rollout-to-training importance ratios.
    p.add_argument("--vllm_importance_sampling_cap", type=float, default=0.0,
                   help="passed to GRPOConfig when >0; 0 keeps the TRL default")
    # Sleep mode lets rollout and optimization reuse single-GPU memory.
    p.add_argument("--vllm_enable_sleep_mode", action="store_true")
    p.add_argument("--w_call", type=float, default=1.0)
    p.add_argument("--w_struct", type=float, default=0.5)
    p.add_argument("--w_extra", type=float, default=0.2)
    p.add_argument("--w_format", type=float, default=1.0)
    p.add_argument("--clip_min", type=float, default=-1.0)
    p.add_argument("--clip_max", type=float, default=1.5)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--use_vllm", action="store_true")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--log_file", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    weights = {"w_call": args.w_call, "w_struct": args.w_struct,
               "w_extra": args.w_extra, "w_format": args.w_format,
               "clip_min": args.clip_min, "clip_max": args.clip_max}

    import inspect as _insp
    import math as _math

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from trl import GRPOConfig, GRPOTrainer

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                        local_files_only=args.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tools = json.loads(open(args.tools_file, encoding="utf-8").read())

    # Golden and sampled emissions use the same model-family response parser.
    parse_fn = parse_calls
    _fam = json.loads((pathlib.Path(args.model_name_or_path) / "config.json")
                      .read_text(encoding="utf-8")).get("model_type") or ""
    if _fam == "lfm2":
        import trl.chat_template_utils as _ctu
        try:
            _ctu.add_response_schema(tok)
        except ValueError:
            tok.response_template = _ctu.lfm2_2_5_template

        def parse_fn(text: str):
            try:
                parsed = tok.parse_response(text, prefix="")
            except Exception:
                return [], True
            calls = [{"tool": c["function"]["name"], "args": c["function"]["arguments"]}
                     for c in (parsed.get("tool_calls") or [])]
            return calls, ("<|tool_call_start|>" in text and not calls)
        print("emission parser: LFM2 response template")

    def render(messages, gen_prompt):
        return tok.apply_chat_template(messages, tools=tools, tokenize=False,
                                       add_generation_prompt=gen_prompt)

    # Pre-rendered dataset: prompt text + golden calls (same domain: render the golden turn, then parse_calls)
    data = json.loads(open(args.train_file, encoding="utf-8").read())
    GOVERNED_PREFIXES = 3632   # the §5.3 governed population
    if len(data) != GOVERNED_PREFIXES:
        raise SystemExit(f"train_file has {len(data)} rows != the governed population {GOVERNED_PREFIXES} - refusing to train")
    rows = []
    for s in data:
        conv = s["conversation"]
        context, golden_msg = normalize_tool_args(conv[:-1]), conv[-1]
        prompt = render(context, gen_prompt=True)
        golden_turn = to_template_message(golden_msg.get("content") or "",
                                          golden_calls_of(golden_msg))
        golden_text = render(context + [golden_turn], gen_prompt=False)[
            len(render(context, gen_prompt=False)):]
        golden_calls, bad = parse_fn(golden_text)
        assert not bad, f"golden turn failed to parse after rendering, idx={s['metadata']['idx']}"
        rows.append({"prompt": prompt,
                     "golden_calls_json": json.dumps(golden_calls, ensure_ascii=False)})
    ds = Dataset.from_list(rows)
    print(f"prompts: {len(ds)}")

    def verifiable_reward(completions, golden_calls_json, **kwargs):
        rewards = []
        for comp, gj in zip(completions, golden_calls_json):
            text = comp if isinstance(comp, str) else str(comp)
            rewards.append(score_emission(json.loads(gj), text, weights, parse_fn))
        return rewards

    # Load through the architecture view whose parameter names match colocated vLLM.
    from transformers import AutoModelForImageTextToText
    _arch = json.loads((pathlib.Path(args.model_name_or_path) / "config.json")
                       .read_text(encoding="utf-8")).get("architectures") or [""]
    if not _arch[0].endswith("ForConditionalGeneration"):
        # A text-only causal model (such as LFM2.5's Lfm2ForCausalLM): the colocated engine names
        # by the CausalLM graph, so a conditional-generation view would misalign weight sync every step.
        AutoModelForImageTextToText = AutoModelForCausalLM
    print(f"base architecture: {_arch[0]}")
    base = AutoModelForImageTextToText.from_pretrained(
        args.model_name_or_path, local_files_only=args.local_files_only,
        dtype=torch.bfloat16 if args.bf16 else None)
    model = PeftModel.from_pretrained(base, args.sft_adapter, is_trainable=True,
                                      local_files_only=args.local_files_only)
    model.print_trainable_parameters()
    if args.gradient_checkpointing:
        model.enable_input_require_grads()

    g_kwargs = dict(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        temperature=args.temperature,
        beta=args.kl_beta,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        use_vllm=args.use_vllm,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=args.vllm_max_model_length,
        **({"vllm_importance_sampling_cap": args.vllm_importance_sampling_cap}
           if args.vllm_importance_sampling_cap > 0 else {}),
        vllm_enable_sleep_mode=args.vllm_enable_sleep_mode,
        optim="adamw_torch",
        report_to=[],
        remove_unused_columns=False,
    )
    params = set(_insp.signature(GRPOConfig.__init__).parameters)
    if "warmup_ratio" not in params:
        steps = _math.ceil(len(ds) / (args.per_device_train_batch_size
                                      * args.gradient_accumulation_steps))
        g_kwargs["warmup_steps"] = _math.ceil(args.warmup_ratio * steps * args.num_train_epochs)
    dropped = sorted(k for k in g_kwargs if k not in params)
    if dropped:
        print(f"GRPOConfig: dropping unsupported args {dropped}")
    config = GRPOConfig(**{k: v for k, v in g_kwargs.items() if k in params})

    tr_kwargs = dict(model=model, args=config, reward_funcs=verifiable_reward,
                     train_dataset=ds)
    tr_params = set(_insp.signature(GRPOTrainer.__init__).parameters)
    tr_kwargs["processing_class" if "processing_class" in tr_params else "tokenizer"] = tok
    trainer = GRPOTrainer(**tr_kwargs)

    last_ckpt = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    if last_ckpt:
        print(f"resuming from checkpoint: {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)

    trainer.save_state()
    trainer.save_model(args.output_dir)

    # Successful delivery includes adapter weights and their SHA-256 identity.
    import hashlib as _hl
    out = pathlib.Path(args.output_dir)
    ad = next((out / n for n in ("adapter_model.safetensors", "adapter_model.bin")
               if (out / n).exists()), None)
    if ad is None:
        raise SystemExit(f"training finished but there is no adapter weight file under {out} - delivery incomplete, treated as failure")
    digest = _hl.sha256(ad.read_bytes()).hexdigest()
    (out / "adapter.sha256").write_text(digest + "\n")
    print("adapter sha256:", digest)
    print("Training complete. Adapter saved to:", args.output_dir)


if __name__ == "__main__":
    main()
