#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§5.3 per-user-turn GRPO: one training unit = one user segment, with the model walking
the tool chain itself.

The one structural difference from §5.2 (one emission per step, history always golden):
the generation loop is driven by TRL's environment_factory - the model calls a tool, the
frozen world answers, and the model continues against the context it created itself, until
it answers without a call or hits max_tool_calling_iterations. The reward comes from the
environment's get_reward (env.py, same standard as §6.2), and TRL masks tool-result tokens
out of the loss.

Difficulty filtering (optional): --task_filter points at the JSON produced by
probe_difficulty.py, so training covers only the segments the current policy does not yet
get reliably right, plus a few solved anchors (anti-forgetting).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))
from env import TravelEnv, build_episodes, install_schema_bridge  # noqa: E402
from env import TOOL_SCHEMAS as CN_TRAVEL_TOOLS  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trajectory-level GRPO (README §5.3)")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--sft_adapter", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--limit", type=int, default=0, help="smoke testing: take only the first N segments")
    p.add_argument("--task_filter", type=str, default="",
                   help="the segment list produced by probe_difficulty.py; empty = all 1,018 segments")
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--kl_beta", type=float, default=0.0)
    p.add_argument("--learning_rate", type=float, default=2e-6)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--max_prompt_length", type=int, default=7168)
    p.add_argument("--max_completion_length", type=int, default=1024)
    p.add_argument("--max_tool_calling_iterations", type=int, default=6)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=10)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.75)
    p.add_argument("--vllm_max_model_length", type=int, default=8192)
    p.add_argument("--vllm_importance_sampling_cap", type=float, default=0.0)
    p.add_argument("--epsilon_high", type=float, default=0.0,
                   help="enables clip-higher (DAPO) when >0; 0 = TRL's default symmetric clipping")
    p.add_argument("--loss_type", type=str, default="")
    p.add_argument("--mask_truncated_completions", action="store_true")
    p.add_argument("--w_call", type=float, default=1.0)
    p.add_argument("--w_struct", type=float, default=0.5)
    p.add_argument("--w_extra", type=float, default=0.2)
    p.add_argument("--clip_min", type=float, default=-1.0)
    p.add_argument("--clip_max", type=float, default=1.5)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--use_vllm", action="store_true")
    p.add_argument("--vllm_enable_sleep_mode", action="store_true")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--log_file", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    import inspect as _insp
    import math as _math

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import (AutoModelForCausalLM, AutoModelForImageTextToText,
                              AutoTokenizer, set_seed)
    from transformers.trainer_utils import get_last_checkpoint
    from trl import GRPOConfig, GRPOTrainer

    set_seed(args.seed)
    install_schema_bridge()      # the rendered tool schemas equal the business originals (train == serve)
    TravelEnv.W = {"w_call": args.w_call, "w_struct": args.w_struct,
                   "w_extra": args.w_extra,
                   "clip_min": args.clip_min, "clip_max": args.clip_max}

    episodes = build_episodes()
    if args.task_filter:
        keep = {(e["idx"], e["seg_no"]) for e in
                json.loads(pathlib.Path(args.task_filter).read_text(encoding="utf-8"))["train_on"]}
        episodes = [e for e in episodes if (e["idx"], e["seg_no"]) in keep]
        if not episodes:
            raise SystemExit("training set is empty after difficulty filtering - the filter list does not match the corpus")
    if args.limit:
        episodes = episodes[: args.limit]
    ds = Dataset.from_list(episodes)
    print(f"episodes: {len(ds)}  (segments; filter={'on' if args.task_filter else 'off'})")

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                        local_files_only=args.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Install a compatible response parser and verify golden-call round trips before training.
    import trl.chat_template_utils as _ctu
    try:
        _ctu.add_response_schema(tok)
    except ValueError:
        fam = (json.loads((pathlib.Path(args.model_name_or_path) / "config.json")
                          .read_text(encoding="utf-8")).get("model_type") or "")
        if fam != "lfm2":
            raise
        tok.response_template = _ctu.lfm2_2_5_template
        probe_calls = [{"tool": "recommend_hotels", "args": {"location": "北京", "requirements": ""}},
                       {"tool": "get_hotel_reviews", "args": {"hotel_name": "国贸大酒店", "location": "北京"}}]
        _tools = json.loads(CN_TRAVEL_TOOLS.read_text(encoding="utf-8"))
        _msg = {"role": "assistant", "content": "",
                "tool_calls": [{"id": f"c{i}", "type": "function",
                                "function": {"name": c["tool"], "arguments": c["args"]}}
                               for i, c in enumerate(probe_calls)]}
        _pre = tok.apply_chat_template([{"role": "user", "content": "x"}], tools=_tools, tokenize=False)
        _full = tok.apply_chat_template([{"role": "user", "content": "x"}, _msg],
                                        tools=_tools, tokenize=False)
        _got = [{"tool": c["function"]["name"], "args": c["function"]["arguments"]}
                for c in (tok.parse_response(_full[len(_pre):], prefix=_pre).get("tool_calls") or [])]
        if _got != probe_calls:
            raise SystemExit(f"LFM2 response-template round-trip check failed: parsed back {_got} != golden {probe_calls}")
        print("response template: TRL lfm2_2_5_template (round-trip verified)")

    # Match the loading view to the checkpoint architecture and vLLM parameter names.
    _arch = json.loads((pathlib.Path(args.model_name_or_path) / "config.json")
                       .read_text(encoding="utf-8")).get("architectures") or [""]
    _cond = _arch[0].endswith("ForConditionalGeneration")
    print(f"base architecture: {_arch[0]} -> "
          f"{'AutoModelForImageTextToText' if _cond else 'AutoModelForCausalLM'}")
    _loader = AutoModelForImageTextToText if _cond else AutoModelForCausalLM
    base = _loader.from_pretrained(
        args.model_name_or_path, local_files_only=args.local_files_only,
        dtype=torch.bfloat16 if args.bf16 else None)
    model = PeftModel.from_pretrained(base, args.sft_adapter, is_trainable=True,
                                      local_files_only=args.local_files_only)
    if args.kl_beta > 0:
        # KL uses the frozen §5.1 SFT adapter as its reference policy.
        # Verify this TRL release supports the named reference adapter before training.
        import inspect as _i

        import trl.trainer.grpo_trainer as _g
        if 'adapter_name="ref"' not in _i.getsource(_g):
            raise SystemExit(
                "this TRL's GRPO does not honor the \"ref\" adapter convention: kl_beta>0 would degrade the "
                "reference policy to the bare base. Set kl_beta=0, or use a TRL that supports the convention.")
        model.load_adapter(args.sft_adapter, adapter_name="ref",
                           is_trainable=False)
        model.set_adapter("default")     # keep the trainable adapter active
        print(f"KL reference: frozen SFT adapter loaded as 'ref' (beta={args.kl_beta})")
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
        max_tool_calling_iterations=args.max_tool_calling_iterations,
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
        vllm_enable_sleep_mode=args.vllm_enable_sleep_mode,
        optim="adamw_torch",
        report_to=[],
        remove_unused_columns=False,      # reset(**row) needs the golden/idx/seg_no columns
    )
    if args.vllm_importance_sampling_cap > 0:
        g_kwargs["vllm_importance_sampling_cap"] = args.vllm_importance_sampling_cap
    if args.epsilon_high > 0:
        g_kwargs["epsilon_high"] = args.epsilon_high
    if args.loss_type:
        g_kwargs["loss_type"] = args.loss_type
    if args.mask_truncated_completions:
        g_kwargs["mask_truncated_completions"] = True

    params = set(_insp.signature(GRPOConfig.__init__).parameters)
    if "warmup_ratio" not in params:
        steps = _math.ceil(len(ds) / (args.per_device_train_batch_size
                                      * args.gradient_accumulation_steps))
        g_kwargs["warmup_steps"] = _math.ceil(args.warmup_ratio * steps * args.num_train_epochs)
    dropped = sorted(k for k in g_kwargs if k not in params)
    if dropped:
        print(f"GRPOConfig: dropping unsupported args {dropped}")
    config = GRPOConfig(**{k: v for k, v in g_kwargs.items() if k in params})

    trainer = GRPOTrainer(model=model, args=config, train_dataset=ds,
                          processing_class=tok,
                          environment_factory=TravelEnv)    # both tools and reward come from the environment

    # The environment must expose exactly the 5 business tools: TRL collects every public method
    # of the instance as a tool, so any new public helper would become a sixth "tool" out of
    # nowhere and be written into the prompt (train != serve).
    exposed = sorted(getattr(f, "__name__", str(f)) for f in trainer.tools)
    expected = sorted(json.loads(CN_TRAVEL_TOOLS.read_text(encoding="utf-8"))
                      and [t["function"]["name"] for t in
                           json.loads(CN_TRAVEL_TOOLS.read_text(encoding="utf-8"))])
    if exposed != expected:
        raise SystemExit(f"environment exposes tools {exposed} != business tools {expected}")
    print(f"env tools verified: {exposed}")

    last_ckpt = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    if last_ckpt:
        print(f"resuming from checkpoint: {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)

    trainer.save_state()
    trainer.save_model(args.output_dir)

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
