#!/usr/bin/env bash
# Read verl_traj_config.json and pass every hyperparameter to verl main_ppo; the config file is authoritative.
# Switch models by changing only the config: CONFIG=.../lfm2_5_350m_lora/GRPO_VERL/verl_traj_config.json run_verl.sh
# Training uses an isolated verl-env (verl requires transformers<5.11) and leaves the TRL train-env untouched.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
APP_ROOT="$(cd "$DIR/../../.." && pwd)"
cd "$APP_ROOT"

CONFIG_PATH="${CONFIG:-$DIR/verl_traj_config.json}"
[[ "$CONFIG_PATH" = /* ]] || CONFIG_PATH="$APP_ROOT/$CONFIG_PATH"
export CN_TRAVEL_VERL_CONFIG="$CONFIG_PATH"
export PYTHONPATH="$APP_ROOT:$APP_ROOT/src:$APP_ROOT/train:$DIR:$DIR/../GRPO_TRL${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Manage Ray explicitly: stale Ray workers retain GPU memory and can prevent the next
# single-GPU run from starting. Stop them before training and again on exit, including failures.
PY_BIN="$(command -v "${PYTHON:-python3}")"
RAY_BIN="$(dirname "$PY_BIN")/ray"
"$RAY_BIN" stop --force >/dev/null 2>&1 || true
trap '"$RAY_BIN" stop --force >/dev/null 2>&1 || true' EXIT

"$PY_BIN" - "$DIR" "$APP_ROOT" "$@" <<'PY'
import json, os, pathlib, sys
d = sys.argv[1]
app_root = pathlib.Path(sys.argv[2])
cfg = json.load(open(os.environ["CN_TRAVEL_VERL_CONFIG"], encoding="utf-8"))
for key in ("model_name_or_path", "sft_adapter", "output_dir", "train_parquet"):
    if cfg.get(key) and not pathlib.Path(cfg[key]).is_absolute():
        cfg[key] = str(app_root / cfg[key])
g = cfg.get                      # verl_loop.py reads reward weights from the same file, not the command line


def container_cpus():
    """Return the container CPU quota rather than the host-visible count."""
    try:                                            # cgroup v2
        q, per = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max":
            return max(1, int(q) // int(per))
    except Exception:
        pass
    try:                                            # cgroup v1
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        per = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0:
            return max(1, q // per)
    except Exception:
        pass
    return os.cpu_count() or 8
ov = {
    "algorithm.adv_estimator": "grpo",
    "algorithm.filter_groups.enable": g("filter_groups", False),
    "algorithm.filter_groups.metric": "score",
    "algorithm.filter_groups.max_num_gen_batches": g("max_num_gen_batches", 3),
    "data.train_files": cfg["train_parquet"],
    "data.val_files": cfg["train_parquet"],
    "data.return_raw_chat": True,
    "data.seed": cfg["seed"],
    "data.train_batch_size": cfg["train_batch_size"],
    "data.max_prompt_length": cfg["max_prompt_length"],
    "data.max_response_length": cfg["max_response_length"],
    "actor_rollout_ref.model.path": cfg["model_name_or_path"],
    "actor_rollout_ref.model.lora_rank": cfg["lora_r"],
    "actor_rollout_ref.model.lora_alpha": cfg["lora_alpha"],
    "actor_rollout_ref.model.lora_adapter_path": cfg["sft_adapter"],
    "actor_rollout_ref.model.enable_gradient_checkpointing": True,
    # verl defaults to flash_attention_2 (the fallback in workers/config/model.py), but this
    # environment lacks flash-attn. The TRL environments in sections 5.2 and 5.3 also use
    # sdpa, which keeps the attention implementation consistent across compared runs.
    "+actor_rollout_ref.model.override_config.attn_implementation": "sdpa",
    # Keep verl's default use_remove_padding=True so logits and entropy use actual token counts.
    # Including padding would materialize logits for an 11k context and 150k-token vocabulary,
    # exhausting GPU memory (entropy_from_logits logsumexp alone requires 4.7 GiB). The upstream
    # attention_utils.py restored in verl-env supplies pure-torch unpad/pad helpers without flash-attn.
    "actor_rollout_ref.actor.optim.lr": cfg["learning_rate"],
    "actor_rollout_ref.actor.optim.warmup_style": cfg["lr_scheduler_type"],
    "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio": cfg["warmup_ratio"],
    "actor_rollout_ref.actor.ppo_mini_batch_size": cfg["train_batch_size"],
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
    "actor_rollout_ref.actor.use_kl_loss": cfg["kl_beta"] > 0,
    "actor_rollout_ref.actor.kl_loss_coef": cfg["kl_beta"],
    "actor_rollout_ref.actor.clip_ratio_low": cfg["epsilon_low"],
    "actor_rollout_ref.actor.clip_ratio_high": cfg["epsilon_high"],
    "actor_rollout_ref.rollout.name": "vllm",
    # verl defaults to TP=2, assuming at least two GPUs. This host has one RTX 4090, so tensor
    # parallelism must be 1; otherwise infer_world_size=2 cannot divide world_size=1 at startup.
    "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
    "actor_rollout_ref.rollout.mode": "async",
    "actor_rollout_ref.rollout.n": cfg["num_generations"],
    "actor_rollout_ref.rollout.temperature": cfg["temperature"],
    "actor_rollout_ref.rollout.gpu_memory_utilization": cfg["vllm_gpu_memory_utilization"],
    "actor_rollout_ref.rollout.max_model_len": cfg["vllm_max_model_length"],
    "actor_rollout_ref.rollout.max_num_batched_tokens": cfg["vllm_max_model_length"],
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
    "actor_rollout_ref.rollout.multi_turn.max_user_turns": cfg["max_user_turns"],
    "actor_rollout_ref.rollout.multi_turn.max_assistant_turns": cfg["max_assistant_turns"],
    "actor_rollout_ref.rollout.agent.agent_loop_config_path": os.path.join(d, "verl_agent.yaml"),
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 1,
    "ray_kwargs.ray_init.num_cpus": container_cpus(),
    # verl 0.9 defaults to V1 TaskRunner, which enables transfer_queue and imports it unconditionally,
    # but that package is neither a verl dependency nor available on PyPI. V0 is the documented,
    # commonly used path and still supports agent loops through AgentLoopManager in ray_trainer.py.
    "trainer.use_v1": False,
    "trainer.n_gpus_per_node": 1,
    "trainer.nnodes": 1,
    "trainer.logger": '["console"]',
    "trainer.project_name": "cn_travel",
    "trainer.experiment_name": cfg["experiment_name"],
    "trainer.default_local_dir": cfg["output_dir"],
    "trainer.save_freq": cfg["save_steps"],
    "trainer.max_actor_ckpt_to_keep": cfg["save_total_limit"],
    "trainer.test_freq": -1,
    # verl validates the entire dataset before training by default. Here val_files is the training
    # set itself (909 rows times n rollouts), so disable it and use the separate section 6.2 evaluation path.
    "trainer.val_before_train": False,
    "trainer.resume_mode": "auto",          # Resume from checkpoints by default
    "trainer.total_epochs": cfg["num_train_epochs"],
}
# Command-line overrides support temporary changes such as smoke tests or step counts.
cmd = ([sys.executable, "-m", "verl.trainer.main_ppo"]
       + [f"{k}={v}" for k, v in ov.items()] + sys.argv[3:])
if os.environ.get("DRYRUN"):
    # Compose configuration without training so invalid keys fail before GPU allocation.
    cmd.append("--cfg=job")
    print("DRYRUN: composing config only", flush=True)
print("launch: verl.trainer.main_ppo", f"({len(ov)} overrides)", flush=True)
os.execv(sys.executable, cmd)
PY
