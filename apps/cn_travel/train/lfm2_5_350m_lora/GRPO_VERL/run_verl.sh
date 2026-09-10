#!/usr/bin/env bash
# Whole-episode §5.4 GRPO. Configuration is selected with CONFIG and sealed.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
APP_ROOT="$(cd "$DIR/../../.." && pwd)"
cd "$APP_ROOT"

CONFIG_PATH="${CONFIG:-$DIR/verl_traj_config.json}"
[[ "$CONFIG_PATH" = /* ]] || CONFIG_PATH="$APP_ROOT/$CONFIG_PATH"
export CN_TRAVEL_VERL_CONFIG="$CONFIG_PATH"
export PYTHONPATH="$APP_ROOT:$APP_ROOT/src:$APP_ROOT/train:$DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Load the pinned in-memory LFM2 LoRA mapper fix in the driver and every
# Ray/vLLM subprocess. See verl_vllm_compat.py.
export VERL_USE_EXTERNAL_MODULES=verl_vllm_compat

PY_BIN="$(command -v "${PYTHON:-python3}" || true)"
if [[ ! -x "$PY_BIN" ]]; then
  echo "VERL Python is not executable: $PY_BIN" >&2
  exit 1
fi
export PATH="$(dirname "$PY_BIN"):$PATH"

# All-episode training loads the workflow-stratified sampler. Probe-filtered
# training loads the VERL/vLLM compatibility module.
TRAINING_POPULATION="$("$PY_BIN" - "$CN_TRAVEL_VERL_CONFIG" <<'PY'
import json
import pathlib
import sys

config = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
population = config.get("training_population")
if population is None and "task_filter" in config:
    population = "probe_filtered"
if population not in {"all_episodes", "probe_filtered"}:
    raise SystemExit(f"unsupported training_population: {population!r}")
print(population)
PY
)"
if [[ "$TRAINING_POPULATION" == "all_episodes" ]]; then
  export VERL_USE_EXTERNAL_MODULES=verl_vllm_compat,verl_workflow_sampler
fi

# Build the sealed episode parquet and reject per-turn rows, duplicate IDs, or
# populations other than 909 EpisodeSpecs.
"$PY_BIN" "$DIR/verl_data.py" --config "$CN_TRAVEL_VERL_CONFIG"

# The merged checkpoint makes adapter-disabled reference logits equal the frozen
# §5.1 SFT policy.  Both source hashes and every merged file are checked here.
"$PY_BIN" "$DIR/prepare_sft_reference.py" --config "$CN_TRAVEL_VERL_CONFIG"

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  exit 0
fi

RAY_BIN="$(dirname "$PY_BIN")/ray"
"$RAY_BIN" stop --force >/dev/null 2>&1 || true
trap '"$RAY_BIN" stop --force >/dev/null 2>&1 || true' EXIT

if (( $# )); then
  echo "run_verl.sh is sealed and does not accept Hydra overrides" >&2
  exit 1
fi

"$PY_BIN" - "$DIR" "$APP_ROOT" <<'PY'
from __future__ import annotations

import hashlib
import inspect
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile


script_dir = pathlib.Path(sys.argv[1]).resolve()
app_root = pathlib.Path(sys.argv[2]).resolve()
config_path = pathlib.Path(os.environ["CN_TRAVEL_VERL_CONFIG"]).resolve()
config = json.loads(config_path.read_text(encoding="utf-8"))
training_population = config.get("training_population")
if training_population is None and "task_filter" in config:
    training_population = "probe_filtered"
uses_workflow_sampler = training_population == "all_episodes"
external_modules = os.environ.get("VERL_USE_EXTERNAL_MODULES", "")


def resolve(value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    return path if path.is_absolute() else app_root / path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


require(
    training_population in {"all_episodes", "probe_filtered"},
    f"unsupported training_population: {training_population!r}",
)
if uses_workflow_sampler:
    sampler_contract = config.get("workflow_stratified_sampler")
    require("task_filter" not in config, "all-episode training cannot use a task filter")
    require(
        not any(key.startswith("probe_") for key in config),
        "all-episode training cannot contain probe settings",
    )
    require(int(config["expected_episode_count"]) == 909,
            "Run #3 requires exactly 909 source episodes")
    require(int(config["train_batch_size"]) == 32,
            "Run #3 requires train_batch_size=32")
    require(int(config["num_generations"]) == 8,
            "Run #3 requires G=8")
    require(int(config["micro_batch_size_per_gpu"]) == 2,
            "Run #3 requires micro_batch_size_per_gpu=2")
    require(config.get("filter_overlong_prompts") is False,
            "Run #3 must disable prompt filtering to retain all 909 episodes")
    require(
        isinstance(sampler_contract, dict)
        and sampler_contract.get("policy") == "workflow_stratified_v1",
        "Run #3 requires the sealed workflow-stratified sampler contract",
    )
    require(
        external_modules == "verl_vllm_compat,verl_workflow_sampler",
        "Run #3 external-module list is incomplete",
    )
else:
    sampler_contract = None
    require("task_filter" in config, "filtered training requires task_filter")
    require(
        external_modules == "verl_vllm_compat",
        "filtered training must use only the LFM/vLLM compatibility module",
    )


def container_cpus() -> int:
    try:
        quota, period = pathlib.Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, int(quota) // int(period))
    except Exception:
        pass
    try:
        quota = int(pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0:
            return max(1, quota // period)
    except Exception:
        pass
    return os.cpu_count() or 8


def verify_reference_contract() -> pathlib.Path:
    try:
        import transformers
        import verl
        import vllm
        from verl.experimental.agent_loop.agent_loop import AgentLoopBase
    except Exception as exc:
        raise SystemExit(f"cannot import the pinned VERL runtime: {exc}") from exc
    require(
        verl.__version__ == config["verl_version"],
        f"§5.4 requires verl=={config['verl_version']}, installed {verl.__version__}",
    )
    require(
        vllm.__version__ == config["vllm_version"],
        f"§5.4 requires vllm=={config['vllm_version']}, installed {vllm.__version__}",
    )
    require(
        transformers.__version__ == config["transformers_version"],
        "§5.4 transformers version differs from the sealed runtime stack",
    )
    if uses_workflow_sampler:
        try:
            import verl_workflow_sampler
        except Exception as exc:
            raise SystemExit(
                f"cannot install the workflow-stratified VERL sampler: {exc}"
            ) from exc
        require(
            getattr(verl_workflow_sampler, "PATCH_ACTIVE", False) is True,
            "workflow-stratified VERL sampler patch is inactive",
        )
    base_source = inspect.getsource(AgentLoopBase)
    require(
        callable(getattr(AgentLoopBase, "apply_chat_template", None))
        and "remove_system_prompt" in inspect.signature(
            AgentLoopBase.apply_chat_template
        ).parameters
        and "turn_separator" in base_source,
        "installed VERL AgentLoopBase does not provide the pinned v0.9 API",
    )
    require(float(config["kl_beta"]) == 0.01, "§5.4 requires KL beta=0.01")
    require(config["kl_loss_type"] == "low_var_kl",
            "§5.4 requires VERL low_var_kl actor loss")
    require(int(config["seed"]) == 42, "§5.4 requires rollout/training seed=42")
    require(int(config["lora_r"]) > 0, "GRPO LoRA rank must be positive")
    reference = resolve(config["sft_reference_model"])
    require((reference / "cn_travel_sft_reference.json").is_file(),
            "verified frozen SFT reference manifest is missing")
    require((reference / "config.json").is_file(), "SFT reference is not a HF checkpoint")
    require(not (reference / "adapter_config.json").exists(),
            "SFT reference must be merged weights, not a live PEFT adapter")

    # Fail closed if the installed VERL no longer implements LoRA reference
    # log-probs by disabling the actor adapter.  With model.path pointing at the
    # merged SFT checkpoint, that disabled-adapter view is exactly §5.1 SFT.
    try:
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    except Exception as exc:
        raise SystemExit(f"cannot import VERL trainer for reference-policy audit: {exc}") from exc
    source = inspect.getsource(RayPPOTrainer)
    require("ref_in_actor" in source and "without lora applied" in source.lower(),
            "installed VERL reference semantics are unrecognized; refusing KL against an unknown policy")
    print(f"KL reference: verified frozen merged SFT policy at {reference}", flush=True)
    return reference


reference_model = verify_reference_contract()
train_parquet = resolve(config["train_parquet"])
require(train_parquet.is_file(), f"missing packed training parquet: {train_parquet}")


def canonical_digest(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def establish_run_contract(reference: pathlib.Path) -> tuple[dict, str]:
    checkpoint_root = resolve(config["checkpoint_dir"])
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    filter_path = None if uses_workflow_sampler else resolve(config["task_filter"])
    reference_manifest = reference / "cn_travel_sft_reference.json"
    if filter_path is not None:
        require(filter_path.is_file(), f"missing sealed task filter: {filter_path}")
    require(reference_manifest.is_file(), "frozen SFT reference manifest is missing")
    behavior_files = [
        script_dir / "run_verl.sh",
        script_dir / "verl_loop.py",
        script_dir / "verl_data.py",
        script_dir / "verl_vllm_compat.py",
        script_dir / "verl_agent.yaml",
        app_root / "train/world_policy/__init__.py",
        app_root / "train/world_policy/actions.py",
        app_root / "train/world_policy/interpreter.py",
        app_root / "train/world_policy/runtime.py",
        app_root / "train/world_policy/world.py",
        app_root / "src/cn_travel/business_logic/contracts.py",
        app_root / "src/cn_travel/business_logic/tool_schemas.json",
        app_root / "data/city_registry.json",
        app_root / "src/cn_travel/service/result.py",
    ]
    behavior_files.append(
        script_dir
        / ("verl_workflow_sampler.py" if uses_workflow_sampler else "probe_difficulty.py")
    )
    for path in behavior_files:
        require(path.is_file(), f"behavior-critical source is missing: {path}")
    behavior_hashes = {
        str(path.relative_to(app_root)): sha256(path)
        for path in behavior_files
    }
    payload = {
        "schema_version": (
            "cn_travel.grpo_run_contract.v2"
            if uses_workflow_sampler
            else "cn_travel.grpo_run_contract.v1"
        ),
        "verl_version": config["verl_version"],
        "config_sha256": sha256(config_path),
        "train_parquet_sha256": sha256(train_parquet),
        "episodes_sha256": config["episodes_sha256"],
        "world_sha256": config["world_sha256"],
        "world_id": config["world_id"],
        "sft_reference_manifest_sha256": sha256(reference_manifest),
        "sft_reference_weights_sha256": config["sft_reference_weights_sha256"],
        "behavior_files_sha256": behavior_hashes,
    }
    if uses_workflow_sampler:
        payload.update(
            {
                "training_population": "all_episodes",
                "workflow_stratified_sampler": sampler_contract,
                "workflow_stratified_sampler_sha256": canonical_digest(
                    sampler_contract
                ),
                "external_modules": external_modules,
                "data_shuffle": False,
                "filter_overlong_prompts": False,
            }
        )
    else:
        assert filter_path is not None
        payload.update({"task_filter_sha256": sha256(filter_path)})
    fingerprint = canonical_digest(payload)
    document = {**payload, "fingerprint_sha256": fingerprint}
    contract_path = checkpoint_root / "run_contract.json"
    if contract_path.exists():
        try:
            existing = json.loads(contract_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SystemExit(f"invalid existing GRPO run contract: {contract_path}") from exc
        require(
            existing == document,
            "checkpoint run contract differs from the current data/config/reference; "
            "refusing an incompatible auto-resume",
        )
    else:
        has_checkpoint_state = any(checkpoint_root.glob("global_step_*")) or (
            checkpoint_root / "latest_checkpointed_iteration.txt"
        ).exists()
        require(
            not has_checkpoint_state,
            "checkpoint state exists without a run contract; refusing an unverified auto-resume",
        )
        temporary = checkpoint_root / ".run_contract.json.tmp"
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, contract_path)
    print(f"run contract: {fingerprint}", flush=True)
    return document, fingerprint


run_contract, run_contract_sha256 = establish_run_contract(reference_model)

overrides = {
    "algorithm.adv_estimator": "grpo",
    "algorithm.use_kl_in_reward": False,
    "algorithm.filter_groups.enable": config.get("filter_groups", False),
    "algorithm.filter_groups.metric": "score",
    "algorithm.filter_groups.max_num_gen_batches": config.get("max_num_gen_batches", 3),
    "data.train_files": str(train_parquet),
    "data.val_files": str(train_parquet),
    "data.return_raw_chat": True,
    "data.seed": config["seed"],
    "data.train_batch_size": config["train_batch_size"],
    "data.max_prompt_length": config["max_prompt_length"],
    "data.max_response_length": config["max_response_length"],
    # The actor starts as merged SFT + zero GRPO LoRA.  VERL's adapter-disabled
    # ref view is therefore frozen SFT, never the bare LFM checkpoint.
    "actor_rollout_ref.model.path": str(reference_model),
    "actor_rollout_ref.model.lora_rank": config["lora_r"],
    "actor_rollout_ref.model.lora_alpha": config["lora_alpha"],
    # Hydra's override lexer rejects the regex backslashes unless the complete
    # value is quoted in Hydra syntax.  Single quotes preserve them verbatim.
    "actor_rollout_ref.model.target_modules": f"'{config['lora_target_modules']}'",
    "actor_rollout_ref.model.lora_adapter_path": None,
    "actor_rollout_ref.model.enable_gradient_checkpointing": True,
    "+actor_rollout_ref.model.override_config.attn_implementation": "sdpa",
    "actor_rollout_ref.actor.optim.lr": config["learning_rate"],
    "actor_rollout_ref.actor.optim.lr_scheduler_type": config["lr_scheduler_type"],
    "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio": config["warmup_ratio"],
    "actor_rollout_ref.actor.ppo_mini_batch_size": config["train_batch_size"],
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": config.get("micro_batch_size_per_gpu", 1),
    "actor_rollout_ref.actor.use_kl_loss": True,
    "actor_rollout_ref.actor.kl_loss_coef": config["kl_beta"],
    "actor_rollout_ref.actor.kl_loss_type": config["kl_loss_type"],
    "actor_rollout_ref.actor.clip_ratio_low": config["epsilon_low"],
    "actor_rollout_ref.actor.clip_ratio_high": config["epsilon_high"],
    "actor_rollout_ref.rollout.name": "vllm",
    "actor_rollout_ref.rollout.load_format": "safetensors",
    "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
    "actor_rollout_ref.rollout.mode": "async",
    "actor_rollout_ref.rollout.n": config["num_generations"],
    "actor_rollout_ref.rollout.temperature": config["temperature"],
    "actor_rollout_ref.rollout.seed": config["seed"],
    "actor_rollout_ref.rollout.gpu_memory_utilization": config["vllm_gpu_memory_utilization"],
    "actor_rollout_ref.rollout.max_model_len": config["vllm_max_model_length"],
    "actor_rollout_ref.rollout.max_num_batched_tokens": config["vllm_max_model_length"],
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": config.get("micro_batch_size_per_gpu", 1),
    "actor_rollout_ref.rollout.multi_turn.enable": True,
    "actor_rollout_ref.rollout.multi_turn.max_user_turns": config["max_user_turns"],
    "actor_rollout_ref.rollout.multi_turn.max_assistant_turns": config["max_assistant_turns"],
    "actor_rollout_ref.rollout.agent.agent_loop_config_path": str(script_dir / "verl_agent.yaml"),
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": config.get("micro_batch_size_per_gpu", 1),
    "ray_kwargs.ray_init.num_cpus": container_cpus(),
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES": "verl_vllm_compat",
    "+ray_kwargs.ray_init.runtime_env.env_vars.CN_TRAVEL_VERL_CONFIG": str(
        config_path
    ),
    "trainer.use_v1": False,
    "trainer.n_gpus_per_node": 1,
    "trainer.nnodes": 1,
    "trainer.logger": '["console"]',
    "trainer.project_name": "cn_travel",
    "trainer.experiment_name": config["experiment_name"],
    "trainer.default_local_dir": str(resolve(config["checkpoint_dir"])),
    "trainer.save_freq": config["save_steps"],
    "trainer.max_actor_ckpt_to_keep": config["save_total_limit"],
    "trainer.test_freq": -1,
    "trainer.val_before_train": False,
    "trainer.resume_mode": "auto",
    "trainer.total_epochs": config["num_train_epochs"],
}
if uses_workflow_sampler:
    # The Run #3 sampler owns both epoch and within-batch order. Filtering here
    # would silently reduce the sealed population before that sampler sees it.
    overrides["data.shuffle"] = False
    overrides["data.filter_overlong_prompts"] = False
    # Quote the comma-separated module list for Hydra; the resulting Ray
    # environment value contains no quote characters.
    overrides[
        "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES"
    ] = f"'{external_modules}'"


def hydra_value(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return value


command = [sys.executable, "-m", "verl.trainer.main_ppo"]
command += [f"{key}={hydra_value(value)}" for key, value in overrides.items()]
dry_run = bool(os.environ.get("DRYRUN"))
if dry_run:
    command.append("--cfg=job")
    print("DRYRUN: composing VERL config only", flush=True)
print(f"launch: verl.trainer.main_ppo ({len(overrides)} sealed overrides)", flush=True)
subprocess.run(command, check=True)
if dry_run:
    raise SystemExit(0)


def latest_actor_checkpoint(checkpoint_root: pathlib.Path) -> tuple[int, pathlib.Path]:
    tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
    require(tracker.is_file(), f"training ended without VERL's checkpoint tracker in {checkpoint_root}")
    try:
        step = int(tracker.read_text(encoding="utf-8").strip())
    except ValueError as exc:
        raise SystemExit(f"invalid VERL checkpoint tracker: {tracker}") from exc
    actor = checkpoint_root / f"global_step_{step}" / "actor"
    require(actor.is_dir(), f"tracked actor checkpoint is missing: {actor}")
    require((actor.parent / "data.pt").is_file(), f"tracked checkpoint is incomplete: {actor.parent}")
    return step, actor


checkpoint_root = resolve(config["checkpoint_dir"])
step, actor_checkpoint = latest_actor_checkpoint(checkpoint_root)
final_root = resolve(config["output_dir"])
final_root.mkdir(parents=True, exist_ok=True)
merge_tmp = pathlib.Path(tempfile.mkdtemp(prefix=".merge.", dir=final_root.parent))
stage = pathlib.Path(tempfile.mkdtemp(prefix=f".global_step_{step}.", dir=final_root))
try:
    subprocess.run(
        [sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
         "--local_dir", str(actor_checkpoint), "--target_dir", str(merge_tmp)],
        check=True,
    )
    adapters = [
        path for path in merge_tmp.rglob("adapter_model.safetensors")
        if (path.parent / "adapter_config.json").is_file()
    ]
    require(len(adapters) == 1,
            f"expected one exported LoRA adapter, found {len(adapters)} under {merge_tmp}")
    adapter_source = adapters[0].parent
    for source in adapter_source.iterdir():
        if source.is_file():
            shutil.copy2(source, stage / source.name)
    adapter_file = stage / "adapter_model.safetensors"
    digest = sha256(adapter_file)
    adapter_config = stage / "adapter_config.json"
    require(adapter_config.is_file(), "exported LoRA adapter_config.json is missing")
    adapter_config_digest = sha256(adapter_config)
    adapter_files = {
        path.name: sha256(path)
        for path in sorted(stage.iterdir())
        if path.is_file()
    }
    (stage / "adapter.sha256").write_text(digest + "\n", encoding="utf-8")
    (stage / "reference.json").write_text(
        json.dumps(
            {
                "schema_version": "cn_travel.grpo_adapter.v1",
                "global_step": step,
                "base_model": config["sft_reference_model"],
                "base_model_manifest": "cn_travel_sft_reference.json",
                "adapter_sha256": digest,
                "adapter_config_sha256": adapter_config_digest,
                "adapter_files_sha256": adapter_files,
                "kl_beta": config["kl_beta"],
                "run_contract_sha256": run_contract_sha256,
                "run_contract": run_contract,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    final_step = final_root / f"global_step_{step}"
    if final_step.exists():
        for name, expected_digest in adapter_files.items():
            existing = final_step / name
            require(existing.is_file() and sha256(existing) == expected_digest,
                    f"refusing to reuse a different final adapter file: {existing}")
        existing_reference = final_step / "reference.json"
        require(existing_reference.is_file(),
                f"existing final adapter lacks provenance: {existing_reference}")
        existing_provenance = json.loads(existing_reference.read_text(encoding="utf-8"))
        require(existing_provenance.get("run_contract_sha256") == run_contract_sha256,
                f"existing final adapter belongs to a different run: {final_step}")
    else:
        os.replace(stage, final_step)
    latest = {
        "global_step": step,
        "adapter": str(final_step.relative_to(app_root)),
        "adapter_sha256": digest,
        "adapter_config_sha256": adapter_config_digest,
        "reference_model": config["sft_reference_model"],
        "run_contract_sha256": run_contract_sha256,
    }
    pointer_tmp = final_root / ".latest_adapter.json.tmp"
    pointer_tmp.write_text(json.dumps(latest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(pointer_tmp, final_root / "latest_adapter.json")
    hash_tmp = final_root / ".adapter.sha256.tmp"
    hash_tmp.write_text(digest + "\n", encoding="utf-8")
    os.replace(hash_tmp, final_root / "adapter.sha256")
    print(f"final adapter: {final_step}", flush=True)
    print(f"adapter sha256: {digest}", flush=True)
finally:
    shutil.rmtree(merge_tmp, ignore_errors=True)
    shutil.rmtree(stage, ignore_errors=True)
PY
