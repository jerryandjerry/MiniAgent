"""Integration tests for the VERL whole-trajectory training implementation."""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.util
import itertools
import json
import pathlib
import sys
import types
from collections import Counter
from types import SimpleNamespace

import pytest

from project_paths import DATA_ROOT, TRAIN_ROOT

VERL_DIR = TRAIN_ROOT / "lfm2_5_350m_lora" / "GRPO_VERL"
EPISODES = DATA_ROOT / "7_world_policy" / "world_policy_episodes.jsonl"
WORLD = DATA_ROOT / "7_world_policy" / "synthetic_china_world_v1.json"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, VERL_DIR / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sampler_without_training_stack(monkeypatch, name: str):
    """Load the pure scheduling logic with the small Sampler protocol it uses."""

    class Sampler:
        @classmethod
        def __class_getitem__(cls, _item):
            return cls

    torch = types.ModuleType("torch")
    utils = types.ModuleType("torch.utils")
    data = types.ModuleType("torch.utils.data")
    data.Sampler = Sampler
    utils.data = data
    torch.utils = utils
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils", utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", data)
    return _load(name, "verl_workflow_sampler.py")


def _episode_rows() -> list[dict]:
    return [json.loads(line) for line in EPISODES.read_text(encoding="utf-8").splitlines()]


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: pathlib.Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def test_sft_reference_is_bound_to_trusted_source_and_merged_bytes(tmp_path):
    reference = _load("world_policy_sft_reference_test", "prepare_sft_reference.py")
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    merged = tmp_path / "merged"
    _write(base / "model.safetensors", "base-weights")
    _write(base / "config.json", "{}")
    _write(adapter / "adapter_model.safetensors", "sft-weights")
    _write(
        adapter / "adapter_config.json",
        json.dumps({"base_model_name_or_path": str(base)}),
    )
    tokenizer_names = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    for name in tokenizer_names:
        _write(adapter / name, f"sealed-{name}")
    _write(merged / "model.safetensors", "merged-weights")

    config = {
        "base_model": str(base),
        "base_weights_file": "model.safetensors",
        "base_model_sha256": _sha256(base / "model.safetensors"),
        "base_config_sha256": _sha256(base / "config.json"),
        "sft_adapter": str(adapter),
        "sft_adapter_sha256": _sha256(adapter / "adapter_model.safetensors"),
        "sft_adapter_config_sha256": _sha256(adapter / "adapter_config.json"),
        "sft_tokenizer_files_sha256": {
            name: _sha256(adapter / name) for name in tokenizer_names
        },
        "sft_reference_model": str(merged),
        "sft_reference_weights_file": "model.safetensors",
        "sft_reference_weights_sha256": _sha256(merged / "model.safetensors"),
    }
    source = reference._source_contract(config)
    manifest = {
        "schema_version": "cn_travel.sft_reference.v1",
        "source": source,
        "merged_weights": {
            "file": "model.safetensors",
            "sha256": config["sft_reference_weights_sha256"],
        },
        "files": {"model.safetensors": config["sft_reference_weights_sha256"]},
    }
    _write(
        merged / reference.MANIFEST_NAME,
        json.dumps(manifest, ensure_ascii=False),
    )
    assert reference.verify(config) == merged

    _write(merged / "model.safetensors", "different-merged-weights")
    with pytest.raises(SystemExit, match="frozen merged SFT weights SHA-256 mismatch"):
        reference.verify(config)

    _write(merged / "model.safetensors", "merged-weights")
    _write(adapter / "chat_template.jinja", "different-template")
    with pytest.raises(SystemExit, match="SFT tokenizer asset chat_template.jinja SHA-256 mismatch"):
        reference.verify(config)


def test_verl_launcher_seals_resume_and_export_provenance():
    launcher = (VERL_DIR / "run_verl.sh").read_text(encoding="utf-8")
    for required in (
        'verl.__version__ == config["verl_version"]',
        '"task_filter_sha256": sha256(filter_path)',
        '"train_parquet_sha256": sha256(train_parquet)',
        '"sft_reference_manifest_sha256": sha256(reference_manifest)',
        'checkpoint_root / "run_contract.json"',
        'checkpoint_root / "latest_checkpointed_iteration.txt"',
        '"adapter_config_sha256": adapter_config_digest',
        '"run_contract_sha256": run_contract_sha256',
        '"behavior_files_sha256": behavior_hashes',
        '"actor_rollout_ref.actor.kl_loss_type": config["kl_loss_type"]',
        '"actor_rollout_ref.rollout.seed": config["seed"]',
        'run_verl.sh is sealed and does not accept Hydra overrides',
    ):
        assert required in launcher
    assert (
        '"actor_rollout_ref.model.target_modules": '
        'f"\'{config[\'lora_target_modules\']}\'"'
    ) in launcher
    assert (
        '"actor_rollout_ref.actor.optim.lr_scheduler_type": '
        'config["lr_scheduler_type"]'
    ) in launcher
    assert '"actor_rollout_ref.actor.optim.warmup_style"' not in launcher
    assert '"actor_rollout_ref.rollout.multi_turn.enable": True' in launcher
    assert "export VERL_USE_EXTERNAL_MODULES=verl_vllm_compat" in launcher
    assert (
        '"+ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES": '
        '"verl_vllm_compat"'
    ) in launcher
    assert "command += sys.argv[2:]" not in launcher


def test_verl_vllm_lfm2_lora_mapper_shim_is_pinned_and_unstacks_names():
    shim = (VERL_DIR / "verl_vllm_compat.py").read_text(encoding="utf-8")

    assert '_EXPECTED_VERL = "0.9.0"' in shim
    assert '_EXPECTED_VLLM_PREFIX = "0.28.0"' in shim
    assert "get_unstacked_mapper" in shim
    assert "LoRAModel.from_lora_tensors = _from_lora_tensors_unstacked" in shim


def _run3_config() -> dict:
    return json.loads(
        (VERL_DIR / "verl_traj_config_run3.json").read_text(encoding="utf-8")
    )


def test_active_lfm_reproduction_configs_preserve_published_adapters():
    expected = {
        "verl_traj_config.json": (
            "checkpoints_reproduction_run2",
            "adapter_verl_reproduction_run2",
        ),
        "verl_traj_config_run3.json": (
            "checkpoints_reproduction_run3",
            "adapter_verl_reproduction_run3",
        ),
    }
    for filename, (checkpoint_name, output_name) in expected.items():
        config = json.loads((VERL_DIR / filename).read_text(encoding="utf-8"))
        assert pathlib.Path(config["checkpoint_dir"]).name == checkpoint_name
        assert pathlib.Path(config["output_dir"]).name == output_name


def _run3_rows() -> list[dict]:
    data = _load("world_policy_verl_data_run3_rows", "verl_data.py")
    return data.build_all_rows(EPISODES, expected_count=909)


def test_run3_config_uses_all_episodes_g8_and_no_probe_contract():
    config = _run3_config()

    assert config["training_population"] == "all_episodes"
    assert config["expected_episode_count"] == 909
    assert config["num_generations"] == 8
    assert config["train_batch_size"] == 32
    assert config["micro_batch_size_per_gpu"] == 2
    assert config["filter_overlong_prompts"] is False
    assert not any(
        key == "task_filter" or key.startswith("probe_")
        for key in config
    )
    sampler = config["workflow_stratified_sampler"]
    assert sampler["policy"] == "workflow_stratified_v1"
    assert sampler["batches_per_epoch"] == 29
    assert sampler["presentations_per_epoch"] == 928
    assert sampler["padding_presentations_per_epoch"] == 19


def test_run3_build_all_rows_preserves_population_workflows_and_prompt_privacy():
    specs = sorted(
        _episode_rows(),
        key=lambda item: (item["source_idx"], item["episode_id"]),
    )
    rows = _run3_rows()

    assert len(rows) == len(specs) == 909
    assert len({row["extra_info"]["episode_id"] for row in rows}) == 909
    assert [row["extra_info"]["episode_id"] for row in rows] == [
        spec["episode_id"] for spec in specs
    ]
    assert Counter(row["extra_info"]["workflow"] for row in rows) == {
        "W1": 405,
        "W2": 108,
        "W3": 216,
        "W4": 90,
        "W5": 90,
    }

    forbidden_private_fields = (
        '"person_state"',
        '"state_graph"',
        '"initial_state_id"',
        '"response_policy"',
        '"reward_milestones"',
        '"metadata"',
    )
    for row, spec in zip(rows, specs):
        assert set(row) == {"data_source", "agent_name", "prompt", "extra_info"}
        assert set(row["extra_info"]) == {"episode_id", "workflow"}
        assert row["prompt"] == spec["initial_messages"]
        roles = [message["role"] for message in row["prompt"]]
        assert roles[0] == "system" and roles[-1] == "user"
        assert set(roles) <= {"system", "user"}
        serialized = json.dumps(row, ensure_ascii=False)
        assert all(field not in serialized for field in forbidden_private_fields)


def test_run3_workflow_sampler_epoch_contract_and_determinism(monkeypatch):
    sampler_module = _load_sampler_without_training_stack(
        monkeypatch, "world_policy_workflow_sampler_epoch_test"
    )
    config = _run3_config()
    rows = _run3_rows()
    sampler = sampler_module.WorkflowStratifiedSampler(rows, config)
    schedules = [sampler.epoch_schedule(epoch) for epoch in (0, 1)]

    workflow_by_index = {
        index: row["extra_info"]["workflow"] for index, row in enumerate(rows)
    }
    expected_workflow_totals = config["workflow_stratified_sampler"]["epoch_counts"]
    for schedule in schedules:
        counts = Counter(schedule)
        assert len(schedule) == 928
        assert len(counts) == 909
        assert sum(number - 1 for number in counts.values()) == 19
        assert Counter(counts.values()) == {1: 890, 2: 19}
        assert max(counts.values()) == 2
        assert Counter(workflow_by_index[index] for index in schedule) == (
            expected_workflow_totals
        )

        batches = [schedule[offset : offset + 32] for offset in range(0, 928, 32)]
        assert len(batches) == 29
        for batch in batches:
            assert len(batch) == len(set(batch)) == 32
            assert {workflow_by_index[index] for index in batch} == {
                "W1", "W2", "W3", "W4", "W5"
            }

    identical = sampler_module.WorkflowStratifiedSampler(rows, config)
    assert identical.epoch_schedule(0) == schedules[0]
    assert identical.epoch_schedule(1) == schedules[1]
    assert schedules[0] != schedules[1]


def test_run3_workflow_sampler_resume_is_exact_and_rejects_stale_state(monkeypatch):
    sampler_module = _load_sampler_without_training_stack(
        monkeypatch, "world_policy_workflow_sampler_resume_test"
    )
    config = _run3_config()
    rows = _run3_rows()
    sampler = sampler_module.WorkflowStratifiedSampler(rows, config)
    schedule = sampler.epoch_schedule(0)

    assert list(itertools.islice(iter(sampler), 137)) == list(schedule[:137])
    state = sampler.state_dict()
    resumed = sampler_module.WorkflowStratifiedSampler(rows, config)
    resumed.load_state_dict(state)
    assert list(resumed) == list(schedule[137:])

    stale = dict(state, schedule_fingerprint="0" * 64)
    rejected = sampler_module.WorkflowStratifiedSampler(rows, config)
    with pytest.raises(
        sampler_module.SamplerContractError,
        match="different data/config",
    ):
        rejected.load_state_dict(stale)


def test_run3_launcher_installs_sampler_without_filtering_or_random_shuffle():
    launcher = (VERL_DIR / "run_verl.sh").read_text(encoding="utf-8")

    assert (
        "export VERL_USE_EXTERNAL_MODULES=verl_vllm_compat,verl_workflow_sampler"
        in launcher
    )
    assert 'overrides["data.shuffle"] = False' in launcher
    assert 'overrides["data.filter_overlong_prompts"] = False' in launcher
    assert 'require("task_filter" not in config' in launcher
    assert 'not any(key.startswith("probe_") for key in config)' in launcher
    assert (
        '"$PY_BIN" "$DIR/verl_data.py" --config "$CN_TRAVEL_VERL_CONFIG"'
        in launcher
    )
    assert "--filter" not in launcher


_RUN2_REWARD = {
    "policy": "terminal_success_gated_v1",
    "maximum": 1.5,
    "success_floor": 0.5,
    "penalty_per_violation": 0.2,
}


def _mini_specs(count: int) -> list[dict]:
    return [
        {
            "episode_id": f"episode-{index}",
            "source_idx": index,
            "metadata": {"route": f"route-{index % 5}"},
            "initial_messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": f"user-{index}"},
            ],
        }
        for index in range(count)
    ]


def _attempt(position: int, number: int, *, goal: bool, u: int = 0) -> dict:
    clean = goal and u == 0
    reward = 0.0 if not goal else max(0.5, 1.5 - 0.2 * u)
    violations = ["wrong_question"] * u
    if not goal:
        violations.append("invalid_final")
    return {
        "attempt": number,
        "phase": "initial" if number < 4 else "retry",
        "temperature": 0.9 if number < 4 else 0.8,
        "seed": 42 + position * 10_000 + number * 100,
        "reward": reward,
        "goal_reached": goal,
        "clean_trace": clean,
        "U": u,
        "outcome": {
            "terminal": True,
            "goal_reached": goal,
            "clean_trace": clean,
            "violations": violations,
            "reward": {"total": reward, "violation_count": u},
        },
    }


def _record(spec: dict, position: int, category: str) -> dict:
    if category == "solved":
        attempts = [_attempt(position, number, goal=True) for number in range(4)]
        final_category = "solved"
    elif category == "mixed":
        attempts = [
            _attempt(position, number, goal=True, u=(1 if number == 0 else 0))
            for number in range(4)
        ]
        final_category = "mixed"
    elif category == "mixed_after_retry":
        attempts = [_attempt(position, number, goal=False) for number in range(4)] + [
            _attempt(position, number, goal=(number == 4)) for number in range(4, 8)
        ]
        final_category = "mixed"
    elif category == "unsolved":
        attempts = [_attempt(position, number, goal=False) for number in range(8)]
        final_category = "unsolved"
    else:
        raise AssertionError(category)
    return {
        "episode_id": spec["episode_id"],
        "source_idx": spec["source_idx"],
        "route": spec["metadata"]["route"],
        "category": final_category,
        "attempts": attempts,
    }


def _filter_document(data, specs: list[dict], categories: list[str], *, digest="episodes"):
    records = [_record(spec, position, category)
               for position, (spec, category) in enumerate(zip(specs, categories))]
    maxima = [record for record in records if record["category"] == "solved"]
    anchors = data._sample_anchors(maxima, fraction=0.15, seed=42)
    anchor_ids = {record["episode_id"] for record in anchors}
    train_on = [
        {
            "episode_id": record["episode_id"],
            "source_idx": record["source_idx"],
            "route": record["route"],
            "selection": (
                "anchor" if record["episode_id"] in anchor_ids else record["category"]
            ),
        }
        for record in records
        if record["episode_id"] in anchor_ids
        or record["category"] == "mixed"
    ]
    counts = {
        name: sum(record["category"] == name for record in records)
        for name in data.CATEGORIES
    }
    initial_counts = {
        name: sum(
            data._category_for(record["attempts"][:4], 1.5) == name
            for record in records
        )
        for name in data.CATEGORIES
    }
    meta = {
        "model": "frozen-sft",
        "base_url": "http://127.0.0.1:8001/v1",
        "initial_count": 4,
        "initial_temperature": 0.9,
        "retry_temperature": 0.8,
        "retry_count": 4,
        "anchor_fraction": 0.15,
        "anchor_sampling": "route_stratified_exact_target",
        "max_tokens": 1024,
        "seed": 42,
        "expected_episode_count": len(specs),
        "episodes_sha256": digest,
        "world_sha256": "world-hash",
        "sft_adapter_sha256": "adapter-hash",
        "sft_reference_weights_sha256": "reference-hash",
        "reward_policy": dict(_RUN2_REWARD),
        "behavior_files_sha256": data._probe_behavior_hashes(),
        "episodes_total": len(specs),
        "complete_population": True,
        **{f"initial_{name}": count for name, count in initial_counts.items()},
        **counts,
        "attempts_total": sum(len(record["attempts"]) for record in records),
        "anchors_kept": len(anchors),
        "train_on": len(train_on),
    }
    return {
        "schema_version": data.PROBE_SCHEMA,
        "meta": meta,
        "records": records,
        "train_on": train_on,
    }


def test_verl_data_contains_only_initial_prompt_and_episode_id(tmp_path):
    data = _load("world_policy_verl_data_test", "verl_data.py")
    specs = _mini_specs(4)
    episodes_path = tmp_path / "episodes.jsonl"
    episodes_path.write_text(
        "\n".join(json.dumps(spec) for spec in specs) + "\n", encoding="utf-8"
    )
    doc = _filter_document(
        data,
        specs,
        ["mixed", "mixed_after_retry", "unsolved", "solved"],
        digest=data._sha256(episodes_path),
    )
    filter_path = tmp_path / "filter.json"
    filter_path.write_text(json.dumps(doc), encoding="utf-8")
    rows = data.build_rows(episodes_path, filter_path, expected_count=4)
    assert len(rows) == 2
    assert set(rows[0]) == {"data_source", "agent_name", "prompt", "extra_info"}
    assert set(rows[0]["extra_info"]) == {"episode_id"}
    assert [message["role"] for message in rows[0]["prompt"]] == ["system", "user"]
    serialized = json.dumps(rows, ensure_ascii=False)
    assert all(name not in serialized for name in ('"attempts"', '"person_state"', '"state_graph"'))


def test_legacy_v2_probe_is_rejected(tmp_path):
    data = _load("world_policy_verl_data_legacy_test", "verl_data.py")
    specs = _mini_specs(1)
    episodes_path = tmp_path / "episodes.jsonl"
    episodes_path.write_text(json.dumps(specs[0]) + "\n", encoding="utf-8")
    filter_path = tmp_path / "filter.json"
    filter_path.write_text(json.dumps({
        "schema_version": "cn_travel.world_policy_probe.v2",
        "meta": {"episodes_sha256": data._sha256(episodes_path)},
    }), encoding="utf-8")
    with pytest.raises(SystemExit, match="requires v3"):
        data.build_rows(episodes_path, filter_path, expected_count=1)


def test_probe_runs_four_initial_and_retries_only_initial_unsolved():
    probe = _load("world_policy_adaptive_calls_test", "probe_difficulty.py")
    calls = []
    outcomes = iter([False] * 4 + [True, False, False, False])

    def run_one(number, phase, temperature):
        calls.append((number, phase, temperature))
        return {"goal_reached": next(outcomes)}

    attempts = probe._adaptive_attempts(
        run_one,
        initial_count=4,
        initial_temperature=0.9,
        retry_temperature=0.8,
        retry_count=4,
    )
    assert len(attempts) == 8
    assert calls == [
        (number, "initial", 0.9) for number in range(4)
    ] + [(number, "retry", 0.8) for number in range(4, 8)]

    calls.clear()
    outcomes = iter([False, True, False, False])
    attempts = probe._adaptive_attempts(
        lambda number, phase, temperature: (
            calls.append((number, phase, temperature))
            or {"goal_reached": next(outcomes)}
        ),
        initial_count=4,
        initial_temperature=0.9,
        retry_temperature=0.8,
        retry_count=4,
    )
    assert len(attempts) == 4
    assert calls == [(number, "initial", 0.9) for number in range(4)]


def test_anchor_sample_is_exact_seeded_and_route_stratified():
    data = _load("world_policy_probe_anchor_test", "verl_data.py")
    maxima = [
        {"episode_id": f"episode-{index}", "source_idx": index,
         "route": f"route-{index % 5}"}
        for index in range(100)
    ]
    first = data._sample_anchors(maxima, fraction=0.15, seed=42)
    second = data._sample_anchors(maxima, fraction=0.15, seed=42)
    assert len(first) == round(0.15 * len(maxima)) == 15
    assert {route: sum(item["route"] == route for item in first)
            for route in {item["route"] for item in maxima}} == {
                f"route-{index}": 3 for index in range(5)
            }
    assert [item["episode_id"] for item in first] == [item["episode_id"] for item in second]


def test_probe_categories_use_all_applicable_outcomes():
    data = _load("world_policy_probe_category_test", "verl_data.py")
    spec = _mini_specs(1)[0]
    assert data._category_for(_record(spec, 0, "solved")["attempts"], 1.5) == "solved"
    assert data._category_for(_record(spec, 0, "mixed")["attempts"], 1.5) == "mixed"
    recovered = _record(spec, 0, "mixed_after_retry")["attempts"]
    assert data._category_for(recovered[:4], 1.5) == "unsolved"
    assert data._category_for(recovered, 1.5) == "mixed"
    assert data._category_for(_record(spec, 0, "unsolved")["attempts"], 1.5) == "unsolved"

    four_failures_then_four_maxima = [
        _attempt(0, number, goal=False) for number in range(4)
    ] + [_attempt(0, number, goal=True) for number in range(4, 8)]
    assert data._category_for(four_failures_then_four_maxima, 1.5) == "mixed"


def test_adaptive_categories_and_exact_retention_contract():
    data = _load("world_policy_probe_filter_test", "verl_data.py")
    specs = _mini_specs(23)
    categories = ["solved"] * 20 + ["mixed", "mixed_after_retry", "unsolved"]
    doc = _filter_document(data, specs, categories)
    episodes = {spec["episode_id"]: spec for spec in specs}
    data._validate_filter_contract(doc, episodes, 23)
    assert doc["meta"]["anchors_kept"] == round(0.15 * 20) == 3
    adaptive = next(item for item in doc["train_on"]
                    if item["episode_id"] == "episode-21")
    assert adaptive["selection"] == "mixed"
    assert "episode-22" not in {item["episode_id"] for item in doc["train_on"]}
    assert doc["meta"]["initial_unsolved"] == 2
    assert doc["meta"]["attempts_total"] == 4 * 23 + 4 * 2

    wrong_category = json.loads(json.dumps(doc))
    wrong_category["records"][21]["category"] = "solved"
    with pytest.raises(SystemExit, match="category does not match"):
        data._validate_filter_contract(wrong_category, episodes, 23)

    retained_failure = json.loads(json.dumps(doc))
    failure = retained_failure["records"][22]
    retained_failure["train_on"].append({
        "episode_id": failure["episode_id"], "source_idx": failure["source_idx"],
        "route": failure["route"], "selection": "unsolved",
    })
    retained_failure["meta"]["train_on"] += 1
    with pytest.raises(SystemExit, match="retained set differs"):
        data._validate_filter_contract(retained_failure, episodes, 23)


@pytest.mark.parametrize("u, expected", [(0, 1.5), (1, 1.3), (2, 1.1), (5, 0.5), (6, 0.5)])
def test_probe_contract_enforces_success_reward_formula(u, expected):
    data = _load(f"world_policy_reward_formula_{u}", "verl_data.py")
    specs = _mini_specs(1)
    category = "solved" if u == 0 else "mixed"
    doc = _filter_document(data, specs, [category])
    attempt = doc["records"][0]["attempts"][0]
    attempt["U"] = u
    attempt["reward"] = expected
    attempt["clean_trace"] = u == 0
    attempt["outcome"]["violations"] = ["wrong_question"] * u
    attempt["outcome"]["clean_trace"] = u == 0
    attempt["outcome"]["reward"] = {"total": expected, "violation_count": u}
    data._validate_filter_contract(doc, {specs[0]["episode_id"]: specs[0]}, 1)


def test_probe_contract_rejects_prefix_credit_and_inconsistent_sampling():
    data = _load("world_policy_probe_fail_closed_test", "verl_data.py")
    specs = _mini_specs(1)
    episodes = {specs[0]["episode_id"]: specs[0]}
    doc = _filter_document(data, specs, ["unsolved"])
    initial = doc["records"][0]["attempts"][0]
    initial["reward"] = 0.4
    initial["outcome"]["reward"]["total"] = 0.4
    with pytest.raises(SystemExit, match="reward formula"):
        data._validate_filter_contract(doc, episodes, 1)

    doc = _filter_document(data, specs, ["unsolved"])
    doc["records"][0]["attempts"].pop()
    doc["meta"]["attempts_total"] -= 1
    with pytest.raises(SystemExit, match="expected 8 adaptive attempts"):
        data._validate_filter_contract(doc, episodes, 1)

    doc = _filter_document(data, specs, ["unsolved"])
    doc["records"][0]["attempts"][4]["temperature"] = 0.9
    with pytest.raises(SystemExit, match="sampling identity"):
        data._validate_filter_contract(doc, episodes, 1)


def _reused_probe_fixture():
    data = _load("world_policy_probe_fixture_data", "verl_data.py")
    specs = _mini_specs(4)
    source = _filter_document(data, specs, [
        "solved", "mixed", "mixed_after_retry", "unsolved",
    ])
    identity = {
        key: value for key, value in source["meta"].items()
        if key in {
            "model", "initial_count", "initial_temperature", "retry_temperature",
            "retry_count",
            "anchor_fraction", "max_tokens", "seed", "expected_episode_count",
            "episodes_sha256", "world_sha256", "sft_adapter_sha256",
            "sft_reference_weights_sha256", "reward_policy",
            "behavior_files_sha256",
        }
    }
    source["records"] = list(reversed(source["records"]))
    return source, specs, identity


def test_probe_from_records_accepts_complete_v3_and_rejects_stale_bytes():
    probe = _load("world_policy_probe_reuse_test", "probe_difficulty.py")
    source, specs, identity = _reused_probe_fixture()
    records = probe._validated_reused_records(
        source, specs=specs, required_identity=identity
    )
    assert [record["episode_id"] for record in records] == [
        spec["episode_id"] for spec in specs
    ]

    stale = json.loads(json.dumps(source))
    stale["meta"]["retry_temperature"] = 0.7
    with pytest.raises(SystemExit, match="different identity"):
        probe._validated_reused_records(stale, specs=specs, required_identity=identity)

    legacy = json.loads(json.dumps(source))
    legacy["schema_version"] = "cn_travel.world_policy_probe.v2"
    with pytest.raises(SystemExit, match="unsupported"):
        probe._validated_reused_records(legacy, specs=specs, required_identity=identity)


def test_verl_data_binds_task_filter_to_exact_adaptive_probe_policy():
    data = _load("world_policy_verl_data_probe_identity_test", "verl_data.py")
    config = json.loads((VERL_DIR / "verl_traj_config.json").read_text(encoding="utf-8"))
    meta = {
        "model": config["probe_model"],
        "initial_count": config["probe_initial_count"],
        "initial_temperature": config["probe_initial_temperature"],
        "retry_temperature": config["probe_retry_temperature"],
        "retry_count": config["probe_retry_count"],
        "anchor_fraction": config["probe_anchor_fraction"],
        "max_tokens": config["probe_max_tokens"],
        "seed": config["seed"],
        "expected_episode_count": config["expected_episode_count"],
        "episodes_sha256": "episodes",
        "world_sha256": "world",
        "sft_adapter_sha256": config["sft_adapter_sha256"],
        "sft_reference_weights_sha256": config["sft_reference_weights_sha256"],
        "reward_policy": config["reward"],
        "behavior_files_sha256": data._probe_behavior_hashes(),
    }
    data._validate_probe_identity(
        meta, config, episodes_sha256="episodes", world_sha256="world"
    )
    stale = dict(meta, retry_count=3)
    with pytest.raises(SystemExit, match="different probe identity"):
        data._validate_probe_identity(
            stale, config, episodes_sha256="episodes", world_sha256="world"
        )


def test_run2_probe_identity_resolves_to_bundled_historical_source():
    data = _load("world_policy_run2_bundled_identity_test", "verl_data.py")
    filter_doc = json.loads(
        (DATA_ROOT / "7_world_policy" / "task_filter_run2.json").read_text(
            encoding="utf-8"
        )
    )
    recorded = filter_doc["meta"]["behavior_files_sha256"]
    data._validate_probe_behavior_identity(recorded)

    corrupted = dict(recorded)
    key = next(iter(corrupted))
    corrupted[key] = "0" * 64
    with pytest.raises(SystemExit, match="unbundled"):
        data._validate_probe_behavior_identity(corrupted)


def _install_verl_stubs(monkeypatch):
    class AgentLoopBase:
        pass

    class AgentLoopOutput:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def register(_name):
        return lambda cls: cls

    @contextlib.contextmanager
    def simple_timer(_name, _metrics):
        yield

    modules = {
        "verl": types.ModuleType("verl"),
        "verl.experimental": types.ModuleType("verl.experimental"),
        "verl.experimental.agent_loop": types.ModuleType("verl.experimental.agent_loop"),
        "verl.experimental.agent_loop.agent_loop": types.ModuleType(
            "verl.experimental.agent_loop.agent_loop"
        ),
        "verl.utils": types.ModuleType("verl.utils"),
        "verl.utils.profiler": types.ModuleType("verl.utils.profiler"),
        "verl.utils.rollout_trace": types.ModuleType("verl.utils.rollout_trace"),
    }
    modules["verl.experimental.agent_loop.agent_loop"].AgentLoopBase = AgentLoopBase
    modules["verl.experimental.agent_loop.agent_loop"].AgentLoopOutput = AgentLoopOutput
    modules["verl.experimental.agent_loop.agent_loop"].register = register
    modules["verl.utils.profiler"].simple_timer = simple_timer
    modules["verl.utils.rollout_trace"].rollout_trace_op = lambda function: function
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_verl_loop_marks_cap_before_scoring_sliced_emission(monkeypatch):
    _install_verl_stubs(monkeypatch)
    loop_module = _load("world_policy_verl_loop_cap_test", "verl_loop.py")

    class Runtime:
        initial_messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]

        def __init__(self):
            self.step_calls = 0
            self.truncation_reason = None

        def reset(self):
            return self

        def interpret_and_step(self, _emission):
            self.step_calls += 1
            raise AssertionError("a capped emission must not be scored")

        def truncate(self, reason):
            self.truncation_reason = reason

        def reward(self):
            return SimpleNamespace(total=0.0)

        def outcome(self):
            return {"truncated": True, "truncation_reason": self.truncation_reason}

    class Server:
        def __init__(self):
            self.sampling = None

        async def generate(self, **kwargs):
            self.sampling = kwargs["sampling_params"]
            return SimpleNamespace(token_ids=[11, 12, 13, 14])

    runtime = Runtime()
    server = Server()
    loop = loop_module.TravelAgentLoop.__new__(loop_module.TravelAgentLoop)
    loop.response_length = 3
    loop.max_assistant_turns = 24
    loop.server_manager = server
    loop.tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: "final")
    loop.turn_separator = []
    loop._runtime = lambda _episode_id: runtime

    async def apply_chat_template(_messages, **_kwargs):
        return [101, 102]

    loop.apply_chat_template = apply_chat_template
    result = asyncio.run(
        loop.run(
            {"temperature": 0.9, "max_tokens": 50},
            raw_prompt=runtime.initial_messages,
            extra_info={"episode_id": "episode-1"},
        )
    )

    assert server.sampling["max_tokens"] == 3
    assert runtime.step_calls == 0
    assert runtime.truncation_reason == "response_length"
    assert result.response_ids == [11, 12, 13]
    assert result.response_mask == [1, 1, 1]
    assert result.reward_score == 0.0


def test_verl_loop_masks_environment_tokens_and_scores_once(monkeypatch):
    _install_verl_stubs(monkeypatch)
    loop_module = _load("world_policy_verl_loop_mask_test", "verl_loop.py")

    class Runtime:
        initial_messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]

        def __init__(self):
            self.steps = 0
            self.reward_calls = 0

        def reset(self):
            return self

        def interpret_and_step(self, _emission):
            self.steps += 1
            if self.steps == 1:
                return SimpleNamespace(
                    terminal=False,
                    messages=({"role": "user", "content": "environment"},),
                )
            return SimpleNamespace(terminal=True, messages=())

        def reward(self):
            self.reward_calls += 1
            return SimpleNamespace(total=1.5)

        def outcome(self):
            return {"goal_reached": True, "clean_trace": True}

    class Server:
        def __init__(self):
            self.outputs = iter(([11], [12]))

        async def generate(self, **_kwargs):
            return SimpleNamespace(token_ids=next(self.outputs))

    runtime = Runtime()
    loop = loop_module.TravelAgentLoop.__new__(loop_module.TravelAgentLoop)
    loop.response_length = 10
    loop.max_assistant_turns = 4
    loop.server_manager = Server()
    loop.tokenizer = SimpleNamespace(decode=lambda tokens, **_kwargs: f"turn-{tokens[0]}")
    loop.turn_separator = []
    loop._runtime = lambda _episode_id: runtime

    async def apply_chat_template(messages, **_kwargs):
        return [101, 102] if messages == runtime.initial_messages else [201, 202]

    loop.apply_chat_template = apply_chat_template
    result = asyncio.run(
        loop.run(
            {"temperature": 0.9, "max_tokens": 5},
            raw_prompt=runtime.initial_messages,
            extra_info={"episode_id": "episode-1"},
        )
    )
    assert result.response_ids == [11, 201, 202, 12]
    assert result.response_mask == [1, 0, 0, 1]
    assert runtime.steps == 2
    assert runtime.reward_calls == 1
    assert result.reward_score == 1.5


def test_verl_loop_strips_only_transport_eos_but_keeps_its_token_and_mask(monkeypatch):
    _install_verl_stubs(monkeypatch)
    loop_module = _load("world_policy_verl_loop_eos_test", "verl_loop.py")

    class Runtime:
        initial_messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]

        def __init__(self):
            self.emission = None

        def reset(self):
            return self

        def interpret_and_step(self, emission):
            self.emission = emission
            return SimpleNamespace(terminal=True, messages=())

        def reward(self):
            return SimpleNamespace(total=1.5)

        def outcome(self):
            return {"goal_reached": True, "clean_trace": True}

    class Server:
        async def generate(self, **_kwargs):
            return SimpleNamespace(token_ids=[11, 12])

    native = "<|tool_call_start|>[search_travel_guide(location='信阳')]<|tool_call_end|>"
    runtime = Runtime()
    loop = loop_module.TravelAgentLoop.__new__(loop_module.TravelAgentLoop)
    loop.response_length = 10
    loop.max_assistant_turns = 4
    loop.server_manager = Server()
    loop.tokenizer = SimpleNamespace(
        eos_token="<|im_end|>",
        decode=lambda *_args, **_kwargs: native + "<|im_end|>",
    )
    loop.turn_separator = []
    loop._runtime = lambda _episode_id: runtime

    async def apply_chat_template(_messages, **_kwargs):
        return [101, 102]

    loop.apply_chat_template = apply_chat_template
    result = asyncio.run(loop.run(
        {"temperature": 0.9, "max_tokens": 5},
        raw_prompt=runtime.initial_messages,
        extra_info={"episode_id": "episode-1"},
    ))
    assert runtime.emission == native
    assert result.response_ids == [11, 12]
    assert result.response_mask == [1, 1]


def test_verl_loop_propagates_external_parser_failure_without_scoring(monkeypatch):
    _install_verl_stubs(monkeypatch)
    loop_module = _load("world_policy_verl_loop_exception_test", "verl_loop.py")

    class Runtime:
        initial_messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]

        def __init__(self):
            self.reward_calls = 0

        def reset(self):
            return self

        def interpret_and_step(self, _emission):
            raise RuntimeError("external parser failure")

        def reward(self):
            self.reward_calls += 1
            return SimpleNamespace(total=0.0)

    class Server:
        async def generate(self, **_kwargs):
            return SimpleNamespace(token_ids=[11])

    runtime = Runtime()
    loop = loop_module.TravelAgentLoop.__new__(loop_module.TravelAgentLoop)
    loop.response_length = 10
    loop.max_assistant_turns = 4
    loop.server_manager = Server()
    loop.tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: "emission")
    loop.turn_separator = []
    loop._runtime = lambda _episode_id: runtime

    async def apply_chat_template(_messages, **_kwargs):
        return [101, 102]

    loop.apply_chat_template = apply_chat_template
    with pytest.raises(RuntimeError, match="external parser failure"):
        asyncio.run(loop.run(
            {"temperature": 0.9, "max_tokens": 5},
            raw_prompt=runtime.initial_messages,
            extra_info={"episode_id": "episode-1"},
        ))
    assert runtime.reward_calls == 0
