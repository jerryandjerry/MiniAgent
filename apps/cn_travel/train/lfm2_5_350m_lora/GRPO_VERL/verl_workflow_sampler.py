#!/usr/bin/env python3
"""Deterministic, resumable workflow-stratified sampler for CN Travel Run #3.

VERL 0.9.0 does not expose a configurable sampler class.  This module is
loaded through ``VERL_USE_EXTERNAL_MODULES`` and replaces only VERL's sampler
factory.  The replacement remains a normal PyTorch sampler and implements the
state protocol consumed by TorchData's ``StatefulDataLoader``.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import pathlib
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Iterator

from torch.utils.data import Sampler


SCHEMA = "cn_travel.workflow_sampler.v1"
POLICY = "workflow_stratified_v1"
EXPECTED_VERL = "0.9.0"
HERE = pathlib.Path(__file__).resolve().parent
APP_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = HERE / "verl_traj_config_run3.json"


class SamplerContractError(ValueError):
    """The data, sampler configuration, or resume state violates Run #3."""


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve(value: str | os.PathLike[str]) -> pathlib.Path:
    path = pathlib.Path(value)
    return path if path.is_absolute() else APP_ROOT / path


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def _stable_permutation(values: Sequence[Any], *, salt: str) -> list[Any]:
    """Return a hash-keyed permutation independent of process hash seeding."""
    decorated = []
    for position, value in enumerate(values):
        identity = json.dumps(value, ensure_ascii=False, sort_keys=True)
        key = hashlib.sha256(
            f"{salt}\0{position}\0{identity}".encode("utf-8")
        ).digest()
        decorated.append((key, position, value))
    decorated.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in decorated]


def _dataset_labels(dataset: Any) -> list[tuple[int, str, str]]:
    """Read sampler-only labels without tokenizing or mutating model prompts."""
    try:
        size = len(dataset)
    except Exception as exc:  # pragma: no cover - defensive integration guard
        raise SamplerContractError("VERL dataset has no stable length") from exc

    dataframe = getattr(dataset, "dataframe", None)
    extras: Any = None
    if dataframe is not None:
        try:
            extras = dataframe["extra_info"]
        except Exception:
            extras = None

    labels: list[tuple[int, str, str]] = []
    for index in range(size):
        if extras is not None:
            extra = extras[index]
        else:
            try:
                row = dataset[index]
            except Exception as exc:
                raise SamplerContractError(
                    f"cannot read sampler labels for dataset row {index}"
                ) from exc
            extra = row.get("extra_info") if isinstance(row, Mapping) else None
        if not isinstance(extra, Mapping):
            raise SamplerContractError(f"dataset row {index} lacks extra_info")
        episode_id = extra.get("episode_id")
        workflow = extra.get("workflow")
        if not isinstance(episode_id, str) or not episode_id:
            raise SamplerContractError(f"dataset row {index} has an invalid episode_id")
        if not isinstance(workflow, str) or not workflow:
            raise SamplerContractError(f"{episode_id}: missing workflow sampler label")
        labels.append((index, episode_id, workflow))
    return labels


def _normalized_contract(config: Mapping[str, Any], labels: Sequence[tuple[int, str, str]]) -> dict[str, Any]:
    raw = config.get("workflow_stratified_sampler")
    if not isinstance(raw, Mapping) or raw.get("policy") != POLICY:
        raise SamplerContractError(f"Run #3 requires sampler policy {POLICY}")

    workflows = raw.get("workflows")
    if not isinstance(workflows, list) or not workflows or len(set(workflows)) != len(workflows):
        raise SamplerContractError("sampler workflows must be a unique non-empty list")
    if any(not isinstance(item, str) or not item for item in workflows):
        raise SamplerContractError("sampler workflow names must be non-empty strings")

    batch_size = config.get("train_batch_size")
    seed = config.get("seed")
    expected_count = config.get("expected_episode_count")
    if batch_size != 32 or seed != 42 or expected_count != 909:
        raise SamplerContractError("Run #3 sampler requires batch_size=32, seed=42, and 909 episodes")
    if len(labels) != expected_count:
        raise SamplerContractError(
            f"sampler received {len(labels)} rows; expected {expected_count}"
        )
    episode_ids = [episode_id for _, episode_id, _ in labels]
    if len(set(episode_ids)) != len(episode_ids):
        raise SamplerContractError("sampler dataset contains duplicate episode IDs")

    def integer_map(name: str, *, positive: bool) -> dict[str, int]:
        value = raw.get(name)
        if not isinstance(value, Mapping) or set(value) != set(workflows):
            raise SamplerContractError(f"sampler {name} must cover exactly {workflows}")
        result: dict[str, int] = {}
        for workflow in workflows:
            number = value[workflow]
            lower = 1 if positive else 0
            if not isinstance(number, int) or isinstance(number, bool) or number < lower:
                raise SamplerContractError(f"sampler {name}.{workflow} is invalid")
            result[workflow] = number
        return result

    source_counts = integer_map("source_counts", positive=True)
    epoch_counts = integer_map("epoch_counts", positive=True)
    observed = Counter(workflow for _, _, workflow in labels)
    if set(observed) != set(workflows) or dict(observed) != source_counts:
        raise SamplerContractError(
            f"workflow source counts differ: observed={dict(observed)}, expected={source_counts}"
        )

    quota_types_raw = raw.get("batch_quota_types")
    type_counts_raw = raw.get("batch_type_counts")
    if not isinstance(quota_types_raw, Mapping) or not quota_types_raw:
        raise SamplerContractError("sampler batch_quota_types is missing")
    if not isinstance(type_counts_raw, Mapping) or set(type_counts_raw) != set(quota_types_raw):
        raise SamplerContractError("sampler batch_type_counts disagrees with quota types")
    quota_types: dict[str, dict[str, int]] = {}
    type_counts: dict[str, int] = {}
    for name, quotas in quota_types_raw.items():
        if not isinstance(name, str) or not isinstance(quotas, Mapping) or set(quotas) != set(workflows):
            raise SamplerContractError(f"invalid batch quota type {name!r}")
        normalized: dict[str, int] = {}
        for workflow in workflows:
            number = quotas[workflow]
            if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
                raise SamplerContractError(f"batch quota {name}.{workflow} must be positive")
            normalized[workflow] = number
        if sum(normalized.values()) != batch_size:
            raise SamplerContractError(f"batch quota type {name} does not sum to {batch_size}")
        occurrences = type_counts_raw[name]
        if not isinstance(occurrences, int) or isinstance(occurrences, bool) or occurrences <= 0:
            raise SamplerContractError(f"batch type count {name} must be positive")
        quota_types[name] = normalized
        type_counts[name] = occurrences

    batches_per_epoch = raw.get("batches_per_epoch")
    presentations = raw.get("presentations_per_epoch")
    padding = raw.get("padding_presentations_per_epoch")
    max_occurrences = raw.get("max_occurrences_per_episode_per_epoch")
    if batches_per_epoch != sum(type_counts.values()) or batches_per_epoch != 29:
        raise SamplerContractError("Run #3 requires exactly 29 batches per epoch")
    if presentations != batches_per_epoch * batch_size or presentations != 928:
        raise SamplerContractError("Run #3 requires exactly 928 presentations per epoch")
    if padding != presentations - expected_count or padding != 19:
        raise SamplerContractError("Run #3 requires the minimum 19 padding presentations")
    if max_occurrences != 2:
        raise SamplerContractError("Run #3 permits at most two occurrences per episode per epoch")
    if raw.get("shuffle_within_workflow") is not True or raw.get("shuffle_batch_types") is not True:
        raise SamplerContractError("both Run #3 sampler shuffle controls must be enabled")

    aggregate = {workflow: 0 for workflow in workflows}
    for name, occurrences in type_counts.items():
        for workflow in workflows:
            aggregate[workflow] += quota_types[name][workflow] * occurrences
    if aggregate != epoch_counts:
        raise SamplerContractError(
            f"batch quotas produce {aggregate}, not configured epoch counts {epoch_counts}"
        )
    if any(epoch_counts[name] < source_counts[name] for name in workflows):
        raise SamplerContractError("epoch counts may not drop source episodes")
    if sum(epoch_counts.values()) != presentations:
        raise SamplerContractError("epoch workflow counts do not sum to presentations")

    return {
        "policy": POLICY,
        "workflows": workflows,
        "source_counts": source_counts,
        "epoch_counts": epoch_counts,
        "batch_quota_types": quota_types,
        "batch_type_counts": type_counts,
        "batches_per_epoch": batches_per_epoch,
        "presentations_per_epoch": presentations,
        "padding_presentations_per_epoch": padding,
        "max_occurrences_per_episode_per_epoch": max_occurrences,
        "batch_size": batch_size,
        "seed": seed,
    }


class WorkflowStratifiedSampler(Sampler[int]):
    """One full, workflow-mixed, deterministic schedule per logical epoch."""

    def __init__(
        self,
        dataset: Any,
        config: Mapping[str, Any],
        *,
        start_epoch: int = 0,
    ) -> None:
        if not isinstance(start_epoch, int) or isinstance(start_epoch, bool) or start_epoch < 0:
            raise SamplerContractError("start_epoch must be a non-negative integer")
        self._labels = _dataset_labels(dataset)
        self._contract = _normalized_contract(config, self._labels)
        self._workflow_by_index = {index: workflow for index, _, workflow in self._labels}
        self._episode_by_index = {index: episode_id for index, episode_id, _ in self._labels}
        self._indices_by_workflow: dict[str, list[int]] = defaultdict(list)
        for index, _, workflow in self._labels:
            self._indices_by_workflow[workflow].append(index)
        identity = {
            "schema": SCHEMA,
            "contract": self._contract,
            "rows": [
                {"index": index, "episode_id": episode_id, "workflow": workflow}
                for index, episode_id, workflow in self._labels
            ],
        }
        self.schedule_fingerprint = _canonical_digest(identity)
        self._epoch = start_epoch
        self._yielded = 0
        self._cache: dict[int, tuple[int, ...]] = {}

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def yielded(self) -> int:
        return self._yielded

    @property
    def batches_per_epoch(self) -> int:
        return int(self._contract["batches_per_epoch"])

    @property
    def batch_size(self) -> int:
        return int(self._contract["batch_size"])

    def __len__(self) -> int:
        return int(self._contract["presentations_per_epoch"])

    def epoch_schedule(self, epoch: int) -> tuple[int, ...]:
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise SamplerContractError("epoch must be a non-negative integer")
        if epoch in self._cache:
            return self._cache[epoch]

        seed = self._contract["seed"]
        workflows: list[str] = self._contract["workflows"]
        quota_types: dict[str, dict[str, int]] = self._contract["batch_quota_types"]
        type_counts: dict[str, int] = self._contract["batch_type_counts"]
        batch_tags = [
            (name, ordinal)
            for name in sorted(type_counts)
            for ordinal in range(type_counts[name])
        ]
        batch_tags = _stable_permutation(
            batch_tags,
            salt=f"{SCHEMA}|batch-types|seed={seed}|epoch={epoch}",
        )
        batches: list[list[int]] = [[] for _ in batch_tags]

        for workflow in workflows:
            pool = _stable_permutation(
                self._indices_by_workflow[workflow],
                salt=f"{SCHEMA}|workflow={workflow}|seed={seed}|epoch={epoch}",
            )
            padding_pool = _stable_permutation(
                pool,
                salt=f"{SCHEMA}|padding={workflow}|seed={seed}|epoch={epoch}",
            )
            cursor = 0
            repeated: set[int] = set()
            for batch_index, (batch_type, _) in enumerate(batch_tags):
                required = quota_types[batch_type][workflow]
                available = min(required, len(pool) - cursor)
                if available:
                    batches[batch_index].extend(pool[cursor : cursor + available])
                    cursor += available
                missing = required - available
                for _ in range(missing):
                    candidate = next(
                        (
                            index
                            for index in padding_pool
                            if index not in repeated and index not in batches[batch_index]
                        ),
                        None,
                    )
                    if candidate is None:
                        raise SamplerContractError(
                            f"cannot allocate non-colliding padding for {workflow}"
                        )
                    batches[batch_index].append(candidate)
                    repeated.add(candidate)
            if cursor != len(pool):
                raise SamplerContractError(f"sampler dropped source rows from {workflow}")

        shuffled_batches = [
            _stable_permutation(
                batch,
                salt=f"{SCHEMA}|within-batch={batch_index}|seed={seed}|epoch={epoch}",
            )
            for batch_index, batch in enumerate(batches)
        ]
        schedule = tuple(index for batch in shuffled_batches for index in batch)
        self._validate_schedule(schedule, batch_tags)
        self._cache[epoch] = schedule
        return schedule

    def _validate_schedule(
        self,
        schedule: Sequence[int],
        batch_tags: Sequence[tuple[str, int]],
    ) -> None:
        if len(schedule) != len(self):
            raise SamplerContractError("epoch schedule has the wrong length")
        counts = Counter(schedule)
        source_indices = set(self._episode_by_index)
        if set(counts) != source_indices:
            raise SamplerContractError("epoch schedule does not cover every source episode")
        if max(counts.values()) > self._contract["max_occurrences_per_episode_per_epoch"]:
            raise SamplerContractError("an episode exceeds its per-epoch occurrence limit")
        repeats = sum(number - 1 for number in counts.values())
        if repeats != self._contract["padding_presentations_per_epoch"]:
            raise SamplerContractError("epoch schedule has the wrong padding count")
        observed_workflows = Counter(self._workflow_by_index[index] for index in schedule)
        if dict(observed_workflows) != self._contract["epoch_counts"]:
            raise SamplerContractError("epoch schedule has the wrong workflow totals")

        size = self.batch_size
        for batch_index, (batch_type, _) in enumerate(batch_tags):
            batch = list(schedule[batch_index * size : (batch_index + 1) * size])
            if len(batch) != size or len(set(batch)) != size:
                raise SamplerContractError("every batch must contain 32 distinct episode IDs")
            observed = Counter(self._workflow_by_index[index] for index in batch)
            if dict(observed) != self._contract["batch_quota_types"][batch_type]:
                raise SamplerContractError(
                    f"batch {batch_index} differs from quota type {batch_type}"
                )

    def __iter__(self) -> Iterator[int]:
        if self._yielded == len(self):
            self._epoch += 1
            self._yielded = 0
        schedule = self.epoch_schedule(self._epoch)
        while self._yielded < len(schedule):
            index = schedule[self._yielded]
            self._yielded += 1
            yield index
        if self._yielded == len(schedule):
            self._epoch += 1
            self._yielded = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "epoch": self._epoch,
            "yielded": self._yielded,
            "schedule_fingerprint": self.schedule_fingerprint,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        expected_keys = {"schema", "epoch", "yielded", "schedule_fingerprint"}
        if not isinstance(state_dict, Mapping) or set(state_dict) != expected_keys:
            raise SamplerContractError("invalid workflow sampler state")
        if state_dict["schema"] != SCHEMA:
            raise SamplerContractError("workflow sampler state schema differs")
        if state_dict["schedule_fingerprint"] != self.schedule_fingerprint:
            raise SamplerContractError("workflow sampler state belongs to different data/config")
        epoch = state_dict["epoch"]
        yielded = state_dict["yielded"]
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise SamplerContractError("workflow sampler state has an invalid epoch")
        if (
            not isinstance(yielded, int)
            or isinstance(yielded, bool)
            or yielded < 0
            or yielded > len(self)
        ):
            raise SamplerContractError("workflow sampler state has an invalid yielded count")
        self._epoch = epoch
        self._yielded = yielded


def _load_run_config() -> tuple[pathlib.Path, dict[str, Any]]:
    path = pathlib.Path(os.environ.get("CN_TRAVEL_VERL_CONFIG", DEFAULT_CONFIG)).resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SamplerContractError(f"cannot load Run #3 config: {path}") from exc
    if not isinstance(value, dict) or value.get("training_population") != "all_episodes":
        raise SamplerContractError("workflow sampler may only load for all-episode Run #3")
    return path, value


def _start_epoch(config: Mapping[str, Any]) -> int:
    contract = config.get("workflow_stratified_sampler")
    if not isinstance(contract, Mapping):
        raise SamplerContractError("Run #3 sampler contract is missing")
    steps_per_epoch = contract.get("batches_per_epoch")
    if steps_per_epoch != 29:
        raise SamplerContractError("Run #3 resume requires 29 steps per epoch")
    tracker = _resolve(config["checkpoint_dir"]) / "latest_checkpointed_iteration.txt"
    if not tracker.exists():
        return 0
    try:
        global_step = int(tracker.read_text(encoding="utf-8").strip())
    except Exception as exc:
        raise SamplerContractError(f"invalid checkpoint tracker: {tracker}") from exc
    total_steps = steps_per_epoch * int(config["num_train_epochs"])
    if global_step < 0 or global_step > total_steps:
        raise SamplerContractError(
            f"checkpoint step {global_step} lies outside Run #3 total {total_steps}"
        )
    return global_step // steps_per_epoch


def _install_verl_patch() -> None:
    global PATCH_ACTIVE
    import verl
    from verl.trainer.ppo import utils as ppo_utils

    if verl.__version__ != EXPECTED_VERL:
        raise SamplerContractError(
            f"workflow sampler requires verl=={EXPECTED_VERL}, installed {verl.__version__}"
        )
    original = ppo_utils.create_rl_sampler
    if getattr(original, "_cn_travel_workflow_stratified", False):
        PATCH_ACTIVE = True
        return
    parameters = tuple(inspect.signature(original).parameters)
    try:
        source = inspect.getsource(original)
    except (OSError, TypeError) as exc:
        raise SamplerContractError("cannot audit VERL's sampler factory source") from exc
    if parameters != ("data_config", "dataset") or "RandomSampler" not in source:
        raise SamplerContractError("installed VERL sampler factory differs from pinned v0.9.0")

    config_path, config = _load_run_config()

    def create_workflow_sampler(data_config: Any, dataset: Any) -> WorkflowStratifiedSampler:
        if _get(data_config, "shuffle") is not False:
            raise SamplerContractError("VERL data.shuffle must be false for the workflow sampler")
        if _get(data_config, "filter_overlong_prompts") is not False:
            raise SamplerContractError(
                "VERL prompt filtering must be disabled so all 909 episodes remain"
            )
        if _get(data_config, "train_batch_size") != config["train_batch_size"]:
            raise SamplerContractError("VERL train batch size differs from the Run #3 contract")
        sampler = WorkflowStratifiedSampler(
            dataset,
            config,
            start_epoch=_start_epoch(config),
        )
        print(
            "Run #3 workflow sampler: "
            f"{len(dataset)} unique rows -> {sampler.batches_per_epoch}x{sampler.batch_size}; "
            f"start_epoch={sampler.epoch}; fingerprint={sampler.schedule_fingerprint}",
            flush=True,
        )
        return sampler

    create_workflow_sampler._cn_travel_workflow_stratified = True  # type: ignore[attr-defined]
    create_workflow_sampler._cn_travel_original = original  # type: ignore[attr-defined]
    ppo_utils.create_rl_sampler = create_workflow_sampler
    for module_name in (
        "verl.trainer.main_ppo_v0",
        "verl.trainer.ppo.ray_trainer",
    ):
        module = sys.modules.get(module_name)
        if module is not None and getattr(module, "create_rl_sampler", None) is original:
            module.create_rl_sampler = create_workflow_sampler
    PATCH_ACTIVE = True
    print(f"installed Run #3 workflow sampler patch from {config_path}", flush=True)


PATCH_ACTIVE = False
if importlib.util.find_spec("verl") is not None:
    _install_verl_patch()
