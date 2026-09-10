import importlib
import json
from pathlib import Path
import sys

import pytest


APPLICATION_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APPLICATION_ROOT / "data" / "scripts"))
evaluation_module = importlib.import_module("_evaluation")


def test_evaluation_config_declares_stable_launcher_fields() -> None:
    config = json.loads(
        (APPLICATION_ROOT / "eval/config.json").read_text(encoding="utf-8")
    )
    assert set(config) == {
        "schema_version",
        "status",
        "command",
        "dataset_manifest",
        "sealed_dataset",
        "evaluator_configuration",
        "output_manifest",
    }
    assert config["schema_version"] == "miniagent.evaluation-run.v1"


def test_promotion_launcher_is_valid_python() -> None:
    source = (APPLICATION_ROOT / "eval/promote.py").read_text(encoding="utf-8")
    compile(source, "eval/promote.py", "exec")


def test_evaluation_evidence_must_use_the_sealed_episode_ids(tmp_path) -> None:
    split = tmp_path / "evaluation.jsonl"
    split.write_text(
        '{"episode_id":"sealed-1"}\n{"episode_id":"sealed-2"}\n',
        encoding="utf-8",
    )
    assert evaluation_module._sealed_case_ids(split, "jsonl") == {
        "sealed-1",
        "sealed-2",
    }
    fabricated = {
        "other-1": {"case_id": "other-1"},
        "other-2": {"case_id": "other-2"},
    }
    with pytest.raises(evaluation_module.ContractError, match="every sealed case"):
        evaluation_module._validate_case_coverage(
            fabricated,
            fabricated,
            evaluation_module._sealed_case_ids(split, "jsonl"),
        )


def test_evaluation_metrics_are_derived_from_score_records() -> None:
    transcript = {"case_id": "sealed-1", "goal_reached": False}
    transcripts = {"sealed-1": transcript}
    scores = {
        "sealed-1": {
            "case_id": "sealed-1",
            "status": "completed",
            "transcript_sha256": evaluation_module._canonical_sha256(transcript),
            "metrics": {"goal_completion": 0.0},
            "slices": [],
        }
    }
    metrics = [
        {
            "name": "goal_completion",
            "value": 1.0,
            "aggregation": "mean",
        }
    ]
    summary = {"attempted": 1, "completed": 1, "errors": 0}
    with pytest.raises(evaluation_module.ContractError, match="from score evidence"):
        evaluation_module._validate_score_records(
            transcripts, scores, metrics, summary
        )
