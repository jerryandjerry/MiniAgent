"""Verification for immutable MiniAgent evaluation results."""

import hashlib
import json
import math
from pathlib import Path
import shlex
from typing import Any, Dict, Optional, Set

from _artifacts import resolve_path, verify_reference, verify_references
from _contract import ContractError, validate_acceptance_metrics, validate_schema
from _dataset import load_split_records


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _case_records(record: Dict[str, Any], label: str) -> Dict[str, Dict[str, Any]]:
    path, _ = verify_reference(record, label)
    if not path.is_file():
        raise ContractError(f"Store {label} as one evidence file.")
    if record["format"] == "jsonl":
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, list):
            raise ContractError(f"Store {label} JSON evidence as an array.")
    if len(values) != record["cases"]:
        raise ContractError(f"Match {label} case count to its evidence file.")
    if not all(
        isinstance(item, dict)
        and isinstance(item.get("case_id"), str)
        and item["case_id"]
        for item in values
    ):
        raise ContractError(f"Every {label} record requires a case_id.")
    records = {item["case_id"]: item for item in values}
    if len(records) != len(values):
        raise ContractError(f"Assign unique case_id values in {label}.")
    return records


def _sealed_case_ids(path: Path, data_format: str) -> Set[str]:
    records = load_split_records(path, data_format)
    if not all(
        isinstance(item.get("episode_id"), str) and item["episode_id"]
        for item in records
    ):
        raise ContractError("Every sealed evaluation record requires an episode_id.")
    case_ids = {item["episode_id"] for item in records}
    if len(case_ids) != len(records):
        raise ContractError("Assign unique episode_id values in the sealed evaluation split.")
    return case_ids


def _canonical_sha256(value: Dict[str, Any]) -> str:
    content = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _validate_case_coverage(
    transcripts: Dict[str, Dict[str, Any]],
    scores: Dict[str, Dict[str, Any]],
    sealed_ids: Set[str],
) -> None:
    if set(transcripts) != sealed_ids or set(scores) != sealed_ids:
        raise ContractError("Cover every sealed case in transcript and score evidence.")


def _aggregate(values: list[float], method: str) -> float:
    if not values:
        raise ContractError("Aggregate evaluation metrics from at least one case.")
    if method == "mean":
        return sum(values) / len(values)
    if method == "sum":
        return sum(values)
    if method == "min":
        return min(values)
    if method == "max":
        return max(values)
    quantile = 0.50 if method == "p50" else 0.95
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _validate_score_records(
    transcripts: Dict[str, Dict[str, Any]],
    scores: Dict[str, Dict[str, Any]],
    metrics: list[Dict[str, Any]],
    summary: Dict[str, Any],
) -> None:
    if set(transcripts) != set(scores):
        raise ContractError("Match score records to transcript records exactly.")
    metric_names = [item["name"] for item in metrics]
    if len(metric_names) != len(set(metric_names)):
        raise ContractError("Assign unique evaluation metric names.")
    expected_metrics = set(metric_names)
    completed = 0
    errors = 0
    for case_id, score in scores.items():
        if not isinstance(score, dict):
            raise ContractError(f"Score record {case_id} must be an object.")
        status = score.get("status")
        expected_fields = {
            "case_id",
            "status",
            "transcript_sha256",
            "metrics",
            "slices",
        }
        if status == "error":
            expected_fields.add("error")
        if set(score) != expected_fields or status not in {"completed", "error"}:
            raise ContractError(
                f"Score record {case_id} must use the stable score envelope."
            )
        if score["transcript_sha256"] != _canonical_sha256(transcripts[case_id]):
            raise ContractError(f"Score record {case_id} differs from its transcript.")
        case_metrics = score["metrics"]
        if not isinstance(case_metrics, dict) or set(case_metrics) != expected_metrics:
            raise ContractError(f"Score record {case_id} must define every metric once.")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in case_metrics.values()
        ):
            raise ContractError(f"Score record {case_id} metrics must be finite numbers.")
        slices = score["slices"]
        if (
            not isinstance(slices, list)
            or len(slices) != len(set(slices))
            or any(not isinstance(item, str) or not item for item in slices)
        ):
            raise ContractError(f"Score record {case_id} slices must be unique names.")
        if status == "completed":
            completed += 1
        else:
            errors += 1
            if not isinstance(score["error"], str) or not score["error"]:
                raise ContractError(f"Score record {case_id} error requires text.")

    if (
        summary["attempted"] != len(scores)
        or summary["completed"] != completed
        or summary["errors"] != errors
    ):
        raise ContractError("Derive evaluation summary counts from score evidence.")
    for metric in metrics:
        name = metric["name"]
        aggregation = metric["aggregation"]
        observed = _aggregate(
            [float(score["metrics"][name]) for score in scores.values()],
            aggregation,
        )
        if not math.isclose(float(metric["value"]), observed, rel_tol=0.0, abs_tol=1e-12):
            raise ContractError(f"Derive evaluation metric {name} from score evidence.")
        for sliced in metric.get("slices", []):
            selected = [
                float(score["metrics"][name])
                for score in scores.values()
                if sliced["name"] in score["slices"]
            ]
            if sliced["cases"] != len(selected):
                raise ContractError(f"Derive evaluation metric {name} slice counts.")
            observed_slice = _aggregate(selected, aggregation)
            if not math.isclose(
                float(sliced["value"]),
                observed_slice,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ContractError(
                    f"Derive evaluation metric {name} slices from scores."
                )


def verify_evaluation_manifest(
    manifest_path: Path,
    contract: Dict[str, Any],
    dataset: Dict[str, Any],
    expected_split: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_schema(manifest, "evaluation-manifest.schema.json", "Evaluation manifest")
    if manifest["application_id"] != contract["application_id"]:
        raise ContractError("Match the evaluation and application contract identities.")
    evaluation_root = resolve_path(
        contract["artifacts"]["evaluation_results"], "artifacts.evaluation_results"
    )
    evaluation_directory = manifest_path.parent
    if (
        evaluation_directory.parent.resolve() != evaluation_root.resolve()
        or manifest_path.name != "evaluation-manifest.json"
        or manifest["evaluation_id"] != evaluation_directory.name
    ):
        raise ContractError("Use the canonical immutable evaluation path and ID.")

    splits = [item for item in dataset["splits"] if item["purpose"] == "evaluation"]
    if len(splits) != 1:
        raise ContractError("Bind one sealed evaluation split into the dataset.")
    split = expected_split or splits[0]
    dataset_path, _ = verify_reference(manifest["dataset"], "evaluation dataset")
    split_path, _ = verify_reference(split, "sealed evaluation split")
    if (
        dataset_path.resolve() != split_path.resolve()
        or manifest["dataset"]["dataset_id"] != dataset["dataset_id"]
        or manifest["dataset"]["sha256"] != split["sha256"]
        or manifest["dataset"]["cases"] != split["records"]
        or manifest["dataset"]["sealed"] is not True
    ):
        raise ContractError("Bind the exact sealed dataset split into the evaluation.")

    evaluator_records = [
        manifest["evaluator"]["source"],
        manifest["evaluator"]["config"],
        manifest["evaluator"]["launcher_configuration"],
        manifest["reproducibility"]["environment"],
    ]
    verify_references(evaluator_records, "evaluation provenance")
    launcher_path, _ = verify_reference(
        manifest["evaluator"]["launcher_configuration"],
        "evaluation launcher configuration",
    )
    launcher = json.loads(launcher_path.read_text(encoding="utf-8"))
    if (
        not isinstance(launcher, dict)
        or set(launcher) != {"schema_version", "configuration", "argv"}
        or launcher["schema_version"] != "miniagent.effective-invocation.v1"
        or launcher["argv"] != manifest["reproducibility"]["argv"]
        or manifest["reproducibility"]["command"]
        != shlex.join(manifest["reproducibility"]["argv"])
    ):
        raise ContractError("Bind the effective evaluation invocation into the result.")
    if any(
        not _inside(resolve_path(item["path"], "evaluation provenance"), evaluation_directory)
        for item in evaluator_records
    ):
        raise ContractError("Snapshot evaluation provenance inside its result directory.")

    candidate = manifest["candidate"]
    verify_references(candidate["artifacts"], "candidate artifacts")
    candidate_keys = [(item["path"], item["sha256"]) for item in candidate["artifacts"]]
    if len(candidate_keys) != len(set(candidate_keys)):
        raise ContractError("List each candidate artifact once.")

    evidence = manifest["evidence"]
    verify_references(evidence, "evaluation evidence")
    evidence_keys = [
        (item["path"], item["sha256"], item["kind"]) for item in evidence
    ]
    if len(evidence_keys) != len(set(evidence_keys)):
        raise ContractError("List each evaluation evidence artifact once.")
    if any(
        not _inside(resolve_path(item["path"], "evaluation evidence"), evaluation_directory)
        for item in evidence
    ):
        raise ContractError("Write evaluation evidence inside its immutable result directory.")
    by_kind: Dict[str, list[Dict[str, Any]]] = {}
    for item in evidence:
        by_kind.setdefault(item["kind"], []).append(item)
    if len(by_kind.get("transcripts", [])) != 1 or len(by_kind.get("scores", [])) != 1:
        raise ContractError("Record exactly one transcript file and one score file.")
    transcript_records = _case_records(
        by_kind["transcripts"][0], "transcript evidence"
    )
    score_records = _case_records(by_kind["scores"][0], "score evidence")
    sealed_ids = _sealed_case_ids(split_path, split["format"])
    _validate_case_coverage(transcript_records, score_records, sealed_ids)

    summary = manifest["summary"]
    if summary["attempted"] != split["records"]:
        raise ContractError("Attempt every case in the sealed evaluation split.")
    if summary["completed"] + summary["errors"] != summary["attempted"]:
        raise ContractError("Account for every attempted evaluation case.")
    _validate_score_records(
        transcript_records, score_records, manifest["metrics"], summary
    )
    validate_acceptance_metrics(contract, manifest)
    if summary["status"] == "passed" and (
        summary["errors"] != 0 or summary["completed"] != summary["attempted"]
    ):
        raise ContractError("A passing evaluation must complete every sealed case.")

    declared = [*evaluator_records, *evidence]
    declared_paths = [
        resolve_path(item["path"], "evaluation artifact").resolve()
        for item in declared
    ]
    if len(declared_paths) != len(set(declared_paths)):
        raise ContractError("Bind each evaluation artifact path once.")
    if any(
        left in right.parents
        for left in declared_paths
        for right in declared_paths
        if left != right
    ):
        raise ContractError("Keep evaluation artifact records non-overlapping.")
    if manifest_path.resolve() in declared_paths:
        raise ContractError("Keep the evaluation manifest outside its hashed artifacts.")
    if any(path.is_symlink() for path in evaluation_directory.rglob("*")):
        raise ContractError("Build evaluations from regular files and directories.")
    unbound = []
    for path in (item for item in evaluation_directory.rglob("*") if item.is_file()):
        if path.resolve() == manifest_path.resolve():
            continue
        if not any(
            path.resolve() == declared
            or (declared.is_dir() and declared in path.resolve().parents)
            for declared in declared_paths
        ):
            unbound.append(path.relative_to(evaluation_directory).as_posix())
    if unbound:
        raise ContractError(
            "Bind every evaluation file into the manifest: "
            + ", ".join(sorted(unbound))
            + "."
        )
    return manifest
