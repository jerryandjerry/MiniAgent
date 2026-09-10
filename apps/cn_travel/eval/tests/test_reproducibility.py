"""Integrity checks for the sealed evaluator and leaderboard publication."""

import gzip
import hashlib
import json
from pathlib import Path

from project_paths import EVAL_ROOT


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot_sha256(path: Path) -> str:
    return hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest()


def test_rebaseline_contract_sources_are_frozen_by_exact_hash():
    contract_path = EVAL_ROOT / "staging" / "20260901_all_entries" / "contract.json"
    assert _sha256(contract_path) == (
        "1872255341ff20e5298d17247b3f0e37f46c9076d5fdd8efc9e17e3c14c888b0"
    )
    contract = _load(contract_path)
    source_dir = EVAL_ROOT / "reproducibility" / "evaluator_sources" / "20260901_all_entries"

    for logical_name, expected in contract["evaluation"]["evaluator_files"].items():
        stem = Path(logical_name).stem
        assert _snapshot_sha256(source_dir / f"{stem}.{expected}.py.gz") == expected

    channel_sources = {
        "codex_proxy": contract["channels"]["openai_cli"]["proxy_sha256"],
        "claude_proxy": contract["channels"]["anthropic_cli"]["proxy_sha256"],
        "claude_shim": contract["channels"]["openai_cli"]["shim_sha256"],
        "claude_cli": contract["channels"]["anthropic_cli"]["transport_sha256"],
    }
    assert (
        contract["channels"]["anthropic_cli"]["shim_sha256"]
        == channel_sources["claude_shim"]
    )
    for stem, expected in channel_sources.items():
        assert _snapshot_sha256(source_dir / f"{stem}.{expected}.py.gz") == expected


def test_rebaseline_publication_amendment_matches_current_names_and_hashes():
    amendment = _load(
        EVAL_ROOT
        / "staging"
        / "20260901_all_entries"
        / "publication_amendments"
        / "20260903_ids_16_17_display_name_only.json"
    )
    contract_path = EVAL_ROOT / "staging" / "20260901_all_entries" / "contract.json"
    assert amendment["parent_contract"]["sha256"] == _sha256(contract_path)
    publication = _load(contract_path)["publication"]
    assert amendment["source_publication"] == {
        "leaderboard_sha256": publication["leaderboard_sha256"],
        "leaderboard_html_sha256": publication["leaderboard_html_sha256"],
    }

    leaderboard = _load(EVAL_ROOT / "leaderboard.json")
    rows = {row["id"]: row for row in leaderboard}
    for change in amendment["changes"]:
        assert rows[change["id"]][change["field"]] == change["after"]
        summary_path = EVAL_ROOT / rows[change["id"]]["run_dir"] / "summary.json"
        summary = _load(summary_path)
        assert summary["system"] == change["after"]
        assert _sha256(summary_path) == change["canonical_summary_after_sha256"]

    assert amendment["resulting_publication"] == {
        "leaderboard_sha256": _sha256(EVAL_ROOT / "leaderboard.json"),
        "leaderboard_html_sha256": _sha256(EVAL_ROOT / "leaderboard.html"),
    }
