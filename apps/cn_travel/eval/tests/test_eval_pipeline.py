#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline tests for frozen-world fidelity and deterministic evaluation."""
import json
import sys
from pathlib import Path

import pytest

from project_paths import DATA_ROOT, EVAL_ROOT

sys.path.insert(0, str(EVAL_ROOT))
import score as ev_score  # noqa: E402
import world as ev_world  # noqa: E402

EVAL_SET = DATA_ROOT / "6_multiturn" / "leaderboard_eval.json"


@pytest.fixture(scope="module")
def eval_rows():
    return json.loads(EVAL_SET.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def frozen_world():
    return ev_world.FrozenWorld()


@pytest.fixture(scope="module")
def golden():
    return ev_score.load_eval_set()


def _reference_run_dir(tmp_path: Path, eval_rows) -> Path:
    run = tmp_path / "202601010000_reference"
    (run / "transcripts").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({
        "system": "reference", "model": "sealed-teacher",
        "price_in_per_mtok": 0.0, "price_out_per_mtok": 0.0,
        "cases": 101, "max_rounds": 6,
        "finished_at": "2026-01-01T00:00:00-06:00"}), encoding="utf-8")
    for row in eval_rows:
        meta = row["metadata"]
        segs = []
        for s in ev_score.golden_segments(row["conversation"]):
            calls = [{"round": r + 1, "id": f"ref_{r}_{i}", "tool": c["tool"],
                      "arguments_raw": json.dumps(c["args"], ensure_ascii=False),
                      "arguments": c["args"]}
                     for r, rnd in enumerate(s["rounds"]) for i, c in enumerate(rnd)]
            segs.append({"user": s["user"], "calls": calls, "final_assistant": "ok",
                         "rounds": len(s["rounds"]) + 1, "cap_hit": False})
        t = {"idx": meta["idx"], "segments": segs, "latency_ms": 0.0,
             "tokens_in": 0, "tokens_out": 0, "requests": 0,
             "world_fallbacks": [], "error": None}
        (run / "transcripts" / f"{meta['idx']:04d}.json").write_text(
            json.dumps(t, ensure_ascii=False), encoding="utf-8")
    return run


def _doctor(run: Path, idx: int, fn) -> None:
    f = run / "transcripts" / f"{idx:04d}.json"
    t = json.loads(f.read_text(encoding="utf-8"))
    fn(t)
    f.write_text(json.dumps(t, ensure_ascii=False), encoding="utf-8")


def _task(run: Path, idx: int) -> dict:
    return json.loads((run / "tasks" / f"{idx:04d}.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------ world -------
def test_world_replays_every_sealed_call_identically(eval_rows, frozen_world):
    n_calls = 0
    for row in eval_rows:
        conv = row["conversation"]
        sealed = {m["tool_call_id"]: json.loads(m["content"])
                  for m in conv if m.get("role") == "tool"}
        for m in conv:
            for tc in (m.get("tool_calls") or []) if m.get("role") == "assistant" else []:
                args = json.loads(tc["function"]["arguments"])
                got = frozen_world.run(tc["function"]["name"], args)
                assert got == sealed[tc["id"]], \
                    f"idx {row['metadata']['idx']} {tc['function']['name']}({args}) 与封存结果不同"
                n_calls += 1
    assert n_calls > 0
    print(f"\n  replayed {n_calls} sealed calls identically")


def test_world_uses_shared_canonical_key_for_hotel_requirements(eval_rows, frozen_world):
    row = next(row for row in eval_rows if row["metadata"]["idx"] == 836)
    call = next(tc for msg in row["conversation"] if msg.get("role") == "assistant"
                for tc in (msg.get("tool_calls") or [])
                if tc["function"]["name"] == "recommend_hotels")
    sealed = next(json.loads(msg["content"]) for msg in row["conversation"]
                  if msg.get("role") == "tool" and msg["tool_call_id"] == call["id"])
    args = json.loads(call["function"]["arguments"])
    assert args == {"location": "湘乡", "requirements": "市中心附近的"}

    before = len(frozen_world.fallbacks)
    got = frozen_world.run("recommend_hotels",
                           {"location": "湘乡", "requirements": "市中心附近的酒店"})
    assert got == sealed
    assert len(frozen_world.fallbacks) == before


def test_world_uses_same_registry_city_alias_as_scorer(eval_rows, frozen_world):
    target = None
    for row in eval_rows:
        for msg in row["conversation"]:
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls") or []:
                args = json.loads(tc["function"]["arguments"])
                loc = args.get("location")
                if (isinstance(loc, str) and not loc.endswith("市")
                        and loc in ev_world._REG_KEYS):
                    target = (row, tc, args, f"{loc}市")
                    break
            if target:
                break
        if target:
            break
    assert target is not None
    row, call, args, location_alias = target
    sealed = next(json.loads(msg["content"]) for msg in row["conversation"]
                  if msg.get("role") == "tool" and msg["tool_call_id"] == call["id"])
    before = len(frozen_world.fallbacks)
    assert frozen_world.run(call["function"]["name"],
                            {**args, "location": location_alias}) == sealed
    assert len(frozen_world.fallbacks) == before


def test_world_is_deterministic_and_never_crashes_on_junk(frozen_world):
    junk = [("get_weather_info", {}), ("get_weather_info", {"location": "天门", "start_date": "2026-09-03", "num_days": "x"}),
            ("query_route", {"start_location": "a"}), ("no_such_tool", {"x": 1}),
            ("recommend_hotels", {"location": "不存在的城市啊"}),
            ("get_hotel_reviews", {"hotel_name": "不存在的酒店啊"})]
    for tool, args in junk:
        r1, r2 = frozen_world.run(tool, dict(args)), frozen_world.run(tool, dict(args))
        assert r1 == r2, f"{tool}({args}) 两次结果不同"
        assert r1.get("status") in {"error", "empty", "ok"}


def test_novel_review_call_gets_empty_shape_not_cached_teacher_evidence(eval_rows, frozen_world):
    gc = next(c for row in eval_rows for m in row["conversation"] if m.get("role") == "assistant"
              for c in (m.get("tool_calls") or []) if c["function"]["name"] == "get_hotel_reviews")
    args = json.loads(gc["function"]["arguments"])
    novel = {**args, "location": "评测探针城市"}       # Same hotel, different args: not exact
    got = frozen_world.run("get_hotel_reviews", novel)
    assert got["status"] == "empty" and got["reviews"] == [] \
        and got["summary"] == "未检索到该酒店的公开评价"


def test_every_non_sealed_answer_is_logged_as_fallback(frozen_world):
    probes = [("search_travel_guide", {"location": "嘉兴"}, "guide_generator"),
              ("get_weather_info", {"location": "嘉兴", "start_date": "2031-01-01", "num_days": 2}, "weather_generator"),
              ("query_route", {"start_location": "嘉兴", "end_location": "上不了榜市", "city": "嘉兴"}, "route_generator")]
    for tool, args, kind in probes:
        n0 = len(frozen_world.fallbacks)
        frozen_world.run(tool, args)
        new = frozen_world.fallbacks[n0:]
        assert new and new[-1]["fallback"] in {kind, "exec_cache"}, \
            f"{tool} 新参数应答未留痕（got {new}）"


# ------------------------------------------------------------ scorer ------
def test_reference_submission_scores_perfect(tmp_path, eval_rows):
    run = _reference_run_dir(tmp_path, eval_rows)
    summary = ev_score.score_run(run)
    o = summary["overall"]
    assert summary["cases"] == 101
    assert o["trace_match"] == 1.0 and o["toolcall_match"] == 1.0
    assert o["calls_extra"] == 0 and o["errors"] == 0
    assert all(g["trace_match"] == 1.0 for g in summary["by_route"].values())
    assert all("p50_latency_ms" in g and "p95_latency_ms" in g
               for g in summary["by_workflow"].values()), "分组切片必须带延迟分位"
    assert (run / "summary.json").exists() and len(list((run / "tasks").glob("*.json"))) == 101


def test_json_bool_is_not_a_number():
    cmp = ev_score._compare
    assert cmp({"num_days": 1}, {"num_days": True}) == "wrong_args:num_days"
    assert cmp({"num_days": True}, {"num_days": 1}) == "wrong_args:num_days"
    assert cmp({"num_days": 1}, {"num_days": 1.0}) is None
    assert cmp({"a": {"b": [1, True]}}, {"a": {"b": [1, 1]}}) == "wrong_args:a"


def test_declared_protocol_without_transcripts_is_refused(tmp_path):
    run = tmp_path / "202601010001_empty"
    (run / "transcripts").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({
        "system": "empty", "model": "none", "cases": 101, "max_rounds": 6,
        "finished_at": "2026-01-01T00:00:00-06:00"}), encoding="utf-8")
    summary = ev_score.score_run(run)
    o = summary["overall"]
    assert o["transcripts"] == 0 and o["latencies_recorded"] == 0
    assert o["p50_latency_ms"] is None and o["p95_latency_ms"] is None
    with pytest.raises(ev_score.ProtocolMismatch):
        ev_score.append_leaderboard(summary, tmp_path / "leaderboard.json")
    assert not (tmp_path / "leaderboard.json").exists()


def test_structurally_empty_transcripts_are_refused_even_when_all_files_exist(tmp_path, eval_rows):
    run = tmp_path / "202601010002_hollow"
    (run / "transcripts").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({
        "system": "hollow", "model": "none", "cases": 101, "max_rounds": 6,
        "finished_at": "2026-01-01T00:00:00-06:00"}), encoding="utf-8")
    for row in eval_rows:
        idx = row["metadata"]["idx"]
        (run / "transcripts" / f"{idx:04d}.json").write_text(json.dumps(
            {"idx": idx, "segments": [], "latency_ms": 1.0,
             "tokens_in": 0, "tokens_out": 0, "error": None}), encoding="utf-8")
    summary = ev_score.score_run(run)
    o = summary["overall"]
    assert o["transcripts"] == 101 and o["latencies_recorded"] == 101
    assert o["rollouts_complete"] == 0
    with pytest.raises(ev_score.ProtocolMismatch):
        ev_score.append_leaderboard(summary, tmp_path / "leaderboard.json")


def test_cap_hit_rollout_stays_board_eligible_but_errored_run_is_refused(tmp_path, eval_rows, golden):
    board = tmp_path / "leaderboard.json"
    idx = next(i for i in sorted(golden) if golden[i]["segments"])
    run = _reference_run_dir(tmp_path, eval_rows)

    def cap(t):
        s = t["segments"][0]                 # Structurally valid cap: six all-call rounds
        s["final_assistant"] = None
        s["cap_hit"] = True
        s["rounds"] = 6
        s["calls"] = [{"round": r, "id": f"c{r}", "tool": "search_travel_guide",
                       "arguments_raw": "{\"location\": \"北京\"}",
                       "arguments": {"location": "北京"}} for r in range(1, 7)]
    _doctor(run, idx, cap)
    summary = ev_score.score_run(run)
    assert summary["overall"]["rollouts_complete"] == 101
    rec = _task(run, idx)
    assert rec["rollout_complete"] and not rec["complete"] and not rec["trace_match"]
    assert ev_score.append_leaderboard(summary, board)[0]["name"] == "reference"

    _doctor(run, idx, lambda t: t.__setitem__("error", "APIError: probe"))
    summary2 = ev_score.score_run(run)
    assert summary2["overall"]["rollouts_complete"] == 100
    with pytest.raises(ev_score.ProtocolMismatch):
        ev_score.append_leaderboard(summary2, board)


def test_unreachable_endpoint_claims_are_not_protocol_complete(tmp_path, eval_rows, golden):
    # Complete file set with structurally unreachable segment endpoints.
    run = tmp_path / "202601010003_forged"
    (run / "transcripts").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({
        "system": "forged", "model": "none", "cases": 101, "max_rounds": 6,
        "finished_at": "2026-01-01T00:00:00-06:00"}), encoding="utf-8")
    g = ev_score.load_eval_set()
    for row in eval_rows:
        idx = row["metadata"]["idx"]
        segs = [{"user": s["user"], "calls": [], "final_assistant": None,
                 "cap_hit": True, "rounds": 0} for s in g[idx]["segments"]]
        (run / "transcripts" / f"{idx:04d}.json").write_text(json.dumps(
            {"idx": idx, "segments": segs, "latency_ms": 1.0,
             "tokens_in": 0, "tokens_out": 0, "error": None}), encoding="utf-8")
    summary = ev_score.score_run(run)
    o = summary["overall"]
    assert o["transcripts"] == 101 and o["latencies_recorded"] == 101
    assert o["rollouts_complete"] == 0
    with pytest.raises(ev_score.ProtocolMismatch):
        ev_score.append_leaderboard(summary, tmp_path / "leaderboard.json")

    # Each unreachable endpoint form is rejected independently.
    zero = next(i for i in sorted(golden)
                if sum(len(s["calls"]) for s in golden[i]["segments"]) == 0)
    ref = _reference_run_dir(tmp_path, eval_rows)
    for forge in (lambda s: s.update(rounds=0),                            # A reply at round zero is unreachable
                  lambda s: s.update(rounds=7),                            # Out of bounds
                  lambda s: s.update(cap_hit=True)):                       # Final reply and cap conflict
        _doctor(ref, zero, lambda t: forge(t["segments"][0]))
        ev_score.score_run(ref)
        rec = _task(ref, zero)
        assert not rec["rollout_complete"] and not rec["complete"] and not rec["trace_match"], forge
        _doctor(ref, zero, lambda t: t["segments"][0].update(              # Restore.
            rounds=1, cap_hit=False, final_assistant="ok"))


def test_empty_segments_transcript_does_not_pass_zero_call_task(tmp_path, eval_rows, golden):
    zero = next(i for i in sorted(golden)
                if sum(len(s["calls"]) for s in golden[i]["segments"]) == 0)
    run = _reference_run_dir(tmp_path, eval_rows)
    _doctor(run, zero, lambda t: t.__setitem__("segments", []))
    ev_score.score_run(run)
    rec = _task(run, zero)
    assert not rec["complete"] and not rec["trace_match"]


def test_summary_slices_carry_task_locators(tmp_path, eval_rows):
    run = _reference_run_dir(tmp_path, eval_rows)
    summary = ev_score.score_run(run)
    assert len(summary["overall"]["tasks"]) == 101
    for grp in ("by_workflow", "by_route", "by_expect"):
        for g in summary[grp].values():
            assert len(g["tasks"]) == g["n"] and all((run / p).exists() for p in g["tasks"])
    assert summary["by_workflow"]["W4"]["n"] == 10


def test_argument_equality_is_strict_beyond_registry_shi_strip():
    cmp = ev_score._compare
    assert cmp({"location": "天门"}, {"location": "天门", "多余": 1}) == "wrong_args:多余"
    assert cmp({"location": "天门", "start_date": "2026-09-03", "num_days": 2},
               {"location": "天门", "start_date": "2026-09-03", "num_days": "2"}) == "wrong_args:num_days"
    assert cmp({"location": "如皋"}, {"location": "如皋市"}) is None      # Registry-backed suffix stripping
    assert cmp({"hotel_name": "假想市"}, {"hotel_name": "假想"}) == "wrong_args:hotel_name"  # Do not strip other fields
    assert cmp({"location": "天门"}, "not a dict") == "unparseable_args"


@pytest.mark.parametrize(("gold", "variant"), [
    ("市中心附近的", "市中心附近"),
    ("市中心附近的", "市中心附近的酒店"),
    ("预算250元左右", "预算250元左右一晚"),
    ("预算400元左右", "预算大约400元左右一晚"),
    ("预算450元左右", "预算大约450元左右一晚就行"),
    ("经济实惠一点", "经济实惠一点的"),
    ("评分高一些的", "一点高评分"),
    ("评分高一些的", "一点高评分的酒店"),
])
def test_recommend_hotel_requirement_equivalents_match(gold, variant):
    cmp = ev_score._compare
    assert cmp({"location": "湘乡", "requirements": gold},
               {"location": "湘乡", "requirements": variant},
               "recommend_hotels") is None


def test_requirement_normalization_is_closed_and_preserves_strict_json():
    cmp = ev_score._compare
    tool = "recommend_hotels"
    assert cmp({"requirements": "预算400元左右"},
               {"requirements": "预算450元左右"}, tool) == "wrong_args:requirements"
    assert cmp({"requirements": "市中心附近的"},
               {"requirements": "郊区附近的"}, tool) == "wrong_args:requirements"
    assert cmp({"requirements": "经济实惠一点"},
               {"requirements": "高档一点的"}, tool) == "wrong_args:requirements"
    assert cmp({"requirements": ""}, {}, tool) == "wrong_args:requirements"
    assert cmp({"requirements": ""}, {"requirements": 0}, tool) == "wrong_args:requirements"
    assert cmp({"requirements": "市中心附近的"},
               {"requirements": "市中心附近的", "extra": True}, tool) == "wrong_args:extra"
    assert cmp({"requirements": "市中心附近的"},
               {"requirements": "市中心附近的旅馆"}, tool) == "wrong_args:requirements"
    # Requirement aliases are tool-specific, not a global relaxation of string equality.
    assert cmp({"requirements": "市中心附近的"},
               {"requirements": "市中心附近"}, "search_travel_guide") == "wrong_args:requirements"


def test_equivalent_requirement_keeps_end_to_end_reference_score_perfect(tmp_path, eval_rows):
    run = _reference_run_dir(tmp_path, eval_rows)
    _doctor(run, 836, lambda t: t["segments"][0]["calls"][0]["arguments"].update(
        requirements="市中心附近的酒店"))
    summary = ev_score.score_run(run)
    rec = _task(run, 836)
    assert rec["trace_match"] and rec["calls_correct"] == rec["calls_required"]
    assert summary["overall"]["trace_match"] == 1.0
    assert summary["overall"]["toolcall_match"] == 1.0


def test_scorer_flags_wrong_args_missing_and_extra(tmp_path, eval_rows, golden):
    run = _reference_run_dir(tmp_path, eval_rows)
    idxs = [i for i in sorted(golden) if golden[i]["segments"]
            and len(golden[i]["segments"][0]["calls"]) >= 1][:3]
    a, b, c = idxs
    _doctor(run, a, lambda t: t["segments"][0]["calls"][0]["arguments"].update(
        {next(iter(t["segments"][0]["calls"][0]["arguments"])): "错误值"}))
    _doctor(run, b, lambda t: t["segments"][0]["calls"].pop(0))
    _doctor(run, c, lambda t: t["segments"][0]["calls"].append(
        {"round": 9, "id": "x", "tool": "search_travel_guide",
         "arguments_raw": "{\"location\": \"多余\"}", "arguments": {"location": "多余"}}))
    ev_score.score_run(run)

    ta = _task(run, a)
    assert not ta["trace_match"] and any(
        str(r.get("reason", "")).startswith(("wrong_args:", "unparseable")) for r in ta["call_detail"])
    tb = _task(run, b)
    assert not tb["trace_match"] and any(r.get("reason") == "missing" for r in tb["call_detail"])
    assert tb["calls_correct"] == tb["calls_required"] - 1
    tc_ = _task(run, c)
    assert not tc_["trace_match"] and tc_["calls_extra"] == 1
    assert tc_["calls_correct"] == tc_["calls_required"], "多余调用不应影响正确计数"


def test_reordered_golden_sequence_fails_trace(tmp_path, eval_rows, golden):
    idx = next(i for i in sorted(golden)
               if golden[i]["segments"] and len(golden[i]["segments"][0]["rounds"]) == 2)
    run = _reference_run_dir(tmp_path, eval_rows)
    _doctor(run, idx, lambda t: t["segments"][0]["calls"].reverse())
    ev_score.score_run(run)
    rec = _task(run, idx)
    assert not rec["trace_match"]
    assert rec["calls_correct"] == rec["calls_required"] - 1
    assert rec["calls_extra"] == 0
    assert any(r.get("reason") == "out_of_order" for r in rec["call_detail"])


def test_task_error_keeps_partial_credit_and_latency(tmp_path, eval_rows, golden):
    idx = next(i for i in sorted(golden) if golden[i]["segments"]
               and len(golden[i]["segments"][0]["calls"]) >= 1)
    run = _reference_run_dir(tmp_path, eval_rows)

    def wreck(t):
        t["error"] = "APIError: probe"
        t["latency_ms"] = 123.4
    _doctor(run, idx, wreck)
    ev_score.score_run(run)
    rec = _task(run, idx)
    assert rec["error"] == "APIError: probe"
    assert not rec["trace_match"]
    assert rec["calls_correct"] == rec["calls_required"], "出错前的正确调用必须保留"
    assert rec["latency_ms"] == 123.4, "出错任务的延迟必须进分位样本"


def test_zero_call_task_fails_on_any_call(tmp_path, eval_rows, golden):
    zero = next(i for i in sorted(golden)
                if sum(len(s["calls"]) for s in golden[i]["segments"]) == 0)
    run = _reference_run_dir(tmp_path, eval_rows)
    _doctor(run, zero, lambda t: t["segments"][0]["calls"].append(
        {"round": 1, "id": "x", "tool": "search_travel_guide",
         "arguments_raw": "{\"location\": \"北京\"}", "arguments": {"location": "北京"}}))
    ev_score.score_run(run)
    rec = _task(run, zero)
    assert rec["calls_required"] == 0 and rec["calls_extra"] == 1 and not rec["trace_match"]


def test_scoring_is_deterministic(tmp_path, eval_rows):
    run = _reference_run_dir(tmp_path, eval_rows)
    s1 = ev_score.score_run(run)
    first = {p.name: p.read_bytes() for p in (run / "tasks").glob("*.json")}
    s2 = ev_score.score_run(run)
    assert s1 == s2
    assert all((run / "tasks" / n).read_bytes() == b for n, b in first.items())


# ------------------------------------------------------- leaderboard ------
def test_evaluate_runs_submission_to_leaderboard_entry(tmp_path, eval_rows, capsys):
    import shutil
    import evaluate as ev_evaluate
    shutil.copyfile(EVAL_ROOT / "leaderboard.html", tmp_path / "leaderboard.html")
    run = _reference_run_dir(tmp_path, eval_rows)
    rc = ev_evaluate.main(["--submission", str(run), "--leaderboard", str(tmp_path / "leaderboard.json")])
    assert rc == 0
    board = json.loads((tmp_path / "leaderboard.json").read_text(encoding="utf-8"))
    assert len(board) == 1 and board[0]["name"] == "reference" and board[0]["trace_match"] == 1.0
    assert board[0]["run_dir"], "榜单行必须带 run 定位符"
    html = (tmp_path / "leaderboard.html").read_text(encoding="utf-8")
    assert '"trace_match": 1.0' in html and '"run_dir"' in html
    assert "trace_match 1.0000" in capsys.readouterr().out
    assert ev_evaluate.main(["--submission", str(tmp_path / "nowhere")]) == 2
    assert ev_evaluate.main(["--leaderboard", "none"]) == 2      # No submission or endpoint arguments


def test_one_run_one_leaderboard_row(tmp_path, eval_rows):
    board = tmp_path / "leaderboard.json"
    run = _reference_run_dir(tmp_path, eval_rows)
    summary = ev_score.score_run(run)
    ev_score.append_leaderboard(summary, board)
    entries = ev_score.append_leaderboard(ev_score.score_run(run), board)
    assert len(entries) == 1 and entries[0]["id"] == 1


def test_non_sealed_protocol_run_is_refused_from_leaderboard(tmp_path, eval_rows):
    board = tmp_path / "leaderboard.json"
    run = _reference_run_dir(tmp_path, eval_rows)
    summary = ev_score.score_run(run)
    for proto in ({"cases_run": 5, "max_rounds": 6}, {"cases_run": 101, "max_rounds": 3}):
        bad = {**summary, "protocol": proto}
        with pytest.raises(ev_score.ProtocolMismatch):
            ev_score.append_leaderboard(bad, board)
    assert not board.exists(), "被拒的 run 不应留下榜单文件"


def test_leaderboard_append_and_html_sync(tmp_path, eval_rows):
    import shutil
    board = tmp_path / "leaderboard.json"
    shutil.copyfile(EVAL_ROOT / "leaderboard.html", tmp_path / "leaderboard.html")
    run = _reference_run_dir(tmp_path, eval_rows)
    summary = ev_score.score_run(run)
    entries = ev_score.append_leaderboard(summary, board)
    assert len(entries) == 1 and entries[0]["id"] == 1 and entries[0]["trace_match"] == 1.0
    assert all(entries[0][f"trace_w{i}"] == 1.0 for i in range(1, 6)), "榜单行必须带每工作流列"
    html = (tmp_path / "leaderboard.html").read_text(encoding="utf-8")
    assert '"name": "reference"' in html and '"trace_match": 1.0' in html
    bad = dict(summary)
    bad["dataset_sha256"] = "0" * 64
    with pytest.raises(ev_score.DatasetMismatch):
        ev_score.append_leaderboard(bad, board)
