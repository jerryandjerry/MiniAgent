"""Focused offline tests for the §5.4 complete-episode world policy."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from cn_travel.business_logic.contracts import validate
from project_paths import DATA_ROOT
from world_policy import (
    Ask,
    AssistantActionInterpreter,
    EpisodeRepository,
    EpisodeRuntime,
    EpisodeSpec,
    EpisodeSpecError,
    Final,
    Invalid,
    Refusal,
    SyntheticChinaWorld,
    ToolCall,
    ToolCalls,
    WorldSnapshotError,
)


GUIDE_ARGS = {"location": "信阳", "search_mode": "hybrid"}
WEATHER_ARGS = {"location": "信阳", "start_date": "2026-08-25", "num_days": 1}


def _guide(status: str = "ok") -> dict:
    return {
        "status": status,
        "source": f"synthetic:test:{status}",
        "location": "信阳",
        "guides": ([{"city": "信阳", "province": "河南", "content": "合成攻略",
                     "score": 1.0}] if status == "ok" else []),
    }


def _weather() -> dict:
    return {
        "status": "ok",
        "source": "synthetic:test:ok",
        "location": "信阳",
        "resolved": {"name": "信阳市", "adcode": "411500", "lat": 32.1, "lng": 114.0},
        "days": [{"date": "2026-08-25", "day_weather": "多云",
                  "night_weather": "晴", "temp_min_c": 21, "temp_max_c": 28}],
    }


def _world(guide_status: str = "ok") -> SyntheticChinaWorld:
    entries = [{"tool": "search_travel_guide", "arguments": GUIDE_ARGS,
                "result": _guide(guide_status)}]
    if guide_status == "ok":
        entries.append({"tool": "get_weather_info", "arguments": WEATHER_ARGS,
                        "result": _weather()})
    return SyntheticChinaWorld({
        "schema_version": "cn_travel.synthetic_china.v1",
        "world_id": f"world-{guide_status}",
        "entries": entries,
    })


def _travel_spec(*, guide_status: str = "ok", world_id: str | None = None) -> EpisodeSpec:
    states = [
        {
            "state_id": "ask_destination",
            "expected_action": {"type": "Ask", "slot": "destination"},
            "user_response": "目的地是信阳",
            "revealed_slots": ["destination"],
            "next_state": "ask_travel_date",
            "milestone_ids": {"ask": "ask_destination"},
        },
        {
            "state_id": "ask_travel_date",
            "expected_action": {"type": "Ask", "slot": "travel_date"},
            "user_response": "就定8月25号那天，一天就行",
            "revealed_slots": ["travel_date"],
            "next_state": "guide",
            "milestone_ids": {"ask": "ask_travel_date"},
        },
        {
            "state_id": "guide",
            "expected_action": {"type": "ToolCalls", "calls": [
                {"name": "search_travel_guide", "arguments": GUIDE_ARGS},
            ]},
            "expected_result_status": guide_status,
            "result_branches": {
                "ok": "weather",
                "empty": "final",
                "error": "final",
            },
            "milestone_ids": {
                "calls": ["call_guide"], "round": "round_guide",
                "branches": {"ok": "branch_guide_ok", "empty": "branch_guide_empty",
                             "error": "branch_guide_error"},
            },
        },
    ]
    if guide_status == "ok":
        states.append({
            "state_id": "weather",
            "expected_action": {"type": "ToolCalls", "calls": [
                {"name": "get_weather_info", "arguments": WEATHER_ARGS},
            ]},
            "expected_result_status": "ok",
            "result_branches": {"ok": "final", "empty": "final", "error": "final"},
            "milestone_ids": {
                "calls": ["call_weather"], "round": "round_weather",
                "branches": {"ok": "branch_weather_ok", "empty": "branch_weather_empty",
                             "error": "branch_weather_error"},
            },
        })
    states.append({
        "state_id": "final",
        "expected_action": {"type": "Final"},
        "next_state": "done",
        "milestone_ids": {"completion": "valid_final"},
    })
    return EpisodeSpec.from_dict({
        "schema_version": "cn_travel.world_policy.v1",
        "episode_id": "cn-travel-0053",
        "source_idx": 53,
        "metadata": {"idx": 53, "workflow": 1, "route": "W1-G"},
        "world_id": world_id or f"world-{guide_status}",
        "initial_messages": [
            {"role": "system", "content": "系统"},
            {"role": "user", "content": "我想安排个出行"},
        ],
        "person_state": {
            "public_context": {"current_city": "高密"},
            "private_slots": {"destination": "信阳", "travel_date": "2026-08-25"},
            "revealed_slots": [],
            "initially_hidden": ["destination", "travel_date"],
            "goal": {"workflow": "travel_plan", "route": "W1-G"},
        },
        "response_policy": {"canonical_refusal": None},
        "initial_state_id": "ask_destination",
        "state_graph": states,
        "reward_milestones": [
            "ask_destination", "ask_travel_date", "call_guide", "round_guide",
            f"branch_guide_{guide_status}",
            *(["call_weather", "round_weather", "branch_weather_ok"]
              if guide_status == "ok" else []),
            "valid_final",
        ],
    })


def _correct_questions(runtime: EpisodeRuntime) -> None:
    first = runtime.interpret_and_step("请告诉我您想去哪个城市旅行？")
    assert first.messages == ({"role": "user", "content": "目的地是信阳"},)
    runtime.interpret_and_step("请问您计划什么时候出行？")


def _correct_tools(runtime: EpisodeRuntime, *, with_weather: bool = True) -> None:
    transition = runtime.step(ToolCalls((ToolCall("search_travel_guide", GUIDE_ARGS, "g"),)))
    assert transition.messages[0]["role"] == "tool"
    assert transition.messages[0]["tool_call_id"] == "g"
    assert transition.environment_loss_mask == 0
    if with_weather:
        runtime.step(ToolCalls((ToolCall("get_weather_info", WEATHER_ARGS, "w"),)))


def test_interpreter_parses_lfm_calls_questions_and_multiple_questions():
    interpreter = AssistantActionInterpreter()
    action = interpreter.parse(
        "<think>ok</think><|tool_call_start|>"
        "[get_weather_info(location='信阳', start_date='2026-08-25', num_days=1)]"
        "<|tool_call_end|>"
    )
    assert isinstance(action, ToolCalls)
    assert action.calls[0].arguments["num_days"] == 1
    assert interpreter.parse("请问您想去哪里？") == Ask(
        slot="destination", text="请问您想去哪里？", matched_rule="ask.destination.v1"
    )
    multi = interpreter.parse("请问想去哪里，哪天出发？")
    assert isinstance(multi, Invalid) and multi.reason == "multiple_questions"
    malformed = interpreter.parse("<|tool_call_start|>[x(a=)]<|tool_call_end|>")
    assert isinstance(malformed, Invalid)

    structured = [{
        "id": "call-1",
        "type": "function",
        "function": {
            "name": "get_weather_info",
            "arguments": json.dumps(WEATHER_ARGS, ensure_ascii=False),
        },
    }]
    assert isinstance(interpreter.parse("<think>ok</think>", structured), ToolCalls)
    mixed = interpreter.parse("先查一下。", structured)
    assert isinstance(mixed, Invalid)
    assert mixed.reason == "text mixed with structured tool calls"


def test_clean_complete_episode_gets_maximum_reward():
    runtime = EpisodeRuntime(_travel_spec(), _world())
    _correct_questions(runtime)
    _correct_tools(runtime)
    done = runtime.step(Final("这是根据工具结果生成的信阳旅行计划。"))
    score = runtime.reward()
    assert done.terminal and done.success and done.clean_trace
    assert score.ask_score == score.call_score == score.structure_score == 1.0
    assert score.goal_reached == score.clean_trace == 1
    assert score.violation_count == 0
    assert score.total == runtime.config.reward.maximum == 1.5
    assert len(score.credited_milestones) == len(score.required_milestones)


def test_short_or_ungrounded_final_cannot_collect_completion_reward():
    for text in ("x", "这是根据工具结果生成的完整旅行计划。"):
        runtime = EpisodeRuntime(_travel_spec(), _world())
        _correct_questions(runtime)
        _correct_tools(runtime)
        stopped = runtime.step(Final(text))
        score = runtime.reward()
        assert stopped.terminal and not stopped.success
        assert stopped.new_violations == ("invalid_final",)
        assert score.goal_reached == score.clean_trace == 0
        assert score.violation_count == 0
        assert score.total == 0.0
        assert "valid_final" not in score.credited_milestones


def test_wrong_question_can_recover_but_clean_trace_stays_false():
    runtime = EpisodeRuntime(_travel_spec(), _world())
    wrong = runtime.interpret_and_step("请问您计划什么时候出行？")
    assert wrong.state_id == "ask_destination"
    assert wrong.messages[0]["content"] == "先告诉我您想去的目的地吧。"
    _correct_questions(runtime)
    _correct_tools(runtime)
    runtime.step(Final("已根据查询结果完成信阳旅行计划，请查收。"))
    score = runtime.reward()
    assert score.goal_reached == 1 and score.clean_trace == 0
    assert score.ask_score == score.call_score == score.structure_score == 1.0
    assert score.violation_count == 1
    assert score.penalties["wrong_question"] == 0.2
    assert score.total == 1.3


def test_repeated_question_never_farms_a_second_milestone():
    runtime = EpisodeRuntime(_travel_spec(), _world())
    runtime.interpret_and_step("请告诉我您想去哪个城市旅行？")
    repeated = runtime.interpret_and_step("请告诉我您想去哪个城市旅行？")
    assert repeated.state_id == "ask_travel_date"
    assert repeated.new_violations == ("repeated_question",)
    runtime.interpret_and_step("请问您计划什么时候出行？")
    _correct_tools(runtime)
    runtime.step(Final("已根据查询结果完成信阳旅行计划，请查收。"))
    score = runtime.reward()
    assert score.ask_score == 1.0
    assert sum(name.startswith("ask_") for name in score.credited_milestones) == 2
    assert score.clean_trace == 0


def test_extra_premature_tool_returns_result_then_allows_recovery():
    runtime = EpisodeRuntime(_travel_spec(), _world())
    _correct_questions(runtime)
    early = runtime.step(ToolCalls((ToolCall("get_weather_info", WEATHER_ARGS),)))
    assert early.state_id == "guide" and early.messages[0]["role"] == "tool"
    assert early.new_violations == ("extra_or_wrong_tool",)
    _correct_tools(runtime)
    runtime.step(Final("已根据查询结果完成信阳旅行计划，请查收。"))
    score = runtime.reward()
    assert score.goal_reached == 1 and score.clean_trace == 0
    assert score.call_score == 1.0  # ordered required calls still completed once
    assert score.violation_count == 1
    assert score.penalties["extra_or_wrong_tool"] == 0.2
    assert score.total == 1.3


def test_two_recoverable_emissions_score_1_1_only_after_completion():
    runtime = EpisodeRuntime(_travel_spec(), _world())
    runtime.interpret_and_step("请问您计划什么时候出行？")
    runtime.step(Invalid("bad"))
    _correct_questions(runtime)
    _correct_tools(runtime)
    runtime.step(Final("已根据查询结果完成信阳旅行计划，请查收。"))
    score = runtime.reward()
    assert score.goal_reached == 1 and score.clean_trace == 0
    assert score.violation_count == 2
    assert score.penalty_total == 0.4
    assert score.total == 1.1


def test_success_reward_has_a_half_point_floor_after_five_violations():
    runtime = EpisodeRuntime(_travel_spec(), _world())
    for _ in range(5):
        runtime.interpret_and_step("请问您计划什么时候出行？")
    _correct_questions(runtime)
    _correct_tools(runtime)
    runtime.step(Final("已根据查询结果完成信阳旅行计划，请查收。"))
    score = runtime.reward()
    assert score.goal_reached == 1 and score.violation_count == 5
    assert score.total == 0.5


def test_success_reward_stays_at_floor_after_six_violations():
    runtime = EpisodeRuntime(
        _travel_spec(), _world(), {"max_person_turns": 20}
    )
    for _ in range(6):
        runtime.interpret_and_step("请问您计划什么时候出行？")
    _correct_questions(runtime)
    _correct_tools(runtime)
    runtime.step(Final("已根据查询结果完成信阳旅行计划，请查收。"))
    score = runtime.reward()
    assert score.goal_reached == 1 and score.violation_count == 6
    assert score.unclipped_total == pytest.approx(0.3)
    assert score.total == 0.5


def test_reached_prefixes_have_diagnostics_but_zero_reward():
    runtime = EpisodeRuntime(_travel_spec(), _world())
    _correct_questions(runtime)
    after_questions = runtime.reward()
    assert after_questions.ask_score == 1.0
    assert after_questions.goal_reached == 0 and after_questions.total == 0.0

    _correct_tools(runtime)
    before_final = runtime.reward()
    assert before_final.call_score == before_final.structure_score == 1.0
    assert before_final.goal_reached == 0 and before_final.total == 0.0


def test_multiple_and_malformed_emissions_each_add_one_u_then_recover():
    runtime = EpisodeRuntime(_travel_spec(), _world())
    runtime.step(Invalid("multiple_questions"))
    runtime.step(Invalid("bad"))
    _correct_questions(runtime)
    _correct_tools(runtime)
    runtime.step(Final("已根据查询结果完成信阳旅行计划，请查收。"))
    score = runtime.reward()
    assert score.violation_count == 2
    assert score.penalties == {"malformed_output": 0.2, "multiple_questions": 0.2}
    assert score.total == 1.1


def test_wrong_multi_call_batch_counts_as_one_illegal_emission():
    runtime = EpisodeRuntime(_travel_spec(), _world())
    _correct_questions(runtime)
    wrong_batch = runtime.step(ToolCalls((
        ToolCall("get_weather_info", WEATHER_ARGS),
        ToolCall("search_travel_guide", GUIDE_ARGS),
    )))
    assert wrong_batch.new_violations == ("extra_or_wrong_tool",)
    _correct_tools(runtime)
    runtime.step(Final("已根据查询结果完成信阳旅行计划，请查收。"))
    score = runtime.reward()
    assert score.goal_reached == 1 and score.violation_count == 1
    assert score.total == 1.3


def test_premature_final_is_terminal_zero_with_no_recoverable_u():
    runtime = EpisodeRuntime(_travel_spec(), _world())
    stopped = runtime.step(Final("现在直接给出答案。"))
    score = runtime.reward()
    assert stopped.terminal and stopped.new_violations == ("premature_final",)
    assert score.goal_reached == 0 and score.violation_count == 0
    assert score.total == 0.0


def test_question_like_wording_inside_legal_final_is_still_text_completion():
    runtime = EpisodeRuntime(_travel_spec(), _world())
    _correct_questions(runtime)
    _correct_tools(runtime)
    done = runtime.interpret_and_step(
        "信阳旅行计划已经根据查询结果整理完成。如需换个出发时间，可以告诉我？"
    )
    assert done.terminal and done.success
    assert runtime.reward().total == 1.5


def test_refusal_syntax_cannot_satisfy_an_entity_free_final_state():
    canonical = "抱歉，我是专门的旅行助手，只能回答旅行相关的问题。"
    spec = EpisodeSpec.from_dict({
        "episode_id": "w4-final",
        "world_id": "w4-final",
        "initial_messages": [{"role": "user", "content": "聊聊旅行"}],
        "person_state": {"goal": {"workflow": "travel_chat"}},
        "response_policy": {"canonical_refusal": canonical},
        "initial_state_id": "final",
        "state_graph": [{
            "state_id": "final",
            "expected_action": {"type": "Final"},
            "next_state": "done",
            "milestone_ids": {"completion": "valid_final"},
        }],
        "reward_milestones": ["valid_final"],
    })
    runtime = EpisodeRuntime(
        spec, SyntheticChinaWorld({"world_id": "w4-final", "entries": []})
    )
    stopped = runtime.interpret_and_step(canonical)
    score = runtime.reward()
    assert stopped.terminal and not stopped.success
    assert stopped.new_violations == ("invalid_final",)
    assert score.goal_reached == 0 and score.total == 0.0


def test_empty_guide_branch_has_fixed_path_without_weather():
    runtime = EpisodeRuntime(_travel_spec(guide_status="empty"), _world("empty"))
    _correct_questions(runtime)
    _correct_tools(runtime, with_weather=False)
    assert runtime.state_id == "final"
    runtime.step(Final("目前没有找到信阳的旅行攻略，请稍后再试。"))
    score = runtime.reward()
    assert score.calls_required == 1
    assert "branch_guide_empty" in score.required_milestones
    assert all("weather" not in item for item in score.required_milestones)
    assert score.is_clean_pass


def test_reward_equivalent_city_alias_uses_canonical_non_ok_world_result():
    runtime = EpisodeRuntime(_travel_spec(guide_status="empty"), _world("empty"))
    _correct_questions(runtime)
    # Xinyang is registry-backed, so the scorer accepts its city-suffix alias. It
    # must still execute the frozen canonical key rather than an always-ok fallback.
    transition = runtime.step(ToolCalls((ToolCall(
        "search_travel_guide", {"location": "信阳市", "search_mode": "hybrid"}
    ),)))
    assert transition.state_id == "final"
    assert transition.tool_results[0].frozen
    assert transition.tool_results[0].result["status"] == "empty"


def test_canonical_w2_ask_is_accepted_in_route_destination_state():
    world = SyntheticChinaWorld({"world_id": "w2", "entries": [{
        "tool": "query_route",
        "arguments": {"start_location": "高密", "end_location": "信阳", "city": "信阳"},
        "result": {
            "status": "ok", "source": "synthetic:test",
            "origin": {"query": "高密", "lat": 1.0, "lng": 2.0},
            "destination": {"query": "信阳", "name": "信阳", "lat": 3.0, "lng": 4.0},
            "routes": {"walking": {"duration_min": 1, "distance_m": 2}},
        },
    }]})
    spec = EpisodeSpec.from_dict({
        "episode_id": "w2", "world_id": "w2",
        "initial_messages": [{"role": "user", "content": "帮我查路线"}],
        "person_state": {"private_slots": {"route_destination": "信阳"},
                         "initially_hidden": ["route_destination"]},
        "initial_state_id": "ask_route_destination",
        "state_graph": [
            {"state_id": "ask_route_destination",
             "expected_action": {"type": "Ask", "slot": "route_destination"},
             "user_response": "目的地是信阳。", "revealed_slots": ["route_destination"],
             "next_state": "route", "milestone_ids": {"ask": "ask_route_destination"}},
            {"state_id": "route", "expected_action": {"type": "ToolCalls", "calls": [{
                "name": "query_route",
                "arguments": {"start_location": "高密", "end_location": "信阳",
                              "city": "信阳"},
             }]}, "expected_result_status": "ok", "result_branches": {"ok": "final"},
             "milestone_ids": {"calls": ["call_route"], "round": "round_route",
                               "branches": {"ok": "branch_route_ok"}}},
            {"state_id": "final", "expected_action": {"type": "Final"},
             "next_state": "done", "milestone_ids": {"completion": "valid_final"}},
        ],
        "reward_milestones": ["ask_route_destination", "call_route", "round_route",
                              "branch_route_ok", "valid_final"],
    })
    runtime = EpisodeRuntime(spec, world)
    parsed = runtime.interpreter.parse("请问您想去哪里？")
    assert isinstance(parsed, Ask) and parsed.slot == "destination"
    transition = runtime.step(parsed)
    assert transition.state_id == "route" and not transition.new_violations


def test_w5_requires_exact_canonical_or_approved_refusal():
    canonical = "抱歉，我是专门的旅行助手，只能回答旅行相关的问题。"
    spec = EpisodeSpec.from_dict({
        "episode_id": "w5", "world_id": "w5",
        "initial_messages": [{"role": "user", "content": "帮我改Python"}],
        "person_state": {"goal": {"workflow": "refusal"}},
        "response_policy": {"canonical_refusal": canonical,
                            "approved_refusals": [canonical + "谢谢理解。"]},
        "initial_state_id": "refuse",
        "state_graph": [{"state_id": "refuse", "expected_action": {"type": "Refusal"},
                         "next_state": "done",
                         "milestone_ids": {"completion": "valid_refusal"}}],
        "reward_milestones": ["valid_refusal"],
    })
    world = SyntheticChinaWorld({"world_id": "w5", "entries": []})
    good = EpisodeRuntime(spec, world)
    assert good.interpret_and_step(canonical).success

    broad_but_unapproved = EpisodeRuntime(spec, world)
    action = broad_but_unapproved.interpreter.parse(
        "抱歉，我是旅行助手，只能处理旅行相关内容。"
    )
    assert isinstance(action, Refusal)
    failed = broad_but_unapproved.step(action)
    assert failed.terminal and not failed.success
    assert failed.new_violations == ("invalid_final",)


def test_w3_pivot_reveals_compiled_slots_not_the_asked_hotel_name():
    spec = EpisodeSpec.from_dict({
        "episode_id": "pivot", "world_id": "pivot",
        "initial_messages": [{"role": "user", "content": "之前那家口碑如何"}],
        "person_state": {
            "private_slots": {"hotel_location": "营口", "requirements": "高档一点"},
            "initially_hidden": ["hotel_name", "hotel_location", "requirements"],
        },
        "initial_state_id": "ask_hotel",
        "state_graph": [
            {"state_id": "ask_hotel", "expected_action": {"type": "Ask", "slot": "hotel_name"},
             "user_response": "算了那家先不纠结了，我要去营口，帮我推荐高档酒店。",
             "revealed_slots": ["hotel_location", "requirements"],
             "concealed_slots": ["hotel_name"], "next_state": "final",
             "milestone_ids": {"ask": "ask_hotel_name"}},
            {"state_id": "final", "expected_action": {"type": "Final"}, "next_state": "done",
             "milestone_ids": {"completion": "valid_final"}},
        ],
        "reward_milestones": ["ask_hotel_name", "valid_final"],
    })
    runtime = EpisodeRuntime(spec, SyntheticChinaWorld({"world_id": "pivot", "entries": []}))
    runtime.interpret_and_step("您想了解哪家酒店的点评呢？")
    assert runtime.person_state.revealed_slots == {"hotel_location", "requirements"}
    assert "hotel_name" not in runtime.person_state.revealed_slots


def test_world_fallback_is_deterministic_schema_valid_and_offline():
    world = SyntheticChinaWorld({"world_id": "fallback", "entries": []})
    cases = [
        ("search_travel_guide", {"location": "北极城"}),
        ("get_weather_info", {"location": "北极城", "start_date": "2026-09-01"}),
        ("query_route", {"start_location": "起点", "end_location": "终点", "city": "北极城"}),
        ("recommend_hotels", {"location": "北极城"}),
        ("get_hotel_reviews", {"hotel_name": "极光酒店"}),
    ]
    for tool, args in cases:
        first, second = world.run(tool, args), world.run(tool, args)
        assert first == second
        validate(tool, first)
    invalid = world.run("get_weather_info", {"location": "北极城", "start_date": "bad"})
    assert invalid["status"] == "error"
    validate("get_weather_info", invalid)


def test_world_rejects_conflicting_results_for_one_normalized_key():
    with pytest.raises(WorldSnapshotError, match="different results"):
        SyntheticChinaWorld({"entries": [
            {"tool": "search_travel_guide", "arguments": GUIDE_ARGS, "result": _guide("ok")},
            {"tool": "search_travel_guide", "arguments": GUIDE_ARGS, "result": _guide("empty")},
        ]})


def test_runtime_rejects_declared_milestone_drift():
    spec = _travel_spec()
    drifted = replace(spec, reward_milestones=spec.reward_milestones[:-1])
    with pytest.raises(EpisodeSpecError, match="reward_milestones drift"):
        EpisodeRuntime(drifted, _world())


def test_sealed_step7_world_hashes_detect_tampering(tmp_path):
    source = DATA_ROOT / "7_world_policy" / "synthetic_china_world_v1.json"
    loaded = SyntheticChinaWorld.load(source)
    assert len(loaded) > 0
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["entries"][0]["result"]["source"] += ":tampered"
    tampered = tmp_path / "tampered-world.json"
    tampered.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(WorldSnapshotError, match="result_sha256 mismatch"):
        SyntheticChinaWorld.load(tampered)


def test_group_of_four_has_isolated_person_and_reward_state():
    repo = EpisodeRepository([_travel_spec()], _world())
    group = repo.reset_group("cn-travel-0053", 4)
    group[0].interpret_and_step("请告诉我您想去哪个城市旅行？")
    assert group[0].state_id == "ask_travel_date"
    assert all(runtime.state_id == "ask_destination" for runtime in group[1:])
    assert all(not runtime.person_state.revealed_slots for runtime in group[1:])
    assert len({id(runtime.person_state) for runtime in group}) == 4
    assert len({id(runtime.world) for runtime in group}) == 1


def test_runtime_caps_and_explicit_truncation_score_exactly_zero():
    runtime = EpisodeRuntime(
        _travel_spec(), _world(),
        {"max_rounds_per_user_turn": 2, "max_assistant_rounds": 10,
         "max_person_turns": 2},
    )
    runtime.step(Invalid("bad"))
    stopped = runtime.step(Invalid("bad"))
    assert stopped.terminal and "truncation" in stopped.new_violations
    score = runtime.reward()
    assert score.goal_reached == 0 and score.clean_trace == 0
    assert score.violation_count == 2
    assert "truncation" not in score.penalties
    assert score.total == score.unclipped_total == 0.0


def test_runtime_rejects_stale_run1_reward_keys():
    with pytest.raises(ValueError, match="unknown RUN #2 reward keys"):
        EpisodeRuntime(
            _travel_spec(),
            _world(),
            {"reward": {"w_ask": 1.0, "clip_max": 4.0}},
        )


def test_all_frozen_training_trajectories_replay_as_clean_run2_completions():
    rows = json.loads(
        (DATA_ROOT / "5_merged" / "train_validation.json").read_text(
            encoding="utf-8"
        )
    )
    repository = EpisodeRepository.load(
        DATA_ROOT / "7_world_policy" / "world_policy_episodes.jsonl",
        DATA_ROOT / "7_world_policy" / "synthetic_china_world_v1.json",
        expected_count=909,
    )
    failures = []
    for row in rows:
        source_idx = row["metadata"]["idx"]
        runtime = repository.new_runtime(source_idx)
        for message in row["conversation"][2:]:
            if message.get("role") == "assistant":
                runtime.interpret_and_step(
                    message.get("content") or "", message.get("tool_calls")
                )
        score = runtime.reward()
        if not score.is_clean_pass or score.total != 1.5:
            failures.append((source_idx, runtime.state_id, runtime.outcome().violations))
    assert failures == []
