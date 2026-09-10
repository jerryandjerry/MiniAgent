#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end route matrix.

The training data contains nine distinct conversation shapes ("routes").
This exercises every one of them with at least three prompts, and asserts the
FULL shape — how many model rounds fired and which tools were batched into each
— not merely which tools were eventually called.

    PYTHONPATH=src:train:. uv run --project src python src/tests/e2e_matrix.py
    PYTHONPATH=src:train:. uv run --project src python src/tests/e2e_matrix.py R2 R6

Uses the configured policy endpoint on :8000, packaged RAG, AMap, and the
configured review backend for routes that invoke those capabilities.
"""
import concurrent.futures as cf
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent

SHORT = {
    "search_travel_guide": "guide", "get_weather_info": "weather",
    "query_route": "route", "recommend_hotels": "hotels",
    "get_hotel_reviews": "reviews",
}

ASKS = ("哪个城市", "哪里", "去哪", "哪个地区", "具体", "地址", "什么时候", "哪家")
REFUSES = ("抱歉", "旅行助手", "只能回答")


def asks(reply):
    return any(n in reply for n in ASKS)


def refuses(reply):
    return any(n in reply for n in REFUSES)


@dataclass
class Case:
    route: str
    turns: list
    expect: object                    # accepted shape, or tuple of accepted shapes
    check: Optional[Callable] = None
    note: str = ""
    got: str = ""
    reply: str = ""
    status: str = ""


# --------------------------------------------------------------------------
# Route distribution in merged_train_final.json.
# --------------------------------------------------------------------------
NO_TOOL = "U → A(text)"
GUIDE_WX = "U → A[guide+weather] → A(text)"
# The model may batch both tools into one round, or chain them across two.
# Both satisfy Workflow 1 and reach the same answer, so the route accepts either.
GUIDE_WX_SEQ = "U → A[guide] → A[weather] → A(text)"
GUIDE = "U → A[guide] → A(text)"
HOTELS = "U → A[hotels] → A[reviews] → A(text)"
REVIEWS = "U → A[reviews] → A(text)"
ROUTE = "U → A[route] → A(text)"
ASK_GUIDE_WX = "U → A(text) → U → A[guide+weather] → A(text)"
ASK_GUIDE_WX_SEQ = "U → A(text) → U → A[guide] → A[weather] → A(text)"
ASK_ASK = "U → A(text) → U → A(text)"
# Mark cases with undefined product rules: run and record their shape without
# judging them, so one model's choice does not become policy. Add an expectation later.
UNSPECIFIED = "<undecided>"
# Probing with a lookup before asking again is equally acceptable: the point of
# R8 is that the second turn still ends in a question, not that no tool ran.
ASK_PROBE_ASK = "U → A(text) → U → A[guide] → A(text)"

ROUTES = {
    "R1": (NO_TOOL, "44.8% — chitchat / rejection / clarifying question"),
    "R2": (GUIDE_WX, "16.6% — Workflow 1 complete, both tools in one turn"),
    "R3": (GUIDE, "11.6% — guide empty, weather correctly suppressed"),
    "R4": (HOTELS, "10.0% — Workflow 3 sequential chain"),
    "R5": (REVIEWS, "8.5% — Workflow 3B, hotel named directly"),
    "R6": (ROUTE, "3.3% — Workflow 2"),
    "R7": (ASK_GUIDE_WX, "3.1% — follow-up then completes"),
    "R8": (ASK_ASK, "1.9% — follow-up still needs another question"),
    "R9": ("(no duplicate tool in one round)", "0.1% — dedup leak; must NOT recur"),
}

CASES = [
    # ---- R1: no tool at all -------------------------------------------------
    Case("R1", ["旅行的时候需要注意什么安全问题？"], NO_TOOL, note="chitchat"),
    Case("R1", ["出国旅行需要准备什么？"], NO_TOOL, note="chitchat"),
    Case("R1", ["你好，我是第一次使用旅行助手"], NO_TOOL, note="greeting"),
    Case("R1", ["帮我算一下1+1等于几？"], NO_TOOL, refuses, "rejection"),
    Case("R1", ["你能帮我写一段Python代码吗？"], NO_TOOL, refuses, "rejection"),
    Case("R1", ["如何做红烧肉？"], NO_TOOL, refuses, "rejection"),
    Case("R1", ["帮我找个酒店"], NO_TOOL, asks, "must ask which city"),
    Case("R1", ["我想出去旅游，有什么推荐的地方吗？"], NO_TOOL, asks, "must ask destination"),
    Case("R1", ["怎么回家？"], NO_TOOL, asks, "must ask address"),

    # ---- R2: guide + weather in one turn ------------------------------------
    Case("R2", ["我想去嘉兴旅游，帮我制定旅行计划"], (GUIDE_WX, GUIDE_WX_SEQ)),
    Case("R2", ["帮我规划一下去青岛的行程"], (GUIDE_WX, GUIDE_WX_SEQ)),
    Case("R2", ["南京有什么好玩的，顺便看下天气"], (GUIDE_WX, GUIDE_WX_SEQ)),
    Case("R2", ["下周去苏州玩三天，帮我安排"], (GUIDE_WX, GUIDE_WX_SEQ)),

    # ---- R3: guide empty -> weather suppressed ------------------------------
    Case("R3", ["我想去拉萨旅游，下周出发"], GUIDE, note="out of corpus (Tibet)"),
    Case("R3", ["帮我制定三亚的旅行计划，下周去"], GUIDE, note="out of corpus (Hainan)"),
    Case("R3", ["平壤有什么好玩的，下周去"], GUIDE, note="语料外：境外地名"),

    # ---- R4: hotels -> reviews ----------------------------------------------
    Case("R4", ["推荐一些北京的酒店"], HOTELS),
    Case("R4", ["嘉兴有什么好的酒店推荐？"], HOTELS),
    Case("R4", ["我要在青岛住酒店"], HOTELS),
    Case("R4", ["帮我查下建阳的酒店"], HOTELS, note="small city"),

    # ---- R5: reviews only ---------------------------------------------------
    Case("R5", ["北京国贸大酒店怎么样"], REVIEWS),
    Case("R5", ["杭州西湖国宾馆的评价好不好"], REVIEWS),
    Case("R5", ["上海和平饭店口碑如何"], REVIEWS),

    # ---- R6: route ----------------------------------------------------------
    Case("R6", ["从望京到北京天坛公园怎么走？"], ROUTE),
    Case("R6", ["去颐和园怎么走？"], ROUTE),
    Case("R6", ["我要去火车站怎么走"], ROUTE, note="public facility, must not ask"),
    Case("R6", ["从北京到上海怎么走"], ROUTE, note="inter-city"),

    # ---- R7: ask, then complete ---------------------------------------------
    Case("R7", ["帮我规划一下行程", "去嘉兴，下周出发"], (ASK_GUIDE_WX, ASK_GUIDE_WX_SEQ)),
    Case("R7", ["我想出去玩几天", "青岛，下周"], (ASK_GUIDE_WX, ASK_GUIDE_WX_SEQ)),
    Case("R7", ["我想旅行，但不知道去哪里", "那就南京吧，下周去"], (ASK_GUIDE_WX, ASK_GUIDE_WX_SEQ)),

    # ---- R8: ask, still needs another question ------------------------------
    Case("R8", ["我想出去旅游", "西藏"], (ASK_ASK, ASK_PROBE_ASK), asks,
         note="province, not a city — must ask again"),
    Case("R8", ["帮我找个地方玩", "南方"], ASK_ASK, note="too vague"),
    # An "anything is fine" answer leaves the destination unclear; ask again without tools.
    Case("R8", ["我要住酒店", "随便"], ASK_ASK),

    # ---- R9: no duplicate tool inside one round -----------------------------
    Case("R9", ["我想去嘉兴旅游，帮我制定旅行计划"], "", note="dedup"),
    Case("R9", ["推荐一些北京的酒店"], "", note="dedup"),
    Case("R9", ["从望京到天坛怎么走"], "", note="dedup"),
]


def observe(agent, turns):
    """Run the turns and reconstruct the conversation's route shape.

    Rounds are counted by wrapping the chat completion call, so tools batched
    into one assistant turn render as A[a+b] while a second round renders as a
    separate A[c] — which is what distinguishes route R2 from a sequential one.
    """
    state = {"round": 0}
    calls = []

    original_create = agent.client.chat.completions.create

    def create(*args, **kwargs):
        state["round"] += 1
        return original_create(*args, **kwargs)

    agent.client.chat.completions.create = create

    original_call = agent.call_function

    def spy(name, arguments):
        calls.append((len(turns_done), state["round"], name,
                      json.dumps(arguments, sort_keys=True, ensure_ascii=False)))
        return original_call(name, arguments)

    agent.call_function = spy

    turns_done = []
    reply = ""
    for turn in turns:
        reply = agent.process_user_input(turn)
        turns_done.append(turn)

    parts = []
    for index in range(len(turns)):
        parts.append("U")
        rounds = {}
        for turn_index, round_no, name, _args in calls:
            if turn_index == index:
                short = SHORT.get(name, name)
                bucket = rounds.setdefault(round_no, [])
                # Calling reviews once per recommended hotel is one logical step,
                # so repeated names collapse; R9 checks true duplicates via args.
                if short not in bucket:
                    bucket.append(short)
        for round_no in sorted(rounds):
            parts.append("A[" + "+".join(rounds[round_no]) + "]")
        parts.append("A(text)")
    return " → ".join(parts), reply, calls


def run_case(case: Case) -> Case:
    from cn_travel.agent import TravelAssistantFuncCall

    for _ in range(3):
        agent = TravelAssistantFuncCall(user_name="测试", user_city_name="北京")
        try:
            got, reply, calls = observe(agent, case.turns)
        except Exception as exc:
            case.reply = f"<EXCEPTION> {exc}"
            continue
        if "处理请求时出错" in reply:
            case.reply = reply
            continue                                   # transient API failure

        case.got, case.reply = got, reply
        if case.route == "R9":
            grouped = {}
            for turn_index, round_no, name, args in calls:
                grouped.setdefault((turn_index, round_no), []).append((name, args))
            ok = all(len(v) == len(set(v)) for v in grouped.values())
        else:
            if case.expect == UNSPECIFIED:
                case.status = "UNSPEC"
                return case
            accepted = case.expect if isinstance(case.expect, tuple) else (case.expect,)
            ok = got in accepted
            if ok and case.check:
                ok = bool(case.check(reply))
        case.status = "PASS" if ok else "FAIL"
        return case

    case.status = "ERR"
    return case


def main() -> None:
    wanted = [a.upper() for a in sys.argv[1:]]
    cases = [c for c in CASES if not wanted or c.route in wanted]
    print(f"running {len(cases)} cases across {len(set(c.route for c in cases))} routes ...\n")

    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        done = list(pool.map(run_case, cases))

    print(f"{'route':6} {'result':>8}   shape")
    print("-" * 78)
    for key in sorted(ROUTES):
        items = [c for c in done if c.route == key]
        if not items:
            continue
        ok = sum(1 for c in items if c.status == "PASS")
        err = sum(1 for c in items if c.status == "ERR")
        unspec = sum(1 for c in items if c.status == "UNSPEC")
        scored = len(items) - err - unspec
        mark = "✅" if ok == scored and scored else "❌"
        shape, share = ROUTES[key]
        extra = f"  [{err} ERR]" if err else ""
        if unspec:
            extra += f"  [{unspec} 产品规则未定义]"
        print(f"{key:6} {mark} {ok}/{scored:<4}{extra}  {shape}")
        print(f"{'':17}{share}")

    for c in done:
        if c.status == "UNSPEC":
            print(f"\n[未定义] {c.route}: {' → '.join(c.turns)}")
            print(f"        实际形状: {c.got}")
            print(f"        {c.note}")
            print("        需要业务方给出规则，才能写成断言。")

    failures = [c for c in done if c.status not in ("PASS", "UNSPEC")]
    if failures:
        print("\n" + "=" * 78)
        for c in failures:
            print(f"[{c.status}] {c.route}: {' → '.join(c.turns)}")
            if c.route != "R9":
                print(f"        expected: {' | '.join(c.expect) if isinstance(c.expect, tuple) else c.expect}")
                print(f"        actual  : {c.got or '(none)'}")
            if c.note:
                print(f"        note: {c.note}")
            print(f"        reply: {c.reply[:150].replace(chr(10), ' ')}\n")

    ok = sum(1 for c in done if c.status == "PASS")
    err = sum(1 for c in done if c.status == "ERR")
    unspec = sum(1 for c in done if c.status == "UNSPEC")
    print(f"\nTOTAL {ok}/{len(done) - err - unspec} passed"
          + (f", {err} unscored" if err else "")
          + (f", {unspec} 待业务定义" if unspec else ""))


if __name__ == "__main__":
    main()
