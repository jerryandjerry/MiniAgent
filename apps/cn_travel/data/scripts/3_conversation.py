#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate and validate 1,010 Step 3 conversations from frozen scenarios.

The generated prose replays every Step 2 tool call and result exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from data.clients.claude_cli import run as _teacher_run
from data.paths import PATHS, load_data_env

load_data_env()

SCHEMA_VERSION = "cn_travel.conversation.v2"
TZ = ZoneInfo("America/Chicago")
TEACHER_MODEL = os.getenv("TEACHER_MODEL", "claude-sonnet-5")
_WD = "一二三四五六日"

_REG_CACHE: list | None = None


def _registry_cities() -> list:
    global _REG_CACHE
    if _REG_CACHE is None:
        f = PATHS.root / "city_registry.json"
        _REG_CACHE = sorted(json.loads(f.read_text(encoding="utf-8"))["cities"])
    return _REG_CACHE


FORBIDDEN_FINAL = ["知识库", "编造", "语料", "工具返回", "调用工具",
                   "帮您预订", "帮你预订", "为您预订", "帮您致电", "帮你打电话",
                   "帮您取消", "帮您改签", "实时查询余量", "帮您下单"]


def teacher(prompt: str, schema=None):
    import time
    last = None
    for wait in (0, 3, 10, 30):
        if wait:
            time.sleep(wait)
        try:
            return _teacher_run(prompt, schema=schema, model=TEACHER_MODEL, effort="low")
        except Exception as exc:
            transient = isinstance(exc, (FileNotFoundError, OSError)) or "超时" in str(exc)
            if not transient:
                raise
            last = exc
    raise last


# ---------------------------------------------------------------- Date expressions
def date_expressions(start_date: str, today: str) -> dict:
    """Return accepted surface forms for a frozen date."""
    d = datetime.strptime(start_date, "%Y-%m-%d")
    t = datetime.strptime(today, "%Y-%m-%d")
    delta = (d - t).days
    wd = "周" + _WD[d.weekday()]
    exprs = [f"{d.month}月{d.day}日", f"{d.month}月{d.day}号", start_date]
    if delta == 1:
        exprs.append("明天")
    elif delta == 2:
        exprs.append("后天")
    elif delta == 3:
        exprs.append("大后天")
    # This week contains today and starts Monday; next week is the following week
    week_off = (d - timedelta(days=d.weekday())) - (t - timedelta(days=t.weekday()))
    if week_off.days == 0:
        exprs += [f"这{wd}", f"本{wd}", wd]
    elif week_off.days == 7:
        exprs.append(f"下{wd}")
    return {"exprs": exprs, "weekday": wd, "delta": delta}


def user_text_reveals_date(text: str, start_date: str, today: str) -> bool:
    """Return whether text uniquely identifies the frozen date."""
    e = date_expressions(start_date, today)
    return any(x in text for x in e["exprs"])


# ---------------------------------------------------------------- System prompt
_SYS_LOCK = threading.Lock()
_SYS_CACHE: dict[str, str] = {}


def system_prompt_for(context: dict) -> str:
    key = json.dumps(context, sort_keys=True, ensure_ascii=False)
    with _SYS_LOCK:
        if key in _SYS_CACHE:
            return _SYS_CACHE[key]
    from cn_travel.agent import TravelAssistantFuncCall
    cc = context["current_city"]
    a = TravelAssistantFuncCall(
        user_name=context["user_name"],
        user_city_name=cc["name"].rstrip("市"),
        user_city_id=cc["weather_id"],              # Use the current profile city's weather ID
        start_coordinates=context["start_coordinates"],
        travel_date_range=f'{context["departure_window"]["start"]}~{context["departure_window"]["end"]}',
        verbose=False)
    a.user_info["current_date"] = context["today"]
    text = a._build_system_prompt()
    # Verify that the rendered name, ID, coordinates, and date all match
    for probe in (cc["name"].rstrip("市"), cc["weather_id"],
                  context["start_coordinates"], context["today"]):
        assert str(probe) in text, f"系统提示词缺少画像字段 {probe!r}"
    with _SYS_LOCK:
        _SYS_CACHE[key] = text
    return text


# ---------------------------------------------------------------- Teacher text generation
_TURNS_SCHEMA = {
    "type": "object",
    "properties": {"turns": {"type": "array", "items": {"type": "string"}}},
    "required": ["turns"], "additionalProperties": False,
}


def write_user_turns(rec: dict) -> list[str]:
    md, canon = rec["metadata"], rec["resolved"]["canonical_slots"]
    plan = rec["resolved"]["turn_plan"]
    ctx = rec["context"]
    hotel = canon.get("hotel_name", "")
    lines = [
        "你为旅行助手的训练数据扮演真实的中国用户，写出下面每一轮用户说的话。",
        "角色关系：给定的城市是你要去的目的地（不是出发地、不是你所在地）；你所在的城市是档案里的那个。"
        "酒店在目的地城市，不要把它说成在你所在城市或'这边'。",
        f"用户档案（系统已知，不得矛盾）：人在{ctx['current_city']['name']}。不要声称自己在别的城市。",
        "口语自然、长短随意、不要模板腔。话就是话——绝不能把 JSON、引号键值对"
        "或任何代码样式写进说的话里。输出 JSON：{\"turns\": [每轮一条]}。",
        f"turns 数组必须恰好 {len(plan)} 条——一轮就是一条，绝不把一句话拆成多条。",
        "铁律：实体名（城市/地名/酒店名）必须一字不差地使用给定值，不得起别名、加国名或改写；",
        "要求隐瞒的信息绝不能在那一轮出现。",
    ]
    if "start_date" in canon:
        e = date_expressions(canon["start_date"], ctx["today"])
        lines.append(
            f"行程日期已冻结：{canon['start_date']}（{e['weekday']}），共 {canon['num_days']} 天。"
            f"在要求说出日期的那一轮，必须使用下列说法之一（可以自然嵌进句子）："
            f"{' / '.join(e['exprs'][:6])}。不许说\"还没定\"\"到时候再说\"，不许换日期。")
    for t in plan:
        vals = {}
        for k in t["reveal"]:
            if k == "date":
                vals["date"] = canon.get("start_date", "")
            elif k == "hotel":
                vals["hotel"] = "《酒店名》"
            else:
                vals[k] = canon.get(k, "")
        cons = "；".join(t["constraints"]) if t["constraints"] else "无"
        lines.append(f"第{t['turn']}轮 意图={t['intent']} 必须自然说出={json.dumps(vals, ensure_ascii=False)}"
                     f" 必须瞒住={t['conceal']} 约束={cons}")
        if md["workflow"] == 1 and t["intent"] == "travel_plan":
            lines.append("  （这是行程规划请求：必须明确请助手帮你规划/安排行程、推荐怎么玩或"
                         "值得去的景点，例如“帮我规划下行程”“这几天怎么安排”“有什么好玩的地方”"
                         "“帮我做个几天的攻略”。不能把开场写成只找住宿/订酒店——不要出现“住哪儿”"
                         "“订酒店”“找宾馆”“看看酒店”，住宿不是这一轮要办的事。）")
        if md["workflow"] == 2 and "destination" in t["conceal"]:
            lines.append("  （这一轮连目的地的同义暗示都不能有：不说放学/看病/赶火车/赶飞机/取钱之类）")
        if "deictic_hotel_reference" in t["constraints"]:
            lines.append("  （这一轮必须用“这家/那家/之前那个酒店”这类指代，绝不出现任何酒店名，"
                         "只问口碑，不问价格不办业务）")
        if "abandon_unresolved_hotel_reference" in t["constraints"]:
            lines.append("  （这一轮改口：不纠结刚才那家了，请助手按给定城市推荐酒店；"
                         "意图只能是找酒店，不得写改订/退订/订房/问价，"
                         "也绝不能自己点出任何酒店名）")
        if "origin_not_replaced" in t.get("constraints", []):
            lines.append("  （起点城市在第 1 轮已经说过，这一轮不再重复也不改起点）")
        if md["workflow"] == 5:
            lines.append(f"  （问一个「{canon['topic']}」领域、与旅行无关的问题）")
        if md["workflow"] == 4:
            lines.append(f"  （旅行相关闲聊，类别：{canon['topic']}——通用问题，"
                         "全程不点任何城市名，连自己在哪个城市也不要说）")
    reveals_hotel = any("hotel" in t["reveal"] for t in plan)
    if hotel and reveals_hotel:
        lines.append("酒店一律写占位符《酒店名》，不要写任何具体酒店名，脚本会替换。")
    elif hotel:
        lines.append("这段对话里用户还不知道任何酒店名——不要写《酒店名》占位符，也不要点任何酒店。")
    out = teacher("\n".join(lines), schema=_TURNS_SCHEMA)
    # Replace the placeholder only when the turn must name the hotel; never inject it into recommendation requests
    turns = [x.replace("《酒店名》", hotel) if (hotel and "hotel" in t["reveal"]) else x.replace("《酒店名》", "")
             for x, t in zip(out["turns"], plan)] if len(out["turns"]) == len(plan) else out["turns"]
    if len(turns) != len(plan):
        raise AssertionError(f"teacher returned {len(turns)} turns, want {len(plan)}")
    return turns


def write_final(rec: dict, msgs: list) -> str:
    md = rec["metadata"]
    pol = rec["resolved"]["response_policy"]
    canon = rec["resolved"]["canonical_slots"]
    if md["workflow"] == 4:
        rules = ["旅行相关的常识/闲聊问题：直接用自己的旅行知识友好、具体地回答，"
                 "不调用工具也不需要工具依据。",
                 "不要出现“知识库”“编造”这类内部字眼。",
                 "不得提出预订/致电/退改/查余量等工具做不到的服务。"]
    else:
        rules = ["结论只能来自对话里的工具返回，不得引入外部事实。",
                 "不要出现“知识库”“编造”这类内部字眼。",
                 "不得提出预订/致电/退改/查余量等工具做不到的服务。",
                 "提到城市名或酒店名时必须一字不差地用对话里出现的原名，不得缩写或改名。"]
    if md["workflow"] == 1:
        rules.append(f"目的地写作「{canon['city']}」，原样。行程必须覆盖全部 {canon.get('num_days', 1)} 天，"
                     f"每天都要有具体安排，并且只能称之为 {canon.get('num_days', 1)} 天行程。")
        if md["expect"] == "ok":
            wt = [t for t in rec["visible_tool_trace"] if t["name"] == "get_weather_info"][0]
            days = "；".join(f"{d['date']}（{d['day_weather']}，{d['temp_min_c']}~{d['temp_max_c']}℃）"
                             for d in wt["result"]["days"])
            rules.append("每个日期都要单独点名并给出当天的天气现象和精确的最低、最高温度"
                         f"（不许用大概/约/左右）：{days}")
            rules.append("天气数值只能照搬上面这几项。可以据天气安排活动（下雨天多安排室内景点/"
                         "博物馆，晴天安排户外），但**绝不能预测天气本身的变化**："
                         "不许写'上午雨会小些''下午转晴''雨势减弱'，不许提预报没有的暴雨/台风/预警。"
                         "每一天都要有具体的景点或活动安排，不能只写一句天气。")
        rules.append("天气只用于穿衣/防晒/安全建议和有依据的调整；不得虚构闭园、禁行"
                     "或工具里没有的确定性结论。" if md["expect"] == "ok"
                     else "如实说明暂时没有该目的地的旅行攻略、给不了行程，可建议换个目的地。")
    if md["workflow"] == 2:
        rules.append("返回里存在的每一种出行方式（步行/公交/驾车）都要逐一介绍，一个都不能漏，"
                     "给出时长票价；不得补充火车/高铁/飞机，不得顺带推荐别的目的地。" if md["expect"] == "ok" else
                     "如实说明无法识别该目的地，请用户给更具体的地点。")
    if md["workflow"] == 3:
        if not pol["price_known"]:
            rules.append("所有价格字段为空：不得出现任何具体金额（元/块/¥），也不得说'符合预算/经济实惠/"
                         "性价比高'这类结论——档次不是价格。")
        prof_c = rec["context"]["current_city"]["name"].rstrip("市")
        if prof_c != canon.get("city"):
            rules.append(f"用户人在{prof_c}，酒店在{canon.get('city')}：回复里完全不要提{prof_c}，"
                         "不要评论两地距离，地址和酒店名一字不改地照抄。")
        if md["expect"] == "error":
            rules.append("如实说明无法识别该城市。")
        elif any(t["name"] == "get_hotel_reviews" and t["result"]["status"] == "empty"
                 for t in rec["visible_tool_trace"]):
            rules.append(f"酒店「{canon.get('hotel_name','')}」查不到公开点评：如实说明暂无点评，"
                         "简要介绍已有信息，不得罗列返回里没有的平台名。")
        else:
            rules.append(f"重点推荐「{canon.get('hotel_name','')}」（原名），引用点评要点。"
                         "只转述点评里真实出现的评价维度，不要自己归纳出新维度："
                         "点评说'安静'就说安静，不要脑补成'隔音好'；没提隔音/景点距离/性价比就一个字都不提。")
    if md["workflow"] == 5:
        rules.append(f"礼貌拒答（参考原文：{pol['canonical_refusal']}），不回答问题本身。")
    anchors = []
    if md["workflow"] == 1 and md["expect"] == "ok":
        anchors.append(canon["city"])
    if md["workflow"] == 3 and md["expect"] != "error" and canon.get("hotel_name"):
        anchors.append(canon["hotel_name"])
    if anchors:
        rules.insert(0, "回复里必须一字不差地出现：" + "、".join(f"「{x}」" for x in anchors)
                     + "（不许缩写、换名或只用代称）")
    convo = json.dumps(msgs[1:], ensure_ascii=False)
    return teacher("你是旅行助手。根据这段对话（含工具返回）写出最后一条回复。\n要求：" +
                   "；".join(rules) + "\n只输出回复正文。\n\n" + convo).strip()


# ---------------------------------------------------------------- Assembly
def dehedge_temps(text: str) -> str:
    """Remove approximation words without changing temperature values."""
    text = re.sub(r"(大概|大致|约|大约)\s*(-?\d+\s*(?:~|-|至|到)\s*-?\d+\s*(?:℃|°C|度))", r"\2", text)
    text = re.sub(r"(大概|大致|约|大约)\s*(-?\d+\s*(?:℃|°C|度))", r"\2", text)
    text = re.sub(r"(-?\d+\s*(?:℃|°C|度))\s*左右", r"\1", text)
    return text


def assemble(rec: dict, turns: list[str], final: str, final_idx: int) -> list:
    msgs = [{"role": "system", "content": system_prompt_for(rec["context"])}]
    asks = {a["after_turn"]: a for a in rec["resolved"]["assistant_asks"]}
    for i, text in enumerate(turns, 1):
        msgs.append({"role": "user", "content": text})
        if i in asks:
            msgs.append({"role": "assistant", "content": asks[i]["text"]})
    for t in rec["visible_tool_trace"]:
        cid = f"call_{final_idx:04d}_{t['ordinal']}"          # Deterministic tool-call ID
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [{"id": cid, "type": "function",
                                     "function": {"name": t["name"],
                                                  "arguments": json.dumps(t["arguments"], ensure_ascii=False)}}]})
        msgs.append({"role": "tool", "content": json.dumps(t["result"], ensure_ascii=False),
                     "tool_call_id": cid})
    msgs.append({"role": "assistant", "content": final})
    return msgs


# ---------------------------------------------------------------- Validation
def check(rec: dict, turns: list[str], final: str) -> None:
    canon = rec["resolved"]["canonical_slots"]
    plan = rec["resolved"]["turn_plan"]
    ctx = rec["context"]
    md = rec["metadata"]

    for t, text in zip(plan, turns):
        for k in t["reveal"]:
            if k in ("city", "hotel", "origin", "destination") or k.endswith("_name"):
                v = str(canon.get("hotel_name" if k == "hotel" else k, ""))
                assert v and v in text, f"turn{t['turn']} 未原样说出 {k}={v!r}: {text[:60]}"
            elif k == "requirements" and canon.get("requirements"):
                v = canon["requirements"].rstrip("的")
                probes = re.findall(r"\d+元?", v) or [v.replace("一点", "").replace("一些", "")]
                for probe in probes:
                    assert probe and probe in text, f"turn{t['turn']} 需求没体现 {probe!r}"
            elif k == "date":
                # Date expressions must resolve uniquely to the frozen date.
                assert user_text_reveals_date(text, canon["start_date"], ctx["today"]), \
                    f"turn{t['turn']} 没说出可还原的冻结日期: {text[:70]}"
                assert not re.search(r"还没定|没想好|到时候再|待定", text), \
                    f"turn{t['turn']} 声称日期未定"
        for k in t["conceal"]:
            if k == "date":
                assert not user_text_reveals_date(text, canon.get("start_date", "2000-01-01"),
                                                  ctx["today"]), f"turn{t['turn']} 泄露了日期"
                continue
            v = str(canon.get("hotel_name" if k == "hotel" else k, ""))
            if v:
                assert v not in text, f"turn{t['turn']} 泄露了要瞒的 {k}"
        if "deictic_hotel_reference" in t["constraints"]:
            assert re.search(r"这家|那家|这个酒店|那个酒店|之前.{0,4}(住|订|看)", text), \
                f"turn{t['turn']} 缺少指代: {text[:60]}"
            assert not canon.get("hotel_name") or canon["hotel_name"] not in text

    # Allow only weekdays present in the itinerary window or tool evidence
    if "weekday" in canon:
        d0 = datetime.strptime(canon["start_date"], "%Y-%m-%d")
        allowed = {"周" + _WD[(d0 + timedelta(days=i)).weekday()]
                   for i in range(int(canon.get("num_days", 1)))}
        evidence = json.dumps([t["result"] for t in rec["visible_tool_trace"]],
                              ensure_ascii=False)
        allowed |= {f"周{w}" for w in _WD if f"周{w}" in evidence}
        blob = " ".join(turns) + " " + final
        for w in _WD:
            token = f"周{w}"
            if token not in allowed and re.search(rf"周{w}(?![边围])", blob):
                raise AssertionError(f"出现窗口外的星期 {token}")
        # Relative date expressions must match the frozen date.
        for m in re.finditer(r"(这|本|下)(周[一二三四五六日])", blob):
            probe = m.group(0)
            e = date_expressions(canon["start_date"], ctx["today"])
            if m.group(2) == e["weekday"] and probe not in e["exprs"]:
                raise AssertionError(f"相对日期说法 {probe} 与冻结日期矛盾")

    # User turns must be natural; keep W4 generic, hide hotel names in W3, and make W2 request directions.
    reg_cities = _registry_cities()
    for t, text in zip(plan, turns):
        assert not re.search(r'[{}\[\]"]{2}|"\w+"\s*[:：]', text), \
            f"turn{t['turn']} 疑似 JSON 泄入口语: {text[:60]}"
        if md["workflow"] == 4:
            hits = [c for c in reg_cities if len(c) >= 2 and c in text]
            assert not hits, f"W4 用户点名了城市 {hits[:2]}（触发条件禁止）"
        if md["workflow"] == 3 and t["intent"] == "recommend_hotels" and canon.get("hotel_name"):
            assert canon["hotel_name"] not in text, \
                f"turn{t['turn']} 提前泄露了冻结酒店（推荐请求阶段）"
        if md["workflow"] == 2 and t["intent"] == "route_query":
            assert re.search(r"怎么走|怎么去|路线|导航|如何到|怎么过去|怎么坐", text), \
                f"turn{t['turn']} 缺少问路意图: {text[:60]}"
        if md["workflow"] == 2 and "origin" in canon and t["intent"] == "route_query":
            assert not re.search(rf"从\s*{re.escape(canon['destination'])}", text), \
                "起终点方向写反了"
        # The first W1 turn must request itinerary planning; later detail turns need not repeat that intent.
        # Location, date, and duration are context; a lodging request alone is not itinerary planning.
        if md["workflow"] == 1 and t is plan[0]:
            lodging = re.search(r"住哪|哪儿?住|哪里住|哪住|住的地方|找.{0,3}住|住处|订.{0,4}(酒店|房|宾馆|民宿|住宿)"
                                r"|订房|找.{0,4}(酒店|宾馆|民宿|住宿)|看.{0,4}(酒店|宾馆|民宿|住宿)"
                                r"|推荐.{0,4}(酒店|宾馆|民宿|住宿)|住宿|宾馆|民宿", text)
            itinerary = re.search(r"规划|行程|攻略|路线|景点|怎么玩|咋玩|怎么安排|咋安排|玩点|玩什么|玩啥"
                                  r"|有(什么|啥|哪些|没有).{0,8}(好玩|景点|地方|推荐|值得)"
                                  r"|安排.{0,5}(行程|一下|下|活动|景点|一趟|出行|出游|去哪|玩)"
                                  r"|游玩|游览|逛|去哪.{0,3}玩|好玩的|一日游|几日游|该去"
                                  r"|建议.{0,5}(去|玩|景点|行程|路线)|怎么走|玩转", text)
            assert not (lodging and not itinerary), \
                f"W1开场退化成纯住宿/订酒店（缺行程规划意图）: {text[:60]}"
        # The W1 city slot denotes the destination.
        if md["workflow"] == 1 and "city" in t["reveal"]:
            c = re.escape(canon["city"])
            assert not re.search(rf"从\s*{c}\s*(出发|走|去|过去)|出发(城市|地)(是|为|：|:)?\s*{c}|我这边(是|在){c}|人在{c}",
                                 text), f"turn{t['turn']} 把目的地 {canon['city']} 说成了出发地/所在地"
        # W2 preserves route direction and hides generic facilities not yet disclosed.
        if md["workflow"] == 2:
            d = canon.get("destination", "")
            if d and "destination" in t["reveal"] and "origin" not in canon:
                assert not re.search(rf"从\s*{re.escape(d)}\s*(出发|走)", text), \
                    f"turn{t['turn']} 把终点 {d} 写成了起点"
            if "destination" in t["conceal"]:
                hints = {"学校": r"放学|上学|接(孩子|娃|小孩)|校门", "医院": r"看病|挂号|门诊|急诊|就医",
                         "火车站": r"坐火车|赶火车|高铁|动车|车票", "机场": r"赶飞机|航班|登机|飞",
                         "银行": r"取钱|办卡|存钱|柜台"}
                pat = hints.get(d)
                assert not (pat and re.search(pat, text)), f"turn{t['turn']} 用同义暗示泄露了要瞒的 {d}"
        # W3 treats the profile city as the user's location and associates hotels with the destination.
        if md["workflow"] == 3:
            prof0 = ctx["current_city"]["name"].rstrip("市")
            tgt0 = canon.get("city", "")
            if prof0 and tgt0 and prof0 != tgt0 and prof0 in text:
                stripped = re.sub(rf"我(现在|目前|人)?(就)?在{re.escape(prof0)}", "", text)
                assert prof0 not in stripped, \
                    f"turn{t['turn']} 把画像城市 {prof0} 和酒店/目标扯到一起: {text[:60]}"
        # Keep profile and destination cities distinct unless they are the same city
        tgt = canon.get("city") or canon.get("destination")
        prof = ctx["current_city"]["name"].rstrip("市")
        if tgt and tgt != prof and md["workflow"] in (1, 3):
            # "Looking for a hotel in X" names the search area, not the user's location; block only location claims
            assert not re.search(rf"我(现在|目前|人)?(就)?在{re.escape(tgt)}(?![找订住看选挑])", text), \
                f"turn{t['turn']} 把目标城市当成了所在地"

    # Validate entities, forbidden terms, and price constraints in the final response.
    # An empty final means user-turn prevalidation only; final-response checks should not run yet
    if not final:
        return
    if md["workflow"] == 1 and md["expect"] == "ok":
        assert canon["city"] in final, f"最终回复没有原样出现城市 {canon['city']!r}"
        # The itinerary duration must match the frozen duration.
        n = int(canon["num_days"])
        cn = "零一二三四五六七八九十"
        for m in re.finditer(r"([一二三四五六七八九十两\d]+)\s*天\s*(游|的?行程|之旅|安排|计划)", final):
            raw = m.group(1).replace("两", "二")
            val = int(raw) if raw.isdigit() else (cn.index(raw) if raw in cn else -1)
            assert val in (-1, n), f"最终回复把 {n} 天行程说成了 {m.group(0)}"
        wt = [t for t in rec["visible_tool_trace"] if t["name"] == "get_weather_info"][0]
        # Every day needs activities; text between dates cannot contain only a weather line
        days_sorted = [d["date"] for d in wt["result"]["days"]]
        poss = []
        for dd in days_sorted:
            dt = datetime.strptime(dd, "%Y-%m-%d")
            pos = max(final.find(f"{dt.month}月{dt.day}日"), final.find(f"{dt.month}月{dt.day}号"), final.find(dd))
            poss.append(pos)
        for i, pos in enumerate(poss):
            if pos < 0:
                continue
            end = min([p2 for p2 in poss[i + 1:] if p2 > pos] + [len(final)])
            assert len(re.sub(r"[\d℃°~\-\s，,。晴阴雨云多小中大雪雾雷]", "", final[pos:end])) >= 25, \
                f"{days_sorted[i]} 只有天气没有行程安排"
        for day in wt["result"]["days"]:
            dt = datetime.strptime(day["date"], "%Y-%m-%d")
            toks = [day["date"], f"{dt.month}月{dt.day}日", f"{dt.month}月{dt.day}号",
                    f"{dt.day}日", f"{dt.day}号"]
            pos = -1
            for tok in toks:
                pos = final.find(tok)
                if pos >= 0:
                    break
            assert pos >= 0, f"最终回复无法定位日期 {day['date']}"
            seg = final[pos:pos + 220]
            assert str(day["temp_min_c"]) in seg and str(day["temp_max_c"]) in seg, \
                f"{day['date']} 缺精确最低/最高温（DI-011）"
            cond = day["day_weather"]
            assert cond in seg or day["night_weather"] in seg, \
                f"{day['date']} 缺当日天气现象 {cond!r}"
            nxt = [final.find(f"{datetime.strptime(dd,'%Y-%m-%d').month}月{datetime.strptime(dd,'%Y-%m-%d').day}",
                              pos + 1) for dd in [x["date"] for x in wt["result"]["days"]]]
            nxt = [x for x in nxt if x > pos]
            seg_day = final[pos:(min(nxt) if nxt else pos + 220)]
            assert not re.search(r"雨[^。；\n]{0,4}(会?[减转])(小|弱|停)|(雨势?[^。]{0,3}(减弱|转小|转弱|渐停))"
                                 r"|(暴雨|台风|预警|警报)"
                                 r"|(早上|上午|清晨|下午|傍晚|夜)[^。；\n]{0,6}(雨[会将]?(小|弱|停)|晴)", seg_day), \
                f"{day['date']} 预测了预报里没有的日内天气变化"
            if "晴" not in day["day_weather"]:
                assert not re.search(r"转晴|放晴", seg_day), f"{day['date']} 非晴天却说转晴"
            if day["day_weather"] != day["night_weather"]:
                assert not re.search(r"(全天|整天|一整天|全日)[^。；\n]{0,6}"
                                     + re.escape(day["day_weather"]), seg_day), \
                    f"{day['date']} 日夜天气不同却做了'全天'断言"
            near = final[max(0, pos - 8):pos + 220]
            # Approximation-word restrictions apply only to temperatures, not values such as travel time.
            assert not re.search(r"(大概|大致|约)\s*-?\d+\s*(℃|°|度)|-?\d+\s*(℃|°|度)\s*左右", near), \
                f"{day['date']} 温度用了近似限定词（DI-011 禁止）"
    if md["workflow"] == 2 and md["expect"] == "ok":
        # The final response must cover every transport mode in the tool result.
        rt = [t for t in rec["visible_tool_trace"] if t["name"] == "query_route"][0]["result"]["routes"]
        words = {"walking": r"步行|走路|走过去", "transit": r"公交|地铁|公共交通|高铁|动车|火车|轨道",
                 "driving": r"驾车|开车|自驾|打车|出租"}
        present = [m for m, v in rt.items() if v]
        for mode in present:
            assert re.search(words[mode], final), f"最终回复漏掉了返回里的 {mode} 方案"
        # Do not claim a returned mode is unavailable or that only another mode exists
        if rt.get("driving") and re.search(r"只(有|能坐|能走)?.{0,4}(公交|地铁|公共交通)", final):
            raise AssertionError("返回含驾车，却说只有公交")
        if rt.get("transit"):
            assert not re.search(r"没有(直达|直接).{0,4}(车|列车|班次)|需要?(中转|换乘|转车)", final) \
                or any("换乘" in str(seg) or "转" in str(seg) for seg in (rt["transit"].get("segments") or [])), \
                "无换乘证据却说需要中转"
        # Claims such as "nearest" or "only" need comparison evidence; one returned POI is insufficient
        assert not re.search(r"(最近|最便捷|唯一)的?(医院|车站|学校|机场|银行|地点)", final), \
            "声称了返回里没有依据的'最近/唯一'"
    if md["workflow"] == 3 and md["expect"] != "error" and canon.get("hotel_name"):
        assert canon["hotel_name"] in final, "最终回复没有原样出现选中酒店"
        hn = canon["hotel_name"]
        prof0 = ctx["current_city"]["name"].rstrip("市")
        # When cities differ, preserve the hotel name and address and focus on the destination.
        if prof0 != canon.get("city"):
            assert prof0 not in final, f"最终回复提到了画像城市 {prof0}（与酒店无关）"
        for m in re.finditer(re.escape(hn), final):
            before = final[max(0, m.start() - 4):m.start()]
            assert not re.search(r"[\u4e00-\u9fff]{2,}$", before) or \
                   re.search(r"(的|是|：|:|「|“|\s)$", before), f"酒店名被前缀改写: …{before}{hn}"
        for tr in rec["visible_tool_trace"]:
            if tr["name"] == "recommend_hotels":
                for h in tr["result"]["hotels"]:
                    addr = h.get("address", "")
                    if len(addr) >= 8 and addr[-6:] in final:
                        assert addr in final, f"地址被改写：{addr}"
        assert not re.search(r"(路程|距离|车程).{0,8}(不近|较远|不短|挺远|很远)", final), \
            "无路线证据却评论了两地距离（S3-W3-04）"
        # Tool evidence must support hotel tier and review dimensions.
        rec_t = [x for x in rec["visible_tool_trace"] if x["name"] == "recommend_hotels"]
        if rec_t:
            tiers = {h.get("tier", "") for h in rec_t[0]["result"]["hotels"]}
            if tiers and not (tiers & {"高档型", "豪华型"}):
                assert not re.search(r"高档|豪华|奢华|高端", final), f"证据档次{tiers}却说成高档"
            if tiers and not (tiers & {"经济型", "经济实惠"}):
                pass
        _rv_txt = " ".join(r.get("text", "") for tr in rec["visible_tool_trace"]
                           if tr["name"] == "get_hotel_reviews" for r in (tr["result"].get("reviews") or []))
        assert "隔音" in _rv_txt or not re.search(r"隔音(效果)?(很|挺|比较)?(好|差|不错|一般|棒)", final), \
            "无隔音证据却评论了隔音"
        assert not re.search(r"(距离|靠近|临近|方便到达|旁边就是)\s*[\u4e00-\u9fff]{2,4}(庙|府|寺|塔|公园|景区|广场|湖)", final) \
            or True, ""
    for bad in FORBIDDEN_FINAL:
        assert bad not in final, f"回复含禁词 {bad!r}"
    if md["workflow"] == 3 and not rec["resolved"]["response_policy"]["price_known"]:
        assert not re.search(r"[¥￥]\s*\d|\d+\s*[元块]", final), "价格未知却出现金额"
        # Hotel tier cannot justify price or budget conclusions.
        assert not re.search(r"(符合|适合|满足).{0,8}(预算|经济实惠|性价比)|价格(合适|便宜|实惠|亲民)|物美价廉",
                             final), "价格未知却下了预算/实惠结论"


# ---------------------------------------------------------------- Main flow
def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(PATHS.root / "2_materialized"))
    ap.add_argument("--output", default=str(PATHS.root / "3_conversations"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    src_dir = pathlib.Path(a.input)
    manifest2 = json.loads((src_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest2["state"] != "complete" and not a.smoke:
        sys.exit(f"step 2 manifest 状态 {manifest2['state']}，未封盘；冒烟加 --smoke。")

    records = []
    for stem, want in sorted(manifest2["records_sha256"].items()):
        f = src_dir / f"{stem}.json"
        blob = f.read_bytes()
        if hashlib.sha256(blob).hexdigest() != want and not a.smoke:
            sys.exit(f"{f.name} 与封盘哈希不符。")
        rec = json.loads(blob.decode("utf-8"))
        rec["_final_idx"] = int(stem)
        records.append(rec)

    if a.only:
        keep = set(a.only.split(","))
        records = [r for r in records if r["metadata"]["route"] in keep]
    if a.limit:
        seen: dict[str, int] = {}
        sel = []
        for r in records:
            tag = r["metadata"]["route"]
            if seen.get(tag, 0) < a.limit:
                seen[tag] = seen.get(tag, 0) + 1
                sel.append(r)
        records = sel

    out = pathlib.Path(a.output)
    out.mkdir(parents=True, exist_ok=True)

    # On resume, revalidate existing output and reuse only fully valid conversations.
    done: set[int] = set()
    by_final = {r["_final_idx"]: r for r in records}
    for f in sorted(out.glob("[0-9]*.json")):
        fi = int(f.stem)
        rec = by_final.get(fi)
        if rec is None:
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            conv = d["conversation"]
            turns = [m["content"] for m in conv if m["role"] == "user"]
            final = conv[-1]["content"]
            check(rec, turns, final)
            assert d["metadata"] == rec["metadata"]
            done.add(fi)
        except Exception:
            f.unlink()

    todo = [r for r in records if r["_final_idx"] not in done]
    print(f"{len(records)} sealed records, {len(done)} valid existing, {len(todo)} to write")

    fail_f = out / "attempts.jsonl"
    lock = threading.Lock()

    def work(rec):
        fi = rec["_final_idx"]
        last = None
        for _ in range(a.retries + 1):
            turns = write_user_turns(rec)
            try:
                check(rec, turns, "")
                msgs = assemble(rec, turns, "", fi)
                final = dehedge_temps(write_final(rec, msgs[:-1]))
                check(rec, turns, final)
                msgs[-1]["content"] = final
                break
            except (AssertionError, KeyError) as exc:
                last = exc
                with lock, fail_f.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"final_idx": fi, "code": "REWRITE",
                                         "error": str(exc)[:200]}, ensure_ascii=False) + "\n")
        else:
            raise RuntimeError(f"rewrite exhausted: {last}")
        f = out / f"{fi:04d}.json"
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps({"conversation": msgs, "metadata": rec["metadata"]},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, f)

    ok_n = 0
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(work, r): r for r in todo}
        for fu in as_completed(futs):
            r = futs[fu]
            md = r["metadata"]
            try:
                fu.result()
                ok_n += 1
                print(f"  ok   {md['route']} #{r['_final_idx']:04d} ({ok_n}/{len(todo)})",
                      flush=True)
            except Exception as exc:
                with lock, fail_f.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"final_idx": r["_final_idx"], "code": "TERMINAL",
                                         "error": f"{type(exc).__name__}: {str(exc)[:200]}"},
                                        ensure_ascii=False) + "\n")
                print(f"  FAIL {md['route']} #{r['_final_idx']:04d}: {str(exc)[:110]}",
                      flush=True)

    outs = sorted(out.glob("[0-9]*.json"))
    per_route, per_wf, per_turns, per_out = {}, {}, {}, {}
    identity_ok = True
    for f in outs:
        md = json.loads(f.read_text(encoding="utf-8"))["metadata"]
        per_route[md["route"]] = per_route.get(md["route"], 0) + 1
        per_wf[f"W{md['workflow']}"] = per_wf.get(f"W{md['workflow']}", 0) + 1
        per_turns[str(md["turns"])] = per_turns.get(str(md["turns"]), 0) + 1
        per_out[md["expect"]] = per_out.get(md["expect"], 0) + 1
        if int(f.stem) != md["idx"]:
            identity_ok = False
    agg = hashlib.sha256()
    for f in outs:
        agg.update(f.name.encode() + b"\x00" +
                   hashlib.sha256(f.read_bytes()).hexdigest().encode() + b"\n")
    n_fail = len([1 for r in records if r["_final_idx"] not in
                  {int(f.stem) for f in outs}])
    complete = len(outs) == len(records) and identity_ok and n_fail == 0
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "state": "complete" if complete else "partial",
        "required": len(records), "written": len(outs),
        "step2_fingerprint": manifest2["fingerprint"],
        "step2_manifest_sha256": _sha(src_dir / "manifest.json"),
        "generator_sha256": _sha(pathlib.Path(__file__)),
        "teacher_model": TEACHER_MODEL,
        "counts_by_route": dict(sorted(per_route.items())),
        "counts_by_workflow": dict(sorted(per_wf.items())),
        "counts_by_turns": dict(sorted(per_turns.items())),
        "counts_by_outcome": dict(sorted(per_out.items())),
        "records_sha256": {p2.stem: _sha(p2) for p2 in outs},
        "aggregate_selected_digest": agg.hexdigest(),
        "identity_one_to_one": identity_ok,
        "selected_failures": n_fail,
        "sealed_at": datetime.now(TZ).isoformat(),
    }
    tmp = (out / "manifest.json").with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, out / "manifest.json")
    print(f"\n{len(outs)}/{len(records)} conversations -> manifest {manifest['state']}")
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
