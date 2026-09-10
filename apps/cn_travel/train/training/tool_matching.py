"""Training-owned tool-call canonicalization and sequence matching."""
from __future__ import annotations

import json
import re

from .paths import DATA_ROOT


_REG_KEYS = frozenset(json.loads(
    (DATA_ROOT / "city_registry.json").read_text(encoding="utf-8")
)["cities"])
_LOCATION_FIELDS = frozenset({"location", "city", "start_location", "end_location"})
_REQUIREMENT_CLASSES = (
    (re.compile(r"市中心附近(?:的)?(?:酒店)?"), "市中心附近的"),
    (re.compile(r"经济实惠(?:一点)?(?:的)?(?:酒店)?"), "经济实惠一点"),
    (re.compile(r"(?:评分高(?:一点|一些)?|(?:一点|一些)?高评分)(?:的)?(?:酒店)?"),
     "评分高一些的"),
    (re.compile(r"交通方便(?:一点)?(?:的)?(?:酒店)?"), "交通方便的"),
    (re.compile(r"高档(?:一点)?(?:的)?(?:酒店)?"), "高档一点的"),
)
_BUDGET_REQUIREMENT = re.compile(
    r"预算(?:大概|大约)?(?P<amount>[1-9]\d{1,4})元左右"
    r"(?:一晚|每晚)?(?:的)?(?:酒店)?(?:就行)?"
)


def canonicalize_requirements(value):
    if not isinstance(value, str):
        return value
    for pattern, canonical in _REQUIREMENT_CLASSES:
        if pattern.fullmatch(value):
            return canonical
    match = _BUDGET_REQUIREMENT.fullmatch(value)
    if match:
        return f"预算{match.group('amount')}元左右"
    return value


def normalize_tool_args(tool: str, args: dict) -> dict:
    out = {}
    for key, value in args.items():
        if (key in _LOCATION_FIELDS and isinstance(value, str)
                and value.endswith("市") and value[:-1] in _REG_KEYS):
            value = value[:-1]
        if tool == "recommend_hotels" and key == "requirements":
            value = canonicalize_requirements(value)
        out[key] = value
    return out


WORLD_ARGS = {
    "search_travel_guide": {"location": str},
    "get_weather_info": {"location": str, "start_date": str, "num_days": int},
    "query_route": {"start_location": str, "end_location": str, "city": "strip市"},
    "recommend_hotels": {"location": str, "requirements": "optional"},
    "get_hotel_reviews": {"hotel_name": str, "location": "optional"},
}


def project_args(tool: str, args: dict) -> dict | None:
    if not isinstance(args, dict):
        return None
    spec = WORLD_ARGS.get(tool)
    if spec is None:
        return None
    args = normalize_tool_args(tool, args)
    out = {}
    for key, kind in spec.items():
        value = args.get(key)
        if kind == "optional":
            out[key] = str(value) if value is not None else ""
            continue
        if value is None:
            return None
        if kind is int:
            try:
                out[key] = int(value)
            except (TypeError, ValueError):
                return None
        elif kind == "strip市":
            if not isinstance(value, str):
                return None
            out[key] = value.rstrip("市")
        else:
            if not isinstance(value, str):
                return None
            out[key] = value
    return out


def golden_segments(conversation: list[dict]) -> list[dict]:
    segments, current = [], None
    for message in conversation:
        role = message.get("role")
        if role == "user":
            if current is not None:
                segments.append(current)
            current = {"user": message["content"], "rounds": [], "calls": []}
        elif role == "assistant" and current is not None:
            calls = [
                {
                    "tool": call["function"]["name"],
                    "args": json.loads(call["function"]["arguments"]),
                }
                for call in message.get("tool_calls") or []
            ]
            if calls:
                current["rounds"].append(calls)
                current["calls"].extend(calls)
    if current is not None:
        segments.append(current)
    return segments


def _json_eq(left, right) -> bool:
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _json_eq(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_eq(a, b) for a, b in zip(left, right)
        )
    if left != right:
        return False
    return type(left) is type(right) or (
        isinstance(left, (int, float)) and isinstance(right, (int, float))
    )


def compare_args(golden_args: dict, made_args, tool: str | None = None) -> str | None:
    if not isinstance(made_args, dict):
        return "unparseable_args"
    golden = normalize_tool_args(tool or "", golden_args)
    made = normalize_tool_args(tool or "", made_args)
    if _json_eq(golden, made):
        return None
    for key in sorted(set(golden) | set(made)):
        if key not in golden or key not in made or not _json_eq(golden[key], made[key]):
            return f"wrong_args:{key}"
    return "wrong_args:?"


def _call_eq(golden: dict, made: dict) -> bool:
    return (
        made.get("tool") == golden["tool"]
        and compare_args(
            golden["args"], made.get("arguments"), golden["tool"]
        ) is None
    )


def lcs_pairs(golden: list[dict], made: list[dict]) -> list[tuple[int, int]]:
    n, m = len(golden), len(made)
    lengths = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            lengths[i][j] = (
                lengths[i + 1][j + 1] + 1
                if _call_eq(golden[i], made[j])
                else max(lengths[i + 1][j], lengths[i][j + 1])
            )
    pairs, i, j = [], 0, 0
    while i < n and j < m:
        if (_call_eq(golden[i], made[j])
                and lengths[i][j] == lengths[i + 1][j + 1] + 1):
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif lengths[i + 1][j] >= lengths[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs
