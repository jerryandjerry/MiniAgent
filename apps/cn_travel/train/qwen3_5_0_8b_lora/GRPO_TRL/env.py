#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training environment for §5.3 per-user-turn GRPO: the §6.2 frozen world wired into
TRL's multi-turn generation loop.

TRL's environment_factory contract: an environment instance's public methods are its
tools (the schema is rendered from signature + docstring), reset(**row) runs before each
rollout, and get_reward() returns that rollout's reward. TRL masks tool-result tokens out
of the loss automatically (tool_mask).

One training unit = one user segment. After the sealed prefix (system plus the golden
messages up to that user turn), the model is free to call tools, read the world's real
answers, and decide what to do next, until it answers without a call or hits the round
cap. The reward looks only at the call sequence at the end of that segment - the same
standard as §6.2 (ordered LCS pairing + argument equality via _compare + a penalty for
extra calls). No new adjudication is introduced.

The world holds training-side sealed evidence only (train_validation.json); eval evidence
is never loaded during training.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

TRAIN_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = TRAIN_ROOT.parent / "src"
for _path in (TRAIN_ROOT, SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from training.frozen_world import FrozenWorld  # noqa: E402
from training.paths import DATA_ROOT, TOOL_SCHEMAS  # noqa: E402
from training.tool_matching import (  # noqa: E402
    compare_args as _compare,
    golden_segments,
    lcs_pairs as _lcs_pairs,
)

TRAIN_FILE = DATA_ROOT / "6_multiturn" / "train_validation.json"
GOVERNED_PREFIXES = 3632
_BIZ_SCHEMAS = {t["function"]["name"]: t
                for t in json.loads(TOOL_SCHEMAS.read_text(encoding="utf-8"))}


def install_schema_bridge() -> None:
    """Make the 5 tool schemas the template renders byte-identical to
        business_logic/tool_schemas.json.

        train == serve is this project's bedrock: at eval time the model sees the business
        schema verbatim, so training must match exactly. But a TRL environment tool can only
        be a bound method, and for methods the template always goes through transformers'
        get_json_schema (inferred from signature + docstring), which cannot express fields
        like enum defaults from the business schema - so the render would disagree with eval.
        Hence at the render entry point we substitute the business schema by function name;
        functions outside this environment keep the original implementation. This is an
        adapter and does not change a single byte of the business schema.
    """
    from transformers.utils import chat_template_utils as _ctu

    if getattr(_ctu.get_json_schema, "_cn_travel_bridge", False):
        return
    _orig = _ctu.get_json_schema

    def bridged(func):
        biz = _BIZ_SCHEMAS.get(getattr(func, "__name__", ""))
        return biz if biz is not None else _orig(func)

    bridged._cn_travel_bridge = True
    _ctu.get_json_schema = bridged

# Building the world reads the Step 2 generator + exec_cache (seconds), so build it once
# per process. Sharing it across environment instances is safe: the world is a read-only,
# deterministic lookup/generation.
_WORLD: FrozenWorld | None = None


def world() -> FrozenWorld:
    global _WORLD
    if _WORLD is None:
        _WORLD = FrozenWorld(sealed_set=TRAIN_FILE, on_conflict="first")
    return _WORLD


def build_episodes() -> list[dict]:
    """3,632 unrolled rows -> 909 complete conversations -> 1,018 user segments
        (one training unit each).

        Per training unit:
          prompt  = the sealed messages before this user turn, plus the turn itself (from system);
          golden  = that segment's golden call sequence (ordered, flattened across turns).
    """
    rows = json.loads(TRAIN_FILE.read_text(encoding="utf-8"))
    if len(rows) != GOVERNED_PREFIXES:
        raise SystemExit(f"train_validation.json has {len(rows)} rows != the governed population {GOVERNED_PREFIXES}")
    by_idx: dict[int, list] = collections.defaultdict(list)
    for r in rows:
        by_idx[r["metadata"]["idx"]].append(r)
    # Among Step 6's cumulative slices the longest is the full parent conversation (the final-turn
    # one is also duplicated x3, so taking the longest deduplicates)
    convs = {i: max(v, key=lambda r: len(r["conversation"]))["conversation"]
             for i, v in by_idx.items()}

    episodes = []
    for idx in sorted(convs):
        conv = convs[idx]
        segs = golden_segments(conv)
        cut = 0                      # messages consumed from conv so far
        for seg_no, seg in enumerate(segs):
            # prefix = the sealed messages up to and including this segment's user turn
            while cut < len(conv) and conv[cut].get("role") != "user":
                cut += 1
            prefix = [dict(m) for m in conv[:cut + 1]]
            cut += 1
            # skip this segment's remaining sealed messages (golden assistant/tool) - they are exactly
            # what the model must produce itself
            while cut < len(conv) and conv[cut].get("role") != "user":
                cut += 1
            episodes.append({
                "prompt": prefix,
                "golden": json.dumps(seg["calls"], ensure_ascii=False),
                "idx": idx, "seg_no": seg_no,
                "route": next((r["metadata"]["route"] for r in by_idx[idx]), ""),
            })
    return episodes


class TravelEnv:
    """5 tools = 5 public methods; docstrings and signatures align verbatim with business_logic/tool_schemas.json."""

    # the train script injects reward weights when it builds the factory (config decides; no implicit defaults)
    W: dict = {}

    def __init__(self) -> None:
        self._w = world()
        self._calls: list[dict] = []
        self._golden: list[dict] = []

    # ---------------------------------------------------------------- tools --
    def search_travel_guide(self, location: str, search_mode: str = "hybrid") -> str:
        """搜索目的地旅行攻略。

        Args:
            location: 目的地城市名称，如'北京'、'上海'
            search_mode: 搜索模式：vector(向量搜索)、keyword(关键词搜索)、hybrid(混合搜索)
        """
        return self._run("search_travel_guide",
                         {"location": location, "search_mode": search_mode})

    def get_weather_info(self, location: str, start_date: str, num_days: int = 1) -> str:
        """查询指定城市指定日期的天气。

        Args:
            location: 城市名称，如'北京'、'上海'，或城市ID如'101010100'
            start_date: 开始日期，格式YYYY-MM-DD，如'2025-09-15'
            num_days: 查询天数，默认1天。根据旅行攻略中的行程天数确定
        """
        return self._run("get_weather_info",
                         {"location": location, "start_date": start_date, "num_days": num_days})

    def query_route(self, start_location: str, end_location: str, city: str) -> str:
        """查询两地之间的路线。

        Args:
            start_location: 起点坐标，如'116.481028,39.989643'
            end_location: 终点地址，如'颐和园'
            city: 所在城市名称，如'北京'、'嘉兴'。用于限定地点搜索范围
        """
        return self._run("query_route", {"start_location": start_location,
                                         "end_location": end_location, "city": city})

    def recommend_hotels(self, location: str, requirements: str = "") -> str:
        """推荐指定城市的酒店。

        Args:
            location: 城市名称，如'北京'、'嘉兴'
            requirements: 档次或类型偏好，如'五星级'、'经济型'、'青年旅舍'
        """
        return self._run("recommend_hotels",
                         {"location": location, "requirements": requirements})

    def get_hotel_reviews(self, hotel_name: str, location: str = "") -> str:
        """查询指定酒店的用户评价。

        Args:
            hotel_name: 酒店名称
            location: 酒店所在城市，可选，用于消歧
        """
        return self._run("get_hotel_reviews",
                         {"hotel_name": hotel_name, "location": location})

    # --------------------------------------------------------------- TRL API --
    def reset(self, **row) -> None:
        """Reset before each rollout: record this segment's golden calls and clear the call log."""
        self._calls = []
        self._golden = json.loads(row.get("golden") or "[]")
        return None

    def get_reward(self) -> float:
        """Segment-level trajectory reward (same standard as §6.2; weights from config).

                    R = w_call * LCS pairs / max(1, golden call count)   ordered pairing + strict argument equality
                      + w_struct * [all matched and no extras]           a flawless segment
                      - w_extra * extra call count
                    Zero-call golden: making no call scores full marks on the first two terms
                    (silence is the correct answer).
        """
        w = self.W
        required = len(self._golden)
        if not required:
            silent = not self._calls
            r = (w["w_call"] * (1.0 if silent else 0.0)
                 + w["w_struct"] * (1.0 if silent else 0.0)
                 - w["w_extra"] * len(self._calls))
            return max(w["clip_min"], min(w["clip_max"], r))
        made = [{"tool": c["tool"], "arguments": c["args"]} for c in self._calls]
        correct = len(_lcs_pairs(self._golden, made))
        extra = max(0, len(made) - required)
        r = (w["w_call"] * correct / required
             + w["w_struct"] * (1.0 if correct == required and extra == 0 else 0.0)
             - w["w_extra"] * extra)
        return max(w["clip_min"], min(w["clip_max"], r))

    # Every helper below is underscore-prefixed: TRL collects *every* public method of the
    # environment instance as a tool and renders it into the prompt, so a public helper would
    # invent an extra "tool" out of nowhere and break train == serve.
    def _calls_made(self) -> int:
        return len(self._calls)

    def _calls_required(self) -> int:
        return len(self._golden)

    def _can_still_pass(self) -> bool:
        """Whether the calls made so far can still earn a perfect score (the early-exit test
                used by the §5.4 probe).

                A perfect score means the call sequence matches golden position for position (LCS
                pairs == golden count, with no extras). So the moment an extra call appears, or a
                tool/argument at some position fails to match, this rollout can no longer be
                perfect and may be abandoned immediately rather than run to its terminal turn. The
                test shares its source with get_reward (the same _compare).
        """
        if len(self._calls) > len(self._golden):
            return False
        return all(m["tool"] == g["tool"] and _compare(g["args"], m["args"]) is None
                   for m, g in zip(self._calls, self._golden))

    # ---------------------------------------------------------------- internal --
    def _run(self, tool: str, args: dict) -> str:
        self._calls.append({"tool": tool, "args": args})
        return json.dumps(self._w.run(tool, args), ensure_ascii=False)
