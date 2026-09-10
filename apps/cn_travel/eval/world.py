#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic tool world for the sealed §6.2 evaluation.

Canonical tool-and-argument matches return the sealed conversation result. Other
valid calls use the sealed execution cache, deterministic local generators, or the
documented deterministic fallback for that tool. Every non-conversation resolution
is recorded in ``fallbacks``. Invalid arguments return the tool contract's error shape.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
from collections import defaultdict

try:
    from .paths import PATHS
except ImportError:  # Direct ``python eval/world.py`` invocation.
    from paths import PATHS

_STEP2 = PATHS.materializer
_CACHE = PATHS.execution_cache

_REG_KEYS = frozenset(json.loads(
    (PATHS.data / "city_registry.json").read_text(encoding="utf-8"))["cities"])
_LOCATION_FIELDS = frozenset({"location", "city", "start_location", "end_location"})

# The sealed generator has seven requirement classes. Natural language may spell the
# same class in a few closed forms; arbitrary text, negation and fuzzy matches remain
# distinct. Each pattern is a full match, so a phrase merely containing one of these
# strings is not accepted as equivalent.
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
    """Map only approved, whole-string hotel requirement aliases to sealed forms."""
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
    """Return the shared scorer/world canonical form without adding or dropping keys."""
    out = {}
    for key, value in args.items():
        if (key in _LOCATION_FIELDS and isinstance(value, str)
                and value.endswith("市") and value[:-1] in _REG_KEYS):
            value = value[:-1]
        if tool == "recommend_hotels" and key == "requirements":
            value = canonicalize_requirements(value)
        out[key] = value
    return out

# Generator argument projections, including deterministic coercions. Scoring uses
# the shared canonicalizer followed by strict whole-object equality.
WORLD_ARGS = {
    "search_travel_guide": {"location": str},
    "get_weather_info": {"location": str, "start_date": str, "num_days": int},
    "query_route": {"start_location": str, "end_location": str, "city": "strip市"},
    "recommend_hotels": {"location": str, "requirements": "optional"},
    "get_hotel_reviews": {"hotel_name": str, "location": "optional"},
}


def _load_step2():
    spec = importlib.util.spec_from_file_location("cn_travel_step2", _STEP2)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._REG_FOR_SYNTH.update(mod.load_registry())
    return mod


def project_args(tool: str, args: dict) -> dict | None:
    """The argument projection the world reads; returns None on a missing required key or an illegal type (the world then rejects it)."""
    if not isinstance(args, dict):
        return None
    spec = WORLD_ARGS.get(tool)
    if spec is None:
        return None
    args = normalize_tool_args(tool, args)
    out = {}
    for key, kind in spec.items():
        v = args.get(key)
        if kind == "optional":                       # generator: a.get(key, "")
            out[key] = str(v) if v is not None else ""
            continue
        if v is None:
            return None
        if kind is int:                              # generator: int(a["num_days"])
            try:
                out[key] = int(v)
            except (TypeError, ValueError):
                return None
        elif kind == "strip市":                      # Generator strips the trailing city suffix
            if not isinstance(v, str):
                return None
            out[key] = v.rstrip("市")
        else:
            if not isinstance(v, str):
                return None
            out[key] = v
    return out


_EVAL_SET = PATHS.eval_set


class FrozenWorld:
    def __init__(self, sealed_set: pathlib.Path | None = None,
                 on_conflict: str = "assert") -> None:
        """sealed_set defaults to the eval set (the §6.2 channel). The training side (§5.3)
                passes train_validation.json so the first-authority layer holds training evidence
                only - eval evidence is never loaded during training.

                on_conflict: what to do when one (tool, args) has two different sealed results.
                  "assert" - the eval standard: evidence must be self-consistent, and inconsistency
                             aborts (§6.2 invariant);
                  "first"  - the training standard: take the first occurrence (row order is fixed, so
                             the world stays byte-reproducible). The training corpus genuinely contains
                             such conflicts: hotel reviews were generated per conversation by the teacher
                             at materialization time, so the same (hotel, city) picked up different review
                             text in different conversations. The reward judges calls, not results, so
                             taking either one does not affect scoring - it only has to be deterministic.
        """
        self._m = _load_step2()
        self.fallbacks: list[dict] = []
        self.sealed_conflicts = 0
        # First authority: canonical (tool, args) from a sealed conversation's tool
        # trace -> the sealed result.
        self._sealed: dict[str, dict] = {}
        for row in json.loads((sealed_set or _EVAL_SET).read_text(encoding="utf-8")):
            conv = row["conversation"]
            results = {m["tool_call_id"]: json.loads(m["content"])
                       for m in conv if m.get("role") == "tool"}
            for msg in conv:
                if msg.get("role") != "assistant":
                    continue
                for tc in msg.get("tool_calls") or []:
                    if tc["id"] not in results:
                        continue          # a slice cut at this assistant turn: the result is not on this row
                    key = self._key(tc["function"]["name"],
                                    json.loads(tc["function"]["arguments"]))
                    prev = self._sealed.get(key)
                    if prev is not None:
                        if prev == results[tc["id"]]:
                            continue
                        if on_conflict == "first":
                            self.sealed_conflicts += 1
                            continue
                        raise AssertionError(f"sealed evidence contradicts itself: {key} has two different results")
                    self._sealed[key] = results[tc["id"]]
        # Second layer: an index over the sealed exec_cache (guides excluded - stale);
        # hotels additionally get a deterministic per-location fallback index (the one approximate
        # fallback §6.2 explicitly permits).
        self._exact: dict[str, dict] = {}
        self._hotels_by_loc: dict[str, list[tuple[str, dict]]] = defaultdict(list)
        for f in sorted(_CACHE.glob("*.json")):
            rec = json.loads(f.read_text(encoding="utf-8"))
            name, args, result = rec["name"], rec["arguments"], rec["result"]
            key = self._key(name, args)
            self._exact[key] = result
            if name == "recommend_hotels":
                norm_args = normalize_tool_args(name, args)
                self._hotels_by_loc[str(norm_args.get("location", ""))].append((key, result))
        for k in self._hotels_by_loc:
            self._hotels_by_loc[k].sort(key=lambda kv: kv[0])   # deterministic: sort by cache key and take the first

    @staticmethod
    def _key(tool: str, args: dict) -> str:
        return json.dumps({"tool": tool, "args": normalize_tool_args(tool, args)},
                          ensure_ascii=False, sort_keys=True)

    def _log(self, tool: str, args, kind: str) -> None:
        self.fallbacks.append({"tool": tool, "args": args, "fallback": kind})

    def run(self, tool: str, args: dict) -> dict:
        from cn_travel.business_logic.contracts import err

        if tool not in WORLD_ARGS:
            self._log(tool, args, "unknown_tool")     # a tool outside the contract: no contract to follow, so return a deterministic error
            return {"status": "error", "source": f"unknown tool {tool!r}"}
        if isinstance(args, dict):                    # first authority: canonical hit on sealed evidence
            hit = self._sealed.get(self._key(tool, args))
            if hit is not None:
                return hit
        proj = project_args(tool, args)
        if proj is None:
            self._log(tool, args, "invalid_args")
            return err(tool, f"invalid arguments for {tool}: {json.dumps(args, ensure_ascii=False, sort_keys=True)}")

        # Guide calls use the local generator. Other tools consult the sealed cache,
        # then use their generator or deterministic fallback. Log every such resolution.
        if tool != "search_travel_guide" and isinstance(args, dict):
            hit = self._exact.get(self._key(tool, args))
            if hit is not None:
                self._log(tool, args, "exec_cache")
                return hit

        if tool == "search_travel_guide":
            self._log(tool, args, "guide_generator")
            return self._m._syn_guide(proj)
        if tool == "get_weather_info":
            self._log(tool, args, "weather_generator")
            return self._m._syn_weather(proj)
        if tool == "query_route":
            self._log(tool, args, "route_generator")
            return self._m._syn_route(proj)      # Generator strips the city suffix; projection is equivalent
        if tool == "recommend_hotels":
            cands = self._hotels_by_loc.get(proj["location"], [])
            if cands:
                self._log(tool, args, "hotels_by_location")
                return cands[0][1]
            self._log(tool, args, "hotels_error_shape")
            if proj["location"] not in self._m._REG_FOR_SYNTH:
                return err(tool, f"could not resolve location {proj['location']!r}")
            return err(tool, f"no hotels found for {proj['location']!r}")
        # Review calls without an exact hit return the deterministic empty shape.
        self._log(tool, args, "reviews_empty_shape")
        return {"status": "empty", "source": "synthetic:none", "hotel_name": proj["hotel_name"],
                "rating": None, "price_hint": None, "reviews": [],
                "summary": "未检索到该酒店的公开评价"}
