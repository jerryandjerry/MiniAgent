"""Deterministic training-side tool world for per-user-turn GRPO."""
from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from pathlib import Path

from .paths import DATA_ROOT, install_import_paths
from .tool_matching import WORLD_ARGS, normalize_tool_args, project_args


_STEP2 = DATA_ROOT / "scripts" / "2_materialize.py"
_CACHE = DATA_ROOT / "2_materialized" / "exec_cache"


def _load_step2():
    install_import_paths()
    spec = importlib.util.spec_from_file_location("cn_travel_step2", _STEP2)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load materializer: {_STEP2}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._REG_FOR_SYNTH.update(module.load_registry())
    return module


class FrozenWorld:
    def __init__(self, sealed_set: Path, on_conflict: str = "assert") -> None:
        self._materializer = _load_step2()
        self.fallbacks: list[dict] = []
        self.sealed_conflicts = 0
        self._sealed: dict[str, dict] = {}
        for row in json.loads(sealed_set.read_text(encoding="utf-8")):
            conversation = row["conversation"]
            results = {
                message["tool_call_id"]: json.loads(message["content"])
                for message in conversation if message.get("role") == "tool"
            }
            for message in conversation:
                if message.get("role") != "assistant":
                    continue
                for call in message.get("tool_calls") or []:
                    if call["id"] not in results:
                        continue
                    key = self._key(
                        call["function"]["name"],
                        json.loads(call["function"]["arguments"]),
                    )
                    previous = self._sealed.get(key)
                    if previous is not None:
                        if previous == results[call["id"]]:
                            continue
                        if on_conflict == "first":
                            self.sealed_conflicts += 1
                            continue
                        raise AssertionError(
                            f"sealed evidence contradicts itself: {key} has two results"
                        )
                    self._sealed[key] = results[call["id"]]

        self._exact: dict[str, dict] = {}
        self._hotels_by_location: dict[str, list[tuple[str, dict]]] = defaultdict(list)
        for path in sorted(_CACHE.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            name, arguments, result = (
                record["name"], record["arguments"], record["result"]
            )
            key = self._key(name, arguments)
            self._exact[key] = result
            if name == "recommend_hotels":
                normalized = normalize_tool_args(name, arguments)
                location = str(normalized.get("location", ""))
                self._hotels_by_location[location].append((key, result))
        for location in self._hotels_by_location:
            self._hotels_by_location[location].sort(key=lambda item: item[0])

    @staticmethod
    def _key(tool: str, arguments: dict) -> str:
        return json.dumps(
            {"tool": tool, "args": normalize_tool_args(tool, arguments)},
            ensure_ascii=False,
            sort_keys=True,
        )

    def _log(self, tool: str, arguments, kind: str) -> None:
        self.fallbacks.append({"tool": tool, "args": arguments, "fallback": kind})

    def run(self, tool: str, arguments: dict) -> dict:
        install_import_paths()
        from cn_travel.business_logic.contracts import err

        if tool not in WORLD_ARGS:
            self._log(tool, arguments, "unknown_tool")
            return {"status": "error", "source": f"unknown tool {tool!r}"}
        if isinstance(arguments, dict):
            hit = self._sealed.get(self._key(tool, arguments))
            if hit is not None:
                return hit
        projected = project_args(tool, arguments)
        if projected is None:
            self._log(tool, arguments, "invalid_args")
            detail = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
            return err(tool, f"invalid arguments for {tool}: {detail}")

        if tool != "search_travel_guide" and isinstance(arguments, dict):
            hit = self._exact.get(self._key(tool, arguments))
            if hit is not None:
                self._log(tool, arguments, "exec_cache")
                return hit

        if tool == "search_travel_guide":
            self._log(tool, arguments, "guide_generator")
            return self._materializer._syn_guide(projected)
        if tool == "get_weather_info":
            self._log(tool, arguments, "weather_generator")
            return self._materializer._syn_weather(projected)
        if tool == "query_route":
            self._log(tool, arguments, "route_generator")
            return self._materializer._syn_route(projected)
        if tool == "recommend_hotels":
            candidates = self._hotels_by_location.get(projected["location"], [])
            if candidates:
                self._log(tool, arguments, "hotels_by_location")
                return candidates[0][1]
            self._log(tool, arguments, "hotels_error_shape")
            if projected["location"] not in self._materializer._REG_FOR_SYNTH:
                return err(
                    tool,
                    f"could not resolve location {projected['location']!r}",
                )
            return err(tool, f"no hotels found for {projected['location']!r}")

        self._log(tool, arguments, "reviews_empty_shape")
        return {
            "status": "empty",
            "source": "synthetic:none",
            "hotel_name": projected["hotel_name"],
            "rating": None,
            "price_hint": None,
            "reviews": [],
            "summary": "未检索到该酒店的公开评价",
        }
