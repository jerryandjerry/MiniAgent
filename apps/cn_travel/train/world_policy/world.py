"""Immutable, local Synthetic China snapshot used by world-policy rollouts."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from cn_travel.business_logic import contracts


class WorldSnapshotError(ValueError):
    """Raised when a world snapshot is malformed or internally inconsistent."""


class InvalidToolArguments(ValueError):
    pass


def normalize_json(value: Any) -> Any:
    """Compiler/runtime frozen-key normalization: recursive NFC, no coercion.

    Object ordering is handled by :func:`canonical_json`; values and JSON types
    are otherwise preserved exactly.  In particular this function does not add
    optional defaults, trim strings, coerce numbers, or remove a trailing 市.
    """

    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [normalize_json(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_json(item) for item in value]
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise TypeError(f"JSON object key is not a string: {raw_key!r}")
            key = unicodedata.normalize("NFC", raw_key)
            if key in normalized:
                raise ValueError(f"keys collide after NFC normalization: {key!r}")
            normalized[key] = normalize_json(item)
        return normalized
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise TypeError(f"not a JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(normalize_json(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def world_key(tool: str, arguments: Mapping[str, Any]) -> str:
    return canonical_json({"tool": tool, "arguments": dict(arguments)})


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _validate_sealed_snapshot(snapshot: Mapping[str, Any]) -> None:
    """Fail closed on every binding emitted by Step 7.

    The content digest protects the complete payload, while the per-entry
    digests make errors local and ensure callers cannot accidentally reproduce
    the outer digest with a differently interpreted key/result contract.
    """

    expected_schema = "cn_travel.synthetic_china_world.v1"
    expected_version = "synthetic-china-world-v1"
    expected_normalization = {
        "algorithm": "json-nfc-sort-keys-v1",
        "string_normalization": "NFC",
        "object_key_order": "lexicographic",
        "coercions": "none",
    }
    if snapshot.get("schema_version") != expected_schema:
        raise WorldSnapshotError(
            f"world schema {snapshot.get('schema_version')!r} != {expected_schema!r}"
        )
    if snapshot.get("world_version") != expected_version:
        raise WorldSnapshotError(
            f"world version {snapshot.get('world_version')!r} != {expected_version!r}"
        )
    if snapshot.get("normalization") != expected_normalization:
        raise WorldSnapshotError("world normalization contract is not Step-7 v1")
    entries = snapshot.get("entries")
    conflicts = snapshot.get("conflicts")
    if not isinstance(entries, list) or not isinstance(conflicts, list):
        raise WorldSnapshotError("sealed world entries/conflicts must be lists")
    if snapshot.get("entry_count") != len(entries):
        raise WorldSnapshotError("world entry_count does not match entries")
    if snapshot.get("conflict_count") != len(conflicts):
        raise WorldSnapshotError("world conflict_count does not match conflicts")

    seen: dict[str, Mapping[str, Any]] = {}
    ordered_keys: list[str] = []
    for ordinal, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise WorldSnapshotError(f"world entry {ordinal} is not an object")
        tool, args, result = entry.get("tool"), entry.get("arguments"), entry.get("result")
        if not isinstance(tool, str) or not isinstance(args, Mapping) \
                or not isinstance(result, Mapping):
            raise WorldSnapshotError(f"world entry {ordinal} lacks tool/arguments/result")
        key = world_key(tool, args)
        key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
        result_hash = _sha256_json(result)
        if entry.get("key_sha256") != key_hash:
            raise WorldSnapshotError(f"world entry {ordinal} key_sha256 mismatch")
        if entry.get("execution_id") != f"sha256:{key_hash}":
            raise WorldSnapshotError(f"world entry {ordinal} execution_id mismatch")
        if entry.get("result_sha256") != result_hash:
            raise WorldSnapshotError(f"world entry {ordinal} result_sha256 mismatch")
        if key in seen:
            raise WorldSnapshotError(f"duplicate normalized world key at entry {ordinal}")
        seen[key] = entry
        ordered_keys.append(key)
    if ordered_keys != sorted(ordered_keys):
        raise WorldSnapshotError("world entries are not sorted by canonical key")

    for ordinal, conflict in enumerate(conflicts):
        if not isinstance(conflict, Mapping):
            raise WorldSnapshotError(f"world conflict {ordinal} is not an object")
        tool, args = conflict.get("tool"), conflict.get("arguments")
        candidates = conflict.get("candidate_results")
        if not isinstance(tool, str) or not isinstance(args, Mapping) \
                or not isinstance(candidates, list) or len(candidates) < 2:
            raise WorldSnapshotError(f"world conflict {ordinal} is malformed")
        key = world_key(tool, args)
        key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
        if conflict.get("key_sha256") != key_hash or key not in seen:
            raise WorldSnapshotError(f"world conflict {ordinal} key binding mismatch")
        candidate_hashes: list[str] = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or not isinstance(candidate.get("result"), Mapping):
                raise WorldSnapshotError(f"world conflict {ordinal} candidate is malformed")
            result_hash = _sha256_json(candidate["result"])
            if candidate.get("result_sha256") != result_hash:
                raise WorldSnapshotError(
                    f"world conflict {ordinal} candidate result_sha256 mismatch"
                )
            candidate_hashes.append(result_hash)
        resolution = conflict.get("resolution")
        selected = min(candidate_hashes)
        if not isinstance(resolution, Mapping) \
                or resolution.get("rule") != "minimum_canonical_result_sha256" \
                or resolution.get("selected_result_sha256") != selected:
            raise WorldSnapshotError(f"world conflict {ordinal} resolution mismatch")
        if seen[key].get("result_sha256") != selected:
            raise WorldSnapshotError(f"world conflict {ordinal} selected entry mismatch")

    content_hash = snapshot.get("content_sha256")
    payload = {key: value for key, value in snapshot.items()
               if key not in {"content_sha256", "world_id"}}
    expected_content_hash = _sha256_json(payload)
    if content_hash != expected_content_hash:
        raise WorldSnapshotError("world content_sha256 mismatch")
    if snapshot.get("world_id") != f"{expected_version}:{expected_content_hash}":
        raise WorldSnapshotError("world_id is not bound to content_sha256")


@dataclass(frozen=True)
class WorldExecution:
    tool: str
    arguments: Mapping[str, Any]
    result: Mapping[str, Any]
    key: str
    frozen: bool


_ARG_SPEC: dict[str, tuple[dict[str, type], dict[str, tuple[type, Any]]]] = {
    "search_travel_guide": (
        {"location": str},
        {"search_mode": (str, "hybrid")},
    ),
    "get_weather_info": (
        {"location": str, "start_date": str},
        {"num_days": (int, 1)},
    ),
    "query_route": (
        {"start_location": str, "end_location": str, "city": str},
        {},
    ),
    "recommend_hotels": (
        {"location": str},
        {"requirements": (str, "")},
    ),
    "get_hotel_reviews": (
        {"hotel_name": str},
        {"location": (str, "")},
    ),
}


def _is_json_type(value: Any, expected: type) -> bool:
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, expected)


def validate_and_fill_arguments(tool: str, arguments: Any) -> dict[str, Any]:
    """Validate against the public five-tool schema and fill only defaults.

    This projection is used exclusively for deterministic fallback generation.
    Frozen lookup always happens first with the uncoerced compiler key.
    """

    if tool not in _ARG_SPEC:
        raise InvalidToolArguments(f"unknown tool {tool!r}")
    if not isinstance(arguments, Mapping):
        raise InvalidToolArguments("arguments must be an object")
    required, optional = _ARG_SPEC[tool]
    allowed = set(required) | set(optional)
    extras = sorted(set(arguments) - allowed)
    if extras:
        raise InvalidToolArguments(f"unexpected arguments: {extras}")
    out: dict[str, Any] = {}
    for key, expected in required.items():
        if key not in arguments or not _is_json_type(arguments[key], expected):
            raise InvalidToolArguments(f"{key} must be {expected.__name__}")
        if expected is str and not arguments[key]:
            raise InvalidToolArguments(f"{key} must not be empty")
        out[key] = arguments[key]
    for key, (expected, default) in optional.items():
        value = arguments.get(key, default)
        if not _is_json_type(value, expected):
            raise InvalidToolArguments(f"{key} must be {expected.__name__}")
        out[key] = value
    if tool == "search_travel_guide" and out["search_mode"] not in {
        "vector", "keyword", "hybrid"
    }:
        raise InvalidToolArguments("search_mode is outside its enum")
    if tool == "get_weather_info":
        try:
            date.fromisoformat(out["start_date"])
        except ValueError as exc:
            raise InvalidToolArguments("start_date must be YYYY-MM-DD") from exc
        if not 1 <= out["num_days"] <= 30:
            raise InvalidToolArguments("num_days must be between 1 and 30")
    return normalize_json(out)


def _entry_records(snapshot: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    """Accept the v1 list form and a few unambiguous legacy map forms."""

    root: Any = snapshot.get("entries")
    if root is None:
        root = snapshot.get("tool_results", snapshot.get("results"))
    if isinstance(root, list):
        yield from root
        return
    if isinstance(root, Mapping):
        for key, value in root.items():
            if isinstance(value, Mapping) and (
                "result" in value and ("tool" in value or "name" in value)
            ):
                yield value
            elif isinstance(value, list):
                for item in value:
                    if not isinstance(item, Mapping):
                        raise WorldSnapshotError(f"invalid world entry under {key!r}")
                    if "tool" not in item and "name" not in item:
                        item = {**item, "tool": key}
                    yield item
            else:
                raise WorldSnapshotError(f"ambiguous world result map entry {key!r}")
        return
    raise WorldSnapshotError("world snapshot must contain an entries list")


class SyntheticChinaWorld:
    """One immutable mapping plus pure, seeded fallbacks.

    No method in this class performs network I/O, calls an LLM, or mutates world
    facts.  ``execute`` returns defensive copies so independent rollouts cannot
    affect one another through a shared snapshot.
    """

    def __init__(self, snapshot: Mapping[str, Any]) -> None:
        self.schema_version = str(snapshot.get("schema_version") or
                                  "cn_travel.synthetic_china.v1")
        self.world_id = str(snapshot.get("world_id") or snapshot.get("id") or "")
        entries: dict[str, tuple[str, dict[str, Any], dict[str, Any]]] = {}
        for ordinal, raw in enumerate(_entry_records(snapshot)):
            tool = raw.get("tool", raw.get("name"))
            args = raw.get("arguments", raw.get("args"))
            result = raw.get("result")
            if not isinstance(tool, str) or not isinstance(args, Mapping) \
                    or not isinstance(result, Mapping):
                raise WorldSnapshotError(f"entry {ordinal} lacks tool/arguments/result")
            if tool not in _ARG_SPEC:
                raise WorldSnapshotError(f"entry {ordinal}: unknown tool {tool!r}")
            normalized_args = normalize_json(dict(args))
            normalized_result = normalize_json(dict(result))
            try:
                contracts.validate(tool, normalized_result)
            except (AssertionError, KeyError, TypeError) as exc:
                raise WorldSnapshotError(f"entry {ordinal}: invalid {tool} result: {exc}") from exc
            key = world_key(tool, normalized_args)
            previous = entries.get(key)
            current = (tool, normalized_args, normalized_result)
            if previous is not None and previous[2] != normalized_result:
                raise WorldSnapshotError(
                    f"one normalized tool key maps to different results: {key}"
                )
            entries[key] = current
        self._entries = entries
        if not self.world_id:
            digest = hashlib.sha256(canonical_json([
                {"tool": tool, "arguments": args, "result": result}
                for _, (tool, args, result) in sorted(entries.items())
            ]).encode("utf-8")).hexdigest()
            self.world_id = f"synthetic-china-world-v1:{digest}"

    @classmethod
    def load(cls, path: str | Path) -> "SyntheticChinaWorld":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise WorldSnapshotError("world snapshot root must be an object")
        _validate_sealed_snapshot(data)
        return cls(data)

    def __len__(self) -> int:
        return len(self._entries)

    def execute(self, tool: str, arguments: Any) -> WorldExecution:
        raw_args = dict(arguments) if isinstance(arguments, Mapping) else arguments
        if isinstance(raw_args, Mapping):
            try:
                key = world_key(tool, raw_args)
            except (TypeError, ValueError):
                key = canonical_json({"tool": str(tool), "arguments": str(raw_args)})
            hit = self._entries.get(key)
            if hit is not None:
                return WorldExecution(tool, copy.deepcopy(hit[1]), copy.deepcopy(hit[2]),
                                      key, True)
        else:
            key = canonical_json({"tool": str(tool), "arguments": str(raw_args)})

        if tool not in _ARG_SPEC:
            result = {"status": "error", "source": "synthetic:unknown-tool",
                      "message": f"unknown tool {tool!r}"}
            return WorldExecution(str(tool), {}, result, key, False)
        try:
            args = validate_and_fill_arguments(tool, arguments)
        except InvalidToolArguments as exc:
            result = contracts.err(tool, f"synthetic:invalid-arguments:{exc}")
            contracts.validate(tool, result)
            return WorldExecution(tool, normalize_json(raw_args) if isinstance(raw_args, Mapping)
                                  else {}, result, key, False)
        fallback_key = world_key(tool, args)
        result = self._fallback(tool, args, fallback_key)
        contracts.validate(tool, result)
        return WorldExecution(tool, copy.deepcopy(args), result, fallback_key, False)

    def run(self, tool: str, arguments: Any) -> dict[str, Any]:
        return dict(self.execute(tool, arguments).result)

    @staticmethod
    def _fallback(tool: str, args: Mapping[str, Any], key: str) -> dict[str, Any]:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        source = f"synthetic:world-v1:{hashlib.sha256(key.encode()).hexdigest()[:16]}"

        if tool == "search_travel_guide":
            location = str(args["location"])
            return {
                "status": "ok",
                "source": source,
                "location": location,
                "guides": [{
                    "city": location,
                    "province": "合成省",
                    "content": f"{location}合成旅行指南：城市公园、历史街区和地方餐饮。",
                    "score": round(0.75 + (seed % 21) / 100, 2),
                }],
            }
        if tool == "get_weather_info":
            location = str(args["location"])
            start = date.fromisoformat(str(args["start_date"]))
            weather = ("晴", "多云", "小雨", "阴")[seed % 4]
            lo = 8 + seed % 18
            days = [{
                "date": (start + timedelta(days=i)).isoformat(),
                "day_weather": weather,
                "night_weather": ("晴", "多云", "阴")[(seed + i) % 3],
                "temp_min_c": int(lo + i % 2),
                "temp_max_c": int(lo + 6 + (seed + i) % 5),
            } for i in range(int(args["num_days"]))]
            lat = round(18 + (seed % 3500000) / 100000, 6)
            lng = round(73 + ((seed // 17) % 6200000) / 100000, 6)
            return {
                "status": "ok",
                "source": source,
                "location": location,
                "resolved": {
                    "name": location,
                    "adcode": f"{seed % 1_000_000:06d}",
                    "lat": lat,
                    "lng": lng,
                },
                "days": days,
            }
        if tool == "query_route":
            distance = 800 + seed % 40_000
            duration = max(2, math.ceil(distance / 450))
            lat = round(18 + (seed % 3500000) / 100000, 6)
            lng = round(73 + ((seed // 19) % 6200000) / 100000, 6)
            end_lat = round(lat + ((seed % 101) - 50) / 1000, 6)
            end_lng = round(lng + (((seed // 101) % 101) - 50) / 1000, 6)
            return {
                "status": "ok",
                "source": source,
                "origin": {"query": str(args["start_location"]), "lat": lat, "lng": lng},
                "destination": {
                    "query": str(args["end_location"]),
                    "name": str(args["end_location"]),
                    "lat": end_lat,
                    "lng": end_lng,
                },
                "routes": {
                    "walking": {"duration_min": max(1, math.ceil(distance / 80)),
                                "distance_m": distance},
                    "transit": {"duration_min": duration, "fare_cny": 2.0,
                                "segments": ["合成公交线路"]},
                    "driving": {"duration_min": max(1, math.ceil(distance / 500)),
                                "distance_m": distance, "tolls_cny": 0.0},
                },
            }
        if tool == "recommend_hotels":
            location = str(args["location"])
            hotels = []
            for i in range(3):
                hotels.append({
                    "name": f"{location}合成酒店{chr(ord('A') + i)}",
                    "address": f"{location}合成大道{1 + (seed + i) % 99}号",
                    "district": "中心区",
                    "rating": round(4.0 + ((seed + i) % 9) / 10, 1),
                    "tier": str(args["requirements"] or "舒适型"),
                    "tel": f"000-{(seed + i) % 10_000:04d}",
                    "lat": round(18 + ((seed + i) % 3500000) / 100000, 6),
                    "lng": round(73 + (((seed // 17) + i) % 6200000) / 100000, 6),
                    "price_cny": None,
                })
            return {"status": "ok", "source": source, "location": location,
                    "hotels": hotels}
        hotel = str(args["hotel_name"])
        rating = round(4.0 + (seed % 9) / 10, 1)
        return {
            "status": "ok",
            "source": source,
            "hotel_name": hotel,
            "rating": rating,
            "price_hint": None,
            "reviews": [{"text": f"住客认为{hotel}整洁，服务稳定。"}],
            "summary": f"{hotel}的合成口碑评分为{rating}。",
        }


# The versioned public name used by the specification.
SyntheticChinaWorldV1 = SyntheticChinaWorld


__all__ = [
    "InvalidToolArguments",
    "SyntheticChinaWorld",
    "SyntheticChinaWorldV1",
    "WorldExecution",
    "WorldSnapshotError",
    "canonical_json",
    "normalize_json",
    "validate_and_fill_arguments",
    "world_key",
]
