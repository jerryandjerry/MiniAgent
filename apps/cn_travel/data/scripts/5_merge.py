#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge each Step 4 partition into an index-ordered JSON array."""
import json
from collections import Counter

from data.paths import PATHS

SPLIT = PATHS.root / "4_split"
OUT = PATHS.root / "5_merged"
SIDES = {"train_validation": 909, "leaderboard_eval": 101}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for side, want in SIDES.items():
        files = sorted((SPLIT / side).glob("[0-9]*.json"), key=lambda p: int(p.stem))
        assert len(files) == want, f"{side}: {len(files)} != {want}"
        arr = [json.loads(f.read_text(encoding="utf-8")) for f in files]
        # Every element must match its source value-for-value and carry metadata.
        for f, obj in zip(files, arr):
            assert obj == json.loads(f.read_text(encoding="utf-8"))
            assert "conversation" in obj and "metadata" in obj
            assert obj["metadata"]["idx"] == int(f.stem)
        (OUT / f"{side}.json").write_text(
            json.dumps(arr, ensure_ascii=False, indent=2), encoding="utf-8")
        rc = Counter(o["metadata"]["route"] for o in arr)
        print(f"{side}: merged {len(arr)} rows, {len(rc)} routes -> {OUT / (side + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
