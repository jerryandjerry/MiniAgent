#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Expand training conversations into cumulative Step 6 assistant targets.

Tool-free conversations remain single samples. Tool conversations produce one
cumulative target per assistant turn and three copies of the final target. The
leaderboard partition remains byte-identical to Step 5.
"""
import copy
import json

from data.paths import PATHS

MERGED = PATHS.root / "5_merged"
OUT = PATHS.root / "6_multiturn"


def expand_one(obj: dict) -> list[dict]:
    """Expand one labeled conversation into cumulative targets."""
    conv = obj["conversation"]
    meta = obj["metadata"]
    if not any(m.get("role") == "tool" for m in conv):
        return [{"conversation": conv, "metadata": meta}]           # Preserve tool-free conversations.
    a_pos = [i for i, m in enumerate(conv) if m.get("role") == "assistant"]
    out = []
    for k, end in enumerate(a_pos):
        sample = {"conversation": conv[:end + 1], "metadata": meta}
        out.append(sample)
        if k == len(a_pos) - 1:                                     # Duplicate the final round twice.
            out.append(copy.deepcopy(sample))
            out.append(copy.deepcopy(sample))
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    train = json.loads((MERGED / "train_validation.json").read_text(encoding="utf-8"))
    expanded = []
    for obj in train:
        expanded.extend(expand_one(obj))

    # Validate that every derived sample has metadata from a training parent.
    parent_meta = {json.dumps(o["metadata"], sort_keys=True, ensure_ascii=False) for o in train}
    for s in expanded:
        assert "conversation" in s and "metadata" in s, "派生样本缺字段"
        assert json.dumps(s["metadata"], sort_keys=True, ensure_ascii=False) in parent_meta, \
            "派生样本的 metadata 不是训练父对话的标签"
        # Each sample ends with assistant, the function-call SFT target, unless tool-free.
        assert s["conversation"][-1]["role"] == "assistant", "样本未以 assistant 结尾"
    (OUT / "train_validation.json").write_text(
        json.dumps(expanded, ensure_ascii=False, indent=2), encoding="utf-8")

    # Keep the leaderboard partition byte-identical.
    lb_bytes = (MERGED / "leaderboard_eval.json").read_bytes()
    (OUT / "leaderboard_eval.json").write_bytes(lb_bytes)
    assert (OUT / "leaderboard_eval.json").read_bytes() == lb_bytes, "leaderboard 未逐字节保留"

    mult = len(expanded) / len(train)
    print(f"Step 6: train {len(train)} -> {len(expanded)} multi-turn samples ({mult:.2f}x); "
          f"leaderboard_eval {len(json.loads(lb_bytes))} rows untouched (byte-identical)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
