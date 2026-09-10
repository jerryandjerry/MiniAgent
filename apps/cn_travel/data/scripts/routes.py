#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compile the first Mermaid flowchart into a deterministic route table.

Node shapes encode terminal, decision, follow-up, and tool-call states. Route
enumeration tracks edges so one follow-up cycle can be traversed once.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

# A -->|"e1 label"| B, or A --> B
_EDGE = re.compile(r'^\s*(.+?)\s*-->\s*(?:\|\s*"?(.*?)"?\s*\|\s*)?(.+?)\s*$')
_SHAPE = re.compile(r'^(\w+)\s*(\(\[|\{|\[/|\[)')


class Node:
    def __init__(self, name):
        self.name = name
        self.kind = ""        # end | decision | ask | tool
        self.text = ""        # Node text; tool names are read from here.
        self.next = []


class Solution:
    def __init__(self):
        self.track = []
        self.routes = []
        self.visited = set()

    def backtrack(self, start):
        self.track.append(start)
        if not start.next:                    # A node without outgoing edges is terminal.
            self.routes.append(list(self.track))
        for node in start.next:
            edge = (start, node)              # Track edges, not nodes.
            if edge in self.visited:
                continue
            self.visited.add(edge)
            self.backtrack(node)
            self.visited.discard(edge)        # Paired with add so the edge is reusable after leaving a cycle.
        self.track.pop()                      # Paired with append.


def _decl(decl: str) -> tuple[str, str, str]:
    """Parse a node declaration into its identifier, kind, and text."""
    m = _SHAPE.match(decl)
    if not m:
        return decl.strip(), "", ""
    kind = {"([": "end", "{": "decision", "[/": "ask", "[": "tool"}[m.group(2)]
    body = re.search(r'(?:\(\[|\{|\[/|\[)\s*"?(.*?)"?\s*(?:\]\)|\}|/\]|\])$', decl)
    return m.group(1), kind, body.group(1) if body else ""


def parse(md: str) -> tuple[dict, dict]:
    """Build node and edge-label maps from the first Mermaid block."""
    block = re.search(r"```mermaid\s*(.*?)```", md, re.S)
    if not block:
        sys.exit("找不到 ```mermaid 代码块")

    nodes: dict[str, Node] = {}
    eid: dict[tuple[str, str], str] = {}

    def get(decl: str) -> Node:
        nid, kind, text = _decl(decl)
        n = nodes.setdefault(nid, Node(nid))
        if kind and not n.kind:            # The first declared shape wins.
            n.kind, n.text = kind, text
        return n

    for line in block.group(1).splitlines():
        if "-->" not in line:
            continue
        m = _EDGE.match(line)
        if not m:
            continue
        src, label, dst = get(m.group(1)), m.group(2) or "", get(m.group(3))
        name = (label.split() or [""])[0]
        name = name if name.startswith("e") else ""
        # Two edges between the same nodes collide because edge names use
        # (source, target) as the key. The second would silently overwrite the
        # first and duplicate route rows. Reject this; equivalent choices belong on one edge.
        if (src.name, dst.name) in eid:
            sys.exit(f"{src.name} → {dst.name} 之间有两条边"
                     f"（{eid[(src.name, dst.name)] or '未命名'} 和 {name or '未命名'}）。"
                     f"两个选项如果不改变轮数和工具调用，就合成一条边；"
                     f"如果会改变，让它们指向不同的节点。")
        src.next.append(dst)
        eid[(src.name, dst.name)] = name

    return nodes, eid


def main() -> int:
    ap = argparse.ArgumentParser(description="mermaid 流程图 → 路线表")
    ap.add_argument("doc", help="含 ```mermaid 块的 Markdown 文件")
    ap.add_argument("--tag", default="R", help="路线标签前缀，比如 W1")
    a = ap.parse_args()

    md = pathlib.Path(a.doc).read_text(encoding="utf-8")
    nodes, eid = parse(md)

    targets = {d for _, d in eid}
    starts = [n for n in nodes.values() if n.name not in targets]
    if len(starts) != 1:
        sys.exit(f"起点必须唯一，实际找到 {[n.name for n in starts] or '无（图里全是环）'}")

    s = Solution()
    s.backtrack(starts[0])
    if not s.routes:
        sys.exit("没有从起点到终态的路径")

    routes = [{"nodes": r, "edges": [e for e in
                                     (eid[(x.name, y.name)] for x, y in zip(r, r[1:])) if e]}
              for r in s.routes]

    # Sort order defines labels, so it must be deterministic and independent of enumeration.
    # Order by fewer follow-ups, queried slot, more tool calls, longer path, then edge order.
    # The main path, with no follow-up and the complete tool chain, is always first.
    #
    # Do not express this as success before empty. Success only happens to be
    # longer with one tool chain; two empty branches can have equal lengths.
    # Compare edge order numerically, because lexical order puts "e10" before "e9".
    def num(e):
        return int(e[1:]) if e[1:].isdigit() else 0

    def key(r):
        asks = [num(e) for e, n in zip(r["edges"], r["nodes"][1:]) if n.kind == "ask"]
        tools = sum(1 for n in r["nodes"] if n.kind == "tool")
        return len(asks), asks, -tools, -len(r["edges"]), [num(e) for e in r["edges"]]

    routes.sort(key=key)

    print("| Tag | Edges | Turns | Tool calls |")
    print("|---|---|---|---|")
    for i, r in enumerate(routes):
        tag = f"{a.tag}-{chr(65 + i)}" if i < 26 else f"{a.tag}-{i + 1}"
        asks = sum(1 for n in r["nodes"] if n.kind == "ask")
        tools = [n.text.replace("tool_call:", "").strip()
                 for n in r["nodes"] if n.kind == "tool"]
        print(f"| {tag} | {'→'.join(r['edges'])} | {asks + 1} | {', '.join(tools) or '—'} |")

    print(f"\n{len(routes)} routes", file=sys.stderr)

    # Common graph errors are edges absent from every route, usually an open cycle,
    # and terminal states not drawn as ([...]), usually due to a missing outgoing edge.
    covered = {e for r in routes for e in r["edges"]}
    orphan = sorted({e for e in eid.values() if e} - covered)
    if orphan:
        print(f"边不在任何路线里（死边，或图画错了）: {orphan}", file=sys.stderr)
    # Unnamed edges silently disappear from routes and leave an empty table cell.
    # This is especially hard to notice in a single-edge graph.
    unnamed = sorted(f"{a}→{b}" for (a, b), e in eid.items() if not e)
    if unnamed:
        print(f"这些边没有命名，不会出现在路线里: {unnamed}", file=sys.stderr)
    odd = sorted(n.name for n in nodes.values() if not n.next and n.kind != "end")
    if odd:
        print(f"这些节点没有出边，被当成终态，但没画成 ([...]): {odd}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
