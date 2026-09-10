#!/usr/bin/env python3
"""Compile one Mermaid workflow in a Markdown file into a route table."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import re
import sys


EDGE_RE = re.compile(r'^\s*(.+?)\s*-->\s*(?:\|\s*"?(.*?)"?\s*\|\s*)?(.+?)\s*$')
SHAPE_RE = re.compile(r'^(\w+)\s*(\(\[|\{|\[/|\[)')
BLOCK_RE = re.compile(r"```mermaid\s*(.*?)```", re.DOTALL)


@dataclass(eq=False)
class Node:
    name: str
    kind: str = ""
    text: str = ""
    outgoing: list["Edge"] = field(default_factory=list)


@dataclass(frozen=True)
class Edge:
    source: Node
    target: Node
    label: str
    ordinal: int


def parse_node(declaration: str) -> tuple[str, str, str]:
    declaration = declaration.strip()
    match = SHAPE_RE.match(declaration)
    if match is None:
        return declaration, "", ""

    kind = {"([": "end", "{": "decision", "[/": "ask", "[": "tool"}[match.group(2)]
    body = re.search(
        r'(?:\(\[|\{|\[/|\[)\s*"?(.*?)"?\s*(?:\]\)|\}|/\]|\])$', declaration
    )
    return match.group(1), kind, body.group(1) if body else ""


def parse_workflow(
    markdown: str, block_number: int = 1
) -> tuple[dict[str, Node], list[Edge]]:
    blocks = BLOCK_RE.findall(markdown)
    if not blocks:
        raise ValueError("The document contains no Mermaid code block.")
    if block_number < 1 or block_number > len(blocks):
        raise ValueError(
            f"Mermaid block {block_number} is unavailable; the document contains "
            f"{len(blocks)} block(s)."
        )
    block = blocks[block_number - 1]

    nodes: dict[str, Node] = {}
    edges: list[Edge] = []
    pairs: set[tuple[str, str]] = set()

    def get_node(declaration: str) -> Node:
        name, kind, text = parse_node(declaration)
        node = nodes.setdefault(name, Node(name=name))
        if kind:
            if node.kind and (node.kind != kind or node.text != text):
                raise ValueError(f"Node {name!r} has conflicting declarations.")
            node.kind = kind
            node.text = text
        return node

    for line in block.splitlines():
        if "-->" not in line:
            declaration = line.strip()
            if SHAPE_RE.match(declaration):
                get_node(declaration)
            continue
        match = EDGE_RE.match(line)
        if match is None:
            raise ValueError(f"Cannot parse Mermaid edge: {line.strip()}")

        source = get_node(match.group(1))
        target = get_node(match.group(3))
        pair = (source.name, target.name)
        if pair in pairs:
            raise ValueError(
                f"Workflow contains multiple edges from {source.name} to {target.name}; "
                "route-distinct choices must lead to distinct nodes."
            )
        pairs.add(pair)

        raw_label = (match.group(2) or "").strip()
        token = raw_label.split(maxsplit=1)[0] if raw_label else ""
        label = token if re.fullmatch(r"e\d+", token) else ""
        edge = Edge(source=source, target=target, label=label, ordinal=len(edges))
        source.outgoing.append(edge)
        edges.append(edge)

    if not edges:
        raise ValueError("The Mermaid block contains no workflow edges.")
    return nodes, edges


def enumerate_routes(start: Node) -> list[list[Edge]]:
    routes: list[list[Edge]] = []
    path: list[Edge] = []
    visited: set[int] = set()

    def walk(node: Node) -> None:
        if not node.outgoing:
            routes.append(list(path))
            return
        for edge in node.outgoing:
            if edge.ordinal in visited:
                continue
            visited.add(edge.ordinal)
            path.append(edge)
            walk(edge.target)
            path.pop()
            visited.remove(edge.ordinal)

    walk(start)
    return routes


def edge_number(label: str) -> int:
    return int(label[1:]) if re.fullmatch(r"e\d+", label) else 0


def route_sort_key(route: list[Edge]) -> tuple[object, ...]:
    nodes = [edge.target for edge in route]
    asks = [edge_number(edge.label) for edge in route if edge.target.kind == "ask"]
    tools = sum(node.kind == "tool" for node in nodes)
    labels = [edge_number(edge.label) for edge in route if edge.label]
    return len(asks), asks, -tools, -len(route), labels


def route_tag(prefix: str, index: int) -> str:
    if index < 26:
        return f"{prefix}-{chr(65 + index)}"
    return f"{prefix}-{index + 1}"


def compile_routes(document: Path, prefix: str, block_number: int = 1) -> str:
    nodes, edges = parse_workflow(
        document.read_text(encoding="utf-8"), block_number=block_number
    )
    connected = {
        node.name
        for edge in edges
        for node in (edge.source, edge.target)
    }
    isolated = sorted(set(nodes).difference(connected))
    if isolated:
        raise ValueError(
            "Workflow contains nodes outside every terminal route: "
            + ", ".join(isolated)
            + "."
        )
    targets = {edge.target.name for edge in edges}
    starts = [node for node in nodes.values() if node.name not in targets]
    if len(starts) != 1:
        names = ", ".join(node.name for node in starts) or "none"
        raise ValueError(f"Workflow must have one start node; found {names}.")

    start = starts[0]
    unnamed = [
        edge
        for edge in edges
        if not edge.label and edge.source is not start
    ]
    if unnamed:
        descriptions = ", ".join(
            f"{edge.source.name}->{edge.target.name}" for edge in unnamed
        )
        raise ValueError(f"Route-defining edges require e<number> labels: {descriptions}.")

    invalid_terminals = [
        node.name for node in nodes.values() if not node.outgoing and node.kind != "end"
    ]
    if invalid_terminals:
        raise ValueError(
            "Terminal nodes require ([...]) declarations: " + ", ".join(invalid_terminals)
        )

    routes = enumerate_routes(start)
    if not routes:
        raise ValueError("Workflow has no start-to-terminal route.")
    covered_edges = {edge.ordinal for route in routes for edge in route}
    uncovered_edges = [
        f"{edge.source.name}->{edge.target.name}"
        for edge in edges
        if edge.ordinal not in covered_edges
    ]
    if uncovered_edges:
        raise ValueError(
            "Workflow contains edges outside every terminal route: "
            + ", ".join(uncovered_edges)
            + "."
        )
    covered_nodes = {start.name}
    covered_nodes.update(
        edge.target.name for route in routes for edge in route
    )
    uncovered_nodes = sorted(set(nodes).difference(covered_nodes))
    if uncovered_nodes:
        raise ValueError(
            "Workflow contains nodes outside every terminal route: "
            + ", ".join(uncovered_nodes)
            + "."
        )
    routes.sort(key=route_sort_key)

    lines = [
        "| Tag | Edges | Turns | Tool calls |",
        "| --- | --- | ---: | --- |",
    ]
    for index, route in enumerate(routes):
        route_edges = [edge.label for edge in route if edge.label]
        nodes_in_route = [start, *(edge.target for edge in route)]
        turns = 1 + sum(node.kind == "ask" for node in nodes_in_route)
        tools = [
            node.text.replace("tool_call:", "", 1).strip()
            for node in nodes_in_route
            if node.kind == "tool"
        ]
        lines.append(
            f"| {route_tag(prefix, index)} | {'→'.join(route_edges)} | {turns} | "
            f"{', '.join(tools) or '—'} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path, help="Markdown file containing a Mermaid workflow")
    parser.add_argument("--tag", default="W1", help="Route tag prefix")
    parser.add_argument(
        "--block",
        type=int,
        default=1,
        help="One-based Mermaid block number for this workflow",
    )
    args = parser.parse_args()

    try:
        table = compile_routes(args.document, args.tag, block_number=args.block)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
