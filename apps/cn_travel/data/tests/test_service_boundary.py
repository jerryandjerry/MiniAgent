"""Dependency-boundary checks for active data-pipeline scripts."""
from __future__ import annotations

import ast
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.update(f"{node.module}.{alias.name}" for alias in node.names)
    return imports


def test_active_data_scripts_use_services_instead_of_tool_modules():
    offenders: list[str] = []
    for path in sorted(SCRIPTS_ROOT.glob("*.py")):
        private_imports = sorted(
            name
            for name in _imports(path)
            if name == "cn_travel.tool" or name.startswith("cn_travel.tool.")
        )
        if private_imports:
            offenders.append(f"{path.name}: {', '.join(private_imports)}")

    assert not offenders, "data scripts import tool implementations:\n" + "\n".join(
        offenders
    )
