#!/usr/bin/env python3
"""Validate the structure and rendered state of a MiniAgent application scaffold."""

from __future__ import annotations

import argparse
import json
import keyword
from pathlib import Path
import re
import sys

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None


REQUIRED_PATHS = (
    "BUSINESS_BRIEF.md",
    "README.md",
    "Makefile",
    "pytest.ini",
    "data/README.md",
    "data/config.json",
    "data/conversation_validator.py",
    "data/scripts",
    "data/specification/application.json",
    "data/specification/schemas/application.schema.json",
    "data/specification/schemas/workflow.schema.json",
    "data/specification/schemas/tool.schema.json",
    "data/specification/schemas/dataset-manifest.schema.json",
    "data/specification/schemas/training-manifest.schema.json",
    "data/specification/schemas/evaluation-manifest.schema.json",
    "data/specification/schemas/release-manifest.schema.json",
    "data/tests",
    "data/reproducibility",
    "train/README.md",
    "train/config.json",
    "train/configurations",
    "train/run.py",
    "train/runs",
    "train/tests",
    "eval/README.md",
    "eval/config.json",
    "eval/configurations",
    "eval/evaluate.py",
    "eval/promote.py",
    "eval/results/runs",
    "eval/releases",
    "eval/leaderboard.json",
    "eval/tests",
    "eval/reproducibility",
    "src/pyproject.toml",
    "src/.python-version",
    "src/.env.example",
    "src/tests",
)

MAKE_TARGETS = (
    "help",
    "install",
    "validate",
    "data-build",
    "train",
    "eval",
    "promote-runtime",
    "test",
    "test-standalone",
    "serve",
    "chat",
)

TEMPLATE_MARKERS = ("{{", "{%", "{#")
TEXT_SUFFIXES = {"", ".md", ".py", ".toml", ".ini", ".yaml", ".yml", ".json"}
IGNORED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}


def load_project_name(pyproject: Path) -> str:
    if tomllib is not None:
        with pyproject.open("rb") as handle:
            data = tomllib.load(handle)
        name = data.get("project", {}).get("name")
    else:
        text = pyproject.read_text(encoding="utf-8")
        project = re.search(r"(?ms)^\[project\]\s*(.*?)(?=^\[|\Z)", text)
        match = (
            re.search(r'''(?m)^name\s*=\s*["']([^"']+)["']\s*$''', project.group(1))
            if project
            else None
        )
        name = match.group(1) if match else None
    if not isinstance(name, str) or not name.strip():
        raise ValueError("src/pyproject.toml requires project.name")
    return name


def validate(root: Path) -> list[str]:
    errors: list[str] = []
    if not root.is_dir():
        return [f"application directory does not exist: {root}"]

    for relative in REQUIRED_PATHS:
        if not (root / relative).exists():
            errors.append(f"missing required path: {relative}")

    makefile = root / "Makefile"
    if makefile.is_file():
        contents = makefile.read_text(encoding="utf-8")
        for target in MAKE_TARGETS:
            if re.search(rf"(?m)^{re.escape(target)}\s*:", contents) is None:
                errors.append(f"Makefile is missing target: {target}")

    pyproject = root / "src/pyproject.toml"
    if pyproject.is_file():
        try:
            project_name = load_project_name(pyproject)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
        else:
            package_name = project_name.replace("-", "_")
            if keyword.iskeyword(package_name):
                errors.append(
                    f"runtime package must not be a Python keyword: {package_name}"
                )
            package = root / "src" / package_name
            if not (package / "__init__.py").is_file():
                errors.append(
                    f"runtime package requires src/{package_name}/__init__.py"
                )
            for relative in (
                "agent.py",
                "api.py",
                "cli.py",
                "release_verifier.py",
                "business_logic/application.json",
                "business_logic/application.schema.json",
                "business_logic/release.schema.json",
                "service/application.py",
                "service/tools.py",
            ):
                if not (package / relative).is_file():
                    errors.append(
                        f"runtime package requires src/{package_name}/{relative}"
                    )

    for path in root.rglob("*"):
        if any(part in IGNORED_PARTS for part in path.relative_to(root).parts):
            continue
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(marker in text for marker in TEMPLATE_MARKERS):
            errors.append(f"unrendered template marker: {path.relative_to(root)}")
        if path.suffix.lower() == ".json":
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                errors.append(f"invalid JSON in {path.relative_to(root)}: {exc}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("application", type=Path)
    args = parser.parse_args()
    root = args.application.resolve()
    errors = validate(root)
    if errors:
        print("MiniAgent scaffold validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"MiniAgent scaffold is valid: {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
