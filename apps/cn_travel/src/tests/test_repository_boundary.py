"""Ownership checks for the self-contained CN Travel application."""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

from project_paths import APPLICATION_ROOT, RUNTIME_PACKAGE_ROOT, SOURCE_ROOT


APP_ROOT = APPLICATION_ROOT


SOURCE_SUFFIXES = {".py", ".sh", ".json", ".toml", ".yaml", ".yml", ".md"}
ROOT_SOURCE_SUFFIXES = SOURCE_SUFFIXES - {".md"}
IMMUTABLE_PARTS = {
    "__pycache__",
    ".venv",
    "archive",
    "reproducibility",
    "runs",
    "staging",
    "run_01_source",
    "run_02_source",
    "run_03_source",
}
IMMUTABLE_PREFIXES = ("adapter", "checkpoint", "logs")
ROOT_DOMAIN_NAMES = re.compile(
    r"\b(?:CN_TRAVEL|TravelAssistantFuncCall|search_travel_guide|"
    r"get_weather_info|query_route|recommend_hotels|get_hotel_reviews|"
    r"travel_assistant|travel_guides|city_code_mapping)\b|apps/cn_travel"
)
ROOT_TRAVEL_LANGUAGE = re.compile(
    r"\b(?:cn[_ -]?travel|travel|hotel|weather|tourism|itinerary|amap)\b|"
    r"旅行|旅游|酒店|天气|攻略",
    re.IGNORECASE,
)
OUTSIDE_APP_IDENTITY = re.compile(
    r"cn[_ -]?travel|README_CN|apps/cn_travel|TravelAssistantFuncCall",
    re.IGNORECASE,
)


def _active(path: Path) -> bool:
    relative = path.relative_to(APP_ROOT)
    return not any(
        part in IMMUTABLE_PARTS or part.startswith(IMMUTABLE_PREFIXES)
        for part in relative.parts
    )


def _python_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def _repository_root() -> Path | None:
    for candidate in (APP_ROOT, *APP_ROOT.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _tracked_files(repo: Path, *prefixes: str) -> list[Path]:
    output = subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "-z", "--", *prefixes]
    )
    return [repo / item for item in output.decode().split("\0") if item]


def test_active_application_python_imports_are_application_local():
    offenders: list[str] = []
    for path in APP_ROOT.rglob("*.py"):
        if not _active(path):
            continue
        external = sorted(
            name
            for name in _python_imports(path)
            if name == "miniagent" or name.startswith("miniagent.")
        )
        if external:
            offenders.append(f"{path.relative_to(APP_ROOT)}: {', '.join(external)}")
    assert not offenders, "application imports repository-root code:\n" + "\n".join(offenders)


def test_runtime_package_does_not_import_offline_areas():
    forbidden_roots = {"data", "eval", "train", "training", "world_policy"}
    offenders: list[str] = []
    for path in RUNTIME_PACKAGE_ROOT.rglob("*.py"):
        imports = _python_imports(path)
        external = sorted(
            name
            for name in imports
            if name.split(".", 1)[0] in forbidden_roots
        )
        if external:
            offenders.append(
                f"{path.relative_to(RUNTIME_PACKAGE_ROOT)}: {', '.join(external)}"
            )
    assert not offenders, "runtime imports offline areas:\n" + "\n".join(offenders)


def test_api_and_agent_reach_tool_implementations_only_through_services():
    offenders: list[str] = []
    for path in (RUNTIME_PACKAGE_ROOT / "api.py", RUNTIME_PACKAGE_ROOT / "agent.py"):
        direct_tool_imports = sorted(
            name
            for name in _python_imports(path)
            if name == "cn_travel.tool" or name.startswith("cn_travel.tool.")
        )
        if direct_tool_imports:
            offenders.append(
                f"{path.relative_to(RUNTIME_PACKAGE_ROOT)}: "
                f"{', '.join(direct_tool_imports)}"
            )
    assert not offenders, "API or agent bypasses the service layer:\n" + "\n".join(offenders)


def test_tool_service_loads_implementations_only_when_called():
    path = RUNTIME_PACKAGE_ROOT / "service" / "tools.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_imports.add(node.module)
    eager_tool_imports = sorted(
        name
        for name in module_imports
        if name == "cn_travel.tool" or name.startswith("cn_travel.tool.")
    )
    assert not eager_tool_imports, (
        "tool service eagerly imports provider-bound implementations: "
        + ", ".join(eager_tool_imports)
    )


def test_active_application_sources_do_not_assume_repository_layout():
    forbidden = {
        "REPO_ROOT": re.compile(r"\bREPO_ROOT\b"),
        "MINIAGENT_ROOT": re.compile(r"\bMINIAGENT_ROOT\b"),
        "root src path": re.compile(r"(?:\$ROOT|\$\{ROOT\})/src\b"),
        "root apps path": re.compile(r"(?:\$ROOT|\$\{ROOT\})/apps\b"),
        "root docs path": re.compile(r"(?:\.\./){2,}docs/"),
    }
    offenders: list[str] = []
    for path in APP_ROOT.rglob("*"):
        if not path.is_file() or not _active(path):
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix not in SOURCE_SUFFIXES and path.name != "Makefile":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        hits = [label for label, pattern in forbidden.items() if pattern.search(text)]
        if hits:
            offenders.append(f"{path.relative_to(APP_ROOT)}: {', '.join(hits)}")
    assert not offenders, "application assumes repository layout:\n" + "\n".join(offenders)


def test_application_symlinks_and_dependency_sources_stay_inside_application():
    escaping = []
    for path in APP_ROOT.rglob("*"):
        if path.is_symlink() and _active(path):
            try:
                path.resolve(strict=False).relative_to(APP_ROOT)
            except ValueError:
                escaping.append(str(path.relative_to(APP_ROOT)))
    assert not escaping, f"application symlinks escape APP_ROOT: {escaping}"

    manifests = "\n".join(
        (SOURCE_ROOT / name).read_text(encoding="utf-8")
        for name in ("pyproject.toml", "uv.lock")
    )
    assert not re.search(r"(?:path|editable)\s*=\s*[\"']\.\.", manifests)


def test_generic_repository_sources_contain_no_travel_contract():
    repo = _repository_root()
    if repo is None:
        pytest.skip("Git worktree metadata is unavailable for this boundary audit")
    offenders: list[str] = []
    for path in _tracked_files(repo, "."):
        relative = path.relative_to(repo)
        if relative.parts and relative.parts[0] == "apps":
            continue
        if not path.is_file() or (
            path.suffix.lower() not in ROOT_SOURCE_SUFFIXES
            and path.name not in {"Makefile", ".gitignore"}
        ):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if ROOT_DOMAIN_NAMES.search(text) or ROOT_TRAVEL_LANGUAGE.search(text):
            offenders.append(str(relative))
    assert not offenders, "generic repository sources contain travel policy:\n" + "\n".join(offenders)


def test_tracked_sources_outside_application_do_not_reference_cn_travel():
    repo = _repository_root()
    if repo is None:
        pytest.skip("Git worktree metadata is unavailable for this boundary audit")
    offenders: list[str] = []
    app_prefix = APP_ROOT.relative_to(repo)
    for path in _tracked_files(repo, "."):
        relative = path.relative_to(repo)
        if relative == app_prefix or app_prefix in relative.parents or not path.is_file():
            continue
        if path.suffix.lower() not in ROOT_SOURCE_SUFFIXES and path.name not in {"Makefile", ".gitignore"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if OUTSIDE_APP_IDENTITY.search(text):
            offenders.append(str(relative))
    assert not offenders, "sources outside the application reference CN Travel:\n" + "\n".join(offenders)
