"""Static checks for the independently deployable CN Travel runtime."""

import json
from argparse import Namespace

from cn_travel import cli
from cn_travel.business_logic.contracts import TOOL_NAMES
from cn_travel.paths import CN_TRAVEL
from cn_travel.policy_server import policy_server_command
from cn_travel.service.config import AppConfig


def test_application_makefile_starts_the_http_api():
    makefile = (CN_TRAVEL.deploy_root.parent / "Makefile").read_text(encoding="utf-8")
    assert "python -m cn_travel serve" in makefile


def test_runtime_package_owns_each_required_surface():
    required = (
        "config.yaml",
        "deployment_manifest.json",
        "agent.py",
        "api.py",
        "business_logic/system_prompt.md",
        "business_logic/user_info.md",
        "business_logic/tool_schemas.json",
        "business_logic/contracts.py",
        "tool",
        "service",
        "model/policy/base/config.json",
        "model/policy/adapter/adapter_config.json",
        "tool/guide/knowledge_base/embedding/embeddinggemma-300m/modules.json",
        "tool/guide/knowledge_base/milvus.db",
    )
    missing = [relative for relative in required if not (CN_TRAVEL.root / relative).exists()]
    assert not missing, f"CN Travel runtime surfaces are missing: {missing}"


def test_tool_schema_and_contract_names_are_identical():
    schemas = json.loads(CN_TRAVEL.tool_schemas.read_text(encoding="utf-8"))
    names = [schema["function"]["name"] for schema in schemas]
    assert len(names) == len(set(names))
    assert set(names) == set(TOOL_NAMES)


def test_runtime_configuration_selects_packaged_embeddings():
    config = AppConfig.load(CN_TRAVEL.config)
    assert config.llm.provider == "openai_compat"
    assert config.llm.model == "cn-travel-policy-cee7c1bfa881"
    assert config.llm.base_url == "http://127.0.0.1:8000/v1"
    assert config.rag.embeddings.provider == "local"
    assert (
        config.rag.embeddings.model
        == "tool/guide/knowledge_base/embedding/embeddinggemma-300m"
    )
    assert (CN_TRAVEL.root / config.rag.embeddings.model).is_dir()


def test_project_declares_runtime_dependencies_and_commands():
    project_path = CN_TRAVEL.deploy_root / "pyproject.toml"
    project = project_path.read_text(encoding="utf-8")
    for dependency in ("fastapi", "uvicorn", "sentence-transformers", "pymilvus"):
        assert f'"{dependency}' in project
    assert 'cn-travel = "cn_travel.cli:main"' in project
    assert 'cn-travel-api = "cn_travel.api:main"' in project
    assert 'cn-travel-policy = "cn_travel.policy_server:main"' in project
    assert '"vllm==0.28.0;' in project


def test_deployment_manifest_paths_resolve_inside_package():
    manifest = json.loads(CN_TRAVEL.deployment_manifest.read_text(encoding="utf-8"))
    paths = (
        manifest["policy"]["base_path"],
        manifest["policy"]["adapter_path"],
        manifest["retrieval"]["embedding_path"],
        manifest["retrieval"]["index_path"],
        manifest["retrieval"]["fingerprint_path"],
    )
    root = CN_TRAVEL.root.resolve()
    for relative in paths:
        resolved = (root / relative).resolve()
        assert resolved.is_relative_to(root)
        assert resolved.exists()


def test_policy_launcher_binds_promoted_adapter_and_parser():
    manifest = json.loads(CN_TRAVEL.deployment_manifest.read_text(encoding="utf-8"))
    command = policy_server_command()
    served_model = manifest["policy"]["served_model_name"]
    assert (
        command[command.index("--served-model-name") + 1]
        == manifest["policy"]["base_served_model_name"]
    )
    assert command[command.index("--tool-call-parser") + 1] == "qwen3_xml"
    assert f"{served_model}={CN_TRAVEL.policy_adapter}" in command
    assert command[command.index("--max-lora-rank") + 1] == "32"


def test_validate_fails_closed_on_artifact_hash_error(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "deployment_status",
        lambda verify_hashes: {
            "ready": False,
            "errors": ["policy_adapter_weights: SHA-256 differs from deployment manifest"],
        },
    )
    assert cli.cmd_validate(Namespace()) == 1
    assert "policy_adapter_weights" in capsys.readouterr().out
