"""Validation for the model and retrieval artifacts promoted into the package."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from cn_travel.artifact_integrity import (
    directory_manifest_errors,
    sha256_file,
)
from cn_travel.paths import CN_TRAVEL
from cn_travel.service.config import AppConfig


def _package_path(relative: str) -> Path:
    path = (CN_TRAVEL.root / relative).resolve()
    try:
        path.relative_to(CN_TRAVEL.root.resolve())
    except ValueError as exc:
        raise ValueError(f"artifact path leaves the package: {relative!r}") from exc
    return path


def deployment_status(verify_hashes: bool = True) -> dict[str, Any]:
    """Return readiness evidence for every promoted runtime artifact."""
    errors: list[str] = []
    manifest: dict[str, Any] = {}
    try:
        manifest = json.loads(CN_TRAVEL.deployment_manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"deployment manifest: {exc}")
        return {"ready": False, "errors": errors, "artifacts": {}}

    if manifest.get("schema_version") != "cn_travel.deployment.v2":
        errors.append("unsupported deployment manifest schema")

    try:
        policy = manifest["policy"]
        retrieval = manifest["retrieval"]
        base = _package_path(policy["base_path"])
        adapter = _package_path(policy["adapter_path"])
        embedding = _package_path(retrieval["embedding_path"])
        index = _package_path(retrieval["index_path"])
        fingerprint = _package_path(retrieval["fingerprint_path"])
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"deployment manifest paths: {exc}")
        return {"ready": False, "errors": errors, "artifacts": {}}

    served_model_name = policy.get("served_model_name", "")
    adapter_files = policy.get("adapter_directory", {}).get("files", [])
    adapter_hash = next(
        (
            entry.get("sha256", "")
            for entry in adapter_files
            if isinstance(entry, dict) and entry.get("path") == "adapter_model.safetensors"
        ),
        "",
    )
    if not adapter_hash or not served_model_name.endswith(adapter_hash[:12]):
        errors.append("served model name is not bound to the adapter SHA-256")
    try:
        configured_model = AppConfig.load(CN_TRAVEL.config).llm.model
        if configured_model != served_model_name:
            errors.append("runtime model name differs from the deployment manifest")
    except Exception as exc:
        errors.append(f"runtime config: {exc}")

    files: dict[str, tuple[Path | None, str | None]] = {
        "knowledge_index": (index, retrieval.get("index_sha256")),
        "knowledge_fingerprint": (
            fingerprint,
            retrieval.get("fingerprint_sha256"),
        ),
        "system_prompt": (CN_TRAVEL.system_prompt, None),
        "user_info": (CN_TRAVEL.user_info, None),
        "tool_schemas": (CN_TRAVEL.tool_schemas, None),
        "runtime_config": (CN_TRAVEL.config, None),
    }

    artifacts: dict[str, dict[str, Any]] = {}
    for name, (path, expected_hash) in files.items():
        if path is None:
            errors.append(f"{name}: expected exactly one model weight file")
            artifacts[name] = {"exists": False}
            continue
        exists = path.is_file()
        item: dict[str, Any] = {
            "path": str(path.relative_to(CN_TRAVEL.root)),
            "exists": exists,
        }
        if not exists:
            errors.append(f"{name}: missing {item['path']}")
        elif verify_hashes and expected_hash:
            actual_hash = sha256_file(path)
            item["sha256"] = actual_hash
            item["verified"] = actual_hash == expected_hash
            if actual_hash != expected_hash:
                errors.append(f"{name}: SHA-256 differs from deployment manifest")
        artifacts[name] = item

    directories = {
        "policy_base_directory": (base, policy.get("base_directory")),
        "policy_adapter_directory": (adapter, policy.get("adapter_directory")),
        "embedding_directory": (embedding, retrieval.get("embedding_directory")),
    }
    for name, (directory, expected) in directories.items():
        directory_errors, actual = directory_manifest_errors(
            directory, expected, name, verify_hashes=verify_hashes
        )
        errors.extend(directory_errors)
        artifacts[name] = {
            "path": str(directory.relative_to(CN_TRAVEL.root)),
            "exists": directory.is_dir(),
            "file_count": actual["file_count"] if actual else None,
            "total_bytes": actual["total_bytes"] if actual else None,
            "sha256": actual["tree_sha256"] if actual else None,
            "verified": not directory_errors if verify_hashes else None,
        }

    return {
        "ready": not errors,
        "policy": policy.get("name"),
        "served_model_name": served_model_name,
        "tool_parser": policy.get("tool_parser"),
        "expected_collection_size": retrieval.get("entity_count"),
        "errors": errors,
        "artifacts": artifacts,
    }


def policy_endpoint_status(timeout: float = 2.0) -> dict[str, Any]:
    """Check that the configured OpenAI-compatible endpoint serves the policy."""
    config = AppConfig.load(CN_TRAVEL.config).llm
    manifest = json.loads(CN_TRAVEL.deployment_manifest.read_text(encoding="utf-8"))
    policy = manifest["policy"]
    transport = os.getenv("LLM_TRANSPORT", config.provider).lower()
    base_url = os.getenv("LLM_BASE_URL", config.base_url or "").rstrip("/")
    model = os.getenv("LLM_MODEL", config.model)
    if transport not in {"local", "openai", "openai_compat"}:
        return {
            "ready": False,
            "transport": transport,
            "base_url": base_url,
            "model": model,
            "error": "deployment requires the packaged OpenAI-compatible policy",
        }
    if not base_url:
        return {
            "ready": False,
            "transport": transport,
            "base_url": base_url,
            "model": model,
            "error": "LLM_BASE_URL or llm.base_url is required",
        }

    request = Request(f"{base_url}/models")
    api_key = os.getenv("LLM_API_KEY")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        cards = [item for item in payload.get("data", []) if isinstance(item, dict)]
        served = [item.get("id") for item in cards if item.get("id")]
        matches = [item for item in cards if item.get("id") == model]
        expected_root = CN_TRAVEL.policy_adapter.resolve()
        identity_matches = [
            item
            for item in matches
            if item.get("parent") == policy["base_served_model_name"]
            and item.get("root")
            and Path(item["root"]).is_absolute()
            and (
                not Path(item["root"]).exists()
                or Path(item["root"]).resolve() == expected_root
            )
        ]
        ready = (
            model == policy["served_model_name"]
            and len(matches) == 1
            and len(identity_matches) == 1
        )
        identity_mode = None
        if ready:
            served_root = Path(identity_matches[0]["root"])
            identity_mode = (
                "local_path" if served_root.exists() else "manifest_model_id"
            )
        status: dict[str, Any] = {
            "ready": ready,
            "transport": transport,
            "base_url": base_url,
            "model": model,
            "served_models": served,
            "model_card": matches[0] if len(matches) == 1 else None,
            "identity_mode": identity_mode,
        }
        if not ready:
            status["error"] = (
                f"configured LoRA {model!r} is not uniquely served on base "
                f"{policy['base_served_model_name']!r} with the manifest identity"
            )
        return status
    except (HTTPError, URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "ready": False,
            "transport": transport,
            "base_url": base_url,
            "model": model,
            "error": str(exc),
        }
