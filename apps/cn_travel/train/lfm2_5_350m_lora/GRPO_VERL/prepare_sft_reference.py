#!/usr/bin/env python3
"""Materialize and verify the frozen SFT policy used as VERL's KL reference.

VERL computes a LoRA reference policy by disabling the trainable adapter.  If
the §5.1 adapter were passed through ``model.lora_adapter_path``, that reference
would silently be the bare LFM base.  We instead merge the sealed SFT adapter
into an immutable Hugging Face checkpoint, train a fresh GRPO LoRA on top, and
verify the merged checkpoint's complete file manifest before every run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import tempfile
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
APP_ROOT = pathlib.Path(__file__).resolve().parents[3]
MANIFEST_NAME = "cn_travel_sft_reference.json"


def _resolve(value: str | os.PathLike[str]) -> pathlib.Path:
    path = pathlib.Path(value)
    if path.is_absolute():
        return path
    resolved = APP_ROOT / path
    if resolved.exists() or not path.parts or path.parts[0] != "models":
        return resolved
    return APP_ROOT / "train" / pathlib.Path(*path.parts[1:])


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(path: pathlib.Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise SystemExit(f"{label} SHA-256 mismatch: {actual} != {expected} ({path})")


def _weight_files(directory: pathlib.Path) -> list[pathlib.Path]:
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        files = sorted(directory.glob("pytorch_model*.bin"))
    if not files:
        raise SystemExit(f"merged SFT checkpoint has no model weights: {directory}")
    return files


def _source_contract(config: dict[str, Any]) -> dict[str, Any]:
    base = _resolve(config["base_model"])
    adapter = _resolve(config["sft_adapter"])
    base_weights = base / config.get("base_weights_file", "model.safetensors")
    adapter_weights = adapter / "adapter_model.safetensors"
    _require_sha(base_weights, config["base_model_sha256"], "base model")
    _require_sha(adapter_weights, config["sft_adapter_sha256"], "SFT adapter")

    base_config_path = base / "config.json"
    adapter_config_path = adapter / "adapter_config.json"
    _require_sha(base_config_path, config["base_config_sha256"], "base model config")
    _require_sha(
        adapter_config_path,
        config["sft_adapter_config_sha256"],
        "SFT adapter config",
    )
    tokenizer_hashes = config.get("sft_tokenizer_files_sha256")
    if not isinstance(tokenizer_hashes, dict) or not tokenizer_hashes:
        raise SystemExit("SFT tokenizer/chat-template hashes are not configured")
    for name, expected in sorted(tokenizer_hashes.items()):
        if not isinstance(name, str) or not isinstance(expected, str):
            raise SystemExit("invalid SFT tokenizer/chat-template hash contract")
        _require_sha(adapter / name, expected, f"SFT tokenizer asset {name}")
    if not adapter_config_path.is_file():
        raise SystemExit(f"missing SFT adapter config: {adapter_config_path}")
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    declared_base = adapter_config.get("base_model_name_or_path")
    if declared_base and _resolve(declared_base).resolve() != base.resolve():
        raise SystemExit(
            "SFT adapter declares a different base model: "
            f"{declared_base!r} != {config['base_model']!r}"
        )
    return {
        "base_model": config["base_model"],
        "base_weights_file": config.get("base_weights_file", "model.safetensors"),
        "base_model_sha256": config["base_model_sha256"],
        "base_config_sha256": config["base_config_sha256"],
        "sft_adapter": config["sft_adapter"],
        "sft_adapter_sha256": config["sft_adapter_sha256"],
        "sft_adapter_config_sha256": config["sft_adapter_config_sha256"],
        "sft_tokenizer_files_sha256": dict(sorted(tokenizer_hashes.items())),
    }


def _same_source_artifacts(recorded: Any, current: dict[str, Any]) -> bool:
    """Compare immutable source bytes while allowing application-local relocation."""

    if not isinstance(recorded, dict) or set(recorded) != set(current):
        return False
    path_labels = {"base_model", "sft_adapter"}
    return all(
        key in path_labels or recorded[key] == current[key]
        for key in current
    )


def verify(config: dict[str, Any]) -> pathlib.Path:
    source = _source_contract(config)
    output = _resolve(config["sft_reference_model"])
    weights_name = str(config["sft_reference_weights_file"])
    expected_weights_sha256 = str(config["sft_reference_weights_sha256"])
    _require_sha(
        output / weights_name,
        expected_weights_sha256,
        "frozen merged SFT weights",
    )
    manifest_path = output / MANIFEST_NAME
    if not manifest_path.is_file():
        raise SystemExit(f"frozen SFT reference manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid SFT reference manifest: {manifest_path}: {exc}") from exc
    if manifest.get("schema_version") != "cn_travel.sft_reference.v1":
        raise SystemExit(f"unsupported SFT reference manifest: {manifest_path}")
    if not _same_source_artifacts(manifest.get("source"), source):
        raise SystemExit("frozen SFT reference source identity does not match the config")
    if manifest.get("merged_weights") != {
        "file": weights_name,
        "sha256": expected_weights_sha256,
    }:
        raise SystemExit("frozen SFT reference weight identity does not match the config")

    expected_files = manifest.get("files")
    if not isinstance(expected_files, dict) or not expected_files:
        raise SystemExit("frozen SFT reference manifest has no file hashes")
    actual_files: dict[str, str] = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != MANIFEST_NAME:
            actual_files[path.name] = _sha256(path)
    if actual_files != expected_files:
        missing = sorted(set(expected_files) - set(actual_files))
        extra = sorted(set(actual_files) - set(expected_files))
        changed = sorted(
            name for name in set(actual_files) & set(expected_files)
            if actual_files[name] != expected_files[name]
        )
        raise SystemExit(
            "frozen SFT reference bytes changed: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    _weight_files(output)
    print(
        "verified frozen SFT reference: "
        f"base={source['base_model_sha256']} adapter={source['sft_adapter_sha256']}"
    )
    return output


def prepare(config: dict[str, Any]) -> pathlib.Path:
    source = _source_contract(config)
    base = _resolve(config["base_model"])
    adapter = _resolve(config["sft_adapter"])
    output = _resolve(config["sft_reference_model"])
    if output.exists():
        return verify(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise SystemExit(
                "preparing the frozen SFT reference requires torch, transformers, and peft"
            ) from exc

        print(f"merging sealed SFT adapter into frozen reference: {tmp}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            base,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(
            model,
            adapter,
            is_trainable=False,
            local_files_only=True,
        )
        merged = model.merge_and_unload(safe_merge=True)
        merged.save_pretrained(tmp, safe_serialization=True)
        tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
        tokenizer.save_pretrained(tmp)
        del merged, model

        _weight_files(tmp)
        weights_name = str(config["sft_reference_weights_file"])
        expected_weights_sha256 = str(config["sft_reference_weights_sha256"])
        _require_sha(
            tmp / weights_name,
            expected_weights_sha256,
            "newly merged SFT weights",
        )
        files = {
            path.name: _sha256(path)
            for path in sorted(tmp.iterdir())
            if path.is_file() and path.name != MANIFEST_NAME
        }
        manifest = {
            "schema_version": "cn_travel.sft_reference.v1",
            "construction": "base_plus_sft_adapter_merge_and_unload_bfloat16",
            "source": source,
            "merged_weights": {
                "file": weights_name,
                "sha256": expected_weights_sha256,
            },
            "files": files,
        }
        (tmp / MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, output)
    except BaseException:
        if tmp.exists():
            shutil.rmtree(tmp)
        raise
    return verify(config)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare/verify the frozen §5.1 SFT reference")
    parser.add_argument("--config", type=pathlib.Path, default=HERE / "verl_traj_config.json")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    (verify if args.verify_only else prepare)(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
