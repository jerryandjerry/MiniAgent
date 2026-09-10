"""Launch the packaged policy through vLLM's OpenAI-compatible server."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from cn_travel.paths import CN_TRAVEL


def policy_server_command(extra_args: Sequence[str] = ()) -> list[str]:
    manifest = json.loads(CN_TRAVEL.deployment_manifest.read_text(encoding="utf-8"))
    policy = manifest["policy"]
    served_model = policy["served_model_name"]
    base = CN_TRAVEL.root / policy["base_path"]
    adapter = CN_TRAVEL.root / policy["adapter_path"]
    adapter_config = json.loads(
        adapter.joinpath("adapter_config.json").read_text(encoding="utf-8")
    )
    ranks = [adapter_config["r"], *adapter_config.get("rank_pattern", {}).values()]
    max_lora_rank = max(int(rank) for rank in ranks)
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(base),
        "--served-model-name",
        policy["base_served_model_name"],
        "--enable-lora",
        "--lora-modules",
        f"{served_model}={adapter}",
        "--max-lora-rank",
        str(max_lora_rank),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        policy["tool_parser"],
        "--host",
        os.getenv("CN_TRAVEL_POLICY_HOST", "127.0.0.1"),
        "--port",
        os.getenv("CN_TRAVEL_POLICY_PORT", "8000"),
        *extra_args,
    ]


def main() -> None:
    if importlib.util.find_spec("vllm") is None:
        raise SystemExit("vLLM is required; install the inference dependency group")
    env = os.environ.copy()
    executable_dir = str(Path(sys.executable).parent)
    env["PATH"] = os.pathsep.join(
        part for part in (executable_dir, env.get("PATH", "")) if part
    )
    raise SystemExit(
        subprocess.call(policy_server_command(sys.argv[1:]), env=env)
    )


if __name__ == "__main__":
    main()
