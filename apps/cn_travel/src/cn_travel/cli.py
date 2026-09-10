#!/usr/bin/env python3
"""Application-owned commands for validating and running CN Travel."""
from __future__ import annotations

import argparse
import os

from cn_travel.paths import CN_TRAVEL
from cn_travel.deployment import deployment_status
from cn_travel.service.config import AppConfig


REQUIRED = (
    ("config.yaml", "应用声明"),
    ("business_logic/system_prompt.md", "从客户业务逻辑编译的 QA 策略"),
    ("business_logic/tool_schemas.json", "工具 schema"),
    ("business_logic/contracts.py", "工具返回值契约"),
    ("tool", "工具实现"),
    ("deployment_manifest.json", "部署资产清单"),
    ("tool/guide/knowledge_base/milvus.db", "检索索引"),
)


def cmd_validate(args: argparse.Namespace) -> int:
    problems: list[str] = []
    for relative, purpose in REQUIRED:
        if not (CN_TRAVEL.root / relative).exists():
            problems.append(f"缺 {relative}  ({purpose})")

    config = None
    config_path = CN_TRAVEL.config
    if config_path.exists():
        try:
            config = AppConfig.load(config_path)
        except Exception as exc:
            problems.append(f"config.yaml 不合法: {exc}")

    if config and config.name != CN_TRAVEL.name:
        problems.append(
            f"config.yaml 的 name={config.name!r} 与应用 {CN_TRAVEL.name!r} 不一致"
        )

    if config and config.tools and (CN_TRAVEL.business_logic / "contracts.py").exists():
        try:
            from cn_travel.business_logic.contracts import TOOL_NAMES

            missing = set(config.tools) - set(TOOL_NAMES)
            if missing:
                problems.append(f"config.yaml 里的工具没有契约: {sorted(missing)}")
        except Exception as exc:
            problems.append(f"contracts 导入失败: {exc}")

    try:
        artifact_status = deployment_status(verify_hashes=True)
        problems.extend(artifact_status["errors"])
    except Exception as exc:
        problems.append(f"deployment artifacts: {exc}")

    if problems:
        print(f"✗ {CN_TRAVEL.name} 有 {len(problems)} 处问题:")
        for problem in problems:
            print("   -", problem)
        return 1

    print(f"✓ {CN_TRAVEL.name} 符合 spec")
    if config:
        print(f"   llm        {config.llm.provider} / {config.llm.model}")
        print(
            f"   embeddings {config.rag.embeddings.provider} / "
            f"{config.rag.embeddings.model} ({config.rag.embeddings.dims}d)"
        )
        print(f"   tools      {', '.join(config.tools) or '(未声明)'}")
    if CN_TRAVEL.knowledge_base.exists():
        file_count = sum(
            1 for path in CN_TRAVEL.knowledge_base.rglob("*") if path.is_file()
        )
        print(f"   知识库     {file_count} 个文件")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "cn_travel.api:app",
        host=os.getenv("CN_TRAVEL_HOST", "0.0.0.0"),
        port=int(os.getenv("CN_TRAVEL_PORT", "8010")),
    )
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    from cn_travel.agent import main as chat_main

    chat_main()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cn-travel", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    for name, handler, help_text in (
        ("validate", cmd_validate, "检查应用是否符合 spec"),
        ("serve", cmd_serve, "启动知识库服务"),
        ("chat", cmd_chat, "与应用对话"),
    ):
        command = subcommands.add_parser(name, help=help_text)
        command.set_defaults(handler=handler)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
