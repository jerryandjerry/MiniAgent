#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Typed schema for the application's ``config.yaml`` file."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Chat and tool-calling backend configuration."""

    provider: str = "openai_compat"       # openai_compat | openai | local
    model: str = "cn-travel-policy-cee7c1bfa881"
    base_url: Optional[str] = "http://127.0.0.1:8000/v1"


class EmbeddingsConfig(BaseModel):
    """Embedding backend whose dimensions must match the Milvus index."""

    provider: str = "dashscope"
    model: str = "text-embedding-v4"
    dims: int = 1024


class RAGConfig(BaseModel):
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    port: int = 8010


class AppConfig(BaseModel):
    """Complete application configuration."""

    name: str
    description: str = ""
    language: str = "zh"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    tools: List[str] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        return cls(**yaml.safe_load(path.read_text(encoding="utf-8")))
