#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared embedding contract for corpus indexing and retrieval.

The index fingerprint records provider, model, and dimensions so queries use
the same vector space that built the index.

Configuration is loaded from the packaged ``cn_travel/config.yaml`` resource:

    rag:
      embeddings:
        provider: dashscope    # dashscope | local
        model: text-embedding-v4
        dims: 1024
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

from cn_travel.paths import CN_TRAVEL

Vector = List[float]

# Store the indexing model fingerprint here and verify it during queries.
FINGERPRINT_NAME = ".embedding_model.json"


# ---------------------------------------------------------------- providers
def _dashscope(texts: List[str], model: str, dims: int) -> List[Vector]:
    """Call DashScope through its OpenAI-compatible embeddings API."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY", ""),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    out: List[Vector] = []
    for i in range(0, len(texts), 10):          # The API accepts at most 10 items per request.
        r = client.embeddings.create(model=model, input=texts[i:i + 10], dimensions=dims)
        out += [d.embedding for d in r.data]
    return out


_LOCAL_CACHE: Dict[str, object] = {}
_LOCAL_LOCK = __import__("threading").Lock()


def _local(texts: List[str], model: str, dims: int) -> List[Vector]:
    """Embed texts with a local sentence-transformers model.

    Indexing and querying intentionally use the same unprefixed text format.
    """
    from sentence_transformers import SentenceTransformer

    # Serialize cold-start initialization so workers share one model instance.
    st = _LOCAL_CACHE.get(model)
    if st is None:
        with _LOCAL_LOCK:
            st = _LOCAL_CACHE.get(model)
            if st is None:
                source = Path(model).expanduser()
                app_source = CN_TRAVEL.root / source
                if not source.is_absolute() and app_source.exists():
                    source = app_source
                st = _LOCAL_CACHE[model] = SentenceTransformer(str(source))
    return [v.tolist() for v in st.encode(texts, normalize_embeddings=True)]


_PROVIDERS: Dict[str, Callable[..., List[Vector]]] = {
    "dashscope": _dashscope,
    "local": _local,
}


# ---------------------------------------------------------------- config
def _settings() -> tuple[str, str, int]:
    """Return provider, model, and dimensions with environment overrides."""
    provider = os.getenv("EMBEDDING_PROVIDER", "")
    model = os.getenv("EMBEDDING_MODEL", "")
    dims = os.getenv("EMBEDDING_DIMS", "")
    if provider and model and dims:
        return provider, model, int(dims)

    try:
        from cn_travel.service.config import AppConfig
        cfg = AppConfig.load(CN_TRAVEL.config)
        e = cfg.rag.embeddings
        return provider or e.provider, model or e.model, int(dims or e.dims)
    except Exception:
        return provider or "dashscope", model or "text-embedding-v4", int(dims or 1024)


def describe() -> str:
    """Return the canonical embedding configuration identifier."""
    provider, model, dims = _settings()
    return f"{provider}:{model}:{dims}"


# ---------------------------------------------------------------- public
def embed(texts: List[str]) -> List[Vector]:
    """Embed text through the configured provider."""
    if not texts:
        return []
    provider, model, dims = _settings()
    fn = _PROVIDERS.get(provider)
    if fn is None:
        raise ValueError(f"未知的 embeddings provider: {provider!r}（可选 {sorted(_PROVIDERS)}）")
    return fn(texts, model, dims)


def embed_one(text: str) -> Optional[Vector]:
    """Embed one text, logging provider failures before returning ``None``."""
    import logging

    try:
        got = embed([text])
        return got[0] if got else None
    except Exception as exc:
        logging.getLogger(__name__).error("向量化失败: %s: %s", type(exc).__name__, exc)
        return None


def warm_up() -> None:
    """Initialize the embedding model before concurrent indexing begins."""
    try:
        embed(["warm up"])
    except Exception:
        pass


# ---------------------------------------------------------------- fingerprint
def write_fingerprint(db_path: Path) -> None:
    """Record the embedding configuration that built the index."""
    p = Path(db_path).with_name(FINGERPRINT_NAME)
    p.write_text(json.dumps({"embedding": describe()}, ensure_ascii=False, indent=2),
                 encoding="utf-8")


def check_fingerprint(db_path: Path) -> Optional[str]:
    """Return a warning when the current and indexed vector spaces differ."""
    p = Path(db_path).with_name(FINGERPRINT_NAME)
    if not p.exists():
        return None                      # Allow legacy databases without a fingerprint.
    try:
        was = json.loads(p.read_text(encoding="utf-8")).get("embedding")
    except Exception:
        return None
    now = describe()
    def canonical(value: object) -> str:
        descriptor = str(value)
        for prefix in (
            ":models/embedding/",
            ":model/embedding/",
            ":tool/guide/knowledge_base/embedding/",
        ):
            descriptor = descriptor.replace(prefix, ":")
        return descriptor

    normalized_was = canonical(was)
    normalized_now = canonical(now)
    if was and normalized_was != normalized_now:
        return (f"向量库是用 {was} 建的，当前配置是 {now}。"
                f"两者不在同一个向量空间，必须重新导入语料："
                f"run make build-kb from the CN Travel application directory")
    return None
