"""Application service facade for guide-index build operations."""
from __future__ import annotations

from pathlib import Path

from cn_travel.tool.guide import embeddings as _embeddings


FINGERPRINT_NAME = _embeddings.FINGERPRINT_NAME


def embedding_dimensions() -> int:
    """Return the configured vector dimension used by indexing and retrieval."""
    return int(_embeddings._settings()[2])


def embed_one(text: str) -> list[float] | None:
    """Create one guide vector with the runtime embedding implementation."""
    return _embeddings.embed_one(text)


def warm_up() -> None:
    """Initialize the embedding model before concurrent guide processing."""
    _embeddings.warm_up()


def write_fingerprint(db_path: Path) -> None:
    """Record the configured embedding identity next to a built index."""
    _embeddings.write_fingerprint(db_path)


__all__ = [
    "FINGERPRINT_NAME",
    "embed_one",
    "embedding_dimensions",
    "warm_up",
    "write_fingerprint",
]
