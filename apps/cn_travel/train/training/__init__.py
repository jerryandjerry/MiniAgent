"""Shared implementation used by CN Travel training entrypoints."""

from typing import Any


__all__ = ["DataCollatorForCausal", "JsonlConversations"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import masking

        return getattr(masking, name)
    raise AttributeError(name)
