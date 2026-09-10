"""Application-owned semantic validator for generated conversations."""

from typing import Any, Dict


VALIDATOR_READY = False


def validate(record: Dict[str, Any], route: Dict[str, Any]) -> bool:
    """Validate question intent and final response content for one route."""
    raise RuntimeError(
        "Implement data/conversation_validator.py and set VALIDATOR_READY = True."
    )
