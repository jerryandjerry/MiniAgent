"""Strict result envelopes and bounded dispatch for application-owned tools."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional


@dataclass(frozen=True)
class ToolError:
    code: str
    message: str
    retryable: bool

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("Tool error code must be a non-empty string.")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("Tool error message must be a non-empty string.")
        if not isinstance(self.retryable, bool):
            raise ValueError("Tool error retryable must be a boolean.")


@dataclass(frozen=True)
class ToolResult:
    status: str
    data: Optional[Mapping[str, Any]] = None
    error: Optional[ToolError] = None

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def ok(cls, data: Mapping[str, Any]) -> "ToolResult":
        return cls(status="ok", data=data)

    @classmethod
    def empty(cls) -> "ToolResult":
        return cls(status="empty")

    @classmethod
    def failure(
        cls, code: str, message: str, retryable: bool = False
    ) -> "ToolResult":
        return cls(
            status="error",
            error=ToolError(code=code, message=message, retryable=retryable),
        )

    def validate(self) -> None:
        if not isinstance(self.status, str) or self.status not in {"ok", "empty", "error"}:
            raise ValueError("Tool result status must be ok, empty, or error.")
        if self.status == "ok":
            if not isinstance(self.data, Mapping) or self.error is not None:
                raise ValueError("An ok tool result requires data and excludes error.")
        elif self.status == "empty":
            if self.data is not None or self.error is not None:
                raise ValueError("An empty tool result carries null data and null error.")
        elif self.data is not None or not isinstance(self.error, ToolError):
            raise ValueError("An error tool result requires error and carries null data.")


ToolHandler = Callable[[Mapping[str, Any]], ToolResult]


class ToolService:
    def __init__(self) -> None:
        self._handlers: Dict[str, ToolHandler] = {}

    @property
    def names(self):
        return tuple(sorted(self._handlers))

    def register(self, name: str, handler: ToolHandler) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Tool names must be non-empty strings.")
        if not callable(handler):
            raise TypeError("Tool handlers must be callable.")
        if name in self._handlers:
            raise ValueError(f"Tool already registered: {name}")
        self._handlers[name] = handler

    def dispatch(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        try:
            handler = self._handlers[name]
        except KeyError as exc:
            raise ValueError(f"Tool is outside the application allowlist: {name}") from exc
        if not isinstance(arguments, Mapping):
            raise TypeError("Tool arguments must be a mapping.")
        result = handler(arguments)
        if not isinstance(result, ToolResult):
            raise TypeError("Tool handlers must return ToolResult.")
        result.validate()
        return result
