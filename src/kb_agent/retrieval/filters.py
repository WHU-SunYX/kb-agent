"""Structured retrieval intent and trusted access scope.

The LLM-facing surface must never accept a raw backend query expression.  It
expresses retrieval intent with schema-backed fields; each storage adapter is
responsible for compiling that intent to its own query language.

Access-control scope is modeled separately from model-selectable filters.  A
caller may ask to narrow results by project or source type, but it cannot use
those fields to replace the server's tenant boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import ClassVar


class RetrievalFilterError(ValueError):
    """Raised when structured retrieval constraints are invalid."""


def _normalize(value: str, *, name: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise RetrievalFilterError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise RetrievalFilterError(f"{name} must not be blank")
    if len(normalized) > max_length:
        raise RetrievalFilterError(
            f"{name} exceeds the maximum length of {max_length}"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class RetrievalScope:
    """Trusted access-control scope supplied by the application, never the LLM."""

    tenant_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            _normalize(self.tenant_id, name="tenant_id", max_length=128),
        )

    def items(self) -> tuple[tuple[str, str], ...]:
        return (("tenant_id", self.tenant_id),)


@dataclass(frozen=True, slots=True)
class RetrievalFilters:
    """Backend-neutral exact-match filters for indexed chunk metadata."""

    document_id: str | None = None
    domain: str | None = None
    project: str | None = None
    classification: str | None = None
    source_type: str | None = None
    provider: str | None = None
    source_id: str | None = None

    _MAX_LENGTHS: ClassVar[dict[str, int]] = {
        "document_id": 128,
        "domain": 128,
        "project": 128,
        "classification": 64,
        "source_type": 64,
        "provider": 128,
        "source_id": 256,
    }

    def __post_init__(self) -> None:
        for item in fields(self):
            name = item.name
            value = getattr(self, name)
            if value is None:
                continue
            object.__setattr__(
                self,
                name,
                _normalize(value, name=name, max_length=self._MAX_LENGTHS[name]),
            )

    def items(self) -> tuple[tuple[str, str], ...]:
        """Return configured filters in stable schema order."""
        return tuple(
            (item.name, value)
            for item in fields(self)
            if (value := getattr(self, item.name)) is not None
        )

    def is_empty(self) -> bool:
        return not self.items()


__all__ = [
    "RetrievalFilterError",
    "RetrievalFilters",
    "RetrievalScope",
]
