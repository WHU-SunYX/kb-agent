"""FastAPI request and response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from kb_agent.models import SecurityClassification, SourceType
from kb_agent.retrieval.filters import RetrievalFilters


class IngestResponse(BaseModel):
    """Compact result returned after one uploaded source is ingested/indexed."""

    ingestion_id: str | None = None
    source_kind: str | None = None
    source_filename: str | None = None
    raw_uri: str | None = None
    raw_object_reused: bool = False
    document_count: int = 0
    accepted_document_count: int = 0
    blocked_document_count: int = 0
    chunk_count: int = 0
    indexed_chunk_count: int = 0


class SearchFilters(BaseModel):
    """Safe, schema-backed retrieval filters exposed by the HTTP API.

    These are exact-match business filters. Raw Milvus expressions are
    intentionally not accepted by the public API.
    """

    model_config = ConfigDict(extra="forbid")

    document_id: str | None = Field(default=None, max_length=128)
    domain: str | None = Field(default=None, max_length=128)
    project: str | None = Field(default=None, max_length=128)
    classification: SecurityClassification | None = None
    source_type: SourceType | None = None
    provider: str | None = Field(default=None, max_length=128)
    source_id: str | None = Field(default=None, max_length=256)

    @field_validator("document_id", "domain", "project", "provider", "source_id")
    @classmethod
    def reject_blank_values(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("search filter values must not be blank")
        return normalized

    def to_retrieval_filters(self) -> RetrievalFilters:
        return RetrievalFilters(**self.model_dump(mode="json", exclude_none=True))


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    filters: SearchFilters | None = None
    top_k: int = Field(default=10, ge=1, le=100)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized


class SearchHit(BaseModel):
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: list[SearchHit] = Field(default_factory=list)
