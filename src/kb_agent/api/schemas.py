"""FastAPI request and response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


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


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    filter_expr: str | None = None
    top_k: int = 10


class SearchHit(BaseModel):
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: list[SearchHit] = Field(default_factory=list)
