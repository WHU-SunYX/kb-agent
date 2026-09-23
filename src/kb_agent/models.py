"""Core domain models for kb-agent.

These models form the stable boundary between ingestion, storage, indexing,
retrieval, and MCP layers. They intentionally do not depend on ChatGPT,
Docling, Milvus, BGE, or MCP-specific types.

Design principles
-----------------
* Source-neutral: ChatGPT, Codex, Claude Code, Trae, PDF, DOCX, etc. are
  represented through a generic source type + provider pair.
* Traceable: every indexed or extracted item can point back to its original
  source through :class:`SourceReference`.
* Multi-tenant/domain ready: tenant, domain, and project are metadata rather
  than hard-coded directory or collection names.
* Security-aware: ACL and classification travel with the data from ingestion
  through retrieval.
* Version-aware: extracted knowledge can be marked current, historical, or
  superseded without deleting the underlying evidence.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _DomainModel(BaseModel):
    """Common Pydantic behavior for kb-agent domain models."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class SourceType(str, Enum):
    """High-level shape of an original source, independent of provider."""

    CONVERSATION = "conversation"
    DOCUMENT = "document"
    CODE = "code"
    ISSUE = "issue"
    WIKI = "wiki"
    EMAIL = "email"
    OTHER = "other"


class SecurityClassification(str, Enum):
    """Coarse information-security classification."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class KnowledgeType(str, Enum):
    """Semantic type of a distilled knowledge unit."""

    ARCHITECTURE_DECISION = "architecture_decision"
    TROUBLESHOOTING = "troubleshooting"
    EXPERIMENT = "experiment"
    CONFIGURATION = "configuration"
    CODE_CHANGE = "code_change"
    TECHNICAL_EXPLANATION = "technical_explanation"
    FAQ = "faq"
    PROCEDURE = "procedure"
    GENERAL = "general"


class KnowledgeStatus(str, Enum):
    """Lifecycle state for distilled knowledge."""

    DRAFT = "draft"
    VALIDATED = "validated"
    CURRENT = "current"
    HISTORICAL = "historical"
    SUPERSEDED = "superseded"


class RetrievalItemType(str, Enum):
    """Kind of object returned by retrieval."""

    CHUNK = "chunk"
    KNOWLEDGE_UNIT = "knowledge_unit"


class SourceReference(_DomainModel):
    """Provider-neutral pointer back to original evidence.

    ``source_type`` describes the shape of the source, while ``provider``
    identifies the concrete producer/format. For example::

        source_type="conversation", provider="chatgpt"
        source_type="conversation", provider="codex"
        source_type="document", provider="pdf"
        source_type="document", provider="docx"

    ``uri`` should identify the raw artifact whenever one exists, e.g.
    ``s3://kb-raw/chatgpt/...`` or ``file:///...``. Provider-specific
    locators such as message IDs, page ranges, thread IDs, repositories, or
    section anchors belong in ``metadata`` so the core schema remains stable.
    """

    source_type: SourceType
    provider: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    source_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SecurityMetadata(_DomainModel):
    """Security classification and retrieval-time access-control metadata."""

    tenant_id: str = Field(default="default", min_length=1)
    classification: SecurityClassification = SecurityClassification.INTERNAL
    owner_id: str | None = None
    allowed_users: list[str] = Field(default_factory=list)
    allowed_groups: list[str] = Field(default_factory=list)
    contains_sensitive_data: bool = False
    sensitive_categories: list[str] = Field(default_factory=list)

    @field_validator("allowed_users", "allowed_groups", "sensitive_categories")
    @classmethod
    def _deduplicate_strings(cls, values: list[str]) -> list[str]:
        """Remove blanks/duplicates while preserving the original order."""

        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = value.strip()
            if normalized and normalized not in seen:
                result.append(normalized)
                seen.add(normalized)
        return result


class NormalizedDocument(_DomainModel):
    """Canonical document produced by every ingestion source adapter.

    A ChatGPT/Codex/Claude/Trae conversation and a PDF/DOCX document all become
    this model before downstream security, chunking, extraction, and indexing.
    """

    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source: SourceReference

    domain: str | None = None
    project: str | None = None
    author: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    content_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    security: SecurityMetadata = Field(default_factory=SecurityMetadata)

    @model_validator(mode="after")
    def _validate_timestamps(self) -> "NormalizedDocument":
        if self.created_at and self.updated_at and self.updated_at < self.created_at:
            raise ValueError("updated_at must be greater than or equal to created_at")
        return self


class KnowledgeChunk(_DomainModel):
    """Retrieval-oriented chunk derived from a normalized document."""

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    source: SourceReference

    title: str | None = None
    section_path: list[str] = Field(default_factory=list)
    token_count: int | None = Field(default=None, ge=0)

    domain: str | None = None
    project: str | None = None
    content_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    security: SecurityMetadata = Field(default_factory=SecurityMetadata)

    @field_validator("section_path")
    @classmethod
    def _clean_section_path(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]


class KnowledgeUnit(_DomainModel):
    """Distilled, version-aware engineering/organizational knowledge.

    Knowledge units are primarily expected in the later Gold Knowledge phase.
    Keeping the model in Step 2 lets ingestion and retrieval APIs share a stable
    contract from the beginning.
    """

    knowledge_id: str = Field(min_length=1)
    knowledge_type: KnowledgeType = KnowledgeType.GENERAL
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)

    domain: str | None = None
    project: str | None = None

    problem: str | None = None
    root_cause: str | None = None
    solution: str | None = None
    conclusion: str | None = None
    verification: str | None = None

    status: KnowledgeStatus = KnowledgeStatus.DRAFT
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    supersedes: list[str] = Field(default_factory=list)

    evidence: list[SourceReference] = Field(default_factory=list)
    content_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    security: SecurityMetadata = Field(default_factory=SecurityMetadata)

    @model_validator(mode="after")
    def _validate_validity_window(self) -> "KnowledgeUnit":
        if self.valid_from and self.valid_to and self.valid_to < self.valid_from:
            raise ValueError("valid_to must be greater than or equal to valid_from")
        if self.knowledge_id in self.supersedes:
            raise ValueError("a knowledge unit cannot supersede itself")
        return self


class RetrievalResult(_DomainModel):
    """Provider-neutral result returned by the retrieval service.

    ``score`` is the final ranking score exposed to callers. Optional component
    scores are kept so later phases can inspect dense, BM25, and reranker
    behavior without changing the public result schema.
    """

    item_type: RetrievalItemType
    item_id: str = Field(min_length=1)
    title: str | None = None
    text: str = Field(min_length=1)
    score: float

    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None

    domain: str | None = None
    project: str | None = None
    source: SourceReference | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "KnowledgeChunk",
    "KnowledgeStatus",
    "KnowledgeType",
    "KnowledgeUnit",
    "NormalizedDocument",
    "RetrievalItemType",
    "RetrievalResult",
    "SecurityClassification",
    "SecurityMetadata",
    "SourceReference",
    "SourceType",
]
