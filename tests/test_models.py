from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from kb_agent.models import (
    KnowledgeChunk,
    KnowledgeStatus,
    KnowledgeType,
    KnowledgeUnit,
    NormalizedDocument,
    RetrievalItemType,
    RetrievalResult,
    SecurityClassification,
    SecurityMetadata,
    SourceReference,
    SourceType,
)


def test_conversation_source_is_provider_neutral() -> None:
    source = SourceReference(
        source_type=SourceType.CONVERSATION,
        provider="chatgpt",
        uri="s3://kb-raw/chatgpt/conversations/abc.json",
        source_id="abc",
        metadata={"message_ids": ["m1", "m2"]},
    )

    assert source.source_type is SourceType.CONVERSATION
    assert source.provider == "chatgpt"
    assert source.metadata["message_ids"] == ["m1", "m2"]


def test_normalized_document_supports_chat_and_documents() -> None:
    chat = NormalizedDocument(
        document_id="chat-1",
        title="Sparse KV discussion",
        content="Discussion content",
        source=SourceReference(
            source_type="conversation",
            provider="codex",
            uri="file:///tmp/thread.json",
        ),
        domain="ai-ssd",
        project="sparse-kv",
    )
    pdf = NormalizedDocument(
        document_id="pdf-1",
        title="Sparse KV Design",
        content="Document content",
        source=SourceReference(
            source_type="document",
            provider="pdf",
            uri="s3://kb-raw/docs/sparse-kv.pdf",
            metadata={"page_count": 42},
        ),
    )

    assert chat.source.source_type is SourceType.CONVERSATION
    assert pdf.source.source_type is SourceType.DOCUMENT


def test_security_metadata_deduplicates_acl_values() -> None:
    security = SecurityMetadata(
        classification=SecurityClassification.CONFIDENTIAL,
        allowed_groups=["aissd", "aissd", "  infra  ", ""],
        sensitive_categories=["token", "token"],
    )

    assert security.allowed_groups == ["aissd", "infra"]
    assert security.sensitive_categories == ["token"]


def test_chunk_requires_non_negative_index() -> None:
    source = SourceReference(
        source_type="document",
        provider="docx",
        uri="file:///tmp/design.docx",
    )

    with pytest.raises(ValidationError):
        KnowledgeChunk(
            chunk_id="c1",
            document_id="d1",
            chunk_index=-1,
            text="hello",
            source=source,
        )


def test_document_rejects_reversed_timestamps() -> None:
    source = SourceReference(
        source_type="document",
        provider="pdf",
        uri="file:///tmp/doc.pdf",
    )

    with pytest.raises(ValidationError):
        NormalizedDocument(
            document_id="d1",
            title="doc",
            content="text",
            source=source,
            created_at=datetime(2026, 9, 11, tzinfo=timezone.utc),
            updated_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
        )


def test_knowledge_unit_tracks_version_and_evidence() -> None:
    evidence = SourceReference(
        source_type="conversation",
        provider="chatgpt",
        uri="s3://kb-raw/chatgpt/conversations/abc.json",
        source_id="abc",
        metadata={"message_ids": ["m10", "m11"]},
    )

    unit = KnowledgeUnit(
        knowledge_id="ku-2",
        knowledge_type=KnowledgeType.TROUBLESHOOTING,
        title="Abnormal LBA causes timeout",
        summary="An invalid LBA caused the SSD read path to stall.",
        status=KnowledgeStatus.CURRENT,
        root_cause="Abnormal LBA mapping",
        solution="Fail fast when the LBA exceeds the configured threshold.",
        supersedes=["ku-1"],
        evidence=[evidence],
        domain="ai-ssd",
        project="sparse-kv",
    )

    assert unit.supersedes == ["ku-1"]
    assert unit.evidence[0].provider == "chatgpt"


def test_knowledge_unit_cannot_supersede_itself() -> None:
    with pytest.raises(ValidationError):
        KnowledgeUnit(
            knowledge_id="ku-1",
            title="Title",
            summary="Summary",
            supersedes=["ku-1"],
        )


def test_retrieval_result_keeps_component_scores() -> None:
    result = RetrievalResult(
        item_type=RetrievalItemType.CHUNK,
        item_id="chunk-7",
        title="Selector timing",
        text="Relevant content",
        score=0.91,
        dense_score=0.82,
        sparse_score=4.3,
        rerank_score=0.91,
    )

    assert result.score == 0.91
    assert result.rerank_score == 0.91
