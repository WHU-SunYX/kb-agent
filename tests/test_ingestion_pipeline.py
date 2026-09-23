from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from kb_agent.ingestion.chunking import AutoChunker, ChunkingConfig
from kb_agent.ingestion.pipeline import (
    IngestionDocumentStatus,
    IngestionPipeline,
    IngestionPipelineError,
    IngestionSourceKind,
    IngestionStage,
)
from kb_agent.ingestion.security import (
    DetectedItem,
    SecurityAction,
    SecurityFindingKind,
    SecurityPolicy,
    SecurityScanner,
)
from kb_agent.models import NormalizedDocument, SecurityMetadata, SourceReference, SourceType
from kb_agent.storage.raw_store import LocalRawStore


class EmptyDetector:
    def detect(self, text: str, *, field_path: str) -> list[DetectedItem]:
        return []


class SecretDetector:
    def detect(self, text: str, *, field_path: str) -> list[DetectedItem]:
        if "BLOCKME" not in text:
            return []
        return [
            DetectedItem(
                kind=SecurityFindingKind.SECRET,
                category="secret:test",
                detector="test-secret",
                field_path=field_path,
            )
        ]


class WordCounter:
    method = "word_test"

    def count(self, text: str) -> int:
        return len(text.split())


class FakeDocumentParser:
    def __init__(self, *, content: str = "# Title\n\nbody text") -> None:
        self.content = content
        self.calls: list[dict] = []

    def parse(self, source, *, raw_uri=None, domain=None, project=None, security=None):
        self.calls.append(
            {
                "source": Path(source),
                "raw_uri": raw_uri,
                "domain": domain,
                "project": project,
                "security": security,
            }
        )
        return [
            NormalizedDocument(
                document_id="doc-1",
                title="Title",
                content=self.content,
                source=SourceReference(
                    source_type=SourceType.DOCUMENT,
                    provider="pdf",
                    uri=raw_uri,
                ),
                domain=domain,
                project=project,
                security=security or SecurityMetadata(),
            )
        ]


class MultiConversationParser:
    def parse(self, source, *, raw_uri=None, domain=None, project=None, security=None):
        base = dict(domain=domain, project=project, security=security or SecurityMetadata())
        return [
            NormalizedDocument(
                document_id="chat:safe",
                title="Safe",
                content="## User\n\nhello\n\n## Assistant\n\nworld",
                source=SourceReference(
                    source_type=SourceType.CONVERSATION,
                    provider="chatgpt",
                    uri=raw_uri,
                ),
                metadata={
                    "turns": [
                        {"role": "user", "text": "hello", "message_id": "u1"},
                        {"role": "assistant", "text": "world", "message_id": "a1"},
                    ]
                },
                **base,
            ),
            NormalizedDocument(
                document_id="chat:blocked",
                title="Blocked",
                content="## User\n\nBLOCKME",
                source=SourceReference(
                    source_type=SourceType.CONVERSATION,
                    provider="chatgpt",
                    uri=raw_uri,
                ),
                metadata={
                    "turns": [
                        {"role": "user", "text": "BLOCKME", "message_id": "u2"},
                    ]
                },
                **base,
            ),
        ]


class ExplodingParser:
    def parse(self, *args, **kwargs):
        raise ValueError("bad parse")


class ExplodingScanner:
    def scan(self, document):
        raise RuntimeError("scan failure")


class ExplodingChunker:
    def chunk(self, document):
        raise RuntimeError("chunk failure")


def make_scanner(*, secret: bool = False) -> SecurityScanner:
    empty = EmptyDetector()
    return SecurityScanner(
        policy=SecurityPolicy(secret_action=SecurityAction.BLOCK),
        secret_detector=SecretDetector() if secret else empty,
        pii_detector=empty,
        context_detector=empty,
    )


def make_chunker() -> AutoChunker:
    return AutoChunker(
        config=ChunkingConfig(max_tokens=64, overlap_tokens=0, conversation_overlap_turns=0),
        token_counter=WordCounter(),
    )


def test_document_pipeline_stores_raw_parses_scans_and_chunks(tmp_path: Path) -> None:
    source = tmp_path / "design.pdf"
    source.write_bytes(b"fake-pdf")
    raw_store = LocalRawStore(tmp_path / "raw")
    parser = FakeDocumentParser()
    security = SecurityMetadata(tenant_id="company", allowed_groups=["aissd"])
    pipeline = IngestionPipeline(
        raw_store=raw_store,
        document_parser=parser,
        security_scanner=make_scanner(),
        chunker=make_chunker(),
    )

    result = pipeline.ingest_file(
        source,
        domain="ai-ssd",
        project="sparse-kv",
        security=security,
        ingestion_id="ing-1",
    )

    assert result.ingestion_id == "ing-1"
    assert result.source_kind is IngestionSourceKind.DOCUMENT
    assert result.raw_sha256 == sha256(b"fake-pdf").hexdigest()
    assert result.raw_size_bytes == len(b"fake-pdf")
    assert result.raw_object_reused is False
    assert result.document_count == 1
    assert result.accepted_document_count == 1
    assert result.blocked_document_count == 0
    assert result.chunk_count == len(result.chunks) >= 1
    assert result.documents[0].domain == "ai-ssd"
    assert result.documents[0].project == "sparse-kv"
    assert result.documents[0].security.allowed_groups == ["aissd"]
    assert result.documents[0].source.uri == result.raw_uri
    assert parser.calls[0]["raw_uri"] == result.raw_uri
    assert raw_store.exists(result.raw_uri)


def test_reingesting_same_raw_file_reuses_content_addressed_object(tmp_path: Path) -> None:
    source = tmp_path / "notes.md"
    source.write_text("hello", encoding="utf-8")
    pipeline = IngestionPipeline(
        raw_store=LocalRawStore(tmp_path / "raw"),
        document_parser=FakeDocumentParser(content="# T\n\nhello"),
        security_scanner=make_scanner(),
        chunker=make_chunker(),
    )

    first = pipeline.ingest_file(source)
    second = pipeline.ingest_file(source)

    assert first.raw_uri == second.raw_uri
    assert first.raw_object_reused is False
    assert second.raw_object_reused is True


def test_chatgpt_export_can_block_one_conversation_without_blocking_others(tmp_path: Path) -> None:
    source = tmp_path / "export.zip"
    source.write_bytes(b"not-a-real-zip-needed-by-fake-parser")
    pipeline = IngestionPipeline(
        raw_store=LocalRawStore(tmp_path / "raw"),
        chatgpt_parser=MultiConversationParser(),
        security_scanner=make_scanner(secret=True),
        chunker=make_chunker(),
    )

    result = pipeline.ingest_file(source, source_kind="chatgpt")

    assert result.source_kind is IngestionSourceKind.CHATGPT
    assert result.document_count == 2
    assert result.accepted_document_count == 1
    assert result.blocked_document_count == 1
    assert [doc.document_id for doc in result.documents] == ["chat:safe"]
    statuses = {item.document_id: item for item in result.document_results}
    assert statuses["chat:safe"].status is IngestionDocumentStatus.ACCEPTED
    assert statuses["chat:blocked"].status is IngestionDocumentStatus.BLOCKED
    assert statuses["chat:blocked"].chunk_count == 0
    assert all(chunk.document_id == "chat:safe" for chunk in result.chunks)
    assert "BLOCKME" not in result.model_dump_json()


def test_auto_source_detection_supports_chatgpt_and_documents(tmp_path: Path) -> None:
    raw = LocalRawStore(tmp_path / "raw")
    pipeline = IngestionPipeline(
        raw_store=raw,
        chatgpt_parser=MultiConversationParser(),
        document_parser=FakeDocumentParser(),
        security_scanner=make_scanner(),
        chunker=make_chunker(),
    )
    chat = tmp_path / "export.json"
    chat.write_text("{}", encoding="utf-8")
    doc = tmp_path / "x.docx"
    doc.write_bytes(b"docx")

    assert pipeline.ingest_file(chat).source_kind is IngestionSourceKind.CHATGPT
    assert pipeline.ingest_file(doc).source_kind is IngestionSourceKind.DOCUMENT


def test_unsupported_extension_fails_before_raw_storage(tmp_path: Path) -> None:
    source = tmp_path / "x.txt"
    source.write_text("x", encoding="utf-8")
    raw_root = tmp_path / "raw"
    pipeline = IngestionPipeline(
        raw_store=LocalRawStore(raw_root),
        security_scanner=make_scanner(),
    )

    with pytest.raises(IngestionPipelineError) as exc_info:
        pipeline.ingest_file(source)

    assert exc_info.value.stage is IngestionStage.VALIDATE
    assert list(raw_root.rglob("*")) == []


def test_directory_input_is_rejected_so_every_file_gets_its_own_raw_uri(tmp_path: Path) -> None:
    pipeline = IngestionPipeline(
        raw_store=LocalRawStore(tmp_path / "raw"),
        security_scanner=make_scanner(),
    )

    with pytest.raises(IngestionPipelineError) as exc_info:
        pipeline.ingest_file(tmp_path)

    assert exc_info.value.stage is IngestionStage.VALIDATE


def test_parse_failure_keeps_raw_uri_for_retry_and_audit(tmp_path: Path) -> None:
    source = tmp_path / "bad.pdf"
    source.write_bytes(b"bad")
    store = LocalRawStore(tmp_path / "raw")
    pipeline = IngestionPipeline(
        raw_store=store,
        document_parser=ExplodingParser(),
        security_scanner=make_scanner(),
    )

    with pytest.raises(IngestionPipelineError) as exc_info:
        pipeline.ingest_file(source)

    error = exc_info.value
    assert error.stage is IngestionStage.PARSE
    assert error.raw_uri is not None
    assert store.exists(error.raw_uri)
    assert "bad parse" in str(error)


def test_security_failure_is_wrapped_with_stage_and_raw_uri(tmp_path: Path) -> None:
    source = tmp_path / "x.md"
    source.write_text("x", encoding="utf-8")
    pipeline = IngestionPipeline(
        raw_store=LocalRawStore(tmp_path / "raw"),
        document_parser=FakeDocumentParser(),
        security_scanner=ExplodingScanner(),
        chunker=make_chunker(),
    )

    with pytest.raises(IngestionPipelineError) as exc_info:
        pipeline.ingest_file(source)

    assert exc_info.value.stage is IngestionStage.SECURITY
    assert exc_info.value.raw_uri is not None


def test_chunk_failure_is_wrapped_with_stage_and_raw_uri(tmp_path: Path) -> None:
    source = tmp_path / "x.md"
    source.write_text("x", encoding="utf-8")
    pipeline = IngestionPipeline(
        raw_store=LocalRawStore(tmp_path / "raw"),
        document_parser=FakeDocumentParser(),
        security_scanner=make_scanner(),
        chunker=ExplodingChunker(),
    )

    with pytest.raises(IngestionPipelineError) as exc_info:
        pipeline.ingest_file(source)

    assert exc_info.value.stage is IngestionStage.CHUNK
    assert exc_info.value.raw_uri is not None


def test_ingest_files_preserves_input_order_and_metadata(tmp_path: Path) -> None:
    first = tmp_path / "a.md"
    second = tmp_path / "b.md"
    first.write_text("a", encoding="utf-8")
    second.write_text("b", encoding="utf-8")
    pipeline = IngestionPipeline(
        raw_store=LocalRawStore(tmp_path / "raw"),
        document_parser=FakeDocumentParser(),
        security_scanner=make_scanner(),
        chunker=make_chunker(),
    )

    results = pipeline.ingest_files(
        [first, second],
        domain="ai-ssd",
        project="dma",
    )

    assert [result.source_filename for result in results] == ["a.md", "b.md"]
    assert all(result.documents[0].domain == "ai-ssd" for result in results)
    assert all(result.documents[0].project == "dma" for result in results)
