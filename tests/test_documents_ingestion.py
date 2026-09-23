from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from kb_agent.ingestion.documents import (
    DoclingDocumentParser,
    DocumentIngestionError,
    parse_documents,
)
from kb_agent.models import SecurityClassification, SecurityMetadata, SourceType


@dataclass
class _FakeStatus:
    value: str = "success"


class _FakeDoclingDocument:
    def __init__(self, markdown: str, *, name: str | None = None, page_count: int = 1):
        self._markdown = markdown
        self.name = name
        self.pages = {index: object() for index in range(1, page_count + 1)}

    def export_to_markdown(self) -> str:
        return self._markdown


class _FakeResult:
    def __init__(self, document, *, status: str = "success"):
        self.document = document
        self.status = _FakeStatus(status)


class _FakeConverter:
    def __init__(self, markdown_by_name: dict[str, str] | None = None):
        self.markdown_by_name = markdown_by_name or {}
        self.calls: list[Path] = []

    def convert(self, source: Path):
        source = Path(source)
        self.calls.append(source)
        markdown = self.markdown_by_name.get(
            source.name,
            f"# {source.stem}\n\nNormalized content for {source.name}",
        )
        return _FakeResult(
            _FakeDoclingDocument(markdown, name=source.stem, page_count=2),
        )


def test_parse_pdf_to_normalized_document(tmp_path: Path):
    source = tmp_path / "design.pdf"
    source.write_bytes(b"synthetic pdf bytes")
    converter = _FakeConverter({"design.pdf": "# Sparse KV Design\n\nBody text"})

    document = parse_documents(
        source,
        domain="ai-ssd",
        project="sparse-kv",
        converter=converter,
    )[0]

    assert document.title == "Sparse KV Design"
    assert document.content == "# Sparse KV Design\n\nBody text"
    assert document.domain == "ai-ssd"
    assert document.project == "sparse-kv"
    assert document.source.source_type == SourceType.DOCUMENT
    assert document.source.provider == "pdf"
    assert document.source.uri == source.resolve().as_uri()
    assert document.source.source_id == document.metadata["raw_sha256"]
    assert document.metadata["conversion_status"] == "success"
    assert document.metadata["page_count"] == 2
    assert document.metadata["mime_type"] == "application/pdf"
    assert document.content_hash is not None
    assert document.document_id.startswith("document:pdf:")


def test_raw_uri_and_security_are_preserved(tmp_path: Path):
    source = tmp_path / "spec.docx"
    source.write_bytes(b"docx")
    security = SecurityMetadata(
        tenant_id="company",
        classification=SecurityClassification.CONFIDENTIAL,
        allowed_groups=["aissd"],
    )

    document = parse_documents(
        source,
        raw_uri="s3://kb-raw/documents/spec.docx",
        security=security,
        converter=_FakeConverter(),
    )[0]

    assert document.source.uri == "s3://kb-raw/documents/spec.docx"
    assert document.source.provider == "docx"
    assert document.security.tenant_id == "company"
    assert document.security.classification == SecurityClassification.CONFIDENTIAL
    assert document.security.allowed_groups == ["aissd"]
    assert document.security is not security


def test_markdown_provider_and_title(tmp_path: Path):
    source = tmp_path / "notes.md"
    source.write_text("# Original source", encoding="utf-8")

    document = parse_documents(
        source,
        converter=_FakeConverter({"notes.md": "## Parsed title\n\nKnowledge"}),
    )[0]

    assert document.source.provider == "markdown"
    assert document.title == "Parsed title"


def test_title_falls_back_to_docling_document_name(tmp_path: Path):
    source = tmp_path / "plain.pdf"
    source.write_bytes(b"pdf")
    converter = _FakeConverter({"plain.pdf": "No markdown heading here."})

    document = parse_documents(source, converter=converter)[0]

    assert document.title == "plain"


def test_directory_ingestion_is_recursive_and_filters_extensions(tmp_path: Path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "a.pdf").write_bytes(b"a")
    (nested / "b.docx").write_bytes(b"b")
    (nested / "c.md").write_text("# c", encoding="utf-8")
    (nested / "ignore.txt").write_text("not phase-1 input", encoding="utf-8")
    converter = _FakeConverter()

    documents = DoclingDocumentParser(converter=converter).parse(tmp_path)

    assert len(documents) == 3
    assert {document.source.provider for document in documents} == {"pdf", "docx", "markdown"}
    assert [path.name for path in converter.calls] == ["a.pdf", "b.docx", "c.md"]


def test_directory_rejects_single_raw_uri(tmp_path: Path):
    (tmp_path / "a.pdf").write_bytes(b"a")

    with pytest.raises(ValueError, match="single file"):
        parse_documents(
            tmp_path,
            raw_uri="s3://kb-raw/documents/batch",
            converter=_FakeConverter(),
        )


def test_unsupported_extension_is_rejected(tmp_path: Path):
    source = tmp_path / "notes.txt"
    source.write_text("text", encoding="utf-8")

    with pytest.raises(DocumentIngestionError, match="unsupported document format"):
        parse_documents(source, converter=_FakeConverter())


def test_empty_docling_output_is_rejected(tmp_path: Path):
    source = tmp_path / "empty.pdf"
    source.write_bytes(b"pdf")

    with pytest.raises(DocumentIngestionError, match="empty content"):
        parse_documents(
            source,
            converter=_FakeConverter({"empty.pdf": "   \n"}),
        )


def test_converter_failure_is_wrapped(tmp_path: Path):
    source = tmp_path / "bad.pdf"
    source.write_bytes(b"pdf")

    class BrokenConverter:
        def convert(self, _source):
            raise RuntimeError("broken parser")

    with pytest.raises(DocumentIngestionError, match="Docling failed to convert"):
        parse_documents(source, converter=BrokenConverter())


def test_document_id_is_content_stable_across_paths(tmp_path: Path):
    left = tmp_path / "left.pdf"
    right = tmp_path / "right.pdf"
    left.write_bytes(b"same raw bytes")
    right.write_bytes(b"same raw bytes")
    converter = _FakeConverter(
        {
            "left.pdf": "# Same\n\nBody",
            "right.pdf": "# Same\n\nBody",
        }
    )

    left_doc = parse_documents(left, converter=converter)[0]
    right_doc = parse_documents(right, converter=converter)[0]

    assert left_doc.document_id == right_doc.document_id
    assert left_doc.content_hash == right_doc.content_hash
