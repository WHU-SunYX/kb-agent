"""General document ingestion through Docling.

This module is intentionally a thin adapter around Docling.  kb-agent does not
implement PDF/DOCX/Markdown parsing itself; it converts Docling's document model
into the source-neutral :class:`~kb_agent.models.NormalizedDocument` contract.

Docling is an optional dependency so local development that only works with
ChatGPT exports does not need to install the document-processing stack. Install
it with::

    pip install -e ".[documents]"

or, for development plus document ingestion::

    pip install -e ".[dev,documents]"
"""

from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256
import mimetypes
from pathlib import Path
import re
from typing import Any

from kb_agent.models import (
    NormalizedDocument,
    SecurityMetadata,
    SourceReference,
    SourceType,
)


class DocumentIngestionError(RuntimeError):
    """Raised when a general document cannot be normalized safely."""


# Phase-1 formats.  Docling supports many more formats, but keeping the adapter
# whitelist explicit prevents kb-agent from silently accepting formats we have
# not yet covered with tests and policy decisions.
DEFAULT_SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".md", ".markdown"})

_PROVIDER_BY_EXTENSION = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".md": "markdown",
    ".markdown": "markdown",
}

_HEADING_RE = re.compile(r"^\s*#{1,6}\s+(.+?)\s*#*\s*$")


class DoclingDocumentParser:
    """Convert local documents to :class:`NormalizedDocument` using Docling.

    Parameters
    ----------
    converter:
        Optional Docling-compatible converter.  Supplying one is useful for
        tests or for applications that want to reuse a preconfigured
        ``DocumentConverter`` instance.  When omitted, Docling is imported
        lazily on the first conversion.
    supported_extensions:
        Optional iterable of filename extensions.  Phase 1 defaults to PDF,
        DOCX, and Markdown.
    """

    def __init__(
        self,
        *,
        converter: Any | None = None,
        supported_extensions: Iterable[str] | None = None,
    ) -> None:
        self._converter = converter
        configured = supported_extensions or DEFAULT_SUPPORTED_EXTENSIONS
        normalized = {_normalize_extension(value) for value in configured}
        if not normalized:
            raise ValueError("supported_extensions must not be empty")
        self.supported_extensions = frozenset(normalized)

    def parse(
        self,
        source: str | Path,
        *,
        raw_uri: str | None = None,
        domain: str | None = None,
        project: str | None = None,
        security: SecurityMetadata | None = None,
    ) -> list[NormalizedDocument]:
        """Parse one document or all supported documents below a directory.

        ``raw_uri`` identifies the immutable artifact in ``RawStore`` and is
        therefore valid only for a single input file.  Directory ingestion must
        let each file keep its own URI (or, later, be orchestrated file-by-file
        by the ingestion pipeline).
        """

        path = Path(source).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"document source not found: {path}")

        if path.is_file():
            return [
                self._parse_file(
                    path,
                    raw_uri=raw_uri,
                    domain=domain,
                    project=project,
                    security=security,
                )
            ]

        if not path.is_dir():
            raise DocumentIngestionError(f"document source is neither file nor directory: {path}")
        if raw_uri is not None:
            raise ValueError("raw_uri can only be used when source is a single file")

        files = sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in self.supported_extensions
        )
        if not files:
            supported = ", ".join(sorted(self.supported_extensions))
            raise DocumentIngestionError(
                f"no supported documents found under {path}; supported extensions: {supported}"
            )

        return [
            self._parse_file(
                candidate,
                raw_uri=None,
                domain=domain,
                project=project,
                security=security,
            )
            for candidate in files
        ]

    def _parse_file(
        self,
        path: Path,
        *,
        raw_uri: str | None,
        domain: str | None,
        project: str | None,
        security: SecurityMetadata | None,
    ) -> NormalizedDocument:
        extension = path.suffix.lower()
        if extension not in self.supported_extensions:
            supported = ", ".join(sorted(self.supported_extensions))
            raise DocumentIngestionError(
                f"unsupported document format {extension or '<none>'!r} for {path}; "
                f"supported extensions: {supported}"
            )

        raw_bytes = path.read_bytes()
        raw_hash = sha256(raw_bytes).hexdigest()
        provider = _PROVIDER_BY_EXTENSION.get(extension, extension.lstrip(".") or "document")

        converter = self._get_converter()
        try:
            result = converter.convert(path)
        except Exception as exc:  # Docling uses format/pipeline-specific errors.
            raise DocumentIngestionError(f"Docling failed to convert {path}: {exc}") from exc

        document = getattr(result, "document", None)
        if document is None:
            raise DocumentIngestionError(f"Docling returned no document for {path}")

        try:
            content = document.export_to_markdown().strip()
        except Exception as exc:
            raise DocumentIngestionError(
                f"Docling failed to export normalized Markdown for {path}: {exc}"
            ) from exc
        if not content:
            raise DocumentIngestionError(f"Docling produced empty content for {path}")

        content_hash = sha256(content.encode("utf-8")).hexdigest()
        source_uri = raw_uri or path.resolve().as_uri()
        title = _extract_title(content) or _document_name(document) or path.stem

        source_metadata: dict[str, Any] = {
            "filename": path.name,
            "extension": extension,
            "raw_sha256": raw_hash,
            "size_bytes": len(raw_bytes),
        }
        mime_type, _ = mimetypes.guess_type(path.name)
        if mime_type:
            source_metadata["mime_type"] = mime_type

        metadata: dict[str, Any] = {
            "format": "docling",
            "schema_version": 1,
            "filename": path.name,
            "extension": extension,
            "provider": provider,
            "raw_sha256": raw_hash,
            "size_bytes": len(raw_bytes),
        }
        if mime_type:
            metadata["mime_type"] = mime_type

        status = _enum_or_string(getattr(result, "status", None))
        if status:
            metadata["conversion_status"] = status

        page_count = _page_count(document)
        if page_count is not None:
            metadata["page_count"] = page_count

        docling_name = _document_name(document)
        if docling_name:
            metadata["docling_document_name"] = docling_name

        source_reference = SourceReference(
            source_type=SourceType.DOCUMENT,
            provider=provider,
            uri=source_uri,
            source_id=raw_hash,
            metadata=source_metadata,
        )

        return NormalizedDocument(
            document_id=f"document:{provider}:{raw_hash[:24]}",
            title=title,
            content=content,
            source=source_reference,
            domain=domain,
            project=project,
            content_hash=content_hash,
            metadata=metadata,
            security=(security or SecurityMetadata()).model_copy(deep=True),
        )

    def _get_converter(self) -> Any:
        if self._converter is not None:
            return self._converter

        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:
            raise DocumentIngestionError(
                "Docling is required for general document ingestion. "
                "Install kb-agent with the 'documents' extra, for example: "
                "pip install -e '.[dev,documents]'"
            ) from exc

        self._converter = DocumentConverter()
        return self._converter


def parse_documents(
    source: str | Path,
    *,
    raw_uri: str | None = None,
    domain: str | None = None,
    project: str | None = None,
    security: SecurityMetadata | None = None,
    converter: Any | None = None,
) -> list[NormalizedDocument]:
    """Convenience wrapper around :class:`DoclingDocumentParser`."""

    return DoclingDocumentParser(converter=converter).parse(
        source,
        raw_uri=raw_uri,
        domain=domain,
        project=project,
        security=security,
    )


def _normalize_extension(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError("document extension must not be blank")
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    return normalized


def _extract_title(markdown: str) -> str | None:
    for line in markdown.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            title = match.group(1).strip()
            if title:
                return title
    return None


def _document_name(document: Any) -> str | None:
    name = getattr(document, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _page_count(document: Any) -> int | None:
    pages = getattr(document, "pages", None)
    if pages is None:
        return None
    try:
        return len(pages)
    except TypeError:
        return None


def _enum_or_string(value: Any) -> str | None:
    if value is None:
        return None
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str) and enum_value:
        return enum_value
    text = str(value).strip()
    return text or None
