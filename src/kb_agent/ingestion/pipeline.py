"""Unified ingestion orchestration for kb-agent.

Step 8 is the first layer that composes the source adapters and processing
stages built in Steps 3-7 into one reusable Python service:

    RawStore -> Parser -> SecurityScanner -> Chunker -> optional ChunkIndexer
        -> IngestionResult

The pipeline deliberately contains no CLI, HTTP, FastAPI, Milvus-specific, or
MCP code.  An optional chunk indexer is injected through a tiny protocol so the
same orchestration can remain storage-agnostic while production wiring persists
retrieval-ready chunks.

Design notes
------------
* One ingestion call represents one original file.  Directory traversal belongs
  in a caller (CLI/API) which can submit files individually, giving every raw
  artifact its own durable URI and status.
* The raw artifact is persisted *before* parsing/security processing so failed
  attempts remain auditable/retryable evidence.
* A security BLOCK applies to one normalized document, not the whole raw file.
  This matters for ChatGPT exports containing many conversations: one blocked
  conversation does not prevent safe conversations in the same export from
  being chunked.
* Parse/storage/security/chunking/indexing infrastructure failures raise a typed
  ``IngestionPipelineError`` with a stage and, when available, ``raw_uri``.
  Step 12 can map that cleanly to an ingestion-job failure.
* Raw object keys are content-addressed. Re-ingesting the same file under the
  same tenant reuses the existing object instead of writing another copy.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from hashlib import sha256
from pathlib import Path
import re
from typing import Any, Protocol, Sequence
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kb_agent.ingestion.chatgpt import ChatGPTExportParser
from kb_agent.ingestion.chunking import AutoChunker
from kb_agent.ingestion.documents import DEFAULT_SUPPORTED_EXTENSIONS, DoclingDocumentParser
from kb_agent.ingestion.security import SecurityAction, SecurityFinding, SecurityScanner
from kb_agent.models import KnowledgeChunk, NormalizedDocument, SecurityMetadata
from kb_agent.storage.raw_store import RawStore


class IngestionSourceKind(str, Enum):
    """Source family used to select the appropriate parser."""

    AUTO = "auto"
    CHATGPT = "chatgpt"
    DOCUMENT = "document"


class IngestionStage(str, Enum):
    """Pipeline stage used for structured failure reporting."""

    VALIDATE = "validate"
    STORE_RAW = "store_raw"
    PARSE = "parse"
    SECURITY = "security"
    CHUNK = "chunk"
    INDEX = "index"


class IngestionDocumentStatus(str, Enum):
    """Outcome for one normalized document inside an ingestion."""

    ACCEPTED = "accepted"
    BLOCKED = "blocked"


class IngestionPipelineError(RuntimeError):
    """Typed failure raised by an ingestion infrastructure stage."""

    def __init__(
        self,
        stage: IngestionStage,
        message: str,
        *,
        raw_uri: str | None = None,
    ) -> None:
        self.stage = stage
        self.raw_uri = raw_uri
        super().__init__(message)

    def __str__(self) -> str:
        suffix = f" [raw_uri={self.raw_uri}]" if self.raw_uri else ""
        return f"ingestion failed at {self.stage.value}: {super().__str__()}{suffix}"


class _PipelineModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class IngestionDocumentResult(_PipelineModel):
    """Safe per-document ingestion summary.

    Findings intentionally contain detector metadata but never matched secret
    values (enforced by Step 6's ``SecurityFinding`` contract).
    """

    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    status: IngestionDocumentStatus
    security_decision: SecurityAction
    findings: list[SecurityFinding] = Field(default_factory=list)
    chunk_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_status(self) -> "IngestionDocumentResult":
        if self.status is IngestionDocumentStatus.BLOCKED and self.chunk_count != 0:
            raise ValueError("blocked documents cannot contain chunks")
        if self.status is IngestionDocumentStatus.BLOCKED:
            if self.security_decision is not SecurityAction.BLOCK:
                raise ValueError("blocked status requires a block security decision")
        return self


class IngestionResult(_PipelineModel):
    """In-memory result returned by one ingestion attempt.

    ``documents`` contains sanitized/indexable documents only. ``chunks`` are
    the retrieval-ready chunks produced from those documents. Blocked documents
    are represented only by ``document_results`` so their unsafe normalized
    content cannot accidentally flow downstream.
    """

    ingestion_id: str = Field(min_length=1)
    source_kind: IngestionSourceKind
    source_filename: str = Field(min_length=1)

    raw_uri: str = Field(min_length=1)
    raw_sha256: str = Field(min_length=64, max_length=64)
    raw_size_bytes: int = Field(ge=0)
    raw_object_reused: bool = False

    documents: list[NormalizedDocument] = Field(default_factory=list)
    chunks: list[KnowledgeChunk] = Field(default_factory=list)
    document_results: list[IngestionDocumentResult] = Field(default_factory=list)

    document_count: int = Field(ge=0)
    accepted_document_count: int = Field(ge=0)
    blocked_document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    indexed_chunk_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_counts(self) -> "IngestionResult":
        if self.document_count != len(self.document_results):
            raise ValueError("document_count does not match document_results")
        if self.accepted_document_count != len(self.documents):
            raise ValueError("accepted_document_count does not match documents")
        if self.blocked_document_count != sum(
            result.status is IngestionDocumentStatus.BLOCKED
            for result in self.document_results
        ):
            raise ValueError("blocked_document_count does not match document_results")
        if self.accepted_document_count + self.blocked_document_count != self.document_count:
            raise ValueError("accepted + blocked counts must equal document_count")
        if self.chunk_count != len(self.chunks):
            raise ValueError("chunk_count does not match chunks")
        if self.indexed_chunk_count > self.chunk_count:
            raise ValueError("indexed_chunk_count cannot exceed chunk_count")
        return self


class _Parser(Protocol):
    def parse(
        self,
        source: str | Path,
        *,
        raw_uri: str | None = None,
        domain: str | None = None,
        project: str | None = None,
        security: SecurityMetadata | None = None,
    ) -> list[NormalizedDocument]:
        ...


class _SecurityScanner(Protocol):
    def scan(self, document: NormalizedDocument) -> Any:
        ...


class _Chunker(Protocol):
    def chunk(self, document: NormalizedDocument) -> list[KnowledgeChunk]:
        ...


class _ChunkIndexer(Protocol):
    def index(self, chunks: Sequence[KnowledgeChunk]) -> int:
        ...


class IngestionPipeline:
    """Reusable ingestion service independent of CLI/API/MCP entry points."""

    CHATGPT_EXTENSIONS = frozenset({".json", ".zip"})

    def __init__(
        self,
        *,
        raw_store: RawStore,
        chatgpt_parser: _Parser | None = None,
        document_parser: _Parser | None = None,
        security_scanner: _SecurityScanner | None = None,
        chunker: _Chunker | None = None,
        indexer: _ChunkIndexer | None = None,
    ) -> None:
        self.raw_store = raw_store
        self.chatgpt_parser = chatgpt_parser or ChatGPTExportParser()
        self.document_parser = document_parser or DoclingDocumentParser()
        self.security_scanner = security_scanner or SecurityScanner()
        self.chunker = chunker or AutoChunker()
        self.indexer = indexer

    def ingest_file(
        self,
        source: str | Path,
        *,
        source_kind: IngestionSourceKind | str = IngestionSourceKind.AUTO,
        domain: str | None = None,
        project: str | None = None,
        security: SecurityMetadata | None = None,
        ingestion_id: str | None = None,
    ) -> IngestionResult:
        """Ingest one local raw file through parse/security/chunking stages.

        The local ``source`` is only the staging path. The parser receives the
        durable URI returned by ``RawStore`` so every normalized document and
        chunk points back to immutable-ish raw evidence instead of ``/tmp``.
        """

        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise IngestionPipelineError(
                IngestionStage.VALIDATE,
                f"source file not found: {path}",
            )
        if not path.is_file():
            raise IngestionPipelineError(
                IngestionStage.VALIDATE,
                "IngestionPipeline.ingest_file accepts one file at a time; "
                "submit directory contents individually",
            )

        try:
            kind = IngestionSourceKind(source_kind)
        except ValueError as exc:
            raise IngestionPipelineError(
                IngestionStage.VALIDATE,
                f"unsupported source_kind: {source_kind!r}",
            ) from exc

        resolved_kind = self._resolve_source_kind(path, kind)
        raw_hash, raw_size = _sha256_file(path)
        safe_security = (security or SecurityMetadata()).model_copy(deep=True)
        raw_key = _raw_object_key(
            path,
            source_kind=resolved_kind,
            tenant_id=safe_security.tenant_id,
            raw_sha256=raw_hash,
        )

        raw_uri = self.raw_store.uri_for(raw_key)
        try:
            raw_reused = self.raw_store.exists(raw_uri)
            if not raw_reused:
                stored_uri = self.raw_store.put_file(raw_key, path)
                if stored_uri != raw_uri:
                    raise IngestionPipelineError(
                        IngestionStage.STORE_RAW,
                        "RawStore returned a URI different from uri_for(key)",
                        raw_uri=stored_uri,
                    )
        except IngestionPipelineError:
            raise
        except Exception as exc:
            raise IngestionPipelineError(
                IngestionStage.STORE_RAW,
                str(exc),
                raw_uri=raw_uri,
            ) from exc

        parser = (
            self.chatgpt_parser
            if resolved_kind is IngestionSourceKind.CHATGPT
            else self.document_parser
        )
        try:
            parsed_documents = parser.parse(
                path,
                raw_uri=raw_uri,
                domain=domain,
                project=project,
                security=safe_security,
            )
        except Exception as exc:
            raise IngestionPipelineError(
                IngestionStage.PARSE,
                str(exc),
                raw_uri=raw_uri,
            ) from exc

        if not parsed_documents:
            raise IngestionPipelineError(
                IngestionStage.PARSE,
                "source parser returned no normalized documents",
                raw_uri=raw_uri,
            )

        accepted_documents: list[NormalizedDocument] = []
        chunks: list[KnowledgeChunk] = []
        summaries: list[IngestionDocumentResult] = []

        for document in parsed_documents:
            try:
                scan = self.security_scanner.scan(document)
            except Exception as exc:
                raise IngestionPipelineError(
                    IngestionStage.SECURITY,
                    f"document {document.document_id}: {exc}",
                    raw_uri=raw_uri,
                ) from exc

            if scan.blocked:
                summaries.append(
                    IngestionDocumentResult(
                        document_id=document.document_id,
                        title=document.title,
                        status=IngestionDocumentStatus.BLOCKED,
                        security_decision=scan.decision,
                        findings=scan.findings,
                        chunk_count=0,
                    )
                )
                continue

            sanitized = scan.document
            if sanitized is None:  # Defensive against custom scanner implementations.
                raise IngestionPipelineError(
                    IngestionStage.SECURITY,
                    f"document {document.document_id}: non-blocked scan returned no document",
                    raw_uri=raw_uri,
                )

            try:
                document_chunks = self.chunker.chunk(sanitized)
            except Exception as exc:
                raise IngestionPipelineError(
                    IngestionStage.CHUNK,
                    f"document {sanitized.document_id}: {exc}",
                    raw_uri=raw_uri,
                ) from exc

            if not document_chunks:
                raise IngestionPipelineError(
                    IngestionStage.CHUNK,
                    f"document {sanitized.document_id}: chunker returned no chunks",
                    raw_uri=raw_uri,
                )

            accepted_documents.append(sanitized)
            chunks.extend(document_chunks)
            summaries.append(
                IngestionDocumentResult(
                    document_id=sanitized.document_id,
                    title=sanitized.title,
                    status=IngestionDocumentStatus.ACCEPTED,
                    security_decision=scan.decision,
                    findings=scan.findings,
                    chunk_count=len(document_chunks),
                )
            )

        blocked_count = sum(
            summary.status is IngestionDocumentStatus.BLOCKED for summary in summaries
        )

        indexed_chunk_count = 0
        if self.indexer is not None and chunks:
            try:
                indexed_chunk_count = self.indexer.index(chunks)
            except Exception as exc:
                raise IngestionPipelineError(
                    IngestionStage.INDEX,
                    str(exc),
                    raw_uri=raw_uri,
                ) from exc
            if indexed_chunk_count != len(chunks):
                raise IngestionPipelineError(
                    IngestionStage.INDEX,
                    "chunk indexer persisted a different number of chunks than expected: "
                    f"{indexed_chunk_count} != {len(chunks)}",
                    raw_uri=raw_uri,
                )

        return IngestionResult(
            ingestion_id=ingestion_id or str(uuid4()),
            source_kind=resolved_kind,
            source_filename=path.name,
            raw_uri=raw_uri,
            raw_sha256=raw_hash,
            raw_size_bytes=raw_size,
            raw_object_reused=raw_reused,
            documents=accepted_documents,
            chunks=chunks,
            document_results=summaries,
            document_count=len(summaries),
            accepted_document_count=len(accepted_documents),
            blocked_document_count=blocked_count,
            chunk_count=len(chunks),
            indexed_chunk_count=indexed_chunk_count,
        )

    def ingest_files(
        self,
        sources: Iterable[str | Path],
        *,
        source_kind: IngestionSourceKind | str = IngestionSourceKind.AUTO,
        domain: str | None = None,
        project: str | None = None,
        security: SecurityMetadata | None = None,
    ) -> list[IngestionResult]:
        """Sequentially ingest many files using the same metadata/policy context.

        The method intentionally fails fast. Step 12's job layer can decide
        whether batch uploads should continue after a per-file failure while
        preserving one independently traceable ingestion result per raw file.
        """

        return [
            self.ingest_file(
                source,
                source_kind=source_kind,
                domain=domain,
                project=project,
                security=security,
            )
            for source in sources
        ]

    def _resolve_source_kind(
        self,
        path: Path,
        requested: IngestionSourceKind,
    ) -> IngestionSourceKind:
        extension = path.suffix.lower()

        if requested is IngestionSourceKind.AUTO:
            if extension in self.CHATGPT_EXTENSIONS:
                return IngestionSourceKind.CHATGPT
            if extension in DEFAULT_SUPPORTED_EXTENSIONS:
                return IngestionSourceKind.DOCUMENT
            supported = sorted(self.CHATGPT_EXTENSIONS | DEFAULT_SUPPORTED_EXTENSIONS)
            raise IngestionPipelineError(
                IngestionStage.VALIDATE,
                f"cannot infer source kind for extension {extension or '<none>'!r}; "
                f"supported extensions: {', '.join(supported)}",
            )

        if requested is IngestionSourceKind.CHATGPT and extension not in self.CHATGPT_EXTENSIONS:
            raise IngestionPipelineError(
                IngestionStage.VALIDATE,
                f"ChatGPT ingestion expects .json or .zip, got {extension or '<none>'!r}",
            )
        if requested is IngestionSourceKind.DOCUMENT and extension not in DEFAULT_SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(DEFAULT_SUPPORTED_EXTENSIONS))
            raise IngestionPipelineError(
                IngestionStage.VALIDATE,
                f"document ingestion does not support {extension or '<none>'!r}; "
                f"supported extensions: {supported}",
            )
        return requested


def _sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> tuple[str, int]:
    digest = sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(block_size)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


_KEY_SEGMENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_key_segment(value: str, *, fallback: str) -> str:
    normalized = _KEY_SEGMENT_RE.sub("-", value.strip()).strip("-._")
    return normalized[:96] or fallback


def _safe_filename(path: Path) -> str:
    stem = _safe_key_segment(path.stem, fallback="source")
    suffix = path.suffix.lower()
    suffix = suffix if re.fullmatch(r"\.[A-Za-z0-9]{1,16}", suffix) else ""
    return f"{stem}{suffix}"


def _raw_object_key(
    path: Path,
    *,
    source_kind: IngestionSourceKind,
    tenant_id: str,
    raw_sha256: str,
) -> str:
    tenant = _safe_key_segment(tenant_id, fallback="default")
    return (
        f"tenants/{tenant}/sources/{source_kind.value}/"
        f"{raw_sha256[:2]}/{raw_sha256}/{_safe_filename(path)}"
    )


__all__ = [
    "IngestionDocumentResult",
    "IngestionDocumentStatus",
    "IngestionPipeline",
    "IngestionPipelineError",
    "IngestionResult",
    "IngestionSourceKind",
    "IngestionStage",
]
