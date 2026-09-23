"""FastAPI HTTP routes.

Only HTTP adaptation lives here. Business logic stays in ingestion/retrieval
services. Ingestion is exposed as a real multipart upload rather than a
server-local path API so browser/admin clients never need filesystem access to
the kb-agent host.
"""

from __future__ import annotations

from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from kb_agent.ingestion.pipeline import IngestionPipelineError, IngestionStage

from .schemas import IngestResponse, SearchHit, SearchRequest, SearchResponse


def create_router(
    ingestion_service: Any | None = None,
    retrieval_service: Any | None = None,
) -> APIRouter:
    router = APIRouter()

    @router.post("/ingest/upload", response_model=IngestResponse)
    def ingest_upload(
        file: UploadFile = File(...),
        domain: str | None = Form(default=None),
        project: str | None = Form(default=None),
        source_kind: str = Form(default="auto"),
    ) -> IngestResponse:
        """Upload one source file and ingest/index it synchronously.

        ``UploadFile`` provides a spooled file object, so the request does not
        need to materialize the whole upload in Python memory. The temporary
        staging copy keeps the original basename/extension because parser
        selection and raw evidence naming depend on them.
        """

        if ingestion_service is None:
            raise HTTPException(
                status_code=503,
                detail="ingestion service unavailable",
            )

        filename = _safe_upload_filename(file.filename)

        try:
            with TemporaryDirectory(prefix="kb-agent-upload-") as staging_dir:
                staging_path = Path(staging_dir) / filename
                file.file.seek(0)
                with staging_path.open("wb") as output:
                    shutil.copyfileobj(file.file, output, length=1024 * 1024)

                result = ingestion_service.ingest_file(
                    staging_path,
                    source_kind=source_kind,
                    domain=domain,
                    project=project,
                )
        except IngestionPipelineError as exc:
            raise _ingestion_http_error(exc) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"ingestion request failed: {exc}",
            ) from exc
        finally:
            file.file.close()

        resolved_source_kind = getattr(result, "source_kind", None)
        if hasattr(resolved_source_kind, "value"):
            resolved_source_kind = resolved_source_kind.value

        return IngestResponse(
            ingestion_id=getattr(result, "ingestion_id", None),
            source_kind=(
                str(resolved_source_kind)
                if resolved_source_kind is not None
                else None
            ),
            source_filename=getattr(result, "source_filename", filename),
            raw_uri=getattr(result, "raw_uri", None),
            raw_object_reused=bool(
                getattr(result, "raw_object_reused", False)
            ),
            document_count=int(getattr(result, "document_count", 0)),
            accepted_document_count=int(
                getattr(result, "accepted_document_count", 0)
            ),
            blocked_document_count=int(
                getattr(result, "blocked_document_count", 0)
            ),
            chunk_count=int(getattr(result, "chunk_count", 0)),
            indexed_chunk_count=int(
                getattr(result, "indexed_chunk_count", 0)
            ),
        )

    @router.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest) -> SearchResponse:
        if retrieval_service is None:
            raise HTTPException(
                status_code=503,
                detail="retrieval service unavailable",
            )

        hits = retrieval_service.search(
            req.query,
            filter_expr=req.filter_expr,
        )

        return SearchResponse(
            results=[
                SearchHit(
                    chunk_id=hit.chunk_id,
                    score=hit.score,
                    text=hit.text,
                    metadata=hit.metadata,
                )
                for hit in hits
            ]
        )

    return router


def _safe_upload_filename(filename: str | None) -> str:
    if not filename:
        raise HTTPException(
            status_code=400,
            detail="uploaded file must have a filename",
        )

    # Normalize both POSIX and Windows separators before taking the basename.
    normalized = filename.replace("\\", "/")
    basename = normalized.rsplit("/", 1)[-1].strip()
    if not basename or basename in {".", ".."}:
        raise HTTPException(
            status_code=400,
            detail="uploaded file has an invalid filename",
        )
    if "\x00" in basename:
        raise HTTPException(
            status_code=400,
            detail="uploaded file has an invalid filename",
        )
    return basename


def _ingestion_http_error(exc: IngestionPipelineError) -> HTTPException:
    if exc.stage is IngestionStage.VALIDATE:
        status_code = 400
    elif exc.stage is IngestionStage.PARSE:
        status_code = 422
    elif exc.stage is IngestionStage.INDEX:
        # Indexing usually means an embedding or remote Milvus dependency
        # failed after the source itself was accepted.
        status_code = 502
    else:
        status_code = 500

    return HTTPException(
        status_code=status_code,
        detail={
            "stage": exc.stage.value,
            "message": str(exc),
            "raw_uri": exc.raw_uri,
        },
    )
