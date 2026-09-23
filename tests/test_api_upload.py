from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from kb_agent.api.routes import create_router
from kb_agent.ingestion.pipeline import IngestionSourceKind


class FakeIngestionService:
    def __init__(self) -> None:
        self.calls = []

    def ingest_file(self, path, **kwargs):
        staged = Path(path)
        self.calls.append(
            {
                "name": staged.name,
                "bytes": staged.read_bytes(),
                **kwargs,
            }
        )
        return SimpleNamespace(
            ingestion_id="ing-1",
            source_kind=IngestionSourceKind.DOCUMENT,
            source_filename=staged.name,
            raw_uri="file:///raw/test.md",
            raw_object_reused=False,
            document_count=1,
            accepted_document_count=1,
            blocked_document_count=0,
            chunk_count=1,
            indexed_chunk_count=1,
        )


def _app(service: FakeIngestionService) -> FastAPI:
    app = FastAPI()
    app.include_router(
        create_router(ingestion_service=service),
        prefix="/api/v1",
    )
    return app


def test_upload_ingest_uses_multipart_and_preserves_original_filename() -> None:
    service = FakeIngestionService()
    client = TestClient(_app(service))

    response = client.post(
        "/api/v1/ingest/upload",
        files={"file": ("test.md", b"# AI SSD\n", "text/markdown")},
        data={
            "domain": "ai-ssd",
            "project": "test",
            "source_kind": "document",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_filename"] == "test.md"
    assert payload["chunk_count"] == 1
    assert payload["indexed_chunk_count"] == 1
    assert service.calls == [
        {
            "name": "test.md",
            "bytes": b"# AI SSD\n",
            "source_kind": "document",
            "domain": "ai-ssd",
            "project": "test",
        }
    ]


def test_upload_ingest_strips_client_side_path_components() -> None:
    service = FakeIngestionService()
    client = TestClient(_app(service))

    response = client.post(
        "/api/v1/ingest/upload",
        files={"file": ("C:\\Users\\eric\\notes.md", b"hello", "text/markdown")},
    )

    assert response.status_code == 200
    assert service.calls[0]["name"] == "notes.md"
