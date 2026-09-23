"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI

from .dependencies import build_application_services
from .routes import create_router


def create_app() -> FastAPI:
    services = build_application_services()

    app = FastAPI(
        title="kb-agent",
        version="0.1.0",
    )

    app.include_router(
        create_router(**services),
        prefix="/api/v1",
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()
