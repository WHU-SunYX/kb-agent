"""Read-only kb-agent MCP stdio adapter backed by the shared FastAPI process.

FastAPI is the only in-process owner of the BGE-M3 embedding backend and
Milvus retriever. Harness starts this lightweight MCP server; queries are
forwarded to the FastAPI search endpoint over localhost by default.

Run: python -m kb_agent.mcp.server
"""

from __future__ import annotations

from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from kb_agent.config import load_settings
from kb_agent.mcp.http_client import search_via_api
from kb_agent.models import SecurityClassification, SourceType
from kb_agent.mcp.search_result import format_search_hits


_settings = load_settings()
mcp = FastMCP(name=_settings.mcp.name)


@mcp.tool()
def kb_search(
    query: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Natural-language evidence query. Do not put database filter "
                "syntax in this field."
            ),
        ),
    ],
    document_id: Annotated[
        str | None,
        Field(
            max_length=128,
            description="Exact indexed document_id, only when already known.",
        ),
    ] = None,
    domain: Annotated[
        str | None,
        Field(max_length=128, description="Exact knowledge domain, only when known."),
    ] = None,
    project: Annotated[
        str | None,
        Field(max_length=128, description="Exact project metadata value, only when known."),
    ] = None,
    classification: Annotated[
        SecurityClassification | None,
        Field(description="Exact security classification metadata value."),
    ] = None,
    source_type: Annotated[
        SourceType | None,
        Field(description="Exact indexed source type, such as document or conversation."),
    ] = None,
    provider: Annotated[
        str | None,
        Field(max_length=128, description="Exact source provider, only when known."),
    ] = None,
    source_id: Annotated[
        str | None,
        Field(max_length=256, description="Exact provider source_id, only when known."),
    ] = None,
) -> dict[str, Any]:
    """Retrieve cited evidence from indexed knowledge sources.

    Use natural-language ``query`` for the semantic/full-text search. Optional
    fields are exact-match metadata constraints backed by the KB schema; leave
    them unset unless the requested scope is known. Do not invent database
    expressions or access-control fields. Cite chunk_id and source title when
    provided, and distinguish conflicting source versions.
    """
    hits = search_via_api(
        query=query,
        document_id=document_id,
        domain=domain,
        project=project,
        classification=classification,
        source_type=source_type,
        provider=provider,
        source_id=source_id,
    )
    return format_search_hits(
        hits,
        max_results=_settings.mcp.search_max_results,
        max_chars=_settings.mcp.search_max_chars,
    )


@mcp.tool()
def kb_get_document(document_id: str) -> dict[str, str]:
    """Reserved read-only tool; full document lookup is not implemented yet."""
    return {"document_id": document_id, "status": "not_implemented"}


if __name__ == "__main__":
    mcp.run()
