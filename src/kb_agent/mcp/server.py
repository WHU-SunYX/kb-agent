"""Read-only kb-agent MCP stdio adapter backed by the shared FastAPI process.

FastAPI is the only in-process owner of the BGE-M3 embedding backend and
Milvus retriever. Harness starts this lightweight MCP server; queries are
forwarded to the FastAPI search endpoint over localhost by default.

Run: python -m kb_agent.mcp.server
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from kb_agent.config import load_settings
from kb_agent.mcp.http_client import search_via_api
from kb_agent.mcp.search_result import format_search_hits


_settings = load_settings()
mcp = FastMCP(name=_settings.mcp.name)


@mcp.tool()
def kb_search(query: str, filter_expr: str | None = None) -> dict[str, Any]:
    """Retrieve cited evidence from indexed knowledge sources.

    Use this tool when answering questions about prior discussions, decisions,
    or documents rather than guessing from general knowledge. Cite chunk_id
    and the source title when provided. Distinguish conflicting source versions.
    Results can be omitted for context size; check omitted_hits.
    """
    hits = search_via_api(query=query, filter_expr=filter_expr)
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
