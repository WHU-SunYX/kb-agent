"""Small stdlib HTTP adapter for MCP read tools.

Embedding, hybrid search and Milvus access stay in the long-lived FastAPI process.
The MCP stdio process never constructs a second in-process embedding model.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


_MAX_RESPONSE_BYTES = 10 * 1024 * 1024
_DEFAULT_API_BASE_URL = "http://127.0.0.1:8080"


def search_via_api(query: str, filter_expr: str | None = None) -> list[dict[str, Any]]:
    """Invoke the existing FastAPI search endpoint and return its hit list.

    KB_AGENT_API_BASE_URL is a deployment setting, not an MCP tool argument.
    Only the fixed /api/v1/search path is reachable through this adapter.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a nonempty string")
    if filter_expr is not None and not isinstance(filter_expr, str):
        raise ValueError("filter_expr must be a string or None")

    base_url = os.environ.get("KB_AGENT_API_BASE_URL", _DEFAULT_API_BASE_URL).strip()
    parsed = urlsplit(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise RuntimeError("KB_AGENT_API_BASE_URL must be an HTTP(S) origin")
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise RuntimeError("KB_AGENT_API_BASE_URL must not contain credentials/query/fragment")
    url = base_url.rstrip("/") + "/api/v1/search"

    payload = json.dumps(
        {"query": query, "filter_expr": filter_expr}, ensure_ascii=False
    ).encode("utf-8")
    request = Request(
        url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )

    try:
        with urlopen(request, timeout=180) as response:
            body = response.read(_MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        # Do not print HTTP response details: they might contain sensitive data.
        raise RuntimeError(f"kb-agent search API returned HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"kb-agent search API unavailable at {url}") from exc

    if len(body) > _MAX_RESPONSE_BYTES:
        raise RuntimeError("kb-agent search API response exceeded 10 MiB")
    try:
        document = json.loads(body)
    except (ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError("kb-agent search API returned invalid JSON") from exc
    if not isinstance(document, dict) or not isinstance(document.get("results"), list):
        raise RuntimeError("kb-agent search API returned an unexpected response shape")
    hits = document["results"]
    if not all(
        isinstance(hit, dict)
        and isinstance(hit.get("chunk_id"), str)
        and isinstance(hit.get("text"), str)
        and isinstance(hit.get("score"), (int, float))
        and isinstance(hit.get("metadata"), dict)
        for hit in hits
    ):
        raise RuntimeError("kb-agent search API returned malformed search hits")
    return hits
