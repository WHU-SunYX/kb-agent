"""Build a compact, source-attributed search result for LLM-facing MCP tools.

The FastAPI search response is the complete application/HTTP representation;
this module is an *adapter* for model context. It removes repeated text and
operational metadata without changing the search ranking or source records.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def _put_str(target: dict[str, Any], key: str, value: Any) -> None:
    if isinstance(value, str) and value.strip():
        target[key] = value


def _put_int(target: dict[str, Any], key: str, value: Any) -> None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        target[key] = value


def _source_for_hit(hit: Mapping[str, Any]) -> dict[str, Any]:
    """Retain attribution across conversation, file and other source types.

    Untrusted record metadata is only evidence, never instructions. Avoid
    copying bulky or sensitive internal metadata (including duplicate text,
    authorization records and raw storage paths) into the model's context.
    Omit unavailable fields rather than manufacturing a source title/URL.
    """
    meta = _mapping(hit.get("metadata"))
    details = _mapping(meta.get("metadata_json"))
    original = _mapping(details.get("source"))
    origin_meta = _mapping(original.get("metadata"))
    chunk = _mapping(details.get("chunk"))
    source: dict[str, Any] = {}

    for key, options in (
        ("title", (details.get("title"), meta.get("title"), original.get("title"), origin_meta.get("title"), hit.get("title"))),
        ("document_id", (meta.get("document_id"), hit.get("document_id"))),
        ("source_type", (meta.get("source_type"), original.get("source_type"))),
        ("provider", (meta.get("provider"), original.get("provider"))),
        ("source_id", (meta.get("source_id"), original.get("source_id"))),
    ):
        for value in options:
            if isinstance(value, str) and value.strip():
                _put_str(source, key, value)
                break

    for key, options in (
        ("chunk_index", (details.get("chunk_index"), chunk.get("chunk_index"))),
        ("page_number", (details.get("page_number"), details.get("page"), meta.get("page_number"))),
        ("turn_start", (chunk.get("turn_start"),)),
        ("turn_end", (chunk.get("turn_end"),)),
    ):
        for value in options:
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                _put_int(source, key, value)
                break
    section_path = details.get("section_path")
    if isinstance(section_path, list) and all(isinstance(item, str) for item in section_path):
        if section_path:
            source["section_path"] = section_path
    return source


def _compact_hit(hit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": hit["chunk_id"],
        "score": hit["score"],
        "text": hit["text"],
        "source": _source_for_hit(hit),
    }


def _serialized_length(value: dict[str, Any]) -> int:
    # A character budget is a lightweight, deterministic proxy, NOT a token
    # budget: actual tokenization is model-specific.
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def format_search_hits(
    hits: Sequence[dict[str, Any]], *, max_results: int = 5, max_chars: int = 10000
) -> dict[str, Any]:
    """Preserve ranked, complete hits where possible within an output budget.

    Stop at a hit boundary rather than silently cutting a relevant passage.
    Only when the *first* passage alone exceeds the budget do we truncate its
    text; that exceptional case is explicitly flagged. A narrower follow-up
    search can be issued whenever results are omitted.
    """
    if max_results <= 0 or max_chars <= 0:
        raise ValueError("max_results and max_chars must be positive")
    total = len(hits)
    result: dict[str, Any] = {
        "results": [],
        "total_hits": total,
        "returned_hits": 0,
        "omitted_hits": total,
        "truncated": False,
    }
    for hit in hits[:max_results]:
        compact = _compact_hit(hit)
        candidate = dict(result, results=[*result["results"], compact])
        candidate["returned_hits"] = len(candidate["results"])
        candidate["omitted_hits"] = total - candidate["returned_hits"]
        candidate["truncated"] = candidate["omitted_hits"] > 0
        if _serialized_length(candidate) <= max_chars:
            result = candidate
            continue
        if result["results"]:
            # Keep the existing evidence intact instead of clipping the next
            # hit and producing a potentially misleading partial quotation.
            break

        # A single giant hit: retain its attribution and the largest possible
        # prefix of its text. Flag loss so the agent cannot assume completeness.
        compact["text_truncated"] = True
        candidate["truncated"] = True
        text = compact["text"]
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            compact["text"] = text[:mid]
            if _serialized_length(candidate) <= max_chars:
                lo = mid
            else:
                hi = mid - 1
        compact["text"] = text[:lo]
        if _serialized_length(candidate) > max_chars:
            raise ValueError("max_chars is too small to preserve source attribution")
        result = candidate
        break

    result["truncated"] = result["truncated"] or result["omitted_hits"] > 0
    return result
