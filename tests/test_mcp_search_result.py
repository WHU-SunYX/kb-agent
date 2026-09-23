"""MCP answer-context formatting: source fidelity and bounded evidence."""

from __future__ import annotations

import json

import pytest

from kb_agent.mcp.search_result import format_search_hits


def _hit(i: int, text: str, metadata: dict | None = None) -> dict:
    return {
        "chunk_id": f"chunk-{i}",
        "score": 0.9 - i / 100,
        "text": text,
        "metadata": {} if metadata is None else metadata,
    }


def _size(value: dict) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def test_source_metadata_is_compact_and_complete_for_attribution():
    meta = {
        "chunk_id": "chunk-0",
        "score": 0.9,
        "text": "duplicate body should not reach model",
        "document_id": "doc-a",
        "source_type": "conversation",
        "provider": "chatgpt",
        "source_id": "conversation-a",
        "security": {"allowed_users": ["secret"]},
        "metadata_json": {
            "title": "Design discussion",
            "chunk_index": 25,
            "section_path": ["Architecture", "DMA"],
            "source": {"uri": "file:///private/internal/path"},
            "chunk": {"turn_start": 3, "turn_end": 5},
        },
    }
    out = format_search_hits([_hit(0, "actual body", meta)])
    assert out == {
        "results": [{
            "chunk_id": "chunk-0",
            "citation": "《Design discussion》 [chunk_id: chunk-0]",
            "score": 0.9, "text": "actual body",
            "source": {
                "title": "Design discussion", "document_id": "doc-a",
                "source_type": "conversation", "provider": "chatgpt",
                "source_id": "conversation-a", "chunk_index": 25,
                "turn_start": 3, "turn_end": 5,
                "section_path": ["Architecture", "DMA"],
            },
        }],
        "total_hits": 1, "returned_hits": 1,
        "omitted_hits": 0, "truncated": False,
    }
    payload = json.dumps(out)
    assert "secret" not in payload
    assert "file:///" not in payload
    assert "duplicate body" not in payload


def test_generic_document_metadata_no_made_up_title():
    out = format_search_hits([_hit(0, "paper content", {
        "document_id": "report-1", "provider": "gdrive",
        "metadata_json": {"page_number": 11, "source": {"metadata": {"title": "Report"}}},
    }), _hit(1, "uncited document")])
    assert out["results"][0]["source"] == {
        "title": "Report", "document_id": "report-1",
        "provider": "gdrive", "page_number": 11,
    }
    assert "title" not in out["results"][1]["source"]


def test_max_results_is_configurable_and_preserves_order():
    out = format_search_hits([_hit(i, "short") for i in range(8)], max_results=3)
    assert [h["chunk_id"] for h in out["results"]] == ["chunk-0", "chunk-1", "chunk-2"]
    assert (out["total_hits"], out["returned_hits"], out["omitted_hits"]) == (8, 3, 5)
    assert out["truncated"] is True


def test_context_budget_drops_whole_later_hits_instead_of_partial_quote():
    hits = [_hit(0, "A" * 600), _hit(1, "B" * 600), _hit(2, "C" * 600)]
    one = format_search_hits(hits, max_results=1, max_chars=3000)
    budget = _size(one) + 20
    out = format_search_hits(hits, max_results=3, max_chars=budget)
    assert out["returned_hits"] == 1
    assert out["omitted_hits"] == 2
    assert out["results"][0]["text"] == hits[0]["text"]
    assert "text_truncated" not in out["results"][0]
    assert _size(out) <= budget


def test_giant_first_hit_truncation_is_explicit_and_bounded():
    out = format_search_hits([_hit(0, "字\n\\\"" * 10000), _hit(1, "short")], max_chars=2048)
    assert out["returned_hits"] == 1
    assert out["omitted_hits"] == 1
    assert out["results"][0]["text_truncated"] is True
    assert out["results"][0]["text"]
    assert out["truncated"] is True
    assert _size(out) <= 2048


def test_no_results_and_validation():
    assert format_search_hits([]) == {
        "results": [], "total_hits": 0, "returned_hits": 0,
        "omitted_hits": 0, "truncated": False,
    }
    with pytest.raises(ValueError):
        format_search_hits([], max_results=0)
    with pytest.raises(ValueError):
        format_search_hits([], max_chars=0)


def test_citation_uses_only_real_title_and_chunk_id():
    result = format_search_hits([
        _hit(0, "alpha", {"metadata_json": {"title": "技术文档"}}),
        _hit(1, "beta", {"document_id": "doc-2", "provider": "gdrive"}),
    ])
    assert result["results"][0]["citation"] == "《技术文档》 [chunk_id: chunk-0]"
    assert result["results"][1]["citation"] == "[chunk_id: chunk-1]"
    assert "《" not in result["results"][1]["citation"]
    assert "file:///" not in json.dumps(result)


def test_citation_counts_toward_budget_without_clipping_later_evidence():
    hits = [_hit(0, "A" * 300, {"metadata_json": {"title": "First"}}),
            _hit(1, "B" * 300, {"metadata_json": {"title": "Second"}})]
    first = format_search_hits(hits, max_results=1)
    budget = _size(first) + 8
    result = format_search_hits(hits, max_results=2, max_chars=budget)
    assert result["returned_hits"] == 1
    assert result["results"][0]["text"] == hits[0]["text"]
    assert result["omitted_hits"] == 1
    assert _size(result) <= budget


def test_too_small_budget_does_not_drop_attribution():
    with pytest.raises(ValueError, match="source attribution"):
        format_search_hits([_hit(0, "A")], max_chars=20)
