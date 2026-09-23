"""No-network tests for MCP -> FastAPI search sharing."""

import json
from urllib.error import HTTPError, URLError

import pytest

from kb_agent.mcp import http_client


class FakeResponse:
    def __init__(self, body: object):
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, count: int):
        return self.body[:count]


def test_search_forwards_query_filter_and_returns_hits(monkeypatch):
    monkeypatch.setenv("KB_AGENT_API_BASE_URL", "http://127.0.0.1:8080")
    hit = {"chunk_id": "c1", "text": "DMA", "score": 0.8, "metadata": {}}
    recorded = {}

    def fake_open(request, timeout):
        recorded["url"] = request.full_url
        recorded["method"] = request.get_method()
        recorded["payload"] = json.loads(request.data)
        recorded["timeout"] = timeout
        return FakeResponse({"results": [hit]})

    monkeypatch.setattr(http_client, "urlopen", fake_open)
    assert http_client.search_via_api("DMA", 'provider == "chatgpt"') == [hit]
    assert recorded == {
        "url": "http://127.0.0.1:8080/api/v1/search",
        "method": "POST",
        "payload": {"query": "DMA", "filter_expr": 'provider == "chatgpt"'},
        "timeout": 180,
    }


def test_empty_query_fails_before_network():
    with pytest.raises(ValueError, match="query"):
        http_client.search_via_api(" ")


def test_http_error_reports_status_only(monkeypatch):
    def fake_open(*_args, **_kwargs):
        raise HTTPError("http://localhost", 502, "bad gateway", None, None)

    monkeypatch.setattr(http_client, "urlopen", fake_open)
    with pytest.raises(RuntimeError, match="HTTP 502"):
        http_client.search_via_api("DMA")


def test_network_error(monkeypatch):
    def fake_open(*_args, **_kwargs):
        raise URLError("connection refused")

    monkeypatch.setattr(http_client, "urlopen", fake_open)
    with pytest.raises(RuntimeError, match="unavailable"):
        http_client.search_via_api("DMA")


def test_invalid_response(monkeypatch):
    monkeypatch.setattr(http_client, "urlopen", lambda *_a, **_k: FakeResponse({"hits": []}))
    with pytest.raises(RuntimeError, match="unexpected response"):
        http_client.search_via_api("DMA")
