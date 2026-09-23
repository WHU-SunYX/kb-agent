"""Regression guard for the public MCP surface.

These tests intentionally inspect the source module rather than starting the
MCP transport so the write/read boundary stays testable even when the optional
``mcp`` dependency is not installed in a minimal unit-test environment.
"""

from __future__ import annotations

import ast
from pathlib import Path


SERVER_PATH = Path(__file__).parents[1] / "src" / "kb_agent" / "mcp" / "server.py"


def _tool_function_names() -> set[str]:
    tree = ast.parse(SERVER_PATH.read_text(encoding="utf-8"))
    result: set[str] = set()

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "mcp"
                and func.attr == "tool"
            ):
                result.add(node.name)
    return result


def test_mcp_surface_is_read_only() -> None:
    assert _tool_function_names() == {"kb_search", "kb_get_document"}


def test_mcp_server_does_not_wire_ingestion_service() -> None:
    source = SERVER_PATH.read_text(encoding="utf-8")
    assert "kb_ingest" not in source
    assert "get_ingestion_pipeline" not in source
    assert "ingestion_service" not in source


def test_mcp_server_does_not_load_embedding_or_retriever() -> None:
    source = SERVER_PATH.read_text(encoding="utf-8")
    assert "get_retriever" not in source
    assert "get_embedding_backend" not in source
    assert "build_application_services" not in source
    assert "search_via_api" in source
