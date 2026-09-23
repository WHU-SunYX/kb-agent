from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from kb_agent.ingestion.chatgpt import ChatGPTExportError, parse_chatgpt_export
from kb_agent.models import SecurityClassification, SecurityMetadata


def _message(
    message_id: str,
    role: str,
    text: str | None,
    created: float,
    *,
    content_type: str = "text",
    parts=None,
    metadata=None,
    recipient: str = "all",
):
    if parts is None:
        parts = [] if text is None else [text]
    return {
        "id": message_id,
        "author": {"role": role, "name": None, "metadata": {}},
        "create_time": created,
        "update_time": None,
        "content": {"content_type": content_type, "parts": parts},
        "metadata": metadata or {},
        "recipient": recipient,
    }


def _conversation(conversation_id: str = "conv-1") -> dict:
    # root -> system -> user -> assistant-old
    #                       \-> assistant-current -> hidden -> user2
    return {
        "id": conversation_id,
        "title": "Sparse KV debug",
        "create_time": 1_700_000_000.0,
        "update_time": 1_700_000_004.0,
        "current_node": "u2",
        "default_model_slug": "gpt-test",
        "mapping": {
            "root": {
                "id": "root",
                "message": None,
                "parent": None,
                "children": ["sys"],
            },
            "sys": {
                "id": "sys",
                "message": _message("m-sys", "system", "internal instruction", 1_700_000_000.1),
                "parent": "root",
                "children": ["u1"],
            },
            "u1": {
                "id": "u1",
                "message": _message("m-u1", "user", "Why is selector slow?", 1_700_000_001.0),
                "parent": "sys",
                "children": ["a-old", "a-current"],
            },
            "a-old": {
                "id": "a-old",
                "message": _message("m-old", "assistant", "Old superseded answer", 1_700_000_002.0),
                "parent": "u1",
                "children": [],
            },
            "a-current": {
                "id": "a-current",
                "message": _message(
                    "m-a1",
                    "assistant",
                    "Current answer",
                    1_700_000_002.5,
                    metadata={"model_slug": "gpt-test"},
                ),
                "parent": "u1",
                "children": ["hidden"],
            },
            "hidden": {
                "id": "hidden",
                "message": _message(
                    "m-hidden",
                    "assistant",
                    "hidden reasoning",
                    1_700_000_003.0,
                    content_type="reasoning_recap",
                ),
                "parent": "a-current",
                "children": ["u2"],
            },
            "u2": {
                "id": "u2",
                "message": _message("m-u2", "user", "Show the measured result.", 1_700_000_004.0),
                "parent": "hidden",
                "children": [],
            },
        },
    }


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_parse_conversations_json_uses_only_active_visible_branch(tmp_path: Path):
    source = tmp_path / "conversations.json"
    _write_json(source, [_conversation()])

    documents = parse_chatgpt_export(source, domain="ai-ssd", project="sparse-kv")

    assert len(documents) == 1
    document = documents[0]
    assert document.document_id == "chatgpt:conv-1"
    assert document.title == "Sparse KV debug"
    assert document.domain == "ai-ssd"
    assert document.project == "sparse-kv"
    assert "Why is selector slow?" in document.content
    assert "Current answer" in document.content
    assert "Show the measured result." in document.content
    assert "Old superseded answer" not in document.content
    assert "internal instruction" not in document.content
    assert "hidden reasoning" not in document.content
    assert document.metadata["turn_count"] == 3
    assert document.metadata["inactive_mapping_node_count"] == 1
    assert [turn["role"] for turn in document.metadata["turns"]] == [
        "user",
        "assistant",
        "user",
    ]


def test_parse_single_conversation_object(tmp_path: Path):
    source = tmp_path / "conversations.json"
    _write_json(source, _conversation("single"))

    documents = parse_chatgpt_export(source)

    assert [document.document_id for document in documents] == ["chatgpt:single"]


def test_parse_numbered_shards_from_directory_and_deduplicate(tmp_path: Path):
    older = _conversation("shared")
    older["update_time"] = 1_700_000_004.0
    newer = _conversation("shared")
    newer["update_time"] = 1_700_000_100.0
    newer["title"] = "Newest title"

    _write_json(tmp_path / "conversations-001.json", [older])
    _write_json(tmp_path / "conversations-002.json", [newer, _conversation("second")])
    _write_json(tmp_path / "unrelated.json", {"hello": "world"})

    documents = parse_chatgpt_export(tmp_path)

    assert len(documents) == 2
    by_id = {document.document_id: document for document in documents}
    assert by_id["chatgpt:shared"].title == "Newest title"
    assert "chatgpt:second" in by_id


def test_parse_zip_and_preserve_raw_uri(tmp_path: Path):
    archive_path = tmp_path / "export.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("account/conversations.json", json.dumps([_conversation()]))

    documents = parse_chatgpt_export(
        archive_path,
        raw_uri="s3://kb-raw/chatgpt/export-20260914.zip",
    )

    source = documents[0].source
    assert source.uri == "s3://kb-raw/chatgpt/export-20260914.zip"
    assert source.metadata["export_member"] == "account/conversations.json"


def test_parse_nested_conversation_zip(tmp_path: Path):
    nested_bytes = BytesIO()
    with ZipFile(nested_bytes, "w") as nested:
        nested.writestr("conversations-001.json", json.dumps([_conversation("nested")]))

    outer = tmp_path / "chatgpt-export.zip"
    with ZipFile(outer, "w") as archive:
        archive.writestr("Conversations__2026_part-001.zip", nested_bytes.getvalue())
        archive.writestr("Files__2026_part-001.zip", b"not a conversation zip")

    documents = parse_chatgpt_export(outer)

    assert [document.document_id for document in documents] == ["chatgpt:nested"]
    assert documents[0].source.metadata["container_chain"] == [
        "Conversations__2026_part-001.zip"
    ]


def test_multimodal_parts_and_attachment_metadata_are_preserved(tmp_path: Path):
    conversation = _conversation("media")
    conversation["mapping"]["u2"]["message"] = _message(
        "m-u2",
        "user",
        None,
        1_700_000_004.0,
        content_type="multimodal_text",
        parts=[
            "Please inspect this screenshot",
            {
                "content_type": "image_asset_pointer",
                "asset_pointer": "file-service://file-123",
                "width": 1024,
                "height": 768,
            },
        ],
        metadata={
            "attachments": [
                {"id": "file-123", "name": "screenshot.png", "mime_type": "image/png"}
            ]
        },
    )
    source = tmp_path / "conversations.json"
    _write_json(source, [conversation])

    document = parse_chatgpt_export(source)[0]
    last_turn = document.metadata["turns"][-1]

    assert "Please inspect this screenshot" in last_turn["text"]
    assert "[Image: file-service://file-123]" in last_turn["text"]
    assert last_turn["metadata"]["attachments"][0]["name"] == "screenshot.png"
    assert last_turn["metadata"]["asset_references"][0]["width"] == 1024


def test_invalid_current_node_falls_back_to_latest_leaf(tmp_path: Path):
    conversation = _conversation("fallback")
    conversation["current_node"] = "does-not-exist"
    source = tmp_path / "conversations.json"
    _write_json(source, [conversation])

    document = parse_chatgpt_export(source)[0]

    assert "invalid_or_missing_current_node_used_latest_leaf" in document.metadata["diagnostics"]
    assert "Show the measured result." in document.content


def test_security_metadata_is_copied_to_document(tmp_path: Path):
    source = tmp_path / "conversations.json"
    _write_json(source, [_conversation()])
    security = SecurityMetadata(
        tenant_id="company",
        classification=SecurityClassification.CONFIDENTIAL,
        allowed_groups=["aissd"],
    )

    document = parse_chatgpt_export(source, security=security)[0]

    assert document.security.tenant_id == "company"
    assert document.security.classification == SecurityClassification.CONFIDENTIAL
    assert document.security.allowed_groups == ["aissd"]


def test_flat_messages_fallback(tmp_path: Path):
    payload = {
        "id": "flat-1",
        "title": "Flat export",
        "messages": [
            _message("u", "user", "hello", 1_700_000_000.0),
            _message("a", "assistant", "hi", 1_700_000_001.0),
        ],
    }
    source = tmp_path / "conversations.json"
    _write_json(source, payload)

    document = parse_chatgpt_export(source)[0]

    assert document.metadata["turn_count"] == 2
    assert "used_flat_messages_fallback" in document.metadata["diagnostics"]


def test_no_conversation_files_in_zip_raises(tmp_path: Path):
    archive_path = tmp_path / "export.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("profile.json", "{}")

    with pytest.raises(ChatGPTExportError, match="no conversations.json"):
        parse_chatgpt_export(archive_path)


def test_unsafe_zip_member_is_rejected(tmp_path: Path):
    archive_path = tmp_path / "export.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("../conversations.json", json.dumps([_conversation()]))

    with pytest.raises(ChatGPTExportError, match="unsafe ZIP member"):
        parse_chatgpt_export(archive_path)
