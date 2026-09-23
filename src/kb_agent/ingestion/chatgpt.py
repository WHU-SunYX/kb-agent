"""ChatGPT data-export source adapter.

The adapter converts ChatGPT export JSON into :class:`NormalizedDocument`
objects without coupling downstream code to OpenAI's export schema.

Supported inputs
----------------
* ``conversations.json``
* numbered shards such as ``conversations-001.json``
* an extracted export directory containing either form
* ZIP archives containing either form
* recent outer ZIP archives that contain nested conversation ZIP parts

Conversation ordering
---------------------
ChatGPT exports store messages in a ``mapping`` graph.  The visible/current
conversation is reconstructed by walking ``current_node -> parent -> ...`` and
reversing that path.  This deliberately avoids mixing superseded sibling
branches created by message edits or regenerated assistant answers.

The raw export remains the source of truth.  This module keeps only normalized
turn text plus provenance/diagnostic metadata; later ingestion stages handle
security scanning, chunking, embedding, and indexing.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO, TextIOWrapper
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping, Sequence
from zipfile import BadZipFile, ZipFile

from kb_agent.models import NormalizedDocument, SecurityMetadata, SourceReference, SourceType


_CONVERSATION_JSON_RE = re.compile(
    r"(?:^|/)conversations(?:[-_]\d+)?\.json$",
    re.IGNORECASE,
)
_HIDDEN_CONTENT_TYPES = {
    "thoughts",
    "reasoning_recap",
    "user_editable_context",
    "model_editable_context",
}
_VISIBLE_ROLES = {"user", "assistant"}
_MAX_NESTED_ZIP_DEPTH = 3


class ChatGPTExportError(ValueError):
    """Raised when a ChatGPT export cannot be parsed safely/reliably."""


class ChatGPTExportParser:
    """Parse ChatGPT account exports into provider-neutral documents.

    The parser is intentionally defensive because the export schema is not a
    public, versioned API.  Unknown fields are ignored, unknown content parts
    become visible placeholders, and provenance metadata is retained so a
    later stage can always trace an indexed document back to the raw artifact.
    """

    def parse(
        self,
        source: str | Path,
        *,
        raw_uri: str | None = None,
        domain: str | None = None,
        project: str | None = None,
        security: SecurityMetadata | None = None,
    ) -> list[NormalizedDocument]:
        """Parse a JSON file, ZIP archive, or extracted export directory.

        Args:
            source: Local path used for parsing.
            raw_uri: Optional durable URI for the raw artifact, e.g. an S3 URI
                returned by ``RawStore``.  When omitted, local ``file://`` URIs
                are generated.
            domain: Optional knowledge domain metadata (e.g. ``ai-ssd``).
            project: Optional project metadata (e.g. ``sparse-kv``).
            security: Security/ACL metadata copied onto every document.

        Returns:
            One :class:`NormalizedDocument` for every non-empty conversation.

        Raises:
            FileNotFoundError: if ``source`` does not exist.
            ChatGPTExportError: for unsupported input or malformed export data.
        """

        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)

        if path.is_dir():
            payloads = list(self._payloads_from_directory(path, raw_uri=raw_uri))
        elif path.suffix.lower() == ".json":
            payloads = [
                (
                    self._load_json_path(path),
                    raw_uri or path.as_uri(),
                    {"export_member": path.name},
                )
            ]
        elif path.suffix.lower() == ".zip":
            base_uri = raw_uri or path.as_uri()
            try:
                with ZipFile(path) as archive:
                    payloads = list(
                        self._payloads_from_zip(
                            archive,
                            base_uri=base_uri,
                            container_chain=[],
                            depth=0,
                        )
                    )
            except BadZipFile as exc:
                raise ChatGPTExportError(f"invalid ZIP archive: {path}") from exc
        else:
            raise ChatGPTExportError(
                f"unsupported ChatGPT export input: {path}; expected .json, .zip, or directory"
            )

        if not payloads:
            raise ChatGPTExportError(
                f"no conversations.json or numbered conversation JSON files found in {path}"
            )

        documents: dict[str, NormalizedDocument] = {}
        for payload, source_uri, source_metadata in payloads:
            for conversation in self._coerce_conversation_list(payload, source_metadata):
                document = self._normalize_conversation(
                    conversation,
                    source_uri=source_uri,
                    source_metadata=source_metadata,
                    domain=domain,
                    project=project,
                    security=security,
                )
                if document is None:
                    continue

                # Defensive de-duplication for exports/shards that happen to
                # repeat a conversation.  Prefer the newest representation.
                existing = documents.get(document.document_id)
                if existing is None or self._is_newer(document, existing):
                    documents[document.document_id] = document

        return sorted(
            documents.values(),
            key=lambda doc: (
                doc.created_at or datetime.min.replace(tzinfo=timezone.utc),
                doc.document_id,
            ),
        )

    def _payloads_from_directory(
        self,
        directory: Path,
        *,
        raw_uri: str | None,
    ) -> Iterable[tuple[Any, str, dict[str, Any]]]:
        json_paths = sorted(
            path
            for path in directory.rglob("*.json")
            if _is_conversation_json_name(path.as_posix())
        )
        for path in json_paths:
            relative = path.relative_to(directory).as_posix()
            source_uri = raw_uri or path.resolve().as_uri()
            metadata: dict[str, Any] = {"export_member": relative}
            if raw_uri:
                metadata["raw_container_uri"] = raw_uri
            yield self._load_json_path(path), source_uri, metadata

        zip_paths = sorted(
            path
            for path in directory.rglob("*.zip")
            if "conversation" in path.name.lower()
        )
        for path in zip_paths:
            base_uri = raw_uri or path.resolve().as_uri()
            try:
                with ZipFile(path) as archive:
                    yield from self._payloads_from_zip(
                        archive,
                        base_uri=base_uri,
                        container_chain=[path.relative_to(directory).as_posix()],
                        depth=0,
                    )
            except BadZipFile as exc:
                raise ChatGPTExportError(f"invalid nested ZIP archive: {path}") from exc

    def _payloads_from_zip(
        self,
        archive: ZipFile,
        *,
        base_uri: str,
        container_chain: list[str],
        depth: int,
    ) -> Iterable[tuple[Any, str, dict[str, Any]]]:
        if depth > _MAX_NESTED_ZIP_DEPTH:
            raise ChatGPTExportError(
                f"nested ZIP depth exceeds {_MAX_NESTED_ZIP_DEPTH}: {'!/'.join(container_chain)}"
            )

        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            if info.is_dir():
                continue

            member = _safe_zip_member_name(info.filename)
            if _is_conversation_json_name(member):
                try:
                    with archive.open(info) as raw:
                        payload = json.load(TextIOWrapper(raw, encoding="utf-8-sig"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    location = "!/".join([*container_chain, member])
                    raise ChatGPTExportError(
                        f"invalid conversation JSON inside ZIP: {location}"
                    ) from exc

                metadata: dict[str, Any] = {
                    "export_member": member,
                }
                if container_chain:
                    metadata["container_chain"] = list(container_chain)
                yield payload, base_uri, metadata
                continue

            # Recent ChatGPT exports can wrap conversation shards in nested ZIP
            # parts.  Restrict recursion to ZIPs whose names mention
            # "conversation" so unrelated user-uploaded ZIP attachments are not
            # parsed as export containers.
            if (
                member.lower().endswith(".zip")
                and "conversation" in PurePosixPath(member).name.lower()
            ):
                try:
                    nested_bytes = archive.read(info)
                    with ZipFile(BytesIO(nested_bytes)) as nested:
                        yield from self._payloads_from_zip(
                            nested,
                            base_uri=base_uri,
                            container_chain=[*container_chain, member],
                            depth=depth + 1,
                        )
                except BadZipFile as exc:
                    location = "!/".join([*container_chain, member])
                    raise ChatGPTExportError(
                        f"invalid nested conversation ZIP: {location}"
                    ) from exc

    @staticmethod
    def _load_json_path(path: Path) -> Any:
        try:
            with path.open("r", encoding="utf-8-sig") as handle:
                return json.load(handle)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ChatGPTExportError(f"invalid JSON file: {path}") from exc

    @staticmethod
    def _coerce_conversation_list(
        payload: Any,
        source_metadata: Mapping[str, Any],
    ) -> list[Mapping[str, Any]]:
        if isinstance(payload, list):
            conversations = payload
        elif isinstance(payload, dict) and _looks_like_conversation(payload):
            conversations = [payload]
        elif isinstance(payload, dict) and isinstance(payload.get("conversations"), list):
            # Defensive compatibility for wrappers used by some export tooling.
            conversations = payload["conversations"]
        else:
            member = source_metadata.get("export_member", "<unknown>")
            raise ChatGPTExportError(
                f"conversation JSON has unsupported top-level shape: {member}"
            )

        invalid = [index for index, item in enumerate(conversations) if not isinstance(item, dict)]
        if invalid:
            member = source_metadata.get("export_member", "<unknown>")
            raise ChatGPTExportError(
                f"conversation JSON contains non-object entries at indexes {invalid[:5]}: {member}"
            )
        return conversations

    def _normalize_conversation(
        self,
        conversation: Mapping[str, Any],
        *,
        source_uri: str,
        source_metadata: Mapping[str, Any],
        domain: str | None,
        project: str | None,
        security: SecurityMetadata | None,
    ) -> NormalizedDocument | None:
        conversation_id = _first_nonempty_string(
            conversation.get("id"),
            conversation.get("conversation_id"),
        )
        if not conversation_id:
            # A stable hash avoids random IDs while still allowing malformed but
            # useful exports to be ingested deterministically.
            identity_blob = json.dumps(conversation, sort_keys=True, default=str).encode("utf-8")
            conversation_id = f"anonymous-{sha256(identity_blob).hexdigest()[:16]}"

        title = _first_nonempty_string(conversation.get("title")) or f"ChatGPT {conversation_id}"
        mapping = conversation.get("mapping")
        diagnostics: list[str] = []

        if isinstance(mapping, dict) and mapping:
            path_nodes = self._active_path_nodes(conversation, mapping, diagnostics)
        else:
            path_nodes = []
            diagnostics.append("missing_or_empty_mapping")

        turns: list[dict[str, Any]] = []
        for node in path_nodes:
            turn = self._normalize_turn(node, diagnostics)
            if turn is not None:
                turns.append(turn)

        # Some third-party/current-conversation exports expose a flat messages
        # array.  It is not the primary account-export shape, but supporting it
        # here costs little and keeps this adapter resilient.
        if not turns and isinstance(conversation.get("messages"), list):
            diagnostics.append("used_flat_messages_fallback")
            for index, message in enumerate(conversation["messages"]):
                if not isinstance(message, dict):
                    continue
                synthetic_node = {"id": f"flat-{index}", "message": message}
                turn = self._normalize_turn(synthetic_node, diagnostics)
                if turn is not None:
                    turns.append(turn)

        if not turns:
            return None

        content = _render_transcript(turns)
        created_at = _epoch_to_datetime(conversation.get("create_time"))
        updated_at = _epoch_to_datetime(conversation.get("update_time"))
        turn_times = [
            _epoch_to_datetime(turn.get("create_time"))
            for turn in turns
            if turn.get("create_time") is not None
        ]
        turn_times = [value for value in turn_times if value is not None]
        if created_at is None and turn_times:
            created_at = min(turn_times)
        if updated_at is None and turn_times:
            updated_at = max(turn_times)
        if created_at and updated_at and updated_at < created_at:
            diagnostics.append("conversation_update_time_precedes_create_time")
            updated_at = created_at

        role_counts = Counter(turn["role"] for turn in turns)
        mapping_count = len(mapping) if isinstance(mapping, dict) else 0
        inactive_count = max(mapping_count - len(path_nodes), 0)

        provider_metadata = dict(source_metadata)
        provider_metadata.update(
            {
                "conversation_id": conversation_id,
                "current_node": conversation.get("current_node"),
            }
        )

        source = SourceReference(
            source_type=SourceType.CONVERSATION,
            provider="chatgpt",
            uri=source_uri,
            source_id=conversation_id,
            metadata=provider_metadata,
        )

        document_metadata: dict[str, Any] = {
            "format": "chatgpt_export",
            "schema_version": 1,
            "conversation_id": conversation_id,
            "current_node": conversation.get("current_node"),
            "turn_count": len(turns),
            "role_counts": dict(role_counts),
            "mapping_node_count": mapping_count,
            "active_path_node_count": len(path_nodes),
            "inactive_mapping_node_count": inactive_count,
            "turns": turns,
        }
        if diagnostics:
            document_metadata["diagnostics"] = list(dict.fromkeys(diagnostics))

        # Keep a small amount of conversation-level metadata useful for later
        # filtering/analysis without copying the entire raw export object.
        for key in (
            "default_model_slug",
            "conversation_template_id",
            "gizmo_id",
            "gizmo_type",
            "is_archived",
            "is_starred",
            "conversation_origin",
            "memory_scope",
            "voice",
        ):
            if key in conversation:
                document_metadata[key] = deepcopy(conversation[key])

        return NormalizedDocument(
            document_id=f"chatgpt:{conversation_id}",
            title=title,
            content=content,
            source=source,
            domain=domain,
            project=project,
            created_at=created_at,
            updated_at=updated_at,
            content_hash=sha256(content.encode("utf-8")).hexdigest(),
            metadata=document_metadata,
            security=(security or SecurityMetadata()).model_copy(deep=True),
        )

    def _active_path_nodes(
        self,
        conversation: Mapping[str, Any],
        mapping: Mapping[str, Any],
        diagnostics: list[str],
    ) -> list[Mapping[str, Any]]:
        current = conversation.get("current_node")
        if not isinstance(current, str) or current not in mapping:
            current = _choose_latest_leaf(mapping)
            diagnostics.append("invalid_or_missing_current_node_used_latest_leaf")
        if current is None:
            return []

        path: list[Mapping[str, Any]] = []
        seen: set[str] = set()
        while isinstance(current, str) and current in mapping:
            if current in seen:
                diagnostics.append("mapping_cycle_detected")
                break
            seen.add(current)
            node = mapping[current]
            if not isinstance(node, dict):
                diagnostics.append("non_object_mapping_node")
                break
            path.append(node)
            parent = node.get("parent")
            if parent is None:
                break
            if not isinstance(parent, str):
                diagnostics.append("invalid_parent_reference")
                break
            current = parent

        path.reverse()
        return path

    def _normalize_turn(
        self,
        node: Mapping[str, Any],
        diagnostics: list[str],
    ) -> dict[str, Any] | None:
        message = node.get("message")
        if not isinstance(message, dict):
            return None

        author = message.get("author")
        role = author.get("role") if isinstance(author, dict) else None
        if role not in _VISIBLE_ROLES:
            return None

        metadata = message.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        if metadata.get("is_visually_hidden_from_conversation") is True:
            return None

        recipient = message.get("recipient")
        if isinstance(recipient, str) and recipient not in {"", "all"}:
            return None

        content = message.get("content")
        content_type = content.get("content_type") if isinstance(content, dict) else None
        if content_type in _HIDDEN_CONTENT_TYPES:
            return None

        text, asset_refs, unknown_parts = _content_to_text(content)
        if unknown_parts:
            diagnostics.append("unknown_content_parts_rendered_as_placeholders")
        if not text.strip():
            return None

        message_metadata = _select_message_metadata(metadata)
        if asset_refs:
            message_metadata["asset_references"] = asset_refs

        return {
            "node_id": _first_nonempty_string(node.get("id")),
            "message_id": _first_nonempty_string(message.get("id")),
            "role": role,
            "author_name": (
                _first_nonempty_string(author.get("name")) if isinstance(author, dict) else None
            ),
            "create_time": message.get("create_time"),
            "update_time": message.get("update_time"),
            "content_type": content_type or "unknown",
            "text": text,
            "metadata": message_metadata,
        }

    @staticmethod
    def _is_newer(candidate: NormalizedDocument, existing: NormalizedDocument) -> bool:
        floor = datetime.min.replace(tzinfo=timezone.utc)
        candidate_time = candidate.updated_at or candidate.created_at or floor
        existing_time = existing.updated_at or existing.created_at or floor
        return candidate_time >= existing_time


def parse_chatgpt_export(
    source: str | Path,
    *,
    raw_uri: str | None = None,
    domain: str | None = None,
    project: str | None = None,
    security: SecurityMetadata | None = None,
) -> list[NormalizedDocument]:
    """Convenience wrapper around :class:`ChatGPTExportParser`."""

    return ChatGPTExportParser().parse(
        source,
        raw_uri=raw_uri,
        domain=domain,
        project=project,
        security=security,
    )


def _is_conversation_json_name(name: str) -> bool:
    normalized = name.replace("\\", "/")
    return _CONVERSATION_JSON_RE.search(normalized) is not None


def _safe_zip_member_name(name: str) -> str:
    """Normalize a ZIP member name without extracting it to the filesystem."""

    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ChatGPTExportError(f"unsafe ZIP member path: {name}")
    return path.as_posix()


def _looks_like_conversation(value: Mapping[str, Any]) -> bool:
    return "mapping" in value or "messages" in value or "current_node" in value


def _first_nonempty_string(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _epoch_to_datetime(value: Any) -> datetime | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    numeric = float(value)
    # Defensive support for millisecond timestamps.
    if abs(numeric) >= 100_000_000_000:
        numeric /= 1000.0
    try:
        return datetime.fromtimestamp(numeric, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _choose_latest_leaf(mapping: Mapping[str, Any]) -> str | None:
    candidates: list[tuple[float, str]] = []
    for node_id, node in mapping.items():
        if not isinstance(node_id, str) or not isinstance(node, dict):
            continue
        children = node.get("children")
        if isinstance(children, list) and children:
            continue
        message = node.get("message")
        timestamp = 0.0
        if isinstance(message, dict):
            raw_time = message.get("create_time")
            if isinstance(raw_time, (int, float)) and not isinstance(raw_time, bool):
                timestamp = float(raw_time)
        candidates.append((timestamp, node_id))

    if not candidates:
        candidates = [(0.0, node_id) for node_id in mapping if isinstance(node_id, str)]
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def _content_to_text(content: Any) -> tuple[str, list[dict[str, Any]], int]:
    """Extract readable text and asset references from one message content."""

    if isinstance(content, str):
        return content.strip(), [], 0
    if not isinstance(content, dict):
        return "", [], 0

    pieces: list[str] = []
    assets: list[dict[str, Any]] = []
    unknown_parts = 0

    parts = content.get("parts")
    if isinstance(parts, list):
        for part in parts:
            text, asset, unknown = _content_part_to_text(part)
            if text:
                pieces.append(text)
            if asset is not None:
                assets.append(asset)
            unknown_parts += unknown
    else:
        for key in ("text", "content", "result"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                pieces.append(value.strip())
                break

    return "\n\n".join(piece for piece in pieces if piece).strip(), assets, unknown_parts


def _content_part_to_text(part: Any) -> tuple[str, dict[str, Any] | None, int]:
    if isinstance(part, str):
        return part.strip(), None, 0
    if not isinstance(part, dict):
        return f"[Unsupported content part: {type(part).__name__}]", None, 1

    for key in ("text", "content"):
        value = part.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), None, 0

    content_type = str(part.get("content_type") or "unknown")
    pointer = _first_nonempty_string(
        part.get("asset_pointer"),
        part.get("image_asset_pointer"),
        part.get("file_id"),
    )

    if "image" in content_type:
        label = f"[Image: {pointer}]" if pointer else "[Image]"
        return label, _asset_metadata(part, content_type, pointer), 0
    if "audio" in content_type:
        label = f"[Audio: {pointer}]" if pointer else "[Audio]"
        return label, _asset_metadata(part, content_type, pointer), 0
    if "file" in content_type or pointer:
        label = f"[File: {pointer}]" if pointer else f"[File: {content_type}]"
        return label, _asset_metadata(part, content_type, pointer), 0

    return f"[Unsupported content: {content_type}]", None, 1


def _asset_metadata(
    part: Mapping[str, Any],
    content_type: str,
    pointer: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"content_type": content_type}
    if pointer:
        result["pointer"] = pointer
    for key in ("size_bytes", "width", "height"):
        if key in part:
            result[key] = deepcopy(part[key])
    return result


def _select_message_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "model_slug",
        "message_type",
        "finish_details",
        "attachments",
        "content_references",
        "citations",
    ):
        if key in metadata:
            result[key] = deepcopy(metadata[key])
    return result


def _render_transcript(turns: Sequence[Mapping[str, Any]]) -> str:
    blocks: list[str] = []
    for turn in turns:
        role = str(turn["role"]).capitalize()
        text = str(turn["text"]).strip()
        blocks.append(f"## {role}\n\n{text}")
    return "\n\n".join(blocks).strip()
