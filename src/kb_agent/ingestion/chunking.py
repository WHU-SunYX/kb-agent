"""Structure-aware chunking for normalized kb-agent documents.

Step 7 deliberately chunks the stable :class:`NormalizedDocument` model rather
than binding downstream code to ChatGPT or Docling internals.

Two strategies are provided:

* ``DocumentChunker``: structure-aware Markdown chunking for PDF/DOCX/Markdown
  documents normalized by Docling. It keeps headings, fenced code blocks,
  paragraphs, lists, and tables together when possible and adds bounded overlap.
* ``ConversationChunker``: turn-aware chunking for ChatGPT/Codex/etc.
  conversations. It groups complete turns whenever possible and preserves turn
  locators in chunk metadata.

Token counting is intentionally injectable. The default counter is a small,
deterministic approximation so Step 7 does not introduce a tokenizer/model
runtime before the embedding layer exists. Step 9 can provide the BGE-M3
Tokenizer-backed implementation through the same ``TokenCounter`` protocol.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterable, Mapping, Protocol, Sequence

from kb_agent.models import KnowledgeChunk, NormalizedDocument, SourceType


class ChunkingError(ValueError):
    """Raised when a normalized document cannot be chunked safely."""


class TokenCounter(Protocol):
    """Minimal interface needed by chunking strategies."""

    def count(self, text: str) -> int:
        """Return the number of tokens (or token-equivalent units) in ``text``."""


# English/number/code identifiers are counted as word-like units. CJK characters
# are counted individually because whitespace-based counting severely
# underestimates Chinese text. Punctuation is also counted to stay conservative.
_TOKEN_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
    r"|[A-Za-z0-9_]+(?:[-./:][A-Za-z0-9_]+)*"
    r"|[^\s]",
    re.UNICODE,
)


class ApproximateTokenCounter:
    """Dependency-free deterministic token estimate used before Step 9.

    The value is intentionally an estimate, not a claim to match BGE-M3's
    tokenizer. Chunks record ``token_count_method=approximate_v1`` so later
    stages can distinguish it from an exact model-tokenizer count.
    """

    method = "approximate_v1"

    def count(self, text: str) -> int:
        return len(_TOKEN_RE.findall(text))


@dataclass(frozen=True)
class ChunkingConfig:
    """Shared chunking limits.

    ``max_tokens`` is a hard limit with respect to the configured ``TokenCounter``.
    ``overlap_tokens`` is used only by document chunks. Conversation overlap is
    expressed in whole turns so a message is not duplicated partially.
    """

    max_tokens: int = 512
    overlap_tokens: int = 64
    conversation_overlap_turns: int = 1

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens must be greater than or equal to 0")
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        if self.conversation_overlap_turns < 0:
            raise ValueError("conversation_overlap_turns must be >= 0")


@dataclass(frozen=True)
class _MarkdownBlock:
    text: str
    section_path: tuple[str, ...]
    kind: str


@dataclass(frozen=True)
class _Turn:
    index: int
    role: str
    text: str
    message_id: str | None
    node_id: str | None
    create_time: Any | None


@dataclass(frozen=True)
class _ConversationPiece:
    text: str
    turn_index: int
    role: str
    message_id: str | None
    node_id: str | None
    create_time: Any | None
    part_index: int
    part_count: int


class BaseChunker:
    """Common helpers for kb-agent chunking strategies."""

    strategy_name = "base"

    def __init__(
        self,
        *,
        config: ChunkingConfig | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.config = config or ChunkingConfig()
        self.token_counter = token_counter or ApproximateTokenCounter()

    @property
    def token_count_method(self) -> str:
        return str(getattr(self.token_counter, "method", self.token_counter.__class__.__name__))

    def chunk(self, document: NormalizedDocument) -> list[KnowledgeChunk]:
        raise NotImplementedError

    def _make_chunk(
        self,
        document: NormalizedDocument,
        *,
        chunk_index: int,
        text: str,
        section_path: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> KnowledgeChunk:
        normalized = text.strip()
        if not normalized:
            raise ChunkingError("refusing to create an empty chunk")

        token_count = self.token_counter.count(normalized)
        if token_count > self.config.max_tokens:
            raise ChunkingError(
                f"chunk exceeds configured max_tokens: {token_count} > {self.config.max_tokens}"
            )

        content_hash = sha256(normalized.encode("utf-8")).hexdigest()
        identity = (
            f"{document.document_id}\n{self.strategy_name}\n{chunk_index}\n{content_hash}"
        ).encode("utf-8")
        chunk_id = f"{document.document_id}:chunk:{sha256(identity).hexdigest()[:20]}"

        chunk_metadata = {
            "chunking_strategy": self.strategy_name,
            "token_count_method": self.token_count_method,
            "max_tokens": self.config.max_tokens,
        }
        if metadata:
            chunk_metadata.update(dict(metadata))

        return KnowledgeChunk(
            chunk_id=chunk_id,
            document_id=document.document_id,
            chunk_index=chunk_index,
            text=normalized,
            source=document.source.model_copy(deep=True),
            title=document.title,
            section_path=list(section_path),
            token_count=token_count,
            domain=document.domain,
            project=document.project,
            content_hash=content_hash,
            metadata=chunk_metadata,
            security=document.security.model_copy(deep=True),
        )


class DocumentChunker(BaseChunker):
    """Chunk normalized Markdown while preserving structural boundaries."""

    strategy_name = "markdown_structure_v1"

    def chunk(self, document: NormalizedDocument) -> list[KnowledgeChunk]:
        blocks = _parse_markdown_blocks(document.content)
        if not blocks:
            raise ChunkingError(f"document {document.document_id!r} has no chunkable content")

        # Oversized blocks are split conservatively before packing. A normal
        # paragraph/code/table/list remains atomic whenever it fits.
        expanded: list[_MarkdownBlock] = []
        for block in blocks:
            pieces = _split_text_to_fit(
                block.text,
                max_tokens=self.config.max_tokens,
                counter=self.token_counter,
            )
            expanded.extend(
                _MarkdownBlock(text=piece, section_path=block.section_path, kind=block.kind)
                for piece in pieces
            )

        groups: list[list[_MarkdownBlock]] = []
        current: list[_MarkdownBlock] = []

        for block in expanded:
            candidate = _join_markdown_blocks([*current, block])
            if current and self.token_counter.count(candidate) > self.config.max_tokens:
                groups.append(current)
                current = self._overlap_tail(current)
                candidate = _join_markdown_blocks([*current, block])
                # Overlap must never make the next block overflow. Drop oldest
                # overlap blocks until the new block fits.
                while current and self.token_counter.count(candidate) > self.config.max_tokens:
                    current = current[1:]
                    candidate = _join_markdown_blocks([*current, block])

            current.append(block)

        if current:
            groups.append(current)

        chunks: list[KnowledgeChunk] = []
        previous_group: list[_MarkdownBlock] | None = None
        for index, group in enumerate(groups):
            text = _join_markdown_blocks(group)
            section_path = _dominant_section_path(group)
            overlap_count = _shared_prefix_count(previous_group, group) if previous_group else 0
            chunks.append(
                self._make_chunk(
                    document,
                    chunk_index=index,
                    text=text,
                    section_path=section_path,
                    metadata={
                        "source_type": "document",
                        "block_count": len(group),
                        "block_kinds": list(dict.fromkeys(block.kind for block in group)),
                        "overlap_block_count": overlap_count,
                    },
                )
            )
            previous_group = group

        return chunks

    def _overlap_tail(self, blocks: Sequence[_MarkdownBlock]) -> list[_MarkdownBlock]:
        if self.config.overlap_tokens <= 0:
            return []

        selected: list[_MarkdownBlock] = []
        for block in reversed(blocks):
            candidate = [block, *selected]
            count = self.token_counter.count(_join_markdown_blocks(candidate))
            if count > self.config.overlap_tokens:
                break
            selected = candidate
        return selected


class ConversationChunker(BaseChunker):
    """Chunk conversations by complete turns, with optional whole-turn overlap."""

    strategy_name = "conversation_turn_window_v1"

    def chunk(self, document: NormalizedDocument) -> list[KnowledgeChunk]:
        turns = _extract_turns(document)
        if not turns:
            # Conversation providers should normally carry structured turns from
            # Step 4. Falling back to document chunking is safer than silently
            # dropping third-party conversation exports that only have content.
            fallback = DocumentChunker(
                config=self.config,
                token_counter=self.token_counter,
            )
            chunks = fallback.chunk(document)
            for chunk in chunks:
                chunk.metadata["conversation_fallback"] = True
            return chunks

        pieces: list[_ConversationPiece] = []
        for turn in turns:
            rendered_parts = _split_turn_to_fit(
                turn.role,
                turn.text,
                max_tokens=self.config.max_tokens,
                counter=self.token_counter,
            )
            for part_index, part in enumerate(rendered_parts):
                pieces.append(
                    _ConversationPiece(
                        text=part,
                        turn_index=turn.index,
                        role=turn.role,
                        message_id=turn.message_id,
                        node_id=turn.node_id,
                        create_time=turn.create_time,
                        part_index=part_index,
                        part_count=len(rendered_parts),
                    )
                )

        groups: list[list[_ConversationPiece]] = []
        current: list[_ConversationPiece] = []
        for piece in pieces:
            candidate = _join_conversation_pieces([*current, piece])
            if current and self.token_counter.count(candidate) > self.config.max_tokens:
                groups.append(current)
                current = self._overlap_turn_tail(current)
                candidate = _join_conversation_pieces([*current, piece])
                while current and self.token_counter.count(candidate) > self.config.max_tokens:
                    oldest_turn = current[0].turn_index
                    current = [item for item in current if item.turn_index != oldest_turn]
                    candidate = _join_conversation_pieces([*current, piece])
            current.append(piece)
        if current:
            groups.append(current)

        chunks: list[KnowledgeChunk] = []
        previous_group: list[_ConversationPiece] | None = None
        for index, group in enumerate(groups):
            turn_indices = list(dict.fromkeys(piece.turn_index for piece in group))
            message_ids = list(
                dict.fromkeys(piece.message_id for piece in group if piece.message_id is not None)
            )
            node_ids = list(
                dict.fromkeys(piece.node_id for piece in group if piece.node_id is not None)
            )
            roles = list(dict.fromkeys(piece.role for piece in group))
            overlap_turn_count = (
                _shared_turn_prefix_count(previous_group, group) if previous_group else 0
            )
            chunks.append(
                self._make_chunk(
                    document,
                    chunk_index=index,
                    text=_join_conversation_pieces(group),
                    metadata={
                        "source_type": "conversation",
                        "turn_start": min(turn_indices),
                        "turn_end": max(turn_indices),
                        "turn_indices": turn_indices,
                        "message_ids": message_ids,
                        "node_ids": node_ids,
                        "roles": roles,
                        "overlap_turn_count": overlap_turn_count,
                        "contains_split_turn": any(piece.part_count > 1 for piece in group),
                    },
                )
            )
            previous_group = group

        return chunks

    def _overlap_turn_tail(
        self, pieces: Sequence[_ConversationPiece]
    ) -> list[_ConversationPiece]:
        turns_to_keep = self.config.conversation_overlap_turns
        if turns_to_keep <= 0 or not pieces:
            return []

        ordered_turns = list(dict.fromkeys(piece.turn_index for piece in pieces))
        selected_turns = set(ordered_turns[-turns_to_keep:])
        return [piece for piece in pieces if piece.turn_index in selected_turns]


class AutoChunker(BaseChunker):
    """Dispatch a normalized document to the correct source-aware strategy."""

    strategy_name = "auto_v1"

    def __init__(
        self,
        *,
        config: ChunkingConfig | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        super().__init__(config=config, token_counter=token_counter)
        self.document_chunker = DocumentChunker(
            config=self.config,
            token_counter=self.token_counter,
        )
        self.conversation_chunker = ConversationChunker(
            config=self.config,
            token_counter=self.token_counter,
        )

    def chunk(self, document: NormalizedDocument) -> list[KnowledgeChunk]:
        if document.source.source_type is SourceType.CONVERSATION:
            return self.conversation_chunker.chunk(document)
        return self.document_chunker.chunk(document)


def chunk_document(
    document: NormalizedDocument,
    *,
    config: ChunkingConfig | None = None,
    token_counter: TokenCounter | None = None,
) -> list[KnowledgeChunk]:
    """Convenience entry point used by the future ingestion pipeline."""

    return AutoChunker(config=config, token_counter=token_counter).chunk(document)


def chunk_documents(
    documents: Iterable[NormalizedDocument],
    *,
    config: ChunkingConfig | None = None,
    token_counter: TokenCounter | None = None,
) -> list[KnowledgeChunk]:
    """Chunk many normalized documents while preserving input order."""

    chunker = AutoChunker(config=config, token_counter=token_counter)
    chunks: list[KnowledgeChunk] = []
    for document in documents:
        chunks.extend(chunker.chunk(document))
    return chunks


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def _parse_markdown_blocks(text: str) -> list[_MarkdownBlock]:
    """Parse normalized Markdown into coarse structure-preserving blocks."""

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    section_stack: list[str] = []
    blocks: list[_MarkdownBlock] = []
    buffer: list[str] = []
    buffer_section: tuple[str, ...] = ()
    in_fence = False
    fence_marker: str | None = None

    def flush(kind: str = "text") -> None:
        nonlocal buffer
        content = "\n".join(buffer).strip()
        if content:
            blocks.append(_MarkdownBlock(content, buffer_section, kind))
        buffer = []

    for line in lines:
        heading = _HEADING_RE.match(line) if not in_fence else None
        if heading:
            flush()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            section_stack[level - 1 :] = []
            while len(section_stack) < level - 1:
                section_stack.append("")
            section_stack.append(title)
            section_stack[:] = [part for part in section_stack if part]
            blocks.append(
                _MarkdownBlock(
                    text=line.strip(),
                    section_path=tuple(section_stack),
                    kind="heading",
                )
            )
            buffer_section = tuple(section_stack)
            continue

        fence = _FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)
            if not in_fence:
                flush()
                in_fence = True
                fence_marker = marker
                buffer_section = tuple(section_stack)
                buffer = [line]
            else:
                buffer.append(line)
                if marker == fence_marker:
                    flush("code")
                    in_fence = False
                    fence_marker = None
                    buffer_section = tuple(section_stack)
            continue

        if in_fence:
            buffer.append(line)
            continue

        if not line.strip():
            flush(_infer_block_kind(buffer))
            buffer_section = tuple(section_stack)
            continue

        if not buffer:
            buffer_section = tuple(section_stack)
        buffer.append(line)

    flush("code" if in_fence else _infer_block_kind(buffer))
    return blocks


def _infer_block_kind(lines: Sequence[str]) -> str:
    stripped = [line.strip() for line in lines if line.strip()]
    if not stripped:
        return "text"
    if all(
        line.startswith(("- ", "* ", "+ ")) or re.match(r"^\d+[.)]\s", line)
        for line in stripped
    ):
        return "list"
    if len(stripped) >= 2 and all("|" in line for line in stripped[:2]):
        return "table"
    return "text"


def _join_markdown_blocks(blocks: Sequence[_MarkdownBlock]) -> str:
    return "\n\n".join(block.text.strip() for block in blocks if block.text.strip()).strip()


def _dominant_section_path(blocks: Sequence[_MarkdownBlock]) -> tuple[str, ...]:
    for block in reversed(blocks):
        if block.section_path:
            return block.section_path
    return ()


def _shared_prefix_count(
    previous: Sequence[_MarkdownBlock] | None,
    current: Sequence[_MarkdownBlock],
) -> int:
    if not previous:
        return 0
    max_shared = min(len(previous), len(current))
    for count in range(max_shared, 0, -1):
        if list(previous[-count:]) == list(current[:count]):
            return count
    return 0


def _extract_turns(document: NormalizedDocument) -> list[_Turn]:
    raw_turns = document.metadata.get("turns")
    if not isinstance(raw_turns, list):
        return []

    turns: list[_Turn] = []
    for index, item in enumerate(raw_turns):
        if not isinstance(item, Mapping):
            continue
        text = item.get("text")
        role = item.get("role")
        if not isinstance(text, str) or not text.strip():
            continue
        if not isinstance(role, str) or not role.strip():
            role = "unknown"
        turns.append(
            _Turn(
                index=index,
                role=role.strip().lower(),
                text=text.strip(),
                message_id=_clean_optional_string(item.get("message_id")),
                node_id=_clean_optional_string(item.get("node_id")),
                create_time=item.get("create_time"),
            )
        )
    return turns


def _render_turn(role: str, text: str) -> str:
    display_role = {
        "user": "User",
        "assistant": "Assistant",
        "system": "System",
        "tool": "Tool",
    }.get(role.lower(), role.strip().title() or "Unknown")
    return f"## {display_role}\n\n{text.strip()}"


def _split_turn_to_fit(
    role: str,
    text: str,
    *,
    max_tokens: int,
    counter: TokenCounter,
) -> list[str]:
    """Split an oversized turn while repeating the role heading in every piece."""

    rendered = _render_turn(role, text)
    if counter.count(rendered) <= max_tokens:
        return [rendered]

    header = _render_turn(role, "").strip()
    header_tokens = counter.count(header)
    body_budget = max_tokens - header_tokens
    if body_budget <= 0:
        raise ChunkingError(
            "max_tokens is too small to preserve the conversation role heading"
        )

    body_parts = _split_text_to_fit(
        text,
        max_tokens=body_budget,
        counter=counter,
    )
    rendered_parts = [_render_turn(role, part) for part in body_parts]
    if any(counter.count(part) > max_tokens for part in rendered_parts):
        raise ChunkingError("unable to split conversation turn within max_tokens")
    return rendered_parts


def _join_conversation_pieces(pieces: Sequence[_ConversationPiece]) -> str:
    return "\n\n".join(piece.text.strip() for piece in pieces if piece.text.strip()).strip()


def _shared_turn_prefix_count(
    previous: Sequence[_ConversationPiece] | None,
    current: Sequence[_ConversationPiece],
) -> int:
    if not previous or not current:
        return 0
    previous_turns = list(dict.fromkeys(piece.turn_index for piece in previous))
    current_turns = list(dict.fromkeys(piece.turn_index for piece in current))
    max_shared = min(len(previous_turns), len(current_turns))
    for count in range(max_shared, 0, -1):
        if previous_turns[-count:] == current_turns[:count]:
            return count
    return 0


def _split_text_to_fit(
    text: str,
    *,
    max_tokens: int,
    counter: TokenCounter,
) -> list[str]:
    """Split oversized text using semantic boundaries before hard word chunks."""

    normalized = text.strip()
    if not normalized:
        return []
    if counter.count(normalized) <= max_tokens:
        return [normalized]

    # Preserve lines (especially code, tables and logs) whenever possible.
    lines = [line for line in normalized.splitlines()]
    if len(lines) > 1:
        packed = _pack_units(lines, separator="\n", max_tokens=max_tokens, counter=counter)
        result: list[str] = []
        for part in packed:
            if counter.count(part) <= max_tokens:
                result.append(part)
            else:
                result.extend(
                    _split_text_to_fit(
                        part,
                        max_tokens=max_tokens,
                        counter=counter,
                    )
                )
        if result:
            return result

    sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?。！？；;])\s*", normalized)
        if item.strip()
    ]
    if len(sentences) > 1:
        packed = _pack_units(
            sentences,
            separator=" ",
            max_tokens=max_tokens,
            counter=counter,
        )
        result: list[str] = []
        for part in packed:
            if counter.count(part) <= max_tokens:
                result.append(part)
            else:
                result.extend(_split_words_to_fit(part, max_tokens=max_tokens, counter=counter))
        return result

    return _split_words_to_fit(normalized, max_tokens=max_tokens, counter=counter)


def _pack_units(
    units: Sequence[str],
    *,
    separator: str,
    max_tokens: int,
    counter: TokenCounter,
) -> list[str]:
    result: list[str] = []
    current: list[str] = []
    for unit in units:
        if not unit and not current:
            continue
        candidate = separator.join([*current, unit])
        if current and counter.count(candidate) > max_tokens:
            result.append(separator.join(current).strip())
            current = [unit]
        else:
            current.append(unit)
    if current:
        result.append(separator.join(current).strip())
    return [item for item in result if item]


def _split_words_to_fit(
    text: str,
    *,
    max_tokens: int,
    counter: TokenCounter,
) -> list[str]:
    words = text.split()
    if not words:
        return _hard_character_split(text, max_tokens=max_tokens, counter=counter)

    result: list[str] = []
    current: list[str] = []
    for word in words:
        if counter.count(word) > max_tokens:
            if current:
                result.append(" ".join(current))
                current = []
            result.extend(_hard_character_split(word, max_tokens=max_tokens, counter=counter))
            continue
        candidate = " ".join([*current, word])
        if current and counter.count(candidate) > max_tokens:
            result.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        result.append(" ".join(current))
    return result


def _hard_character_split(
    text: str,
    *,
    max_tokens: int,
    counter: TokenCounter,
) -> list[str]:
    result: list[str] = []
    current = ""
    for char in text:
        candidate = current + char
        if current and counter.count(candidate) > max_tokens:
            result.append(current)
            current = char
        else:
            current = candidate
    if current:
        result.append(current)
    return result


def _clean_optional_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


__all__ = [
    "ApproximateTokenCounter",
    "AutoChunker",
    "BaseChunker",
    "ChunkingConfig",
    "ChunkingError",
    "ConversationChunker",
    "DocumentChunker",
    "TokenCounter",
    "chunk_document",
    "chunk_documents",
]
