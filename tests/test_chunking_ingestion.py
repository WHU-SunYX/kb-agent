from __future__ import annotations

from kb_agent.ingestion.chunking import (
    ApproximateTokenCounter,
    ChunkingConfig,
    ConversationChunker,
    DocumentChunker,
    chunk_document,
    chunk_documents,
)
from kb_agent.models import (
    NormalizedDocument,
    SecurityClassification,
    SecurityMetadata,
    SourceReference,
    SourceType,
)


class _WordCounter:
    method = "word_test"

    def count(self, text: str) -> int:
        return len(text.split())


def _document(content: str, *, document_id: str = "doc-1") -> NormalizedDocument:
    return NormalizedDocument(
        document_id=document_id,
        title="Sparse KV Design",
        content=content,
        source=SourceReference(
            source_type=SourceType.DOCUMENT,
            provider="pdf",
            uri="s3://kb-raw/design.pdf",
            source_id="raw-1",
        ),
        domain="ai-ssd",
        project="sparse-kv",
        security=SecurityMetadata(
            tenant_id="company",
            classification=SecurityClassification.CONFIDENTIAL,
            allowed_groups=["aissd"],
        ),
    )


def _conversation(turns: list[dict]) -> NormalizedDocument:
    rendered = "\n\n".join(
        f"## {turn['role'].title()}\n\n{turn['text']}" for turn in turns
    )
    return NormalizedDocument(
        document_id="chatgpt:conv-1",
        title="Sparse KV discussion",
        content=rendered,
        source=SourceReference(
            source_type=SourceType.CONVERSATION,
            provider="chatgpt",
            uri="s3://kb-raw/chatgpt/export.zip",
            source_id="conv-1",
        ),
        domain="ai-ssd",
        project="sparse-kv",
        metadata={"turns": turns},
        security=SecurityMetadata(
            tenant_id="company",
            allowed_groups=["aissd"],
        ),
    )


def test_approximate_counter_handles_chinese_without_spaces():
    counter = ApproximateTokenCounter()
    assert counter.count("稀疏注意力") >= 5
    assert counter.count("hello world") >= 2


def test_document_chunker_preserves_headings_and_section_paths():
    document = _document(
        "# Architecture\n\nintro text\n\n"
        "## Selector\n\nselector details\n\n"
        "## IO\n\nio details"
    )
    chunker = DocumentChunker(
        config=ChunkingConfig(max_tokens=8, overlap_tokens=0),
        token_counter=_WordCounter(),
    )

    chunks = chunker.chunk(document)

    assert len(chunks) >= 2
    assert any(chunk.section_path == ["Architecture", "Selector"] for chunk in chunks)
    assert any(chunk.section_path == ["Architecture", "IO"] for chunk in chunks)
    assert all(chunk.token_count <= 8 for chunk in chunks)


def test_document_chunker_keeps_fenced_code_together_when_it_fits():
    document = _document(
        "# Example\n\nBefore\n\n```bash\nexport A=1\nexport B=2\n```\n\nAfter"
    )
    chunks = DocumentChunker(
        config=ChunkingConfig(max_tokens=20, overlap_tokens=0),
        token_counter=_WordCounter(),
    ).chunk(document)

    joined = "\n".join(chunk.text for chunk in chunks)
    assert "```bash\nexport A=1\nexport B=2\n```" in joined
    assert any("code" in chunk.metadata["block_kinds"] for chunk in chunks)


def test_document_overlap_is_bounded_and_reported():
    document = _document("# H\n\none two\n\nthree four\n\nfive six\n\nseven eight")
    chunks = DocumentChunker(
        config=ChunkingConfig(max_tokens=5, overlap_tokens=2),
        token_counter=_WordCounter(),
    ).chunk(document)

    assert len(chunks) >= 2
    assert all(chunk.token_count <= 5 for chunk in chunks)
    assert any(chunk.metadata["overlap_block_count"] > 0 for chunk in chunks[1:])


def test_oversized_document_block_is_split_to_hard_limit():
    document = _document("# H\n\n" + " ".join(f"w{i}" for i in range(30)))
    chunks = DocumentChunker(
        config=ChunkingConfig(max_tokens=7, overlap_tokens=0),
        token_counter=_WordCounter(),
    ).chunk(document)

    assert len(chunks) > 1
    assert all(chunk.token_count <= 7 for chunk in chunks)


def test_conversation_chunker_uses_structured_turns_and_locators():
    document = _conversation(
        [
            {
                "role": "user",
                "text": "why selector slow",
                "message_id": "u1",
                "node_id": "n1",
                "create_time": 1.0,
            },
            {
                "role": "assistant",
                "text": "because qk scoring costs time",
                "message_id": "a1",
                "node_id": "n2",
                "create_time": 2.0,
            },
            {
                "role": "user",
                "text": "show benchmark",
                "message_id": "u2",
                "node_id": "n3",
                "create_time": 3.0,
            },
        ]
    )
    chunks = ConversationChunker(
        config=ChunkingConfig(max_tokens=20, overlap_tokens=0, conversation_overlap_turns=0),
        token_counter=_WordCounter(),
    ).chunk(document)

    assert chunks[0].metadata["turn_start"] == 0
    assert chunks[-1].metadata["turn_end"] == 2
    assert "u1" in chunks[0].metadata["message_ids"]
    assert "n1" in chunks[0].metadata["node_ids"]
    assert "## User" in chunks[0].text
    assert all(chunk.token_count <= 20 for chunk in chunks)


def test_conversation_overlap_keeps_whole_turns():
    turns = [
        {"role": "user", "text": "u one two", "message_id": "u1"},
        {"role": "assistant", "text": "a one two", "message_id": "a1"},
        {"role": "user", "text": "u three four", "message_id": "u2"},
        {"role": "assistant", "text": "a three four", "message_id": "a2"},
    ]
    chunks = ConversationChunker(
        config=ChunkingConfig(max_tokens=10, overlap_tokens=0, conversation_overlap_turns=1),
        token_counter=_WordCounter(),
    ).chunk(_conversation(turns))

    assert len(chunks) >= 2
    assert any(chunk.metadata["overlap_turn_count"] == 1 for chunk in chunks[1:])


def test_oversized_single_turn_is_split_and_marked():
    document = _conversation(
        [{"role": "user", "text": " ".join(f"word{i}" for i in range(40)), "message_id": "u1"}]
    )
    chunks = ConversationChunker(
        config=ChunkingConfig(max_tokens=10, overlap_tokens=0, conversation_overlap_turns=0),
        token_counter=_WordCounter(),
    ).chunk(document)

    assert len(chunks) > 1
    assert all(chunk.metadata["contains_split_turn"] for chunk in chunks)
    assert all(chunk.token_count <= 10 for chunk in chunks)
    assert all(chunk.text.startswith("## User") for chunk in chunks)


def test_conversation_without_turn_metadata_falls_back_to_markdown_chunking():
    document = NormalizedDocument(
        document_id="conversation:x",
        title="fallback",
        content="## User\n\nhello\n\n## Assistant\n\nworld",
        source=SourceReference(
            source_type=SourceType.CONVERSATION,
            provider="other-agent",
            uri="file:///tmp/x.json",
        ),
    )

    chunks = chunk_document(
        document,
        config=ChunkingConfig(max_tokens=20, overlap_tokens=0),
        token_counter=_WordCounter(),
    )

    assert chunks
    assert all(chunk.metadata["conversation_fallback"] is True for chunk in chunks)


def test_chunk_inherits_acl_domain_project_and_source():
    chunk = chunk_document(
        _document("# H\n\nbody"),
        config=ChunkingConfig(max_tokens=20, overlap_tokens=0),
        token_counter=_WordCounter(),
    )[0]

    assert chunk.domain == "ai-ssd"
    assert chunk.project == "sparse-kv"
    assert chunk.security.tenant_id == "company"
    assert chunk.security.allowed_groups == ["aissd"]
    assert chunk.source.uri == "s3://kb-raw/design.pdf"


def test_chunk_ids_are_deterministic():
    document = _document("# H\n\nalpha beta gamma")
    kwargs = {
        "config": ChunkingConfig(max_tokens=20, overlap_tokens=0),
        "token_counter": _WordCounter(),
    }
    first = chunk_document(document, **kwargs)
    second = chunk_document(document, **kwargs)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert [chunk.content_hash for chunk in first] == [chunk.content_hash for chunk in second]


def test_chunk_documents_preserves_document_order():
    first = _document("# A\n\nalpha", document_id="doc-a")
    second = _document("# B\n\nbeta", document_id="doc-b")

    chunks = chunk_documents(
        [first, second],
        config=ChunkingConfig(max_tokens=20, overlap_tokens=0),
        token_counter=_WordCounter(),
    )

    assert [chunk.document_id for chunk in chunks] == ["doc-a", "doc-b"]
