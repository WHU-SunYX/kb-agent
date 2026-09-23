"""Retrieval-layer adapters and services."""

from kb_agent.retrieval.embedding import (
    ChunkEmbeddingBatch,
    EmbeddingBackend,
    EmbeddingBatch,
    EmbeddingError,
    EmbeddingKind,
    EmbeddingVector,
    LocalBGEBackend,
    LocalBGEConfig,
    create_embedding_backend,
    embed_chunks,
)

__all__ = [
    "ChunkEmbeddingBatch",
    "EmbeddingBackend",
    "EmbeddingBatch",
    "EmbeddingError",
    "EmbeddingKind",
    "EmbeddingVector",
    "LocalBGEBackend",
    "LocalBGEConfig",
    "create_embedding_backend",
    "embed_chunks",
]
