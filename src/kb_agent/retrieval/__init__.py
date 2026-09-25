"""Retrieval-layer adapters and services."""

from kb_agent.retrieval.filters import (
    RetrievalFilterError,
    RetrievalFilters,
    RetrievalScope,
)
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
    "RetrievalFilterError",
    "RetrievalFilters",
    "RetrievalScope",
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
