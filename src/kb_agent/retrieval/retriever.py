"""Hybrid retrieval service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .embedding import EmbeddingBackend
from .filters import RetrievalFilters, RetrievalScope


class RetrievalError(RuntimeError):
    pass


class VectorStoreProtocol(Protocol):
    def search_dense(
        self,
        vector: Sequence[float],
        limit: int = 10,
        filters: RetrievalFilters | None = None,
        scope: RetrievalScope | None = None,
    ) -> Any:
        ...

    def search_bm25(
        self,
        text: str,
        limit: int = 10,
        filters: RetrievalFilters | None = None,
        scope: RetrievalScope | None = None,
    ) -> Any:
        ...


@dataclass(slots=True)
class RetrievalConfig:
    top_k_dense: int = 20
    top_k_bm25: int = 20
    top_k_final: int = 10


@dataclass(slots=True)
class RetrievalHit:
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]


class HybridRetrievalService:
    def __init__(
        self,
        embedding: EmbeddingBackend,
        store: VectorStoreProtocol,
        config: RetrievalConfig | None = None,
        *,
        tenant_id: str | None = None,
    ):
        self.embedding = embedding
        self.store = store
        self.config = config or RetrievalConfig()
        self.scope = RetrievalScope(tenant_id) if tenant_id is not None else None

    def search(
        self,
        query: str,
        filters: RetrievalFilters | None = None,
        limit: int | None = None,
    ) -> list[RetrievalHit]:
        requested_limit = self.config.top_k_final if limit is None else limit
        if requested_limit <= 0:
            raise ValueError("limit must be greater than 0")

        query_vec = self.embedding.embed_queries([query]).vectors[0]

        dense = self.store.search_dense(
            query_vec,
            limit=max(self.config.top_k_dense, requested_limit),
            filters=filters,
            scope=self.scope,
        )

        bm25 = []
        if hasattr(self.store, "search_bm25"):
            bm25 = self.store.search_bm25(
                query,
                limit=max(self.config.top_k_bm25, requested_limit),
                filters=filters,
                scope=self.scope,
            )

        return self._fuse(dense, bm25)[:requested_limit]

    def _fuse(self, dense: Any, bm25: Any) -> list[RetrievalHit]:
        """Initial placeholder fusion.

        Production version can replace this with RRF/weighted fusion.
        """
        results: list[RetrievalHit] = []

        for item in self._normalize_hits(dense):
            results.append(item)

        existing = {x.chunk_id for x in results}
        for item in self._normalize_hits(bm25):
            if item.chunk_id not in existing:
                results.append(item)

        return results

    @staticmethod
    def _normalize_hits(raw: Any) -> list[RetrievalHit]:
        if not raw:
            return []

        output = []
        for item in raw:
            if isinstance(item, RetrievalHit):
                output.append(item)
            elif isinstance(item, dict):
                output.append(
                    RetrievalHit(
                        chunk_id=str(item.get("id") or item.get("chunk_id")),
                        score=float(item.get("score", 0)),
                        text=item.get("text", ""),
                        metadata=item,
                    )
                )
        return output
