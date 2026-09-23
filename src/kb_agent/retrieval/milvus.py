"""Milvus integration for kb-agent.

This module keeps Milvus-specific persistence/search code behind a small service
boundary. It stores retrieval-ready KnowledgeChunk records, dense BGE vectors,
and optionally uses Milvus' built-in BM25 Function for full-text retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence

from kb_agent.models import KnowledgeChunk
from kb_agent.retrieval.embedding import EmbeddingBackend, embed_chunks
from kb_agent.retrieval.schema import (
    MilvusSchemaConfig,
    build_pymilvus_index_params,
    build_pymilvus_schema,
)


class MilvusError(RuntimeError):
    pass


@dataclass(slots=True)
class MilvusConfig:
    uri: str = "http://localhost:19530"
    collection_name: str = "kb_chunks"
    dimension: int = 1024
    metric_type: str = "COSINE"
    consistency_level: str = "Bounded"
    enable_bm25: bool = True


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    text: str
    vector: Sequence[float]
    metadata: dict[str, Any]


class MilvusClientProtocol(Protocol):
    def has_collection(self, *args: Any, **kwargs: Any) -> Any: ...
    def describe_collection(self, *args: Any, **kwargs: Any) -> Any: ...
    def create_collection(self, *args: Any, **kwargs: Any) -> Any: ...
    def prepare_index_params(self, *args: Any, **kwargs: Any) -> Any: ...
    def create_index(self, *args: Any, **kwargs: Any) -> Any: ...
    def load_collection(self, *args: Any, **kwargs: Any) -> Any: ...
    def insert(self, *args: Any, **kwargs: Any) -> Any: ...
    def upsert(self, *args: Any, **kwargs: Any) -> Any: ...
    def search(self, *args: Any, **kwargs: Any) -> Any: ...


class MilvusStore:
    """Thin adapter around pymilvus.

    Retrieval ranking/fusion does not belong here. This class owns collection
    lifecycle, persistence, and the individual dense/BM25 search primitives.
    """

    def __init__(
        self,
        config: MilvusConfig,
        client: MilvusClientProtocol | None = None,
    ):
        self.config = config
        self._client = client
        self._collection_ready = False

    @property
    def collection_name(self) -> str:
        return self.config.collection_name

    def _ensure_client(self) -> MilvusClientProtocol:
        if self._client is None:
            try:
                from pymilvus import MilvusClient
            except ImportError as exc:
                raise MilvusError(
                    "pymilvus is required for MilvusStore"
                ) from exc

            self._client = MilvusClient(uri=self.config.uri)

        return self._client

    def ensure_collection(self) -> None:
        """Create/load the kb-agent collection with dense and optional BM25 indexes."""

        if self._collection_ready:
            return

        client = self._ensure_client()

        exists = bool(client.has_collection(collection_name=self.collection_name))
        if exists:
            self._validate_existing_collection(client)
        else:
            schema_config = MilvusSchemaConfig(
                collection_name=self.collection_name,
                dimension=self.config.dimension,
                metric_type=self.config.metric_type,
                enable_bm25=self.config.enable_bm25,
            )
            schema = build_pymilvus_schema(schema_config)
            index_params = build_pymilvus_index_params(client, schema_config)

            client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
                consistency_level=self.config.consistency_level,
            )

        if hasattr(client, "load_collection"):
            client.load_collection(
                collection_name=self.collection_name,
            )

        self._collection_ready = True

    def _validate_existing_collection(self, client: MilvusClientProtocol) -> None:
        """Fail clearly if an older quick-schema collection already exists."""

        if not hasattr(client, "describe_collection"):
            return

        description = client.describe_collection(
            collection_name=self.collection_name,
        )
        if not isinstance(description, dict):
            return

        raw_fields = description.get("fields")
        if not isinstance(raw_fields, list):
            return

        names: set[str] = set()
        for field in raw_fields:
            if not isinstance(field, dict):
                continue
            name = field.get("name") or field.get("field_name")
            if name:
                names.add(str(name))

        required = {
            "chunk_id",
            "document_id",
            "text",
            "dense_vector",
            "tenant_id",
            "domain",
            "project",
            "classification",
            "source_type",
            "provider",
            "source_id",
            "metadata_json",
        }
        if self.config.enable_bm25:
            required.add("sparse_vector")

        missing = sorted(required - names)
        if missing:
            raise MilvusError(
                f"existing collection {self.collection_name!r} is incompatible with "
                f"the kb-agent schema; missing fields: {', '.join(missing)}. "
                "Drop/recreate that development collection or migrate it before indexing."
            )

    def upsert_chunks(self, chunks: Iterable[ChunkRecord]) -> int:
        records = list(chunks)

        if not records:
            return 0

        self.ensure_collection()
        client = self._ensure_client()

        data: list[dict[str, Any]] = []
        for record in records:
            vector = list(record.vector)
            if len(vector) != self.config.dimension:
                raise MilvusError(
                    f"chunk {record.chunk_id!r} vector dimension {len(vector)} "
                    f"does not match Milvus dimension {self.config.dimension}"
                )

            metadata = dict(record.metadata)
            data.append(
                {
                    "chunk_id": record.chunk_id,
                    "document_id": record.document_id,
                    "text": record.text,
                    "dense_vector": vector,
                    "tenant_id": str(metadata.pop("tenant_id", "default")),
                    "domain": str(metadata.pop("domain", "") or ""),
                    "project": str(metadata.pop("project", "") or ""),
                    "classification": str(metadata.pop("classification", "internal")),
                    "source_type": str(metadata.pop("source_type", "") or ""),
                    "provider": str(metadata.pop("provider", "") or ""),
                    "source_id": str(metadata.pop("source_id", "") or ""),
                    "metadata_json": metadata.pop("metadata_json", metadata),
                }
            )

        if hasattr(client, "upsert"):
            client.upsert(
                collection_name=self.collection_name,
                data=data,
            )
        else:
            # Compatibility path for lightweight test doubles.
            client.insert(
                collection_name=self.collection_name,
                data=data,
            )

        return len(records)

    def search_dense(
        self,
        vector: Sequence[float],
        limit: int = 10,
        filter_expr: str | None = None,
    ) -> list[dict[str, Any]]:
        self.ensure_collection()
        client = self._ensure_client()

        if len(vector) != self.config.dimension:
            raise MilvusError(
                f"query vector dimension {len(vector)} does not match "
                f"Milvus dimension {self.config.dimension}"
            )

        kwargs: dict[str, Any] = {
            "collection_name": self.collection_name,
            "data": [list(vector)],
            "anns_field": "dense_vector",
            "limit": limit,
            "output_fields": _SEARCH_OUTPUT_FIELDS,
            "search_params": {
                "metric_type": self.config.metric_type,
            },
            "consistency_level": self.config.consistency_level,
        }

        if filter_expr:
            kwargs["filter"] = filter_expr

        return _normalize_search_results(client.search(**kwargs))

    def search_bm25(
        self,
        text: str,
        limit: int = 10,
        filter_expr: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.config.enable_bm25:
            return []

        self.ensure_collection()
        client = self._ensure_client()

        kwargs: dict[str, Any] = {
            "collection_name": self.collection_name,
            "data": [text],
            "anns_field": "sparse_vector",
            "limit": limit,
            "output_fields": _SEARCH_OUTPUT_FIELDS,
            "search_params": {
                "metric_type": "BM25",
            },
            "consistency_level": self.config.consistency_level,
        }

        if filter_expr:
            kwargs["filter"] = filter_expr

        return _normalize_search_results(client.search(**kwargs))


class MilvusChunkIndexer:
    """Embed KnowledgeChunks and persist them to Milvus as one indexing stage."""

    def __init__(
        self,
        *,
        embedding: EmbeddingBackend,
        store: MilvusStore,
    ) -> None:
        self.embedding = embedding
        self.store = store

    def index(self, chunks: Sequence[KnowledgeChunk]) -> int:
        chunk_list = list(chunks)
        if not chunk_list:
            return 0

        embedded = embed_chunks(chunk_list, self.embedding)
        if embedded.dimension is not None and embedded.dimension != self.store.config.dimension:
            raise MilvusError(
                "embedding dimension does not match Milvus collection dimension: "
                f"{embedded.dimension} != {self.store.config.dimension}"
            )

        records = [
            _chunk_record(chunk, vector)
            for chunk, vector in zip(chunk_list, embedded.vectors, strict=True)
        ]
        return self.store.upsert_chunks(records)


_SEARCH_OUTPUT_FIELDS = [
    "text",
    "document_id",
    "tenant_id",
    "domain",
    "project",
    "classification",
    "source_type",
    "provider",
    "source_id",
    "metadata_json",
]


def _chunk_record(
    chunk: KnowledgeChunk,
    vector: Sequence[float],
) -> ChunkRecord:
    security = chunk.security.model_dump(mode="json")
    source = chunk.source.model_dump(mode="json")

    classification = security.get("classification", "internal")
    if hasattr(classification, "value"):
        classification = classification.value

    return ChunkRecord(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        text=chunk.text,
        vector=vector,
        metadata={
            "tenant_id": security.get("tenant_id", "default"),
            "domain": chunk.domain or "",
            "project": chunk.project or "",
            "classification": str(classification),
            "source_type": str(source.get("source_type", "") or ""),
            "provider": str(source.get("provider", "") or ""),
            "source_id": str(source.get("source_id", "") or ""),
            "metadata_json": {
                "title": chunk.title,
                "section_path": list(chunk.section_path),
                "chunk_index": chunk.chunk_index,
                "content_hash": chunk.content_hash,
                "source": source,
                "security": security,
                "chunk": chunk.metadata,
            },
        },
    )


def _normalize_search_results(raw: Any) -> list[dict[str, Any]]:
    """Flatten one-query Milvus search output into retriever-friendly dictionaries."""

    if not raw:
        return []

    rows = raw
    if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], (list, tuple)):
        rows = raw[0]

    output: list[dict[str, Any]] = []
    for hit in rows:
        if not isinstance(hit, dict):
            continue

        entity = hit.get("entity")
        if not isinstance(entity, dict):
            entity = {}

        chunk_id = hit.get("id") or hit.get("chunk_id") or entity.get("chunk_id")
        if chunk_id is None:
            continue

        score = hit.get("distance", hit.get("score", 0.0))
        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            numeric_score = 0.0

        output.append(
            {
                "id": str(chunk_id),
                "chunk_id": str(chunk_id),
                "score": numeric_score,
                "text": entity.get("text", hit.get("text", "")),
                "document_id": entity.get("document_id", hit.get("document_id")),
                "tenant_id": entity.get("tenant_id", hit.get("tenant_id")),
                "domain": entity.get("domain", hit.get("domain")),
                "project": entity.get("project", hit.get("project")),
                "classification": entity.get(
                    "classification",
                    hit.get("classification"),
                ),
                "source_type": entity.get("source_type", hit.get("source_type")),
                "provider": entity.get("provider", hit.get("provider")),
                "source_id": entity.get("source_id", hit.get("source_id")),
                "metadata_json": entity.get(
                    "metadata_json",
                    hit.get("metadata_json", {}),
                ),
            }
        )

    return output


__all__ = [
    "ChunkRecord",
    "MilvusChunkIndexer",
    "MilvusConfig",
    "MilvusError",
    "MilvusStore",
]
