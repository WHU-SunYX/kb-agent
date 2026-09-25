"""Application dependency assembly.

This module is the single place where kb-agent runtime objects are created.
FastAPI routes and MCP tools reuse the same application services.

Configuration source:
    kb_agent.config.load_settings()
"""

from __future__ import annotations

from functools import lru_cache

from kb_agent.config import load_settings

from kb_agent.storage.raw_store import LocalRawStore

from kb_agent.ingestion.pipeline import IngestionPipeline
from kb_agent.ingestion.security import SecurityScanner

from kb_agent.retrieval.embedding import (
    EmbeddingBackend,
    create_embedding_backend,
)

from kb_agent.retrieval.milvus import (
    MilvusChunkIndexer,
    MilvusConfig,
    MilvusStore,
)

from kb_agent.retrieval.retriever import HybridRetrievalService


@lru_cache(maxsize=1)
def get_settings():
    """Load validated application settings once."""
    return load_settings()


@lru_cache(maxsize=1)
def get_raw_store():
    cfg = get_settings().raw_store

    if cfg.type != "local":
        raise NotImplementedError(
            "S3 raw store backend is not wired in Step 13"
        )

    return LocalRawStore(
        root=cfg.local_root,
    )


@lru_cache(maxsize=1)
def get_embedding_backend() -> EmbeddingBackend:
    cfg = get_settings().embedding
    return create_embedding_backend(cfg)


@lru_cache(maxsize=1)
def get_milvus_store():
    cfg = get_settings().milvus

    return MilvusStore(
        MilvusConfig(
            uri=cfg.uri,
            collection_name=cfg.collection_name,
            dimension=cfg.dimension,
            metric_type=cfg.metric_type,
            consistency_level=cfg.consistency_level,
            enable_bm25=cfg.enable_bm25,
        )
    )


@lru_cache(maxsize=1)
def get_chunk_indexer():
    return MilvusChunkIndexer(
        embedding=get_embedding_backend(),
        store=get_milvus_store(),
    )


@lru_cache(maxsize=1)
def get_ingestion_pipeline():
    return IngestionPipeline(
        raw_store=get_raw_store(),
        indexer=get_chunk_indexer(),
        security_scanner=SecurityScanner(pii_device=get_settings().security.pii_device),
    )


@lru_cache(maxsize=1)
def get_retriever():
    return HybridRetrievalService(
        embedding=get_embedding_backend(),
        store=get_milvus_store(),
        tenant_id=get_settings().kb.default_tenant,
    )


def build_application_services():
    return {
        "ingestion_service": get_ingestion_pipeline(),
        "retrieval_service": get_retriever(),
    }
