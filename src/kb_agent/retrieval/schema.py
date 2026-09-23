"""Milvus schema and index builders for kb-agent.

This module centralizes pymilvus-specific schema/index construction so the rest
of kb-agent can depend on stable application-level interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class MilvusSchemaConfig:
    collection_name: str = "kb_chunks"
    dimension: int = 1024
    metric_type: str = "COSINE"
    enable_bm25: bool = True


# Backward-compatible alias for code created before the production_ naming
# cleanup. New code should use MilvusSchemaConfig.
ProductionMilvusSchemaConfig = MilvusSchemaConfig


def build_pymilvus_schema(config: MilvusSchemaConfig) -> Any:
    try:
        from pymilvus import DataType, Function, FunctionType, MilvusClient
    except ImportError as exc:
        raise RuntimeError(
            "pymilvus is required for Milvus schema creation"
        ) from exc

    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )

    schema.add_field(
        field_name="chunk_id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=128,
    )
    schema.add_field(
        field_name="document_id",
        datatype=DataType.VARCHAR,
        max_length=128,
    )
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=65535,
        enable_analyzer=True,
    )
    schema.add_field(
        field_name="dense_vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=config.dimension,
    )
    schema.add_field(
        field_name="tenant_id",
        datatype=DataType.VARCHAR,
        max_length=128,
    )
    schema.add_field(
        field_name="domain",
        datatype=DataType.VARCHAR,
        max_length=128,
    )
    schema.add_field(
        field_name="project",
        datatype=DataType.VARCHAR,
        max_length=128,
    )
    schema.add_field(
        field_name="classification",
        datatype=DataType.VARCHAR,
        max_length=64,
    )
    schema.add_field(
        field_name="source_type",
        datatype=DataType.VARCHAR,
        max_length=64,
    )
    schema.add_field(
        field_name="provider",
        datatype=DataType.VARCHAR,
        max_length=128,
    )
    schema.add_field(
        field_name="source_id",
        datatype=DataType.VARCHAR,
        max_length=256,
    )
    schema.add_field(
        field_name="metadata_json",
        datatype=DataType.JSON,
    )

    if config.enable_bm25:
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )
        schema.add_function(
            Function(
                name="text_bm25",
                input_field_names=["text"],
                output_field_names=["sparse_vector"],
                function_type=FunctionType.BM25,
            )
        )

    return schema


def build_pymilvus_index_params(
    client: Any,
    config: MilvusSchemaConfig,
) -> Any:
    """Build dense and optional BM25 index definitions for one collection."""

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="AUTOINDEX",
        metric_type=config.metric_type,
    )

    if config.enable_bm25:
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )

    return index_params


__all__ = [
    "MilvusSchemaConfig",
    "ProductionMilvusSchemaConfig",
    "build_pymilvus_index_params",
    "build_pymilvus_schema",
]
