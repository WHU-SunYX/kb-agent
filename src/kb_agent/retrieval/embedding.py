"""Embedding abstractions and a local BGE-M3 backend for kb-agent.

The retrieval layer depends on the small :class:`EmbeddingBackend` protocol,
not directly on FlagEmbedding.  The first implementation, :class:`LocalBGEBackend`,
loads a BGE-M3 model in the kb-agent process and supports either a local model
path or a Hugging Face model identifier.

This separation is intentional: a future HTTP/gRPC embedding service can
implement the same protocol without changing ingestion, Milvus indexing, or
retrieval orchestration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol, Sequence, runtime_checkable

from kb_agent.models import KnowledgeChunk


class EmbeddingError(RuntimeError):
    """Raised when an embedding backend cannot produce a valid dense vector."""


EmbeddingKind = Literal["query", "document"]
EmbeddingVector = list[float]


@dataclass(frozen=True)
class EmbeddingBatch:
    """Dense vectors returned by an embedding backend.

    ``vectors`` always contains plain Python ``float`` values so this object is
    independent of NumPy, PyTorch, or any particular inference runtime.
    """

    vectors: list[EmbeddingVector]
    model_name: str
    kind: EmbeddingKind
    dimension: int | None
    normalized: bool


@dataclass(frozen=True)
class ChunkEmbeddingBatch:
    """Dense vectors aligned one-to-one with knowledge-chunk identifiers."""

    chunk_ids: list[str]
    vectors: list[EmbeddingVector]
    model_name: str
    dimension: int | None
    normalized: bool


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Provider-neutral dense embedding contract used by kb-agent."""

    @property
    def model_name(self) -> str:
        """Human-readable model identifier or local model path."""

    @property
    def dimension(self) -> int | None:
        """Dense vector dimension, if known before inference."""

    def embed_queries(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Embed search queries."""

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Embed corpus/document text."""


@dataclass(frozen=True)
class LocalBGEConfig:
    """Configuration for the in-process BGE-M3 backend.

    ``model_name_or_path`` may be a local directory such as
    ``/models/BAAI/bge-m3``.  It may also be a Hugging Face model identifier for
    development environments that allow downloads.

    BGE-M3's dense dimension is 1024, therefore ``expected_dimension`` defaults
    to 1024.  A locally fine-tuned compatible model normally keeps that
    dimension; callers can override it when intentionally using another shape.
    """

    model_name_or_path: str = "BAAI/bge-m3"
    devices: str | list[str] | None = None
    use_fp16: bool = False
    use_bf16: bool = False
    normalize_embeddings: bool = True
    batch_size: int = 32
    query_max_length: int = 512
    passage_max_length: int = 512
    expected_dimension: int | None = 1024
    pooling_method: str = "cls"
    cache_dir: str | None = None
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if not self.model_name_or_path.strip():
            raise ValueError("model_name_or_path must not be blank")
        if self.use_fp16 and self.use_bf16:
            raise ValueError("use_fp16 and use_bf16 cannot both be enabled")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if self.query_max_length <= 0:
            raise ValueError("query_max_length must be greater than 0")
        if self.passage_max_length <= 0:
            raise ValueError("passage_max_length must be greater than 0")
        if self.expected_dimension is not None and self.expected_dimension <= 0:
            raise ValueError("expected_dimension must be greater than 0 when set")
        if not self.pooling_method.strip():
            raise ValueError("pooling_method must not be blank")
        if isinstance(self.devices, list) and not self.devices:
            raise ValueError("devices must not be an empty list")


ModelFactory = Callable[[LocalBGEConfig], Any]


def _default_bge_model_factory(config: LocalBGEConfig) -> Any:
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:  # pragma: no cover - depends on optional runtime dependency
        raise EmbeddingError(
            "LocalBGEBackend requires FlagEmbedding. Install kb-agent with "
            '`pip install -e ".[embedding]"` (or include the embedding extra '
            "with your other extras)."
        ) from exc

    try:
        return BGEM3FlagModel(
            config.model_name_or_path,
            normalize_embeddings=config.normalize_embeddings,
            use_fp16=config.use_fp16,
            use_bf16=config.use_bf16,
            devices=config.devices,
            pooling_method=config.pooling_method,
            trust_remote_code=config.trust_remote_code,
            cache_dir=config.cache_dir,
            batch_size=config.batch_size,
            query_max_length=config.query_max_length,
            passage_max_length=config.passage_max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
    except Exception as exc:  # pragma: no cover - model/runtime specific
        raise EmbeddingError(
            f"failed to load local BGE model {config.model_name_or_path!r}: {exc}"
        ) from exc


class LocalBGEBackend:
    """In-process BGE-M3 dense embedding backend.

    The model is loaded lazily on the first non-empty embedding request.  This
    keeps CLI/API startup cheap and avoids allocating GPU/CPU model memory when
    a process only performs non-embedding work.
    """

    def __init__(
        self,
        config: LocalBGEConfig | None = None,
        *,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self.config = config or LocalBGEConfig()
        self._model_factory = model_factory or _default_bge_model_factory
        self._model: Any | None = None
        self._dimension: int | None = self.config.expected_dimension

    @property
    def model_name(self) -> str:
        return self.config.model_name_or_path

    @property
    def dimension(self) -> int | None:
        return self._dimension

    @property
    def is_loaded(self) -> bool:
        """Whether the heavyweight model runtime has already been created."""

        return self._model is not None

    def embed_queries(self, texts: Sequence[str]) -> EmbeddingBatch:
        return self._embed(texts, kind="query")

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch:
        return self._embed(texts, kind="document")

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = self._model_factory(self.config)
        return self._model

    def _embed(self, texts: Sequence[str], *, kind: EmbeddingKind) -> EmbeddingBatch:
        normalized_texts = _validate_texts(texts)
        if not normalized_texts:
            return EmbeddingBatch(
                vectors=[],
                model_name=self.model_name,
                kind=kind,
                dimension=self.dimension,
                normalized=self.config.normalize_embeddings,
            )

        model = self._get_model()
        try:
            if kind == "query":
                output = model.encode_queries(
                    normalized_texts,
                    batch_size=self.config.batch_size,
                    max_length=self.config.query_max_length,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
            else:
                output = model.encode_corpus(
                    normalized_texts,
                    batch_size=self.config.batch_size,
                    max_length=self.config.passage_max_length,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                f"{self.model_name!r} failed to embed {kind} batch: {exc}"
            ) from exc

        vectors = _extract_dense_vectors(output)
        dimension = _validate_vectors(
            vectors,
            expected_count=len(normalized_texts),
            expected_dimension=self.config.expected_dimension,
        )
        self._dimension = dimension

        return EmbeddingBatch(
            vectors=vectors,
            model_name=self.model_name,
            kind=kind,
            dimension=dimension,
            normalized=self.config.normalize_embeddings,
        )


def create_embedding_backend(
    config: Any,
    *,
    model_factory: ModelFactory | None = None,
) -> EmbeddingBackend:
    """Create the configured embedding backend from application settings.

    The argument is intentionally duck-typed so the retrieval module does not
    require Pydantic. It is compatible with ``kb_agent.config.EmbeddingConfig``.
    """

    backend_name = getattr(config, "backend", None)
    if backend_name != "local_bge":
        raise ValueError(f"unsupported embedding backend: {backend_name!r}")

    runtime = LocalBGEConfig(
        model_name_or_path=config.model_name_or_path,
        devices=config.devices,
        use_fp16=config.use_fp16,
        use_bf16=config.use_bf16,
        normalize_embeddings=config.normalize_embeddings,
        batch_size=config.batch_size,
        query_max_length=config.query_max_length,
        passage_max_length=config.passage_max_length,
        expected_dimension=config.expected_dimension,
        pooling_method=config.pooling_method,
        cache_dir=config.cache_dir,
        trust_remote_code=config.trust_remote_code,
    )
    return LocalBGEBackend(runtime, model_factory=model_factory)


def embed_chunks(
    chunks: Sequence[KnowledgeChunk],
    backend: EmbeddingBackend,
) -> ChunkEmbeddingBatch:
    """Embed chunk text while preserving explicit chunk/vector alignment."""

    chunk_list = list(chunks)
    batch = backend.embed_documents([chunk.text for chunk in chunk_list])
    if len(batch.vectors) != len(chunk_list):
        raise EmbeddingError(
            "embedding backend returned a different number of vectors than chunks: "
            f"{len(batch.vectors)} != {len(chunk_list)}"
        )

    return ChunkEmbeddingBatch(
        chunk_ids=[chunk.chunk_id for chunk in chunk_list],
        vectors=batch.vectors,
        model_name=batch.model_name,
        dimension=batch.dimension,
        normalized=batch.normalized,
    )


def _validate_texts(texts: Sequence[str]) -> list[str]:
    if isinstance(texts, (str, bytes)):
        raise TypeError("texts must be a sequence of strings, not a single string")

    normalized = list(texts)
    for index, text in enumerate(normalized):
        if not isinstance(text, str):
            raise TypeError(f"texts[{index}] must be str, got {type(text).__name__}")
        if not text.strip():
            raise ValueError(f"texts[{index}] must not be blank")
    return normalized


def _extract_dense_vectors(output: Any) -> list[EmbeddingVector]:
    if not isinstance(output, dict) or "dense_vecs" not in output:
        raise EmbeddingError("embedding backend output does not contain 'dense_vecs'")

    dense = output["dense_vecs"]
    if dense is None:
        raise EmbeddingError("embedding backend returned dense_vecs=None")

    if hasattr(dense, "tolist"):
        dense = dense.tolist()

    if not isinstance(dense, (list, tuple)):
        raise EmbeddingError(
            f"dense_vecs must be an array-like object, got {type(dense).__name__}"
        )

    # Be defensive for runtimes that collapse a batch-of-one to a single vector.
    if dense and _is_number(dense[0]):
        dense = [dense]

    vectors: list[EmbeddingVector] = []
    for row_index, row in enumerate(dense):
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, (list, tuple)):
            raise EmbeddingError(f"dense_vecs[{row_index}] is not a vector")

        vector: EmbeddingVector = []
        for col_index, value in enumerate(row):
            if not _is_number(value):
                raise EmbeddingError(
                    f"dense_vecs[{row_index}][{col_index}] is not numeric"
                )
            numeric = float(value)
            if not math.isfinite(numeric):
                raise EmbeddingError(
                    f"dense_vecs[{row_index}][{col_index}] is not finite"
                )
            vector.append(numeric)
        vectors.append(vector)
    return vectors


def _validate_vectors(
    vectors: Sequence[EmbeddingVector],
    *,
    expected_count: int,
    expected_dimension: int | None,
) -> int:
    if len(vectors) != expected_count:
        raise EmbeddingError(
            "embedding backend returned an unexpected number of vectors: "
            f"{len(vectors)} != {expected_count}"
        )
    if not vectors:
        raise EmbeddingError("embedding backend returned no vectors for a non-empty batch")

    dimension = len(vectors[0])
    if dimension <= 0:
        raise EmbeddingError("embedding vectors must not be empty")

    for index, vector in enumerate(vectors):
        if len(vector) != dimension:
            raise EmbeddingError(
                f"embedding vector dimension mismatch at row {index}: "
                f"{len(vector)} != {dimension}"
            )

    if expected_dimension is not None and dimension != expected_dimension:
        raise EmbeddingError(
            "embedding dimension does not match configured expected_dimension: "
            f"{dimension} != {expected_dimension}"
        )
    return dimension


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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
