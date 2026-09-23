from __future__ import annotations

from dataclasses import replace

import pytest

from kb_agent.models import KnowledgeChunk, SourceReference, SourceType
from kb_agent.retrieval.embedding import (
    EmbeddingError,
    LocalBGEBackend,
    LocalBGEConfig,
    embed_chunks,
)


class _Array:
    """Tiny NumPy-like test object exposing tolist()."""

    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


class _FakeBGEModel:
    def __init__(self, *, dimension: int = 3):
        self.dimension = dimension
        self.calls: list[tuple[str, list[str], dict]] = []

    def _result(self, texts):
        rows = [
            [float(index + 1)] * self.dimension
            for index, _ in enumerate(texts)
        ]
        return {"dense_vecs": _Array(rows)}

    def encode_queries(self, texts, **kwargs):
        self.calls.append(("query", list(texts), kwargs))
        return self._result(texts)

    def encode_corpus(self, texts, **kwargs):
        self.calls.append(("document", list(texts), kwargs))
        return self._result(texts)


def _config(**changes) -> LocalBGEConfig:
    base = LocalBGEConfig(
        model_name_or_path="/models/bge-m3",
        expected_dimension=3,
        batch_size=8,
        query_max_length=128,
        passage_max_length=512,
    )
    return replace(base, **changes)


def _chunk(chunk_id: str, text: str) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        chunk_index=int(chunk_id.rsplit("-", 1)[-1]),
        text=text,
        source=SourceReference(
            source_type=SourceType.DOCUMENT,
            provider="pdf",
            uri="s3://kb-raw/doc.pdf",
        ),
    )


def test_local_bge_is_lazy_loaded_and_empty_batch_does_not_load_model():
    calls = []

    def factory(config):
        calls.append(config.model_name_or_path)
        return _FakeBGEModel()

    backend = LocalBGEBackend(_config(), model_factory=factory)

    assert backend.is_loaded is False
    empty = backend.embed_documents([])
    assert empty.vectors == []
    assert empty.dimension == 3
    assert backend.is_loaded is False
    assert calls == []


def test_local_bge_uses_distinct_query_and_corpus_paths():
    model = _FakeBGEModel()
    backend = LocalBGEBackend(_config(), model_factory=lambda _: model)

    queries = backend.embed_queries(["selector latency"])
    documents = backend.embed_documents(["QK scoring dominates latency"])

    assert queries.kind == "query"
    assert documents.kind == "document"
    assert queries.vectors == [[1.0, 1.0, 1.0]]
    assert documents.vectors == [[1.0, 1.0, 1.0]]
    assert model.calls[0][0] == "query"
    assert model.calls[1][0] == "document"
    assert model.calls[0][2]["max_length"] == 128
    assert model.calls[1][2]["max_length"] == 512
    assert model.calls[0][2]["return_sparse"] is False
    assert model.calls[1][2]["return_colbert_vecs"] is False


def test_model_factory_receives_local_model_configuration():
    received = []
    config = _config(devices=["cuda:0"], use_fp16=True)

    backend = LocalBGEBackend(
        config,
        model_factory=lambda cfg: received.append(cfg) or _FakeBGEModel(),
    )
    backend.embed_documents(["text"])

    assert received == [config]
    assert received[0].model_name_or_path == "/models/bge-m3"
    assert received[0].devices == ["cuda:0"]
    assert received[0].use_fp16 is True


def test_local_bge_validates_expected_dimension():
    backend = LocalBGEBackend(
        _config(expected_dimension=4),
        model_factory=lambda _: _FakeBGEModel(dimension=3),
    )

    with pytest.raises(EmbeddingError, match="expected_dimension"):
        backend.embed_documents(["text"])


def test_local_bge_rejects_inconsistent_vector_dimensions():
    class BadModel(_FakeBGEModel):
        def encode_corpus(self, texts, **kwargs):
            return {"dense_vecs": [[1.0, 2.0, 3.0], [1.0, 2.0]]}

    backend = LocalBGEBackend(
        _config(expected_dimension=None),
        model_factory=lambda _: BadModel(),
    )

    with pytest.raises(EmbeddingError, match="dimension mismatch"):
        backend.embed_documents(["a", "b"])


def test_local_bge_rejects_non_finite_vectors():
    class BadModel(_FakeBGEModel):
        def encode_corpus(self, texts, **kwargs):
            return {"dense_vecs": [[1.0, float("nan"), 3.0]]}

    backend = LocalBGEBackend(_config(), model_factory=lambda _: BadModel())

    with pytest.raises(EmbeddingError, match="not finite"):
        backend.embed_documents(["text"])


def test_local_bge_rejects_blank_text_and_single_string_argument():
    backend = LocalBGEBackend(_config(), model_factory=lambda _: _FakeBGEModel())

    with pytest.raises(ValueError, match="must not be blank"):
        backend.embed_documents(["   "])
    with pytest.raises(TypeError, match="sequence of strings"):
        backend.embed_queries("not-a-sequence")


def test_runtime_failure_is_wrapped_as_embedding_error():
    class FailingModel(_FakeBGEModel):
        def encode_queries(self, texts, **kwargs):
            raise RuntimeError("CUDA OOM")

    backend = LocalBGEBackend(_config(), model_factory=lambda _: FailingModel())

    with pytest.raises(EmbeddingError, match="failed to embed query batch"):
        backend.embed_queries(["query"])


def test_embed_chunks_preserves_chunk_vector_alignment():
    model = _FakeBGEModel()
    backend = LocalBGEBackend(_config(), model_factory=lambda _: model)
    chunks = [_chunk("chunk-0", "alpha"), _chunk("chunk-1", "beta")]

    result = embed_chunks(chunks, backend)

    assert result.chunk_ids == ["chunk-0", "chunk-1"]
    assert result.vectors == [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
    assert result.model_name == "/models/bge-m3"
    assert result.dimension == 3


def test_local_bge_config_rejects_invalid_runtime_settings():
    with pytest.raises(ValueError, match="cannot both"):
        LocalBGEConfig(use_fp16=True, use_bf16=True)
    with pytest.raises(ValueError, match="batch_size"):
        LocalBGEConfig(batch_size=0)
    with pytest.raises(ValueError, match="expected_dimension"):
        LocalBGEConfig(expected_dimension=0)
    with pytest.raises(ValueError, match="empty list"):
        LocalBGEConfig(devices=[])


def test_application_embedding_config_creates_local_backend(tmp_path):
    from kb_agent.config import EmbeddingConfig
    from kb_agent.retrieval.embedding import create_embedding_backend

    app_config = EmbeddingConfig(
        model_name_or_path=str(tmp_path / "bge-m3"),
        devices="cuda:0",
        use_fp16=True,
        expected_dimension=3,
    )
    received = []
    backend = create_embedding_backend(
        app_config,
        model_factory=lambda runtime: received.append(runtime) or _FakeBGEModel(),
    )

    result = backend.embed_documents(["text"])

    assert result.dimension == 3
    assert received[0].model_name_or_path == str(tmp_path / "bge-m3")
    assert received[0].devices == "cuda:0"
    assert received[0].use_fp16 is True
