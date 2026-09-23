from kb_agent.retrieval.schema import build_kb_chunk_schema
from kb_agent.retrieval.milvus import MilvusSchemaConfig


def test_schema():
    s = build_kb_chunk_schema(MilvusSchemaConfig())
    assert s["fields"][3].name == "dense_vector"
    assert s["functions"]["bm25"] is True
