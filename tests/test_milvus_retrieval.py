from kb_agent.retrieval.milvus import (
    ChunkRecord,
    MilvusConfig,
    MilvusStore,
)


class FakeMilvus:
    def __init__(self):
        self.inserted = []

    def has_collection(self, name):
        return False

    def create_collection(self, **kwargs):
        self.created = kwargs

    def insert(self, **kwargs):
        self.inserted.extend(kwargs["data"])

    def search(self, **kwargs):
        return kwargs


def test_insert_chunks():
    fake = FakeMilvus()
    store = MilvusStore(
        MilvusConfig(dimension=3),
        client=fake,
    )

    store.ensure_collection()

    count = store.upsert_chunks(
        [
            ChunkRecord(
                chunk_id="c1",
                document_id="d1",
                text="hello",
                vector=[0.1, 0.2, 0.3],
                metadata={"domain": "ai-ssd"},
            )
        ]
    )

    assert count == 1
    assert fake.inserted[0]["id"] == "c1"


def test_dense_search():
    fake = FakeMilvus()
    store = MilvusStore(MilvusConfig(), client=fake)

    result = store.search_dense([0.1], limit=5)

    assert result["limit"] == 5
