from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from kb_agent.config import RawStoreConfig
from kb_agent.storage.raw_store import (
    InvalidRawStoreKey,
    InvalidRawStoreUri,
    LocalRawStore,
    RawObjectNotFound,
    S3RawStore,
    create_raw_store,
)


class FakeS3NotFound(Exception):
    def __init__(self) -> None:
        self.response = {"Error": {"Code": "NoSuchKey"}}
        super().__init__("NoSuchKey")


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
        self.objects[(Bucket, Key)] = bytes(Body)

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, BytesIO]:
        try:
            value = self.objects[(Bucket, Key)]
        except KeyError as exc:
            raise FakeS3NotFound() from exc
        return {"Body": BytesIO(value)}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        if (Bucket, Key) not in self.objects:
            raise FakeS3NotFound()
        return {}

    def delete_object(self, *, Bucket: str, Key: str) -> None:
        self.objects.pop((Bucket, Key), None)

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.objects[(bucket, key)] = Path(filename).read_bytes()

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        try:
            value = self.objects[(bucket, key)]
        except KeyError as exc:
            raise FakeS3NotFound() from exc
        Path(filename).write_bytes(value)


def test_local_raw_store_round_trip(tmp_path: Path) -> None:
    store = LocalRawStore(tmp_path / "raw")

    uri = store.put_bytes("chatgpt/conversations/abc.json", b'{"id":"abc"}')

    assert uri.startswith("file://")
    assert store.uri_for("chatgpt/conversations/abc.json") == uri
    assert store.exists(uri)
    assert store.exists("chatgpt/conversations/abc.json")
    assert store.get_bytes(uri) == b'{"id":"abc"}'

    store.delete(uri)
    assert not store.exists(uri)
    with pytest.raises(RawObjectNotFound):
        store.get_bytes(uri)


def test_local_raw_store_put_and_download_file(tmp_path: Path) -> None:
    store = LocalRawStore(tmp_path / "raw")
    source = tmp_path / "source.pdf"
    source.write_bytes(b"fake-pdf")

    uri = store.put_file("documents/ai-ssd/design.pdf", source)
    downloaded = store.download_file(uri, tmp_path / "downloads" / "design.pdf")

    assert downloaded.read_bytes() == b"fake-pdf"


def test_local_raw_store_rejects_traversal_and_external_file_uri(tmp_path: Path) -> None:
    store = LocalRawStore(tmp_path / "raw")

    with pytest.raises(InvalidRawStoreKey):
        store.put_bytes("../secret.txt", b"secret")

    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    with pytest.raises(InvalidRawStoreUri):
        store.get_bytes(outside.resolve().as_uri())


def test_s3_raw_store_round_trip_with_injected_client(tmp_path: Path) -> None:
    client = FakeS3Client()
    store = S3RawStore(bucket="kb-raw", client=client)

    uri = store.put_bytes("chatgpt/conversations/a b.json", b"conversation")

    assert uri == "s3://kb-raw/chatgpt/conversations/a%20b.json"
    assert store.exists(uri)
    assert store.get_bytes(uri) == b"conversation"

    destination = tmp_path / "copy.json"
    store.download_file(uri, destination)
    assert destination.read_bytes() == b"conversation"

    store.delete(uri)
    assert not store.exists(uri)
    with pytest.raises(RawObjectNotFound):
        store.get_bytes(uri)


def test_s3_raw_store_put_file(tmp_path: Path) -> None:
    client = FakeS3Client()
    store = S3RawStore(bucket="kb-raw", client=client)
    source = tmp_path / "notes.docx"
    source.write_bytes(b"docx")

    uri = store.put_file("documents/notes.docx", source)

    assert store.get_bytes(uri) == b"docx"


def test_s3_raw_store_rejects_uri_for_another_bucket() -> None:
    store = S3RawStore(bucket="kb-raw", client=FakeS3Client())

    with pytest.raises(InvalidRawStoreUri):
        store.get_bytes("s3://another-bucket/documents/a.pdf")


def test_create_raw_store_local_backend(tmp_path: Path) -> None:
    config = RawStoreConfig(type="local", local_root=tmp_path / "raw")

    store = create_raw_store(config)

    assert isinstance(store, LocalRawStore)
    assert store.root == (tmp_path / "raw").resolve()


def test_s3_config_requires_credentials_as_a_pair() -> None:
    with pytest.raises(ValueError):
        RawStoreConfig(
            type="s3",
            bucket="kb-raw",
            access_key_id="access-only",
        )


def test_create_raw_store_s3_backend_without_connecting() -> None:
    config = RawStoreConfig(
        type="s3",
        endpoint_url="http://127.0.0.1:8333",
        bucket="kb-raw",
        access_key_id="access",
        secret_access_key="secret",
    )

    store = create_raw_store(config)

    assert isinstance(store, S3RawStore)
    assert store.bucket == "kb-raw"
    assert store.endpoint_url == "http://127.0.0.1:8333"
