"""Raw-source storage backends for kb-agent.

The raw store preserves original evidence (ChatGPT exports, PDFs, DOCX files,
etc.) separately from retrieval indexes such as Milvus.  Callers address
objects with logical keys and persist the URI returned by ``put_*`` methods in
``SourceReference.uri``.

Two backends are provided:

* :class:`LocalRawStore` for development and tests, using ``file://`` URIs.
* :class:`S3RawStore` for S3-compatible object stores, using ``s3://`` URIs.

The S3 backend intentionally relies on boto3 rather than implementing the S3
protocol.  boto3 is an optional dependency so local development does not need
it installed.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, unquote, urlparse

if TYPE_CHECKING:
    from kb_agent.config import RawStoreConfig


class RawStoreError(RuntimeError):
    """Base exception for raw-store operations."""


class InvalidRawStoreKey(RawStoreError, ValueError):
    """Raised when an object key is unsafe or malformed."""


class InvalidRawStoreUri(RawStoreError, ValueError):
    """Raised when a URI does not belong to the selected raw store."""


class RawObjectNotFound(RawStoreError, FileNotFoundError):
    """Raised when a referenced raw object does not exist."""


def _normalize_key(key: str) -> str:
    """Validate and normalize a logical raw-store object key.

    Keys use POSIX-style separators on every backend.  Parent traversal,
    absolute paths, empty path components, backslashes, and NUL bytes are
    rejected so the same logical key is safe for both local and S3 storage.
    """

    normalized = key.strip()
    if not normalized:
        raise InvalidRawStoreKey("raw-store key must not be empty")
    if "\x00" in normalized:
        raise InvalidRawStoreKey("raw-store key must not contain NUL bytes")
    if "\\" in normalized:
        raise InvalidRawStoreKey("raw-store keys must use '/' separators")
    if normalized.startswith("/"):
        raise InvalidRawStoreKey("raw-store key must be relative")

    parts = normalized.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise InvalidRawStoreKey(
            "raw-store key must not contain empty, '.', or '..' path components"
        )
    return "/".join(parts)


class RawStore(ABC):
    """Backend-neutral interface for immutable-ish raw evidence storage.

    ``key`` arguments are logical object keys such as
    ``chatgpt/2026-09-11/conversations/abc.json``.  Read/delete methods accept
    either such a key or a URI previously returned by this store.
    """

    @abstractmethod
    def uri_for(self, key: str) -> str:
        """Return the canonical URI for ``key`` without writing it."""

    @abstractmethod
    def put_bytes(self, key: str, data: bytes) -> str:
        """Store bytes at ``key`` and return the canonical URI."""

    @abstractmethod
    def put_file(self, key: str, source_path: str | Path) -> str:
        """Store a local file at ``key`` and return the canonical URI."""

    @abstractmethod
    def get_bytes(self, ref: str) -> bytes:
        """Read an object by logical key or canonical URI."""

    @abstractmethod
    def download_file(self, ref: str, destination: str | Path) -> Path:
        """Copy an object to ``destination`` and return the destination path."""

    @abstractmethod
    def exists(self, ref: str) -> bool:
        """Return whether an object exists."""

    @abstractmethod
    def delete(self, ref: str) -> None:
        """Delete an object if it exists."""


class LocalRawStore(RawStore):
    """Filesystem-backed raw store intended for local development."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        key = _normalize_key(key)
        target = (self.root / key).resolve()
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise InvalidRawStoreKey(f"raw-store key escapes local root: {key}") from exc
        return target

    def _resolve_ref(self, ref: str) -> Path:
        parsed = urlparse(ref)
        if not parsed.scheme:
            return self._path_for_key(ref)
        if parsed.scheme != "file":
            raise InvalidRawStoreUri(
                f"LocalRawStore accepts logical keys or file:// URIs, got: {ref}"
            )
        if parsed.netloc not in {"", "localhost"}:
            raise InvalidRawStoreUri(f"unsupported file URI host: {parsed.netloc}")

        candidate = Path(unquote(parsed.path)).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise InvalidRawStoreUri(
                f"file URI is outside configured raw-store root: {ref}"
            ) from exc
        return candidate

    def uri_for(self, key: str) -> str:
        return self._path_for_key(key).as_uri()

    @staticmethod
    def _atomic_replace_from_writer(target: Path, writer: Any) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
                writer(tmp)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_path, target)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def put_bytes(self, key: str, data: bytes) -> str:
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        target = self._path_for_key(key)
        self._atomic_replace_from_writer(target, lambda stream: stream.write(data))
        return target.as_uri()

    def put_file(self, key: str, source_path: str | Path) -> str:
        source = Path(source_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"source file not found: {source}")

        target = self._path_for_key(key)

        def writer(stream: Any) -> None:
            with source.open("rb") as src:
                shutil.copyfileobj(src, stream, length=1024 * 1024)

        self._atomic_replace_from_writer(target, writer)
        return target.as_uri()

    def get_bytes(self, ref: str) -> bytes:
        path = self._resolve_ref(ref)
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise RawObjectNotFound(f"raw object not found: {ref}") from exc

    def download_file(self, ref: str, destination: str | Path) -> Path:
        source = self._resolve_ref(ref)
        if not source.is_file():
            raise RawObjectNotFound(f"raw object not found: {ref}")

        destination_path = Path(destination).expanduser().resolve()
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination_path)
        return destination_path

    def exists(self, ref: str) -> bool:
        return self._resolve_ref(ref).is_file()

    def delete(self, ref: str) -> None:
        self._resolve_ref(ref).unlink(missing_ok=True)


class S3RawStore(RawStore):
    """S3-compatible raw store backed by boto3.

    ``client`` exists primarily for dependency injection in tests.  In normal
    use it is omitted and a boto3 S3 client is created lazily from the supplied
    endpoint/region/credentials.  For custom endpoints (for example Ceph RGW
    or SeaweedFS), path-style addressing is used for broad compatibility.
    """

    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: str | None = None,
        region_name: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        client: Any | None = None,
    ):
        bucket = bucket.strip()
        if not bucket:
            raise ValueError("S3 bucket must not be empty")

        self.bucket = bucket
        self.endpoint_url = endpoint_url
        self.region_name = region_name
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "S3RawStore requires boto3. Install kb-agent with the S3 extra: "
                "pip install -e '.[s3]'"
            ) from exc

        kwargs: dict[str, Any] = {}
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
            kwargs["config"] = Config(s3={"addressing_style": "path"})
        if self.region_name:
            kwargs["region_name"] = self.region_name
        if self.access_key_id:
            kwargs["aws_access_key_id"] = self.access_key_id
        if self.secret_access_key:
            kwargs["aws_secret_access_key"] = self.secret_access_key

        self._client = boto3.client("s3", **kwargs)
        return self._client

    def _key_from_ref(self, ref: str) -> str:
        parsed = urlparse(ref)
        if not parsed.scheme:
            return _normalize_key(ref)
        if parsed.scheme != "s3":
            raise InvalidRawStoreUri(
                f"S3RawStore accepts logical keys or s3:// URIs, got: {ref}"
            )
        if parsed.netloc != self.bucket:
            raise InvalidRawStoreUri(
                f"S3 URI bucket {parsed.netloc!r} does not match configured "
                f"bucket {self.bucket!r}"
            )
        return _normalize_key(unquote(parsed.path.lstrip("/")))

    @staticmethod
    def _is_not_found(exc: Exception) -> bool:
        response = getattr(exc, "response", None)
        if not isinstance(response, dict):
            return False
        error = response.get("Error")
        if not isinstance(error, dict):
            return False
        return str(error.get("Code")) in {"404", "NoSuchKey", "NotFound"}

    def uri_for(self, key: str) -> str:
        key = _normalize_key(key)
        return f"s3://{self.bucket}/{quote(key, safe='/-_.~')}"

    def put_bytes(self, key: str, data: bytes) -> str:
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        key = _normalize_key(key)
        self._get_client().put_object(Bucket=self.bucket, Key=key, Body=data)
        return self.uri_for(key)

    def put_file(self, key: str, source_path: str | Path) -> str:
        key = _normalize_key(key)
        source = Path(source_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"source file not found: {source}")
        self._get_client().upload_file(str(source), self.bucket, key)
        return self.uri_for(key)

    def get_bytes(self, ref: str) -> bytes:
        key = self._key_from_ref(ref)
        try:
            response = self._get_client().get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            if self._is_not_found(exc):
                raise RawObjectNotFound(f"raw object not found: {ref}") from exc
            raise

        body = response["Body"]
        try:
            return body.read()
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()

    def download_file(self, ref: str, destination: str | Path) -> Path:
        key = self._key_from_ref(ref)
        destination_path = Path(destination).expanduser().resolve()
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._get_client().download_file(self.bucket, key, str(destination_path))
        except Exception as exc:
            if self._is_not_found(exc):
                raise RawObjectNotFound(f"raw object not found: {ref}") from exc
            raise
        return destination_path

    def exists(self, ref: str) -> bool:
        key = self._key_from_ref(ref)
        try:
            self._get_client().head_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            if self._is_not_found(exc):
                return False
            raise
        return True

    def delete(self, ref: str) -> None:
        key = self._key_from_ref(ref)
        self._get_client().delete_object(Bucket=self.bucket, Key=key)


def create_raw_store(config: "RawStoreConfig") -> RawStore:
    """Create a raw-store backend from :class:`RawStoreConfig`."""

    if config.type == "local":
        return LocalRawStore(config.local_root)
    if config.type == "s3":
        if not config.bucket:  # Defensive: RawStoreConfig already validates this.
            raise ValueError("raw_store.bucket is required for the S3 backend")
        return S3RawStore(
            bucket=config.bucket,
            endpoint_url=config.endpoint_url,
            region_name=config.region,
            access_key_id=config.access_key_id,
            secret_access_key=config.secret_access_key,
        )
    raise ValueError(f"unsupported raw-store backend: {config.type}")


__all__ = [
    "InvalidRawStoreKey",
    "InvalidRawStoreUri",
    "LocalRawStore",
    "RawObjectNotFound",
    "RawStore",
    "RawStoreError",
    "S3RawStore",
    "create_raw_store",
]
