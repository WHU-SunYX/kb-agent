"""Application configuration for kb-agent.

Configuration is loaded in two layers:

1. YAML file (defaults to ``configs/config.yaml``)
2. Environment-variable overrides using ``KB_AGENT__`` and ``__`` nesting
"""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseModel):
    """General application settings."""

    model_config = ConfigDict(extra="forbid")

    name: str = "kb-agent"
    environment: Literal["development", "test", "production"] = "development"


class KnowledgeBaseConfig(BaseModel):
    """Domain-neutral knowledge-base defaults."""

    model_config = ConfigDict(extra="forbid")

    default_tenant: str = "default"
    default_domain: str = "ai-ssd"


class EmbeddingConfig(BaseModel):
    """Dense embedding backend configuration."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["local_bge"] = "local_bge"
    model_name_or_path: str = "BAAI/bge-m3"
    devices: str | list[str] | None = None
    use_fp16: bool = False
    use_bf16: bool = False
    normalize_embeddings: bool = True
    batch_size: int = Field(default=32, gt=0)
    query_max_length: int = Field(default=512, gt=0)
    passage_max_length: int = Field(default=512, gt=0)
    expected_dimension: int | None = Field(default=1024, gt=0)
    pooling_method: str = "cls"
    cache_dir: str | None = None
    trust_remote_code: bool = False

    @model_validator(mode="after")
    def validate_precision(self) -> "EmbeddingConfig":
        if self.use_fp16 and self.use_bf16:
            raise ValueError(
                "embedding.use_fp16 and embedding.use_bf16 cannot both be enabled"
            )
        if not self.model_name_or_path.strip():
            raise ValueError(
                "embedding.model_name_or_path must not be blank"
            )
        if not self.pooling_method.strip():
            raise ValueError(
                "embedding.pooling_method must not be blank"
            )
        if isinstance(self.devices, list) and not self.devices:
            raise ValueError(
                "embedding.devices must not be an empty list"
            )
        return self


class SecurityConfig(BaseModel):
    """PII detector runtime placement; secret/context policies are unchanged.

    Presidio's device selector is process-wide.  The default is CPU so a
    security scan cannot unexpectedly allocate memory on CUDA device zero.
    """

    model_config = ConfigDict(extra="forbid")

    pii_device: str = "cpu"

    @field_validator("pii_device")
    @classmethod
    def validate_pii_device(cls, value: str) -> str:
        device = value.strip().lower()
        if not re.fullmatch(r"cpu|cuda(?::(?:0|[1-9][0-9]*))?", device):
            raise ValueError("security.pii_device must be 'cpu', 'cuda' or 'cuda:N'")
        return device


class RawStoreConfig(BaseModel):
    """Raw-source storage configuration."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["local", "s3"] = "local"
    local_root: Path = Path("./data/dev/raw")

    endpoint_url: str | None = None
    bucket: str | None = None
    region: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None

    @model_validator(mode="after")
    def validate_backend(self) -> "RawStoreConfig":
        if self.type == "s3" and not self.bucket:
            raise ValueError(
                "raw_store.bucket is required when raw_store.type='s3'"
            )
        if bool(self.access_key_id) != bool(self.secret_access_key):
            raise ValueError(
                "raw_store.access_key_id and raw_store.secret_access_key "
                "must be set together"
            )
        return self


class MilvusConfig(BaseModel):
    """Milvus vector database configuration."""

    model_config = ConfigDict(extra="forbid")

    uri: str = "http://localhost:19530"
    collection_name: str = "kb_chunks"
    dimension: int = Field(default=1024, gt=0)
    metric_type: str = "COSINE"
    consistency_level: str = "Bounded"
    enable_bm25: bool = True


class MCPConfig(BaseModel):
    """MCP server configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    name: str = "kb-agent"
    transport: Literal["stdio"] = "stdio"
    # MCP-only presentation limits; the FastAPI search response remains intact.
    search_max_results: int = Field(default=5, ge=1, le=100)
    search_max_chars: int = Field(default=10000, ge=2048, le=500000)


class Settings(BaseSettings):
    """Top-level kb-agent settings."""

    model_config = SettingsConfigDict(
        env_prefix="KB_AGENT__",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="forbid",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            env_settings,
            init_settings,
            dotenv_settings,
            file_secret_settings,
        )

    app: AppConfig = Field(default_factory=AppConfig)
    kb: KnowledgeBaseConfig = Field(default_factory=KnowledgeBaseConfig)
    raw_store: RawStoreConfig = Field(default_factory=RawStoreConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)


def _default_config_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "configs" / "config.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"kb-agent configuration file not found: {path}"
        )
    if not path.is_file():
        raise ValueError(
            f"kb-agent configuration path is not a file: {path}"
        )

    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"kb-agent configuration root must be a mapping: {path}"
        )

    return data


def load_settings(config_path: str | Path | None = None) -> Settings:
    selected_path = Path(
        config_path
        or os.getenv("KB_AGENT_CONFIG")
        or _default_config_path()
    ).expanduser()

    yaml_values = _read_yaml(selected_path)

    return Settings(**yaml_values)
