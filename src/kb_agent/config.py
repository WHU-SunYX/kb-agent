"""Application configuration for kb-agent.

Configuration is loaded in two layers:

1. YAML file (defaults to ``configs/config.yaml``)
2. Environment-variable overrides using ``KB_AGENT__`` and ``__`` nesting

Example::

    export KB_AGENT__APP__ENVIRONMENT=production
    export KB_AGENT__KB__DEFAULT_DOMAIN=ai-ssd
    export KB_AGENT__RAW_STORE__LOCAL_ROOT=/var/lib/kb-agent/raw

A different YAML file can be selected with ``KB_AGENT_CONFIG``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
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


class RawStoreConfig(BaseModel):
    """Raw-source storage configuration.

    Step 1 enables the local backend only. ``s3`` is already represented in the
    schema so Step 3 can add the S3-compatible implementation without changing
    the public configuration shape.
    """

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
            raise ValueError("raw_store.bucket is required when raw_store.type='s3'")
        return self


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
        # For kb-agent, environment variables intentionally override values
        # loaded from YAML and passed as init settings.
        return (
            env_settings,
            init_settings,
            dotenv_settings,
            file_secret_settings,
        )

    app: AppConfig = Field(default_factory=AppConfig)
    kb: KnowledgeBaseConfig = Field(default_factory=KnowledgeBaseConfig)
    raw_store: RawStoreConfig = Field(default_factory=RawStoreConfig)


def _default_config_path() -> Path:
    """Return the default YAML path for an editable project checkout."""

    project_root = Path(__file__).resolve().parents[2]
    return project_root / "configs" / "config.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML configuration file and return a mapping."""

    if not path.exists():
        raise FileNotFoundError(f"kb-agent configuration file not found: {path}")
    if not path.is_file():
        raise ValueError(f"kb-agent configuration path is not a file: {path}")

    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"kb-agent configuration root must be a mapping: {path}")
    return data


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load validated kb-agent settings.

    Precedence, from lowest to highest:

    1. Pydantic model defaults
    2. YAML configuration
    3. ``KB_AGENT__...`` environment variables

    ``KB_AGENT_CONFIG`` selects the YAML file when ``config_path`` is omitted.
    """

    selected_path = Path(
        config_path
        or os.getenv("KB_AGENT_CONFIG")
        or _default_config_path()
    ).expanduser()

    yaml_values = _read_yaml(selected_path)

    # BaseSettings reads environment variables itself. Passing YAML values as
    # init data gives environment variables higher priority than the YAML layer.
    return Settings(**yaml_values)
