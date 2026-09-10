# kb-agent

`kb-agent` is an independent knowledge-base service intended to provide retrieval capabilities to agent runtimes such as DeepSeek Harness through MCP.

The first version is intentionally small. Later steps will add ingestion, raw storage, Milvus hybrid retrieval, reranking, security/ACL, and MCP tools without coupling the project to a specific knowledge domain.

## Requirements

- Python 3.10+
- `pip`

## Editable installation

```bash
cd kb-agent
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For development tools:

```bash
python -m pip install -e '.[dev]'
```

## Verify

```bash
python - <<'PY'
from kb_agent import __version__
from kb_agent.config import load_settings

settings = load_settings()
print("kb-agent version:", __version__)
print("environment:", settings.app.environment)
print("default domain:", settings.kb.default_domain)
print("raw store type:", settings.raw_store.type)
PY
```

By default, configuration is loaded from `configs/config.yaml`. Set `KB_AGENT_CONFIG` to use another YAML file.

Environment variables can override individual values with the `KB_AGENT__` prefix and double underscores for nesting. For example:

```bash
export KB_AGENT__APP__ENVIRONMENT=production
export KB_AGENT__KB__DEFAULT_DOMAIN=ai-ssd
export KB_AGENT__RAW_STORE__TYPE=local
export KB_AGENT__RAW_STORE__LOCAL_ROOT=/var/lib/kb-agent/raw
```
