# DeepSeek Harness MCP integration

## Architecture

```text
                          kb-agent

DeepSeek Harness              FastAPI clients / UI
       |                              |
       | MCP stdio                    | HTTP multipart / JSON
       v                              v
kb-agent MCP server              kb-agent FastAPI
       |                              |
       +-- kb_search                   +-- POST /api/v1/ingest/upload
       +-- kb_get_document             +-- POST /api/v1/search
       |                              |
       +--------------+---------------+
                      |
               retrieval / ingestion
                      |
                  Remote Milvus
```

The MCP surface is intentionally read-only. Knowledge ingestion and other
administrative/write operations go through FastAPI rather than MCP. In
particular, `kb_ingest` is not an MCP tool.

## Start kb-agent MCP server

```bash
pip install -e ".[mcp]"
python -m kb_agent.mcp.server
```

The default transport is stdio, so an MCP client normally starts the process
itself instead of running it as a standalone daemon.

## Verify the MCP tool surface

Using MCP Inspector CLI:

```bash
PYTHONPATH=/home/sunyaxin/pycharm-project/kb-agent/src \
mcp-inspector --cli \
  /home/sunyaxin/miniforge3/envs/XInfer/bin/python \
  -m kb_agent.mcp.server \
  -- \
  --method tools/list
```

The expected tools are:

```text
kb_search
kb_get_document
```

`kb_ingest` must not appear. Upload/import uses FastAPI
`POST /api/v1/ingest/upload`.

## DeepSeek Harness client configuration

DeepSeek Harness integration is handled in Step 14.5. The Harness-side
configuration should launch this stdio server and expose the two read-only MCP
tools to the agent.

## LLM-facing search result size and citations

`kb_search` forwards a query to the shared FastAPI search endpoint, then returns a
**compact, source-attributed** MCP result rather than the full HTTP record. The
HTTP endpoint, stored vectors, retrieval ranks and full metadata are unchanged.

The response is an object containing `results`, `total_hits`, `returned_hits`,
`omitted_hits` and `truncated`. Each hit contains `chunk_id`, `score`, `text`
and a small `source` object populated only with **actual** source metadata
(e.g. `title`, `document_id`, `source_type`, `provider`, `source_id`,
`chunk_index`, `section_path`, `page_number`, `turn_start`, `turn_end`). A
missing title is omitted; the model must not treat a chunk ID as a title.
Repeated `metadata.text`, security fields and internal raw-store paths are not
copied into the MCP response.

Defaults in `configs/config.yaml`:

```yaml
mcp:
  search_max_results: 5
  search_max_chars: 10000
```

Change them through YAML or `KB_AGENT__MCP__SEARCH_MAX_RESULTS` and
`KB_AGENT__MCP__SEARCH_MAX_CHARS`. Limits apply **only** to the result presented
to an LLM through MCP, not the FastAPI endpoint. Results keep their original
ranking. Whole passages are preferred: if the next passage does not fit, it is
omitted and reported in `omitted_hits`; only a single first passage that
exceeds the entire limit is shortened, with `text_truncated=true`.
`search_max_chars` uses serialized Unicode character count as a deterministic
size proxy, **not** model-token count. Harness may still enforce its own context
compaction. If the tool result is pruned, lower these values or increase the
*actual* vLLM model context capacity before changing Harness's context setting.
For high-recall tasks, raise the limits or issue a narrower follow-up query.

The MCP tool is read-only and is normally spawned automatically by Harness.
A standalone `python -m kb_agent.mcp.server` is not the Harness process's
Tool Call log. Examine the Harness session events for `tool/call`, `tool/result`
and context compaction; use MCP Inspector to test the server independently.

## Structured search filters

`kb_search` intentionally does **not** expose a raw Milvus `filter_expr` to the
LLM.  The tool accepts only schema-backed exact-match fields such as `domain`,
`project`, `source_type`, `provider`, `classification`, `source_id`, and
`document_id`.  The MCP HTTP adapter sends those fields as a structured
`filters` object and the FastAPI/retrieval stack compiles them to the Milvus
boolean expression internally.

This boundary is deliberate:

- an LLM cannot invent backend syntax such as `type:document` or
  `workspace-write` and pass it directly to Milvus;
- field names come from the application schema rather than model-generated
  text;
- string values are quoted by the backend compiler;
- `tenant_id` is an access-scope field and is injected from trusted server-side
  configuration instead of being exposed as an MCP tool argument;
- stale/unknown HTTP fields are rejected with FastAPI/Pydantic validation
  instead of surfacing as a Milvus query-plan HTTP 500.

Example MCP intent:

```text
query="STAR2000 DMA optimization"
source_type="document"
project="engram"
```

Equivalent FastAPI payload:

```json
{
  "query": "STAR2000 DMA optimization",
  "filters": {
    "source_type": "document",
    "project": "engram"
  },
  "top_k": 10
}
```

The concrete Milvus expression is an implementation detail and is never part
of the public tool contract.
