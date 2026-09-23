# kb-agent × DeepSeek Harness (Steps 14.5 + 14.6)

This directory is an **external DeepSeek Harness bundle/plugin**. It adds:

1. **Step 14.5 — read-only MCP** (`kb_search`, `kb_get_document`) by starting
   `python -m kb_agent.mcp.server` through `@deepseek-ai/dsh-mcp-client`.
2. **Step 14.6 — human/admin ingestion UI** in the Web sidebar. The browser
   uploads to the Harness same-origin route `/api/kb-agent/ingest`; the Harness
   Host streams that multipart request to kb-agent
   `/api/v1/ingest/upload` on `127.0.0.1:8080` by default.

No DeepSeek Harness source file is modified.

## Why two channels

- **MCP is read-only** and belongs to the agent/model plane.
- **Ingestion is a write/admin operation** and stays on FastAPI.
- The browser does not call `http://127.0.0.1:8080` directly, avoiding CORS and
  the ambiguity of what `localhost` means in a remote browser.

## Files

- `cordis.patch.yml` — bundle rows for MCP + UI Host plugin.
- `src/index.js` — Host exact Fetch route and streaming proxy.
- `src/client.cjs` — browser slot UI source.
- `lib/index.js`, `lib/client.js` — ready-to-run artifacts consumed by Harness.
- `scripts/build.mjs` — deterministic local rebuild; no npm dependencies.
- `test/` — Host proxy and client-bundle smoke tests.

## Install

Copy this whole `plugin/` directory into:

```text
/home/sunyaxin/pycharm-project/kb-agent/integrations/deepseek-harness/plugin
```

Make sure kb-agent FastAPI is already listening on `127.0.0.1:8080`, then:

```bash
cd /home/sunyaxin/pycharm-project/kb-agent/integrations/deepseek-harness/plugin
npm test

dsh plugin --profile web add .
```

Adding/removing a **bundle** changes profile bundle membership, so restart the
currently running `web` profile once after installation. Normal later edits to
profile/home `cordis.patch.yml` can follow the profile's configured reload
policy.

## Optional environment overrides

Defaults match the current kb-agent machine:

```bash
export KB_AGENT_PYTHON=/home/sunyaxin/miniforge3/envs/XInfer/bin/python
export KB_AGENT_PROJECT_ROOT=/home/sunyaxin/pycharm-project/kb-agent
export KB_AGENT_PYTHONPATH=/home/sunyaxin/pycharm-project/kb-agent/src
export KB_AGENT_API_BASE_URL=http://127.0.0.1:8080
```

Set them **before starting DeepSeek Harness** when a path/port differs.

## Verify composition

```bash
dsh --profile web --dump-config | grep -n -A16 -B3 'kb-agent'
```

You should see both rows:

- `kb-agent-mcp`
- `kb-agent-ui`

Then open the Web UI. A new **Import data / 导入数据** action should appear in
`sidebar.footer.action` above Settings. It opens a global `shell.overlay`
dialog. Importing a file should return counts including `chunk_count` and
`indexed_chunk_count`.

## Verify MCP remains read-only

Use your existing MCP Inspector flow, or ask Harness to list its MCP tools. The
kb-agent MCP server should expose only:

- `kb_search`
- `kb_get_document`

It should **not** expose `kb_ingest`.

## Uninstall

```bash
dsh plugin --profile web remove @kb-agent/dsh-kb-agent
```

Restart the running Web profile after bundle removal.

## Notes

- The proxy forwards only multipart `Content-Type`, optional `Content-Length`,
  the body stream, and the request abort signal. It does **not** forward Harness
  cookies/authorization/Host/Origin headers to kb-agent.
- The upstream response status/body are preserved; network failures become a
  small HTTP 502 JSON error.
- The UI supports `source_kind=auto|document|chatgpt`, matching the current
  kb-agent ingestion pipeline.
