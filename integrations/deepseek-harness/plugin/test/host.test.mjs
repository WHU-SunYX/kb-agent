import assert from 'node:assert/strict'
import test from 'node:test'

import {
  KB_AGENT_INGEST_PROXY_PATH,
  createIngestProxyHandler,
  normalizeConfig,
  apply,
} from '../lib/index.js'

test('normalizes defaults and rejects non-http API base', () => {
  const defaults = normalizeConfig()
  assert.equal(defaults.apiBaseUrl, 'http://127.0.0.1:8080/')
  assert.equal(defaults.ingestPath, '/api/v1/ingest/upload')
  assert.throws(() => normalizeConfig({ apiBaseUrl: 'file:///tmp/kb' }), /http:\/\/ or https:\/\//)
})

test('registers one streaming exact POST route', () => {
  let captured
  const ctx = {
    connection: {
      fetch: {
        register(route) { captured = route; return async () => {} },
      },
    },
  }
  apply(ctx, {})
  assert.equal(captured.path, KB_AGENT_INGEST_PROXY_PATH)
  assert.deepEqual(captured.methods, ['POST'])
  assert.equal(captured.requestBody, 'streaming')
  assert.equal(typeof captured.fetch, 'function')
})

test('rejects a non-multipart body before upstream fetch', async () => {
  let called = false
  const handler = createIngestProxyHandler({}, async () => { called = true; return new Response() })
  const request = new Request('http://dsh.internal/api/kb-agent/ingest', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: '{}',
  })
  const response = await handler(request)
  assert.equal(response.status, 415)
  assert.equal(called, false)
})

test('streams multipart upload and strips Harness credentials', async () => {
  const boundary = '----kb-agent-test-boundary'
  const body = `--${boundary}\r\nContent-Disposition: form-data; name="project"\r\n\r\ntest\r\n--${boundary}--\r\n`
  let seen
  const handler = createIngestProxyHandler(
    { apiBaseUrl: 'http://127.0.0.1:8080', ingestPath: '/api/v1/ingest/upload' },
    async (target, init) => {
      seen = { target: String(target), init }
      // Drain the streamed request body to prove it is consumable.
      const text = await new Response(init.body).text()
      assert.match(text, /name="project"/)
      return new Response(JSON.stringify({ indexed_chunk_count: 3 }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    },
  )
  const request = new Request('http://dsh.internal/api/kb-agent/ingest', {
    method: 'POST',
    headers: {
      'content-type': `multipart/form-data; boundary=${boundary}`,
      authorization: 'Bearer must-not-leak',
      cookie: 'secret=must-not-leak',
    },
    body,
    duplex: 'half',
  })
  const response = await handler(request)
  assert.equal(response.status, 200)
  assert.equal(seen.target, 'http://127.0.0.1:8080/api/v1/ingest/upload')
  assert.equal(seen.init.headers.get('authorization'), null)
  assert.equal(seen.init.headers.get('cookie'), null)
  assert.match(seen.init.headers.get('content-type'), /^multipart\/form-data;/)
  assert.equal((await response.json()).indexed_chunk_count, 3)
})

test('maps an upstream connection failure to 502', async () => {
  const boundary = '----kb-agent-test-boundary'
  const handler = createIngestProxyHandler({}, async () => { throw new Error('offline') })
  const request = new Request('http://dsh.internal/api/kb-agent/ingest', {
    method: 'POST',
    headers: { 'content-type': `multipart/form-data; boundary=${boundary}` },
    body: `--${boundary}--\r\n`,
    duplex: 'half',
  })
  const original = console.error
  console.error = () => {}
  try {
    const response = await handler(request)
    assert.equal(response.status, 502)
    assert.deepEqual(await response.json(), { detail: 'kb-agent ingestion service is unavailable' })
  } finally {
    console.error = original
  }
})
