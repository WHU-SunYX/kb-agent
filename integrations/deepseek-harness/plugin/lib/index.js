/**
 * Host half of the kb-agent DeepSeek Harness integration.
 *
 * The browser never contacts kb-agent :8080 directly. Instead it POSTs the
 * multipart form to this same-origin Harness exact Fetch route. Connection
 * authenticates/authorizes the browser request before dispatching the route;
 * this plugin then forwards only the multipart content to kb-agent.
 */

export const name = 'kb-agent-harness'
export const inject = ['connection']

export const KB_AGENT_INGEST_PROXY_PATH = '/api/kb-agent/ingest'
export const DEFAULT_KB_AGENT_API_BASE_URL = 'http://127.0.0.1:8080'
export const DEFAULT_KB_AGENT_INGEST_PATH = '/api/v1/ingest/upload'

function trimmedString(value, fallback) {
  return typeof value === 'string' && value.trim() !== '' ? value.trim() : fallback
}

/** Normalize and validate the small Host-side proxy configuration. */
export function normalizeConfig(config = {}) {
  const apiBaseUrl = trimmedString(config.apiBaseUrl, DEFAULT_KB_AGENT_API_BASE_URL)
  const ingestPath = trimmedString(config.ingestPath, DEFAULT_KB_AGENT_INGEST_PATH)

  let base
  try {
    base = new URL(apiBaseUrl)
  } catch {
    throw new Error(`kb-agent: invalid apiBaseUrl ${JSON.stringify(apiBaseUrl)}`)
  }
  if (base.protocol !== 'http:' && base.protocol !== 'https:') {
    throw new Error('kb-agent: apiBaseUrl must use http:// or https://')
  }
  if (!ingestPath.startsWith('/')) {
    throw new Error('kb-agent: ingestPath must be an absolute URL path beginning with /')
  }

  return Object.freeze({
    apiBaseUrl: base.toString(),
    ingestPath,
  })
}

function jsonResponse(status, detail) {
  return new Response(JSON.stringify({ detail }), {
    status,
    headers: {
      'content-type': 'application/json; charset=utf-8',
      'cache-control': 'no-store',
    },
  })
}

function responseHeaders(upstream) {
  const headers = new Headers()
  const contentType = upstream.headers.get('content-type')
  if (contentType !== null) headers.set('content-type', contentType)
  headers.set('cache-control', 'no-store')
  return headers
}

/**
 * Build the protected same-origin ingestion proxy handler.
 *
 * Only the multipart body and Content-Type are forwarded. Browser cookies,
 * Harness authorization headers, Host and Origin are deliberately not leaked
 * to kb-agent.
 */
export function createIngestProxyHandler(config = {}, fetchImpl = globalThis.fetch) {
  const resolved = normalizeConfig(config)
  if (typeof fetchImpl !== 'function') throw new Error('kb-agent: global fetch is unavailable')

  return async function ingestProxy(request) {
    if (request.method !== 'POST') {
      return jsonResponse(405, 'method not allowed')
    }

    const contentType = request.headers.get('content-type') ?? ''
    if (!contentType.toLowerCase().startsWith('multipart/form-data;')) {
      return jsonResponse(415, 'expected multipart/form-data upload')
    }
    if (request.body === null) {
      return jsonResponse(400, 'missing upload body')
    }

    const target = new URL(resolved.ingestPath, resolved.apiBaseUrl)
    const headers = new Headers({ 'content-type': contentType })
    const contentLength = request.headers.get('content-length')
    if (contentLength !== null) headers.set('content-length', contentLength)

    try {
      const upstream = await fetchImpl(target, {
        method: 'POST',
        headers,
        body: request.body,
        signal: request.signal,
        // Required by Node/undici for a streamed request body. Browsers never
        // see this option because this code runs in the Harness Host process.
        duplex: 'half',
      })
      return new Response(upstream.body, {
        status: upstream.status,
        statusText: upstream.statusText,
        headers: responseHeaders(upstream),
      })
    } catch (error) {
      if (request.signal.aborted) request.signal.throwIfAborted()
      console.error('[kb-agent] ingestion proxy failed:', error)
      return jsonResponse(502, 'kb-agent ingestion service is unavailable')
    }
  }
}

/** Register the exact POST route owned by this feature plugin. */
export function apply(ctx, config = {}) {
  const fetchHandler = createIngestProxyHandler(config)
  ctx.connection.fetch.register({
    path: KB_AGENT_INGEST_PROXY_PATH,
    methods: ['POST'],
    requestBody: 'streaming',
    fetch: fetchHandler,
  })
}
