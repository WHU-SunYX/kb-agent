'use strict';

const React = require('react');

const NS = 'kb-agent';
const PROXY_PATH = '/api/kb-agent/ingest';

const zh = {
  'action.import': '导入数据',
  'dialog.title': '导入知识库数据',
  'dialog.description': '上传文件后，DeepSeek Harness 会通过同源 Host 路由转发给 kb-agent。',
  'field.file': '文件',
  'field.fileHint': '支持 PDF、DOCX、Markdown，以及 ChatGPT 导出的 ZIP/JSON。',
  'field.domain': 'Domain',
  'field.project': 'Project',
  'field.sourceKind': '数据类型',
  'source.auto': '自动识别',
  'source.document': '文档',
  'source.chatgpt': 'ChatGPT 导出',
  'placeholder.domain': '例如 ai-ssd',
  'placeholder.project': '例如 sparse-kv',
  'button.cancel': '取消',
  'button.close': '关闭',
  'button.import': '导入',
  'button.importing': '导入中…',
  'error.fileRequired': '请选择要导入的文件。',
  'error.requestFailed': '导入失败',
  'result.success': '导入完成',
  'result.filename': '文件',
  'result.documents': '文档',
  'result.accepted': '已接受',
  'result.blocked': '被阻止',
  'result.chunks': 'Chunks',
  'result.indexed': '已索引',
  'result.rawReused': '原始对象复用',
  'value.yes': '是',
  'value.no': '否',
};

const en = {
  'action.import': 'Import data',
  'dialog.title': 'Import knowledge data',
  'dialog.description': 'DeepSeek Harness forwards the uploaded file to kb-agent through a same-origin Host route.',
  'field.file': 'File',
  'field.fileHint': 'PDF, DOCX, Markdown, and ChatGPT export ZIP/JSON are supported.',
  'field.domain': 'Domain',
  'field.project': 'Project',
  'field.sourceKind': 'Source type',
  'source.auto': 'Auto detect',
  'source.document': 'Document',
  'source.chatgpt': 'ChatGPT export',
  'placeholder.domain': 'e.g. ai-ssd',
  'placeholder.project': 'e.g. sparse-kv',
  'button.cancel': 'Cancel',
  'button.close': 'Close',
  'button.import': 'Import',
  'button.importing': 'Importing…',
  'error.fileRequired': 'Choose a file to import.',
  'error.requestFailed': 'Import failed',
  'result.success': 'Import complete',
  'result.filename': 'File',
  'result.documents': 'Documents',
  'result.accepted': 'Accepted',
  'result.blocked': 'Blocked',
  'result.chunks': 'Chunks',
  'result.indexed': 'Indexed',
  'result.rawReused': 'Raw object reused',
  'value.yes': 'Yes',
  'value.no': 'No',
};

let dialogSnapshot = Object.freeze({ open: false, generation: 0 });
const dialogListeners = new Set();

function publishDialog(next) {
  dialogSnapshot = Object.freeze({ ...dialogSnapshot, ...next });
  for (const listener of [...dialogListeners]) listener();
}

function subscribeDialog(listener) {
  dialogListeners.add(listener);
  return () => { dialogListeners.delete(listener); };
}

function getDialogSnapshot() {
  return dialogSnapshot;
}

function openDialog() {
  publishDialog({ open: true, generation: dialogSnapshot.generation + 1 });
}

function closeDialog() {
  publishDialog({ open: false });
}

function useDialogSnapshot() {
  return React.useSyncExternalStore(subscribeDialog, getDialogSnapshot, getDialogSnapshot);
}

function UploadGlyph() {
  return React.createElement(
    'svg',
    {
      width: 18,
      height: 18,
      viewBox: '0 0 24 24',
      fill: 'none',
      stroke: 'currentColor',
      strokeWidth: 1.8,
      strokeLinecap: 'round',
      strokeLinejoin: 'round',
      'aria-hidden': true,
    },
    React.createElement('path', { d: 'M12 16V4' }),
    React.createElement('path', { d: 'm7 9 5-5 5 5' }),
    React.createElement('path', { d: 'M5 20h14' }),
  );
}

const actionStyleBase = {
  boxSizing: 'border-box',
  minHeight: 38,
  border: 0,
  borderRadius: 8,
  background: 'transparent',
  color: 'var(--dsw-alias-label-primary)',
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: 8,
  font: 'inherit',
};

function ImportAction({ wide, t }) {
  useDialogSnapshot();
  const label = t('action.import');
  return React.createElement(
    'button',
    {
      type: 'button',
      onClick: openDialog,
      'aria-label': label,
      title: wide ? undefined : label,
      style: {
        ...actionStyleBase,
        width: wide ? '100%' : 40,
        padding: wide ? '0 10px' : 0,
        justifyContent: wide ? 'flex-start' : 'center',
      },
    },
    React.createElement(UploadGlyph),
    wide ? React.createElement('span', null, label) : null,
  );
}

const fieldStyle = {
  display: 'grid',
  gap: 6,
};

const controlStyle = {
  boxSizing: 'border-box',
  width: '100%',
  minHeight: 36,
  padding: '7px 10px',
  borderRadius: 8,
  border: '1px solid color-mix(in srgb, var(--dsw-alias-label-primary) 18%, transparent)',
  background: 'var(--dsw-alias-bg-layer-1)',
  color: 'var(--dsw-alias-label-primary)',
  font: 'inherit',
};

const secondaryButtonStyle = {
  minHeight: 36,
  padding: '0 14px',
  borderRadius: 8,
  border: '1px solid color-mix(in srgb, var(--dsw-alias-label-primary) 18%, transparent)',
  background: 'var(--dsw-alias-bg-layer-1)',
  color: 'var(--dsw-alias-label-primary)',
  cursor: 'pointer',
  font: 'inherit',
};

const primaryButtonStyle = {
  ...secondaryButtonStyle,
  border: '1px solid color-mix(in srgb, var(--dsw-alias-label-primary) 35%, transparent)',
  background: 'color-mix(in srgb, var(--dsw-alias-label-primary) 12%, var(--dsw-alias-bg-layer-1))',
  fontWeight: 600,
};

function messageOfPayload(payload, response, t) {
  if (payload && typeof payload === 'object' && typeof payload.detail === 'string') return payload.detail;
  if (typeof payload === 'string' && payload.trim() !== '') return payload.trim();
  return `${t('error.requestFailed')}: HTTP ${response.status}`;
}

async function readPayload(response) {
  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    try { return await response.json(); } catch { return null; }
  }
  try { return await response.text(); } catch { return null; }
}

function ResultGrid({ result, t }) {
  if (!result) return null;
  const rows = [
    [t('result.filename'), result.source_filename || '—'],
    [t('result.documents'), result.document_count ?? 0],
    [t('result.accepted'), result.accepted_document_count ?? 0],
    [t('result.blocked'), result.blocked_document_count ?? 0],
    [t('result.chunks'), result.chunk_count ?? 0],
    [t('result.indexed'), result.indexed_chunk_count ?? 0],
    [t('result.rawReused'), result.raw_object_reused ? t('value.yes') : t('value.no')],
  ];
  return React.createElement(
    'div',
    {
      style: {
        display: 'grid',
        gridTemplateColumns: 'minmax(120px, auto) 1fr',
        gap: '6px 12px',
        padding: 12,
        borderRadius: 8,
        background: 'color-mix(in srgb, var(--dsw-alias-label-primary) 5%, transparent)',
      },
    },
    ...rows.flatMap(([label, value], index) => [
      React.createElement('div', { key: `l-${index}`, style: { opacity: 0.7 } }, label),
      React.createElement('div', { key: `v-${index}`, style: { overflowWrap: 'anywhere' } }, String(value)),
    ]),
  );
}

function Field({ label, children, hint }) {
  return React.createElement(
    'label',
    { style: fieldStyle },
    React.createElement('span', { style: { fontWeight: 600 } }, label),
    children,
    hint ? React.createElement('span', { style: { fontSize: 12, opacity: 0.7 } }, hint) : null,
  );
}

function ImportOverlay({ t }) {
  const dialog = useDialogSnapshot();
  const [file, setFile] = React.useState(null);
  const [domain, setDomain] = React.useState('');
  const [project, setProject] = React.useState('');
  const [sourceKind, setSourceKind] = React.useState('auto');
  const [status, setStatus] = React.useState('idle');
  const [error, setError] = React.useState(null);
  const [result, setResult] = React.useState(null);
  const abortRef = React.useRef(null);

  React.useEffect(() => {
    if (!dialog.open) return;
    setFile(null);
    setDomain('');
    setProject('');
    setSourceKind('auto');
    setStatus('idle');
    setError(null);
    setResult(null);
    abortRef.current = null;
  }, [dialog.generation]);

  React.useEffect(() => () => { abortRef.current?.abort(); }, []);

  React.useEffect(() => {
    if (!dialog.open) return undefined;
    const onKeyDown = (event) => {
      if (event.key !== 'Escape') return;
      if (status === 'uploading') abortRef.current?.abort();
      closeDialog();
    };
    globalThis.addEventListener?.('keydown', onKeyDown);
    return () => { globalThis.removeEventListener?.('keydown', onKeyDown); };
  }, [dialog.open, status]);

  if (!dialog.open) return null;

  const busy = status === 'uploading';

  const dismiss = () => {
    if (busy) abortRef.current?.abort();
    closeDialog();
  };

  const submit = async (event) => {
    event.preventDefault();
    if (!file) {
      setError(t('error.fileRequired'));
      setStatus('error');
      return;
    }

    const abort = new AbortController();
    abortRef.current = abort;
    setStatus('uploading');
    setError(null);
    setResult(null);

    const body = new FormData();
    body.append('file', file, file.name);
    const trimmedDomain = domain.trim();
    const trimmedProject = project.trim();
    if (trimmedDomain) body.append('domain', trimmedDomain);
    if (trimmedProject) body.append('project', trimmedProject);
    body.append('source_kind', sourceKind);

    try {
      const response = await fetch(PROXY_PATH, {
        method: 'POST',
        body,
        signal: abort.signal,
      });
      const payload = await readPayload(response);
      if (!response.ok) throw new Error(messageOfPayload(payload, response, t));
      setResult(payload && typeof payload === 'object' ? payload : null);
      setStatus('success');
    } catch (caught) {
      if (abort.signal.aborted) {
        setStatus('idle');
        return;
      }
      setError(caught instanceof Error ? caught.message : String(caught));
      setStatus('error');
    } finally {
      if (abortRef.current === abort) abortRef.current = null;
    }
  };

  return React.createElement(
    'div',
    {
      role: 'presentation',
      onMouseDown: (event) => {
        if (event.target === event.currentTarget && !busy) dismiss();
      },
      style: {
        position: 'fixed',
        inset: 0,
        zIndex: 1000,
        display: 'grid',
        placeItems: 'center',
        padding: 20,
        background: 'rgba(0, 0, 0, 0.46)',
        pointerEvents: 'auto',
      },
    },
    React.createElement(
      'div',
      {
        role: 'dialog',
        'aria-modal': true,
        'aria-labelledby': 'kb-agent-import-title',
        style: {
          boxSizing: 'border-box',
          width: 'min(560px, calc(100vw - 32px))',
          maxHeight: 'min(760px, calc(100vh - 32px))',
          overflow: 'auto',
          borderRadius: 12,
          border: '1px solid color-mix(in srgb, var(--dsw-alias-label-primary) 14%, transparent)',
          background: 'var(--dsw-alias-bg-layer-1)',
          color: 'var(--dsw-alias-label-primary)',
          boxShadow: '0 18px 60px rgba(0, 0, 0, 0.28)',
          padding: 20,
        },
      },
      React.createElement(
        'div',
        { style: { display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 16 } },
        React.createElement(
          'div',
          null,
          React.createElement('h2', { id: 'kb-agent-import-title', style: { margin: 0, fontSize: 18 } }, t('dialog.title')),
          React.createElement('p', { style: { margin: '8px 0 0', opacity: 0.72, lineHeight: 1.45 } }, t('dialog.description')),
        ),
        React.createElement(
          'button',
          {
            type: 'button',
            onClick: dismiss,
            'aria-label': t('button.close'),
            disabled: false,
            style: { ...secondaryButtonStyle, minWidth: 36, width: 36, padding: 0, flex: '0 0 auto' },
          },
          '×',
        ),
      ),
      React.createElement(
        'form',
        { onSubmit: submit, style: { display: 'grid', gap: 14, marginTop: 18 } },
        React.createElement(
          Field,
          { label: t('field.file'), hint: t('field.fileHint') },
          React.createElement('input', {
            type: 'file',
            accept: '.pdf,.docx,.md,.markdown,.zip,.json,application/pdf,application/json,application/zip',
            disabled: busy,
            onChange: (event) => { setFile(event.target.files?.[0] || null); setError(null); },
            style: controlStyle,
          }),
        ),
        React.createElement(
          'div',
          { style: { display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 12 } },
          React.createElement(
            Field,
            { label: t('field.domain') },
            React.createElement('input', {
              type: 'text',
              value: domain,
              disabled: busy,
              placeholder: t('placeholder.domain'),
              onChange: (event) => setDomain(event.target.value),
              style: controlStyle,
            }),
          ),
          React.createElement(
            Field,
            { label: t('field.project') },
            React.createElement('input', {
              type: 'text',
              value: project,
              disabled: busy,
              placeholder: t('placeholder.project'),
              onChange: (event) => setProject(event.target.value),
              style: controlStyle,
            }),
          ),
        ),
        React.createElement(
          Field,
          { label: t('field.sourceKind') },
          React.createElement(
            'select',
            {
              value: sourceKind,
              disabled: busy,
              onChange: (event) => setSourceKind(event.target.value),
              style: controlStyle,
            },
            React.createElement('option', { value: 'auto' }, t('source.auto')),
            React.createElement('option', { value: 'document' }, t('source.document')),
            React.createElement('option', { value: 'chatgpt' }, t('source.chatgpt')),
          ),
        ),
        error ? React.createElement(
          'div',
          { role: 'alert', style: { padding: '10px 12px', borderRadius: 8, background: 'color-mix(in srgb, #d33 12%, transparent)', overflowWrap: 'anywhere' } },
          error,
        ) : null,
        status === 'success' ? React.createElement(
          'div',
          { style: { display: 'grid', gap: 10 } },
          React.createElement('strong', null, t('result.success')),
          React.createElement(ResultGrid, { result, t }),
        ) : null,
        React.createElement(
          'div',
          { style: { display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 4 } },
          React.createElement(
            'button',
            { type: 'button', onClick: dismiss, style: secondaryButtonStyle },
            status === 'success' ? t('button.close') : t('button.cancel'),
          ),
          status === 'success' ? null : React.createElement(
            'button',
            { type: 'submit', disabled: busy, style: { ...primaryButtonStyle, opacity: busy ? 0.65 : 1 } },
            busy ? t('button.importing') : t('button.import'),
          ),
        ),
      ),
    ),
  );
}

exports.inject = ['slots', 'locale'];

exports.apply = function apply(ctx) {
  ctx.effect(() => ctx.locale.register(NS, { zh, en }), 'kb-agent: browser dictionaries');

  ctx.slots.inject('sidebar.footer.action', () => ctx.slots.register({
    name: 'sidebar.footer.action',
    id: 'kb-agent-import',
    order: 80,
    locale: NS,
  }, ImportAction));

  ctx.slots.inject('shell.overlay', () => ctx.slots.register({
    name: 'shell.overlay',
    id: 'kb-agent-import-dialog',
    order: 80,
    locale: NS,
  }, ImportOverlay));
};
