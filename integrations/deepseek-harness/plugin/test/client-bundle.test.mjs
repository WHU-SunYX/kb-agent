import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import vm from 'node:vm'
import test from 'node:test'

const bundleUrl = new URL('../lib/client.js', import.meta.url)


test('client bundle registers through the Harness lazy module wrapper', async () => {
  const source = await readFile(bundleUrl, 'utf8')
  let contribution
  vm.runInNewContext(source, {
    window: { __ModuleLoader__: { load(value) { contribution = value } } },
  }, { filename: 'lib/client.js' })

  assert.equal(contribution.id, '@kb-agent/dsh-kb-agent')
  assert.equal(typeof contribution.factory, 'function')

  const fakeReact = {
    createElement() { return null },
    useSyncExternalStore() { return { open: false, generation: 0 } },
    useState(initial) { return [initial, () => {}] },
    useRef(initial) { return { current: initial } },
    useEffect() {},
  }
  const client = contribution.factory((specifier) => {
    if (specifier === 'react') return fakeReact
    throw new Error(`unexpected require: ${specifier}`)
  })

  assert.deepEqual(Array.from(client.inject), ['slots', 'locale'])
  assert.equal(typeof client.apply, 'function')

  const registrations = []
  const dictionaries = []
  const ctx = {
    effect(setup) { return setup() },
    locale: {
      register(ns, dicts) { dictionaries.push({ ns, dicts }); return () => {} },
    },
    slots: {
      inject(name, setup) { assert.ok(name === 'sidebar.footer.action' || name === 'shell.overlay'); return setup() },
      register(options, component) { registrations.push({ options, component }); return () => {} },
    },
  }
  client.apply(ctx)

  assert.equal(dictionaries.length, 1)
  assert.equal(dictionaries[0].ns, 'kb-agent')
  assert.deepEqual(registrations.map(row => row.options.name), [
    'sidebar.footer.action',
    'shell.overlay',
  ])
  assert.equal(registrations[0].options.id, 'kb-agent-import')
  assert.equal(registrations[1].options.id, 'kb-agent-import-dialog')
})
