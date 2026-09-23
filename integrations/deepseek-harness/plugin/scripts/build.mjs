import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const root = resolve(here, '..')
await mkdir(resolve(root, 'lib'), { recursive: true })
await copyFile(resolve(root, 'src/index.js'), resolve(root, 'lib/index.js'))

const source = await readFile(resolve(root, 'src/client.cjs'), 'utf8')
const wrapped = [
  "window.__ModuleLoader__.load({",
  "  id: '@kb-agent/dsh-kb-agent',",
  "  factory: (require) => {",
  "    var module = { exports: {} };",
  "    var exports = module.exports;",
  source.split('\n').map(line => `    ${line}`).join('\n'),
  "    return module.exports;",
  "  },",
  "});",
  "",
].join('\n')
await writeFile(resolve(root, 'lib/client.js'), wrapped, 'utf8')
