// Run after `npm run content`: validate reading destinations and URL-driven views.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { createServer } from 'vite'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const read = (file) => JSON.parse(fs.readFileSync(path.join(root, file), 'utf8'))
const graph = read('src/data/graph.json')
const paths = read('src/data/learning-paths.json')
const ids = new Set(graph.nodes.map((topic) => topic.id))
assert.equal(ids.size, graph.nodes.length, 'Topic IDs must be unique')
for (const topic of graph.nodes) {
  assert.ok(graph.groups[topic.group], `Unknown area: ${topic.group}`)
  const [route, hash] = topic.route.split('#')
  const page = read(`public/_content/${route.replace(/\/$/, '')}.json`)
  if (hash) assert.ok(page.html.includes(`id="${hash}"`), `Missing section: ${topic.route}`)
}
for (const edge of graph.edges) {
  assert.equal(edge.length, 2, 'Connections must have two endpoints')
  for (const id of edge) assert.ok(ids.has(id), `Unknown connected topic: ${id}`)
}
assert.equal(new Set(paths.map((item) => item.id)).size, paths.length, 'Path IDs must be unique')
for (const item of paths) {
  assert.equal(item.topics.length, 5, `${item.id} should have five stops`)
  assert.equal(new Set(item.topics).size, item.topics.length, 'Path stops must be unique')
  for (const id of item.topics) assert.ok(ids.has(id), `Unknown path stop: ${id}`)
}

const server = await createServer({ root, server: { middlewareMode: true }, appType: 'custom' })
try {
  const { default: TopicMap } = await server.ssrLoadModule('/src/pages/Map.tsx')
  const render = (url) => renderToStaticMarkup(createElement(MemoryRouter, { initialEntries: [url] }, createElement(TopicMap)))
  const title = (html) => html.match(/<h2 id="tm-topic-title">(.*?)<\/h2>/)?.[1]
  const count = (html) => [...html.matchAll(/class="tm-topic(?: is-selected)?"/g)].length
  const home = render('/map')
  assert.equal(count(home), graph.nodes.length, 'Every topic should be browsable')
  assert.equal(title(home), 'Self-attention')

  for (const [id] of Object.entries(graph.groups)) {
    const html = render(`/map?group=${id}`)
    assert.equal(count(html), graph.nodes.filter((topic) => topic.group === id).length)
  }
  for (const item of paths) {
    const first = render(`/map?path=${item.id}`)
    assert.equal(count(first), 5)
    assert.match(first, /Stop 1 of 5/)
    assert.match(first, /Next in this path/)
    const last = render(`/map?path=${item.id}&topic=${item.topics.at(-1)}`)
    assert.match(last, /Stop 5 of 5/)
    assert.match(last, /Last stop in this path/)
    assert.doesNotMatch(last, /Next in this path/)
  }

  const selected = render('/map?topic=kv-cache')
  assert.equal(title(selected), 'KV cache')
  assert.match(selected, /href="\/writings\/llm-inference-2-kv-cache\/#what-exactly-gets-stored"/)
  const related = selected.slice(selected.indexOf('class="tm-connections"'))
  assert.match(related, /Prefill and decode/, 'Direct connections should appear')
  assert.doesNotMatch(related, /Images as patches/, 'Unconnected topics should stay out of the detail panel')

  const search = render('/map?q=KV%20cache')
  assert.ok(count(search) > 0 && count(search) < graph.nodes.length)
  assert.match(search, /<strong>KV cache<\/strong>/)
  assert.equal(count(render('/map?q=KV-CACHE')), count(search), 'Search should ignore punctuation and case')
  const empty = render('/map?q=zzzznoresults')
  assert.equal(count(empty), 0)
  assert.match(empty, /No topics found/)
  assert.equal(title(empty), undefined)
  assert.equal(title(render('/map?group=invalid&path=invalid&topic=invalid')), 'Self-attention')
  assert.doesNotMatch(render('/map?q=%3Cscript%3Ealert%281%29%3C%2Fscript%3E'), /<script>/, 'Search must stay plain text')
  console.log(`Topic map checks passed: ${graph.nodes.length} destinations, ${graph.edges.length} connections, ${paths.length} paths, search and shared URL views.`)
} finally {
  await server.close()
}
