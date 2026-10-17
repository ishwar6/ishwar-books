// Validate rendered navigation and initial states without downloading or running models.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { createElement as h } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { createServer } from 'vite'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const server = await createServer({ root, server: { middlewareMode: true }, appType: 'custom' })
const render = (Component, url, props = {}) => renderToStaticMarkup(h(MemoryRouter, { initialEntries: [url] }, h(Component, props)))
const selectedTab = (html) => html.match(/id="([^"]+)" role="tab" aria-selected="true"/)?.[1]
const visiblePanels = (html) => [...html.matchAll(/<div class="demo-tab-panel"[^>]*>/g)].filter(([tag]) => !tag.includes('hidden=')).length

try {
  const { default: Tabs } = await server.ssrLoadModule('/src/components/DemoTabs.tsx')
  const { default: Gallery } = await server.ssrLoadModule('/src/pages/Demos.tsx')
  const { default: Sam } = await server.ssrLoadModule('/src/demos/sam/Demo.tsx')
  const { default: Vit } = await server.ssrLoadModule('/src/demos/vit/Demo.tsx')
  const tabs = [
    { id: 'first', label: 'First', content: h('p', null, 'Retained first view') },
    { id: 'second', label: 'Second', content: h('p', null, 'Retained second view') },
    { id: 'last', label: 'Last', content: h('p', null, 'Retained final view') },
  ]
  for (const [url, expected] of [
    ['/demo', 'first'], ['/demo?view=second', 'second'], ['/demo?view=last', 'last'],
    ['/demo#second', 'second'], ['/demo?view=missing', 'first'],
    ['/demo?view=last#second', 'last'],
  ]) {
    const html = render(Tabs, url, { id: 'test', label: 'Test journey', tabs })
    assert.equal(selectedTab(html), `test-tab-${expected}`)
    assert.equal(visiblePanels(html), 1, 'Only the selected panel should be visible')
    assert.equal((html.match(/role="tabpanel"/g) ?? []).length, 3, 'All panels stay mounted to retain experiments')
    assert.equal((html.match(/role="tab" aria-selected="true"/g) ?? []).length, 1)
    for (const tab of tabs) {
      assert.ok(html.includes(`aria-controls="test-panel-${tab.id}"`))
      assert.ok(html.includes(`aria-labelledby="test-tab-${tab.id}"`))
    }
  }

  const gallery = render(Gallery, '/demos')
  for (const slug of ['sam', 'vit']) {
    assert.ok(gallery.includes(`href="/papers/${slug}/demo"`))
    assert.ok(fs.existsSync(path.join(root, `public/demos/${slug}/thumb.jpg`)))
  }
  assert.match(gallery, /Choose an experiment/)

  for (const view of ['segment', 'inside', 'compare', 'automatic', 'about']) {
    const html = render(Sam, `/papers/sam/demo?view=${view}`)
    assert.equal(selectedTab(html), `sam-tab-${view}`)
    assert.equal(visiblePanels(html), 1)
    assert.match(html, /Loading the list of photos/)
    for (const [button] of html.matchAll(/<button\b[^>]*>[\s\S]*?<\/button>/g)) {
      assert.doesNotMatch(button, /<a\b/, 'Controls cannot contain nested interactive links')
    }
  }
  assert.equal(selectedTab(render(Sam, '/papers/sam/demo?view=unknown')), 'sam-tab-segment')
  const vit = render(Vit, '/papers/vit/demo')
  assert.match(vit, /Choose an image/)
  assert.match(vit, /Explore a saved run/)
  assert.match(vit, /Run on my device/)
  assert.match(vit, /aria-label="Use your own photo"/)
  assert.doesNotMatch(vit, /role="radio"/, 'Image and model choices use native toggle buttons')
  console.log('Demo checks passed: gallery destinations, tab URLs, legacy anchors, retained panels, SAM initial views, and ViT input controls.')
} finally {
  await server.close()
}
