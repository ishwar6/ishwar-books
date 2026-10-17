// Checks the demo page against Python.
// 1. Toy model: types each date of results/toy_reference.json into the page, writes the output greedily in the browser
//    (onnxruntime-web, wasm) and compares the text and the logits of the last decoder call with PyTorch (model.py).
// 2. Saved translator runs: selects a sentence, a kind of attention, a layer, a head and a row in the page and compares
//    the weights the page shows (decoded from 8-bit levels) with the float32 weights of the model (marian_reference.json).
// Usage: node code/papers/transformer/demo/check_browser.mjs [http://localhost:4190]   (needs playwright; set
// PLAYWRIGHT_DIR to a folder whose node_modules has it). Writes results/browser_check.txt.
import { createRequire } from 'node:module'
import fs from 'node:fs'
import path from 'node:path'
import os from 'node:os'

const require = createRequire(path.join(process.env.PLAYWRIGHT_DIR ?? path.join(os.homedir(), '.venvs/shots'), 'x.js'))
const { chromium } = require('playwright')
const here = path.dirname(new URL(import.meta.url).pathname)
const base = process.argv[2] ?? 'http://localhost:4190'
const toyRef = JSON.parse(fs.readFileSync(path.join(here, 'results/toy_reference.json'), 'utf8'))
const marRef = JSON.parse(fs.readFileSync(path.join(here, 'results/marian_reference.json'), 'utf8'))

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } })
const errors = []
page.on('pageerror', (e) => errors.push(e.message))
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()) })
const t0 = Date.now()
await page.goto(base + '/papers/transformer/demo')
await page.waitForSelector('#tx-tokens', { timeout: 120000 })
const firstMs = Date.now() - t0
const lines = [`page ${base}/papers/transformer/demo; first result ${firstMs} ms after navigation (playwright, this machine)`,
  'toy model: browser (onnxruntime-web 1.22, wasm, float32) vs PyTorch (scratch/model.py), greedy decoding']
let worst = 0, same = 0
for (const c of toyRef) {
  await page.fill('#tx-date', c.input)
  await page.click('form.transformer-demo-input button[type=submit]')
  await page.waitForFunction((x) => window.__txDemo?.encode?.input === x, c.input)
  await page.waitForFunction(() => !document.querySelector('.transformer-demo-toy button.transformer-demo-btn.is-on[disabled]') || document.querySelector('.transformer-demo-ok, .transformer-demo-bad'))
  await page.click('text=Write to the end')
  await page.waitForFunction((x) => window.__txDemo?.lastDecode?.input === x && window.__txDemo.lastDecode.ids.at(-1) === 2, c.input, { timeout: 60000 })
  const got = await page.evaluate(() => window.__txDemo.lastDecode)
  const V = got.V
  let d = 0
  for (let i = 0; i < c.logits.length; i++) for (let j = 0; j < V; j++) d = Math.max(d, Math.abs(c.logits[i][j] - got.logits[i * V + j]))
  const text = await page.innerText('.transformer-demo-final > div:nth-child(2) code')
  const ok = JSON.stringify(got.ids) === JSON.stringify(c.ids) && text.trim() === c.output
  same += ok
  worst = Math.max(worst, d)
  lines.push(`  ${c.input} -> browser "${text.trim()}", Python "${c.output}"  same tokens: ${ok}  max |logit diff| over ${c.logits.length} x ${V}: ${d.toExponential(2)}`)
}
lines.push(`  same output ${same}/${toyRef.length}; max |logit difference| ${worst.toExponential(2)}`)

await page.click('#transformer-tab-translator')
await page.waitForSelector('.transformer-demo-headlist', { state: 'visible', timeout: 60000 })
lines.push('saved translator runs: weights shown in the page (8-bit levels, row scale) vs float32 model weights')
let mw = 0
for (const r of marRef) {
  await page.selectOption('.transformer-demo-select select', r.slug)
  const kindLabel = { enc: 'encoder self-attention', cross: 'cross-attention', dec: 'decoder self-attention' }[r.kind]
  await page.click(`.transformer-demo-marian [role=radiogroup][aria-label="Kind of attention"] >> text="${kindLabel}"`)
  await page.click(`.transformer-demo-marian [role=radiogroup][aria-label="Layer"] >> text="${r.layer + 1}"`)
  for (let h = 0; h < r.heads.length; h++) {
    await page.click(`.transformer-demo-marian [role=radiogroup][aria-label="Head"] >> text="${h + 1}"`)
    const rows = r.heads[h].length
    for (const q of [0, Math.floor(rows / 2), rows - 1]) {
      await page.click(`.transformer-demo-marian [role=radiogroup][aria-label="Query token"] button >> nth=${q}`)
      await page.waitForFunction(([s, k, l, hh, qq]) => { const m = window.__txDemo?.marian; return m && m.slug === s && m.kind === k && m.layer === l && m.head === hh && m.row === qq },
        [r.slug, r.kind, r.layer, h, q])
      const got = await page.evaluate(() => window.__txDemo.marian.w)
      const want = r.heads[h][q]
      for (let j = 0; j < want.length; j++) mw = Math.max(mw, Math.abs(want[j] - got[j]))
    }
  }
  lines.push(`  ${r.slug}, ${r.kind}, layer ${r.layer + 1}, ${r.heads.length} heads x 3 rows: running max |difference| ${mw.toFixed(4)}`)
}
// the paper's Figure 3 check from Part 6: encoder layer 5, head 3, the word "making"
await page.selectOption('.transformer-demo-select select', 'making')
await page.click('.transformer-demo-marian [role=radiogroup][aria-label="Kind of attention"] >> text="encoder self-attention"')
await page.click('.transformer-demo-marian [role=radiogroup][aria-label="Layer"] >> text="5"')
await page.click('.transformer-demo-marian [role=radiogroup][aria-label="Head"] >> text="3"')
await page.click('.transformer-demo-marian [role=radiogroup][aria-label="Query token"] button >> text="making"')
const fig3 = await page.innerText('.transformer-demo-headlist .transformer-demo-headrow.is-sel')
lines.push(`  "making", encoder layer 5, head 3 in the page: ${fig3.replace(/\s+/g, ' ')} (Part 6 prints: difficult 0.71, more 0.29, process 0.00)`)
lines.push(`console errors: ${errors.length ? errors.join(' | ') : 'none'}`)
await browser.close()
const out = lines.join('\n') + '\n'
fs.writeFileSync(path.join(here, 'results/browser_check.txt'), out)
process.stdout.write(out)
