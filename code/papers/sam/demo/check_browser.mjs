// Replays the prompts of results/reference.json in the demo page (onnxruntime-web, float16 embedding) and compares
// each mask with the Python one (transformers SamModel, fp32 embedding).
// Usage: node code/papers/sam/demo/check_browser.mjs [http://localhost:4190]   (needs playwright; set PLAYWRIGHT_DIR
// to a folder whose node_modules has it). Writes results/browser_check.txt.
import { createRequire } from 'node:module'
import fs from 'node:fs'
import path from 'node:path'
import os from 'node:os'

const require = createRequire(path.join(process.env.PLAYWRIGHT_DIR ?? path.join(os.homedir(), '.venvs/shots'), 'x.js'))
const { chromium } = require('playwright')
const here = path.dirname(new URL(import.meta.url).pathname)
const base = process.argv[2] ?? 'http://localhost:4190'
const ref = JSON.parse(fs.readFileSync(path.join(here, 'results/reference.json'), 'utf8'))

const decode = (runs, n) => {
  const m = new Uint8Array(n)
  let o = 0, v = 0
  for (const r of runs) { if (v) m.fill(1, o, o + r); o += r; v ^= 1 }
  return m
}
const iou = (a, b) => {
  let i = 0, u = 0
  for (let k = 0; k < a.length; k++) { if (a[k] | b[k]) { u++; if (a[k] & b[k]) i++ } }
  return u ? i / u : 1
}

const browser = await chromium.launch()
const page = await browser.newPage()
await page.goto(base + '/papers/sam/demo')
await page.waitForSelector('.sam-demo-badge', { timeout: 240000 })
await page.waitForFunction(() => !!window.__samDemo)
const got = await page.evaluate((cases) => window.__samDemo.check(cases), ref.map((c) => ({ photo: c.photo, rounds: c.rounds })))
await browser.close()

const lines = ['browser (onnxruntime-web 1.22, wasm, float16 embedding) vs Python (transformers SamModel, fp32 embedding)']
let worstIou = 1, worstScore = 0, sameTop = 0
ref.forEach((c, i) => {
  const g = got[i], n = c.width * c.height
  const ious = [0, 1, 2].map((k) => iou(decode(c.masks[k], n), decode(g.masks[k], n)))
  const ds = Math.max(...[0, 1, 2].map((k) => Math.abs(c.scores[k] - g.scores[k])))
  const top = g.scores.indexOf(Math.max(...g.scores))
  worstIou = Math.min(worstIou, ...ious)
  worstScore = Math.max(worstScore, ds)
  sameTop += top === c.top ? 1 : 0
  lines.push(`${c.photo.padEnd(6)} ${c.name.padEnd(58)} mask IoU ${ious.map((v) => v.toFixed(4)).join(', ')}  ` +
    `score diff ${ds.toExponential(1)}  first: python ${c.top + 1}, browser ${top + 1}  decoder ${g.ms.toFixed(0)} ms`)
})
lines.push(`${ref.length} cases, ${ref.length * 3} masks: lowest IoU ${worstIou.toFixed(4)}, largest score difference ` +
  `${worstScore.toExponential(1)}, same first mask in ${sameTop} of ${ref.length}`)
console.log(lines.join('\n'))
fs.writeFileSync(path.join(here, 'results/browser_check.txt'), lines.join('\n') + '\n')
