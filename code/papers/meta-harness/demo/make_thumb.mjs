// The demo card picture: the run chart of the scripted search, dark theme, saved as a PNG to resize to 800 x 420.
// Usage: node code/papers/meta-harness/demo/make_thumb.mjs <base url> <out.png>   (PLAYWRIGHT_DIR as in check_browser.mjs)
import { createRequire } from 'node:module'
import path from 'node:path'
import os from 'node:os'
const require = createRequire(path.join(process.env.PLAYWRIGHT_DIR ?? path.join(os.homedir(), '.venvs/shots'), 'x.js'))
const { chromium } = require('playwright')
const [base = 'http://localhost:4190', out = 'thumb.png'] = process.argv.slice(2)
const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1100, height: 900 }, deviceScaleFactor: 2 })
await page.addInitScript(() => { try { localStorage.setItem('theme', 'dark') } catch {} })
await page.goto(`${base}/papers/meta-harness/demo?view=run`)
await page.waitForSelector('.meta-harness-demo-run .meta-harness-demo-svg')
await page.addStyleTag({ content: '.meta-harness-demo-run .meta-harness-demo-step > :not(.meta-harness-demo-fig), .meta-harness-demo-run .meta-harness-demo-fig .meta-harness-demo-src { visibility: hidden }' })
await page.waitForTimeout(300)
const fig = await page.$('.meta-harness-demo-run .meta-harness-demo-fig')
await fig.scrollIntoViewIfNeeded()
const b = await fig.boundingBox()
const w = b.width + 24
const h = Math.round((w * 420) / 800)
await page.screenshot({ path: out, clip: { x: b.x - 12, y: b.y - Math.max(12, (h - b.height) / 2), width: w, height: h } })
await browser.close()
