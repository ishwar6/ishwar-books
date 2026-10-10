// Copy of ch2_blogshot.mjs for Chapter 3: captures the TABLE that contains a phrase, and highlights in yellow the table
// row that contains a second phrase.
// usage: node ch3_blogshot.mjs <url> <out.png> "<phrase inside the table>" "<phrase of the row to highlight>" [--pad 16]
import { chromium } from '/Users/admin/.npm/_npx/e41f203b7505f1fb/node_modules/playwright/index.mjs'
const [url, out, phrase, rowPhrase, ...rest] = process.argv.slice(2)
const opt = (k, d) => { const i = rest.indexOf(k); return i >= 0 ? rest[i + 1] : d }
const pad = Number(opt('--pad', 16))
const UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36'
const b = await chromium.launch()
const p = await b.newPage({ viewport: { width: 1100, height: 1400 }, deviceScaleFactor: 2, userAgent: UA, colorScheme: 'light' })
await p.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 }); await p.waitForTimeout(3500)
let el = p.getByText(phrase, { exact: false }).first()
await el.waitFor({ timeout: 20000 })
el = el.locator('xpath=ancestor-or-self::table[1]')
await el.evaluate((node, rowPhrase) => {
  for (const tr of node.querySelectorAll('tr')) if (tr.textContent.includes(rowPhrase)) tr.style.background = '#ffdb33'
}, rowPhrase)
await el.scrollIntoViewIfNeeded(); await p.waitForTimeout(400)
const r = await el.boundingBox()
const clip = { x: Math.max(0, r.x - pad), y: Math.max(0, r.y - pad), width: Math.min(1100, r.width + 2 * pad), height: r.height + 2 * pad }
await p.screenshot({ path: out, clip })
console.log(`${out}: ${Math.round(clip.width)}x${Math.round(clip.height)} css px (2x) from ${url}`)
await b.close()
