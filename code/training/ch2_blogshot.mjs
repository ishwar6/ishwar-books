// Copy of blogshot.mjs for Chapter 2 with two changes: a desktop browser user agent (the OpenAI blog refuses the default
// headless one) and a yellow highlight on the matched phrase.
// usage: node ch2_blogshot.mjs <url> <out.png> "<phrase to highlight>" [--up N] [--pad 24]
import { chromium } from '/Users/admin/.npm/_npx/e41f203b7505f1fb/node_modules/playwright/index.mjs'
const [url, out, phrase, ...rest] = process.argv.slice(2)
const opt = (k, d) => { const i = rest.indexOf(k); return i >= 0 ? rest[i + 1] : d }
const pad = Number(opt('--pad', 24)), up = Number(opt('--up', 0))
const UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36'
const b = await chromium.launch()
const p = await b.newPage({ viewport: { width: 1100, height: 1400 }, deviceScaleFactor: 2, userAgent: UA, colorScheme: 'light' })
await p.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 }); await p.waitForTimeout(3500)
for (const t of ['Accept all', 'Accept', 'I agree', 'Got it']) { const btn = p.getByRole('button', { name: t, exact: false }).first(); if (await btn.count()) { try { await btn.click({ timeout: 1500 }) } catch {} } }
let el = p.getByText(phrase, { exact: false }).first()
await el.waitFor({ timeout: 20000 })
el = el.locator('xpath=ancestor-or-self::*[self::p or self::li or self::blockquote or self::section or self::div][1]')
for (let i = 0; i < up; i++) el = el.locator('xpath=..')
await el.evaluate((node, phrase) => {             // wrap the phrase in a yellow <mark>
  const walk = document.createTreeWalker(node, NodeFilter.SHOW_TEXT)
  while (walk.nextNode()) {
    const t = walk.currentNode, i = t.data.indexOf(phrase)
    if (i >= 0) { const r = document.createRange(); r.setStart(t, i); r.setEnd(t, i + phrase.length)
      const m = document.createElement('mark'); m.style.background = '#ffdb33'; m.style.color = 'inherit'; r.surroundContents(m); return }
  }
}, phrase)
await el.scrollIntoViewIfNeeded(); await p.waitForTimeout(400)
const r = await el.boundingBox()
const clip = { x: Math.max(0, r.x - pad), y: Math.max(0, r.y - pad), width: Math.min(1100, r.width + 2 * pad), height: r.height + 2 * pad }
await p.screenshot({ path: out, clip })
console.log(`${out}: ${Math.round(clip.width)}x${Math.round(clip.height)} css px (2x) from ${url}`)
await b.close()
