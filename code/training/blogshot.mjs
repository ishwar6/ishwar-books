// Screenshot one passage or figure of a web page (an engineering blog, a docs page) for a "From the source" box.
// usage: node blogshot.mjs <url> <out.png> <selector-or-"text=...">  [--width 1100] [--pad 24] [--full]
// Example: node blogshot.mjs https://www.anthropic.com/research/building-effective-agents public/img/training/ch1-anthropic-workflows.png "text=Workflows are systems"
// With "text=..." the element containing that text (its nearest block ancestor) is captured, plus `pad` pixels around it.
import { chromium } from '/Users/admin/.npm/_npx/e41f203b7505f1fb/node_modules/playwright/index.mjs'
const [url, out, target, ...rest] = process.argv.slice(2)
const opt = (k, d) => { const i = rest.indexOf(k); return i >= 0 ? rest[i + 1] : d }
const width = Number(opt('--width', 1100)), pad = Number(opt('--pad', 24)), full = rest.includes('--full')
// --wait <state>: load | domcontentloaded | networkidle (default); --up N: capture the Nth ancestor of the matched block (e.g. the whole list)
const waitUntil = opt('--wait', 'networkidle'), up = Number(opt('--up', 0))
const b = await chromium.launch(); const p = await b.newPage({ viewport: { width, height: 1400 }, deviceScaleFactor: 2 })
await p.goto(url, { waitUntil, timeout: 60000 }); await p.waitForTimeout(1200)
// dismiss common cookie banners
for (const t of ['Accept all', 'Accept', 'I agree', 'Got it', 'OK']) { const btn = p.getByRole('button', { name: t, exact: false }).first(); if (await btn.count()) { try { await btn.click({ timeout: 1500 }) } catch {} } }
if (full) { await p.screenshot({ path: out, fullPage: true }); console.log('full page ->', out); await b.close(); process.exit(0) }
let el = target.startsWith('text=') ? p.getByText(target.slice(5), { exact: false }).first() : p.locator(target).first()
await el.waitFor({ timeout: 20000 })
if (target.startsWith('text=')) el = el.locator('xpath=ancestor-or-self::*[self::p or self::li or self::blockquote or self::figure or self::table or self::pre or self::h1 or self::h2 or self::h3 or self::section][1]')
for (let i = 0; i < up; i++) el = el.locator('xpath=..')
await el.scrollIntoViewIfNeeded(); await p.waitForTimeout(400)
const r = await el.boundingBox()
const clip = { x: Math.max(0, r.x - pad), y: Math.max(0, r.y - pad), width: Math.min(width, r.width + 2 * pad), height: r.height + 2 * pad }
await p.screenshot({ path: out, clip })
console.log(`${out}: ${Math.round(clip.width)}x${Math.round(clip.height)} css px (2x) from ${url}`)
await b.close()
