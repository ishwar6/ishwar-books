// Screenshot table rows of a docs page: from the row containing <first> to the row containing <last>, full table width.
// usage: node code/serving/rowshot.mjs <url> <out.png> "<first text>" "<last text>" [--width 1100]
import { chromium } from '/Users/admin/.npm/_npx/e41f203b7505f1fb/node_modules/playwright/index.mjs'
const [url, out, first, last, ...rest] = process.argv.slice(2)
const i = rest.indexOf('--width'); const width = i >= 0 ? Number(rest[i + 1]) : 1100
const b = await chromium.launch(); const p = await b.newPage({ viewport: { width, height: 1400 }, deviceScaleFactor: 2, colorScheme: 'light' })
await p.goto(url, { waitUntil: 'networkidle', timeout: 60000 }); await p.waitForTimeout(1200)
const r = await p.evaluate(([f, l]) => {
  const row = (t) => [...document.querySelectorAll('tr')].find((tr) => tr.innerText.includes(t))
  const a = row(f), z = row(l)
  if (!a || !z) return null
  const table = a.closest('table')
  const head = table.querySelector('thead tr')
  const A = a.getBoundingClientRect(), Z = z.getBoundingClientRect(), T = table.getBoundingClientRect()
  return { x: T.left, w: T.width, y: A.top + scrollY, h: Z.bottom - A.top, head: head ? head.getBoundingClientRect().height : 0 }
}, [first, last])
if (!r) { console.error('row not found'); process.exit(1) }
await p.screenshot({ path: out, clip: { x: Math.max(0, r.x - 8), y: r.y - 8, width: Math.min(width, r.w + 16), height: r.h + 16 }, fullPage: true })
console.log(`${out}: rows "${first}" .. "${last}" from ${url}`)
await b.close()
