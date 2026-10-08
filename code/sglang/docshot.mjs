// Screenshot a passage of a docs page from the block holding <start> text to the block holding <end> text.
// usage: node code/sglang/docshot.mjs <url> <out.png> "<start text>" "<end text>" [--width 1100] [--pad 16]
// Needed for pages whose prose sits in <span>/<div> rather than <p>, where blogshot.mjs cannot pick a block.
import { chromium } from '/Users/admin/.npm/_npx/e41f203b7505f1fb/node_modules/playwright/index.mjs'
const [url, out, start, end, ...rest] = process.argv.slice(2)
const opt = (k, d) => { const i = rest.indexOf(k); return i >= 0 ? rest[i + 1] : d }
const width = Number(opt('--width', 1100)), pad = Number(opt('--pad', 16))
const b = await chromium.launch(); const p = await b.newPage({ viewport: { width, height: 1400 }, deviceScaleFactor: 2, colorScheme: 'light' })
await p.goto(url, { waitUntil: 'networkidle', timeout: 60000 }); await p.waitForTimeout(1500)
const r = await p.evaluate(([s, e]) => {
  const block = (t) => {
    const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT); let n
    while ((n = w.nextNode())) {
      if (!n.textContent.includes(t)) continue
      let el = n.parentElement
      while (el && getComputedStyle(el).display.startsWith('inline')) el = el.parentElement
      const rc = el.getBoundingClientRect(); if (rc.width && rc.height) return rc
    }
    // the phrase may span several inline elements: fall back to the smallest block whose text contains it
    let best = null
    for (const el of document.querySelectorAll('div,p,li,td,section')) {
      if (el.innerText && el.innerText.includes(t)) { const rc = el.getBoundingClientRect(); if (rc.height && (!best || rc.height < best.height)) best = rc }
    }
    return best
  }
  const a = block(s), z = block(e)
  if (!a || !z) return null
  return { x: Math.min(a.left, z.left), y: a.top + scrollY, w: Math.max(a.right, z.right) - Math.min(a.left, z.left), h: z.bottom - a.top }
}, [start, end])
if (!r) { console.error('phrase not found'); process.exit(1) }
const clip = { x: Math.max(0, r.x - pad), y: Math.max(0, r.y - pad), width: Math.min(width, r.w + 2 * pad), height: r.h + 2 * pad }
await p.screenshot({ path: out, clip, fullPage: true })
console.log(`${out}: ${Math.round(clip.width)}x${Math.round(clip.height)} css px (2x) from ${url}`)
await b.close()
