// Screenshot passages and figures of the paper's web page, with key words highlighted in yellow.
// The paper is an HTML article (no PDF), so each excerpt is found on the live page and cropped.
// usage (from the repo root): node code/papers/circuits/paper_shots.mjs code/papers/circuits/shots_part1.json
// Each entry of the JSON list:
//   { "out": "p1-title.png",              file name in public/img/papers/circuits/
//     "find": "a phrase",                 text inside the block to capture (its nearest p / li / figure / h1 ...)
//     "figure": 3,                        or: capture the Nth <figure> of the page instead (0-based)
//     "upto": "a phrase",                 optional: extend the crop down to the block containing this phrase
//     "up": 0,                            optional: go up N parents from the matched block
//     "mark": ["word", "two words"],      phrases to highlight in yellow inside the crop
//     "pad": 16, "width": 1100 }          optional
// Pattern copied from code/training/blogshot.mjs (book-training worktree).
import fs from 'node:fs'
import path from 'node:path'
import { chromium } from '/Users/admin/.npm/_npx/e41f203b7505f1fb/node_modules/playwright/index.mjs'

const PAGE = 'https://transformer-circuits.pub/2021/framework/index.html'
const OUT = path.join(path.dirname(new URL(import.meta.url).pathname), '..', '..', '..', 'public', 'img', 'papers', 'circuits')
const shots = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'))
const only = process.argv[3]
fs.mkdirSync(OUT, { recursive: true })

const b = await chromium.launch()
const p = await b.newPage({ viewport: { width: 1280, height: 1400 }, deviceScaleFactor: 2, colorScheme: 'light' })
await p.goto(PAGE, { waitUntil: 'load', timeout: 90000 })
await p.waitForTimeout(2500)

for (const s of shots) {
  if (only && s.out !== only) continue
  const res = await p.evaluate((s) => {
    document.querySelectorAll('mark.shot').forEach((m) => m.replaceWith(...m.childNodes))
    const BLOCK = 'p, li, figure, h1, h2, h3, d-title, table, ul, ol, .l-body, d-math[block], .d-byline .published'
    const findBlock = (phrase) => {
      const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT)
      const norm = (t) => t.replace(/\s+/g, ' ')
      // match across child nodes: walk elements whose textContent contains the phrase, keep the deepest block
      let best = null
      for (const el of document.querySelectorAll(BLOCK)) {
        if (norm(el.textContent).includes(phrase) && (!best || best.contains(el))) best = el
      }
      return best
    }
    let el = s.figure !== undefined ? document.querySelectorAll('figure')[s.figure] : findBlock(s.find)
    if (!el) return { err: 'not found: ' + (s.find || s.figure) }
    for (let i = 0; i < (s.up || 0); i++) el = el.parentElement
    let top = el, bottom = el
    if (s.upto) { bottom = findBlock(s.upto); if (!bottom) return { err: 'upto not found: ' + s.upto } }
    // highlight
    for (const phrase of s.mark || []) {
      const scope = [top]; if (bottom !== top) { let n = top; while (n && n !== bottom) { n = n.nextElementSibling || n.parentElement?.nextElementSibling; if (n) scope.push(n) } }
      for (const root of scope) {
        const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT)
        let node
        while ((node = w.nextNode())) {
          const i = node.data.indexOf(phrase)
          if (i < 0 || node.parentElement.closest('mark.shot')) continue
          const r = document.createRange(); r.setStart(node, i); r.setEnd(node, i + phrase.length)
          const m = document.createElement('mark'); m.className = 'shot'
          m.style.cssText = 'background:#fff176;color:inherit;padding:0 1px;border-radius:2px'
          r.surroundContents(m)
          break
        }
      }
    }
    top.scrollIntoView({ block: 'start' })
    const a = top.getBoundingClientRect(), z = bottom.getBoundingClientRect()
    const x0 = Math.min(a.left, z.left), x1 = Math.max(a.right, z.right)
    return { x: x0 + window.scrollX, y: a.top + window.scrollY, w: x1 - x0, h: z.bottom - a.top, marks: document.querySelectorAll('mark.shot').length }
  }, s)
  if (res.err) { console.log(s.out, 'ERROR', res.err); continue }
  const pad = s.pad ?? 16
  const clip = { x: Math.max(0, res.x - pad), y: Math.max(0, res.y - pad), width: Math.min(1280, res.w + 2 * pad), height: res.h + 2 * pad }
  await p.screenshot({ path: path.join(OUT, s.out), clip, fullPage: true })
  console.log(`${s.out}: ${Math.round(clip.width)}x${Math.round(clip.height)} css px, ${res.marks} highlights${(s.mark || []).length > res.marks ? ' (MISSING ' + ((s.mark || []).length - res.marks) + ')' : ''}`)
}
await b.close()
