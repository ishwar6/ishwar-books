// Print the paper's web page to a PDF, for the side-by-side reader (/papers/transformer-circuits/read).
// The paper is an HTML article with no PDF, so we make our own: Chrome prints the live page once every maths
// formula, font and figure has loaded. The site's top bar is hidden.
//
// Each excerpt of shots_part*.json (the "From the paper" screenshots) is found on the page with the same rules as
// paper_shots.mjs, and tagged so that reader.py can find it in the PDF without guessing:
//   - the excerpt (from the `find` block down to the `upto` block) gets 1 px links .../start/<shot name>/0 and
//     .../end/<shot name>/0 where its text starts and ends, and invisible links .../box/<shot name>/<k> over
//     its first and last block (they give its width),
//   - every highlighted phrase is painted yellow and wrapped in a link to https://reader.invalid/hl/<shot name>
//     (hl/<name>,<name> on words that several excerpts highlight).
// Chrome writes these links into the PDF with their exact rectangles (one per page or per line); reader.py reads
// them, draws the numbered markers and removes the links.
//
// usage (from the repo root): node code/papers/circuits/paper_pdf.mjs   ->  ~/.cache/papers/circuits-framework.pdf
// Needs Playwright (the copy in ~/.venvs/shots/node_modules works).
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createRequire } from 'node:module'

const require = createRequire(path.join(os.homedir(), '.venvs', 'shots', 'node_modules', '.keep'))
const { chromium } = require('playwright')

const PAGE = 'https://transformer-circuits.pub/2021/framework/index.html'
const HERE = path.dirname(new URL(import.meta.url).pathname)
const OUT = process.argv[2] || path.join(os.homedir(), '.cache', 'papers', 'circuits-framework.pdf')
const shots = fs.readdirSync(HERE).filter((f) => /^shots_part\d+\.json$/.test(f)).sort()
  .flatMap((f) => JSON.parse(fs.readFileSync(path.join(HERE, f), 'utf8')))

const b = await chromium.launch()
const p = await b.newPage({ viewport: { width: 1280, height: 1400 }, colorScheme: 'light' })
await p.goto(PAGE, { waitUntil: 'networkidle', timeout: 120000 })

// scroll through the whole page so that anything loaded on demand is there, then wait for maths, fonts and images
await p.evaluate(async () => {
  for (let y = 0; y < document.body.scrollHeight; y += 900) { window.scrollTo(0, y); await new Promise((r) => setTimeout(r, 40)) }
  window.scrollTo(0, 0)
})
await p.waitForFunction(() => [...document.querySelectorAll('d-math')].every((m) => m.shadowRoot?.querySelector('.katex')), null, { timeout: 60000 })
await p.evaluate(async () => {
  await Promise.all([...document.images].map((i) => i.complete ? null : new Promise((r) => { i.onload = i.onerror = r })))
  await document.fonts.ready
})

const report = await p.evaluate((shots) => {
  const out = []
  const found = new Map()         // text node -> [{a, b, name}]: the phrases to paint, as character ranges
  const BLOCK = 'p, li, figure, h1, h2, h3, d-title, table, ul, ol, .l-body, d-math[block], .d-byline .published, div.math-grid'
  const norm = (t) => t.replace(/\s+/g, ' ')
  const findBlock = (phrase) => {
    let best = null
    for (const el of document.querySelectorAll(BLOCK)) if (norm(el.textContent).includes(phrase) && (!best || best.contains(el))) best = el
    return best
  }
  for (const s of shots) {
    const name = s.out.replace(/\.png$/, '')
    let top = s.figure !== undefined ? document.querySelectorAll('figure')[s.figure] : findBlock(s.find)
    if (!top) { out.push(`${name}: NOT FOUND ${s.find || s.figure}`); continue }
    for (let i = 0; i < (s.up || 0); i++) top = top.parentElement
    const bottom = s.upto ? findBlock(s.upto) : top
    if (!bottom) { out.push(`${name}: upto NOT FOUND ${s.upto}`); continue }
    const scope = [top]
    // the blocks from top down to the one that holds `upto` (stop at a block that contains it)
    if (bottom !== top && !top.contains(bottom)) {
      let n = top
      while (n && n !== bottom && !n.contains(bottom)) { n = n.nextElementSibling || n.parentElement?.nextElementSibling; if (n) scope.push(n) }
      if (!n) { out.push(`${name}: upto is not below find: ${s.upto}`); continue }
    }
    // The excerpt runs from the top of the `find` block to the bottom of the `upto` block, like the screenshot.
    // Tags: an invisible link over each of the two blocks (it gives the width; Chrome writes it on one page only
    // when a block breaks across pages), and a 1 px link where the excerpt's text starts and where it ends, so that
    // reader.py knows exactly where an excerpt that crosses a page break begins and ends.
    const link = (kind, k) => {
      const a = document.createElement('a')
      a.href = `https://reader.invalid/${kind}/${name}/${k}`
      a.setAttribute('aria-hidden', 'true')
      a.style.cssText = kind === 'box'
        ? 'position:absolute;inset:0;display:block;border:0;background:none;pointer-events:none'
        : 'display:inline-block;width:1px;height:1em;margin-right:-1px;vertical-align:baseline;border:0;background:none'
      return a
    }
    const texts = (el) => {          // the text nodes that show on the page (not the TeX source inside d-math, nor footnotes)
      const w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, { acceptNode: (n) =>
        n.parentElement.closest('d-math, d-footnote, d-cite, style, script') || !n.data.trim() ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT })
      const list = []; let n
      while ((n = w.nextNode())) list.push(n)
      return list
    }
    const edge = (el, kind) => {
      if (el.tagName === 'D-MATH') { out.push(`${name}: the ${kind} is a display formula; no tag`); return }
      const t = texts(el)
      if (t.length) { if (kind === 'start') t[0].before(link(kind, 0)); else t[t.length - 1].after(link(kind, 0)); return }
      const a = link(kind, 0)                       // a figure: its top or bottom edge
      a.style.cssText += kind === 'start' ? ';position:absolute;left:0;top:0' : ';position:absolute;left:0;bottom:0'
      el.append(a)
    }
    for (const [k, el] of [top, bottom].entries()) {
      if (k === 1 && el === top) break
      if (el.tagName === 'D-MATH') continue
      if (getComputedStyle(el).position === 'static') el.style.position = 'relative'
      el.appendChild(link('box', k))
    }
    edge(top, 'start')
    edge(bottom, 'end')
    // the key phrases: yellow, like the screenshots (first match in the excerpt, as paper_shots.mjs does). They are
    // only found here; the painting waits until every excerpt is found (below), so that two excerpts that quote the
    // same words both get them, as both screenshots do.
    let marks = 0
    for (const phrase of s.mark || []) {
      let done = false
      for (const root of scope) {
        const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT)
        let node
        while ((node = w.nextNode())) {
          const i = node.data.indexOf(phrase)
          if (i < 0) continue
          if (!found.has(node)) found.set(node, [])
          found.get(node).push({ a: i, b: i + phrase.length, name })
          marks++; done = true
          break
        }
        if (done) break
      }
    }
    out.push(`${name}: ${scope.length} block(s), ${marks}/${(s.mark || []).length} highlights`)
  }
  // Paint the phrases. Where phrases of several excerpts overlap, the text is cut at every phrase edge and each piece
  // links to all the excerpts that hold it (https://reader.invalid/hl/<name>,<name>), so each excerpt gets exactly
  // its own words. The pieces of a node are wrapped from right to left, so the offsets on the left stay valid.
  for (const [node, rs] of found) {
    const cuts = [...new Set(rs.flatMap((r) => [r.a, r.b]))].sort((x, y) => y - x)
    for (let j = 1; j < cuts.length; j++) {
      const a = cuts[j], b = cuts[j - 1]
      const names = [...new Set(rs.filter((r) => r.a <= a && r.b >= b).map((r) => r.name))]
      if (!names.length) continue
      const r = document.createRange(); r.setStart(node, a); r.setEnd(node, b)
      const m = document.createElement('a'); m.className = 'rd-hl'; m.href = `https://reader.invalid/hl/${names.join(',')}`
      // a cut inside yellow text (where another piece goes on) gets no padding and square corners: no seam
      if (rs.some((x) => x.a < a && x.b >= a)) m.style.cssText += ';padding-left:0;border-top-left-radius:0;border-bottom-left-radius:0'
      if (rs.some((x) => x.a <= b && x.b > b)) m.style.cssText += ';padding-right:0;border-top-right-radius:0;border-bottom-right-radius:0'
      r.surroundContents(m)
    }
  }
  return out
}, shots)
console.log(report.join('\n'))

// one column, like a printed paper. The page is printed with the screen styles at the width of the printed text
// (701 px), where the article already uses one column; the figures that run into the right margin on a wide screen
// are fitted to the text, and the small margin figures sit in the text. (The site's own print styles give a fixed
// grid wider than the page, so they are not used.)
await p.setViewportSize({ width: 701, height: 1000 })
await p.emulateMedia({ media: 'screen' })
await p.addStyleTag({ content: `
  .article-header, d-contents, distill-header { display: none !important; }
  figure.gdoc-image, figure.gdoc-image-gutter, .l-gutter, .l-page, .l-screen, .l-body-outset { grid-column: text !important; }
  figure img, figure svg, figure canvas { max-width: 100% !important; height: auto !important; }
  figure.gdoc-image-gutter { margin: 0 0 1em; }
  a.rd-hl { color: inherit !important; text-decoration: none !important; border-bottom: 0 !important;
    background: rgba(255, 219, 51, 0.42) !important; border-radius: 2px; padding: 0 1px; }
  figure, img, d-math[block], .math-grid, table, tr { break-inside: avoid; }
  h2, h3, h4 { break-after: avoid; }
` })
await p.waitForTimeout(800)
fs.mkdirSync(path.dirname(OUT), { recursive: true })
await p.pdf({
  path: OUT, printBackground: true, width: '8.5in', height: '11in',
  margin: { top: '0.5in', bottom: '0.55in', left: '0.6in', right: '0.6in' },
})
await b.close()
console.log(`${OUT}: ${(fs.statSync(OUT).size / 1e6).toFixed(1)} MB`)
