// Run locally after adding a writing: npm run build && npx vite preview --port 4173, then
// npx -y -p playwright node scripts/og.mjs   (writes public/og/<slug>.png; commit the images)
// PORT=<n> picks another preview port.
// Social preview cards (1200x630): one per writing (its cover + title) and one for the site.
import { chromium } from 'playwright'
import fs from 'node:fs'
const OUT = new URL('../public/og', import.meta.url).pathname
fs.mkdirSync(OUT, { recursive: true })
const manifest = JSON.parse(fs.readFileSync(new URL('../src/generated/manifest.json', import.meta.url).pathname, 'utf8'))
const b = await chromium.launch()
const p = await b.newPage({ viewport: { width: 1440, height: 1200 } })
await p.goto(`http://localhost:${process.env.PORT ?? 4173}/writings`); await p.waitForTimeout(1200)
const covers = await p.evaluate(() => Object.fromEntries([...document.querySelectorAll('.writing-card')].map((c) => [c.querySelector('h3 a').getAttribute('href'), c.querySelector('.cover').innerHTML])))
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;')
const card = (title, kicker, cover) => `<!doctype html><html><head><meta charset="utf-8"><style>
  *{margin:0;box-sizing:border-box} body{width:1200px;height:630px;background:#0d0e16;font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;color:#ececef;overflow:hidden;position:relative}
  .art{position:absolute;inset:0;opacity:.9} .art svg{width:100%;height:100%}
  .shade{position:absolute;inset:0;background:linear-gradient(90deg,rgba(13,14,22,.97) 0%,rgba(13,14,22,.9) 45%,rgba(13,14,22,.25) 100%)}
  .text{position:absolute;left:72px;top:0;bottom:0;width:640px;display:flex;flex-direction:column;justify-content:center;gap:22px}
  .kicker{font-size:22px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#8fb4ff}
  h1{font-size:58px;line-height:1.08;letter-spacing:-.03em;font-weight:800}
  .brand{position:absolute;left:72px;bottom:56px;display:flex;align-items:center;gap:14px;font-size:24px;font-weight:700}
  .mark{width:44px;height:44px;border-radius:11px;background:linear-gradient(135deg,#7c9cff,#c4a1ff);color:#111114;display:grid;place-items:center;font-size:19px;font-weight:800}
  .url{color:#8e8e97;font-weight:500;margin-left:6px}
</style></head><body><div class="art">${cover}</div><div class="shade"></div>
<div class="text">${kicker ? `<div class="kicker">${esc(kicker)}</div>` : ''}<h1>${esc(title)}</h1></div>
<div class="brand"><span class="mark">IJ</span>Ishwar Jangid<span class="url">ishwarj.com</span></div></body></html>`
const shot = async (html, file) => {
  const q = await b.newPage({ viewport: { width: 1200, height: 630 } })
  await q.setContent(html); await q.waitForTimeout(150)
  await q.screenshot({ path: `${OUT}/${file}` }); await q.close()
}
for (const w of manifest.writings) {
  const cover = covers['/' + w.route] ?? ''
  await shot(card(w.title, w.series ? `${w.series} · Part ${w.seriesPart}` : 'Writing', cover), `${w.slug}.png`)
}
const first = Object.values(covers)[0] ?? ''
await shot(card('First-principles guides to LLM inference, RAG and GPUs', 'Writings and books', first), 'site.png')
for (const bk of manifest.books) await shot(card(bk.title, 'Book', first), `book-${bk.slug}.png`)

// research papers: a "sheet of paper" card in the paper theme's teal
const paperArt = (short, lines) => `<svg viewBox="0 0 1200 630" xmlns="http://www.w3.org/2000/svg">
  <rect width="1200" height="630" fill="#0d0e16"/>
  <g transform="translate(790 70) rotate(4)"><rect width="330" height="440" rx="14" fill="#16303a" stroke="#4fc3d9" stroke-opacity=".5"/></g>
  <g transform="translate(770 90)"><rect width="330" height="440" rx="14" fill="#f4f6f8"/>
    <text x="34" y="92" font-family="Georgia, serif" font-size="64" font-weight="700" fill="#111">${esc(short)}</text>
    ${Array.from({ length: lines }, (_, i) => `<rect x="34" y="${130 + i * 26}" width="${i % 3 === 2 ? 180 : 262}" height="9" rx="4" fill="${i === 3 || i === 4 ? '#ffdc33' : '#c9ced6'}"/>`).join('')}
  </g></svg>`
const papers = manifest.papers ?? []
const paperCard = (title, kicker, short) => card(title, kicker, paperArt(short, 12)).replace('color:#8fb4ff', 'color:#4fc3d9')
await shot(paperCard('Research papers, one section at a time', 'Research papers', 'BERT'), 'papers.png')
for (const pp of papers) {
  await shot(paperCard(`${pp.short}, explained`, `Research paper · ${pp.parts.length} parts`, pp.short), `paper-${pp.slug}.png`)
  for (const x of pp.parts) await shot(paperCard(x.title, `${pp.short} · Part ${x.partNumber} of ${pp.parts.length}`, pp.short), `paper-${pp.slug}-${x.slug}.png`)
}
console.log(fs.readdirSync(OUT))
await b.close()
