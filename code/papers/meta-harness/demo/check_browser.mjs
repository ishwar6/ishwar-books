// Checks what the demo page shows against the source files.
// 1. The paper's numbers: every TerminalBench-2 bar (Table 7), every Table 6 dumbbell for every method, every Figure 3
//    point's card (Tables 2 and 9) and the Table 3 ablation rows, against paper_numbers.json.
// 2. The search runs: for every candidate of every exported run, clicks its folder and reads the search and test scores
//    and the number of wrong traces, against the run's own summary.json and scores.json in ../scratch/runs/.
// Usage: node code/papers/meta-harness/demo/check_browser.mjs [http://localhost:4190]   (needs playwright; set
// PLAYWRIGHT_DIR to a folder whose node_modules has it). Writes results/browser_check.txt.
import { createRequire } from 'node:module'
import fs from 'node:fs'
import path from 'node:path'
import os from 'node:os'

const require = createRequire(path.join(process.env.PLAYWRIGHT_DIR ?? path.join(os.homedir(), '.venvs/shots'), 'x.js'))
const { chromium } = require('playwright')
const here = path.dirname(new URL(import.meta.url).pathname)
const base = process.argv[2] ?? 'http://localhost:4190'
const P = JSON.parse(fs.readFileSync(path.join(here, 'paper_numbers.json'), 'utf8'))
const index = JSON.parse(fs.readFileSync(path.join(here, '../../../../public/demos/meta-harness/run/index.json'), 'utf8'))

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } })
const errors = []
page.on('pageerror', (e) => errors.push(e.message))
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()) })
const out = []
let checked = 0, bad = 0
const eq = (what, got, want) => { checked++; if (String(got) !== String(want)) { bad++; out.push(`MISMATCH ${what}: page ${got}, file ${want}`) } }

// ---- 1. the paper's numbers ----
await page.goto(`${base}/papers/meta-harness/demo?view=results`)
await page.waitForSelector('.meta-harness-demo-hbars')
for (const g of P.table7.groups) {
  await page.getByRole('radio', { name: g.model, exact: true }).click()
  const rows = await page.$$eval('.meta-harness-demo-hbars li', (lis) => lis.map((li) => [li.querySelector('.meta-harness-demo-hname').firstChild.textContent.trim(), li.querySelector('.meta-harness-demo-hval').textContent]))
  eq(`Table 7 ${g.model} rows`, rows.length, g.rows.length)
  for (const r of g.rows) eq(`Table 7 ${g.model} ${r.harness}`, rows.find((x) => x[0] === r.harness)?.[1], r.pass.toFixed(1))
}
for (const r of P.table6.rows.slice(1)) {
  await page.getByRole('radio', { name: r.method, exact: true }).click()
  const vals = await page.$$eval('.meta-harness-demo-dumb li .meta-harness-demo-ddelta', (es) => es.map((e) => e.firstChild.textContent.trim()))
  r.acc.forEach((a, i) => eq(`Table 6 ${r.method} ${P.table6.models[i].name}`, vals[i], a.toFixed(1)))
  eq(`Table 6 ${r.method} average`, vals[5], r.avg.toFixed(1))
}
const pts = await page.$$('.meta-harness-demo-pt')
for (const pt of pts) {
  await pt.focus()
  const name = await page.$eval('.meta-harness-demo-card h4', (e) => e.textContent)
  const avg = await page.$eval('.meta-harness-demo-card dd', (e) => e.textContent)
  const star = P.table9.rows.find((r) => name === `Meta-Harness (${r.variant})`)
  const base2 = P.table2.rows.find((r) => r.harness === name)
  eq(`Figure 3 point ${name}`, avg, `${(star ?? base2).avg.toFixed(1)}%`)
}
eq('Figure 3 points', pts.length, P.table9.rows.length + P.figure3.baselines.length)
await page.goto(`${base}/papers/meta-harness/demo?view=proposer`)
await page.waitForSelector('.meta-harness-demo-abl')
const abl = await page.$$eval('.meta-harness-demo-abl li', (lis) => lis.map((li) => [...li.querySelectorAll('em')].map((e) => e.textContent).concat(li.querySelector('.meta-harness-demo-azs b').textContent)))
P.table3.rows.forEach((r, i) => eq(`Table 3 ${r.method}`, abl[i].join('|'), `median ${r.median.toFixed(1)}|best ${r.best.toFixed(1)}|${r.above_zs}`))
out.push(`paper numbers: ${checked - bad}/${checked} values on the page equal paper_numbers.json`)

// ---- 2. the runs ----
const t0 = Date.now()
await page.goto(`${base}/papers/meta-harness/demo?view=run`)
await page.waitForSelector('.meta-harness-demo-cand h4')
out.push(`run tab: first candidate shown ${Date.now() - t0} ms after navigation`)
const before = checked, badBefore = bad
for (const r of index.runs) {
  const dir = path.join(here, '../scratch/runs', r.name)
  const summary = JSON.parse(fs.readFileSync(path.join(dir, 'summary.json'), 'utf8'))
  if (index.runs.length > 1) await page.getByRole('radio', { name: new RegExp(`^${r.name} `) }).click()
  await page.waitForSelector(`text=runs/${r.name}/`)
  for (const s of summary) {
    await page.click(`.meta-harness-demo-tnode:has-text("${s.folder}/")`)
    await page.waitForSelector(`.meta-harness-demo-candhead h4:text-is("${s.folder}/")`)
    const cards = await page.$$eval('.meta-harness-demo-score b', (bs) => bs.map((b) => b.textContent))
    eq(`${r.name}/${s.folder} search`, cards[0], `${s.search_acc.toFixed(1)}%`)
    eq(`${r.name}/${s.folder} test`, cards[1], s.test_acc === undefined ? '–' : `${s.test_acc.toFixed(1)}%`)
    eq(`${r.name}/${s.folder} prompt chars`, cards[2], s.ctx_chars.toLocaleString('en-US'))
    await page.getByRole('radio', { name: 'Traces', exact: true }).click()
    const wrongTab = await page.getByRole('radio', { name: /^Wrong \(/ }).textContent()
    const sc = JSON.parse(fs.readFileSync(path.join(dir, 'candidates', s.folder, 'scores.json'), 'utf8'))
    eq(`${r.name}/${s.folder} wrong`, wrongTab, `Wrong (${sc.n - sc.correct})`)
    await page.getByRole('radio', { name: 'proposal.md', exact: true }).click()
  }
}
out.push(`runs: ${checked - before - (bad - badBefore)}/${checked - before} values equal the run files (${index.runs.map((r) => r.name).join(', ')})`)
out.push(`console or page errors: ${errors.length}${errors.length ? ' ' + errors.join(' | ') : ''}`)
await browser.close()
fs.mkdirSync(path.join(here, 'results'), { recursive: true })
fs.writeFileSync(path.join(here, 'results/browser_check.txt'), out.join('\n') + '\n')
console.log(out.join('\n'))
process.exit(bad || errors.length ? 1 : 0)
