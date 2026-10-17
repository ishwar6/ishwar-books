// Part 7 screenshots of the LATER papers (each one is its own web page on transformer-circuits.pub).
// Same rules as paper_shots.mjs (find / upto / up / figure / mark / pad), plus a `url` per entry, so that one run
// covers several pages. These boxes carry the later paper's name, so the side-by-side reader does not use them.
// usage (from code/papers/circuits): node shots_part7_later.mjs [only-this-file.png]
import fs from 'node:fs'
import path from 'node:path'
import { chromium } from '/Users/admin/.npm/_npx/e41f203b7505f1fb/node_modules/playwright/index.mjs'

const ICL = 'https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html'
const TOY = 'https://transformer-circuits.pub/2022/toy_model/index.html'
const MONO = 'https://transformer-circuits.pub/2023/monosemantic-features/index.html'
const SCALE = 'https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html'
const TRACE = 'https://transformer-circuits.pub/2025/attribution-graphs/methods.html'
const BIO = 'https://transformer-circuits.pub/2025/attribution-graphs/biology.html'

export const SHOTS = [
  { url: ICL, out: 'p7-icl-def.png', find: 'Formally, we define an induction head as one which exhibits',
    upto: 'The head’s output increases the logit corresponding to the attended-to token',
    mark: ['Prefix matching', 'Copying', 'The head attends back to previous tokens that were followed by the current and/or recent tokens',
      'The head’s output increases the logit corresponding to the attended-to token'] },
  { url: ICL, out: 'p7-icl-score.png', find: 'Throughout this work we compute a simple heuristic measure of in-context learning',
    upto: 'We chose the 500th and 50th token indices somewhat arbitrarily',
    mark: ['the loss of the 500th token in the context minus the average loss of the 50th token in the context', 'somewhat arbitrarily'] },
  { url: ICL, out: 'p7-icl-phase.png', find: 'it develops abruptly in a narrow window early in training',
    mark: ['develops abruptly in a narrow window early in training', 'roughly 2.5 to 5 billion tokens', 'less than 0.15 nats',
      'roughly 0.4 nats', 'except for the one layer model where not much in-context learning ever forms'] },
  { url: ICL, out: 'p7-icl-case.png', find: 'Together the claims establish a circumstantial case',
    mark: ['circumstantial case', 'only the beginnings of evidence'] },
  { url: TOY, out: 'p7-toy-intro.png', find: 'Consider a toy model where we train an embedding of five features',
    mark: ['five features of varying importance', 'in two dimensions', 'With dense features',
      'orthogonal basis of the most important two features', 'if we make the features sparse'] },
  { url: TOY, out: 'p7-toy-intro-fig.png', find: 'Consider a toy model where we train an embedding of five features', next: 1, pad: 10 },
  { url: TOY, out: 'p7-toy-hypothesis.png', find: 'Roughly, the idea of superposition is that neural networks',
    upto: 'Compressed sensing. In general',
    mark: ['want to represent more features than they have neurons', 'almost orthogonal', 'possible to recover the original vector'] },
  { url: TOY, out: 'p7-toy-model.png', find: 'The superposition hypothesis suggests that each feature in the higher-dimensional model',
    upto: 'The final step is whether to add an activation function',
    mark: ['each column', 'use the transpose of the same matrix', "allows the model to set features it doesn't represent to their expected value",
      'whether to add an activation function', 'critical to whether superposition occurs'] },
  { url: MONO, out: 'p7-mono-poly.png', find: 'Unfortunately, the most natural computational unit of the neural network',
    mark: ['polysemantic', 'academic citations, English dialogue, HTTP requests, and Korean text'] },
  { url: MONO, out: 'p7-mono-setup.png', find: 'The goal of this paper is to provide a detailed demonstration of a sparse autoencoder',
    mark: ['one-layer transformer with a 512-neuron MLP layer', '8 billion data points', '1× (512 features) to 256× (131,072 features)', '4,096 features'] },
  { url: SCALE, out: 'p7-scale-eq.png', find: 'As a preprocessing step we apply a scalar normalization',
    upto: 'Including the factor of', mark: ['L1 penalty on feature activations'] },
  { url: SCALE, out: 'p7-scale-runs.png', find: 'We trained three SAEs of varying sizes',
    upto: 'For all three SAEs, the average number of features active',
    mark: ['33,554,432 (~34M) features',
      'fewer than 300', 'at least 65% of the variance'] },
  { url: SCALE, out: 'p7-scale-ggb.png', find: 'clamping the Golden Gate Bridge feature',
    mark: ['10× its maximum activation value', 'the model starts to self-identify as the Golden Gate Bridge'] },
  { url: SCALE, out: 'p7-scale-neurons.png', find: 'for 82% of our features, the most-correlated neuron',
    mark: ['for 82% of our features, the most-correlated neuron has a correlation of 0.3 or smaller'] },
  { url: TRACE, out: 'p7-trace-linear.png', find: 'Linear Attribution Between Features.',
    mark: ['the direct interactions between features are linear', 'we freeze attention patterns and normalization denominators'] },
  { url: TRACE, out: 'p7-trace-local.png', find: 'The local replacement model can be viewed as a very large fully connected neural network',
    upto: 'The only nonlinearities in the local replacement model',
    mark: ['including via the residual stream and through attention', 'virtual weights', 'not instantiated in the underlying model'] },
  { url: TRACE, out: 'p7-trace-induction.png', find: 'Let\'s consider for a moment a much simpler model',
    upto: 'These models will have induction heads attend back to',
    mark: ['a humble 2-layer attention-only model', 'the answer isn’t very informative'] },
  { url: TRACE, out: 'p7-trace-induction2.png', find: 'This misses the entire interesting story!',
    mark: ['This misses the entire interesting story!', 'because it was preceded by', 'a kind of regression'] },
  { url: BIO, out: 'p7-bio-dallas.png', find: 'Fact: the capital of the state containing Dallas is',
    upto: 'In this section we provide evidence that, in this example',
    mark: ['first, inferring that the state containing Dallas is Texas', 'genuine two-step reasoning internally'] },
]

const OUT = path.join(path.dirname(new URL(import.meta.url).pathname), '..', '..', '..', 'public', 'img', 'papers', 'circuits')
const only = process.argv[2]
fs.mkdirSync(OUT, { recursive: true })
const b = await chromium.launch()
const p = await b.newPage({ viewport: { width: 1280, height: 1400 }, deviceScaleFactor: 2, colorScheme: 'light' })
let loaded = null
for (const s of SHOTS) {
  if (only && s.out !== only) continue
  if (loaded !== s.url) {
    await p.goto(s.url, { waitUntil: 'load', timeout: 120000 })
    await p.evaluate(async () => {   // scroll once so that lazy figures and maths are drawn
      for (let y = 0; y < document.body.scrollHeight; y += 1200) { window.scrollTo(0, y); await new Promise((r) => setTimeout(r, 30)) }
    })
    await p.waitForTimeout(2500)
    loaded = s.url
  }
  const res = await p.evaluate((s) => {
    document.querySelectorAll('mark.shot').forEach((m) => m.replaceWith(...m.childNodes))
    const BLOCK = 'p, li, figure, h1, h2, h3, d-title, table, ul, ol, .l-body, d-math[block], div.math-grid'
    const norm = (t) => t.replace(/\s+/g, ' ')
    const findBlock = (phrase) => {
      let best = null
      for (const el of document.querySelectorAll(BLOCK)) if (norm(el.textContent).includes(phrase) && (!best || best.contains(el))) best = el
      return best
    }
    let el = findBlock(s.find)
    if (!el) return { err: 'not found: ' + s.find }
    for (let i = 0; i < (s.up || 0); i++) el = el.parentElement
    for (let i = 0; i < (s.next || 0); i++) { el = el.nextElementSibling; if (!el) return { err: 'no next sibling' } }
    let top = el, bottom = el
    if (s.upto) { bottom = findBlock(s.upto); if (!bottom) return { err: 'upto not found: ' + s.upto } }
    let done = 0
    for (const phrase of s.mark || []) {
      const scope = [top]
      if (bottom !== top) { let n = top; while (n && n !== bottom) { n = n.nextElementSibling || n.parentElement?.nextElementSibling; if (n) scope.push(n) } }
      let hit = false
      for (const root of scope) {
        if (hit) break
        const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT)
        let node
        while ((node = w.nextNode())) {
          const t = node.data.replace(/\s/g, ' ')
          const i = t.indexOf(phrase)
          if (i < 0 || node.parentElement.closest('mark.shot')) continue
          const r = document.createRange(); r.setStart(node, i); r.setEnd(node, i + phrase.length)
          const m = document.createElement('mark'); m.className = 'shot'
          m.style.cssText = 'background:#fff176;color:inherit;padding:0 1px;border-radius:2px'
          r.surroundContents(m)
          hit = true; done++
          break
        }
      }
    }
    top.scrollIntoView({ block: 'start' })
    const a = top.getBoundingClientRect(), z = bottom.getBoundingClientRect()
    const x0 = Math.min(a.left, z.left), x1 = Math.max(a.right, z.right)
    return { x: x0 + window.scrollX, y: a.top + window.scrollY, w: x1 - x0, h: z.bottom - a.top, marks: done }
  }, s)
  if (res.err) { console.log(s.out, 'ERROR', res.err); continue }
  const pad = s.pad ?? 16, px = s.padx ?? 40, py = s.pady ?? (s.pad ?? 6)
  const clip = { x: Math.max(0, res.x - px), y: Math.max(0, res.y - py), width: Math.min(1280, res.w + 2 * px), height: res.h + 2 * py }
  await p.screenshot({ path: path.join(OUT, s.out), clip, fullPage: true })
  const want = (s.mark || []).length
  console.log(`${s.out}: ${Math.round(clip.width)}x${Math.round(clip.height)} css px, ${res.marks} highlights${want > res.marks ? ' (MISSING ' + (want - res.marks) + ')' : ''}`)
}
await b.close()
