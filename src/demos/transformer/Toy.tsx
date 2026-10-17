/* Part 1 of the demo: our own from-scratch Transformer (code/papers/transformer/scratch/model.py), trained to turn
   "2021-03-12" into "12 March 2021", running live in the browser. Every step from the characters to the output. */
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { toy, loadToyMeta, type ToyMeta, type Encoded, type Decoded, type Timing } from './data'
import { softmax, argmax, topK, sample, humanDate, randomDate, fix } from './txMath'
import { AttnGrid, PEWaves, Seg, TokenWeights, TopBars, VecHeat, tokText } from './views'
import { P1, P2, P3, P5, P6, P7, CODE } from './links'

type LiveState = { phase: 'idle' | 'loading' | 'ready' | 'error'; done: number; total: number; timing?: Timing; error?: string }
const sec = (ms: number) => (ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`)
const MAX_OUT = 24

declare global { interface Window { __txDemo?: Record<string, unknown> } }
const debug = (k: string, v: unknown) => { window.__txDemo = { ...(window.__txDemo ?? {}), [k]: v } }

export function Step({ n, title, read, children, id }: { n: string; title: string; read: ReactNode; children: ReactNode; id: string }) {
  return (
    <section className="transformer-demo-step" id={id} aria-labelledby={`${id}-h`}>
      <header className="transformer-demo-stephead">
        <span className="transformer-demo-stepn">{n}</span>
        <h3 id={`${id}-h`}>{title}</h3>
        <span className="transformer-demo-read">{read}</span>
      </header>
      {children}
    </section>
  )
}

/** Mean over heads of one layer, or one head: (L, H, r, c) -> (r, c). */
function pick(att: Float32Array, H: number, r: number, c: number, layer: number, head: number): Float32Array {
  const size = r * c
  if (head >= 0) return att.slice((layer * H + head) * size, (layer * H + head + 1) * size)
  const out = new Float32Array(size)
  for (let h = 0; h < H; h++) for (let i = 0; i < size; i++) out[i] += att[(layer * H + h) * size + i] / H
  return out
}

function p99(a: Float32Array) {
  const s = Array.from(a, Math.abs).sort((x, y) => x - y)
  return s[Math.floor(0.99 * (s.length - 1))] || 1
}

export default function Toy() {
  const [meta, setMeta] = useState<ToyMeta | null>(null)
  const [live, setLive] = useState<LiveState>({ phase: 'idle', done: 0, total: 1 })
  const [draft, setDraft] = useState('2021-03-12')
  const [input, setInput] = useState('2021-03-12')
  const [pos, setPos] = useState(true)
  const [masked, setMasked] = useState(true)
  const [enc, setEnc] = useState<Encoded | null>(null)
  const [decs, setDecs] = useState<Decoded[]>([])
  const [out, setOut] = useState<number[]>([])          // written tokens, starting with <s>
  const [view, setView] = useState<number | null>(null)  // which decoder call to inspect (null = the last)
  const [mode, setMode] = useState<'greedy' | 'sample'>('greedy')
  const [temp, setTemp] = useState(1)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [encLayer, setEncLayer] = useState(0)
  const [encHead, setEncHead] = useState(0)
  const [encRow, setEncRow] = useState(8)
  const [decLayer, setDecLayer] = useState(1)
  const [decHead, setDecHead] = useState(-1)
  const [selfRow, setSelfRow] = useState<number | null>(null)
  const [firstAt, setFirstAt] = useState<number | null>(null)
  const opened = useRef(performance.now())
  const run = useRef(0)

  useEffect(() => { loadToyMeta().then(setMeta).catch((e) => setError(String(e))) }, [])
  useEffect(() => {
    setLive((s) => ({ ...s, phase: 'loading' }))
    toy.start((done, total) => setLive((s) => ({ ...s, done, total })))
      .then((timing) => { setLive({ phase: 'ready', done: timing.bytes, total: timing.bytes, timing }); debug('timing', timing) })
      .catch((e) => setLive((s) => ({ ...s, phase: 'error', error: (e as Error).message })))
  }, [])

  const vocab = meta?.vocab ?? []
  const stoi = useMemo(() => new Map(vocab.map((s, i) => [s, i])), [vocab])
  const bad = useMemo(() => [...draft].filter((c) => !stoi.has(c) || c.length !== 1 || ['<pad>', '<s>', '</s>'].includes(c)), [draft, stoi])
  const name = (i: number) => (vocab[i] === ' ' ? '␣' : vocab[i] ?? '?')

  /** Encode the input and make the first decoder call (prefix = <s>). */
  const start = useCallback(async () => {
    if (live.phase !== 'ready' || !meta) return
    const my = ++run.current
    setBusy(true)
    setError(null)
    try {
      const src = [...input].map((c) => stoi.get(c)!).concat(meta.eos)
      const e = await toy.encode(src, pos)
      const d = await toy.decode(e, [meta.bos], pos, masked)
      if (my !== run.current) return
      setEnc(e); setDecs([d]); setOut([meta.bos]); setView(null)
      setEncRow((r) => Math.min(r, src.length - 1))
      if (firstAt == null) { const t = performance.now() - opened.current; setFirstAt(t); debug('firstResult', t) }
      debug('encode', { input, src, ms: e.ms })
    } catch (err) { setError((err as Error).message) }
    setBusy(false)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [live.phase, meta, input, pos, masked, stoi])
  useEffect(() => { start() }, [start])

  const done = out.length > 1 && out[out.length - 1] === meta?.eos
  const full = out.length > MAX_OUT

  /** Choose the next token from the last call (greedy or sampled), then call the decoder on the longer prefix. */
  const stepOnce = useCallback(async (state: { out: number[]; decs: Decoded[] }) => {
    if (!enc || !meta) return null
    const last = state.decs[state.decs.length - 1]
    const V = meta.vocab.length
    const t = last.tgt.length
    const row = last.logits.slice((t - 1) * V, t * V)
    const p = softmax(row, temp)
    const next = mode === 'greedy' ? argmax(row) : sample(p)
    const o2 = [...state.out, next]
    if (next === meta.eos || o2.length > MAX_OUT) return { out: o2, decs: state.decs }
    const d = await toy.decode(enc, o2, pos, masked)
    return { out: o2, decs: [...state.decs, d] }
  }, [enc, meta, temp, mode, pos, masked])

  const next = async (toEnd: boolean) => {
    if (busy || done || full) return
    const my = run.current
    setBusy(true)
    try {
      let s = { out, decs }
      for (;;) {
        const r = await stepOnce(s)
        if (!r || my !== run.current) break
        s = r
        setOut(s.out); setDecs(s.decs); setView(null)
        if (!toEnd || s.out[s.out.length - 1] === meta!.eos || s.out.length > MAX_OUT) break
      }
      const last = s.decs[s.decs.length - 1]
      debug('lastDecode', { input, ids: s.out, prefix: last.tgt, logits: Array.from(last.logits), V: meta!.vocab.length, ms: last.ms })
    } catch (err) { setError((err as Error).message) }
    setBusy(false)
  }

  if (!meta) return error ? <div className="transformer-demo-error" role="alert">{error}</div> : <div className="transformer-demo-busy">Loading…</div>
  const c = meta.config
  const H = c.n_heads, D = c.d_model
  const n = enc?.src.length ?? 0
  const srcToks = enc ? enc.src.map((i) => vocab[i]) : []
  const k = view ?? decs.length - 1
  const dsel = decs[k]
  const right = humanDate(input)
  const written = vocab.length ? out.slice(1).filter((i) => i !== meta.eos).map((i) => vocab[i]).join('') : ''

  const apply = () => { if (!bad.length && draft.length > 0 && draft.length < c.max_len) setInput(draft) }
  const ab = meta.ablation

  return (
    <div className="transformer-demo-toy">
      <p>
        This is the Transformer that the series builds in <Link to={P7}>Part 7</Link>, from our own code
        (<a href={CODE} target="_blank" rel="noreferrer">model.py</a>): no library attention, every step written out. We trained a small copy of it
        on one task: turn a date like <code>2021-03-12</code> into <code>12 March 2021</code>. The day, the month and the year
        change places and the month becomes a word, so each output letter must find the right input characters. It has the
        paper's design at a smaller size: {c.n_encoder_layers} encoder and {c.n_decoder_layers} decoder layers, d_model = {D},{' '}
        {H} heads, d_ff = {c.d_ff}, {meta.params.toLocaleString('en-US')} weights. It got all {meta.n_test.toLocaleString('en-US')} held-out
        dates right ({meta.n_test.toLocaleString('en-US')} dates it never saw in training). It runs here, in your browser.
      </p>

      <div className="transformer-demo-bar">
        <form className="transformer-demo-input" onSubmit={(e) => { e.preventDefault(); apply() }}>
          <label htmlFor="tx-date">Input</label>
          <input id="tx-date" value={draft} maxLength={c.max_len - 1} spellCheck={false} autoComplete="off"
            onChange={(e) => setDraft(e.target.value)} aria-invalid={bad.length > 0} />
          <button type="submit" className="transformer-demo-btn is-on" disabled={bad.length > 0 || !draft}>Run</button>
          <button type="button" className="transformer-demo-btn" onClick={() => { const d = randomDate(); setDraft(d); setInput(d) }}>Random date</button>
        </form>
        <div className="transformer-demo-toggles">
          <Seg label="Position encoding" value={pos ? 1 : 0} onChange={(v) => setPos(!!v)} options={[{ v: 1, label: 'positions on' }, { v: 0, label: 'off' }]} />
          <Seg label="Decoder mask" value={masked ? 1 : 0} onChange={(v) => setMasked(!!v)} options={[{ v: 1, label: 'mask on' }, { v: 0, label: 'off' }]} />
        </div>
        {bad.length > 0 && <p className="transformer-demo-warn" role="alert">The model knows only digits, “-”, a space and the letters of the month names. Not known: {[...new Set(bad)].map((b) => `“${b}”`).join(' ')}</p>}
        {!bad.length && draft === input && !right && <p className="transformer-demo-warn">This is not a date from {meta.first.slice(0, 4)} to {meta.last.slice(0, 4)} in the form YYYY-MM-DD. The model never saw such an input; see what it does.</p>}
        {(!pos || !masked) && <p className="transformer-demo-warn">You switched off {!pos && 'the position encoding'}{!pos && !masked && ' and '}{!masked && 'the decoder mask'}. Every panel below shows the broken model. <a href="#tx-break">What breaks</a></p>}
        {live.phase === 'loading' && (
          <div className="transformer-demo-progress" role="progressbar" aria-valuemin={0} aria-valuemax={live.total} aria-valuenow={live.done}>
            <i style={{ width: `${(100 * live.done) / Math.max(1, live.total)}%` }} />
            <span>Downloading the model: {(live.done / 1e6).toFixed(1)} of {(live.total / 1e6).toFixed(1)} MB</span>
          </div>
        )}
        {live.phase === 'error' && <div className="transformer-demo-error" role="alert">The model did not load: {live.error}</div>}
        {live.timing && (
          <p className="transformer-demo-note">
            Model: two ONNX files, {(live.timing.bytes / 1e6).toFixed(1)} MB, downloaded in {sec(live.timing.download)} and ready in {sec(live.timing.session)}.
            {enc && <> Encoder: {sec(enc.ms)}. Last decoder call: {sec(decs[decs.length - 1]?.ms ?? 0)}.</>}
            {firstAt != null && <> First result {sec(firstAt)} after the page opened.</>}
          </p>
        )}
      </div>
      {error && <div className="transformer-demo-error" role="alert">{error}</div>}

      {enc && dsel && (
        <>
          <Step n="1" id="tx-tokens" title="Characters become tokens, tokens become ids"
            read={<>Read: <Link to={`${P2}#embeddings-and-the-shared-matrix`}>Part 2, embeddings</Link> · <Link to={`${P3}#what-an-attention-function-is`}>Part 3, tokens</Link></>}>
            <p>
              A model reads numbers, not text. Our toy model cuts the input into single characters; each one is a token
              with a number, its id, in a vocabulary of {vocab.length} tokens. The released translator in the second tab of this
              page uses word pieces in the same way, with 58,101 of them. The encoder input ends with the end token &lt;/s&gt;.
            </p>
            <div className="transformer-demo-chips" role="list" aria-label="Input tokens and their ids">
              {enc.src.map((id, i) => (
                <span key={i} role="listitem" className="transformer-demo-chip"><b>{tokText(vocab[id])}</b><small>{id}</small></span>
              ))}
            </div>
            <details className="transformer-demo-details">
              <summary>The whole vocabulary ({vocab.length} tokens)</summary>
              <div className="transformer-demo-chips is-small">
                {vocab.map((s, i) => <span key={i} className="transformer-demo-chip"><b>{s === ' ' ? '␣' : s}</b><small>{i}</small></span>)}
              </div>
              <p className="transformer-demo-note">One vocabulary serves the input and the output, as in §3.4 of the paper. &lt;s&gt; starts the output and &lt;pad&gt; fills short rows of a batch.</p>
            </details>
          </Step>

          <EmbedStep enc={enc} toks={srcToks} D={D} pos={pos} layout={c.pos_layout} />

          <Step n="3" id="tx-encoder" title="The encoder: every character looks at every character"
            read={<>Read: <Link to={`${P3}#the-three-uses-of-attention`}>Part 3, self-attention</Link> · <Link to={`${P3}#multi-head-attention`}>multi-head attention</Link> · <Link to={`${P2}#the-encoder-stack`}>Part 2, the encoder stack</Link></>}>
            <p>
              Each encoder layer runs {H} attention heads side by side. In a head, each token sends a query, compares it with
              the key of every token, and takes a weighted mix of their values. A row of the grid is one query; it adds up
              to 1. After attention, a feed-forward network works on each token alone. The model has {c.n_encoder_layers} such layers.
              Pick a head, then hover or tap a row.
            </p>
            <div className="transformer-demo-heads" role="radiogroup" aria-label="Encoder layer and head">
              {Array.from({ length: c.n_encoder_layers }, (_, l) => Array.from({ length: H }, (_, h) => (
                <button key={`${l}-${h}`} type="button" role="radio" aria-checked={encLayer === l && encHead === h}
                  className="transformer-demo-mini" onClick={() => { setEncLayer(l); setEncHead(h) }}>
                  <MiniGrid w={pick(enc.att, H, n, n, l, h)} r={n} c={n} />
                  <span>layer {l + 1}, head {h + 1}</span>
                </button>
              )))}
            </div>
            <div className="transformer-demo-two">
              <AttnGrid rows={srcToks} cols={srcToks} w={pick(enc.att, H, n, n, encLayer, encHead)} sel={encRow} onSel={setEncRow} numbers
                label={`Encoder self-attention, layer ${encLayer + 1}, head ${encHead + 1}`} rowTitle="query" colTitle="keys →" />
              <div>
                <p className="transformer-demo-label">Layer {encLayer + 1}, head {encHead + 1}: where “{tokText(srcToks[encRow])}” (position {encRow}) looks</p>
                <TokenWeights toks={srcToks} w={pick(enc.att, H, n, n, encLayer, encHead).slice(encRow * n, (encRow + 1) * n)} q={encRow} label="Attention weights of the selected query" />
                <p className="transformer-demo-note">
                  Compare the eight small grids: each head has its own pattern, and no one told it which. Some heads put
                  almost all of a row on one character; others spread it over a group. With the position encoding off,
                  the two “2”s of 2021 get exactly the same row: the model cannot tell them apart.
                </p>
              </div>
            </div>
          </Step>

          <Step n="4" id="tx-decode" title="The decoder writes one token at a time"
            read={<>Read: <Link to={`${P1}#training-is-parallel-writing-is-not`}>Part 1, writing is not parallel</Link> · <Link to={`${P3}#the-mask-no-looking-ahead`}>Part 3, the mask</Link> · <Link to={`${P2}#the-decoder-stack`}>Part 2, the decoder stack</Link></>}>
            <p>
              The decoder starts from &lt;s&gt;. Each call reads everything written so far and gives a score (a logit) to each
              of the {vocab.length} tokens for the next place. A softmax turns the scores into probabilities. We pick one token, add it,
              and call the decoder again, until it writes &lt;/s&gt;. The encoder ran only once; its output is reused at every step.
            </p>
            <div className="transformer-demo-controls">
              <button type="button" className="transformer-demo-btn is-on" disabled={busy || done || full} onClick={() => next(false)}>Next step</button>
              <button type="button" className="transformer-demo-btn" disabled={busy || done || full} onClick={() => next(true)}>Write to the end</button>
              <button type="button" className="transformer-demo-btn" disabled={busy || out.length < 2} onClick={() => start()}>Start again</button>
              <Seg label="Decoding" value={mode} onChange={setMode} options={[{ v: 'greedy', label: 'greedy' }, { v: 'sample', label: 'sampling' }]} />
              <label className="transformer-demo-slider">
                <span>temperature {temp.toFixed(1)}</span>
                <input type="range" min={0.1} max={3} step={0.1} value={temp} onChange={(e) => setTemp(+e.target.value)} />
              </label>
            </div>
            <div className="transformer-demo-written" role="list" aria-label="Tokens written so far; pick one to inspect its step">
              {out.map((id, i) => {
                const isCall = i < decs.length
                return (
                  <button key={i} type="button" role="listitem" disabled={!isCall}
                    className={`transformer-demo-chip${i === k ? ' is-sel' : ''}${id === meta.eos ? ' is-end' : ''}`}
                    onClick={() => setView(i === decs.length - 1 ? null : i)} title={isCall ? `inspect the decoder call that read up to here` : undefined}>
                    <b>{name(id)}</b><small>{i}</small>
                  </button>
                )
              })}
              {!done && !full && <span className="transformer-demo-chip is-next" aria-label="next token">?</span>}
            </div>
            <DecodeView meta={meta} enc={enc} d={dsel} srcToks={srcToks} temp={temp} mode={mode} masked={masked}
              chosen={out[k + 1]} layer={decLayer} setLayer={setDecLayer} head={decHead} setHead={setDecHead} selfRow={selfRow} setSelfRow={setSelfRow} />
          </Step>

          <Step n="5" id="tx-output" title="The output"
            read={<>Read: <Link to={`${P6}#averaged-checkpoints-and-beam-search`}>Part 6, greedy and beam search</Link> · <Link to={`${P5}#label-smoothing`}>Part 5, label smoothing</Link></>}>
            <div className="transformer-demo-final">
              <div><span className="transformer-demo-label">Input</span><code>{input}</code></div>
              <div><span className="transformer-demo-label">Model output{done ? '' : ' so far'}</span><code>{written || ' '}</code></div>
              <div><span className="transformer-demo-label">Right answer</span><code>{right ?? 'not a valid date'}</code></div>
            </div>
            <p className={done && right ? (written === right ? 'transformer-demo-ok' : 'transformer-demo-bad') : 'transformer-demo-note'}>
              {!done ? 'Press “Write to the end” to finish the output.'
                : !right ? 'There is no right answer for this input. The model still writes something: it has no way to say “I do not know”.'
                : written === right ? 'Right, letter for letter.' : 'Wrong. Look at the step where the output went off track, and at its top 5.'}
            </p>
            <p className="transformer-demo-note">
              Greedy decoding takes the most probable token at each step. Sampling draws a token at random with the
              probabilities shown; a low temperature makes the top token more likely, a high one flattens the choice. The paper
              used beam search, which keeps the 4 best partial outputs at each step. The probabilities of this model are rarely
              exactly 1, partly because it was trained with label smoothing, as in the paper.
            </p>
          </Step>

          <Step n="6" id="tx-break" title="Break it: no positions, no mask"
            read={<>Read: <Link to={`${P2}#positional-encoding`}>Part 2, positional encoding</Link> · <Link to={`${P3}#the-mask-no-looking-ahead`}>Part 3, the mask</Link></>}>
            <div className="transformer-demo-controls">
              <Seg label="Position encoding" value={pos ? 1 : 0} onChange={(v) => setPos(!!v)} options={[{ v: 1, label: 'positions on' }, { v: 0, label: 'off' }]} />
              <Seg label="Decoder mask" value={masked ? 1 : 0} onChange={(v) => setMasked(!!v)} options={[{ v: 1, label: 'mask on' }, { v: 0, label: 'off' }]} />
            </div>
            <p>
              <b>No position encoding.</b> The page feeds a table of zeros in place of the sine waves, to the encoder and the
              decoder. Attention alone does not know the order of the tokens, so “2021-03-12” and “2012-03-21” look like the same
              bag of characters. The decoder also loses track of how much it has written. In Python, on all {ab.no_position.n.toLocaleString('en-US')} held-out
              dates, the model gets {ab.no_position.right.toLocaleString('en-US')} right ({fix(100 * ab.no_position.exact, 1)}%), against {ab.normal.right.toLocaleString('en-US')} with positions.
              For example, {ab.no_position.wrong_examples[0][0]} gave “{ab.no_position.wrong_examples[0][2]}”.
            </p>
            <p>
              <b>No mask.</b> The model was trained with the mask, so each position learned to read only the positions before it.
              When writing, the future does not exist yet, so the last row never sees it. But the earlier rows now also look at
              later rows, so their vectors are not the ones the model learned to read. In Python the model then gets {ab.no_mask.right.toLocaleString('en-US')} of {ab.no_mask.n.toLocaleString('en-US')} right
              ({fix(100 * ab.no_mask.exact, 1)}%); for example {ab.no_mask.wrong_examples[0][0]} gave “{ab.no_mask.wrong_examples[0][2]}”
              and {ab.no_mask.wrong_examples[1][0]} gave “{ab.no_mask.wrong_examples[1][2]}”. In training the damage would be worse: the whole answer is in the input,
              and without the mask each position could copy the next character.
            </p>
            <p className="transformer-demo-note">
              Both switches change the inputs of the same ONNX model: the position table and the mask are inputs of the graph,
              so nothing else changes. The numbers above come from <code>export_toy.py</code>, which runs the same switches in PyTorch.
            </p>
          </Step>
        </>
      )}
      {busy && !enc && <div className="transformer-demo-busy" aria-live="polite">Running the model…</div>}
    </div>
  )
}

function MiniGrid({ w, r, c }: { w: Float32Array; r: number; c: number }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const cv = ref.current
    if (!cv) return
    const ctx = cv.getContext('2d')!
    const img = ctx.createImageData(c, r)
    for (let i = 0; i < r * c; i++) {
      const t = Math.min(1, w[i]) ** 0.6
      const v = Math.round(20 + 235 * t)
      img.data.set([Math.round(v * 0.35), Math.round(v * 0.8), v, 255], 4 * i)
    }
    ctx.putImageData(img, 0, 0)
  }, [w, r, c])
  return <canvas ref={ref} width={c} height={r} aria-hidden="true" />
}

function EmbedStep({ enc, toks, D, pos, layout }: { enc: Encoded; toks: string[]; D: number; pos: boolean; layout: string }) {
  const n = toks.length
  const scale = useMemo(() => Math.max(1, p99(enc.emb)), [enc])
  const [which, setWhich] = useState<'emb' | 'pe' | 'x0'>('x0')
  const data = which === 'emb' ? enc.emb : which === 'pe' ? enc.pe : enc.x0
  return (
    <Step n="2" id="tx-embed" title="Embedding plus position encoding"
      read={<>Read: <Link to={`${P2}#embeddings-and-the-shared-matrix`}>Part 2, embeddings</Link> · <Link to={`${P2}#positional-encoding`}>Part 2, positional encoding</Link></>}>
      <p>
        Each id picks one row of a learned table: a vector of {D} numbers, multiplied by √{D} = {Math.sqrt(D).toFixed(1)}. Attention
        compares tokens but has no sense of order, so the model adds a second vector that depends only on the position: the
        sine and cosine waves of §3.5. Their sum is what the first encoder layer reads.
      </p>
      <div className="transformer-demo-rowhead">
        <Seg label="Matrix" value={which} onChange={setWhich} options={[
          { v: 'emb', label: 'embedding' }, { v: 'pe', label: '+ position' }, { v: 'x0', label: '= encoder input' }]} />
        <span className="transformer-demo-note">{n} tokens × {D} numbers · orange positive, blue negative</span>
      </div>
      <VecHeat data={data} rows={n} d={D} scale={scale} labels={toks} label={`The ${n} by ${D} matrix: ${which}`} />
      <p className="transformer-demo-label">The position waves for a few of the {D} dimensions (dots: the {n} input positions)</p>
      <PEWaves pe={enc.pe} d={D} dims={[0, 1, 8, 16, 32, 64]} n={n} maxPos={20} off={!pos} />
      <p className="transformer-demo-note">
        {pos
          ? <>Dimension 0 turns fast: a full wave every 2π = 6.3 positions, so it tells neighbours apart. Dimension 64 turns about 100 times more slowly. Together they give each position its own pattern, like the hands of a clock{layout === 'interleaved' ? ' (sines in even dimensions, cosines in odd ones, as in the paper)' : ''}.</>
          : <>The position encoding is off: the page feeds zeros, so the encoder input is the embedding alone, and the same character gets the same vector wherever it is.</>}
      </p>
    </Step>
  )
}

function DecodeView({ meta, enc, d, srcToks, temp, mode, masked, chosen, layer, setLayer, head, setHead, selfRow, setSelfRow }: {
  meta: ToyMeta; enc: Encoded; d: Decoded; srcToks: string[]; temp: number; mode: 'greedy' | 'sample'; masked: boolean
  chosen: number | undefined; layer: number; setLayer: (l: number) => void; head: number; setHead: (h: number) => void
  selfRow: number | null; setSelfRow: (r: number | null) => void
}) {
  const V = meta.vocab.length, H = meta.config.n_heads, L = meta.config.n_decoder_layers
  const t = d.tgt.length, n = enc.src.length
  const row = d.logits.slice((t - 1) * V, t * V)
  const p = softmax(row, temp)
  const top = topK(p, 5)
  const tg = d.tgt.map((i) => meta.vocab[i])
  const sa = pick(d.self, H, t, t, layer, head)
  const ca = pick(d.cross, H, t, n, layer, head)
  const r = selfRow != null && selfRow < t ? selfRow : t - 1
  const nm = (i: number) => (meta.vocab[i] === ' ' ? '␣ (space)' : meta.vocab[i])
  const ci = chosen != null ? top.findIndex(([i]) => i === chosen) : -1
  return (
    <div className="transformer-demo-decode">
      <div className="transformer-demo-controls">
        <span className="transformer-demo-label">Decoder layer</span>
        <Seg small label="Decoder layer" value={layer} onChange={setLayer} options={Array.from({ length: L }, (_, l) => ({ v: l, label: String(l + 1) }))} />
        <span className="transformer-demo-label">head</span>
        <Seg small label="Decoder head" value={head} onChange={setHead} options={[...Array.from({ length: H }, (_, h) => ({ v: h, label: String(h + 1) })), { v: -1, label: 'mean' }]} />
      </div>
      <div className="transformer-demo-three">
        <div>
          <p className="transformer-demo-label">Masked self-attention over the {t} token{t > 1 ? 's' : ''} written so far</p>
          <AttnGrid rows={tg} cols={tg} w={sa} sel={r} onSel={setSelfRow} numbers={t <= 14} maxPx={30}
            masked={masked ? (i, j) => j > i : undefined} label="Decoder self-attention" rowTitle="query" colTitle="keys →" />
          <p className="transformer-demo-note">{masked
            ? 'Hatched cells are hidden by the mask: a token sees itself and the tokens before it. The last row makes the prediction.'
            : 'The mask is off: earlier rows now also see later tokens. The model never learned to read that.'}</p>
        </div>
        <div>
          <p className="transformer-demo-label">Cross-attention onto the input</p>
          <AttnGrid rows={tg} cols={srcToks} w={ca} sel={r} onSel={setSelfRow} numbers={t <= 14} maxPx={30}
            label="Decoder cross-attention" rowTitle="query" colTitle="input →" />
          <p className="transformer-demo-label">Row “{tokText(tg[r])}” (position {r}) looks at</p>
          <TokenWeights toks={srcToks} w={ca.slice(r * n, (r + 1) * n)} label="Cross-attention weights of the selected row" />
          <p className="transformer-demo-note">The queries come from the decoder, the keys and values from the encoder output. A row is the call that has just read its token and writes the next one. With layer 2 and the mean of the heads, the rows that write the day look at the last two digits, and the rows that write the year look at the first four. The rows that write the month look mostly at the dash after the month: after the encoder, the vector there has probably gathered the month digits next to it.</p>
        </div>
        <div>
          <p className="transformer-demo-label">Next token: the top 5{temp !== 1 && <> at temperature {temp.toFixed(1)}</>}</p>
          <TopBars items={top.map(([i, q]) => ({ label: nm(i), p: q }))} chosen={ci >= 0 ? ci : undefined} />
          <p className="transformer-demo-mono">
            logits: {top.map(([i]) => `${nm(i).slice(0, 4)} ${fix(row[i], 2)}`).join(', ')}
          </p>
          <p className="transformer-demo-note">
            {chosen != null
              ? <>At this step the page took “{nm(chosen)}”.</>
              : mode === 'greedy' ? <>“Next step” takes the top token.</> : <>“Next step” draws a token with these probabilities.</>}
            {' '}The probabilities are softmax(logits / temperature).
          </p>
        </div>
      </div>
    </div>
  )
}
