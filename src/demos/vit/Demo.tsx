import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import {
  BASE, LIVE_BYTES, loadPictures, loadModel, loadSaved, fullAttention, pixelsOf, live, runLive, runExperiment,
  type Picture, type Result, type Timing, type Experiment, type Model,
} from './data'
import { G, N, T, patchNumbers, shuffleTiles, diverging, type Top } from './vitMath'
import { PictureView, Strip, Matrix, Heat, SimGrid, DistanceChart } from './views'
import './vit-demo.css'

const P1 = '/papers/vit/part-1-the-big-idea'
const P2 = '/papers/vit/part-2-the-model'

type Source = { kind: 'picture'; slug: string } | { kind: 'upload'; name: string }
type LiveState = { phase: 'idle' | 'loading' | 'ready' | 'error'; done: number; total: number; timing?: Timing; error?: string }

const fmt = (n: number) => n.toLocaleString('en-US')
const pct = (p: number) => (p >= 0.1 ? `${(100 * p).toFixed(1)}%` : p >= 0.001 ? `${(100 * p).toFixed(2)}%` : p >= 0.00001 ? `${(100 * p).toFixed(3)}%` : `${(100 * p).toExponential(1)}%`)
const rc = (n: number) => `row ${Math.floor(n / G) + 1}, column ${(n % G) + 1}`
const sec = (ms: number) => (ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`)

declare global { interface Window { __vitDemo?: Record<string, unknown> } }
const debug = (k: string, v: unknown) => { window.__vitDemo = { ...(window.__vitDemo ?? {}), [k]: v } }

function Step({ n, title, read, children, id }: { n: string; title: string; read: ReactNode; children: ReactNode; id: string }) {
  return (
    <section className="vit-demo-step" id={id} aria-labelledby={`${id}-h`}>
      <header className="vit-demo-stephead">
        <span className="vit-demo-stepn">{n}</span>
        <h2 id={`${id}-h`}>{title}</h2>
        <span className="vit-demo-read">{read}</span>
      </header>
      {children}
    </section>
  )
}

function TopList({ top, compact }: { top: Top; compact?: boolean }) {
  return (
    <ol className={`vit-demo-top${compact ? ' is-compact' : ''}`}>
      {(compact ? top.slice(0, 3) : top).map(([i, name, p]) => (
        <li key={i}>
          <span className="vit-demo-topname">{name}</span>
          <span className="vit-demo-topbar"><i style={{ width: `${Math.max(0.5, 100 * p)}%` }} /></span>
          <span className="vit-demo-topp">{pct(p)}</span>
        </li>
      ))}
    </ol>
  )
}

export default function Demo() {
  const [pictures, setPictures] = useState<Picture[]>([])
  const [source, setSource] = useState<Source>({ kind: 'picture', slug: 'cats' })
  const [useLive, setUseLive] = useState(false)
  const [rgba, setRgba] = useState<Uint8ClampedArray | null>(null)
  const [upload, setUpload] = useState<Uint8ClampedArray | null>(null)
  const [result, setResult] = useState<Result | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [liveState, setLiveState] = useState<LiveState>({ phase: 'idle', done: 0, total: LIVE_BYTES })
  const [att, setAtt] = useState<Float32Array | null>(null)
  const [firstRun, setFirstRun] = useState<number | null>(null)
  const startedAt = useRef(performance.now())
  const liveCache = useRef(new Map<string, Result>())

  // the panels' own state
  const [sel, setSel] = useState(5 * G + 4)
  const [hover, setHover] = useState<number | null>(null)
  const [patchSize, setPatchSize] = useState(16)
  const [afterPos, setAfterPos] = useState(true)
  const [posSel, setPosSel] = useState(3 * G + 10)
  const [layer, setLayer] = useState(0)
  const [head, setHead] = useState<number>(0)          // -1 = mean of the heads
  const [query, setQuery] = useState<number>(-1)       // -1 = the [class] token, else a patch
  const [exp, setExp] = useState<Experiment | null>(null)
  const [expBusy, setExpBusy] = useState(false)

  useEffect(() => { loadPictures().then(setPictures).catch((e) => setError(String(e))) }, [])

  const isLive = source.kind === 'upload' || useLive
  const key = source.kind === 'upload' ? `upload:${source.name}` : source.slug

  const ensureLive = useCallback(async () => {
    if (liveState.phase === 'ready') return
    setLiveState((s) => ({ ...s, phase: 'loading' }))
    try {
      const timing = await live.start((done, total) => setLiveState((s) => ({ ...s, phase: 'loading', done, total })))
      setLiveState({ phase: 'ready', done: timing.bytes, total: timing.bytes, timing })
      debug('timing', { ...timing, openedToReady: performance.now() - startedAt.current })
    } catch (e) {
      setLiveState((s) => ({ ...s, phase: 'error', error: (e as Error).message }))
      throw e
    }
  }, [liveState.phase])

  // load the picture and its result
  useEffect(() => {
    let stop = false
    setError(null)
    setAtt(null)
    setExp(null)
    ;(async () => {
      try {
        const px = source.kind === 'upload' ? upload! : await pixelsOf(`img/${source.slug}.png`)
        if (stop) return
        setRgba(px)
        let r: Result
        if (isLive) {
          const cached = liveCache.current.get(key)
          if (cached) r = cached
          else {
            setBusy('Loading the model')
            await ensureLive()
            if (stop) return
            setBusy('Running the model')
            r = await runLive(px)
            liveCache.current.set(key, r)
            if (firstRun == null) {
              const t = performance.now() - startedAt.current
              setFirstRun(t)
              debug('firstLivePrediction', t)
            }
          }
        } else {
          setBusy('Loading the saved run')
          r = await loadSaved(source.slug)
        }
        if (stop) return
        setResult(r)
        setBusy(null)
        setLayer((l) => Math.min(l, r.model.L - 1))
        setHead((h) => Math.min(h, r.model.H - 1))
        debug('result', { key, live: r.live, model: r.model.key, top5: r.top5, logits: r.logits ? Array.from(r.logits) : null,
                          rollout: Array.from(r.rollout), distance: r.distance, ms: r.ms })
        if (r.att) setAtt(r.att)
        else if (source.kind === 'picture') {
          const a = await fullAttention(source.slug)
          if (!stop) setAtt(a)
        }
      } catch (e) {
        if (!stop) { setError((e as Error).message); setBusy(null) }
      }
    })()
    return () => { stop = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, isLive, upload])

  const onFile = async (f: File | undefined) => {
    if (!f) return
    try {
      const px = await pixelsOf(f)
      liveCache.current.delete(`upload:${f.name}`)
      setUpload(px)
      setSource({ kind: 'upload', name: f.name })
    } catch {
      setError('This file could not be read as a picture.')
    }
  }

  const pic = source.kind === 'picture' ? pictures.find((p) => p.slug === source.slug) : null
  const m = result?.model
  const ready = !!(result && rgba && !busy)

  return (
    <div className="vit-demo">
      <p className="vit-demo-intro">
        A Vision Transformer (ViT) cuts a picture into small squares, turns each square into a list of numbers, lets the
        squares exchange information through attention, and reads the class from one extra token. Every panel below
        is one step of that path, with the real numbers of the model. The series explains each step in{' '}
        <Link to={P1}>Part 1</Link> and <Link to={P2}>Part 2</Link>.
      </p>

      <div className="vit-demo-pick">
        <div className="vit-demo-thumbs" role="radiogroup" aria-label="Picture">
          {pictures.map((p) => (
            <button
              key={p.slug}
              type="button"
              role="radio"
              aria-checked={source.kind === 'picture' && source.slug === p.slug}
              className="vit-demo-thumb"
              onClick={() => setSource({ kind: 'picture', slug: p.slug })}
              title={p.name}
            >
              <img src={`${BASE}img/${p.slug}.png`} alt={p.name} width={72} height={72} />
            </button>
          ))}
          <label className={`vit-demo-thumb vit-demo-upload${source.kind === 'upload' ? ' is-on' : ''}`}>
            <input type="file" accept="image/*" onChange={(e) => { onFile(e.target.files?.[0]); e.target.value = '' }} />
            <span>Your photo</span>
          </label>
        </div>
        <p className="vit-demo-source">
          {pic ? (
            <>
              {pic.name}. Photo from <a href={pic.flickr} target="_blank" rel="noreferrer">Flickr</a>, in the COCO 2017
              validation set (image {pic.coco}), licence <a href={pic.licence_url} target="_blank" rel="noreferrer">{pic.licence}</a>.
              {pic.slug === 'cats' && ' Part 1 of the series uses the same photo.'}
              {pic.slug === 'giraffes' && ' ImageNet has no giraffe class, so watch what the model answers.'}
            </>
          ) : (
            <>Your photo: {source.kind === 'upload' ? source.name : ''}. It stays in this browser. Nothing is uploaded; the model runs on your device.</>
          )}
        </p>

        <div className="vit-demo-models">
          {source.kind === 'picture' && (
            <div className="vit-demo-seg" role="radiogroup" aria-label="Model">
              <button type="button" role="radio" aria-checked={!useLive} onClick={() => setUseLive(false)}>
                ViT-B/16, saved run
              </button>
              <button type="button" role="radio" aria-checked={useLive} onClick={() => setUseLive(true)}>
                DeiT-Ti/16, live in your browser
              </button>
            </div>
          )}
          <p className="vit-demo-modelnote">
            {isLive ? (
              <>
                <b>Running here:</b> DeiT-Ti/16 (facebook/deit-tiny-patch16-224), the same ViT design at a smaller width:
                5.7 million weights, D = 192, 3 heads, 12 layers, 16 × 16 patches. The paper's ViT-B/16 would be a 173 MB
                download, too much for a web page, so the browser runs this one (11.6 MB).
              </>
            ) : (
              <>
                <b>Shown:</b> ViT-B/16 (google/vit-base-patch16-224), the model of the series: 86.6 million weights,
                D = 768, 12 heads, 12 layers. We ran it in Python and saved its numbers, so they show at once. Switch to the
                live model, or use your own photo, to run a ViT in this browser.
              </>
            )}
          </p>
          {isLive && liveState.phase === 'loading' && (
            <div className="vit-demo-progress" role="progressbar" aria-valuemin={0} aria-valuemax={liveState.total} aria-valuenow={liveState.done}>
              <i style={{ width: `${(100 * liveState.done) / liveState.total}%` }} />
              <span>Downloading the model: {(liveState.done / 1e6).toFixed(1)} of {(liveState.total / 1e6).toFixed(1)} MB</span>
            </div>
          )}
          {isLive && liveState.timing && result?.live && (
            <p className="vit-demo-timing">
              Model: {(liveState.timing.bytes / 1e6).toFixed(1)} MB downloaded in {sec(liveState.timing.download)}, ready
              in {sec(liveState.timing.session)}. This picture: {sec(result.ms ?? 0)} for one pass through the model.
              {firstRun != null && <> First prediction {sec(firstRun)} after the page opened.</>}
            </p>
          )}
        </div>
      </div>

      {error && <div className="vit-demo-error" role="alert">{error}</div>}
      {busy && <div className="vit-demo-busy" aria-live="polite">{busy}…</div>}

      {ready && m && rgba && (
        <>
          <PatchStep rgba={rgba} sel={sel} setSel={setSel} hover={hover} setHover={setHover} patchSize={patchSize} setPatchSize={setPatchSize} />
          <EmbedStep r={result!} sel={sel} afterPos={afterPos} setAfterPos={setAfterPos} />
          <PosStep m={m} sel={posSel} setSel={setPosSel} />
          <EncoderStep r={result!} rgba={rgba} att={att} layer={layer} setLayer={setLayer} head={head} setHead={setHead} query={query} setQuery={setQuery} />
          <RolloutStep r={result!} rgba={rgba} />
          <HeadStep r={result!} />
          <ExperimentStep
            r={result!}
            rgba={rgba}
            exp={exp ?? result!.exp}
            busy={expBusy}
            onRun={async () => {
              setExpBusy(true)
              try { setExp(await runExperiment(rgba, m.labels)) } catch (e) { setError((e as Error).message) }
              setExpBusy(false)
            }}
          />
        </>
      )}
    </div>
  )
}

/* ---------------- a. patches ---------------- */

function PatchStep({ rgba, sel, setSel, hover, setHover, patchSize, setPatchSize }: {
  rgba: Uint8ClampedArray; sel: number; setSel: (n: number) => void; hover: number | null; setHover: (n: number | null) => void
  patchSize: number; setPatchSize: (p: number) => void
}) {
  const shown = patchSize === 16 ? (hover ?? sel) : sel
  const nums = useMemo(() => patchNumbers(rgba, shown), [rgba, shown])
  const side = 224 / patchSize, n = side * side, seq = n + 1
  const base = 197 * 197
  const color = useCallback((v: number, i: number): [number, number, number] => {
    const t = (v + 1) / 2 * 255
    return i < 256 ? [t, 0.15 * t, 0.15 * t] : i < 512 ? [0.15 * t, t, 0.15 * t] : [0.2 * t, 0.3 * t, t]
  }, [])
  return (
    <Step n="1" id="patches" title="Cut the picture into patches"
      read={<>Read: <Link to={`${P2}#from-pixels-to-a-sequence-the-reshape`}>Part 2, the reshape</Link> · <Link to={`${P1}#the-quadratic-cost-of-attention-over-pixels`}>Part 1, the cost of attention</Link></>}>
      <div className="vit-demo-two">
        <div className="vit-demo-visual">
          <PictureView rgba={rgba} grid={patchSize} sel={sel} onPick={setSel} onHover={patchSize === 16 ? setHover : undefined}
            label={`The picture at 224 by 224 pixels with a grid of ${side} by ${side} patches`} />
        </div>
        <div className="vit-demo-text">
          <p>
            The model first squeezes the picture to 224 × 224 pixels. A 16-pixel grid then cuts it into 14 × 14 = 196
            patches. Each patch is 16 × 16 pixels with 3 colours, so it is a list of 16 × 16 × 3 = 768 numbers.
            Hover over or tap a patch.
          </p>
          <p className="vit-demo-label">
            Patch {shown + 1} of 196 ({rc(shown)}): its 768 numbers, red, then green, then blue, scaled to −1 … 1
          </p>
          <Strip values={nums} color={color} label="The 768 numbers of the patch" />
          <p className="vit-demo-mono">[{Array.from(nums.slice(0, 6)).map((v) => v.toFixed(3)).join(', ')}, …, {nums[767].toFixed(3)}]</p>
          <div className="vit-demo-seg" role="radiogroup" aria-label="Patch size">
            {[8, 16, 32].map((p) => (
              <button key={p} type="button" role="radio" aria-checked={patchSize === p} onClick={() => setPatchSize(p)}>{p} × {p}</button>
            ))}
          </div>
          <table className="vit-demo-table">
            <tbody>
              <tr><th>patches per side</th><td>224 / {patchSize} = {side}</td></tr>
              <tr><th>patches N</th><td>{side} × {side} = {fmt(n)}</td></tr>
              <tr><th>numbers per patch</th><td>{patchSize} × {patchSize} × 3 = {fmt(3 * patchSize * patchSize)}</td></tr>
              <tr><th>tokens with [class]</th><td>{fmt(seq)}</td></tr>
              <tr><th>attention scores per head</th><td>{fmt(seq)}² = {fmt(seq * seq)}{patchSize !== 16 && <> ({(seq * seq / base).toFixed(seq * seq > base ? 1 : 3)} × the 16-pixel cost)</>}</td></tr>
            </tbody>
          </table>
          <p className="vit-demo-note">
            {patchSize === 16
              ? 'The models here use 16 × 16 patches. Attention compares every token with every token, so its cost grows with the square of the sequence length.'
              : `This is arithmetic only: the models here run with 16 × 16 patches. Halving the patch side makes 4 times more patches and about 16 times more attention scores.`}
          </p>
        </div>
      </div>
    </Step>
  )
}

/* ---------------- b. projection and [class] ---------------- */

function scaleOf(a: Float32Array) {
  const s = Array.from(a, Math.abs).sort((x, y) => x - y)
  return s[Math.floor(0.99 * (s.length - 1))] || 1
}

function EmbedStep({ r, sel, afterPos, setAfterPos }: { r: Result; sel: number; afterPos: boolean; setAfterPos: (b: boolean) => void }) {
  const D = r.model.D
  const row = useMemo(() => r.patch.slice(sel * D, (sel + 1) * D), [r, sel, D])
  const mat = useMemo(() => {
    if (afterPos) return r.tokens
    const z = new Float32Array(T * D)
    z.set(r.model.cls, 0)
    z.set(r.patch, D)
    return z
  }, [r, afterPos, D])
  const scale = useMemo(() => scaleOf(r.tokens), [r])
  const rowScale = useMemo(() => scaleOf(row), [row])
  const color = useCallback((v: number): [number, number, number] => diverging(v / rowScale), [rowScale])
  const big = Array.from(row.keys()).sort((a, b) => Math.abs(row[b]) - Math.abs(row[a])).slice(0, 3)
  return (
    <Step n="2" id="embed" title="Project each patch, add the [class] token"
      read={<>Read: <Link to={`${P2}#equation-1-the-input-sequence`}>Part 2, Equation 1</Link> · <Link to={`${P2}#the-class-token-and-the-head`}>the [class] token</Link></>}>
      <p>
        One learned matrix E multiplies every patch's 768 numbers and gives {D} numbers per patch: the patch embedding.
        The same E serves all {N} patches. In front of them the model puts one learned [class] token. Its row starts the
        same for every picture, and at the end it carries the answer.
      </p>
      <p className="vit-demo-label">Patch {sel + 1} ({rc(sel)}) after the projection: {D} numbers (pick a patch in step 1)</p>
      <Strip values={row} color={color} label={`The ${D} numbers of the patch embedding`} />
      <p className="vit-demo-mono">
        largest: {big.map((i) => `#${i + 1} = ${row[i].toFixed(3)}`).join(', ')} · orange = positive, blue = negative
      </p>
      <div className="vit-demo-rowhead">
        <p className="vit-demo-label">The whole sequence: {T} tokens × {D} numbers ({fmt(T * D)} numbers)</p>
        <div className="vit-demo-seg" role="radiogroup" aria-label="Sequence">
          <button type="button" role="radio" aria-checked={!afterPos} onClick={() => setAfterPos(false)}>[x_class; x_p E]</button>
          <button type="button" role="radio" aria-checked={afterPos} onClick={() => setAfterPos(true)}>z₀ = … + E_pos</button>
        </div>
      </div>
      <div className="vit-demo-matrixwrap">
        <span className="vit-demo-matrixlab"><b>row 1: [class]</b><span>rows 2 to 197: the patches, left to right, top to bottom</span></span>
        <Matrix data={mat} rows={T} cols={D} scale={scale} sel={sel + 1} label={`The ${T} by ${D} token matrix`} />
      </div>
      <p className="vit-demo-note">
        Each row is one token, each column one of the {D} numbers. The yellow ticks mark patch {sel + 1}.
        Switch to z₀ to see the sequence after the position embeddings of step 3 are added: this is the input of the encoder.
      </p>
    </Step>
  )
}

/* ---------------- c. position embeddings ---------------- */

function PosStep({ m, sel, setSel }: { m: Model; sel: number; setSel: (n: number) => void }) {
  const map = useMemo(() => m.possim.slice(sel * N, (sel + 1) * N), [m, sel])
  const others = Array.from(map).map((v, i) => [v, i] as const).filter(([, i]) => i !== sel).sort((a, b) => b[0] - a[0])
  return (
    <Step n="3" id="positions" title="Add the position embeddings"
      read={<>Read: <Link to={`${P2}#position-embeddings`}>Part 2, position embeddings</Link> · the paper's Figure 7 (centre)</>}>
      <div className="vit-demo-two">
        <div className="vit-demo-text">
          <p>
            Attention does not know where a patch came from. So the model adds a learned vector E_pos to each token, one
            per position. Nobody told the model that the patches form a grid. Yet after training, the vector of each
            position is most similar to the vectors of its neighbours, its row and its column.
          </p>
          <p>
            Click a patch. The map shows the cosine similarity of its position vector with all 196 others (bright = similar,
            dark = opposite; the scale runs from −1 to 1). These are the weights of {m.key === 'vit-b16' ? 'ViT-B/16' : 'DeiT-Ti/16'}.
          </p>
          <p className="vit-demo-mono">
            position {sel + 1} ({rc(sel)}): most similar {rc(others[0][1])} ({others[0][0].toFixed(3)}), least similar{' '}
            {rc(others[others.length - 1][1])} ({others[others.length - 1][0].toFixed(3)})
          </p>
        </div>
        <div className="vit-demo-posviz">
          <figure>
            <Heat values={map} lo={-1} hi={1} mark={sel} size={220} label={`Cosine similarity of position ${sel + 1} with every position`} />
            <figcaption>Position {sel + 1} against all</figcaption>
          </figure>
          <figure>
            <SimGrid sim={m.possim} sel={sel} onPick={setSel} />
            <figcaption>All 196 maps, each in its own place (click one)</figcaption>
          </figure>
        </div>
      </div>
    </Step>
  )
}

/* ---------------- d. the encoder ---------------- */

function EncoderStep({ r, rgba, att, layer, setLayer, head, setHead, query, setQuery }: {
  r: Result; rgba: Uint8ClampedArray; att: Float32Array | null; layer: number; setLayer: (l: number) => void
  head: number; setHead: (h: number) => void; query: number; setQuery: (q: number) => void
}) {
  const { L, H } = r.model
  const row = useCallback((l: number, h: number): Float32Array | null => {
    const q = query + 1
    if (q === 0) return r.clsAtt.subarray((l * H + h) * T, (l * H + h + 1) * T)
    if (!att) return null
    const o = ((l * H + h) * T + q) * T
    return att.subarray(o, o + T)
  }, [r, att, query, H])
  const rows = useMemo(() => Array.from({ length: H }, (_, h) => row(layer, h)), [row, layer, H])
  const shown = useMemo(() => {
    if (rows.some((x) => !x)) return null
    if (head >= 0) return rows[head]!
    const mean = new Float32Array(T)
    for (const x of rows) for (let j = 0; j < T; j++) mean[j] += x![j] / H
    return mean
  }, [rows, head, H])
  const patches = shown ? shown.subarray(1) : null
  const best = patches ? Array.from(patches.keys()).sort((a, b) => patches[b] - patches[a])[0] : 0
  const self = shown && query >= 0 ? shown[query + 1] : null
  return (
    <Step n="4" id="encoder" title="The encoder: 12 layers of attention"
      read={<>Read: <Link to={`${P2}#equation-2-layernorm-attention-residual`}>Part 2, Equations 2 and 3</Link> · <Link to={`${P2}#equations-5-to-7-one-head`}>Appendix A, one head</Link> · <Link to={`${P2}#equation-8-many-heads`}>many heads</Link></>}>
      <p>
        Each of the {L} layers lets every token look at every token. In each of the {H} heads, a token's query is compared
        with all {T} keys, and a softmax turns the scores into weights that add up to 1. The token then takes that
        weighted mix of the values. Pick a layer and a head, then click a patch to see where it looks.
      </p>
      <div className="vit-demo-controls">
        <label className="vit-demo-slider">
          <span>Layer {layer + 1}</span>
          <input type="range" min={1} max={L} value={layer + 1} onChange={(e) => setLayer(Number(e.target.value) - 1)} aria-label="Layer" />
        </label>
        <div className="vit-demo-seg vit-demo-heads" role="radiogroup" aria-label="Head">
          {Array.from({ length: H }, (_, h) => (
            <button key={h} type="button" role="radio" aria-checked={head === h} onClick={() => setHead(h)}>{h + 1}</button>
          ))}
          <button type="button" role="radio" aria-checked={head === -1} onClick={() => setHead(-1)}>mean</button>
        </div>
        <button type="button" className={`vit-demo-btn${query === -1 ? ' is-on' : ''}`} onClick={() => setQuery(-1)} aria-pressed={query === -1}>
          Query: [class] token
        </button>
      </div>
      <div className="vit-demo-two">
        <div className="vit-demo-visual">
          {patches ? (
            <PictureView rgba={rgba} heat={patches} mark={query >= 0 ? query : null} onPick={setQuery}
              label={`Where ${query < 0 ? 'the [class] token' : `patch ${query + 1}`} attends in layer ${layer + 1}`} />
          ) : (
            <div className="vit-demo-wait">Loading the attention maps of this picture (about 2 MB)…</div>
          )}
          <p className="vit-demo-note">
            Bright patches get more attention from {query < 0 ? 'the [class] token' : <>patch {query + 1} (blue box)</>}; dark ones get little.
            Click any patch to make it the query.
          </p>
        </div>
        <div className="vit-demo-text">
          {shown && patches && (
            <p className="vit-demo-mono">
              layer {layer + 1}, {head >= 0 ? `head ${head + 1}` : 'mean of the heads'}, query {query < 0 ? '[class]' : `patch ${query + 1}`}:
              weight on [class] {pct(shown[0])}{self != null && <>, on itself {pct(self)}</>}, largest on patch {best + 1} ({pct(patches[best])})
            </p>
          )}
          <div className="vit-demo-heads-row">
            {rows.map((x, h) => x && (
              <figure key={h}>
                <Heat values={x.subarray(1)} lo={0} hi={Math.max(...x.subarray(1))} mark={query} size={64}
                  label={`Head ${h + 1}`} onClick={() => setHead(h)} active={head === h} />
                <figcaption>{h + 1}</figcaption>
              </figure>
            ))}
          </div>
          <p className="vit-demo-note">All {H} heads of layer {layer + 1} for this query. Each head learns its own way of looking.</p>
          <DistanceChart dist={r.distance} layer={layer} head={Math.max(head, 0)} onPick={(l, h) => { setLayer(l); setHead(h) }} />
          <p className="vit-demo-note">
            Mean attention distance, as in the paper's Figure 7 (right), for this picture: for each head, the
            attention-weighted distance in pixels from each patch to the patches it attends to (the [class] token left
            out), averaged over the 196 patches. In early layers some heads look near and some look far; in deep layers
            most heads look far.
          </p>
        </div>
      </div>
    </Step>
  )
}

/* ---------------- e. rollout ---------------- */

function RolloutStep({ r, rgba }: { r: Result; rgba: Uint8ClampedArray }) {
  const { L, H } = r.model
  const last = useMemo(() => {
    const v = new Float32Array(N)
    for (let h = 0; h < H; h++) for (let j = 0; j < N; j++) v[j] += r.clsAtt[((L - 1) * H + h) * T + 1 + j] / H
    return v
  }, [r, L, H])
  return (
    <Step n="5" id="rollout" title="Attention rollout: what the [class] token sees"
      read={<>The paper's Figure 6 and Appendix D.8</>}>
      <p>
        One attention map shows one layer. Information also flows through the residual path and through many layers. Attention
        rollout follows it: average the heads, add the identity for the residual path, and multiply the {L} layers. The
        [class] row of the product says how much each patch fed the final answer.
      </p>
      <div className="vit-demo-three">
        <figure><PictureView rgba={rgba} label="The picture" /><figcaption>The picture</figcaption></figure>
        <figure><PictureView rgba={rgba} heat={r.rollout} gamma={1} label="Attention rollout from the [class] token" /><figcaption>Rollout over all {L} layers</figcaption></figure>
        <figure><PictureView rgba={rgba} heat={last} gamma={1} label="Attention of the [class] token in the last layer" /><figcaption>Last layer alone, mean of heads</figcaption></figure>
      </div>
      <p className="vit-demo-note">
        The last layer alone often puts much of its weight on a few background patches. Later work (Darcet et al., 2023,
        "Vision Transformers Need Registers") found that large ViTs reuse such patches to hold information about the
        whole picture. Rollout also counts the earlier layers and the residual path, so its map is smoother.
      </p>
    </Step>
  )
}

/* ---------------- f. the head ---------------- */

function HeadStep({ r }: { r: Result }) {
  return (
    <Step n="6" id="head" title="The head: from the [class] token to a class"
      read={<>Read: <Link to={`${P2}#equation-4-the-image-representation`}>Part 2, Equation 4</Link></>}>
      <div className="vit-demo-two">
        <div className="vit-demo-text">
          <p>
            After the last layer, the model takes only the [class] token, normalises it with LayerNorm (y = LN(z_L⁰)), and
            one linear layer turns its {r.model.D} numbers into 1,000 scores, one per ImageNet class. A softmax turns the
            scores into probabilities that add up to 1.
          </p>
          <p className="vit-demo-note">
            {r.live ? 'DeiT-Ti/16, computed now in this browser. DeiT was trained with mixed pictures and softened labels, so its top probability is often lower than ViT-B/16\'s, even when the class is right.' : 'ViT-B/16, saved from the Python run.'} The labels are the 1,000 ImageNet classes.
          </p>
        </div>
        <TopList top={r.top5} />
      </div>
    </Step>
  )
}

/* ---------------- g. the experiment ---------------- */

function ExperimentStep({ r, rgba, exp, busy, onRun }: { r: Result; rgba: Uint8ClampedArray; exp: Experiment | null; busy: boolean; onRun: () => void }) {
  const shuffled = useMemo(() => (exp ? shuffleTiles(rgba, exp.perm) : null), [rgba, exp])
  return (
    <Step n="7" id="experiment" title="Experiment: shuffle the patches"
      read={<>Read: <Link to={`${P2}#position-embeddings`}>Part 2, why position embeddings are needed</Link></>}>
      <p>
        Attention treats its input as a set: shuffle the tokens and the outputs shuffle the same way. Only the position
        embeddings tell the model where each patch was. Here the same picture goes through the model four ways.
      </p>
      {!exp ? (
        <button type="button" className="vit-demo-btn is-on" onClick={onRun} disabled={busy}>
          {busy ? 'Running…' : 'Run the experiment (3 more passes)'}
        </button>
      ) : (
        <>
          <div className="vit-demo-cards">
            <figure className="vit-demo-card"><PictureView rgba={rgba} label="The picture" /><figcaption><b>As it is</b><TopList top={r.top5} compact /></figcaption></figure>
            <figure className="vit-demo-card"><PictureView rgba={shuffled} label="The patches shuffled" /><figcaption><b>Patches shuffled</b><TopList top={exp.shuffled} compact /></figcaption></figure>
            <figure className="vit-demo-card"><PictureView rgba={rgba} label="The picture" /><figcaption><b>No position embeddings</b><TopList top={exp.nopos} compact /></figcaption></figure>
            <figure className="vit-demo-card"><PictureView rgba={shuffled} label="The patches shuffled" /><figcaption><b>Shuffled, no position embeddings</b><TopList top={exp.noposShuffled} compact /></figcaption></figure>
          </div>
          <p>
            With the position embeddings, shuffling changes the answer: the model uses where each patch is. Without them,
            the last two runs give the same answer, because the [class] token sees the same set of patches in a
            different order.{exp.noposShuffleDiff != null && <> Their 1,000 scores differ by at most {exp.noposShuffleDiff.toExponential(1)}, which is rounding error.</>}{' '}
            The model was never trained without positions, so that answer is usually poor.
          </p>
          {r.live ? (
            <button type="button" className="vit-demo-btn" onClick={onRun} disabled={busy}>{busy ? 'Running…' : 'Shuffle again'}</button>
          ) : (
            <p className="vit-demo-note">The saved run uses one fixed shuffle. Switch to the live model to try new shuffles.</p>
          )}
        </>
      )}
    </Step>
  )
}
