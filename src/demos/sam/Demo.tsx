import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { SamClient, fetchWithProgress, type Box, type Decoded, type Prompts, type AmgMask, type AmgCounts } from './client'
import { LOW, frameOf, iou as maskIou, maskStats, toMask, unpack } from './samMath'
import './sam-demo.css'

const BASE = import.meta.env.BASE_URL + 'demos/sam/'
const P1 = '/papers/sam/part-1-the-big-idea'
const P2 = '/papers/sam/part-2-the-task'
const P3 = '/papers/sam/part-3-the-model'
const P4 = '/papers/sam/part-4-data-engine-and-sa-1b'

type Photo = {
  key: string; title: string; file: string; emb: string; width: number; height: number
  source: string; why: string; url: string; licence?: string; licence_url?: string; click?: [number, number]
}
type Entry = { prompts: Prompts; mask: Float32Array | null }
type Saved = { key: string; mask: Uint8Array; label: string; area: number }

// colours of the pictures (photos and masks are images, not page chrome)
const MASK_RGB = [64, 156, 255]
const SAVED_RGB = [255, 160, 64]
const FG = '#2fd67b', BG = '#ff4d5e'

const fmt = (n: number) => n.toLocaleString('en-US')
const pct = (a: number, b: number) => `${((100 * a) / b).toFixed(2)}%`

export default function Demo() {
  const client = useRef<SamClient | null>(null)
  const [photos, setPhotos] = useState<Photo[]>([])
  const [key, setKey] = useState('cats')
  const [img, setImg] = useState<HTMLImageElement | null>(null)
  const [status, setStatus] = useState<{ text: string; frac?: number } | null>({ text: 'Loading the list of photos…' })
  const [error, setError] = useState<string | null>(null)
  const [ready, setReady] = useState(false)          // model loaded
  const [embKey, setEmbKey] = useState<string | null>(null)   // whose embedding the worker holds
  const [loadMs, setLoadMs] = useState<{ model: number; modelBytes: number; first?: number } | null>(null)

  const [mode, setMode] = useState<'fg' | 'bg'>('fg')
  const [history, setHistory] = useState<Entry[]>([])
  const [result, setResult] = useState<Decoded | null>(null)
  const [chosen, setChosen] = useState(1)
  const [feedback, setFeedback] = useState(true)
  const [thr, setThr] = useState(0)
  const [busy, setBusy] = useState(false)
  const [saved, setSaved] = useState<Saved | null>(null)
  const [isPreset, setIsPreset] = useState(false)
  const [amgRunning, setAmgRunning] = useState(false)   // one decoder at a time: clicks wait for the automatic run
  const t0 = useRef(performance.now())

  const photo = photos.find((p) => p.key === key)
  const prompts: Prompts = history.length ? history[history.length - 1].prompts : { points: [], box: null }

  // ---------------------------------------------------------------- loading: photo list and model, then a photo
  useEffect(() => {
    const c = new SamClient()
    client.current = c
    let alive = true
    ;(async () => {
      try {
        const list = await (await fetch(BASE + 'photos.json')).json()
        if (!alive) return
        setPhotos(list.photos)
        const t = performance.now()
        const model = await fetchWithProgress(BASE + 'sam_decoder.onnx', (got, total) =>
          alive && setStatus({ text: `Downloading the prompt encoder and mask decoder: ${(got / 1e6).toFixed(1)} of ${(total / 1e6 || 16.5).toFixed(1)} MB`, frac: total ? got / total : undefined }))
        const bytes = model.byteLength
        if (!alive) return
        setStatus({ text: 'Starting the decoder (WebAssembly)…' })
        await c.init(model)
        if (!alive) return
        setLoadMs({ model: performance.now() - t, modelBytes: bytes })
        setReady(true)
      } catch (e) {
        if (alive) setError(String(e))
      }
    })()
    return () => { alive = false; c.dispose() }
  }, [])

  // the photo: picture and embedding
  useEffect(() => {
    if (!photo || !ready) return
    let alive = true
    setImg(null); setResult(null); setHistory([]); setThr(0); setIsPreset(false)
    ;(async () => {
      try {
        const im = new Image()
        im.src = BASE + photo.file
        const embP = fetchWithProgress(BASE + photo.emb, (got, total) =>
          alive && setStatus({ text: `Downloading the image embedding of this photo: ${(got / 1e6).toFixed(1)} of ${(total / 1e6 || 2.1).toFixed(1)} MB`, frac: total ? got / total : undefined }))
        await im.decode()
        const buf = await embP
        if (!alive) return
        await client.current!.embed(photo.key, buf)
        if (!alive) return
        setEmbKey(photo.key)
        setImg(im)
        setStatus(null)
        if (photo.click) {
          const p: Prompts = { points: [{ x: photo.click[0], y: photo.click[1], label: 1 }], box: null }
          await run({ prompts: p, mask: null }, photo, true)
          setIsPreset(true)
        }
      } catch (e) {
        if (alive) setError(String(e))
      }
    })()
    return () => { alive = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [photo?.key, ready])

  // ---------------------------------------------------------------- running the decoder
  const run = useCallback(async (entry: Entry, ph: Photo, replace = false) => {
    setBusy(true)
    try {
      const r = await client.current!.decode(entry.prompts, entry.mask, ph.width, ph.height)
      setResult(r)
      const best = 1 + [r.iou[1], r.iou[2], r.iou[3]].indexOf(Math.max(r.iou[1], r.iou[2], r.iou[3]))
      setChosen(best)
      setThr(0)
      setHistory((h) => (replace ? [entry] : [...h, entry]))
      setLoadMs((l) => (l && l.first === undefined ? { ...l, first: performance.now() - t0.current } : l))
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }, [])

  const addPrompt = useCallback((next: Prompts) => {
    if (!photo || busy) return
    const mask = feedback && result ? result.low.slice(chosen * LOW * LOW, (chosen + 1) * LOW * LOW) : null
    setIsPreset(false)
    run({ prompts: next, mask }, photo)
  }, [photo, busy, feedback, result, chosen, run])

  const undo = () => {
    if (!photo || busy || !history.length) return
    setIsPreset(false)
    if (history.length === 1) { setHistory([]); setResult(null); return }
    const prev = history.slice(0, -1)
    const last = prev[prev.length - 1]
    setHistory(prev.slice(0, -1))
    run(last, photo)
  }
  const clear = () => { if (!busy) { setHistory([]); setResult(null); setIsPreset(false) } }

  // the chosen mask at the current threshold
  const current = useMemo(() => {
    if (!result) return null
    const lg = result.up[chosen]
    return { logits: lg, mask: toMask(lg, thr), stats: maskStats(lg, thr) }
  }, [result, chosen, thr])

  const ranked = result ? 1 + [result.iou[1], result.iou[2], result.iou[3]].indexOf(Math.max(result.iou[1], result.iou[2], result.iou[3])) : 0

  // a test hook for the browser check (code/papers/sam/demo/check_browser.mjs)
  useEffect(() => {
    if (!ready || !photos.length) return
    ;(window as any).__samDemo = {
      async check(cases: { photo: string; rounds: { pts?: number[][]; box?: Box; prev?: boolean }[] }[]) {
        const c = client.current!
        const out = []
        for (const cs of cases) {
          const ph = photos.find((p) => p.key === cs.photo)!
          await c.embed(ph.key, await (await fetch(BASE + ph.emb)).arrayBuffer())
          let prev: Float32Array | null = null
          let r: Decoded | null = null
          for (const rd of cs.rounds) {
            const p: Prompts = { points: (rd.pts ?? []).map(([x, y, l]) => ({ x, y, label: l as 0 | 1 })), box: rd.box ?? null }
            r = await c.decode(p, rd.prev ? prev : null, ph.width, ph.height)
            const best = 1 + [r.iou[1], r.iou[2], r.iou[3]].indexOf(Math.max(r.iou[1], r.iou[2], r.iou[3]))
            prev = r.low.slice(best * LOW * LOW, (best + 1) * LOW * LOW)
          }
          out.push({ scores: [r!.iou[1], r!.iou[2], r!.iou[3]], ms: r!.ms,
            masks: [1, 2, 3].map((k) => rle(toMask(r!.up[k], 0))) })
        }
        if (embKey) {
          const ph = photos.find((p) => p.key === embKey)!
          await c.embed(ph.key, await (await fetch(BASE + ph.emb)).arrayBuffer())
        }
        return out
      },
    }
  }, [ready, photos, embKey])

  if (error) return <div className="sam-demo-error" role="alert">The demo could not start: {error}</div>

  const w = photo?.width ?? 640, h = photo?.height ?? 480
  const fr = frameOf(w, h)
  const nTokens = prompts.points.length + (prompts.box ? 2 : 1)

  return (
    <div className="sam-demo">
      <p className="sam-demo-intro">
        This page runs the real SAM ViT-H, the model of the paper, on your clicks. The heavy image encoder ran once for
        each photo on our computer; your browser downloads its output. The prompt encoder and the mask decoder run here,
        in your browser, for every click. Each step below links to the part of the series that explains it.
      </p>

      <Pipeline result={result} nTokens={nTokens} w={w} h={h} rw={fr.rw} rh={fr.rh} busy={busy} />

      <div className="sam-demo-photos" role="radiogroup" aria-label="Photo">
        {photos.map((p) => (
          <button key={p.key} type="button" role="radio" aria-checked={p.key === key}
            className={`sam-demo-thumb${p.key === key ? ' on' : ''}`} onClick={() => !busy && !amgRunning && setKey(p.key)}
            title={`${p.title}: ${p.why}`}>
            <img src={BASE + p.file} alt="" loading="lazy" />
            <span>{p.title}</span>
          </button>
        ))}
      </div>

      <div className="sam-demo-main">
        <div className="sam-demo-stage">
          <div className="sam-demo-toolbar">
            <div className="sam-demo-seg" role="radiogroup" aria-label="Kind of click">
              <button type="button" role="radio" aria-checked={mode === 'fg'} className={mode === 'fg' ? 'on' : ''} onClick={() => setMode('fg')}>
                <i className="sam-demo-dot fg" /> Foreground point
              </button>
              <button type="button" role="radio" aria-checked={mode === 'bg'} className={mode === 'bg' ? 'on' : ''} onClick={() => setMode('bg')}>
                <i className="sam-demo-dot bg" /> Background point
              </button>
            </div>
            <button type="button" className="sam-demo-btn" onClick={undo} disabled={!history.length || busy || amgRunning}>Undo</button>
            <button type="button" className="sam-demo-btn" onClick={clear} disabled={!history.length || busy || amgRunning}>Clear</button>
          </div>
          <PhotoCanvas img={img} w={w} h={h} prompts={prompts} mask={current?.mask ?? null} mode={mode} busy={busy || amgRunning}
            onPoint={(x, y, label) => addPrompt({ points: [...prompts.points, { x, y, label }], box: prompts.box })}
            onBox={(b) => addPrompt({ points: prompts.points, box: b })} />
          <p className="sam-demo-help">
            Click or tap to add a point. Right-click, or press and hold on a phone, for a background point. Drag to draw a
            box (one box at a time). <Link to={`${P2}#what-a-prompt-can-be`}>Part 2: what a prompt can be</Link>
          </p>
          {photo && (
            <p className="sam-demo-credit">
              {photo.why} Photo: <a href={photo.url} rel="noopener">{photo.source}</a>
              {photo.licence && <>, licence <a href={photo.licence_url} rel="noopener license">{photo.licence}</a></>}. {w} × {h} pixels.
            </p>
          )}
          {status && (
            <div className="sam-demo-status" role="status">
              <span>{status.text}</span>
              {status.frac !== undefined && <progress max={1} value={status.frac} />}
            </div>
          )}
        </div>

        <div className="sam-demo-side">
          {isPreset && result && photo && (
            <div className="sam-demo-hint">
              <strong>Why three masks?</strong> We clicked once for you, at ({photo.click![0]}, {photo.click![1]}).
              One point can mean a part, a larger part or the whole object, so SAM returns one mask for each level and
              a score for each. <Link to={`${P2}#figure-3-one-point-three-valid-masks`}>Part 2, Figure 3</Link>
            </div>
          )}
          <MaskCards result={result} img={img} w={w} h={h} chosen={chosen} ranked={ranked}
            onPick={(k) => { setChosen(k); setThr(0) }} />
          <label className="sam-demo-switch">
            <input type="checkbox" checked={feedback} onChange={(e) => setFeedback(e.target.checked)} />
            <span>
              Feed the chosen mask back as a prompt on the next click. Its 256 × 256 logits go into the prompt encoder,
              as in the game of clicks of training. <Link to={`${P3}#training-a-game-of-clicks`}>Part 3</Link>
            </span>
          </label>
          {result && (
            <p className="sam-demo-timing">
              Decoder in this browser: <strong>{result.ms.toFixed(0)} ms</strong> for this prompt
              ({result.tokens} prompt tokens{result.usedMask ? ', plus the previous mask' : ''}), one thread of
              WebAssembly. Upscaling the four masks to {w} × {h}: {result.upMs.toFixed(0)} ms.
            </p>
          )}
        </div>
      </div>

      {result && current && photo && img && (
        <Inside result={result} chosen={chosen} thr={thr} setThr={setThr} current={current} prompts={prompts}
          w={w} h={h} rw={fr.rw} rh={fr.rh} s={fr.s} />
      )}

      {photo && img && (
        <Compare current={current?.mask ?? null} saved={saved} photoKey={photo.key} img={img} w={w} h={h}
          onSave={() => current && setSaved({ key: photo.key, mask: current.mask.slice(), area: current.stats.area,
            label: `${photo.title}, mask ${chosen === 0 ? '0 (single)' : chosen}, ${prompts.points.length} point${prompts.points.length === 1 ? '' : 's'}${prompts.box ? ' and a box' : ''}` })} />
      )}

      {photo && img && ready && embKey === photo.key && (
        <Everything key={photo.key} client={client.current!} img={img} w={w} h={h} disabled={busy} msPerRun={result?.ms} onRunning={setAmgRunning} />
      )}

      <section className="sam-demo-panel sam-demo-notes">
        <h2>How this page was made</h2>
        <p>
          The image encoder of ViT-H has 637 million weights and needs 5.96 trillion floating-point operations (TFLOPs)
          for one photo. In our run of Part 3 on an Apple M5 Pro, it took 2,999 ms on the laptop GPU and 6,779 ms on the
          CPU. The prompt encoder and the mask decoder need 3.62 billion operations, 0.06% of that: 6.5 ms on the same
          GPU. <Link to={`${P3}#efficiency`}>Part 3: efficiency</Link>
        </p>
        <p>
          So we ran the encoder once for each photo, in Python, and saved its 256 × 64 × 64 output as float16 numbers
          (2 MB a photo). Float16 changes the masks very little: in our check, the lowest IoU against the float32
          embedding was 0.997. The prompt encoder and the mask decoder of the same model are one ONNX file of
          {loadMs ? ` ${(loadMs.modelBytes / 1e6).toFixed(1)} MB` : ' 16.5 MB'}, run by onnxruntime-web.
          {loadMs && <> On this visit, the download and start of the decoder took {(loadMs.model / 1000).toFixed(1)} s{loadMs.first !== undefined && <>, and the first masks came {(loadMs.first / 1000).toFixed(1)} s after the page opened</>}.</>}
        </p>
        <p>
          The scripts that made every file of this page are in{' '}
          <a href="https://github.com/ishwar6/ishwar-books/tree/main/code/papers/sam/demo" rel="noopener">code/papers/sam/demo</a>.
          The masks of this page agree with the Python model (transformers' SamModel) on the same prompts.
        </p>
      </section>
    </div>
  )
}

function rle(m: Uint8Array) {
  const runs: number[] = []
  let cur = 0, n = 0
  for (let i = 0; i < m.length; i++) {
    if (m[i] === cur) n++
    else { runs.push(n); cur = m[i]; n = 1 }
  }
  runs.push(n)
  return runs
}

// ------------------------------------------------------------------ the pipeline strip
function Pipeline({ result, nTokens, w, h, rw, rh, busy }: { result: Decoded | null; nTokens: number; w: number; h: number; rw: number; rh: number; busy: boolean }) {
  return (
    <ol className="sam-demo-pipe" aria-label="The pipeline of SAM">
      <li className="sam-demo-step done">
        <span className="sam-demo-step-kicker">1 · once per photo, precomputed</span>
        <strong>Image encoder</strong>
        <span className="sam-demo-shape">{w} × {h} → {rw} × {rh} → padded 1024 × 1024 → 256 × 64 × 64</span>
        <span className="sam-demo-step-note">ViT-H, 637M weights, 5.96 TFLOPs. We ran it in Python; you download its output.</span>
        <Link to={`${P3}#the-image-encoder`}>Part 3: the image encoder</Link>
      </li>
      <li className={`sam-demo-step${busy ? ' active' : result ? ' done' : ''}`}>
        <span className="sam-demo-step-kicker">2 · this click, in your browser</span>
        <strong>Prompt encoder</strong>
        <span className="sam-demo-shape">sparse: 1 × {nTokens} × 256 · dense: 1 × 256 × 64 × 64</span>
        <span className="sam-demo-step-note">
          Each point or box corner becomes 256 numbers. The dense part is {result?.usedMask ? 'the previous mask, after two small convolutions' : 'one learned "no mask" vector in every cell'}.
        </span>
        <Link to={`${P3}#the-prompt-encoder`}>Part 3: the prompt encoder</Link>
      </li>
      <li className={`sam-demo-step${busy ? ' active' : result ? ' done' : ''}`}>
        <span className="sam-demo-step-kicker">3 · this click, in your browser</span>
        <strong>Mask decoder</strong>
        <span className="sam-demo-shape">tokens 1 × {5 + nTokens} × 256 → 4 masks of 256 × 256 + 4 scores → {w} × {h}</span>
        <span className="sam-demo-step-note">
          {result ? `${result.ms.toFixed(0)} ms here. ` : ''}Two two-way attention layers, then a dot product per pixel.
        </span>
        <Link to={`${P3}#the-mask-decoder`}>Part 3: the mask decoder</Link>
      </li>
    </ol>
  )
}

// ------------------------------------------------------------------ the photo with prompts and the chosen mask
function PhotoCanvas({ img, w, h, prompts, mask, mode, busy, onPoint, onBox }: {
  img: HTMLImageElement | null; w: number; h: number; prompts: Prompts; mask: Uint8Array | null; mode: 'fg' | 'bg'; busy: boolean
  onPoint: (x: number, y: number, label: 0 | 1) => void; onBox: (b: Box) => void
}) {
  const ref = useRef<HTMLCanvasElement>(null)
  const drag = useRef<{ x: number; y: number; px: number; py: number; moved: boolean; timer: number; used: boolean } | null>(null)
  const [rubber, setRubber] = useState<Box | null>(null)

  useEffect(() => {
    const cv = ref.current
    if (!cv) return
    const ctx = cv.getContext('2d', { willReadFrequently: true })!
    ctx.clearRect(0, 0, w, h)
    if (img) ctx.drawImage(img, 0, 0, w, h)
    if (mask && img) drawMask(ctx, mask, w, h, MASK_RGB, 0.45)
    const scale = Math.max(1, w / (cv.clientWidth || w))
    const b = rubber ?? prompts.box
    if (b) {
      ctx.save()
      ctx.lineWidth = 2.5 * scale
      ctx.strokeStyle = '#ffffff'
      ctx.setLineDash([8 * scale, 5 * scale])
      ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1])
      ctx.restore()
    }
    for (const p of prompts.points) drawPoint(ctx, p.x, p.y, p.label, 7 * scale)
  }, [img, w, h, prompts, mask, rubber])

  const toPhoto = (e: React.PointerEvent) => {
    const r = ref.current!.getBoundingClientRect()
    return { x: Math.min(w - 1, Math.max(0, ((e.clientX - r.left) / r.width) * w)), y: Math.min(h - 1, Math.max(0, ((e.clientY - r.top) / r.height) * h)) }
  }

  const down = (e: React.PointerEvent) => {
    if (busy || !img || e.button === 2) return
    const p = toPhoto(e)
    ref.current!.setPointerCapture(e.pointerId)
    const d = { ...p, px: e.clientX, py: e.clientY, moved: false, used: false, timer: 0 }
    if (e.pointerType !== 'mouse') {
      d.timer = window.setTimeout(() => {      // press and hold: a background point
        if (drag.current && !drag.current.moved) { drag.current.used = true; onPoint(Math.round(d.x), Math.round(d.y), 0) }
      }, 550)
    }
    drag.current = d
  }
  const move = (e: React.PointerEvent) => {
    const d = drag.current
    if (!d) return
    if (Math.hypot(e.clientX - d.px, e.clientY - d.py) > 8) { d.moved = true; clearTimeout(d.timer) }
    if (d.moved) {
      const p = toPhoto(e)
      setRubber([Math.min(d.x, p.x), Math.min(d.y, p.y), Math.max(d.x, p.x), Math.max(d.y, p.y)].map(Math.round) as Box)
    }
  }
  const up = (e: React.PointerEvent) => {
    const d = drag.current
    drag.current = null
    if (!d) return
    clearTimeout(d.timer)
    if (d.used) return
    if (d.moved) {
      const p = toPhoto(e)
      const b = [Math.min(d.x, p.x), Math.min(d.y, p.y), Math.max(d.x, p.x), Math.max(d.y, p.y)].map(Math.round) as Box
      setRubber(null)
      if (b[2] - b[0] >= 4 && b[3] - b[1] >= 4) onBox(b)
    } else {
      onPoint(Math.round(d.x), Math.round(d.y), mode === 'fg' ? 1 : 0)
    }
  }
  const context = (e: React.MouseEvent) => {
    e.preventDefault()
    if (busy || !img) return
    const r = ref.current!.getBoundingClientRect()
    onPoint(Math.round(((e.clientX - r.left) / r.width) * w), Math.round(((e.clientY - r.top) / r.height) * h), 0)
  }

  return (
    <div className={`sam-demo-canvaswrap${busy ? ' busy' : ''}`} style={{ aspectRatio: `${w} / ${h}` }}>
      <canvas ref={ref} width={w} height={h} className="sam-demo-canvas" aria-label="The photo. Click to add a point, drag to draw a box."
        onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={() => { if (drag.current) clearTimeout(drag.current.timer); drag.current = null; setRubber(null) }}
        onContextMenu={context} />
      {busy && <span className="sam-demo-spinner" aria-hidden="true" />}
    </div>
  )
}

function drawPoint(ctx: CanvasRenderingContext2D, x: number, y: number, label: number, r: number) {
  ctx.save()
  ctx.beginPath()
  ctx.arc(x, y, r, 0, Math.PI * 2)
  ctx.fillStyle = label ? FG : BG
  ctx.fill()
  ctx.lineWidth = r * 0.35
  ctx.strokeStyle = '#ffffff'
  ctx.stroke()
  ctx.beginPath()
  ctx.moveTo(x - r * 0.5, y); ctx.lineTo(x + r * 0.5, y)
  if (label) { ctx.moveTo(x, y - r * 0.5); ctx.lineTo(x, y + r * 0.5) }
  ctx.lineWidth = r * 0.3
  ctx.stroke()
  ctx.restore()
}

/** Paint a 0/1 mask of w x h over the canvas, with a stronger edge. */
function drawMask(ctx: CanvasRenderingContext2D, m: Uint8Array, w: number, h: number, rgb: number[], alpha: number) {
  const id = ctx.getImageData(0, 0, w, h)
  const d = id.data
  const a = alpha, ae = 0.95
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x
      if (!m[i]) continue
      const edge = x === 0 || y === 0 || x === w - 1 || y === h - 1 || !m[i - 1] || !m[i + 1] || !m[i - w] || !m[i + w]
      const k = edge ? ae : a
      const j = i * 4
      d[j] = d[j] * (1 - k) + (edge ? 255 : rgb[0]) * k
      d[j + 1] = d[j + 1] * (1 - k) + (edge ? 255 : rgb[1]) * k
      d[j + 2] = d[j + 2] * (1 - k) + (edge ? 255 : rgb[2]) * k
    }
  }
  ctx.putImageData(id, 0, 0)
}

/** A small copy of the photo with one mask (logits > 0) over it, sampled to the canvas size. */
function Thumb({ img, logits, w, h, rgb = MASK_RGB, width = 240 }: { img: HTMLImageElement | null; logits: Float32Array | null; w: number; h: number; rgb?: number[]; width?: number }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const tw = Math.min(width, w), th = Math.round((tw * h) / w)
  useEffect(() => {
    const cv = ref.current
    if (!cv || !img) return
    const ctx = cv.getContext('2d', { willReadFrequently: true })!
    ctx.drawImage(img, 0, 0, tw, th)
    if (!logits) return
    const id = ctx.getImageData(0, 0, tw, th)
    const d = id.data
    for (let y = 0; y < th; y++) {
      const sy = Math.min(h - 1, Math.floor(((y + 0.5) * h) / th)) * w
      for (let x = 0; x < tw; x++) {
        const v = logits[sy + Math.min(w - 1, Math.floor(((x + 0.5) * w) / tw))]
        const j = (y * tw + x) * 4
        if (v > 0) { d[j] = d[j] * 0.4 + rgb[0] * 0.6; d[j + 1] = d[j + 1] * 0.4 + rgb[1] * 0.6; d[j + 2] = d[j + 2] * 0.4 + rgb[2] * 0.6 }
        else { d[j] *= 0.55; d[j + 1] *= 0.55; d[j + 2] *= 0.55 }
      }
    }
    ctx.putImageData(id, 0, 0)
  }, [img, logits, w, h, tw, th, rgb])
  return <canvas ref={ref} width={tw} height={th} className="sam-demo-thumbcanvas" />
}

// ------------------------------------------------------------------ the three masks and the single-mask output
function MaskCards({ result, img, w, h, chosen, ranked, onPick }: {
  result: Decoded | null; img: HTMLImageElement | null; w: number; h: number; chosen: number; ranked: number; onPick: (k: number) => void
}) {
  const areas = useMemo(() => result?.up.map((u) => maskStats(u, 0).area) ?? [], [result])
  return (
    <section className="sam-demo-masks" aria-label="The masks of the decoder">
      <h2>Three masks and their scores</h2>
      <p className="sam-demo-sub">
        The number is SAM's predicted IoU: its own guess of how good the mask is. SAM ranks the mask with the highest
        score first. Pick a mask to show it on the photo. <Link to={`${P3}#resolving-ambiguity`}>Part 3: resolving ambiguity</Link>
      </p>
      <div className="sam-demo-cards">
        {[1, 2, 3].map((k) => (
          <button key={k} type="button" className={`sam-demo-card${chosen === k ? ' on' : ''}`} onClick={() => onPick(k)}
            disabled={!result} aria-pressed={chosen === k}>
            <Thumb img={img} logits={result ? result.up[k] : null} w={w} h={h} />
            <span className="sam-demo-card-row">
              <strong>Mask {k}</strong>
              {result && ranked === k && <em className="sam-demo-badge">SAM's first</em>}
            </span>
            <span className="sam-demo-card-row">
              <span>score {result ? result.iou[k].toFixed(3) : '…'}</span>
              <span>{result ? `${fmt(areas[k])} px` : ''}</span>
            </span>
          </button>
        ))}
      </div>
      <button type="button" className={`sam-demo-card sam-demo-card0${chosen === 0 ? ' on' : ''}`} onClick={() => onPick(0)}
        disabled={!result} aria-pressed={chosen === 0}>
        <Thumb img={img} logits={result ? result.up[0] : null} w={w} h={h} width={120} />
        <span>
          <strong>Mask 0, the single-mask output</strong> · score {result ? result.iou[0].toFixed(3) : '…'}
          {result && ` · ${fmt(areas[0])} px`}
          <br />
          <span className="sam-demo-muted">A fourth output token. SAM uses it when several points remove the ambiguity. <Link to={`${P3}#resolving-ambiguity`}>Part 3</Link></span>
        </span>
      </button>
    </section>
  )
}

// ------------------------------------------------------------------ inside the chosen mask: logits, threshold, stability
function Inside({ result, chosen, thr, setThr, current, prompts, w, h, rw, rh, s }: {
  result: Decoded; chosen: number; thr: number; setThr: (t: number) => void
  current: { logits: Float32Array; mask: Uint8Array; stats: ReturnType<typeof maskStats> }; prompts: Prompts
  w: number; h: number; rw: number; rh: number; s: number
}) {
  const heat = useRef<HTMLCanvasElement>(null)
  const bin = useRef<HTMLCanvasElement>(null)
  const low = result.low.subarray(chosen * LOW * LOW, (chosen + 1) * LOW * LOW)
  const range = useMemo(() => {
    let mn = Infinity, mx = -Infinity
    for (let i = 0; i < low.length; i++) { if (low[i] < mn) mn = low[i]; if (low[i] > mx) mx = low[i] }
    return [mn, mx]
  }, [low])

  useEffect(() => {
    const cv = heat.current
    if (!cv) return
    const ctx = cv.getContext('2d', { willReadFrequently: true })!
    const id = ctx.createImageData(LOW, LOW)
    const d = id.data
    for (let i = 0; i < LOW * LOW; i++) {
      const v = low[i]
      const c = heatColour(v)
      const above = v > thr
      // the cut-off as a thin white line: a pixel above it with a neighbour below it
      const x = i % LOW, y = (i / LOW) | 0
      const edge = above && ((x > 0 && !(low[i - 1] > thr)) || (x < LOW - 1 && !(low[i + 1] > thr)) || (y > 0 && !(low[i - LOW] > thr)) || (y < LOW - 1 && !(low[i + LOW] > thr)))
      d[i * 4] = edge ? 255 : c[0]; d[i * 4 + 1] = edge ? 255 : c[1]; d[i * 4 + 2] = edge ? 255 : c[2]; d[i * 4 + 3] = 255
    }
    ctx.putImageData(id, 0, 0)
    // the part of the 256 grid that the photo covers; the rest is the padding
    ctx.save()
    ctx.setLineDash([4, 3])
    ctx.strokeStyle = 'rgba(255,255,255,0.8)'
    ctx.lineWidth = 1
    ctx.strokeRect(0.5, 0.5, rw / 4 - 1, rh / 4 - 1)
    ctx.restore()
    for (const p of prompts.points) drawPoint(ctx, (p.x * s) / 4, (p.y * s) / 4, p.label, 3.2)
    if (prompts.box) {
      ctx.strokeStyle = '#ffffff'
      ctx.setLineDash([3, 2])
      const b = prompts.box.map((v) => (v * s) / 4)
      ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1])
    }
  }, [low, thr, prompts, rw, rh, s])

  useEffect(() => {
    const cv = bin.current
    if (!cv) return
    const ctx = cv.getContext('2d', { willReadFrequently: true })!
    const id = ctx.createImageData(w, h)
    const d = id.data
    const lg = current.logits
    for (let i = 0; i < w * h; i++) {
      const v = lg[i]
      // inside: white; the unsure band between t - 1 and t + 1: orange (inside) or grey (outside)
      const band = v > thr - 1 && v <= thr + 1
      const inside = v > thr
      const c = inside ? (band ? [255, 170, 60] : [245, 245, 245]) : band ? [120, 120, 130] : [24, 24, 28]
      d[i * 4] = c[0]; d[i * 4 + 1] = c[1]; d[i * 4 + 2] = c[2]; d[i * 4 + 3] = 255
    }
    ctx.putImageData(id, 0, 0)
  }, [current, thr, w, h])

  const st = current.stats
  return (
    <section className="sam-demo-panel">
      <h2>Inside mask {chosen}: logits and a cut-off</h2>
      <p className="sam-demo-sub">
        The decoder does not return a yes or no for each pixel. It returns a logit: a number above 0 means "inside", below
        0 means "outside", and far from 0 means "sure". The mask is the set of pixels above the cut-off, 0 by default.{' '}
        <Link to={`${P2}#pre-training-a-game-of-prompts`}>Part 2: logits</Link>
      </p>
      <div className="sam-demo-inside">
        <figure>
          <canvas ref={heat} width={LOW} height={LOW} className="sam-demo-heat" aria-label="Heatmap of the 256 by 256 logits" />
          <figcaption>
            The decoder's output: 256 × 256 logits, from {range[0].toFixed(1)} (blue) to +{range[1].toFixed(1)} (yellow).
            The white line is the cut-off. The dashed frame is the photo: the grid covers the padded 1024 × 1024 square,
            so {rh < 1024 ? `the rows below ${Math.round(rh / 4)} are` : rw < 1024 ? `the columns right of ${Math.round(rw / 4)} are` : 'nothing is'} padding.{' '}
            <Link to={`${P3}#from-tokens-to-a-mask`}>Part 3: from tokens to a mask</Link>
          </figcaption>
        </figure>
        <figure>
          <canvas ref={bin} width={w} height={h} className="sam-demo-bin" aria-label="The upscaled mask" />
          <figcaption>
            The same logits upscaled to the photo ({LOW} → 1024, cut to {rw} × {rh}, then → {w} × {h}). White: above the
            cut-off. Orange and grey: the unsure pixels, less than 1 away from it.
          </figcaption>
        </figure>
      </div>
      <div className="sam-demo-slider">
        <label htmlFor="sam-demo-thr">Cut-off on the logits: <strong>{thr > 0 ? '+' : ''}{thr.toFixed(2)}</strong></label>
        <input id="sam-demo-thr" type="range" min={-8} max={8} step={0.25} value={thr} onChange={(e) => setThr(Number(e.target.value))} />
        <button type="button" className="sam-demo-btn" onClick={() => setThr(0)} disabled={thr === 0}>Back to 0</button>
      </div>
      <div className="sam-demo-stats">
        <div><span>Mask area</span><strong>{fmt(st.area)} px</strong><em>{pct(st.area, w * h)} of the photo</em></div>
        <div><span>Pixels above {fmtT(thr + 1)}</span><strong>{fmt(st.hi)}</strong></div>
        <div><span>Pixels above {fmtT(thr - 1)}</span><strong>{fmt(st.lo)}</strong></div>
        <div className={st.stability >= 0.95 ? 'ok' : 'bad'}>
          <span>Stability score</span><strong>{st.stability.toFixed(3)}</strong>
          <em>{fmt(st.hi)} / {fmt(st.lo)}; {st.stability >= 0.95 ? 'passes' : 'fails'} the 0.95 filter</em>
        </div>
      </div>
      <p className="sam-demo-sub">
        The stability score is the IoU of two masks: the pixels above the cut-off + 1 and the pixels above the cut-off − 1.
        The first mask sits inside the second, so the IoU is the ratio of their sizes. Near 1 means that few pixels are
        unsure. The data engine kept only masks with a score of at least 0.95.{' '}
        <Link to={`${P4}#filter-2-is-the-mask-stable`}>Part 4: is the mask stable?</Link>
      </p>
    </section>
  )
}

const fmtT = (t: number) => (t > 0 ? `+${t.toFixed(2)}` : t.toFixed(2))

/** Blue for negative logits, dark at 0, yellow for positive ones; tanh keeps far values from washing out. */
function heatColour(v: number): [number, number, number] {
  const a = Math.tanh(v / 10)
  if (a >= 0) return [Math.round(40 + 215 * a), Math.round(30 + 185 * a), Math.round(40 - 10 * a)]
  const b = -a
  return [Math.round(40 - 20 * b), Math.round(30 + 80 * b), Math.round(40 + 200 * b)]
}

// ------------------------------------------------------------------ compare with a saved mask
function Compare({ current, saved, photoKey, img, w, h, onSave }: {
  current: Uint8Array | null; saved: Saved | null; photoKey: string; img: HTMLImageElement; w: number; h: number; onSave: () => void
}) {
  const ref = useRef<HTMLCanvasElement>(null)
  const same = saved && saved.key === photoKey && current
  const r = useMemo(() => (same ? maskIou(current!, saved!.mask) : null), [same, current, saved])
  useEffect(() => {
    const cv = ref.current
    if (!cv || !same) return
    const ctx = cv.getContext('2d', { willReadFrequently: true })!
    ctx.drawImage(img, 0, 0, w, h)
    const id = ctx.getImageData(0, 0, w, h)
    const d = id.data
    for (let i = 0; i < w * h; i++) {
      const a = current![i], b = saved!.mask[i]
      if (!a && !b) { d[i * 4] *= 0.45; d[i * 4 + 1] *= 0.45; d[i * 4 + 2] *= 0.45; continue }
      const c = a && b ? [70, 220, 140] : a ? MASK_RGB : SAVED_RGB
      d[i * 4] = d[i * 4] * 0.35 + c[0] * 0.65; d[i * 4 + 1] = d[i * 4 + 1] * 0.35 + c[1] * 0.65; d[i * 4 + 2] = d[i * 4 + 2] * 0.35 + c[2] * 0.65
    }
    ctx.putImageData(id, 0, 0)
  }, [same, current, saved, img, w, h])
  return (
    <section className="sam-demo-panel">
      <h2>Compare two masks with IoU</h2>
      <p className="sam-demo-sub">
        IoU (intersection over union) tells how much two masks overlap. Count the pixels in both masks, and divide by the
        pixels in at least one mask. 1 means the same mask; 0 means no pixel in common.{' '}
        <Link to={`${P1}#let-us-run-it`}>Part 1: IoU</Link>
      </p>
      <div className="sam-demo-compare">
        <div>
          <button type="button" className="sam-demo-btn primary" onClick={onSave} disabled={!current}>Save the current mask</button>
          {saved ? (
            <p className="sam-demo-muted">Saved: {saved.label} ({fmt(saved.area)} px).{saved.key !== photoKey && ' It is on another photo: go back to that photo to compare.'}</p>
          ) : (
            <p className="sam-demo-muted">Save a mask, then change the prompts or pick another mask.</p>
          )}
          {r && (
            <div className="sam-demo-iou">
              <div><span>In both</span><strong>{fmt(r.inter)} px</strong></div>
              <div><span>In at least one</span><strong>{fmt(r.union)} px</strong></div>
              <div className="big"><span>IoU</span><strong>{fmt(r.inter)} / {fmt(r.union)} = {r.iou.toFixed(3)}</strong></div>
              <p className="sam-demo-legend">
                <i style={{ background: 'rgb(70,220,140)' }} /> both <i style={{ background: `rgb(${MASK_RGB})` }} /> current only{' '}
                <i style={{ background: `rgb(${SAVED_RGB})` }} /> saved only
              </p>
            </div>
          )}
        </div>
        {same && <canvas ref={ref} width={w} height={h} className="sam-demo-comparecanvas" aria-label="The two masks over the photo" />}
      </div>
    </section>
  )
}

// ------------------------------------------------------------------ everything: the automatic mask generator
const AMG = { iouT: 0.88, stabT: 0.95, bigT: 0.95, nmsT: 0.7 }

function Everything({ client, img, w, h, disabled, msPerRun, onRunning }: { client: SamClient; img: HTMLImageElement; w: number; h: number; disabled: boolean; msPerRun?: number; onRunning: (r: boolean) => void }) {
  const [n, setN] = useState(16)
  const [state, setState] = useState<'idle' | 'running' | 'done'>('idle')
  const [prog, setProg] = useState<{ done: number; total: number; ms: number; counts: AmgCounts } | null>(null)
  const [res, setRes] = useState<{ counts: AmgCounts; masks: AmgMask[]; ms: number; n: number } | null>(null)
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => () => { client.cancelAmg(); onRunning(false) }, [client, onRunning])

  const start = async () => {
    setState('running'); setRes(null); setProg(null); onRunning(true)
    try {
      const r = await client.amg({ n, w, h, ...AMG }, (p) => setProg(p))
      if (r) { setRes({ counts: r.counts, masks: r.masks, ms: r.ms, n }); setState('done') } else setState('idle')
    } finally {
      onRunning(false)
    }
  }

  useEffect(() => {
    const cv = ref.current
    if (!cv || !res) return
    const ctx = cv.getContext('2d', { willReadFrequently: true })!
    ctx.drawImage(img, 0, 0, w, h)
    const id = ctx.getImageData(0, 0, w, h)
    const d = id.data
    res.masks.forEach((m, k) => {
      const c = hsl((k * 137.508) % 360)
      const bits = unpack(m.bits, w * h)
      for (let i = 0; i < w * h; i++) if (bits[i]) { d[i * 4] = d[i * 4] * 0.4 + c[0] * 0.6; d[i * 4 + 1] = d[i * 4 + 1] * 0.4 + c[1] * 0.6; d[i * 4 + 2] = d[i * 4 + 2] * 0.4 + c[2] * 0.6 }
    })
    ctx.putImageData(id, 0, 0)
    ctx.fillStyle = '#ffffff'
    for (let k = 0; k < res.n * res.n; k++) {
      ctx.beginPath()
      ctx.arc(((k % res.n) + 0.5) / res.n * w, (Math.floor(k / res.n) + 0.5) / res.n * h, Math.max(1.5, w / 400), 0, Math.PI * 2)
      ctx.fill()
    }
  }, [res, img, w, h])

  const c = res?.counts ?? prog?.counts
  const perPoint = prog && prog.done ? prog.ms / prog.done : 0
  return (
    <section className="sam-demo-panel">
      <h2>Everything: the automatic stage of the data engine</h2>
      <p className="sam-demo-sub">
        To make SA-1B with no people, SAM was prompted with a grid of points. Each point gives three masks. Three filters
        and NMS then keep the good and different ones. Here it runs in your browser on the full photo, with the paper's
        filters: predicted IoU above 0.88, stability of at least 0.95, less than 95% of the photo, and box NMS at 0.7. It
        skips the zoomed-in crops and the clean-up of small pieces. <Link to={`${P4}#the-whole-funnel`}>Part 4: the whole funnel</Link>
      </p>
      <div className="sam-demo-toolbar">
        <div className="sam-demo-seg" role="radiogroup" aria-label="Grid size">
          {[8, 16].map((k) => (
            <button key={k} type="button" role="radio" aria-checked={n === k} className={n === k ? 'on' : ''}
              onClick={() => setN(k)} disabled={state === 'running'}>{k} × {k} points</button>
          ))}
        </div>
        {state !== 'running' ? (
          <button type="button" className="sam-demo-btn primary" onClick={start} disabled={disabled}>Run on this photo</button>
        ) : (
          <button type="button" className="sam-demo-btn" onClick={() => client.cancelAmg()}>Stop</button>
        )}
        <span className="sam-demo-muted">
          {n * n} decoder runs
          {state === 'running' && perPoint ? `, about ${Math.ceil((perPoint * (n * n - (prog?.done ?? 0))) / 1000)} s left` : ''}
          {state !== 'running' && msPerRun ? `, about ${Math.ceil((msPerRun * n * n) / 1000)} s at ${msPerRun.toFixed(0)} ms each` : ''}
          . The paper used 32 × 32.
        </span>
      </div>
      {state === 'running' && prog && <progress className="sam-demo-progress" max={prog.total} value={prog.done} />}
      {c && (
        <table className="sam-demo-funnel">
          <tbody>
            <tr><td>grid points</td><td>{fmt(c.points)}</td><td /></tr>
            <tr><td>candidate masks (3 per point)</td><td>{fmt(c.candidates)}</td><td /></tr>
            <tr><td>predicted IoU &gt; 0.88</td><td>{fmt(c.passIou)}</td><td>removed {fmt(c.candidates - c.passIou)}</td></tr>
            <tr><td>and stability ≥ 0.95</td><td>{fmt(c.passStab)}</td><td>removed {fmt(c.passIou - c.passStab)}</td></tr>
            <tr><td>and covers &lt; 95% of the photo</td><td>{fmt(c.passBig)}</td><td>removed {fmt(c.passStab - c.passBig)}</td></tr>
            <tr><td>box NMS at 0.7 (duplicates)</td><td>{res ? fmt(c.final) : '…'}</td><td>{res ? `removed ${fmt(c.passBig - c.final)}` : ''}</td></tr>
          </tbody>
        </table>
      )}
      {res && (
        <>
          <canvas ref={ref} width={w} height={h} className="sam-demo-amgcanvas" aria-label="The automatic masks over the photo" />
          <p className="sam-demo-muted">
            {res.counts.final} masks, one colour each; the white dots are the grid. {fmt(res.n * res.n)} decoder runs took
            {' '}{(res.ms / 1000).toFixed(1)} s here ({(res.ms / (res.n * res.n)).toFixed(0)} ms each).
          </p>
        </>
      )}
    </section>
  )
}

function hsl(hue: number): [number, number, number] {
  const f = (n: number) => {
    const k = (n + hue / 30) % 12
    return Math.round(255 * (0.55 - 0.45 * Math.max(-1, Math.min(k - 3, 9 - k, 1))))
  }
  return [f(0), f(8), f(4)]
}
