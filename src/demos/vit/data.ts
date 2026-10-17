/* Loading for the ViT demo: the saved ViT-B/16 runs of the curated pictures, the live model (DeiT-Ti/16) in a
   worker, and one shape for both kinds of result so every panel reads the same fields. */
import { f16ToF32, gunzip, toPlanes, shuffleTiles, randomPerm, top5, rollout, meanDistance, T, N, type Top } from './vitMath'

export const BASE = import.meta.env.BASE_URL + 'demos/vit/'
export const LIVE_FILE = 'model/deit-ti16-fp16.onnx'
export const LIVE_BYTES = 11572760              // size of LIVE_FILE, for the progress bar if the server sends no length

export type ModelKey = 'vit-b16' | 'deit-ti16'
export type Model = {
  key: ModelKey; name: string; D: number; H: number; L: number; params: number
  labels: string[]; cls: Float32Array; pos: Float32Array; possim: Float32Array; attLevels: number
}
export type Picture = { slug: string; name: string; coco: number; licence: string; licence_url: string; flickr: string }
export type Experiment = { perm: number[]; shuffled: Top; nopos: Top; noposShuffled: Top; noposShuffleDiff: number | null }
export type Result = {
  model: Model
  live: boolean
  patch: Float32Array              // (196, D) patch embeddings x_p E
  tokens: Float32Array             // (197, D) z_0
  clsAtt: Float32Array             // (L, H, 197) the [class] row of every attention map
  att: Float32Array | null         // (L, H, 197, 197), null until loaded (saved runs load it on demand)
  distance: number[][]             // (L, H) mean attention distance in pixels
  rollout: ArrayLike<number>       // (196,)
  top5: Top
  logits: Float32Array | null
  exp: Experiment | null
  ms: number | null                // model time in the browser
}

const cache = new Map<string, Promise<unknown>>()
function once<T>(key: string, f: () => Promise<T>): Promise<T> {
  if (!cache.has(key)) cache.set(key, f().catch((e) => { cache.delete(key); throw e }))
  return cache.get(key) as Promise<T>
}
async function get(path: string): Promise<Response> {
  const r = await fetch(BASE + path)
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`)
  return r
}
const json = <T,>(p: string) => get(p).then((r) => r.json() as Promise<T>)
const f16 = (p: string) => get(p).then((r) => r.arrayBuffer()).then((b) => f16ToF32(new Uint16Array(b)))

export const loadPictures = () => once('pictures', () => json<Picture[]>('pictures.json'))

export function loadModel(key: ModelKey): Promise<Model> {
  return once(`model:${key}`, async () => {
    const [m, pos, possim] = await Promise.all([
      json<{ model: string; D: number; heads: number; layers: number; params: number; labels: string[]; cls: number[]; att_levels: number }>(`${key}/meta.json`),
      f16(`${key}/pos.bin`),
      f16(`${key}/possim.bin`),
    ])
    return { key, name: m.model, D: m.D, H: m.heads, L: m.layers, params: m.params, labels: m.labels.map((s) => s.split(',')[0]),
             cls: new Float32Array(m.cls), pos, possim, attLevels: m.att_levels }
  })
}

/** The 224 x 224 RGBA pixels of a picture file (the PNG is already 224 x 224: the browser reads it unchanged). */
export async function pixelsOf(src: string | Blob): Promise<Uint8ClampedArray> {
  const blob = typeof src === 'string' ? await get(src).then((r) => r.blob()) : src
  const bmp = await createImageBitmap(blob)
  const c = document.createElement('canvas')
  c.width = c.height = 224
  const ctx = c.getContext('2d', { willReadFrequently: true })!
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(bmp, 0, 0, 224, 224)               // squeezed to a square, like the model's preprocessing
  bmp.close()
  return ctx.getImageData(0, 0, 224, 224).data
}

/** z_0 = [x_class; x_p E] + E_pos (Eq. 1), from the saved patch embeddings and the model's weights. */
function tokensOf(m: Model, patch: Float32Array) {
  const z = new Float32Array(T * m.D)
  z.set(m.cls, 0)
  z.set(patch, m.D)
  for (let i = 0; i < z.length; i++) z[i] += m.pos[i]
  return z
}

type Saved = { top5: [number, string, number][]; shuffled: Top; nopos: Top; nopos_shuffled: Top; shuffle: number[]
  distance: number[][]; rollout: number[]; nopos_shuffle_max_abs_diff: number }

/** A curated picture's saved ViT-B/16 run. The full attention maps (about 2 MB) come later, from fullAttention(). */
export function loadSaved(slug: string): Promise<Result> {
  return once(`saved:${slug}`, async () => {
    const m = await loadModel('vit-b16')
    const [s, bin] = await Promise.all([json<Saved>(`vit-b16/${slug}.json`), f16(`vit-b16/${slug}.bin`)])
    const patch = bin.slice(0, N * m.D)
    const name = (t: Top) => t.map(([i, , p]) => [i, m.labels[i], p] as [number, string, number])
    return {
      model: m, live: false, patch, tokens: tokensOf(m, patch), clsAtt: bin.slice(N * m.D), att: null,
      distance: s.distance, rollout: s.rollout, top5: name(s.top5), logits: null, ms: null,
      exp: { perm: s.shuffle, shuffled: name(s.shuffled), nopos: name(s.nopos), noposShuffled: name(s.nopos_shuffled), noposShuffleDiff: s.nopos_shuffle_max_abs_diff },
    }
  })
}

export function fullAttention(slug: string): Promise<Float32Array> {
  return once(`att:${slug}`, async () => {
    const m = await loadModel('vit-b16')
    const got = await get(`vit-b16/${slug}.att.bin`).then((r) => r.arrayBuffer())
    const head = new Uint8Array(got, 0, 2)
    const raw = head[0] === 0x1f && head[1] === 0x8b ? await gunzip(got) : got     // a server may already have unzipped it
    const rows = m.L * m.H * T
    const cw = f16ToF32(new Uint16Array(raw, 0, rows))                  // weight on [class]
    const mx = f16ToF32(new Uint16Array(raw, rows * 2, rows))           // largest weight on a patch
    const q = new Uint8Array(raw, rows * 4)                             // (rows, 196) patch weights in 63 levels
    const out = new Float32Array(rows * T)
    for (let r = 0; r < rows; r++) {
      const s = mx[r] / m.attLevels
      out[r * T] = cw[r]
      for (let j = 0; j < N; j++) out[r * T + 1 + j] = q[r * N + j] * s
    }
    return out
  })
}

/* ---------------- the live model ---------------- */

type Out = { patch: Float32Array; tokens: Float32Array; att: Float32Array; logits: Float32Array; ms: number }
export type Timing = { download: number; session: number; bytes: number }

class Live {
  worker: Worker | null = null
  ready: Promise<Timing> | null = null
  next = 1
  wait = new Map<number | string, { ok: (v: any) => void; fail: (e: Error) => void }>()

  start(onProgress: (done: number, total: number) => void): Promise<Timing> {
    if (this.ready) return this.ready
    this.ready = (async () => {
      const t0 = performance.now()
      const r = await get(LIVE_FILE)
      const total = Number(r.headers.get('content-length')) || LIVE_BYTES
      const reader = r.body!.getReader()
      const parts: Uint8Array[] = []
      let done = 0
      for (;;) {
        const { value, done: end } = await reader.read()
        if (end) break
        parts.push(value)
        done += value.length
        onProgress(done, total)
      }
      const buf = new Uint8Array(done)
      let o = 0
      for (const p of parts) { buf.set(p, o); o += p.length }
      const download = performance.now() - t0
      this.worker = new Worker(new URL('./vitWorker.ts', import.meta.url), { type: 'module' })
      this.worker.onmessage = (e) => {
        const m = e.data
        const key = m.type === 'ready' ? 'init' : m.id
        const w = this.wait.get(key)
        if (!w) return
        this.wait.delete(key)
        if (m.type === 'error') w.fail(new Error(m.message))
        else w.ok(m)
      }
      this.worker.onerror = (e) => {
        for (const w of this.wait.values()) w.fail(new Error(e.message || 'worker error'))
        this.wait.clear()
      }
      const t1 = performance.now()
      await this.ask('init', { type: 'init', model: buf.buffer }, [buf.buffer])
      return { download, session: performance.now() - t1, bytes: done }
    })()
    this.ready.catch(() => { this.ready = null })
    return this.ready
  }

  private ask<R>(key: number | string, msg: unknown, transfer: Transferable[] = []) {
    return new Promise<R>((ok, fail) => {
      this.wait.set(key, { ok, fail })
      this.worker!.postMessage(msg, transfer)
    })
  }

  run(rgba: Uint8ClampedArray, pos: number): Promise<Out> {
    const id = this.next++
    const pixels = toPlanes(rgba)
    return this.ask<Out>(id, { type: 'run', id, pixels, pos }, [pixels.buffer])
  }
}
export const live = new Live()

/** Run the live model on a picture and compute what the panels need, in the browser. */
export async function runLive(rgba: Uint8ClampedArray): Promise<Result> {
  const m = await loadModel('deit-ti16')
  const o = await live.run(rgba, 1)
  const clsAtt = new Float32Array(m.L * m.H * T)
  for (let k = 0; k < m.L * m.H; k++) clsAtt.set(o.att.subarray(k * T * T, k * T * T + T), k * T)
  return {
    model: m, live: true, patch: o.patch, tokens: o.tokens, clsAtt, att: o.att,
    distance: meanDistance(o.att, m.L, m.H), rollout: rollout(o.att, m.L, m.H),
    top5: top5(o.logits, m.labels), logits: o.logits, exp: null, ms: o.ms,
  }
}

/** The experiment with the live model: a new random shuffle, with and without the position embeddings. */
export async function runExperiment(rgba: Uint8ClampedArray, labels: string[]): Promise<Experiment> {
  const perm = randomPerm(N)
  const sh = shuffleTiles(rgba, perm)
  const a = await live.run(sh, 1)
  const b = await live.run(rgba, 0)
  const c = await live.run(sh, 0)
  let d = 0
  for (let i = 0; i < b.logits.length; i++) d = Math.max(d, Math.abs(b.logits[i] - c.logits[i]))
  return { perm, shuffled: top5(a.logits, labels), nopos: top5(b.logits, labels), noposShuffled: top5(c.logits, labels), noposShuffleDiff: d }
}
