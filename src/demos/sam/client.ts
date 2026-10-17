/* The page side of the SAM worker: promises for each request, progress callbacks for the automatic generator. */
import { upscale, LOW } from './samMath'

export type Point = { x: number; y: number; label: 0 | 1 }
export type Box = [number, number, number, number]
export type Prompts = { points: Point[]; box: Box | null }
export type Decoded = {
  low: Float32Array          // (4, 256, 256): token 0 = single-mask output, 1..3 = the three masks
  iou: Float32Array          // (4,) predicted IoU of each
  up: Float32Array[]         // the four logit maps at the photo's size
  ms: number                 // decoder time (session.run) in the worker
  upMs: number               // time to upscale the four maps on the page
  tokens: number             // sparse prompt tokens that went into the decoder
  usedMask: boolean
}
export type AmgMask = { bits: Uint8Array; box: number[]; score: number; stab: number; area: number; point: number[] }
export type AmgCounts = { points: number; candidates: number; passIou: number; passStab: number; passBig: number; final: number }

export class SamClient {
  worker: Worker
  private next = 1
  private wait = new Map<number | string, { ok: (v: any) => void; fail: (e: Error) => void }>()
  private amgProgress: ((p: { done: number; total: number; ms: number; counts: AmgCounts }) => void) | null = null

  constructor() {
    this.worker = new Worker(new URL('./samWorker.ts', import.meta.url), { type: 'module' })
    this.worker.onmessage = (e) => {
      const m = e.data
      if (m.type === 'amg-progress') { this.amgProgress?.(m); return }
      const key = m.type === 'ready' ? 'init' : m.type === 'embedded' ? `embed:${m.key}` : m.id
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
  }

  private ask<T>(key: number | string, msg: unknown, transfer: Transferable[] = []) {
    return new Promise<T>((ok, fail) => {
      this.wait.set(key, { ok, fail })
      this.worker.postMessage(msg, transfer)
    })
  }

  init(model: ArrayBuffer) {
    return this.ask<{ ms: number }>('init', { type: 'init', model }, [model])
  }

  embed(key: string, buf: ArrayBuffer) {
    return this.ask<void>(`embed:${key}`, { type: 'embed', key, buf }, [buf])
  }

  /** Run the prompt encoder + mask decoder for the prompts (in photo pixels) on a photo of w x h pixels. */
  async decode(p: Prompts, mask: Float32Array | null, w: number, h: number): Promise<Decoded> {
    const s = 1024 / Math.max(w, h)
    const c: number[] = [], l: number[] = []
    for (const q of p.points) { c.push(q.x * s, q.y * s); l.push(q.label) }
    if (p.box) { c.push(p.box[0] * s, p.box[1] * s, p.box[2] * s, p.box[3] * s); l.push(2, 3) }
    else { c.push(0, 0); l.push(-1) }      // the "not a point" padding token, as in SAM when there is no box
    const id = this.next++
    const m = mask ? mask.slice() : null
    const r = await this.ask<{ low: Float32Array; iou: Float32Array; ms: number }>(id,
      { type: 'decode', id, coords: new Float32Array(c), labels: new Float32Array(l), mask: m }, m ? [m.buffer] : [])
    const t = performance.now()
    const up = [0, 1, 2, 3].map((k) => upscale(r.low, k * LOW * LOW, w, h))
    return { low: r.low, iou: r.iou, ms: r.ms, up, upMs: performance.now() - t, tokens: l.length, usedMask: !!mask }
  }

  amg(opts: { n: number; w: number; h: number; iouT: number; stabT: number; bigT: number; nmsT: number },
    onProgress: (p: { done: number; total: number; ms: number; counts: AmgCounts }) => void) {
    const id = this.next++
    this.amgProgress = onProgress
    return this.ask<{ type: string; counts: AmgCounts; masks: AmgMask[]; ms: number }>(id, { type: 'amg', id, ...opts })
      .then((r) => (r.type === 'amg-cancelled' ? null : r))
  }

  cancelAmg() { this.worker.postMessage({ type: 'amg-cancel' }) }

  dispose() { this.worker.terminate() }
}

/** Fetch a file with a progress callback (bytes so far, total bytes if known). */
export async function fetchWithProgress(url: string, onProgress: (got: number, total: number) => void) {
  const res = await fetch(url)
  if (!res.ok || !res.body) throw new Error(`${url}: ${res.status}`)
  const total = Number(res.headers.get('content-length')) || 0
  const reader = res.body.getReader()
  const parts: Uint8Array[] = []
  let got = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    parts.push(value)
    got += value.length
    onProgress(got, total)
  }
  const out = new Uint8Array(got)
  let o = 0
  for (const p of parts) { out.set(p, o); o += p.length }
  return out.buffer
}
