/* Loading for the Transformer demo: our toy model (two ONNX files run in a worker) and the saved runs of the
   released translation model (Helsinki-NLP/opus-mt-en-de). Files are made by code/papers/transformer/demo/. */
import { f16ToF32, peTable } from './txMath'

export const BASE = import.meta.env.BASE_URL + 'demos/transformer/'

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

/* ---------------- the toy model ---------------- */

export type Ablation = { exact: number; right: number; n: number; wrong_examples: [string, string, string][] }
export type ToyMeta = {
  task: string
  config: { vocab_size: number; d_model: number; n_heads: number; d_ff: number; n_encoder_layers: number; n_decoder_layers: number
            max_len: number; pad_id: number; pos_layout: string; activation: string }
  vocab: string[]; bos: number; eos: number; pad: number; params: number; first: string; last: string
  n_train: number; n_test: number
  train: { steps: number | null; final_loss: number | null; seconds: number | null }
  ablation: Record<'normal' | 'no_position' | 'no_mask' | 'neither', Ablation>
  check: { max_logit_diff_onnx_vs_torch: number; same_greedy_text: number; n: number }
  bytes: { encoder: number; decoder: number }
  test_sample: string[]
}

export const loadToyMeta = () => once('toy-meta', () => get('toy/meta.json').then((r) => r.json() as Promise<ToyMeta>))

export type Encoded = { src: number[]; emb: Float32Array; x0: Float32Array; memory: Float32Array; att: Float32Array; pe: Float32Array; ms: number }
export type Decoded = { tgt: number[]; logits: Float32Array; self: Float32Array; cross: Float32Array; ms: number }
export type Timing = { download: number; session: number; bytes: number }

async function download(path: string, onBytes: (n: number) => void): Promise<ArrayBuffer> {
  const r = await get(path)
  const reader = r.body!.getReader()
  const parts: Uint8Array[] = []
  let done = 0
  for (;;) {
    const { value, done: end } = await reader.read()
    if (end) break
    parts.push(value)
    done += value.length
    onBytes(value.length)
  }
  const buf = new Uint8Array(done)
  let o = 0
  for (const p of parts) { buf.set(p, o); o += p.length }
  return buf.buffer
}

class Toy {
  worker: Worker | null = null
  ready: Promise<Timing> | null = null
  next = 1
  wait = new Map<number | string, { ok: (v: any) => void; fail: (e: Error) => void }>()
  meta: ToyMeta | null = null

  start(onProgress: (done: number, total: number) => void): Promise<Timing> {
    if (this.ready) return this.ready
    this.ready = (async () => {
      const meta = await loadToyMeta()
      this.meta = meta
      const total = meta.bytes.encoder + meta.bytes.decoder
      let got = 0
      const tick = (n: number) => { got += n; onProgress(got, total) }
      const t0 = performance.now()
      const [encoder, decoder] = await Promise.all([download('toy/encoder.onnx', tick), download('toy/decoder.onnx', tick)])
      const dl = performance.now() - t0
      this.worker = new Worker(new URL('./toyWorker.ts', import.meta.url), { type: 'module' })
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
      await this.ask('init', { type: 'init', encoder, decoder }, [encoder, decoder])
      return { download: dl, session: performance.now() - t1, bytes: got }
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

  /** The position table the model gets: the sinusoids of §3.5, or zeros when the reader turns positions off. */
  pe(rows: number, on: boolean): Float32Array {
    const c = this.meta!.config
    return on ? peTable(rows, c.d_model, c.pos_layout) : new Float32Array(rows * c.d_model)
  }

  async encode(src: number[], pos: boolean): Promise<Encoded> {
    const id = this.next++
    const pe = this.pe(src.length, pos)
    const r = await this.ask<Omit<Encoded, 'src' | 'pe'>>(id, { type: 'encode', id, src, pe: pe.slice(), d: this.meta!.config.d_model })
    return { ...r, src, pe }
  }

  /** One decoder call on the whole prefix tgt (no cache), as greedy_decode in model.py does. */
  async decode(enc: Encoded, tgt: number[], pos: boolean, masked: boolean): Promise<Decoded> {
    const id = this.next++
    const t = tgt.length
    const mask = new Float32Array(t * t)
    for (let i = 0; i < t; i++) for (let j = 0; j < t; j++) mask[i * t + j] = !masked || j <= i ? 1 : 0
    const r = await this.ask<Omit<Decoded, 'tgt'>>(id, {
      type: 'decode', id, tgt, src: enc.src, memory: enc.memory, pe: this.pe(t, pos), mask, d: this.meta!.config.d_model,
    })
    return { ...r, tgt }
  }
}
export const toy = new Toy()

/* ---------------- the released translator, saved runs ---------------- */

export type Sentence = {
  slug: string; en: string; de: string; note: string
  src: string[]; tgt: string[]; out: string[]; top5: [string, number][][]
  enc: number[]; cross: number[]; dec: number[]; bin: string; bytes: number
}
export type Marian = { model: string; layers: number; heads: number; d_model: number; params: number; max_abs_diff: number; sentences: Sentence[] }
export type Attn = { enc: Float32Array; cross: Float32Array; dec: Float32Array }

export const loadMarian = () => once('marian', () => get('marian/sentences.json').then((r) => r.json() as Promise<Marian>))

/** Decode one block: rows float16 row scales, then rows x cols uint8 levels; weight = level / 255 * scale. */
function unpack(buf: ArrayBuffer, offset: number, shape: number[]): [Float32Array, number] {
  const cols = shape[shape.length - 1]
  const rows = shape.reduce((a, b) => a * b, 1) / cols
  const scale = f16ToF32(new Uint16Array(buf.slice(offset, offset + rows * 2)))
  const lev = new Uint8Array(buf, offset + rows * 2, rows * cols)
  const out = new Float32Array(rows * cols)
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out[r * cols + c] = (lev[r * cols + c] / 255) * scale[r]
  return [out, offset + rows * 2 + rows * cols]
}

export function loadAttention(s: Sentence): Promise<Attn> {
  return once(`att:${s.slug}`, async () => {
    const buf = await get(`marian/${s.bin}`).then((r) => r.arrayBuffer())
    const [enc, o1] = unpack(buf, 0, s.enc)
    const [cross, o2] = unpack(buf, o1, s.cross)
    const [dec] = unpack(buf, o2, s.dec)
    return { enc, cross, dec }
  })
}
