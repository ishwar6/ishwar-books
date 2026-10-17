/// <reference lib="webworker" />
/* The Transformer demo's worker: it holds the two ONNX sessions of our from-scratch toy model (the encoder, and one
   decoder call) and runs them. Running them here keeps the page responsive. */
import * as ort from 'onnxruntime-web/wasm'

// an absolute URL, so the dev server does not rewrite the runtime's own import of its .mjs file
ort.env.wasm.wasmPaths = new URL(import.meta.env.BASE_URL + 'ort/', self.location.origin).href
ort.env.wasm.numThreads = 1

let enc: ort.InferenceSession | null = null
let dec: ort.InferenceSession | null = null
const post = (m: unknown, transfer: Transferable[] = []) => (self as DedicatedWorkerGlobalScope).postMessage(m, transfer)

type Msg =
  | { type: 'init'; encoder: ArrayBuffer; decoder: ArrayBuffer }
  | { type: 'encode'; id: number; src: number[]; pe: Float32Array; d: number }
  | { type: 'decode'; id: number; tgt: number[]; src: number[]; memory: Float32Array; pe: Float32Array; mask: Float32Array; d: number }

const ids = (a: number[]) => new ort.Tensor('int64', BigInt64Array.from(a.map(BigInt)), [1, a.length])
const f32 = (t: ort.Tensor) => t.data as Float32Array

self.onmessage = async (e: MessageEvent<Msg>) => {
  const m = e.data
  try {
    if (m.type === 'init') {
      const t = performance.now()
      const opt = { executionProviders: ['wasm'] }
      enc = await ort.InferenceSession.create(new Uint8Array(m.encoder), opt)
      dec = await ort.InferenceSession.create(new Uint8Array(m.decoder), opt)
      post({ type: 'ready', ms: performance.now() - t })
    } else if (m.type === 'encode') {
      if (!enc) throw new Error('the model is not loaded')
      const n = m.src.length
      const t = performance.now()
      const o = await enc.run({ src_ids: ids(m.src), pe: new ort.Tensor('float32', m.pe, [n, m.d]) })
      const ms = performance.now() - t
      const r = { emb: f32(o.emb), x0: f32(o.x0), memory: f32(o.memory), att: f32(o.enc_att) }
      post({ type: 'result', id: m.id, ...r, ms }, [r.emb.buffer, r.x0.buffer, r.memory.buffer, r.att.buffer])
    } else if (m.type === 'decode') {
      if (!dec) throw new Error('the model is not loaded')
      const n = m.src.length, t = m.tgt.length
      const t0 = performance.now()
      const o = await dec.run({
        tgt_ids: ids(m.tgt), memory: new ort.Tensor('float32', m.memory, [1, n, m.d]), src_ids: ids(m.src),
        pe: new ort.Tensor('float32', m.pe, [t, m.d]), mask: new ort.Tensor('float32', m.mask, [t, t]),
      })
      const ms = performance.now() - t0
      const r = { logits: f32(o.logits), self: f32(o.self_att), cross: f32(o.cross_att) }
      post({ type: 'result', id: m.id, ...r, ms }, [r.logits.buffer, r.self.buffer, r.cross.buffer])
    }
  } catch (err) {
    post({ type: 'error', id: m.type === 'init' ? 'init' : m.id, message: (err as Error).message ?? String(err) })
  }
}
