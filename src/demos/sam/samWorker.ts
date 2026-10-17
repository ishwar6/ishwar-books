/// <reference lib="webworker" />
/* The SAM demo's worker: it holds the ONNX session (prompt encoder + mask decoder of SAM ViT-H) and the image
   embedding of the current photo, runs the decoder for each prompt, and runs the automatic mask generator. Running
   it here keeps the page responsive while the decoder works. */
import * as ort from 'onnxruntime-web/wasm'
import { f16ToF32, upscale, maskStats, toMask, boxOf, nms, pack, LOW } from './samMath'

// a full URL: the dev server rewrites dynamic imports of paths that start with '/', and public files cannot be imported
ort.env.wasm.wasmPaths = new URL(import.meta.env.BASE_URL + 'ort/', self.location.origin).href
ort.env.wasm.numThreads = 1

let session: ort.InferenceSession | null = null
let embedding: ort.Tensor | null = null
let cancelled = false
const noMask = new Float32Array(LOW * LOW)

async function decode(coords: Float32Array, labels: Float32Array, mask: Float32Array | null) {
  if (!session || !embedding) throw new Error('not ready')
  const n = labels.length
  const feeds = {
    image_embeddings: embedding,
    point_coords: new ort.Tensor('float32', coords, [1, n, 2]),
    point_labels: new ort.Tensor('float32', labels, [1, n]),
    mask_input: new ort.Tensor('float32', mask ?? noMask, [1, 1, LOW, LOW]),
    has_mask_input: new ort.Tensor('float32', new Float32Array([mask ? 1 : 0]), [1]),
  }
  const t = performance.now()
  const out = await session.run(feeds)
  const ms = performance.now() - t
  return { low: out.low_res_logits.data as Float32Array, iou: out.iou_predictions.data as Float32Array, ms }
}

type Msg =
  | { type: 'init'; model: ArrayBuffer }
  | { type: 'embed'; key: string; buf: ArrayBuffer }
  | { type: 'decode'; id: number; coords: Float32Array; labels: Float32Array; mask: Float32Array | null }
  | { type: 'amg'; id: number; n: number; w: number; h: number; iouT: number; stabT: number; bigT: number; nmsT: number }
  | { type: 'amg-cancel' }

const post = (m: unknown, transfer: Transferable[] = []) => (self as DedicatedWorkerGlobalScope).postMessage(m, transfer)

self.onmessage = async (e: MessageEvent<Msg>) => {
  const m = e.data
  try {
    if (m.type === 'init') {
      const t = performance.now()
      session = await ort.InferenceSession.create(new Uint8Array(m.model), {
        executionProviders: ['wasm'], graphOptimizationLevel: 'all',
      })
      post({ type: 'ready', ms: performance.now() - t })
    } else if (m.type === 'embed') {
      embedding = new ort.Tensor('float32', f16ToF32(new Uint16Array(m.buf)), [1, 256, 64, 64])
      post({ type: 'embedded', key: m.key })
    } else if (m.type === 'decode') {
      const r = await decode(m.coords, m.labels, m.mask)
      const low = new Float32Array(r.low), iou = new Float32Array(r.iou)
      post({ type: 'decoded', id: m.id, low, iou, ms: r.ms }, [low.buffer, iou.buffer])
    } else if (m.type === 'amg-cancel') {
      cancelled = true
    } else if (m.type === 'amg') {
      cancelled = false
      await amg(m)
    }
  } catch (err) {
    post({ type: 'error', id: 'id' in m ? m.id : undefined, message: String(err) })
  }
}

/** The fully automatic stage of the data engine (Part 4), on the full photo only: an n x n grid of points, three
 *  masks per point, then the filters (predicted IoU, stability, not the whole picture) and box NMS. */
async function amg(m: Extract<Msg, { type: 'amg' }>) {
  const { n, w, h, iouT, stabT, bigT, nmsT } = m
  const s = 1024 / Math.max(w, h)
  const counts = { points: n * n, candidates: 0, passIou: 0, passStab: 0, passBig: 0, final: 0 }
  const kept: { bits: Uint8Array; box: number[]; score: number; stab: number; area: number; point: number[] }[] = []
  let total = 0
  for (let k = 0; k < n * n; k++) {
    if (cancelled) { post({ type: 'amg-cancelled', id: m.id }); return }
    const gx = ((k % n) + 0.5) / n * w, gy = (Math.floor(k / n) + 0.5) / n * h      // the centre of each grid cell
    const r = await decode(new Float32Array([gx * s, gy * s, 0, 0]), new Float32Array([1, -1]), null)
    total += r.ms
    for (let j = 1; j <= 3; j++) {
      counts.candidates++
      const score = r.iou[j]
      if (!(score > iouT)) continue
      counts.passIou++
      const lg = upscale(r.low, j * LOW * LOW, w, h)
      const st = maskStats(lg, 0)
      if (!(st.stability >= stabT)) continue
      counts.passStab++
      if (st.area >= bigT * w * h || st.area === 0) continue
      counts.passBig++
      const b = boxOf(lg, w, h)!
      kept.push({ bits: pack(toMask(lg)), box: b, score, stab: st.stability, area: st.area, point: [gx, gy] })
    }
    post({ type: 'amg-progress', id: m.id, done: k + 1, total: n * n, ms: total, counts: { ...counts } })
  }
  const keep = nms(kept.map((c) => c.box), kept.map((c) => c.score), nmsT)
  counts.final = keep.length
  const masks = keep.map((i) => kept[i]).sort((a, b) => b.area - a.area)
  post({ type: 'amg-done', id: m.id, counts, masks, ms: total }, masks.map((x) => x.bits.buffer))
}
