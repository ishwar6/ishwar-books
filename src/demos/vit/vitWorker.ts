/// <reference lib="webworker" />
/* The ViT demo's worker: it holds the ONNX session of the live model (DeiT-Ti/16, the ViT design at a small width) and
   runs it on a 224 x 224 picture. Running it here keeps the page responsive while the model works. */
import * as ort from 'onnxruntime-web/wasm'

// an absolute URL, so the dev server does not rewrite the runtime's own import of its .mjs file
ort.env.wasm.wasmPaths = new URL(import.meta.env.BASE_URL + 'ort/', self.location.origin).href
ort.env.wasm.numThreads = 1

let session: ort.InferenceSession | null = null
const post = (m: unknown, transfer: Transferable[] = []) => (self as DedicatedWorkerGlobalScope).postMessage(m, transfer)

type Msg =
  | { type: 'init'; model: ArrayBuffer }
  | { type: 'run'; id: number; pixels: Float32Array; pos: number }

self.onmessage = async (e: MessageEvent<Msg>) => {
  const m = e.data
  try {
    if (m.type === 'init') {
      const t = performance.now()
      session = await ort.InferenceSession.create(new Uint8Array(m.model), { executionProviders: ['wasm'] })
      post({ type: 'ready', ms: performance.now() - t })
    } else if (m.type === 'run') {
      if (!session) throw new Error('the model is not loaded')
      const feeds = {
        pixels: new ort.Tensor('float32', m.pixels, [1, 3, 224, 224]),
        pos_scale: new ort.Tensor('float32', new Float32Array([m.pos]), [1]),
      }
      const t = performance.now()
      const out = await session.run(feeds)
      const ms = performance.now() - t
      const patch = out.patch_embeddings.data as Float32Array
      const tokens = out.tokens.data as Float32Array
      const att = out.attentions.data as Float32Array
      const logits = out.logits.data as Float32Array
      post({ type: 'result', id: m.id, patch, tokens, att, logits, ms }, [patch.buffer, tokens.buffer, att.buffer, logits.buffer])
    }
  } catch (err) {
    post({ type: 'error', id: m.type === 'run' ? m.id : 'init', message: (err as Error).message ?? String(err) })
  }
}
