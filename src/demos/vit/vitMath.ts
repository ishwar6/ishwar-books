/* The arithmetic of the ViT demo: the same formulas as code/papers/vit/demo/precompute.py (rollout, mean attention
   distance), float16 decoding, the patch shuffle and the colour maps of the heatmaps. */

export const G = 14          // patches per side
export const P = 16          // pixels per patch side
export const N = G * G       // 196 patches
export const T = N + 1       // 197 tokens with [class]

export function f16ToF32(src: Uint16Array): Float32Array {
  const out = new Float32Array(src.length)
  for (let i = 0; i < src.length; i++) {
    const h = src[i]
    const s = h & 0x8000 ? -1 : 1
    const e = (h >> 10) & 0x1f
    const f = h & 0x3ff
    out[i] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15)
  }
  return out
}

export async function gunzip(buf: ArrayBuffer): Promise<ArrayBuffer> {
  const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'))
  return await new Response(stream).arrayBuffer()
}

/** RGBA pixels of a 224 x 224 picture -> (1, 3, 224, 224) float32, values 0 to 255 (the model scales them itself). */
export function toPlanes(rgba: Uint8ClampedArray): Float32Array {
  const n = 224 * 224
  const out = new Float32Array(3 * n)
  for (let i = 0; i < n; i++) {
    out[i] = rgba[4 * i]
    out[n + i] = rgba[4 * i + 1]
    out[2 * n + i] = rgba[4 * i + 2]
  }
  return out
}

/** Slot k of the new picture gets patch perm[k] of the old one (as shuffle_tiles in precompute.py). */
export function shuffleTiles(rgba: Uint8ClampedArray, perm: number[]): Uint8ClampedArray {
  const out = new Uint8ClampedArray(rgba.length)
  for (let k = 0; k < N; k++) {
    const src = perm[k]
    const sr = Math.floor(src / G) * P, sc = (src % G) * P
    const dr = Math.floor(k / G) * P, dc = (k % G) * P
    for (let y = 0; y < P; y++) {
      const a = ((sr + y) * 224 + sc) * 4, b = ((dr + y) * 224 + dc) * 4
      out.set(rgba.subarray(a, a + P * 4), b)
    }
  }
  return out
}

export function randomPerm(n: number): number[] {
  const p = Array.from({ length: n }, (_, i) => i)
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[p[i], p[j]] = [p[j], p[i]]
  }
  return p
}

/** The 768 numbers of patch n, in the order the series flattens them: all red values, then green, then blue,
 *  each 16 x 16 block row by row; scaled to [-1, 1] like the model's preprocessing. */
export function patchNumbers(rgba: Uint8ClampedArray, n: number): Float32Array {
  const out = new Float32Array(3 * P * P)
  const r0 = Math.floor(n / G) * P, c0 = (n % G) * P
  for (let ch = 0; ch < 3; ch++)
    for (let y = 0; y < P; y++)
      for (let x = 0; x < P; x++)
        out[ch * P * P + y * P + x] = rgba[((r0 + y) * 224 + c0 + x) * 4 + ch] / 127.5 - 1
  return out
}

export function softmax(logits: Float32Array | number[]): Float64Array {
  let mx = -Infinity
  for (const v of logits) mx = Math.max(mx, v)
  const out = new Float64Array(logits.length)
  let s = 0
  for (let i = 0; i < logits.length; i++) s += out[i] = Math.exp(logits[i] - mx)
  for (let i = 0; i < out.length; i++) out[i] /= s
  return out
}

export type Top = [number, string, number][]

export function top5(logits: Float32Array, labels: string[]): Top {
  const p = softmax(logits)
  const idx = Array.from(p.keys()).sort((a, b) => p[b] - p[a]).slice(0, 5)
  return idx.map((i) => [i, labels[i], p[i]])
}

/** Distances in pixels between the centres of every pair of patches. */
const DIST = (() => {
  const d = new Float64Array(N * N)
  for (let i = 0; i < N; i++)
    for (let j = 0; j < N; j++) {
      const dy = (Math.floor(i / G) - Math.floor(j / G)) * P, dx = ((i % G) - (j % G)) * P
      d[i * N + j] = Math.hypot(dy, dx)
    }
  return d
})()

/** att: (L, H, 197, 197). For each query patch, the attention-weighted mean distance to the 196 key patches (the
 *  weights over the patches renormalised to sum to 1), then the mean over the query patches. Returns [L][H] pixels. */
export function meanDistance(att: Float32Array, L: number, H: number): number[][] {
  const out: number[][] = []
  for (let l = 0; l < L; l++) {
    const row: number[] = []
    for (let h = 0; h < H; h++) {
      const base = (l * H + h) * T * T
      let total = 0
      for (let i = 0; i < N; i++) {
        const o = base + (i + 1) * T + 1
        let s = 0, d = 0
        for (let j = 0; j < N; j++) {
          const a = att[o + j]
          s += a
          d += a * DIST[i * N + j]
        }
        total += d / s
      }
      row.push(total / N)
    }
    out.push(row)
  }
  return out
}

/** Attention rollout (Figure 6): average the heads, add the identity for the residual path, renormalise, multiply
 *  the layers from the first to the last; the [class] row over the 196 patches, scaled to sum to 1. */
export function rollout(att: Float32Array, L: number, H: number): Float64Array {
  let R = new Float64Array(T * T)
  for (let i = 0; i < T; i++) R[i * T + i] = 1
  const A = new Float64Array(T * T)
  for (let l = 0; l < L; l++) {
    A.fill(0)
    for (let h = 0; h < H; h++) {
      const base = (l * H + h) * T * T
      for (let k = 0; k < T * T; k++) A[k] += att[base + k] / H
    }
    for (let k = 0; k < T * T; k++) A[k] *= 0.5
    for (let i = 0; i < T; i++) A[i * T + i] += 0.5
    const next = new Float64Array(T * T)
    for (let i = 0; i < T; i++)
      for (let k = 0; k < T; k++) {
        const a = A[i * T + k]
        if (a === 0) continue
        const ro = k * T, no = i * T
        for (let j = 0; j < T; j++) next[no + j] += a * R[ro + j]
      }
    R = next
  }
  const v = R.slice(1, T)
  let s = 0
  for (const x of v) s += x
  return v.map((x) => x / s)
}

/* ---------- colour maps (data images keep their own colours in both themes) ---------- */

// magma-like ramp: black, purple, orange, light yellow
const MAGMA = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]]

export function magma(t: number): [number, number, number] {
  t = Math.min(1, Math.max(0, t)) * (MAGMA.length - 1)
  const i = Math.min(MAGMA.length - 2, Math.floor(t)), f = t - i
  const a = MAGMA[i], b = MAGMA[i + 1]
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]
}

/** Blue for negative, near-black for 0, orange for positive (for embeddings; t in [-1, 1]). */
export function diverging(t: number): [number, number, number] {
  t = Math.max(-1, Math.min(1, t))
  const z = [24, 24, 30]
  const c = t < 0 ? [80, 150, 255] : [255, 160, 60]
  const k = Math.abs(t) ** 0.7
  return [z[0] + (c[0] - z[0]) * k, z[1] + (c[1] - z[1]) * k, z[2] + (c[2] - z[2]) * k]
}
