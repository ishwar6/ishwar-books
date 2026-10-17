/* The arithmetic of the Transformer demo: the position table of §3.5 (the same formula as sinusoid_table in
   code/papers/transformer/scratch/model.py), softmax with a temperature, sampling, float16 decoding, the date
   helpers of the toy task (toy_task.py) and the colours of the data canvases. */

/** PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(...): the paper's interleaved layout. (rows, d) */
export function peTable(rows: number, d: number, layout = 'interleaved'): Float32Array {
  const out = new Float32Array(rows * d)
  const half = d / 2
  for (let pos = 0; pos < rows; pos++) {
    for (let i = 0; i < half; i++) {
      // float32 like torch: freq = 1 / 10000 ** (2i / d); angle = pos * freq
      const freq = Math.fround(1 / Math.fround(10000 ** Math.fround((2 * i) / d)))
      const angle = Math.fround(pos * freq)
      const s = Math.sin(angle), c = Math.cos(angle)
      if (layout === 'interleaved') { out[pos * d + 2 * i] = s; out[pos * d + 2 * i + 1] = c }
      else { out[pos * d + i] = s; out[pos * d + half + i] = c }
    }
  }
  return out
}

export function softmax(x: ArrayLike<number>, temperature = 1): Float64Array {
  const t = Math.max(temperature, 1e-6)
  let m = -Infinity
  for (let i = 0; i < x.length; i++) m = Math.max(m, x[i] / t)
  const out = new Float64Array(x.length)
  let s = 0
  for (let i = 0; i < x.length; i++) { out[i] = Math.exp(x[i] / t - m); s += out[i] }
  for (let i = 0; i < x.length; i++) out[i] /= s
  return out
}

export function argmax(x: ArrayLike<number>): number {
  let b = 0
  for (let i = 1; i < x.length; i++) if (x[i] > x[b]) b = i
  return b
}

export function topK(p: ArrayLike<number>, k: number): [number, number][] {
  return Array.from(p, (v, i) => [i, v] as [number, number]).sort((a, b) => b[1] - a[1]).slice(0, k)
}

/** Draw one index from the probabilities p. */
export function sample(p: ArrayLike<number>, r = Math.random()): number {
  let acc = 0
  for (let i = 0; i < p.length; i++) { acc += p[i]; if (r < acc) return i }
  return p.length - 1
}

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

/* ---- the toy task (code/papers/transformer/scratch/toy_task.py) ---- */
export const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
  'November', 'December']

/** "2021-03-12" -> "12 March 2021" when it is a real date from 1900 to 2099, else null. */
export function humanDate(iso: string): string | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso)
  if (!m) return null
  const y = +m[1], mo = +m[2], d = +m[3]
  if (y < 1900 || y > 2099 || mo < 1 || mo > 12 || d < 1) return null
  const days = new Date(Date.UTC(y, mo, 0)).getUTCDate()
  if (d > days) return null
  return `${d} ${MONTHS[mo - 1]} ${y}`
}

export function randomDate(): string {
  const a = Date.UTC(1900, 0, 1), b = Date.UTC(2099, 11, 31)
  const t = new Date(a + Math.floor(Math.random() * ((b - a) / 864e5 + 1)) * 864e5)
  return t.toISOString().slice(0, 10)
}

/* ---- colours of the data canvases (they keep their own colours in every theme) ---- */

/** Blue for negative, near-black for 0, orange for positive (t in [-1, 1]). */
export function diverging(t: number): [number, number, number] {
  t = Math.max(-1, Math.min(1, t))
  const z = [24, 24, 30]
  const c = t < 0 ? [80, 150, 255] : [255, 160, 60]
  const k = Math.abs(t) ** 0.7
  return [z[0] + (c[0] - z[0]) * k, z[1] + (c[1] - z[1]) * k, z[2] + (c[2] - z[2]) * k]
}

export const fix = (v: number, d = 3) => (Object.is(v, -0) ? 0 : v).toFixed(d).replace(/^-(0\.0+)$/, '$1')
export const pct = (p: number) => (p >= 0.1 ? `${(100 * p).toFixed(1)}%` : p >= 0.001 ? `${(100 * p).toFixed(2)}%` : p > 0 ? `${(100 * p).toExponential(1)}%` : '0%')
