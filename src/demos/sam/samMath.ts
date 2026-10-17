/* Small numeric helpers of the SAM demo: float16 decoding, the upscaling of SAM's 256 x 256 mask logits to the photo
   (the same two bilinear steps as transformers' post_process_masks), mask statistics, IoU and box NMS. */

export const LOW = 256 // the decoder's mask grid
export const FRAME = 1024 // the square that the image encoder reads

/** Decode little-endian float16 numbers into float32. */
export function f16ToF32(src: Uint16Array): Float32Array {
  const table = new Float32Array(65536)
  for (let x = 0; x < 65536; x++) {
    const s = x >> 15, e = (x >> 10) & 31, m = x & 1023
    const v = e === 0 ? m * 2 ** -24 : e === 31 ? (m ? NaN : Infinity) : (1 + m / 1024) * 2 ** (e - 15)
    table[x] = s ? -v : v
  }
  const out = new Float32Array(src.length)
  for (let i = 0; i < src.length; i++) out[i] = table[src[i]]
  return out
}

/** The scale from photo pixels to the 1024 frame, and the size of the photo inside that frame. */
export function frameOf(w: number, h: number) {
  const s = FRAME / Math.max(w, h)
  return { s, rw: Math.floor(w * s + 0.5), rh: Math.floor(h * s + 0.5) }
}

/** PyTorch bilinear interpolation (align_corners=False, no antialias) on one axis: index pairs and weights. */
function axis(inSize: number, outSize: number) {
  const scale = inSize / outSize
  const i0 = new Int32Array(outSize), i1 = new Int32Array(outSize), l1 = new Float32Array(outSize)
  for (let o = 0; o < outSize; o++) {
    let src = (o + 0.5) * scale - 0.5
    if (src < 0) src = 0
    const a = Math.min(Math.floor(src), inSize - 1)
    i0[o] = a
    i1[o] = Math.min(a + 1, inSize - 1)
    l1[o] = src - a
  }
  return { i0, i1, l1 }
}

const axisCache = new Map<string, ReturnType<typeof axis>>()
function axisC(inSize: number, outSize: number) {
  const k = `${inSize}:${outSize}`
  let a = axisCache.get(k)
  if (!a) { a = axis(inSize, outSize); axisCache.set(k, a) }
  return a
}

/** Resize the top-left (inH x inW) part of a map whose rows are sw long to (dh x dw). */
function resize(src: Float32Array, sw: number, dw: number, dh: number, inW: number, inH: number) {
  const ax = axisC(inW, dw), ay = axisC(inH, dh)
  const out = new Float32Array(dw * dh)
  for (let y = 0; y < dh; y++) {
    const r0 = ay.i0[y] * sw, r1 = ay.i1[y] * sw, wy = ay.l1[y], wy0 = 1 - wy
    const row = y * dw
    for (let x = 0; x < dw; x++) {
      const c0 = ax.i0[x], c1 = ax.i1[x], wx = ax.l1[x], wx0 = 1 - wx
      out[row + x] = wy0 * (wx0 * src[r0 + c0] + wx * src[r0 + c1]) + wy * (wx0 * src[r1 + c0] + wx * src[r1 + c1])
    }
  }
  return out
}

/** One 256 x 256 logit map -> the logits at every photo pixel (w x h): 256 -> 1024 x 1024, keep the photo part
 *  (rw x rh, the rest is padding), then -> w x h. Both steps bilinear, as in post_process_masks. */
export function upscale(low: Float32Array, offset: number, w: number, h: number): Float32Array {
  const { rw, rh } = frameOf(w, h)
  const src = low.subarray(offset, offset + LOW * LOW)
  // step 1: only the rows and columns of the 1024 frame that the photo covers are needed
  const ax = axisC(LOW, FRAME), ay = axisC(LOW, FRAME)
  const mid = new Float32Array(rw * rh)
  for (let y = 0; y < rh; y++) {
    const r0 = ay.i0[y] * LOW, r1 = ay.i1[y] * LOW, wy = ay.l1[y], wy0 = 1 - wy
    for (let x = 0; x < rw; x++) {
      const c0 = ax.i0[x], c1 = ax.i1[x], wx = ax.l1[x], wx0 = 1 - wx
      mid[y * rw + x] = wy0 * (wx0 * src[r0 + c0] + wx * src[r0 + c1]) + wy * (wx0 * src[r1 + c0] + wx * src[r1 + c1])
    }
  }
  if (rw === w && rh === h) return mid
  return resize(mid, rw, w, h, rw, rh)
}

/** Pixels above t, above t + 1 and above t - 1, and the stability score of Part 4: IoU(M_{t+1}, M_{t-1}). */
export function maskStats(logits: Float32Array, t = 0) {
  let area = 0, hi = 0, lo = 0
  for (let i = 0; i < logits.length; i++) {
    const v = logits[i]
    if (v > t - 1) {
      lo++
      if (v > t) { area++; if (v > t + 1) hi++ }
    }
  }
  return { area, hi, lo, stability: lo > 0 ? hi / lo : 0 }
}

export function toMask(logits: Float32Array, t = 0): Uint8Array {
  const m = new Uint8Array(logits.length)
  for (let i = 0; i < logits.length; i++) m[i] = logits[i] > t ? 1 : 0
  return m
}

/** Intersection, union and IoU of two masks of the same size (Part 1). */
export function iou(a: Uint8Array, b: Uint8Array) {
  let inter = 0, union = 0
  for (let i = 0; i < a.length; i++) {
    const x = a[i], y = b[i]
    if (x | y) { union++; if (x & y) inter++ }
  }
  return { inter, union, iou: union ? inter / union : 1 }
}

/** Bounding box [x0, y0, x1, y1] (inclusive) of the pixels above t, or null when the mask is empty. */
export function boxOf(logits: Float32Array, w: number, h: number, t = 0): [number, number, number, number] | null {
  let x0 = w, y0 = h, x1 = -1, y1 = -1
  for (let y = 0; y < h; y++) {
    const row = y * w
    for (let x = 0; x < w; x++) {
      if (logits[row + x] > t) {
        if (x < x0) x0 = x
        if (x > x1) x1 = x
        if (y < y0) y0 = y
        y1 = y
      }
    }
  }
  return x1 < 0 ? null : [x0, y0, x1, y1]
}

export function boxIou(a: number[], b: number[]) {
  // torchvision's box_iou on [x0, y0, x1, y1] boxes (no +1)
  const iw = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]))
  const ih = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]))
  const inter = iw * ih
  const ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
  return ua > 0 ? inter / ua : 0
}

/** Greedy NMS: keep the best box, drop every box that overlaps it by more than thr, repeat. */
export function nms(boxes: number[][], scores: number[], thr: number): number[] {
  const order = scores.map((_, i) => i).sort((i, j) => scores[j] - scores[i])
  const gone = new Uint8Array(boxes.length)
  const keep: number[] = []
  for (const i of order) {
    if (gone[i]) continue
    keep.push(i)
    for (const j of order) if (!gone[j] && j !== i && boxIou(boxes[i], boxes[j]) > thr) gone[j] = 1
  }
  return keep
}

/** Pack a 0/1 mask into bits (8 pixels per byte), and back. */
export function pack(m: Uint8Array) {
  const out = new Uint8Array((m.length + 7) >> 3)
  for (let i = 0; i < m.length; i++) if (m[i]) out[i >> 3] |= 1 << (i & 7)
  return out
}
export function unpack(p: Uint8Array, n: number) {
  const out = new Uint8Array(n)
  for (let i = 0; i < n; i++) out[i] = (p[i >> 3] >> (i & 7)) & 1
  return out
}
