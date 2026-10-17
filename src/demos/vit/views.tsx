/* The drawing pieces of the ViT demo: the picture with a patch grid and a heatmap, number strips, the token matrix,
   small heatmaps and the attention-distance chart. Canvases draw data in their own colours; the chrome uses theme tokens. */
import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { G, N, magma, diverging } from './vitMath'

type Rgb = [number, number, number]

function paintBase(ctx: CanvasRenderingContext2D, rgba: Uint8ClampedArray, size: number) {
  const off = document.createElement('canvas')
  off.width = off.height = 224
  off.getContext('2d')!.putImageData(new ImageData(new Uint8ClampedArray(rgba), 224, 224), 0, 0)
  ctx.imageSmoothingEnabled = true
  ctx.drawImage(off, 0, 0, size, size)
}

/** The 224 x 224 picture, drawn at 448 x 448, with an optional patch grid, a spotlight heatmap (one value per patch:
 *  the highest patch at full brightness, the lowest dark) and a selected patch. Click or tap picks a patch. */
export function PictureView(props: {
  rgba: Uint8ClampedArray | null
  grid?: number                       // patch size in pixels for the grid lines (0: none)
  heat?: ArrayLike<number> | null     // 196 values, any scale
  sel?: number | null                 // selected patch (outlined)
  mark?: number | null                // a second outlined patch (for example the query)
  onPick?: (n: number) => void
  onHover?: (n: number | null) => void
  label: string
  gamma?: number                      // brightness = t ** gamma (0.5 lifts the faint patches)
}) {
  const ref = useRef<HTMLCanvasElement>(null)
  const S = 448
  const { rgba, grid = 0, heat, sel, mark, onPick, onHover, label, gamma = 0.5 } = props
  useEffect(() => {
    const c = ref.current
    if (!c || !rgba) return
    const ctx = c.getContext('2d')!
    paintBase(ctx, rgba, S)
    if (heat) {
      let mx = -Infinity, mn = Infinity
      for (let i = 0; i < N; i++) { mx = Math.max(mx, heat[i]); mn = Math.min(mn, heat[i]) }
      const cell = S / G
      for (let i = 0; i < N; i++) {
        const t = mx > mn ? (heat[i] - mn) / (mx - mn) : 0          // lowest patch dark, highest patch bright
        ctx.fillStyle = `rgba(0,0,0,${(0.88 * (1 - t ** gamma)).toFixed(3)})`
        ctx.fillRect((i % G) * cell, Math.floor(i / G) * cell, cell, cell)
      }
    }
    if (grid) {
      ctx.strokeStyle = 'rgba(255,255,255,0.45)'
      ctx.lineWidth = 1
      const step = (S * grid) / 224
      ctx.beginPath()
      for (let x = step; x < S - 0.5; x += step) {
        ctx.moveTo(Math.round(x) + 0.5, 0); ctx.lineTo(Math.round(x) + 0.5, S)
        ctx.moveTo(0, Math.round(x) + 0.5); ctx.lineTo(S, Math.round(x) + 0.5)
      }
      ctx.stroke()
    }
    const cell = S / G
    const box = (n: number, col: string) => {
      ctx.strokeStyle = 'rgba(0,0,0,0.7)'
      ctx.lineWidth = 5
      ctx.strokeRect((n % G) * cell + 1.5, Math.floor(n / G) * cell + 1.5, cell - 3, cell - 3)
      ctx.strokeStyle = col
      ctx.lineWidth = 2.5
      ctx.strokeRect((n % G) * cell + 1.5, Math.floor(n / G) * cell + 1.5, cell - 3, cell - 3)
    }
    if (mark != null && mark >= 0) box(mark, '#4fc3ff')
    if (sel != null && sel >= 0 && (grid === 16 || !grid)) box(sel, '#ffd23f')
  }, [rgba, grid, heat, sel, mark, gamma])

  const at = (e: { clientX: number; clientY: number }) => {
    const r = ref.current!.getBoundingClientRect()
    const col = Math.floor(((e.clientX - r.left) / r.width) * G)
    const row = Math.floor(((e.clientY - r.top) / r.height) * G)
    if (col < 0 || row < 0 || col >= G || row >= G) return null
    return row * G + col
  }
  const onKey = (e: KeyboardEvent) => {
    if (!onPick) return
    const cur = sel ?? mark ?? 0
    const r = Math.floor(Math.max(cur, 0) / G), c = Math.max(cur, 0) % G
    const moves: Record<string, [number, number]> = { ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0] }
    const m = moves[e.key]
    if (!m) return
    e.preventDefault()
    onPick(Math.min(G - 1, Math.max(0, r + m[0])) * G + Math.min(G - 1, Math.max(0, c + m[1])))
  }
  return (
    <canvas
      ref={ref}
      width={S}
      height={S}
      className={`vit-demo-picture${onPick ? ' is-pickable' : ''}`}
      role={onPick ? 'img' : undefined}
      aria-label={label}
      tabIndex={onPick ? 0 : undefined}
      onKeyDown={onKey}
      onClick={(e) => { const n = at(e); if (n != null) onPick?.(n) }}
      onMouseMove={(e) => onHover?.(at(e))}
      onMouseLeave={() => onHover?.(null)}
    />
  )
}

/** A row of numbers as coloured cells, one pixel column per number. */
export function Strip({ values, color, height = 26, label }: { values: ArrayLike<number>; color: (v: number, i: number) => Rgb; height?: number; label: string }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    if (!c) return
    const ctx = c.getContext('2d')!
    const img = ctx.createImageData(values.length, 1)
    for (let i = 0; i < values.length; i++) {
      const [r, g, b] = color(values[i], i)
      img.data.set([r, g, b, 255], 4 * i)
    }
    ctx.putImageData(img, 0, 0)
  }, [values, color])
  return <canvas ref={ref} width={values.length} height={1} className="vit-demo-strip" style={{ height }} role="img" aria-label={label} />
}

/** The token matrix: one row per token, one column per number; colour = sign and size (diverging scale). */
export function Matrix({ data, rows, cols, scale, sel, label }: { data: Float32Array; rows: number; cols: number; scale: number; sel: number | null; label: string }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    if (!c) return
    const ctx = c.getContext('2d')!
    const img = ctx.createImageData(cols, rows)
    for (let i = 0; i < rows * cols; i++) {
      const [r, g, b] = diverging(data[i] / scale)
      img.data.set([r, g, b, 255], 4 * i)
    }
    ctx.putImageData(img, 0, 0)
    if (sel != null) {
      ctx.fillStyle = 'rgba(255,210,63,0.9)'
      ctx.fillRect(0, sel, Math.max(4, cols / 120), 1)
      ctx.fillRect(cols - Math.max(4, cols / 120), sel, Math.max(4, cols / 120), 1)
    }
  }, [data, rows, cols, scale, sel])
  return <canvas ref={ref} width={cols} height={rows} className="vit-demo-matrix" role="img" aria-label={label} />
}

/** A 14 x 14 heatmap in the magma colours (lo..hi mapped to dark..bright). */
export function Heat({ values, lo, hi, mark, size = 112, label, onClick, active }: {
  values: ArrayLike<number>; lo: number; hi: number; mark?: number | null; size?: number; label: string
  onClick?: () => void; active?: boolean
}) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    if (!c) return
    const ctx = c.getContext('2d')!
    const img = ctx.createImageData(G, G)
    for (let i = 0; i < N; i++) {
      const [r, g, b] = magma((values[i] - lo) / (hi - lo || 1))
      img.data.set([r, g, b, 255], 4 * i)
    }
    ctx.putImageData(img, 0, 0)
  }, [values, lo, hi])
  const canvas = (
    <span className="vit-demo-heat" style={{ width: size, height: size }}>
      <canvas ref={ref} width={G} height={G} role="img" aria-label={label} />
      {mark != null && mark >= 0 && (
        <i style={{ left: `${((mark % G) / G) * 100}%`, top: `${(Math.floor(mark / G) / G) * 100}%`, width: `${100 / G}%`, height: `${100 / G}%` }} />
      )}
    </span>
  )
  if (!onClick) return canvas
  return <button type="button" className={`vit-demo-heatbtn${active ? ' is-on' : ''}`} onClick={onClick} aria-pressed={active}>{canvas}</button>
}

/** All 196 position-similarity maps in one 14 x 14 grid of 14 x 14 maps, like the paper's Figure 7 (centre). */
export function SimGrid({ sim, sel, onPick }: { sim: Float32Array; sel: number; onPick: (n: number) => void }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const C = G + 1                                   // 14 cells and a 1-pixel gap
  useEffect(() => {
    const c = ref.current
    if (!c) return
    const ctx = c.getContext('2d')!
    const W = G * C
    const img = ctx.createImageData(W, W)
    for (let k = 0; k < W * W; k++) img.data.set([0, 0, 0, 0], 4 * k)
    for (let q = 0; q < N; q++) {
      const oy = Math.floor(q / G) * C, ox = (q % G) * C
      for (let j = 0; j < N; j++) {
        const [r, g, b] = magma((sim[q * N + j] + 1) / 2)
        img.data.set([r, g, b, 255], 4 * ((oy + Math.floor(j / G)) * W + ox + (j % G)))
      }
    }
    ctx.putImageData(img, 0, 0)
  }, [sim])
  return (
    <span className="vit-demo-simgrid">
      <canvas
        ref={ref}
        width={G * C}
        height={G * C}
        role="img"
        aria-label="The 196 position-similarity maps, one per patch, laid out in the 14 by 14 grid"
        onClick={(e) => {
          const r = (e.target as HTMLCanvasElement).getBoundingClientRect()
          const col = Math.floor(((e.clientX - r.left) / r.width) * G), row = Math.floor(((e.clientY - r.top) / r.height) * G)
          if (col >= 0 && row >= 0 && col < G && row < G) onPick(row * G + col)
        }}
      />
      <i style={{ left: `${((sel % G) / G) * 100}%`, top: `${(Math.floor(sel / G) / G) * 100}%`, width: `${100 / G}%`, height: `${100 / G}%` }} />
    </span>
  )
}

/** Mean attention distance per head (dots) for every layer, like the paper's Figure 7 (right). */
export function DistanceChart({ dist, layer, head, onPick }: { dist: number[][]; layer: number; head: number; onPick: (l: number, h: number) => void }) {
  const L = dist.length
  const W = 420, Hh = 250, ml = 44, mr = 12, mt = 12, mb = 40
  const ymax = Math.max(140, Math.ceil(Math.max(...dist.flat()) / 20) * 20)
  const x = (l: number) => ml + ((l + 0.5) / L) * (W - ml - mr)
  const y = (v: number) => mt + (1 - v / ymax) * (Hh - mt - mb)
  const [tip, setTip] = useState<string | null>(null)
  const ticks = Array.from({ length: ymax / 20 + 1 }, (_, i) => i * 20).filter((v) => v % 40 === 0)
  return (
    <figure className="vit-demo-chart">
      <svg viewBox={`0 0 ${W} ${Hh}`} role="img" aria-label="Mean attention distance in pixels for every head of every layer">
        {ticks.map((v) => (
          <g key={v}>
            <line className="vit-demo-grid" x1={ml} x2={W - mr} y1={y(v)} y2={y(v)} />
            <text className="vit-demo-axis" x={ml - 6} y={y(v) + 4} textAnchor="end">{v}</text>
          </g>
        ))}
        {dist.map((_, l) => (
          <text key={l} className="vit-demo-axis" x={x(l)} y={Hh - mb + 16} textAnchor="middle">{l + 1}</text>
        ))}
        <text className="vit-demo-axis" x={(ml + W - mr) / 2} y={Hh - 6} textAnchor="middle">layer</text>
        <text className="vit-demo-axis" transform={`translate(12 ${(mt + Hh - mb) / 2}) rotate(-90)`} textAnchor="middle">pixels</text>
        <rect className="vit-demo-band" x={x(layer) - (W - ml - mr) / L / 2} width={(W - ml - mr) / L} y={mt} height={Hh - mt - mb} />
        {dist.map((row, l) =>
          row.map((v, h) => {
            const on = l === layer && h === head
            return (
              <circle
                key={`${l}-${h}`}
                className={`vit-demo-dot${on ? ' is-on' : ''}`}
                cx={x(l) + (h - (row.length - 1) / 2) * (row.length > 6 ? 1.2 : 3)}
                cy={y(v)}
                r={on ? 6 : 3.6}
                onClick={() => onPick(l, h)}
                onMouseEnter={() => setTip(`layer ${l + 1}, head ${h + 1}: ${v.toFixed(1)} pixels`)}
                onMouseLeave={() => setTip(null)}
              >
                <title>{`layer ${l + 1}, head ${h + 1}: ${v.toFixed(1)} pixels`}</title>
              </circle>
            )
          }),
        )}
      </svg>
      <figcaption>{tip ?? `Selected: layer ${layer + 1}, head ${head + 1}, ${dist[layer][head].toFixed(1)} pixels. Click a dot to pick its layer and head.`}</figcaption>
    </figure>
  )
}
