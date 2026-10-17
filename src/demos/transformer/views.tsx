/* The drawing pieces of the Transformer demo: segmented controls, attention grids (rows = queries, columns = keys),
   a row of tokens shaded by attention, a heatmap of vectors, the position-encoding waves and the top-5 bars.
   Chrome and attention cells use theme tokens; the vector heatmaps are data canvases with their own colours. */
import { useEffect, useRef, type ReactNode } from 'react'
import { diverging, fix, pct } from './txMath'

export function Seg<T extends string | number>({ value, options, onChange, label, small }: {
  value: T; options: { v: T; label: ReactNode; title?: string }[]; onChange: (v: T) => void; label: string; small?: boolean
}) {
  return (
    <div className={`transformer-demo-seg${small ? ' is-small' : ''}`} role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button key={String(o.v)} type="button" role="radio" aria-checked={value === o.v} title={o.title} onClick={() => onChange(o.v)}>
          {o.label}
        </button>
      ))}
    </div>
  )
}

/** How a token is shown: the word-start mark of SentencePiece (▁) becomes a gap, a space becomes ␣. */
export function tokText(t: string): string {
  if (t === ' ') return '␣'
  return t.startsWith('▁') ? t.slice(1) || '▁' : t
}

/** Attention weights as a grid: one row per query, one column per key. Cells are shaded by weight (0 to 1).
 *  masked(i, j) marks cells that the mask hides. Hover or tap a row to pick it. */
export function AttnGrid({ rows, cols, w, sel, onSel, masked, numbers, label, rowTitle, colTitle, maxPx = 34 }: {
  rows: string[]; cols: string[]; w: ArrayLike<number>; sel?: number | null; onSel?: (i: number) => void
  masked?: (i: number, j: number) => boolean; numbers?: boolean; label: string; rowTitle?: string; colTitle?: string; maxPx?: number
}) {
  const nc = cols.length
  return (
    <div className="transformer-demo-gridwrap">
      {colTitle && <div className="transformer-demo-coltitle">{colTitle}</div>}
      <div
        className="transformer-demo-grid"
        role="grid"
        aria-label={label}
        style={{ gridTemplateColumns: `max-content repeat(${nc}, minmax(0, ${maxPx}px))` }}
      >
        <span className="transformer-demo-corner">{rowTitle}</span>
        {cols.map((c, j) => <span key={`c${j}`} className="transformer-demo-collab" title={c}>{tokText(c)}</span>)}
        {rows.map((r, i) => (
          <div key={`r${i}`} role="row" className={`transformer-demo-row${sel === i ? ' is-sel' : ''}`} style={{ display: 'contents' }}>
            <button type="button" className="transformer-demo-rowlab" title={r} aria-pressed={sel === i}
              onClick={() => onSel?.(i)} onMouseEnter={() => onSel?.(i)} onFocus={() => onSel?.(i)}>
              {tokText(r)}
            </button>
            {cols.map((_, j) => {
              const v = w[i * nc + j]
              const hide = masked?.(i, j)
              return (
                <span
                  key={j}
                  role="gridcell"
                  className={`transformer-demo-cell${hide ? ' is-masked' : ''}${sel === i ? ' is-sel' : ''}`}
                  style={hide ? undefined : { ['--w' as string]: `${Math.round(Math.min(1, v) * 100)}%` }}
                  title={`${tokText(r)} → ${tokText(cols[j])}: ${hide ? 'masked' : fix(v, 3)}`}
                  onMouseEnter={() => onSel?.(i)}
                >
                  {numbers && !hide && v >= 0.005 ? (v >= 0.995 ? '1' : v.toFixed(2).slice(1)) : ''}
                </span>
              )
            })}
          </div>
        ))}
      </div>
    </div>
  )
}

/** A row of tokens shaded by the weight that one query puts on each of them, with the weight under it. */
export function TokenWeights({ toks, w, q, label }: { toks: string[]; w: ArrayLike<number>; q?: number | null; label: string }) {
  return (
    <div className="transformer-demo-tokw" role="list" aria-label={label}>
      {toks.map((t, j) => (
        <span key={j} role="listitem" className={`transformer-demo-tokwi${q === j ? ' is-q' : ''}`}
          style={{ ['--w' as string]: `${Math.round(Math.min(1, w[j]) * 100)}%` }}>
          <b>{tokText(t)}</b>
          <small>{w[j] >= 0.995 ? '1.00' : fix(w[j], 2)}</small>
        </span>
      ))}
    </div>
  )
}

/** A matrix of vectors (rows x d) as a heatmap: orange positive, blue negative. */
export function VecHeat({ data, rows, d, scale, labels, label, sel }: {
  data: Float32Array; rows: number; d: number; scale: number; labels: string[]; label: string; sel?: number | null
}) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    if (!c) return
    const ctx = c.getContext('2d')!
    const img = ctx.createImageData(d, rows)
    for (let i = 0; i < rows * d; i++) {
      const [r, g, b] = diverging(data[i] / scale)
      img.data.set([r, g, b, 255], 4 * i)
    }
    ctx.putImageData(img, 0, 0)
  }, [data, rows, d, scale])
  return (
    <div className="transformer-demo-vec">
      <div className="transformer-demo-veclabs" style={{ gridTemplateRows: `repeat(${rows}, 16px)` }}>
        {labels.map((l, i) => <span key={i} className={sel === i ? 'is-sel' : ''}>{tokText(l)}</span>)}
      </div>
      <canvas ref={ref} width={d} height={rows} role="img" aria-label={label} style={{ height: rows * 16 }} />
    </div>
  )
}

/** The position-encoding waves: PE(pos, dim) against pos for a few dimensions, with a dot at each token. */
export function PEWaves({ pe, d, dims, n, maxPos, off }: { pe: Float32Array; d: number; dims: number[]; n: number; maxPos: number; off: boolean }) {
  const W = 560, H = 46, ml = 70, mr = 10
  const x = (p: number) => ml + (p / maxPos) * (W - ml - mr)
  const fine = 8
  const val = (pos: number, k: number) => {
    // the continuous wave for drawing; the dots use the table the model gets
    const i = Math.floor(k / 2)
    const a = pos / 10000 ** ((2 * i) / d)
    return k % 2 === 0 ? Math.sin(a) : Math.cos(a)
  }
  return (
    <svg className="transformer-demo-waves" viewBox={`0 0 ${W} ${H * dims.length + 22}`} role="img"
      aria-label={`Position encoding values for dimensions ${dims.join(', ')} at positions 0 to ${maxPos}`}>
      {dims.map((k, r) => {
        const y0 = r * H + H / 2
        const y = (v: number) => y0 - v * (H / 2 - 6)
        let path = ''
        for (let s = 0; s <= maxPos * fine; s++) {
          const p = s / fine
          path += `${s ? 'L' : 'M'}${x(p).toFixed(1)},${y(off ? 0 : val(p, k)).toFixed(1)}`
        }
        return (
          <g key={k}>
            <text className="transformer-demo-axis" x={ml - 8} y={y0 + 4} textAnchor="end">dim {k} {k % 2 ? 'cos' : 'sin'}</text>
            <line className="transformer-demo-gridline" x1={ml} x2={W - mr} y1={y0} y2={y0} />
            <path className="transformer-demo-wave" d={path} />
            {Array.from({ length: n }, (_, p) => (
              <circle key={p} className="transformer-demo-wavedot" cx={x(p)} cy={y(pe[p * d + k])} r={3}>
                <title>{`position ${p}, dimension ${k}: ${fix(pe[p * d + k], 4)}`}</title>
              </circle>
            ))}
          </g>
        )
      })}
      {Array.from({ length: Math.floor(maxPos / 5) + 1 }, (_, i) => i * 5).map((p) => (
        <text key={p} className="transformer-demo-axis" x={x(p)} y={H * dims.length + 16} textAnchor="middle">{p}</text>
      ))}
    </svg>
  )
}

export function TopBars({ items, chosen }: { items: { label: string; p: number }[]; chosen?: number }) {
  return (
    <ol className="transformer-demo-top">
      {items.map((it, i) => (
        <li key={i} className={chosen === i ? 'is-chosen' : ''}>
          <span className="transformer-demo-topname">{it.label}</span>
          <span className="transformer-demo-topbar"><i style={{ width: `${Math.max(0.5, 100 * it.p)}%` }} /></span>
          <span className="transformer-demo-topp">{pct(it.p)}</span>
        </li>
      ))}
    </ol>
  )
}
