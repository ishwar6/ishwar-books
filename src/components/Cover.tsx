import { useMemo } from 'react'
import type { PageMeta } from '../lib/content'

// Generated cover art. Deterministic per article (seeded by its slug), so a new
// writing gets a cover with no image to make. Frontmatter `cover:` overrides it
// with a real image; `motif` and `accent` steer the generated one.

type Motif = NonNullable<PageMeta['motif']>
const W = 640
const H = 360
const ACCENTS = ['#7c9cff', '#c4a1ff', '#5fd4b0', '#ff9ecf', '#f5b942', '#38bdf8']
const TAG_MOTIF: Record<string, Motif> = {
  hnsw: 'graph', 'vector-search': 'graph', retrieval: 'graph', rag: 'graph', agents: 'graph',
  gpu: 'grid', cuda: 'grid', performance: 'grid', kernels: 'grid',
  meta: 'code', writing: 'code', python: 'code', tooling: 'code',
  llm: 'waves', training: 'waves', ml: 'waves', inference: 'timeline', 'kv-cache': 'kv', vllm: 'blocks',
}

function hash(s: string) {
  let h = 2166136261
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619)
  return h >>> 0
}

function rng(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function motifOf(meta: PageMeta): Motif {
  if (meta.motif) return meta.motif
  for (const t of meta.tags) if (TAG_MOTIF[t]) return TAG_MOTIF[t]
  return (['graph', 'code', 'grid', 'waves'] as const)[hash(meta.slug) % 4]
}

function Graph({ r, a, gid }: { r: () => number; a: string; gid: string }) {
  const planes = [
    { y: 80, n: 5 },
    { y: 185, n: 11 },
    { y: 290, n: 24 },
  ]
  const pts = planes.map((p, li) =>
    Array.from({ length: p.n }, () => ({ x: 120 + r() * 400, y: p.y + (r() - 0.5) * 44, l: li })),
  )
  const edges: [number, number, number][] = []
  pts.forEach((layer, li) =>
    layer.forEach((p, i) => {
      const near = layer
        .map((q, j) => ({ j, d: Math.hypot(q.x - p.x, q.y - p.y) }))
        .filter((o) => o.j !== i)
        .sort((x, y) => x.d - y.d)
        .slice(0, li === 2 ? 3 : 2)
      near.forEach((o) => edges.push([li, i, o.j]))
    }),
  )
  // a descending search path: best-first toward a target on the bottom layer
  const target = pts[2][Math.floor(r() * pts[2].length)]
  const closest = (layer: typeof pts[number]) =>
    layer.reduce((best, q) => (Math.abs(q.x - target.x) < Math.abs(best.x - target.x) ? q : best))
  const path = [pts[0][0], closest(pts[0]), closest(pts[1]), target].filter((p, i, all) => i === 0 || p !== all[i - 1])
  return (
    <g>
      {planes.map((p) => (
        <polygon key={p.y} points={`90,${p.y - 42} 590,${p.y - 42} 550,${p.y + 42} 50,${p.y + 42}`} fill="#ffffff" fillOpacity="0.035" stroke="#ffffff" strokeOpacity="0.1" />
      ))}
      {edges.map(([li, i, j], k) => (
        <line key={k} x1={pts[li][i].x} y1={pts[li][i].y} x2={pts[li][j].x} y2={pts[li][j].y} stroke="#ffffff" strokeOpacity="0.18" strokeWidth="1.5" />
      ))}
      {pts[1].concat(pts[0]).map((p, i) => (
        <line key={`v${i}`} x1={p.x} y1={p.y + 8} x2={p.x} y2={290 + 30} stroke="#ffffff" strokeOpacity="0.06" strokeDasharray="2 5" />
      ))}
      <g filter={`url(#${gid})`}>
        {path.map((p, i) =>
          i ? <line key={`p${i}`} x1={path[i - 1].x} y1={path[i - 1].y} x2={p.x} y2={p.y} stroke={a} strokeWidth="3.5" strokeLinecap="round" /> : null,
        )}
      </g>
      {pts.flat().map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={p.l === 0 ? 7 : p.l === 1 ? 5.5 : 4.5} fill={path.includes(p) ? a : '#e9e9f0'} fillOpacity={path.includes(p) ? 1 : 0.55} />
      ))}
      <circle cx={target.x} cy={target.y} r="12" fill="none" stroke={a} strokeWidth="2.5" filter={`url(#${gid})`} />
      <circle cx={target.x} cy={target.y} r="22" fill="none" stroke={a} strokeOpacity="0.35" strokeWidth="1.5" />
    </g>
  )
}

function Code({ r, a }: { r: () => number; a: string }) {
  const colors = [a, '#5fd4b0', '#f5b942', '#ff9ecf', '#e9e9f0']
  const lines = Array.from({ length: 9 }, (_, i) => {
    const indent = i === 0 || i === 8 ? 0 : [1, 1, 2, 2, 1, 2, 1][i - 1]
    let x = 120 + indent * 34
    return Array.from({ length: 2 + Math.floor(r() * 3) }, () => {
      const w = 26 + r() * 90
      const seg = { x, w, c: colors[Math.floor(r() * colors.length)] }
      x += w + 12
      return seg
    }).filter((s) => s.x + s.w < 540)
  })
  return (
    <g>
      <rect x="84" y="46" width="472" height="268" rx="18" fill="#0b0b12" fillOpacity="0.55" stroke="#ffffff" strokeOpacity="0.12" />
      {['#ff5f57', '#febc2e', '#28c840'].map((c, i) => <circle key={c} cx={112 + i * 20} cy={72} r="6" fill={c} fillOpacity="0.85" />)}
      {lines.map((segs, li) =>
        segs.map((s, si) => <rect key={`${li}-${si}`} x={s.x} y={102 + li * 23} width={s.w} height="9" rx="4.5" fill={s.c} fillOpacity={s.c === '#e9e9f0' ? 0.35 : 0.8} />),
      )}
      <rect x={120 + 34} y={102 + 4 * 23 - 6} width="4" height="21" fill={a} />
    </g>
  )
}

function Grid({ r, a }: { r: () => number; a: string }) {
  const cells = []
  const cx = 8 + r() * 10
  const cy = 3 + r() * 4
  for (let y = 0; y < 9; y++)
    for (let x = 0; x < 18; x++) {
      const d = Math.hypot(x - cx, (y - cy) * 1.6)
      const v = Math.max(0, 1 - d / 9) * (0.55 + r() * 0.45)
      cells.push(<rect key={`${x}-${y}`} x={86 + x * 26} y={62 + y * 26} width="21" height="21" rx="5" fill={v > 0.12 ? a : '#ffffff'} fillOpacity={v > 0.12 ? 0.15 + v * 0.85 : 0.05} />)
    }
  return <g>{cells}</g>
}

function Waves({ r, a }: { r: () => number; a: string }) {
  const waves = Array.from({ length: 7 }, (_, i) => {
    const amp = 18 + r() * 40
    const f = 0.008 + r() * 0.012
    const ph = r() * 6.28
    const y0 = 90 + i * 30
    const d = Array.from({ length: 65 }, (_, k) => {
      const x = k * 10
      return `${k ? 'L' : 'M'}${x},${(y0 + Math.sin(x * f + ph) * amp * Math.sin((x / W) * Math.PI)).toFixed(1)}`
    }).join('')
    return <path key={i} d={d} fill="none" stroke={i === 3 ? a : '#ffffff'} strokeOpacity={i === 3 ? 1 : 0.14 + i * 0.03} strokeWidth={i === 3 ? 3.5 : 2} />
  })
  return <g>{waves}</g>
}

function Timeline({ r, a }: { r: () => number; a: string }) {
  const rows = [0, 1, 2, 3]
  return (
    <g>
      {rows.map((i) => {
        const y = 92 + i * 52
        const pre = 90 + r() * 130
        const n = 9 + Math.floor(r() * 7)
        const out = [<rect key={`p${i}`} x={90} y={y} width={pre} height={30} rx={7} fill={a} fillOpacity={i === 1 ? 1 : 0.55} />]
        let x = 90 + pre + 6
        for (let k = 0; k < n && x < 560; k++) {
          out.push(<rect key={`d${i}-${k}`} x={x} y={y} width={18} height={30} rx={4} fill="#ff9ecf" fillOpacity={i === 1 ? 0.95 : 0.4} />)
          x += 24
        }
        return <g key={i}>{out}</g>
      })}
      <line x1="90" y1="300" x2="560" y2="300" stroke="#ffffff" strokeOpacity="0.15" strokeWidth="2" />
    </g>
  )
}

function Kv({ r, a }: { r: () => number; a: string }) {
  const cells = []
  const n = 11
  for (let row = 0; row < n; row++)
    for (let col = 0; col <= row; col++) {
      const fresh = col === row
      cells.push(
        <rect key={`${row}-${col}`} x={150 + col * 32} y={52 + row * 24} width={26} height={18} rx={4}
          fill={fresh ? '#ff9ecf' : a} fillOpacity={fresh ? 0.95 : 0.25 + (col / n) * 0.55 * (0.7 + r() * 0.3)} />,
      )
    }
  return <g>{cells}</g>
}

function Blocks({ r, a }: { r: () => number; a: string }) {
  const hues = [a, '#ff9ecf', '#5fd4b0', '#f5b942']
  const cells = []
  for (let y = 0; y < 6; y++)
    for (let x = 0; x < 12; x++) {
      const v = r()
      const owner = v < 0.2 ? -1 : Math.floor(r() * 4)
      cells.push(
        <rect key={`${x}-${y}`} x={100 + x * 38} y={70 + y * 38} width={32} height={32} rx={6}
          fill={owner < 0 ? '#ffffff' : hues[owner]} fillOpacity={owner < 0 ? 0.05 : 0.35 + r() * 0.55} />,
      )
    }
  return <g>{cells}</g>
}

export default function Cover({ meta, className = '' }: { meta: PageMeta; className?: string }) {
  const art = useMemo(() => {
    const seed = hash(meta.slug)
    const r = rng(seed)
    const a = meta.accent ?? ACCENTS[seed % ACCENTS.length]
    const motif = motifOf(meta)
    const body =
      motif === 'graph' ? <Graph r={r} a={a} gid={`c${seed.toString(36)}-glow`} />
      : motif === 'code' ? <Code r={r} a={a} />
      : motif === 'grid' ? <Grid r={r} a={a} />
      : motif === 'timeline' ? <Timeline r={r} a={a} />
      : motif === 'kv' ? <Kv r={r} a={a} />
      : motif === 'blocks' ? <Blocks r={r} a={a} />
      : <Waves r={r} a={a} />
    return { a, body, id: `c${seed.toString(36)}` }
  }, [meta])

  if (meta.cover) return <div className={`cover ${className}`}><img src={meta.cover} alt="" loading="lazy" /></div>

  return (
    <div className={`cover ${className}`} aria-hidden="true">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid slice">
        <defs>
          <radialGradient id={`${art.id}-g1`} cx="18%" cy="12%" r="75%">
            <stop offset="0" stopColor={art.a} stopOpacity="0.55" />
            <stop offset="1" stopColor={art.a} stopOpacity="0" />
          </radialGradient>
          <radialGradient id={`${art.id}-g2`} cx="92%" cy="100%" r="70%">
            <stop offset="0" stopColor="#ff9ecf" stopOpacity="0.28" />
            <stop offset="1" stopColor="#ff9ecf" stopOpacity="0" />
          </radialGradient>
          <filter id={`${art.id}-glow`} x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="4" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <pattern id={`${art.id}-dots`} width="22" height="22" patternUnits="userSpaceOnUse">
            <circle cx="2" cy="2" r="1" fill="#ffffff" fillOpacity="0.07" />
          </pattern>
        </defs>
        <rect width={W} height={H} fill="#0d0e16" />
        <rect width={W} height={H} fill={`url(#${art.id}-g1)`} />
        <rect width={W} height={H} fill={`url(#${art.id}-g2)`} />
        <rect width={W} height={H} fill={`url(#${art.id}-dots)`} />
        {art.body}
      </svg>
    </div>
  )
}
