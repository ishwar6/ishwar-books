import { useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import graph from '../data/graph.json'
import { useTitle } from '../lib/hooks'

type Node = { id: string; label: string; group: keyof typeof graph.groups; route: string; blurb: string }
type Pos = { x: number; y: number }
const W = 960, H = 720

/** A small force-directed layout: springs on edges, repulsion between every pair, a pull to the centre.
 *  Deterministic (seeded by group and index), so the map looks the same on every visit. */
function layout(nodes: Node[], edges: string[][]): Record<string, Pos> {
  const groups = Object.keys(graph.groups)
  const pos: Record<string, Pos & { vx: number; vy: number }> = {}
  nodes.forEach((n, i) => {
    const a = (groups.indexOf(n.group) / groups.length) * Math.PI * 2 + i * 0.37
    pos[n.id] = { x: W / 2 + Math.cos(a) * (180 + (i % 5) * 22), y: H / 2 + Math.sin(a) * (130 + (i % 3) * 20), vx: 0, vy: 0 }
  })
  const deg: Record<string, number> = {}
  edges.forEach(([a, b]) => { deg[a] = (deg[a] ?? 0) + 1; deg[b] = (deg[b] ?? 0) + 1 })
  for (let it = 0; it < 420; it++) {
    const t = 1 - it / 420
    for (const a of nodes) for (const b of nodes) {
      if (a.id >= b.id) continue
      const pa = pos[a.id], pb = pos[b.id]
      let dx = pa.x - pb.x, dy = pa.y - pb.y
      const d2 = Math.max(dx * dx + dy * dy, 1), d = Math.sqrt(d2)
      const f = 15000 / d2
      dx /= d; dy /= d
      pa.vx += dx * f; pa.vy += dy * f; pb.vx -= dx * f; pb.vy -= dy * f
    }
    for (const [a, b] of edges) {
      const pa = pos[a], pb = pos[b]
      const dx = pb.x - pa.x, dy = pb.y - pa.y, d = Math.max(Math.sqrt(dx * dx + dy * dy), 1)
      const f = (d - 150) * 0.02
      pa.vx += (dx / d) * f; pa.vy += (dy / d) * f; pb.vx -= (dx / d) * f; pb.vy -= (dy / d) * f
    }
    for (const n of nodes) {
      const p = pos[n.id]
      p.vx += (W / 2 - p.x) * 0.004; p.vy += (H / 2 - p.y) * 0.006
      p.x += p.vx * 0.5 * (0.3 + t); p.y += p.vy * 0.5 * (0.3 + t); p.vx *= 0.6; p.vy *= 0.6
      const r = 14 + 3 * (deg[n.id] ?? 1)
      p.x = Math.min(W - 70 - r, Math.max(70 + r, p.x)); p.y = Math.min(H - 40 - r, Math.max(40 + r, p.y))
    }
  }
  // push apart nodes whose label boxes overlap (labels sit under the dot, up to two lines)
  const box = (n: Node) => {
    const r = 14 + 3 * (deg[n.id] ?? 1), words = n.label.split(' ')
    const longest = words.length > 3 ? Math.max(words.slice(0, Math.ceil(words.length / 2)).join(' ').length, words.slice(Math.ceil(words.length / 2)).join(' ').length) : n.label.length
    return { w: Math.max(2 * r, longest * 7.4) + 10, top: r, bottom: r + 18 + (words.length > 3 ? 15 : 0) + 6 }
  }
  for (let it = 0; it < 80; it++) {
    for (const a of nodes) for (const b of nodes) {
      if (a.id >= b.id) continue
      const pa = pos[a.id], pb = pos[b.id], ba = box(a), bb = box(b)
      const ox = (ba.w + bb.w) / 2 - Math.abs(pa.x - pb.x)
      const oy = Math.min(pa.y + ba.bottom, pb.y + bb.bottom) - Math.max(pa.y - ba.top, pb.y - bb.top)
      if (ox > 0 && oy > 0) {
        if (ox < oy) { const s = Math.sign(pa.x - pb.x) || 1; pa.x += s * ox / 2; pb.x -= s * ox / 2 }
        else { const s = Math.sign(pa.y - pb.y) || 1; pa.y += s * oy / 2; pb.y -= s * oy / 2 }
      }
    }
    for (const n of nodes) { const p = pos[n.id], b = box(n); p.x = Math.min(W - b.w / 2 - 6, Math.max(b.w / 2 + 6, p.x)); p.y = Math.min(H - b.bottom - 4, Math.max(b.top + 8, p.y)) }
  }
  return pos
}

export default function TopicMap() {
  useTitle('Topic map', 'Every important topic on this site, as one connected map: attention, BERT, ViT, LLM inference, RAG, agents and GPUs. Click a topic to read about it.')
  const nodes = graph.nodes as Node[]
  const edges = graph.edges
  const pos = useMemo(() => layout(nodes, edges), [nodes, edges])
  const deg = useMemo(() => { const d: Record<string, number> = {}; edges.forEach(([a, b]) => { d[a] = (d[a] ?? 0) + 1; d[b] = (d[b] ?? 0) + 1 }); return d }, [edges])
  const [active, setActive] = useState<string | null>(null)
  const [pinned, setPinned] = useState<string | null>(null)
  const navigate = useNavigate()
  const sel = pinned ?? active
  const near = useMemo(() => { const s = new Set<string>(); if (sel) { s.add(sel); edges.forEach(([a, b]) => { if (a === sel) s.add(b); if (b === sel) s.add(a) }) } return s }, [sel, edges])
  const selNode = nodes.find((n) => n.id === sel)
  const color = (n: Node) => graph.groups[n.group].color

  return (
    <div className="container">
      <header className="page-head">
        <span className="eyebrow">Topic map</span>
        <h1>Everything on this site, connected</h1>
        <p>Each dot is one important idea from the writings, books, videos and paper breakdowns. Lines join ideas that depend on each other. Hover to see what a topic is; click to read about it.</p>
      </header>

      <div className="map-legend">
        {Object.entries(graph.groups).map(([k, g]) => <span key={k} className="map-key"><i style={{ background: g.color }} />{g.label}</span>)}
      </div>

      <div className="map-wrap">
        <svg className="map-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="A graph of the site's topics, grouped by colour and joined by lines" onMouseLeave={() => setActive(null)}>
          {edges.map(([a, b]) => {
            const on = sel ? a === sel || b === sel : false
            const dim = sel ? !on : false
            return <line key={a + b} className={`map-edge${on ? ' on' : ''}${dim ? ' dim' : ''}`} x1={pos[a].x} y1={pos[a].y} x2={pos[b].x} y2={pos[b].y} />
          })}
          {nodes.map((n) => {
            const p = pos[n.id], r = 12 + 2.5 * (deg[n.id] ?? 1)
            const dim = sel ? !near.has(n.id) : false
            const words = n.label.split(' ')
            const lines = words.length > 3 ? [words.slice(0, Math.ceil(words.length / 2)).join(' '), words.slice(Math.ceil(words.length / 2)).join(' ')] : [n.label]
            return (
              <g key={n.id} className={`map-node${dim ? ' dim' : ''}${sel === n.id ? ' sel' : ''}`} transform={`translate(${p.x},${p.y})`}
                 onMouseEnter={() => setActive(n.id)} onFocus={() => setActive(n.id)}
                 onClick={(e) => { e.preventDefault(); if (pinned === n.id || window.matchMedia('(hover: hover)').matches) navigate(`/${n.route}`); else setPinned(n.id) }}
                 tabIndex={0} role="link" aria-label={`${n.label}: ${n.blurb}`}>
                <circle r={r + 6} className="map-halo" style={{ fill: color(n) }} />
                <circle r={r} style={{ fill: color(n) }} />
                {lines.map((l, i) => <rect key={'b' + i} className="map-label-bg" x={-(l.length * 7.3 + 12) / 2} y={r + 5 + i * 16} width={l.length * 7.3 + 12} height={16} rx={5} />)}
                {lines.map((l, i) => <text key={i} y={r + 17 + i * 16} textAnchor="middle" className="map-label">{l}</text>)}
              </g>
            )
          })}
        </svg>
        <aside className={`map-panel${selNode ? ' show' : ''}`} aria-live="polite">
          {selNode ? (
            <>
              <span className="map-key"><i style={{ background: color(selNode) }} />{graph.groups[selNode.group].label}</span>
              <h3>{selNode.label}</h3>
              <p>{selNode.blurb}</p>
              <Link to={`/${selNode.route}`} className="more-link">Read about it →</Link>
              {pinned && <button className="map-clear" onClick={() => setPinned(null)}>Clear</button>}
            </>
          ) : <p className="map-hint">Hover or tap a topic.</p>}
        </aside>
      </div>

      <section className="map-list">
        {Object.entries(graph.groups).map(([k, g]) => (
          <div key={k} className="map-list-group">
            <h2><i style={{ background: g.color }} />{g.label}</h2>
            <ul>{nodes.filter((n) => n.group === k).map((n) => <li key={n.id}><Link to={`/${n.route}`}>{n.label}</Link><span>{n.blurb}</span></li>)}</ul>
          </div>
        ))}
      </section>
    </div>
  )
}
