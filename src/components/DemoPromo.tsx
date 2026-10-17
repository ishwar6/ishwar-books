import { Link } from 'react-router-dom'
import type { CSSProperties } from 'react'
import { papers, type Paper } from '../lib/content'
import { Arrow } from './icons'
import { demoGuides } from '../demos/catalog'
import '../styles/demo.css'

const BASE = import.meta.env.BASE_URL
const accentStyle = (paper: Paper) => ({ '--paper': paper.accent } as CSSProperties)
/** Every paper that has an interactive demo (src/demos/<paper>/meta.json), in the papers' order. */
export const demos = papers.filter((p) => p.demo)

/** A cursor over a frame: "try it". */
export function Play() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="3" y="4" width="18" height="13" rx="2" />
      <path d="M11 9l2.5 8 1.2-3.3 3.3-1.2z" fill="currentColor" />
    </svg>
  )
}

/** A card for one demo: its picture, its name, what it does. */
export function DemoCard({ paper }: { paper: Paper }) {
  if (!paper.demo) return null
  const guide = demoGuides[paper.slug]
  return (
    <Link to={`/${paper.route}/demo`} className="demo-card" style={accentStyle(paper)}>
      <span className="dc-preview"><img src={`${BASE}demos/${paper.slug}/thumb.jpg`} alt="" loading="lazy" width={800} height={420} /><span className="dc-model">{paper.short}</span><span className="dc-preview-arrow"><Arrow /></span></span>
      <span className="dc-body">
        <span className="dp-kicker">{guide?.category ?? 'Interactive demo'}</span>
        <span className="dp-title">{guide?.title ?? paper.demo.title}</span>
        <span className="dp-text">{guide?.summary ?? paper.demo.description}</span>
        {guide && <span className="dc-actions">{guide.actions.map((action) => <span key={action}>{action}</span>)}</span>}
        <span className="dc-foot"><span>{guide?.mode ?? 'In your browser'}</span><span className="dc-go">Explore demo <Arrow /></span></span>
      </span>
    </Link>
  )
}

/** The demos as a grid of cards. */
export function DemoGrid({ only }: { only?: string }) {
  const list = only ? demos.filter((p) => p.slug === only) : demos
  return list.length ? <div className="demo-grid">{list.map((p) => <DemoCard key={p.slug} paper={p} />)}</div> : null
}

/** The big link on a paper's series page. */
export function DemoPromo({ slug }: { slug: string }) {
  const paper = papers.find((p) => p.slug === slug)
  if (!paper?.demo) return null
  return (
    <Link to={`/${paper.route}/demo`} className="demo-promo" style={accentStyle(paper)}>
      <img src={`${BASE}demos/${paper.slug}/thumb.jpg`} alt="" loading="lazy" width={800} height={420} />
      <span className="dc-body">
        <span className="dp-kicker"><Play /> Interactive demo</span>
        <span className="dp-title">{paper.demo.title}</span>
        <span className="dp-text">{paper.demo.description}</span>
        <span className="btn demo-btn">Try it in your browser</span>
      </span>
    </Link>
  )
}

/** A small button, for a part's header and a paper card. */
export function DemoButton({ paper, label = 'Try the demo' }: { paper: Paper; label?: string }) {
  if (!paper.demo) return null
  return <Link to={`/${paper.route}/demo`} className="btn demo-btn" style={accentStyle(paper)}><Play /> {label}</Link>
}
