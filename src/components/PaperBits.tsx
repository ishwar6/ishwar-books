import { Link } from 'react-router-dom'
import type { CSSProperties } from 'react'
import type { PageMeta, Paper } from '../lib/content'
import { Arrow } from './icons'
import { DemoButton, Play } from './DemoPromo'

const arxivAbs = (id: string) => `https://arxiv.org/abs/${id}`
const arxivPdf = (id: string) => `https://arxiv.org/pdf/${id}`

export const accentStyle = (paper: Paper) => ({ '--paper': paper.accent } as CSSProperties)

/** "Devlin, Chang, Lee, Toutanova" style author line. */
export const authorLine = (authors: string[]) => authors.join(', ')

/** The paper's title card: a small "sheet of paper" with the citation, links and (optionally) what you will learn. */
export function PaperSheet({ paper, compact = false, part }: { paper: Paper; compact?: boolean; part?: string }) {
  return (
    <section className={`paper-sheet${compact ? ' compact' : ''}`} aria-label="About the paper">
      <div className="paper-sheet-top">
        <span className="paper-kicker">Research paper</span>
        <span className="paper-cite">{[paper.venue, paper.arxiv && `arXiv:${paper.arxiv}`].filter(Boolean).join(' · ')}</span>
      </div>
      {compact ? (
        <Link to={`/${paper.route}`} className="paper-title">{paper.title}</Link>
      ) : (
        <div className="paper-title">{paper.title}</div>
      )}
      <div className="paper-authors">
        {authorLine(paper.authors)}
        {paper.org && <span className="paper-org"> · {paper.org}</span>}
        {paper.year && <span className="paper-org"> · {paper.year}</span>}
      </div>
      <div className="paper-links">
        {paper.reader && <Link to={`/${paper.route}/read${part ? `?part=${part}` : ''}`} className="paper-read">Read beside the paper <Arrow /></Link>}
        {paper.arxiv && <a href={arxivAbs(paper.arxiv)} target="_blank" rel="noreferrer">arXiv page <Arrow /></a>}
        {paper.arxiv && <a href={arxivPdf(paper.arxiv)} target="_blank" rel="noreferrer">PDF <Arrow /></a>}
        {paper.url && <a href={paper.url} target="_blank" rel="noreferrer">Paper <Arrow /></a>}
        {paper.code && <a href={paper.code} target="_blank" rel="noreferrer">Original code <Arrow /></a>}
        {!compact && <span className="paper-time">{paper.parts.length} {paper.parts.length === 1 ? 'part' : 'parts'} · about {paper.minutes} min of reading</span>}
      </div>
      {!compact && paper.learn.length > 0 && (
        <div className="paper-learn">
          <h2>What you will learn</h2>
          <ul>{paper.learn.map((l) => <li key={l}>{l}</li>)}</ul>
        </div>
      )}
    </section>
  )
}

/** Part 1..N of a paper, with the current part highlighted. */
export function PartNav({ paper, current }: { paper: Paper; current?: string }) {
  return (
    <nav className="part-nav" aria-label={`Parts of the ${paper.short} breakdown`}>
      <ol>
        {paper.parts.map((p) => {
          const on = p.slug === current
          const body = (
            <>
              <span className="pn">{p.partLabel}</span>
              <span className="pt">{p.title}</span>
              {p.covers && <span className="pc">{p.covers}</span>}
            </>
          )
          return (
            <li key={p.slug} className={on ? 'current' : ''}>
              {on ? <span aria-current="page">{body}</span> : <Link to={`/${p.route}`}>{body}</Link>}
            </li>
          )
        })}
      </ol>
    </nav>
  )
}

export function PaperCard({ paper }: { paper: Paper }) {
  const first = paper.parts[0] as PageMeta | undefined
  return (
    <article className="paper-card" style={accentStyle(paper)}>
      <div className="paper-card-sheet" aria-hidden="true">
        <span className="ps-short" style={{ ['--ps-size' as string]: `${Math.min(46, Math.floor(172 / (0.58 * paper.short.length)))}px` }}>{paper.short}</span>
        <span className="ps-lines"><i /><i /><i /><i /><i /><i /></span>
        <span className="ps-year">{paper.year}</span>
      </div>
      <div className="paper-card-body">
        <div className="paper-cite">{[paper.venue, paper.arxiv && `arXiv:${paper.arxiv}`].filter(Boolean).join(' · ')}</div>
        <h3><Link to={`/${paper.route}`}>{paper.title}</Link></h3>
        {paper.reader && <span className="reader-badge"><SideBySide /> Side-by-side reader</span>}
        {paper.demo && <span className="reader-badge"><Play /> Interactive demo</span>}
        <div className="paper-authors">{authorLine(paper.authors)}</div>
        <p>{paper.description}</p>
        <div className="paper-card-foot">
          <span>{paper.parts.length} {paper.parts.length === 1 ? 'part' : 'parts'} · {paper.minutes} min of reading</span>
          <span className="paper-card-actions">
            {paper.demo && <DemoButton paper={paper} />}
            {paper.reader && <Link to={`/${paper.route}/read`} className="btn reader-btn"><SideBySide /> Read side by side</Link>}
            {first && <Link to={`/${first.route}`} className="btn primary">Start with Part 1</Link>}
          </span>
        </div>
      </div>
    </article>
  )
}

/** Two panes: the paper and its notes. */
export function SideBySide() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
      <rect x="2.5" y="4" width="8" height="16" rx="1.5" /><rect x="13.5" y="4" width="8" height="16" rx="1.5" />
      <path d="M5 9h3M5 12h3M16 9h3M16 12h3M16 15h2" />
    </svg>
  )
}

/** The banner that shows the side-by-side reader: a small moving picture of the two panes, and a button. */
export function ReaderPromo({ paper, compact = false }: { paper: Paper; compact?: boolean }) {
  return (
    <section className={`reader-promo${compact ? ' compact' : ''}`} style={accentStyle(paper)} aria-label="Read the paper side by side">
      <div className="rp-art" aria-hidden="true">
        <div className="rp-sheet">
          <i /><i className="hl a" /><i className="hl a" /><i /><i /><i className="hl b" /><i className="hl b" /><i />
          <b className="rp-dot a">1</b><b className="rp-dot b">2</b>
        </div>
        <div className="rp-notes">
          <div className="rp-card a"><span>1</span><i /><i /><i /></div>
          <div className="rp-card b"><span>2</span><i /><i /><i /></div>
        </div>
      </div>
      <div className="rp-text">
        <span className="rp-kicker"><SideBySide /> New: read side by side</span>
        <h2>The real paper on the left, its explanation on the right</h2>
        <p>
          The {paper.short} paper opens with its key lines highlighted and numbered. As you scroll, the explanation of each
          highlighted passage moves into view beside it. Click a passage to see its explanation, or a number to find it in the paper.
        </p>
        <Link to={`/${paper.route}/read`} className="btn reader-btn big"><SideBySide /> Read {paper.short} side by side</Link>
      </div>
    </section>
  )
}
