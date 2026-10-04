import { Link } from 'react-router-dom'
import type { CSSProperties } from 'react'
import type { PageMeta, Paper } from '../lib/content'
import { Arrow } from './icons'

const arxivAbs = (id: string) => `https://arxiv.org/abs/${id}`
const arxivPdf = (id: string) => `https://arxiv.org/pdf/${id}`

export const accentStyle = (paper: Paper) => ({ '--paper': paper.accent } as CSSProperties)

/** "Devlin, Chang, Lee, Toutanova" style author line. */
export const authorLine = (authors: string[]) => authors.join(', ')

/** The paper's title card: a small "sheet of paper" with the citation, links and (optionally) what you will learn. */
export function PaperSheet({ paper, compact = false }: { paper: Paper; compact?: boolean }) {
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
        {paper.arxiv && <a href={arxivAbs(paper.arxiv)} target="_blank" rel="noreferrer">arXiv page <Arrow /></a>}
        {paper.arxiv && <a href={arxivPdf(paper.arxiv)} target="_blank" rel="noreferrer">PDF <Arrow /></a>}
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
              <span className="pn">Part {p.partNumber}</span>
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
        <span className="ps-short">{paper.short}</span>
        <span className="ps-lines"><i /><i /><i /><i /><i /><i /></span>
        <span className="ps-year">{paper.year}</span>
      </div>
      <div className="paper-card-body">
        <div className="paper-cite">{[paper.venue, paper.arxiv && `arXiv:${paper.arxiv}`].filter(Boolean).join(' · ')}</div>
        <h3><Link to={`/${paper.route}`}>{paper.title}</Link></h3>
        <div className="paper-authors">{authorLine(paper.authors)}</div>
        <p>{paper.description}</p>
        <div className="paper-card-foot">
          <span>{paper.parts.length} {paper.parts.length === 1 ? 'part' : 'parts'} · {paper.minutes} min of reading</span>
          {first && <Link to={`/${first.route}`} className="btn primary">Start with Part 1</Link>}
        </div>
      </div>
    </article>
  )
}
