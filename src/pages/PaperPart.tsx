import { Link, useParams } from 'react-router-dom'
import { papers } from '../lib/content'
import { useTitle } from '../lib/hooks'
import Article from '../components/Article'
import { PaperSheet, PartNav, accentStyle } from '../components/PaperBits'
import { Chevron } from '../components/icons'
import NotFound from './NotFound'

export default function PaperPart() {
  const { paper: slug, part: partSlug } = useParams()
  const paper = papers.find((p) => p.slug === slug)
  const idx = paper ? paper.parts.findIndex((p) => p.slug === partSlug) : -1
  const meta = idx >= 0 ? paper!.parts[idx] : undefined
  useTitle(meta && paper ? `${paper.short}, Part ${meta.partNumber}: ${meta.title}` : undefined, meta?.description)
  if (!paper || !meta) return <NotFound />
  const prev = paper.parts[idx - 1]
  const next = paper.parts[idx + 1]
  return (
    <div style={accentStyle(paper)}>
      <Article
        route={meta.route}
        className="paper-theme"
        header={
          <header className="article-head paper-head">
            <nav className="breadcrumbs">
              <Link to="/">Home</Link><span className="sep"><Chevron /></span>
              <Link to="/papers">Research papers</Link><span className="sep"><Chevron /></span>
              <Link to={`/${paper.route}`}>{paper.short}</Link>
            </nav>
            <PaperSheet paper={paper} compact />
            <div className="part-label">Part {meta.partNumber} of {paper.parts.length}{meta.covers ? <span className="covers"> · Covers {meta.covers}</span> : null}</div>
            <h1>{meta.title}</h1>
            {meta.description && <p className="lede">{meta.description}</p>}
            <div className="meta"><span>{meta.minutes} min read</span>{meta.tags.map((t) => <span key={t} className="tag">{t}</span>)}</div>
            <PartNav paper={paper} current={meta.slug} />
          </header>
        }
        footer={
          <nav className="pager">
            {prev && <Link to={`/${prev.route}`}><span className="dir">← Part {prev.partNumber}</span><span className="t">{prev.title}</span></Link>}
            {next
              ? <Link to={`/${next.route}`} className="next"><span className="dir">Part {next.partNumber} →</span><span className="t">{next.title}</span></Link>
              : <Link to={`/${paper.route}`} className="next"><span className="dir">Back to the overview →</span><span className="t">{paper.short}, explained</span></Link>}
          </nav>
        }
      />
    </div>
  )
}
