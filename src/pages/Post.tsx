import { Link, useParams } from 'react-router-dom'
import { formatDate, seriesOf, writings } from '../lib/content'
import { useTitle } from '../lib/hooks'
import Article from '../components/Article'
import { Chevron } from '../components/icons'
import NotFound from './NotFound'

export default function Post() {
  const { slug } = useParams()
  const meta = writings.find((w) => w.slug === slug)
  useTitle(meta?.title)
  if (!meta) return <NotFound />
  const parts = seriesOf(meta.series)
  const idx = parts.findIndex((p) => p.slug === meta.slug)
  const next = idx >= 0 ? parts[idx + 1] : undefined
  const prev = idx > 0 ? parts[idx - 1] : undefined
  const seriesBox = parts.length > 1 && (
    <nav className="series-box" aria-label="Series">
      <div className="series-name"><span>Series</span>{meta.series}</div>
      <ol>
        {parts.map((p) => (
          <li key={p.slug} className={p.slug === meta.slug ? 'current' : ''}>
            {p.slug === meta.slug ? <span>{p.title}</span> : <Link to={`/${p.route}`}>{p.title}</Link>}
          </li>
        ))}
      </ol>
    </nav>
  )
  return (
    <Article
      route={meta.route}
      header={
        <header className="article-head">
          <nav className="breadcrumbs">
            <Link to="/">Home</Link><span className="sep"><Chevron /></span>
            <Link to="/writings">Writings</Link>
          </nav>
          <h1>{meta.title}</h1>
          {meta.description && <p className="lede">{meta.description}</p>}
          <div className="meta">
            {meta.date && <time dateTime={meta.date}>{formatDate(meta.date)}</time>}
            <span className={meta.date ? 'dot' : ''}>{meta.minutes} min read</span>
            {meta.tags.map((t) => <span key={t} className="tag">{t}</span>)}
          </div>
          {seriesBox}
        </header>
      }
      footer={
        parts.length > 1 && (
          <nav className="pager">
            {prev && <Link to={`/${prev.route}`}><span className="dir">← Part {prev.seriesPart}</span><span className="t">{prev.title}</span></Link>}
            {next && <Link to={`/${next.route}`} className="next"><span className="dir">Part {next.seriesPart} →</span><span className="t">{next.title}</span></Link>}
          </nav>
        )
      }
    />
  )
}
