import { Link, useParams } from 'react-router-dom'
import { formatDate, writings } from '../lib/content'
import { useTitle } from '../lib/hooks'
import Article from '../components/Article'
import { Chevron } from '../components/icons'
import NotFound from './NotFound'

export default function Post() {
  const { slug } = useParams()
  const meta = writings.find((w) => w.slug === slug)
  useTitle(meta?.title)
  if (!meta) return <NotFound />
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
        </header>
      }
    />
  )
}
