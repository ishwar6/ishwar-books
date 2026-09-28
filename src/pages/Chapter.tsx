import { Link, useParams } from 'react-router-dom'
import { useEffect, useState, type CSSProperties } from 'react'
import { books } from '../lib/content'
import { useTitle } from '../lib/hooks'
import Article from '../components/Article'
import ChapterNav from '../components/ChapterNav'
import { Chevron, Menu } from '../components/icons'
import NotFound from './NotFound'

export default function Chapter() {
  const { book: bookSlug, chapter } = useParams()
  const [navOpen, setNavOpen] = useState(false)
  const book = books.find((b) => b.slug === bookSlug)
  const idx = book?.chapters.findIndex((c) => c.slug === chapter) ?? -1
  const meta = book?.chapters[idx]
  useTitle(meta && book ? `${meta.title} · ${book.title}` : undefined, meta?.description || book?.description)
  useEffect(() => setNavOpen(false), [chapter])
  if (!book || !meta) return <NotFound />
  const prev = book.chapters[idx - 1]
  const next = book.chapters[idx + 1]

  return (
    <div style={{ '--book': book.accent } as CSSProperties}>
      <Article
        route={meta.route}
        sidebar={<ChapterNav book={book} current={meta.slug} open={navOpen} />}
        header={
          <header className="article-head">
            <button className="btn chapter-toggle" onClick={() => setNavOpen((o) => !o)}><Menu /> Chapters</button>
            <nav className="breadcrumbs">
              <Link to="/">Home</Link><span className="sep"><Chevron /></span>
              <Link to="/books">Books</Link><span className="sep"><Chevron /></span>
              <Link to={`/books/${book.slug}`}>{book.title}</Link>
            </nav>
            {meta.part && <div className="part-label">{meta.part}</div>}
            <h1>{meta.title}</h1>
            {meta.description && <p className="lede">{meta.description}</p>}
            <div className="meta"><span>Chapter {idx + 1} of {book.chapters.length}</span><span className="dot">{meta.minutes} min read</span></div>
          </header>
        }
        footer={
          <nav className="pager">
            {prev && <Link to={`/${prev.route}`}><span className="dir">← Previous</span><span className="t">{prev.title}</span></Link>}
            {next && <Link to={`/${next.route}`} className="next"><span className="dir">Next →</span><span className="t">{next.title}</span></Link>}
          </nav>
        }
      />
    </div>
  )
}
