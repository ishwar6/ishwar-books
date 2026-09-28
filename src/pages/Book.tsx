import { Link, useParams } from 'react-router-dom'
import type { CSSProperties } from 'react'
import { books } from '../lib/content'
import { useTitle } from '../lib/hooks'
import Article from '../components/Article'
import { groupByPart } from '../components/ChapterNav'
import { Chevron } from '../components/icons'
import NotFound from './NotFound'

export default function Book() {
  const { book: slug } = useParams()
  const book = books.find((b) => b.slug === slug)
  useTitle(book?.title, book?.description)
  if (!book) return <NotFound />
  let n = 0
  return (
    <div style={{ '--book': book.accent } as CSSProperties}>
      <Article
        route={`books/${book.slug}`}
        header={
          <header className="article-head">
            <nav className="breadcrumbs">
              <Link to="/">Home</Link><span className="sep"><Chevron /></span>
              <Link to="/books">Books</Link>
            </nav>
            <div className="part-label">{book.subtitle}</div>
            <h1>{book.title}</h1>
            <p className="lede">{book.description}</p>
            <div className="tags">{book.topics.map((t) => <span key={t} className="tag">{t}</span>)}</div>
          </header>
        }
        footer={
          <section style={{ marginTop: 56 }}>
            <h2 style={{ fontSize: 28, fontWeight: 800, color: 'var(--text-strong)', letterSpacing: '-0.02em', margin: '0 0 8px' }}>Chapters</h2>
            {book.chapters.length ? (
              <ol className="chapter-list">
                {groupByPart(book).map((g, i) => (
                  <li key={g.part ?? i}>
                    {g.part && <div className="part-head">{g.part}</div>}
                    {g.chapters.map((c) => (
                      <Link key={c.slug} to={`/${c.route}`}>
                        <span className="n">{String(++n).padStart(2, '0')}</span>
                        <span className="t">{c.title}</span>
                        <span className="m">{c.minutes} min</span>
                      </Link>
                    ))}
                  </li>
                ))}
              </ol>
            ) : (
              <div className="empty">{book.plannedChapters} chapters are being edited for publication. Check back soon.</div>
            )}
          </section>
        }
      />
    </div>
  )
}
