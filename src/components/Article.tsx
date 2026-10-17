import { useEffect, useState, type ReactNode } from 'react'
import { loadPage, formatDate, type Rendered } from '../lib/content'
import { site } from '../data/site'
import { Link } from 'react-router-dom'
import Prose from './Prose'
import Toc from './Toc'
import ReadingProgress from './ReadingProgress'

/** Fetches a pre-rendered page and lays it out with a header, body and table of contents. */
export default function Article({ route, header, sidebar, footer, className = '' }: { route: string; header: ReactNode; sidebar?: ReactNode; footer?: ReactNode; className?: string }) {
  const [page, setPage] = useState<Rendered | null>(null)
  const [error, setError] = useState(false)

  useEffect(() => {
    let live = true
    setPage(null)
    setError(false)
    loadPage(route).then((p) => live && setPage(p)).catch(() => live && setError(true))
    return () => { live = false }
  }, [route])

  useEffect(() => {
    if (page && location.hash) document.getElementById(decodeURIComponent(location.hash.slice(1)))?.scrollIntoView()
  }, [page])

  return (
    <div className={`container wide reader${sidebar ? ' with-sidebar' : ''}${className ? ` ${className}` : ''}`}>
      <ReadingProgress />
      {sidebar}
      <article className="reader-main">
        {header}
        {error ? (
          <div className="empty">This page could not be loaded. Try refreshing.</div>
        ) : page ? (
          <Prose html={page.html} />
        ) : (
          <div aria-busy="true">{[92, 100, 86, 97, 64].map((w, i) => <div key={i} className="skeleton" style={{ width: `${w}%` }} />)}</div>
        )}
        {page?.meta && <aside className="article-credit" aria-label="Author and citation">
          <p>Written by <Link to="/about" rel="author">{site.name}</Link>{page.meta.updated && <> · Updated <time dateTime={page.meta.updated}>{formatDate(page.meta.updated)}</time></>}</p>
          <details><summary>Cite or read this page as Markdown</summary>
            <p>{site.name}. {page.meta.title}. <a href={`${site.url}${import.meta.env.BASE_URL}${route}/`}>{site.url}{import.meta.env.BASE_URL}{route}/</a></p>
            <p><a href={`${import.meta.env.BASE_URL}_content/${route}.md`}>Read Markdown</a>{page.meta.source && <> · <a href={`https://github.com/ishwar6/ishwar-books/blob/main/${page.meta.source}`} target="_blank" rel="noreferrer">View source and revision history</a></>}</p>
          </details>
        </aside>}
        {footer}
      </article>
      <Toc items={page?.toc ?? []} />
    </div>
  )
}
