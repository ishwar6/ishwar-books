import { useEffect, useState, type ReactNode } from 'react'
import { loadPage, type Rendered } from '../lib/content'
import Prose from './Prose'
import Toc from './Toc'
import ReadingProgress from './ReadingProgress'

/** Fetches a pre-rendered page and lays it out with a header, body and table of contents. */
export default function Article({ route, header, sidebar, footer }: { route: string; header: ReactNode; sidebar?: ReactNode; footer?: ReactNode }) {
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
    <div className={`container reader${sidebar ? ' with-sidebar' : ''}`}>
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
        {footer}
      </article>
      <Toc items={page?.toc ?? []} />
    </div>
  )
}
