import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { books } from '../lib/content'
import { loadIndex, search, snippet, type SearchHit } from '../lib/search'
import { SearchIcon } from './icons'

function highlight(text: string, terms: string[]): ReactNode {
  if (!terms.length) return text
  const re = new RegExp(`(${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi')
  return text.split(re).map((part, i) => (i % 2 ? <mark key={i}>{part}</mark> : part))
}

/** Header button + ⌘K / "/" dialog. Inside a book it searches that book by default. */
export default function Search() {
  const [open, setOpen] = useState(false)
  const { pathname } = useLocation()
  const bookSlug = pathname.match(/^\/books\/([^/]+)/)?.[1]
  const book = books.find((b) => b.slug === bookSlug)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = /^(INPUT|TEXTAREA|SELECT)$/.test((e.target as HTMLElement).tagName)
      if ((e.key === 'k' && (e.metaKey || e.ctrlKey)) || (e.key === '/' && !typing)) {
        e.preventDefault()
        setOpen(true)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  useEffect(() => setOpen(false), [pathname])

  return (
    <>
      <button className="search-btn" onClick={() => setOpen(true)} onMouseEnter={() => loadIndex().catch(() => {})} aria-label="Search">
        <SearchIcon />
        <span className="label">{book ? 'Search this book' : 'Search'}</span>
        <kbd>⌘K</kbd>
      </button>
      {open && <SearchDialog book={book} onClose={() => setOpen(false)} />}
    </>
  )
}

function SearchDialog({ book, onClose }: { book?: (typeof books)[number]; onClose: () => void }) {
  const [query, setQuery] = useState('')
  const [scope, setScope] = useState<'book' | 'site'>(book ? 'book' : 'site')
  const [hits, setHits] = useState<SearchHit[]>([])
  const [loading, setLoading] = useState(false)
  const [active, setActive] = useState(0)
  const input = useRef<HTMLInputElement>(null)
  const list = useRef<HTMLUListElement>(null)
  const navigate = useNavigate()
  const bookTitle = useMemo(() => Object.fromEntries(books.map((b) => [b.slug, b.title])), [])

  useEffect(() => {
    input.current?.focus()
    document.body.style.overflow = 'hidden'
    return () => { document.body.style.overflow = '' }
  }, [])

  useEffect(() => {
    const q = query.trim()
    if (q.length < 2) { setHits([]); return }
    let live = true
    setLoading(true)
    const t = setTimeout(() => {
      search(q, scope === 'book' ? book?.slug : undefined)
        .then((h) => { if (live) { setHits(h); setActive(0) } })
        .catch(() => live && setHits([]))
        .finally(() => live && setLoading(false))
    }, 80)
    return () => { live = false; clearTimeout(t) }
  }, [query, scope, book])

  useEffect(() => {
    list.current?.querySelector<HTMLElement>(`[data-i="${active}"]`)?.scrollIntoView({ block: 'nearest' })
  }, [active])

  const go = (h: SearchHit) => {
    onClose()
    navigate(`/${h.r}${h.a ? `#${h.a}` : ''}`)
  }

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') onClose()
    else if (e.key === 'ArrowDown') { e.preventDefault(); setActive((a) => Math.min(a + 1, hits.length - 1)) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActive((a) => Math.max(a - 1, 0)) }
    else if (e.key === 'Enter' && hits[active]) go(hits[active])
    else if (e.key === 'Tab' && book) { e.preventDefault(); setScope((s) => (s === 'book' ? 'site' : 'book')) }
  }

  return (
    <div className="search-overlay" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="search-dialog" role="dialog" aria-modal="true" aria-label="Search" onKeyDown={onKey}>
        <div className="search-input">
          <SearchIcon />
          <input
            ref={input}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={scope === 'book' && book ? `Search ${book.title}` : 'Search the whole site'}
            aria-label="Search query"
            spellCheck={false}
          />
          <button className="esc" onClick={onClose} aria-label="Close search">Esc</button>
        </div>
        {book && (
          <div className="search-scope" role="tablist">
            <button role="tab" aria-selected={scope === 'book'} onClick={() => setScope('book')}>This book</button>
            <button role="tab" aria-selected={scope === 'site'} onClick={() => setScope('site')}>Entire site</button>
          </div>
        )}
        <ul className="search-results" ref={list}>
          {query.trim().length < 2 ? (
            <li className="search-empty">
              {scope === 'book' && book ? <>Searching <strong>{book.title}</strong>. Press Tab to search the entire site.</> : 'Search every writing, research paper and book chapter.'}
            </li>
          ) : loading && !hits.length ? (
            <li className="search-empty">Searching…</li>
          ) : !hits.length ? (
            <li className="search-empty">No results for “{query}”.</li>
          ) : (
            hits.map((h, i) => (
              <li key={h.id} data-i={i}>
                <button className={`search-hit${i === active ? ' active' : ''}`} onMouseMove={() => setActive(i)} onClick={() => go(h)}>
                  <span className="where">
                    {scope === 'site' && h.b && <span className="book">{bookTitle[h.b]}</span>}
                    <span className="page">{h.t}</span>
                  </span>
                  {h.h && <span className="heading">{highlight(h.h, h.terms)}</span>}
                  <span className="text">{highlight(snippet(h.x, h.terms), h.terms)}</span>
                </button>
              </li>
            ))
          )}
        </ul>
        <div className="search-foot"><span><kbd>↑</kbd><kbd>↓</kbd> to move</span><span><kbd>Enter</kbd> to open</span>{book && <span><kbd>Tab</kbd> switch scope</span>}<span><kbd>Esc</kbd> to close</span></div>
      </div>
    </div>
  )
}
