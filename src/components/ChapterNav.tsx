import { Link } from 'react-router-dom'
import type { Book } from '../lib/content'

/** Chapters grouped by part, preserving book order. */
export function groupByPart(book: Book) {
  const groups: { part?: string; chapters: Book['chapters'] }[] = []
  for (const c of book.chapters) {
    const last = groups[groups.length - 1]
    if (last && last.part === c.part) last.chapters.push(c)
    else groups.push({ part: c.part, chapters: [c] })
  }
  return groups
}

export default function ChapterNav({ book, current, open }: { book: Book; current?: string; open: boolean }) {
  return (
    <aside className={`chapter-nav${open ? ' open' : ''}`}>
      <Link to={`/books/${book.slug}`} className="book-title">{book.title}</Link>
      {groupByPart(book).map((g, i) => (
        <div key={g.part ?? i}>
          {g.part && <div className="part">{g.part}</div>}
          {g.chapters.map((c) => (
            <Link key={c.slug} to={`/${c.route}`} className={`ch${c.slug === current ? ' active' : ''}`}>{c.title}</Link>
          ))}
        </div>
      ))}
    </aside>
  )
}
