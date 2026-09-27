import { Link } from 'react-router-dom'
import type { CSSProperties } from 'react'
import { formatDate, type Book, type PageMeta } from '../lib/content'
import type { Project } from '../data/site'
import { Arrow } from './icons'

const STATUS: Record<Book['status'], string> = { 'in-progress': 'In progress', complete: 'Complete', planned: 'Planned' }
const LANG_COLOR: Record<string, string> = { Python: '#3572a5', HCL: '#844fba', JavaScript: '#f1e05a', TypeScript: '#3178c6', CSS: '#663399' }

export function BookCard({ book }: { book: Book }) {
  const published = book.chapters.length
  return (
    <Link to={`/books/${book.slug}`} className="book-card" style={{ '--b': book.accent } as CSSProperties}>
      <span className={`status status-${book.status}`}>{STATUS[book.status]}</span>
      <h3>{book.title}</h3>
      {book.subtitle && <div className="subtitle">{book.subtitle}</div>}
      <p>{book.description}</p>
      <div className="tags">{book.topics.slice(0, 4).map((t) => <span key={t} className="tag">{t}</span>)}</div>
      <div className="foot">
        <span>{published ? `${published} of ${book.plannedChapters} chapters published` : `${book.plannedChapters} chapters · coming soon`}</span>
      </div>
    </Link>
  )
}

export function PostList({ posts }: { posts: PageMeta[] }) {
  if (!posts.length) return <div className="empty">Nothing published here yet.</div>
  return (
    <ul className="post-list">
      {posts.map((p) => (
        <li key={p.route}>
          <Link to={`/${p.route}`}>
            <time dateTime={p.date}>{formatDate(p.date)}</time>
            <div>
              <div className="title">{p.title}</div>
              {p.description && <div className="desc">{p.description}</div>}
            </div>
          </Link>
        </li>
      ))}
    </ul>
  )
}

export function ProjectCard({ project }: { project: Project }) {
  return (
    <a href={project.url} target="_blank" rel="noreferrer" className="project-card">
      <h3>{project.name}<Arrow /></h3>
      <p>{project.description}</p>
      <div className="tags" style={{ alignItems: 'center' }}>
        <span className="lang" style={{ '--lc': LANG_COLOR[project.language] } as CSSProperties}>{project.language}</span>
        {project.tags.map((t) => <span key={t} className="tag">{t}</span>)}
      </div>
    </a>
  )
}
