import { Link } from 'react-router-dom'
import type { CSSProperties } from 'react'
import { formatDate, type Book, type PageMeta } from '../lib/content'
import type { Project } from '../data/site'
import { Arrow } from './icons'
import Cover from './Cover'

const STATUS: Record<Book['status'], string> = { 'in-progress': 'In progress', complete: 'Complete', planned: 'Planned' }

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

export function WritingCard({ post, featured = false, onTag }: { post: PageMeta; featured?: boolean; onTag?: (t: string) => void }) {
  return (
    <article className={`writing-card${featured ? ' featured' : ''}`}>
      <Link to={`/${post.route}`} className="writing-cover-link" tabIndex={-1} aria-hidden="true">
        <Cover meta={post} className="writing-cover" />
      </Link>
      <div className="writing-body">
        {post.series && <div className="series-badge" title={post.series}>{featured ? `${post.series} · Part ${post.seriesPart}` : `Series · Part ${post.seriesPart}`}</div>}
        <div className="writing-meta">
          <time dateTime={post.date}>{formatDate(post.date)}</time>
          <span className="dot">{post.minutes} min read</span>
        </div>
        <h3><Link to={`/${post.route}`}>{post.title}</Link></h3>
        {post.description && <p>{post.description}</p>}
        {post.tags.length > 0 && (
          <div className="tags">
            {post.tags.map((t) =>
              onTag
                ? <button key={t} className="tag tag-btn" onClick={() => onTag(t)}>#{t}</button>
                : <span key={t} className="tag">#{t}</span>,
            )}
          </div>
        )}
      </div>
    </article>
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

export function ProjectCard({ project, full = false }: { project: Project; full?: boolean }) {
  const body = (
    <>
      {project.status && <span className="project-status">{project.status}</span>}
      <h3>{project.name}{project.url && <Arrow />}</h3>
      <div className="project-tagline">{project.tagline}</div>
      <p>{project.description}</p>
      {full && project.highlights.length > 0 && (
        <ul className="project-highlights">{project.highlights.map((h) => <li key={h}>{h}</li>)}</ul>
      )}
      <div className="tags">{project.tech.map((t) => <span key={t} className="tag">{t}</span>)}</div>
    </>
  )
  const cls = `project-card${full ? ' full' : ''}`
  return project.url
    ? <a href={project.url} target="_blank" rel="noreferrer" className={cls}>{body}</a>
    : <div className={cls}>{body}</div>
}
