import { Link } from 'react-router-dom'
import { projects, site } from '../data/site'
import { books, videos, writings } from '../lib/content'
import { useTitle } from '../lib/hooks'
import { BookCard, ProjectCard, VideoCard, WritingCard } from '../components/Cards'
import { GitHub, Mail, X, YouTube } from '../components/icons'

export default function Home() {
  useTitle()
  const chapters = books.reduce((n, b) => n + b.plannedChapters, 0)
  const explore = [
    { to: '/writings', count: writings.length, label: 'Writings', desc: 'Long-form essays and deep dives, written from first principles.' },
    { to: '/books', count: books.length, label: 'Books', desc: `${chapters}+ chapters on GPUs, LLMs, RAG and agents.` },
    { to: '/projects', count: projects.length, label: 'Projects', desc: 'Tools and pipelines I build, like an explainer-video studio.' },
    { to: '/videos', count: videos.length, label: 'Videos', desc: 'Animated explainers, each with a written companion.' },
  ]

  return (
    <div className="container">
      <section className="hero">
        <div>
          <h1>Hey, I am <span className="wave">{site.firstName}</span></h1>
          <div className="tagline">{site.tagline}</div>
          <div className="bio">{site.bio.map((p) => <p key={p}>{p}</p>)}</div>
          <div className="socials">
            <a className="btn primary" href={`mailto:${site.email}`}><Mail /> Get in touch</a>
            <a className="btn" href={site.links.github} target="_blank" rel="noreferrer"><GitHub /> GitHub</a>
            <a className="btn" href={site.links.youtube} target="_blank" rel="noreferrer"><YouTube /> YouTube</a>
            <a className="btn" href={site.links.x} target="_blank" rel="noreferrer"><X /> X</a>
          </div>
        </div>
        <div className="hero-avatar" aria-hidden="true">IJ</div>
      </section>

      <section className="section">
        <div className="section-head"><h2>Explore my work</h2></div>
        <div className="explore">
          {explore.map((e) => {
            const body = (
              <>
                <span className="count">{e.count}</span>
                <span className="label">{e.label}</span>
                <span className="desc">{e.desc}</span>
              </>
            )
            return <Link key={e.label} to={e.to} className="explore-card">{body}</Link>
          })}
        </div>
      </section>

      {videos.length > 0 && (
        <section className="section">
          <div className="section-head">
            <h2>Latest video</h2>
            <Link to="/videos" className="more-link">All videos →</Link>
          </div>
          <VideoCard video={videos[0]} featured />
        </section>
      )}

      <section className="section">
        <div className="section-head">
          <h2>Books</h2>
          <Link to="/books" className="more-link">All books →</Link>
        </div>
        <div className="book-grid">{books.map((b) => <BookCard key={b.slug} book={b} />)}</div>
      </section>

      <section className="section">
        <div className="section-head">
          <h2>Recent writings</h2>
          <Link to="/writings" className="more-link">All writings →</Link>
        </div>
        <div className="writing-grid">{writings.slice(0, 3).map((w) => <WritingCard key={w.route} post={w} />)}</div>
      </section>

      <section className="section">
        <div className="section-head">
          <h2>Projects</h2>
          <Link to="/projects" className="more-link">All projects →</Link>
        </div>
        <div className="project-grid">{projects.slice(0, 3).map((p) => <ProjectCard key={p.name} project={p} />)}</div>
      </section>
    </div>
  )
}
