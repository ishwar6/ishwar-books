import { projects, site } from '../data/site'
import { useTitle } from '../lib/hooks'
import { ProjectCard } from '../components/Cards'

export default function Projects() {
  useTitle('Projects')
  return (
    <div className="container">
      <header className="page-head">
        <span className="eyebrow">Projects</span>
        <h1>Things I have built</h1>
        <p>Open-source code, course material and production templates. More on <a href={site.links.github} target="_blank" rel="noreferrer" style={{ color: 'var(--link)' }}>GitHub</a>.</p>
      </header>
      <div className="project-grid">{projects.map((p) => <ProjectCard key={p.url} project={p} />)}</div>
    </div>
  )
}
