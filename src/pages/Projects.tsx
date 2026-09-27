import { projects } from '../data/site'
import { useTitle } from '../lib/hooks'
import { ProjectCard } from '../components/Cards'

export default function Projects() {
  useTitle('Projects')
  return (
    <div className="container">
      <header className="page-head">
        <span className="eyebrow">Projects</span>
        <h1>Things I have built</h1>
        <p>Tools and systems I build, mostly around explaining hard ideas well. More are on the way.</p>
      </header>
      <div className="project-list">{projects.map((p) => <ProjectCard key={p.name} project={p} full />)}</div>
    </div>
  )
}
