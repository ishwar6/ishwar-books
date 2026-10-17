import { lazy, Suspense, useMemo, type ComponentType } from 'react'
import { Link, useParams } from 'react-router-dom'
import { papers } from '../lib/content'
import { useTitle } from '../lib/hooks'
import { accentStyle } from '../components/PaperBits'
import { Chevron } from '../components/icons'
import NotFound from './NotFound'
import '../styles/demo.css'

/** Each paper's interactive demo lives in src/demos/<paper>/Demo.tsx (its default export) and is loaded only on its
 *  own page, so its models and code never weigh on the rest of the site. src/demos/<paper>/meta.json gives the title
 *  and description, which scripts/build-content.mjs copies into the manifest (paper.demo). */
const modules = import.meta.glob<{ default: ComponentType }>('../demos/*/Demo.tsx')
const demos = Object.fromEntries(
  Object.entries(modules).map(([file, load]) => [file.split('/')[2], lazy(load)]),
)

export default function PaperDemo() {
  const { paper: slug } = useParams()
  const paper = papers.find((p) => p.slug === slug)
  const Demo = useMemo(() => (slug ? demos[slug] : undefined), [slug])
  useTitle(paper?.demo ? `${paper.demo.title} · ${paper.short}` : undefined, paper?.demo?.description)
  if (!paper?.demo || !Demo) return <NotFound />
  return (
    <div className="demo-page paper-theme" style={accentStyle(paper)}>
      <header className="demo-head">
        <nav className="breadcrumbs">
          <Link to="/">Home</Link><span className="sep"><Chevron /></span>
          <Link to="/papers">Research papers</Link><span className="sep"><Chevron /></span>
          <Link to={`/${paper.route}`}>{paper.short}</Link>
        </nav>
        <h1>{paper.demo.title}</h1>
        <p className="lede">{paper.demo.description}</p>
      </header>
      <Suspense fallback={<div className="demo-loading">Loading the demo…</div>}>
        <Demo />
      </Suspense>
    </div>
  )
}
