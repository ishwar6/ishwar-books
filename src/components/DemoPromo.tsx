import { Link } from 'react-router-dom'
import { papers } from '../lib/content'
import { accentStyle } from './PaperBits'
import '../styles/demo.css'

/** The link to a paper's demo, for the series page. */
export function DemoPromo({ slug }: { slug: string }) {
  const paper = papers.find((p) => p.slug === slug)
  if (!paper?.demo) return null
  return (
    <Link to={`/${paper.route}/demo`} className="demo-promo" style={accentStyle(paper)}>
      <span className="dp-kicker">Interactive demo</span>
      <span className="dp-title">{paper.demo.title}</span>
      <span className="dp-text">{paper.demo.description}</span>
    </Link>
  )
}
