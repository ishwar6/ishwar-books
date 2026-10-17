import { Link, useParams } from 'react-router-dom'
import { papers } from '../lib/content'
import { useTitle } from '../lib/hooks'
import Article from '../components/Article'
import { PaperSheet, PartNav, ReaderPromo, accentStyle } from '../components/PaperBits'
import { Chevron } from '../components/icons'
import { DemoButton, DemoPromo } from '../components/DemoPromo'
import NotFound from './NotFound'

export default function Paper() {
  const { paper: slug } = useParams()
  const paper = papers.find((p) => p.slug === slug)
  useTitle(paper ? `${paper.short}, explained` : undefined, paper?.description)
  if (!paper) return <NotFound />
  return (
    <div style={accentStyle(paper)}>
      <Article
        route={paper.route}
        className="paper-theme"
        header={
          <header className="article-head paper-head">
            <nav className="breadcrumbs">
              <Link to="/">Home</Link><span className="sep"><Chevron /></span>
              <Link to="/papers">Research papers</Link>
            </nav>
            <h1>{paper.short}, explained</h1>
            <p className="lede">{paper.description}</p>
            {paper.demo && <div className="part-demo"><DemoButton paper={paper} label={`Try ${paper.short} in your browser`} /></div>}
            <PaperSheet paper={paper} />
            <DemoPromo slug={paper.slug} />
            {paper.reader && <ReaderPromo paper={paper} compact />}
          </header>
        }
        footer={
          <section className="paper-parts">
            <h2>The parts</h2>
            <PartNav paper={paper} />
          </section>
        }
      />
    </div>
  )
}
