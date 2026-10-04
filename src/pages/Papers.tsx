import { papers } from '../lib/content'
import { useTitle } from '../lib/hooks'
import { PaperCard } from '../components/PaperBits'

export default function Papers() {
  useTitle('Research Papers', 'Landmark research papers read slowly, section by section: highlighted excerpts, plain-English explanations, figures, definitions and code you can run.')
  return (
    <div className="container paper-theme">
      <header className="page-head papers-head">
        <span className="eyebrow">Research papers</span>
        <h1>Read the papers, <span className="grad">one section at a time</span></h1>
        <p>
          Each paper is split into a few parts that follow its own sections. Every part shows the exact lines of the paper,
          explains them in plain English, defines every new word, draws a picture, and runs real code to check the claims.
        </p>
      </header>
      {papers.length ? (
        <div className="paper-list">{papers.map((p) => <PaperCard key={p.slug} paper={p} />)}</div>
      ) : (
        <div className="empty">The first paper is being written.</div>
      )}
    </div>
  )
}
