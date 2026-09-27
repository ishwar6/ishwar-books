import { books } from '../lib/content'
import { useTitle } from '../lib/hooks'
import { BookCard } from '../components/Cards'

export default function Books() {
  useTitle('Books')
  return (
    <div className="container">
      <header className="page-head">
        <span className="eyebrow">Books</span>
        <h1>Books, from first principles</h1>
        <p>Complete, self-contained courses, each one written to be read in order, with code you run and numbers you measure at every step. Chapters are published as they are edited.</p>
      </header>
      <div className="book-grid">{books.map((b) => <BookCard key={b.slug} book={b} />)}</div>
    </div>
  )
}
