import { writings } from '../lib/content'
import { useTitle } from '../lib/hooks'
import { PostList } from '../components/Cards'

export default function Writings() {
  useTitle('Writings')
  return (
    <div className="container">
      <header className="page-head">
        <span className="eyebrow">Writings</span>
        <h1>Essays and deep dives</h1>
        <p>Long-form pieces on systems, machine learning and GPUs, written to be read start to finish.</p>
      </header>
      <PostList posts={writings} />
    </div>
  )
}
