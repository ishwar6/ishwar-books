import { Link } from 'react-router-dom'
import { useTitle } from '../lib/hooks'

export default function NotFound() {
  useTitle('Not found')
  return (
    <div className="container">
      <header className="page-head">
        <span className="eyebrow">404</span>
        <h1>This page does not exist</h1>
        <p>It may not be published yet. <Link to="/" style={{ color: 'var(--link)' }}>Go home →</Link></p>
      </header>
    </div>
  )
}
