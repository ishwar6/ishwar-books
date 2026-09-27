import { Link, NavLink, Outlet, useLocation } from 'react-router-dom'
import { useEffect } from 'react'
import { site } from '../data/site'
import { books, writings } from '../lib/content'
import { useFontSize, useTheme } from '../lib/hooks'
import Search from './Search'
import { Dim, GitHub, Mail, Moon, Sun, X, YouTube } from './icons'

function ScrollToTop() {
  const { pathname, hash } = useLocation()
  useEffect(() => {
    if (hash) document.getElementById(decodeURIComponent(hash.slice(1)))?.scrollIntoView()
    else window.scrollTo(0, 0)
  }, [pathname, hash])
  return null
}

const THEME_LABEL = { dark: 'Dark', dim: 'Dim', light: 'Light' } as const

function FontSizeControl() {
  const [size, setSize] = useFontSize()
  const opts = [['small', 'A-', 'sm', 'Smaller text'], ['normal', 'A', '', 'Normal text'], ['large', 'A+', 'lg', 'Larger text']] as const
  return (
    <div className="font-size" role="group" aria-label="Text size">
      {opts.map(([value, label, cls, title]) => (
        <button key={value} className={cls} aria-pressed={size === value} title={title} onClick={() => setSize(value)}>{label}</button>
      ))}
    </div>
  )
}

export default function Layout() {
  const [theme, toggle] = useTheme()
  const { pathname } = useLocation()
  const reading = /^\/(writings|books)\/[^/]+/.test(pathname)
  return (
    <>
      <ScrollToTop />
      <header className="site-header">
        <div className={`container${reading ? ' wide' : ''}`}>
          <Link to="/" className="brand">
            <span className="brand-mark">IJ</span>
            {site.name}
          </Link>
          <nav className="nav">
            <NavLink to="/writings">Writings</NavLink>
            <NavLink to="/books">Books</NavLink>
            <NavLink to="/projects">Projects</NavLink>
            <NavLink to="/about">About</NavLink>
          </nav>
          <div className="header-actions">
            <Search />
            {reading && <FontSizeControl />}
            <button className="icon-btn" onClick={toggle} aria-label={`Theme: ${THEME_LABEL[theme]}. Switch theme`} title={`Theme: ${THEME_LABEL[theme]}`}>
              {theme === 'dark' ? <Moon /> : theme === 'dim' ? <Dim /> : <Sun />}
            </button>
          </div>
        </div>
      </header>
      <main>
        <Outlet />
      </main>
      <footer className="site-footer">
        <div className="container">
          <div className="footer-grid">
            <div>
              <div className="brand" style={{ marginBottom: 12 }}>
                <span className="brand-mark">IJ</span>
                {site.name}
              </div>
              <p style={{ margin: 0, maxWidth: 420, lineHeight: 1.6 }}>{site.tagline}</p>
              <div className="socials" style={{ marginTop: 18 }}>
                <a className="icon-btn" href={site.links.github} target="_blank" rel="noreferrer" aria-label="GitHub"><GitHub /></a>
                <a className="icon-btn" href={site.links.youtube} target="_blank" rel="noreferrer" aria-label="YouTube"><YouTube /></a>
                <a className="icon-btn" href={site.links.x} target="_blank" rel="noreferrer" aria-label="X"><X /></a>
                <a className="icon-btn" href={`mailto:${site.email}`} aria-label="Email"><Mail /></a>
              </div>
            </div>
            <div>
              <h4>Read</h4>
              <Link to="/writings">Writings ({writings.length})</Link>
              <Link to="/books">Books ({books.length})</Link>
              <Link to="/projects">Projects</Link>
            </div>
            <div>
              <h4>Connect</h4>
              <Link to="/about">About</Link>
              <a href={`mailto:${site.email}`}>Email</a>
              <a href={site.links.youtube} target="_blank" rel="noreferrer">YouTube</a>
            </div>
          </div>
          <div className="footer-bottom">© {new Date().getFullYear()} {site.name}. Written in Markdown, built with React.</div>
        </div>
      </footer>
    </>
  )
}
