import { Link, useParams } from 'react-router-dom'
import { useCallback, useEffect, useRef } from 'react'
import { formatDate, videos } from '../lib/content'
import { useTitle } from '../lib/hooks'
import Article from '../components/Article'
import { Chevron, YouTube } from '../components/icons'
import NotFound from './NotFound'

export default function Video() {
  const { slug } = useParams()
  const v = videos.find((x) => x.slug === slug)
  useTitle(v?.title, v?.description)
  const frame = useRef<HTMLIFrameElement>(null)

  // Jump the embedded player to a moment (YouTube IFrame API via postMessage).
  const seek = useCallback((t: number) => {
    const win = frame.current?.contentWindow
    if (!win) return
    const cmd = (func: string, args: unknown[] = []) => win.postMessage(JSON.stringify({ event: 'command', func, args }), '*')
    cmd('seekTo', [t, true])
    cmd('playVideo')
    frame.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }, [])

  useEffect(() => {
    const onSeek = (e: Event) => seek((e as CustomEvent<number>).detail)
    window.addEventListener('video-seek', onSeek)
    return () => window.removeEventListener('video-seek', onSeek)
  }, [seek])

  if (!v) return <NotFound />
  const watch = `https://www.youtube.com/watch?v=${v.youtube}`
  const origin = typeof window !== 'undefined' ? encodeURIComponent(window.location.origin) : ''

  return (
    <Article
      route={v.route}
      header={
        <header className="article-head">
          <nav className="breadcrumbs">
            <Link to="/">Home</Link><span className="sep"><Chevron /></span>
            <Link to="/videos">Videos</Link>
          </nav>
          {v.series && <div className="part-label">{v.series} · Part {v.seriesPart}</div>}
          <h1>{v.title}</h1>
          <p className="lede">{v.description}</p>
          <div className="meta">
            {v.date && <time dateTime={v.date}>{formatDate(v.date)}</time>}
            {v.duration && <span className="dot">{v.duration} video</span>}
            <a className="yt-link" href={watch} target="_blank" rel="noreferrer"><YouTube /> Watch on YouTube</a>
          </div>
          <div className="player">
            <iframe
              ref={frame}
              src={`https://www.youtube-nocookie.com/embed/${v.youtube}?enablejsapi=1&rel=0&modestbranding=1&origin=${origin}`}
              title={v.title}
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
              referrerPolicy="strict-origin-when-cross-origin"
              allowFullScreen
            />
          </div>
          {v.chapters.length > 0 && (
            <details className="chapters" open>
              <summary>Chapters <span>{v.chapters.length}</span></summary>
              <ol>
                {v.chapters.map((c) => (
                  <li key={c.t}>
                    <button onClick={() => seek(c.t)}><span className="stamp">{c.stamp}</span>{c.label}</button>
                  </li>
                ))}
              </ol>
            </details>
          )}
        </header>
      }
    />
  )
}
