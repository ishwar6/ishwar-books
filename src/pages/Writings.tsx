import { useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import { writings } from '../lib/content'
import { useTitle } from '../lib/hooks'
import { WritingCard } from '../components/Cards'
import { SearchIcon } from '../components/icons'

export default function Writings() {
  useTitle('Writings')
  const [params, setParams] = useSearchParams()
  const tag = params.get('tag') ?? ''
  const q = params.get('q') ?? ''

  const tags = useMemo(() => {
    const counts = new Map<string, number>()
    writings.forEach((w) => w.tags.forEach((t) => counts.set(t, (counts.get(t) ?? 0) + 1)))
    return [...counts].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
  }, [])

  const shown = useMemo(() => {
    const needle = q.trim().toLowerCase()
    return writings.filter(
      (w) =>
        (!tag || w.tags.includes(tag)) &&
        (!needle || `${w.title} ${w.description} ${w.tags.join(' ')}`.toLowerCase().includes(needle)),
    )
  }, [tag, q])

  const update = (next: { tag?: string; q?: string }) => {
    const p = new URLSearchParams(params)
    for (const [k, v] of Object.entries(next)) v ? p.set(k, v) : p.delete(k)
    setParams(p, { replace: true })
  }
  const pickTag = (t: string) => update({ tag: t === tag ? '' : t })
  const filtering = Boolean(tag || q.trim())
  const [first, ...rest] = shown

  return (
    <div className="container">
      <header className="page-head writings-head">
        <span className="eyebrow">Writings</span>
        <h1>Essays and <span className="grad">deep dives</span></h1>
        <p>Long-form pieces on systems, machine learning and GPUs, written to be read start to finish. Every number measured, every diagram drawn from the real thing.</p>
      </header>

      <div className="writing-filters">
        <label className="filter-input">
          <SearchIcon />
          <input value={q} onChange={(e) => update({ q: e.target.value })} placeholder="Filter writings" aria-label="Filter writings" />
        </label>
        <div className="pills" role="toolbar" aria-label="Filter by tag">
          <button className="pill" aria-pressed={!tag} onClick={() => update({ tag: '' })}>
            All <span className="n">{writings.length}</span>
          </button>
          {tags.map(([t, n]) => (
            <button key={t} className="pill" aria-pressed={tag === t} onClick={() => pickTag(t)}>
              #{t} <span className="n">{n}</span>
            </button>
          ))}
        </div>
      </div>

      {!shown.length ? (
        <div className="empty">
          Nothing matches {tag && <strong>#{tag}</strong>} {q && <>“{q}”</>}.{' '}
          <button className="link-btn" onClick={() => update({ tag: '', q: '' })}>Clear filters</button>
        </div>
      ) : (
        <>
          {!filtering && first && <WritingCard post={first} featured onTag={pickTag} />}
          <div className="writing-grid">
            {(filtering ? shown : rest).map((w) => <WritingCard key={w.route} post={w} onTag={pickTag} />)}
          </div>
        </>
      )}
    </div>
  )
}
