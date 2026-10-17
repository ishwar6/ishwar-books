import { useEffect, useRef, useState, type CSSProperties } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { Arrow, Chevron, SearchIcon } from '../components/icons'
import graph from '../data/graph.json'
import paths from '../data/learning-paths.json'
import { useTitle } from '../lib/hooks'
import '../styles/topic-map.css'

type Group = keyof typeof graph.groups
type Topic = { id: string; label: string; group: Group; route: string; blurb: string }
const topics = graph.nodes as Topic[]
const groups = Object.entries(graph.groups) as [Group, { label: string; color: string }][]
const byId = new Map(topics.map((topic) => [topic.id, topic]))
const accent = (group: Group) => ({ '--topic-color': graph.groups[group].color }) as CSSProperties
const normalize = (value: string) => value.toLowerCase().normalize('NFKD').replace(/[\u0300-\u036f]/g, '').replace(/[^a-z0-9]+/g, ' ').trim()
const resourceType = (topic: Topic) => topic.route.startsWith('books/') ? 'Book' : topic.route.startsWith('papers/') ? 'Paper breakdown' : topic.route.startsWith('videos/') ? 'Video' : 'Article'
const topicHref = (topic: Topic) => {
  const [route, hash] = topic.route.split('#')
  return `/${route.replace(/\/$/, '')}/${hash ? `#${hash}` : ''}`
}

function GroupLabel({ group }: { group: Group }) {
  return <span className="tm-group-label" style={accent(group)}><i aria-hidden="true" />{graph.groups[group].label}</span>
}

export default function TopicMap() {
  useTitle('Topic map', 'Find a learning path through attention, LLM inference, RAG, agents and GPUs. Search topics, explore their connections, and jump to the exact explanation.')
  const [params, setParams] = useSearchParams()
  const [copyStatus, setCopyStatus] = useState('')
  const explorerRef = useRef<HTMLElement>(null)
  const detailRef = useRef<HTMLElement>(null)
  const listRef = useRef<HTMLDivElement>(null)
  const query = params.get('q') ?? ''
  const groupParam = params.get('group')
  const group = groups.some(([id]) => id === groupParam) ? groupParam as Group : 'all'
  const path = paths.find((item) => item.id === params.get('path'))
  const words = normalize(query).split(' ').filter(Boolean)
  const candidates = path ? path.topics.map((id) => byId.get(id)!) : topics.filter((topic) => group === 'all' || topic.group === group)
  const visible = candidates.filter((topic) => words.every((word) => normalize(`${topic.label} ${topic.blurb} ${graph.groups[topic.group].label}`).includes(word)))
  const selected = visible.find((topic) => topic.id === params.get('topic')) ?? visible[0]
  const neighborIds = new Set(graph.edges.flatMap(([a, b]) => a === selected?.id ? [b] : b === selected?.id ? [a] : []))
  const relatedGroups = groups.map(([id, value]) => ({ id, ...value, topics: topics.filter((topic) => topic.group === id && neighborIds.has(topic.id)) })).filter((item) => item.topics.length)
  const pathIndex = selected && path ? path.topics.indexOf(selected.id) : -1
  const nextTopic = path && pathIndex >= 0 ? byId.get(path.topics[pathIndex + 1]) : undefined
  const listTitle = path?.title ?? (group === 'all' ? 'All topics' : graph.groups[group].label)

  // Keep deep links, connected topics, and browser back/forward visible in the list.
  useEffect(() => {
    const list = listRef.current
    const item = list?.querySelector<HTMLElement>('[aria-pressed="true"]')
    if (!list || !item) return
    const bounds = list.getBoundingClientRect()
    const row = item.getBoundingClientRect()
    if (row.top < bounds.top) list.scrollTop += row.top - bounds.top - 8
    else if (row.bottom > bounds.bottom) list.scrollTop += row.bottom - bounds.bottom + 8
    setCopyStatus('')
  }, [selected?.id])

  function resetList() {
    setCopyStatus('')
    listRef.current?.scrollTo({ top: 0 })
  }
  function chooseGroup(id: string) {
    setParams(id === 'all' ? {} : { group: id })
    resetList()
  }
  function choosePath(id: string) {
    setParams({ path: id })
    resetList()
    requestAnimationFrame(() => explorerRef.current?.scrollIntoView({ behavior: 'instant', block: 'start' }))
  }
  function chooseTopic(topic: Topic, connected = false) {
    const next = connected ? new URLSearchParams({ group: topic.group }) : new URLSearchParams(params)
    next.set('topic', topic.id)
    setParams(next)
    setCopyStatus('')
    if (connected) listRef.current?.scrollTo({ top: 0 })
    requestAnimationFrame(() => {
      if (window.matchMedia('(max-width: 720px)').matches) detailRef.current?.scrollIntoView({ behavior: 'instant', block: 'start' })
      detailRef.current?.focus({ preventScroll: true })
    })
  }
  function search(value: string) {
    setParams(value ? { q: value } : {}, { replace: true })
    resetList()
  }
  async function copyLink() {
    if (!selected) return
    const url = new URL(window.location.href)
    url.searchParams.set('topic', selected.id)
    try {
      await navigator.clipboard.writeText(url.toString())
      setCopyStatus('Link copied')
    } catch {
      setCopyStatus('Copy the page address to share this view.')
    }
  }

  return (
    <div className="container topic-map">
      <header className="tm-intro">
        <div>
          <span className="eyebrow">Topic map</span>
          <h1>Find your way through AI.</h1>
          <p>Start with a learning path, or follow an idea that interests you. Every topic leads to an explanation and the ideas around it.</p>
          <div className="tm-stats"><span><strong>{topics.length}</strong> connected topics</span><span><strong>{groups.length}</strong> areas to explore</span></div>
        </div>
        <div className="tm-motif" aria-hidden="true">
          <svg viewBox="0 0 210 120"><path d="M22 60H76L126 23H188M76 60H188M76 60L126 97H188" />{[[22, 60, 'attention'], [76, 60, 'pretrain'], [126, 23, 'agents'], [188, 23, 'agents'], [126, 60, 'inference'], [188, 60, 'inference'], [126, 97, 'retrieval'], [188, 97, 'retrieval']].map(([x, y, id], i) => <circle key={i} cx={x} cy={y} r={i === 1 ? 8 : 5} fill={graph.groups[id as Group].color} />)}</svg>
          <span>One idea leads to another.</span>
        </div>
      </header>

      <section className="tm-paths" aria-labelledby="tm-paths-heading">
        <div className="tm-section-label"><h2 id="tm-paths-heading">Not sure where to start?</h2><span>Pick a learning path · 5 stops each</span></div>
        <div className="tm-path-grid">
          {paths.map((item, index) => (
            <button key={item.id} className={`tm-path${path?.id === item.id ? ' is-active' : ''}`} aria-pressed={path?.id === item.id} onClick={() => choosePath(item.id)}>
              <span className="tm-path-top"><span>0{index + 1}</span><Arrow /></span>
              <strong>{item.title}</strong><span className="tm-path-description">{item.description}</span>
              <span className="tm-trail" aria-hidden="true">{item.topics.map((id) => <i key={id} style={accent(byId.get(id)!.group)} />)}</span>
            </button>
          ))}
        </div>
      </section>

      <section className="tm-explorer" ref={explorerRef} aria-labelledby="tm-explorer-heading">
        <div className="tm-explorer-head">
          <div><h2 id="tm-explorer-heading">Explore the connections</h2><p>Choose a topic to see where it leads.</p></div>
          <div className="tm-search"><SearchIcon /><input type="search" aria-label="Search all topics" placeholder="Search topics, e.g. KV cache" value={query} onChange={(event) => search(event.target.value)} />{query && <button onClick={() => search('')} aria-label="Clear search">×</button>}</div>
        </div>
        <div className="tm-mobile-filter"><label htmlFor="tm-area">Browse an area</label><select id="tm-area" value={path ? 'path' : group} onChange={(event) => chooseGroup(event.target.value)}>{path && <option value="path">Learning path: {path.title}</option>}<option value="all">All topics ({topics.length})</option>{groups.map(([id, value]) => <option key={id} value={id}>{value.label} ({topics.filter((topic) => topic.group === id).length})</option>)}</select></div>
        <div className="tm-workspace">
          <nav className="tm-areas" aria-label="Topic areas">
            <span className="tm-overline">Browse by area</span>
            <button className={!path && group === 'all' ? 'is-active' : ''} aria-pressed={!path && group === 'all'} onClick={() => chooseGroup('all')}><span>All topics</span><span className="tm-count">{topics.length}</span></button>
            {groups.map(([id, value]) => <button key={id} style={accent(id)} className={!path && group === id ? 'is-active' : ''} aria-pressed={!path && group === id} onClick={() => chooseGroup(id)}><i aria-hidden="true" /><span>{value.label}</span><span className="tm-count">{topics.filter((topic) => topic.group === id).length}</span></button>)}
            <p className="tm-area-hint">Explore freely. Connections show related ideas, so you can follow your curiosity.</p>
          </nav>

          <div className="tm-topic-column">
            <div className="tm-list-head"><span className="tm-overline">{path ? 'Learning path' : query ? 'Search results' : 'Topics'}</span><h3>{query ? `Results for “${query}”` : listTitle}</h3><p role="status">{visible.length} {visible.length === 1 ? 'topic' : 'topics'}{path ? ' · follow the numbered stops' : ' · select to explore'}</p>{path && <button className="tm-text-button" onClick={() => chooseGroup('all')}>Browse all topics →</button>}</div>
            <div className={`tm-topic-list${path ? ' is-path' : ''}`} ref={listRef} tabIndex={0} aria-label={path ? `${path.title} stops` : 'Topics'}>
              {visible.length ? visible.map((topic, index) => (
                <button key={topic.id} style={accent(topic.group)} className={`tm-topic${selected?.id === topic.id ? ' is-selected' : ''}`} aria-pressed={selected?.id === topic.id} onClick={() => chooseTopic(topic)}>
                  <span className="tm-topic-marker" aria-hidden="true">{path ? index + 1 : <i />}</span>
                  <span className="tm-topic-text"><strong>{topic.label}</strong><span>{path ? topic.blurb : graph.groups[topic.group].label}</span></span><Chevron />
                </button>
              )) : <div className="tm-empty"><SearchIcon /><strong>No topics found</strong><p>Try “attention”, “agents”, or a shorter search.</p><button className="tm-text-button" onClick={() => search('')}>Clear search →</button></div>}
            </div>
          </div>

          {selected ? <article className="tm-detail" ref={detailRef} tabIndex={-1} aria-labelledby="tm-topic-title" style={accent(selected.group)}>
            <button className="tm-back tm-text-button" onClick={() => { listRef.current?.scrollIntoView({ behavior: 'instant', block: 'start' }); listRef.current?.focus({ preventScroll: true }) }}>← Back to topics</button>
            <div className="tm-detail-body">
              <GroupLabel group={selected.group} />
              {path && <span className="tm-step">Stop {pathIndex + 1} of {path.topics.length}</span>}
              <h2 id="tm-topic-title">{selected.label}</h2><p className="tm-blurb">{selected.blurb}</p>
              <Link className="tm-read" to={topicHref(selected)}>Read the explanation <Arrow /></Link>
              <div className="tm-source"><span>{resourceType(selected)}{selected.route.includes('#') ? ' · opens at this topic' : ''}</span><button onClick={copyLink}>Copy link</button></div>
              <span className="tm-copy-status" role="status">{copyStatus}</span>
            </div>
            {path && <div className="tm-next"><span className="tm-overline">{nextTopic ? 'Next in this path' : 'Last stop in this path'}</span>{nextTopic ? <button onClick={() => chooseTopic(nextTopic)}>{nextTopic.label}<Chevron /></button> : <p>Keep exploring with a related topic below.</p>}</div>}
            <div className="tm-connections"><div className="tm-related-heading"><h3>Connected ideas</h3><span>{neighborIds.size}</span></div><p>Follow a connection to explore further.</p>
              {relatedGroups.map((item) => <div className="tm-related-group" key={item.id}><GroupLabel group={item.id} />{item.topics.map((topic) => <button key={topic.id} onClick={() => chooseTopic(topic, true)}><span>{topic.label}</span><Chevron /></button>)}</div>)}
              {!relatedGroups.length && <p>Browse another area to find more ideas.</p>}
            </div>
          </article> : <div className="tm-detail tm-detail-empty"><span className="tm-overline">Room to explore</span><h2>Your next idea is out there.</h2><p>Search all {topics.length} topics, or choose an area to start again.</p></div>}
        </div>
      </section>
    </div>
  )
}
