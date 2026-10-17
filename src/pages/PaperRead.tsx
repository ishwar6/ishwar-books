import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import * as pdfjs from 'pdfjs-dist'
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url'
import { loadPage, papers, type Paper } from '../lib/content'
import { useTitle } from '../lib/hooks'
import { accentStyle } from '../components/PaperBits'
import Prose from '../components/Prose'
import NotFound from './NotFound'
import '../styles/reader.css'

pdfjs.GlobalWorkerOptions.workerSrc = workerUrl
const BASE = import.meta.env.BASE_URL

/** The map written by code/papers/<paper>/reader.py: page sizes in PDF points, and where each box of the articles is. */
type Region = { page: number; clip: number[]; hl: number[][] }
type Box = { n: number; id: string; part: string; regions: Region[] }
type ReaderMap = { pdf: string; pages: [number, number][]; boxes: Box[] }
type Side = 'paper' | 'notes'

const LINE = 0.28          // the "reading line": this fraction of a pane's height, from its top
const pct = (v: number, of: number) => `${(100 * v) / of}%`
/** One box in scroll coordinates: its span in the notes, and its passages in the paper (x: centre, 0..1 of the page).
 *  part: the index of its part. aside: the box quotes a passage out of the paper's order (a footnote, an appendix,
 *  a figure already shown), so the panes do not follow it; its number tag still shows it. */
type Spot = { n: number; top: number; bot: number; part: number; aside: boolean; regs: { top: number; bot: number; x: number }[] }

/** The boxes of one part that follow the paper's order: the longest run whose passages go down the paper.
 *  The others are asides. Returns the box numbers on the run. */
function inOrder(list: Spot[]) {
  const pos = list.map((s) => s.regs[0]?.top ?? -1)
  const len = pos.map(() => 1), prev = pos.map(() => -1)
  for (let i = 0; i < pos.length; i++) {
    if (pos[i] < 0) continue
    for (let j = 0; j < i; j++) if (pos[j] >= 0 && pos[j] <= pos[i] && len[j] + 1 > len[i]) { len[i] = len[j] + 1; prev[i] = j }
  }
  const keep = new Set<number>()
  let i = len.reduce((b, l, k) => (pos[k] >= 0 && l > (len[b] ?? 0) ? k : b), -1)
  for (; i >= 0; i = prev[i]) keep.add(list[i].n)
  return keep
}

/** One page of the PDF: a box of the right shape at once, the drawing when it comes near the screen. */
function PdfPage({ doc, index, size, boxes, active, onPick }: {
  doc: pdfjs.PDFDocumentProxy | null; index: number; size: [number, number]; boxes: { box: Box; r: Region }[]
  active: number; onPick: (n: number) => void
}) {
  const wrap = useRef<HTMLDivElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const [near, setNear] = useState(false)
  const [width, setWidth] = useState(0)

  useEffect(() => {
    const el = wrap.current!
    const io = new IntersectionObserver(([e]) => setNear(e.isIntersecting), { rootMargin: '150% 0px' })
    io.observe(el)
    const ro = new ResizeObserver(() => setWidth(Math.round(el.clientWidth)))
    ro.observe(el)
    return () => { io.disconnect(); ro.disconnect() }
  }, [])

  useEffect(() => {
    if (!doc || !near || !width) return
    let task: pdfjs.RenderTask | null = null
    let live = true
    doc.getPage(index + 1).then((page) => {
      if (!live) return
      const scale = (width / size[0]) * Math.min(window.devicePixelRatio || 1, 2.5)
      const vp = page.getViewport({ scale })
      const c = canvas.current!
      c.width = Math.floor(vp.width)
      c.height = Math.floor(vp.height)
      task = page.render({ canvas: c, viewport: vp })
      task.promise.catch(() => {})
    })
    return () => { live = false; task?.cancel() }
  }, [doc, near, width, index, size])

  const [w, h] = size
  return (
    <div ref={wrap} className="pr-page" style={{ aspectRatio: `${w} / ${h}` }} data-page={index}>
      <canvas ref={canvas} aria-label={`Page ${index + 1} of the paper`} />
      {boxes.map(({ box, r }, k) => (
        <button
          key={`${box.n}-${k}`}
          type="button"
          className={`pr-region${box.n === active ? ' on' : ''}`}
          style={{ left: pct(r.clip[0], w), top: pct(r.clip[1], h), width: pct(r.clip[2] - r.clip[0], w), height: pct(r.clip[3] - r.clip[1], h) }}
          onClick={() => onPick(box.n)}
          aria-label={`Explanation ${box.n}`}
          data-n={box.n}
        >
          {box.n === active && r.hl.map((q, i) => (
            <span key={i} className="pr-hl" style={{
              left: pct(q[0] - r.clip[0], r.clip[2] - r.clip[0]), top: pct(q[1] - r.clip[1], r.clip[3] - r.clip[1]),
              width: pct(q[2] - q[0], r.clip[2] - r.clip[0]), height: pct(q[3] - q[1], r.clip[3] - r.clip[1]),
            }} />
          ))}
        </button>
      ))}
      <span className="pr-pno">{index + 1}</span>
    </div>
  )
}

function Reader({ paper }: { paper: Paper }) {
  const [params] = useSearchParams()
  const [map, setMap] = useState<ReaderMap | null>(null)
  const [doc, setDoc] = useState<pdfjs.PDFDocumentProxy | null>(null)
  const [html, setHtml] = useState<(string | null)[]>([])
  const [failed, setFailed] = useState(false)
  const [active, setActive] = useState(0)
  const [zoom, setZoom] = useState(1)
  const [split, setSplit] = useState(0.5)
  const [top, setTop] = useState(64)
  const [later, setLater] = useState(0)     // the page at the reading line, when no part explains it yet
  const paperPane = useRef<HTMLDivElement>(null)
  const notesPane = useRef<HTMLDivElement>(null)
  const rootRef = useRef<HTMLDivElement>(null)
  const driver = useRef<Side>('paper')
    const spots = useRef<Spot[]>([])
  const pointerX = useRef(0.25)
  const moved = useRef(false)    // nothing follows until the reader moves a pane
  const activeRef = useRef(0)
  activeRef.current = active
  const [asides, setAsides] = useState('')
  const parts = useMemo(() => paper.parts.filter((p) => p.partNumber), [paper])

  // the reader fills the window under the site header; the page itself does not scroll
  useLayoutEffect(() => {
    const header = document.querySelector<HTMLElement>('.site-header')
    const measure = () => setTop(header ? header.getBoundingClientRect().height : 64)
    measure()
    window.addEventListener('resize', measure)
    document.documentElement.classList.add('pr-lock')
    return () => { window.removeEventListener('resize', measure); document.documentElement.classList.remove('pr-lock') }
  }, [])

  useEffect(() => {
    let live = true
    fetch(`${BASE}${paper.reader}`).then((r) => r.json() as Promise<ReaderMap>).then((m) => {
      if (!live) return
      setMap(m)
      const task = pdfjs.getDocument({ url: `${BASE}${m.pdf.replace(/^\//, '')}` })
      task.promise.then((d) => live && setDoc(d)).catch(() => live && setFailed(true))
    }).catch(() => live && setFailed(true))
    Promise.all(parts.map((p) => loadPage(p.route).then((r) => r.html).catch(() => null))).then((h) => live && setHtml(h))
    return () => { live = false }
  }, [paper, parts])

  const byN = useMemo(() => new Map((map?.boxes ?? []).map((b) => [b.n, b])), [map])
  const onPage = useMemo(() => {
    const out: { box: Box; r: Region }[][] = (map?.pages ?? []).map(() => [])
    for (const box of map?.boxes ?? []) for (const r of box.regions) out[r.page]?.push({ box, r })
    return out
  }, [map])
  const onPageRef = useRef(onPage)
  onPageRef.current = onPage

  /** Where each box is in both panes, in scroll coordinates. */
  const measure = useCallback(() => {
    const P = paperPane.current, N = notesPane.current
    if (!P || !N || !map) return
    const pTop = P.getBoundingClientRect().top - P.scrollTop
    const nTop = N.getBoundingClientRect().top - N.scrollTop
    const list: Spot[] = []
    for (const el of N.querySelectorAll<HTMLElement>('[data-box]')) {
      const box = byN.get(Number(el.dataset.box))
      if (!box) continue
      // follow only the passages next to the first one; a footnote or appendix far away would make the paper jump
      const regs = box.regions.filter((r) => Math.abs(r.page - box.regions[0].page) <= 1).flatMap((r) => {
        const page = P.querySelector<HTMLElement>(`[data-page="${r.page}"]`)
        if (!page) return []
        const pr = page.getBoundingClientRect(), [w, h] = map.pages[r.page]
        return [{ top: pr.top + (r.clip[1] / h) * pr.height - pTop, bot: pr.top + (r.clip[3] / h) * pr.height - pTop, x: (r.clip[0] + r.clip[2]) / 2 / w }]
      })
      const er = el.getBoundingClientRect()
      const part = parts.findIndex((p) => p.slug === box.part)
      list.push({ n: box.n, top: er.top - nTop, bot: er.bottom - nTop, part, aside: false, regs })
    }
    for (let p = 0; p < parts.length; p++) {
      const mine = list.filter((s) => s.part === p)
      const keep = inOrder(mine)
      for (const s of mine) s.aside = s.regs.length > 0 && !keep.has(s.n)
    }
    spots.current = list
    setAsides(list.filter((s) => s.aside).map((s) => s.n).join(','))
  }, [map, byN, parts])

  /** Move a pane: small steps at once (it follows the hand), large jumps smoothly. */
  const put = (el: HTMLElement, top: number) => {
    const d = Math.abs(el.scrollTop - top)
    if (d < 2) return
    el.scrollTo({ top, behavior: d > 0.6 * el.clientHeight ? 'smooth' : 'instant' })
  }

  /** The pane that the reader moves leads; the other pane shows the same place. While the reading line goes
   *  through a highlighted passage of the paper, its explanation goes by at the same pace (and the reverse).
   *  Between passages the other pane stays still, so the reader can look at it. */
  const follow = useCallback((from: Side) => {
    const P = paperPane.current, N = notesPane.current
    if (!P || !N) return
    if (from === 'paper') {
      const line = P.scrollTop + LINE * P.clientHeight
      let hits = spots.current.flatMap((s) => (s.aside ? [] : s.regs.map((r, k) => ({ s, r, k })))).filter(({ r }) => r.top <= line && line <= r.bot)
      // stay with the part being read while it still has a passage on screen; the paper's later pages bring the next part
      const cur = spots.current.find((s) => s.n === activeRef.current)?.part ?? -1
      const view0 = P.scrollTop, view1 = P.scrollTop + P.clientHeight
      const curOnScreen = cur >= 0 && spots.current.some((s) => s.part === cur && !s.aside && s.regs.some((r) => r.bot >= view0 && r.top <= view1))
      if (curOnScreen) {
        hits = hits.filter((h) => h.s.part === cur)
        if (!hits.length) { setLater(0); return }
      }
      if (!hits.length) {
        // a page that no part explains yet: say so, and leave the notes where they are
        const lineY = P.getBoundingClientRect().top + LINE * P.clientHeight
        const page = [...P.querySelectorAll<HTMLElement>('[data-page]')].find((el) => el.getBoundingClientRect().bottom >= lineY)
        const i = Number(page?.dataset.page ?? -1)
        setLater(i >= 0 && !spots.current.some((s) => s.regs.length) ? 0 : i >= 0 && !(onPageRef.current[i]?.length) ? i + 1 : 0)
        return
      }
      setLater(0)
      // two columns can both cross the line: take the one under the pointer, else the first in reading order
      // then the part nearest the one being read (the next one first)
      const far = (s: Spot) => (cur < 0 ? s.part : s.part >= cur ? 2 * (s.part - cur) : 2 * (cur - s.part) + 1)
      hits.sort((a, b) => Math.abs(a.r.x - pointerX.current) - Math.abs(b.r.x - pointerX.current) || far(a.s) - far(b.s) || a.s.n - b.s.n)
      const { s, r, k } = hits[0]
      const share = 1 / s.regs.length
      const f = (k + (line - r.top) / Math.max(1, r.bot - r.top)) * share
      put(N, s.top + f * (s.bot - s.top) - LINE * N.clientHeight)
      setActive(s.n)
    } else {
      const line = N.scrollTop + LINE * N.clientHeight
      let last = 0
      let lastPart = -1
      for (const s of spots.current) {
        if (s.top > line) break
        last = s.n
        lastPart = s.part
        // an aside quotes another place in the paper: the paper stays, the number tag can show it
        if (line > s.bot || !s.regs.length || s.aside) continue
        const f = ((line - s.top) / Math.max(1, s.bot - s.top)) * s.regs.length
        const k = Math.min(s.regs.length - 1, Math.floor(f)), r = s.regs[k]
        put(P, r.top + (f - k) * (r.bot - r.top) - LINE * P.clientHeight)
      }
      // at the start of a part, before its first box: name that part's first box, not the last box of the part before
      const nTop = N.getBoundingClientRect().top - N.scrollTop
      let partAt = -1
      N.querySelectorAll<HTMLElement>('[data-part]').forEach((sec, i) => { if (sec.getBoundingClientRect().top - nTop <= line) partAt = i })
      if (partAt >= 0 && partAt !== lastPart) last = spots.current.find((s) => s.part === partAt)?.n ?? last
      setActive(last)
      setLater(0)
    }
  }, [])

  // the notes: mark each box of this paper with its number, and hide its screenshot (the paper is on the left)
  useEffect(() => {
    const N = notesPane.current
    if (!N || !map || !html.length) return
    const byId = new Map(map.boxes.map((b) => [`${b.part}/${b.id}`, b]))
    for (const sec of N.querySelectorAll<HTMLElement>('[data-part]')) {
      for (const aside of sec.querySelectorAll<HTMLElement>('.callout-paper')) {
        if (aside.dataset.box) continue
        const src = aside.querySelector('.pp-shot img')?.getAttribute('src') ?? ''
        const box = byId.get(`${sec.dataset.part}/${src.split('/').pop()?.replace(/\.png$/, '')}`)
        if (!box) continue
        aside.dataset.box = String(box.n)
        aside.classList.add('pr-box')
        const tag = document.createElement('button')
        tag.type = 'button'
        tag.className = 'pr-tag'
        tag.dataset.go = String(box.n)
        tag.innerHTML = `<b>${box.n}</b><span>page ${box.regions[0].page + 1}</span>`
        tag.title = 'Show this place in the paper'
        aside.querySelector('.pp-head')?.prepend(tag)
      }
    }
    measure()
  }, [map, html, measure])

  // keep the curve right when pages, pictures or maths change size
  useEffect(() => {
    const P = paperPane.current, N = notesPane.current
    if (!P || !N) return
    let t = 0
    const ro = new ResizeObserver(() => { cancelAnimationFrame(t); t = requestAnimationFrame(() => { measure(); if (moved.current) follow(driver.current) }) })
    ro.observe(P.firstElementChild!)
    ro.observe(N.firstElementChild!)
    return () => { ro.disconnect(); cancelAnimationFrame(t) }
  }, [measure, follow, map])

  const onScroll = (side: Side) => () => { if (driver.current === side) follow(side) }
  const lead = (side: Side) => () => { driver.current = side; moved.current = true }

  /** Show box n: the pane that was not clicked scrolls to it. */
  const go = useCallback((n: number, pane: Side) => {
    const s = spots.current.find((x) => x.n === n)
    const el = pane === 'paper' ? paperPane.current : notesPane.current
    if (!s || !el) return
    driver.current = pane === 'paper' ? 'notes' : 'paper'      // the clicked pane leads, so the move does not echo back
    const top = pane === 'paper' ? (s.regs[0]?.top ?? 0) - 24 : s.top - 12
    el.scrollTo({ top: top - (pane === 'paper' ? LINE * el.clientHeight - 24 : LINE * el.clientHeight - 12), behavior: 'smooth' })
    setActive(n)
  }, [])

  // a click on a number tag in the notes shows that place in the paper
  useEffect(() => {
    const N = notesPane.current
    if (!N) return
    const onClick = (e: MouseEvent) => {
      const t = (e.target as HTMLElement).closest<HTMLElement>('[data-go]')
      if (t) go(Number(t.dataset.go), 'paper')
    }
    N.addEventListener('click', onClick)
    return () => N.removeEventListener('click', onClick)
  }, [go])

  /** Open the notes at a part, and the paper at the first passage that the part explains. */
  const showPart = (slug: string, behavior: ScrollBehavior) => {
    const N = notesPane.current, P = paperPane.current
    const sec = N?.querySelector<HTMLElement>(`[data-part="${slug}"]`)
    if (!N || !P || !sec) return
    driver.current = 'notes'
    N.scrollTo({ top: sec.offsetTop - 8, behavior })
    const idx = parts.findIndex((p) => p.slug === slug)
    const s = spots.current.find((x) => x.part === idx && !x.aside && x.regs.length) ?? spots.current.find((x) => x.part === idx)
    if (s?.regs[0]) P.scrollTo({ top: s.regs[0].top - 40, behavior })
    if (s) setActive(s.n)
  }

  // ?part=part-2-the-task opens the notes at that part
  const opened = useRef(false)
  useEffect(() => {
    const want = params.get('part')
    if (opened.current || !want || !html.length || !spots.current.length) return
    const sec = notesPane.current?.querySelector<HTMLElement>(`[data-part="${want}"]`)
    if (!sec) return
    opened.current = true
    showPart(want, 'instant')
  }, [params, html, map])

  // drag the bar between the panes
  const drag = (e: React.PointerEvent) => {
    const root = rootRef.current
    if (!root) return
    e.preventDefault()
    const box = root.getBoundingClientRect()
    const move = (ev: PointerEvent) => setSplit(Math.min(0.72, Math.max(0.28, (ev.clientX - box.left) / box.width)))
    const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up) }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
  }

  const jump = (slug: string) => showPart(slug, 'smooth')

  useEffect(() => {
    const N = notesPane.current
    N?.querySelector('.pr-box.on')?.classList.remove('on')
    N?.querySelector(`[data-box="${active}"]`)?.classList.add('on')
  }, [active])

  const activeBox = byN.get(active)
  return (
    <div ref={rootRef} className="pr-root paper-theme" style={{ ...accentStyle(paper), top, ['--split' as string]: `${split * 100}%` }}>
      <div className="pr-bar">
        <Link to={`/${paper.route}`} className="pr-back">← {paper.short}, explained</Link>
        <select className="pr-jump" aria-label="Go to a part" value="" onChange={(e) => jump(e.target.value)}>
          <option value="" disabled>Go to a part</option>
          {parts.map((p) => <option key={p.slug} value={p.slug}>{p.partLabel}: {p.title}</option>)}
        </select>
        <span className="pr-where">{later ? <>Page {later}: a later part explains it</> : activeBox ? <>Box <b>{activeBox.n}</b> of {map?.boxes.length} · page {activeBox.regions[0].page + 1}{asides.split(',').includes(String(activeBox.n)) && <>: click its number to see it</>}</> : 'Scroll the paper or the notes'}</span>
        <span className="pr-zoom" role="group" aria-label="Paper size">
          <button type="button" onClick={() => setZoom((z) => Math.max(0.6, +(z - 0.1).toFixed(2)))} aria-label="Smaller paper">−</button>
          <button type="button" onClick={() => setZoom(1)} aria-label="Fit the paper to the pane">{Math.round(zoom * 100)}%</button>
          <button type="button" onClick={() => setZoom((z) => Math.min(2, +(z + 0.1).toFixed(2)))} aria-label="Larger paper">+</button>
        </span>
        {map && <a className="pr-dl" href={`${BASE}${map.pdf.replace(/^\//, '')}`} target="_blank" rel="noreferrer">Annotated PDF</a>}
      </div>
      <div className="pr-split">
        <div ref={paperPane} className="pr-paper" onScroll={onScroll('paper')} onWheel={lead('paper')} onTouchStart={lead('paper')} onPointerDown={lead('paper')} onKeyDown={lead('paper')}
          onPointerMove={(e) => { const r = e.currentTarget.querySelector('.pr-pages')!.getBoundingClientRect(); pointerX.current = (e.clientX - r.left) / r.width }}
          tabIndex={0} aria-label="The paper">
          <div className="pr-pages" style={{ width: `${zoom * 100}%` }}>
            {failed && <div className="pr-msg">The paper could not be loaded. Try refreshing the page.</div>}
            {map?.pages.map((size, i) => (
              <PdfPage key={i} doc={doc} index={i} size={size} boxes={onPage[i]} active={active} onPick={(n) => go(n, 'notes')} />
            ))}
          </div>
        </div>
        <div className="pr-handle" onPointerDown={drag} role="separator" aria-orientation="vertical" aria-label="Drag to resize the panes" />
        <div ref={notesPane} className="pr-notes" onScroll={onScroll('notes')} onWheel={lead('notes')} onTouchStart={lead('notes')} onPointerDown={lead('notes')} onKeyDown={lead('notes')}
          tabIndex={0} aria-label="The explanation">
          <div className="pr-notes-in">
            {parts.map((p, i) => (
              <section key={p.slug} data-part={p.slug} className="pr-part">
                <div className="pr-part-head">
                  <span className="pr-part-label">{p.partLabel}</span>
                  <Link to={`/${p.route}`} className="pr-part-title">{p.title}</Link>
                </div>
                {html[i] ? <Prose html={html[i]!} /> : html.length ? <div className="pr-msg">This part could not be loaded.</div> : <div className="skeleton" />}
              </section>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export default function PaperRead() {
  const { paper: slug } = useParams()
  const paper = papers.find((p) => p.slug === slug)
  useTitle(paper ? `${paper.short}, read beside the paper` : undefined, paper ? `Read the ${paper.short} paper with its explanation beside it.` : undefined)
  if (!paper?.reader) return <NotFound />
  return <Reader paper={paper} />
}
