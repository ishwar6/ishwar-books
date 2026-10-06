// Turns content/**/*.md into:
//   src/generated/manifest.json   metadata for every writing and book (imported by the app)
//   public/_content/**.json       pre-rendered HTML + table of contents per page (fetched on demand)
//
// Layout of content/:
//   content/writings/<slug>.md                    → /writings/<slug>
//   content/books/<book>/index.md                 → /books/<book>          (book metadata + intro)
//   content/books/<book>/[<part>/]<chapter>.md    → /books/<book>/<chapter>
//   content/papers/<paper>/index.md               → /papers/<paper>        (paper metadata + overview)
//   content/papers/<paper>/<part>.md              → /papers/<paper>/<part> (one part of a paper breakdown)
import fs from 'node:fs'
import path from 'node:path'
import matter from 'gray-matter'
import { unified } from 'unified'
import remarkParse from 'remark-parse'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import remarkRehype from 'remark-rehype'
import rehypeRaw from 'rehype-raw'
import rehypeSlug from 'rehype-slug'
import rehypeAutolinkHeadings from 'rehype-autolink-headings'
import rehypeKatex from 'rehype-katex'
import rehypeShiki from '@shikijs/rehype'
import rehypeStringify from 'rehype-stringify'
import { visit, SKIP } from 'unist-util-visit'
import { toString } from 'hast-util-to-string'

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..')
const CONTENT = path.join(ROOT, 'content')
const OUT_PUBLIC = path.join(ROOT, 'public', '_content')
const OUT_MANIFEST = path.join(ROOT, 'src', 'generated', 'manifest.json')
const BASE = (process.env.BASE_PATH ?? '/').replace(/\/?$/, '/')

const CALLOUTS = {
  note: 'Note', info: 'Info', tip: 'Tip', important: 'Important', warning: 'Warning',
  caution: 'Caution', danger: 'Danger', example: 'Example', question: 'Question',
  quote: 'Quote', summary: 'Summary', success: 'Success', bug: 'Bug', abstract: 'Summary',
  definition: 'Definition', define: 'Definition', term: 'Definition',
  paper: 'From the paper',
  takeaways: 'Key takeaways',
}

// ---------- helpers ----------
const walk = (dir) =>
  fs.existsSync(dir)
    ? fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
        const p = path.join(dir, e.name)
        if (e.name.startsWith('.') || e.name.startsWith('_')) return []
        return e.isDirectory() ? walk(p) : [p]
      })
    : []

const natural = new Intl.Collator('en', { numeric: true, sensitivity: 'base' }).compare
const pretty = (s) =>
  s.replace(/^part\s*(\d+)[-_ ]*/i, 'Part $1 · ').replace(/^\d+[-_. ]+/, '').replace(/[-_]+/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
const slugOf = (file) => path.basename(file, '.md').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')
const firstHeading = (body) => body.match(/^#\s+(.+)$/m)?.[1]?.trim()
const firstParagraph = (body) =>
  body.split(/\n\s*\n/).map((b) => b.trim())
    .find((b) => b && !/^(#|>|\||```|<|-{3,}|\*\*Concepts|!\[)/.test(b))
    ?.replace(/\[([^\]]+)\]\([^)]*\)/g, '$1').replace(/[*_`]/g, '').replace(/\s+/g, ' ')
const clip = (s, n = 220) => (s && s.length > n ? s.slice(0, s.lastIndexOf(' ', n)).replace(/[,;:.\s]+$/, '') + '…' : s)
// words of prose only: code blocks, figures/SVG and other HTML don't count toward reading time
const readingMinutes = (body) =>
  Math.max(1, Math.round(body.replace(/```[\s\S]*?```/g, '').replace(/<figure[\s\S]*?<\/figure>/g, '').replace(/<svg[\s\S]*?<\/svg>/g, '')
    .replace(/<\/?[a-zA-Z][^>]*>/g, ' ').split(/\s+/).filter(Boolean).length / 220))
const toDate = (d) => (d ? new Date(d).toISOString().slice(0, 10) : undefined)

// ---------- pass 1: discover pages ----------
const pages = [] // { file, route, collection, meta, body }

for (const file of walk(path.join(CONTENT, 'writings')).filter((f) => f.endsWith('.md'))) {
  const { data, content } = matter(fs.readFileSync(file, 'utf8'))
  if (data.draft && process.env.NODE_ENV === 'production') continue
  const slug = data.slug ?? slugOf(file)
  pages.push({ file, collection: 'writings', slug, route: `writings/${slug}`, data, body: content })
}

const videosDir = path.join(CONTENT, 'videos')
const videoFiles = fs.existsSync(videosDir) ? fs.readdirSync(videosDir).filter((f) => f.endsWith('.md')).map((f) => path.join(videosDir, f)) : []
for (const file of videoFiles) {
  const { data, content } = matter(fs.readFileSync(file, 'utf8'))
  if (data.draft && process.env.NODE_ENV === 'production') continue
  const slug = data.slug ?? slugOf(file)
  pages.push({ file, collection: 'videos', slug, route: `videos/${slug}`, data, body: content })
}

const books = []
const bookDirs = fs.existsSync(path.join(CONTENT, 'books'))
  ? fs.readdirSync(path.join(CONTENT, 'books'), { withFileTypes: true }).filter((d) => d.isDirectory() && !d.name.startsWith('.'))
  : []
for (const dir of bookDirs) {
  const bookRoot = path.join(CONTENT, 'books', dir.name)
  const bookSlug = slugOf(dir.name)
  const indexFile = path.join(bookRoot, 'index.md')
  const index = fs.existsSync(indexFile) ? matter(fs.readFileSync(indexFile, 'utf8')) : { data: {}, content: '' }
  if (index.data.hidden) continue   // `hidden: true` in a book's index.md keeps it in the repo but off the site
  const book = { slug: bookSlug, data: index.data, chapters: [] }
  books.push(book)
  pages.push({ file: indexFile, collection: 'book-index', slug: bookSlug, book: bookSlug, route: `books/${bookSlug}`, data: index.data, body: index.content })

  const used = new Set()
  for (const file of walk(bookRoot).filter((f) => f.endsWith('.md') && f !== indexFile).sort(natural)) {
    const { data, content } = matter(fs.readFileSync(file, 'utf8'))
    if (data.draft && process.env.NODE_ENV === 'production') continue
    const rel = path.relative(bookRoot, file)
    const folder = path.dirname(rel)
    let slug = data.slug ?? slugOf(file)
    if (used.has(slug)) slug = `${slugOf(folder)}-${slug}`
    used.add(slug)
    const page = {
      file, collection: 'chapter', slug, book: bookSlug, route: `books/${bookSlug}/${slug}`, data, body: content,
      part: data.part ?? (folder === '.' ? undefined : pretty(folder.split(path.sep)[0])), rel,
    }
    book.chapters.push(page)
    pages.push(page)
  }
  // appendices always come after the numbered parts
  const rank = (c) => (/^appendix/i.test(c.rel) ? 1 : 0)
  book.chapters.sort((a, b) => (a.data.order ?? 1e9) - (b.data.order ?? 1e9) || rank(a) - rank(b) || natural(a.rel, b.rel))
}

const papers = []
const paperDirs = fs.existsSync(path.join(CONTENT, 'papers'))
  ? fs.readdirSync(path.join(CONTENT, 'papers'), { withFileTypes: true }).filter((d) => d.isDirectory() && !d.name.startsWith('.'))
  : []
for (const dir of paperDirs) {
  const root = path.join(CONTENT, 'papers', dir.name)
  const paperSlug = slugOf(dir.name)
  const indexFile = path.join(root, 'index.md')
  const index = fs.existsSync(indexFile) ? matter(fs.readFileSync(indexFile, 'utf8')) : { data: {}, content: '' }
  const paper = { slug: paperSlug, data: index.data, parts: [] }
  papers.push(paper)
  pages.push({ file: indexFile, collection: 'paper-index', slug: paperSlug, paper: paperSlug, route: `papers/${paperSlug}`, data: index.data, body: index.content })
  for (const file of fs.readdirSync(root).filter((f) => f.endsWith('.md') && f !== 'index.md').map((f) => path.join(root, f))) {
    const { data, content } = matter(fs.readFileSync(file, 'utf8'))
    if (data.draft && process.env.NODE_ENV === 'production') continue
    const slug = data.slug ?? slugOf(file)
    const page = { file, collection: 'paper-part', slug, paper: paperSlug, route: `papers/${paperSlug}/${slug}`, data, body: content }
    paper.parts.push(page)
    pages.push(page)
  }
  paper.parts.sort((a, b) => (a.data.part ?? 1e9) - (b.data.part ?? 1e9) || natural(a.slug, b.slug))
}

const byFile = new Map(pages.map((p) => [path.resolve(p.file), p]))
const byName = new Map(pages.map((p) => [path.basename(p.file, '.md').toLowerCase(), p]))

// ---------- plugins ----------
/** GitHub / Obsidian callouts:  > [!TIP] Optional title */
function remarkCallouts() {
  return (tree) =>
    visit(tree, 'blockquote', (node) => {
      const para = node.children[0]
      const text = para?.type === 'paragraph' && para.children[0]?.type === 'text' ? para.children[0] : null
      const m = text?.value.match(/^\[!(\w+)\][+-]?[ \t]*([^\n]*)\n?/)
      if (!m) return
      const kind = m[1].toLowerCase()
      const title = m[2].trim() || CALLOUTS[kind] || pretty(kind)
      text.value = text.value.slice(m[0].length)
      if (!text.value && para.children.length === 1) node.children.shift()
      node.data = { hName: 'aside', hProperties: { className: ['callout', `callout-${CALLOUTS[kind] ? kind : 'note'}`] } }
      const isDef = ['definition', 'define', 'term'].includes(kind)
      node.children.unshift({ type: 'paragraph', data: { hProperties: { className: ['callout-title'] } }, children: [{ type: 'text', value: title }] })
      if (isDef) node.data.hProperties.className = ['callout', 'callout-definition']
      if (kind === 'paper') shapePaperBox(node, title)
    })
}

/** "From the paper" boxes: split the title into the paper's name and location pills, and tag the
 *  screenshot and the Context / What it says / Why it matters paragraphs so CSS can lay them out. */
// paper crops are rendered at 190 dpi; showing them at 1/1.6 of their pixel width makes the paper's text about 16px
const PAPER_SCALE = 1.6
function pngWidth(url) {
  try {
    const buf = fs.readFileSync(path.join(ROOT, 'public', decodeURI(url.replace(BASE, '/'))))
    return buf.readUInt32BE(16)
  } catch { return 0 }
}
const PAPER_LABELS = { 'Context:': ['pp-ctx', 'Where this is'], 'What it says:': ['pp-says', 'What it says'], 'Why it matters:': ['pp-why', 'Why it matters'] }
const el = (tag, cls, children) => ({ type: 'pp', data: { hName: tag, hProperties: { className: cls } }, children })
function shapePaperBox(node, title) {
  const [name, ...where] = title.split(' · ').map((t) => t.trim())
  node.children[0] = el('div', ['pp-head'], [
    el('span', ['pp-kicker'], [{ type: 'text', value: /blog|docs|documentation|guide \(|report/i.test(title) ? 'From the source' : 'From the paper' }]),
    el('span', ['pp-name'], [{ type: 'text', value: name }]),
    ...(where.length ? [el('span', ['pp-where'], where.map((w) => el('span', ['pp-pill'], [{ type: 'text', value: w }])))] : []),
  ])
  for (const p of node.children.slice(1)) {
    if (p.type !== 'paragraph') continue
    const kids = p.children.filter((c) => !(c.type === 'text' && !c.value.trim()))
    const isShot = kids.length && kids.every((c) => c.type === 'image' || (c.type === 'link' && c.children.length === 1 && c.children[0].type === 'image'))
    if (isShot) {
      p.data = { hName: 'div', hProperties: { className: ['pp-shot'] } }
      p.children = kids.map((c) => {
        if (c.type === 'link') c.data = { hProperties: { className: ['pp-zoom'], title: 'Click to enlarge' } }
        const img = c.type === 'image' ? c : c.children[0]
        const w = pngWidth(img.url)
        if (w) img.data = { hProperties: { width: Math.round(w / PAPER_SCALE), loading: 'lazy' } }
        return c
      })
      continue
    }
    const first = p.children[0]
    const label = first?.type === 'strong' && first.children[0]?.type === 'text' ? PAPER_LABELS[first.children[0].value.trim()] : null
    if (label) {
      p.data = { hName: 'div', hProperties: { className: ['pp-note', label[0]] } }
      p.children = [el('span', ['pp-label'], [{ type: 'text', value: label[1] }]), { type: 'paragraph', children: p.children.slice(1).map((c, i) => (i === 0 && c.type === 'text' ? { ...c, value: c.value.replace(/^\s+/, '').replace(/^[a-z]/, (ch) => ch.toUpperCase()) } : c)) }]
      continue
    }
    if (kids.length === 1 && kids[0].type === 'link') p.data = { hProperties: { className: ['pp-link'] } }
  }
}

/** Paper section markers:  ## Masked LM {§3.1}  → <h2 data-sec="§3.1">Masked LM</h2> (shown as a "Paper §3.1" badge) */
function remarkSectionMarks() {
  return (tree) =>
    visit(tree, 'heading', (node) => {
      const last = node.children[node.children.length - 1]
      const m = last?.type === 'text' && last.value.match(/\s*\{(§[^}]+)\}\s*$/)
      if (!m) return
      last.value = last.value.slice(0, m.index)
      const raw = m[1].trim()
      const sec = /^§\s*([A-Z](\.\d+)*|\d+(\.\d+)*)$/.test(raw) ? raw : raw.replace(/^§\s*/, '')   // §3.1, §A.2 keep the §; "§Abstract" → "Abstract"
      node.data = { ...(node.data ?? {}), hProperties: { ...(node.data?.hProperties ?? {}), dataSec: sec } }
    })
}

/** [[wikilink]] and [[wikilink|label]] → links to other pages */
function remarkWikilinks() {
  return (tree) =>
    visit(tree, 'text', (node, index, parent) => {
      if (!parent || !node.value.includes('[[')) return
      const parts = []
      let last = 0
      for (const m of node.value.matchAll(/\[\[([^\]|#]+)(#[^\]|]+)?(?:\|([^\]]+))?\]\]/g)) {
        if (m.index > last) parts.push({ type: 'text', value: node.value.slice(last, m.index) })
        const target = byName.get(path.basename(m[1].trim()).toLowerCase())
        const label = m[3] ?? path.basename(m[1])
        parts.push(target
          ? { type: 'link', url: BASE + target.route + (m[2] ?? ''), children: [{ type: 'text', value: label }] }
          : { type: 'text', value: label })
        last = m.index + m[0].length
      }
      if (!parts.length) return
      if (last < node.value.length) parts.push({ type: 'text', value: node.value.slice(last) })
      parent.children.splice(index, 1, ...parts)
      return [SKIP, index + parts.length]
    })
}

/** ```mermaid blocks become <div class="mermaid"> rendered in the browser */
function rehypeMermaid() {
  return (tree) =>
    visit(tree, 'element', (node, index, parent) => {
      if (node.tagName !== 'pre') return
      const code = node.children[0]
      if (code?.tagName !== 'code' || !(code.properties.className ?? []).includes('language-mermaid')) return
      parent.children[index] = { type: 'element', tagName: 'div', properties: { className: ['mermaid'] }, children: [{ type: 'text', value: toString(code) }] }
    })
}

/** Rewrites relative .md links to routes, copies relative images, opens external links in a new tab. */
function rehypeLinks({ page }) {
  return (tree) =>
    visit(tree, 'element', (node) => {
      if (node.tagName === 'a' && typeof node.properties.href === 'string') {
        const href = node.properties.href
        if (/^[a-z]+:/i.test(href)) {
          node.properties.target = '_blank'
          node.properties.rel = 'noopener noreferrer'
          return
        }
        if (href.startsWith('#') || href.startsWith('/')) return
        const [p, hash = ''] = href.split('#')
        const target = byFile.get(path.resolve(path.dirname(page.file), decodeURIComponent(p)))
        if (target) node.properties.href = BASE + target.route + (hash ? `#${hash}` : '')
        else if (p.endsWith('.md')) { node.tagName = 'span'; node.properties = { className: ['dead-link'], title: 'Not published yet' } }
      }
      if (node.tagName === 'img' && typeof node.properties.src === 'string' && !/^([a-z]+:|\/)/i.test(node.properties.src)) {
        const src = path.resolve(path.dirname(page.file), decodeURIComponent(node.properties.src))
        if (fs.existsSync(src)) {
          const rel = path.relative(CONTENT, src)
          const dest = path.join(OUT_PUBLIC, 'files', rel)
          fs.mkdirSync(path.dirname(dest), { recursive: true })
          fs.copyFileSync(src, dest)
          node.properties.src = BASE + '_content/files/' + rel.split(path.sep).map(encodeURIComponent).join('/')
        }
        node.properties.loading = 'lazy'
      }
    })
}

/** Wraps code blocks with a header (language + copy button) and tables with a scroll container. */
function rehypeChrome() {
  return (tree) =>
    visit(tree, 'element', (node, index, parent) => {
      if (!parent) return
      if (node.tagName === 'pre' && node.properties['data-lang'] !== undefined) {
        const lang = String(node.properties['data-lang'] || 'text')
        parent.children[index] = {
          type: 'element', tagName: 'div', properties: { className: ['code-block'] },
          children: [
            { type: 'element', tagName: 'div', properties: { className: ['code-head'] }, children: [
              { type: 'element', tagName: 'span', properties: { className: ['code-lang'] }, children: [{ type: 'text', value: lang === 'text' ? 'plain text' : lang }] },
              { type: 'element', tagName: 'button', properties: { className: ['code-copy'], type: 'button', ariaLabel: 'Copy code' }, children: [{ type: 'text', value: 'Copy' }] },
            ] },
            node,
          ],
        }
        return SKIP
      }
      if (node.tagName === 'table') {
        parent.children[index] = { type: 'element', tagName: 'div', properties: { className: ['table-wrap'] }, children: [node] }
        return SKIP
      }
    })
}

function rehypeToc({ toc }) {
  return (tree) =>
    visit(tree, 'element', (node) => {
      if ((node.tagName === 'h2' || node.tagName === 'h3') && node.properties.id)
        toc.push({ depth: Number(node.tagName[1]), id: String(node.properties.id), text: toString(node).replace(/#$/, '').trim(), ...(node.properties.dataSec ? { sec: String(node.properties.dataSec) } : {}) })
    })
}

/** Splits the page into sections at h2/h3 for the search index (code blocks are left out). */
function rehypeSections({ sections }) {
  return (tree) => {
    let current = { id: '', heading: '', text: [] }
    sections.push(current)
    for (const node of tree.children) {
      if (node.type !== 'element') continue
      if ((node.tagName === 'h2' || node.tagName === 'h3') && node.properties.id) {
        current = { id: String(node.properties.id), heading: toString(node).trim(), text: [] }
        sections.push(current)
      } else if (node.tagName !== 'pre' && !(node.properties?.className ?? []).includes('mermaid')) {
        current.text.push(toString(node))
      }
    }
  }
}

async function render(page) {
  const toc = []
  const sections = []
  const file = await unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkMath, { singleDollarTextMath: false })
    .use(remarkCallouts)
    .use(remarkSectionMarks)
    .use(remarkWikilinks)
    .use(remarkRehype, { allowDangerousHtml: true })
    .use(rehypeRaw)
    .use(rehypeMermaid)
    .use(rehypeSlug)
    .use(rehypeToc, { toc })
    .use(rehypeSections, { sections })
    .use(rehypeAutolinkHeadings, { behavior: 'append', properties: { className: ['anchor'], ariaHidden: 'true', tabIndex: -1 }, content: { type: 'text', value: '#' } })
    .use(rehypeLinks, { page })
    .use(rehypeKatex)
    .use(rehypeShiki, {
      themes: { dark: 'one-dark-pro', light: 'github-light' },
      defaultColor: false,
      defaultLanguage: 'text',
      fallbackLanguage: 'text',
      langAlias: { cuda: 'cpp', cu: 'cpp', ptx: 'asm', triton: 'python', sh: 'bash', zsh: 'bash', console: 'shellsession' },
      transformers: [{ pre(node) { node.properties['data-lang'] = this.options.lang } }],
    })
    .use(rehypeChrome)
    .use(rehypeStringify)
    .process(page.body)
  return { html: String(file), toc, sections }
}

// ---------- pass 2: render ----------
fs.rmSync(OUT_PUBLIC, { recursive: true, force: true })
fs.mkdirSync(path.dirname(OUT_MANIFEST), { recursive: true })

/** Cover art: `cover: ./image.png` uses that image; otherwise the site draws one (`motif`, `accent` pick its style). */
function coverOf(p) {
  const out = {}
  if (p.data.motif) out.motif = p.data.motif
  if (p.data.accent) out.accent = p.data.accent
  const c = p.data.cover ?? p.data.thumbnail
  if (typeof c === 'string') {
    if (/^(https?:)?\/\//.test(c)) out.cover = c
    else {
      const src = path.resolve(path.dirname(p.file), c)
      if (fs.existsSync(src)) {
        const rel = path.relative(CONTENT, src)
        const dest = path.join(OUT_PUBLIC, 'files', rel)
        fs.mkdirSync(path.dirname(dest), { recursive: true })
        fs.copyFileSync(src, dest)
        out.cover = BASE + '_content/files/' + rel.split(path.sep).map(encodeURIComponent).join('/')
      }
    }
  }
  return out
}

/** "12:46 Searching down the layers" -> { t: 766, stamp: '12:46', label: 'Searching down the layers' } */
function videoMeta(d) {
  const secs = (st) => st.split(':').map(Number).reduce((a, x) => a * 60 + x, 0)
  const chapters = (d.chapters ?? []).map((line) => {
    const m = String(line).match(/^(\d+(?::\d{2}){1,2})\s+(.+)$/)
    return m ? { t: secs(m[1]), stamp: m[1], label: m[2].trim() } : null
  }).filter(Boolean)
  return { youtube: d.youtube, duration: d.duration, durationSeconds: d.duration ? secs(d.duration) : undefined, chapters }
}

const metaOf = (p) => {
  const title = p.data.title ?? firstHeading(p.body) ?? pretty(p.slug)
  // The page header shows the title, so drop a leading H1 that duplicates it.
  p.body = p.body.replace(/^\s*#\s+.+\n/, (h) => (h.replace(/^\s*#\s+/, '').trim() === title ? '' : h))
  return {
    slug: p.slug, route: p.route, title,
    description: p.data.description ?? clip(firstParagraph(p.body)) ?? '',
    date: toDate(p.data.date), updated: toDate(p.data.updated),
    tags: p.data.tags ?? [], minutes: readingMinutes(p.body),
    ...coverOf(p),
    ...(p.data.series ? { series: p.data.series, seriesPart: p.data.series_part ?? 1 } : {}),
    ...(p.collection === 'videos' ? videoMeta(p.data) : {}),
    ...(p.part ? { part: p.part } : {}),
    ...(p.collection === 'paper-part' ? { partNumber: p.data.part, covers: p.data.covers ?? '', short: p.data.short ?? '' } : {}),
  }
}

const paperShort = (slug) => papers.find((x) => x.slug === slug)?.data.short ?? pretty(slug)
const pageLabel = (p) => `${paperShort(p.paper)}, Part ${p.data.part}`

let rendered = 0
const search = [] // one document per section: route, anchor, page title, heading, text, book
for (const p of pages) {
  p.meta = metaOf(p)
  const { html, toc, sections } = await render(p)
  const dest = path.join(OUT_PUBLIC, `${p.route}.json`)
  fs.mkdirSync(path.dirname(dest), { recursive: true })
  fs.writeFileSync(dest, JSON.stringify({ html, toc }))
  for (const sec of sections) {
    const text = sec.text.join(' ').replace(/\s+/g, ' ').trim()
    if (!text && !sec.heading) continue
    const label = p.collection === 'paper-part' ? `${pageLabel(p)}: ${p.meta.title}` : p.meta.title
    search.push({ id: search.length, r: p.route, a: sec.id, t: label, h: sec.heading, x: text, b: p.book ?? null })
  }
  rendered++
}

const writings = pages.filter((p) => p.collection === 'writings').map((p) => p.meta)
  .sort((a, b) => (b.date ?? '').localeCompare(a.date ?? '') || (a.seriesPart ?? 0) - (b.seriesPart ?? 0))

const bookList = books.map((b) => ({
  slug: b.slug,
  title: b.data.title ?? pretty(b.slug),
  subtitle: b.data.subtitle ?? '',
  description: b.data.description ?? '',
  status: b.data.status ?? 'in-progress',
  accent: b.data.accent ?? '#7c9cff',
  topics: b.data.topics ?? [],
  plannedChapters: b.data.chapters ?? b.chapters.length,
  order: b.data.order ?? 1e9,
  chapters: b.chapters.map((c) => c.meta),
})).sort((a, b) => a.order - b.order || natural(a.slug, b.slug))

const paperList = papers.map((x) => {
  const d = x.data
  return {
    slug: x.slug, route: `papers/${x.slug}`,
    title: d.title ?? pretty(x.slug), short: d.short ?? pretty(x.slug),
    description: d.description ?? '', authors: d.authors ?? [], org: d.org ?? '', year: d.year, venue: d.venue ?? '',
    arxiv: d.arxiv ? String(d.arxiv) : undefined, code: d.code, accent: d.accent ?? '#4fc3d9',
    learn: d.learn ?? [], tags: d.tags ?? [], date: toDate(d.date), updated: toDate(d.updated), order: d.order ?? 1e9,
    plannedParts: d.parts ?? x.parts.length,
    minutes: x.parts.reduce((n, p) => n + p.meta.minutes, 0),
    parts: x.parts.map((p) => p.meta),
  }
}).sort((a, b) => a.order - b.order || (b.date ?? '').localeCompare(a.date ?? ''))

fs.writeFileSync(path.join(OUT_PUBLIC, 'search.json'), JSON.stringify(search))
const videos = pages.filter((p) => p.collection === 'videos').map((p) => p.meta)
  .sort((a, b) => (b.date ?? '').localeCompare(a.date ?? '') || (a.seriesPart ?? 0) - (b.seriesPart ?? 0))
fs.writeFileSync(OUT_MANIFEST, JSON.stringify({ writings, books: bookList, videos, papers: paperList }, null, 2))
console.log(`content: rendered ${rendered} pages, ${search.length} search sections (${writings.length} writings, ${bookList.length} books, ${videos.length} videos, ${paperList.length} papers)`)
