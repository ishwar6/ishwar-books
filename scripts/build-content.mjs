// Turns content/**/*.md into:
//   src/generated/manifest.json   metadata for every writing and book (imported by the app)
//   public/_content/**.json       pre-rendered HTML + table of contents per page (fetched on demand)
//
// Layout of content/:
//   content/writings/<slug>.md                    → /writings/<slug>
//   content/books/<book>/index.md                 → /books/<book>          (book metadata + intro)
//   content/books/<book>/[<part>/]<chapter>.md    → /books/<book>/<chapter>
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
const BASE = (process.env.BASE_PATH ?? '/ishwar-books/').replace(/\/?$/, '/')

const CALLOUTS = {
  note: 'Note', info: 'Info', tip: 'Tip', important: 'Important', warning: 'Warning',
  caution: 'Caution', danger: 'Danger', example: 'Example', question: 'Question',
  quote: 'Quote', summary: 'Summary', success: 'Success', bug: 'Bug', abstract: 'Summary',
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
const readingMinutes = (body) => Math.max(1, Math.round(body.replace(/```[\s\S]*?```/g, '').split(/\s+/).length / 220))
const toDate = (d) => (d ? new Date(d).toISOString().slice(0, 10) : undefined)

// ---------- pass 1: discover pages ----------
const pages = [] // { file, route, collection, meta, body }

for (const file of walk(path.join(CONTENT, 'writings')).filter((f) => f.endsWith('.md'))) {
  const { data, content } = matter(fs.readFileSync(file, 'utf8'))
  if (data.draft && process.env.NODE_ENV === 'production') continue
  const slug = data.slug ?? slugOf(file)
  pages.push({ file, collection: 'writings', slug, route: `writings/${slug}`, data, body: content })
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
      node.children.unshift({ type: 'paragraph', data: { hProperties: { className: ['callout-title'] } }, children: [{ type: 'text', value: title }] })
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
        toc.push({ depth: Number(node.tagName[1]), id: String(node.properties.id), text: toString(node).replace(/#$/, '').trim() })
    })
}

async function render(page) {
  const toc = []
  const file = await unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkMath, { singleDollarTextMath: false })
    .use(remarkCallouts)
    .use(remarkWikilinks)
    .use(remarkRehype, { allowDangerousHtml: true })
    .use(rehypeRaw)
    .use(rehypeMermaid)
    .use(rehypeSlug)
    .use(rehypeToc, { toc })
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
  return { html: String(file), toc }
}

// ---------- pass 2: render ----------
fs.rmSync(OUT_PUBLIC, { recursive: true, force: true })
fs.mkdirSync(path.dirname(OUT_MANIFEST), { recursive: true })

const metaOf = (p) => {
  const title = p.data.title ?? firstHeading(p.body) ?? pretty(p.slug)
  // The page header shows the title, so drop a leading H1 that duplicates it.
  p.body = p.body.replace(/^\s*#\s+.+\n/, (h) => (h.replace(/^\s*#\s+/, '').trim() === title ? '' : h))
  return {
    slug: p.slug, route: p.route, title,
    description: p.data.description ?? clip(firstParagraph(p.body)) ?? '',
    date: toDate(p.data.date), updated: toDate(p.data.updated),
    tags: p.data.tags ?? [], minutes: readingMinutes(p.body),
    ...(p.part ? { part: p.part } : {}),
  }
}

let rendered = 0
for (const p of pages) {
  p.meta = metaOf(p)
  const out = await render(p)
  const dest = path.join(OUT_PUBLIC, `${p.route}.json`)
  fs.mkdirSync(path.dirname(dest), { recursive: true })
  fs.writeFileSync(dest, JSON.stringify(out))
  rendered++
}

const writings = pages.filter((p) => p.collection === 'writings').map((p) => p.meta)
  .sort((a, b) => (b.date ?? '').localeCompare(a.date ?? ''))

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

fs.writeFileSync(OUT_MANIFEST, JSON.stringify({ writings, books: bookList }, null, 2))
console.log(`content: rendered ${rendered} pages (${writings.length} writings, ${bookList.length} books)`)
