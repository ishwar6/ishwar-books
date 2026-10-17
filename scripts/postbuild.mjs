// Static pages for search engines and link previews.
//
// GitHub Pages has no server rendering, so for every route this writes a real index.html that contains:
//   - the page's own <title>, description, canonical URL, Open Graph and X/Twitter tags
//   - JSON-LD structured data (Person, WebSite, BlogPosting, Book, TechArticle, BreadcrumbList)
//   - the full readable content (articles, chapters, lists) inside #root, which React replaces on load
// plus sitemap.xml, robots.txt, rss.xml and a 404.html that tells crawlers not to index it.
import fs from 'node:fs'
import path from 'node:path'
import { contentDates, latestDate } from './lib/content-dates.mjs'

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..')
const DIST = path.join(ROOT, 'dist')
const { site, projects } = JSON.parse(fs.readFileSync(path.join(ROOT, 'src/data/site.json'), 'utf8'))
const manifest = JSON.parse(fs.readFileSync(path.join(ROOT, 'src/generated/manifest.json'), 'utf8'))
const shell = fs.readFileSync(path.join(DIST, 'index.html'), 'utf8')
const BASE = (process.env.BASE_PATH ?? '/').replace(/\/?$/, '/')
const ORIGIN = site.url.replace(/\/$/, '')
const url = (route = '') => `${ORIGIN}${BASE}${route ? route.replace(/\/?$/, '/') : ''}`
const esc = (s = '') => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
const ogImage = (name) => (fs.existsSync(path.join(DIST, 'og', `${name}.png`)) ? `${ORIGIN}${BASE}og/${name}.png` : `${ORIGIN}${BASE}og/site.png`)
const content = (route) => {
  const f = path.join(DIST, '_content', `${route}.json`)
  return fs.existsSync(f) ? JSON.parse(fs.readFileSync(f, 'utf8')).html : ''
}
const iso = (d) => (d ? new Date(d + 'T00:00:00Z').toISOString() : undefined)
const sourceDates = contentDates(ROOT)
const imageSizes = new Map()
function imageSize(image) {
  if (!imageSizes.has(image)) {
    let size = {}
    const imageUrl = new URL(image)
    if (imageUrl.origin === ORIGIN && imageUrl.pathname.startsWith(BASE)) {
      const file = path.join(DIST, imageUrl.pathname.slice(BASE.length))
      if (fs.existsSync(file)) {
        const bytes = fs.readFileSync(file)
        if (bytes.length >= 24 && bytes.subarray(1, 4).toString() === 'PNG') size = { 'image:width': String(bytes.readUInt32BE(16)), 'image:height': String(bytes.readUInt32BE(20)) }
      }
    }
    imageSizes.set(image, size)
  }
  return imageSizes.get(image)
}

const person = {
  '@type': 'Person', '@id': `${ORIGIN}/#person`, name: site.name, url: ORIGIN + BASE,
  email: `mailto:${site.email}`, jobTitle: 'Software Engineer', address: { '@type': 'PostalAddress', addressLocality: site.location },
  sameAs: Object.values(site.links),
  knowsAbout: ['LLM inference', 'vLLM', 'Retrieval-augmented generation', 'Vector search', 'GPU programming', 'CUDA', 'Machine learning systems'],
}
const author = { '@type': 'Person', '@id': person['@id'], name: site.name, url: ORIGIN + BASE }
const website = { '@type': 'WebSite', '@id': `${ORIGIN}/#website`, url: ORIGIN + BASE, name: site.name, description: site.description, author: { '@id': person['@id'] }, inLanguage: 'en' }
const crumbs = (items) => ({
  '@type': 'BreadcrumbList',
  itemListElement: items.map(([name, route], i) => ({ '@type': 'ListItem', position: i + 1, name, item: url(route) })),
})

// ---------------------------------------------------------------- page definitions
const pages = []
const papers = manifest.papers ?? []
const nav = `<nav><a href="${BASE}">Home</a> · <a href="${BASE}writings/">Writings</a> · <a href="${BASE}books/">Books</a> · <a href="${BASE}papers/">Research papers</a> · <a href="${BASE}demos/">Demos</a> · <a href="${BASE}videos/">Videos</a> · <a href="${BASE}map/">Topic map</a> · <a href="${BASE}projects/">Projects</a> · <a href="${BASE}about/">About</a></nav>`
const list = (items) => `<ul>${items.map(([href, title, desc]) => `<li><a href="${href}">${esc(title)}</a>${desc ? `<p>${esc(desc)}</p>` : ''}</li>`).join('')}</ul>`

pages.push({
  route: '', title: `${site.name}: LLM inference, RAG and GPUs, from first principles`, description: site.description,
  image: ogImage('site'), type: 'website', ld: [website, person],
  body: `<h1>${esc(site.name)}</h1><p>${esc(site.tagline)}</p>${site.bio.map((p) => `<p>${esc(p)}</p>`).join('')}
    <h2>Writings</h2>${list(manifest.writings.map((w) => [`${BASE}${w.route}/`, w.title, w.description]))}
    <h2>Videos</h2>${list((manifest.videos ?? []).map((v) => [`${BASE}${v.route}/`, v.title, v.description]))}
    <h2>Research papers</h2>${list(papers.map((p) => [`${BASE}${p.route}/`, `${p.short}, explained`, p.description]))}
    <h2>Books</h2>${list(manifest.books.map((b) => [`${BASE}books/${b.slug}/`, b.title, b.description]))}
    <h2>Projects</h2>${list(projects.map((p) => [`${BASE}projects/`, p.name, p.tagline]))}`,
})
pages.push({
  route: 'writings', title: `Writings · ${site.name}`,
  description: 'Long-form essays and deep dives on LLM inference, vLLM, the KV cache, speculative decoding, vector search and HNSW, written from first principles with real measurements.',
  image: ogImage('site'), type: 'website',
  ld: [{ '@type': 'CollectionPage', name: 'Writings', url: url('writings'), isPartOf: { '@id': website['@id'] } }, crumbs([['Home', ''], ['Writings', 'writings']])],
  body: `<h1>Writings</h1>${list(manifest.writings.map((w) => [`${BASE}${w.route}/`, w.title, w.description]))}`,
})
pages.push({
  route: 'books', title: `Books · ${site.name}`,
  description: 'Free online books written from first principles: RAG, GPU programming, LLMs from scratch and LangGraph, with runnable code in every chapter.',
  image: ogImage('site'), type: 'website',
  ld: [{ '@type': 'CollectionPage', name: 'Books', url: url('books') }, crumbs([['Home', ''], ['Books', 'books']])],
  body: `<h1>Books</h1>${list(manifest.books.map((b) => [`${BASE}books/${b.slug}/`, b.title, b.description]))}`,
})
pages.push({
  route: 'projects', title: `Projects · ${site.name}`, description: projects.map((p) => `${p.name}: ${p.tagline}.`).join(' '),
  image: ogImage('site'), type: 'website',
  ld: [{
    '@type': 'CollectionPage', name: 'Projects', url: url('projects'),
    hasPart: projects.map((p) => ({ '@type': 'SoftwareSourceCode', name: p.name, description: p.description, programmingLanguage: p.tech, author: { '@id': person['@id'] } })),
  }, crumbs([['Home', ''], ['Projects', 'projects']])],
  body: `<h1>Projects</h1>${projects.map((p) => `<article><h2>${esc(p.name)}</h2><p><strong>${esc(p.tagline)}</strong></p><p>${esc(p.description)}</p><ul>${p.highlights.map((h) => `<li>${esc(h)}</li>`).join('')}</ul><p>${esc(p.tech.join(', '))}</p></article>`).join('')}`,
})
const topicGraph = JSON.parse(fs.readFileSync(path.join(ROOT, 'src/data/graph.json'), 'utf8'))
pages.push({
  route: 'map', title: `Topic map · ${site.name}`,
  description: 'Find a learning path through attention, LLM inference, RAG, agents and GPUs. Search topics, explore their connections, and jump to the exact explanation.',
  image: ogImage('site'), type: 'website',
  ld: [{ '@type': 'CollectionPage', name: 'Topic map', url: url('map') }, crumbs([['Home', ''], ['Topic map', 'map']])],
  body: `<h1>Topic map</h1>${Object.entries(topicGraph.groups).map(([k, g]) => `<h2>${esc(g.label)}</h2>${list(topicGraph.nodes.filter((n) => n.group === k).map((n) => [`${BASE}${n.route.replace(/#.*/, '')}/${n.route.includes('#') ? '#' + n.route.split('#')[1] : ''}`, n.label, n.blurb]))}`).join('')}`,
})
pages.push({
  route: 'about', title: `About ${site.name}`,
  description: `${site.name} is a software engineer in ${site.location} who builds production AI systems and writes first-principles guides to LLM inference, RAG and GPU programming.`,
  image: ogImage('site'), type: 'profile',
  ld: [{ '@type': 'ProfilePage', url: url('about'), mainEntity: person }, crumbs([['Home', ''], ['About', 'about']])],
  body: `<h1>About ${esc(site.name)}</h1>${site.bio.map((p) => `<p>${esc(p)}</p>`).join('')}<p>Email: <a href="mailto:${esc(site.email)}">${esc(site.email)}</a></p>
    <ul>${Object.entries(site.links).map(([k, v]) => `<li><a href="${esc(v)}" rel="me">${esc(k)}</a></li>`).join('')}</ul>`,
})

for (const w of manifest.writings) {
  const series = w.series ? { isPartOf: { '@type': 'CreativeWorkSeries', name: w.series }, position: w.seriesPart } : {}
  pages.push({
    route: w.route, title: `${w.title} · ${site.name}`, description: w.description, image: ogImage(w.slug), type: 'article',
    published: iso(w.date), modified: iso(w.updated ?? w.date), tags: w.tags,
    ld: [{
      '@type': 'BlogPosting', headline: w.title, description: w.description, url: url(w.route), mainEntityOfPage: url(w.route),
      image: ogImage(w.slug), datePublished: iso(w.date), dateModified: iso(w.updated ?? w.date), keywords: w.tags.join(', '),
      timeRequired: `PT${w.minutes}M`, inLanguage: 'en', author, publisher: author, ...series,
    }, crumbs([['Home', ''], ['Writings', 'writings'], [w.title, w.route]])],
    body: `<article><h1>${esc(w.title)}</h1><p>${esc(w.description)}</p>${content(w.route)}</article>`,
  })
}

pages.push({
  route: 'papers', title: `Research papers · ${site.name}`,
  description: 'Landmark research papers read slowly, section by section: highlighted excerpts of the paper, clear explanations, definitions, figures, code you can run, and a reader that shows the paper and its explanation side by side.',
  image: ogImage('papers'), type: 'website',
  ld: [{ '@type': 'CollectionPage', name: 'Research papers', url: url('papers'), isPartOf: { '@id': website['@id'] } }, crumbs([['Home', ''], ['Research papers', 'papers']])],
  body: `<h1>Research papers</h1>${list(papers.map((p) => [`${BASE}${p.route}/`, `${p.short}, explained`, p.description]))}`,
})
pages.push({
  route: 'demos', title: `Interactive demos · ${site.name}`,
  description: 'Explore Transformers, Vision Transformers and Segment Anything with interactive browser demos, runnable models, attention visualizations and step-by-step explanations.',
  image: ogImage('papers'), type: 'website',
  ld: [{ '@type': 'CollectionPage', name: 'Interactive demos', url: url('demos'), isPartOf: { '@id': website['@id'] } }, crumbs([['Home', ''], ['Interactive demos', 'demos']])],
  body: `<h1>Interactive demos</h1>${list(papers.filter((p) => p.demo).map((p) => [`${BASE}${p.route}/demo/`, p.demo.title, p.demo.description]))}`,
})
for (const p of papers) {
  const scholarly = {
    '@type': 'ScholarlyArticle', name: p.title, headline: p.title, datePublished: p.year ? String(p.year) : undefined,
    author: p.authors.map((name) => ({ '@type': 'Person', name })), ...(p.arxiv ? { url: `https://arxiv.org/abs/${p.arxiv}`, sameAs: `https://arxiv.org/abs/${p.arxiv}` } : {}),
  }
  const cite = `<p>${esc(p.title)}. ${esc(p.authors.join(', '))}. ${esc([p.venue, p.year].filter(Boolean).join(', '))}.${p.arxiv ? ` <a href="https://arxiv.org/abs/${p.arxiv}">arXiv:${p.arxiv}</a>` : ''}</p>`
  pages.push({
    route: p.route, title: `${p.short}, explained: ${p.title} · ${site.name}`, description: p.description, image: ogImage(`paper-${p.slug}`), type: 'article',
    published: iso(p.date), modified: iso(p.updated ?? p.date), tags: p.tags,
    ld: [{
      '@type': 'TechArticle', headline: `${p.short}, explained`, description: p.description, url: url(p.route), mainEntityOfPage: url(p.route),
      image: ogImage(`paper-${p.slug}`), inLanguage: 'en', author, publisher: author, about: scholarly, keywords: p.tags.join(', '),
      hasPart: p.parts.map((x) => ({ '@type': 'TechArticle', headline: x.title, position: x.partNumber, url: url(x.route) })),
    }, crumbs([['Home', ''], ['Research papers', 'papers'], [p.short, p.route]])],
    body: `<article><h1>${esc(p.short)}, explained</h1><p>${esc(p.description)}</p>${cite}${p.learn.length ? `<h2>What you will learn</h2><ul>${p.learn.map((l) => `<li>${esc(l)}</li>`).join('')}</ul>` : ''}${content(p.route)}
      <h2>Parts</h2>${list(p.parts.map((x) => [`${BASE}${x.route}/`, `${x.partLabel}: ${x.title}`, x.description]))}</article>`,
  })
  p.parts.forEach((x, i) => {
    const prev = p.parts[i - 1], next = p.parts[i + 1]
    const img = ogImage(`paper-${p.slug}-${x.slug}`)
    pages.push({
      route: x.route, title: `${p.short}, ${x.partLabel}: ${x.title} · ${site.name}`, description: x.description, image: img, type: 'article',
      published: iso(x.date), modified: iso(x.updated ?? x.date), tags: x.tags,
      ld: [{
        '@type': 'TechArticle', headline: `${p.short}, ${x.partLabel}: ${x.title}`, description: x.description, url: url(x.route), mainEntityOfPage: url(x.route),
        image: img, datePublished: iso(x.date), dateModified: iso(x.updated ?? x.date), inLanguage: 'en', timeRequired: `PT${x.minutes}M`,
        keywords: x.tags.join(', '), author, publisher: author, about: scholarly, position: x.partNumber,
        isPartOf: { '@type': 'TechArticle', headline: `${p.short}, explained`, url: url(p.route) },
      }, crumbs([['Home', ''], ['Research papers', 'papers'], [p.short, p.route], [x.partLabel, x.route]])],
      body: `<article><p><a href="${BASE}${p.route}/">${esc(p.short)}, explained</a> · ${x.partOf}${x.covers ? ` · Covers ${esc(x.covers)}` : ''}</p><h1>${esc(x.title)}</h1><p>${esc(x.description)}</p>${cite}${content(x.route)}
        <nav>${prev ? `<a href="${BASE}${prev.route}/">Previous: ${prev.partLabel}, ${esc(prev.title)}</a>` : ''} ${next ? `<a href="${BASE}${next.route}/">Next: ${next.partLabel}, ${esc(next.title)}</a>` : ''}</nav></article>`,
    })
  })
  if (p.demo) pages.push({
    route: `${p.route}/demo`, title: `${p.demo.title} · ${p.short} · ${site.name}`, image: ogImage(`paper-${p.slug}`), type: 'website',
    description: p.demo.description,
    ld: [crumbs([['Home', ''], ['Research papers', 'papers'], [p.short, p.route], [p.demo.title, `${p.route}/demo`]])],
    body: `<h1>${esc(p.demo.title)}</h1><p>${esc(p.demo.description)}</p><p><a href="${BASE}${p.route}/">${esc(p.short)}, explained</a></p>`,
  })
  if (p.reader) pages.push({
    route: `${p.route}/read`, title: `${p.short}, read beside the paper · ${site.name}`, image: ogImage(`paper-${p.slug}`), type: 'website',
    description: `Read the ${p.short} paper with its explanation beside it: scroll the paper, and the explanation of each highlighted passage follows.`,
    ld: [crumbs([['Home', ''], ['Research papers', 'papers'], [p.short, p.route], ['Read beside the paper', `${p.route}/read`]])],
    body: `<h1>${esc(p.short)}, read beside the paper</h1><p>The paper and its explanation side by side.</p>${list(p.parts.filter((x) => x.partNumber).map((x) => [`${BASE}${x.route}/`, `${x.partLabel}: ${x.title}`, x.description]))}`,
  })
}

const videos = manifest.videos ?? []
const isoDuration = (sec) => `PT${Math.floor(sec / 60)}M${sec % 60}S`
pages.push({
  route: 'videos', title: `Videos · ${site.name}`,
  description: 'Animated explainers on vector databases, search and AI systems, each with a written companion, the key numbers and the full transcript.',
  image: videos[0]?.cover ? ORIGIN + videos[0].cover : ogImage('site'), type: 'website',
  ld: [{ '@type': 'CollectionPage', name: 'Videos', url: url('videos') }, crumbs([['Home', ''], ['Videos', 'videos']])],
  body: `<h1>Videos</h1>${list(videos.map((v) => [`${BASE}${v.route}/`, v.title, v.description]))}`,
})
for (const v of videos) {
  const thumb = v.cover ? ORIGIN + v.cover : ogImage('site')
  const watch = `https://www.youtube.com/watch?v=${v.youtube}`
  const clips = v.chapters.map((c, i) => ({
    '@type': 'Clip', name: c.label, startOffset: c.t, endOffset: v.chapters[i + 1]?.t ?? v.durationSeconds, url: `${watch}&t=${c.t}s`,
  }))
  pages.push({
    route: v.route, title: `${v.title} · ${site.name}`, description: v.description, image: thumb, type: 'video.other',
    published: iso(v.date), tags: v.tags, video: { url: `https://www.youtube.com/embed/${v.youtube}`, watch },
    ld: [{
      '@type': 'VideoObject', name: v.title, description: v.description, thumbnailUrl: [thumb, `https://i.ytimg.com/vi/${v.youtube}/maxresdefault.jpg`],
      uploadDate: iso(v.date), duration: v.durationSeconds ? isoDuration(v.durationSeconds) : undefined,
      embedUrl: `https://www.youtube.com/embed/${v.youtube}`, inLanguage: 'en', keywords: v.tags.join(', '),
      author, publisher: author, hasPart: clips,
      ...(v.series ? { isPartOf: { '@type': 'CreativeWorkSeries', name: v.series }, position: v.seriesPart } : {}),
    }, crumbs([['Home', ''], ['Videos', 'videos'], [v.title, v.route]])],
    body: `<article><h1>${esc(v.title)}</h1><p>${esc(v.description)}</p><p><a href="${watch}">Watch on YouTube (${esc(v.duration ?? '')})</a></p>
      <h2>Chapters</h2><ol>${v.chapters.map((c) => `<li><a href="${watch}&amp;t=${c.t}s">${c.stamp} ${esc(c.label)}</a></li>`).join('')}</ol>${content(v.route)}</article>`,
  })
}

for (const b of manifest.books) {
  const bookUrl = url(`books/${b.slug}`)
  const img = ogImage(`book-${b.slug}`)
  pages.push({
    route: `books/${b.slug}`, title: `${b.title}: ${b.subtitle} · ${site.name}`, description: b.description, image: img, type: 'book', tags: b.topics,
    ld: [{
      '@type': 'Book', '@id': `${bookUrl}#book`, name: b.title, alternativeHeadline: b.subtitle, description: b.description, url: bookUrl, image: img,
      author, inLanguage: 'en', isAccessibleForFree: true, keywords: b.topics.join(', '),
      hasPart: b.chapters.map((c, i) => ({ '@type': 'Chapter', name: c.title, position: i + 1, url: url(c.route) })),
    }, crumbs([['Home', ''], ['Books', 'books'], [b.title, `books/${b.slug}`]])],
    body: `<article><h1>${esc(b.title)}</h1><p>${esc(b.subtitle)}</p><p>${esc(b.description)}</p>${content(`books/${b.slug}`)}
      <h2>Chapters</h2>${b.chapters.length ? list(b.chapters.map((c) => [`${BASE}${c.route}/`, c.title, c.description])) : '<p>Chapters are being edited for publication.</p>'}</article>`,
  })
  b.chapters.forEach((c, i) => {
    const prev = b.chapters[i - 1], next = b.chapters[i + 1]
    pages.push({
      route: c.route, title: `${c.title} · ${b.title}`, description: c.description || b.description, image: img, type: 'article', tags: b.topics,
      ld: [{
        '@type': 'TechArticle', headline: c.title, description: c.description || b.description, url: url(c.route), mainEntityOfPage: url(c.route),
        image: img, inLanguage: 'en', timeRequired: `PT${c.minutes}M`, keywords: b.topics.join(', '), author,
        isPartOf: { '@type': 'Book', '@id': `${bookUrl}#book`, name: b.title, url: bookUrl }, position: i + 1,
      }, crumbs([['Home', ''], ['Books', 'books'], [b.title, `books/${b.slug}`], [c.title, c.route]])],
      body: `<article><p><a href="${BASE}books/${b.slug}/">${esc(b.title)}</a>${c.part ? ` · ${esc(c.part)}` : ''}</p><h1>${esc(c.title)}</h1>${c.description ? `<p>${esc(c.description)}</p>` : ''}${content(c.route)}
        <nav>${prev ? `<a href="${BASE}${prev.route}/">Previous: ${esc(prev.title)}</a>` : ''} ${next ? `<a href="${BASE}${next.route}/">Next: ${esc(next.title)}</a>` : ''}</nav></article>`,
    })
  })
}

// ---------------------------------------------------------------- shared static and client metadata
// Keep the complete, descriptive title: fixed character clipping was removing chapter subjects.
const clip = (text = '', length = 160) => {
  if (text.length <= length) return text
  const cut = text.slice(0, length - 1)
  const space = cut.lastIndexOf(' ')
  return cut.slice(0, space > length * 0.6 ? space : cut.length).replace(/[,;:\s]+$/, '') + '…'
}
const sourceMeta = new Map([
  ...manifest.writings, ...videos, ...papers, ...papers.flatMap((p) => p.parts),
  ...manifest.books.map((b) => ({ ...b, route: `books/${b.slug}` })), ...manifest.books.flatMap((b) => b.chapters),
].map((p) => [p.route, p]))
const credit = (meta, route) => `<aside class="article-credit" aria-label="Author and citation"><p>Written by <a href="${BASE}about/" rel="author">${esc(site.name)}</a>${meta.updated ? ` · Updated <time datetime="${meta.updated}">${meta.updated}</time>` : ''}</p><details><summary>Cite or read this page as Markdown</summary><p>${esc(site.name)}. ${esc(meta.title)}. <a href="${url(route)}">${url(route)}</a></p><p><a href="${BASE}_content/${route}.md">Read Markdown</a> · <a href="https://github.com/ishwar6/ishwar-books/blob/main/${meta.source}">View source and revision history</a></p></details></aside>`

for (const pg of pages) {
  const meta = sourceMeta.get(pg.route)
  if (meta) {
    pg.published = iso(meta.date)
    pg.modified = iso(meta.updated)
    pg.markdown = `${ORIGIN}${BASE}_content/${pg.route}.md`
    if (pg.body.endsWith('</article>')) pg.body = pg.body.slice(0, -10) + credit(meta, pg.route) + '</article>'
  } else {
    const children = [...sourceMeta.values()].filter((x) => !pg.route || x.route.startsWith(pg.route + '/'))
    const pageFile = { '': 'Home', writings: 'Writings', books: 'Books', papers: 'Papers', demos: 'Demos', videos: 'Videos', map: 'Map', about: 'About', projects: 'Projects' }[pg.route]
    const paper = papers.find((p) => pg.route.startsWith(p.route + '/'))
    pg.modified = iso(latestDate(children.map((x) => x.updated), pageFile && sourceDates.get(`src/pages/${pageFile}.tsx`), ['','about','projects'].includes(pg.route) && sourceDates.get('src/data/site.json'), pg.route === 'map' && sourceDates.get('src/data/graph.json'), paper?.updated, paper && [...sourceDates].filter(([file]) => file.startsWith(`src/demos/${paper.slug}/`)).map(([, date]) => date)))
  }
  const main = pg.ld.find((node) => ['BlogPosting', 'TechArticle', 'Book', 'VideoObject'].includes(node['@type']))
  if (main) {
    main['@id'] ??= `${url(pg.route)}#article`
    main.datePublished = pg.published
    main.dateModified = pg.modified
    main.isAccessibleForFree = true
    main.author ??= author
    main.publisher ??= author
  }
  const webpage = pg.ld.find((node) => ['WebPage', 'CollectionPage', 'ProfilePage'].includes(node['@type'])) ?? { '@type': 'WebPage' }
  Object.assign(webpage, { '@id': `${url(pg.route)}#webpage`, url: url(pg.route), name: pg.title, description: pg.description, inLanguage: 'en', isPartOf: { '@id': website['@id'] }, datePublished: pg.published, dateModified: pg.modified, ...(main ? { mainEntity: { '@id': main['@id'] } } : {}) })
  if (!pg.ld.includes(webpage)) pg.ld.push(webpage)
  pg.ld = [person, website, ...pg.ld.filter((node) => ![person['@id'], website['@id']].includes(node['@id']))]
}

function headFor(pg, { noindex = false } = {}) {
  const canonical = url(pg.route)
  const description = clip(pg.description)
  const nodes = []
  const meta = (key, value, property = false) => { if (value) nodes.push({ tag: 'meta', attrs: { [property ? 'property' : 'name']: key, content: value } }) }
  const link = (attrs) => nodes.push({ tag: 'link', attrs })
  nodes.push({ tag: 'title', text: pg.title })
  meta('description', description)
  meta('theme-color', '#0d0e16')
  meta('author', site.name)
  if (!pg.route && !noindex) {
    meta('google-site-verification', process.env.GOOGLE_SITE_VERIFICATION)
    meta('msvalidate.01', process.env.BING_SITE_VERIFICATION)
  }
  meta('robots', noindex ? 'noindex, follow' : 'index, follow, max-image-preview:large, max-snippet:-1, max-video-preview:-1')
  if (!noindex) link({ rel: 'canonical', href: canonical })
  for (const [key, value] of Object.entries({ 'site_name': site.name, type: pg.type, title: pg.title, description, url: noindex ? undefined : canonical, image: pg.image, ...imageSize(pg.image), 'image:alt': pg.title, locale: 'en_US' })) meta(`og:${key}`, value, true)
  if (pg.type === 'article') {
    meta('article:author', url('about'), true)
    meta('article:published_time', pg.published, true)
    meta('article:modified_time', pg.modified, true)
    for (const tag of pg.tags ?? []) meta('article:tag', tag, true)
  }
  if (pg.video) {
    meta('og:video', pg.video.url, true)
    meta('og:video:secure_url', pg.video.url, true)
    meta('og:video:type', 'text/html', true)
    meta('og:video:width', '1280', true)
    meta('og:video:height', '720', true)
  }
  meta('twitter:card', 'summary_large_image')
  meta('twitter:creator', '@' + site.links.x.split('/').pop())
  for (const [key, value] of Object.entries({ title: pg.title, description, image: pg.image, 'image:alt': pg.title })) meta(`twitter:${key}`, value)
  link({ rel: 'alternate', type: 'application/rss+xml', title: `${site.name}: latest writing`, href: `${ORIGIN}${BASE}rss.xml` })
  if (pg.markdown) link({ rel: 'alternate', type: 'text/markdown', title: 'Markdown version', href: pg.markdown })
  nodes.push({ tag: 'script', attrs: { type: 'application/ld+json', id: 'page-structured-data', 'data-seo-route': pg.route }, text: JSON.stringify({ '@context': 'https://schema.org', '@graph': pg.ld }).replace(/</g, '\\u003c') })
  return nodes
}
function render(pg, options = {}) {
  const head = headFor(pg, options).map(({ tag, attrs = {}, text = '' }) => {
    const open = `<${tag} data-page-meta ${Object.entries(attrs).map(([k, v]) => `${k}="${esc(v)}"`).join(' ')}>`
    return ['meta', 'link'].includes(tag) ? open : `${open}${tag === 'script' ? text : esc(text)}</${tag}>`
  }).join('\n    ')
  return shell
    .replace(/<title>[^<]*<\/title>/, '')
    .replace(/\s*<meta name="description"[^>]*>/, '')
    .replace(/\s*<meta property="og:[^>]*>/g, '')
    .replace(/\s*<link rel="canonical"[^>]*>/, '')
    .replace('</head>', `    ${head}\n  </head>`)
    .replace('<div id="root"></div>', `<div id="root"><div class="container prerender">${nav}<main class="prose">${pg.body}</main><footer><a href="${BASE}rss.xml">RSS feed</a> · <a href="${BASE}llms.txt">Content index for AI tools</a></footer></div></div>`)
}

for (const pg of pages) {
  const dest = path.join(DIST, pg.route, 'index.html')
  fs.mkdirSync(path.dirname(dest), { recursive: true })
  fs.writeFileSync(dest, render(pg))
  const metadata = path.join(DIST, '_meta', `${pg.route || 'home'}.json`)
  fs.mkdirSync(path.dirname(metadata), { recursive: true })
  fs.writeFileSync(metadata, JSON.stringify(headFor(pg)))
}
fs.writeFileSync(path.join(DIST, '404.html'), render({
  route: '', title: `Page not found · ${site.name}`, description: 'This page does not exist. Explore the writings, books and research paper explanations.', image: ogImage('site'), type: 'website', ld: [], body: '<h1>Page not found</h1><p>Use the navigation to find a published page.</p>',
}, { noindex: true }))

// ---------------------------------------------------------------- sitemap, robots, RSS
fs.writeFileSync(path.join(DIST, 'sitemap.xml'), `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${pages.map((pg) => `  <url><loc>${url(pg.route)}</loc>${pg.modified ? `<lastmod>${pg.modified.slice(0, 10)}</lastmod>` : ''}</url>`).join('\n')}
</urlset>
`)
fs.writeFileSync(path.join(DIST, 'robots.txt'), `# Public content is available to search engines and AI search crawlers.\n# OAI-SearchBot serves ChatGPT search; training crawler policy remains unchanged.\nUser-agent: *\nAllow: /\n\nUser-agent: OAI-SearchBot\nAllow: /\n\nSitemap: ${ORIGIN}${BASE}sitemap.xml\n`)
fs.writeFileSync(path.join(DIST, 'rss.xml'), `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom"><channel>
<title>${esc(site.name)}: writings, books, papers and videos</title><link>${ORIGIN}${BASE}</link><description>${esc(site.description)}</description><language>en</language>
<atom:link href="${ORIGIN}${BASE}rss.xml" rel="self" type="application/rss+xml" />
${[...manifest.writings, ...videos, ...manifest.books.flatMap((b) => b.chapters.map((c) => ({ ...c, title: `${c.title} · ${b.title}` }))), ...papers.flatMap((p) => p.parts.map((x) => ({ ...x, title: `${p.short}, ${x.partLabel}: ${x.title}` })))]
  .sort((a, b) => (b.updated ?? b.date ?? '').localeCompare(a.updated ?? a.date ?? '')).map((w) => `<item><title>${esc(w.title)}</title><link>${url(w.route)}</link><guid>${url(w.route)}</guid>${w.date || w.updated ? `<pubDate>${new Date((w.date || w.updated) + 'T00:00:00Z').toUTCString()}</pubDate>` : ''}<description>${esc(w.description)}</description></item>`).join('\n')}
</channel></rss>
`)
fs.writeFileSync(path.join(DIST, '.nojekyll'), '')
console.log(`postbuild: ${pages.length} pages with SEO metadata + prerendered content, sitemap.xml, robots.txt, rss.xml, 404.html`)

// Optional text discovery index. This supplements HTML/sitemaps; it is not a ranking directive.
const mdLink = (p) => `- [${p.title.replace(/[\[\]\n]/g, ' ')}](${url(p.route)}): ${(p.description ?? '').replace(/\s+/g, ' ')}${p.markdown ? ` [Markdown](${p.markdown})` : ''}`
const sections = [
  ['Start here', pages.filter((p) => ['', 'about', 'map', 'books', 'papers', 'demos', 'videos', 'writings'].includes(p.route))],
  ['Writings', pages.filter((p) => p.route.startsWith('writings/'))],
  ...manifest.books.map((b) => [b.title, pages.filter((p) => p.route === `books/${b.slug}` || p.route.startsWith(`books/${b.slug}/`))]),
  ['Research paper explanations and demos', pages.filter((p) => p.route.startsWith('papers/'))],
  ['Videos and transcripts', pages.filter((p) => p.route.startsWith('videos/'))],
]
fs.writeFileSync(path.join(DIST, 'llms.txt'), `# ${site.name}\n\n> ${site.description}\n\nOriginal explanations by ${site.name}. Research paper authors are credited separately on the relevant pages.\nCanonical HTML pages contain the full readable text and stable section anchors. Each article also has a Markdown alternative. Cite the canonical page or the relevant section, and preserve the scope of experiments and measurements.\n\n[Sitemap](${ORIGIN}${BASE}sitemap.xml) · [RSS](${ORIGIN}${BASE}rss.xml)\n\n${sections.map(([title, items]) => `## ${title}\n\n${items.map(mdLink).join('\n')}`).join('\n\n')}\n`)
