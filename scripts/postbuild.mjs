// Static pages for search engines and link previews.
//
// GitHub Pages has no server rendering, so for every route this writes a real index.html that contains:
//   - the page's own <title>, description, canonical URL, Open Graph and X/Twitter tags
//   - JSON-LD structured data (Person, WebSite, BlogPosting, Book, TechArticle, BreadcrumbList)
//   - the full readable content (articles, chapters, lists) inside #root, which React replaces on load
// plus sitemap.xml, robots.txt, rss.xml and a 404.html that tells crawlers not to index it.
import fs from 'node:fs'
import path from 'node:path'

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
const nav = `<nav><a href="${BASE}">Home</a> · <a href="${BASE}writings/">Writings</a> · <a href="${BASE}books/">Books</a> · <a href="${BASE}papers/">Research papers</a> · <a href="${BASE}videos/">Videos</a> · <a href="${BASE}map/">Topic map</a> · <a href="${BASE}projects/">Projects</a> · <a href="${BASE}about/">About</a></nav>`
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
  description: 'Every important topic on this site as one connected map: attention, BERT, the Vision Transformer, LLM inference, RAG, agents and GPUs. Click a topic to read about it.',
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
  description: 'Landmark research papers read slowly, section by section: highlighted excerpts of the paper, plain-English explanations, definitions, figures and code you can run.',
  image: ogImage('papers'), type: 'website',
  ld: [{ '@type': 'CollectionPage', name: 'Research papers', url: url('papers'), isPartOf: { '@id': website['@id'] } }, crumbs([['Home', ''], ['Research papers', 'papers']])],
  body: `<h1>Research papers</h1>${list(papers.map((p) => [`${BASE}${p.route}/`, `${p.short}, explained`, p.description]))}`,
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
      <h2>Parts</h2>${list(p.parts.map((x) => [`${BASE}${x.route}/`, `Part ${x.partNumber}: ${x.title}`, x.description]))}</article>`,
  })
  p.parts.forEach((x, i) => {
    const prev = p.parts[i - 1], next = p.parts[i + 1]
    const img = ogImage(`paper-${p.slug}-${x.slug}`)
    pages.push({
      route: x.route, title: `${p.short}, Part ${x.partNumber}: ${x.title} · ${site.name}`, description: x.description, image: img, type: 'article',
      published: iso(x.date), modified: iso(x.updated ?? x.date), tags: x.tags,
      ld: [{
        '@type': 'TechArticle', headline: `${p.short}, Part ${x.partNumber}: ${x.title}`, description: x.description, url: url(x.route), mainEntityOfPage: url(x.route),
        image: img, datePublished: iso(x.date), dateModified: iso(x.updated ?? x.date), inLanguage: 'en', timeRequired: `PT${x.minutes}M`,
        keywords: x.tags.join(', '), author, publisher: author, about: scholarly, position: x.partNumber,
        isPartOf: { '@type': 'TechArticle', headline: `${p.short}, explained`, url: url(p.route) },
      }, crumbs([['Home', ''], ['Research papers', 'papers'], [p.short, p.route], [`Part ${x.partNumber}`, x.route]])],
      body: `<article><p><a href="${BASE}${p.route}/">${esc(p.short)}, explained</a> · Part ${x.partNumber} of ${p.parts.length}${x.covers ? ` · Covers ${esc(x.covers)}` : ''}</p><h1>${esc(x.title)}</h1><p>${esc(x.description)}</p>${cite}${content(x.route)}
        <nav>${prev ? `<a href="${BASE}${prev.route}/">Previous: Part ${prev.partNumber}, ${esc(prev.title)}</a>` : ''} ${next ? `<a href="${BASE}${next.route}/">Next: Part ${next.partNumber}, ${esc(next.title)}</a>` : ''}</nav></article>`,
    })
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
      contentUrl: watch, embedUrl: `https://www.youtube.com/embed/${v.youtube}`, inLanguage: 'en', keywords: v.tags.join(', '),
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

// ---------------------------------------------------------------- head + body injection
// Search engines show about 155 to 160 characters of a description and about 60 to 70 of a title,
// so the tags get a clipped copy (cut at a word) while the page text and JSON-LD keep the full versions.
const clip = (s, n) => {
  if (!s || s.length <= n) return s ?? ''
  const cut = s.slice(0, n - 1)
  const end = Math.max(cut.lastIndexOf('. '), cut.lastIndexOf('; '), cut.lastIndexOf(', '), cut.lastIndexOf(' '))
  return cut.slice(0, end > n * 0.6 ? end : cut.length).replace(/[,;:\s]+$/, '') + (cut.slice(0, end).endsWith('.') ? '' : '…')
}
const clipTitle = (t) => {
  if (t.length <= 72) return t
  const i = t.lastIndexOf(' · ')
  const [main, tail] = i > 0 ? [t.slice(0, i), t.slice(i)] : [t, '']
  const room = 68 - tail.length
  if (main.length <= room) return t
  const colon = main.lastIndexOf(': ', room)
  return (colon > room * 0.5 ? main.slice(0, colon) : clip(main, room)) + tail
}

function render(pg, { noindex = false } = {}) {
  const canonical = url(pg.route)
  const title = clipTitle(pg.title), description = clip(pg.description, 160)
  const meta = [
    `<title>${esc(title)}</title>`,
    `<meta name="description" content="${esc(description)}" />`,
    '<meta name="theme-color" content="#0d0e16" />',
    `<link rel="canonical" href="${canonical}" />`,
    `<meta name="author" content="${esc(site.name)}" />`,
    noindex ? '<meta name="robots" content="noindex" />' : '<meta name="robots" content="index, follow, max-image-preview:large" />',
    pg.tags?.length ? `<meta name="keywords" content="${esc(pg.tags.join(', '))}" />` : '',
    `<meta property="og:site_name" content="${esc(site.name)}" />`,
    `<meta property="og:type" content="${pg.type === 'book' ? 'book' : pg.type}" />`,
    `<meta property="og:title" content="${esc(title)}" />`,
    `<meta property="og:description" content="${esc(description)}" />`,
    `<meta property="og:url" content="${canonical}" />`,
    `<meta property="og:image" content="${pg.image}" />`,
    '<meta property="og:image:width" content="1200" />',
    '<meta property="og:image:height" content="630" />',
    '<meta property="og:locale" content="en_US" />',
    pg.published ? `<meta property="article:published_time" content="${pg.published}" />` : '',
    pg.modified ? `<meta property="article:modified_time" content="${pg.modified}" />` : '',
    ...(pg.type === 'article' ? (pg.tags ?? []).map((t) => `<meta property="article:tag" content="${esc(t)}" />`) : []),
    ...(pg.video ? [`<meta property="og:video" content="${pg.video.url}" />`, `<meta property="og:video:secure_url" content="${pg.video.url}" />`, '<meta property="og:video:type" content="text/html" />', '<meta property="og:video:width" content="1280" />', '<meta property="og:video:height" content="720" />'] : []),
    ...(pg.type === 'video.other' ? (pg.tags ?? []).map((t) => `<meta property="video:tag" content="${esc(t)}" />`) : []),
    '<meta name="twitter:card" content="summary_large_image" />',
    `<meta name="twitter:creator" content="@${site.links.x.split('/').pop()}" />`,
    `<meta name="twitter:title" content="${esc(title)}" />`,
    `<meta name="twitter:description" content="${esc(description)}" />`,
    `<meta name="twitter:image" content="${pg.image}" />`,
    `<link rel="alternate" type="application/rss+xml" title="${esc(site.name)}: writings" href="${ORIGIN}${BASE}rss.xml" />`,
    `<script type="application/ld+json">${JSON.stringify({ '@context': 'https://schema.org', '@graph': pg.ld }).replace(/</g, '\\u003c')}</script>`,
  ].filter(Boolean).join('\n    ')
  return shell
    .replace(/<title>[^<]*<\/title>/, '')
    .replace(/\s*<meta name="description"[^>]*>/, '')
    .replace(/\s*<meta property="og:[^>]*>/g, '')
    .replace(/\s*<link rel="canonical"[^>]*>/, '')
    .replace('</head>', `    ${meta}\n  </head>`)
    .replace('<div id="root"></div>', `<div id="root"><div class="container prerender">${nav}<main class="prose">${pg.body}</main></div></div>`)
}

for (const pg of pages) {
  const dest = path.join(DIST, pg.route, 'index.html')
  fs.mkdirSync(path.dirname(dest), { recursive: true })
  fs.writeFileSync(dest, render(pg))
}
fs.writeFileSync(path.join(DIST, '404.html'), render({
  route: '', title: `Page not found · ${site.name}`, description: site.description, image: ogImage('site'), type: 'website', ld: [website], body: '<h1>Page not found</h1>',
}, { noindex: true }))

// ---------------------------------------------------------------- sitemap, robots, RSS
const today = new Date().toISOString().slice(0, 10)
fs.writeFileSync(path.join(DIST, 'sitemap.xml'), `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${pages.map((pg) => `  <url><loc>${url(pg.route)}</loc><lastmod>${(pg.modified ?? '').slice(0, 10) || today}</lastmod></url>`).join('\n')}
</urlset>
`)
fs.writeFileSync(path.join(DIST, 'robots.txt'), `User-agent: *\nAllow: /\n\nSitemap: ${ORIGIN}${BASE}sitemap.xml\n`)
fs.writeFileSync(path.join(DIST, 'rss.xml'), `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom"><channel>
<title>${esc(site.name)}: writings and research papers</title><link>${ORIGIN}${BASE}</link><description>${esc(site.description)}</description><language>en</language>
<atom:link href="${ORIGIN}${BASE}rss.xml" rel="self" type="application/rss+xml" />
${[...manifest.writings, ...papers.flatMap((p) => p.parts.map((x) => ({ ...x, title: `${p.short}, Part ${x.partNumber}: ${x.title}` })))]
  .sort((a, b) => (b.date ?? '').localeCompare(a.date ?? '')).map((w) => `<item><title>${esc(w.title)}</title><link>${url(w.route)}</link><guid>${url(w.route)}</guid><pubDate>${new Date(w.date + 'T00:00:00Z').toUTCString()}</pubDate><description>${esc(w.description)}</description></item>`).join('\n')}
</channel></rss>
`)
fs.writeFileSync(path.join(DIST, '.nojekyll'), '')
console.log(`postbuild: ${pages.length} pages with SEO metadata + prerendered content, sitemap.xml, robots.txt, rss.xml, 404.html`)
