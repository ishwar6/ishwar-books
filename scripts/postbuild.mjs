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
const nav = `<nav><a href="${BASE}">Home</a> · <a href="${BASE}writings/">Writings</a> · <a href="${BASE}books/">Books</a> · <a href="${BASE}projects/">Projects</a> · <a href="${BASE}about/">About</a></nav>`
const list = (items) => `<ul>${items.map(([href, title, desc]) => `<li><a href="${href}">${esc(title)}</a>${desc ? `<p>${esc(desc)}</p>` : ''}</li>`).join('')}</ul>`

pages.push({
  route: '', title: `${site.name}: LLM inference, RAG and GPU programming, from first principles`, description: site.description,
  image: ogImage('site'), type: 'website', ld: [website, person],
  body: `<h1>${esc(site.name)}</h1><p>${esc(site.tagline)}</p>${site.bio.map((p) => `<p>${esc(p)}</p>`).join('')}
    <h2>Writings</h2>${list(manifest.writings.map((w) => [`${BASE}${w.route}/`, w.title, w.description]))}
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
function render(pg, { noindex = false } = {}) {
  const canonical = url(pg.route)
  const meta = [
    `<title>${esc(pg.title)}</title>`,
    `<meta name="description" content="${esc(pg.description)}" />`,
    `<link rel="canonical" href="${canonical}" />`,
    `<meta name="author" content="${esc(site.name)}" />`,
    noindex ? '<meta name="robots" content="noindex" />' : '<meta name="robots" content="index, follow, max-image-preview:large" />',
    pg.tags?.length ? `<meta name="keywords" content="${esc(pg.tags.join(', '))}" />` : '',
    `<meta property="og:site_name" content="${esc(site.name)}" />`,
    `<meta property="og:type" content="${pg.type === 'book' ? 'book' : pg.type}" />`,
    `<meta property="og:title" content="${esc(pg.title)}" />`,
    `<meta property="og:description" content="${esc(pg.description)}" />`,
    `<meta property="og:url" content="${canonical}" />`,
    `<meta property="og:image" content="${pg.image}" />`,
    '<meta property="og:image:width" content="1200" />',
    '<meta property="og:image:height" content="630" />',
    '<meta property="og:locale" content="en_US" />',
    pg.published ? `<meta property="article:published_time" content="${pg.published}" />` : '',
    pg.modified ? `<meta property="article:modified_time" content="${pg.modified}" />` : '',
    ...(pg.type === 'article' ? (pg.tags ?? []).map((t) => `<meta property="article:tag" content="${esc(t)}" />`) : []),
    '<meta name="twitter:card" content="summary_large_image" />',
    `<meta name="twitter:creator" content="@${site.links.x.split('/').pop()}" />`,
    `<meta name="twitter:title" content="${esc(pg.title)}" />`,
    `<meta name="twitter:description" content="${esc(pg.description)}" />`,
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
<title>${esc(site.name)}: writings</title><link>${ORIGIN}${BASE}</link><description>${esc(site.description)}</description><language>en</language>
<atom:link href="${ORIGIN}${BASE}rss.xml" rel="self" type="application/rss+xml" />
${manifest.writings.map((w) => `<item><title>${esc(w.title)}</title><link>${url(w.route)}</link><guid>${url(w.route)}</guid><pubDate>${new Date(w.date + 'T00:00:00Z').toUTCString()}</pubDate><description>${esc(w.description)}</description></item>`).join('\n')}
</channel></rss>
`)
fs.writeFileSync(path.join(DIST, '.nojekyll'), '')
console.log(`postbuild: ${pages.length} pages with SEO metadata + prerendered content, sitemap.xml, robots.txt, rss.xml, 404.html`)
