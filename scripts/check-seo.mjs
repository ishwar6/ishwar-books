// Check deployed HTML, not just templates. Runs after the production build and in CI.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { parseHTML, DOMParser } from 'linkedom'
import { createServer } from 'vite'

const root = path.resolve(import.meta.dirname, '..')
const dist = path.join(root, 'dist')
const read = (file) => fs.readFileSync(path.join(dist, file), 'utf8')
const sitemap = new DOMParser().parseFromString(read('sitemap.xml'), 'text/xml')
const urls = [...sitemap.querySelectorAll('url > loc')].map((node) => node.textContent)
assert.ok(urls.length > 0, 'Sitemap must contain pages')
assert.equal(new Set(urls).size, urls.length, 'Sitemap contains duplicates')
const origin = new URL(urls[0]).origin
const base = new URL(urls[0]).pathname
const routeOf = (url) => new URL(url).pathname.slice(base.length).replace(/\/$/, '')
const localFile = (url) => path.join(dist, decodeURIComponent(new URL(url).pathname.slice(base.length)))
const documents = new Map()
const links = []
const titles = new Set()
const descriptions = new Set()
let markdownCount = 0

for (const url of urls) {
  assert.equal(new URL(url).origin, origin, `Noncanonical origin: ${url}`)
  assert.ok(url.endsWith('/') && !url.includes('?') && !url.includes('#'), `Noncanonical URL: ${url}`)
  const route = routeOf(url)
  const { document } = parseHTML(read(`${route ? route + '/' : ''}index.html`))
  const one = (selector) => {
    const nodes = document.querySelectorAll(selector)
    assert.equal(nodes.length, 1, `${url}: expected exactly one ${selector}`)
    return nodes[0]
  }
  const title = one('head title').textContent
  assert.ok(title && !titles.has(title), `${url}: missing or duplicated title`)
  titles.add(title)
  const description = one('head meta[name="description"]').getAttribute('content')
  assert.ok(description && !descriptions.has(description), `${url}: missing or duplicated description`)
  descriptions.add(description)
  assert.equal(one('head link[rel="canonical"]').getAttribute('href'), url)
  assert.equal(one('head meta[property="og:url"]').getAttribute('content'), url)
  assert.equal(one('head meta[name="twitter:title"]').getAttribute('content'), title)
  assert.doesNotMatch(one('head meta[name="robots"]').getAttribute('content'), /noindex/)
  one('h1')
  assert.ok(document.querySelector('main').textContent.trim().length > 40, `${url}: empty prerendered content`)
  const ld = JSON.parse(one('script[type="application/ld+json"]').textContent)
  assert.equal(ld['@context'], 'https://schema.org')
  assert.ok(ld['@graph'].some((node) => node['@type'] === 'Person'))
  assert.ok(ld['@graph'].some((node) => node['@type'] === 'WebSite'))
  assert.ok(ld['@graph'].some((node) => node['@id'] === url + '#webpage'))
  const ids = ld['@graph'].map((node) => node['@id']).filter(Boolean)
  assert.equal(ids.length, new Set(ids).size, `${url}: duplicated structured-data IDs`)
  const serialized = JSON.stringify(ld)
  assert.doesNotMatch(serialized, /Invalid Date|NaN|undefined/)
  for (const node of ld['@graph']) {
    for (const key of ['datePublished', 'dateModified']) if (node[key]) assert.ok(Number.isFinite(Date.parse(node[key])), `${url}: invalid ${key}`)
    if (node.datePublished && node.dateModified) assert.ok(node.dateModified >= node.datePublished, `${url}: modified before publication`)
    if (node['@type'] === 'VideoObject') assert.ok(!node.contentUrl?.includes('youtube.com/watch'), 'A YouTube watch page is not a video file contentUrl')
  }
  const metadata = JSON.parse(read(`_meta/${route || 'home'}.json`))
  assert.equal(metadata.find((node) => node.tag === 'title').text, title)
  assert.deepEqual(JSON.parse(metadata.find((node) => node.tag === 'script').text), ld)
  const markdown = document.querySelector('link[type="text/markdown"]')
  if (markdown) {
    const text = fs.readFileSync(localFile(markdown.getAttribute('href')), 'utf8')
    assert.ok(text.includes(`Canonical URL: ${url}`) && text.includes('Author: Ishwar Jangid'))
    assert.ok(document.querySelector('[rel="author"]'), `${url}: missing visible authorship`)
    markdownCount++
  }
  for (const element of document.querySelectorAll('a[href], img[src], meta[property="og:image"]')) {
    const href = element.getAttribute('href') ?? element.getAttribute('src') ?? element.getAttribute('content')
    if (!href || /^(mailto:|tel:|data:)/.test(href)) continue
    const target = new URL(href, url)
    if (target.origin === origin) links.push({ from: url, target })
  }
  documents.set(url, new Set([...document.querySelectorAll('[id]')].map((node) => node.id)))
}
const broken = []
const brokenAnchors = []
for (const { from, target } of links) {
  const file = localFile(target)
  if (!fs.existsSync(file) && !fs.existsSync(path.join(file, 'index.html'))) broken.push(`${from} → ${target}`)
  const canonical = `${target.origin}${target.pathname.replace(/\/?$/, '/')}`
  if (target.hash && documents.has(canonical) && !documents.get(canonical).has(decodeURIComponent(target.hash.slice(1)))) brokenAnchors.push(`${from} → ${target}`)
}
assert.deepEqual([...new Set(broken)], [], 'Broken internal destinations')
assert.deepEqual([...new Set(brokenAnchors)], [], 'Broken section links')
const notFound = parseHTML(read('404.html')).document
assert.match(notFound.querySelector('meta[name="robots"]').getAttribute('content'), /noindex/)
assert.equal(notFound.querySelector('link[rel="canonical"]'), null, '404 must not canonicalize to home')
for (const date of sitemap.querySelectorAll('lastmod')) assert.match(date.textContent, /^\d{4}-\d{2}-\d{2}$/)
const rss = new DOMParser().parseFromString(read('rss.xml'), 'text/xml')
assert.ok(rss.querySelectorAll('item').length >= markdownCount - 15, 'Feed omits published content types')
for (const date of rss.querySelectorAll('pubDate')) assert.ok(Number.isFinite(Date.parse(date.textContent)))
assert.ok(read('robots.txt').includes(`Sitemap: ${origin}${base}sitemap.xml`))
assert.match(read('robots.txt'), /User-agent: OAI-SearchBot\nAllow: \//)
for (const url of urls.filter((url) => routeOf(url) !== 'projects')) assert.ok(read('llms.txt').includes(url), `Missing discovery index entry: ${url}`)

// Exercise actual client head replacement with a DOM: article → video → 404 → article.
const server = await createServer({ root, configFile: false, server: { middlewareMode: true, hmr: false, watch: null }, logLevel: 'error' })
try {
  const { applyHead } = await server.ssrLoadModule('/src/lib/seo-head.ts')
  const sample = urls.find((url) => url.includes('/writings/'))
  const { document } = parseHTML(fs.readFileSync(path.join(localFile(sample), 'index.html'), 'utf8'))
  for (const destination of [urls.find((url) => url.includes('/videos/')), urls[0], sample].filter(Boolean)) {
    const nodes = JSON.parse(read(`_meta/${routeOf(destination) || 'home'}.json`))
    applyHead(nodes, document.head)
    applyHead(nodes, document.head) // StrictMode/repeated application must not duplicate tags.
    assert.equal(document.querySelectorAll('link[rel="canonical"]').length, 1)
    assert.equal(document.querySelector('link[rel="canonical"]').getAttribute('href'), destination)
    assert.equal(document.querySelectorAll('script[type="application/ld+json"]').length, 1)
    assert.equal(document.title, nodes.find((node) => node.tag === 'title').text)
    if (destination === urls[0]) {
      assert.equal(document.querySelector('meta[property="article:published_time"]'), null)
      assert.equal(document.querySelector('meta[property="og:video"]'), null)
      assert.equal(document.querySelector('link[type="text/markdown"]'), null)
    }
    applyHead([{ tag: 'title', text: 'Not found' }, { tag: 'meta', attrs: { name: 'robots', content: 'noindex, follow' } }], document.head)
    assert.equal(document.querySelector('link[rel="canonical"]'), null)
    assert.equal(document.querySelector('script[type="application/ld+json"]'), null)
  }
} finally { await server.close() }
console.log(`SEO: ${urls.length} canonical HTML pages, ${markdownCount} Markdown alternatives, ${links.length} internal links, structured data, sitemap, RSS and client navigation passed.`)
