// GitHub Pages has no SPA rewrites, so write a real index.html for every route (with its own
// <title> and description for link previews) and a 404.html fallback for anything else.
import fs from 'node:fs'
import path from 'node:path'

const DIST = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..', 'dist')
const manifest = JSON.parse(fs.readFileSync(path.join(DIST, '..', 'src', 'generated', 'manifest.json'), 'utf8'))
const shell = fs.readFileSync(path.join(DIST, 'index.html'), 'utf8')
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;')

const routes = [
  { route: 'writings', title: 'Writings' },
  { route: 'books', title: 'Books' },
  { route: 'projects', title: 'Projects' },
  { route: 'about', title: 'About' },
  ...manifest.writings,
  ...manifest.books.flatMap((b) => [{ route: `books/${b.slug}`, title: b.title, description: b.description }, ...b.chapters]),
]

for (const r of routes) {
  let html = shell.replace(/<title>[^<]*<\/title>/, `<title>${esc(r.title)} · Ishwar Jangid</title>`)
  if (r.description) {
    html = html
      .replace(/(<meta name="description" content=")[^"]*/, `$1${esc(r.description)}`)
      .replace(/(<meta property="og:description" content=")[^"]*/, `$1${esc(r.description)}`)
  }
  html = html.replace(/(<meta property="og:title" content=")[^"]*/, `$1${esc(r.title)}`)
  const dest = path.join(DIST, r.route, 'index.html')
  fs.mkdirSync(path.dirname(dest), { recursive: true })
  fs.writeFileSync(dest, html)
}
fs.writeFileSync(path.join(DIST, '404.html'), shell)
fs.writeFileSync(path.join(DIST, '.nojekyll'), '')
console.log(`postbuild: wrote ${routes.length} route pages + 404.html`)
