# ishwar-books

Personal site of **Ishwar Jangid**: portfolio, writings and long-form books, written in Markdown and published to GitHub Pages.

Live at **https://ishwarj.com**

## Writing

Everything is Markdown in `content/`:

```
content/
  writings/<slug>.md                     → /writings/<slug>
  books/<book>/index.md                  → /books/<book>          (book card + intro)
  books/<book>/<part-folder>/<file>.md   → /books/<book>/<file>   (chapters, sidebar grouped by part folder)
```

Frontmatter is optional. The title falls back to the first `# heading`, the description to the first paragraph:

```yaml
---
title: Why RAG exists
description: One line for cards and link previews.
date: 2026-09-27
tags: [rag]
order: 1        # chapters: explicit order (default: sorted by path, numerically)
part: Foundations  # chapters: override the part name taken from the folder
draft: true     # skipped in the production build
hidden: true    # books: kept in the repo but not built or listed
cover: ./cover.png   # writings: your own cover image (optional)
motif: graph    # writings: style of the generated cover when there is no image: graph | code | grid | waves
accent: "#7c9cff"  # writings: colour of the generated cover
---
```

Book `index.md` also takes `subtitle`, `status` (`planned` | `in-progress` | `complete`), `accent` (hex colour), `order`, `chapters` (planned count) and `topics`.

Supported Markdown: GitHub-flavoured Markdown (tables, task lists, footnotes), syntax-highlighted code, callouts (`> [!NOTE]`, `[!TIP]`, `[!WARNING]`, `[!IMPORTANT]`, `[!CAUTION]`), math with `$$…$$`, ` ```mermaid ` diagrams, relative images, links between `.md` files and Obsidian `[[wikilinks]]`. See `content/writings/how-this-site-renders-markdown.md` for every element.

Profile, links and projects live in `src/data/site.json`.

## Develop

```bash
npm install
npm run dev            # content build + Vite dev server
npm run content:watch  # (second terminal) rebuild content when a .md changes
npm run build          # production build into dist/
```

Pushing to `main` builds and deploys via `.github/workflows/deploy.yml`. The site is served from the custom domain in `public/CNAME` (DNS: four A records to GitHub Pages and a `www` CNAME to `ishwar6.github.io`).

## SEO

Every page is built as real HTML for search engines (`scripts/postbuild.mjs`): its own title, description, canonical URL, Open Graph and X card tags, JSON-LD structured data (Person, WebSite, BlogPosting, Book, TechArticle, BreadcrumbList), and the full article text inside the page. The build also writes `sitemap.xml`, `robots.txt` and `rss.xml`.

Social preview images live in `public/og/<slug>.png`. A new writing falls back to `public/og/site.png` until you generate its own card with `scripts/og.mjs` (instructions at the top of that file).

Profile, links and projects are in `src/data/site.json`.
