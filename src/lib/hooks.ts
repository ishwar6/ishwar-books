import { useEffect, useState } from 'react'
import { site } from '../data/site'
import { useLocation } from 'react-router-dom'
import { books, papers, videos, writings } from './content'
import { applyHead, loadHead, type HeadNode } from './seo-head'

const routes = new Set(['', 'writings', 'books', 'papers', 'videos', 'demos', 'map', 'projects', 'about',
  ...writings.map((p) => p.route), ...videos.map((p) => p.route),
  ...books.flatMap((b) => [`books/${b.slug}`, ...b.chapters.map((c) => c.route)]),
  ...papers.flatMap((p) => [p.route, ...p.parts.map((x) => x.route), ...(p.demo ? [`${p.route}/demo`] : []), ...(p.reader ? [`${p.route}/read`] : [])]),
])

/** Keeps the document title, description and canonical URL in sync with in-app navigation. */
export function useTitle(title?: string, description?: string) {
  const { pathname } = useLocation()
  useEffect(() => {
    const route = pathname.replace(/^\/+|\/+$/g, '')
    const known = routes.has(route)
    // Keep server-rendered metadata intact on the initial load.
    if (known && document.getElementById('page-structured-data')?.getAttribute('data-seo-route') === route) return
    const full = !known ? `Page not found · ${site.name}` : title ? `${title} · ${site.name}` : `${site.name}: LLM inference, RAG and GPUs, from first principles`
    const desc = description ?? site.description
    const canonical = `${site.url.replace(/\/$/, '')}${import.meta.env.BASE_URL}${route ? route + '/' : ''}`
    const fallback: HeadNode[] = [
      { tag: 'title', text: full },
      { tag: 'meta', attrs: { name: 'description', content: desc } },
      { tag: 'meta', attrs: { name: 'robots', content: known ? 'index, follow, max-image-preview:large, max-snippet:-1, max-video-preview:-1' : 'noindex, follow' } },
      { tag: 'meta', attrs: { property: 'og:title', content: full } },
      { tag: 'meta', attrs: { property: 'og:description', content: desc } },
      { tag: 'meta', attrs: { property: 'og:type', content: 'website' } },
      { tag: 'meta', attrs: { property: 'og:image', content: `${site.url}${import.meta.env.BASE_URL}og/site.png` } },
    ]
    if (known) fallback.push({ tag: 'link', attrs: { rel: 'canonical', href: canonical } }, { tag: 'meta', attrs: { property: 'og:url', content: canonical } })
    applyHead(fallback)
    let active = true
    if (known && import.meta.env.PROD) loadHead(route).then((nodes) => { if (active) applyHead(nodes) }).catch(() => { /* Keep the current route's fallback metadata on network failure. */ })
    return () => { active = false }
  }, [title, description, pathname])
}

export const THEMES = ['dark', 'dim', 'light'] as const
export type Theme = (typeof THEMES)[number]

export function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() => {
    const t = document.documentElement.dataset.theme as Theme
    return THEMES.includes(t) ? t : 'dark'
  })
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try { localStorage.setItem('theme', theme) } catch { /* storage unavailable */ }
    window.dispatchEvent(new Event('themechange'))
  }, [theme])
  return [theme, () => setTheme((t) => THEMES[(THEMES.indexOf(t) + 1) % THEMES.length])]
}

export const FONT_SIZES = ['small', 'normal', 'large'] as const
export type FontSize = (typeof FONT_SIZES)[number]

export function useFontSize(): [FontSize, (f: FontSize) => void] {
  const [size, setSize] = useState<FontSize>(() => {
    const f = document.documentElement.dataset.font as FontSize
    return FONT_SIZES.includes(f) ? f : 'normal'
  })
  useEffect(() => {
    document.documentElement.dataset.font = size
    try { localStorage.setItem('font', size) } catch { /* storage unavailable */ }
  }, [size])
  return [size, setSize]
}

/** Id of the heading currently at the top of the viewport. */
export function useActiveHeading(ids: string[]) {
  const [active, setActive] = useState<string>()
  useEffect(() => {
    if (!ids.length) return
    const onScroll = () => {
      let current = ids[0]
      for (const id of ids) {
        const el = document.getElementById(id)
        if (el && el.getBoundingClientRect().top < 120) current = id
      }
      setActive(current)
    }
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [ids])
  return active
}
