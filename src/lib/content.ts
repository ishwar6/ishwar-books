import manifest from '../generated/manifest.json'

export type TocItem = { depth: number; id: string; text: string }
export type PageMeta = {
  slug: string
  route: string
  title: string
  description: string
  date?: string
  updated?: string
  tags: string[]
  minutes: number
  part?: string
  cover?: string
  motif?: 'graph' | 'code' | 'grid' | 'waves' | 'timeline' | 'kv' | 'blocks'
  series?: string
  seriesPart?: number
  accent?: string
}
export type Book = {
  slug: string
  title: string
  subtitle: string
  description: string
  status: 'planned' | 'in-progress' | 'complete'
  accent: string
  topics: string[]
  plannedChapters: number
  chapters: PageMeta[]
}
export type Rendered = { html: string; toc: TocItem[] }

export const writings = manifest.writings as PageMeta[]
export const books = manifest.books as Book[]

const cache = new Map<string, Promise<Rendered>>()

export function loadPage(route: string): Promise<Rendered> {
  if (!cache.has(route)) {
    const p = fetch(`${import.meta.env.BASE_URL}_content/${route}.json`).then((r) => {
      if (!r.ok) throw new Error(`Could not load ${route}`)
      return r.json() as Promise<Rendered>
    })
    p.catch(() => cache.delete(route))
    cache.set(route, p)
  }
  return cache.get(route)!
}

/** Every part of a series, in order. */
export const seriesOf = (name?: string) =>
  name ? writings.filter((w) => w.series === name).sort((a, b) => (a.seriesPart ?? 0) - (b.seriesPart ?? 0)) : []

export const formatDate = (d?: string) =>
  d ? new Date(d + 'T00:00:00').toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' }) : ''
