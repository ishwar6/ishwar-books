import MiniSearch, { type SearchResult } from 'minisearch'

export type SearchDoc = { id: number; r: string; a: string; t: string; h: string; x: string; b: string | null }
export type SearchHit = SearchDoc & { score: number; terms: string[] }

let index: Promise<MiniSearch<SearchDoc>> | null = null

/** Loads and indexes the search data the first time search is opened. */
export function loadIndex() {
  index ??= fetch(`${import.meta.env.BASE_URL}_content/search.json`)
    .then((r) => r.json() as Promise<SearchDoc[]>)
    .then((docs) => {
      const ms = new MiniSearch<SearchDoc>({
        fields: ['t', 'h', 'x'],
        storeFields: ['r', 'a', 't', 'h', 'x', 'b'],
        searchOptions: { boost: { t: 3, h: 2 }, prefix: true, fuzzy: 0.15, combineWith: 'AND' },
      })
      ms.addAll(docs)
      return ms
    })
    .catch((e) => { index = null; throw e })
  return index
}

export async function search(query: string, book?: string): Promise<SearchHit[]> {
  const ms = await loadIndex()
  const filter = book ? (r: SearchResult) => r.b === book : undefined
  let hits = ms.search(query, { filter })
  if (!hits.length) hits = ms.search(query, { filter, combineWith: 'OR' })
  return hits.slice(0, 40) as unknown as SearchHit[]
}

/** ~180 characters of text around the first matching term. */
export function snippet(text: string, terms: string[], size = 180) {
  const lower = text.toLowerCase()
  let at = -1
  for (const t of terms) {
    const i = lower.indexOf(t.toLowerCase())
    if (i !== -1 && (at === -1 || i < at)) at = i
  }
  const start = Math.max(0, at - 50)
  const end = Math.min(text.length, start + size)
  return (start > 0 ? '…' : '') + text.slice(start, end).trim() + (end < text.length ? '…' : '')
}
