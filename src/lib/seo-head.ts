export type HeadNode = { tag: 'title' | 'meta' | 'link' | 'script'; attrs?: Record<string, string>; text?: string }

/** Use the same complete head as the static page, removing metadata from the previous route. */
export function applyHead(nodes: HeadNode[], head: HTMLHeadElement = document.head) {
  head.querySelectorAll('[data-page-meta], title, meta[name="description"], meta[property^="og:"], link[rel="canonical"]').forEach((node) => node.remove())
  for (const { tag, attrs = {}, text } of nodes) {
    if (!['title', 'meta', 'link', 'script'].includes(tag)) continue
    if (tag === 'script' && attrs.type !== 'application/ld+json') continue
    const node = head.ownerDocument.createElement(tag)
    node.setAttribute('data-page-meta', '')
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value)
    if (text !== undefined) node.textContent = text
    head.append(node)
  }
}

const cache = new Map<string, Promise<HeadNode[]>>()
export function loadHead(route: string): Promise<HeadNode[]> {
  if (!cache.has(route)) {
    const request = fetch(`${import.meta.env.BASE_URL}_meta/${route || 'home'}.json`).then((response) => {
      if (!response.ok) throw new Error('Page metadata unavailable')
      return response.json() as Promise<HeadNode[]>
    })
    request.catch(() => cache.delete(route))
    cache.set(route, request)
  }
  return cache.get(route)!
}
