import { useMemo } from 'react'
import type { TocItem } from '../lib/content'
import { useActiveHeading } from '../lib/hooks'

export default function Toc({ items }: { items: TocItem[] }) {
  const ids = useMemo(() => items.map((i) => i.id), [items])
  const active = useActiveHeading(ids)
  if (items.length < 2) return <aside className="toc" />
  return (
    <aside className="toc">
      <h4>On this page</h4>
      <ul>
        {items.map((i) => (
          <li key={i.id}>
            <a
              href={`#${i.id}`}
              className={`depth-${i.depth}${active === i.id ? ' active' : ''}`}
              onClick={(e) => {
                e.preventDefault()
                history.replaceState(null, '', `#${i.id}`)
                document.getElementById(i.id)?.scrollIntoView({ behavior: 'smooth' })
              }}
            >
              {i.text}
            </a>
          </li>
        ))}
      </ul>
    </aside>
  )
}
