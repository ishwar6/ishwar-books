import { useEffect, useRef, type KeyboardEvent, type ReactNode } from 'react'
import { useLocation, useSearchParams } from 'react-router-dom'
import { Chevron } from './icons'

export type DemoTab = { id: string; label: string; content: ReactNode }

/** Keep panels mounted so switching views preserves experiments and in-flight work. */
export default function DemoTabs({ id, label, tabs }: { id: string; label: string; tabs: DemoTab[] }) {
  const [params, setParams] = useSearchParams()
  const { hash } = useLocation()
  const root = useRef<HTMLDivElement>(null)
  const tablist = useRef<HTMLDivElement>(null)
  const buttons = useRef<(HTMLButtonElement | null)[]>([])
  const requested = params.get('view') ?? hash.slice(1)
  const index = Math.max(0, tabs.findIndex((tab) => tab.id === requested))

  useEffect(() => {
    const strip = tablist.current
    const button = buttons.current[index]
    if (!strip || !button) return
    const bounds = strip.getBoundingClientRect()
    const item = button.getBoundingClientRect()
    if (item.left < bounds.left) strip.scrollLeft += item.left - bounds.left - 5
    else if (item.right > bounds.right) strip.scrollLeft += item.right - bounds.right + 5
  }, [index])

  function select(next: number, focus = false, scroll = false) {
    const nextParams = new URLSearchParams(params)
    nextParams.set('view', tabs[next].id)
    setParams(nextParams)
    requestAnimationFrame(() => {
      if (focus) buttons.current[next]?.focus({ preventScroll: true })
      if (scroll) root.current?.scrollIntoView({ block: 'start', behavior: 'instant' })
    })
  }
  function navigate(event: KeyboardEvent, current: number) {
    const next = event.key === 'ArrowRight' ? (current + 1) % tabs.length
      : event.key === 'ArrowLeft' ? (current - 1 + tabs.length) % tabs.length
        : event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : undefined
    if (next === undefined) return
    event.preventDefault()
    select(next, true)
  }

  return <div className="demo-explorer" ref={root}>
    <div className="demo-explorer-label"><span>{label}</span><span>{index + 1} / {tabs.length}</span></div>
    <div className="demo-tabs" ref={tablist} role="tablist" aria-label={label}>
      {tabs.map((tab, i) => <button key={tab.id} ref={(element) => { buttons.current[i] = element }}
        id={`${id}-tab-${tab.id}`} role="tab" aria-selected={i === index} aria-controls={`${id}-panel-${tab.id}`}
        tabIndex={i === index ? 0 : -1} onClick={() => select(i)} onKeyDown={(event) => navigate(event, i)}>
        <span className="demo-tab-number" aria-hidden="true">{String(i + 1).padStart(2, '0')}</span>{tab.label}
      </button>)}
    </div>
    {tabs.map((tab, i) => <div key={tab.id} className="demo-tab-panel" id={`${id}-panel-${tab.id}`} role="tabpanel"
      aria-labelledby={`${id}-tab-${tab.id}`} hidden={i !== index} tabIndex={0}>{tab.content}</div>)}
    <div className="demo-step-footer">
      <button disabled={index === 0} onClick={() => select(index - 1, true, true)}><span className="demo-prev"><Chevron /></span>Previous</button>
      <span>{tabs[index].label}</span>
      {index < tabs.length - 1 ? <button onClick={() => select(index + 1, true, true)}>Next: {tabs[index + 1].label}<Chevron /></button>
        : <button onClick={() => select(0, true, true)}>Back to the start<Chevron /></button>}
    </div>
  </div>
}
