import { useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'

const BASE = import.meta.env.BASE_URL

async function renderMermaid(root: HTMLElement) {
  const nodes = [...root.querySelectorAll<HTMLElement>('.mermaid')]
  if (!nodes.length) return
  const { default: mermaid } = await import('mermaid')
  const dark = document.documentElement.dataset.theme !== 'light'
  mermaid.initialize({ startOnLoad: false, theme: dark ? 'dark' : 'default', fontFamily: 'Helvetica Neue, Helvetica, Inter, Arial, sans-serif' })
  for (const el of nodes) {
    el.dataset.source ??= el.textContent ?? ''
    el.removeAttribute('data-processed')
    el.textContent = el.dataset.source
  }
  await mermaid.run({ nodes }).catch(() => {})
}

/** Shrinks display equations that are wider than the column (phones), down to 60% size; anything wider still scrolls. */
function fitMath(root: HTMLElement) {
  for (const box of root.querySelectorAll<HTMLElement>('.katex-display')) {
    const k = box.querySelector<HTMLElement>(':scope > .katex')
    if (!k) continue
    k.style.fontSize = ''
    let size = 1
    for (let i = 0; i < 4; i++) {           // spacing does not scale exactly with font size, so re-measure
      const over = box.scrollWidth / box.clientWidth
      if (over <= 1.01 || size <= 0.6) break
      size = Math.max(0.6, size * 0.98 / over)
      k.style.fontSize = `${1.08 * size}em`
    }
  }
}

/** Renders build-time HTML and wires up copy buttons, in-app links and Mermaid diagrams. */
export default function Prose({ html }: { html: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const navigate = useNavigate()

  useEffect(() => {
    const root = ref.current
    if (!root) return
    renderMermaid(root)
    fitMath(root)
    const onResize = () => fitMath(root)
    window.addEventListener('resize', onResize)
    const onTheme = () => renderMermaid(root)
    window.addEventListener('themechange', onTheme)

    const onClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement
      const copy = target.closest<HTMLButtonElement>('.code-copy')
      if (copy) {
        const code = copy.closest('.code-block')?.querySelector('pre')?.innerText ?? ''
        navigator.clipboard?.writeText(code).then(() => {
          copy.textContent = 'Copied'
          copy.classList.add('copied')
          setTimeout(() => { copy.textContent = 'Copy'; copy.classList.remove('copied') }, 1600)
        })
        return
      }
      const zoom = target.closest<HTMLAnchorElement>('a.pp-zoom')
      if (zoom && !e.metaKey && !e.ctrlKey && !e.shiftKey) {     // paper screenshot: open it full screen
        e.preventDefault()
        const box = document.createElement('div')
        box.className = 'pp-lightbox'
        box.innerHTML = `<img src="${zoom.getAttribute('href')}" alt=""><span>Click or press Esc to close</span>`
        box.querySelector('img')!.alt = zoom.querySelector('img')?.alt ?? ''
        const close = () => { box.remove(); window.removeEventListener('keydown', onKey) }
        const onKey = (ev: KeyboardEvent) => { if (ev.key === 'Escape') close() }
        box.addEventListener('click', close)
        window.addEventListener('keydown', onKey)
        document.body.appendChild(box)
        return
      }
      const a = target.closest<HTMLAnchorElement>('a[href]')
      if (!a || a.target === '_blank' || e.metaKey || e.ctrlKey || e.shiftKey) return
      const href = a.getAttribute('href')!
      if (a.hasAttribute('data-video-time') || href.startsWith('#t=')) {                    // timestamp link on a video page
        e.preventDefault()
        window.dispatchEvent(new CustomEvent('video-seek', { detail: Number(a.getAttribute('data-video-time') ?? href.slice(3)) }))
      } else if (href.startsWith('#')) {
        e.preventDefault()
        history.replaceState(null, '', href)
        document.getElementById(decodeURIComponent(href.slice(1)))?.scrollIntoView({ behavior: 'smooth' })
      } else if (href.startsWith(BASE)) {
        e.preventDefault()
        navigate('/' + href.slice(BASE.length))
      }
    }
    root.addEventListener('click', onClick)
    return () => {
      root.removeEventListener('click', onClick)
      window.removeEventListener('themechange', onTheme)
      window.removeEventListener('resize', onResize)
    }
  }, [html, navigate])

  return <div ref={ref} className="prose" dangerouslySetInnerHTML={{ __html: html }} />
}
