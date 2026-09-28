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

/** Renders build-time HTML and wires up copy buttons, in-app links and Mermaid diagrams. */
export default function Prose({ html }: { html: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const navigate = useNavigate()

  useEffect(() => {
    const root = ref.current
    if (!root) return
    renderMermaid(root)
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
      const a = target.closest<HTMLAnchorElement>('a[href]')
      if (!a || a.target === '_blank' || e.metaKey || e.ctrlKey || e.shiftKey) return
      const href = a.getAttribute('href')!
      if (href.startsWith('#t=')) {                    // timestamp link on a video page
        e.preventDefault()
        window.dispatchEvent(new CustomEvent('video-seek', { detail: Number(href.slice(3)) }))
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
    }
  }, [html, navigate])

  return <div ref={ref} className="prose" dangerouslySetInnerHTML={{ __html: html }} />
}
