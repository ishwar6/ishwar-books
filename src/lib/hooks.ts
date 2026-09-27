import { useEffect, useState } from 'react'
import { site } from '../data/site'

export function useTitle(title?: string) {
  useEffect(() => {
    document.title = title ? `${title} · ${site.name}` : `${site.name} · engineering, ML systems and GPUs`
  }, [title])
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
