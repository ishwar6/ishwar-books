/* Small shared pieces of the Meta-Harness demo: a segmented control, loading and error states, a source line,
   and a line diff for two versions of a harness file. */
import { useEffect, useState, type ReactNode } from 'react'

export function Seg<T extends string | number>({ value, options, onChange, label }: {
  value: T; options: { v: T; label: ReactNode; title?: string }[]; onChange: (v: T) => void; label: string
}) {
  return (
    <div className="meta-harness-demo-seg" role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button key={String(o.v)} type="button" role="radio" aria-checked={value === o.v} title={o.title} onClick={() => onChange(o.v)}>
          {o.label}
        </button>
      ))}
    </div>
  )
}

export const Busy = ({ text }: { text: string }) => <div className="meta-harness-demo-busy" role="status">{text}</div>
export const Err = ({ msg }: { msg: string }) => <div className="meta-harness-demo-error" role="alert">{msg}</div>
export const Src = ({ children }: { children: ReactNode }) => <p className="meta-harness-demo-src">{children}</p>

export type DiffLine = { kind: ' ' | '+' | '-'; text: string; a?: number; b?: number }

/** A line diff (longest common subsequence), enough for files of a few hundred lines. */
export function diffLines(a: string, b: string): DiffLine[] {
  const A = a.split('\n'), B = b.split('\n')
  const n = A.length, m = B.length
  const L = Array.from({ length: n + 1 }, () => new Int32Array(m + 1))
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) L[i][j] = A[i] === B[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1])
  const out: DiffLine[] = []
  let i = 0, j = 0
  while (i < n && j < m) {
    if (A[i] === B[j]) { out.push({ kind: ' ', text: A[i], a: i + 1, b: j + 1 }); i++; j++ }
    else if (L[i + 1][j] >= L[i][j + 1]) { out.push({ kind: '-', text: A[i], a: i + 1 }); i++ }
    else { out.push({ kind: '+', text: B[j], b: j + 1 }); j++ }
  }
  while (i < n) { out.push({ kind: '-', text: A[i], a: i + 1 }); i++ }
  while (j < m) { out.push({ kind: '+', text: B[j], b: j + 1 }); j++ }
  return out
}

/** Show a diff with 3 lines of context around each change; long unchanged runs fold into one line. */
export function Diff({ lines, label }: { lines: DiffLine[]; label: string }) {
  const keep = new Array(lines.length).fill(false)
  lines.forEach((l, k) => { if (l.kind !== ' ') for (let d = -3; d <= 3; d++) if (lines[k + d]) keep[k + d] = true })
  const out: ReactNode[] = []
  let skipped = 0
  lines.forEach((l, k) => {
    if (!keep[k]) { skipped++; return }
    if (skipped) { out.push(<div key={`s${k}`} className="meta-harness-demo-dfold">… {skipped} unchanged lines</div>); skipped = 0 }
    out.push(<div key={k} className={`meta-harness-demo-dl is-${l.kind === '+' ? 'add' : l.kind === '-' ? 'del' : 'same'}`}>
      <span aria-hidden="true">{l.kind}</span><code>{l.text || ' '}</code></div>)
  })
  if (skipped) out.push(<div key="end" className="meta-harness-demo-dfold">… {skipped} unchanged lines</div>)
  const adds = lines.filter((l) => l.kind === '+').length, dels = lines.filter((l) => l.kind === '-').length
  return (
    <div className="meta-harness-demo-diff" role="region" aria-label={label} tabIndex={0}>
      <div className="meta-harness-demo-dsum">+{adds} lines, −{dels} lines</div>
      {adds + dels === 0 ? <div className="meta-harness-demo-dfold">No change.</div> : out}
    </div>
  )
}

/** True on a phone-width screen, so charts can draw with a narrower viewBox and keep their text readable. */
export function useNarrow(px = 640) {
  const q = `(max-width: ${px}px)`
  const [narrow, setNarrow] = useState(() => typeof window !== 'undefined' && window.matchMedia(q).matches)
  useEffect(() => {
    const m = window.matchMedia(q)
    const f = () => setNarrow(m.matches)
    m.addEventListener('change', f)
    return () => m.removeEventListener('change', f)
  }, [q])
  return narrow
}
