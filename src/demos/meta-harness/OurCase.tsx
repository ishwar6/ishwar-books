/* Tab 2, second case: one real wrong answer from our saved run, shown as a proposer would get it under each view.
   The case is picked from the run: the child harness with the largest search gain over its parent, and a search
   message the parent got wrong and the child got right. Every line shown comes from the run's files. */
import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { loadRun, loadRunIndex, loadTraces, type Cand, type Run, type TraceFull } from './data'
import { Busy, Err, Src } from './ui'
import { TraceCard } from './Run'
import type { View } from './Proposer'

const short = (f: string) => f.replace(/^\d{3}_/, '')
const pct = (x: number) => `${x.toFixed(1)}%`
const goldShown = (t: TraceFull) => t.calls.some((c) => c.prompt.includes(`Label: ${t.gold}\n`) || c.prompt.includes(`: ${t.gold}\n`))

function pickCase(run: Run) {
  const byName = new Map(run.candidates.map((c) => [c.folder, c]))
  let best: { parent: Cand; child: Cand; gain: number } | null = null
  for (const c of run.candidates) {
    const p = c.parent ? byName.get(c.parent) : undefined
    if (!p || c.seed) continue
    const gain = c.search.accuracy - p.search.accuracy
    if (!best || gain > best.gain) best = { parent: p, child: c, gain }
  }
  return best
}

export default function OurCase({ view }: { view: View }) {
  const [run, setRun] = useState<Run | null>(null)
  const [tr, setTr] = useState<TraceFull[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    loadRunIndex().then((ix) => loadRun(ix.runs[0].file)).then(setRun).catch((e) => setError(String(e)))
  }, [])
  const kase = useMemo(() => (run ? pickCase(run) : null), [run])
  useEffect(() => {
    if (!run || !kase) return
    loadTraces(run.name, kase.parent.folder).then(setTr).catch((e) => setError(String(e)))
  }, [run, kase])
  if (error) return <Err msg={`The saved run could not be loaded (${error}).`} />
  if (!run || (kase && !tr)) return <Busy text="Loading our run…" />
  if (!kase || !tr) return <p className="meta-harness-demo-note">This run has no proposed harness with a parent to compare.</p>
  const { parent: P, child: C } = kase
  const fixed = tr.filter((t) => !t.ok && C.ok[t.id])
  const shown = fixed.find((t) => !goldShown(t)) ?? fixed[0] ?? tr.find((t) => !t.ok)!
  const wrong = tr.filter((t) => !t.ok)
  const noGold = wrong.filter((t) => !goldShown(t)).length
  const before = run.candidates.filter((c) => c.iter < C.iter)
  const edit = (C.proposal.match(/^(Edit|IDEA|Change|Changes|Hypothesis)[^:\n]*:\s*(.+)$/im)?.[2] ?? C.proposal.split('\n').filter((l) => l.trim() && !l.startsWith('#'))[0] ?? '').trim()

  return (
    <div className="meta-harness-demo-case">
      <p className="meta-harness-demo-note">
        From our saved run <code>{run.name}</code>. Before iteration {C.iter}, the proposer has {before.length} harnesses in its folder.
        One of them, <code>{short(P.folder)}</code>, gets {pct(P.search.accuracy)} of the search messages right. What should it change?
      </p>
      {view === 'scores' && <>
        <Term title="grep accuracy candidates/*/scores.json">
          <pre>{before.map((c) => `${c.folder.padEnd(34)} ${pct(c.search.accuracy).padStart(6)}`).join('\n')}</pre>
        </Term>
        <Verdict>which harness is best so far, and that {short(P.folder)} misses {wrong.length} of {P.search.n} messages. Not which ones,
          and not why. A next harness can only be a guess: more examples, fewer examples, another wording.</Verdict>
      </>}
      {view === 'summary' && <>
        <Term title="a summary of each candidate">
          <pre className="is-wrap">{before.map((c) => summary(c)).join('\n\n')}</pre>
        </Term>
        <Verdict>which labels {short(P.folder)} confuses. That narrows the search, but a summary still hides the prompts. It cannot
          show that the model often never saw an example of the right label.</Verdict>
        <Src>We made these summaries from each candidate's scores.json (accuracy, the most common confusion, the weakest label): the short, fixed form of feedback that summary-based optimizers pass on.</Src>
      </>}
      {view === 'full' && <>
        <p className="meta-harness-demo-src">One line of <code>{P.folder}/traces.jsonl</code>: the message, the answer, and the exact prompt the harness built.</p>
        <TraceCard t={shown} open />
        <Verdict>
          the gold label <code>{shown.gold}</code> {goldShown(shown) ? 'does appear' : 'has no example'} in this prompt.
          Across all {wrong.length} wrong answers of {short(P.folder)}, {noGold} prompts had no example of the gold label. That points at
          one part of the code: how the harness picks its examples. The next proposal, <code>{short(C.folder)}</code>, says: “{edit.replace(/[.\s]+$/, '')}.”
          It scores {pct(C.search.accuracy)} on the search set ({C.search.accuracy - P.search.accuracy >= 0 ? '+' : '−'}{Math.abs(C.search.accuracy - P.search.accuracy).toFixed(1)} points)
          {C.ok[shown.id] ? ', and it gets this message right.' : '.'}
        </Verdict>
      </>}
      <Src>The proposer in this run is {run.meta.proposer === 'scripted' ? 'a script with fixed rules, not an LLM: it reads the traces and edits a small design space. It shows what the traces make possible; the paper\'s proposer is a strong coding agent' : run.meta.proposer}.</Src>
    </div>
  )
}

function summary(c: Cand) {
  const conf = Object.entries(c.search.top_confusions)[0]
  const worst = Object.entries(c.search.accuracy_by_label).map(([l, s]) => { const [a, b] = s.split('/').map(Number); return { l, a, b } })
    .sort((x, y) => x.a / x.b - y.a / y.b)[0]
  return `${short(c.folder)}: ${pct(c.search.accuracy)}.` +
    (conf ? ` Most common mistake: ${conf[0].replace(' -> ', ' answered as ')} (${conf[1]}x).` : '') +
    (worst ? ` Weakest label: ${worst.l} (${worst.a}/${worst.b}).` : '')
}

function Term({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="meta-harness-demo-term">
      <div className="meta-harness-demo-termbar"><i /><i /><i /><span>{title}</span></div>
      <div className="meta-harness-demo-termbody" tabIndex={0}>{children}</div>
    </div>
  )
}
function Verdict({ children }: { children: ReactNode }) {
  return <div className="meta-harness-demo-verdict"><b>What the proposer can tell:</b> {children}</div>
}
