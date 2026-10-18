/* Tab 2: what the proposer sees. The same failure shown three ways (scores, a summary, the full traces and code),
   the paper's ablation for each (Table 3), how much each earlier optimizer reads (Table 1), and what the proposer
   really read in one run (Table 8). The paper's case is Appendix A.2 (TerminalBench-2, iterations 1 to 3); the second
   case is a real wrong answer from our own run (OurCase.tsx). */
import { lazy, Suspense, useEffect, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { loadPaper, type Paper } from './data'
import { Seg, Busy, Err, Src } from './ui'
import { P2, P3, P5 } from './links'

const OurCase = lazy(() => import('./OurCase'))

export type View = 'scores' | 'summary' | 'full'
const VIEWS: { v: View; label: string }[] = [
  { v: 'scores', label: 'Scores only' },
  { v: 'summary', label: 'Scores + a summary' },
  { v: 'full', label: 'Full traces and code' },
]

export default function Proposer() {
  const [p, setP] = useState<Paper | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [view, setView] = useState<View>('scores')
  const [kase, setKase] = useState<'paper' | 'ours'>('paper')
  useEffect(() => { loadPaper().then(setP).catch((e) => setError(String(e))) }, [])
  if (error) return <Err msg={error} />
  if (!p) return <Busy text="Loading the paper's numbers…" />
  return (
    <div className="meta-harness-demo-proposer">
      <section className="meta-harness-demo-step" aria-labelledby="mh-view-h">
        <div className="meta-harness-demo-stephead">
          <span className="meta-harness-demo-stepn">1</span><h3 id="mh-view-h">One failure, three ways to see it</h3>
          <span className="meta-harness-demo-read">Explained in <Link to={P2}>Part 2</Link> (Table 1) and <Link to={P3}>Part 3</Link> (Table 3)</span>
        </div>
        <p>
          The proposer is the agent that writes the next harness. It can only fix what it can see. Earlier optimizers give it
          a score, or a short summary of each attempt. Meta-Harness gives it a folder with everything: the code, the scores,
          and the traces. A <b>trace</b> is the record of one run: what the harness put in the prompt, what the model answered,
          and whether it was right. Choose a case and what the proposer may read.
        </p>
        <div className="meta-harness-demo-controls">
          <span className="meta-harness-demo-clabel">Case</span>
          <Seg label="Case" value={kase} onChange={setKase} options={[
            { v: 'paper', label: 'The paper: TerminalBench-2' },
            { v: 'ours', label: 'Our run: one wrong answer' },
          ]} />
          <span className="meta-harness-demo-clabel">The proposer may read</span>
          <Seg label="What the proposer may read" value={view} onChange={setView} options={VIEWS} />
        </div>
        {kase === 'paper'
          ? <PaperCase p={p} view={view} />
          : <Suspense fallback={<Busy text="Loading our run…" />}><OurCase view={view} /></Suspense>}
      </section>

      <Ablation p={p} view={view} />
      <Budget p={p} />
    </div>
  )
}

/* ---------------- the paper's case: Appendix A.2 ---------------- */

function Terminal({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="meta-harness-demo-term">
      <div className="meta-harness-demo-termbar"><i /><i /><i /><span>{title}</span></div>
      <div className="meta-harness-demo-termbody">{children}</div>
    </div>
  )
}

function Verdict({ children }: { children: ReactNode }) {
  return <div className="meta-harness-demo-verdict"><b>What the proposer can tell:</b> {children}</div>
}

function PaperCase({ p, view }: { p: Paper; view: View }) {
  const a = p.appendixA
  const [t1, t2, t3] = a.tries
  const pp = (x: number) => `${x > 0 ? '+' : '−'}${Math.abs(x).toFixed(1)} points`
  const changes: [string, boolean, boolean][] = [
    ['Strip leaked terminal markers from what the model sees', true, true],
    ['Add a loop breaker for turns with no tool call', true, true],
    ['New prompt: cleanup directives and a checklist', true, true],
    ['Remove the double-confirmation step', false, true],
  ]
  return (
    <div className="meta-harness-demo-case">
      <p className="meta-harness-demo-note">
        The proposer starts from Terminus-KIRA ({a.baseline}% on the 89 tasks with Opus 4.6) and tries two new harnesses.
        Both get worse. What should it try next?
      </p>
      {view === 'scores' && <>
        <Terminal title="cat runs/*/score.json">
          <pre>{`terminus_kira (start)   ${a.baseline.toFixed(1)}%
${t1.name.padEnd(22)}  ${t1.pass.toFixed(1)}%   ${t1.change.toFixed(1)}
${t2.name.padEnd(22)}  ${t2.pass.toFixed(1)}%   ${t2.change.toFixed(1)}`}</pre>
        </Terminal>
        <Verdict>both new harnesses are worse ({pp(t1.change)} and {pp(t2.change)}). Nothing says why, or which of their changes did
          the harm. The safe move is to throw both away, with any good ideas inside them.</Verdict>
      </>}
      {view === 'summary' && <>
        <Terminal title="summaries of each attempt">
          <pre>{`${t1.name}  ${t1.pass.toFixed(1)}%
  Strips leaked terminal markers, adds a loop breaker,
  a cleanup prompt and a verification checklist. Worse.

${t2.name}  ${t2.pass.toFixed(1)}%
  Removes the double-confirmation step. Keeps the marker
  stripping and the new prompt. Worse.`}</pre>
        </Terminal>
        <Verdict>what each attempt changed, one attempt at a time. A summary of one run does not say what went wrong inside it, so
          the cause is still a guess. A reader could blame the marker fix, which was fine.</Verdict>
        <Src>We wrote these two summaries from Appendix A.2, in the short form a summarizing optimizer passes on.</Src>
      </>}
      {view === 'full' && <>
        <div className="meta-harness-demo-tablewrap">
          <table className="meta-harness-demo-table is-changes">
            <caption>What changed in each harness's code (from Appendix A.2)</caption>
            <thead><tr><th>Change</th><th>{t1.name}</th><th>{t2.name}</th></tr></thead>
            <tbody>
              {changes.map(([c, x, y], i) => (
                <tr key={c} className={i === 2 ? 'is-sel' : ''}><th>{c}</th><td>{x ? '✓' : ''}</td><td>{y ? '✓' : ''}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
        <Terminal title="the proposer's notes, iteration 3 (quoted from the paper)">
          <pre className="is-wrap">{`Prior attempts: evo marker fix (58.9%, -5.6pp), evo single confirm (57.8%, -6.7pp) --- both regressed.
Root cause of regressions: Prompt template changes (cleanup directives) caused the agent to delete necessary state before task completion. The structural bugfixes were confounded with harmful prompt changes.
evo strip only isolates the two proven structural fixes.`}</pre>
        </Terminal>
        <Verdict>both failures share one change, the cleanup prompt, and the traces show agents deleting state they still needed. So
          it keeps the fixes and drops the prompt. That harness ({t3.name}) scores {t3.pass}% ({pp(t3.change)}): much closer to the start,
          which supports the diagnosis. Four iterations later, a purely additive change (a snapshot of the sandbox in the first
          prompt) becomes the best harness of the run.</Verdict>
      </>}
      <p className="meta-harness-demo-note">
        In this run the proposer read a median of {p.table8.files_median} files per iteration (range {p.table8.files_range}):{' '}
        {p.table8.breakdown.map((b) => `${b.kind.toLowerCase()} ${b.pct}%`).join(', ')} ({p.table8.source}). The whole story is in <Link to={P5}>Part 5</Link>.
      </p>
      <div className="meta-harness-demo-stack" role="img" aria-label="What the proposer read: harness code 41%, traces 40%, scores 6%, other 13%">
        {p.table8.breakdown.map((b, i) => <span key={b.kind} className={`is-${i}`} style={{ flexBasis: `${b.pct}%` }} title={`${b.kind}: ${b.pct}%`} />)}
      </div>
      <div className="meta-harness-demo-legend" aria-hidden="true">
        {p.table8.breakdown.map((b, i) => <span key={b.kind}><i className={`is-stack${i}`} />{b.kind} {b.pct}%</span>)}
      </div>
    </div>
  )
}

/* ---------------- Table 3 and Table 4 ---------------- */

function Ablation({ p, view }: { p: Paper; view: View }) {
  const max = 60
  const rows = p.table3.rows
  return (
    <section className="meta-harness-demo-step" aria-labelledby="mh-abl-h">
      <div className="meta-harness-demo-stephead">
        <span className="meta-harness-demo-stepn">2</span><h3 id="mh-abl-h">What each view is worth: the paper's ablation</h3>
        <span className="meta-harness-demo-read">Explained in <Link to={P3}>Part 3</Link> (Table 3)</span>
      </div>
      <p>
        The paper ran the same search on text classification three times, with the same budget. Only what the proposer could read
        changed. In all three runs it could read the harness code and the scores. The bars are the search-set accuracy of the
        harnesses each proposer wrote. Your choice above is highlighted.
      </p>
      <figure className="meta-harness-demo-fig">
        <ul className="meta-harness-demo-abl" aria-label="Median and best accuracy for each proposer view">
          {rows.map((r) => (
            <li key={r.id} className={r.id === view ? 'is-sel' : ''}>
              <span className="meta-harness-demo-aname">{r.method}<small>{[r.scores && 'scores', r.code && 'code', r.summary && 'summaries', r.traces && 'traces'].filter(Boolean).join(' + ')}</small></span>
              <span className="meta-harness-demo-abars">
                <span className="meta-harness-demo-abar is-med"><span><i style={{ width: `${(100 * r.median) / max}%` }} /></span><em>median {r.median.toFixed(1)}</em></span>
                <span className="meta-harness-demo-abar is-best"><span><i style={{ width: `${(100 * r.best) / max}%` }} /></span><em>best {r.best.toFixed(1)}</em></span>
              </span>
              <span className="meta-harness-demo-azs"><b>{r.above_zs}</b> above zero-shot</span>
            </li>
          ))}
        </ul>
        <Src>{p.table3.source}, accuracy in % on the search set. "Above zero-shot" counts the candidate harnesses that beat a prompt with no examples.</Src>
      </figure>
      <p>
        {view === 'scores' && 'With scores and code alone, the median harness reaches 34.6. The proposer knows a harness failed but must guess where.'}
        {view === 'summary' && 'Adding summaries does not help: the median is 34.9 and the best is lower, 38.7. A summary is written before anyone knows which detail matters, so it can drop the one clue that would explain the failure.'}
        {view === 'full' && 'With the raw traces, the median harness reaches 50.0, above the best harness of either other run. The proposer can read the exact examples a harness got wrong and the prompt it built for them, so it can trace the failure back to one line of code.'}
      </p>
      <details className="meta-harness-demo-details">
        <summary>Against other optimizers, same budget (Table 4)</summary>
        <div className="meta-harness-demo-tablewrap">
          <table className="meta-harness-demo-table">
            <thead><tr><th>Optimizer</th><th>Median</th><th>Best</th></tr></thead>
            <tbody>{p.table4.rows.map((r) => <tr key={r.method} className={r.method === 'Meta-Harness' ? 'is-sel' : ''}><th>{r.method}</th><td>{r.median.toFixed(1)}</td><td>{r.best.toFixed(1)}</td></tr>)}</tbody>
          </table>
        </div>
        <Src>{p.table4.source}, search-set accuracy (%) of the harnesses each optimizer proposed.</Src>
      </details>
    </section>
  )
}

/* ---------------- Table 1: how much each optimizer reads ---------------- */

function Budget({ p }: { p: Paper }) {
  const rows = p.table1.rows
  const lo = Math.log10(0.001), hi = Math.log10(20)
  const w = (v: number) => `${(100 * (Math.log10(v) - lo)) / (hi - lo)}%`
  return (
    <section className="meta-harness-demo-step" aria-labelledby="mh-budget-h">
      <div className="meta-harness-demo-stephead">
        <span className="meta-harness-demo-stepn">3</span><h3 id="mh-budget-h">How much each optimizer can read</h3>
        <span className="meta-harness-demo-read">Explained in <Link to={P2}>Part 2</Link> (Table 1)</span>
      </div>
      <p>
        For each method, the paper estimates how much text one evaluation produces for the optimizer, in millions of tokens
        (MTok). Each step on this scale is ten times the one before. Meta-Harness works with about 400 times more than the
        next method, because it never squeezes the logs into one prompt: the proposer opens the files it needs.
      </p>
      <figure className="meta-harness-demo-fig">
        <ul className="meta-harness-demo-log" aria-label="Context per iteration for each method, log scale">
          {rows.map((r) => (
            <li key={r.method} className={r.method === 'Meta-Harness' ? 'is-sel' : ''}>
              <span className="meta-harness-demo-lname">{r.method}<small>{r.history.toLowerCase()} · {r.log}</small></span>
              <span className="meta-harness-demo-ltrack"><i style={{ width: w(r.mtok) }} /></span>
              <span className="meta-harness-demo-lval">{r.mtok}</span>
            </li>
          ))}
        </ul>
        <div className="meta-harness-demo-lticks" aria-hidden="true">
          {[0.001, 0.01, 0.1, 1, 10].map((v) => <span key={v} style={{ left: w(v) }}>{v}</span>)}
        </div>
        <Src>{p.table1.source}. MTok per iteration, log scale. History: which earlier attempts the optimizer sees (a window of recent ones, the last one, a summary, or all of them).</Src>
      </figure>
    </section>
  )
}
