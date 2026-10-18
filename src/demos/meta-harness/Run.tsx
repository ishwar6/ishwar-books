/* Tab 1: inside a real search run of our small Meta-Harness (code/papers/meta-harness/scratch/, exported by
   code/papers/meta-harness/demo/export_run.py). The chart shows search-set accuracy per candidate with the best-so-far
   line and the test-set accuracy beside it; the explorer shows the filesystem the proposer reads, one folder per
   candidate, with its code, a diff against its parent, its scores and its traces. */
import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { loadRun, loadRunIndex, loadTraces, proposerName, type Cand, type Rejected, type Run as R, type RunIndex, type TraceFull } from './data'
import { Busy, Diff, diffLines, Err, Seg, Src, useNarrow } from './ui'
import { P2, P5, P6, SCRATCH, DEMO_CODE } from './links'

type Tab = 'proposal' | 'code' | 'diff' | 'traces' | 'scores' | 'test'
const short = (f: string) => f.replace(/^\d{3}_/, '')
const pct = (x: number) => `${x.toFixed(1)}%`
const kb = (b: number) => (b < 1024 ? `${b} B` : `${(b / 1024).toFixed(1)} KB`)

export default function Run() {
  const [index, setIndex] = useState<RunIndex | null>(null)
  const [file, setFile] = useState<string | null>(null)
  const [run, setRun] = useState<R | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    loadRunIndex().then((ix) => { setIndex(ix); setFile(ix.runs[0]?.file ?? null) }).catch((e) => setError(String(e)))
  }, [])
  useEffect(() => {
    if (!file) return
    let stop = false
    loadRun(file).then((r) => { if (!stop) setRun(r) }).catch((e) => setError(String(e)))
    return () => { stop = true }
  }, [file])
  if (error) return <Err msg={`The saved run could not be loaded (${error}).`} />
  if (!index || !run || run.name + '.json' !== file) return <Busy text="Loading the saved search run…" />
  return (
    <div className="meta-harness-demo-run">
      <p className="meta-harness-demo-note">
        This is a real search, run on a laptop by the code of <Link to={P6}>Part 6</Link>. The task model is
        frozen: <code>{run.meta.task_model}</code>. It labels a bank customer's message with one of {run.meta.labels.length} card
        intents from Banking77. A harness decides which of the {run.meta.sizes.memory} labelled messages it has seen go into
        the prompt, and how. Each candidate is scored on {run.meta.sizes.search} search messages. The {run.meta.sizes.test} test
        messages were scored once, after the search, and the proposer never read them.
      </p>
      {index.runs.length > 1 && (
        <div className="meta-harness-demo-controls">
          <span className="meta-harness-demo-clabel">Run</span>
          <Seg label="Saved run" value={file!} onChange={setFile}
            options={index.runs.map((r) => ({ v: r.file, label: `${r.name} (${r.candidates} candidates)` }))} />
        </div>
      )}
      <RunView key={run.name} run={run} />
    </div>
  )
}

function RunView({ run }: { run: R }) {
  const m = run.meta
  const cands = run.candidates
  const [sel, setSel] = useState<string>(m.selected)
  const [tab, setTab] = useState<Tab>('proposal')
  const cur = cands.find((c) => c.folder === sel)
  const rej = run.rejected.find((c) => c.folder === sel)
  const selected = cands.find((c) => c.folder === m.selected)!
  const seeds = cands.filter((c) => c.seed)
  const bestSeed = seeds.reduce((a, b) => (b.search.accuracy > a.search.accuracy ? b : a), seeds[0])
  const bestTest = cands.reduce<Cand | null>((a, b) => (b.test && (!a?.test || b.test.acc > a.test.acc) ? b : a), null)
  function pick(folder: string, t?: Tab) { setSel(folder); if (t) setTab(t) }

  return (
    <>
      <ul className="meta-harness-demo-facts">
        <li>Proposer: <b>{proposerName(m.proposer)}</b>{m.proposer_model ? <> ({m.proposer_model})</> : null}</li>
        <li><b>{m.iterations}</b> iterations × <b>{m.k}</b> proposals</li>
        <li><b>{cands.length}</b> harnesses evaluated ({seeds.length} seeds)</li>
        <li><b>{m.stats.rejected}</b> rejected before evaluation{m.stats.duplicates ? ` (${m.stats.duplicates} duplicates)` : ''}</li>
        <li>Search took <b>{m.minutes_search}</b> min, all {m.minutes_total} min ({m.device})</li>
      </ul>
      <p className="meta-harness-demo-note">
        {m.proposer === 'scripted'
          ? <>The proposer here is a script, not an LLM. It reads the parent's traces, counts the kinds of mistakes, and changes one
            setting of a template harness. It is the honest baseline: it shows what a search over this folder can find without a
            strong coding agent.</>
          : <>Here a small local model, <code>{m.proposer_model ?? 'an LLM'}</code>, asks for files from the folder, reads them, and
            writes each harness. It is far weaker than the paper's proposer: of its {m.stats.proposed} proposals, {m.stats.rejected} were
            rejected ({m.stats.duplicates} were copies of earlier code; the others failed the interface check). Its proposals
            name the files it read.</>}
      </p>

      <section className="meta-harness-demo-step" aria-labelledby="mh-chart-h">
        <div className="meta-harness-demo-stephead">
          <span className="meta-harness-demo-stepn">1</span><h3 id="mh-chart-h">Search score over the run, and the test score beside it</h3>
          <span className="meta-harness-demo-read">Search and test sets: <Link to={P2}>Part 2</Link> · overfitting: <Link to={P5}>Part 5</Link></span>
        </div>
        <p>
          Each dot is one harness, in the order it was evaluated. Teal is its accuracy on the search set: the only score the
          proposer sees. The line is the best so far. Purple is the same harness on the held-out test set. The distance
          between them is the <b>overfitting gap</b>: how much of the search score came from fitting those {m.sizes.search} messages.
          Click a dot to open its folder.
        </p>
        <Chart run={run} sel={sel} onSel={(f) => pick(f)} />
        <p className="meta-harness-demo-note">
          The search picks <code>{short(selected.folder)}</code> by its search score alone: {pct(selected.search.accuracy)} on search,
          {selected.test ? <> {pct(selected.test.acc)} on test (gap {gapText(selected.search.accuracy - selected.test.acc)}).</> : ' not scored on test.'}
          {' '}The best seed, <code>{short(bestSeed.folder)}</code>, has {pct(bestSeed.search.accuracy)} on search
          {bestSeed.test ? <> and {pct(bestSeed.test.acc)} on test</> : null}.
          {selected.test && bestSeed.test ? <> So on new messages the search gained {gapText(selected.test.acc - bestSeed.test.acc, true)} over its
            best starting point, against {gapText(selected.search.accuracy - bestSeed.search.accuracy, true)} on the messages it was tuned on.</> : null}
          {bestTest && bestTest.folder !== selected.folder && bestTest.test && <> The best harness on the test set was <code>{short(bestTest.folder)}</code> ({pct(bestTest.test.acc)}),
            but the search cannot know that: it never sees the test set. With {m.sizes.search} search messages, one message is {+(100 / m.sizes.search).toFixed(1)} points, so small search gains can be luck.</>}
        </p>
      </section>

      <section className="meta-harness-demo-step" aria-labelledby="mh-fs-h">
        <div className="meta-harness-demo-stephead">
          <span className="meta-harness-demo-stepn">2</span><h3 id="mh-fs-h">The filesystem the proposer reads</h3>
          <span className="meta-harness-demo-read">Explained in <Link to={P2}>Part 2</Link> (Figure 2, Algorithm 1) · the code: <a href={SCRATCH}>scratch/</a></span>
        </div>
        <p>
          Every evaluated harness gets a folder: its code (<code>harness.py</code>), its scores, one trace line per search
          message, and the proposal that explains why it was written. The proposer reads these files to write the next
          harness. Harnesses that fail the interface check go to <code>rejected/</code>. The test results in <code>heldout/</code> are
          written only after the search ends.
        </p>
        <div className="meta-harness-demo-explorer">
          <Tree run={run} sel={sel} tab={tab} onPick={pick} />
          <div className="meta-harness-demo-cand">
            {cur ? <CandView run={run} c={cur} tab={tab} setTab={setTab} onPick={pick} />
              : rej ? <RejView r={rej} onPick={pick} /> : null}
          </div>
        </div>
        <Src>Saved run <code>scratch/runs/{run.name}/</code>, exported by <a href={DEMO_CODE}>export_run.py</a>. Every number here is from the run's own files.</Src>
      </section>
    </>
  )
}

const gapText = (g: number, signed = false) => `${signed && g > 0 ? '+' : g < 0 ? '−' : ''}${Math.abs(g).toFixed(1)} points`

/* ---------------- the chart ---------------- */

function Chart({ run, sel, onSel }: { run: R; sel: string; onSel: (f: string) => void }) {
  const cands = run.candidates
  const narrow = useNarrow()
  const W = narrow ? 420 : 760, H = narrow ? 280 : 300, ml = 36, mr = narrow ? 58 : 84, mt = 16, mb = 40
  const n = cands.length
  const accs = cands.flatMap((c) => [c.search.accuracy, c.test?.acc ?? c.search.accuracy])
  const lo = Math.max(0, Math.floor((Math.min(...accs) - 5) / 10) * 10)
  const hi = Math.min(100, Math.ceil((Math.max(...accs) + 5) / 10) * 10)
  const x = (i: number) => ml + ((i + 0.5) / n) * (W - ml - mr)
  const y = (v: number) => mt + (1 - (v - lo) / (hi - lo)) * (H - mt - mb)
  // best so far by search accuracy (ties keep the earlier one, as a search would)
  const bestIdx: number[] = []
  cands.forEach((c, i) => { bestIdx.push(i && c.search.accuracy <= cands[bestIdx[i - 1]].search.accuracy ? bestIdx[i - 1] : i) })
  const step = (val: (i: number) => number | null) => {
    let d = ''
    bestIdx.forEach((b, i) => {
      const v = val(b)
      if (v === null) return
      const x0 = ml + (i / n) * (W - ml - mr), x1 = ml + ((i + 1) / n) * (W - ml - mr)
      d += `${d ? 'L' : 'M'}${x0.toFixed(1)},${y(v).toFixed(1)}L${x1.toFixed(1)},${y(v).toFixed(1)}`
    })
    return d
  }
  // iteration bands
  const iters = [...new Set(cands.map((c) => c.iter))]
  const last = cands.length - 1, lb = cands[bestIdx[last]]
  const ticks = Array.from({ length: (hi - lo) / 10 + 1 }, (_, i) => lo + i * 10)
  const curI = cands.findIndex((c) => c.folder === sel)
  const cur = cands[curI]
  return (
    <figure className="meta-harness-demo-fig">
      <svg className="meta-harness-demo-svg" viewBox={`0 0 ${W} ${H}`} role="img"
        aria-label="Search-set and test-set accuracy of every candidate, in the order they were evaluated, with the best-so-far line">
        {ticks.map((t) => (
          <g key={t}>
            <line className="meta-harness-demo-grid" x1={ml} x2={W - mr} y1={y(t)} y2={y(t)} />
            <text className="meta-harness-demo-axis" x={ml - 8} y={y(t) + 4} textAnchor="end">{t}</text>
          </g>
        ))}
        {iters.map((it) => {
          const idx = cands.map((c, i) => (c.iter === it ? i : -1)).filter((i) => i >= 0)
          const x0 = ml + (idx[0] / n) * (W - ml - mr), x1 = ml + ((idx[idx.length - 1] + 1) / n) * (W - ml - mr)
          return (
            <g key={it}>
              {it > 0 && <line className="meta-harness-demo-ref" x1={x0} x2={x0} y1={mt} y2={H - mb} />}
              <text className="meta-harness-demo-axis" x={(x0 + x1) / 2} y={H - mb + 16} textAnchor="middle">{it === 0 ? 'seeds' : n > 24 || narrow ? it : `iter ${it}`}</text>
            </g>
          )
        })}
        <text className="meta-harness-demo-axis" x={(ml + W - mr) / 2} y={H - 4} textAnchor="middle">candidates in evaluation order, by iteration</text>
        <text className="meta-harness-demo-axis" transform={`translate(12 ${(mt + H - mb) / 2}) rotate(-90)`} textAnchor="middle">accuracy (%)</text>
        <path className="meta-harness-demo-best" d={step((b) => cands[b].search.accuracy)} />
        <path className="meta-harness-demo-besttest" d={step((b) => cands[b].test?.acc ?? null)} />
        {lb.test && (
          <g>
            <path className="meta-harness-demo-bracket" d={`M${W - mr + 4},${y(lb.search.accuracy)}h6V${y(lb.test.acc)}h-6`} />
            <text className="meta-harness-demo-gaptag" x={W - mr + 14} y={(y(lb.search.accuracy) + y(lb.test.acc)) / 2 - 3}>gap</text>
            <text className="meta-harness-demo-gaptag" x={W - mr + 14} y={(y(lb.search.accuracy) + y(lb.test.acc)) / 2 + 13}>{(lb.search.accuracy - lb.test.acc).toFixed(1)} pts</text>
          </g>
        )}
        {cands.map((c, i) => (
          <g key={c.folder} className={`meta-harness-demo-chartpt${c.folder === sel ? ' is-sel' : ''}`} role="button" tabIndex={0}
            aria-label={`${c.folder}: search ${pct(c.search.accuracy)}${c.test ? `, test ${pct(c.test.acc)}` : ''}`} aria-pressed={c.folder === sel}
            onClick={() => onSel(c.folder)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSel(c.folder) } }}>
            <rect className="meta-harness-demo-hit" x={x(i) - (W - ml - mr) / n / 2} width={(W - ml - mr) / n} y={mt} height={H - mt - mb} rx={4} />
            {c.test && <line className="meta-harness-demo-gap" x1={x(i)} x2={x(i)} y1={y(c.search.accuracy)} y2={y(c.test.acc)} />}
            {c.test && <circle className="meta-harness-demo-tdot" cx={x(i)} cy={y(c.test.acc)} r={4.5} />}
            <circle className="meta-harness-demo-sdot" cx={x(i)} cy={y(c.search.accuracy)} r={5.5} />
          </g>
        ))}
      </svg>
      <div className="meta-harness-demo-legend">
        <span><i className="is-search" />search accuracy</span>
        <span><i className="is-line" />best so far (search)</span>
        <span><i className="is-test" />test accuracy</span>
        <span><i className="is-line" style={{ background: 'var(--mh-test)' }} />test accuracy of the best so far</span>
      </div>
      {cur && <p className="meta-harness-demo-src">Selected: <code>{cur.folder}</code>, search {pct(cur.search.accuracy)}
        {cur.test ? `, test ${pct(cur.test.acc)}` : ''}, {cur.search.ctx_chars.toLocaleString()} prompt characters per message.</p>}
    </figure>
  )
}

/* ---------------- the file tree ---------------- */

function Tree({ run, sel, tab, onPick }: { run: R; sel: string; tab: Tab; onPick: (f: string, t?: Tab) => void }) {
  const best = run.meta.selected
  const fileTab: Record<string, Tab> = { 'harness.py': 'code', 'scores.json': 'scores', 'traces.jsonl': 'traces', 'proposal.md': 'proposal' }
  return (
    <nav className="meta-harness-demo-tree" aria-label="The run's folders">
      <div className="meta-harness-demo-troot">runs/{run.name}/</div>
      <ul>
        <li><div className="meta-harness-demo-troot">candidates/</div>
          <ul>
            {run.candidates.map((c) => (
              <li key={c.folder}>
                <button type="button" className={`meta-harness-demo-tnode${c.folder === best ? ' is-best' : ''}`} aria-current={c.folder === sel}
                  aria-expanded={c.folder === sel} onClick={() => onPick(c.folder)}>
                  <span aria-hidden="true">{c.folder === sel ? '▾' : '▸'}</span>
                  <span className="meta-harness-demo-tname">{c.folder}/</span>
                  <span className="meta-harness-demo-tscore">{c.search.accuracy.toFixed(0)}%</span>
                </button>
                {c.folder === sel && (
                  <ul>
                    {c.files.map((f) => (
                      <li key={f.path}>
                        <button type="button" className="meta-harness-demo-tfile is-btn" aria-current={fileTab[f.path] === tab}
                          onClick={() => onPick(c.folder, fileTab[f.path] ?? 'proposal')}>
                          <span>{f.path}</span><span>{kb(f.bytes)}</span>
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </li>
            ))}
          </ul>
        </li>
        {run.rejected.length > 0 && (
          <li><div className="meta-harness-demo-troot">rejected/</div>
            <ul>
              {run.rejected.map((r) => (
                <li key={r.folder}>
                  <button type="button" className="meta-harness-demo-tnode is-bad" aria-current={r.folder === sel} onClick={() => onPick(r.folder)}>
                    <span aria-hidden="true">×</span><span className="meta-harness-demo-tname">{r.folder}/</span>
                    <span className="meta-harness-demo-tscore">rejected</span>
                  </button>
                </li>
              ))}
            </ul>
          </li>
        )}
        <li><div className="meta-harness-demo-troot">heldout/ <span>(not readable during the search)</span></div>
          <ul>
            {run.candidates.some((c) => c.folder === sel && c.test) ? (
              <li>
                <button type="button" className="meta-harness-demo-tfile is-btn" aria-current={tab === 'test'} onClick={() => onPick(sel, 'test')}>
                  <span>{sel}.json</span><span>test</span>
                </button>
              </li>
            ) : <li className="meta-harness-demo-tfile"><span>(no test result for this folder)</span></li>}
          </ul>
        </li>
      </ul>
    </nav>
  )
}

/* ---------------- one candidate ---------------- */

function CandView({ run, c, tab, setTab, onPick }: { run: R; c: Cand; tab: Tab; setTab: (t: Tab) => void; onPick: (f: string, t?: Tab) => void }) {
  const parent = c.parent ? run.candidates.find((x) => x.folder === c.parent) : undefined
  const diff = useMemo(() => (parent ? diffLines(parent.code, c.code) : null), [parent, c.code])
  const tabs: { v: Tab; label: string }[] = [
    { v: 'proposal', label: 'proposal.md' }, { v: 'code', label: 'harness.py' },
    ...(parent ? [{ v: 'diff' as Tab, label: 'Diff vs parent' }] : []),
    { v: 'traces', label: 'Traces' }, { v: 'scores', label: 'scores.json' }, { v: 'test', label: 'Test (held out)' },
  ]
  const t = tab === 'diff' && !parent ? 'code' : tab
  return (
    <>
      <div className="meta-harness-demo-candhead">
        <h4>{c.folder}/</h4>
        <span>{c.seed ? 'a seed harness' : `iteration ${c.iter}`}
          {parent && <> · parent <button type="button" className="meta-harness-demo-linkbtn" onClick={() => onPick(parent.folder)}>{parent.folder}</button></>}
          {c.folder === run.meta.selected && ' · selected by the search'}{c.frontier && ' · on the Pareto frontier'}</span>
      </div>
      <div className="meta-harness-demo-scores">
        <div className="meta-harness-demo-score is-search"><small>Search set ({c.search.n})</small><b>{pct(c.search.accuracy)}</b>
          {parent && <small>{gapText(c.search.accuracy - parent.search.accuracy, true)} vs parent</small>}</div>
        <div className="meta-harness-demo-score is-test"><small>Test set ({c.test?.n ?? '–'})</small><b>{c.test ? pct(c.test.acc) : '–'}</b>
          {c.test && <small>gap {gapText(c.search.accuracy - c.test.acc)}</small>}</div>
        <div className="meta-harness-demo-score"><small>Prompt per message</small><b>{c.search.ctx_chars.toLocaleString()}</b><small>characters</small></div>
        <div className="meta-harness-demo-score"><small>Model calls per message</small><b>{c.search.calls_per_message}</b></div>
      </div>
      <Seg label="File to show" value={t} onChange={setTab} options={tabs} />
      <div className="meta-harness-demo-candbody">
        {t === 'proposal' && <Term title={`${c.folder}/proposal.md`}><pre className="is-wrap">{c.proposal || '(no proposal file)'}</pre></Term>}
        {t === 'code' && <Term title={`${c.folder}/harness.py`}><Code text={c.code} /></Term>}
        {t === 'diff' && diff && parent && <>
          <p className="meta-harness-demo-src">From <code>{parent.folder}/harness.py</code> to <code>{c.folder}/harness.py</code>. Lines marked + are new; lines marked − are gone.</p>
          <Diff lines={diff} label={`Diff from ${parent.folder} to ${c.folder}`} />
        </>}
        {t === 'traces' && <Traces run={run} c={c} />}
        {t === 'scores' && <Term title={`${c.folder}/scores.json`}><pre>{JSON.stringify(c.search, null, 1)}</pre></Term>}
        {t === 'test' && <TestView c={c} />}
      </div>
    </>
  )
}

function Term({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="meta-harness-demo-term">
      <div className="meta-harness-demo-termbar"><i /><i /><i /><span>{title}</span></div>
      <div className="meta-harness-demo-termbody" tabIndex={0}>{children}</div>
    </div>
  )
}

function Code({ text }: { text: string }) {
  return <pre className="meta-harness-demo-code">{text.replace(/\n$/, '').split('\n').map((l, i) => <span key={i}>{l || ' '}</span>)}</pre>
}

function Traces({ run, c }: { run: R; c: Cand }) {
  const [tr, setTr] = useState<TraceFull[] | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [show, setShow] = useState<'wrong' | 'right'>('wrong')
  useEffect(() => {
    let stop = false
    setTr(null)
    loadTraces(run.name, c.folder).then((x) => { if (!stop) setTr(x) }).catch((e) => setErr(String(e)))
    return () => { stop = true }
  }, [run.name, c.folder])
  const wrong = c.ok.filter((v) => !v).length
  const conf = Object.entries(c.search.top_confusions)
  return (
    <>
      <p className="meta-harness-demo-src">
        One line of <code>traces.jsonl</code> per search message: the message, the gold label, the answer, and every prompt the
        harness sent with the model's reply. {wrong} of {c.search.n} are wrong.
        {conf.length > 0 && <> Most common mistake: <code>{conf[0][0].replace(' -> ', ' → ')}</code> ({conf[0][1]} times).</>}
      </p>
      <div className="meta-harness-demo-okstrip" role="img" aria-label={`Right and wrong per search message: ${c.search.correct} right, ${wrong} wrong`}>
        {c.ok.map((v, i) => <i key={i} className={v ? 'is-right' : 'is-wrong'} title={`message ${i}: ${v ? 'right' : 'wrong'}`} />)}
      </div>
      <div className="meta-harness-demo-controls">
        <Seg label="Which traces" value={show} onChange={setShow} options={[{ v: 'wrong', label: `Wrong (${wrong})` }, { v: 'right', label: `Right (${c.search.correct})` }]} />
      </div>
      {err ? <Err msg={err} /> : !tr ? <Busy text="Loading the traces…" /> : (
        <div className="meta-harness-demo-traces">
          {tr.filter((x) => (show === 'wrong' ? !x.ok : x.ok)).slice(0, 8).map((x) => <TraceCard key={x.id} t={x} />)}
          {tr.filter((x) => (show === 'wrong' ? !x.ok : x.ok)).length > 8 && <p className="meta-harness-demo-src">Showing the first 8.</p>}
        </div>
      )}
    </>
  )
}

export function TraceCard({ t, open = false }: { t: TraceFull; open?: boolean }) {
  return (
    <div className={`meta-harness-demo-trace ${t.ok ? 'is-right' : 'is-wrong'}`}>
      <p>“{t.text}”</p>
      <div className="meta-harness-demo-tmeta">
        <span>gold <b>{t.gold}</b></span>
        <span className={t.ok ? 'is-right' : 'is-wrong'}>answer <b>{t.pred ?? 'none'}</b></span>
        <span>{t.calls.length} call{t.calls.length === 1 ? '' : 's'}, {t.chars.toLocaleString()} prompt characters</span>
      </div>
      {t.error && <p className="meta-harness-demo-src">Error: {t.error}</p>}
      {t.calls.map((c, i) => (
        <details key={i} open={open}>
          <summary>Call {i + 1}: the prompt and the reply</summary>
          <pre>{c.prompt}</pre>
          <pre>{'→ ' + c.reply}</pre>
        </details>
      ))}
    </div>
  )
}

function TestView({ c }: { c: Cand }) {
  if (!c.test) return <p className="meta-harness-demo-note">This candidate was not scored on the test set.</p>
  return (
    <>
      <p className="meta-harness-demo-note">
        <code>heldout/{c.folder}.json</code>: {c.test.correct} of {c.test.n} test messages right ({pct(c.test.acc)}), against
        {' '}{c.search.correct} of {c.search.n} on the search set ({pct(c.search.accuracy)}). The proposer never reads this folder,
        so it cannot tune a harness to these messages. Only the search score chose which harness to keep.
      </p>
      <div className="meta-harness-demo-okstrip is-test" role="img" aria-label={`Test messages: ${c.test.correct} right`}>
        {c.test.ok.map((v, i) => <i key={i} className={v ? 'is-right' : 'is-wrong'} />)}
      </div>
    </>
  )
}

function RejView({ r, onPick }: { r: Rejected; onPick: (f: string, t?: Tab) => void }) {
  return (
    <>
      <div className="meta-harness-demo-candhead">
        <h4>{r.folder}/</h4>
        <span>iteration {r.iter} · rejected before evaluation
          {r.parent && <> · parent <button type="button" className="meta-harness-demo-linkbtn" onClick={() => onPick(r.parent!)}>{r.parent}</button></>}</span>
      </div>
      <p className="meta-harness-demo-note">
        The cheap interface check (Algorithm 1, line 11) runs the harness on a few examples with a fake model before any real
        evaluation. This one failed, so it cost almost nothing:
      </p>
      <Term title={`${r.folder}/error.txt`}><pre className="is-wrap">{r.error}</pre></Term>
      {r.proposal && <Term title={`${r.folder}/proposal.md`}><pre className="is-wrap">{r.proposal}</pre></Term>}
      {r.code && <Term title={`${r.folder}/harness.py`}><Code text={r.code} /></Term>}
    </>
  )
}
