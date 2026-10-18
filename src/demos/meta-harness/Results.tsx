/* Tab 3: the paper's own numbers, explorable. TerminalBench-2 (Table 7, Figure 1 right), the text-classification
   Pareto frontier (Figure 3, with Table 9 and Table 2), and math retrieval across held-out models (Table 6).
   Every number comes from public/demos/meta-harness/paper.json (checked against the PDF by export_paper.py). */
import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { loadPaper, type Paper } from './data'
import { Seg, Busy, Err, Src, useNarrow } from './ui'
import { P1, P3, P4, P5 } from './links'

export default function Results() {
  const [p, setP] = useState<Paper | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => { loadPaper().then(setP).catch((e) => setError(String(e))) }, [])
  if (error) return <Err msg={error} />
  if (!p) return <Busy text="Loading the paper's numbers…" />
  return (
    <div className="meta-harness-demo-results">
      <TB2 p={p} />
      <Pareto p={p} />
      <MathPanel p={p} />
    </div>
  )
}

/* ---------------- TerminalBench-2 ---------------- */

function TB2({ p }: { p: Paper }) {
  const [g, setG] = useState(1)
  const group = p.table7.groups[g]
  const rows = [...group.rows].sort((a, b) => b.pass - a.pass)
  const max = g === 0 ? 100 : 50
  const ticks = g === 0 ? [0, 25, 50, 75, 100] : [0, 10, 20, 30, 40, 50]
  return (
    <section className="meta-harness-demo-step" aria-labelledby="mh-tb2-h">
      <div className="meta-harness-demo-stephead">
        <span className="meta-harness-demo-stepn">A</span><h3 id="mh-tb2-h">TerminalBench-2: an agent in a terminal</h3>
        <span className="meta-harness-demo-read">Explained in <Link to={P4}>Part 4</Link> · the Haiku 4.5 chart is <Link to={P1}>Figure 1, right</Link></span>
      </div>
      <p>
        TerminalBench-2 has 89 hard tasks. An agent must finish each one alone, with shell commands in a sandbox. A task
        passes when its tests pass. The model stays the same in each chart; only the harness changes. Meta-Harness started
        from Terminus-KIRA. The main change the search found: a snapshot of the sandbox (files, languages, package managers) in the first prompt.
      </p>
      <div className="meta-harness-demo-controls">
        <Seg label="Model" value={g} onChange={setG} options={p.table7.groups.map((x, i) => ({ v: i, label: x.model }))} />
      </div>
      <figure className="meta-harness-demo-fig">
        <figcaption>Pass rate (%) on 89 tasks, {group.model}. Teal: found by the automatic search. Grey: written by people.</figcaption>
        <ol className="meta-harness-demo-hbars" aria-label={`TerminalBench-2 pass rates with ${group.model}`}>
          {rows.map((r) => (
            <li key={r.harness} className={r.auto ? 'is-auto' : ''}>
              <span className="meta-harness-demo-hname">{r.harness}{r.auto && <small> automatic</small>}</span>
              <span className="meta-harness-demo-htrack"><i style={{ width: `${(100 * r.pass) / max}%` }} /></span>
              <span className="meta-harness-demo-hval">{r.pass.toFixed(1)}</span>
            </li>
          ))}
        </ol>
        <div className="meta-harness-demo-hticks" aria-hidden="true">
          {ticks.map((t) => <span key={t} style={{ left: `${(100 * t) / max}%` }}>{t}</span>)}
        </div>
        <Src>{p.table7.source}. Other harnesses' numbers are from the official leaderboard, as the paper reports them.</Src>
      </figure>
      <p className="meta-harness-demo-note">
        {g === 1
          ? 'With Haiku 4.5, the searched harness is first: 37.6% against 35.5% for Goose, the best harness written by people.'
          : 'With Opus 4.6, it is second: 76.4% against 74.7% for Terminus-KIRA, its starting point. The paper could not reproduce ForgeCode\'s 81.8% from its public code.'}
        {' '}One caution: here the search and the final score use the same 89 tasks, because the benchmark is small. The paper
        checked the harness by hand and with regex audits for task-specific strings. <Link to={P5}>Part 5</Link> discusses this.
      </p>
    </section>
  )
}

/* ---------------- Figure 3: the Pareto frontier ---------------- */

type Pt = { key: string; name: string; x: number; y: number; kind: 'star' | 'zero' | 'few' | 'mce' | 'ace'; rows: [string, string][]; approx?: boolean }

function Pareto({ p }: { p: Paper }) {
  const pts: Pt[] = useMemo(() => {
    const stars: Pt[] = p.table9.rows.map((r) => ({
      key: r.variant, name: `Meta-Harness (${r.variant})`, x: r.ctx_kchar, y: r.avg, kind: 'star',
      rows: [['Average accuracy', `${r.avg.toFixed(1)}%`], ['USPTO', `${r.uspto.toFixed(1)}%`], ['Symptom2Disease', `${r.s2d.toFixed(1)}%`],
        ['LawBench', `${r.law.toFixed(1)}%`], ['Extra context', `${r.ctx_kchar.toFixed(1)}K characters (Table 9)`]],
    }))
    const base: Pt[] = p.figure3.baselines.map((b) => {
      const r = p.table2.rows.find((x) => x.harness === b.table2)!
      const kind: Pt['kind'] = b.harness === 'Zero-Shot' ? 'zero' : b.harness === 'MCE' ? 'mce' : b.harness === 'ACE' ? 'ace' : 'few'
      return {
        key: b.harness, name: b.harness, x: b.x_kchar_from_plot, y: r.avg, kind, approx: true,
        rows: [['Average accuracy', `${r.avg.toFixed(1)}%`], ['USPTO', `${r.uspto.toFixed(1)}%`], ['Symptom2Disease', `${r.s2d.toFixed(1)}%`],
          ['LawBench', `${r.law.toFixed(1)}%`], ['Extra context', `${r.ctx_ktok.toFixed(1)}K tokens (Table 2); drawn near ${b.x_kchar_from_plot}K characters, as in Figure 3`]],
      }
    })
    return [...base, ...stars]
  }, [p])
  const [sel, setSel] = useState('Label-Primed Query')
  const cur = pts.find((x) => x.key === sel) ?? pts[pts.length - 1]

  // a broken x axis like the paper's: 0 to 52K characters, then a gap, then 100K to 210K
  const narrow = useNarrow()
  const W = narrow ? 420 : 640, H = narrow ? 320 : 330, ml = 40, mr = 14, mt = 14, mb = 44
  const a1 = ml + (W - ml - mr) * 0.74, b0 = a1 + 24
  const x = (v: number) => (v <= 55 ? ml + (v / 52) * (a1 - ml) : b0 + ((v - 100) / 110) * (W - mr - b0))
  const y = (v: number) => mt + (1 - (v - 25) / (52 - 25)) * (H - mt - mb)
  const stars = pts.filter((q) => q.kind === 'star').sort((a, b) => a.x - b.x)
  const few = pts.filter((q) => q.kind === 'zero' || q.kind === 'few').sort((a, b) => a.x - b.x)
  const line = (q: Pt[]) => q.map((r, i) => `${i ? 'L' : 'M'}${x(r.x).toFixed(1)},${y(r.y).toFixed(1)}`).join('')

  return (
    <section className="meta-harness-demo-step" aria-labelledby="mh-pareto-h">
      <div className="meta-harness-demo-stephead">
        <span className="meta-harness-demo-stepn">B</span><h3 id="mh-pareto-h">Text classification: accuracy against context</h3>
        <span className="meta-harness-demo-read">Explained in <Link to={P3}>Part 3</Link> (Figure 3, Tables 2 and 9)</span>
      </div>
      <p>
        A harness can buy accuracy by putting more examples in the prompt. That costs context: extra text the model reads on
        every call. So each harness is a point: how much extra text (right) and how often it is right (up). A point is on the
        <b> Pareto frontier</b> when no other point is both cheaper and more accurate. Hover or tap a point.
      </p>
      <div className="meta-harness-demo-two">
        <figure className="meta-harness-demo-fig">
          <svg className="meta-harness-demo-svg" viewBox={`0 0 ${W} ${H}`} role="img"
            aria-label="Test accuracy against extra context in thousands of characters, for the searched harnesses and the baselines">
            {[25, 30, 35, 40, 45, 50].map((t) => (
              <g key={t}>
                <line className="meta-harness-demo-grid" x1={ml} x2={W - mr} y1={y(t)} y2={y(t)} />
                <text className="meta-harness-demo-axis" x={ml - 8} y={y(t) + 4} textAnchor="end">{t}</text>
              </g>
            ))}
            {(narrow ? [0, 20, 40, 115, 200] : [0, 10, 20, 30, 40, 50, 115, 200]).map((t) => (
              <text key={t} className="meta-harness-demo-axis" x={x(t)} y={H - mb + 18} textAnchor="middle">{t ? `${t}K` : '0'}</text>
            ))}
            <line className="meta-harness-demo-axisline" x1={ml} x2={a1} y1={H - mb} y2={H - mb} />
            <line className="meta-harness-demo-axisline" x1={b0} x2={W - mr} y1={H - mb} y2={H - mb} />
            <path className="meta-harness-demo-axisline" d={`M${a1 - 3},${H - mb + 5}l6,-10M${b0 - 3},${H - mb + 5}l6,-10`} />
            <text className="meta-harness-demo-axis" x={(ml + W - mr) / 2} y={H - 6} textAnchor="middle">Extra context (thousands of characters)</text>
            <text className="meta-harness-demo-axis" transform={`translate(12 ${(mt + H - mb) / 2}) rotate(-90)`} textAnchor="middle">Test accuracy (%)</text>
            <path className="meta-harness-demo-fewline" d={line(few)} />
            <path className="meta-harness-demo-frontier" d={line(stars)} />
            {pts.map((q) => (
              <g key={q.key} className={`meta-harness-demo-pt is-${q.kind}${q.key === cur.key ? ' is-sel' : ''}`} tabIndex={0} role="button"
                aria-label={`${q.name}: ${q.y.toFixed(1)}% accuracy`} aria-pressed={q.key === cur.key}
                onMouseEnter={() => setSel(q.key)} onFocus={() => setSel(q.key)} onClick={() => setSel(q.key)}>
                <circle className="meta-harness-demo-hit" cx={x(q.x)} cy={y(q.y)} r={13} />
                {q.kind === 'star' ? <path d={starPath(x(q.x), y(q.y), q.key === cur.key ? 11 : 8)} />
                  : q.kind === 'zero' ? <path d={`M${x(q.x) - 6},${y(q.y) - 6}l12,12M${x(q.x) + 6},${y(q.y) - 6}l-12,12`} />
                    : q.kind === 'mce' ? <path d={`M${x(q.x)},${y(q.y) - 8}l7,12h-14z`} />
                      : q.kind === 'ace' ? <path d={`M${x(q.x)},${y(q.y) - 8}l8,8l-8,8l-8,-8z`} />
                        : <circle cx={x(q.x)} cy={y(q.y)} r={6} />}
              </g>
            ))}
          </svg>
          <div className="meta-harness-demo-legend" aria-hidden="true">
            <span><i className="is-star" />Meta-Harness, Pareto variants (Table 9)</span>
            <span><i className="is-few" />Zero-shot and few-shot</span>
            <span><i className="is-mce" />MCE</span>
            <span><i className="is-ace" />ACE</span>
          </div>
          <Src>Figure 3 with numbers from Table 9 (stars, characters) and Table 2 (baselines, accuracy). Table 2 gives the
            baselines' context in tokens; Figure 3 draws them at about four characters per token, so their left-right place is
            read from the plot. The paper's few-shot 4 and 16 points and its faint non-Pareto candidates have no printed numbers, so they are left out.</Src>
        </figure>
        <div className="meta-harness-demo-card" aria-live="polite">
          <h4>{cur.name}</h4>
          <dl>{cur.rows.map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}</dl>
          {cur.key === 'Label-Primed Query' && <p>The paper's selected system: the most accurate point. It lists every valid label first, then one
            example per label, then pairs of near-identical examples with different labels.</p>}
          {cur.key === 'Draft Verification' && <p>The cheapest frontier point. Call 1 guesses a label from 5 similar examples. Call 2 shows
            examples that agree and disagree with that guess, and the model keeps or changes it.</p>}
          {cur.kind === 'ace' && <p>ACE was the best harness written by people. It uses about four times more context than the selected
            Meta-Harness (50.8K against 11.4K tokens) and is 7.7 points less accurate.</p>}
          {cur.kind === 'mce' && <p>MCE keeps a library of natural-language skills for building the context.</p>}
          {cur.kind === 'few' && <p>Few-shot puts N labelled examples in the prompt. More examples cost more context.</p>}
          {cur.kind === 'zero' && <p>Zero-shot: the question alone, with no examples. No extra context.</p>}
        </div>
      </div>
      <p className="meta-harness-demo-note">
        For every people-written harness except zero-shot (which adds no context), some star is both more accurate and cheaper. The cheapest star, Draft Verification, matches MCE (40.1% against 40.0%) with a small part of its context.
        The test sets were held out until the end; the search only saw search-set scores.
      </p>
    </section>
  )
}

function starPath(cx: number, cy: number, r: number) {
  let d = ''
  for (let i = 0; i < 10; i++) {
    const a = -Math.PI / 2 + (i * Math.PI) / 5
    const rr = i % 2 ? r * 0.45 : r
    d += `${i ? 'L' : 'M'}${(cx + rr * Math.cos(a)).toFixed(1)},${(cy + rr * Math.sin(a)).toFixed(1)}`
  }
  return d + 'Z'
}

/* ---------------- Table 6: math retrieval across models ---------------- */

function MathPanel({ p }: { p: Paper }) {
  const t = p.table6
  const base = t.rows[0]
  const methods = t.rows.slice(1)
  const [mi, setMi] = useState(methods.length - 1)
  const m = methods[mi]
  const lo = 20, hi = 52
  const pos = (v: number) => `${(100 * (v - lo)) / (hi - lo)}%`
  const rows = [...t.models.map((md, i) => ({ name: md.name, b: base.acc[i], v: m.acc[i], d: m.delta![i] })),
    { name: 'Average of the five', b: base.avg, v: m.avg, d: m.avg_delta! }]
  return (
    <section className="meta-harness-demo-step" aria-labelledby="mh-math-h">
      <div className="meta-harness-demo-stephead">
        <span className="meta-harness-demo-stepn">C</span><h3 id="mh-math-h">Math: one retrieval harness, five models</h3>
        <span className="meta-harness-demo-read">Explained in <Link to={P4}>Part 4</Link> (Table 6, Figure 8)</span>
      </div>
      <p>
        Here the harness decides which solved problems from a corpus of 535K go into the prompt of an olympiad problem. The
        search picked one harness with one model (GPT-OSS-20B) on 250 search problems. The test is 200 new IMO-level
        problems, with that model and four models the search never used. Pick a method and compare it with no retrieval.
      </p>
      <div className="meta-harness-demo-controls">
        <Seg label="Method" value={mi} onChange={setMi} options={methods.map((r, i) => ({ v: i, label: r.method }))} />
      </div>
      <figure className="meta-harness-demo-fig">
        <figcaption>Accuracy (%, pass@1 averaged over three samples). Hollow dot: no retriever. Filled dot: {m.method}.</figcaption>
        <ul className="meta-harness-demo-dumb" aria-label={`${m.method} against no retriever, per model`}>
          {rows.map((r) => (
            <li key={r.name} className={r.name.startsWith('Average') ? 'is-avg' : ''}>
              <span className="meta-harness-demo-dname">{r.name}</span>
              <span className="meta-harness-demo-dtrack">
                <i className={`meta-harness-demo-dseg ${r.d >= 0 ? 'is-up' : 'is-down'}`}
                  style={{ left: pos(Math.min(r.b, r.v)), width: `${(100 * Math.abs(r.v - r.b)) / (hi - lo)}%` }} />
                <b className="meta-harness-demo-dbase" style={{ left: pos(r.b) }} title={`No retriever: ${r.b.toFixed(1)}%`} />
                <b className={`meta-harness-demo-dval ${r.d >= 0 ? 'is-up' : 'is-down'}`} style={{ left: pos(r.v) }} title={`${m.method}: ${r.v.toFixed(1)}%`} />
              </span>
              <span className={`meta-harness-demo-ddelta ${r.d >= 0 ? 'is-up' : 'is-down'}`}>{r.v.toFixed(1)} <small>({r.d >= 0 ? '+' : '−'}{Math.abs(r.d).toFixed(1)})</small></span>
            </li>
          ))}
        </ul>
        <div className="meta-harness-demo-dticks" aria-hidden="true">
          {[20, 30, 40, 50].map((v) => <span key={v} style={{ left: pos(v) }}>{v}</span>)}
        </div>
      </figure>
      <details className="meta-harness-demo-details">
        <summary>All methods: change against no retriever (points)</summary>
        <div className="meta-harness-demo-tablewrap">
          <table className="meta-harness-demo-table">
            <thead><tr><th>Method</th>{t.models.map((md) => <th key={md.id}>{md.short}</th>)}<th>Avg.</th></tr></thead>
            <tbody>
              <tr><th>No Retriever</th>{base.acc.map((v, i) => <td key={i}>{v.toFixed(1)}</td>)}<td>{base.avg.toFixed(1)}</td></tr>
              {methods.map((r, k) => (
                <tr key={r.method} className={k === mi ? 'is-sel' : ''}>
                  <th>{r.method}</th>
                  {r.delta!.map((d, i) => <td key={i} className={d >= 0 ? 'is-up' : 'is-down'}>{d >= 0 ? '+' : '−'}{Math.abs(d).toFixed(1)}</td>)}
                  <td className={r.avg_delta! >= 0 ? 'is-up' : 'is-down'}>{r.avg_delta! >= 0 ? '+' : '−'}{Math.abs(r.avg_delta!).toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <Src>{t.source}. The first row is accuracy; the other rows are the change from it, as in the paper.</Src>
      <p className="meta-harness-demo-note">
        {mi === methods.length - 1
          ? 'The searched harness helps all five models: +4.7 points on average, and no model gets worse. Dense retrieval with k=5 helps more on two models but hurts two others.'
          : m.delta!.some((d) => d < 0)
            ? `${m.method} hurts ${m.delta!.filter((d) => d < 0).length} of the five models. Retrieval is not free: the wrong examples can distract the model.`
            : `${m.method} helps all five models, by ${m.avg_delta!.toFixed(1)} points on average; the searched harness adds ${(methods[methods.length - 1].avg_delta! - m.avg_delta!).toFixed(1)} more on the same BM25 index.`}
      </p>
    </section>
  )
}
