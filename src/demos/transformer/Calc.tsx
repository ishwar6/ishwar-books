/* The attention calculator: Equation 1 of the paper, softmax(Q K^T / sqrt(d_k)) V, on 3 or 4 tokens with d_k = 2.
   The reader edits the numbers or drags the arrows, and every table updates. The first load is the worked example
   of Part 3 (queries (1,0), (0,1), (1,1); keys (2,0), (0,2), (1,1); values (10,0), (0,10), (5,5)). */
import { useMemo, useRef, useState, type PointerEvent as RPointerEvent } from 'react'
import { Link } from 'react-router-dom'
import { fix } from './txMath'
import { Seg } from './views'

type Mat = number[][]
const START: { Q: Mat; K: Mat; V: Mat } = {
  Q: [[1, 0], [0, 1], [1, 1], [0.5, -1]],
  K: [[2, 0], [0, 2], [1, 1], [-1, 0.5]],
  V: [[10, 0], [0, 10], [5, 5], [0, 0]],
}
const copy = (m: Mat) => m.map((r) => [...r])
const P3 = '/papers/transformer/part-3-attention'
const NAMES = ['t1', 't2', 't3', 't4']

function NumIn({ v, on, label }: { v: number; on: (x: number) => void; label: string }) {
  const [text, setText] = useState<string | null>(null)
  return (
    <input
      className="transformer-demo-num"
      type="number"
      inputMode="decimal"
      step={0.1}
      aria-label={label}
      value={text ?? String(Math.round(v * 100) / 100)}
      onChange={(e) => {
        setText(e.target.value)
        const x = parseFloat(e.target.value)
        if (Number.isFinite(x)) on(Math.max(-99, Math.min(99, x)))
      }}
      onBlur={() => setText(null)}
    />
  )
}

function MatIn({ name, m, n, set, color }: { name: string; m: Mat; n: number; set: (i: number, j: number, x: number) => void; color: string }) {
  return (
    <div className="transformer-demo-matin">
      <div className="transformer-demo-matname"><i className={color} />{name}</div>
      {m.slice(0, n).map((row, i) => (
        <div key={i} className="transformer-demo-matrow">
          <span>{NAMES[i]}</span>
          {row.map((v, j) => <NumIn key={j} v={v} on={(x) => set(i, j, x)} label={`${name}, ${NAMES[i]}, number ${j + 1}`} />)}
        </div>
      ))}
    </div>
  )
}

function Table({ title, rows, cols, data, fmt, hl, note, cellClass }: {
  title: React.ReactNode; rows: string[]; cols: string[]; data: (number | null)[][]; fmt: (v: number) => string; hl?: number
  note?: React.ReactNode; cellClass?: (i: number, j: number) => string
}) {
  return (
    <figure className="transformer-demo-calctab">
      <figcaption>{title}</figcaption>
      <table>
        <thead><tr><th />{cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
        <tbody>
          {data.map((r, i) => (
            <tr key={i} className={hl === i ? 'is-hl' : ''}>
              <th>{rows[i]}</th>
              {r.map((v, j) => <td key={j} className={cellClass?.(i, j) ?? ''}>{v === null ? '−∞' : fmt(v)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
      {note && <p className="transformer-demo-note">{note}</p>}
    </figure>
  )
}

/** The 2D plane: the selected query and all keys as arrows; drag an arrow's tip to change it. */
function Plane({ Q, K, n, row, setQ, setK }: { Q: Mat; K: Mat; n: number; row: number; setQ: (i: number, v: number[]) => void; setK: (i: number, v: number[]) => void }) {
  const S = 260, R = 3.2
  const ref = useRef<SVGSVGElement>(null)
  const drag = useRef<{ kind: 'q' | 'k'; i: number } | null>(null)
  const px = (v: number) => S / 2 + (v / R) * (S / 2 - 12)
  const py = (v: number) => S / 2 - (v / R) * (S / 2 - 12)
  const toVal = (e: RPointerEvent) => {
    const r = ref.current!.getBoundingClientRect()
    const x = ((e.clientX - r.left) / r.width) * S, y = ((e.clientY - r.top) / r.height) * S
    const cl = (t: number) => Math.max(-R, Math.min(R, Math.round(t * 10) / 10))
    return [cl(((x - S / 2) / (S / 2 - 12)) * R), cl((-(y - S / 2) / (S / 2 - 12)) * R)]
  }
  const move = (e: RPointerEvent) => {
    const d = drag.current
    if (!d) return
    const v = toVal(e)
    if (d.kind === 'q') setQ(d.i, v)
    else setK(d.i, v)
  }
  const arrow = (v: number[], cls: string, kind: 'q' | 'k', i: number, label: string) => (
    <g key={`${kind}${i}`} className={cls}>
      <line x1={px(0)} y1={py(0)} x2={px(v[0])} y2={py(v[1])} />
      <circle cx={px(v[0])} cy={py(v[1])} r={9} className="transformer-demo-handle"
        onPointerDown={(e) => { (e.target as Element).setPointerCapture(e.pointerId); drag.current = { kind, i } }}>
        <title>{`${label}: (${v[0]}, ${v[1]}). Drag to change it.`}</title>
      </circle>
      <text x={px(v[0]) + 11} y={py(v[1]) - 9}>{label}</text>
    </g>
  )
  return (
    <figure className="transformer-demo-plane">
      <svg ref={ref} viewBox={`0 0 ${S} ${S}`} role="img" aria-label="The query and the keys as arrows in the plane"
        onPointerMove={move} onPointerUp={() => (drag.current = null)} onPointerCancel={() => (drag.current = null)}>
        {[-3, -2, -1, 1, 2, 3].map((t) => (
          <g key={t}>
            <line className="transformer-demo-gridline" x1={px(t)} x2={px(t)} y1={0} y2={S} />
            <line className="transformer-demo-gridline" y1={py(t)} y2={py(t)} x1={0} x2={S} />
          </g>
        ))}
        <line className="transformer-demo-axisline" x1={0} x2={S} y1={py(0)} y2={py(0)} />
        <line className="transformer-demo-axisline" y1={0} y2={S} x1={px(0)} x2={px(0)} />
        {K.slice(0, n).map((k, i) => arrow(k, 'transformer-demo-karrow', 'k', i, `k${i + 1}`))}
        {arrow(Q[row], 'transformer-demo-qarrow', 'q', row, `q${row + 1}`)}
      </svg>
      <figcaption>Drag the tips. The query q{row + 1} is the bright arrow; the keys are the others. A key that points the same way as the query gets a large dot product.</figcaption>
    </figure>
  )
}

export default function Calc() {
  const [n, setN] = useState(3)
  const [Q, setQm] = useState(() => copy(START.Q))
  const [K, setKm] = useState(() => copy(START.K))
  const [V, setVm] = useState(() => copy(START.V))
  const [scale, setScale] = useState(true)
  const [mask, setMask] = useState(false)
  const [row, setRow] = useState(0)
  const [ver, setVer] = useState(0)          // re-mounts the inputs on reset
  const dk = 2
  const r = row < n ? row : 0

  const calc = useMemo(() => {
    const q = Q.slice(0, n), k = K.slice(0, n), v = V.slice(0, n)
    const raw = q.map((a) => k.map((b) => a[0] * b[0] + a[1] * b[1]))
    const sc = raw.map((rr) => rr.map((s) => (scale ? s / Math.sqrt(dk) : s)))
    const msk: (number | null)[][] = sc.map((rr, i) => rr.map((s, j) => (mask && j > i ? null : s)))
    const w = msk.map((rr) => {
      const m = Math.max(...rr.filter((x): x is number => x !== null))
      const e = rr.map((x) => (x === null ? 0 : Math.exp(x - m)))
      const z = e.reduce((a, b) => a + b, 0)
      return e.map((x) => x / z)
    })
    const out = w.map((rr) => [0, 1].map((c) => rr.reduce((a, wj, j) => a + wj * v[j][c], 0)))
    return { raw, sc, msk, w, out }
  }, [Q, K, V, n, scale, mask])

  const upd = (set: React.Dispatch<React.SetStateAction<Mat>>) => (i: number, j: number, x: number) =>
    set((m) => { const c = copy(m); c[i][j] = x; return c })
  const names = NAMES.slice(0, n)
  const qn = names.map((s) => 'q' + s.slice(1)), kn = names.map((s) => 'k' + s.slice(1))
  const msk = calc.msk[r]
  const eRow = msk.map((x) => (x === null ? 0 : Math.exp(x)))
  const zRow = eRow.reduce((a, b) => a + b, 0)
  const isStart = n === 3 && scale && !mask && JSON.stringify([Q, K, V].map((m) => m.slice(0, 3))) === JSON.stringify([START.Q, START.K, START.V].map((m) => m.slice(0, 3)))

  return (
    <div className="transformer-demo-calc" key={ver}>
      <div className="transformer-demo-controls">
        <Seg label="Tokens" value={n} onChange={(v) => { setN(v); if (row >= v) setRow(0) }} options={[{ v: 3, label: '3 tokens' }, { v: 4, label: '4 tokens' }]} />
        <Seg label="Scale" value={scale ? 1 : 0} onChange={(v) => setScale(!!v)} options={[{ v: 1, label: '÷ √d_k on' }, { v: 0, label: 'off' }]} />
        <Seg label="Mask" value={mask ? 1 : 0} onChange={(v) => setMask(!!v)} options={[{ v: 0, label: 'no mask' }, { v: 1, label: 'decoder mask' }]} />
        <button type="button" className="transformer-demo-btn" onClick={() => { setQm(copy(START.Q)); setKm(copy(START.K)); setVm(copy(START.V)); setN(3); setScale(true); setMask(false); setRow(0); setVer((x) => x + 1) }}>
          Reset to Part 3's example
        </button>
      </div>
      <div className="transformer-demo-calcin">
        <MatIn name="Q (queries)" m={Q} n={n} set={upd(setQm)} color="is-q" />
        <MatIn name="K (keys)" m={K} n={n} set={upd(setKm)} color="is-k" />
        <MatIn name="V (values)" m={V} n={n} set={upd(setVm)} color="is-v" />
        <Plane Q={Q} K={K} n={n} row={r} setQ={(i, v) => setQm((m) => { const c = copy(m); c[i] = v; return c })} setK={(i, v) => setKm((m) => { const c = copy(m); c[i] = v; return c })} />
      </div>
      <div className="transformer-demo-controls">
        <span className="transformer-demo-label">Follow the query of</span>
        <Seg label="Query row" value={r} onChange={setRow} options={names.map((s, i) => ({ v: i, label: s }))} />
      </div>
      <div className="transformer-demo-calcout">
        <Table title={<>1. Scores <b>Q Kᵀ</b></>} rows={qn} cols={kn} data={calc.raw} fmt={(v) => fix(v, 2)} hl={r}
          note="Row i, column j: the dot product of query i with key j." />
        <Table title={scale ? <>2. Divide by <b>√d_k = √2 = 1.414</b></> : <>2. No scale (switched off)</>} rows={qn} cols={kn} data={calc.sc} fmt={(v) => fix(v, 3)} hl={r} />
        <Table title={mask ? <>3. Mask: future keys get <b>−∞</b></> : <>3. No mask (encoder style)</>} rows={qn} cols={kn} data={calc.msk} fmt={(v) => fix(v, 3)} hl={r}
          cellClass={(i, j) => (mask && j > i ? 'is-masked' : '')} />
        <Table title={<>4. <b>softmax</b>, row by row</>} rows={qn} cols={kn} data={calc.w} fmt={(v) => fix(v, 3)} hl={r}
          note="Each row adds up to 1. These are the attention weights." />
        <Table title={<>5. Output <b>weights × V</b></>} rows={names.map((s) => 'out ' + s.slice(1))} cols={['1', '2']} data={calc.out} fmt={(v) => fix(v, 2)} hl={r} />
      </div>
      <div className="transformer-demo-calcrow">
        <p className="transformer-demo-label">The query of {names[r]} = ({Q[r][0]}, {Q[r][1]}), step by step</p>
        <ul className="transformer-demo-mono transformer-demo-steps">
          {K.slice(0, n).map((k, j) => (
            <li key={j}>
              q{r + 1} · k{j + 1} = {Q[r][0]} × {k[0]} + {Q[r][1]} × {k[1]} = {fix(calc.raw[r][j], 2)}
              {scale && <> ; ÷ 1.414 = {fix(calc.sc[r][j], 3)}</>}
              {msk[j] === null ? <> ; masked: −∞, e^−∞ = 0</> : <> ; e^{fix(msk[j], 3)} = {fix(eRow[j], 3)}</>}
            </li>
          ))}
          <li>sum = {fix(zRow, 3)} ; weights = {eRow.map((e) => fix(e / zRow, 3)).join(', ')}</li>
          <li>output = {calc.w[r].map((w, j) => `${fix(w, 3)} × (${V[j][0]}, ${V[j][1]})`).join(' + ')} = ({fix(calc.out[r][0], 2)}, {fix(calc.out[r][1], 2)})</li>
        </ul>
        <p className="transformer-demo-note">
          {isStart
            ? <>These are the numbers of <Link to={`${P3}#equation-1-step-by-step`}>Part 3's worked example</Link>: weights 0.576, 0.140 and 0.284, output (7.18, 2.82).</>
            : <>Part 3's example starts from queries (1, 0), (0, 1), (1, 1), keys (2, 0), (0, 2), (1, 1) and values (10, 0), (0, 10), (5, 5). The reset button brings it back.</>}
          {' '}Switch the scale off: every score grows by a factor of 1.414 and the weights get sharper. A head of the paper has d_k = 64, so there the factor would be 8, and the softmax would put most of the weight on one key; this is <Link to={`${P3}#why-divide-by-the-square-root-of-d_k`}>why the paper divides by √d_k</Link>.
          {' '}Switch on the mask and look at row 1: it can only see itself, so its weight on t1 is 1 (<Link to={`${P3}#the-mask-no-looking-ahead`}>the mask</Link>).
        </p>
      </div>
    </div>
  )
}
