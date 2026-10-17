/* Part 2 of the demo: the released translator Helsinki-NLP/opus-mt-en-de, run in Python and saved
   (code/papers/transformer/demo/marian_precompute.py). The reader picks a sentence, a kind of attention, a layer and a
   head, and hovers a token to see where it looks. */
import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { loadMarian, loadAttention, type Marian as M, type Attn } from './data'
import { AttnGrid, Seg, TokenWeights, TopBars, tokText } from './views'
import { P1, P2, P3, P6, DEMO_CODE } from './links'
import { fix } from './txMath'

type Kind = 'enc' | 'cross' | 'dec'

export default function Marian() {
  const [m, setM] = useState<M | null>(null)
  const [slug, setSlug] = useState('cat')
  const [loaded, setLoaded] = useState<{ slug: string; att: Attn } | null>(null)
  const [kind, setKind] = useState<Kind>('cross')
  const [layer, setLayer] = useState(2)
  const [head, setHead] = useState(-1)
  const [row, setRow] = useState(10)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => { loadMarian().then(setM).catch((e) => setError(String(e))) }, [])
  const s = m?.sentences.find((x) => x.slug === slug)
  useEffect(() => {
    if (!s) return
    let stop = false
    loadAttention(s).then((a) => { if (!stop) setLoaded({ slug: s.slug, att: a }) }).catch((e) => setError(String(e)))
    return () => { stop = true }
  }, [s])

  const att = loaded && s && loaded.slug === s.slug ? loaded.att : null
  const shape = s ? s[kind] : null
  const rowsT = s ? (kind === 'enc' ? s.src : s.tgt) : []
  const colsT = s ? (kind === 'cross' ? s.src : kind === 'enc' ? s.src : s.tgt) : []
  const r = Math.min(row, Math.max(0, rowsT.length - 1))
  const w = useMemo(() => {
    if (!att || !shape) return null
    const [, H, R, C] = shape
    const a = att[kind]
    const size = R * C
    if (head >= 0) return a.slice((layer * H + head) * size, (layer * H + head + 1) * size)
    const out = new Float32Array(size)
    for (let h = 0; h < H; h++) for (let i = 0; i < size; i++) out[i] += a[(layer * H + h) * size + i] / H
    return out
  }, [att, shape, kind, layer, head])

  useEffect(() => {
    if (!w || !s) return
    const C0 = (kind === 'dec' ? s.tgt : s.src).length
    window.__txDemo = { ...(window.__txDemo ?? {}), marian: { slug: s.slug, kind, layer, head, row: r, w: Array.from(w.slice(r * C0, (r + 1) * C0)) } }
  }, [w, s, kind, layer, head, r])

  if (error) return <div className="transformer-demo-error" role="alert">{error}</div>
  if (!m || !s) return <div className="transformer-demo-busy">Loading the saved runs…</div>
  const H = m.heads, L = m.layers
  const C = colsT.length
  const heads = att && shape ? Array.from({ length: H }, (_, h) => {
    const [, , R, CC] = shape
    const size = R * CC
    return att[kind].slice((layer * H + h) * size + r * CC, (layer * H + h) * size + (r + 1) * CC)
  }) : []
  const rowLabel = (i: number) => (kind === 'enc' ? tokText(s.src[i]) : tokText(s.tgt[i]))
  const wrote = kind !== 'enc' ? s.out[r] : null

  return (
    <div className="transformer-demo-marian">
      <p>
        This part shows a full-size translator: Helsinki-NLP/opus-mt-en-de, the released English to German model that the
        series uses since <Link to={`${P1}#a-released-transformer-translating-one-sentence`}>Part 1</Link>. It has the paper's base shapes: {L} encoder and {L} decoder
        layers, d_model = {m.d_model}, {H} heads, {(m.params / 1e6).toFixed(1)} million weights. <b>These numbers are precomputed.</b>{' '}
        The weights are about 300 MB as 32-bit numbers, too heavy to download into a web page. So we ran the model in Python
        on {m.sentences.length} sentences and saved its translation (greedy, one token per step) and every attention weight:
        each layer and head of the encoder self-attention, the decoder self-attention and the cross-attention. Each weight
        is stored in 8 bits with one scale per row, within {fix(m.max_abs_diff, 4)} of the model's own value
        (<a href={DEMO_CODE} target="_blank" rel="noreferrer">the script</a>).
      </p>
      <div className="transformer-demo-bar">
        <label className="transformer-demo-select">
          <span>Sentence</span>
          <select value={slug} onChange={(e) => { setSlug(e.target.value); setRow(0) }}>
            {m.sentences.map((x) => <option key={x.slug} value={x.slug}>{x.en.length > 70 ? x.en.slice(0, 68) + '…' : x.en}</option>)}
          </select>
        </label>
        <div className="transformer-demo-pair">
          <p><span className="transformer-demo-label">English</span> {s.en}</p>
          <p><span className="transformer-demo-label">German (the model)</span> {s.de}</p>
          <p className="transformer-demo-note">{s.note}</p>
        </div>
      </div>
      <div className="transformer-demo-controls">
        <Seg label="Kind of attention" value={kind} onChange={(k) => { setKind(k); setRow(0) }} options={[
          { v: 'enc', label: 'encoder self-attention' }, { v: 'cross', label: 'cross-attention' }, { v: 'dec', label: 'decoder self-attention' }]} />
      </div>
      <div className="transformer-demo-controls">
        <span className="transformer-demo-label">layer</span>
        <Seg small label="Layer" value={layer} onChange={setLayer} options={Array.from({ length: L }, (_, l) => ({ v: l, label: String(l + 1) }))} />
        <span className="transformer-demo-label">head</span>
        <Seg small label="Head" value={head} onChange={setHead} options={[...Array.from({ length: H }, (_, h) => ({ v: h, label: String(h + 1) })), { v: -1, label: 'mean' }]} />
      </div>
      <p className="transformer-demo-note">
        {kind === 'enc' && <>Encoder self-attention: each English token (a row) looks at all English tokens. This is where “it” can gather information from “cat” or “street” before any German is written. Read <Link to={`${P3}#the-three-uses-of-attention`}>Part 3, the three uses</Link>.</>}
        {kind === 'cross' && <>Cross-attention: each German position (a row) looks at the English tokens. Read across a row to see which English words the model used for the next German word. Read <Link to={`${P3}#the-three-uses-of-attention`}>Part 3, the three uses</Link>.</>}
        {kind === 'dec' && <>Decoder self-attention: each German position looks at the German written before it. The mask hides the future, so the upper triangle is empty. Read <Link to={`${P3}#the-mask-no-looking-ahead`}>Part 3, the mask</Link>.</>}
        {kind !== 'enc' && <> A row is labelled with the token the decoder has just read; it chooses the next token. Row “&lt;pad&gt;” is the start signal of this model.</>}
      </p>

      {!att || !w ? <div className="transformer-demo-busy">Loading the attention weights…</div> : (
        <>
          <div className="transformer-demo-hover">
            <p className="transformer-demo-label">Hover or tap a {kind === 'enc' ? 'word' : 'German token'}:</p>
            <div className="transformer-demo-chips" role="radiogroup" aria-label="Query token">
              {rowsT.map((t, i) => (
                <button key={i} type="button" role="radio" aria-checked={i === r} className={`transformer-demo-chip${i === r ? ' is-sel' : ''}`}
                  onMouseEnter={() => setRow(i)} onFocus={() => setRow(i)} onClick={() => setRow(i)}>
                  <b>{tokText(t)}</b>
                </button>
              ))}
            </div>
            <p className="transformer-demo-label">
              Layer {layer + 1}, {head >= 0 ? `head ${head + 1}` : 'mean of the 8 heads'}: “{rowLabel(r)}”{wrote ? <> (it writes “{tokText(wrote)}” next)</> : null} looks at
            </p>
            <TokenWeights toks={colsT} w={w.slice(r * C, (r + 1) * C)} q={kind === 'cross' ? null : r} label="Where the selected token looks" />
          </div>
          <div className="transformer-demo-two is-wide">
            <AttnGrid rows={rowsT} cols={colsT} w={w} sel={r} onSel={setRow} numbers={C <= 16}
              masked={kind === 'dec' ? (i, j) => j > i : undefined} maxPx={C > 20 ? 22 : 30}
              label={`${kind} attention, layer ${layer + 1}`} rowTitle={kind === 'enc' ? 'English' : 'German'} colTitle={kind === 'dec' ? 'German →' : 'English →'} />
            <div>
              <p className="transformer-demo-label">The {H} heads of layer {layer + 1} for “{rowLabel(r)}”</p>
              <div className="transformer-demo-headlist">
                {heads.map((hw, h) => {
                  const best = Array.from(hw.keys()).sort((a, b) => hw[b] - hw[a]).slice(0, 3)
                  return (
                    <button key={h} type="button" className={`transformer-demo-headrow${head === h ? ' is-sel' : ''}`} onClick={() => setHead(h)}>
                      <span>head {h + 1}</span>
                      <span>{best.map((j) => `${tokText(colsT[j])} ${fix(hw[j], 2)}`).join(', ')}</span>
                    </button>
                  )
                })}
              </div>
              {kind !== 'enc' && (
                <>
                  <p className="transformer-demo-label">After reading “{rowLabel(r)}”, the model's top 5 for the next token</p>
                  <TopBars items={s.top5[r].map(([t, p]) => ({ label: tokText(t), p }))} chosen={0} />
                </>
              )}
              <p className="transformer-demo-note">
                Each head has its own job, and many look at the end token &lt;/s&gt; or the full stop when they have nothing to do.
                The paper's appendix shows the same kind of pictures for layer 5 (<Link to={`${P6}#inside-the-heads-the-appendix`}>Part 6</Link>); pick
                the “making” or the “Law” sentence and layer 5 to compare. <Link to={`${P2}#the-decoder-stack`}>Part 2</Link> shows where each attention sits in the stacks.
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
