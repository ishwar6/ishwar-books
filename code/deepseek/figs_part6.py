"""Figures for Part 6 (DeepSeek-V4 and V4.1). Writes results/figs_part6.json (key -> SVG string).
Numbers come from results/part6_kv_per_token.json and results/part6_mhc_toy.json (our scripts) or from the
papers and configs, as noted next to each figure."""
import json, math, os
from figlib import svg, text, box, arrow, bar_path, esc
from common import RESULTS, save

KV = json.load(open(os.path.join(RESULTS, 'part6_kv_per_token.json')))
MHC = json.load(open(os.path.join(RESULTS, 'part6_mhc_toy.json')))
F = {}
W = 680


def label_box(x, y, w, h, cls, title, sub=None, tcls='t-note'):
    b = [box(x, y, w, h, cls, rx=8)]
    if sub:
        b += [text(x + w / 2, y + h / 2 - 3, title, tcls, 'middle'), text(x + w / 2, y + h / 2 + 14, sub, 't-muted', 'middle')]
    else:
        b.append(text(x + w / 2, y + h / 2 + 5, title, tcls, 'middle'))
    return b


def line(x1, y1, x2, y2, cls='edge'):
    return f'<line class="{cls}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>'


# 1. the 2026 releases (dossier 01 dates) -----------------------------------------------------------------------
ev = [('Apr 24', 'V4 preview', 'V4-Pro 1.6T / 49B, V4-Flash 284B / 13B', 'box-1'),
      ('Jun 27', 'DSpark checkpoints', 'V4 + speculative-decoding drafter', 'box'),
      ('Jul 31', 'V4-Flash-0731', 'official V4-Flash, with DSpark', 'box-1'),
      ('Aug 13', 'V4-Pro-0813', 'official V4-Pro, with DSpark', 'box-1'),
      ('Aug 21', 'V4-Flash-Vision-Exp', 'first V4 that reads images', 'box'),
      ('Sep 10', 'V4.1-Flash', '552B + 196B Engram, 8B / 16B active', 'box-2'),
      ('Oct 17', 'V4.1-Flash-1017', 'minor update', 'box-2')]
b = [text(20, 24, 'DeepSeek releases in 2026 (all open weights, MIT license)', 't-title')]
x0 = 112
b.append(line(x0, 44, x0, 44 + 46 * len(ev) - 18))
for i, (d, name, sub, cls) in enumerate(ev):
    y = 44 + i * 46
    b.append(f'<circle class="dot" cx="{x0}" cy="{y + 14}" r="5"/>')
    b.append(text(x0 - 14, y + 19, d, 't-tick', 'end'))
    b.append(box(x0 + 18, y, 520, 30, cls, rx=6))
    b.append(text(x0 + 30, y + 20, name, 't-note'))
    b.append(text(x0 + 220, y + 20, sub, 't-muted'))
F['releases'] = svg(W, 44 + 46 * len(ev) + 4, 'Timeline of the 2026 DeepSeek releases from the V4 preview in April to V4.1-Flash-1017 in October.', b)

# 2. headline: global KV cache per token across generations (our script; matches V4.1 paper Figure 1b) ------
rows = KV['rows']
names = {'DeepSeek LLM 67B': ('DeepSeek LLM 67B', 'GQA, BF16'), 'DeepSeek-V3': ('DeepSeek-V3', 'MLA, BF16'),
         'DeepSeek-V3.2': ('DeepSeek-V3.2', 'MLA + DSA, FP8'), 'DeepSeek-V4-Flash': ('DeepSeek-V4-Flash', 'CSA + HCA'),
         'DeepSeek-V4.1-Flash': ('DeepSeek-V4.1-Flash', 'CSA2 shared, FP4')}
L, T, bw = 190, 50, 330
lo, hi = math.log10(500), math.log10(500000)
X = lambda v: L + (math.log10(v) - lo) / (hi - lo) * bw
b = [text(20, 24, 'Global KV cache per token (bytes, log scale)', 't-title')]
for t in (1000, 10000, 100000):
    b += [line(X(t), T - 8, X(t), T + 5 * 50 - 14, 'grid'), text(X(t), T + 5 * 50 + 4, f'{t:,}', 't-tick', 'middle')]
for i, r in enumerate(rows):
    y = T + i * 50
    nm, sub = names[r['name']]
    b.append(text(L - 10, y + 12, nm, 't-note', 'end'))
    b.append(text(L - 10, y + 28, f'{r["date"]}  {sub}', 't-muted', 'end'))
    cls = 's2' if 'V4' in r['name'] else 's1'
    b.append(f'<g class="mark"><title>{esc(nm)}: {r["bytes"]:,.0f} bytes per token</title>'
             f'<path class="{cls} bar-mark" d="{bar_path(X(500), y, X(r["bytes"]) - X(500), 26, horizontal=True)}"/></g>')
    b.append(text(X(r['bytes']) + 8, y + 14, f'{r["bytes"]:,.0f} B', 't-val'))
    b.append(text(X(r['bytes']) + 8, y + 29, f'{r["bytes"] * 1048576 / 1e9:,.2f} GB at 1M tokens', 't-muted'))
F['kvgen'] = svg(W, T + 5 * 50 + 14, 'Global KV cache per token across five DeepSeek generations, from 389,120 bytes for the 67B model to 890 bytes for V4.1-Flash.', b)

# 3. what one query sees in a CSA layer and an HCA layer ------------------------------------------------------
b = [text(20, 24, 'One query at position 40. What it can read in each kind of layer', 't-title')]
N, cw, xs = 40, 13.5, 30
def tokens(y, label):
    out = [text(xs, y - 8, label, 't-note')]
    for j in range(N):
        out.append(f'<rect class="box" x="{xs + j * cw:.1f}" y="{y}" width="{cw - 2:.1f}" height="14" rx="2"/>')
    return out
# CSA: groups of 4 -> 10 entries; indexer picks 3 (the toy top-k); window = last 8
y = 56
b += tokens(y, 'CSA (ratio 4): every 4 tokens become one entry, the indexer picks the top-k entries')
pick = {1, 5, 7}
for g in range(N // 4):
    cls = 'box-1' if g in pick else 'box-ghost'
    b.append(f'<rect class="{cls}" x="{xs + g * 4 * cw:.1f}" y="{y + 24}" width="{4 * cw - 2:.1f}" height="18" rx="4"/>')
    if g in pick:
        b.append(text(xs + g * 4 * cw + 2 * cw, y + 37, 'top-k', 't-tick', 'middle'))
b.append(f'<rect class="box-2" x="{xs + (N - 8) * cw:.1f}" y="{y + 48}" width="{8 * cw - 2:.1f}" height="18" rx="4"/>')
b.append(text(xs + (N - 4) * cw, y + 61, 'window', 't-tick', 'middle'))
# HCA: one entry per 128 tokens -> drawn as one entry for the whole strip; window
y = 166
b += tokens(y, 'HCA (ratio 128): one entry per 128 tokens, all of them read, no indexer')
b.append(f'<rect class="box-1" x="{xs:.1f}" y="{y + 24}" width="{N * cw - 2:.1f}" height="18" rx="4"/>')
b.append(text(xs + N * cw / 2, y + 37, 'one heavily compressed entry (stands for 128 tokens)', 't-tick', 'middle'))
b.append(f'<rect class="box-2" x="{xs + (N - 8) * cw:.1f}" y="{y + 48}" width="{8 * cw - 2:.1f}" height="18" rx="4"/>')
b.append(text(xs + (N - 4) * cw, y + 61, 'window', 't-tick', 'middle'))
b.append(text(20, 268, 'Drawn small: the real window is 128 tokens, V4-Flash keeps top-512 CSA entries, V4-Pro top-1024.', 't-muted'))
F['csa_hca'] = svg(W, 280, 'In a CSA layer each group of 4 tokens becomes one entry and the indexer selects a few entries; in an HCA layer 128 tokens become one entry and all entries are read; both layers also read a sliding window of recent tokens.', b)

# 4. CSA's overlapping compression (V4 paper Eq. 11 to 12), m = 4 ---------------------------------------------
b = [text(20, 24, 'CSA compression with m = 4: each entry mixes 8 tokens, but there is one entry per 4 tokens', 't-title')]
m, cw, xs, y = 4, 30, 40, 50
for j in range(16):
    b.append(box(xs + j * cw, y, cw - 4, 26, 'box', rx=4))
    b.append(text(xs + j * cw + (cw - 4) / 2, y + 18, str(j), 't-tick', 'middle'))
for i, cls in zip((1, 2, 3), ('box-1', 'box-2', 'box-1')):
    yy = y + 40 + (i - 1) * 40
    xb, xa = xs + m * (i - 1) * cw, xs + m * i * cw
    b.append(f'<rect class="box-ghost" x="{xb:.1f}" y="{yy}" width="{m * cw - 4:.1f}" height="24" rx="4"/>')
    b.append(f'<rect class="{cls}" x="{xa:.1f}" y="{yy}" width="{m * cw - 4:.1f}" height="24" rx="4"/>')
    b.append(text(xb + m * cw / 2, yy + 17, 'previous 4 (series b)', 't-tick', 'middle'))
    b.append(text(xa + m * cw / 2, yy + 17, 'own 4 (series a)', 't-tick', 'middle'))
    b.append(text(xa + m * cw + 6, yy + 17, f'entry {i}', 't-note'))
b.append(text(20, 222, 'One softmax over the 8 scores decides how much each token adds. Neighbouring entries share 4 tokens,', 't-muted'))
b.append(text(20, 240, 'so a boundary between groups does not cut a phrase in half. HCA uses one series and no overlap.', 't-muted'))
F['csa_overlap'] = svg(W, 252, 'CSA builds entry i from its own 4 tokens and the previous 4 tokens, weighted by one softmax over 8 scores, so entries overlap but there is still one entry per 4 tokens.', b)

# 5. the layer plans, from compress_ratios in the configs -----------------------------------------------------
def strip(y, label, kinds, sub):
    out = [text(20, y - 8, label, 't-note'), text(W - 20, y - 8, sub, 't-muted', 'end')]
    n = len(kinds)
    cw = (W - 40) / n
    for j, k in enumerate(kinds):
        cls = {'SWA': 'box', 'CSA': 'box-1', 'HCA': 'box-2', 'F': 'box-on', 'R': 'box-1', 'X': 'box-3', 'U': 'box-ghost'}[k]
        out.append(f'<g class="mark"><title>layer {j}: {k}</title><rect class="{cls}" x="{20 + j * cw:.1f}" y="{y}" width="{max(cw - 1.5, 1):.1f}" height="22" rx="2"/></g>')
    return out
flash = ['SWA', 'SWA'] + ['CSA', 'HCA'] * 20 + ['CSA']
pro = ['HCA', 'HCA'] + ['CSA', 'HCA'] * 29 + ['CSA']
enc = ['SWA', 'SWA'] + (['F'] + ['U'] * 5) * 3
dec = ['F', 'U', 'U', 'U'] + (['X', 'U', 'U', 'U']) * 4
b = [text(20, 24, 'Which attention each layer uses (read from compress_ratios in config.json)', 't-title')]
b += strip(62, 'V4-Flash, 43 layers', flash, '2 window-only, 21 CSA, 20 HCA')
b += strip(122, 'V4-Pro, 61 layers', pro, '30 CSA, 31 HCA')
b += strip(182, 'V4.1-Flash, 40 layers: 20 encoder + 20 decoder', enc + dec, 'KV stored in 4 layers only')
b.append(line(20 + 20 * (W - 40) / 40, 176, 20 + 20 * (W - 40) / 40, 210, 'drop'))
keys = [('box', 'window only'), ('box-1', 'CSA'), ('box-2', 'HCA'), ('box-on', 'CSA2 Full'), ('box-3', 'CSA2 Reindex'), ('box-ghost', 'CSA2 Reuse')]
for i, (c, k) in enumerate(keys):
    x = 20 + (i % 3) * 210
    yy = 228 + (i // 3) * 24
    b += [box(x, yy, 16, 14, c, rx=3), text(x + 24, yy + 12, k, 't-tick')]
F['layers'] = svg(W, 284, 'Layer plans of V4-Flash, V4-Pro and V4.1-Flash. V4 alternates CSA and HCA layers; V4.1 uses CSA2 layers in Full, Reindex or Reuse mode, and only four Full layers store KV.', b)

# 6. where the bytes come from (our script) ---------------------------------------------------------------------
fl, pr = KV['flash'], KV['pro']
parts = [('V4-Flash', [('21 CSA layers', fl['csa_bytes'], 's1'), ('20 HCA layers', fl['hca_bytes'], 's2')]),
         ('V4-Pro', [('30 CSA layers', pr['csa_bytes'], 's1'), ('31 HCA layers', pr['hca_bytes'], 's2')]),
         ('V4.1-Flash', [('3 encoder Full layers', 534, 's1'), ('1 decoder Full layer', 356, 's3')])]
L, bw = 120, 400
mx = 5100
b = [text(20, 24, 'Bytes per token, split by the layers that store them', 't-title')]
for i, (nm, segs) in enumerate(parts):
    y = 50 + i * 56
    b.append(text(L - 10, y + 18, nm, 't-note', 'end'))
    x = L
    for lab, v, c in segs:
        w = v / mx * bw
        b.append(f'<g class="mark"><title>{esc(nm)}, {esc(lab)}: {v:,.1f} B</title><rect class="{c} bar-mark" x="{x:.1f}" y="{y}" width="{max(w, 2):.1f}" height="26" rx="3"/></g>')
        x += w
    tot = sum(v for _, v, _ in segs)
    b.append(text(x + 8, y + 18, f'{tot:,.0f} B', 't-val'))
    b.append(text(L, y + 44, '  +  '.join(f'{lab}: {v:,.0f} B' for lab, v, _ in segs), 't-muted'))
F['bytes_split'] = svg(W, 216, 'In V4 almost all bytes come from the CSA layers; HCA layers add under 150 bytes per token. V4.1-Flash stores KV in only four layers.', b)

# 7. residual connection vs hyper-connections ------------------------------------------------------------------
b = [text(W / 4, 24, 'Residual connection', 't-title', 'middle'), text(3 * W / 4, 24, 'Hyper-connections, n = 4 streams', 't-title', 'middle')]
# left
b += label_box(40, 200, 120, 36, 'box', 'x (one stream)')
b += label_box(180, 120, 120, 36, 'box-1', 'layer F')
b += [arrow(100, 200, 100, 70), arrow(160, 218, 240, 156), arrow(240, 120, 110, 70)]
b += [f'<circle class="dot" cx="100" cy="60" r="10"/>', text(100, 64, '+', 't-strong', 'middle')]
b.append(text(40, 270, 'x + F(x): the input passes through untouched', 't-muted'))
# right
for k in range(4):
    b.append(box(372 + k * 5, 226 - k * 5, 92, 28, 'box', rx=6))
b.append(text(425, 272, 'X: 4 copies, each d wide', 't-muted', 'middle'))
b += label_box(372, 130, 120, 36, 'box-2', 'B mixes streams')
b += label_box(520, 200, 130, 36, 'box', 'A: 4 streams to 1')
b += label_box(520, 120, 130, 36, 'box-1', 'layer F')
b += label_box(520, 50, 130, 36, 'box', 'C: 1 back to 4')
b += [arrow(432, 208, 432, 166), arrow(474, 225, 520, 220), arrow(585, 200, 585, 156), arrow(585, 120, 585, 86),
      arrow(432, 130, 432, 70), arrow(520, 66, 446, 60)]
b += [f'<circle class="dot" cx="432" cy="60" r="10"/>', text(432, 64, '+', 't-strong', 'middle')]
b.append(text(372, 294, 'new X = B X + C F(A X)', 't-note'))
F['residual_hc'] = svg(W, 304, 'A residual connection adds the layer output to an untouched input. Hyper-connections keep four copies of the stream, mix them with a 4 x 4 matrix B, read one input with A and write the output back with C.', b)

# 8. the stability toy (our script) ------------------------------------------------------------------------------
depth = MHC['depth']
xs_ = list(range(1, depth + 1))
Lp, Tp, pw, ph = 64, 40, 420, 210
X = lambda l: Lp + (l - 1) / (depth - 1) * pw
Y = lambda v: Tp + ph - (math.log10(v) - 0) / 3 * ph
b = [text(20, 24, f'Largest gain of the product of {depth} mixing matrices (4 streams, 5 random seeds)', 't-title')]
for e in (0, 1, 2, 3):
    b += [line(Lp, Y(10 ** e), Lp + pw, Y(10 ** e), 'grid'), text(Lp - 8, Y(10 ** e) + 4, f'{10 ** e:,}', 't-tick', 'end')]
for l in (1, 30, 60, 90, 120):
    b.append(text(X(l), Tp + ph + 20, str(l), 't-tick', 'middle'))
b.append(text(Lp + pw / 2, Tp + ph + 40, 'layer', 't-tick', 'middle'))
series = [('HC, worst seed', MHC['hc_max'], 'l2', 's2'), ('HC, median', MHC['hc_median'], 'l1', 's1'), ('mHC, all seeds', MHC['mhc_max'], 'l3', 's3')]
for name, vals, lc, dc in series:
    pts = ' '.join(f'{X(l):.1f},{Y(max(v, 1)):.1f}' for l, v in zip(xs_, vals))
    b.append(f'<polyline class="{lc}" points="{pts}"/>')
    yl = Y(max(vals[-1], 1))
    b.append(f'<circle class="{dc}" cx="{X(depth):.1f}" cy="{yl:.1f}" r="4"/>')
    b.append(text(X(depth) + 10, yl + (4 if 'mHC' not in name else -6), f'{name}: {vals[-1]:,.0f}' if vals[-1] > 2 else f'{name}: {vals[-1]:.3f}', 't-note'))
F['mhc_gain'] = svg(W, Tp + ph + 52, 'In our toy the gain of unconstrained hyper-connections grows to hundreds after 120 layers, while the Sinkhorn-projected matrices of mHC keep it at 1.', b)

# 9. Sinkhorn on a 2 x 2 matrix (our script output) --------------------------------------------------------------
mats = [('exp(raw)', [[1, 3], [1, 1]], 'rows 4, 2  cols 2, 4'),
        ('columns / sum', [[0.5, 0.75], [0.5, 0.25]], 'rows 1.25, 0.75'),
        ('rows / sum', [[0.4, 0.6], [0.667, 0.333]], 'cols 1.067, 0.933'),
        ('after 20 steps', [[0.366, 0.634], [0.634, 0.366]], 'rows 1, 1  cols 1, 1')]
b = [text(20, 24, 'Sinkhorn-Knopp: divide columns by their sums, then rows, and repeat', 't-title')]
for i, (ttl, M, note) in enumerate(mats):
    x = 20 + i * 165
    b.append(text(x + 70, 54, ttl, 't-note', 'middle'))
    for r in range(2):
        for c in range(2):
            cls = 'box-1' if i == 3 else 'box'
            b.append(box(x + 10 + c * 62, 66 + r * 40, 56, 32, cls, rx=4))
            v = M[r][c]
            b.append(text(x + 38 + c * 62, 87 + r * 40, f'{v:g}' if isinstance(v, int) else f'{v:.3f}'.rstrip('0').rstrip('.'), 't-val', 'middle'))
    b.append(text(x + 70, 168, note, 't-muted', 'middle'))
    if i < 3:
        b.append(arrow(x + 142, 106, x + 168, 106))
F['sinkhorn'] = svg(W, 184, 'Sinkhorn-Knopp on the 2 x 2 matrix [[1, 3], [1, 1]]: alternate column and row normalisation until every row and every column sums to 1.', b)

# 10. Engram lookup ------------------------------------------------------------------------------------------------
b = [text(20, 24, 'Engram at one position: hash the last tokens, read rows, gate, add', 't-title')]
toks = ['obeyed', 'Alexander', 'the', 'Great']
for i, t in enumerate(toks):
    b += label_box(20 + i * 104, 44, 96, 30, 'box-1' if i >= 1 else 'box', t)
b += label_box(20, 100, 200, 34, 'box', '2-gram: the Great')
b += label_box(232, 100, 200, 34, 'box', '3-gram: Alexander the Great')
b += [arrow(278, 74, 140, 100), arrow(330, 74, 330, 100)]
for j, (x, lab) in enumerate([(20, 'head 1'), (122, 'head 2'), (232, 'head 1'), (334, 'head 2')]):
    b += label_box(x, 158, 96, 30, 'box-2', f'hash, {lab}')
    b.append(arrow(x + 48, 134, x + 48, 158))
    for r in range(5):
        cls = 'box-on' if r == (j * 2 + 1) % 5 else 'box-ghost'
        b.append(f'<rect class="{cls}" x="{x + 18}" y="{206 + r * 12}" width="60" height="9" rx="2"/>')
    b.append(arrow(x + 48, 188, x + 48, 204))
b.append(text(20, 286, 'one row from each table (prime-sized, about 16M rows each in V4.1)', 't-muted'))
b.append(line(40, 272, 420, 272, 'drop'))
b += label_box(460, 216, 160, 34, 'box', 'concatenate: e')
b += label_box(460, 158, 160, 34, 'box-3', 'gate: does e fit h?')
b += label_box(460, 100, 160, 34, 'box-1', 'add to the stream')
b += label_box(632, 158, 36, 34, 'box', 'h')
b += [arrow(420, 233, 460, 233), arrow(540, 216, 540, 192), arrow(540, 158, 540, 134), arrow(632, 175, 620, 175)]
b.append(text(460, 84, 'h = hidden state at this position', 't-muted'))
F['engram'] = svg(W, 300, 'Engram hashes the last two and three tokens with several hash heads, reads one row from each table, joins the rows, weights them with a gate that compares them with the hidden state, and adds the result to the residual stream.', b)

# 11. Causal encoder-decoder ---------------------------------------------------------------------------------------
b = [text(20, 24, 'V4.1-Flash: the decoder gets its long-range KV from the encoder', 't-title')]
b += label_box(20, 220, 250, 40, 'box', 'prompt tokens')
b += label_box(20, 150, 250, 50, 'box-1', 'causal encoder, layers 0 to 19', 'run on every prompt token')
b += label_box(20, 70, 250, 50, 'box-2', 'decoder, layers 20 to 39', 'run on the newest tokens only')
b += [arrow(145, 220, 145, 200), arrow(145, 150, 145, 120)]
b += label_box(330, 150, 330, 50, 'box-on', 'encoder output H (layer 19)', 'projected once into decoder KV')
b += [arrow(270, 175, 330, 175), arrow(495, 150, 270, 100)]
b.append(text(330, 232, 'Prefill of a long prompt runs only the 20 encoder layers', 't-note'))
b.append(text(330, 252, 'for most tokens: about half the work (8B active, not 16B).', 't-note'))
b.append(text(330, 272, 'Each decoder layer still keeps its own 128-token window.', 't-muted'))
F['ced'] = svg(W, 290, 'In the causal encoder-decoder of V4.1-Flash, the long-range KV of the decoder layers is projected from the output of the last encoder layer, so a long prompt only needs the encoder half during prefill.', b)

save('figs_part6', F)
print(f'wrote {len(F)} figures: {", ".join(F)}')
