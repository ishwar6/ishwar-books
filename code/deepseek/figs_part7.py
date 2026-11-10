"""Figures for Part 7 (running DeepSeek). Writes results/figs_part7.json (key -> SVG). CSS classes only."""
import json, math, os
from figlib import svg, text, box, arrow, esc, lines_chart
from common import RESULTS, save

F = json.load(open(os.path.join(RESULTS, 'part7_fits.json')))
S = json.load(open(os.path.join(RESULTS, 'part7_serving.json')))
FIGS = {}
W = 760


def hbars(label, rows, title, unit_fmt, xmax, w=W, lab_w=250, bar_h=24, gap=10, log=False, marks=(), note=None):
    """rows: (name, value, cls, value_label). Horizontal bars with direct labels. marks: (value, text) dashed lines."""
    top = 44
    h = top + len(rows) * (bar_h + gap) + 44 + (18 if note else 0) + (14 if len(marks) > 1 else 0)
    x0, x1 = lab_w, w - 90
    if log:
        lo = math.log10(xmax[0]); hi = math.log10(xmax[1])
        X = lambda v: x0 + (math.log10(v) - lo) / (hi - lo) * (x1 - x0)
    else:
        X = lambda v: x0 + v / xmax * (x1 - x0)
    b = [text(20, 24, title, 't-title')]
    yb = top + len(rows) * (bar_h + gap)
    for k, (v, t) in enumerate(marks):
        b.append(f'<line class="edge-dim" x1="{X(v):.1f}" y1="{top - 8}" x2="{X(v):.1f}" y2="{yb + (12 if k % 2 else 0)}"/>')
        b.append(text(X(v), yb + 16 + (14 if k % 2 else 0), t, 't-muted', 'middle'))
    for i, (name, v, cls, vl) in enumerate(rows):
        y = top + i * (bar_h + gap)
        b.append(text(x0 - 10, y + bar_h / 2 + 5, name, 't-tick', 'end'))
        b.append(f'<g class="mark"><title>{esc(name)}: {esc(vl)}</title>'
                 f'<rect class="{cls}" x="{x0}" y="{y}" width="{max(2, X(v) - x0):.1f}" height="{bar_h}" rx="4"/></g>')
        b.append(text(X(v) + 8, y + bar_h / 2 + 5, vl, 't-val'))
    yb = top + len(rows) * (bar_h + gap)
    b.append(f'<line class="axis" x1="{x0}" y1="{yb - gap / 2}" x2="{x1}" y2="{yb - gap / 2}"/>')
    extra = 14 if len(marks) > 1 else 0
    b.append(text(x0, yb + 34 + extra, unit_fmt, 't-muted'))
    if note:
        b.append(text(x0, yb + 52 + extra, note, 't-muted'))
    return svg(w, h, label, b)


# 1. the size ladder of the 671B model
files = [f for f in F['files'] if f['model'] == 'V3 / R1 (671B)']
rows = [(f"{f['fmt']} ({f['bits']:.2f} bits)", f['gb'], 's1' if 'official' in f['fmt'] else 's2', f"{f['gb']:,.1f} GB") for f in files]
FIGS['ladder'] = hbars('File sizes of the same 671B model in six formats, with three memory sizes marked.', rows,
                       'One 671B model, six file sizes (DeepSeek-V3 / R1)', 'GB on disk (decimal). Dashed lines: memory of common machines.',
                       1500, marks=[(512, '512 GB Mac'), (640, '8x H100'), (1128, '8x H200')])

# 2. where V3's weights live, and what one token touches
v3 = F['v3']
routed, rest, used = v3['routed'] / 1e9, v3['rest'] / 1e9, v3['per_token_experts'] / 1e9
b = [text(20, 24, "DeepSeek-V3's 671B weights: stored versus used for one token", 't-title')]
x0, x1 = 20, W - 20
sc = (x1 - x0) / (routed + rest)
for row, (lab, y) in enumerate([('stored (all weights must be somewhere)', 60), ('read for one token (37B)', 150)]):
    b.append(text(x0, y - 10, lab, 't-note'))
    if row == 0:
        b.append(box(x0, y, routed * sc, 40, 'box-2', 4))
        b.append(box(x0 + routed * sc, y, max(rest * sc, 6), 40, 'box-on', 4))
        b.append(text(x0 + 12, y + 25, f'routed experts: {routed:.1f}B (97.4%)', 't-note'))
        b.append(text(x1, y + 62, f'everything else: {rest:.1f}B (2.6%), always used', 't-tick', 'end'))
    else:
        b.append(box(x0, y, (routed + rest) * sc, 40, 'box-ghost', 4))
        b.append(box(x0, y, used * sc, 40, 'box-2', 4))
        b.append(box(x0 + used * sc, y, rest * sc, 40, 'box-on', 4))
        b.append(text(x0 + used * sc + rest * sc + 12, y + 25, f'{used:.1f}B of experts (8 of 256 per layer) + {rest:.1f}B always', 't-note'))
b.append(text(x0, 228, 'Memory size follows the top bar. Decode speed follows the bottom bar.', 't-muted'))
FIGS['v3weights'] = svg(W, 244, 'Bar chart: 97.4 percent of DeepSeek-V3 weights are routed experts; one token reads about 20 billion expert weights plus 17 billion other weights.', b)

# 3. KV cache bytes per token, log scale
kv = F['kv']
order = [('naive multi-head (V3 shape)', F['naive_v3'], 's3'), ('R1-Distill-Llama-70B (GQA)', kv['Distill-Llama-70B']['bytes_per_token'], 's3'),
         ('R1-Distill-Llama-8B (GQA)', kv['Distill-Llama-8B']['bytes_per_token'], 's3'),
         ('V3 / R1 / V3.1 (MLA, BF16)', kv['V3 / R1 / V3.1']['bytes_per_token'], 's1'), ('V3.2 (FP8 MLA + indexer)', kv['V3.2']['bytes_per_token'], 's1'),
         ('V4-Pro (CSA + HCA), estimate', kv['V4-Pro']['bytes_per_token'], 's2'), ('V4-Flash (CSA + HCA)', kv['V4-Flash']['bytes_per_token'], 's2'),
         ('V4.1-Flash (CSA2, FP4)', kv['V4.1-Flash']['bytes_per_token'], 's2')]
rows = [(n, v, c, f'{v:,.0f} B') for n, v, c in order]
FIGS['kvarc'] = hbars('KV cache bytes per token on a log scale, from about 5 MB for a naive cache down to 890 bytes for V4.1-Flash.', rows,
                      'KV cache per token (log scale)', 'bytes per token, log scale: each grid step is 10x', (300, 2e7), log=True,
                      marks=[(1e3, '1 KB'), (1e4, '10 KB'), (1e5, '100 KB'), (1e6, '1 MB')])

# 4. what fits where grid
ms = [m[0] for m in F['machines']]
fits = F['fits']
cw, rh, lw, top = 54, 26, 236, 96
w = lw + cw * len(ms) + 20
h = top + rh * len(fits) + 30
b = [text(20, 24, 'Weights + one 32K-token KV cache within 90% of memory', 't-title')]
for j, m in enumerate(ms):
    cx, cy = lw + j * cw + cw / 2, top - 10
    b.append(text(cx, cy, m, 't-tick', 'start', f' transform="rotate(-40 {cx:.1f} {cy:.1f})"'))
for i, r in enumerate(fits):
    y = top + i * rh
    b.append(text(lw - 10, y + rh / 2 + 4, f"{r['setup']} ({r['need']:,.0f})", 't-tick', 'end'))
    for j, ok in enumerate(r['fits']):
        b.append(box(lw + j * cw + 3, y + 3, cw - 6, rh - 6, 'box-1' if ok else 'box-ghost', 3))
        if ok:
            b.append(text(lw + j * cw + cw / 2, y + rh / 2 + 4, 'fits', 't-tick', 'middle'))
b.append(text(lw, h - 8, 'Number in brackets: GB needed. Filled cell: fits by memory alone (not a speed claim).', 't-muted'))
FIGS['fitsgrid'] = svg(w, h, 'A grid of 17 DeepSeek setups against 9 machines showing which fit in memory.', b)

# 5. tensor, data and expert parallelism
b = [text(20, 24, 'Three ways to spread one model over 4 GPUs', 't-title')]
cols = [('Tensor parallel (TP)', 'every GPU holds 1/4 of every matrix;', 'they talk at every layer'),
        ('Data parallel (DP)', 'every GPU holds a full copy;', 'each serves different requests'),
        ('Expert parallel (EP)', 'each GPU holds different experts;', 'tokens travel to their experts')]
for k, (t1, t2, t3) in enumerate(cols):
    X0 = 20 + k * 248
    b.append(text(X0, 58, t1, 't-note'))
    for g in range(4):
        y = 72 + g * 46
        b.append(box(X0, y, 220, 38, 'box', 6))
        b.append(text(X0 + 8, y + 24, f'GPU {g}', 't-tick'))
        if k == 0:
            for layer in range(4):
                b.append(box(X0 + 62 + layer * 38, y + 8, 30, 22, 'box-1', 3))
                b.append(text(X0 + 77 + layer * 38, y + 24, f'L{layer + 1}', 't-tick', 'middle'))
        elif k == 1:
            b.append(box(X0 + 62, y + 8, 146, 22, 'box-1', 3))
            b.append(text(X0 + 135, y + 24, f'full model, requests {chr(65 + g)}', 't-tick', 'middle'))
        else:
            b.append(box(X0 + 62, y + 8, 146, 22, 'box-2', 3))
            b.append(text(X0 + 135, y + 24, f'experts {g * 64}-{g * 64 + 63}', 't-tick', 'middle'))
    b.append(text(X0, 274, t2, 't-muted'))
    b.append(text(X0, 290, t3, 't-muted'))
b.append(text(20, 312, 'TP: box L1 on GPU 0 is one quarter of layer 1. Real engines mix all three: DP for attention, EP for experts.', 't-muted'))
FIGS['parallel'] = svg(W, 324, 'Three panels: tensor parallel splits every matrix, data parallel copies the model, expert parallel splits the experts.', b)

# 6. bandwidth ceiling versus measured on an M3 Ultra
rows = [('ceiling, V3/R1 Q4_K_M (37B active)', 35.9, 's1', 'at most 35.9 tok/s'),
        ('measured, R1 4-bit, short prompt', 17.5, 's2', 'about 17-18 tok/s'),
        ('measured, V3-0324 Q4_K_M, 4.8K prompt', 10.89, 's2', '10.89 tok/s'),
        ('measured, V3-0324 Q4_K_M, 12.7K prompt', 6.67, 's2', '6.67 tok/s'),
        ('ceiling, dense 70B distill Q4_K_M', 18.8, 's3', 'at most 18.8 tok/s')]
FIGS['ceiling'] = hbars('On an M3 Ultra, the bandwidth ceiling for V3 is 35.9 tokens per second; reports measure 17-18 for short prompts and less for long ones.', rows,
                        'M3 Ultra (over 800 GB/s): ceiling versus reported speed', 'output tokens per second (one user)', 40, lab_w=300)

# 7. experts on GPU, RAM and SSD
b = [text(20, 24, 'Where the experts can live: three tiers, one token', 't-title')]
tiers = [('GPU memory (fast, small)', 'attention, shared expert, KV cache, hot experts', 54, 6, [1]),
         ('RAM (slower, big)', 'most routed experts', 140, 12, [2, 8]),
         ('SSD (slowest, huge)', 'cold experts or Engram tables, read on demand', 226, 12, [5])]
targets = []
for name, what, y, n, chosen in tiers:
    b.append(box(20, y, 600, 72, 'box', 8))
    b.append(text(32, y + 22, name, 't-note'))
    b.append(text(32, y + 40, what, 't-muted'))
    for e in range(n):
        ex = 330 + e * 23
        b.append(box(ex, y + 46 - 26 + 10, 18, 26, 'box-2' if e in chosen else 'box-ghost', 3))
        if e in chosen:
            targets.append((ex + 18, y + 43))
b.append(box(660, 156, 80, 30, 'box-on', 6))
b.append(text(700, 176, 'token', 't-note', 'middle'))
for tx, ty in targets:
    b.append(arrow(660, 171, tx + 4, ty, on=True))
b.append(text(20, 322, 'A token reads only its chosen experts (filled boxes). The tier that holds each one sets how long the read takes.', 't-muted'))
FIGS['tiers'] = svg(W, 336, 'Three memory tiers, GPU, RAM and SSD, each holding experts; one token reads four chosen experts spread over the tiers.', b)

# 8. prefill / decode disaggregation
b = [text(20, 24, "DeepSeek's serving split (Feb 2025, V3/R1 on H800)", 't-title')]
b.append(box(20, 50, 300, 210, 'box-1', 10))
b.append(text(36, 74, 'Prefill unit: 4 nodes, 32 GPUs', 't-note'))
b.append(text(36, 94, 'reads the whole prompt at once', 't-muted'))
for g in range(32):
    gx, gy = 36 + (g % 8) * 34, 110 + (g // 8) * 30
    b.append(box(gx, gy, 28, 24, 'box', 3))
    b.append(text(gx + 14, gy + 16, '9', 't-tick', 'middle'))
b.append(text(36, 250, 'each GPU: 9 routed + 1 shared expert (EP32)', 't-tick'))
b.append(arrow(328, 155, 412, 155, on=True))
b.append(text(370, 142, 'KV cache', 't-tick', 'middle'))
b.append(box(420, 50, 320, 210, 'box-2', 10))
b.append(text(436, 74, 'Decode unit: 18 nodes, 144 GPUs', 't-note'))
b.append(text(436, 94, 'writes one token per step for every user', 't-muted'))
for g in range(144):
    gx, gy = 436 + (g % 24) * 12, 110 + (g // 24) * 20
    b.append(box(gx, gy, 9, 15, 'box', 2))
b.append(text(436, 250, 'each GPU: 2 routed + 1 shared expert (EP144)', 't-tick'))
b.append(text(20, 286, 'Both units: 256 routed experts + 32 redundant copies = 288 slots (32 x 9 = 144 x 2).', 't-muted'))
FIGS['pd'] = svg(W, 300, 'A prefill unit of 32 GPUs with 9 experts each sends the KV cache to a decode unit of 144 GPUs with 2 experts each.', b)

# 9. the 545% arithmetic
d6 = S['day6']
b = [text(20, 24, 'One day of V3 + R1 (27-28 Feb 2025): cost versus theoretical revenue', 't-title')]
mx = 600000
X = lambda v: 160 + v / mx * 520
b.append(text(150, 76, 'cost', 't-tick', 'end'))
b.append(f'<rect class="s3" x="160" y="58" width="{X(d6["cost"]) - 160:.1f}" height="30" rx="4"/>')
b.append(text(X(d6['cost']) + 8, 78, '$87,072 (226.75 nodes x 8 x 24 h x $2)', 't-val'))
b.append(text(150, 136, 'if all billed', 't-tick', 'end'))
b.append(text(150, 152, 'at R1 prices', 't-tick', 'end'))
x = 160
for v, cls, lab in [(47880, 's1', 'cache-hit input'), (146300, 's2', 'cache-miss input'), (367920, 's3', 'output')]:
    b.append(f'<g class="mark"><title>{lab}: ${v:,}</title><rect class="{cls}" x="{x:.1f}" y="120" width="{v / mx * 520:.1f}" height="30" rx="2"/></g>')
    x += v / mx * 520
for k, (v, cls, lab) in enumerate([(47880, 's1', 'cache-hit input'), (146300, 's2', 'cache-miss input'), (367920, 's3', 'output')]):
    lx = 160 + k * 180
    b.append(f'<rect class="{cls}" x="{lx}" y="166" width="12" height="12" rx="2"/>')
    b.append(text(lx + 18, 177, f'{lab}: ${v:,}', 't-tick'))
b.append(text(x + 8, 140, '$562,027', 't-val'))
b.append(text(20, 222, '(562,027 - 87,072) / 87,072 = 5.45, the "545%". Real revenue is lower: web and app are free, V3 is cheaper, night discounts.', 't-muted'))
FIGS['day6'] = svg(W, 236, 'Bar chart: a daily cost of 87,072 dollars against a theoretical revenue of 562,027 dollars split into cache-hit input, cache-miss input and output.', b)

# 10. DeepEP low-latency kernels: latency barely grows with EP size
eps = [8, 16, 32, 64, 128, 256]
FIGS['deepep'] = lines_chart(W, 300, 'DeepEP low-latency kernels: dispatch latency grows from 163 to 194 microseconds and combine from 318 to 360 as EP grows from 8 to 256.',
                             eps, [('combine', [318, 329, 350, 353, 369, 360], 'l2', 's2'), ('dispatch', [163, 173, 182, 186, 192, 194], 'l1', 's1')],
                             'expert-parallel group size (GPUs), log scale', 'microseconds per call (H800, 128 tokens)', 0, 400, [0, 100, 200, 300, 400],
                             xlog=True, right=150)

# 11. reported speeds at home
rows = [('V4-Flash Q2, M5 Max 128 GB, 2K ctx', 39.35, 's2', '39.35 (ds4)'),
        ('V4-Flash Q2, M5 Max 128 GB, 65K ctx', 27.64, 's2', '27.64 (ds4)'),
        ('V4-Flash, 1x RTX 5090 + CPU', 20, 's2', '"20+" (KTransformers)'),
        ('V4-Flash Q2, DGX Spark, 2K ctx', 18.05, 's2', '18.05 (ds4)'),
        ('R1 4-bit, M3 Ultra 512 GB', 17.5, 's1', 'about 17-18 (review)'),
        ('V3 Q4_K_M, 2x Xeon + 4090D, 6 experts', 13.69, 's1', '13.69 (KTransformers)'),
        ('V3 Q4_K_M, 2x Xeon + 4090D, 8 experts', 12.21, 's1', '12.21 (KTransformers)'),
        ('V3-0324 Q4_K_M, M3 Ultra, 4.8K prompt', 10.89, 's1', '10.89 (third party)'),
        ('V3 Q4_K_M, same Xeon box, llama.cpp', 4.51, 's3', '4.51 (KTransformers)'),
        ('R1 1.58-bit, one RTX 4090 + RAM', 2.5, 's3', '"2-3" (Unsloth staff)')]
FIGS['homespeeds'] = hbars('Reported decode speeds of DeepSeek models on home and workstation hardware, from 2-3 to about 39 tokens per second.', rows,
                           'Reported decode speeds outside the data centre', 'output tokens per second, one user; source in brackets', 45, lab_w=300,
                           note='Top four: V4-Flash (13B active). Rest: V3/R1 (37B active). Each from a different test.')

json.dump(FIGS, open(os.path.join(RESULTS, 'figs_part7.json'), 'w'))
print(f'{len(FIGS)} figures:', ', '.join(FIGS))
