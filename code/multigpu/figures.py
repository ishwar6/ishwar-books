"""Theme-aware SVG figures for Part 7, drawn from the saved results. Classes (box-1, s1, l1, t-note, ...) are styled
by the site's CSS, so every figure follows the dark and light themes. No colours are hard-coded here.
Run: python code/multigpu/figures.py  -> results/figures.json (assemble.py inlines them)."""
import json, math
from pathlib import Path
from html import escape
R = Path(__file__).resolve().parent
J = lambda n: json.loads((R / 'results' / f'{n}.json').read_text())
MEM, COL, LAY, KV = J('memory_math'), J('collectives'), J('layout_model'), J('kv_transfer')
DIS, MOE, MPS, TP = J('disagg_sim'), J('moe_routing'), J('measure_mps'), J('tp_demo')

DEFS = ('<defs><marker id="ah7" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path class="arrow" d="M0,0L10,5L0,10z"/></marker></defs>')


def text(x, y, s, cls='t-note', anchor='middle', extra=''):
    return f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{extra}>{escape(str(s))}</text>'


def rect(x, y, w, h, cls='box-1', rx=8):
    return f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{h:.1f}" rx="{rx}"/>'


def line(x, y, a, b, cls='edge', arrow=False):
    tip = ' marker-end="url(#ah7)"' if arrow else ''
    return f'<line class="{cls}" x1="{x:.1f}" y1="{y:.1f}" x2="{a:.1f}" y2="{b:.1f}"{tip}/>'


def poly(pts, cls='l1'):
    return f'<polyline class="{cls}" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}"/>'


def dot(x, y, cls='s1', r=4):
    return f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="{r}"/>'


def mark(title, body):
    return f'<g class="mark"><title>{escape(title)}</title>{body}</g>'


def svg(h, body, label):
    return f'<svg viewBox="0 0 760 {h}" role="img" aria-label="{escape(label)}">{DEFS}{body}</svg>'


def legend(x, y, items):
    out = ''
    for cls, lab in items:
        out += rect(x, y - 10, 12, 12, cls, 3) + text(x + 18, y, lab, 't-tick', 'start')
        x += 26 + 7.2 * len(lab)
    return out


def logscale(v, lo, hi, a, b):
    return a + (b - a) * (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))


f = {}
GB, GiB = 1e9, 2 ** 30

# 1. Weights + KV cache against GPU memory.
rows = MEM['models']
mx = max(r['total'] for r in rows) / GB
X0, W = 190, 400
sc = lambda v: W * v / mx
b = legend(190, 22, [('s1', 'weights'), ('s2', 'KV cache: 32 chats x 8,192 tokens')])
for i, r in enumerate(rows):
    y = 44 + i * 58
    w = r.get('weights_bf16', r.get('weights_fp8')) / GB
    kv = r['kv_32x8k'] / GB
    name = r['model'].replace(' (FP8)', '')
    b += text(X0 - 10, y + 17, name, 't-note', 'end')
    b += text(X0 - 10, y + 34, 'FP8 weights' if 'fp8' in str(r.keys()) and 'weights_fp8' in r else 'BF16 weights', 't-tick', 'end')
    b += mark(f'{name}: weights {w:.0f} GB, KV {kv:.0f} GB', rect(X0, y + 4, sc(w), 22, 's1', 3) + rect(X0 + sc(w), y + 4, sc(kv), 22, 's2', 3))
    b += text(X0 + sc(w + kv) + 8, y + 20, f"{w + kv:,.0f} GB -> {r['h100s']} x H100", 't-val', 'start')
for n in [1, 8]:
    x = X0 + sc(80 * n * 0.9)
    b += line(x, 34, x, 44 + 4 * 58 - 4, 'base-line')
    b += text(x, 44 + 4 * 58 + 12, f'{n} x 80 GB (90% usable)' if n == 1 else '8 x 80 GB: one node', 't-tick')
f['fit'] = svg(44 + 4 * 58 + 24, b, 'Weights plus the KV cache of 32 conversations of 8,192 tokens: Llama 3.1 8B 50 GB fits one H100; 70B needs 227 GB, four H100s; 405B in BF16 needs 947 GB, more than one 8-GPU node; DeepSeek-V3 in FP8 needs 689 GB, ten H100s.')

# 2. Decode floor vs number of GPUs.
fl = MEM['decode_floor_ms']
X0, X1, Y0, Y1 = 90, 700, 250, 40
ns = [1, 2, 4, 8, 16]
xs = lambda n: X0 + (X1 - X0) * math.log2(n) / 4
ys = lambda v: logscale(v, 0.2, 300, Y0, Y1)
b = ''
for v in [0.3, 1, 3, 10, 30, 100]:
    b += line(X0, ys(v), X1, ys(v), 'grid') + text(X0 - 8, ys(v) + 4, f'{v:g} ms', 't-tick', 'end')
for n in ns:
    b += text(xs(n), Y0 + 20, f'{n} GPU' + ('s' if n > 1 else ''), 't-tick')
b += line(X0, Y0, X1, Y0, 'axis')
for k, (name, cls) in enumerate([('Llama 3.1 8B', 'l1'), ('Llama 3.1 70B', 'l2'), ('Llama 3.1 405B', 'l3')]):
    pts = [(xs(int(n)), ys(v)) for n, v in fl[name].items()]
    fits = {'Llama 3.1 8B': 1, 'Llama 3.1 70B': 2, 'Llama 3.1 405B': 16}[name]
    pts_fit = [(xs(int(n)), ys(v)) for n, v in fl[name].items() if int(n) >= fits]
    pts_no = [(xs(int(n)), ys(v)) for n, v in fl[name].items() if int(n) <= fits]
    b += poly(pts_no, 'long') + poly(pts_fit, cls)
    for n, v in fl[name].items():
        if int(n) >= fits:
            b += mark(f'{name} on {n} GPUs: {v:.1f} ms', dot(xs(int(n)), ys(v), cls.replace('l', 's')))
    x, y = pts_fit[0]
    b += text(x - 12 if fits == 16 else x + 10, y + 5 if fits == 16 else y - 10, f'{name}: {fl[name][str(fits)]:.1f} ms', 't-val', 'end' if fits == 16 else 'start')
b += text(380, Y0 + 44, 'Dashed: the weights do not fit there. BF16 weights read once per step at 3.35 TB/s per H100, perfect split.', 't-tick')
f['floor'] = svg(Y0 + 56, b, 'Lowest possible time per decode step at batch 1: reading the weights once, split perfectly over 1 to 16 H100s. 8B: 4.8 ms on one GPU. 70B: 21.1 ms on two, 5.3 ms on eight. 405B: 15.1 ms on sixteen.')

# 3. A node: 8 GPUs, NVSwitch, NICs, and the network.
b = rect(20, 20, 470, 300, 'plane', 14) + text(255, 44, 'One server (node): 8 GPUs', 't-title')
for i in range(8):
    x = 40 + (i % 4) * 110; y = 64 if i < 4 else 236
    b += rect(x, y, 90, 50, 'box-1') + text(x + 45, y + 22, f'GPU {i}', 't-note') + text(x + 45, y + 40, '80 GB HBM', 't-tick')
    b += line(x + 45, y + 50 if i < 4 else y, x + 45, 150 if i < 4 else 198, 'edge')
b += rect(40, 150, 420, 48, 'box-2') + text(250, 172, 'NVSwitch: any GPU to any GPU', 't-note') + text(250, 190, '900 GB/s per GPU (both directions)', 't-tick')
b += rect(560, 64, 170, 74, 'box-3') + text(645, 92, '8 network cards', 't-note') + text(645, 112, '400 Gb/s each = 50 GB/s', 't-tick') + text(645, 128, '(InfiniBand or Ethernet)', 't-tick')
b += line(490, 101, 556, 101, 'edge', True)
b += rect(560, 236, 170, 74, 'box') + text(645, 264, 'other nodes', 't-note') + text(645, 284, 'through network switches', 't-tick')
b += line(645, 138, 645, 232, 'long', True)
b += rect(520, 160, 0, 0) + text(660, 190, 'about 9x slower', 't-tick', 'start') + text(660, 206, 'than NVLink', 't-tick', 'start')
b += text(255, 345, 'Inside: NVLink through NVSwitch. Between nodes: one NIC per GPU, over the network.', 't-tick')
f['node'] = svg(360, b, 'An 8-GPU server. The GPUs talk to each other through NVSwitch at 900 GB/s per GPU. Each GPU also has a 400 Gb/s network card (50 GB/s) for traffic to other servers.')

# 4. Link bandwidths, log scale.
links = [('HBM inside one H100', 3350, 's1'), ('NVLink 4 per GPU (both directions)', 900, 's2'),
         ('PCIe Gen5 x16 (both directions)', 128, 's3'), ('InfiniBand NDR, one 400 Gb/s NIC', 50, 's4'),
         ('Ethernet 100 Gb/s', 12.5, 's4')]
X0, X1 = 300, 690
b = ''
for v in [10, 100, 1000]:
    x = logscale(v, 5, 5000, X0, X1)
    b += line(x, 18, x, 18 + 5 * 44, 'grid') + text(x, 18 + 5 * 44 + 16, f'{v:,} GB/s', 't-tick')
for i, (lab, v, cls) in enumerate(links):
    y = 22 + i * 44
    w = logscale(v, 5, 5000, X0, X1) - X0
    b += text(X0 - 10, y + 18, lab, 't-note', 'end') + mark(f'{lab}: {v} GB/s', rect(X0, y + 4, w, 22, cls, 3))
    b += text(X0 + w + 8, y + 20, f'{v:,g}', 't-val', 'start')
b += text(380, 18 + 5 * 44 + 40, 'Spec-sheet peaks (NVIDIA H100, ConnectX-7). Log scale: each grid line is 10x.', 't-tick')
f['links'] = svg(18 + 5 * 44 + 52, b, 'Peak bandwidths on a log scale: HBM 3,350 GB/s, NVLink 900 GB/s, PCIe Gen5 128 GB/s, one 400 Gb/s InfiniBand card 50 GB/s, 100 Gb/s Ethernet 12.5 GB/s.')

# 5. Tensor parallel MLP: column split then row split, one all-reduce.
b = text(20, 24, 'Megatron MLP on 2 GPUs:  Y = GeLU(X A),  Z = Y B', 't-title', 'start')
for g in range(2):
    y = 50 + g * 120
    b += rect(20, y, 80, 70, 'box') + text(60, y + 32, 'X', 't-node') + text(60, y + 54, 'full copy', 't-tick')
    b += line(100, y + 35, 136, y + 35, 'edge', True)
    b += rect(140, y, 130, 70, 'box-1') + text(205, y + 30, f'A{g+1}', 't-node') + text(205, y + 54, 'half the columns', 't-tick')
    b += line(270, y + 35, 306, y + 35, 'edge', True)
    b += rect(310, y, 100, 70, 'box-1') + text(360, y + 30, f'Y{g+1} = GeLU(X A{g+1})', 't-tick') + text(360, y + 52, 'no talking', 't-tick')
    b += line(410, y + 35, 446, y + 35, 'edge', True)
    b += rect(450, y, 120, 70, 'box-2') + text(510, y + 30, f'B{g+1}', 't-node') + text(510, y + 54, 'half the rows', 't-tick')
    b += line(570, y + 35, 618, 140 if g == 0 else 160, 'edge', True)
    b += text(110 + 270, y - 6 if g == 0 else y + 88, f'GPU {g}', 't-label') if False else ''
    b += text(16, y + 35, '', 't-tick')
b += rect(620, 110, 120, 80, 'box-3') + text(680, 140, 'all-reduce', 't-note') + text(680, 160, 'Z = Y1 B1 + Y2 B2', 't-tick') + text(680, 176, 'every GPU gets Z', 't-tick')
b += text(255, 135, 'GPU 0 above, GPU 1 below: each holds half of A and half of B', 't-tick')
b += text(380, 262, 'The only communication: one sum of the partial outputs, the size of the activations (tokens x d).', 't-tick')
f['tp_mlp'] = svg(276, b, 'Tensor-parallel MLP on two GPUs. Each GPU has the full input X, half the columns of A and the matching half of the rows of B. It computes its part with no communication. One all-reduce adds the two partial outputs.')

# 6. Attention heads split, KV cache split.
b = text(20, 24, 'Attention on 2 GPUs: whole heads per GPU, and each GPU caches only its own heads', 't-title', 'start')
for g in range(2):
    x = 30 + g * 360
    b += rect(x, 44, 330, 170, 'plane', 12) + text(x + 165, 66, f'GPU {g}', 't-note')
    for h in range(4):
        hx = x + 16 + h * 78
        b += rect(hx, 80, 66, 40, 'box-1', 6) + text(hx + 33, 105, f'head {g*4+h}', 't-tick')
    b += rect(x + 16, 132, 144, 34, 'box-2', 6) + text(x + 88, 154, f'KV head {g}', 't-tick')
    b += rect(x + 170, 132, 144, 34, 'box', 6) + text(x + 242, 154, 'KV cache: half', 't-tick')
    b += rect(x + 16, 176, 298, 30, 'box-3', 6) + text(x + 165, 196, 'partial output = its heads x its rows of W_o', 't-tick')
b += line(195, 214, 330, 248, 'edge', True) + line(555, 214, 430, 248, 'edge', True)
b += rect(300, 248, 160, 36, 'box-3') + text(380, 271, 'all-reduce (sum)', 't-note')
b += text(380, 306, 'Qwen2.5-0.5B in our demo: 14 query heads and 2 KV heads, so TP=2 gives 7 + 1 per GPU.', 't-tick')
f['tp_attn'] = svg(318, b, 'Attention split by heads across two GPUs. Each GPU runs its own query heads with its own key-value heads, keeps only those heads in its KV cache, and produces a partial output. An all-reduce sums the partial outputs.')

# 7. Where the two all-reduces sit in a layer, per token.
steps = [('attention (my heads)', 'box-1', 150), ('all-reduce', 'box-3', 70), ('MLP (my slice)', 'box-1', 190), ('all-reduce', 'box-3', 70)]
b = text(20, 24, 'One transformer layer under tensor parallelism, repeated for every layer', 't-title', 'start')
x = 30
for lab, cls, w in steps:
    b += rect(x, 44, w, 46, cls, 6) + text(x + w / 2, 72, lab, 't-tick')
    x += w + 14
b += text(x + 4, 72, '... next layer', 't-tick', 'start')
b += text(380, 120, 'Llama 3.1 70B: 80 layers x 2 = 160 all-reduces for every decode step.', 't-note')
b += text(380, 142, 'Each carries batch x 8,192 x 2 bytes: 16 KiB at batch 1, 1 MiB at batch 64.', 't-tick')
b += text(380, 162, 'Every GPU waits at each all-reduce, so its cost adds directly to the time per token.', 't-tick')
f['ar_layer'] = svg(178, b, 'Each layer runs attention on the local heads, an all-reduce, the local slice of the MLP, and a second all-reduce. Llama 3.1 70B has 160 all-reduces per decode step.')

# 8. Ring all-reduce trace (4 ranks).
tr = COL['ring_trace']
cols = [('start', tr['start'])] + [(f"{'reduce' if s['phase'].startswith('reduce') else 'gather'} {s['step']}", s['ranks']) for s in tr['steps']]
cw, x0 = 92, 74
b = ''
for j, (lab, ranks) in enumerate(cols):
    x = x0 + j * cw
    b += text(x + 42, 26, lab.replace('reduce-scatter', 'reduce-scatter').replace('all-gather', 'all-gather'), 't-tick')
    for r, vals in enumerate(ranks):
        y = 40 + r * 52
        for c, v in enumerate(vals):
            done = (j >= 3 and v == [10, 20, 30, 40][c])
            cls = 'box-3' if done else ('box-1' if j == 0 else 'box')
            b += rect(x + c * 21, y, 20, 30, cls, 3) + text(x + c * 21 + 10, y + 20, v, 't-cell')
for r in range(4):
    b += text(x0 - 12, 40 + r * 52 + 20, f'rank {r}', 't-note', 'end')
b += line(x0 + 4 * cw - 7, 34, x0 + 4 * cw - 7, 244, 'drop')
b += text(x0 + 2 * cw, 258, 'reduce-scatter: add chunks round the ring', 't-tick')
b += text(x0 + 5.5 * cw, 258, 'all-gather: pass the sums round', 't-tick')
b += text(380, 284, 'Highlighted: a chunk that already holds its final sum. After 2(p-1) = 6 steps every rank has [10, 20, 30, 40].', 't-tick')
f['ring'] = svg(298, b, 'Ring all-reduce on four ranks holding [1,2,3,4] times 1 to 4. Three reduce-scatter steps leave each rank with one fully summed chunk; three all-gather steps spread the sums, so every rank ends with [10, 20, 30, 40].')

# 9. The four collectives with the demo's numbers.
cd = COL['collectives_demo']
b = ''
panels = [('all-reduce', 'all get the full sum', 'tensor parallel (Megatron)'),
          ('all-gather', 'all get every piece', 'data-parallel attention'),
          ('reduce-scatter', 'rank r: sum of piece r', 'half of a ring all-reduce'),
          ('all-to-all', 'piece r goes to rank r', 'expert parallel (MoE)')]
for i, (name, what, use) in enumerate(panels):
    x = 10 + i * 188
    b += rect(x, 14, 178, 260, 'plane', 10) + text(x + 89, 38, name, 't-title') + text(x + 89, 58, what, 't-tick')
    for r in range(4):
        y = 76 + r * 40
        if name == 'all-reduce':
            v = [sum(cd['start'][q][c] for q in range(4)) for c in range(4)]
        elif name == 'all-gather':
            v = cd['all_gather']
        elif name == 'reduce-scatter':
            v = [cd['reduce_scatter'][r]]
        else:
            v = cd['all_to_all'][r]
        b += text(x + 14, y + 19, f'r{r}', 't-tick', 'start')
        for c, val in enumerate(v):
            b += rect(x + 38 + c * 34, y, 32, 28, 'box-1' if name != 'all-to-all' else ('box-2' if c == r else 'box-1'), 3) + text(x + 54 + c * 34, y + 19, val, 't-cell')
    b += text(x + 89, 260, use, 't-tick')
b += text(380, 298, 'Start: rank r holds [10r, 10r+1, 10r+2, 10r+3]. All-gather shares each rank\'s own piece (piece r).', 't-tick')
f['collectives'] = svg(312, b, 'Four collectives on four ranks. All-reduce gives everyone the elementwise sum [60, 64, 68, 72]. All-gather gives everyone [0, 11, 22, 33]. Reduce-scatter gives rank r the sum of piece r. All-to-all sends piece r of every rank to rank r, a transpose.')

# 10. TP scaling: stacked memory / communication for 70B.
rows = [r for r in LAY['tp_scaling_70b'] if r['fits']]
mx = max(r['total_ms'] for r in rows)
X0, W = 170, 420
b = legend(170, 20, [('s1', 'read weights + KV (memory)'), ('s3', 'all-reduces (communication)')])
for i, r in enumerate(rows):
    y = 36 + i * 40
    m = max(r['mem_ms'], r['compute_ms'])
    b += text(X0 - 10, y + 18, f"batch {r['batch']}, TP={r['tp']}", 't-note', 'end')
    b += mark(f"batch {r['batch']} TP={r['tp']}: {m:.2f} ms memory/compute, {r['comm_ms']:.2f} ms all-reduce",
              rect(X0, y + 4, W * m / mx, 22, 's1', 3) + rect(X0 + W * m / mx, y + 4, W * r['comm_ms'] / mx, 22, 's3', 3))
    b += text(X0 + W * r['total_ms'] / mx + 8, y + 20, f"{r['total_ms']:.1f} ms  (comm {100*r['comm_ms']/r['total_ms']:.0f}%)", 't-val', 'start')
y = 36 + len(rows) * 40
b += text(380, y + 16, 'Roofline model with H100 peaks; all-reduce = 2 x 5 us + bytes / 450 GB/s, 160 per step (alpha assumed).', 't-tick')
f['tp_scaling'] = svg(y + 28, b, 'Modelled decode step for Llama 3.1 70B. At batch 1: 22.9 ms on TP=2, 12.2 ms on TP=4, 6.9 ms on TP=8, where all-reduces are 23% of the step. At batch 64: 19.1 ms on TP=4 and 10.7 ms on TP=8.')

# 11. Measured all-reduce time vs size on this laptop (gloo, CPU processes).
ab = TP['allreduce_bench']
X0, X1, Y0, Y1 = 90, 700, 250, 30
allv = [r['us'] for k in ab for r in ab[k]['rows']]
lo, hi = min(allv) / 1.5, max(allv) * 1.5
xs = lambda n: logscale(n, 4, 2 ** 26, X0, X1)
ys = lambda v: logscale(v, lo, hi, Y0, Y1)
b = ''
for v in [10, 100, 1000, 10000, 100000]:
    if lo < v < hi:
        b += line(X0, ys(v), X1, ys(v), 'grid') + text(X0 - 8, ys(v) + 4, f'{v:,} us', 't-tick', 'end')
for n, lab in [(4, '4 B'), (1024, '1 KiB'), (2 ** 20, '1 MiB'), (2 ** 26, '64 MiB')]:
    b += text(xs(n), Y0 + 18, lab, 't-tick')
b += line(X0, Y0, X1, Y0, 'axis')
for k, cls in [('2', 'l1'), ('4', 'l2')]:
    pts = [(xs(r['bytes']), ys(r['us'])) for r in ab[k]['rows']]
    b += poly(pts, cls)
    for (x, y), r in zip(pts, ab[k]['rows']):
        b += mark(f"{k} processes, {r['bytes']:,} B: {r['us']:.0f} us", dot(x, y, cls.replace('l', 's'), 3.5))
b += legend(110, Y1 + 4, [('s1', f"2 processes: alpha {ab['2']['alpha_us']:.0f} us, beta {ab['2']['beta_GBps']:.1f} GB/s"),
                          ('s2', f"4 processes: alpha {ab['4']['alpha_us']:.0f} us, beta {ab['4']['beta_GBps']:.1f} GB/s")])
b += text(380, Y0 + 42, 'Measured: torch.distributed all_reduce, gloo backend, CPU processes on this laptop. Flat, then a slope.', 't-tick')
f['alpha_beta'] = svg(Y0 + 54, b, 'Measured all-reduce time against message size on this laptop with 2 and 4 processes: almost flat for small messages (the fixed cost alpha), then growing in proportion to size (the bandwidth term).')

# 12. Measured shard time on mps vs tokens.
sh = MPS['shard']
toks = sorted({r['tokens'] for r in sh})
b = legend(150, 20, [('s1', 'whole matrix'), ('s2', 'TP=2 shard'), ('s3', 'TP=4 shard')])
X0, W = 150, 470
for i, T in enumerate(toks):
    y = 36 + i * 64
    rs = [r for r in sh if r['tokens'] == T]
    mx = max(r['ms'] for r in rs)
    b += text(X0 - 10, y + 26, f'{T:,} token' + ('s' if T > 1 else ''), 't-note', 'end')
    for j, r in enumerate(rs):
        cls = ['s1', 's2', 's3'][j]
        b += mark(f"{T} tokens TP={r['tp']}: {r['ms']:.3f} ms", rect(X0, y + j * 17, W * r['ms'] / mx, 14, cls, 3))
        b += text(X0 + W * r['ms'] / mx + 6, y + j * 17 + 12, f"{r['ms']:.3f} ms" + (f"  {r['speedup']:.2f}x" if r['tp'] > 1 else ''), 't-tick', 'start')
y = 36 + len(toks) * 64
b += text(380, y + 6, 'Measured on the Apple GPU (mps), BF16, 14,336 x 4,096 matrix. Bars scaled per row.', 't-tick')
f['mps_shard'] = svg(y + 18, b, 'Measured time to multiply one Llama 3.1 8B MLP matrix by 1 to 4,096 tokens, whole and as the half and quarter shards that TP=2 and TP=4 would give each GPU.')

# 13. Pipeline schedule.
ps = LAY['pp_schedule']
b = ''
y0 = 30
for k, (M, title) in enumerate([('1', 'one sequence (M = 1): each stage works 1 slot in 4'), ('4', 'four micro-batches in flight (M = 4)')]):
    g = ps[M]['grid']
    b += text(20, y0 - 8, f"{title}: {ps[M]['tokens']} tokens in {ps[M]['steps']} slots, GPUs busy {ps[M]['utilisation']:.0%}", 't-note', 'start')
    for s, row in enumerate(g):
        y = y0 + s * 26
        b += text(84, y + 16, f'GPU {s}', 't-tick', 'end')
        for t, c in enumerate(row):
            x = 92 + t * 42
            b += rect(x, y, 40, 22, 'cell-masked' if c == '.' else f'box-{int(c) % 3 + 1}', 3)
            if c != '.':
                b += text(x + 20, y + 16, f'mb{c}', 't-cell')
    y0 += 4 * 26 + 46
b += text(380, y0 - 16, 'A token must pass all 4 stages before the next token of the same sequence can start.', 't-tick')
f['pp_schedule'] = svg(y0 - 4, b, 'Pipeline schedule on 4 GPUs. With one sequence each GPU is busy one slot in four (25%). With four micro-batches the GPUs are busy 80% of the time, but each sequence still waits 4 slots per token.')

# 14. 405B across two nodes: TP16 vs TP8xPP2.
pp = LAY['pp_405b']
b = text(20, 22, 'Llama 3.1 405B (BF16) on two 8-GPU nodes, modelled', 't-title', 'start')
items = []
for bs in ['1', '32']:
    r = pp[bs]
    items += [(f'batch {bs}: TP=16 over InfiniBand', r['tp16_ms'], r['tp16_tokens_s'], 's3'),
              (f'batch {bs}: TP=8 x PP=2, 1 micro-batch', r['pp_latency_ms'], r['pp_tokens_s_1mb'], 's1'),
              (f'batch {bs}: TP=8 x PP=2, 2 micro-batches', r['pp_latency_ms'], r['pp_tokens_s_2mb'], 's2')]
b += text(350, 44, 'ms per token (latency)', 't-label') + text(620, 44, 'tokens/s (throughput)', 't-label')
mxl, mxt = max(i[1] for i in items), max(i[2] for i in items)
for i, (lab, lat, thr, cls) in enumerate(items):
    y = 56 + i * 34 + (12 if i >= 3 else 0)
    b += text(270, y + 16, lab, 't-tick', 'end')
    b += mark(f'{lab}: {lat:.1f} ms', rect(280, y + 2, 170 * lat / mxl, 18, cls, 3)) + text(286 + 170 * lat / mxl, y + 16, f'{lat:.1f}', 't-val', 'start')
    b += mark(f'{lab}: {thr:,.0f} tokens/s', rect(520, y + 2, 170 * thr / mxt, 18, cls, 3)) + text(526 + 170 * thr / mxt, y + 16, f'{thr:,.0f}', 't-val', 'start')
y = 56 + 6 * 34 + 12
b += text(380, y + 14, 'Pipeline: no faster per token, but more tokens per second. TP=16: lower latency, many network messages.', 't-tick')
f['pp_vs_tp'] = svg(y + 26, b, 'Modelled 405B on two nodes. At batch 32, TP=16 over InfiniBand takes 31.4 ms per token and makes 1,020 tokens per second; TP=8 with PP=2 takes 36.4 ms per token and makes 1,760 tokens per second with two micro-batches in flight.')

# 15. MoE dispatch and combine.
b = text(20, 22, 'One MoE layer with expert parallelism (EP = 4 GPUs, 2 experts each, top-2)', 't-title', 'start')
for g in range(4):
    x = 30 + g * 182
    b += rect(x, 40, 160, 60, 'box') + text(x + 80, 62, f'GPU {g}: attention', 't-note') + text(x + 80, 84, 'router picks 2 experts/token', 't-tick')
    b += rect(x, 190, 160, 60, 'box-1') + text(x + 80, 214, f'experts {2*g}, {2*g+1}', 't-note') + text(x + 80, 236, 'FFN on received tokens', 't-tick')
    b += rect(x, 330, 160, 44, 'box') + text(x + 80, 357, 'weighted sum of 2 outputs', 't-tick')
for a, c in [(0, 2), (0, 3), (1, 0), (1, 3), (2, 1), (2, 2), (3, 0), (3, 1)]:
    b += line(110 + a * 182, 100, 110 + c * 182 + (a - 1.5) * 8, 186, 'edge', True)
    b += line(110 + c * 182 + (a - 1.5) * 8, 250, 110 + a * 182, 326, 'drop', True)
b += rect(300, 128, 160, 30, 'box-3', 6) + text(380, 148, 'dispatch: all-to-all', 't-note')
b += rect(300, 270, 160, 30, 'box-3', 6) + text(380, 290, 'combine: all-to-all', 't-note')
b += text(380, 396, 'Two all-to-alls per MoE layer. Where tokens go depends on the router, so traffic is uneven.', 't-tick')
f['moe_dispatch'] = svg(408, b, 'Expert parallelism on four GPUs. Each GPU runs attention for its tokens, the router picks experts, a dispatch all-to-all sends each token to the GPUs holding its experts, the experts compute, and a combine all-to-all sends the outputs back.')

# 16. Real expert shares in one layer (OLMoE), sorted, wikitext vs code.
w_share = MOE['wikitext']['share']; c_share = MOE['python code']['share']
L = len(w_share)
lay = max(range(L), key=lambda l: max(w_share[l]))   # the most skewed layer
lay_med = sorted(range(L), key=lambda l: max(w_share[l]))[L // 2]
b = ''
for k, (l, title) in enumerate([(lay_med, 'a typical layer'), (lay, 'the most uneven layer')]):
    y0 = 40 + k * 170
    vals = sorted(w_share[l], reverse=True)
    mxv = max(vals) * 1.1
    b += text(20, y0 - 14, f'Layer {l} ({title}), WikiText: share of the 8 expert slots per token', 't-note', 'start')
    for e, v in enumerate(vals):
        h = 110 * v / mxv
        b += mark(f'expert rank {e+1}: {v*100:.2f}% of visits', rect(60 + e * 10, y0 + 110 - h, 8, h, 's1', 1))
    fair = 110 * (1 / 64) / mxv
    b += line(56, y0 + 110 - fair, 704, y0 + 110 - fair, 'base-line') + text(708, y0 + 114 - fair, 'fair 1/64', 't-tick', 'start')
    b += text(64, y0 + 6, f'hottest: {vals[0]*64:.1f}x fair', 't-val', 'start')
    b += line(56, y0 + 110, 704, y0 + 110, 'axis')
b += text(380, 380, 'Measured: OLMoE-1B-7B (64 experts, top-8) routing 12,288 WikiText tokens. Experts sorted by load.', 't-tick')
f['moe_share'] = svg(392, b, 'Measured share of routing slots per expert in OLMoE-1B-7B, sorted from busiest to idlest, for a typical layer and the most uneven layer. A few experts get several times their fair share; some get almost nothing.')

# 17. Imbalance vs EP size: real vs random.
im = MOE['wikitext']['imbalance']
b = legend(170, 20, [('s2', 'random routing (bad luck only)'), ('s1', 'real routing, WikiText (measured)')])
mx = max(v['real'] for v in im.values())
X0, W = 170, 470
rowsI = [(G, B) for G in [8, 16, 64] for B in [64, 4096]]
for i, (G, B) in enumerate(rowsI):
    y = 36 + i * 46 + (10 if G > 8 else 0) + (10 if G > 16 else 0)
    v = im[f'EP{G}_B{B}']
    b += text(X0 - 10, y + 18, f'EP={G}, {B:,} tokens', 't-note', 'end')
    b += mark(f'EP={G} batch {B}: random {v["random"]:.2f}', rect(X0, y, W * v['random'] / mx, 15, 's2', 3))
    b += mark(f'EP={G} batch {B}: real {v["real"]:.2f}', rect(X0, y + 18, W * v['real'] / mx, 15, 's1', 3))
    b += text(X0 + W * v['random'] / mx + 6, y + 12, f"{v['random']:.2f}", 't-tick', 'start')
    b += text(X0 + W * v['real'] / mx + 6, y + 30, f"{v['real']:.2f}x", 't-val', 'start')
y = 36 + len(rowsI) * 46 + 20
b += text(380, y + 6, 'Busiest GPU / average GPU. 1.00 would be perfect. The whole layer waits for the busiest GPU.', 't-tick')
f['moe_imbalance'] = svg(y + 18, b, 'Load on the busiest GPU divided by the average, OLMoE routing on WikiText. With 4,096-token batches random routing would be nearly even (1.02 to 1.10) but real routing gives 1.32 at EP=8, 1.57 at EP=16 and 3.18 at EP=64.')

# 18. Redundant experts: default vs balanced vs redundant.
red = MOE['wikitext']['redundant']
b = legend(170, 20, [('s2', 'experts in order'), ('s3', 'placed by past load'), ('s1', 'plus redundant copies')])
keys = list(red)
mx = max(max(v.values()) for v in red.values())
for i, k in enumerate(keys):
    y = 38 + i * 70
    v = red[k]
    G, extra = k.replace('EP', '').split('+')
    b += text(160, y + 30, f'EP={G}, {extra} copies', 't-note', 'end')
    for j, (kk, cls) in enumerate([('default', 's2'), ('balanced', 's3'), ('redundant', 's1')]):
        b += mark(f'{k} {kk}: {v[kk]:.2f}', rect(170, y + j * 19, 450 * v[kk] / mx, 16, cls, 3))
        b += text(176 + 450 * v[kk] / mx, y + j * 19 + 13, f'{v[kk]:.2f}', 't-tick', 'start')
y = 38 + len(keys) * 70
b += text(380, y + 4, 'Simulated on measured routing: placement learned on the first half of the text, tested on the second half.', 't-tick')
f['moe_redundant'] = svg(y + 16, b, 'Busiest GPU over average for OLMoE routing on WikiText with experts placed in order, placed by past load, and with extra copies of the hottest experts.')

# 19. DeepSeek-V3 deployment units (from the report).
b = text(20, 22, 'DeepSeek-V3 serving on H800s (technical report, Section 3.4)', 't-title', 'start')
for k, (title, nodes, attn, moe, note) in enumerate([
        ('Prefill unit', '4 nodes, 32 GPUs', 'attention: TP4 + SP, DP8', 'MoE: EP32, 8 experts + 1 redundant per GPU', 'two micro-batches overlap compute and all-to-all'),
        ('Decode unit', '40 nodes, 320 GPUs', 'attention: TP4 + SP, DP80', 'MoE: EP320, 1 expert per GPU (64 GPUs for redundant + shared)', 'point-to-point IB transfers, IBGDA')]):
    y = 40 + k * 150
    b += rect(20, y, 720, 130, 'plane', 12) + text(36, y + 26, f'{title}: {nodes}', 't-note', 'start')
    n = 4 if k == 0 else 40
    for i in range(min(n, 40)):
        cx = 40 + (i % 20) * 16 if k else 40 + i * 70
        cy = y + 44 + (i // 20) * 14 if k else y + 44
        b += rect(cx, cy, 12 if k else 60, 10, 'box-1', 2)
    b += text(36, y + 92, attn, 't-tick', 'start') + text(36, y + 110, moe, 't-tick', 'start')
    b += text(724, y + 26, note, 't-tick', 'end')
b += text(380, 352, 'Prefill and decode run on separate units: disaggregation plus very wide expert parallelism.', 't-tick')
f['dsv3'] = svg(364, b, 'DeepSeek-V3 deployment from its technical report. Prefill units of 4 nodes (32 GPUs) use TP4 with DP8 for attention and EP32 for experts. Decode units of 40 nodes (320 GPUs) use TP4 with DP80 for attention and EP320, one expert per GPU.')

# 20. Interference timeline: colocated vs disaggregated (from the cost model).
b = text(20, 22, 'What one user sees while a 2,000-token prompt arrives (cost model of disagg_sim.py)', 't-title', 'start')
ce = DIS['cost_examples']
dec, pre, mixed = ce['decode_b32_ms'] / 1e3, ce['prefill_2000_ms'] / 1e3, ce['chunk_b32_512_ms'] / 1e3
sc = 2900
def tl(y, label, segs):
    out = text(20, y + 16, label, 't-note', 'start')
    x = 250
    for dur, cls, lab in segs:
        w = dur * sc
        out += rect(x, y, w - 2, 24, cls, 3)
        if lab:
            out += text(x + w / 2, y + 16, lab, 't-cell')
        x += w
    return out
b += tl(44, 'colocated, prefill first', [(dec, 'box-1', 'tok'), (dec, 'box-1', 'tok'), (pre, 'waste', f'prefill {pre*1e3:.0f} ms: no tokens'), (dec, 'box-1', 'tok'), (dec, 'box-1', 'tok')])
b += tl(90, 'colocated, chunked (512)', [(dec, 'box-1', 'tok'), (dec, 'box-1', 'tok')] + [(mixed, 'box-2', '')] * 4 + [(dec, 'box-1', '')])
b += tl(136, 'disaggregated: decode GPU', [(dec, 'box-1', 'tok')] * 7)
b += text(250, 186, f'decode step {dec*1e3:.1f} ms (batch 32);  with a 512-token chunk {mixed*1e3:.1f} ms;  whole prefill {pre*1e3:.0f} ms', 't-tick', 'start')
b += text(380, 210, 'The prompt belongs to another user; ours is one of the 32 sequences already decoding.', 't-tick')
f['interference'] = svg(222, b, 'Token timeline for a user who is decoding while another user\'s 2,000-token prompt arrives. Prefill-first stalls the user for the whole prefill; chunked prefill slows several steps a little; on a dedicated decode GPU the steps are unaffected.')

# 21. KV transfer time vs prefill time (8,192-token prompt).
rowsK = [r for r in KV['rows'] if r['tokens'] == 8192]
linksK = ['NVLink (same node)', '8 x 400G NICs (TP=8 to TP=8)', '1 x 400G NIC', '100G Ethernet']
X0, X1 = 260, 690
lo, hi = 1, 1000
b = ''
for v in [1, 10, 100, 1000]:
    x = logscale(v, lo, hi, X0, X1)
    b += line(x, 26, x, 26 + 3 * 120, 'grid') + text(x, 26 + 3 * 120 + 16, f'{v:,} ms', 't-tick')
for i, r in enumerate(rowsK):
    y = 30 + i * 120
    b += text(20, y + 10, f"{r['model']}: {r['kv_bytes']/GiB:.2f} GiB of KV", 't-note', 'start')
    items = [('prefill itself (~)', r['prefill_ms'], 's3')] + [(ln.replace(' (TP=8 to TP=8)', ''), r[ln]['transfer_ms'], 's1') for ln in linksK]
    for j, (lab, v, cls) in enumerate(items):
        yy = y + 18 + j * 19
        b += text(X0 - 8, yy + 12, lab, 't-tick', 'end')
        b += mark(f'{lab}: {v:.1f} ms', rect(X0, yy, max(2, logscale(max(v, 1), lo, hi, X0, X1) - X0), 15, cls, 3))
        b += text(logscale(max(v, 1), lo, hi, X0, X1) + 6, yy + 12, f'{v:.1f}', 't-tick', 'start')
b += text(380, 26 + 3 * 120 + 40, 'Arithmetic: BF16 cache, peak link bandwidth, prefill at 50% of H100 peak FLOP/s. Log scale.', 't-tick')
f['kv_transfer'] = svg(26 + 3 * 120 + 52, b, 'Time to move the KV cache of one 8,192-token prompt compared with the time to prefill it. Over NVLink or one NIC per GPU it is a few milliseconds; over one 400 Gb/s card it is 12 to 54 ms; over 100 Gb/s Ethernet 46 to 215 ms.')

# 22. Serialized vs layer-wise transfer (Splitwise idea), Llama 70B 8K over one NIC.
r70 = [r for r in rowsK if '70B' in r['model']][0]
pf, xf = r70['prefill_ms'], r70['1 x 400G NIC']['transfer_ms']
sc = 470 / (pf + xf + 30)
b = text(20, 22, f'Llama 3.1 70B, 8,192-token prompt, KV over one 400 Gb/s NIC ({xf:.0f} ms of transfer)', 't-title', 'start')
b += text(20, 62, 'send at the end', 't-note', 'start') + rect(170, 46, pf * sc, 24, 'box-2', 3) + text(170 + pf * sc / 2, 63, f'prefill {pf:.0f} ms', 't-cell')
b += rect(170 + pf * sc, 46, xf * sc, 24, 'waste', 3) + text(170 + pf * sc + xf * sc / 2, 63, f'+{xf:.0f}', 't-cell')
b += rect(170 + (pf + xf) * sc, 46, 22, 24, 'box-1', 3) + text(170 + (pf + xf) * sc + 26, 63, '2nd token', 't-tick', 'start')
b += text(20, 112, 'layer by layer', 't-note', 'start') + rect(170, 96, pf * sc, 24, 'box-2', 3) + text(170 + pf * sc / 2, 113, f'prefill {pf:.0f} ms', 't-cell')
for i in range(10):
    x = 170 + (i + 1) * pf * sc / 10
    b += rect(x - xf * sc / 80, 124, xf * sc / 80 * 8 / 10, 10, 's1', 2)
ex = r70['1 x 400G NIC']['exposed_layerwise_ms']
b += rect(170 + pf * sc + ex * sc, 96, 22, 24, 'box-1', 3) + text(170 + pf * sc + ex * sc + 26, 113, f'2nd token (+{ex:.1f} ms)', 't-tick', 'start')
b += text(380, 160, 'Each layer\'s KV leaves while the next layers compute, so only the last layer\'s share is left at the end.', 't-tick')
f['layerwise'] = svg(174, b, 'Sending the KV cache only after prefill adds 54 ms before the second token for a 70B model over one 400 Gb/s card. Sending each layer as soon as it is ready hides almost all of it.')

# 23. Goodput: SLO attainment vs rate, two panels.
series = [('colocated, prefill first (4 GPUs)', 'l3', 'prefill first'), ('colocated, chunked prefill (4 GPUs)', 'l1', 'chunked'),
          ('disaggregated 2P + 2D', 'l2', '2P + 2D'), ('disaggregated 3P + 1D', 'l4', '3P + 1D')]
b = ''
for k, (kind, key, title) in enumerate([('chat', 'tight (TTFT 0.5 s, TPOT 15 ms)', 'chat, tight SLO (0.5 s, 15 ms)'),
                                        ('long', 'strict (TTFT 1 s, TPOT 12 ms)', 'long prompts, strict SLO (1 s, 12 ms)')]):
    X0 = 60 + k * 360; X1 = X0 + 290; Y0, Y1 = 250, 50
    sw0 = DIS[kind][series[0][0]]['sweep']
    rates = [x['rate'] for x in sw0]
    xs = lambda r, X0=X0, X1=X1, rates=rates: X0 + (X1 - X0) * (r - rates[0]) / (rates[-1] - rates[0])
    ys = lambda v: Y0 - (Y0 - Y1) * v
    b += text((X0 + X1) / 2, 24, title, 't-note')
    for v in [0, 0.5, 0.9, 1.0]:
        b += line(X0, ys(v), X1, ys(v), 'grid' if v != 0.9 else 'base-line')
        if k == 0 or v == 0.9:
            b += text(X0 - 6, ys(v) + 4, f'{v:.0%}', 't-tick', 'end')
    step = 8 if kind == 'chat' else 2
    for r in range(rates[0] if kind == 'long' else 8, rates[-1] + 1, step):
        b += text(xs(r), Y0 + 18, r, 't-tick')
    b += text((X0 + X1) / 2, Y0 + 36, 'requests per second;  goodput (90% met):', 't-tick')
    for j, (name, cls, lab) in enumerate(series):
        sw = DIS[kind][name]['sweep']
        b += poly([(xs(x['rate']), ys(x['slo_attainment'][key])) for x in sw], cls)
        g = DIS[kind][name]['goodput_rps'][key]
        lx, ly = X0 + (j % 2) * 150, Y0 + 56 + (j // 2) * 18
        b += rect(lx, ly - 9, 10, 10, cls.replace('l', 's'), 2) + text(lx + 16, ly, f'{lab}: {g} req/s', 't-tick', 'start')
f['goodput'] = svg(340, b, 'Simulated share of requests meeting the SLO as load rises on 4 GPUs. Chat workload with a tight SLO: chunked prefill keeps 90% up to 28 requests per second, 2P+2D up to 26, prefill-first 16. Long prompts with a strict SLO: 2P+2D reaches 3 requests per second, chunked and 3P+1D 2, prefill-first 1.')

# 24. 70B on one node: TP8x1 vs TP4x2 latency vs throughput.
cv = LAY['node_70b']
X0, X1, Y0, Y1 = 90, 690, 250, 30
allp = [p for c in cv.values() for p in c['points']]
mxT = max(p['node_tokens_per_s'] for p in allp) * 1.08
mxL = max(p['latency_ms'] for p in allp) * 1.1
xs = lambda t: X0 + (X1 - X0) * t / mxT
ys = lambda l: Y0 - (Y0 - Y1) * l / mxL
b = ''
for t in range(0, int(mxT), 2000):
    b += line(xs(t), Y0, xs(t), Y1, 'grid') + text(xs(t), Y0 + 18, f'{t:,}', 't-tick')
for l in range(0, int(mxL), 5):
    b += text(X0 - 8, ys(l) + 4, f'{l} ms', 't-tick', 'end')
b += line(X0, Y0, X1, Y0, 'axis') + text((X0 + X1) / 2, Y0 + 36, 'tokens per second for the whole node', 't-tick')
for name, cls in [('TP=8 x 1', 'l1'), ('TP=4 x 2', 'l2'), ('TP=2 x 4', 'l3')]:
    pts = cv[name]['points']
    if not pts:
        continue
    b += poly([(xs(p['node_tokens_per_s']), ys(p['latency_ms'])) for p in pts], cls)
    for p in pts:
        b += mark(f"{name}, batch {p['batch_per_replica']} per replica: {p['latency_ms']:.1f} ms, {p['node_tokens_per_s']:,.0f} tok/s",
                  dot(xs(p['node_tokens_per_s']), ys(p['latency_ms']), cls.replace('l', 's'), 3.5))
    q = pts[-1]
    left = name == 'TP=2 x 4'
    b += text(xs(q['node_tokens_per_s']) + (10 if left else -6), ys(q['latency_ms']) + (4 if left else -10), f"{name} (max {cv[name]['max_batch']}/replica)", 't-tick', 'start' if left else 'end')
f['node70b'] = svg(Y0 + 48, b, 'Modelled latency against throughput for Llama 3.1 70B on one 8-GPU node. One TP=8 replica gives lower latency and, because it has room for more sequences, the highest throughput; two TP=4 replicas give more throughput at equal batch per replica but run out of KV space at 109 sequences each.')

# 25. Decision guide.
b = ''
b += rect(150, 4, 460, 30, 'box', 6) + text(380, 24, 'Do weights + KV for your traffic fit on one GPU?', 't-note')
b += line(300, 34, 160, 70, 'edge', True) + line(460, 34, 600, 70, 'edge', True)
b += rect(20, 72, 280, 52, 'box-1', 6) + text(160, 94, 'Yes: one GPU per replica', 't-note') + text(160, 114, 'add replicas for more load (Part 6)', 't-tick')
b += rect(460, 72, 280, 30, 'box', 6) + text(600, 92, 'No: does it fit on one node?', 't-note')
b += line(560, 102, 470, 140, 'edge', True) + line(640, 102, 690, 140, 'edge', True)
b += rect(320, 142, 300, 52, 'box-1', 6) + text(470, 164, 'Yes: tensor parallel inside the node', 't-note') + text(470, 184, 'compare TP sizes by KV room and latency', 't-tick')
b += rect(630, 142, 120, 52, 'box-2', 6) + text(690, 164, 'No: TP in node', 't-note') + text(690, 184, '+ PP across', 't-tick')
b += rect(20, 222, 720, 40, 'box-3', 6) + text(380, 247, 'Mixture of experts? Spread the experts with EP; run attention data-parallel.', 't-note')
b += rect(20, 274, 720, 40, 'box-3', 6) + text(380, 299, 'Long prompts, strict time per token, many GPUs? Split prefill and decode onto separate GPUs.', 't-note')
b += text(380, 340, 'Then measure: the right layout is the one that meets your SLO at the lowest cost per token.', 't-tick')
f['decision'] = svg(352, b, 'Decision guide: one GPU per replica if it fits; otherwise tensor parallel inside a node; tensor plus pipeline parallel across nodes; expert parallel for mixture-of-experts; disaggregation for long prompts with strict per-token latency at scale.')

(R / 'results/figures.json').write_text(json.dumps(f, indent=2) + '\n')
print('Wrote', len(f), 'figures')
