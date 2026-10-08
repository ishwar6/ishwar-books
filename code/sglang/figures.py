"""Theme-aware SVG figures for Part 5, drawn from saved results. Classes (box-1, s1, l1, t-note, ...) are
styled by the site's CSS, so every figure follows the dark and light themes.
Run: python code/sglang/figures.py  -> results/figures.json (assemble.py inlines them)."""
import json, math
from pathlib import Path
from html import escape
R = Path(__file__).resolve().parent
D = json.loads((R / 'results/experiments.json').read_text())
M = json.loads((R / 'results/measure_prefix.json').read_text())
S = json.loads((R / 'results/simulate.json').read_text())

DEFS = ('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path class="arrow" d="M0,0L10,5L0,10z"/></marker></defs>')
def text(x, y, s, cls='t-note', anchor='middle', extra=''):
    return f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{extra}>{escape(str(s))}</text>'
def rect(x, y, w, h, cls='box-1', rx=8):
    return f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}"/>'
def line(x, y, a, b, cls='edge', arrow=False):
    tip = ' marker-end="url(#ah)"' if arrow else ''
    return f'<line class="{cls}" x1="{x:.1f}" y1="{y:.1f}" x2="{a:.1f}" y2="{b:.1f}"{tip}/>'
def mark(title, body):
    return f'<g class="mark"><title>{escape(title)}</title>{body}</g>'
def svg(h, body, label):
    return f'<svg viewBox="0 0 760 {h}" role="img" aria-label="{escape(label)}">{DEFS}{body}</svg>'
f = {}

# 1. One request through the engine (kept from the first version).
b = text(30, 25, 'INPUT', cls='t-tick', anchor='start') + rect(20, 40, 720, 52) + text(380, 73, 'Instructions + handbook + question')
b += line(380, 92, 380, 118) + rect(100, 118, 560, 60, 'box-2') + text(380, 144, 'PREFILL: process the supplied text') + text(380, 165, 'Build KV state and select the first answer token', 't-tick')
b += line(380, 178, 380, 204) + rect(100, 204, 560, 60, 'box-1') + text(380, 230, 'DECODE: continue the answer') + text(380, 251, 'Read saved KV, generate the next token, repeat', 't-tick')
b += line(380, 264, 380, 291) + text(380, 315, 'OUTPUT: the answer streams back to the user')
f['request'] = svg(345, b, 'Instructions, handbook and question enter prefill, which builds KV and predicts the first output token. Decode continues the answer.')

# 2. KV bytes per token as a product of five factors (Llama 3.1 8B), with the handbook total.
kv = S['kv_bytes']
facs = [('2', 'K and V'), ('32', 'layers'), ('8', 'KV heads'), ('128', 'numbers/head'), ('2', 'bytes (BF16)')]
b = text(20, 24, 'Llama 3.1 8B: bytes of KV cache for one token', 't-title', 'start')
x = 22
for i, (v, lab) in enumerate(facs):
    b += rect(x, 44, 92, 64, 'box-1' if i else 'box-2') + text(x + 46, 74, v, 't-title') + text(x + 46, 96, lab, 't-tick')
    x += 92
    b += text(x + 14, 82, '×' if i < 4 else '=', 't-title')
    x += 28
b += rect(x, 44, 118, 64, 'box-3') + text(x + 59, 74, '131,072 B', 't-title') + text(x + 59, 96, '= 128 KiB', 't-tick')
q = kv['Qwen2.5-0.5B']; l8 = kv['Llama 3.1 8B']; q7 = kv['Qwen2.5-7B']
rows = [('Llama 3.1 8B', l8), ('Qwen2.5-7B', q7), ('Qwen2.5-0.5B', q)]
b += text(20, 146, 'A 6,000-token handbook, stored once:', 't-note', 'start')
mx = max(r['handbook_6000_tokens_MiB'] for _, r in rows)
for i, (name, r) in enumerate(rows):
    y = 162 + i * 34; w = 430 * r['handbook_6000_tokens_MiB'] / mx
    b += text(150, y + 17, name, 't-note', 'end')
    b += mark(f"{name}: {r['bytes_per_token']//1024} KiB per token, {r['handbook_6000_tokens_MiB']:.0f} MiB per handbook",
              rect(162, y + 3, w, 20, 's1', 4))
    b += text(170 + w, y + 18, f"{r['handbook_6000_tokens_MiB']:.0f} MiB  ({r['bytes_per_token']//1024} KiB/token)", 't-tick', 'start')
f['kv_factors'] = svg(272, b, 'KV bytes per token for Llama 3.1 8B are 2 times 32 times 8 times 128 times 2, which is 128 KiB. A 6,000-token handbook takes 750 MiB on Llama 3.1 8B, 328 MiB on Qwen2.5-7B and 70 MiB on Qwen2.5-0.5B.')

# 3. Measured cold vs warm prefill on a real model.
rows = M['prefix_sweep']; mx = max(r['cold_ms'] for r in rows)
b = rect(170, 12, 12, 12, 's2', 3) + text(188, 23, 'cold: prefill handbook + question', 't-tick', 'start')
b += rect(430, 12, 12, 12, 's1', 3) + text(448, 23, 'warm: handbook cached, prefill the question', 't-tick', 'start')
for i, r in enumerate(rows):
    y = 44 + i * 52
    b += text(150, y + 22, f"P = {r['P']:,}", 't-note', 'end')
    wc, ww = 440 * r['cold_ms'] / mx, 440 * r['warm_ms'] / mx
    b += mark(f"P={r['P']}, S=64: cold {r['cold_ms']} ms", rect(162, y, wc, 17, 's2', 4))
    b += text(168 + wc, y + 13, f"{r['cold_ms']:.0f} ms", 't-tick', 'start')
    b += mark(f"P={r['P']}, S=64: warm {r['warm_ms']} ms", rect(162, y + 20, ww, 17, 's1', 4))
    b += text(168 + ww, y + 33, f"{r['warm_ms']:.0f} ms", 't-tick', 'start')
    b += text(745, y + 24, f"{r['speedup']:.1f}x", 't-val', 'end')
b += line(162, 40, 162, 44 + 5 * 52 - 10, 'axis')
b += text(380, 44 + 5 * 52 + 14, 'Qwen2.5-0.5B, BF16, Apple M5 Pro, PyTorch. S = 64 new tokens. Median of 9 runs.', 't-tick')
f['measured'] = svg(44 + 5 * 52 + 26, b, 'Measured prefill time, cold versus warm, for shared prefixes of 512 to 8,192 tokens. The speedup grows from 2.1x to 20.7x.')

# 4. Radix tree after two requests (kept).
b = text(25, 25, '1. After the first request', anchor='start')
b += rect(90, 48, 580, 50) + text(380, 79, '10 11 12 13 14 15 20 21')
b += text(380, 123, 'One stored path: eight computed token positions', 't-tick')
b += text(25, 177, '2. The second request matches six IDs, then branches', anchor='start')
b += rect(210, 198, 340, 52) + text(380, 230, '10 11 12 13 14 15')
b += line(380, 250, 190, 293) + line(380, 250, 570, 293)
b += rect(90, 293, 200, 48, 'box-1') + text(190, 323, '20 21')
b += rect(470, 293, 200, 48, 'box-2') + text(570, 323, '30 31')
b += text(190, 367, 'First question: already stored', 't-tick') + text(570, 367, 'Second question: compute now', 't-tick')
b += text(380, 405, 'The shared path refers to one reusable set of KV states.', 't-tick')
f['radix_steps'] = svg(430, b, 'First request stores eight tokens. The second shares token IDs 10 through 15 and splits into question branches 20 21 and 30 31.')

# 5. LRU eviction: three panels from the saved trace (capacity 10 tokens).
def panel(x0, title, sub, kids):
    out = text(x0 + 118, 24, title, 't-note') + text(x0 + 118, 44, sub, 't-tick')
    out += rect(x0 + 48, 62, 140, 36, 'box') + text(x0 + 118, 85, '1 2 3 4 5 6', 't-note')
    for j, (lab, cls, note) in enumerate(kids):
        cx = x0 + 118 + (j - (len(kids) - 1) / 2) * 78
        out += line(x0 + 118, 98, cx, 140)
        out += rect(cx - 33, 140, 66, 34, cls) + text(cx, 162, lab, 't-note')
        out += text(cx, 194, note, 't-tick')
    return out
b = panel(0, 'q1, q2 stored', '10 of 10 slots used', [('20 21', 'box-1', 'used: step 1'), ('30 31', 'box-1', 'used: step 2')])
b += panel(254, 'q3 arrives', 'evict the LRU leaf', [('20 21', 'waste', 'evicted'), ('30 31', 'box-1', 'kept'), ('40 41', 'box-2', 'new')])
b += panel(508, 'q2 hits, then q1 returns', 'evict the new LRU leaf', [('30 31', 'box-1', 'hit: 8 of 8'), ('40 41', 'waste', 'evicted'), ('20 21', 'box-2', 'recomputed')])
b += line(250, 14, 250, 200, 'drop') + line(504, 14, 504, 200, 'drop')
b += text(380, 228, 'The shared handbook node is never a leaf while a question hangs below it, so it survives every eviction.', 't-tick')
f['lru'] = svg(244, b, 'Leaf-first LRU eviction with room for 10 tokens. The third question evicts the oldest question branch; the handbook node stays.')

# 6. Memory: no sharing vs radix tree.
sh = S['sharing']
b = ''
rows = [('no sharing', sh['naive_tokens'], 's2', '8.25 GiB on Llama 3.1 8B'), ('radix tree', sh['tree_tokens'], 's1', '1.06 GiB on Llama 3.1 8B')]
for i, (lab, v, cls, gib) in enumerate(rows):
    y = 20 + i * 64; w = 470 * v / sh['naive_tokens']
    b += text(140, y + 22, lab, 't-note', 'end') + mark(f'{lab}: {v:,} token slots', rect(152, y + 6, w, 24, cls, 4))
    b += text(160 + w, y + 23, f'{v:,} slots', 't-val', 'start') + text(152, y + 48, gib, 't-tick', 'start')
b += text(380, 162, '32 active requests: 512 instruction + 1,536 handbook (one of 4) + 64 question tokens each.', 't-tick')
b += text(380, 182, f"Tree = 512 + 4 × 1,536 + 32 × 64 = {sh['formula_tree']:,} slots, {sh['ratio']:.1f}x fewer.", 't-tick')
f['sharing'] = svg(196, b, 'Thirty-two requests need 67,584 KV slots without sharing and 8,704 in a radix tree, 7.8 times fewer.')

# 7. Token granularity: page 1 radix vs 4-token blocks on request 2.
toks = [10, 11, 12, 13, 14, 15, 30, 31]
b = ''
for row, (lab, kinds, res) in enumerate([
        ('radix, page 1', ['r'] * 6 + ['c'] * 2, 'reuse 6, compute 2'),
        ('blocks of 4', ['r'] * 4 + ['l', 'l'] + ['c'] * 2, 'reuse 4, compute 4')]):
    y = 28 + row * 92
    b += text(20, y + 26, lab, 't-note', 'start')
    for i, (t, k) in enumerate(zip(toks, kinds)):
        x = 170 + i * 58
        cls = {'r': 'box-1', 'c': 'box-2', 'l': 'waste'}[k]
        b += rect(x, y, 52, 40, cls, 6) + text(x + 26, y + 26, t, 't-note')
    b += text(170, y + 62, res, 't-tick', 'start')
    if row == 1:
        b += line(170, y - 8, 398, y - 8, 'axis') + line(402, y - 8, 630, y - 8, 'axis')
        b += text(284, y - 13, 'block 1: same', 't-tick') + text(516, y - 13, 'block 2: differs at 30', 't-tick')
b += rect(170, 214, 14, 12, 'box-1', 3) + text(190, 225, 'reused', 't-tick', 'start')
b += rect(270, 214, 14, 12, 'box-2', 3) + text(290, 225, 'computed', 't-tick', 'start')
b += rect(380, 214, 14, 12, 'waste', 3) + text(400, 225, 'matched, but its block is not full', 't-tick', 'start')
f['blocks'] = svg(240, b, 'With page size 1 the second request reuses six tokens. With four-token blocks it reuses four: tokens 14 and 15 match, but their block also holds 30 and 31, so the block hash differs.')

# 8. Capacity sweep: hit rate vs cache size.
cs = S['capacity_sweep']
X0, X1, Y0, Y1 = 90, 690, 272, 72
xs = lambda i: X0 + (X1 - X0) * i / (len(cs) - 1)
ys = lambda h: Y0 - (Y0 - Y1) * (h - 0.5) / 0.5
b = ''
for h in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    b += line(X0, ys(h), X1 + 20, ys(h), 'grid') + text(X0 - 10, ys(h) + 4, f'{h:.0%}', 't-tick', 'end')
for i, r in enumerate(cs):
    b += text(xs(i), Y0 + 22, f"{r['capacity_tokens']:,}", 't-tick')
b += text((X0 + X1) / 2, Y0 + 44, 'cache capacity, tokens (log scale)', 't-tick')
series = [('radix_page1', 'l1', 's1', 'radix, page 1, whole-leaf eviction'), ('radix_trim', 'l3', 's3', 'radix, trims leaf tails (ablation)'),
          ('block16', 'l2', 's2', 'blocks of 16, LRU'), ('block64', 'l4', 's4', 'blocks of 64, LRU')]
for k, (key, lc, sc, lab) in enumerate(series):
    pts = ' '.join(f'{xs(i):.1f},{ys(r[key]):.1f}' for i, r in enumerate(cs))
    b += f'<polyline class="{lc}" points="{pts}"/>'
    for i, r in enumerate(cs):
        b += mark(f"{lab}, {r['capacity_tokens']:,} tokens: {r[key]:.1%}", f'<circle class="{sc} ring" cx="{xs(i):.1f}" cy="{ys(r[key]):.1f}" r="5"/>')
    lx = 20 + (k % 2) * 370; ly = 14 + (k // 2) * 20
    b += f'<rect class="{sc}" x="{lx}" y="{ly}" width="12" height="12" rx="3"/>' + text(lx + 18, ly + 11, lab, 't-tick', 'start')
b += line(X0, Y0, X1 + 20, Y0, 'axis')
f['capacity'] = svg(332, b, 'Hit rate against cache capacity for four caches on the same 600-request trace. Blocks beat the whole-leaf radix cache at small capacity; trimming leaf tails closes the gap.')

# 9. 64-request work comparison (kept).
b = ''
rows = [('No reuse', 86016), ('Radix index', D['aligned']['radix']['computed_tokens']), ('Block-hash index', D['aligned']['blocks16']['computed_tokens'])]
for i, (label, val) in enumerate(rows):
    y = 25 + i * 65; b += text(180, y + 24, label, anchor='end') + rect(200, y, 450 * val / 86016, 36, 'box-2' if i == 0 else 'box-1') + text(210 + 450 * val / 86016, y + 24, f'{val:,}', anchor='start')
b += text(380, 235, 'Prompt token positions computed across 64 requests', 't-tick')
f['work'] = svg(255, b, 'Both teaching cache indexes compute 6,144 tokens versus 86,016 without reuse.')

# 10. Expected prefill time against hit rate (measured end points, P=2048, S=64).
r2 = next(r for r in M['prefix_sweep'] if r['P'] == 2048)
cold, warm = r2['cold_ms'], r2['warm_ms']
X0, X1, Y0, Y1 = 90, 700, 220, 30
xh = lambda h: X0 + (X1 - X0) * h
yt = lambda t: Y0 - (Y0 - Y1) * t / 110
b = ''
for t in [0, 25, 50, 75, 100]:
    b += line(X0, yt(t), X1, yt(t), 'grid') + text(X0 - 10, yt(t) + 4, f'{t}', 't-tick', 'end')
for h in [0, 0.25, 0.5, 0.75, 1]:
    b += text(xh(h), Y0 + 20, f'{h:.0%}', 't-tick')
b += text((X0 + X1) / 2, Y0 + 42, 'probability h that the 2,048-token handbook is still cached', 't-tick')
b += text(16, 18, 'expected prefill, ms', 't-tick', 'start')
b += f'<polyline class="l1" points="{xh(0):.1f},{yt(cold):.1f} {xh(1):.1f},{yt(warm):.1f}"/>'
for h in [0, 0.5, 0.9, 1]:
    t = (1 - h) * cold + h * warm
    b += mark(f'h = {h:.0%}: {t:.1f} ms', f'<circle class="s1 ring" cx="{xh(h):.1f}" cy="{yt(t):.1f}" r="6"/>')
    if h < 0.9:
        b += text(xh(h) + 10, yt(t) - 10, f'{t:.0f} ms', 't-val', 'start')
    elif h < 1:
        b += text(xh(h) - 4, yt(t) - 14, f'{t:.0f} ms', 't-val', 'end')
    else:
        b += text(xh(h) + 6, yt(t) - 14, f'{t:.0f} ms', 't-val', 'end')
b += line(X0, Y0, X1, Y0, 'axis')
f['hitrate'] = svg(276, b, 'Expected prefill time falls in a straight line from 100 ms with no hit to 13 ms with a certain hit; at a 50 percent hit rate it is 57 ms.')

# 11. Scheduling timelines: FCFS vs LPM, colour = handbook.
off = S['offline_scheduling']
tmax = off['fcfs']['total_ms']
X0, W = 70, 650
xt = lambda t: X0 + W * t / tmax
b = ''
for row, (pol, lab) in enumerate([('fcfs', 'FCFS'), ('lpm', 'LPM')]):
    y = 40 + row * 78
    r = off[pol]
    b += text(20, y + 22, lab, 't-note', 'start')
    for e in r['timeline']:
        x0, x1 = xt(e['start']), xt(e['end'])
        b += mark(f"{lab}: handbook {'ABCD'[e['book']]}, {e['end'] - e['start']:.0f} ms, {e['computed']:,} tokens computed",
                  f'<rect class="s{e["book"] + 1}" x="{x0:.2f}" y="{y}" width="{max(0.8, x1 - x0 - 0.8):.2f}" height="32" rx="2"/>')
    b += text(min(xt(r['total_ms']) + 8, 745), y + 52, f"all 32 done at {r['total_ms']:,.0f} ms · {r['computed_tokens']:,} tokens computed · hit rate {r['hit_rate']:.0%}",
              't-tick', 'end' if xt(r['total_ms']) > 500 else 'start')
for k, nm in enumerate('ABCD'):
    b += f'<rect class="s{k + 1}" x="{70 + k * 120}" y="14" width="12" height="12" rx="3"/>' + text(88 + k * 120, 25, f'handbook {nm}', 't-tick', 'start')
b += line(X0, 196, X0 + W, 196, 'axis') + text(X0, 214, '0 ms', 't-tick', 'start') + text(X0 + W, 214, f'{tmax:,.0f} ms', 't-tick', 'end')
f['schedule'] = svg(226, b, 'Thirty-two waiting requests served in arrival order switch handbooks every request and take 2,630 ms. Longest-prefix-match order groups them and finishes in 740 ms.')

# 12. Starvation: waits under FCFS and LPM.
st = S['starvation']
rows = [('FCFS: the D question', st['fcfs']['d_wait_ms'], 's2'), ('FCFS: A questions, mean', st['fcfs']['a_mean_ms'], 's1'),
        ('LPM: the D question', st['lpm']['d_wait_ms'], 's2'), ('LPM: A questions, mean', st['lpm']['a_mean_ms'], 's1')]
mx = max(v for _, v, _ in rows)
b = ''
for i, (lab, v, cls) in enumerate(rows):
    y = 16 + i * 40 + (12 if i >= 2 else 0); w = 430 * v / mx
    b += text(210, y + 17, lab, 't-note', 'end') + mark(f'{lab}: {v:.0f} ms', rect(222, y + 2, w, 22, cls, 4))
    b += text(230 + w, y + 18, f'{v:,.0f} ms', 't-val', 'start')
b += text(380, 200, '60 handbook-A questions arrive every 10 ms; one handbook-D question arrives at 25 ms.', 't-tick')
f['starvation'] = svg(214, b, 'Under FCFS the handbook-D question waits 187 ms. Under longest-prefix-match it waits 987 ms because cached A questions keep jumping ahead.')

# 13. Jump-forward: the same JSON, token by token vs with forced runs merged.
jf = S['jump_forward']
toks, kinds = jf['tokens_text'], jf['kinds']
show = lambda t: t.replace(' ', '·')
def width(t): return max(24, 8.2 * len(t) + 12)
b = text(20, 22, f"token by token: {jf['passes_normal']} forward passes", 't-note', 'start')
x, y = 20, 34
for t, k in zip(toks, kinds):
    w = width(show(t))
    if x + w > 744: x, y = 20, y + 40
    b += rect(x, y, w, 30, 'box-2' if k == 'forced' else 'box-1', 5) + text(x + w / 2, y + 20, show(t), 't-tick')
    x += w + 4
y2 = y + 66
b += text(20, y2 - 12, f"jump-forward: {jf['passes_jump']} forward passes (each forced run is one extend step)", 't-note', 'start')
groups, cur = [], None
for t, k in zip(toks, kinds):
    if k == 'forced' and cur and cur[1] == 'forced':
        cur[0] += t
    else:
        cur = [t, k]; groups.append(cur)
x, y = 20, y2
for t, k in groups:
    w = width(show(t))
    if x + w > 744: x, y = 20, y + 40
    b += rect(x, y, w, 30, 'box-2' if k == 'forced' else 'box-1', 5) + text(x + w / 2, y + 20, show(t), 't-tick')
    x += w + 4
y += 50
b += rect(20, y, 14, 12, 'box-2', 3) + text(40, y + 11, 'forced by the schema', 't-tick', 'start')
b += rect(210, y, 14, 12, 'box-1', 3) + text(230, y + 11, 'chosen by the model', 't-tick', 'start')
b += text(400, y + 11, 'Qwen2.5 tokenizer; · marks a space', 't-tick', 'start')
f['jump'] = svg(y + 24, b, 'The JSON answer is 22 tokens. Decoded one token at a time it needs 22 forward passes; merging the four forced runs needs 11.')

# 14. CPU scheduling overlapped with GPU work (kept).
b = ''
for row, base in enumerate([25, 130]):
    b += text(15, base - 7, 'Serial' if row == 0 else 'Overlap', anchor='start') + text(15, base + 18, 'CPU', cls='t-tick', anchor='start')
    for i in range(5):
        start = 110 + i * (100 if row == 0 else 80)
        b += rect(start, base, 20, 25, 'box-2') + rect(start + 20, base + 32, 80, 25, 'box-1')
    b += text(15, base + 48, 'GPU', cls='t-tick', anchor='start')
b += text(380, 240, 'Illustrative: CPU preparation 2 ms, GPU work 8 ms; each unit is 1 ms.', 't-tick')
b += text(380, 263, 'After startup, the CPU prepares the next batch while the GPU runs.', 't-tick')
f['overlap'] = svg(285, b, 'CPU scheduling overlaps GPU execution. Ideal steady state changes from 10 milliseconds to 8 milliseconds per batch.')

(R / 'results/figures.json').write_text(json.dumps(f, indent=2) + '\n')
print('Wrote', len(f), 'figures')
