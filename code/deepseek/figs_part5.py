"""Part 5 figures (SVG, CSS classes only) -> results/figs_part5.json.
Data comes from results/part5_cost.json, part5_fig3.json and part5_indexer.json."""
import json, os, random
from figlib import svg, text, box, arrow, lines_chart, esc
from common import RESULTS

R = lambda n: json.load(open(os.path.join(RESULTS, n + '.json')))
cost, f3, toy = R('part5_cost'), R('part5_fig3'), R('part5_indexer')
F = {}


def line(x1, y1, x2, y2, cls='edge'):
    return f'<line class="{cls}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>'


def hbars(w, label, rows, unit_fmt, xmax, left=210, bar_h=26, gap=14, title=None, note=None):
    """rows: (name, value, class). Horizontal bars with the value printed at the end."""
    top = 40 if title else 14
    h = top + len(rows) * (bar_h + gap) + (34 if note else 10)
    b = [text(16, 24, title, 't-title')] if title else []
    pw = w - left - 90
    for i, (name, v, cls) in enumerate(rows):
        y = top + i * (bar_h + gap)
        b.append(text(left - 10, y + bar_h / 2 + 5, name, 't-note', 'end'))
        b.append(f'<g class="mark"><title>{esc(name)}: {esc(unit_fmt(v))}</title>'
                 f'<rect class="{cls}" x="{left}" y="{y}" width="{max(2, pw * v / xmax):.1f}" height="{bar_h}" rx="4"/></g>')
        b.append(text(left + pw * v / xmax + 8, y + bar_h / 2 + 5, unit_fmt(v), 't-val'))
    if note:
        b.append(text(16, h - 12, note, 't-muted'))
    return svg(w, h, label, b)


# 1. Attention patterns: which (query, key) pairs get computed --------------------------------------------
def grid(x0, y0, n, cell, on, title, sub):
    b = [text(x0 + n * cell / 2, y0 - 26, title, 't-title', 'middle'), text(x0 + n * cell / 2, y0 - 9, sub, 't-muted', 'middle')]
    for i in range(n):
        for j in range(n):
            x, y = x0 + j * cell, y0 + i * cell
            if j > i:
                continue
            cls = 'cell' if on(i, j) else 'cell-masked'
            b.append(f'<rect class="{cls}" x="{x + 1:.1f}" y="{y + 1:.1f}" width="{cell - 2:.1f}" height="{cell - 2:.1f}" rx="2"/>')
    return b


rnd = random.Random(7)
N, C = 16, 9.5
dsa_pick = {}
for i in range(N):
    cand = list(range(i + 1))
    rnd.shuffle(cand)
    dsa_pick[i] = set(cand[:5])                                  # top-5 per query, scattered tokens
nsa_blocks = {i: {0, i // 4} | ({rnd.randrange(0, i // 4 + 1)} if i >= 4 else set()) for i in range(N)}
b = []
b += grid(20, 60, N, C, lambda i, j: True, 'Dense', 'every earlier token')
b += grid(195, 60, N, C, lambda i, j: i - j < 4, 'Sliding window', 'last 4 tokens')
b += grid(370, 60, N, C, lambda i, j: (j // 4) in nsa_blocks[i] or i - j < 2, 'NSA selection', 'whole blocks of 4')
b += grid(545, 60, N, C, lambda i, j: j in dsa_pick[i], 'DSA top-k', 'any 5 single tokens')
b.append(text(20, 238, 'Rows are queries (newest at the bottom), columns are earlier tokens. Filled = computed.', 't-muted'))
F['patterns'] = svg(720, 250, 'Four attention patterns on 16 tokens: dense, sliding window, NSA block selection and DSA token selection', b)

# 2. Decode: FLOPs per new token -------------------------------------------------------------------------
xs = cost['ctx']
dec = cost['decode']
F['decode_flops'] = lines_chart(
    680, 330, 'Attention FLOPs for one new token versus context length, dense MLA versus DSA, with the expert and projection FLOPs for comparison',
    xs, [('Dense MLA', [d['dense'] for d in dec], 'l1', 's1'),
         ('DSA', [d['dsa'] for d in dec], 'l2', 's2'),
         ('Weights (37B active)', [d['weights'] for d in dec], 'l3', 's3')],
    'context length (tokens, log scale)', 'GFLOP per new token', 0, 2400, [0, 400, 800, 1200, 1600, 2000, 2400],
    xlog=True, fmt=lambda v: f'{v:,.0f}', right=200)

# 3. Decode: cache bytes read per new token ---------------------------------------------------------------
by = cost['bytes']
F['decode_bytes'] = lines_chart(
    680, 330, 'Cache bytes read for one new token of one sequence versus context, dense MLA versus DSA',
    xs, [('Dense, BF16 cache', [d['dense_bf16'] / 1000 for d in by], 'l3', 's3'),
         ('Dense, FP8 cache', [d['dense_fp8'] / 1000 for d in by], 'l1', 's1'),
         ('DSA', [d['dsa'] / 1000 for d in by], 'l2', 's2')],
    'context length (tokens, log scale)', 'GB read per new token', 0, 10, [0, 2, 4, 6, 8, 10],
    xlog=True, fmt=lambda v: f'{v:.2f}', right=200)

# 4. Prefill: whole-prompt attention compute ------------------------------------------------------------
pf = cost['prefill']
F['prefill_flops'] = lines_chart(
    680, 330, 'Attention compute for a whole prompt versus prompt length: dense MHA mode, DSA, and the weights',
    xs, [('Dense (MHA mode)', [d['mha'] / 1000 for d in pf], 'l1', 's1'),
         ('DSA', [d['dsa'] / 1000 for d in pf], 'l2', 's2'),
         ('Weights (37B active)', [d['weights'] / 1000 for d in pf], 'l3', 's3')],
    'prompt length (tokens, log scale)', 'PFLOP per prompt', 0, 45, [0, 10, 20, 30, 40],
    xlog=True, fmt=lambda v: f'{v:.1f}', right=200)

# 5. NSA: three branches -----------------------------------------------------------------------------------
b = []
b.append(text(20, 26, 'Native Sparse Attention: one query, three views of the past', 't-title'))
y = 50
b.append(text(20, y + 14, 'earlier tokens', 't-muted'))
for i in range(32):
    cls = 'box'
    if i // 4 in (1, 5):
        cls = 'box-2'
    if i >= 28:
        cls = 'box-3'
    b.append(box(120 + i * 17, y, 15, 22, cls, rx=3))
for k in range(8):
    b.append(line(120 + k * 68 + 1, y + 28, 120 + k * 68 + 65, y + 28, 'axis'))
b.append(text(120, y + 44, 'blocks of l = 32 tokens (drawn as 4)', 't-muted'))
cols = [(60, 'box-1', 'Compression', ['each block -> 1 key', 'by a small MLP', 'coarse view of all']),
        (270, 'box-2', 'Selection', ['top-n blocks, using', 'the compressed scores', 'fine view of a few']),
        (480, 'box-3', 'Sliding window', ['last w = 512', 'tokens', 'local view'])]
for x, cls, name, lines_ in cols:
    b.append(arrow(x + 90, 112, x + 90, 138))
    b.append(box(x, 140, 180, 92, cls))
    b.append(text(x + 90, 162, name, 't-note', 'middle'))
    for k, s in enumerate(lines_):
        b.append(text(x + 90, 184 + k * 16, s, 't-tick', 'middle'))
    b.append(arrow(x + 90, 234, 330, 268))
b.append(box(250, 270, 160, 44, 'box-on'))
b.append(text(330, 297, 'gated sum', 't-note', 'middle'))
b.append(text(430, 290, 'gates g in [0, 1] come from an MLP + sigmoid', 't-muted'))
b.append(text(430, 306, 'of the query token', 't-muted'))
F['nsa'] = svg(680, 330, 'NSA splits attention into compression, selection and sliding-window branches and mixes them with learned gates', b)

# 6. NSA keys read --------------------------------------------------------------------------------------
rows = []
for r in cost['nsa']:
    rows.append((f'{r["t"] // 1024}K full', r['t'], 'box-1'))
    rows.append((f'{r["t"] // 1024}K NSA', r['nsa'], 'box-2'))
F['nsa_reads'] = hbars(680, 'Keys read per decoding step, full attention versus NSA, from 8K to 64K tokens', rows,
                       lambda v: f'{v:,}', 65536, left=120, bar_h=18, gap=8,
                       note='NSA: t/16 compressed + 16 x 64 selected + 512 window keys (paper settings)')

# 7. The lightning indexer pipeline -------------------------------------------------------------------
b = [text(20, 26, 'One query token through DSA (one layer of DeepSeek-V3.2)', 't-title')]
b.append(box(20, 50, 150, 44, 'box')); b.append(text(95, 77, 'query token h_t', 't-note', 'middle'))
b.append(box(20, 190, 150, 44, 'box')); b.append(text(95, 210, 'every earlier', 't-note', 'middle')); b.append(text(95, 226, 'token h_s', 't-note', 'middle'))
b.append(arrow(172, 72, 218, 72)); b.append(arrow(172, 212, 218, 212))
b.append(box(220, 46, 190, 54, 'box-2')); b.append(text(315, 68, '64 queries q^I (128 dims)', 't-tick', 'middle')); b.append(text(315, 86, '+ 64 weights w', 't-tick', 'middle'))
b.append(box(220, 186, 190, 54, 'box-2')); b.append(text(315, 208, 'ONE key k^I (128 dims)', 't-tick', 'middle')); b.append(text(315, 226, 'FP8, kept in a cache', 't-tick', 'middle'))
b.append(arrow(412, 72, 452, 128)); b.append(arrow(412, 212, 452, 150))
b.append(box(454, 110, 210, 60, 'box-on'))
b.append(text(559, 134, 'I = sum_j w_j ReLU(q_j . k)', 't-tick', 'middle')); b.append(text(559, 154, 'one score per earlier token', 't-tick', 'middle'))
b.append(arrow(559, 172, 559, 268))
b.append(box(454, 270, 210, 44, 'box-3')); b.append(text(559, 297, 'keep the top 2,048', 't-note', 'middle'))
b.append(arrow(452, 292, 412, 292))
b.append(box(170, 268, 240, 76, 'box-1'))
b.append(text(290, 290, 'MLA core attention, MQA mode', 't-note', 'middle'))
b.append(text(290, 310, '128 heads read the same', 't-tick', 'middle'))
b.append(text(290, 327, '2,048 latents (656 bytes each)', 't-tick', 'middle'))
b.append(arrow(168, 306, 120, 306)); b.append(text(20, 311, 'output', 't-note'))
b.append(text(20, 372, 'Top: cheap, but touches every earlier token. Bottom left: expensive, but touches only 2,048.', 't-muted'))
F['indexer'] = svg(680, 384, 'The lightning indexer pipeline: small FP8 queries and one key per token give a score per earlier token, the top 2,048 go to MLA', b)

# 8. Where DSA sits in the V3.2 block ---------------------------------------------------------------
b = [text(20, 26, 'One DeepSeek-V3.2 layer (x 61). Only the box at the bottom is new.', 't-title')]
xs_ = [(20, 'box', 'RMSNorm', ''), (140, 'box-1', 'MLA attention', 'same as V3'), (330, 'box', 'RMSNorm', ''), (450, 'box-1', 'MoE FFN', '1 shared + 8 of 256')]
for x, cls, a1, a2 in xs_:
    w = 170 if 'MoE' in a1 or 'MLA' in a1 else 100
    b.append(box(x, 70, w, 56, cls)); b.append(text(x + w / 2, 94, a1, 't-note', 'middle'))
    if a2:
        b.append(text(x + w / 2, 113, a2, 't-tick', 'middle'))
b.append(arrow(122, 98, 138, 98)); b.append(arrow(312, 98, 328, 98)); b.append(arrow(432, 98, 448, 98))
b.append(box(140, 160, 170, 64, 'box-on'))
b.append(text(225, 184, 'lightning indexer', 't-note', 'middle'))
b.append(text(225, 204, '+ top-k selector', 't-note', 'middle'))
b.append(arrow(225, 158, 225, 130))
b.append(text(330, 180, 'tells MLA which 2,048 cached latents', 't-tick'))
b.append(text(330, 198, 'to read; about 14M weights per layer,', 't-tick'))
b.append(text(330, 216, 'about 0.85B in all (on top of 671B)', 't-tick'))
b.append(text(20, 254, 'Experts, router, MTP, tokenizer and the 128K context are unchanged from V3.1-Terminus.', 't-muted'))
F['block'] = svg(680, 266, 'A DeepSeek-V3.2 layer: RMSNorm, MLA attention with the lightning indexer and top-k selector attached, RMSNorm, MoE', b)

# 9. Training stages --------------------------------------------------------------------------------
b = [text(20, 26, 'From V3.1-Terminus to V3.2: continued training with sparse attention', 't-title')]
stages = [(20, 'box', 'V3.1-Terminus', ['dense MLA', '128K context']),
          (190, 'box-2', '1. Dense warm-up', ['train ONLY the indexer', 'KL to head-summed attention', '1,000 steps, 2.1B tokens']),
          (370, 'box-1', '2. Sparse training', ['train everything, top-2,048', 'indexer: KL on chosen set', '15,000 steps, 943.7B tokens']),
          (550, 'box-3', '3. Post-training', ['distillation + RL', 'with DSA on'])]
for x, cls, name, ls in stages:
    w = 160 if x in (190, 370) else 130
    b.append(box(x, 50, w, 104, cls)); b.append(text(x + w / 2, 74, name, 't-note', 'middle'))
    for k, s in enumerate(ls):
        b.append(text(x + w / 2, 98 + k * 18, s, 't-tick', 'middle'))
b.append(arrow(152, 102, 188, 102)); b.append(arrow(352, 102, 368, 102)); b.append(arrow(532, 102, 548, 102))
tot = 2.1 + 943.7
b.append(text(20, 186, 'Tokens per stage, to scale:', 't-muted'))
b.append(f'<rect class="s2" x="200" y="174" width="{max(2, 460 * 2.1 / tot):.1f}" height="16" rx="2"/>')
b.append(f'<rect class="s1" x="{200 + 460 * 2.1 / tot + 2:.1f}" y="174" width="{460 * 943.7 / tot:.1f}" height="16" rx="2"/>')
b.append(text(200, 210, 'warm-up 0.2%', 't-tick')); b.append(text(660, 210, 'sparse stage 99.8%', 't-tick', 'end'))
F['stages'] = svg(700, 222, 'The training path of DSA: V3.1-Terminus, a dense warm-up of the indexer, sparse training of the whole model, then post-training', b)

# 10. Figure 3 numbers, as ratios --------------------------------------------------------------------
pos = [8192, 16384, 32768, 65536, 98304, 131072]
ratio = lambda part: [f3[part]['at'][str(x)][0] / f3[part]['at'][str(x)][1] for x in pos]
F['fig3_ratio'] = lines_chart(
    680, 320, 'How many times cheaper V3.2 is than V3.1-Terminus per token, by token position, read from the vector data of the paper Figure 3',
    pos, [('Decode', ratio('decode'), 'l1', 's1'), ('Prefill', ratio('prefill'), 'l2', 's2')],
    'token position', 'Terminus cost / V3.2 cost', 0, 10, [0, 2, 4, 6, 8, 10],
    xticks=[8192, 32768, 65536, 98304, 131072], fmt=lambda v: f'{v:.1f}x', right=150)

# 11. KV cache bytes per token -------------------------------------------------------------------------
kv = cost['kv']
F['kv'] = hbars(680, 'KV cache bytes per token: V3 in BF16, V3 in the FP8 layout, V3.2 with the indexer keys', [
    ('V3, BF16 latent', kv['v3_bf16'], 'box-3'),
    ('V3 / V3.1, FP8 layout', kv['v3_fp8'], 'box-1'),
    ('V3.2, FP8 + indexer key', kv['v32'], 'box-2')],
    lambda v: f'{v:,} B', 75000, note='61 layers. Indexer key: 128 FP8 bytes + 4-byte scale per layer = +20%.')

# 12. Toy experiment: attention kept versus k ----------------------------------------------------------
sw = toy['sweep']
F['toy'] = lines_chart(
    680, 320, 'Share of the true attention kept by the selected tokens, versus k, for the trained indexer, the oracle and random choice',
    [r['k'] for r in sw], [('Oracle (true top-k)', [100 * r['oracle'] for r in sw], 'l3', 's3'),
                           ('Trained indexer', [100 * r['trained'] for r in sw], 'l2', 's2'),
                           ('Random tokens', [100 * r['random'] for r in sw], 'l1', 's1')],
    'k, tokens kept out of 4,096 (log scale)', '% of attention kept', 0, 100, [0, 20, 40, 60, 80, 100],
    xlog=True, fmt=lambda v: f'{v:.0f}%', right=190)

# 13. API prices -----------------------------------------------------------------------------------------
rows = [('Input, cache hit: V3.1', 0.07, 'box-1'), ('V3.2-Exp', 0.028, 'box-2'),
        ('Input, cache miss: V3.1', 0.56, 'box-1'), ('V3.2-Exp', 0.28, 'box-2'),
        ('Output: V3.1', 1.68, 'box-1'), ('V3.2-Exp', 0.42, 'box-2')]
F['prices'] = hbars(680, 'DeepSeek API prices per million tokens before and after the V3.2-Exp launch on 29 September 2025', rows,
                    lambda v: f'${v:.3f}'.rstrip('0').rstrip('.') if v < 0.1 else f'${v:.2f}', 1.8, left=200, bar_h=20, gap=8,
                    note='USD per 1M tokens, from the archived pricing pages (Sept 2025 vs Oct 2025).')

json.dump(F, open(os.path.join(RESULTS, 'figs_part5.json'), 'w'))
print(f'wrote {len(F)} figures:', ', '.join(F))
