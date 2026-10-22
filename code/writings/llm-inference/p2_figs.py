"""SVG figures for Part 2 of "LLM Inference from the Ground Up" (the KV cache).
Every figure uses the site's CSS classes only (no colours), so it follows the dark and light themes.
The numbers drawn here come from results/p2_kv_numbers.json, the published measurements of the series,
and the vLLM paper's run in code/papers/vllm/results/part2_stdout.txt.
Run: python p2_figs.py  ->  results/p2_figs.json"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'p2_kv_numbers.json')))

DEFS = ('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        'orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker>'
        '<marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
        'orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs>')


def svg(w, h, label, body):
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{label}">{DEFS}{"".join(body)}</svg>'


def T(x, y, s, cls='t-tick', anchor='start'):
    return f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{s}</text>'


def B(x, y, w, h, cls='box', rx=4):
    return f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}"/>'


def L(x1, y1, x2, y2, cls='edge', arrow=True, on=False):
    m = f' marker-end="url(#{"ah-on" if on else "ah"})"' if arrow else ''
    return f'<line class="{cls}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"{m}/>'


# slot kinds used by the memory rows
KIND = {'P': 'box-1', 'W': 'box-3', 'R': 'box-4', 'I': 'waste', '_': 'box-ghost', 'N': 'box-2', 'X': 'waste'}


def row(x, y, cells, w=26, h=30, gap=3, labels=None):
    out = []
    for i, c in enumerate(cells):
        out.append(B(x + i * (w + gap), y, w, h, KIND.get(c, c), 3))
        if labels and i < len(labels) and labels[i]:
            out.append(T(x + i * (w + gap) + w / 2, y + h / 2 + 4, labels[i], 't-tick', 'middle'))
    return out


def legend(x, y, items, step=None):
    out, cx = [], x
    for cls, text in items:
        out.append(B(cx, y - 11, 14, 14, cls, 3))
        out.append(T(cx + 20, y, text))
        cx += step or (34 + 7 * len(text))
    return out


def brace(x1, x2, y, text, cls='t-tick', up=False):
    d = -6 if up else 6
    return [f'<path class="edge" d="M{x1},{y} v{d} H{x2} v{-d}" style="stroke-width:1.3"/>',
            T((x1 + x2) / 2, y + (d * 3 if up else 20), text, cls, 'middle')]


F = {}

# ------------------------------------------------------------------ recap: one step serves a batch, each with its notes
b = [T(20, 26, 'one decode step, four conversations in the batch', 't-title')]
b += [B(250, 44, 260, 46, 'box-2', 10), T(380, 66, 'model weights', 't-title', 'middle'),
      T(380, 82, 'read from memory once per step', 't-tick', 'middle')]
for i, n in enumerate('ABCD'):
    x = 40 + i * 180
    b += [L(380, 92, x + 70, 138), B(x, 140, 140, 36, 'box-1', 8), T(x + 70, 163, f'conversation {n}', 't-title', 'middle'),
          B(x + 10, 186, 120, 26, 'box-3', 6), T(x + 70, 204, f'notes on {n}\'s past', 't-tick', 'middle')]
b += [T(20, 244, 'GPU memory', 't-note')]
xs = 120
for w, cls, lab in ((250, 'box-2', 'weights'), (70, 'box-3', 'A'), (110, 'box-3', 'B'), (50, 'box-3', 'C'),
                    (90, 'box-3', 'D'), (60, 'box-ghost', 'free')):
    b += [B(xs, 228, w, 26, cls, 4), T(xs + w / 2, 246, lab, 't-tick', 'middle')]
    xs += w + 3
b += [T(20, 284, 'Every conversation in the batch keeps its own notes in GPU memory, next to the weights.', 't-tick')]
F['recap'] = svg(760, 300, 'One decode step reads the weights once for four conversations; each conversation keeps its own notes in GPU memory.', b)

# ------------------------------------------------------------------ the attention lookup with tiny numbers, 3 panels
toy = R['toy']
K = [[0, 1], [3, 0], [1, 1]]
V = [[1, 0], [0, 4], [2, 2]]
names = ['The', 'cat', 'sat']
b = [T(20, 24, 'the new token "it" asks with its query q = [1, 0]', 't-title')]
b += [T(20, 54, 'step 1: compare q with every key', 't-note'), T(270, 54, 'step 2: turn scores into weights', 't-note'),
      T(520, 54, 'step 3: mix the values', 't-note')]
for i, (n, k, s, sc, wt, v) in enumerate(zip(names, K, toy['scores'], toy['scaled'], toy['weights'], V)):
    y = 78 + i * 62
    b += [B(20, y, 52, 30, 'box', 6), T(46, y + 20, n, 't-title', 'middle'),
          B(80, y, 76, 30, 'box-1', 6), T(118, y + 20, f'k = [{k[0]}, {k[1]}]', 't-tick', 'middle'),
          T(166, y + 20, f'q·k = {s:.0f}', 't-val')]
    b += [T(270, y + 20, f'{s:.0f} / √2 = {sc:.2f}', 't-tick')]
    wbar = 150 * wt
    b += [B(370, y + 4, 120, 22, 'box-ghost', 4), B(370, y + 4, max(wbar * 0.8, 2), 22, 'box-on', 4),
          T(370 + max(wbar * 0.8, 2) + 6, y + 20, f'{wt:.3f}', 't-val')]
    b += [B(520, y, 76, 30, 'box-3', 6), T(558, y + 20, f'v = [{v[0]}, {v[1]}]', 't-tick', 'middle'),
          T(606, y + 20, f'× {wt:.3f}', 't-tick')]
b += [T(270, 278, 'softmax: the weights sum to 1', 't-tick'),
      L(640, 270, 640, 262, arrow=False),
      B(520, 268, 220, 36, 'box-2', 8), T(630, 291, f'answer = [{toy["out"][0]:.2f}, {toy["out"][1]:.2f}]', 't-title', 'middle'),
      T(20, 326, 'The answer is mostly the value of "cat", because its key matched the query best.', 't-tick')]
F['lookup'] = svg(760, 340, 'A tiny attention lookup: the query is compared with three keys, the scores become weights 0.088, 0.734 and 0.178, and the answer is the weighted mix of the values.', b)

# ------------------------------------------------------------------ where q, k and v come from
b = [T(20, 24, 'one token at one layer', 't-title')]
b += [B(30, 110, 110, 40, 'box', 8), T(85, 135, 'vector of "cat"', 't-title', 'middle')]
for i, (m, out, cls, note) in enumerate((('W_q', 'query q', 'box-2', 'used now, then thrown away'),
                                         ('W_k', 'key k', 'box-1', 'kept: later tokens compare with it'),
                                         ('W_v', 'value v', 'box-3', 'kept: later tokens mix it in'))):
    y = 50 + i * 70
    b += [L(142, 130, 218, y + 18), B(220, y, 90, 36, 'box-ghost', 8), T(265, y + 23, f'× {m}', 't-mono', 'middle'),
          L(312, y + 18, 368, y + 18), B(370, y, 110, 36, cls, 8), T(425, y + 23, out, 't-title', 'middle'),
          T(496, y + 23, note, 't-tick')]
b += [T(20, 270, 'Three fixed weight matrices turn the same token vector into three different vectors.', 't-tick')]
F['qkv'] = svg(760, 286, 'Each token vector is multiplied by three weight matrices to give its query, key and value. The query is used once; the key and value are kept.', b)

# ------------------------------------------------------------------ what is stored for one token
b = [T(20, 24, 'what the cache keeps for ONE token of Llama 3.1 8B', 't-title')]
b += [B(20, 60, 90, 40, 'box', 8), T(65, 85, '"cat"', 't-title', 'middle')]
for i, lab in enumerate(['layer 1', 'layer 2', 'layer 3', '...', 'layer 32']):
    y = 44 + i * 40
    if lab == '...':
        b += [T(200, y + 22, '⋮', 't-title', 'middle')]
        continue
    b += [L(112, 80, 146, y + 16, arrow=False), B(150, y, 100, 32, 'box-ghost', 6), T(200, y + 21, lab, 't-tick', 'middle')]
b += [L(252, 60, 300, 60)]
b += [T(310, 50, 'inside one layer: 8 KV heads', 't-note')]
for h in range(8):
    x = 310 + h * 54
    b += [B(x, 62, 24, 24, 'box-1', 4), T(x + 12, 79, 'K', 't-tick', 'middle'),
          B(x + 26, 62, 24, 24, 'box-3', 4), T(x + 38, 79, 'V', 't-tick', 'middle')]
b += [T(310, 112, 'each K and each V is a list of 128 numbers (head_dim)', 't-tick'),
      T(310, 132, 'each number takes 2 bytes (bfloat16)', 't-tick')]
b += [B(310, 156, 430, 70, 'box-2', 10),
      T(525, 180, '2 (K and V) × 32 layers × 8 heads × 128 numbers × 2 bytes', 't-note', 'middle'),
      T(525, 206, '= 131,072 bytes = 128 KiB per token', 't-title', 'middle')]
b += [T(20, 262, 'One token leaves 512 small vectors behind: a key and a value for every head of every layer.', 't-tick')]
F['stored'] = svg(760, 278, 'For one token, Llama 3.1 8B stores a key and a value for each of 8 heads in each of 32 layers, each 128 numbers of 2 bytes: 128 KiB.', b)

# ------------------------------------------------------------------ an 80 GB H100 with Llama 3.1 8B
scale = 680 / 80
b = [T(20, 24, 'one 80 GB H100 serving Llama 3.1 8B, conversations at the full 128K tokens', 't-title')]
x = 40
for gb, cls, lab in ((16.1, 'box-2', 'weights 16.1 GB'), (17.2, 'box-3', 'conv. 1: 17.2 GB'),
                     (17.2, 'box-3', 'conv. 2: 17.2 GB'), (17.2, 'box-3', 'conv. 3: 17.2 GB'),
                     (80 - 16.1 - 3 * 17.2, 'box-ghost', '12.3 GB')):
    w = gb * scale
    b += [B(x, 50, w - 2, 44, cls, 6), T(x + w / 2, 77, lab, 't-tick', 'middle')]
    x += w
b += [T(40, 116, '0 GB', 't-tick'), T(720, 116, '80 GB', 't-tick', 'end'),
      T(40, 146, 'A fourth full conversation needs 17.2 GB and does not fit in the 12.3 GB left over.', 't-tick'),
      T(40, 166, '(The model also needs some working memory, so the real room is a little smaller.)', 't-tick')]
F['h100mem'] = svg(760, 180, 'An 80 GB H100 holds the 16.1 GB of Llama 3.1 8B weights and only three 128K-token conversations of 17.2 GB each.', b)

# ------------------------------------------------------------------ 16-bit vs 8-bit numbers
b = [T(20, 24, 'one key of 8 numbers, stored two ways', 't-title')]
vals = ['0.81', '-1.2', '0.05', '2.3', '-0.4', '1.1', '0.66', '-2.0']
b += [T(20, 66, 'bfloat16', 't-note')]
for i, v in enumerate(vals):
    b += [B(110 + i * 66, 46, 62, 30, 'box-1', 4), T(141 + i * 66, 66, v, 't-tick', 'middle')]
b += [T(650, 66, '16 bytes', 't-val')]
b += [T(20, 124, 'FP8', 't-note')]
for i, v in enumerate(vals):
    b += [B(110 + i * 36, 104, 32, 30, 'box-3', 4)]
b += [T(410, 124, '8 bytes: half the memory, slightly rounder numbers', 't-tick')]
b += [T(20, 168, 'Each number keeps fewer digits. The cache halves; quality usually changes very little, but measure it.', 't-tick')]
F['fp8'] = svg(760, 184, 'The same key stored with 2 bytes per number takes 16 bytes; with 1 byte per number (FP8) it takes 8 bytes.', b)

# ------------------------------------------------------------------ what one decode step reads (H100 floor)
h = R['h100']
b = [T(20, 24, 'the least time one decode step can take on an H100 (3.35 TB/s), Llama 3.1 8B', 't-title')]
sc = 60
rows = [('short chat (cache tiny)', [(h['weights_ms'], 'box-2', f'weights {h["weights_ms"]:.1f} ms')]),
        ('one 128K-token chat', [(h['weights_ms'], 'box-2', f'weights {h["weights_ms"]:.1f} ms'),
                                 (h['cache_ms'], 'box-3', f'its cache {h["cache_ms"]:.1f} ms')])]
for i, (lab, parts) in enumerate(rows):
    y = 52 + i * 56
    b += [T(200, y + 24, lab, 't-note', 'end')]
    x = 214
    for ms, cls, t in parts:
        b += [B(x, y, ms * sc - 2, 36, cls, 6), T(x + ms * sc / 2, y + 23, t, 't-tick', 'middle')]
        x += ms * sc
b += [T(20, 186, 'Reading a long cache costs as much time as reading the whole model again, on every step.', 't-tick')]
F['stepread'] = svg(760, 200, 'On an H100 a decode step must read the 16 GB of weights (4.8 ms) and, for one 128K-token chat, also its 17.2 GB cache (5.1 ms more).', b)

# ------------------------------------------------------------------ answer lengths for one question
lens = [2, 3, 3, 4, 5, 6, 15, 19, 23, 27, 28, 29, 35, 48, 73, 88, 151, 200, 200, 200]
b = [T(20, 24, '20 answers to "What is the capital of France?" (OPT-125M, random sampling, limit 200 tokens)', 't-title')]
for i, n in enumerate(lens):
    y = 40 + i * 15
    b += [B(60, y, n * 3.0, 11, 'box-3' if n < 200 else 'box-2', 2), T(60 + n * 3.0 + 6, y + 10, str(n), 't-tick')]
b += [T(60 + 200 * 3.0 + 36, 40 + 18 * 15 + 10, 'hit the limit', 't-tick'),
      T(60, 352, 'answer length in tokens (one bar per answer, sorted from short to long)', 't-tick')]
F['lengths'] = svg(760, 366, 'Twenty sampled answers to one question range from 2 to 200 tokens; three hit the 200-token limit.', b)

# ------------------------------------------------------------------ contiguous: address arithmetic and no room to grow
b = [T(20, 24, 'a contiguous cache: token i sits at start + i × (bytes per token)', 't-title')]
b += row(40, 44, 'PPPPPWW', labels=['0', '1', '2', '3', '4', '5', '6'])
b += [T(40, 96, 'request A, start address 1000', 't-tick'),
      T(40, 114, 'token 5 is at 1000 + 5 × 128 KiB: one multiply, one add, no lookup', 't-tick')]
b += [T(20, 156, 'the catch: A cannot grow if someone sits right behind it', 't-title')]
b += row(40, 172, 'PPPPPWW', labels=['A'] * 7)
b += row(40 + 7 * 29, 172, 'NNNNNN', labels=['B'] * 6)
b += [B(40 + 13 * 29, 172, 26, 30, 'box-ghost', 3), L(40 + 7 * 29 - 4, 228, 40 + 7 * 29 - 4, 206, 'path bad', False),
      T(40, 244, 'A\'s next token must go in slot 7, but B already owns it.', 't-tick'),
      T(40, 262, 'So a contiguous system books all the room A might ever need on day one.', 't-tick')]
F['contig'] = svg(760, 276, 'A contiguous cache finds token i by simple arithmetic, but cannot grow when another request sits right behind it, so systems book the maximum room up front.', b)

LEG = [('box-1', 'prompt'), ('box-3', 'written'), ('box-4', 'reserved'), ('waste', 'never used'), ('box-ghost', 'free')]

# ------------------------------------------------------------------ (1) reserved slots, 3 steps
b = [T(20, 24, 'request A books 8 slots; its prompt has 3 tokens; it will write 5', 't-title')]
for i, (lab, cells, note) in enumerate((('after the prompt', 'PPPRRRRR', '3 hold tokens, 5 wait'),
                                        ('after 2 new tokens', 'PPPWWRRR', '5 hold tokens, 3 wait'),
                                        ('finished', 'PPPWWWWW', 'all 8 hold tokens'))):
    y = 50 + i * 50
    b += [T(170, y + 20, lab, 't-note', 'end')] + row(184, y, cells) + [T(184 + 8 * 29 + 12, y + 20, note, 't-tick')]
b += legend(184, 220, LEG[:3])
b += [T(20, 256, 'The reserved slots are empty for now. They will be used, but until then no other request can use them.', 't-tick')]
F['reserved'] = svg(760, 270, 'Reserved slots: request A books 8 slots; after its prompt 5 of them wait for future tokens, after 2 new tokens 3 still wait.', b)

# ------------------------------------------------------------------ (2) internal fragmentation
b = [T(20, 24, 'request B books 8 slots; its prompt has 2 tokens; its answer stops after 2', 't-title')]
for i, (lab, cells, note) in enumerate((('after the prompt', 'PPRRRRRR', 'nobody knows yet'),
                                        ('answer done', 'PPWWIIII', '4 slots never used'))):
    y = 50 + i * 50
    b += [T(170, y + 20, lab, 't-note', 'end')] + row(184, y, cells) + [T(184 + 8 * 29 + 12, y + 20, note, 't-tick')]
b += legend(184, 170, LEG[:4])
b += [T(20, 206, 'The red slots sat empty for the whole life of B. That memory did nothing at all.', 't-tick')]
F['internal'] = svg(760, 220, 'Internal fragmentation: request B books 8 slots but its answer ends early, so 4 slots inside its booking are never used.', b)

# ------------------------------------------------------------------ (3) external fragmentation
b = [T(20, 24, 'a GPU memory of 24 slots, booked by four requests of different sizes', 't-title')]
labs1 = list('AAAAAAAABBBBCCCCCCDDDDDD')
r1 = ''.join({'A': 'P', 'B': 'W', 'C': 'P', 'D': 'W'}[c] for c in labs1)
b += [T(170, 70, 'all four running', 't-note', 'end')] + row(184, 50, r1, w=20, gap=2, labels=labs1)
labs2 = [c if c in 'AC' else '' for c in labs1]
r2 = ''.join({'A': 'P', 'C': 'P'}.get(c, '_') for c in labs1)
b += [T(170, 130, 'B and D finish', 't-note', 'end')] + row(184, 110, r2, w=20, gap=2, labels=labs2)
b += brace(184 + 8 * 22, 184 + 12 * 22 - 2, 146, 'gap of 4') + brace(184 + 18 * 22, 184 + 24 * 22 - 2, 146, 'gap of 6')
b += [T(170, 210, 'E needs 7', 't-note', 'end')] + row(184, 190, 'XXXXXXX', w=20, gap=2, labels=['E'] * 7)
b += [T(184 + 7 * 22 + 12, 210, 'refused: 10 slots are free, but no single gap holds 7', 't-tick')]
b += [T(20, 250, 'Free memory exists, yet it is cut into pieces that are each too small for the next request.', 't-tick')]
F['external'] = svg(760, 264, 'External fragmentation: after B and D finish, 10 slots are free in gaps of 4 and 6, so a request for 7 slots in one piece is refused.', b)

# ------------------------------------------------------------------ all three at once
t = R['toy_all']
cells = 'PPPWWRRI___PPWRIIII_____'
b = [T(20, 24, 'one moment in a 24-slot memory: request A, a gap, request B, free slots', 't-title')]
b += row(40, 74, cells, w=24, gap=4, labels=['A'] * 8 + [''] * 3 + ['B'] * 8 + [''] * 5)
b += brace(40, 40 + 8 * 28 - 4, 112, 'A books 8') + brace(40 + 11 * 28, 40 + 19 * 28 - 4, 112, 'B books 8')
b += brace(40 + 8 * 28, 40 + 11 * 28 - 4, 68, 'gap 3', up=True) + brace(40 + 19 * 28, 40 + 24 * 28 - 4, 68, 'free 5', up=True)
b += legend(40, 170, LEG, step=130)
b += [T(40, 206, f'holding tokens: {t["tokens"]} slots;  reserved for later tokens: {t["reserved"]};  never used (internal fragmentation): {t["internal"]}'),
      T(40, 226, f'free: {t["free"]}, but in gaps of 3 and 5, and the next request needs 6 in one piece (external fragmentation)'),
      T(40, 262, f'Only {t["tokens"]} of 24 slots ({t["share"]:.0%}) hold real keys and values.', 't-title')]
F['all3'] = svg(760, 278, 'All three wastes in one 24-slot memory: 8 slots hold tokens, 3 are reserved, 5 are never used and 8 are free but in gaps too small for the next request.', b)

# ------------------------------------------------------------------ how many conversations fit on real GPUs
CAP = json.load(open(os.path.join(HERE, 'results', 'p2_gpu_capacity.json')))
b = [T(20, 24, 'Llama 3.1 8B (BF16 weights) on one GPU, conversations of 4,096 tokens', 't-title')]
Lx, Wx, top = 130, 600, 180e9
rows8 = [('H100 80GB', 80e9), ('H200 141GB', 141e9), ('B200 180GB', 180e9)]
w8 = CAP['Llama 3.1 8B']['weights_gb'] * 1e9
for i, (g, cap) in enumerate(rows8):
    y = 44 + i * 64
    r = CAP['Llama 3.1 8B'][g][0]
    sc = lambda v: v / top * Wx
    b += [T(Lx - 10, y + 20, g.replace('GB', ' GB'), 't-note', 'end')]
    b += [B(Lx, y, sc(w8), 30, 'box-2', 3), B(Lx + sc(w8), y, sc(r['kv_gb'] * 1e9), 30, 'box-3', 3),
          B(Lx + sc(w8 + r['kv_gb'] * 1e9), y, sc(cap * 0.1), 30, 'box-ghost', 3)]
    b += [T(Lx + sc(w8) / 2, y + 20, 'weights', 't-tick', 'middle'),
          T(Lx + sc(w8) + sc(r['kv_gb'] * 1e9) / 2, y + 20, f"KV cache {r['kv_gb']:.0f} GB", 't-note', 'middle'),
          T(Lx, y + 48, f"{r['bf16_4096']} conversations with a BF16 cache, {r['fp8_4096']} with FP8", 't-tick')]
y = 44 + 3 * 64 + 6
b += [T(20, y, 'Llama 3.1 70B (141.1 GB of weights) needs several GPUs; bars show the total over all of them', 't-title')]
rows70 = [('4 x H100 80GB', 'H100 80GB', 1, 4 * 80e9), ('2 x H200 141GB', 'H200 141GB', 0, 2 * 141e9), ('2 x B200 180GB', 'B200 180GB', 1, 2 * 180e9)]
w70 = CAP['Llama 3.1 70B']['weights_gb'] * 1e9
top70 = 2 * 180e9
for i, (lab, g, k, cap) in enumerate(rows70):
    yy = y + 18 + i * 64
    r = CAP['Llama 3.1 70B'][g][k]
    sc = lambda v: v / top70 * Wx
    b += [T(Lx - 10, yy + 20, lab.replace('GB', ' GB'), 't-note', 'end')]
    b += [B(Lx, yy, sc(w70), 30, 'box-2', 3), B(Lx + sc(w70), yy, sc(r['kv_gb'] * 1e9), 30, 'box-3', 3),
          B(Lx + sc(w70 + r['kv_gb'] * 1e9), yy, sc(cap * 0.1), 30, 'box-ghost', 3)]
    b += [T(Lx + sc(w70) / 2, yy + 20, 'weights', 't-tick', 'middle'),
          T(Lx + sc(w70) + sc(r['kv_gb'] * 1e9) / 2, yy + 20, f"KV {r['kv_gb']:.0f} GB", 't-note', 'middle'),
          T(Lx, yy + 48, f"{r['bf16_4096']} conversations with a BF16 cache, {r['fp8_4096']} with FP8", 't-tick')]
yy = y + 18 + 3 * 64
b += legend(20, yy + 8, [('box-2', 'weights'), ('box-3', 'KV cache'), ('box-ghost', '10% kept for working memory')], step=150)
F['gpu_capacity'] = svg(760, yy + 24, 'How many 4,096-token conversations fit after the weights, on real GPUs, for Llama 3.1 8B and 70B, with a BF16 or an FP8 cache.', b)

json.dump(F, open(os.path.join(HERE, 'results', 'p2_figs.json'), 'w'), indent=0)
print('figures:', ', '.join(F))
