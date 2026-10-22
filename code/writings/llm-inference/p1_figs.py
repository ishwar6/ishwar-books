"""SVG figures for Part 1 of "LLM Inference from the Ground Up" (prefill and decode).
Numbers come from results/p1_roofline_stdout.txt, results/p1_tokens_stdout.txt and the published Part 1 runs.
CSS classes only, so the figures follow the dark, dim and light themes.
Run: python p1_figs.py  ->  results/p1_figs.json (key -> SVG string)"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'papers', 'sam'))
from figlib import svg, text, box, arrow, bar_path, esc

F = json.load(open(os.path.join(HERE, 'results', 'p1_existing_figs.json')))
OUT = {k: F[k] for k in ('loop', 'request_steps', 'read_vs_write', 'weights_trip', 'batch_bars')}

# published Part 1 measurements (Qwen2.5-0.5B, BF16, Apple M5 Pro)
BATCH = [(1, 10.0, 100), (2, 10.6, 190), (4, 11.1, 361), (8, 11.9, 670), (16, 12.8, 1247),
         (32, 15.1, 2120), (64, 19.7, 3242), (128, 29.3, 4367)]
FIX, PER = 10.6, 0.146          # least-squares line, p1_roofline_stdout.txt


def line(x1, y1, x2, y2, cls='edge'):
    return f'<line class="{cls}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>'


def mark(title, inner):
    return f'<g class="mark"><title>{esc(title)}</title>{inner}</g>'


# ---------------------------------------------------------------- tokens
def tokens():
    b = []
    rows = [('The cat sat on the', ['The', ' cat', ' sat', ' on', ' the'], [785, 8251, 7578, 389, 279]),
            ('Tokenization is unbelievably useful', ['Token', 'ization', ' is', ' unbelie', 'vably', ' useful'],
             [3323, 2022, 374, 38937, 88134, 5390])]
    y = 30
    for txt, toks, ids in rows:
        b.append(text(30, y + 6, f'"{txt}"', 't-title'))
        b.append(text(30, y + 26, f'{len(toks)} tokens', 't-tick'))
        x = 30
        y2 = y + 44
        for t, i in zip(toks, ids):
            w = max(64, 9.5 * len(t.strip()) + 26)
            b.append(box(x, y2, w, 34, 'box-1', 6))
            b.append(text(x + w / 2, y2 + 22, t.replace(' ', '·'), 't-title', 'middle'))
            b.append(text(x + w / 2, y2 + 52, f'{i:,}', 't-tick', 'middle'))
            x += w + 10
        y += 120
    b.append(text(30, y + 4, 'The dot · marks a space. The number under each token is its id in the vocabulary of 151,665 tokens.', 't-tick'))
    return svg(760, y + 20, 'Two sentences cut into tokens by the Qwen2.5 tokenizer. Common words are one token; a rare word is cut into pieces.', b)


# ---------------------------------------------------------------- timeline to scale (redrawn, no fixed colours)
def timeline():
    b = []
    sc = 91.2 / 27.4                                   # px per ms, as in the published figure
    x0 = 80
    b.append(text(x0, 40, 'time', 't-tick'))
    b.append(arrow(114, 36, 200, 36))
    b.append(mark('prefill: all 512 prompt tokens in one pass, 27.4 ms', box(x0, 70, 27.4 * sc, 54, 'box-1', 6)))
    b.append(text(x0 + 27.4 * sc / 2, 102, 'prefill', 't-title', 'middle'))
    x = x0 + 27.4 * sc + 3
    for i in range(14):
        b.append(mark(f'decode step {i + 1}: one new token, 10.0 ms', box(x, 70, 10.0 * sc - 3, 54, 'box-2', 4)))
        x += 10.0 * sc
    b.append(text(x + 8, 103, '...', 't-title'))
    b += [line(x0, 140, x0 + 27.4 * sc, 140, 'axis'), line(x0, 134, x0, 146, 'axis'),
          line(x0 + 27.4 * sc, 134, x0 + 27.4 * sc, 146, 'axis')]
    b.append(text(x0, 166, 'time to first token (TTFT)', 't-note'))
    xs = x0 + 27.4 * sc + 3 + 4 * 10.0 * sc
    b += [line(xs, 140, xs + 10.0 * sc - 3, 140, 'axis'), line(xs, 134, xs, 146, 'axis'), line(xs + 10.0 * sc - 3, 134, xs + 10.0 * sc - 3, 146, 'axis')]
    b.append(text(xs, 166, 'time per output token (TPOT)', 't-note'))
    b.append(text(x0, 194, 'measured: 512-token prompt, Qwen2.5-0.5B on an Apple M5 Pro. Prefill 27 ms, each decode step 10 ms.', 't-tick'))
    return svg(760, 210, 'A request has two phases. Prefill reads the whole prompt in one pass and produces the first token. Decode then produces one token per step.', b)


# ---------------------------------------------------------------- what a GPU is made of
def gpu_parts():
    b = []
    # memory side
    b.append(box(30, 50, 210, 230, 'box', 14))
    b.append(text(135, 80, 'memory (HBM)', 't-title', 'middle'))
    b.append(text(135, 100, 'big, but far from the cores', 't-tick', 'middle'))
    for i in range(4):
        b.append(box(60, 116 + i * 34, 150, 26, 'box-2', 5))
        b.append(text(135, 134 + i * 34, 'weights', 't-tick', 'middle'))
    b.append(text(135, 266, 'H100: 80 GB', 't-note', 'middle'))
    # the link
    b.append(f'<path class="path" d="M246,165 L470,165" marker-end="url(#ah-on)"/>')
    b.append(text(358, 140, 'memory bandwidth', 't-title', 'middle'))
    b.append(text(358, 196, 'bytes per second through the link', 't-tick', 'middle'))
    b.append(text(358, 216, 'H100: 3.35 TB/s', 't-note', 'middle'))
    b.append(text(358, 236, 'our M5 Pro: 262 GB/s (measured)', 't-note', 'middle'))
    # compute side
    b.append(box(480, 50, 250, 230, 'box', 14))
    b.append(text(605, 80, 'compute cores', 't-title', 'middle'))
    b.append(text(605, 100, 'thousands of multiply-add units', 't-tick', 'middle'))
    for r in range(4):
        for c in range(8):
            b.append(box(508 + c * 25, 116 + r * 25, 19, 19, 'box-1', 3))
    b.append(text(605, 236, 'FLOPs per second', 't-tick', 'middle'))
    b.append(text(605, 266, 'H100: 989 trillion FLOP/s', 't-note', 'middle'))
    b.append(text(380, 26, 'Two speeds decide everything: how fast the cores calculate, and how fast memory feeds them.', 't-tick', 'middle'))
    return svg(760, 300, 'A GPU, reduced to two parts: memory that holds the weights, compute cores that do the arithmetic, and a link between them with a fixed bandwidth.', b)


# ---------------------------------------------------------------- the kitchen
def kitchen():
    b = []
    # pantry
    b.append(box(30, 40, 200, 200, 'box-2', 14))
    b.append(text(130, 68, 'the pantry', 't-title', 'middle'))
    for r in range(3):
        for c in range(4):
            b.append(box(52 + c * 40, 86 + r * 40, 30, 30, 'box', 5))
    b.append(text(130, 222, 'every ingredient jar', 't-tick', 'middle'))
    # door
    b.append(box(300, 120, 26, 60, 'box-on', 4))
    b.append(text(313, 112, 'narrow door', 't-note', 'middle'))
    b.append(arrow(236, 150, 296, 150, True))
    b.append(arrow(330, 150, 400, 150, True))
    # chef + counter
    b.append(box(410, 40, 320, 200, 'box-1', 14))
    b.append(text(570, 68, 'the chef at a small counter', 't-title', 'middle'))
    b.append('<circle class="box" cx="490" cy="130" r="28"/>')
    b.append(text(490, 135, 'chef', 't-title', 'middle'))
    b.append(box(540, 112, 160, 40, 'box', 6))
    b.append(text(620, 137, 'room for a few jars', 't-tick', 'middle'))
    b.append(text(570, 196, 'cooks very fast, but only with', 't-tick', 'middle'))
    b.append(text(570, 214, 'what is on the counter', 't-tick', 'middle'))
    # mapping
    rows = [('pantry', 'GPU memory (HBM)'), ('ingredient jars', 'the model\'s weights'),
            ('the door', 'memory bandwidth'), ('chef\'s speed', 'FLOPs per second')]
    y = 274
    for i, (a, c) in enumerate(rows):
        x = 30 + (i % 2) * 360
        yy = y + (i // 2) * 26
        b.append(text(x, yy, a, 't-title'))
        b.append(text(x + 140, yy, '= ' + c, 't-note'))
    return svg(760, 330, 'The kitchen picture: a fast chef with a small counter, a pantry full of jars, and a narrow door between them.', b)


# ---------------------------------------------------------------- matrix times vector / matrix times matrix
def matmul():
    W = [[1, 2, 0], [0, 1, 3], [2, 0, 1]]
    X = [[1, 2, 1], [0, 1, 2], [2, 0, 1], [1, 1, 0]]
    out = [[sum(W[i][k] * x[k] for k in range(3)) for i in range(3)] for x in X]
    b = []

    def grid(x, y, M, cls, cell=30):
        for i, row in enumerate(M):
            for j, v in enumerate(row):
                b.append(box(x + j * cell, y + i * cell, cell - 3, cell - 3, cls, 4))
                b.append(text(x + j * cell + (cell - 3) / 2, y + i * cell + 19, v, 't-title', 'middle'))

    def col(x, y, v, cls, cell=30):
        grid(x, y, [[e] for e in v], cls, cell)

    # panel A: one token
    b.append(text(30, 26, 'A. decode, 1 token', 't-title'))
    grid(30, 50, W, 'box-2'); b.append(text(73, 155, 'weights W', 't-tick', 'middle'))
    b.append(text(130, 100, '×', 't-big', 'middle'))
    col(150, 50, X[0], 'box-1'); b.append(text(164, 155, 'token', 't-tick', 'middle'))
    b.append(text(210, 100, '=', 't-big', 'middle'))
    col(230, 50, out[0], 'box-3'); b.append(text(244, 155, 'output', 't-tick', 'middle'))
    b.append(text(30, 186, '9 weights read = 18 bytes', 't-note'))
    b.append(text(30, 206, '9 multiply-adds = 18 FLOPs', 't-note'))
    b.append(text(30, 230, '18 FLOPs / 18 bytes = 1 FLOP per byte', 't-title'))
    # panel B: four tokens
    b.append(text(390, 26, 'B. four tokens at once', 't-title'))
    grid(390, 50, W, 'box-2', 28); b.append(text(431, 155, 'the same W', 't-tick', 'middle'))
    b.append(text(487, 100, '×', 't-big', 'middle'))
    XT = [[X[j][i] for j in range(4)] for i in range(3)]
    OT = [[out[j][i] for j in range(4)] for i in range(3)]
    grid(502, 50, XT, 'box-1', 27); b.append(text(554, 155, '4 tokens', 't-tick', 'middle'))
    b.append(text(620, 100, '=', 't-big', 'middle'))
    grid(636, 50, OT, 'box-3', 27); b.append(text(688, 155, '4 outputs', 't-tick', 'middle'))
    b.append(text(390, 186, '9 weights read = 18 bytes (once)', 't-note'))
    b.append(text(390, 206, '4 × 9 multiply-adds = 72 FLOPs', 't-note'))
    b.append(text(390, 230, '72 FLOPs / 18 bytes = 4 FLOPs per byte', 't-title'))
    b.append(line(362, 40, 362, 236, 'drop'))
    return svg(760, 250, 'A 3 by 3 weight matrix multiplied by one token and by four tokens. The weights are read once in both cases; four tokens get four times the work out of them.', b)


# ---------------------------------------------------------------- where the 10 ms of a decode step go
def step_budget():
    b = []
    L, R = 40, 720
    sc = (R - L) / 10.0
    b.append(text(L, 30, 'one decode step, batch 1: 10.0 ms (measured)', 't-title'))
    b.append(mark('reading 0.99 GB of weights at 262 GB/s: at least 3.77 ms', box(L, 50, 3.77 * sc, 46, 'box-2', 6)))
    b.append(text(L + 3.77 * sc / 2, 78, 'read the weights: 3.8 ms', 't-title', 'middle'))
    b.append(mark('the rest of the step: about 6.2 ms', box(L + 3.77 * sc + 3, 50, (10 - 3.77) * sc - 3, 46, 'box', 6)))
    b.append(text(L + 3.77 * sc + (10 - 3.77) * sc / 2, 78, 'everything else: about 6.2 ms', 't-title', 'middle'))
    b.append(text(L, 122, 'the floor: 0.99 GB ÷ 262 GB/s', 't-tick'))
    b.append(text(L + 3.77 * sc + 8, 122, 'starting hundreds of small GPU jobs one after another, and gaps between them', 't-tick'))
    for t in range(0, 11, 2):
        b.append(line(L + t * sc, 100, L + t * sc, 106, 'axis'))
    b.append(text(L, 152, 'The arithmetic itself, 0.99 GFLOP, is a tiny part: at full speed an H100 would finish it in about 1 microsecond.', 't-tick'))
    return svg(760, 166, 'A 10 ms decode step at batch 1: at least 3.8 ms reading weights, most of the rest fixed overhead, and almost no time doing arithmetic.', b)


# ---------------------------------------------------------------- roofline helpers
def roof_axes(b, L, T, W, H, xs_log, ys_log, xt, yt, fmt_y=lambda v: f'{v:,g}'):
    X = lambda v: L + (math.log10(v) - xs_log[0]) / (xs_log[1] - xs_log[0]) * W
    Y = lambda v: T + H - (math.log10(v) - ys_log[0]) / (ys_log[1] - ys_log[0]) * H
    for v in yt:
        b.append(line(L, Y(v), L + W, Y(v), 'grid'))
        b.append(text(L - 8, Y(v) + 4, fmt_y(v), 't-tick', 'end'))
    for v in xt:
        b.append(text(X(v), T + H + 18, f'{v:,g}', 't-tick', 'middle'))
    b.append(line(L, T + H, L + W, T + H, 'axis'))
    return X, Y


def roofline_build():
    """Three small panels: the memory slope, the compute roof, and the lower of the two."""
    b = []
    PEAK, BW = 989, 3.35
    for p, (title, show_slope, show_roof) in enumerate([('1. memory limit', 1, 0), ('2. compute limit', 0, 1), ('3. the roofline: the lower of the two', 1, 1)]):
        L = 52 + p * 245; T = 46; Wd = 190; Ht = 160
        b.append(text(L - 40, 22, title, 't-title'))
        X, Y = roof_axes(b, L, T, Wd, Ht, (0, 3.5), (0, 3.2), [1, 10, 100, 1000], [1, 10, 100, 1000])
        xr = PEAK / BW
        if show_slope:
            x2 = 3000 if not show_roof else xr
            cls = 'l2' if not show_roof else 'l1'
            b.append(f'<polyline class="{cls}" points="{X(1):.1f},{Y(BW):.1f} {X(min(x2, 3000)):.1f},{Y(min(BW * x2, 1500)):.1f}"/>')
        if show_roof:
            x1 = 1 if not show_slope else xr
            cls = 'l3' if not show_slope else 'l1'
            b.append(f'<polyline class="{cls}" points="{X(x1):.1f},{Y(PEAK):.1f} {X(3000):.1f},{Y(PEAK):.1f}"/>')
        if p == 0:
            b.append(text(L + Wd, T + Ht - 40, 'speed = bandwidth', 't-tick', 'end'))
            b.append(text(L + Wd, T + Ht - 24, '× FLOPs per byte', 't-tick', 'end'))
        if p == 1:
            b.append(text(L + 6, Y(PEAK) + 18, 'the cores cannot go faster', 't-tick'))
            b.append(text(L + 6, Y(PEAK) + 34, 'than 989 TFLOP/s', 't-tick'))
        if p == 2:
            b.append(line(X(xr), Y(PEAK), X(xr), T + Ht, 'drop'))
            b.append(text(X(xr) + 4, T + Ht - 8, 'ridge 295', 't-tick'))
            b.append(text(X(3) , Y(60), 'memory-bound', 't-note'))
            b.append(text(X(400) + 2, Y(PEAK) + 18, 'compute-', 't-note'))
            b.append(text(X(400) + 2, Y(PEAK) + 34, 'bound', 't-note'))
        b.append(text(L + Wd / 2, T + Ht + 38, 'FLOPs per byte', 't-tick', 'middle'))
    b.append(text(12, 252 + 18, 'Vertical axes: TFLOP/s the GPU can reach. Both axes use a log scale. Numbers: an NVIDIA H100 (BF16).', 't-tick'))
    return svg(760, 284, 'The roofline built in three steps for an H100: a sloped memory limit, a flat compute limit, and the lower of the two.', b)


def roofline(points, label, note=None, h=340):
    b = []
    PEAK, BW = 989, 3.35
    L, T, Wd, Ht = 70, 30, 660, 250
    X, Y = roof_axes(b, L, T, Wd, Ht, (0, math.log10(4096)), (0, 3.2), [1, 4, 16, 64, 295, 1024, 4096], [1, 10, 100, 1000])
    xr = PEAK / BW
    b.append(f'<polyline class="l1" points="{X(1):.1f},{Y(BW):.1f} {X(xr):.1f},{Y(PEAK):.1f} {X(4096):.1f},{Y(PEAK):.1f}"/>')
    b.append(line(X(xr), Y(PEAK), X(xr), T + Ht, 'drop'))
    b.append(text(X(4096), Y(PEAK) - 10, 'compute roof: 989 TFLOP/s', 't-tick', 'end'))
    ang = -math.degrees(math.atan2(Y(BW) - Y(PEAK), X(xr) - X(1)))
    b.append(text(X(6), Y(BW * 6) + 22, 'memory slope: 3.35 TB/s × FLOPs per byte', 't-tick', 'start',
                  f' transform="rotate({ang:.1f} {X(6):.1f} {Y(BW * 6) + 22:.1f})"'))
    for (name, I, cls, dx, dy) in points:
        v = min(PEAK, BW * I)
        b.append(mark(f'{name}: about {I:g} FLOPs per byte, at most {v:,.0f} TFLOP/s ({v / PEAK:.1%} of the roof)',
                      f'<circle class="{cls} ring" cx="{X(I):.1f}" cy="{Y(v):.1f}" r="6"/>'))
        b.append(text(X(I) + dx, Y(v) + dy, name, 't-note'))
    b.append(text(L + Wd / 2, T + Ht + 44, 'FLOPs per byte read from memory (log scale)', 't-tick', 'middle'))
    b.append(text(10, 16, 'TFLOP/s', 't-tick'))
    if note:
        b.append(text(L, h - 8, note, 't-tick'))
    return svg(760, h, label, b)


# ---------------------------------------------------------------- the bus
def bus():
    b = []

    def draw_bus(x, y, n, title, sub):
        b.append(text(x, y - 14, title, 't-title'))
        b.append(box(x, y, 300, 92, 'box', 18))
        for i in range(6):
            b.append(box(x + 18 + i * 44, y + 16, 34, 30, 'box-1' if i < n else 'box-ghost', 5))
            if i < n:
                b.append(f'<circle class="s2" cx="{x + 35 + i * 44:.1f}" cy="{y + 31:.1f}" r="8"/>')
        b.append(f'<circle class="box" cx="{x + 60}" cy="{y + 92}" r="14"/>')
        b.append(f'<circle class="box" cx="{x + 240}" cy="{y + 92}" r="14"/>')
        b.append(text(x + 150, y + 74, sub, 't-note', 'middle'))

    draw_bus(30, 50, 1, 'one passenger', 'one trip: 10.0 ms')
    draw_bus(420, 50, 4, 'four passengers', 'one trip: 11.1 ms')
    b.append(text(30, 190, 'the bus = one read of all the weights', 't-tick'))
    b.append(text(30, 208, 'a passenger = one sequence that needs its next token', 't-tick'))
    b.append(text(420, 190, '1 token per trip: 100 tokens/s', 't-title'))
    b.append(text(420, 208, '4 tokens per trip: 361 tokens/s', 't-title'))
    return svg(760, 222, 'A bus trip costs about the same with one passenger or four. One read of the weights is the trip; each sequence is a passenger.', b)


# ---------------------------------------------------------------- one decode step, batch 1 vs batch 4
def batch_step():
    b = []
    seqs = [('A', 'The capital is', 'Paris'), ('B', 'The sky is', 'blue'), ('C', '2 + 2 =', '4'), ('D', 'def hello', '(')]
    # row 1: batch 1
    b.append(text(20, 24, 'batch 1', 't-title'))
    b.append(text(20, 42, '10.0 ms', 't-tick'))
    y = 60
    b.append(box(100, y, 150, 30, 'box', 6)); b.append(text(175, y + 20, 'A: The capital is', 't-tick', 'middle'))
    b.append(arrow(254, y + 15, 300, y + 15))
    b.append(box(304, y - 6, 170, 42, 'box-2', 8)); b.append(text(389, y + 12, 'read all weights', 't-title', 'middle'))
    b.append(text(389, y + 28, '0.99 GB, once', 't-tick', 'middle'))
    b.append(arrow(478, y + 15, 524, y + 15))
    b.append(box(528, y, 90, 30, 'box-3', 6)); b.append(text(573, y + 21, 'Paris', 't-big', 'middle'))
    b.append(text(630, y + 20, '1 token out', 't-note'))
    b.append(line(20, 116, 740, 116, 'drop'))
    # row 2: batch 4, three numbered steps
    b.append(text(20, 146, 'batch 4', 't-title'))
    b.append(text(20, 164, '11.1 ms', 't-tick'))
    b.append(text(175, 146, '1. stack the newest token', 't-note', 'middle'))
    b.append(text(175, 162, 'of each sequence', 't-note', 'middle'))
    b.append(text(389, 146, '2. read each weight once,', 't-note', 'middle'))
    b.append(text(389, 162, 'use it for all four', 't-note', 'middle'))
    b.append(text(600, 146, '3. each sequence gets', 't-note', 'middle'))
    b.append(text(600, 162, 'its own next token', 't-note', 'middle'))
    y0 = 180
    for i, (k, ctx, nxt) in enumerate(seqs):
        yy = y0 + i * 38
        b.append(box(100, yy, 150, 30, 'box', 6)); b.append(text(175, yy + 20, f'{k}: {ctx}', 't-tick', 'middle'))
        b.append(arrow(254, yy + 15, 300, y0 + 72))
        b.append(arrow(478, y0 + 72, 524, yy + 15))
        b.append(box(528, yy, 90, 30, 'box-3', 6)); b.append(text(573, yy + 21, nxt, 't-big', 'middle'))
    b.append(box(304, y0 + 46, 170, 54, 'box-2', 8)); b.append(text(389, y0 + 68, 'read all weights', 't-title', 'middle'))
    b.append(text(389, y0 + 86, '0.99 GB, once', 't-tick', 'middle'))
    b.append(text(630, y0 + 72, '4 tokens out', 't-note'))
    b.append(text(630, y0 + 90, 'for 1.1 ms more', 't-note'))
    b.append(text(20, 352, 'The four sequences do not need to be the same length or the same topic. They only share the trip to memory.', 't-tick'))
    return svg(760, 366, 'One decode step for one sequence and for four. With four, the newest token of each is stacked, the weights are read once, and four next tokens come out.', b)


# ---------------------------------------------------------------- step time = fixed part + part per sequence
def step_parts():
    b = []
    L, T, Wd, Ht = 70, 40, 640, 220
    bs = [1, 4, 8, 32, 64, 128]
    meas = {bb: ms for bb, ms, _ in BATCH}
    ymax = 32
    Y = lambda v: T + Ht - v / ymax * Ht
    slot = Wd / len(bs)
    for v in range(0, 31, 10):
        b.append(line(L, Y(v), L + Wd, Y(v), 'grid'))
        b.append(text(L - 8, Y(v) + 4, f'{v} ms', 't-tick', 'end'))
    for i, bb in enumerate(bs):
        x = L + i * slot + slot * 0.25; w = slot * 0.5
        per = PER * bb
        b.append(mark(f'batch {bb}: fixed part {FIX} ms', f'<path class="s2 bar-mark" d="{bar_path(x, Y(FIX), w, Y(0) - Y(FIX), r=0)}"/>'))
        b.append(mark(f'batch {bb}: part per sequence {per:.1f} ms', f'<path class="s1 bar-mark" d="{bar_path(x, Y(FIX + per), w, Y(FIX) - Y(FIX + per), r=3)}"/>'))
        b.append(mark(f'batch {bb}: measured {meas[bb]} ms', f'<circle class="s3 ring" cx="{x + w / 2:.1f}" cy="{Y(meas[bb]):.1f}" r="5"/>'))
        b.append(text(x + w / 2, Y(max(meas[bb], FIX + per)) - 10, f'{meas[bb]} ms', 't-val', 'middle'))
        b.append(text(x + w / 2, T + Ht + 18, f'{bb}', 't-tick', 'middle'))
    b.append(line(L, T + Ht, L + Wd, T + Ht, 'axis'))
    b.append(text(L + Wd / 2, T + Ht + 40, 'sequences in the batch', 't-tick', 'middle'))
    lx = L + 10
    b.append(box(lx, 8, 12, 12, 's2', 2)); b.append(text(lx + 18, 18, f'fixed part, {FIX} ms (weights + overhead)', 't-tick'))
    b.append(box(lx + 300, 8, 12, 12, 's1', 2)); b.append(text(lx + 318, 18, f'{PER} ms per sequence', 't-tick'))
    b.append(f'<circle class="s3" cx="{lx + 476}" cy="14" r="5"/>'); b.append(text(lx + 486, 18, 'measured', 't-tick'))
    return svg(760, 310, 'Decode step time split into a fixed part and a part that grows with each sequence. The line fits the measured step times within half a millisecond from batch 4 to 128.', b)


# ---------------------------------------------------------------- GPU memory: weights + notes
def memory_fill():
    b = []
    L, Wd = 40, 680

    def bar(y, total, parts, title, scale_note):
        b.append(text(L, y - 12, title, 't-title'))
        x = L
        for (name, size, cls, lab) in parts:
            w = size / total * Wd
            b.append(mark(f'{name}: {lab}', box(x, y, max(w, 1.5), 40, cls, 3)))
            x += w
        b.append(box(L, y, Wd, 40, 'box-ghost', 3))
        b.append(text(L, y + 60, scale_note, 't-tick'))

    # Qwen run: weights 0.99 GB, notes 0.81 GB at batch 128 x 512
    bar(40, 0.99 + 0.81, [('weights', 0.99, 'box-2', '0.99 GB'), ('notes of 128 sequences', 0.81, 'box-1', '0.81 GB')],
        'our run: Qwen2.5-0.5B, 128 sequences of 512 tokens', '')
    b.append(text(L + 0.99 / 1.8 * Wd / 2, 65, 'weights 0.99 GB', 't-title', 'middle'))
    b.append(text(L + (0.99 + 0.405) / 1.8 * Wd, 65, 'notes 0.81 GB', 't-title', 'middle'))
    b.append(text(L, 100, 'Each token\'s notes take 12 KiB; 128 × 512 tokens × 12 KiB = 0.81 GB, almost as much as the model.', 't-tick'))
    # Llama 3.1 8B on an H100
    y = 160
    total = 80.0
    parts = [('weights', 16.1, 'box-2', '16.1 GB')]
    each = 131072 * 4096 / 1e9          # 0.537 GB per 4,096-token conversation
    for i in range(119):
        parts.append((f'conversation {i + 1}', each, 'box-1' if i % 2 == 0 else 'box-3', '4,096 tokens, 512 MiB of notes'))
    bar(y, total, parts, 'Llama 3.1 8B on an 80 GB H100, conversations of 4,096 tokens', '')
    b.append(text(L + 16.1 / 80 * Wd / 2, y + 25, 'weights', 't-title', 'middle'))
    b.append(text(L, y + 60, 'Each thin stripe is one conversation\'s notes (512 MiB). At most 119 fit, before any working memory.', 't-tick'))
    b.append(text(L, y + 80, 'At the full 128K context, one conversation needs 16 GiB of notes, and at most 3 fit.', 't-tick'))
    return svg(760, 260, 'GPU memory split between the weights and the notes of every sequence in the batch. The notes grow with the batch and with the length of each conversation.', b)


# ---------------------------------------------------------------- latency vs throughput
def trade():
    b = []
    bs = [b_ for b_, _, _ in BATCH]
    per_user = [1000 / ms for _, ms, _ in BATCH]
    total = [t for _, _, t in BATCH]

    def panel(x0, title, vals, ymax, cls, fmt, ticks):
        Wd, Ht, T = 300, 170, 50
        b.append(text(x0, 24, title, 't-title'))
        Y = lambda v: T + Ht - v / ymax * Ht
        for t in ticks:
            b.append(line(x0 + 40, Y(t), x0 + 40 + Wd, Y(t), 'grid'))
            b.append(text(x0 + 34, Y(t) + 4, f'{t:,}', 't-tick', 'end'))
        slot = Wd / len(bs)
        for i, (bb, v) in enumerate(zip(bs, vals)):
            x = x0 + 40 + i * slot + slot * 0.18; w = slot * 0.64
            b.append(mark(f'batch {bb}: {fmt(v)}', f'<path class="{cls} bar-mark" d="{bar_path(x, Y(v), w, Y(0) - Y(v), r=3)}"/>'))
            b.append(text(x + w / 2, T + Ht + 16, f'{bb}', 't-tick', 'middle'))
        b.append(text(x0 + 40 + slot * 0.5, Y(vals[0]) - 6, fmt(vals[0]), 't-val', 'middle'))
        b.append(text(x0 + 40 + slot * 7.5, Y(vals[-1]) - 6, fmt(vals[-1]), 't-val', 'middle'))
        b.append(line(x0 + 40, T + Ht, x0 + 40 + Wd, T + Ht, 'axis'))
        b.append(text(x0 + 40 + Wd / 2, T + Ht + 36, 'sequences in the batch', 't-tick', 'middle'))

    panel(20, 'what one user sees: tokens/s', per_user, 110, 's2', lambda v: f'{v:.0f}', [0, 50, 100])
    panel(400, 'what the GPU makes: tokens/s', total, 4800, 's1', lambda v: f'{v:,.0f}', [0, 2000, 4000])
    return svg(760, 270, 'As the batch grows from 1 to 128, each user gets tokens about three times slower, while the GPU as a whole makes 44 times more.', b)


# ---------------------------------------------------------------- the hint: 16 tokens cost about as much as 1
def spec_hint():
    b = []
    L = 250; sc = 40
    rows = [('decode: 1 token', 10.0, 's2'), ('prefill: 16 tokens', 10.8, 's1')]
    for i, (name, ms, cls) in enumerate(rows):
        y = 30 + i * 50
        b.append(text(L - 12, y + 22, name, 't-note', 'end'))
        b.append(mark(f'{name}: {ms} ms', f'<path class="{cls} bar-mark" d="{bar_path(L, y, ms * sc, 30, True)}"/>'))
        b.append(text(L + ms * sc + 8, y + 21, f'{ms} ms', 't-val'))
    b.append(text(L, 146, 'Sixteen times the tokens for 8% more time.', 't-tick'))
    return svg(760, 160, 'One decode step for 1 token takes 10.0 ms; one pass over 16 tokens takes 10.8 ms.', b)


OUT['tokens'] = tokens()
OUT['timeline'] = timeline()
OUT['gpu_parts'] = gpu_parts()
OUT['kitchen'] = kitchen()
OUT['matmul'] = matmul()
OUT['step_budget'] = step_budget()
OUT['roofline_build'] = roofline_build()
OUT['roofline_phases'] = roofline(
    [('decode, batch 1', 1, 's2', 10, -12), ('prefill, 512 tokens', 512, 's1', 10, 26)],
    'Roofline for an H100: decode at batch 1 sits at the bottom of the memory slope; prefill sits on the compute roof.')
OUT['bus'] = bus()
OUT['batch_step'] = batch_step()
OUT['step_parts'] = step_parts()
OUT['roofline_batch'] = roofline(
    [('batch 1', 1, 's2', 10, -12), ('batch 4', 4, 's2', -62, -10), ('batch 16', 16, 's2', -70, -10),
     ('batch 64', 64, 's2', -70, -10), ('batch 128', 128, 's2', -78, -10), ('prefill, 512 tokens', 512, 's1', 10, 26)],
    'Decode climbs the memory slope of the H100 roofline as the batch grows: each sequence in the batch adds about one FLOP per byte.',
    note='Weights only. Each sequence also reads its own notes (Part 2), which pulls real points lower.', h=356)
OUT['memory_fill'] = memory_fill()
OUT['trade'] = trade()
OUT['spec_hint'] = spec_hint()


# ---------------------------------------------------------------- CUDA: from a Python line to the GPU
GPUS = json.load(open(os.path.join(HERE, 'results', 'p1_gpus.json')))
KER = open(os.path.join(HERE, 'results', 'p1_kernels_stdout.txt')).read()


def cuda_path():
    b = []
    # 1. CPU side
    b.append(text(20, 24, '1. on the CPU', 't-title'))
    b.append(box(20, 36, 220, 54, 'box-2', 8))
    b.append(text(130, 58, 'Python: y = x @ W', 't-mono', 'middle'))
    b.append(text(130, 78, 'PyTorch picks a GPU kernel', 't-tick', 'middle'))
    b.append(arrow(130, 92, 130, 118))
    b.append(box(20, 120, 220, 66, 'box-2', 8))
    b.append(text(130, 142, 'launch the kernel:', 't-note', 'middle'))
    b.append(text(130, 160, 'its code, its inputs and', 't-tick', 'middle'))
    b.append(text(130, 176, 'how many blocks of threads', 't-tick', 'middle'))
    b.append(f'<path class="path" d="M242,153 L300,153" marker-end="url(#ah-on)"/>')
    b.append(text(271, 143, 'launch', 't-tick', 'middle'))
    # 2. GPU: grid of blocks spread over SMs
    b.append(text(310, 24, '2. on the GPU: the blocks of threads are spread over the SMs', 't-title'))
    b.append(box(305, 36, 440, 184, 'box', 12))
    for i in range(4):
        x = 320 + i * 106
        b.append(box(x, 50, 96, 120, 'box-1', 8))
        b.append(text(x + 48, 68, f'SM {i if i < 3 else 131}', 't-note', 'middle'))
        for r in range(2):
            for c in range(4):
                b.append(box(x + 10 + c * 20, 78 + r * 20, 15, 15, 'box-3', 2))
        b.append(text(x + 48, 134, 'registers', 't-tick', 'middle'))
        b.append(text(x + 48, 150, 'shared memory', 't-tick', 'middle'))
        b.append(text(x + 48, 164, '(on-chip SRAM)', 't-tick', 'middle'))
    b.append(text(638, 116, '...', 't-big', 'middle'))
    b.append(box(320, 182, 414, 26, 'box-4', 6))
    b.append(text(527, 200, 'L2 cache, shared by all SMs', 't-note', 'middle'))
    # 3. HBM
    b.append(f'<path class="path" d="M420,276 L420,224" marker-end="url(#ah-on)"/>')
    b.append(text(428, 262, '3. read W and x', 't-tick'))
    b.append(f'<path class="path" d="M640,224 L640,276" marker-end="url(#ah-on)"/>')
    b.append(text(632, 262, '4. write y back', 't-tick', 'end'))
    b.append(box(305, 280, 440, 46, 'box-2', 8))
    b.append(text(525, 300, 'HBM: the GPU\'s main memory (the weights live here)', 't-note', 'middle'))
    b.append(text(525, 318, 'H100: 80 GB at 3.35 TB/s', 't-tick', 'middle'))
    b.append(text(20, 222, 'The CPU only gives orders.', 't-tick'))
    b.append(text(20, 240, 'All the arithmetic happens', 't-tick'))
    b.append(text(20, 258, 'on the GPU.', 't-tick'))
    b.append(box(20, 286, 12, 12, 'box-3', 2))
    b.append(text(40, 296, 'one block of threads', 't-tick'))
    b.append(text(20, 316, '(many threads, run in warps of 32)', 't-tick'))
    return svg(760, 340, 'What happens for one line of PyTorch: the CPU launches a kernel, its blocks of threads run on the streaming multiprocessors, they read the inputs from HBM through the on-chip memories and write the result back.', b)


def memory_ladder():
    b = []
    rows = [('registers', '256 KB per SM', 256e3 * 132, 'inside each SM, the fastest'),
            ('shared memory + L1', '256 KB per SM', 256e3 * 132, 'inside each SM'),
            ('L2 cache', '50 MB', 50e6, 'on the chip, shared by all 132 SMs'),
            ('HBM', '80 GB', 80e9, 'beside the chip, 3.35 TB/s')]
    L, Wd = 230, 330
    lo, hi = math.log10(1e6), math.log10(1e11)
    X = lambda v: L + (math.log10(v) - lo) / (hi - lo) * Wd
    b.append(text(20, 22, 'The memories of an H100, from the smallest and fastest to the largest (log scale)', 't-title'))
    for i, (name, size, tot, note) in enumerate(rows):
        y = 44 + i * 52
        b.append(text(L - 12, y + 17, name, 't-note', 'end'))
        b.append(text(L - 12, y + 33, size, 't-tick', 'end'))
        b.append(mark(f'{name}: {size}', f'<path class="{"s1" if i < 3 else "s2"} bar-mark" d="{bar_path(L, y, X(tot) - L, 28, True)}"/>'))
        total = f'{tot / 1e6:.0f} MB in all' if i < 2 else ''
        b.append(text(X(tot) + 8, y + 19, (total + ', ' if total else '') + note, 't-tick'))
    y = 44 + 4 * 52
    b.append(line(X(16.06e9), 44 + 3 * 52 - 8, X(16.06e9), y - 6, 'drop'))
    b.append(text(X(16.06e9) - 6, y + 10, 'Llama 3.1 8B weights: 16.1 GB', 't-tick', 'end'))
    for v, lab in ((1e6, '1 MB'), (1e8, '100 MB'), (1e10, '10 GB'), (1e11, '100 GB')):
        b.append(text(X(v), y + 30, lab, 't-tick', 'middle'))
    b.append(text(20, y + 52, 'Only HBM is big enough for the model, so every decode step must bring the weights from HBM again.', 't-tick'))
    return svg(760, y + 64, 'The memory levels of an H100 drawn on a log scale. The on-chip memories add up to tens of megabytes; only HBM can hold a model of 16 GB.', b)


def launch_gaps():
    b = []
    n = 12
    def lane(y0, title, graph):
        b.append(text(20, y0, title, 't-title'))
        b.append(text(110, y0 + 30, 'CPU', 't-note', 'end'))
        b.append(text(110, y0 + 66, 'GPU', 't-note', 'end'))
        x = 124
        if graph:
            b.append(box(124, y0 + 16, 60, 22, 'box-2', 4))
            b.append(text(154, y0 + 31, 'replay', 't-tick', 'middle'))
            x = 190
            for i in range(n):
                b.append(box(x, y0 + 52, 34, 22, 'box-1', 3)); x += 36
            b.append(text(x + 8, y0 + 67, 'kernels back to back', 't-tick'))
        else:
            for i in range(n):
                b.append(box(x, y0 + 16, 12, 22, 'box-2', 2))
                b.append(box(x + 18, y0 + 52, 34, 22, 'box-1', 3))
                x += 50
            b.append(text(x + 6, y0 + 67, 'gaps', 't-tick'))
    lane(24, 'without a CUDA graph: one launch per kernel', False)
    lane(124, 'with a CUDA graph: one launch replays the whole recorded step', True)
    b.append(box(20, 214, 12, 12, 'box-2', 2)); b.append(text(38, 224, 'CPU launch work', 't-tick'))
    b.append(box(170, 214, 12, 12, 'box-1', 2)); b.append(text(188, 224, 'a kernel running on the GPU', 't-tick'))
    b.append(text(20, 248, 'Small kernels finish faster than the CPU can launch the next one, so the GPU waits in the gaps.', 't-tick'))
    return svg(760, 262, 'Two timelines of one decode step. Without a CUDA graph the CPU launches every kernel in turn and the GPU waits between them; with a graph one launch replays all of them back to back.', b)


def fusion():
    b = []
    b.append(text(20, 22, 'three small kernels', 't-title'))
    names = ['multiply', 'add', 'normalise']
    for i, nm in enumerate(names):
        x = 40 + i * 120
        b.append(box(x, 40, 96, 34, 'box-1', 6)); b.append(text(x + 48, 62, nm, 't-note', 'middle'))
        b.append(arrow(x + 30, 120, x + 30, 78)); b.append(arrow(x + 66, 78, x + 66, 120, True))
    b.append(box(20, 124, 380, 30, 'box-2', 6)); b.append(text(210, 144, 'HBM: 3 reads + 3 writes', 't-note', 'middle'))
    b.append(text(440, 22, 'one fused kernel', 't-title'))
    b.append(box(460, 40, 260, 34, 'box-on', 6)); b.append(text(590, 62, 'multiply, add, normalise', 't-note', 'middle'))
    b.append(arrow(540, 120, 540, 78)); b.append(arrow(640, 78, 640, 120, True))
    b.append(box(440, 124, 300, 30, 'box-2', 6)); b.append(text(590, 144, 'HBM: 1 read + 1 write', 't-note', 'middle'))
    b.append(text(20, 184, 'The fused kernel keeps the in-between numbers in registers, so they never make the trip to HBM.', 't-tick'))
    return svg(760, 198, 'Kernel fusion. Three small kernels each read their input from HBM and write their output back; one fused kernel does all three steps with one read and one write.', b)


def gpu_bars():
    b = []
    names = list(GPUS)
    def panel(x0, title, key, fmt, vmax, cls, sub):
        b.append(text(x0, 22, title, 't-title'))
        b.append(text(x0, 40, sub, 't-tick'))
        Lb = x0 + 110; Wb = 230
        for i, nm in enumerate(names):
            y = 58 + i * 40; v = GPUS[nm][key]
            b.append(text(Lb - 8, y + 18, nm.replace(' 80GB SXM', '').replace(' (HGX)', ''), 't-note', 'end'))
            b.append(mark(f'{nm}: {fmt(v)}', f'<path class="{cls} bar-mark" d="{bar_path(Lb, y, v / vmax * Wb, 26, True)}"/>'))
            b.append(text(Lb + v / vmax * Wb + 6, y + 18, fmt(v), 't-val'))
    panel(10, 'one user, Llama 3.1 8B, batch 1', 'max_tps', lambda v: f'{v:.0f} tok/s', 520, 's2', 'the most tokens/s: bandwidth ÷ 16.1 GB')
    panel(390, 'ridge point', 'ridge', lambda v: f'{v:.0f}', 320, 's1', 'FLOPs per byte = TFLOP/s ÷ TB/s')
    b.append(text(10, 272, 'Vendor specs: dense BF16, no sparsity. Upper bounds from bandwidth alone; real steps are slower.', 't-tick'))
    return svg(760, 286, 'Left: the most tokens per second one user can get from Llama 3.1 8B at batch 1 on each GPU. Right: each GPU\'s ridge point, the batch size where decode stops being limited by memory.', b)


def server8():
    b = []
    b.append(text(20, 22, 'one server: 8 GPUs joined by NVLink switches', 't-title'))
    for i in range(8):
        x = 30 + i * 90
        b.append(box(x, 40, 76, 56, 'box-1', 8)); b.append(text(x + 38, 64, f'GPU {i}', 't-note', 'middle'))
        b.append(text(x + 38, 82, '80 GB', 't-tick', 'middle'))
        b.append(line(x + 38, 98, x + 38, 128, 'edge-on'))
    b.append(box(30, 130, 706, 34, 'box-on', 8)); b.append(text(383, 152, 'NVLink switch: every GPU talks to every other GPU', 't-note', 'middle'))
    b.append(text(20, 192, 'H100: 900 GB/s of NVLink per GPU. B200: 1.8 TB/s. Both are far faster than the PCIe link to the CPU.', 't-tick'))
    return svg(760, 206, 'An 8-GPU server. Every GPU connects to NVLink switches, so any GPU can read from any other at high speed.', b)


OUT['cuda_path'] = cuda_path()
OUT['memory_ladder'] = memory_ladder()
OUT['launch_gaps'] = launch_gaps()
OUT['fusion'] = fusion()
OUT['gpu_bars'] = gpu_bars()
OUT['server8'] = server8()

json.dump(OUT, open(os.path.join(HERE, 'results', 'p1_figs.json'), 'w'), indent=1)
print(f'{len(OUT)} figures:', ', '.join(OUT))
