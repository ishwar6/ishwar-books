"""Figures for Chapter 4 (pretraining). Writes results/figs_ch4.json (key -> svg).
Colours come only from CSS classes (box, box-1..4, s1..s4, l1..l4, edge, path ...), so every figure follows the site's themes.
Text widths: about 7.2 px per character for t-tick / t-muted (12 px), 7.8 for t-note (13 px), 9 px for t-title; 17 px between lines.
Every number drawn here is read from a results/ch4_*.json file written by a script."""
import json, math
from figlib import svg, text, box, arrow, esc
from agentfig import line, hbars
from alammar import frame, brace, vec
from ch4_gpt import lr_at

J = lambda n: json.load(open(f'results/{n}.json'))
PRE, COMP, DATA, TOK, MINI, FIT = J('ch4_pretrain'), J('ch4_compute'), J('ch4_data'), J('ch4_tokenizer'), J('ch4_scaling_mini'), J('ch4_fit')
PROBE, BATCH, PREC, PREP = J('ch4_probe'), J('ch4_batch'), J('ch4_precision'), J('ch4_prep')
F = {}


def para(x, y, lines, cls='t-muted', anchor='start', lh=17):
    return [text(x, y + i * lh, l, cls, anchor) for i, l in enumerate(lines)]


def block(x, y, w, h, title, lines=(), cls='box', tcls='t-note', lcls='t-tick'):
    b = [box(x, y, w, h, cls, 10)]
    b.append(text(x + w / 2, y + (h / 2 + 5 if not lines else 22), title, tcls, 'middle'))
    b += para(x + w / 2, y + 41, lines, lcls, 'middle')
    return b


def sci(v):
    e = int(math.floor(math.log10(v)))
    m = v / 10 ** e
    return f'{m:.1f}e{e}' if abs(m - 1) > 0.05 else f'1e{e}'


def chart(x0, y0, w, h, series, xr, yr, xticks, yticks, xlog=False, ylog=False, fx=str, fy=str, xlabel='', ylabel='',
          end_labels=True, dots=True):
    """A small line chart. series: [(name, xs, ys, line_cls, dot_cls)]. Returns (parts, X, Y)."""
    tx = (lambda v: math.log10(v)) if xlog else (lambda v: v)
    ty = (lambda v: math.log10(v)) if ylog else (lambda v: v)
    X = lambda v: x0 + (tx(v) - tx(xr[0])) / (tx(xr[1]) - tx(xr[0])) * w
    Y = lambda v: y0 + h - (ty(v) - ty(yr[0])) / (ty(yr[1]) - ty(yr[0])) * h
    b = []
    for t in yticks:
        b += [f'<line class="grid" x1="{x0}" y1="{Y(t):.1f}" x2="{x0 + w}" y2="{Y(t):.1f}"/>', text(x0 - 8, Y(t) + 4, fy(t), 't-tick', 'end')]
    for t in xticks:
        b.append(text(X(t), y0 + h + 18, fx(t), 't-tick', 'middle'))
    b.append(f'<line class="axis" x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}"/>')
    ends = []
    for name, xs, ys, lc, dc in series:
        b.append(f'<polyline class="{lc}" points="' + ' '.join(f'{X(a):.1f},{Y(c):.1f}' for a, c in zip(xs, ys)) + '"/>')
        if dots:
            for a, c in zip(xs, ys):
                b.append(f'<g class="mark"><title>{esc(name)}: {fx(a)}, {fy(c)}</title><circle class="{dc} ring" cx="{X(a):.1f}" cy="{Y(c):.1f}" r="4"/></g>')
        ends.append([Y(ys[-1]), name, dc, X(xs[-1]), Y(ys[-1])])
    if end_labels:
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 16)
        for ly, name, dc, ex, ey in ends:
            b.append(f'<rect class="{dc}" x="{ex + 12:.1f}" y="{ly - 5:.1f}" width="10" height="10" rx="2"/>')
            b.append(text(ex + 27, ly + 4, name, 't-tick'))
    if xlabel:
        b.append(text(x0 + w / 2, y0 + h + 38, xlabel, 't-muted', 'middle'))
    if ylabel:
        b.append(text(x0 - 8, y0 - 16, ylabel, 't-muted', 'start'))
    return b, X, Y


# ---------------------------------------------------------------- 1. the recipe at a glance
def recipe():
    b = [text(380, 24, 'Pretraining in one picture: five ingredients and one loop', 't-title', 'middle')]
    xs = [20, 170, 320, 470, 620]
    items = [('raw text', ['web, books, code', 'trillions of words'], 'box-1'),
             ('clean + mix', ['filter, dedup,', 'choose proportions'], 'box-1'),
             ('tokenize', ['BPE: text to ids', 'one long stream'], 'box-3'),
             ('Transformer', ['random weights', 'N parameters'], 'box-2'),
             ('base model', ['predicts the', 'next token well'], 'box-4')]
    for x, (t, l, c) in zip(xs, items):
        b += block(x, 50, 125, 70, t, l, c)
    for x in xs[:-1]:
        b.append(arrow(x + 125, 85, x + 170, 85))
    # the loop under the model
    b.append(box(200, 150, 400, 112, 'box-ghost', 12))
    b.append(text(400, 172, 'the training loop, repeated for S steps', 't-note', 'middle'))
    steps = ['1. take a batch of B token windows', '2. predict every next token, cross-entropy loss',
             '3. backpropagate: gradient of the loss', '4. AdamW update, learning rate from the schedule']
    b += para(225, 196, steps, 't-tick')
    b.append(line(532, 120, 532, 150, 'edge', 'ah'))
    b.append(line(470, 150, 470, 120, 'edge', 'ah'))
    b.append(text(400, 290, 'Chapter 1 showed the loss and one small training loop. This chapter is about everything around it, at scale.', 't-muted', 'middle'))
    return svg(760, 305, 'The pretraining recipe: raw text is cleaned and mixed, tokenized, and fed to a Transformer through a training loop, producing a base model.', b)


F['ch4_recipe'] = recipe()


# ---------------------------------------------------------------- 2. the data funnel on real Common Crawl text
def funnel():
    fun = DATA['funnel'] + [['MinHash dedup', DATA['after_dedup']], ['FineWeb-Edu score >= 3', sum(DATA['edu_hist'][3:])]]
    names = {'raw WET records': 'raw pages (one WET file)', 'English': 'English only', 'Gopher rules': '+ Gopher quality/repetition',
             'C4 rules': '+ C4 rules', 'FineWeb rules': '+ FineWeb custom rules'}
    b = [text(20, 24, f'{fun[0][1]:,} real Common Crawl pages through a FineWeb-style pipeline (ch4_data.py)', 't-title')]
    top = fun[0][1]
    for i, (k, v) in enumerate(fun):
        y = 46 + i * 34
        w = max(3, 400 * v / top)
        b.append(text(230, y + 18, names.get(k, k), 't-tick', 'end'))
        b.append(f'<g class="mark"><title>{esc(k)}: {v}</title><rect class="{"s1" if i < 5 else ("s3" if i == 5 else "s4")}" x="240" y="{y + 4}" width="{w:.1f}" height="20" rx="3"/></g>')
        b.append(text(248 + w, y + 18, f'{v:,}  ({v / top:.1%})', 't-val'))
    b.append(text(20, 46 + len(fun) * 34 + 14, 'Each bar is the number of pages still alive after that step. Most raw pages are not English; most English pages fail a quality rule.', 't-muted'))
    return svg(760, 46 + len(fun) * 34 + 26, 'A funnel of document counts through language, quality, deduplication and classifier filters.', b)


F['ch4_funnel'] = funnel()


# ---------------------------------------------------------------- 3. why pages were removed
def reasons():
    items = [(k, v) for k, v in DATA['reasons'] if k != 'not English'][:10]
    b = hbars(20, 44, items, 330, row_h=24, cls='s2', label_w=330, fmt=lambda v: f'{v}', vmax=items[0][1],
              title=f'Why the {DATA["funnel"][1][1]:,} English pages were removed (first rule that fired)')
    return svg(760, 44 + 24 * len(items) + 14, 'Counts of pages removed by each filter rule.', b)


F['ch4_reasons'] = reasons()


# ---------------------------------------------------------------- 4. MinHash banding curve
def minhash_curve():
    cur = DATA['band_curve']
    b, X, Y = chart(70, 64, 560, 220, [('14 bands x 8 rows', [c[0] for c in cur], [c[1] for c in cur], 'l1', 's1')],
                    (0, 1), (0, 1), [0, 0.2, 0.4, 0.6, 0.75, 1.0], [0, 0.25, 0.5, 0.75, 1.0], fx=lambda v: f'{v:g}', fy=lambda v: f'{v:g}',
                    xlabel='true Jaccard similarity of two documents (shared word 5-grams / all word 5-grams)',
                    ylabel='probability the pair is flagged as duplicate', dots=False, end_labels=False)
    b.insert(0, text(20, 24, 'MinHash with 112 hashes in 14 buckets of 8: a soft threshold near 75% similarity', 't-title'))
    for s in [0.6, 0.75, 0.9]:
        p = 1 - (1 - s ** 8) ** 14
        b += [f'<circle class="s2 ring" cx="{X(s):.1f}" cy="{Y(p):.1f}" r="5"/>', text(X(s) + (-10 if s > 0.8 else 10), Y(p) + (16 if s > 0.8 else -6), f'J={s}: {p:.2f}', 't-val', 'end' if s > 0.8 else 'start')]
    b.append(text(650, 120, 'P = 1 - (1 - J^8)^14', 't-math'))
    b.append(text(650, 142, 'all 8 hashes of one', 't-muted'))
    b.append(text(650, 158, 'bucket must match', 't-muted'))
    return svg(820, 334, 'The probability that MinHash with 14 buckets of 8 hashes flags a pair, as a function of their Jaccard similarity.', b)


F['ch4_minhash'] = minhash_curve()


# ---------------------------------------------------------------- 5. FineWeb-Edu score histogram
def edu_hist():
    h = DATA['edu_hist']
    n = sum(h)
    b = [text(20, 24, f'FineWeb-Edu classifier on our {n} surviving pages: most web text is not "educational"', 't-title')]
    m = max(h)
    for i, v in enumerate(h):
        x = 90 + i * 100
        bh = 180 * v / m
        cls = 's3' if i >= 3 else 's1'
        b.append(f'<g class="mark"><title>score {i}: {v}</title><rect class="{cls}" x="{x}" y="{240 - bh:.1f}" width="70" height="{bh:.1f}" rx="3"/></g>')
        b.append(text(x + 35, 234 - bh, f'{v}', 't-val', 'middle'))
        b.append(text(x + 35, 258, f'score {i}', 't-tick', 'middle'))
    b.append(f'<line class="axis" x1="80" y1="240" x2="690" y2="240"/>')
    b.append(line(375, 50, 375, 250, 'edge-dim'))
    b.append(text(385, 64, f'threshold 3: keep {sum(h[3:])} of {n} ({sum(h[3:]) / n:.1%})', 't-note'))
    b.append(text(20, 284, 'Rounded regression score, 0 = no educational value, 5 = excellent for teaching. FineWeb-Edu keeps scores of 3 and above.', 't-muted'))
    return svg(760, 296, 'Histogram of FineWeb-Edu classifier scores on the pages that survived the rule-based filters.', b)


F['ch4_edu_hist'] = edu_hist()


# ---------------------------------------------------------------- 6. BPE merges as frames
def bpe_frames():
    steps = TOK['bpe']
    show = [(0, None), (1, steps[0]), (3, steps[2]), (5, steps[4]), (8, steps[7])]
    start = ['l o w </w>', 'l o w e r </w>', 'n e w e s t </w>', 'w i d e s t </w>']
    counts = [5, 2, 6, 3]
    b = [text(20, 24, 'Byte-pair encoding, merge by merge (the toy example of Sennrich et al., run by ch4_tokenizer.py)', 't-title')]
    y = 44
    for k, (n, st) in enumerate(show):
        words = start if st is None else st['vocab']
        title = 'start: every word split into characters' if st is None else f'after merge {n}: "{st["pair"][0]}" + "{st["pair"][1]}" (seen {st["count"]} times)'
        b += frame(20, y, 760, 62, k + 1, title, 'box' if k else 'box-ghost')
        x = 56
        for wds, c in zip(words, counts):
            syms = wds.split()
            for s in syms:
                wpx = max(18, 7.6 * len(s) + 8)
                b += [box(x, y + 34, wpx, 22, 'box-1' if len(s) > 1 and s != '</w>' else 'box', 4), text(x + wpx / 2, y + 49.5, s, 't-tick', 'middle')]
                x += wpx + 2
            b.append(text(x + 2, y + 49.5, f'x{c}', 't-muted'))
            x += 24
        y += 72
    b.append(text(20, y + 8, 'Coloured blocks are merged symbols. After 10 merges, "low", "newest" and "est" are single tokens; real tokenizers do 50,000 to 150,000 merges.', 't-muted'))
    return svg(800, y + 22, 'Five frames showing byte-pair encoding merging the most frequent adjacent pair of symbols step by step.', b)


F['ch4_bpe'] = bpe_frames()


# ---------------------------------------------------------------- 7. token counts of three tokenizers
def tok_counts():
    names = ['GPT-2 (50,257)', 'Qwen2.5 (151,665)', 'TinyStories (4,096)']
    labels = ['English sentence', 'rare word', 'Python code', 'numbers', 'Hindi', 'TinyStories style']
    b = [text(20, 24, 'How many tokens does the same text become? (ch4_tokenizer.py)', 't-title')]
    cls = ['s1', 's2', 's3']
    y = 50
    for row, lab in zip(TOK['real'], labels):
        b.append(text(170, y + 22, lab, 't-tick', 'end'))
        b.append(text(170, y + 37, f'{len(row["text"])} characters', 't-muted', 'end'))
        for j, n in enumerate(names):
            v = len(row[n])
            b.append(f'<g class="mark"><title>{esc(n)}: {v} tokens</title><rect class="{cls[j]}" x="180" y="{y + j * 15:.1f}" width="{v * 9:.1f}" height="12" rx="2"/></g>')
            b.append(text(186 + v * 9, y + j * 15 + 10, str(v), 't-tick'))
        y += 58
    for j, n in enumerate(names):
        b += [f'<rect class="{cls[j]}" x="{200 + j * 190}" y="{y + 4}" width="10" height="10" rx="2"/>', text(216 + j * 190, y + 13, n, 't-tick')]
    return svg(760, y + 30, 'Token counts for six strings under the GPT-2, Qwen2.5 and a 4,096-token TinyStories tokenizer.', b)


F['ch4_tokcount'] = tok_counts()


# ---------------------------------------------------------------- 8. the forward pass with shapes from our tiny GPT
def forward():
    b = [text(20, 24, 'One forward pass of the tiny GPT, with the real shapes (B = 32 windows, T = 256 tokens, d = 384, V = 4,096)', 't-title')]
    y = 60
    stages = [('token ids', '32 x 256', 'integers 0..4095', 'box'),
              ('embeddings', '32 x 256 x 384', 'token + position', 'box-1'),
              ('6 blocks', '32 x 256 x 384', 'attention + MLP', 'box-2'),
              ('logits', '32 x 256 x 4096', 'a score per token', 'box-3'),
              ('loss', '1 number', 'mean cross-entropy', 'box-4')]
    x = 20
    for i, (t, s, d, c) in enumerate(stages):
        b += block(x, y, 132, 76, t, [s, d], c)
        if i < 4:
            b.append(arrow(x + 132, y + 38, x + 150, y + 38))
        x += 150
    # inside one block
    b.append(box(170, 170, 420, 120, 'box-ghost', 12))
    b.append(text(380, 192, 'inside each block (pre-norm, residual)', 't-note', 'middle'))
    b += block(190, 205, 180, 70, 'x + Attention(LN(x))', ['mixes information', 'across positions (causal)'], 'box-2', 't-tick', 't-muted')
    b += block(390, 205, 180, 70, 'x + MLP(LN(x))', ['transforms each position', 'on its own (384 to 1536)'], 'box-2', 't-tick', 't-muted')
    b.append(arrow(370, 240, 390, 240))
    b.append(line(390, 136, 380, 170, 'edge-dim'))
    b.append(text(380, 318, 'Chapter 1 explained logits, softmax and cross-entropy. Pretraining just does this on trillions of tokens.', 't-muted', 'middle'))
    return svg(780, 330, 'The forward pass of a small GPT: token ids become embeddings, pass through six blocks, become logits and then one loss number.', b)


F['ch4_forward'] = forward()


# ---------------------------------------------------------------- 9. where 6N comes from
def six_n():
    N = PREP['params']
    b = [text(20, 24, 'Why training costs about 6 N FLOPs per token', 't-title')]
    parts = [('forward', '2N', 'each weight: one multiply + one add', 'box-1', 2),
             ('backward: gradient for inputs', '2N', 'pass the error back through each layer', 'box-2', 2),
             ('backward: gradient for weights', '2N', 'how each weight should change', 'box-2', 2)]
    x = 20
    for t, f, d, c, k in parts:
        w = 230
        b += [box(x, 50, w, 70, c, 10), text(x + w / 2, 74, f'{t}: {f}', 't-note', 'middle'), text(x + w / 2, 96, d, 't-muted', 'middle')]
        x += w + 12
    b += brace(20, x - 12, 136, 'total per token: 2N + 2N + 2N = 6N FLOPs', 'edge', False, 't-note')
    b.append(text(20, 190, f'Worked: our tiny GPT has N = {N:,} parameters and saw D = {1500 * 32 * 256:,} tokens.', 't-tick'))
    b.append(text(20, 210, f'C = 6 x {N:,} x {1500 * 32 * 256:,} = {COMP["runs"][0]["C"]:.3g} FLOPs, about {COMP["runs"][0]["C"] / COMP["tiny_flops_s"] / 60:.1f} minutes at the {COMP["tiny_flops_s"] / 1e12:.2f} TFLOP/s the laptop GPU reached.', 't-tick'))
    return svg(760, 225, 'The forward pass costs 2N FLOPs per token and the backward pass about 4N, for a total of 6N.', b)


F['ch4_sixn'] = six_n()


# ---------------------------------------------------------------- 10. compute of famous runs, log scale
def compute_bars():
    runs = COMP['runs']
    b = [text(20, 24, 'Training compute C = 6ND, on a log scale (each grid line is 1,000x more)', 't-title')]
    lo, hi = 14, 26
    X = lambda c: 260 + (math.log10(c) - lo) / (hi - lo) * 430
    for e in range(lo, hi + 1, 3):
        b += [f'<line class="grid" x1="{X(10 ** e):.1f}" y1="40" x2="{X(10 ** e):.1f}" y2="{48 + 30 * len(runs)}"/>', text(X(10 ** e), 60 + 30 * len(runs), f'1e{e}', 't-tick', 'middle')]
    for i, r in enumerate(runs):
        y = 48 + i * 30
        b.append(text(250, y + 15, r['name'].replace(' (ch4_pretrain.py)', ''), 't-tick', 'end'))
        w = X(r['C']) - 260
        b.append(f'<g class="mark"><title>{esc(r["name"])}: {r["C"]:.3g} FLOPs</title><rect class="{"s3" if i == 0 else "s1"}" x="260" y="{y + 3}" width="{w:.1f}" height="17" rx="3"/></g>')
        b.append(text(266 + w, y + 16, f'{r["C"]:.2g}', 't-val'))
    b.append(text(20, 84 + 30 * len(runs), f'Llama 3 405B used about {runs[4]["C"] / runs[0]["C"] / 1e9:.0f} billion times the compute of our laptop run.', 't-muted'))
    return svg(760, 96 + 30 * len(runs), 'Bars of training compute for our tiny model and five well-known models, on a log scale.', b)


F['ch4_compute'] = compute_bars()


# ---------------------------------------------------------------- 11. tokens per parameter
def tpp():
    runs = COMP['runs'][1:]
    b = [text(20, 24, 'Tokens seen per parameter: from "Kaplan-style" to "Chinchilla-optimal" to "over-trained"', 't-title')]
    lo, hi = 0, 5
    X = lambda v: 220 + (math.log10(v) - lo) / (hi - lo) * 480
    for e in range(lo, hi + 1):
        b += [f'<line class="grid" x1="{X(10 ** e):.1f}" y1="40" x2="{X(10 ** e):.1f}" y2="{46 + 30 * len(runs)}"/>', text(X(10 ** e), 60 + 30 * len(runs), f'{10 ** e:,}', 't-tick', 'middle')]
    b.append(line(X(20), 36, X(20), 46 + 30 * len(runs), 'key-line'))
    b.append(text(X(20) + 6, 44, 'about 20 (Chinchilla)', 't-tick'))
    for i, r in enumerate(runs):
        y = 50 + i * 30
        b.append(text(210, y + 15, r['name'], 't-tick', 'end'))
        b.append(f'<g class="mark"><title>{esc(r["name"])}: {r["tpp"]:,.0f}</title><rect class="s1" x="220" y="{y + 3}" width="{X(r["tpp"]) - 220:.1f}" height="17" rx="3"/></g>')
        b.append(text(X(r['tpp']) + 6, y + 16, f'{r["tpp"]:,.0f}', 't-val'))
    b.append(text(20, 84 + 30 * len(runs), 'Small models meant to be run cheaply are trained far past 20 tokens per parameter: it costs more training, but less at use time.', 't-muted'))
    return svg(760, 96 + 30 * len(runs), 'Tokens per parameter for GPT-3, Chinchilla, Llama 3 and Qwen2.5, on a log scale.', b)


F['ch4_tpp'] = tpp()


# ---------------------------------------------------------------- 12. IsoFLOP slice from the Chinchilla fit
def isoflop():
    iso = COMP['iso']
    Ns = [r[0] for r in iso]; Ls = [r[2] for r in iso]
    b, X, Y = chart(80, 64, 520, 220, [('C = 1e21 FLOPs', Ns, Ls, 'l1', 's1')], (1.5e8, 1.3e10), (2.30, 2.52),
                    [2e8, 5e8, 1e9, 2e9, 5e9, 1e10], [2.32, 2.36, 2.40, 2.44, 2.48, 2.52], xlog=True,
                    fx=lambda v: f'{v / 1e9:g}B', fy=lambda v: f'{v:.2f}', xlabel='model size N (log scale); D = C / 6N is whatever the budget leaves',
                    ylabel='predicted loss L(N, D)', end_labels=False)
    b.insert(0, text(20, 24, 'One compute budget, many ways to spend it (Chinchilla Approach 3 fit, ch4_compute.py)', 't-title'))
    best = min(iso, key=lambda r: r[2])
    b += [f'<circle class="s2 ring" cx="{X(best[0]):.1f}" cy="{Y(best[2]):.1f}" r="7"/>',
          text(X(best[0]), Y(best[2]) + 26, f'lowest: N = {best[0] / 1e9:g}B, D = {best[1] / 1e9:.0f}B', 't-val', 'middle')]
    b.append(text(X(2.2e8), Y(iso[0][2]) - 12, 'too small, too many tokens', 't-muted'))
    b.append(text(X(1e10), Y(iso[-1][2]) - 12, 'too big, too few tokens', 't-muted', 'end'))
    return svg(760, 334, 'An isoFLOP curve: predicted loss for different model sizes at a fixed compute budget of 1e21 FLOPs, with a minimum in the middle.', b)


F['ch4_isoflop'] = isoflop()


# ---------------------------------------------------------------- 13. our laptop scaling runs
def mini_scaling():
    cls = [('l1', 's1'), ('l2', 's2'), ('l3', 's3'), ('l4', 's4'), ('edge-on', 'box-on')]
    ser = []
    for (lc, dc), r in zip(cls, MINI):
        ser.append((f'{r["N"] / 1e6:.1f}M', [p['C'] for p in r['curve']], [p['val'] for p in r['curve']], lc, dc))
    b, X, Y = chart(70, 64, 300, 230, ser, (8e11, 6e14), (2.2, 6.2), [1e12, 1e13, 1e14], [2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6], xlog=True,
                    fx=lambda v: f'1e{int(round(math.log10(v)))}', fy=lambda v: f'{v:g}', xlabel='training compute C = 6ND (FLOPs, log)',
                    ylabel='validation loss', dots=False)
    b.insert(0, text(20, 24, 'Five model sizes, same data and recipe (ch4_scaling_mini.py)', 't-title'))
    # right panel: final loss vs N with the fit
    N, L = FIT['N'], FIT['L']
    fitx = [10 ** (5 + i * 0.1) for i in range(23)]
    b2, X2, Y2 = chart(500, 64, 230, 230, [('measured', N, L, 'l2', 's2')], (8e4, 1.5e7), (2.3, 3.5),
                       [1e5, 1e6, 1e7], [2.4, 2.8, 3.2], xlog=True, fx=lambda v: f'1e{int(round(math.log10(v)))}', fy=lambda v: f'{v:g}',
                       xlabel='non-embedding parameters N (log)', ylabel='final loss after 6.6M tokens', end_labels=False)
    b2.append('<polyline class="edge-dim" points="' + ' '.join(f'{X2(n):.1f},{Y2((FIT["Nc"] / n) ** FIT["alpha"]):.1f}' for n in fitx) + '"/>')
    b += b2
    b.append(text(730, 84, f'dashed fit: L = (Nc/N)^{FIT["alpha"]:.3f}', 't-muted', 'end'))
    return svg(780, 354, 'Left: validation loss against training compute for five model sizes. Right: final loss against parameter count with a fitted power law.', b)


F['ch4_mini_scaling'] = mini_scaling()


# ---------------------------------------------------------------- 14. compute-optimal allocation
def allocation():
    opt = COMP['opt']
    Cs = [o['C'] for o in opt]
    ser = [('N_opt (parameters)', Cs, [o['N'] for o in opt], 'l1', 's1'), ('D_opt (tokens)', Cs, [o['D'] for o in opt], 'l2', 's2')]
    b, X, Y = chart(80, 64, 400, 230, ser, (5e17, 8e25), (3e7, 1e14), [1e18, 1e20, 1e22, 1e24], [1e8, 1e10, 1e12, 1e14], xlog=True, ylog=True,
                    fx=lambda v: f'1e{int(round(math.log10(v)))}', fy=lambda v: f'1e{int(round(math.log10(v)))}',
                    xlabel='compute budget C (FLOPs)', ylabel='compute-optimal size (log)')
    b.insert(0, text(20, 24, 'Grow the model and the data together (Chinchilla Approach 3 fit)', 't-title'))
    for o in [opt[0], opt[3], opt[5]]:
        b.append(text(X(o['C']), Y(o['D']) - 12, f'{o["tpp"]:.0f} tok/param', 't-muted', 'middle'))
    b += para(610, 150, ['N grows as C^0.45', 'D grows as C^0.55', '', 'Approaches 1 and 2 in the', 'paper give C^0.50 for both:', 'about 20 tokens per', 'parameter at every scale.'], 't-muted')
    return svg(800, 344, 'Compute-optimal parameters and tokens grow together as the compute budget grows.', b)


F['ch4_allocation'] = allocation()


# ---------------------------------------------------------------- 15. learning-rate schedules
def lr_fig():
    S = PRE['step']
    b, X, Y = chart(80, 64, 560, 200, [('our run: warmup 100, cosine to 10%', S, PRE['lr'], 'l1', 's1')], (0, 1500), (0, 1.1e-3),
                    [0, 100, 300, 600, 900, 1200, 1500], [0, 2.5e-4, 5e-4, 7.5e-4, 1e-3], fx=lambda v: f'{v:,}', fy=lambda v: f'{v:.1e}' if v else '0',
                    xlabel='training step', ylabel='learning rate', dots=False, end_labels=False)
    # a warmup-stable-decay alternative, drawn for comparison
    wsd = [1e-3 * (s + 1) / 100 if s < 100 else (1e-3 if s < 1200 else 1e-3 * (1 - 0.9 * (s - 1200) / 300)) for s in S]
    b.append(f'<polyline class="l2" style="stroke-dasharray:6 4" points="' + ' '.join(f'{X(s):.1f},{Y(v):.1f}' for s, v in zip(S, wsd)) + '"/>')
    b.insert(0, text(20, 24, 'Learning-rate schedules: warm up, hold high, come down', 't-title'))
    b += [f'<rect class="s1" x="80" y="318" width="10" height="10" rx="2"/>', text(96, 327, 'warmup + cosine decay (used in ch4_pretrain.py, GPT-3, Llama 3)', 't-tick'),
          f'<rect class="s2" x="80" y="338" width="10" height="10" rx="2"/>', text(96, 347, 'warmup-stable-decay (MiniCPM): flat, then a short final decay (drawn for comparison)', 't-tick')]
    b.append(text(X(100) + 8, Y(5e-4), '<- warmup: 100 steps', 't-muted'))
    return svg(760, 360, 'The learning rate of our run rises linearly for 100 steps and then follows a cosine down to 10% of its peak; a warmup-stable-decay schedule is drawn for comparison.', b)


F['ch4_lr'] = lr_fig()


# ---------------------------------------------------------------- 16. the loss curve of our run
def loss_curve():
    S, L = PRE['step'], PRE['loss']
    b, X, Y = chart(80, 64, 560, 260, [('train loss', S, L, 'l1', 's1'), ('validation loss', PRE['val_step'], PRE['val'], 'l2', 's2')],
                    (0, 1500), (1.8, 8.6), [0, 300, 600, 900, 1200, 1500], [2, 3, 4, 5, 6, 7, 8], fx=lambda v: f'{v:,}', fy=lambda v: f'{v:g}',
                    xlabel=f'training step (each step = 8,192 tokens; 1,500 steps = {1500 * 8192 / 1e6:.1f}M tokens)', ylabel='cross-entropy loss (nats per token)')
    b = [p for p in b if 'circle class="s1' not in p]
    b.insert(0, text(20, 24, f'Pretraining the 12M-parameter GPT on TinyStories: {PRE["seconds"] / 60:.1f} minutes on the laptop GPU', 't-title'))
    b.append(f'<line class="base-line" x1="80" y1="{Y(math.log(4096)):.1f}" x2="640" y2="{Y(math.log(4096)):.1f}"/>')
    b.append(text(300, Y(math.log(4096)) - 6, f'ln(4096) = {math.log(4096):.2f}: a uniform guess over the vocabulary', 't-muted'))
    b.append(text(X(1500) - 4, Y(PRE['val'][-1]) - 12, f'{PRE["val"][-1]:.2f}', 't-val', 'end'))
    return svg(780, 374, 'Training and validation loss of the tiny GPT over 1,500 steps, starting near ln(4096) and ending near 2.07.', b)


F['ch4_loss'] = loss_curve()


# ---------------------------------------------------------------- 17. samples at checkpoints
def samples():
    sm = PRE['samples']
    val = dict(zip(PRE['val_step'], PRE['val']))
    pick = ['0', '50', '150', '600', '1500']
    b = [text(20, 24, 'The same prompt, "Once upon a time", at five checkpoints (temperature 0.8, same random seed)', 't-title')]
    y = 42
    import textwrap
    for k, s in enumerate(pick):
        t = sm[s][:250].replace('\ufffd', '?')
        lines = textwrap.wrap(t, 98)[:3]
        h = 34 + 17 * len(lines)
        v = val.get(int(s))
        title = f'step {int(s):,}' + (f'   (validation loss {v:.2f})' if v is not None else '')
        b += frame(20, y, 740, h, k + 1, title, 'box-ghost' if k == 0 else ('box-1' if k == 4 else 'box'))
        b += para(40, y + 44, lines, 't-tick')
        y += h + 8
    return svg(780, y + 4, 'Five text samples from the tiny GPT at steps 0, 50, 150, 600 and 1,500, from random tokens to simple stories.', b)


F['ch4_samples'] = samples()


# ---------------------------------------------------------------- 18. gradient agreement vs batch size
def batch_fig():
    Bs = [r['tokens'] for r in BATCH]; cs = [r['cos'] for r in BATCH]
    b, X, Y = chart(80, 64, 520, 210, [('cosine', Bs, cs, 'l1', 's1')], (200, 40000), (0, 0.26), [256, 1024, 4096, 16384, 32768],
                    [0, 0.05, 0.1, 0.15, 0.2, 0.25], xlog=True, fx=lambda v: f'{v:,}', fy=lambda v: f'{v:.2f}',
                    xlabel='tokens per batch (log scale)', ylabel='cosine similarity of two independent batch gradients', end_labels=False)
    b.insert(0, text(20, 24, 'Bigger batches give gradients that agree more (trained tiny GPT, ch4_batch.py)', 't-title'))
    b.append(line(X(8192), 64, X(8192), 274, 'edge-dim'))
    b.append(text(X(8192) - 6, 80, 'our training batch', 't-muted', 'end'))
    for r in BATCH[-3:]:
        b.append(text(X(r['tokens']), Y(r['cos']) - 10, f'{r["cos"]:.3f}', 't-val', 'middle'))
    return svg(760, 324, 'Cosine similarity between gradients from two independent batches grows with the batch size.', b)


F['ch4_batch'] = batch_fig()


# ---------------------------------------------------------------- 19. floating-point formats
def precision():
    b = [text(20, 24, 'Three ways to store a number in bits: sign, exponent (range), mantissa (precision)', 't-title')]
    y = 46
    for name in ['fp32', 'fp16', 'bf16']:
        p = PREC[name]
        b.append(text(70, y + 20, name, 't-note', 'end'))
        x = 80
        for n, cls, lab in [(1, 'box-4', 'sign'), (p['exp'], 'box-1', 'exponent'), (p['mant'], 'box-2', 'mantissa')]:
            w = n * 14
            b.append(box(x, y + 4, w, 24, cls, 3))
            b.append(text(x + w / 2, y + 20, f'{n}' if n < 3 else f'{lab} {n}', 't-tick', 'middle'))
            x += w + 2
        b.append(text(x + 12, y + 14, f'max {p["max"]:.3g}', 't-tick'))
        b.append(text(x + 12, y + 30, f'step after 1.0: {p["eps"]:.3g}', 't-muted'))
        y += 50
    b += para(20, y + 6, ['fp16 has precision but little range: 70,000 overflows to inf and 1e-8 rounds to 0.',
                          'bf16 keeps fp32\'s range with less precision: 1 + 0.001 rounds to exactly 1.',
                          f'So adding a 1e-4 update to a weight of 1.0 a thousand times gives {PREC["acc_bf16"]:.4f} in bf16 and {PREC["acc_fp32"]:.4f} in fp32.'], 't-tick')
    return svg(820, y + 60, 'Bit layouts of fp32, fp16 and bf16 with their range and precision.', b)


F['ch4_precision'] = precision()


# ---------------------------------------------------------------- 20. memory per parameter
def memory():
    parts = COMP['mem_parts']
    b = [text(20, 24, f'Training memory per parameter with mixed precision and AdamW: {COMP["mem_per"]} bytes', 't-title')]
    x = 20
    cls = ['box-1', 'box-1', 'box-2', 'box-3', 'box-3']
    for (k, v), c in zip(parts.items(), cls):
        w = v * 45
        b += [box(x, 46, w, 44, c, 6), text(x + w / 2, 66, f'{v} B', 't-note', 'middle'), text(x + w / 2, 82, k.split(' (')[0], 't-muted', 'middle')]
        x += w + 4
    rows = [('our tiny GPT', 1.23e7), ('Qwen2.5-0.5B', 0.494e9), ('Llama 3 8B', 8e9), ('Llama 3 405B', 405e9)]
    y = 120
    for name, n in rows:
        gb = n * COMP['mem_per'] / 2 ** 30
        b.append(text(20, y, f'{name}: {n:.3g} parameters x 16 B = {gb:,.1f} GiB, before activations', 't-tick'))
        y += 18
    b.append(text(20, y + 6, 'An 80 GB GPU holds the training state of about a 5B-parameter model at most, so big runs split it across many GPUs.', 't-muted'))
    return svg(760, y + 18, 'Bytes of training memory per parameter: bf16 weights and gradients, fp32 master weights and two Adam moments.', b)


F['ch4_memory'] = memory()


# ---------------------------------------------------------------- 21. in-context learning curves
def icl():
    ks = [0, 1, 2, 4, 8]
    names = list(PROBE['icl'])
    short = {names[0]: 'English to French', names[1]: 'antonyms', names[2]: 'made-up labels (blue/red)'}
    ser = [(short[n], [k + 0.0 for k in ks], [PROBE['icl'][n][str(k)] for k in ks], f'l{i + 1}', f's{i + 1}') for i, n in enumerate(names)]
    xs_pos = {0: 0, 1: 1, 2: 2, 4: 3, 8: 4}
    ser = [(a, [xs_pos[int(k)] for k in xs], ys, l, d) for a, xs, ys, l, d in ser]
    b, X, Y = chart(80, 64, 440, 220, ser, (0, 4), (0, 1.0), [0, 1, 2, 3, 4], [0, 0.25, 0.5, 0.75, 1.0],
                    fx=lambda v: str([0, 1, 2, 4, 8][int(v)]), fy=lambda v: f'{v:.0%}', xlabel='number of solved examples in the prompt (k)',
                    ylabel='accuracy on 28 held-out items')
    b.insert(0, text(20, 24, 'In-context learning in Qwen2.5-0.5B base: no training, only examples in the prompt (ch4_probe.py)', 't-title'))
    return svg(780, 334, 'Accuracy of the base model on three tasks as the number of examples in the prompt grows from 0 to 8.', b)


F['ch4_icl'] = icl()


# ---------------------------------------------------------------- 22. Llama 3 data mix and the phases of pretraining
def phases():
    b = [text(20, 24, 'The phases of a modern pretraining run (numbers from the Llama 3 paper)', 't-title')]
    mix = [('general knowledge', 50, 's1'), ('math + reasoning', 25, 's2'), ('code', 17, 's3'), ('multilingual', 8, 's4')]
    x = 20
    b.append(text(20, 50, 'final data mix (share of tokens)', 't-muted'))
    for name, p, c in mix:
        w = p * 7.2
        b.append(f'<g class="mark"><title>{name}: {p}%</title><rect class="{c}" x="{x:.1f}" y="58" width="{w - 2:.1f}" height="26" rx="3" style="fill-opacity:0.8"/></g>')
        b.append(text(x + w / 2, 100, f'{name} {p}%', 't-tick', 'middle'))
        x += w
    y = 130
    stages = [('1. initial pretraining', 'about 15T tokens, cosine LR, batch 4M to 16M tokens, context 8K', 'box-1', 470),
              ('2. long-context', 'context grown to 128K in steps, ~800B tokens', 'box-3', 150),
              ('3. annealing', 'last 40M tokens: LR to 0, high-quality data upsampled', 'box-2', 100)]
    x = 20
    for t, d, c, w in stages:
        b += [box(x, y, w, 40, c, 8), text(x + 8, y + 25, t, 't-tick')]
        x += w + 5
    b += para(20, y + 66, ['1: ' + stages[0][1], '2: ' + stages[1][1], '3: ' + stages[2][1] + ', then average the checkpoints'], 't-tick')
    b.append(text(20, y + 126, 'Widths are not to scale: annealing is a tiny fraction of the tokens but has an outsized effect on benchmarks.', 't-muted'))
    return svg(760, y + 140, 'The Llama 3 data mix and the three stages of its pretraining: initial pretraining, long-context training and annealing.', b)


F['ch4_phases'] = phases()


for k in ['ch4_memory', 'ch4_reasons']:      # drawn but not used in the text (a table and a code block say the same)
    F.pop(k)
json.dump(F, open('results/figs_ch4.json', 'w'))
print(f'{len(F)} figures:', ', '.join(F))
