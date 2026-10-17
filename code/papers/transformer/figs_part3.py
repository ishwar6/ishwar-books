"""SVG figures for Part 3, drawn from results/part3.json (run tx_part3.py first). Writes results/figs_part3.json."""
import json, os
from figlib import svg, heatmap, text

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'part3.json')))
F = {}


def clean(t):
    return t.replace('▁', '').replace('</s>', 'end').replace('<pad>', 'start')


def fig(key, M, rows, cols, label, mask_upper=False, cell=26):
    lw = 70
    w = 20 + lw + cell * len(cols) + 20
    h = 40 + cell * len(rows) + 70
    body = heatmap(20, 40, M, [clean(r) for r in rows], [clean(c) for c in cols], cell, mask_upper=mask_upper, label_w=lw)
    body.append(text(20 + lw - 8, 26, 'query', 't-muted', 'end'))
    F[key] = svg(w, h, label, body)


d = R['dec_self']
fig('p3_causal', d['W'], d['tokens'], d['tokens'],
    'Decoder self-attention weights of the last decoder layer, mean of 8 heads: every cell above the diagonal is empty.',
    mask_upper=True)
c = R['cross']
fig('p3_cross', c['W'], c['tgt'], c['src'],
    'Cross-attention weights of decoder layer 3, mean of 8 heads: German queries on the rows, English keys on the columns.')


# ---------------------------------------------------------------- figures of the code walkthroughs (tx_part3b.py)
from figlib import box
from samfig import line
B = json.load(open(os.path.join(HERE, 'results', 'part3b.json')))


def grid(x0, y0, M, cell, title, kind):
    """One 5 x 5 table with the value in each cell. kind: 'scores' (shade by rank), 'masked' or 'weights'."""
    n = len(M)
    b = [text(x0 + cell * n / 2, y0 - 12, title, 't-title', 'middle')]
    flat = [v for r in M for v in r]
    lo, hi = min(flat), max(flat)
    for i in range(n):
        b.append(text(x0 - 8, y0 + i * cell + cell / 2 + 4, f'q{i + 1}', 't-tick', 'end'))
        for j in range(n):
            x, y = x0 + j * cell, y0 + i * cell
            v = M[i][j]
            if kind == 'masked' and j > i:
                b.append(f'<rect class="cell-masked" x="{x + 1:.1f}" y="{y + 1:.1f}" width="{cell - 2:.1f}" height="{cell - 2:.1f}" rx="2"/>')
                b.append(text(x + cell / 2, y + cell / 2 + 4, '-inf', 't-cell', 'middle'))
                continue
            op = v if kind == 'weights' else 0.08 + 0.75 * (v - lo) / (hi - lo)
            b.append(f'<g class="mark"><title>query {i + 1}, key {j + 1}: {v:.3f}</title><rect class="cell" x="{x + 1:.1f}" '
                     f'y="{y + 1:.1f}" width="{cell - 2:.1f}" height="{cell - 2:.1f}" rx="2" style="fill-opacity:{max(0.04, op):.3f}"/></g>')
            b.append(text(x + cell / 2, y + cell / 2 + 4, f'{v:.2f}', 't-cell' + (' on' if op > 0.55 else ''), 'middle'))
    for j in range(n):
        b.append(text(x0 + j * cell + cell / 2, y0 + n * cell + 16, f'k{j + 1}', 't-tick', 'middle'))
    return b


def mask_steps():
    m = B['mask']
    cell, gap = 44, 70
    W = 30 + 3 * 5 * cell + 2 * gap + 20
    b = []
    b += grid(40, 50, m['scores'], cell, '1. scores Q K^T / sqrt(d_k)', 'scores')
    masked = [[v if j <= i else 0.0 for j, v in enumerate(r)] for i, r in enumerate(m['scores'])]
    b += grid(40 + 5 * cell + gap, 50, masked, cell, '2. future set to -inf', 'masked')
    b += grid(40 + 2 * (5 * cell + gap), 50, m['weights'], cell, '3. softmax of each row', 'weights')
    for t in range(2):
        x = 40 + (t + 1) * 5 * cell + t * gap
        b.append(line(x + 8, 50 + 2.5 * cell, x + gap - 32, 50 + 2.5 * cell, 'edge-on', 'ah-on'))
    b.append(text(20, 50 + 5 * cell + 46, 'Row i is query i, column j is key j. Each weight row adds up to 1, and every future weight is exactly 0.', 't-muted'))
    return svg(W, 50 + 5 * cell + 60, 'Three steps of masked attention on 5 tokens: the scores, the scores with every future cell set to minus '
               'infinity, and the weights after the softmax, where the future cells are zero and each row adds up to 1.', b)


F['p3_mask_steps'] = mask_steps()


def split_heads_fig():
    """One token vector of d_model = 16 numbers cut into h = 4 heads of d_k = 4, then heads moved in front of tokens."""
    cw, ch = 22, 22
    b = [text(20, 24, 'Split heads: (batch, n, d_model) -> (batch, n, h, d_k) -> (batch, h, n, d_k)', 't-title')]
    cls = ['box-1', 'box-2', 'box-3', 'box-4']
    # left: n = 5 tokens, each a row of 16 numbers coloured by head
    x0, y0 = 30, 60
    b.append(text(x0, y0 - 12, 'Q: 5 tokens x 16 numbers', 't-note'))
    for i in range(5):
        b.append(text(x0 - 6, y0 + i * ch + 15, f't{i + 1}', 't-tick', 'end'))
        for j in range(16):
            b.append(box(x0 + j * cw, y0 + i * ch, cw - 2, ch - 2, cls[j // 4], 3))
    for hh in range(4):
        b.append(text(x0 + (hh * 4 + 2) * cw - 1, y0 + 5 * ch + 16, f'head {hh + 1}', 't-tick', 'middle'))
    b.append(text(x0, y0 + 5 * ch + 40, '.view(batch, n, h, d_k) cuts each row', 't-muted'))
    b.append(text(x0, y0 + 5 * ch + 56, 'into 4 pieces; no number moves.', 't-muted'))
    # arrow
    ax = x0 + 16 * cw + 20
    b.append(line(ax, y0 + 2.5 * ch, ax + 60, y0 + 2.5 * ch, 'edge-on', 'ah-on'))
    b.append(text(ax + 30, y0 + 2.5 * ch - 10, 'transpose(1, 2)', 't-tick', 'middle'))
    # right: 4 small tables, one per head, each 5 tokens x 4 numbers
    x1 = ax + 80
    for hh in range(4):
        xx = x1 + hh * (4 * cw + 18)
        b.append(text(xx + 2 * cw, y0 - 12, f'head {hh + 1}', 't-note', 'middle'))
        for i in range(5):
            for j in range(4):
                b.append(box(xx + j * cw, y0 + i * ch, cw - 2, ch - 2, cls[hh], 3))
    b.append(text(x1, y0 + 5 * ch + 40, 'Each head is now its own 5 x 4 table: its own', 't-muted'))
    b.append(text(x1, y0 + 5 * ch + 56, 'queries for the same 5 tokens. Attention runs on each.', 't-muted'))
    W = x1 + 4 * (4 * cw + 18) + 10
    return svg(W, y0 + 5 * ch + 70, 'The split of multi-head attention. On the left, 5 tokens with 16 numbers each, coloured in 4 groups of 4. '
               'View cuts each row into 4 pieces. Transpose then makes 4 separate tables, one per head, each with 5 tokens of 4 numbers.', b)


F['p3_split'] = split_heads_fig()
json.dump(F, open(os.path.join(HERE, 'results', 'figs_part3.json'), 'w'))
print('figures:', ', '.join(F))
