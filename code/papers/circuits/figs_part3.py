"""Figures for Part 3 (paths through the model), drawn from results/part3.json and results/part3_toy.json.
Writes results/figs_part3.json (key -> svg string). Colours come only from CSS classes, so every figure follows the
site's dark, dim and light themes."""
import json
from figlib import svg, text, box, arrow, esc
from cfig import token, line

R = json.load(open('results/part3.json'))
T = json.load(open('results/part3_toy.json'))
F = {}


def tok(s):
    """A token for display: drop the leading space (every token in these figures is a whole word or a comma)."""
    if s == '<|BOS|>':
        return 'BOS'
    return s.strip()


def vec(x, y, vals, cls, cw=46, ch=28):
    """A short row vector drawn as cells with their numbers."""
    b = []
    m = max(1e-9, max(abs(v) for v in vals))
    for j, v in enumerate(vals):
        op = 0.10 + 0.55 * abs(v) / m
        b.append(f'<rect class="{cls}" x="{x + j * cw:.1f}" y="{y:.1f}" width="{cw - 4}" height="{ch}" rx="4" style="fill-opacity:{op:.2f}"/>')
        b.append(text(x + j * cw + (cw - 4) / 2, y + ch / 2 + 5, f'{v:g}', 't-val', 'middle'))
    return b


def dbars(x0, y0, items, half, row_h=26, label_w=190, vmax=None, fmt=lambda v: f'{v:+.3f}', hi=()):
    """Diverging horizontal bars around a zero line: items = [(label, value)]. Positive s1, negative s2."""
    vmax = vmax or max(abs(v) for _, v in items)
    zx = x0 + label_w + half
    b = [f'<line class="axis" x1="{zx:.1f}" y1="{y0 - 4:.1f}" x2="{zx:.1f}" y2="{y0 + row_h * len(items):.1f}"/>']
    for i, (lab, v) in enumerate(items):
        y = y0 + i * row_h
        w = max(2, half * abs(v) / vmax)
        xx = zx if v >= 0 else zx - w
        b.append(text(x0 + label_w - 10, y + row_h / 2 + 4, lab, 't-tick t-strong' if lab in hi else 't-tick', 'end'))
        b.append(f'<g class="mark"><title>{esc(lab)}: {fmt(v)}</title><rect class="{"s1" if v >= 0 else "s2"}" x="{xx:.1f}" y="{y + 4:.1f}" width="{w:.1f}" height="{row_h - 9:.1f}" rx="3"/></g>')
        tx = zx + w + 6 if v >= 0 else zx - w - 6
        b.append(text(tx, y + row_h / 2 + 4, fmt(v), 't-val', 'start' if v >= 0 else 'end'))
    return b


# ---------------------------------------------------------------- 1. the toy: product = sum of paths
def toy():
    B = T['B']
    b = [text(24, 26, 'A toy model on the text "a b": the product and the sum of paths give the same logits', 't-title')]
    # left: the product, one layer after the other
    b.append(text(24, 62, 'The product: layer after layer (position 2)', 't-note'))
    y = 80
    steps = [('x0 = W_E t', B['x0'][1], 'box'), ('x1 = x0 + head', B['x1'][1], 'box'), ('logits = W_U x1', B['prod'][1], 'box-3')]
    for k, (lab, v, cls) in enumerate(steps):
        yy = y + k * 66
        b += token(24, yy, 150, lab, cls, 30)
        b += vec(190, yy + 1, v, 's3' if k == 2 else 's1')
        if k < 2:
            b.append(arrow(99, yy + 34, 99, yy + 62))
    b.append(text(24, 300, 'the head at position 2 reads a and b, half each,', 't-muted'))
    b.append(text(24, 318, 'and swaps their numbers: (0.5, 0.5)', 't-muted'))
    # right: the sum of two paths
    X = 400
    b.append(text(X, 62, 'The sum: one term per end-to-end path', 't-note'))
    b += token(X, 80, 170, 'direct: W_U W_E t', 'box-1', 30)
    b += vec(X + 186, 81, B['direct'][1], 's1')
    b.append(text(X, 132, 'the current token b, straight to the logits', 't-muted'))
    b += token(X, 150, 170, 'head: A ⊗ W_U W_OV W_E', 'box-2', 30)
    b += vec(X + 186, 151, B['head'][1], 's2')
    b.append(text(X, 202, '0.5 × column a + 0.5 × column b of the OV table', 't-muted'))
    b.append(line(X, 222, X + 330, 222, 'axis'))
    b += token(X, 236, 170, 'sum', 'box-3', 30)
    b += vec(X + 186, 237, [a + c for a, c in zip(B['direct'][1], B['head'][1])], 's3')
    b.append(text(X, 290, 'the same logits as the product', 't-muted'))
    b.append(text(X, 318, 'columns of each vector: logit of a, of b, of c', 't-tick'))
    return svg(760, 336, 'A toy one-layer model with tokens a, b, c on the text "a b", at position 2. Left, the product: the embedding gives (0, 1), the head adds (0.5, 0.5) so the stream is (0.5, 1.5), and the unembedding gives logits (0.5, 1.5, 0). Right, the sum of paths: the direct path gives (0, 1, 0), the head term gives (0.5, 0.5, 0), and their sum is (0.5, 1.5, 0), the same.', b)


F['p3_toy'] = toy()


# ---------------------------------------------------------------- 2. the real terms and their share of the " tea" logit
def terms():
    D = R['terms']
    names = {'direct path (token)': 'direct path', 'direct path (position)': 'position, direct', 'heads (positions)': 'position, via heads',
             'biases b_O and b_U': 'biases'}
    items = [(names.get(k, k.replace(' (tokens)', '')), v['tea']) for k, v in D.items()]
    b = [text(24, 26, 'attn-only-1l at the last "green": how much each path adds to the logit of "tea"', 't-title'),
         text(24, 48, 'the first 9 rows are the paper\'s terms; the last 3 are parts the paper leaves out', 't-muted')]
    b += dbars(24, 66, items, 230, vmax=6.0, hi=('direct path', 'head 0.3'))
    y = 66 + 26 * len(items) + 18
    b.append(text(24, y, f'sum of all 12 rows = the model\'s logit of "tea", {R["tea_logit"]:.3f}', 't-note'))
    desc = ', '.join(f'{lab} {v:+.3f}' for lab, v in items)
    return svg(760, y + 16, f'Diverging bars: how much each path term of attn-only-1l adds to the logit of tea at the last position: {desc}. They add up to {R["tea_logit"]:.3f}, the model logit.', b)


F['p3_terms'] = terms()


# ---------------------------------------------------------------- 3. one head term in tokens
def head3():
    S = R['head3_sources']
    b = [text(24, 26, 'Head 0.3 at the last "green": weight × OV column value = what each source adds to "tea"', 't-title')]
    cols = [(24, 'source token'), (170, 'weight A'), (290, 'OV column, tea row'), (450, 'adds to "tea"')]
    for x, lab in cols:
        b.append(text(x, 56, lab, 't-note'))
    y0, rh = 68, 26
    mx = max(abs(d['contrib']) for d in S)
    for k, d in enumerate(S):
        y = y0 + k * rh
        hi = d['tok'] == ' tea'
        b.append(text(24, y + 17, tok(d['tok']) + f'  (j={k})', 't-tick t-strong' if hi else 't-tick'))
        b.append(f'<rect class="cell" x="170" y="{y + 4}" width="{max(2, 100 * d["a"]):.1f}" height="{rh - 9}" rx="3" style="fill-opacity:0.7"/>')
        b.append(text(176 + max(2, 100 * d['a']), y + 17, f'{d["a"]:.3f}', 't-val'))
        b.append(text(290, y + 17, f'{d["col_tea"]:+.2f}', 't-val'))
        w = max(2, 230 * abs(d['contrib']) / mx)
        cls = 's1' if d['contrib'] >= 0 else 's2'
        b.append(f'<rect class="{cls}" x="450" y="{y + 4}" width="{w:.1f}" height="{rh - 9}" rx="3"/>')
        b.append(text(456 + w, y + 17, f'{d["contrib"]:+.3f}', 't-val'))
    y = y0 + len(S) * rh + 22
    b.append(text(24, y, f'sum {sum(d["contrib"] for d in S):+.3f}: the head 0.3 term of the expansion. Almost all of it comes from "tea" itself.', 't-note'))
    return svg(760, y + 14, 'A table for head 0.3 at the last position. For each of the 10 source tokens: the attention weight, the value of its OV column in the row for tea, and their product. Tea has weight 0.370 and column value +12.15, so it adds +4.497. Drink adds +0.979, BOS -0.167, the rest almost nothing. The total is +5.465.', b)


F['p3_head3'] = head3()


# ---------------------------------------------------------------- 4. the freezing trick: run twice
def freeze():
    Fz = R['freeze']
    b = [text(24, 26, 'The freezing trick: run the model twice', 't-title')]
    b += token(24, 50, 190, 'run 1: real text', 'box', 32)
    b.append(arrow(218, 66, 262, 66))
    b += token(266, 50, 220, 'record A of every head', 'box-1', 32)
    b.append(text(496, 71, 'this run uses only', 't-muted'))
    b.append(text(496, 89, 'the QK circuits', 't-muted'))
    b += token(24, 124, 190, 'run 2: any tokens', 'box', 32)
    b.append(arrow(218, 140, 262, 140))
    b += token(266, 124, 220, 'heads use the recorded A', 'box-2', 32)
    b.append(arrow(490, 140, 534, 140))
    b += token(538, 124, 196, 'logits: linear in tokens', 'box-3', 32)
    b.append(line(376, 86, 376, 120, 'path', 'ah-on'))
    y = 196
    b.append(text(24, y, 'Test on attn-only-1l: is g(a + b) = g(a) + g(b)?  (g = the part of the logits that depends on the tokens)', 't-note'))
    rows = [('patterns frozen', Fz['frozen']['sum'], 's1'), ('patterns recomputed', Fz['not frozen']['sum'], 's2')]
    for k, (lab, v, cls) in enumerate(rows):
        yy = y + 18 + k * 34
        b.append(text(24, yy + 18, lab, 't-tick'))
        w = max(3, 400 * v / 17.0)
        b.append(f'<rect class="{cls}" x="190" y="{yy + 5}" width="{min(w, 400):.1f}" height="20" rx="3"/>')
        b.append(text(200 + min(w, 400), yy + 19, f'largest error {v:.1e}', 't-val'))
    b.append(text(24, y + 100, f'typical size of g(a): {Fz["frozen"]["size"]:.2f}. Frozen, the error is rounding noise; recomputed, it is larger than the logits.', 't-muted'))
    return svg(760, y + 116, 'A diagram of the freezing trick. Run 1 on real text records the attention pattern A of every head, which depends only on the QK circuits. Run 2 on any tokens reuses the recorded patterns, and the logits become a linear function of the tokens. Below, a test on attn-only-1l: with frozen patterns, g(a + b) and g(a) + g(b) differ by at most 0.00004; with recomputed patterns they differ by up to 17.', b)


F['p3_freeze'] = freeze()


# ---------------------------------------------------------------- 5. the footnote: one row depends on every source
def row():
    t1, t2 = R['tokens'], R['tokens_d']
    r1, r2 = R['row_before'], R['row_after']
    b = [text(24, 26, 'Head 0.3, last row of the pattern: change one source token, every weight moves', 't-title')]
    cw, x0 = 66, 110
    for k, (toks, rr, lab) in enumerate([(t1, r1, 'before'), (t2, r2, 'after')]):
        y = 50 + k * 92
        b.append(text(24, y + 40, lab, 't-note'))
        for j, (t, v) in enumerate(zip(toks, rr)):
            x = x0 + j * cw
            changed = t1[j] != t2[j]
            b.append(text(x + cw / 2 - 3, y + 14, tok(t), 't-tick t-strong' if changed else 't-tick', 'middle'))
            b.append(f'<g class="mark"><title>{esc(tok(t))}: {v:.3f}</title><rect class="cell" x="{x:.1f}" y="{y + 22}" width="{cw - 6}" height="34" rx="3" style="fill-opacity:{max(0.05, v):.3f}"/></g>')
            b.append(text(x + cw / 2 - 3, y + 44, f'{v:.2f}', 't-cell' + (' on' if v > 0.55 else ''), 'middle'))
    b.append(text(24, 246, 'Only "drink" became "want", and the head now ignores it. The scores of "tea" and BOS did not change,', 't-muted'))
    b.append(text(24, 264, 'but the softmax shares out the freed weight: "tea" goes from 0.37 to 0.43.', 't-muted'))
    return svg(780, 278, 'Two rows of attention weights of head 0.3 from the last token. Before: BOS 0.37, tea 0.37, drink 0.15, comma 0.09, the rest near zero. After drink is replaced by want: BOS 0.43, tea 0.43, want 0.00. The scores of tea and BOS are unchanged, but their weights grow because the softmax divides by a smaller total.', b)


F['p3_row'] = row()


# ---------------------------------------------------------------- 6. the two-layer model in outline
def two():
    Tw = R['two']
    G = Tw['groups']
    names = {'direct path': 'direct path (1)', 'layer-0 heads': 'layer-0 heads (8)', 'layer-1 heads': 'layer-1 heads (8)',
             'virtual heads': 'virtual heads (64)', 'positions and biases': 'positions and biases'}
    items = [(names[k], v['pred']) for k, v in G.items()]
    b = [text(24, 26, f'attn-only-2l at the last "green": what each group of paths adds to the logit of "{tok(Tw["pred"])}"', 't-title'),
         text(24, 48, 'the paper\'s 81 terms in four groups, plus the parts the paper leaves out', 't-muted')]
    b += dbars(24, 66, items, 250, row_h=30, vmax=16.0, hi=('layer-1 heads (8)',))
    y = 66 + 30 * len(items) + 18
    b.append(text(24, y, f'sum: {Tw["logit"]:+.3f} = the model\'s logit. Part 5 takes the groups apart.', 't-note'))
    desc = ', '.join(f'{lab} {v:+.3f}' for lab, v in items)
    return svg(760, y + 16, f'Diverging bars for attn-only-2l at the last position: the logit of tea split into groups of path terms: {desc}.', b)


F['p3_two'] = two()

json.dump(F, open('results/figs_part3.json', 'w'))
print('wrote', len(F), 'figures:', ', '.join(F))
