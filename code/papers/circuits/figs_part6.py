"""Figures for Part 6 (induction heads), drawn from results/part6.json and results/part6_toy.json.
Writes results/figs_part6.json (key -> svg string). Colours come only from CSS classes, so every figure follows the
site's dark, dim and light themes."""
import json
from figlib import svg, text, box, arrow, heatmap, lines_chart, esc
from cfig import token, row, line, hbars

R = json.load(open('results/part6.json'))
T = json.load(open('results/part6_toy.json'))
F = {}
PT, IH = R['prev_head'], R['ind_head']


# ---------------------------------------------------------------- 1. the algorithm, [a][b] ... [a] -> [b]
def algorithm():
    b = [text(24, 26, 'An induction head on "... Mrs Brindlewick ... Mrs Br" (two steps, two layers)', 't-title')]
    words = ['Mrs', 'Br', 'indle', 'wick', '...', 'Mrs', 'Br', '?']
    cls = ['box', 'box', 'box-2', 'box', 'box-ghost', 'box', 'box-1', 'box-ghost']
    y = 150
    parts, cx = row(40, y, words, w=78, gap=14, classes=cls)
    b += parts
    # layer 0: the previous-token head writes "the token before me" into each position
    b.append(text(40, 66, 'Layer 0, previous-token head: each token gets a note "the token before me was ...".', 't-note'))
    b.append(text(cx[2] + 46, y - 22, 'note: "Br"', 't-s2', 'start'))
    b.append(f'<path class="key-line" d="M{cx[1]:.1f},{y - 4} C{cx[1]:.1f},{y - 30} {cx[2]:.1f},{y - 34} {cx[2]:.1f},{y - 3}" marker-end="url(#ah-2)"/>')
    b.append(text(cx[6], y - 16, 'query: "Br"', 't-s1', 'middle'))
    # layer 1: the induction head
    yb = y + 30
    b.append(f'<path class="path" d="M{cx[6]:.1f},{yb + 4} C{cx[6]:.1f},{yb + 70} {cx[2]:.1f},{yb + 70} {cx[2]:.1f},{yb + 8}" marker-end="url(#ah-on)"/>')
    b.append(text((cx[2] + cx[6]) / 2, yb + 74, 'Layer 1, induction head: the query "Br" matches the note "Br", so it attends to "indle"', 't-on', 'middle'))
    b.append(text((cx[2] + cx[6]) / 2, yb + 96, 'and its OV circuit copies "indle": the prediction for "?" is "indle".', 't-note', 'middle'))
    return svg(800, 300, 'Two steps of an induction head. In layer 0 a previous-token head writes into the position of indle a note that the token before it was Br. In layer 1, the induction head standing on the second Br has the query Br; it matches the note at indle, attends there, and copies indle as the prediction.', b)


F['p6_algorithm'] = algorithm()


# ---------------------------------------------------------------- 2. loss per position on repeated random tokens
def loss_curve():
    c = R['rep']['curves']
    xs = list(range(1, 101))
    sm = lambda v: [sum(v[max(0, i - 2):i + 3]) / len(v[max(0, i - 2):i + 3]) for i in range(len(v))]
    series = [('one layer', sm(c['attn-only-1l']), 'l2', 's2'), ('two layers', sm(c['attn-only-2l']), 'l1', 's1')]
    # lines_chart draws a dot for every point; keep the data, thin the dots by sampling every 3rd point
    keep = [i for i in range(100) if i % 3 == 0 or i == 99]
    xs2 = [xs[i] for i in keep]
    series2 = [(n, [v[i] for i in keep], lc, dc) for n, v, lc, dc in series]
    return lines_chart(800, 330, 'Loss at each position of 50 random tokens repeated twice, averaged over 100 sequences and smoothed over 5 positions. The one-layer model stays near 15 nats on both copies. The two-layer model drops from about 15 to about 3 nats as soon as the second copy starts at position 51.',
                       xs2, series2, 'position in the sequence (the second copy starts at 51)', 'loss (nats)', 0, 20, [0, 5, 10, 15, 20],
                       xticks=[1, 25, 51, 75, 100], fmt=lambda v: f'{v:.1f}' if isinstance(v, float) else f'{v}', right=150)


F['p6_loss'] = loss_curve()


# ---------------------------------------------------------------- 3. the two attention patterns
def patterns():
    s = R['short']
    lab = ['BOS'] + [f't{i}' for i in range(1, 9)] * 2
    cell = 17
    b = []
    b += heatmap(10, 50, s['prev'], lab, lab, cell, title=f'head 0.{PT} (previous token)', mask_upper=True, label_w=40)
    b += heatmap(410, 50, s['ind'], lab, lab, cell, title=f'head 1.{IH} (induction)', mask_upper=True, label_w=40)
    return svg(800, 400, f'Attention patterns of head 0.{PT} and head 1.{IH} on BOS followed by 8 random tokens t1 to t8, repeated. Rows are destinations and columns are sources. Head 0.{PT} puts weight just left of the diagonal: the previous token. Head 1.{IH} puts weight on BOS during the first copy, and on a stripe 7 positions back during the second copy: the token after the earlier copy of the present token.', b)


F['p6_patterns'] = patterns()


# ---------------------------------------------------------------- 4. the scatter: OV positivity vs K-composition positivity
def scatter():
    W, H, L, Tp = 800, 380, 90, 30
    pw, ph = 420, 300
    X = lambda v: L + (v + 1) / 2 * pw
    Y = lambda v: Tp + ph - (v + 1) / 2 * ph
    b = []
    for t in [-1, -0.5, 0, 0.5, 1]:
        b.append(f'<line class="grid" x1="{L}" y1="{Y(t):.1f}" x2="{L + pw}" y2="{Y(t):.1f}"/>')
        b.append(f'<line class="grid" x1="{X(t):.1f}" y1="{Tp}" x2="{X(t):.1f}" y2="{Tp + ph}"/>')
        b.append(text(L - 8, Y(t) + 4, f'{t:+.1f}' if t else '0', 't-tick', 'end'))
        b.append(text(X(t), Tp + ph + 18, f'{t:+.1f}' if t else '0', 't-tick', 'middle'))
    b.append(text(L + pw / 2, Tp + ph + 40, 'OV circuit positivity (copying)', 't-tick', 'middle'))
    b.append(text(L - 80, Tp - 12, 'QK term with head 0.%d (same matching)' % PT, 't-tick'))
    pts = []
    for h in range(8):
        k = f'1.{h}'
        x, y = R['ov_pos'][k], R['qk_pos_prev'][k]
        on = R['ind1'][h] > 0.02
        pts.append((x, y, k, on))
    for x, y, k, on in pts:
        cls = 's1' if on else 's2'
        b.append(f'<g class="mark"><title>head {k}: OV {x:+.2f}, QK {y:+.2f}, induction score {R["ind1"][int(k[2:])]:.2f}</title>'
                 f'<circle class="{cls} ring" cx="{X(x):.1f}" cy="{Y(y):.1f}" r="7"/></g>')
    # labels, nudged so the two corner points do not overlap
    for x, y, k, on in pts:
        dy = -12 if k == f'1.{IH}' else (16 if on else -10)
        anchor = 'end' if x > 0.85 else 'start'
        dx = -10 if anchor == 'end' else 10
        b.append(text(X(x) + dx, Y(y) + dy, k, 't-strong t-tick' if on else 't-tick', anchor))
    lx = L + pw + 40
    b += [f'<circle class="s1 ring" cx="{lx}" cy="80" r="7"/>', text(lx + 14, 84, 'induction score above 0.02', 't-note'),
          f'<circle class="s2 ring" cx="{lx}" cy="106" r="7"/>', text(lx + 14, 110, 'the other layer-1 heads', 't-note'),
          text(lx, 150, f'1.{IH}: OV {R["ov_pos"][f"1.{IH}"]:+.2f}, QK {R["qk_pos_prev"][f"1.{IH}"]:+.2f}', 't-tick'),
          text(lx, 170, f'induction score {R["ind1"][IH]:.2f}', 't-tick'),
          text(lx, 200, f'1.7: OV {R["ov_pos"]["1.7"]:+.2f}, QK {R["qk_pos_prev"]["1.7"]:+.2f}', 't-tick'),
          text(lx, 220, f'induction score {R["ind1"][7]:.2f}', 't-tick')]
    return svg(W, H, f'The eight layer-1 heads of attn-only-2l as points. Across: the eigenvalue positivity of the OV circuit. Up: the eigenvalue positivity of the K-composition term with the previous-token head 0.{PT}. Heads 1.{IH} and 1.7 sit in the top right corner, near (1, 1). The other six are spread out.', b)


F['p6_scatter'] = scatter()


# ---------------------------------------------------------------- 5. where the key comes from
def key_parts():
    kp = R['key_parts']
    items = sorted(((k if k != 'embed+pos' else 'embed + pos', v[0] - v[1]) for k, v in kp.items() if k != 'bias'), key=lambda kv: -kv[1])
    items = [(('head ' + k) if k[0] == '0' else k, max(0.0, v)) for k, v in items]
    b = hbars(30, 50, items, 500, row_h=26, cls='s1', label_w=110, fmt=lambda v: f'+{v:.2f}', vmax=max(v for _, v in items),
              title=f'How much each part of the stream lifts the score of the right source (head 1.{IH})', hi=f'head 0.{PT}')
    return svg(800, 50 + 26 * len(items) + 20, f'The lift each part of the stream gives to the attention score of head 1.{IH} at the right source, compared with an average allowed source. Head 0.{PT} gives +{kp[f"0.{PT}"][0] - kp[f"0.{PT}"][1]:.2f}, head 0.0 gives +{kp["0.0"][0] - kp["0.0"][1]:.2f}, the token and position embeddings give +{kp["embed+pos"][0] - kp["embed+pos"][1]:.2f}, and every other head gives less than 0.3.', b)


F['p6_key_parts'] = key_parts()


# ---------------------------------------------------------------- 6. ablations
def ablations():
    a = R['ablate']
    rows = [('no ablation', R['rep']['attn-only-2l']['second'])]
    names = {f'0.{PT}': 'previous-token head', '0.0': 'second previous-token head', f'1.{IH}': 'induction head'}
    for k, v in a.items():
        rows.append((f'{k} ' + ('(' + names[k] + ')' if k in names else ''), v['second']))
    b = hbars(30, 50, [(r[0].strip(), r[1]) for r in rows], 380, row_h=28, cls='s2', label_w=260, fmt=lambda v: f'{v:.2f}', vmax=16,
              title='Loss on the second copy (nats) after replacing one head with its average output')
    return svg(800, 50 + 28 * len(rows) + 20, 'Loss on the second copy of the repeated random tokens. With no ablation it is 3.41 nats. Replacing the previous-token head 0.3 raises it to 11.85, and replacing the induction head 1.6 raises it to 13.04. Replacing head 0.0 raises it to 6.33. The other heads change it by less than 0.8.', b)


F['p6_ablate'] = ablations()

json.dump(F, open('results/figs_part6.json', 'w'), indent=0)
print('wrote', len(F), 'figures:', ', '.join(F))
