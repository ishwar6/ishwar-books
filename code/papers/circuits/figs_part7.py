"""Figures for Part 7 (what the paper left out, and what came after), drawn from results/part7.json.
Writes results/figs_part7.json (key -> svg string). Colours come only from CSS classes, so every figure follows the
site's dark, dim and light themes."""
import json, math
from figlib import svg, text, box, lines_chart
from cfig import line, hbars

R = json.load(open('results/part7.json'))
F = {}


# ---------------------------------------------------------------- 1. where the parameters are
def params():
    b = [text(24, 26, 'Parameters inside the layers (embedding and unembedding not counted)', 't-title')]
    rows = [('gelu-1l', 'gelu-1l'), ('gelu-2l', 'gelu-2l'), ('gpt2', 'gpt2 (12 layers)')]
    y, x0, W = 56, 150, 520
    for k, lab in rows:
        p = R['params'][k]
        tot = p['attn'] + p['mlp']
        wa, wm = W * p['attn'] / tot, W * p['mlp'] / tot
        b.append(text(x0 - 10, y + 20, lab, 't-tick', 'end'))
        b.append(f'<g class="mark"><title>{lab}: attention {p["attn"]:,}</title>{box(x0, y + 4, wa - 2, 24, "box-2", 4)}</g>')
        b.append(f'<g class="mark"><title>{lab}: MLP {p["mlp"]:,}</title>{box(x0 + wa, y + 4, wm, 24, "box-1", 4)}</g>')
        b.append(text(x0 + wa / 2, y + 21, f'attention {p["attn"] / tot:.1%}', 't-tick', 'middle'))
        b.append(text(x0 + wa + wm / 2, y + 21, f'MLP {p["mlp"] / tot:.1%}', 't-tick', 'middle'))
        y += 40
    b.append(text(x0, y + 14, 'Each MLP has 4 × d_model neurons, so it holds twice the weights of the attention layer beside it.', 't-muted'))
    return svg(700, y + 30, 'Share of the layer parameters in attention and in the MLPs: about one third and two thirds for gelu-1l, gelu-2l and gpt2.', b)


F['p7_params'] = params()


# ---------------------------------------------------------------- 2. loss by position: in-context learning
def icl():
    idx = [1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500]
    c = R['icl']
    series = [('attn-only-1l', [c['attn-only-1l']['curve'][str(k)] for k in idx], 'l2', 's2'),
              ('gelu-1l', [c['gelu-1l']['curve'][str(k)] for k in idx], 'l3', 's3'),
              ('attn-only-2l', [c['attn-only-2l']['curve'][str(k)] for k in idx], 'l1', 's1'),
              ('gelu-2l', [c['gelu-2l']['curve'][str(k)] for k in idx], 'l4', 's4')]
    return lines_chart(800, 340, 'Mean loss at each token index of 300 Wikipedia windows of 512 tokens. All four models start near 9.6 to 9.8 nats. '
                       'By token 500 the one-layer models are at 4.86 and 4.53, the two-layer models at 4.20 and 3.83.',
                       idx, series, 'token index in the window (log scale)', 'loss (nats)', 3, 10, [3, 4, 5, 6, 7, 8, 9, 10],
                       xlog=True, xticks=[1, 10, 50, 100, 500], fmt=lambda v: f'{v:.2f}' if isinstance(v, float) else f'{v}', right=190)


F['p7_icl'] = icl()


# ---------------------------------------------------------------- 3. the toy model: five features in two dimensions
def toy():
    panels = [('relu S=0.0', 'ReLU, dense (S = 0)'), ('relu S=0.7', 'ReLU, S = 0.7'),
              ('relu S=0.9', 'ReLU, S = 0.9'), ('linear S=0.9', 'linear, S = 0.9')]
    b, pw = [], 180
    for k, (key, title) in enumerate(panels):
        d = R['toy'][key]
        cx, cy, sc = 20 + k * pw + pw / 2, 150, 62
        b.append(text(cx, 30, title, 't-title', 'middle'))
        b.append(line(cx - 80, cy, cx + 80, cy, 'grid'))
        b.append(line(cx, cy - 80, cx, cy + 80, 'grid'))
        for i in range(5):
            wx, wy = d['W'][0][i], d['W'][1][i]
            n = math.hypot(wx, wy)
            ex, ey = cx + wx * sc, cy - wy * sc
            cls = 'path' if n > 0.5 else 'edge'
            b.append(f'<g class="mark"><title>feature {i + 1}: length {n:.2f}, angle {d["angles"][i]:.0f}°</title>'
                     f'{line(cx, cy, ex, ey, cls)}<circle class="{"s1" if n > 0.5 else "s2"}" cx="{ex:.1f}" cy="{ey:.1f}" r="4"/></g>')
            if n > 0.5:
                lx, ly = cx + wx / n * (n * sc + 14), cy - wy / n * (n * sc + 14)
                b.append(text(lx, ly + 4, str(i + 1), 't-tick', 'middle'))
        b.append(text(cx, 262, f'{d["represented"]} of 5 features', 't-tick', 'middle'))
        b.append(text(cx, 280, f'loss {d["loss"]:.4f}', 't-muted', 'middle'))
    return svg(4 * pw + 40, 296, 'The columns W_i of the trained toy models, drawn in their two dimensions. Dense ReLU: two features at right angles. '
               'S = 0.7: four features in two opposite pairs. S = 0.9: all five as a pentagon, 72 degrees apart. Linear at S = 0.9: only two.', b)


F['p7_toy'] = toy()


# ---------------------------------------------------------------- 4. two neurons of gelu-1l: what they write
def neurons():
    b = []
    tok = lambda s: s.strip() if s.startswith(' ') else '-' + s
    for k, n in enumerate(['847', '1996']):
        items = [(tok(t), v) for t, v in R['neurons'][n]['up'][:6]]
        b += hbars(20 + k * 360, 52, items, 170, 24, 's1' if k == 0 else 's2', 110, fmt=lambda v: f'+{v:.2f}', vmax=1.5,
                   title=f'neuron {n}: tokens W_U W_O raises')
    return svg(720, 220, 'The direct effect on the logits of two neurons of gelu-1l. Neuron 847 raises days of the week, Friday by 1.10. '
               'Neuron 1996 raises words in capital letters, such as ATT by 1.44.', b)


F['p7_neurons'] = neurons()

json.dump(F, open('results/figs_part7.json', 'w'))
print('figures:', ', '.join(F))
