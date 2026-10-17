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
json.dump(F, open(os.path.join(HERE, 'results', 'figs_part3.json'), 'w'))
print('figures:', ', '.join(F))
