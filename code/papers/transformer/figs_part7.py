"""Figures for Part 7 (build it from scratch), drawn from results/part7_toy.json. Writes results/figs_part7.json.
CSS classes only, no colours."""
import json
import os

from figlib import svg, text, box, arrow, heatmap, lines_chart

HERE = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(HERE, 'results', 'part7_toy.json')))
F = {}

# 1. The package: Figure 1 of the paper as our classes ------------------------------------------------------------
b = []


def block(x, y, w, h, title, sub, cls='box'):
    return [box(x, y, w, h, cls, 8), text(x + w / 2, y + 20, title, 't-title', 'middle'),
            text(x + w / 2, y + 38, sub, 't-tick', 'middle')]


# encoder column (left), decoder column (right), bottom to top
b += block(40, 382, 280, 48, 'TokenEmbedding × √d_model', 'source ids → (b, n, d_model)')
b += block(40, 316, 280, 48, 'PositionalEncoding', '+ fixed sinusoids (§3.5)')
b += [box(30, 112, 300, 186, 'box-1', 10), text(180, 132, 'EncoderLayer  × N', 't-title', 'middle')]
b += block(50, 146, 260, 48, 'MultiHeadAttention + AddAndNorm', 'self-attention, padding mask')
b += block(50, 222, 260, 48, 'FeedForward + AddAndNorm', 'Eq. 2, each position alone')
b += block(40, 34, 280, 48, 'memory', 'encoder output (b, n, d_model)')
b += block(420, 382, 300, 48, 'TokenEmbedding (the same table)', 'target ids, shifted right')
b += block(420, 316, 300, 48, 'PositionalEncoding', '+ fixed sinusoids')
b += [box(410, 84, 320, 214, 'box-2', 10), text(570, 104, 'DecoderLayer  × N', 't-title', 'middle')]
b += block(430, 238, 280, 48, 'masked self-attention + AddAndNorm', 'causal mask')
b += block(430, 178, 280, 48, 'cross-attention + AddAndNorm', 'queries: decoder · keys, values: memory')
b += block(430, 118, 280, 48, 'FeedForward + AddAndNorm', 'Eq. 2')
b += block(420, 20, 300, 48, 'generator: y · Eᵀ (+ bias)', 'logits (b, t, vocab), tied to the embedding')
b += [arrow(180, 382, 180, 366), arrow(180, 316, 180, 300), arrow(180, 112, 180, 84),
      arrow(570, 382, 570, 366), arrow(570, 316, 570, 300), arrow(570, 84, 570, 70),
      arrow(320, 58, 428, 200, on=True)]
F['p7_package'] = svg(760, 440, 'The classes of model.py arranged like Figure 1 of the paper. Left: TokenEmbedding, PositionalEncoding '
                      'and N EncoderLayers, each with MultiHeadAttention and FeedForward wrapped in AddAndNorm, giving the memory. '
                      'Right: the same embedding table, PositionalEncoding and N DecoderLayers with masked self-attention, '
                      'cross-attention to the memory and FeedForward, then the generator tied to the embedding.', b)

# 2. The training curve -------------------------------------------------------------------------------------------
KEEP = (1, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000, 2500, 3000)
curve = [r for r in T['curve'] if r['step'] in KEEP]
xs = [r['step'] for r in curve]
F['p7_loss'] = lines_chart(
    780, 320, f'Training loss of the date model over 3,000 steps of 128 dates. It falls from 4.38 at step 1 to about 1.0 at '
    f'step 100 and 0.70 by step 300, then stays just above the lowest value label smoothing allows, {T["loss_floor"]:.3f}. '
    'The step axis is logarithmic.',
    xs, [('training loss', [r['loss'] for r in curve], 'l1', 's1'),
         ('lowest possible (label smoothing)', [T['loss_floor']] * len(xs), 'l2', 's2')],
    'training step (log scale)', 'loss', 0, 4.5, [0, 1, 2, 3, 4], xlog=True, xticks=[1, 50, 100, 300, 1000, 3000],
    fmt=lambda v: f'{v:.3f}' if isinstance(v, float) else f'{v}', right=280)

ex = [r for r in T['curve'] if 'test_exact' in r]
F['p7_exact'] = lines_chart(
    780, 280, 'Exact match on 500 held-out dates, measured every 250 steps with greedy decoding. 0 at step 1, '
    '0.896 at step 250, 0.998 at step 750, 1.000 at step 1,000 and at the end, with dips to 0.954 at step 1,250 and 0.978 at step 2,750.',
    [r['step'] for r in ex], [('exact match, held-out dates', [r['test_exact'] for r in ex], 'l3', 's3')],
    'training step', 'share of dates right', 0, 1, [0, 0.25, 0.5, 0.75, 1.0], xticks=[1, 500, 1000, 1500, 2000, 2500, 3000],
    fmt=lambda v: f'{v:.2f}' if isinstance(v, float) else f'{v}', right=260)

# 3. Cross-attention of the trained model: decoder layer 2, heads 2 and 3 ----------------------------------------
e = T['example']
src, tgt = e['src'], e['tgt']
cell = 24
b = []
for k, (hd, title) in enumerate([(2, 'layer 2, head 2: copy the digit'), (3, 'layer 2, head 3: find the month')]):
    W = e['weights'][1][hd]
    b += heatmap(10 + k * 380, 40, W, [('␣' if c == ' ' else c) for c in tgt], src, cell, title=title, label_w=56)
F['p7_attn'] = svg(770, 400, f'Cross-attention of the trained date model for {"".join(src[:-1])} → {"".join(tgt[:-1])}. '
                   'Rows are the output characters, columns the input characters, darker means more weight. '
                   'Head 2 of decoder layer 2 sends each output digit to the input digit it copies, including the right one '
                   'of the two ones. Head 3 sends the first month letter to the month digits.', b)

json.dump(F, open(os.path.join(HERE, 'results', 'figs_part7.json'), 'w'))
print('figures:', ', '.join(F))
