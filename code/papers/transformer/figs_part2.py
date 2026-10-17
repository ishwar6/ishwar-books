"""Figures for Part 2 (the architecture), drawn from results/part2.json.
Writes results/figs_part2.json (key -> svg string). Colours come only from CSS classes."""
import json, os
import numpy as np
from figlib import svg, text, box, arrow, lines_chart

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'part2.json')))
F = {}


# ---------------------------------------------------------------- 1. one encoder layer with the real shapes
def layer():
    b = [text(20, 24, 'One encoder layer (the model has 6), for the 12 tokens of our sentence', 't-title')]
    cx, w = 330, 240
    ys = [(48, 'x: 12 × 512', 'box'), (108, 'multi-head self-attention', 'box-2'), (168, 'add: x + Sublayer(x)', 'box'),
          (228, 'layer norm → 12 × 512', 'box-1'), (288, 'feed-forward: 512 → 2048 → 512', 'box-2'),
          (348, 'add: x + Sublayer(x)', 'box'), (408, 'layer norm → 12 × 512', 'box-1')]
    for k, (y, lab, cls) in enumerate(ys):
        b += [box(cx - w / 2, y, w, 36, cls, 8), text(cx, y + 23, lab, 't-note', 'middle')]
        if k:
            b.append(arrow(cx, ys[k - 1][0] + 36, cx, y - 2))
    # residual paths
    for y_from, y_to in [(48, 168), (228, 348)]:
        xl = cx - w / 2 - 40
        b += [f'<path class="edge" d="M{cx - w / 2},{y_from + 18} H{xl} V{y_to + 18} H{cx - w / 2 - 2}" fill="none" marker-end="url(#ah)"/>',
              text(xl - 8, (y_from + y_to) / 2 + 22, 'residual', 't-tick', 'end')]
    b += [text(cx + w / 2 + 16, 130, 'sub-layer 1: 1,050,624 weights', 't-tick'),
          text(cx + w / 2 + 16, 310, 'sub-layer 2: 2,099,712 weights', 't-tick'),
          text(cx + w / 2 + 16, 250, 'layer norm: 1,024 weights', 't-tick'),
          text(cx + w / 2 + 16, 430, 'layer norm: 1,024 weights', 't-tick')]
    return svg(680, 460, 'One encoder layer of the released English to German model: multi-head self-attention, add the input, '
               'layer norm, feed-forward 512 to 2048 to 512, add the input, layer norm. Every arrow carries 12 vectors of 512 numbers.', b)


F['p2_layer'] = layer()


# ---------------------------------------------------------------- 2. four position-encoding dimensions as waves
def waves():
    P = np.array(R['pe']['P'])
    xs = list(range(P.shape[0]))
    dims = [(0, 'dim 0, turn 6.3', 'l1', 's1'), (64, 'dim 64, turn 19.9', 'l2', 's2'),
            (128, 'dim 128, turn 62.8', 'l3', 's3')]
    series = [(lab, [round(float(v), 4) for v in P[:, d]], lc, dc) for d, lab, lc, dc in dims]
    return lines_chart(780, 300, 'Three dimensions of the sinusoidal position encoding for positions 0 to 63. Dimension 0 turns once every '
                       '6.3 positions, dimension 64 every 19.9 positions and dimension 128 every 62.8 positions.',
                       xs, series, 'position', 'value', -1, 1, [-1, -0.5, 0, 0.5, 1], fmt=lambda v: f'{v:g}',
                       xticks=[0, 10, 20, 30, 40, 50, 63], right=250)


F['p2_waves'] = waves()


# ---------------------------------------------------------------- 3. dot product against the offset
def dots():
    P = np.array(R['pe']['P'])
    ks = list(range(0, 60))
    a = [round(float(P[0] @ P[k]), 2) for k in ks]
    b = [round(float(P[3] @ P[3 + k]), 2) if 3 + k < len(P) else None for k in ks]
    ks2 = [k for k in ks if 3 + k < len(P)]
    return lines_chart(780, 300, 'The dot product of two position vectors depends only on how far apart they are. From position 0 and from '
                       'position 3 the two curves lie on top of each other: 256 at offset 0, then 144.2 at offset 30 and 125.1 at offset 60.',
                       ks2, [('from position 0', a[:len(ks2)], 'l1', 's1'), ('from position 3', b[:len(ks2)], 'l2', 's2')],
                       'offset k', 'PE(pos) · PE(pos + k)', 100, 260, [100, 150, 200, 250], fmt=lambda v: f'{v:g}',
                       xticks=[0, 10, 20, 30, 40, 50, 60], right=200)


F['p2_dots'] = dots()

json.dump(F, open(os.path.join(HERE, 'results', 'figs_part2.json'), 'w'))
print('figures:', ', '.join(F))
