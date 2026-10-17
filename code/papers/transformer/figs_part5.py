"""Figures for Part 5 (training), drawn from results/part5.json. Writes results/figs_part5.json. CSS classes only."""
import json, os
from figlib import svg, text, lines_chart
from samfig import hbars

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'part5.json')))
F = {}

# 1. the learning rate of Eq. 3
cur = [c for c in R['curve'] if c[0] % 500 == 0]
xs = [0] + [c[0] for c in cur]
F['p5_lr'] = lines_chart(
    780, 320, 'The learning rate of Eq. 3 over the first 20,000 steps. It rises in a straight line to 7.0 in units of 0.0001 at step 4,000 '
    'for d_model 512, and to 4.9 for d_model 1024, then falls with one over the square root of the step.',
    xs, [('d_model 512 (base)', [0] + [c[1] * 1e4 for c in cur], 'l2', 's2'),
         ('d_model 1024 (big)', [0] + [c[2] * 1e4 for c in cur], 'l3', 's3')],
    'training step', 'learning rate (× 0.0001)', 0, 8, [0, 2, 4, 6, 8],
    xticks=[0, 4000, 8000, 12000, 16000, 20000], fmt=lambda v: f'{v:.1f}' if isinstance(v, float) else f'{v}', right=200)

# 2. Adam with beta2 = 0.98 against 0.999 after the gradient jumps
a, b = R['adam_0.98'], R['adam_0.999']
ts = sorted(int(t) for t in a)
F['p5_adam'] = lines_chart(
    780, 300, 'The size of one Adam step, in units of the learning rate, when the gradient jumps from 1 to 10 at step 200. '
    'With beta2 = 0.98 the step grows to about 1.56 and is back near 1 by step 300. With beta2 = 0.999 it grows to about 2.76 and is still 1.34 at step 400.',
    ts, [('β2 = 0.98 (paper)', [a[str(t)] for t in ts], 'l2', 's2'), ('β2 = 0.999 (common default)', [b[str(t)] for t in ts], 'l3', 's3')],
    'training step', 'step size ÷ learning rate', 0, 3, [0, 1, 2, 3],
    xticks=[200, 250, 300, 350, 400], fmt=lambda v: f'{v:.2f}' if isinstance(v, float) else f'{v}', right=240)

# 3. label smoothing on a 5-word vocabulary
words = ['word 0 ✓', 'word 1', 'word 2', 'word 3', 'word 4']
hard = [1.0, 0, 0, 0, 0]
soft = [0.92, 0.02, 0.02, 0.02, 0.02]
body = hbars(20, 44, list(zip(words, hard)), 220, label_w=80, fmt=lambda v: f'{v:.2f}', title='Target without smoothing', hi='word 0 ✓')
body += hbars(410, 44, list(zip(words, soft)), 220, cls='s2', label_w=80, fmt=lambda v: f'{v:.2f}', title='Target with ε = 0.1', hi='word 0 ✓')
F['p5_smooth'] = svg(780, 180, 'Two target distributions over a vocabulary of five words. Without smoothing, the correct word gets 1.00 and '
                     'the others 0. With label smoothing of 0.1, the correct word gets 0.92 and each other word gets 0.02.', body)

json.dump(F, open(os.path.join(HERE, 'results', 'figs_part5.json'), 'w'))
print('figures:', ', '.join(F))
