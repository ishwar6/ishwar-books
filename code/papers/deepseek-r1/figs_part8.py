"""Figures for Part 8 (train a reasoner from scratch): the minir1 package map with a 'you are here' version for each
file, the reward and length curves of our run, and accuracy before and after. Writes results/figs_part8.json.
CSS classes only, no colours."""
import json, os, sys

from figlib import svg, text, box, arrow
from figlib import lines_chart

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
TAG = sys.argv[1] if len(sys.argv) > 1 else 'instruct'
F = {}

# 1. The package map ---------------------------------------------------------------------------------------------
FILES = [  # key, x, y, title, two notes
    ('template', 20, 30, 'template.py', 'Table 1: the prompt', 'question → text'),
    ('grpo_sample', 250, 30, 'grpo.py: sample_group', 'G answers per question', 'from π_old'),
    ('rewards', 480, 30, 'rewards.py', 'accuracy + format', 'rules, no network'),
    ('grpo_update', 480, 170, 'grpo.py: Eq. 3, 1, 2', 'advantages, clipped ratio,', 'KL to π_ref → loss'),
    ('train', 250, 170, 'train.py', 'the loop, the optimiser,', 'one log line per step'),
    ('eval', 20, 170, 'eval.py', '200 held-out questions', 'before and after'),
]


def pkg_map(on=None):
    b = []
    for key, x, y, title, n1, n2 in FILES:
        hit = on is not None and key in on
        b += [box(x, y, 200, 90, 'box-on' if hit else 'box', 10), text(x + 100, y + 30, title, 't-title', 'middle'),
              text(x + 100, y + 52, n1, 't-tick', 'middle'), text(x + 100, y + 70, n2, 't-tick', 'middle')]
        if hit:
            b.append(text(x + 100, y - 8, 'you are here', 't-note', 'middle'))
    b += [arrow(220, 75, 248, 75), arrow(450, 75, 478, 75), arrow(580, 120, 580, 168), arrow(480, 215, 452, 215),
          arrow(350, 170, 350, 122, on=True), arrow(250, 215, 222, 215)]
    b.append(text(358, 150, 'next step', 't-tick'))
    who = 'none' if on is None else ', '.join(on)
    return svg(700, 280, f'The minir1 package as a loop. template.py turns a question into the Table 1 prompt; '
               f'sample_group in grpo.py samples G answers; rewards.py scores them; the update part of grpo.py turns '
               f'rewards into advantages and a loss; train.py repeats this; eval.py measures held-out questions. '
               f'Highlighted: {who}.', b)


F['p8_map'] = pkg_map()
for key in ['template', 'rewards', 'grpo_sample', 'grpo_update', 'train', 'eval']:
    F[f'p8_map_{key}'] = pkg_map([key])

# 2. Curves from our logs (one pair of charts per run) ---------------------------------------------------------
def curves(tag, suffix, who):
    logp = os.path.join(R, f'r1_part8_log_{tag}.jsonl')
    if not os.path.exists(logp):
        return
    rows = [json.loads(l) for l in open(logp)]
    W = 10                                                       # moving average over 10 steps

    def smooth(key):
        return [sum(r[key] for r in rows[max(0, i - W + 1): i + 1]) / len(rows[max(0, i - W + 1): i + 1])
                for i in range(len(rows))]
    every = max(1, len(rows) // 30)
    idx = list(range(0, len(rows), every))
    if idx[-1] != len(rows) - 1:
        idx.append(len(rows) - 1)
    xs = [rows[i]['step'] for i in idx]
    acc, fmt, ln = smooth('acc'), smooth('fmt'), smooth('length')
    last = xs[-1]
    step = 10 if last <= 60 else 25 if last <= 150 else 50
    ticks = [1] + list(range(step, last + 1, step))
    F[f'p8_reward{suffix}'] = lines_chart(
        800, 330, f'Training rewards of {who}, averaged over 10 steps. Format reward goes from {fmt[0]:.2f} to '
        f'{fmt[-1]:.2f}; accuracy reward from {acc[0]:.2f} to {acc[-1]:.2f}, over {last} steps.',
        xs, [('format reward', [fmt[i] for i in idx], 'l2', 's2'), ('accuracy reward', [acc[i] for i in idx], 'l1', 's1')],
        'training step', 'mean reward (10-step average)', 0, 1, [0, 0.25, 0.5, 0.75, 1.0], xticks=ticks,
        fmt=lambda v: f'{v:.2f}' if isinstance(v, float) else f'{v}', right=230)
    top = round(max(ln) * 1.15, -1)
    F[f'p8_length{suffix}'] = lines_chart(
        800, 300, f'Mean answer length in tokens of {who}, averaged over 10 steps: {ln[0]:.0f} at the start '
        f'and {ln[-1]:.0f} at step {last}.',
        xs, [('mean length (tokens)', [ln[i] for i in idx], 'l3', 's3')],
        'training step', 'tokens per answer (10-step average)', 0, top, [0, round(top / 2, -1), top], xticks=ticks,
        fmt=lambda v: f'{v:.0f}' if isinstance(v, float) else f'{v}', right=220)


curves('base', '_base', 'the base model run')
curves(TAG, '', 'the Instruct model run')

# 3. Before and after -------------------------------------------------------------------------------------------
evp = os.path.join(R, f'r1_part8_eval_{TAG}.json')
if os.path.exists(evp):
    E = json.load(open(evp))
    b, x0, base, H = [], 120, 250, 190
    groups = [('accuracy', 'acc'), ('format', 'fmt')]
    for gi, (lab, k) in enumerate(groups):
        for j, (when, cls) in enumerate([('before', 'box-2'), ('after', 'box-1')]):
            v = E[when][k]
            x = x0 + gi * 260 + j * 90
            h = v * H
            b += [box(x, base - h, 70, max(h, 1), cls, 4), text(x + 35, base - h - 8, f'{v:.3f}', 't-note', 'middle'),
                  text(x + 35, base + 18, when, 't-tick', 'middle')]
        b.append(text(x0 + gi * 260 + 80, base + 40, lab, 't-title', 'middle'))
    b.append(f'<line class="axis" x1="{x0 - 20}" y1="{base}" x2="{x0 + 500}" y2="{base}"/>')
    F['p8_before_after'] = svg(680, 300, f'Held-out GSM8K, 200 questions, greedy: accuracy {E["before"]["acc"]:.3f} '
                               f'before and {E["after"]["acc"]:.3f} after training; format {E["before"]["fmt"]:.3f} '
                               f'before and {E["after"]["fmt"]:.3f} after.', b)

json.dump(F, open(os.path.join(R, 'figs_part8.json'), 'w'))
print('figures:', ', '.join(F))
