"""SVG figures for Part 2 (GRPO). Reads results/r1_part2_memory.json and results/r1_part2_run.json
(run r1_part2_memory.py and r1_part2_run.py first). Writes results/figs_part2.json. CSS classes only, no colours."""
import json, math, os
from figlib import svg, text, box, arrow
from samfig import line

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
MEM = json.load(open(os.path.join(RES, 'r1_part2_memory.json')))
RUN = json.load(open(os.path.join(RES, 'r1_part2_run.json')))
F = {}


def node(x, y, w, h, s, cls='box', sub=None, tcls='t-note'):
    b = [box(x, y, w, h, cls, 8), text(x + w / 2, y + (h / 2 + 4 if not sub else h / 2 - 3), s, tcls, 'middle')]
    if sub:
        b.append(text(x + w / 2, y + h / 2 + 13, sub, 't-muted', 'middle'))
    return b


# ---------------------------------------------------------------- 1. you are here
b = []
b.append(text(20, 24, 'DeepSeek-R1-Zero (§2.2)', 't-title'))
x, y, h = 20, 38, 46
for i, (s, sub, cls) in enumerate([('DeepSeek-V3-Base', 'pre-trained', 'box'), ('RL with GRPO', 'this part', 'box-on'),
                                   ('DeepSeek-R1-Zero', 'the result', 'box')]):
    xx = x + i * 215
    b += node(xx, y, 180, h, s, cls, sub)
    if i:
        b.append(arrow(xx - 35, y + h / 2, xx - 2, y + h / 2))
b.append(text(20, 126, 'DeepSeek-R1 (§2.3): GRPO runs in two of the four stages', 't-title'))
stages = [('cold-start SFT', 'box'), ('RL with GRPO', 'box-on'), ('sampling + SFT', 'box'), ('RL with GRPO', 'box-on'),
          ('DeepSeek-R1', 'box')]
for i, (s, cls) in enumerate(stages):
    xx = 20 + i * 132
    b += node(xx, 140, 112, 42, s, cls)
    if i:
        b.append(arrow(xx - 20, 161, xx - 2, 161))
F['p2_here'] = svg(680, 200, 'You are here: GRPO is the reinforcement learning step that turns DeepSeek-V3-Base into '
                   'R1-Zero, and it runs again in two stages of the R1 pipeline.', b)

# ---------------------------------------------------------------- 2. PPO and GRPO side by side, with memory
b = []
m7 = MEM['7B policy']
cols = [('PPO', 20, [('policy', 'trained', 'box-on'), ('reference', 'frozen', 'box'), ('reward model', 'frozen', 'box'),
                     ('value model (critic)', 'trained, same size', 'box-on')],
         f'7B policy: {m7["PPO"]:.0f} GB of weights and optimizer state'),
        ('GRPO in R1-Zero', 360, [('policy', 'trained', 'box-on'), ('reference', 'frozen', 'box'),
                                  ('reward: rules', 'a program, no weights', 'box-ghost'), ('no critic', 'the group is the baseline', 'box-ghost')],
         f'7B policy: {m7["GRPO, rule rewards"]:.0f} GB of weights and optimizer state')]
for title, x0, items, foot in cols:
    b.append(text(x0 + 150, 24, title, 't-title', 'middle'))
    for i, (s, sub, cls) in enumerate(items):
        b += node(x0, 40 + i * 58, 300, 46, s, cls, sub)
    b.append(text(x0 + 150, 288, foot, 't-strong t-note', 'middle'))
b.append(line(340, 14, 340, 296, 'edge-dim'))
b.append(text(20, 316, 'Filled boxes are trained (weights, gradients and Adam state: 16 bytes per parameter).', 't-muted'))
b.append(text(20, 334, 'Plain boxes only run forward passes (2 bytes per parameter). Dashed boxes need no memory.', 't-muted'))
F['p2_ppo_grpo'] = svg(680, 346, 'PPO keeps four models, two of them trained. GRPO as used for R1-Zero keeps a trained '
                       'policy and a frozen reference; the reward is a program and there is no critic.', b)

# ---------------------------------------------------------------- 3. the GRPO loop, with our real group
b = []
rw, adv = RUN['rewards'], RUN['advantages']
G = len(rw)
b += node(20, 150, 110, 50, 'question q', 'box', 'GSM8K')
b.append(text(250, 24, f'{G} answers from pi_old', 't-title', 'middle'))
b.append(text(410, 24, 'reward r_i', 't-title', 'middle'))
b.append(text(560, 24, 'advantage A_i', 't-title', 'middle'))
for i in range(G):
    y = 38 + i * 36
    b += node(190, y, 120, 28, f'answer {i}', 'box', None, 't-tick')
    b.append(line(130, 175, 188, y + 14, 'edge'))
    b += node(380, y, 60, 28, f'{int(rw[i])}', 'box-1' if rw[i] > 0 else 'box', None, 't-tick')
    b.append(arrow(312, y + 14, 378, y + 14))
    b += node(520, y, 80, 28, f'{adv[i]:+.2f}', 'box-on' if adv[i] > 0 else 'box', None, 't-tick')
    b.append(arrow(442, y + 14, 518, y + 14))
yb = 38 + G * 36 + 14
b.append(text(560, yb, 'Equation 3', 't-muted', 'middle'))
b.append(text(410, yb, 'is it right?', 't-muted', 'middle'))
b.append(f'<path class="edge" d="M600,{yb - 30} C660,{yb + 20} 120,{yb + 40} 75,202" marker-end="url(#ah)"/>')
b.append(text(340, yb + 46, 'update the policy with Equation 1: raise the answers with A > 0, lower the ones with A < 0', 't-note', 'middle'))
F['p2_loop'] = svg(680, yb + 60, f'The GRPO loop on our real group: one question, {G} sampled answers, a 0 or 1 reward '
                   'for each, the group-normalised advantages, and the update.', b)

# ---------------------------------------------------------------- 4. the clip: one token's term as a function of the ratio
def clip_panel(x0, A, title, eps=0.2):
    W, H, L, T = 300, 220, 40, 30
    rmin, rmax = 0.0, 2.0
    ymin, ymax = (-0.1, 1.6) if A > 0 else (-2.1, 0.1)
    X = lambda r: x0 + L + (r - rmin) / (rmax - rmin) * (W - L - 10)
    Y = lambda v: T + (ymax - v) / (ymax - ymin) * (H - T - 30)
    b = [text(x0 + W / 2, 18, title, 't-title', 'middle')]
    b.append(f'<rect class="plane" x="{X(1 - eps):.1f}" y="{T}" width="{X(1 + eps) - X(1 - eps):.1f}" height="{H - T - 30}"/>')
    b.append(f'<line class="axis" x1="{X(rmin)}" y1="{Y(0):.1f}" x2="{X(rmax)}" y2="{Y(0):.1f}"/>')
    for r in (0, 0.8, 1, 1.2, 2):
        b.append(text(X(r), H - 12, f'{r:g}', 't-tick', 'middle'))
    rs = [i / 100 for i in range(0, 201)]
    unc = ' '.join(f'{X(r):.1f},{Y(r * A):.1f}' for r in rs if ymin <= r * A <= ymax)
    obj = ' '.join(f'{X(r):.1f},{Y(min(r * A, min(max(r, 1 - eps), 1 + eps) * A)):.1f}' for r in rs)
    b.append(f'<polyline class="base-line" points="{unc}"/>')
    b.append(f'<polyline class="l1" style="stroke-width:3" points="{obj}"/>')
    b.append(f'<circle class="s2" cx="{X(1):.1f}" cy="{Y(A):.1f}" r="5"/>')
    if A > 0:
        b.append(text(X(1.62), Y(1.2 * A) + 20, 'flat: this term stops pushing', 't-note', 'middle'))
        b.append(text(X(0.42), Y(0.5) - 12, 'slope A', 't-muted', 'middle'))
    else:
        b.append(text(X(0.38), Y(0.8 * A) - 12, 'flat: this term stops pushing', 't-note', 'middle'))
        b.append(text(X(1.45), Y(1.6 * A) + 6, 'slope A: still penalised', 't-muted', 'middle'))
    b.append(text(x0 + W / 2, H + 4, 'ratio rho = pi_theta / pi_old', 't-tick', 'middle'))
    return b


b = clip_panel(10, 1.0, 'A = +1 (a good answer)') + clip_panel(360, -1.0, 'A = -1 (a bad answer)')
F['p2_clip'] = svg(680, 232, 'One token of Equation 1 as a function of the ratio, with epsilon 0.2. For a good answer '
                   'the term stops growing above 1.2; for a bad answer it stops falling below 0.8. The dashed line is '
                   'the term without the clip.', b)

# ---------------------------------------------------------------- 5. k1 and k3 as functions of x = pi_ref / pi_theta
W, H, L, T = 640, 250, 60, 26
xmin, xmax, ymin, ymax = 0.1, 3.0, -1.2, 2.4
X = lambda v: L + (v - xmin) / (xmax - xmin) * (W - L - 150)
Y = lambda v: T + (ymax - v) / (ymax - ymin) * (H - T - 40)
b = []
for t in (-1, 0, 1, 2):
    b += [f'<line class="grid" x1="{L}" y1="{Y(t):.1f}" x2="{X(xmax):.1f}" y2="{Y(t):.1f}"/>', text(L - 8, Y(t) + 4, f'{t}', 't-tick', 'end')]
for v in (0.5, 1, 2, 3):
    b.append(text(X(v), H - 20, f'{v:g}', 't-tick', 'middle'))
xs = [xmin + i * (xmax - xmin) / 200 for i in range(201)]
k1 = ' '.join(f'{X(v):.1f},{Y(-math.log(v)):.1f}' for v in xs if -math.log(v) <= ymax)
k3 = ' '.join(f'{X(v):.1f},{Y(v - math.log(v) - 1):.1f}' for v in xs if v - math.log(v) - 1 <= ymax)
b.append(f'<polyline class="l2" points="{k1}"/>')
b.append(f'<polyline class="l1" style="stroke-width:3" points="{k3}"/>')
b.append(f'<circle class="s1" cx="{X(1):.1f}" cy="{Y(0):.1f}" r="5"/>')
b.append(text(X(3) + 10, Y(-math.log(3)) + 4, 'k1 = -log x: can be < 0', 't-s2'))
b.append(text(X(3) + 10, Y(3 - math.log(3) - 1) + 4, 'k3 = x - log x - 1', 't-s1'))
b.append(text(X(1), Y(0) + 22, 'both 0 at x = 1', 't-muted', 'middle'))
b.append(text(L + (X(xmax) - L) / 2, H - 2, 'x = pi_ref(o) / pi_theta(o)', 't-tick', 'middle'))
F['p2_k3'] = svg(W, H, 'Two per-token estimates of the KL as functions of the probability ratio x. The plain log '
                 'ratio k1 is negative for x above 1; Equation 2, k3, never goes below zero and touches zero only at x = 1.', b)

# ---------------------------------------------------------------- 6. one step on the real group: change of log-prob per answer
rows = RUN['rows']
W, H = 680, 60 + 34 * len(rows)
mx = max(abs(r['after'] - r['before']) for r in rows)
c0, half = 400, 200
b = [text(20, 22, 'answer', 't-tick'), text(110, 22, 'reward', 't-tick'), text(190, 22, 'A', 't-tick'),
     text(c0, 22, 'change of log pi_theta(o_i | q) after one step', 't-tick', 'middle')]
b.append(line(c0, 30, c0, H - 20, 'axis'))
for i, r in enumerate(rows):
    y = 36 + i * 34
    d = r['after'] - r['before']
    w = half * abs(d) / mx
    x = c0 if d >= 0 else c0 - w
    b.append(text(20, y + 16, f'{r["i"]}', 't-tick'))
    b.append(text(110, y + 16, f'{int(r["reward"])}', 't-tick'))
    b.append(text(190, y + 16, f'{r["A"]:+.2f}', 't-tick'))
    b.append(f'<g class="mark"><title>answer {r["i"]}: {r["before"]:.2f} to {r["after"]:.2f}</title>'
             f'<rect class="{"s1" if d >= 0 else "s2"}" x="{x:.1f}" y="{y + 4}" width="{max(w, 1.5):.1f}" height="20" rx="3"/></g>')
    b.append(text(x + w + 8 if d >= 0 else x - 10, y + 18, f'{d:+.3f}', 't-val', 'start' if d >= 0 else 'end'))
F['p2_step'] = svg(W, H, 'After one GRPO step on our group, the log-probability of every right answer went up and '
                   'every wrong answer went down.', b)

json.dump(F, open(os.path.join(RES, 'figs_part2.json'), 'w'))
print('figures:', ', '.join(F))
