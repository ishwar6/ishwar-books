"""Figures for Part 4 (R1 and reasoning by RL). Writes results/figs_part4.json (key -> SVG string).
Numbers come from results/part4_grpo.json (our runs) or from the papers, as noted next to each figure."""
import json, os
from figlib import svg, text, box, arrow, bar_path, esc
from common import RESULTS

R = json.load(open(os.path.join(RESULTS, 'part4_grpo.json')))
F = {}
W = 680


def label_box(x, y, w, h, cls, title, sub=None, tcls='t-note'):
    b = [box(x, y, w, h, cls, rx=8)]
    if sub:
        b += [text(x + w / 2, y + h / 2 - 3, title, tcls, 'middle'), text(x + w / 2, y + h / 2 + 14, sub, 't-muted', 'middle')]
    else:
        b.append(text(x + w / 2, y + h / 2 + 5, title, tcls, 'middle'))
    return b


# 1. PPO vs GRPO: what sits in memory ------------------------------------------------------------
b = [text(W / 4, 24, 'PPO: four models', 't-title', 'middle'), text(3 * W / 4, 24, 'GRPO: three, or two plus rules', 't-title', 'middle')]
for x0 in (20, W / 2 + 20):
    pass
b += label_box(30, 44, 135, 52, 'box-on', 'policy', 'trained')
b += label_box(180, 44, 135, 52, 'box', 'reference', 'frozen')
b += label_box(30, 110, 135, 52, 'box', 'reward model', 'frozen')
b += label_box(180, 110, 135, 52, 'box-on', 'value model', 'trained, same size')
b += label_box(W / 2 + 25, 44, 135, 52, 'box-on', 'policy', 'trained')
b += label_box(W / 2 + 175, 44, 135, 52, 'box', 'reference', 'frozen')
b += label_box(W / 2 + 25, 110, 285, 52, 'box', 'reward: model or rules', 'R1: answer checker + format check')
b.append(f'<line class="edge-dim" x1="{W / 2}" y1="12" x2="{W / 2}" y2="250"/>')
b += [text(30, 190, 'Advantage of each token comes from the', 't-muted'),
      text(30, 207, 'value model (GAE). The value model must', 't-muted'),
      text(30, 224, 'guess the final reward from a half answer.', 't-muted'),
      text(W / 2 + 25, 190, 'Advantage of each answer comes from its', 't-muted'),
      text(W / 2 + 25, 207, 'group: sample G answers to one question,', 't-muted'),
      text(W / 2 + 25, 224, 'compare each reward with the group mean.', 't-muted')]
F['ppo_grpo'] = svg(W, 250, 'PPO keeps a policy, a reference, a reward model and a trained value model. GRPO drops the value model and uses the group of answers as the baseline.', b)

# 2. our group of four -------------------------------------------------------------------------
rows = [('o1', 'think tags, boxed 56', 1, 1), ('o2', 'think tags, boxed 54', 0, 1),
        ('o3', 'think tags, boxed 63', 0, 1), ('o4', '"The answer is 56."', 0, 0)]
A = R['adv_group4']
b = [text(20, 24, 'Question: What is 7 x 8? Four sampled answers, scored by rules', 't-title')]
hdr = [(250, 'accuracy'), (330, 'format'), (410, 'reward'), (560, 'advantage')]
for x, h in hdr:
    b.append(text(x, 52, h, 't-tick', 'middle'))
for i, (n, desc, acc, fmt) in enumerate(rows):
    y = 64 + i * 46
    cls = 'box-1' if acc + fmt == 2 else ('box-2' if acc + fmt == 0 else 'box')
    b.append(box(20, y, 190, 36, cls, rx=6))
    b.append(text(32, y + 23, n, 't-note'))
    b.append(text(62, y + 23, desc, 't-muted'))
    b.append(text(250, y + 23, f'{acc}', 't-val', 'middle'))
    b.append(text(290, y + 23, '+', 't-muted', 'middle'))
    b.append(text(330, y + 23, f'{fmt}', 't-val', 'middle'))
    b.append(text(370, y + 23, '=', 't-muted', 'middle'))
    b.append(text(410, y + 23, f'{acc + fmt}', 't-val', 'middle'))
    a = A[i]
    cx = 560
    w = abs(a) / 1.5 * 60
    if a > 0.01:
        b.append(f'<path class="s1" d="{bar_path(cx, y + 8, w, 20, horizontal=True)}"/>')
        b.append(text(cx + w + 6, y + 23, f'{a:+.3f}', 't-s1'))
    elif a < -0.01:
        b.append(f'<path class="s2" d="{bar_path(cx - w, y + 8, w, 20, horizontal=True)}" transform="rotate(180 {cx - w / 2} {y + 18})"/>')
        b.append(text(cx - w - 6, y + 23, f'{a:+.3f}', 't-bad', 'end'))
    else:
        b.append(text(cx + 6, y + 23, '0.000', 't-muted'))
b.append(f'<line class="axis" x1="560" y1="60" x2="560" y2="244"/>')
b.append(text(20, 270, 'mean = 1, std = 0.707 (divide by G = 4).  A = (reward - mean) / std', 't-note'))
b.append(text(20, 290, 'o4 holds the right number but gets the lowest score: no box, no tags, nothing a rule can check.', 't-muted'))
F['group4'] = svg(W, 300, 'Four answers to 7 x 8 get rewards 2, 1, 1, 0 and advantages +1.414, 0, 0, -1.414.', b)

# 3. clipped surrogate -------------------------------------------------------------------------
cur = R['clip_curve']
L, T, PW, PH = 70, 40, 520, 200
X = lambda r: L + (r - 0.3) / (2.0 - 0.3) * PW
Y = lambda v: T + PH / 2 - v / 2.2 * (PH / 2)
b = [text(20, 22, 'What one token adds to the objective, as its probability ratio moves (eps = 0.2)', 't-title')]
for v in (-2, -1, 0, 1, 2):
    b += [f'<line class="grid" x1="{L}" y1="{Y(v):.1f}" x2="{L + PW}" y2="{Y(v):.1f}"/>', text(L - 8, Y(v) + 4, f'{v:+d}' if v else '0', 't-tick', 'end')]
for rr in (0.5, 0.8, 1.0, 1.2, 1.5, 2.0):
    b.append(text(X(rr), T + PH + 18, f'{rr}', 't-tick', 'middle'))
b.append(f'<rect class="box-ghost" x="{X(0.8):.1f}" y="{T}" width="{X(1.2) - X(0.8):.1f}" height="{PH}"/>')
b.append(text((X(0.8) + X(1.2)) / 2, T + 14, 'trust region', 't-muted', 'middle'))
b.append(f'<polyline class="l1" points="' + ' '.join(f'{X(r):.1f},{Y(p):.1f}' for r, p, n in cur) + '"/>')
b.append(f'<polyline class="l2" points="' + ' '.join(f'{X(r):.1f},{Y(n):.1f}' for r, p, n in cur) + '"/>')
b.append(text(X(2.0) + 6, Y(1.2) + 4, 'A = +1: flat', 't-s1'))
b.append(text(X(2.0) + 6, Y(1.2) + 20, 'above 1.2', 't-s1'))
b.append(text(X(0.32), Y(-1.55), 'A = -1: keeps falling', 't-bad'))
b.append(text(X(0.32), Y(-1.55) + 17, 'as the ratio grows past 1.2', 't-bad'))
b.append(text(L + PW / 2, T + PH + 40, 'ratio = new probability / old probability of the sampled token', 't-tick', 'middle'))
F['clip'] = svg(W, 290, 'The clipped surrogate: for a positive advantage the gain stops growing above ratio 1.2; for a negative advantage the gain stops below 0.8 but the loss keeps growing above 1.2.', b)

# 4. R1-Zero training loop ---------------------------------------------------------------------
b = [text(20, 22, 'One step of DeepSeek-R1-Zero (numbers from the Nature version)', 't-title')]
b += label_box(20, 60, 130, 60, 'box-2', '32 questions', 'maths, code, logic')
b += label_box(190, 60, 140, 60, 'box-on', 'policy', 'V3-Base at step 0')
b += label_box(370, 50, 130, 80, 'box', '16 answers each', '512 in total')
b += label_box(540, 40, 120, 44, 'box-1', 'answer checker')
b += label_box(540, 96, 120, 44, 'box-1', 'format checker')
b += [arrow(150, 90, 188, 90), arrow(330, 90, 368, 90), arrow(500, 74, 538, 62), arrow(500, 106, 538, 118)]
b += label_box(370, 190, 290, 56, 'box', 'rewards -> group advantages', 'reward = accuracy + format, A = (r - mean)/std')
b += label_box(20, 190, 310, 56, 'box-on', 'GRPO update of the policy', 'lr 3e-6, KL weight 0.001, clip ratio')
b += [arrow(600, 140, 600, 188), arrow(368, 218, 332, 218), arrow(260, 190, 260, 124, on=True)]
b.append(text(20, 280, 'Repeat 10,400 steps. Every 400 steps the reference model becomes a copy of the policy.', 't-muted'))
b.append(text(20, 298, 'Answer length cap: 32,768 tokens, raised to 65,536 at step 8,200.', 't-muted'))
F['r1zero_loop'] = svg(W, 310, 'R1-Zero training loop: 32 questions, 16 answers each, rule-based checks, group advantages, a GRPO update.', b)

# 5. AIME numbers by version -------------------------------------------------------------------
items = [('start of RL', 15.6, 'both versions', 's3'), ('R1-Zero, arXiv v1 (Jan 2025)', 71.0, 'pass@1', 's2'),
         ('R1-Zero, Nature / arXiv v2', 77.9, 'pass@1', 's1'), ('R1-Zero, majority vote', 86.7, 'cons@16 (v2), cons@64 (v1)', 's4')]
b = [text(20, 22, 'R1-Zero on AIME 2024: which number, from which version', 't-title')]
for i, (n, v, note, cls) in enumerate(items):
    y = 44 + i * 52
    b.append(text(20, y + 20, n, 't-note'))
    b.append(text(20, y + 37, note, 't-muted'))
    w = v / 100 * 330
    b.append(f'<path class="{cls}" d="{bar_path(270, y + 6, w, 28, horizontal=True)}"/>')
    b.append(text(270 + w + 8, y + 26, f'{v}%', 't-val'))
F['aime_versions'] = svg(W, 260, 'R1-Zero AIME 2024 accuracy: 15.6% at the start, 71.0% in arXiv v1, 77.9% in the Nature version, 86.7% with majority voting.', b)

# 6. the R1 pipeline --------------------------------------------------------------------------
b = [text(20, 22, 'DeepSeek-R1: four stages, each starting from a checkpoint', 't-title')]
st = [('1. cold start SFT', 'thousands of long, readable CoT', 'V3-Base -> Dev1'),
      ('2. reasoning RL', 'rules + language consistency', 'Dev1 -> Dev2'),
      ('3. rejection sampling + SFT', '600K reasoning + 200K other', 'fresh V3-Base -> Dev3'),
      ('4. RL for all scenarios', 'rules + reward models', 'Dev3 -> R1')]
for i, (t1, t2, t3) in enumerate(st):
    y = 44 + i * 66
    b += label_box(20, y, 300, 52, 'box-on' if i in (1, 3) else 'box-1', t1, t2)
    b.append(text(340, y + 31, t3, 't-note'))
    if i < 3:
        b.append(arrow(170, y + 52, 170, y + 64))
b.append(f'<path class="edge" d="M 470 136 C 530 136, 530 202, 470 202" marker-end="url(#ah)"/>')
b.append(text(540, 166, 'Dev2 writes the', 't-muted'))
b.append(text(540, 182, '600K reasoning', 't-muted'))
b.append(text(540, 198, 'samples', 't-muted'))
b += [text(20, 320, 'Stage 3 trains a fresh copy of V3-Base for two epochs on about 800K samples.', 't-muted'),
      text(20, 338, 'The same 800K samples later train the distilled Qwen and Llama models.', 't-muted')]
F['pipeline'] = svg(W, 350, 'The four stages of DeepSeek-R1: cold-start SFT, reasoning RL, rejection sampling with SFT on 800K samples, and RL for all scenarios.', b)

# 7. stage by stage results (R1 Nature version, Table 3) --------------------------------------
cols = ['R1-Zero', 'Dev1', 'Dev2', 'Dev3', 'R1']
series = [('AIME 2024', [77.9, 59.0, 74.0, 78.1, 79.8], 'l1', 's1'),
          ('IF-Eval', [46.6, 71.7, 72.0, 78.1, 83.3], 'l3', 's3'),
          ('AlpacaEval 2.0', [24.7, 50.1, 55.8, 62.1, 87.6], 'l2', 's2')]
L, T, PW, PH = 60, 40, 440, 200
X = lambda i: L + i * PW / 4
Y = lambda v: T + PH - v / 100 * PH
b = [text(20, 22, 'What each stage buys (Nature version, Table 3)', 't-title')]
for v in (0, 25, 50, 75, 100):
    b += [f'<line class="grid" x1="{L}" y1="{Y(v):.1f}" x2="{L + PW}" y2="{Y(v):.1f}"/>', text(L - 8, Y(v) + 4, str(v), 't-tick', 'end')]
for i, c in enumerate(cols):
    b.append(text(X(i), T + PH + 20, c, 't-tick', 'middle'))
ends = []
for n, vals, lc, dc in series:
    b.append(f'<polyline class="{lc}" points="' + ' '.join(f'{X(i):.1f},{Y(v):.1f}' for i, v in enumerate(vals)) + '"/>')
    for i, v in enumerate(vals):
        b.append(f'<g class="mark"><title>{esc(n)}, {cols[i]}: {v}</title><circle class="{dc} ring" cx="{X(i):.1f}" cy="{Y(v):.1f}" r="5"/></g>')
    ends.append([Y(vals[-1]), f'{n}: {vals[-1]}', dc])
ends.sort()
for i in range(1, len(ends)):
    ends[i][0] = max(ends[i][0], ends[i - 1][0] + 18)
for ly, lab, dc in ends:
    b += [f'<rect class="{dc}" x="{X(4) + 16}" y="{ly - 5:.1f}" width="10" height="10" rx="2"/>', text(X(4) + 32, ly + 4, lab, 't-note')]
b.append(text(L, T + PH + 44, 'The cold start (Dev1) costs AIME points; reasoning RL (Dev2) wins them back.', 't-muted'))
F['stages'] = svg(W, 300, 'Scores of R1-Zero, Dev1, Dev2, Dev3 and R1 on AIME 2024, IF-Eval and AlpacaEval 2.0.', b)

# 8. distillation vs RL on a 32B base ---------------------------------------------------------
bench = ['AIME 2024', 'MATH-500', 'GPQA Diamond', 'LiveCodeBench']
rl = [47.0, 91.6, 55.0, 40.2]
di = [72.6, 94.3, 62.1, 57.2]
b = [text(20, 22, 'Same Qwen2.5-32B base, two ways to teach it to reason', 't-title'),
     f'<rect class="s3" x="20" y="36" width="12" height="12" rx="2"/>', text(38, 47, 'large-scale RL, over 10K steps', 't-note'),
     f'<rect class="s1" x="300" y="36" width="12" height="12" rx="2"/>', text(318, 47, 'SFT on 800K R1 samples (distilled)', 't-note')]
for i, (n, a, d) in enumerate(zip(bench, rl, di)):
    y = 70 + i * 58
    b.append(text(20, y + 28, n, 't-note'))
    for j, (v, cls) in enumerate(((a, 's3'), (d, 's1'))):
        w = v / 100 * 380
        b.append(f'<path class="{cls}" d="{bar_path(160, y + j * 22, w, 18, horizontal=True)}"/>')
        b.append(text(160 + w + 6, y + j * 22 + 14, f'{v}', 't-val'))
b.append(text(20, 312, 'Pass@1 in %, arXiv v1 Table 6 (same numbers in the Nature version, Table 16).', 't-muted'))
F['distill'] = svg(W, 322, 'On Qwen2.5-32B, distillation from R1 beats large-scale RL on all four benchmarks.', b)

# 9. cost --------------------------------------------------------------------------------------
items = [('R1-Zero RL', 101, 's1'), ('SFT data creation', 5, 's3'), ('R1 (all its stages)', 41, 's1'),
         ('R1 total', 147, 's2'), ('V3 base model, for scale', 2788, 's4')]
b = [text(20, 22, 'GPU hours (thousands of H800 hours), at the papers\' $2 per hour', 't-title')]
mx = 2788
for i, (n, v, cls) in enumerate(items):
    y = 42 + i * 44
    b.append(text(20, y + 22, n, 't-note'))
    w = max(2, v / mx * 330)
    b.append(f'<path class="{cls}" d="{bar_path(190, y + 6, w, 24, horizontal=True)}"/>')
    b.append(text(190 + w + 8, y + 23, f'{v:,}K = ${v * 2:,}K' if v < 1000 else f'{v:,}K = $5.576M', 't-val'))
b.append(text(20, 280, 'R1 rows: Nature version, Supplementary Table 7. V3 row: V3 report, Table 1. Neither counts research runs.', 't-muted'))
F['cost'] = svg(W, 292, 'R1 cost 147K GPU hours, or $294K: 101K for R1-Zero, 5K for SFT data, 41K for R1. The V3 base model cost 2,788K GPU hours.', b)

# 10. keep routing -----------------------------------------------------------------------------
b = [text(20, 22, 'Keep Routing: train the same experts that wrote the answer', 't-title')]
for k, (lab, y, exps, cls) in enumerate([('sampling engine', 50, (2, 5), 'box-1'), ('training, routes recomputed', 140, (2, 6), 'box-2'),
                                         ('training with Keep Routing', 230, (2, 5), 'box-1')]):
    b.append(text(20, y + 26, lab, 't-note'))
    b += label_box(220, y + 6, 70, 32, 'box', 'token')
    for e in range(8):
        x = 330 + e * 40
        b.append(box(x, y + 6, 32, 32, 'box-on' if e in exps else 'box', rx=5))
        b.append(text(x + 16, y + 27, str(e), 't-tick', 'middle'))
    b.append(arrow(290, y + 22, 326, y + 22))
b.append(text(330, 122, 'expert 5 picked: its weights produced the token', 't-muted'))
b.append(text(330, 212, 'tiny number differences pick expert 6 instead', 't-bad'))
b.append(text(330, 302, 'the recorded route is replayed: same experts', 't-muted'))
F['keep_routing'] = svg(W, 312, 'In sampling the router picks experts 2 and 5. Recomputing the route during training may pick 2 and 6. Keep Routing replays the recorded route.', b)

# 11. on-policy distillation -------------------------------------------------------------------
b = [text(20, 22, 'V4: many specialists teach one student, on the student\'s own answers', 't-title')]
teachers = ['maths', 'code', 'agents', 'instructions', '...']
for i, t in enumerate(teachers):
    b += label_box(20 + i * 130, 44, 116, 40, 'box-1', t)
b.append(text(20, 102, 'each specialist: SFT, then GRPO on its own domain (more than ten teachers in total)', 't-muted'))
b += label_box(220, 140, 240, 52, 'box-on', 'student', 'writes its own answer')
b += label_box(170, 236, 340, 52, 'box', 'loss: sum of w_i x KL(student || teacher_i)', 'full vocabulary, every token')
b += [arrow(340, 192, 340, 234), arrow(340, 112, 340, 138)]
b.append(text(480, 158, 'teachers score every token', 't-muted'))
b.append(text(480, 176, 'of the student\'s answer', 't-muted'))
F['opd'] = svg(W, 300, 'On-policy distillation in V4: the student writes answers; more than ten specialist teachers give their full token distributions; the student minimises a weighted reverse KL.', b)

# 12. toy RL run ---------------------------------------------------------------------------------
cur = R['toy_run']
L, T, PW, PH = 60, 40, 440, 200
X = lambda s: L + s / 40 * PW
Y = lambda v: T + PH - v * PH
b = [text(20, 22, 'Toy GRPO run: probability of each answer to 7 x 8 (our run)', 't-title')]
for v in (0, 0.25, 0.5, 0.75, 1.0):
    b += [f'<line class="grid" x1="{L}" y1="{Y(v):.1f}" x2="{L + PW}" y2="{Y(v):.1f}"/>', text(L - 8, Y(v) + 4, f'{v:.2f}', 't-tick', 'end')]
for s in (0, 10, 20, 30, 40):
    b.append(text(X(s), T + PH + 18, str(s), 't-tick', 'middle'))
names = ['o1 right + format', 'o2 wrong (54)', 'o3 wrong (63)', 'o4 right, no format']
cls = [('l1', 's1'), ('l3', 's3'), ('l4', 's4'), ('l2', 's2')]
ends = []
for k in range(4):
    b.append(f'<polyline class="{cls[k][0]}" points="' + ' '.join(f'{X(c[0]):.1f},{Y(c[k + 1]):.1f}' for c in cur) + '"/>')
    ends.append([Y(cur[-1][k + 1]), f'{names[k]}: {cur[-1][k + 1]:.3f}', cls[k][1]])
ends.sort()
for i in range(1, len(ends)):
    ends[i][0] = max(ends[i][0], ends[i - 1][0] + 17)
for ly, lab, dc in ends:
    b += [f'<rect class="{dc}" x="{X(40) + 12}" y="{ly - 5:.1f}" width="10" height="10" rx="2"/>', text(X(40) + 28, ly + 4, lab, 't-tick')]
b.append(text(L + PW / 2, T + PH + 40, 'GRPO step (8 samples per step, eps 0.2)', 't-tick', 'middle'))
F['toy_run'] = svg(W, 290, 'In our toy run, the right and well-formatted answer rises from 0.126 to 0.970 in 30 steps, then the curve stops moving.', b)

# 13. timeline of post-training --------------------------------------------------------------
ev = [('Feb 2024', 'DeepSeekMath', 'GRPO: no value model'),
      ('Jan 2025', 'R1-Zero and R1', 'pure RL; four-stage pipeline'),
      ('Aug 2025', 'V3.1', 'thinking + non-thinking in one model'),
      ('Nov 2025', 'DeepSeekMath-V2', 'a verifier as the reward'),
      ('Dec 2025', 'V3.2, V3.2-Speciale', 'RL > 10% of pre-training cost'),
      ('Apr 2026', 'V4', 'mixed RL replaced by on-policy distillation'),
      ('Sep 2026', 'V4.1-Flash', 'same algorithms; better data and environments')]
b = [text(20, 22, 'How DeepSeek post-training changed, 2024 to 2026', 't-title'), f'<line class="axis" x1="110" y1="40" x2="110" y2="{40 + 40 * len(ev)}"/>']
for i, (d, m, what) in enumerate(ev):
    y = 56 + i * 40
    b.append(f'<circle class="s1 ring" cx="110" cy="{y}" r="6"/>')
    b.append(text(98, y + 4, d, 't-tick', 'end'))
    b.append(text(126, y + 4, m, 't-note'))
    b.append(text(310, y + 4, what, 't-muted'))
F['timeline'] = svg(W, 60 + 40 * len(ev), 'Timeline: GRPO in DeepSeekMath, R1, V3.1 hybrid thinking, DeepSeekMath-V2, V3.2, V4 on-policy distillation, V4.1 data and environments.', b)

json.dump(F, open(os.path.join(RESULTS, 'figs_part4.json'), 'w'))
print(f'wrote {len(F)} figures to results/figs_part4.json: {", ".join(F)}')
