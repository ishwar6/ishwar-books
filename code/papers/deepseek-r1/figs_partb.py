"""Figures for the Background page (the road to R1), partly from results/r1_partb_*.json.
Writes results/figs_partb.json (key -> svg string). Colours come only from CSS classes."""
import json, os
from figlib import svg, text, box, arrow, lines_chart
from samfig import token, line, hbars

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
RL = json.load(open(os.path.join(RES, 'r1_partb_rl.json')))
COT = json.load(open(os.path.join(RES, 'r1_partb_cot.json')))
F = {}


def legend(x, y, items):
    b = []
    for cls, lab in items:
        b += [f'<rect class="{cls}" x="{x}" y="{y - 9}" width="10" height="10" rx="2"/>', text(x + 16, y, lab, 't-tick')]
        x += 16 + 7.2 * len(lab) + 22
    return b


# ---------------------------------------------------------------- 1. timeline
def timeline():
    ev = [('2017', 'PPO', ['clipped policy', 'updates'], 's1', 'up'),
          ('2018', 'GPT-1', ['pre-train, then', 'fine-tune'], 's1', 'down'),
          ('2022', 'Chain of thought', ['write the steps', 'before the answer'], 's2', 'up'),
          ('2022', 'Self-consistency', ['sample many,', 'take the vote'], 's2', 'down'),
          ('2022', 'InstructGPT', ['SFT, reward model,', 'PPO (RLHF)'], 's1', 'up'),
          ('2023', 'Process rewards', ['a score for', 'every step'], 's2', 'down'),
          ('Feb 2024', 'DeepSeekMath', ['GRPO: PPO', 'without a critic'], 's3', 'up'),
          ('Sep 2024', 'OpenAI o1', ['long thinking,', 'method not public'], 's2', 'down'),
          ('Dec 2024', 'DeepSeek-V3', ['the base model', 'for R1'], 's3', 'up'),
          ('Jan 2025', 'DeepSeek-R1', ['reasoning from', 'RL with rules'], 's3', 'down')]
    x0, dx, y = 50, 74, 150
    b = [line(24, y, 746, y, 'axis', 'ah')]
    for k, (yr, name, ls, cls, side) in enumerate(ev):
        x = x0 + dx * k
        b.append(f'<circle class="{cls}" cx="{x}" cy="{y}" r="7"/>')
        if side == 'up':
            b += [line(x, y - 9, x, y - 26, 'grid'), text(x, y - 98, yr, 't-title', 'middle'), text(x, y - 78, name, 't-note', 'middle')]
            for j, l in enumerate(ls):
                b.append(text(x, y - 58 + 15 * j, l, 't-muted', 'middle'))
        else:
            b += [line(x, y + 9, x, y + 26, 'grid'), text(x, y + 44, yr, 't-title', 'middle'), text(x, y + 64, name, 't-note', 'middle')]
            for j, l in enumerate(ls):
                b.append(text(x, y + 84 + 15 * j, l, 't-muted', 'middle'))
    b += legend(24, 290, [('s1', 'training methods'), ('s2', 'reasoning and test time'), ('s3', 'DeepSeek')])
    return svg(770, 304, 'Timeline: PPO 2017, GPT-1 2018, chain of thought, self-consistency and InstructGPT 2022, process '
               'reward models 2023, DeepSeekMath with GRPO, OpenAI o1 and DeepSeek-V3 in 2024, and DeepSeek-R1 in January 2025.', b)


F['pb_timeline'] = timeline()


# ---------------------------------------------------------------- 2. the stages of training: you are here
def stages():
    st = [('Pre-training', 'next token on', 'trillions of tokens', 'box-1'),
          ('SFT', 'copy written', 'answers', 'box-1'),
          ('Reward model', 'learn from', 'human rankings', 'box-2'),
          ('RL with PPO', 'raise the score,', 'stay near SFT', 'box-2')]
    b = []
    for k, (name, a, c, cls) in enumerate(st):
        x = 20 + 185 * k
        b += [box(x, 30, 160, 84, cls), text(x + 80, 56, name, 't-title', 'middle'), text(x + 80, 80, a, 't-muted', 'middle'),
              text(x + 80, 98, c, 't-muted', 'middle')]
        if k:
            b.append(arrow(x - 24, 72, x - 2, 72))
    b += [text(20 + 80, 140, 'base model', 't-tick', 'middle'), text(205 + 80, 140, 'chat model', 't-tick', 'middle'),
          text(205 + 185 * 1.5 + 80 - 92, 140, 'RLHF (InstructGPT, 2022)', 't-tick', 'middle')]
    return svg(760, 152, 'Four stages: pre-training gives a base model, SFT gives a chat model, then a reward model and RL '
               'with PPO make up RLHF.', b)


F['pb_stages'] = stages()


# ---------------------------------------------------------------- 3. RLHF with PPO: the four models
def rlhf():
    b = []
    b += token(20, 128, 90, 'prompt x', 'box', 34, 't-note')
    b += [box(160, 110, 150, 70, 'box-on'), text(235, 138, 'Policy', 't-title', 'middle'), text(235, 160, 'trained (the LLM)', 't-muted', 'middle')]
    b.append(arrow(112, 145, 158, 145, True))
    b += token(360, 128, 110, 'response y', 'box', 34, 't-note')
    b.append(arrow(312, 145, 358, 145, True))
    # the three helpers
    helpers = [(20, 'Reference', 'frozen copy of SFT', 'box-1', 'KL penalty'),
               (300, 'Reward model', 'frozen, from rankings', 'box-1', 'score r(x, y)'),
               (580, 'Critic (value)', 'trained, same size', 'box-on', 'baseline V')]
    for x, name, sub, cls, out in helpers:
        b += [box(x, 250, 160, 66, cls), text(x + 80, 276, name, 't-title', 'middle'), text(x + 80, 298, sub, 't-muted', 'middle'),
              text(x + 80, 336, out, 't-note', 'middle')]
        b.append(line(415, 164, x + 80, 248, 'edge', 'ah'))
    b += [box(560, 108, 180, 74, 'box-2'), text(650, 134, 'PPO update', 't-title', 'middle'),
          text(650, 154, 'advantage = reward', 't-muted', 'middle'), text(650, 170, 'minus baseline, clipped', 't-muted', 'middle')]
    b.append(arrow(472, 145, 558, 145))
    b.append(line(650, 106, 650, 70, 'path'))
    b.append(line(650, 70, 235, 70, 'path'))
    b.append(line(235, 70, 235, 108, 'path', 'ah-on'))
    b.append(text(440, 62, 'new weights for the policy', 't-tick', 'middle'))
    b += legend(20, 30, [('box-on', 'trained (two big models)'), ('box-1', 'frozen (two more models)')])
    return svg(760, 350, 'RLHF with PPO. The policy writes a response. A frozen reference model gives a KL penalty, a frozen '
               'reward model gives a score, and a trained critic gives a baseline. PPO uses them to update the policy.', b)


F['pb_rlhf'] = rlhf()


# ---------------------------------------------------------------- 4. REINFORCE training curves
def reinforce():
    c = RL['curves']
    xs = list(range(1, 301, 10))
    ser = [('+5, no baseline', c['rewards +5, no baseline']['mean'], 'l1', 's1'),
           ('+5, baseline', c['rewards +5, baseline']['mean'], 'l2', 's2'),
           ('0/1, no baseline', c['rewards +0, no baseline']['mean'], 'l3', 's3')]
    return lines_chart(760, 300, 'Probability of the right answer during REINFORCE training, average of 200 seeds. With every '
                       'reward shifted up by 5, training without a baseline stalls; with a baseline it learns as fast as with 0/1 rewards.',
                       xs, ser, 'training step (one sample per step)', 'P(right answer), mean of 200 seeds', 0, 1,
                       [0, 0.25, 0.5, 0.75, 1], xticks=[1, 51, 101, 151, 201, 251, 291], fmt=lambda v: f'{v:.2f}', right=200)


F['pb_reinforce'] = reinforce()


# ---------------------------------------------------------------- 5. PPO clip: one term as a function of the ratio
def ppo_clip():
    b = []
    for k, (A, title) in enumerate([(1, 'advantage A = +1 (a good answer)'), (-1, 'advantage A = -1 (a bad answer)')]):
        X0, Y0, W, H = 60 + 370 * k, 40, 280, 200
        X = lambda r: X0 + (r - 0.4) / (1.7 - 0.4) * W
        Y = lambda v: Y0 + H / 2 - v * H / 2 / 1.8
        b.append(text(X0 + W / 2, 22, title, 't-title', 'middle'))
        b.append(line(X0, Y(0), X0 + W, Y(0), 'axis'))
        for r in [0.8, 1.0, 1.2]:
            b += [line(X(r), Y0, X(r), Y0 + H, 'grid'), text(X(r), Y0 + H + 16, f'{r:.1f}', 't-tick', 'middle')]
        rs = [0.4 + 0.01 * i for i in range(131)]
        un = ' '.join(f'{X(r):.1f},{Y(r * A):.1f}' for r in rs)
        cl = ' '.join(f'{X(r):.1f},{Y(min(r * A, min(max(r, 0.8), 1.2) * A)):.1f}' for r in rs)
        b.append(f'<polyline class="l3" style="stroke-dasharray:5 4" points="{un}"/>')
        b.append(f'<polyline class="l1" points="{cl}"/>')
        b.append(f'<circle class="s1" cx="{X(1.0):.1f}" cy="{Y(A):.1f}" r="5"/>')
        b.append(text(X0 + W / 2, Y0 + H + 36, 'ratio = new probability / old probability', 't-tick', 'middle'))
        flat = 'flat: no gain past 1.2' if A > 0 else 'flat: no gain below 0.8'
        b.append(text(X(1.45) if A > 0 else X(0.62), Y(1.2 * A) - 10 * A, flat, 't-note', 'middle'))
    b += legend(60, 300, [('s1', 'PPO clipped objective'), ('s3', 'unclipped ratio x A')])
    return svg(760, 312, 'One term of the PPO clipped objective against the probability ratio. For a good answer the gain stops '
               'at ratio 1.2; for a bad answer it stops at 0.8. The dashed line is the unclipped product.', b)


F['pb_ppo_clip'] = ppo_clip()


# ---------------------------------------------------------------- 6. chain of thought and voting, our run
def cot():
    n = COT['N']
    items = [('direct answer', COT['direct'] / n), ('step by step', COT['cot'] / n), ('one sample', COT['one']),
             ('vote of 4', COT['maj']['4']), ('vote of 8', COT['maj']['8']), ('vote of 16', COT['maj']['16'])]
    b = hbars(20, 40, items, 470, row_h=34, cls='s2', label_w=130, fmt=lambda v: f'{100 * v:.0f}%', vmax=1.0,
              title=f'Qwen2.5-0.5B-Instruct, first {n} GSM8K test questions', hi='vote of 16')
    return svg(760, 260, f'Accuracy of Qwen2.5-0.5B-Instruct on {n} GSM8K questions: a direct answer, one greedy chain of '
               'thought, one sampled chain, and majority votes over 4, 8 and 16 sampled chains.', b)


F['pb_cot'] = cot()

json.dump(F, open(os.path.join(RES, 'figs_partb.json'), 'w'))
print('figures:', ', '.join(F))
