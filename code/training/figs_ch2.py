"""Figures for Chapter 2 (how it started, and how it evolved). Writes results/figs_ch2.json (key -> svg).
Colours come only from CSS classes (box, box-1..4, s1..s4, l1..l4, edge, path ...), so every figure follows the site's themes.
Text widths: about 7 px per character for t-tick / t-muted, 7.5 for t-note, 8.5 for t-title; 17 px between baselines.
Numbers come from results/ch2_*.json, written by the experiment scripts."""
import json, math
from figlib import svg, text, box, arrow, esc
from agentfig import line

R = lambda n: json.load(open(f'results/{n}.json'))
BT, KL, DPO, GRPO = R('ch2_bradley_terry'), R('ch2_kl'), R('ch2_dpo'), R('ch2_grpo')
F = {}


# ---------------------------------------------------------------- helpers (no colours, only classes)
def para(x, y, lines, cls='t-muted', anchor='start', lh=17):
    return [text(x, y + i * lh, l, cls, anchor) for i, l in enumerate(lines)]


def block(x, y, w, h, title, lines=(), cls='box', tcls='t-note', lcls='t-tick', rx=10):
    b = [box(x, y, w, h, cls, rx)]
    ty = y + (h / 2 + 5 if not lines else 22)
    b.append(text(x + w / 2, ty, title, tcls, 'middle'))
    b += para(x + w / 2, y + 40, lines, lcls, 'middle')
    return b


def hpath(points, cls='edge', marker='ah'):
    d = 'M' + ' L'.join(f'{x:.1f},{y:.1f}' for x, y in points)
    return f'<path class="{cls}" d="{d}" marker-end="url(#{marker})"/>'


def dot(x, y, r, cls):
    return f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="{r}"/>'


def rect(x, y, w, h, cls, rx=3, title=None):
    r = f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{max(w, 0.5):.1f}" height="{max(h, 0.5):.1f}" rx="{rx}"/>'
    return f'<g class="mark"><title>{esc(title)}</title>{r}</g>' if title else r


# ---------------------------------------------------------------- 1. the big timeline
THREAD = {'p': ('s1', 'pretraining and scale'), 'r': ('s2', 'learning from preferences (RL)'),
          'i': ('s3', 'instruction data (SFT)'), 'x': ('s4', 'RL for reasoning')}
EVENTS = [
    ('1948', 'p', 'Shannon: n-gram approximations of English'),
    ('1951', 'p', 'Shannon: people predicting the next letter'),
    ('2003', 'p', 'Bengio et al.: neural network language model'),
    ('2013', 'p', 'word2vec: words as learned vectors'),
    ('2014', 'p', 'seq2seq and attention for translation'),
    ('2017', 'p', 'Transformer (June)'),
    ('2017', 'r', 'Christiano et al.: RL from human preferences (June)'),
    ('2017', 'r', 'PPO, the RL algorithm RLHF will use (July)'),
    ('2018', 'p', 'GPT-1 (June), BERT (October): pretrain, then fine-tune'),
    ('2019', 'p', 'GPT-2: one model, many tasks, zero-shot (February)'),
    ('2019', 'r', 'Ziegler et al.: RLHF on GPT-2, KL penalty (September)'),
    ('2020', 'p', 'Scaling laws (January), GPT-3 and few-shot prompts (May)'),
    ('2020', 'r', 'Stiennon et al.: summaries from human feedback (September)'),
    ('2021', 'i', 'Natural Instructions (April), FLAN (September), T0 (October)'),
    ('2021', 'r', 'HHH assistant (Askell et al.), WebGPT (December)'),
    ('2022', 'r', 'InstructGPT: SFT, reward model, PPO (March)'),
    ('2022', 'p', 'Chinchilla: more tokens per parameter (March)'),
    ('2022', 'r', 'HH-RLHF (April), Sparrow (September)'),
    ('2022', 'i', 'Super-NaturalInstructions (April), Flan-PaLM (October)'),
    ('2022', 'r', 'ChatGPT (November 30)'),
    ('2022', 'r', 'Constitutional AI: feedback from AI (December)'),
    ('2022', 'i', 'Self-Instruct: a model writes its own instructions (December)'),
    ('2023', 'p', 'LLaMA: strong open base models (February)'),
    ('2023', 'i', 'Alpaca (March), LIMA: 1,000 examples (May)'),
    ('2023', 'r', 'DPO: preferences without RL (May)'),
    ('2023', 'x', "Let's Verify Step by Step: process rewards (May)"),
    ('2023', 'r', 'Llama 2-Chat: rejection sampling + PPO (July)'),
    ('2023', 'r', 'Zephyr (October), Tulu 2 (November): DPO goes open'),
    ('2024', 'x', 'DeepSeekMath: GRPO (February)'),
    ('2024', 'x', 'OpenAI o1: RL to think before answering (September)'),
    ('2024', 'x', 'Tulu 3: RL with verifiable rewards (November)'),
    ('2025', 'x', 'DeepSeek-R1 and R1-Zero (January)'),
]


def timeline():
    W, top, lh = 760, 92, 25
    b = [text(20, 26, 'From next-word prediction to reasoning models: the events in this chapter', 't-title')]
    for (k, (c, lab)), x in zip(THREAD.items(), [20, 190, 420, 600]):
        b += [dot(x + 6, 52, 6, c), text(x + 18, 56, lab, 't-tick')]
    x0 = 92
    y_end = top + lh * (len(EVENTS) - 1)
    b.append(line(x0, top - 12, x0, y_end + 12, 'axis'))
    prev = None
    for i, (yr, th, s) in enumerate(EVENTS):
        y = top + i * lh
        if yr != prev:
            b.append(text(x0 - 16, y + 4, yr, 't-note', 'end'))
            if prev is not None:
                b.append(line(30, y - lh / 2, W - 20, y - lh / 2, 'grid'))
            prev = yr
        b.append(dot(x0, y, 6, THREAD[th][0]))
        b.append(text(x0 + 16, y + 4, s, 't-tick'))
    b.append(text(20, y_end + 40, 'Months are the first public version (usually arXiv). Rows are in time order, not to scale.', 't-muted'))
    return svg(W, y_end + 56, 'A vertical timeline of the events in this chapter from 1948 to 2025, coloured by thread: pretraining and scale, learning from preferences, instruction data, and RL for reasoning.', b)


F['ch2_timeline'] = timeline()


# ---------------------------------------------------------------- 2. three threads merging
def threads():
    b = [text(380, 24, 'Three threads that met in 2022', 't-title', 'middle')]
    rows = [('box-1', 'Pretraining and scale', 'Transformer, GPT-1/2/3, scaling laws', 'gives: knowledge and fluency'),
            ('box-2', 'Learning from preferences', 'Christiano 2017, Ziegler 2019, Stiennon 2020', 'gives: a reward for "better"'),
            ('box-3', 'Instruction data', 'Natural Instructions, FLAN, T0', 'gives: the habit of following a request')]
    ys = [50, 140, 230]
    for (c, t, s, g), y in zip(rows, ys):
        b += [box(20, y, 330, 70, c, 10), text(36, y + 24, t, 't-note'), text(36, y + 44, s, 't-tick'), text(36, y + 61, g, 't-muted')]
        b.append(hpath([(350, y + 35), (410, y + 35), (440, 175)], 'edge'))
    b += [box(440, 125, 150, 100, 'box-on', 12), text(515, 160, 'InstructGPT', 't-note', 'middle'),
          text(515, 180, 'March 2022', 't-tick', 'middle'), text(515, 198, 'SFT + RM + PPO', 't-tick', 'middle')]
    b += [arrow(590, 175, 630, 175, True), box(630, 140, 110, 70, 'box-on', 12), text(685, 170, 'ChatGPT', 't-note', 'middle'),
          text(685, 190, 'Nov 30, 2022', 't-tick', 'middle')]
    b.append(text(380, 325, 'The base model supplies what the assistant knows; demonstrations show the format;', 't-muted', 'middle'))
    b.append(text(380, 342, 'preferences turn "which answer is better" into a number that RL can push up.', 't-muted', 'middle'))
    return svg(760, 356, 'Three research threads, pretraining at scale, learning from human preferences, and instruction data, merging into InstructGPT in March 2022 and then ChatGPT in November 2022.', b)


F['ch2_threads'] = threads()


# ---------------------------------------------------------------- 3. the problem-then-fix chain
CHAIN = [('A base model only continues text', 'GPT-3 (2020) learns tasks from examples in the prompt', 'p'),
         ('Prompts are fragile; models ignore instructions', 'instruction tuning (FLAN, T0, 2021)', 'i'),
         ('"Good" is hard to write as a rule or a metric', 'learn a reward from comparisons (2017 to 2020)', 'r'),
         ('Optimising a learned reward breaks it', 'a KL penalty keeps the policy near the start', 'r'),
         ('Labels are slow and expensive', 'AI feedback from a written constitution (2022)', 'r'),
         ('PPO needs four models and is fiddly', 'DPO: train on pairs directly (2023)', 'r'),
         ('Preference rewards are fuzzy for maths and code', 'verifiable rewards: run a checker (2024)', 'x'),
         ('A value network is as big as the policy', 'GRPO: compare answers within a group (2024)', 'x'),
         ('Models answer too fast on hard problems', 'RL that rewards correct long reasoning (o1, R1)', 'x')]


def chain():
    b = [text(20, 24, 'Each step fixes a problem left by the step before', 't-title'),
         text(170, 50, 'problem', 't-label', 'middle'), text(560, 50, 'fix', 't-label', 'middle')]
    y0, h, g = 62, 38, 14
    for i, (p, f, th) in enumerate(CHAIN):
        y = y0 + i * (h + g)
        c = {'p': 'box-1', 'i': 'box-3', 'r': 'box-2', 'x': 'box-4'}[th]
        b += [box(20, y, 300, h, 'box', 8), text(170, y + 24, p, 't-tick', 'middle')]
        b += [arrow(320, y + h / 2, 384, y + h / 2), box(384, y, 356, h, c, 8), text(562, y + 24, f, 't-tick', 'middle')]
        if i < len(CHAIN) - 1:
            b.append(hpath([(562, y + h), (562, y + h + 6), (170, y + h + 6), (170, y + h + g)], 'edge-dim', 'ah'))
    yb = y0 + len(CHAIN) * (h + g)
    b.append(text(380, yb + 6, 'the dotted line: each fix leaves a new problem, which becomes the next row', 't-muted', 'middle'))
    return svg(760, yb + 20, 'A chain of nine rows. Each row is a problem on the left and the fix that answered it on the right; each fix leads to the problem on the next row, from base models that only continue text to reinforcement learning for reasoning.', b)


F['ch2_chain'] = chain()


# ---------------------------------------------------------------- 4. Bradley-Terry: the sigmoid of the reward gap
def bt_curve():
    W, H, L, T, pw, ph = 760, 300, 70, 40, 420, 200
    X = lambda d: L + (d + 6) / 12 * pw
    Y = lambda p: T + ph - p * ph
    b = [text(20, 24, 'Bradley-Terry: P(A preferred over B) = sigmoid(r_A - r_B)', 't-title')]
    for p in [0, 0.25, 0.5, 0.75, 1]:
        b += [line(L, Y(p), L + pw, Y(p), 'grid'), text(L - 8, Y(p) + 4, f'{p:.2f}', 't-tick', 'end')]
    for d in range(-6, 7, 2):
        b.append(text(X(d), T + ph + 18, f'{d:+d}' if d else '0', 't-tick', 'middle'))
    pts = ' '.join(f'{X(d / 10):.1f},{Y(1 / (1 + math.exp(-d / 10))):.1f}' for d in range(-60, 61))
    b.append(f'<polyline class="l1" points="{pts}"/>')
    b.append(text(L + pw / 2, T + ph + 38, 'reward gap r_A - r_B', 't-tick', 'middle'))
    gt = BT['gap_table']
    for d in ['0', '1', '2', '3']:
        p = gt[d]
        b += [dot(X(float(d)), Y(p), 5, 's2 ring'), text(X(float(d)) + 9, Y(p) + 15, f'{p:.3f}', 't-val')]
    # the worked example
    rA, rB = BT['rA'], BT['rB']
    x = 530
    b += [box(x, 60, 210, 150, 'box', 10), text(x + 105, 84, 'worked example', 't-note', 'middle'),
          text(x + 16, 110, f'r_A = {rA},  r_B = {rB}', 't-tick'), text(x + 16, 132, f'gap = {rA - rB:.1f}', 't-tick'),
          text(x + 16, 154, f'P(A over B) = sigmoid(0.8) = {1 / (1 + math.exp(-(rA - rB))):.3f}', 't-tick'),
          text(x + 16, 180, 'add 10 to both rewards:', 't-muted'), text(x + 16, 197, 'P is unchanged, only gaps count', 't-muted')]
    return svg(W, H, 'The S-shaped sigmoid curve: probability that A is preferred, as a function of the reward gap. A gap of 0 gives 0.5, 1 gives 0.731, 2 gives 0.881, 3 gives 0.953. Worked example: rewards 1.2 and 0.4 give 0.690.', b)


F['ch2_bt_curve'] = bt_curve()


# ---------------------------------------------------------------- 5. a preference pair becomes a training signal for the reward model
def pref_pair():
    b = [text(20, 24, 'From one human choice to one gradient step on the reward model', 't-title')]
    b += block(20, 50, 160, 60, 'prompt x', ['"Explain the moon landing"'], 'box')
    b += block(230, 40, 200, 44, 'answer A (chosen)', [], 'box-3', 't-tick')
    b += block(230, 100, 200, 44, 'answer B (rejected)', [], 'box', 't-tick')
    b += [arrow(180, 70, 230, 62), arrow(180, 90, 230, 122)]
    b += [box(480, 40, 110, 104, 'box-2', 10), text(535, 80, 'reward', 't-note', 'middle'), text(535, 98, 'model r', 't-note', 'middle')]
    b += [arrow(430, 62, 480, 62), arrow(430, 122, 480, 122)]
    b += [text(610, 66, 'r_A = 1.2', 't-val'), text(610, 126, 'r_B = 0.4', 't-val'), arrow(590, 62, 606, 62), arrow(590, 122, 606, 122)]
    y = 190
    steps = [('gap', 'r_A - r_B = 0.8'), ('probability', 'sigmoid(0.8) = 0.690'), ('loss', '-log 0.690 = 0.371'),
             ('update', 'raise r_A, lower r_B')]
    for i, (t, v) in enumerate(steps):
        x = 20 + i * 185
        b += block(x, y, 165, 60, t, [v], 'box-on' if i == 2 else 'box', 't-note', 't-tick')
        if i:
            b.append(arrow(x - 20, y + 30, x, y + 30))
    b.append(text(380, 285, 'A labeler only says "A is better". The loss is small when the model already agrees, large when it does not.', 't-muted', 'middle'))
    return svg(760, 300, 'A prompt and two answers go into the reward model, which scores them 1.2 and 0.4. The gap 0.8 becomes a probability 0.690 through the sigmoid, the loss is minus log 0.690 = 0.371, and the update raises the chosen score and lowers the rejected one.', b)


F['ch2_pref_pair'] = pref_pair()


# ---------------------------------------------------------------- 6. the toy reward model learns the hidden weights
def bt_weights():
    feats = ['correct', 'detail', 'polite', 'length']
    tw, lw = BT['true_w'], BT['learned_w']
    W, L, T = 760, 150, 70
    Xz = 420
    sc = 75
    b = [text(20, 24, 'The reward model recovers the labeler\'s hidden taste from choices alone', 't-title'),
         dot(30, 46, 6, 's1'), text(42, 50, 'true weight (hidden)', 't-tick'), dot(210, 46, 6, 's2'),
         text(222, 50, f'learned from {len(BT["curve"]) and 994} comparisons', 't-tick')]
    b.append(line(Xz, T - 6, Xz, T + 4 * 54, 'axis'))
    for i, (f, t, l) in enumerate(zip(feats, tw, lw)):
        y = T + i * 54
        b.append(text(L, y + 24, f, 't-note', 'end'))
        for j, (v, c) in enumerate([(t, 's1'), (l, 's2')]):
            x0 = Xz if v >= 0 else Xz + v * sc
            b.append(rect(x0, y + 4 + j * 20, abs(v) * sc, 16, c, 3, f'{f}: {v:+.2f}'))
            tx = Xz + v * sc + (8 if v >= 0 else -8)
            b.append(text(tx, y + 17 + j * 20, f'{v:+.2f}', 't-val', 'start' if v >= 0 else 'end'))
    c = BT['curve']
    b.append(text(20, T + 4 * 54 + 22, f'loss {c[0]["loss"]:.3f} (= ln 2, a coin flip) to {c[-1]["loss"]:.3f}; test accuracy {c[-1]["test_acc"]:.3f} '
                  f'against a ceiling of {BT["ceiling"]:.3f}; correlation with the true reward {BT["corr"]:.4f}', 't-muted'))
    return svg(W, T + 4 * 54 + 38, 'Paired bars for four features. True weights 3.0, 1.5, 1.0, -0.5; learned weights 2.93, 1.53, 0.95, -0.23. The model recovers the ranking of what the labeler cares about.', b)


F['ch2_bt_weights'] = bt_weights()


# ---------------------------------------------------------------- 7. the RLHF loop
def rlhf_loop():
    b = [text(20, 24, 'The RLHF loop (Christiano 2017 to InstructGPT 2022)', 't-title')]
    b += block(20, 60, 150, 70, 'prompt x', ['from a dataset'], 'box')
    b += block(220, 60, 150, 70, 'policy π', ['the model being trained'], 'box-on')
    b += block(420, 60, 150, 70, 'answer y', ['sampled'], 'box')
    b += block(595, 40, 130, 54, 'reward model', ['r(x, y)'], 'box-2', 't-note', 't-tick')
    b += block(595, 110, 130, 54, 'frozen copy π_ref', ['log π / π_ref'], 'box-1', 't-note', 't-tick')
    b += [arrow(170, 95, 220, 95), arrow(370, 95, 420, 95), arrow(570, 85, 595, 67), arrow(570, 105, 595, 137)]
    b += [box(420, 200, 320, 56, 'box-3', 10), text(580, 222, 'penalised reward', 't-note', 'middle'),
          text(580, 242, 'R = r(x, y) - β · log(π(y|x) / π_ref(y|x))', 't-tick', 'middle')]
    b += [hpath([(725, 67), (750, 67), (750, 228), (742, 228)]), arrow(660, 164, 660, 200)]
    b += [box(140, 200, 230, 56, 'box', 10), text(255, 222, 'PPO update', 't-note', 'middle'),
          text(255, 242, 'make high-R answers more likely', 't-tick', 'middle')]
    b += [arrow(420, 228, 370, 228), hpath([(255, 200), (255, 165), (295, 165), (295, 130)], 'path', 'ah-on')]
    b += [box(20, 290, 720, 50, 'box-ghost', 10),
          text(380, 311, 'every so often: show new answers to people, collect fresh comparisons, retrain the reward model', 't-tick', 'middle'),
          text(380, 329, '(Christiano 2017 did this continuously; Anthropic\'s HH-RLHF in 2022 did it weekly)', 't-muted', 'middle')]
    return svg(760, 352, 'The RLHF loop: a prompt goes to the policy, which samples an answer; the reward model scores it and a frozen reference copy gives the KL term; the penalised reward drives a PPO update of the policy; from time to time new human comparisons retrain the reward model.', b)


F['ch2_rlhf_loop'] = rlhf_loop()


# ---------------------------------------------------------------- 8. InstructGPT's three steps
def instructgpt_steps():
    b = [text(20, 24, 'InstructGPT (Ouyang et al., 2022): three steps', 't-title')]
    cols = [('box-3', 'Step 1: SFT', ['labelers write good answers', 'to real prompts', 'fine-tune GPT-3 on them'], 'about 13k prompts'),
            ('box-2', 'Step 2: reward model', ['labelers rank 4 to 9 answers', 'per prompt', 'fit r with Bradley-Terry'], 'about 33k prompts'),
            ('box-on', 'Step 3: PPO', ['policy writes answers', 'reward model scores them', 'maximise r - β·KL'], 'about 31k prompts')]
    for i, (c, t, ls, n) in enumerate(cols):
        x = 20 + i * 250
        b += [box(x, 50, 220, 150, c, 12), text(x + 110, 76, t, 't-note', 'middle')]
        b += para(x + 110, 104, ls, 't-tick', 'middle', 20)
        b.append(text(x + 110, 186, n, 't-val', 'middle'))
        if i:
            b.append(arrow(x - 30, 125, x, 125))
    b += [text(130, 222, 'demonstrations', 't-muted', 'middle'), text(380, 222, 'comparisons', 't-muted', 'middle'),
          text(630, 222, 'no new human labels', 't-muted', 'middle')]
    b.append(text(380, 252, 'Result: the 1.3B InstructGPT model was preferred to the 175B GPT-3 by the labelers.', 't-note', 'middle'))
    return svg(760, 266, 'InstructGPT in three columns: step 1, supervised fine-tuning on about 13 thousand prompts with labeler-written answers; step 2, a reward model trained on rankings for about 33 thousand prompts; step 3, PPO against the reward model on about 31 thousand prompts.', b)


F['ch2_instructgpt_steps'] = instructgpt_steps()


# ---------------------------------------------------------------- 9. the KL trade-off in the toy (proxy reward vs true quality)
def kl_tradeoff():
    rows = KL['toy']['rows']
    betas = [r['beta'] for r in rows]
    W, L, T, pw, ph = 760, 70, 50, 470, 220
    xs = list(range(len(rows)))
    X = lambda i: L + i / (len(rows) - 1) * pw
    Y = lambda v: T + ph - (v + 3.2) / 6 * ph
    b = [text(20, 24, 'Less KL penalty (smaller β): the reward model is pleased, the person is not', 't-title')]
    for v in [-3, -2, -1, 0, 1, 2]:
        b += [line(L, Y(v), L + pw, Y(v), 'grid'), text(L - 8, Y(v) + 4, f'{v:+d}' if v else '0', 't-tick', 'end')]
    for i, bt in enumerate(betas):
        b.append(text(X(i), T + ph + 18, f'β = {bt}', 't-tick', 'middle'))
    b.append(text(L + pw / 2, T + ph + 38, 'KL coefficient, from strong penalty (left) to weak penalty (right)', 't-tick', 'middle'))
    for key, lc, dc, name in [('reward', 'l2', 's2', 'reward model score'), ('true', 'l1', 's1', 'true quality')]:
        pts = ' '.join(f'{X(i):.1f},{Y(r[key]):.1f}' for i, r in enumerate(rows))
        b.append(f'<polyline class="{lc}" points="{pts}"/>')
        for i, r in enumerate(rows):
            b.append(f'<g class="mark"><title>β {r["beta"]}: {name} {r[key]:+.3f}, KL {r["kl"]:.3f}</title>{dot(X(i), Y(r[key]), 5, dc + " ring")}</g>')
    b += [dot(L + pw + 30, Y(2.5), 6, 's2'), text(L + pw + 42, Y(2.5) + 4, 'reward model score: +2.50', 't-note'),
          dot(L + pw + 30, Y(-3.0), 6, 's1'), text(L + pw + 42, Y(-3.0) + 4, 'true quality: -3.00', 't-note')]
    b += para(L + pw + 30, Y(0.9), ['at β = 0.5 both agree:', f'reward {rows[3]["reward"]:+.3f}, true {rows[3]["true"]:+.3f}',
                                    f'KL only {rows[3]["kl"]:.3f} nats'], 't-muted')
    return svg(W, T + ph + 52, 'Two lines over seven values of beta. With a strong penalty both the reward model score and the true quality rise together; at beta 0.1 and below the policy jumps to the over-rated answer D: the reward model score climbs to 2.5 while the true quality falls to -3.', b)


F['ch2_kl_tradeoff'] = kl_tradeoff()


# ---------------------------------------------------------------- 10. per-token log-ratio of a real answer
def kl_tokens():
    s = KL['real']['rows'][0]
    toks, vals = s['tokens'][:10], s['per_token'][:10]
    W, x0, y0 = 760, 30, 70
    cw = 70
    mid = 170
    sc = 34
    b = [text(20, 24, 'Where the instruct model differs from its base model: log π - log π_ref, token by token', 't-title'),
         text(20, 46, f'first 10 tokens of a sampled answer to "{KL["real"]["prompt"]}"', 't-muted')]
    b.append(line(x0, mid, x0 + cw * len(toks) + 10, mid, 'axis'))
    for i, (t, v) in enumerate(zip(toks, vals)):
        x = x0 + 10 + i * cw
        h = abs(v) * sc
        b.append(rect(x + 6, mid - h if v > 0 else mid, cw - 14, h, 's2' if v > 0 else 's3', 3, f'{t!r}: {v:+.3f}'))
        b.append(text(x + cw / 2 - 1, (mid - h - 6) if v > 0 else (mid + h + 14), f'{v:+.2f}', 't-tick', 'middle'))
        b.append(text(x + cw / 2 - 1, 246, t.strip() or '·', 't-note', 'middle'))
    b.append(text(20, 280, f'Summed over all {s["n"]} tokens: {s["logratio"]:.2f} nats. Averaged over 8 samples: {KL["real"]["mean_logratio"]:.2f} nats. '
                  f'With β = 0.02: a penalty of {0.02 * KL["real"]["mean_logratio"]:.3f}.', 't-muted'))
    return svg(W, 294, 'Bars for the first ten tokens of an answer, showing the log-probability under the instruct model minus that under the base model. Most tokens are near zero; the opening "The" and "appears" are much more likely under the instruct model.', b)


F['ch2_kl_tokens'] = kl_tokens()


# ---------------------------------------------------------------- 11. PPO vs DPO side by side
def ppo_vs_dpo():
    b = [text(190, 24, 'RLHF with PPO', 't-title', 'middle'), text(570, 24, 'DPO', 't-title', 'middle'), line(380, 12, 380, 330, 'edge-dim')]
    # PPO: 4 models
    ms = [('policy π', 'trained', 'box-on'), ('reference π_ref', 'frozen', 'box-1'), ('reward model r', 'frozen, trained first', 'box-2'),
          ('value model V', 'trained', 'box-on')]
    for i, (t, s, c) in enumerate(ms):
        x, y = 20 + (i % 2) * 180, 46 + (i // 2) * 76
        b += block(x, y, 160, 60, t, [s], c, 't-note', 't-tick')
    b += para(190, 214, ['1. sample answers from π (slow: generation)', '2. score them with r, add the KL term',
                         '3. estimate advantages with V', '4. clipped PPO update of π and V', 'repeat; many settings to tune'], 't-tick', 'middle', 20)
    # DPO: 2 models
    b += block(420, 46, 150, 60, 'policy π', ['trained'], 'box-on', 't-note', 't-tick')
    b += block(590, 46, 150, 60, 'reference π_ref', ['frozen'], 'box-1', 't-note', 't-tick')
    b += [box(420, 122, 320, 60, 'box-ghost', 10), text(580, 146, 'no reward model, no value model,', 't-tick', 'middle'),
          text(580, 164, 'no sampling during training', 't-tick', 'middle')]
    b += para(580, 214, ['1. take a fixed pair (x, chosen, rejected)', '2. four log-probabilities: π and π_ref', '   on chosen and on rejected',
                         '3. loss = -log σ(β · margin)', 'an ordinary supervised training loop'], 't-tick', 'middle', 20)
    return svg(760, 340, 'Left, RLHF with PPO needs four models (policy, reference, reward model, value model) and sampling at every step. Right, DPO needs only the policy and a frozen reference, and trains on fixed preference pairs with a classification-style loss.', b)


F['ch2_ppo_vs_dpo'] = ppo_vs_dpo()


# ---------------------------------------------------------------- 12. the DPO loss on one pair, after training (real numbers)
def dpo_numbers():
    a = DPO['A']['after']
    beta = DPO['beta']
    cw, cl = a['lw'] - a['rw'], a['ll'] - a['rl']
    b = [text(20, 24, 'The DPO loss on one pair, after 40 training steps (numbers from ch2_dpo.py)', 't-title')]
    hdr = ['', 'log π (policy)', 'log π_ref (frozen)', 'difference', 'β · difference']
    xs = [20, 230, 380, 520, 630]
    for x, h in zip(xs, hdr):
        b.append(text(x, 58, h, 't-label'))
    rows = [('chosen  "In short: Tokyo. ..."', a['lw'], a['rw'], cw), ('rejected  "Tokyo has been ..."', a['ll'], a['rl'], cl)]
    for i, (n, p, r, d) in enumerate(rows):
        y = 70 + i * 40
        b += [box(14, y, 730, 32, 'box-3' if i == 0 else 'box', 6), text(26, y + 21, n, 't-tick')]
        b += [text(230, y + 21, f'{p:.2f}', 't-val'), text(380, y + 21, f'{r:.2f}', 't-val'), text(520, y + 21, f'{d:+.2f}', 't-val'),
              text(630, y + 21, f'{beta * d:+.3f}', 't-val')]
    m = beta * (cw - cl)
    y = 170
    steps = [('margin', f'{beta * cw:+.3f} - ({beta * cl:+.3f}) = {m:.3f}'), ('sigmoid', f'σ({m:.3f}) = {1 / (1 + math.exp(-m)):.4f}'),
             ('loss', f'-log σ = {math.log1p(math.exp(-m)):.4f}')]
    for i, (t, v) in enumerate(steps):
        x = 20 + i * 250
        b += block(x, y, 220, 60, t, [v], 'box-on' if i == 2 else 'box', 't-note', 't-tick')
        if i:
            b.append(arrow(x - 30, y + 30, x, y + 30))
    b.append(text(380, 262, 'Before training the two models were identical: every difference was 0, the margin 0, the loss ln 2 = 0.693.', 't-muted', 'middle'))
    return svg(760, 276, 'A table of four log-probabilities for the chosen and rejected answers under the trained policy and the frozen reference, their differences, the implicit rewards, then the margin, its sigmoid and the loss.', b)


F['ch2_dpo_numbers'] = dpo_numbers()


# ---------------------------------------------------------------- 13. the tiny DPO run
def dpo_run():
    h = DPO['hist']
    steps = [x['step'] for x in h]
    W, L, T, pw, ph = 760, 70, 50, 440, 200
    X = lambda s: L + s / steps[-1] * pw
    Y = lambda v: T + ph - (v + 95) / 75 * ph
    b = [text(20, 24, 'A tiny DPO run: log-probabilities of the chosen and rejected answer (pair 0)', 't-title')]
    for v in [-90, -75, -60, -45, -30]:
        b += [line(L, Y(v), L + pw, Y(v), 'grid'), text(L - 8, Y(v) + 4, f'{v}', 't-tick', 'end')]
    for s in steps:
        b.append(text(X(s), T + ph + 18, str(s), 't-tick', 'middle'))
    b.append(text(L + pw / 2, T + ph + 38, 'training step (batch of 4 pairs, learning rate 1e-6)', 't-tick', 'middle'))
    for key, lc, dc, name in [('lw', 'l1', 's1', 'chosen'), ('ll', 'l3', 's3', 'rejected')]:
        pts = ' '.join(f'{X(x["step"]):.1f},{Y(x[key]):.1f}' for x in h)
        b.append(f'<polyline class="{lc}" points="{pts}"/>')
        for x in h:
            b.append(f'<g class="mark"><title>step {x["step"]}: {name} {x[key]:.2f}</title>{dot(X(x["step"]), Y(x[key]), 4, dc + " ring")}</g>')
        b += [dot(L + pw + 24, Y(h[-1][key]), 6, dc), text(L + pw + 36, Y(h[-1][key]) + 4, f'{name}: {h[0][key]:.1f} to {h[-1][key]:.1f}', 't-note')]
    b += para(L + pw + 24, Y(-55), [f'train loss {h[0]["loss"]:.3f} to {h[-1]["loss"]:.4f}', f'held-out answers starting', f'"In short:" {DPO["n_before"]}/8 to {DPO["n_after"]}/8'], 't-muted')
    return svg(W, T + ph + 52, 'Two lines over 40 training steps: the log-probability of the chosen answer rises from -57.3 to -30.0, and that of the rejected answer falls from -40.7 to -89.9.', b)


F['ch2_dpo_run'] = dpo_run()


# ---------------------------------------------------------------- 14. the GRPO group
def grpo_group():
    g = next((x for x in GRPO['groups'] if 0 < sum(s['reward'] for s in x['samples']) < len(x['samples'])), GRPO['groups'][0])
    S = g['samples']
    r = [s['reward'] for s in S]
    mean = sum(r) / len(r)
    std = (sum((v - mean) ** 2 for v in r) / len(r)) ** 0.5
    b = [text(20, 24, 'GRPO: score a group of answers to the same question, compare each to the group', 't-title')]
    b += [box(20, 40, 720, 34, 'box', 8), text(32, 62, g['q'], 't-tick'), text(728, 62, f'key: {g["gold"]}', 't-val', 'end')]
    for i, s in enumerate(S):
        y = 96 + i * 46
        ok = s['reward'] > 0
        b += [box(20, y, 250, 36, 'box-3' if ok else 'box', 8), text(32, y + 23, f'o{i + 1}: final answer {s["pred"]:g}' if s['pred'] is not None else f'o{i + 1}: no answer found', 't-tick')]
        b += [arrow(270, y + 18, 320, y + 18), box(320, y, 110, 36, 'box-2', 8), text(375, y + 23, f'reward {s["reward"]:.0f}', 't-tick', 'middle')]
        b += [arrow(430, y + 18, 560, y + 18), box(560, y, 180, 36, 'box-on' if s['adv'] > 0 else 'box', 8),
              text(650, y + 23, f'advantage {s["adv"]:+.3f}', 't-val', 'middle')]
    b += [box(440, 96, 110, 176, 'box-ghost', 10), text(495, 150, 'group', 't-note', 'middle'), text(495, 172, f'mean {mean:.2f}', 't-tick', 'middle'),
          text(495, 192, f'std {std:.2f}', 't-tick', 'middle'), text(495, 218, '(r - mean)', 't-muted', 'middle'), text(495, 236, '/ std', 't-muted', 'middle')]
    b.append(text(380, 300, 'Right answers get a positive advantage and become more likely; wrong ones get a negative one. No value model.', 't-muted', 'middle'))
    return svg(760, 314, f'A question and four sampled answers from Qwen2.5-0.5B-Instruct, each checked against the correct answer: rewards {r}, group mean {mean:.2f}, standard deviation {std:.2f}, giving the advantages shown.', b)


F['ch2_grpo_group'] = grpo_group()


# ---------------------------------------------------------------- 15. reward from a judge vs reward from a checker
def judge_vs_checker():
    b = [text(190, 24, 'Learned reward (RLHF)', 't-title', 'middle'), text(570, 24, 'Verifiable reward (RLVR, R1)', 't-title', 'middle'),
         line(380, 12, 380, 250, 'edge-dim')]
    b += block(20, 46, 340, 56, 'reward model (a neural network)', ['trained on human comparisons'], 'box-2', 't-note', 't-tick')
    b += para(190, 130, ['+ works for any task: tone, helpfulness, safety', '- it is an imperfect copy of people:', '  push too hard and it can be fooled',
                         '- needs a KL leash and fresh labels'], 't-tick', 'middle', 20)
    b += block(400, 46, 340, 56, 'a program that checks the answer', ['final number equals the key? tests pass?'], 'box-4', 't-note', 't-tick')
    b += para(570, 130, ['+ cannot be flattered: right is right', '+ cheap: no labelers, no reward network', '- only for tasks with a checkable answer',
                         '  (maths, code, some formats)'], 't-tick', 'middle', 20)
    return svg(760, 236, 'Left, a learned reward model trained on human comparisons works for any task but can be over-optimised. Right, a verifiable reward from a checking program cannot be fooled in the same way but only exists for tasks with checkable answers.', b)


F['ch2_judge_vs_checker'] = judge_vs_checker()


# ---------------------------------------------------------------- 16. RLHF vs RLAIF: who writes the comparison
def rlaif():
    b = [text(20, 24, 'Constitutional AI (Bai et al., 2022): the same pipeline, a different labeler', 't-title')]
    for i, (lab, who, c) in enumerate([('RLHF', 'a person compares A and B', 'box-3'), ('RLAIF', 'a model compares A and B, using principles', 'box-4')]):
        y = 50 + i * 80
        b += [text(20, y + 34, lab, 't-note')]
        b += block(80, y, 130, 56, 'two answers', ['A, B'], 'box', 't-note', 't-tick')
        b += [arrow(210, y + 28, 240, y + 28), box(240, y, 290, 56, c, 10), text(385, y + 33, who, 't-tick', 'middle')]
        b += [arrow(530, y + 28, 560, y + 28)]
        b += block(560, y, 180, 56, 'preference model', ['then RL as before'], 'box-2', 't-note', 't-tick')
    b.append(text(380, 226, 'In the paper, harmlessness labels came from the model; helpfulness labels still came from people.', 't-muted', 'middle'))
    return svg(760, 240, 'Two rows. In RLHF a person compares two answers; in RLAIF a model compares them following a written list of principles. Both comparisons train a preference model, followed by RL.', b)


F['ch2_rlaif'] = rlaif()


# ---------------------------------------------------------------- 17. GRPO step: what one gradient step did
def grpo_step():
    rows = GRPO['step']
    mixed = [r for r in rows if r['adv'] != 0]
    zero = [r['after'] - r['before'] for r in rows if r['adv'] == 0]
    items = [(f'o{r["i"]}: reward {r["reward"]:.0f}, advantage {r["adv"]:+.3f}', r['after'] - r['before'], 's3' if r['adv'] > 0 else 's1') for r in mixed]
    items.append((f'{len(zero)} answers with advantage 0 (average)', sum(zero) / len(zero), 's4'))
    W, T, mid, sc = 760, 76, 520, 800
    b = [text(20, 24, 'One GRPO-style gradient step: change in mean log-probability per token', 't-title'),
         text(20, 46, 'the books-and-pens group (3 right, 1 wrong); the other 3 groups were all right, so they had nothing to teach', 't-muted')]
    b.append(line(mid, T - 8, mid, T + 34 * len(items), 'axis'))
    for i, (lab, d, c) in enumerate(items):
        y = T + i * 34
        w = abs(d) * sc
        b.append(text(20, y + 17, lab, 't-tick'))
        b.append(rect(mid if d >= 0 else mid - w, y + 4, w, 18, c, 3, f'{d:+.4f}'))
        b.append(text(mid + w + 6 if d >= 0 else mid - w - 6, y + 18, f'{d:+.4f}', 't-val', 'start' if d >= 0 else 'end'))
    return svg(W, T + 34 * len(items) + 10, 'Bars showing how one gradient step changed the average log-probability per token of each answer: the three right answers rose by about 0.04 to 0.05, the wrong answer fell by 0.256, and the twelve zero-advantage answers barely moved.', b)


F['ch2_grpo_step'] = grpo_step()

json.dump(F, open('results/figs_ch2.json', 'w'))
print(f'{len(F)} figures:', ', '.join(F))
