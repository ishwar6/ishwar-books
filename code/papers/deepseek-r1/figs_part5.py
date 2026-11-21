"""SVG figures for Part 5 (the R1 pipeline). Writes results/figs_part5.json.
Reads results/r1_part5_lang.json (run r1_part5_lang.py) and results/r1_part5_reject.json (run r1_part5_reject.py)."""
import json, os
from figlib import svg, text, box, arrow
from samfig import line

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
F = {}

# ---------------------------------------------------------------- the four-stage pipeline, "you are here"
STAGES = [
    ('1  Cold start', ['SFT of V3-Base on', 'thousands of long', 'readable CoT examples'], 'no reward (SFT)'),
    ('2  Reasoning RL', ['GRPO, as for R1-Zero', 'maths, code, science,', 'logic prompts'], 'rules + language'),
    ('3  Rejection sampling', ['600k reasoning +', '200k non-reasoning', '= 800k; SFT of V3-Base'], 'filters, 2 epochs'),
    ('4  RL, all scenarios', ['reasoning: rules', 'general: reward', 'models'], 'helpful + harmless'),
]


def pipeline(on=None):
    W, H = 840, 300
    bw, bh, gap, x0, y0 = 180, 132, 30, 15, 92
    b = []
    xs = [x0 + i * (bw + gap) for i in range(4)]
    # the base model enters at stage 1, and again at stage 3
    for i in (0, 2):
        cx = xs[i] + bw / 2
        b += [box(cx - 62, 14, 124, 34, 'box-ghost', 8), text(cx, 36, 'DeepSeek-V3-Base', 't-tick', 'middle'),
              arrow(cx, 48, cx, y0 - 4)]
    b.append(text(xs[2] + bw / 2 + 70, 72, 'a fresh start', 't-muted', 'start'))
    for i, (title, lines, rew) in enumerate(STAGES):
        x = xs[i]
        is_on = on == i + 1
        b.append(box(x, y0, bw, bh, 'box-on' if is_on else 'box', 10))
        b.append(text(x + 12, y0 + 24, title, 't-on' if is_on else 't-title'))
        for k, l in enumerate(lines):
            b.append(text(x + 12, y0 + 48 + k * 18, l, 't-note'))
        b.append(line(x + 10, y0 + bh - 34, x + bw - 10, y0 + bh - 34, 'grid'))
        b.append(text(x + 12, y0 + bh - 12, rew, 't-tick'))
        if i < 3:
            dashed = i == 1
            b.append(arrow(x + bw + 2, y0 + bh / 2, x + bw + gap - 3, y0 + bh / 2, on=False) if not dashed else
                     f'<line class="edge-dim" x1="{x + bw + 2:.1f}" y1="{y0 + bh / 2:.1f}" x2="{x + bw + gap - 3:.1f}" '
                     f'y2="{y0 + bh / 2:.1f}" marker-end="url(#ah)"/>')
    b.append(text(xs[1] + bw + gap / 2, y0 + bh + 22, 'its answers, not its weights', 't-muted', 'middle'))
    cx = xs[3] + bw / 2
    b += [arrow(cx, y0 + bh + 2, cx, y0 + bh + 34), box(cx - 55, y0 + bh + 36, 110, 32, 'box-1', 8),
          text(cx, y0 + bh + 57, 'DeepSeek-R1', 't-s1', 'middle')]
    if on:
        b.append(text(xs[on - 1] + bw / 2, y0 - 8, 'you are here', 't-on', 'middle'))
    label = ('The four stages of DeepSeek-R1: cold-start SFT, reasoning RL, rejection sampling with SFT on 800k samples, '
             'and RL for all scenarios' + (f'; stage {on} is highlighted.' if on else '.'))
    return svg(W, H, label, b)


F['p5_pipeline'] = pipeline()
for k in range(1, 5):
    F[f'p5_pipeline_{k}'] = pipeline(k)

# ---------------------------------------------------------------- stage 3 restarts from V3-Base
def restart():
    W, H = 840, 250
    b = []
    names_top = ['V3-Base', 'cold-start SFT', 'R1 Dev-1', 'reasoning RL', 'R1 Dev-2']
    xt = [15, 175, 335, 495, 655]
    for i, (x, n) in enumerate(zip(xt, names_top)):
        cls = 'box' if i % 2 else 'box-ghost'
        if n == 'R1 Dev-2':
            cls = 'box-2'
        b += [box(x, 30, 140, 40, cls, 8), text(x + 70, 55, n, 't-tick', 'middle')]
        if i < 4:
            b.append(arrow(x + 142, 50, x + 172, 50))
    b.append(text(15, 18, 'weights that learn', 't-label'))
    # Dev-2 -> samples (down)
    b += [arrow(725, 72, 725, 112), box(600, 114, 225, 40, 'box-2', 8),
          text(712, 132, 'sample many answers, keep', 't-tick', 'middle'), text(712, 147, 'the correct, readable ones', 't-tick', 'middle')]
    b.append(text(835, 98, 'Dev-2 stops here', 't-muted', 'end'))
    # bottom lane
    names_bot = ['V3-Base', 'SFT, 2 epochs', 'R1 Dev-3', 'RL, all scenarios', 'DeepSeek-R1']
    yb = 190
    for i, (x, n) in enumerate(zip(xt, names_bot)):
        cls = 'box' if i % 2 else 'box-ghost'
        if n == 'DeepSeek-R1':
            cls = 'box-1'
        b += [box(x, yb, 140, 40, cls, 8), text(x + 70, yb + 25, n, 't-s1' if n == 'DeepSeek-R1' else 't-tick', 'middle')]
        if i < 4:
            b.append(arrow(x + 142, yb + 20, x + 172, yb + 20))
    b.append(text(15, yb - 12, 'a second start from the same base', 't-label'))
    # data flows to the SFT box of the bottom lane
    b += [f'<path class="edge-2" d="M600,134 H245 V{yb - 4}" marker-end="url(#ah-2)"/>',
          text(420, 128, '600k + 200k from V3 = 800k samples', 't-s2', 'middle')]
    return svg(W, H, 'Stage 3 does not continue from the RL checkpoint. The checkpoint only writes data; '
               'a fresh copy of DeepSeek-V3-Base is fine-tuned on that data.', b)


F['p5_restart'] = restart()

# ---------------------------------------------------------------- the language reward, word by word
L = json.load(open(os.path.join(RES, 'r1_part5_lang.json')))
words = L['split_B']
W = 840
b = [text(15, 22, 'English question: "What is 15% of 80?"  The thinking, split into words:', 't-note')]
x, y = 15, 40
for w, lang in words:
    ww = max(34, 11 * len(w) + 16)
    if x + ww > W - 15:
        x, y = 15, y + 44
    b.append(box(x, y, ww, 30, 'box-1' if lang == 'en' else 'box-2', 6))
    b.append(text(x + ww / 2, y + 20, w, 't-tick', 'middle'))
    x += ww + 8
n_en = sum(1 for _, l in words if l == 'en')
y += 60
b += [box(15, y - 14, 14, 14, 'box-1', 3), text(36, y - 2, f'English (the target): {n_en} words', 't-note'),
      box(300, y - 14, 14, 14, 'box-2', 3), text(321, y - 2, f'Chinese: {len(words) - n_en} words (one per character)', 't-note')]
b.append(text(15, y + 30, f'Reward_language = {n_en} / {len(words)} = {n_en / len(words):.3f}', 't-on'))
b.append(text(15, y + 52, 'Digits, %, +, = and . are not words, so they do not count either way.', 't-muted'))
F['p5_lang_words'] = svg(W, y + 66, 'The thinking of case B split into words: English words in one colour, Chinese '
                         f'characters in another; {n_en} of {len(words)} words are English.', b)

# ---------------------------------------------------------------- the funnel of our rejection sampling run
rp = os.path.join(RES, 'r1_part5_reject.json')
if os.path.exists(rp):
    R = json.load(open(rp))
    steps = R['funnel']
    short = ['sampled (10 problems x 8)', 'finished thinking', 'correct', 'no mixed language', 'no long paragraph',
             'no code block (kept)']
    W, rowh, lw = 840, 40, 220
    top = steps[0][1]
    b = []
    cx = lw + (W - lw - 20) / 2
    maxw = W - lw - 40
    for i, (name, n) in enumerate(steps):
        y = 14 + i * rowh
        w = max(4, maxw * n / top)
        cls = 's1' if i == len(steps) - 1 else ('s2' if i == 2 else 'box')
        b.append(text(lw - 10, y + 22, short[i], 't-tick', 'end'))
        b.append(f'<g class="mark"><title>{short[i]}: {n}</title><rect class="{cls}" x="{cx - w / 2:.1f}" y="{y + 4:.1f}" '
                 f'width="{w:.1f}" height="{rowh - 10}" rx="4"/></g>')
        b.append(text(cx, y + 23, str(n), 't-strong', 'middle'))
    F['p5_funnel'] = svg(W, 14 + len(steps) * rowh + 10,
                         'Our rejection-sampling funnel: ' + ', '.join(f'{s} {n}' for s, (_, n) in zip(short, steps)) + '.', b)

json.dump(F, open(os.path.join(RES, 'figs_part5.json'), 'w'))
print('figures:', ', '.join(F))
