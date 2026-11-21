"""Figures for Part 4. Paper numbers come from Table 2 and the text of section 2.2.4 (v1); our numbers come from
results/r1_part4.json (r1_part4_analyze.py). Writes results/figs_part4.json (key -> svg string).
Colours come only from CSS classes, so the figures follow the dark, dim and light themes."""
import json, os
from figlib import svg, text, box, arrow, esc, bar_path

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'r1_part4.json')))
F = {}
DASH = ' stroke-dasharray="6 5"'

# ---------- 1. you are here ----------
b = []
W = 760
steps = [('DeepSeek-V3-Base', 'pre-trained, no reasoning SFT', 'box'),
         ('RL with GRPO', 'rule rewards + template (Parts 2, 3)', 'box'),
         ('DeepSeek-R1-Zero', 'this part: what came out', 'box-on')]
x = 20
for i, (t1, t2, cls) in enumerate(steps):
    b.append(box(x, 40, 210, 70, cls))
    b.append(text(x + 105, 70, t1, 't-title', 'middle'))
    b.append(text(x + 105, 92, t2, 't-note', 'middle'))
    if i < 2:
        b.append(arrow(x + 212, 75, x + 252, 75, on=True))
    x += 255
b.append(box(530, 150, 210, 56, 'box-ghost'))
b.append(text(635, 174, 'DeepSeek-R1 (Part 5)', 't-note', 'middle'))
b.append(text(635, 193, 'fixes readability', 't-muted', 'middle'))
b.append(arrow(635, 112, 635, 148))
b.append(text(20, 24, 'You are here: the result of pure RL on the base model', 't-tick'))
F['p4_pipeline'] = svg(W, 220, 'Pipeline: DeepSeek-V3-Base goes through RL with GRPO, rule rewards and the template, and '
                       'becomes DeepSeek-R1-Zero, the subject of this part. Its readability problems lead to DeepSeek-R1 in Part 5.', b)

# ---------- 2. Table 2 as bars ----------
T2 = [('AIME pass@1', '%', [63.6, 74.4, 71.0]), ('AIME cons@64', '%', [80.0, 83.3, 86.7]),
      ('MATH-500 pass@1', '%', [90.0, 94.8, 95.9]), ('GPQA Diamond', '%', [60.0, 77.3, 73.3]),
      ('LiveCodeBench', '%', [53.8, 63.4, 50.0]), ('Codeforces rating', '', [1820, 1843, 1444])]
MODELS = ['o1-mini', 'o1-0912', 'R1-Zero']
CLS = ['s3', 's2', 's1']
b = []
pw, ph, gx, gy = 220, 150, 30, 86
for n, (name, unit, vals) in enumerate(T2):
    col, row = n % 3, n // 3
    x0, y0 = 20 + col * (pw + gx), 76 + row * (ph + gy)
    vmax = 2000 if unit == '' else 100
    b.append(text(x0, y0 - 26, name, 't-title'))
    b.append(f'<line class="axis" x1="{x0}" y1="{y0 + ph}" x2="{x0 + pw}" y2="{y0 + ph}"/>')
    bw = 52
    best = max(vals)
    for i, v in enumerate(vals):
        h = ph * v / vmax
        bx = x0 + 12 + i * (bw + 16)
        b.append(f'<g class="mark"><title>{MODELS[i]}, {esc(name)}: {v}</title>'
                 f'<path class="{CLS[i]}" d="{bar_path(bx, y0 + ph - h, bw, h)}"/></g>')
        lab = f'{v:g}' if unit == '' else f'{v:.1f}'
        b.append(text(bx + bw / 2, y0 + ph - h - 6, lab, 't-val t-strong' if v == best else 't-val', 'middle'))
        b.append(text(bx + bw / 2, y0 + ph + 16, MODELS[i], 't-tick', 'middle'))
b.append(text(20, 22, 'Table 2 redrawn: higher is better in every panel; the best value of each panel is bold', 't-tick'))
F['p4_table2'] = svg(20 + 3 * pw + 2 * gx + 10, 76 + 2 * ph + gy + 30,
                     'Six bar panels from Table 2. AIME pass@1: o1-mini 63.6, o1-0912 74.4, R1-Zero 71.0. AIME cons@64: 80.0, '
                     '83.3, 86.7. MATH-500: 90.0, 94.8, 95.9. GPQA Diamond: 60.0, 77.3, 73.3. LiveCodeBench: 53.8, 63.4, '
                     '50.0. Codeforces rating: 1820, 1843, 1444.', b)

# ---------- 3. sketch of Figure 2 from the stated numbers ----------
b = []
L, T, pw, ph = 70, 40, 470, 230
Y = lambda v: T + ph - v / 100 * ph
for t in [0, 20, 40, 60, 80, 100]:
    b += [f'<line class="grid" x1="{L}" y1="{Y(t):.1f}" x2="{L + pw}" y2="{Y(t):.1f}"/>', text(L - 8, Y(t) + 4, f'{t}%', 't-tick', 'end')]
b.append(f'<line class="axis" x1="{L}" y1="{T + ph}" x2="{L + pw}" y2="{T + ph}"/>')
b.append(f'<line class="base-line" x1="{L}" y1="{Y(74.4):.1f}" x2="{L + pw}" y2="{Y(74.4):.1f}"/>')
b.append(f'<line class="base-line" x1="{L}" y1="{Y(83.3):.1f}" x2="{L + pw}" y2="{Y(83.3):.1f}"/>')
b.append(f'<line class="l1"{DASH} x1="{L + 10}" y1="{Y(15.6):.1f}" x2="{L + pw - 10}" y2="{Y(71.0):.1f}"/>')
for v, xx, cls in [(15.6, L + 10, 's1'), (71.0, L + pw - 10, 's1'), (86.7, L + pw - 10, 's2')]:
    b.append(f'<circle class="{cls} ring" cx="{xx}" cy="{Y(v):.1f}" r="6"/>')
b.append(text(L + 20, Y(15.6) + 4, 'pass@1 at the start: 15.6%', 't-note'))
b.append(text(L + pw - 20, Y(71.0) + 30, 'pass@1 at the end: 71.0%', 't-note', 'end'))
b.append(text(L + pw - 20, Y(86.7) - 10, 'majority vote at the end: 86.7%', 't-note', 'end'))
b.append(text(L + pw + 6, Y(74.4) + 4, 'o1-0912 pass@1 74.4%', 't-tick'))
b.append(text(L + pw + 6, Y(83.3) + 4, 'o1-0912 cons@64 83.3%', 't-tick'))
b.append(text(L + pw / 2, T + ph + 24, 'RL training steps (start to end, thousands of steps)', 't-tick', 'middle'))
b.append(text(L - 60, T - 16, 'AIME 2024 accuracy', 't-tick'))
F['p4_fig2_sketch'] = svg(L + pw + 170, T + ph + 40,
                          'Sketch of Figure 2 from the numbers in the text: pass@1 on AIME 2024 rises from 15.6 percent to 71.0 '
                          'percent during RL; majority voting reaches 86.7 percent at the end; dashed reference lines at 74.4 and '
                          '83.3 percent for OpenAI o1-0912. The straight line only joins two points; the real curve is in the screenshot.', b)

# ---------- 4. sketch of Figure 3: hundreds to thousands ----------
b = []
L, T, pw, ph = 70, 40, 470, 200
b.append(f'<line class="axis" x1="{L}" y1="{T + ph}" x2="{L + pw}" y2="{T + ph}"/>')
b.append(f'<line class="axis" x1="{L}" y1="{T}" x2="{L}" y2="{T + ph}"/>')
pts = ' '.join(f'{L + 10 + i * (pw - 20) / 40:.1f},{T + ph - 14 - (ph - 40) * (i / 40) ** 1.25:.1f}' for i in range(41))
b.append(f'<polyline class="l1" points="{pts}"/>')
b.append(text(L + 18, T + ph - 60, 'hundreds of tokens', 't-note'))
b.append(text(L + pw - 14, T + 18, 'thousands of tokens', 't-note', 'end'))
b.append(text(L + pw / 2, T + ph + 24, 'RL training steps', 't-tick', 'middle'))
b.append(text(L - 60, T - 16, 'average length of a response', 't-tick'))
b.append(text(L + pw - 14, T + ph - 20, 'no length reward, no instruction to think longer', 't-muted', 'end'))
F['p4_fig3_sketch'] = svg(L + pw + 30, T + ph + 40,
                          'Sketch of the shape of Figure 3: the average response length grows during RL from hundreds of tokens '
                          'to thousands of tokens, with no length reward. The shape is drawn by hand; the real curve is in the screenshot.', b)

# ---------- 5. why a correctness reward favours more thinking: a group of 4 answers ----------
b = []
rows = [('answer 1', 300, 0), ('answer 2', 450, 0), ('answer 3', 1200, 1), ('answer 4', 1500, 1)]
# GRPO equation (3) for rewards 0, 0, 1, 1: mean 0.5, std 0.5, so the advantages are -1, -1, +1, +1
x0, y0 = 120, 50
for i, (name, n, r) in enumerate(rows):
    y = y0 + i * 44
    b.append(text(x0 - 10, y + 20, name, 't-tick', 'end'))
    w = n / 1500 * 300
    b.append(f'<g class="mark"><title>{name}: {n} tokens, reward {r}</title>'
             f'<path class="{"s1" if r else "s3"}" d="{bar_path(x0, y + 6, w, 22, horizontal=True)}"/></g>')
    b.append(text(x0 + w + 8, y + 22, f'{n} tokens', 't-val'))
    b.append(text(x0 + 400, y + 22, f'reward {r}', 't-note'))
    b.append(text(x0 + 500, y + 22, f'A = {"+1" if r else "-1"}', 't-on' if r else 't-bad'))
    b.append(text(x0 + 570, y + 22, 'made more likely' if r else 'made less likely', 't-note'))
b.append(text(x0, 30, 'One question, four sampled answers (made-up lengths)', 't-title'))
b.append(text(x0, y0 + 4 * 44 + 26, 'mean reward 0.5, standard deviation 0.5, so A = (r - 0.5) / 0.5 = +1 or -1 (equation 3)', 't-muted'))
F['p4_group'] = svg(x0 + 720, y0 + 4 * 44 + 44,
                    'A group of four sampled answers to one question: two short answers of 300 and 450 tokens are wrong and get '
                    'advantage minus 1; two long answers of 1,200 and 1,500 tokens are right and get advantage plus 1, so RL '
                    'makes the long, careful answers more likely.', b)

# ---------- 6. the aha equation, step by step ----------
b = []
st = [('1', 'square', 'a - √(a+x) = x²', 'needs x ≥ 0'),
      ('2', 'square again', 'a + x = (a - x²)²', 'needs a - x² ≥ 0'),
      ('3', 'a quartic', 'x⁴-2ax²-x+a²-a = 0', 'no x³ term: 4 roots sum to 0'),
      ('4', 'factor', '(x²+x+1-a)(x²-x-a) = 0', 'four candidates'),
      ('5', 'check each', 'in the first equation', 'only one survives'),
      ('6', 'answer', 'x = (√(4a-3) - 1)/2', 'the sum is this one root')]
for i, (n, t1, t2, t3) in enumerate(st):
    col, row = i % 3, i // 3
    x, y = 20 + col * 250, 40 + row * 120
    b.append(box(x, y, 230, 90, 'box-on' if n == '6' else ('box-2' if n == '3' else 'box')))
    b.append(text(x + 14, y + 24, f'{n}. {t1}', 't-title'))
    b.append(text(x + 14, y + 50, t2, 't-mono'))
    b.append(text(x + 14, y + 74, t3, 't-note'))
    if col < 2:
        b.append(arrow(x + 232, y + 45, x + 248, y + 45))
b.append(arrow(20 + 2 * 250 + 115, 40 + 92, 20 + 115, 40 + 118))
F['p4_aha_steps'] = svg(780, 290, 'Six steps to solve the square root equation: square, square again, get a quartic whose four roots '
                        'sum to zero, factor it into two quadratics, check the four candidates in the first equation, and keep '
                        'the one root x equals square root of 4a minus 3, minus 1, over 2.', b)

# ---------- 7. our length histogram, correct vs wrong ----------
def hist_fig(tag, title):
    d = R[tag]
    edges, hc, hw = d['hist_edges'], d['hist_correct'], d['hist_wrong']
    b = []
    L, T, pw, ph = 60, 46, 560, 200
    vmax = max(max(hc), max(hw), 1)
    vmax = ((vmax + 9) // 10) * 10
    Y = lambda v: T + ph - v / vmax * ph
    for t in range(0, vmax + 1, max(10, vmax // 5 // 10 * 10)):
        b += [f'<line class="grid" x1="{L}" y1="{Y(t):.1f}" x2="{L + pw}" y2="{Y(t):.1f}"/>', text(L - 8, Y(t) + 4, str(t), 't-tick', 'end')]
    nb = len(hc)
    cw = pw / nb
    for i in range(nb):
        for j, (v, cls, nm) in enumerate([(hc[i], 's1', 'correct'), (hw[i], 's3', 'wrong')]):
            bx = L + i * cw + 6 + j * (cw - 12) / 2
            bwid = (cw - 12) / 2 - 2
            h = ph * v / vmax
            b.append(f'<g class="mark"><title>{edges[i]} to {edges[i + 1]} tokens, {nm}: {v}</title>'
                     f'<path class="{cls}" d="{bar_path(bx, T + ph - h, bwid, max(h, 0.01))}"/></g>')
            if v:
                b.append(text(bx + bwid / 2, T + ph - h - 4, str(v), 't-val', 'middle'))
    for i, e in enumerate(edges):
        b.append(text(L + i * cw, T + ph + 16, f'{e:,}', 't-tick', 'middle'))
    b.append(f'<line class="axis" x1="{L}" y1="{T + ph}" x2="{L + pw}" y2="{T + ph}"/>')
    b.append(text(L + pw / 2, T + ph + 36, 'answer length in new tokens (the last bin ends at our 2,048 limit)', 't-tick', 'middle'))
    b.append(text(L - 40, T - 26, title, 't-title'))
    b.append(f'<rect class="s1" x="{L + pw - 190}" y="{T - 22}" width="10" height="10" rx="2"/>')
    b.append(text(L + pw - 174, T - 13, f'correct ({sum(hc)})', 't-note'))
    b.append(f'<rect class="s3" x="{L + pw - 90}" y="{T - 22}" width="10" height="10" rx="2"/>')
    b.append(text(L + pw - 74, T - 13, f'wrong ({sum(hw)})', 't-note'))
    return svg(L + pw + 20, T + ph + 46, f'Histogram of answer lengths for {title}: correct answers {hc}, wrong answers {hw}, '
               f'in bins of 256 tokens from 0 to 2,048.', b)


F['p4_len_distill'] = hist_fig('distill', 'DeepSeek-R1-Distill-Qwen-1.5B, 160 answers')
F['p4_len_base'] = hist_fig('base', 'Qwen2.5-Math-1.5B with the R1-Zero template, 160 answers')

# ---------- 8. reflection words, base vs distill ----------
b = []
words = ['wait', 'let me check', 'alternatively', 're-evaluate']
L, T, rowh = 130, 50, 56
vmax = max(max(R[t]['reflect_per_answer'][w] for w in words) for t in ['base', 'distill'])
bw = 470
for i, w in enumerate(words):
    y = T + i * rowh
    b.append(text(L - 10, y + 24, f'"{w}"', 't-tick', 'end'))
    for j, (t, cls, nm) in enumerate([('base', 's3', 'base'), ('distill', 's1', 'distill')]):
        v = R[t]['reflect_per_answer'][w]
        ww = max(2, bw * v / vmax)
        b.append(f'<g class="mark"><title>{nm}, {w}: {v:.2f} per answer</title>'
                 f'<path class="{cls}" d="{bar_path(L, y + 4 + j * 22, ww, 18, horizontal=True)}"/></g>')
        b.append(text(L + ww + 6, y + 18 + j * 22, f'{v:.2f}', 't-val'))
b.append(f'<rect class="s3" x="{L}" y="{T - 30}" width="10" height="10" rx="2"/>')
b.append(text(L + 16, T - 21, 'Qwen2.5-Math-1.5B (base)', 't-note'))
b.append(f'<rect class="s1" x="{L + 220}" y="{T - 30}" width="10" height="10" rx="2"/>')
b.append(text(L + 236, T - 21, 'R1-Distill-Qwen-1.5B', 't-note'))
b.append(text(L, T + 4 * rowh + 14, 'mean count per answer, same 20 problems x 8 samples for each model', 't-muted'))
F['p4_reflect'] = svg(L + bw + 80, T + 4 * rowh + 26,
                      'Reflection phrases per answer. ' + '; '.join(
                          f'{w}: base {R["base"]["reflect_per_answer"][w]:.2f}, distill {R["distill"]["reflect_per_answer"][w]:.2f}'
                          for w in words) + '.', b)

json.dump(F, open(os.path.join(HERE, 'results', 'figs_part4.json'), 'w'), indent=1)
print('wrote results/figs_part4.json:', ', '.join(F))
