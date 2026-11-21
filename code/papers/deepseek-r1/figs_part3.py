"""SVG figures for Part 3, drawn from results/part3.json and results/part3_run.json (run r1_part3.py and
r1_part3_run.py first). Writes results/figs_part3.json. CSS classes only, never colours."""
import json, os
from figlib import svg, text, box, arrow
from samfig import token, line

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'part3_run.json')))
F = {}

# ---------------------------------------------------------------- 1. you are here: the RL loop of R1-Zero
b = []
steps = [('question', 'from the data'), ('Table 1 template', 'adds the tags'), ('base model', 'samples G outputs'),
         ('rule rewards', 'format + accuracy'), ('GRPO update', 'Part 2')]
w, h, gap, y = 118, 54, 26, 46
for i, (a, s) in enumerate(steps):
    x = 20 + i * (w + gap)
    on = a in ('rule rewards', 'Table 1 template')
    b.append(box(x, y, w, h, 'box-on' if on else 'box-1', 8))
    b.append(text(x + w / 2, y + 22, a, 't-note', 'middle'))
    b.append(text(x + w / 2, y + 40, s, 't-tick', 'middle'))
    if i:
        b.append(arrow(x - gap + 2, y + h / 2, x - 3, y + h / 2))
x_last = 20 + 4 * (w + gap) + w / 2
x_model = 20 + 2 * (w + gap) + w / 2
b.append(f'<path class="edge" d="M{x_last:.1f},{y + h} V{y + h + 34} H{x_model:.1f} V{y + h + 4}" fill="none" marker-end="url(#ah)"/>')
b.append(text((x_last + x_model) / 2, y + h + 28, 'new weights, next step', 't-tick', 'middle'))
b.append(text(20, 24, 'You are here: the two highlighted boxes are this part', 't-title'))
F['p3_here'] = svg(20 + 5 * w + 4 * gap + 20, 180, 'The training loop of DeepSeek-R1-Zero: a question goes into the Table 1 '
                   'template, the base model samples a group of outputs, rule-based rewards score them, and GRPO updates '
                   'the model. The template and the rewards are highlighted.', b)

# ---------------------------------------------------------------- 2. the reward pipeline for one output
b = []
y0 = 40
cols = [('model output', ['<think> ... </think>', '<answer> \\boxed{18} </answer>']),
        ('1. format check', ['tags in order?', 'nothing outside?']),
        ('2. extract answer', ['last \\boxed{...}', 'count the braces']),
        ('3. compare', ['normalise, then', 'equal to gold 18?']),
        ('4. reward', ['format + accuracy', '= 1 + 1 = 2'])]
CW, ch, cg = [244, 150, 150, 150, 150], 70, 22
X = [20 + sum(CW[:i]) + i * cg for i in range(len(CW))]
for i, (title, lines_) in enumerate(cols):
    x, cw = X[i], CW[i]
    b.append(box(x, y0, cw, ch, 'box-on' if i == 4 else ('box-2' if i == 0 else 'box-1'), 8))
    b.append(text(x + cw / 2, y0 + 20, title, 't-note', 'middle'))
    for k, l in enumerate(lines_):
        b.append(text(x + cw / 2, y0 + 40 + 16 * k, l, 't-mono' if i in (0, 2) else 't-tick', 'middle'))
    if i:
        b.append(arrow(x - cg + 2, y0 + ch / 2, x - 3, y0 + ch / 2, on=True))
# the failure branches
fails = [(1, 'format reward 0'), (2, 'no box: accuracy 0'), (3, 'not equal: accuracy 0')]
for i, s_ in fails:
    x = X[i] + CW[i] / 2
    b.append(line(x, y0 + ch, x, y0 + ch + 30, 'edge', 'ah'))
    b.append(box(x - 78, y0 + ch + 32, 156, 28, 'box-ghost', 6))
    b.append(text(x, y0 + ch + 51, s_, 't-tick', 'middle'))
b.append(text(20, 22, 'One output, four steps, one number', 't-title'))
F['p3_pipeline'] = svg(X[-1] + CW[-1] + 20, y0 + ch + 80, 'The reward pipeline for one output: check the format, '
                       'extract the last boxed answer, normalise and compare it with the gold answer, then add the format '
                       'reward and the accuracy reward. Each check that fails gives zero for its part.', b)

# ---------------------------------------------------------------- 3. what a raw base model already does (our run)
T = R['totals']
metrics = [('think', 'has <think> and </think>'), ('answer_tag', 'has <answer> and </answer>'),
           ('fmt', 'strict format (reward 1)'), ('acc', 'right answer'), ('both', 'right answer and format')]
models = list(T)
names = {'Qwen/Qwen2.5-Math-1.5B': 'Qwen2.5-Math-1.5B', 'Qwen/Qwen2.5-0.5B': 'Qwen2.5-0.5B'}
b = []
L, top, rowh, barh, pw = 210, 52, 46, 16, 330
for j, mid in enumerate(models):
    b.append(box(L + j * 190, 14, 12, 12, f's{j + 1}', 2))
    b.append(text(L + j * 190 + 18, 24, names[mid], 't-tick'))
for i, (k, lab) in enumerate(metrics):
    y = top + i * rowh
    b.append(text(L - 10, y + barh + 4, lab, 't-tick', 'end'))
    for j, mid in enumerate(models):
        v = T[mid][k]
        ww = max(2, pw * v / 32)
        yy = y + j * (barh + 2)
        b.append(f'<g class="mark"><title>{names[mid]}, {lab}: {v} of 32</title>'
                 f'<rect class="s{j + 1}" x="{L}" y="{yy:.1f}" width="{ww:.1f}" height="{barh}" rx="3"/></g>')
        b.append(text(L + ww + 6, yy + barh - 3, f'{v}/32', 't-val'))
for t in (0, 8, 16, 24, 32):
    xx = L + pw * t / 32
    b.append(f'<line class="grid" x1="{xx:.1f}" y1="{top - 6}" x2="{xx:.1f}" y2="{top + len(metrics) * rowh - 6}"/>')
    b.append(text(xx, top + len(metrics) * rowh + 10, str(t), 't-tick', 'middle'))
b.append(text(L + pw / 2, top + len(metrics) * rowh + 30, 'answers out of 32 (4 questions x 8 samples)', 't-tick', 'middle'))
F['p3_counts'] = svg(L + pw + 70, top + len(metrics) * rowh + 44, 'Bar chart of 32 sampled answers per base model in the '
                     'Table 1 template: how many have think tags, answer tags, the strict format, the right answer, and '
                     'both.', b)

json.dump(F, open(os.path.join(HERE, 'results', 'figs_part3.json'), 'w'))
print('wrote results/figs_part3.json:', ', '.join(F))
