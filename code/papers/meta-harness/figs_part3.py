"""SVG figures for Part 3, drawn from results/part3.json (run mh_part3.py first). Writes results/figs_part3.json."""
import json, os
from figlib import svg, text, box, arrow, esc

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'part3.json')))
F = {}
SHORT = {'zero-shot': 'zero-shot', 'few-shot (8)': 'few-shot (8)', 'draft-verification': 'draft-verification',
         'label-primed query': 'label-primed query', 'TF-IDF 5-NN vote (no model)': '5-NN vote, no model'}


# ------------------------------------------------------------------ the online classification setup
def online():
    b = []
    b.append(text(20, 24, 'labeled examples arrive one at a time', 't-title'))
    for i, (t, y) in enumerate([('"burning in my stomach..."', 'peptic ulcer'), ('"itchy red rash..."', 'fungal infection'),
                                ('"high fever, chills..."', 'malaria')]):
        x = 20 + i * 190
        b.append(box(x, 40, 175, 50, 'box-2'))
        b.append(text(x + 10, 60, t, 't-tick'))
        b.append(text(x + 10, 80, '-> ' + y, 't-note'))
    b.append(arrow(300, 92, 300, 128))
    b.append(box(150, 130, 300, 46, 'box'))
    b.append(text(300, 158, 'memory: every labeled example so far', 't-note', 'middle'))
    b.append(box(20, 220, 170, 60, 'box-3'))
    b.append(text(105, 245, 'new description', 't-note', 'middle'))
    b.append(text(105, 265, '(no label)', 't-tick', 'middle'))
    b.append(box(240, 210, 220, 80, 'box-1'))
    b.append(text(350, 238, 'HARNESS (the code)', 't-title', 'middle'))
    b.append(text(350, 258, 'what to retrieve, what to show,', 't-tick', 'middle'))
    b.append(text(350, 276, 'how many calls', 't-tick', 'middle'))
    b.append(arrow(190, 250, 238, 250))
    b.append(arrow(320, 178, 320, 208))
    b.append(box(510, 220, 130, 60, 'box'))
    b.append(text(575, 245, 'frozen model', 't-note', 'middle'))
    b.append(text(575, 265, 'GPT-OSS-120B', 't-tick', 'middle'))
    b.append(arrow(462, 240, 508, 240))
    b.append(arrow(508, 262, 462, 262))
    b.append(box(510, 320, 130, 40, 'box-2'))
    b.append(text(575, 345, 'one label', 't-note', 'middle'))
    b.append(arrow(400, 292, 520, 318))
    b.append(text(20, 330, 'Meta-Harness rewrites only the HARNESS box.', 't-note'))
    b.append(text(20, 350, 'The model and the data stay the same.', 't-tick'))
    F['p3_online'] = svg(660, 375, 'Online text classification. Labeled examples arrive one at a time and go into a '
                         'memory. For a new description, the harness decides what to take from the memory and what '
                         'prompt to send to a frozen model, which returns one label.', b)


# ------------------------------------------------------------------ scatter helper
def scatter(key, pts, xlabel, ylabel, xmax, ymin, ymax, xticks, yticks, label, frontier=None, diagonal=False,
            xmin=0, fmtx=lambda v: f'{v:,}', w=680, h=380):
    L, T, Rm, B = 64, 24, 30, 54
    pw, ph = w - L - Rm, h - T - B
    X = lambda v: L + (v - xmin) / (xmax - xmin) * pw
    Y = lambda v: T + ph - (v - ymin) / (ymax - ymin) * ph
    b = []
    for t in yticks:
        b += [f'<line class="grid" x1="{L}" y1="{Y(t):.1f}" x2="{L + pw}" y2="{Y(t):.1f}"/>',
              text(L - 8, Y(t) + 4, f'{t}', 't-tick', 'end')]
    for t in xticks:
        b.append(text(X(t), T + ph + 20, fmtx(t), 't-tick', 'middle'))
    b.append(f'<line class="axis" x1="{L}" y1="{T + ph}" x2="{L + pw}" y2="{T + ph}"/>')
    if diagonal:
        lo, hi = max(xmin, ymin), min(xmax, ymax)
        b.append(f'<line class="base-line" x1="{X(lo):.1f}" y1="{Y(lo):.1f}" x2="{X(hi):.1f}" y2="{Y(hi):.1f}"/>')
        b.append(text(X(hi) - 6, Y(hi) + 16, 'y = x', 't-tick', 'end'))
    if frontier:
        fp = sorted([p for p in pts if p[0] in frontier], key=lambda p: p[1])
        d = ' '.join(f'{X(a):.1f},{Y(c):.1f}' for _, a, c, *_ in fp)
        b.append(f'<polyline class="l1" points="{d}" style="stroke-dasharray:6 4"/>')
    for name, x, y, cls, dx, dy, anchor in pts:
        b.append(f'<g class="mark"><title>{esc(name)}: {fmtx(round(x, 1))}, {y:.1f}</title>'
                 f'<circle class="{cls} ring" cx="{X(x):.1f}" cy="{Y(y):.1f}" r="7"/></g>')
        b.append(text(X(x) + dx, Y(y) + dy, f'{name} ({y:.1f})', 't-note', anchor))
    b.append(text(L + pw / 2, h - 12, xlabel, 't-tick', 'middle'))
    b.append(text(L - 50, T - 8, ylabel, 't-tick'))
    F[key] = svg(w, h, label, b)


def pareto():
    rows = R['rows']
    hs = ['zero-shot', 'few-shot (8)', 'draft-verification', 'label-primed query']
    front = R['frontier']
    pts = []
    place = {'zero-shot': (12, -10, 'start'), 'few-shot (8)': (10, 22, 'start'),
             'draft-verification': (0, -14, 'middle'), 'label-primed query': (-10, 24, 'end')}
    for h in hs:
        dx, dy, a = place[h]
        pts.append((SHORT[h], rows[h]['extra'], rows[h]['test'], 's1' if h in front else 's3', dx, dy, a))
    xmax = (int(max(p[1] for p in pts) / 1000) + 1) * 1000
    lo = int(min(p[2] for p in pts) / 10) * 10 - 10
    hi = (int(max(p[2] for p in pts) / 10) + 1) * 10
    scatter('p3_pareto', pts, 'extra characters in the prompt, compared with zero-shot (average per test example)',
            'test accuracy (%)', xmax, lo, hi, list(range(0, xmax + 1, 1000)), list(range(lo, hi + 1, 10)),
            'Our four harnesses on the Symptom2Disease test split: extra prompt characters on the x axis, test accuracy '
            'on the y axis. The dashed line joins the points on the Pareto frontier.',
            frontier=[SHORT[h] for h in front])


def search_test():
    rows = R['rows']
    hs = ['zero-shot', 'few-shot (8)', 'draft-verification', 'label-primed query', 'TF-IDF 5-NN vote (no model)']
    place = {'zero-shot': (12, -8, 'start'), 'few-shot (8)': (12, 18, 'start'), 'draft-verification': (-12, 4, 'end'),
             'label-primed query': (-12, -8, 'end'), 'TF-IDF 5-NN vote (no model)': (-12, -12, 'end')}
    pts = [(SHORT[h], rows[h]['search'], rows[h]['test'], 's2' if 'vote' in h else 's1', *place[h]) for h in hs]
    lo = int(min(min(p[1], p[2]) for p in pts) / 10) * 10
    scatter('p3_search_test', pts, 'search accuracy (%), 100 held-out training examples', 'test accuracy (%), 212 test examples',
            100, lo, 100, list(range(lo, 101, 10)), list(range(lo, 101, 10)),
            'Search-set accuracy against test accuracy for our five harnesses. Points on the dashed line y = x '
            'score the same on both sets.', diagonal=True, xmin=lo, fmtx=lambda v: f'{v:g}')


def per_label():
    acc = R['per_label']
    a, b_ = acc['zero-shot'], acc['label-primed query']
    labs = sorted(a, key=lambda l: b_[l] - a[l], reverse=True)
    rh, L, W = 22, 210, 360
    h = 50 + rh * len(labs) + 30
    b = [text(20, 22, 'test accuracy per disease (%)', 't-title'),
         f'<rect class="s3" x="{L + 150}" y="12" width="12" height="12" rx="2"/>', text(L + 168, 22, 'zero-shot', 't-tick'),
         f'<rect class="s1" x="{L + 240}" y="12" width="12" height="12" rx="2"/>', text(L + 258, 22, 'label-primed query', 't-tick')]
    for t in [0, 50, 100]:
        x = L + W * t / 100
        b += [f'<line class="grid" x1="{x:.1f}" y1="38" x2="{x:.1f}" y2="{h - 26}"/>', text(x, h - 10, f'{t}', 't-tick', 'middle')]
    for i, l in enumerate(labs):
        y = 42 + i * rh
        b.append(text(L - 8, y + 13, l, 't-tick', 'end'))
        b.append(f'<g class="mark"><title>{esc(l)}: zero-shot {a[l]:.0f}%, label-primed query {b_[l]:.0f}%</title>'
                 f'<rect class="s3" x="{L}" y="{y + 2:.1f}" width="{W * a[l] / 100:.1f}" height="8" rx="2"/>'
                 f'<rect class="s1" x="{L}" y="{y + 11:.1f}" width="{W * b_[l] / 100:.1f}" height="8" rx="2"/></g>')
    F['p3_per_label'] = svg(L + W + 40, h, 'Test accuracy for each of the 22 diseases, zero-shot against label-primed '
                            'query, sorted by the gain.', b)


online()
pareto()
search_test()
per_label()
json.dump(F, open(os.path.join(HERE, 'results', 'figs_part3.json'), 'w'))
print('figures:', ', '.join(F))
