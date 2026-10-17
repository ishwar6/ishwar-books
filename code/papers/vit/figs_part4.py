"""Figures for Part 4 (results: Table 2, Figure 2, Figures 3 and 4, Table 5, Table 9), drawn from
results/part4_tables.json, part4_fewshot.json, part4_scratch.json and part4_top5.json.
Writes results/figs_part4.json (key -> svg string). CSS classes only, so every theme works."""
import json, math, os
from figlib import svg, text, box, esc

HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda n: json.load(open(os.path.join(HERE, 'results', n + '.json'))) if os.path.exists(os.path.join(HERE, 'results', n + '.json')) else None
T = J('part4_tables')
F = {}
SER = ['s1', 's2', 's3', 's4']
LIN = ['l1', 'l2', 'l3', 'l4']


def axes(x0, y0, w, h, ymin, ymax, yticks, fmt=lambda v: f'{v:g}', ylabel=None):
    Y = lambda v: y0 + h - (v - ymin) / (ymax - ymin) * h
    b = []
    for t in yticks:
        b += [f'<line class="grid" x1="{x0}" y1="{Y(t):.1f}" x2="{x0 + w}" y2="{Y(t):.1f}"/>', text(x0 - 8, Y(t) + 4, fmt(t), 't-tick', 'end')]
    b.append(f'<line class="axis" x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}"/>')
    if ylabel:
        b.append(text(x0 - 46, y0 - 14, ylabel, 't-tick'))
    return b, Y


def dot(cx, cy, cls, title, r=6, shape='circle'):
    if shape == 'square':
        el = f'<rect class="{cls} ring" x="{cx - r:.1f}" y="{cy - r:.1f}" width="{2 * r}" height="{2 * r}" rx="1.5"/>'
    else:
        el = f'<circle class="{cls} ring" cx="{cx:.1f}" cy="{cy:.1f}" r="{r}"/>'
    return f'<g class="mark"><title>{esc(title)}</title>{el}</g>'


def legend(x, y, items, dx=150):
    b = []
    for i, (lab, cls, shape) in enumerate(items):
        xx = x + i * dx
        b.append(dot(xx + 6, y - 4, cls, lab, 5, shape))
        b.append(text(xx + 16, y, lab, 't-tick'))
    return b


# ---------------------------------------------------------------- 1. Table 2: compute against ImageNet accuracy
def compute_scatter():
    pts = [('ViT-H/14 (JFT)', 2500, 88.55, 's1', 'circle'), ('ViT-L/16 (JFT)', 680, 87.76, 's1', 'circle'),
           ('ViT-L/16 (I21k)', 230, 85.30, 's3', 'circle'), ('BiT-L (ResNet152x4)', 9900, 87.54, 's2', 'square'),
           ('Noisy Student (EfficientNet-L2)', 12300, 88.4, 's4', 'square')]
    x0, y0, w, h = 80, 40, 560, 250
    b, Y = axes(x0, y0, w, h, 84.5, 89.5, [85, 86, 87, 88, 89], fmt=lambda v: f'{v:.0f}%', ylabel='ImageNet top-1')
    lx = lambda v: x0 + (math.log10(v) - 2) / (math.log10(30000) - 2) * w
    for t in [100, 300, 1000, 3000, 10000, 30000]:
        b += [f'<line class="grid" x1="{lx(t):.1f}" y1="{y0}" x2="{lx(t):.1f}" y2="{y0 + h}"/>', text(lx(t), y0 + h + 20, f'{t:,}', 't-tick', 'middle')]
    b.append(text(x0 + w / 2, y0 + h + 44, 'TPUv3-core-days to pre-train (log scale)', 't-tick', 'middle'))
    off = {'ViT-H/14 (JFT)': (-12, 4), 'ViT-L/16 (JFT)': (10, 20), 'ViT-L/16 (I21k)': (10, 4), 'BiT-L (ResNet152x4)': (12, 4),
           'Noisy Student (EfficientNet-L2)': (12, 4)}
    for n, c, a, cls, shp in pts:
        b.append(dot(lx(c), Y(a), cls, f'{n}: {c:,} core-days, {a}%', 7, shp))
        dx, dy = off[n]
        b.append(text(lx(c) + dx, Y(a) + dy, f'{n}: {a}%', 't-note', 'start' if dx > 0 else 'end'))
    b.append(f'<line class="long" x1="{lx(680):.1f}" y1="{Y(87.76):.1f}" x2="{lx(9900):.1f}" y2="{Y(87.76):.1f}"/>')
    b.append(text((lx(680) + lx(9900)) / 2, Y(87.76) - 8, f'14.6× less compute, +0.22 points', 't-tick', 'middle'))
    return svg(880, 350, 'Scatter of pre-training compute in TPUv3-core-days, log scale, against ImageNet top-1 accuracy for the five columns of Table 2. The two JFT ViTs sit up and to the left of BiT-L and Noisy Student: as accurate or more, for 4 to 15 times less compute.', b)


# ---------------------------------------------------------------- 2. Table 2 as error rates
def errors():
    rows = ['ImageNet', 'ImageNet ReaL', 'CIFAR-10', 'CIFAR-100', 'Oxford-IIIT Pets', 'Oxford Flowers-102', 'VTAB (19 tasks)']
    E = T['errors']
    b = [text(16, 22, 'error = 100 − accuracy; each row on its own scale', 't-tick')]
    b += legend(380, 22, [('ViT-H/14 (JFT)', 's1', 'circle'), ('BiT-L (ResNet152x4)', 's2', 'square')], 170)
    y = 44
    for r in rows:
        e = E[r]
        m = max(e['h14'], e['bit'])
        b.append(text(170, y + 22, r, 't-tick', 'end'))
        for i, (v, cls) in enumerate([(e['h14'], 's1'), (e['bit'], 's2')]):
            ww = 380 * v / m
            b.append(f'<g class="mark"><title>{esc(r)}: {v:.2f}% error</title><rect class="{cls}" x="180" y="{y + 4 + i * 17}" width="{ww:.1f}" height="14" rx="3"/></g>')
            b.append(text(186 + ww, y + 15 + i * 17, f'{v:.2f}%', 't-tick'))
        b.append(text(690, y + 24, f'{100 * e["cut"]:.0f}% fewer', 't-val', 'end'))
        y += 44
    return svg(700, y + 6, 'Table 2 rewritten as error rates. For each benchmark, two bars: the errors of ViT-H/14 pre-trained on JFT-300M and of BiT-L. ViT-H/14 makes fewer errors on every row, from 2% fewer on ImageNet ReaL to 28% fewer on Oxford-IIIT Pets.', b)


# ---------------------------------------------------------------- 3. Table 9 grouped as Figure 2
def vtab_groups():
    V = T['vtab']
    names = ['ViT-H/14 (JFT)', 'ViT-L/16 (JFT)', 'ViT-L/16 (I21k)']
    groups = [('All 19 tasks', None), ('Natural (7)', 'Natural'), ('Specialized (4)', 'Specialized'), ('Structured (8)', 'Structured')]
    x0, y0, w, h = 70, 50, 600, 230
    b, Y = axes(x0, y0, w, h, 50, 90, [50, 60, 70, 80, 90], fmt=lambda v: f'{v:.0f}%', ylabel='mean accuracy')
    b += legend(110, 24, [(n, SER[i], 'circle') for i, n in enumerate(names)], 175)
    gw = w / 4
    for gi, (lab, key) in enumerate(groups):
        for i, n in enumerate(names):
            v = V[n]['mean'] if key is None else V[n]['groups'][key]
            x = x0 + gi * gw + 22 + i * 38
            b.append(f'<g class="mark"><title>{esc(n)}, {esc(lab)}: {v:.1f}%</title><rect class="{SER[i]}" x="{x:.1f}" y="{Y(v):.1f}" width="32" height="{y0 + h - Y(v):.1f}" rx="3"/></g>')
            b.append(text(x + 16, Y(v) - 6, f'{v:.1f}', 't-cell', 'middle'))
        b.append(text(x0 + gi * gw + gw / 2, y0 + h + 20, lab, 't-tick', 'middle'))
    return svg(700, 320, 'The 19 VTAB tasks of Table 9 averaged into the three groups of Figure 2, for the three ViT models of Table 2. Natural and Specialized tasks score in the 80s; the Structured tasks (counting, distances, angles) are much harder, 57.5 to 65.0, and gain the most from JFT pre-training and a larger model.', b)


# ---------------------------------------------------------------- 4. Table 5 (Figure 3) for three datasets
def table5():
    sizes = ['ViT-B/16', 'ViT-B/32', 'ViT-L/16', 'ViT-L/32', 'ViT-H/14']
    cls = {'ViT-B/16': ('s1', 'l1'), 'ViT-B/32': ('s1', 'l1'), 'ViT-L/16': ('s2', 'l2'), 'ViT-L/32': ('s2', 'l2'), 'ViT-H/14': ('s3', 'l3')}
    T5 = {'ImageNet': {'ImageNet': [77.91, 73.38, 76.53, 71.16, None], 'CIFAR-100': [87.13, 86.31, 86.35, 87.07, None], 'Pets': [93.81, 92.04, 93.64, 91.35, None]},
          'ImageNet-21k': {'ImageNet': [83.97, 81.28, 85.15, 80.99, 85.13], 'CIFAR-100': [91.67, 91.97, 93.44, 93.04, 93.82], 'Pets': [94.43, 93.02, 94.73, 93.09, 94.82]},
          'JFT-300M': {'ImageNet': [84.15, 80.73, 87.12, 84.37, 88.04], 'CIFAR-100': [91.87, 90.49, 94.04, 92.52, 94.55], 'Pets': [95.80, 93.40, 97.11, 95.83, 97.56]}}
    pre = ['ImageNet', 'ImageNet-21k', 'JFT-300M']
    panels = [('ImageNet', 70, 90, [70, 75, 80, 85, 90]), ('CIFAR-100', 85, 95, [85, 90, 95]), ('Pets', 90, 98, [90, 94, 98])]
    b = legend(40, 22, [('ViT-B (solid: /16, dashed: /32)', 's1', 'circle'), ('ViT-L', 's2', 'circle'), ('ViT-H/14', 's3', 'circle')], 250)
    for p, (ds, lo, hi, ticks) in enumerate(panels):
        x0, y0, w, h = 58 + p * 236, 60, 170, 210
        a, Y = axes(x0, y0, w, h, lo, hi, ticks, fmt=lambda v: f'{v:.0f}')
        b += a + [text(x0 + w / 2, y0 - 12, f'fine-tuned on {ds}', 't-title', 'middle')]
        X = lambda i: x0 + 14 + i * (w - 28) / 2
        for i, d in enumerate(pre):
            b.append(text(X(i), y0 + h + 18, ['IN-1k', 'IN-21k', 'JFT'][i], 't-tick', 'middle'))
        for j, n in enumerate(sizes):
            vals = [(i, T5[d][ds][j]) for i, d in enumerate(pre) if T5[d][ds][j] is not None]
            dash = ' style="stroke-dasharray:5 4"' if n.endswith('/32') else ''
            b.append(f'<polyline class="{cls[n][1]}"{dash} points="' + ' '.join(f'{X(i):.1f},{Y(v):.1f}' for i, v in vals) + '"/>')
            for i, v in vals:
                b.append(dot(X(i), Y(v), cls[n][0], f'{n}, pre-trained on {pre[i]}, fine-tuned on {ds}: {v}%', 4.5))
    b.append(text(380, 318, 'pre-training dataset: ImageNet (1.3M pictures), ImageNet-21k (14M), JFT-300M (303M)', 't-tick', 'middle'))
    return svg(760, 330, 'Table 5 drawn as three small charts, one per fine-tuning dataset (ImageNet, CIFAR-100, Oxford-IIIT Pets). Each line is one ViT size across the three pre-training datasets. On ImageNet the smaller ViT-B/16 leads after ImageNet pre-training, and ViT-L/16 and ViT-H/14 lead after JFT-300M.', b)


# ---------------------------------------------------------------- 5. the tiny ridge example
def ridge_tiny():
    pts = [((1, 0), 0), ((2, 1), 0), ((0, 2), 1), ((1, 3), 1)]
    S = 62
    ox, oy = 70, 290
    P = lambda x, y: (ox + x * S, oy - y * S)
    b = []
    for t in range(0, 5):
        b += [f'<line class="grid" x1="{P(t, 0)[0]}" y1="{P(0, 0)[1]}" x2="{P(t, 0)[0]}" y2="{P(0, 4.2)[1]}"/>', text(P(t, 0)[0], oy + 18, str(t), 't-tick', 'middle')]
        b += [f'<line class="grid" x1="{ox}" y1="{P(0, t)[1]}" x2="{P(4.2, 0)[0]}" y2="{P(0, t)[1]}"/>', text(ox - 10, P(0, t)[1] + 4, str(t), 't-tick', 'end')]
    # boundary: x W[:,0] = x W[:,1]  <=>  1.25 x1 - 0.95 x2 = 0  <=>  x2 = 1.3158 x1
    x1e = 4.2 / 1.3158
    b.append(f'<line class="key-line" x1="{ox}" y1="{oy}" x2="{P(x1e, 4.2)[0]:.1f}" y2="{P(x1e, 4.2)[1]:.1f}"/>')
    b.append(text(P(3.25, 4.0)[0] + 8, P(3.25, 4.0)[1] + 4, 'scores equal', 't-tick'))
    for (x, y), c in pts:
        cx, cy = P(x, y)
        b.append(dot(cx, cy, 's1' if c == 0 else 's3', f'image ({x}, {y}), class {c}', 8, 'square' if c == 0 else 'circle'))
        b.append(text(cx + 10, cy + 22, f'({x}, {y})', 't-tick') if x == 0 else text(cx + 12, cy - 8, f'({x}, {y})', 't-tick'))
    cx, cy = P(1, 2)
    b.append(f'<circle class="ring" cx="{cx}" cy="{cy}" r="8" style="fill:none;stroke:var(--text-strong);stroke-width:2;stroke-dasharray:3 2"/>')
    b.append(text(cx, cy - 14, 'new (1, 2)', 't-note', 'middle'))
    b.append(text(P(3.9, 0.5)[0], P(3.9, 0.5)[1], 'class 0 side', 't-note', 'end'))
    b.append(text(P(0.15, 3.8)[0], P(0.15, 3.8)[1], 'class 1 side', 't-note'))
    b += [box(390, 60, 290, 182, 'box', 10),
          text(406, 86, 'W (from the closed form, λ = 1)', 't-title'),
          text(406, 112, 'class 0 column:  ( 0.625, −0.475)', 't-tick'),
          text(406, 132, 'class 1 column:  (−0.625,  0.475)', 't-tick'),
          text(406, 162, 'new image x = (1, 2):', 't-tick'),
          text(406, 182, 'score 0 = 0.625 − 0.950 = −0.325', 't-tick'),
          text(406, 202, 'score 1 = −0.625 + 0.950 = 0.325', 't-tick'),
          text(406, 222, 'larger score wins: class 1', 't-note')]
    b.append(text(ox + 130, oy + 40, 'feature 1', 't-tick', 'middle'))
    b.append(text(ox - 56, 22, 'feature 2', 't-tick'))
    return svg(700, 340, 'The tiny few-shot example drawn. Two class-0 images at (1, 0) and (2, 1), two class-1 images at (0, 2) and (1, 3). The closed-form weights split the plane along the line where both scores are equal; the new image (1, 2) falls on the class-1 side, with scores -0.325 and 0.325.', b)


F['p4_compute'] = compute_scatter()
F['p4_errors'] = errors()
F['p4_vtab_groups'] = vtab_groups()
F['p4_table5'] = table5()
F['p4_ridge'] = ridge_tiny()


# ---------------------------------------------------------------- 6. our few-shot curves (Figure 4's axis, turned around)
FS = J('part4_fewshot')
if FS:
    def fewshot():
        models = [('ViT-B/32', 's1', 'l1', 'circle', ' style="stroke-dasharray:5 4"'), ('ViT-B/16', 's1', 'l1', 'circle', ''),
                  ('ViT-L/32', 's2', 'l2', 'circle', ' style="stroke-dasharray:5 4"'), ('ViT-L/16', 's2', 'l2', 'circle', ''),
                  ('ViT-H/14', 's3', 'l3', 'circle', ''), ('BiT R50x1', 's4', 'l4', 'square', ''),
                  ('ResNet-50 (ImageNet-1k)', 's4', 'l4', 'square', ' style="stroke-dasharray:5 4"')]
        shots = [1, 2, 5, 10]
        b = []
        for p, ds in enumerate(['CIFAR-100', 'Oxford-IIIT Pets']):
            x0, y0, w, h = 60 + p * 430, 50, 230, 250
            lo, hi = (20, 90) if ds == 'CIFAR-100' else (60, 100)
            a, Y = axes(x0, y0, w, h, lo, hi, list(range(lo, hi + 1, 10)), fmt=lambda v: f'{v:.0f}%')
            b += a + [text(x0 + w / 2, y0 - 16, f'{ds} ({100 if p == 0 else 37} classes)', 't-title', 'middle')]
            X = lambda k: x0 + 10 + math.log(k) / math.log(10) * (w - 20)
            for k in shots:
                b.append(text(X(k), y0 + h + 18, str(k), 't-tick', 'middle'))
            b.append(text(x0 + w / 2, y0 + h + 38, 'labelled images per class (log scale)', 't-tick', 'middle'))
            ends = []
            for n, sc, lc, shp, dash in models:
                if n not in FS[ds]:
                    continue
                vals = [FS[ds][n]['acc'][str(k)][0] for k in shots]
                b.append(f'<polyline class="{lc}"{dash} points="' + ' '.join(f'{X(k):.1f},{Y(v):.1f}' for k, v in zip(shots, vals)) + '"/>')
                for k, v in zip(shots, vals):
                    b.append(dot(X(k), Y(v), sc, f'{n}, {ds}, {k}-shot: {v:.1f}%', 4.5, shp))
                ends.append([Y(vals[-1]), f'{n.replace(" (ImageNet-1k)", " IN-1k")} {vals[-1]:.1f}', sc, Y(vals[-1]), shp])
            ends.sort()
            for i in range(1, len(ends)):
                ends[i][0] = max(ends[i][0], ends[i - 1][0] + 15)
            for ly, lab, sc, ty, shp in ends:
                b.append(f'<line class="grid" x1="{X(10) + 6:.1f}" y1="{ty:.1f}" x2="{X(10) + 14:.1f}" y2="{ly:.1f}"/>')
                b.append(dot(X(10) + 20, ly, sc, lab, 4, shp))
                b.append(text(X(10) + 28, ly + 4, lab, 't-tick'))
        return svg(900, 350, 'Our few-shot linear accuracy on frozen features, against the number of labelled images per class, for CIFAR-100 and Oxford-IIIT Pets. Larger ViTs and smaller patches give higher curves; the ResNet-50s are below every ViT pre-trained on ImageNet-21k.', b)
    F['p4_fewshot'] = fewshot()

# ---------------------------------------------------------------- 7. our from-scratch runs (Section 4.3 in miniature)
SC = J('part4_scratch')
if SC and len(SC['runs']) >= 2:
    def scratch():
        sizes = sorted(int(n) for n in SC['runs'])
        x0, y0, w, h = 80, 50, 470, 250
        vals = [SC['runs'][str(n)][m] for n in sizes for m in ('vit', 'resnet')]
        lo = 10 * math.floor(min(vals) / 10)
        a, Y = axes(x0, y0, w, h, lo, 100, list(range(lo, 101, 10)), fmt=lambda v: f'{v:.0f}%', ylabel='best validation accuracy')
        b = a
        X = lambda n: x0 + 10 + (math.log10(n) - math.log10(sizes[0])) / (math.log10(sizes[-1]) - math.log10(sizes[0])) * (w - 20)
        for n in sizes:
            b.append(text(X(n), y0 + h + 18, f'{n:,}', 't-tick', 'middle'))
        b.append(text(x0 + w / 2, y0 + h + 40, 'CIFAR-10 training images (log scale)', 't-tick', 'middle'))
        for m, sc, lc, shp, lab in [('vit', 's1', 'l1', 'circle', 'ViT'), ('resnet', 's2', 'l2', 'square', 'ResNet')]:
            v = [SC['runs'][str(n)][m] for n in sizes]
            b.append(f'<polyline class="{lc}" points="' + ' '.join(f'{X(n):.1f},{Y(a):.1f}' for n, a in zip(sizes, v)) + '"/>')
            for n, a2 in zip(sizes, v):
                b.append(dot(X(n), Y(a2), sc, f'{lab}, {n:,} images: {a2:.1f}%', 5, shp))
                b.append(text(X(n), Y(a2) + (-10 if m == 'resnet' else 18), f'{a2:.1f}', 't-tick', 'middle'))
            b.append(text(X(sizes[-1]) + 14, Y(v[-1]) + 4, lab, 't-note'))
        return svg(640, 350, 'Best validation accuracy of a small ViT and a small ResNet trained from scratch on growing subsets of CIFAR-10 with the same recipe. The ResNet is ahead at every size we can afford, by 15.5 to 20.5 points.', b)
    F['p4_scratch'] = scratch()

json.dump(F, open(os.path.join(HERE, 'results', 'figs_part4.json'), 'w'))
print('figures:', ', '.join(F))
