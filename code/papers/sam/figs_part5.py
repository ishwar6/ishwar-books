"""Figures for Part 5 (zero-shot experiments). Paper numbers are typed from the paper (Figure 18 legends,
Table 4, Figure 13 axis); our numbers come from results/part5.json.
Writes results/figs_part5.json (key -> svg string). Colours come only from CSS classes."""
import json, os
from figlib import svg, text, box, arrow, esc, lines_chart
from samfig import hbars, line

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'part5.json')))
F = {}


def legend(x, y, items):
    b = []
    for cls, lab in items:
        b += [f'<rect class="{cls}" x="{x}" y="{y - 9}" width="10" height="10" rx="2"/>', text(x + 16, y, lab, 't-tick')]
        x += 16 + 7.0 * len(lab) + 22
    return b


# ---------------------------------------------------------------- 1. the five tasks, from low to high level
def ladder():
    rows = [('low', 'Edge detection', '§7.2', '16 × 16 grid of points', 'the edges between regions'),
            ('mid', 'Object proposals', '§7.3', '64 × 64 grid of points', 'about 1,000 masks of possible objects'),
            ('high', 'Instance segmentation', '§7.4', 'the boxes of a detector', 'one mask for each detected object'),
            ('higher', 'Text to mask', '§7.5', 'a sentence, through CLIP', 'the mask of the named object')]
    b = [text(20, 24, 'Core test (§7.1): one point → one valid mask', 't-title'),
         box(20, 36, 740, 40, 'box-on', 8),
         text(36, 61, 'prompt: the "center" of a mask drawn by people', 't-tick'),
         text(744, 61, 'output: SAM\'s most confident mask', 't-tick', 'end'),
         text(20, 106, 'Four tasks that SAM never trained for, from low-level to high-level vision', 't-title')]
    y = 120
    for k, (lvl, name, sec, prompt, out) in enumerate(rows):
        cls = ['box-1', 'box-2', 'box-3', 'box-4'][k]
        b += [box(20 + k * 22, y, 720 - k * 22 + 20, 46, cls, 8),
              text(34 + k * 22, y + 20, f'{name}  {sec}', 't-strong'),
              text(34 + k * 22, y + 37, lvl + '-level', 't-tick'),
              text(440, y + 20, 'prompt: ' + prompt, 't-tick'),
              text(440, y + 37, 'output: ' + out, 't-tick')]
        y += 56
    b.append(text(20, y + 14, 'The same weights in every row. Only the prompts change: "prompt engineering".', 't-note'))
    return svg(780, y + 26, 'The five experiments of section 7. The core test gives one point and expects one valid mask. Then four tasks from low-level to high-level vision: edge detection from a 16 by 16 grid of points, object proposals from a 64 by 64 grid, instance segmentation from the boxes of a detector, and text to mask through CLIP.', b)


F['p5_ladder'] = ladder()


# ---------------------------------------------------------------- 2. our 40 objects: top-scored mask vs oracle
def oracle():
    per = sorted(R['per_object'], key=lambda r: r['iou']['1'])
    W, H = 780, 330
    L, T, B = 56, 30, 46
    pw, ph = W - L - 30, H - T - B
    n = len(per)
    X = lambda i: L + (i + 0.5) / n * pw
    Y = lambda v: T + ph - v * ph
    b = []
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        b += [f'<line class="grid" x1="{L}" y1="{Y(t):.1f}" x2="{L + pw}" y2="{Y(t):.1f}"/>', text(L - 8, Y(t) + 4, f'{t:.2f}', 't-tick', 'end')]
    for i, r in enumerate(per):
        top, best = r['iou']['1'], max(r['ious3'])
        if best - top > 0.005:
            b.append(f'<line class="edge-dim" x1="{X(i):.1f}" y1="{Y(top):.1f}" x2="{X(i):.1f}" y2="{Y(best):.1f}"/>')
        b.append(f'<g class="mark"><title>{esc(r["cat"])}: top-scored {top:.3f}, oracle {best:.3f}</title>'
                 f'<circle class="s3 ring" cx="{X(i):.1f}" cy="{Y(best):.1f}" r="5"/>'
                 f'<circle class="s2 ring" cx="{X(i):.1f}" cy="{Y(top):.1f}" r="5"/></g>')
    S = R['vit_h_center']
    for v, cls, lab in ((S['1'], 'l2', f'mean, top-scored: {S["1"]:.3f}'), (S['oracle'], 'l3', f'mean, oracle: {S["oracle"]:.3f}')):
        b.append(f'<line class="{cls}" x1="{L}" y1="{Y(v):.1f}" x2="{L + pw}" y2="{Y(v):.1f}" style="stroke-dasharray:5 4"/>')
    b.append(text(L + 8, Y(S['oracle']) - 6, f'mean of the oracle masks: {S["oracle"]:.3f}', 't-s3'))
    b.append(text(L + pw - 8, Y(S['1']) + 18, f'mean of the top-scored masks: {S["1"]:.3f}', 't-s2', 'end'))
    b.append(text(L - 44, T - 12, 'IoU with the mask drawn by people', 't-tick'))
    b.append(text(L + pw / 2, H - 12, f'the {n} objects, sorted by the IoU of the top-scored mask', 't-tick', 'middle'))
    b += legend(L + 260, H - 28, [('s2', 'top-scored mask'), ('s3', 'oracle: best of 3')])
    return svg(W, H, 'IoU of SAM ViT-H for each of our 40 objects with one center point. Orange: the mask with the highest score. Green: the best of the three masks. The mean is 0.601 for the top-scored mask and 0.789 for the oracle.', b)


F['p5_oracle'] = oracle()


# ---------------------------------------------------------------- 3. more points
def points():
    xs = [1, 2, 3, 5, 9]
    a, m, r = R['vit_h_center'], R['vit_h_center_with_mask'], R['vit_h_random']
    return lines_chart(780, 320, 'Mean IoU of SAM ViT-H on our 40 objects after 1 to 9 points placed by the error-region rule. With the previous mask as an extra prompt, the IoU grows from 0.601 to 0.884. With points only, it grows more slowly. With a random first point, it starts lower.',
                       xs, [('points + previous mask', [m[str(x)] for x in xs], 'l3', 's3'),
                            ('points only', [a[str(x)] for x in xs], 'l1', 's1'),
                            ('random first point', [r[str(x)] for x in xs], 'l2', 's2')],
                       'number of points', 'mIoU, our 40 objects', 0.4, 1.0, [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                       fmt=lambda v: f'{v:.3f}' if v not in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0) else f'{v:.1f}', right=230)


F['p5_points'] = points()


# ---------------------------------------------------------------- 4. the human study ratings (Figure 18 of the paper)
RATINGS = {   # mean ratings from the legends of Figure 18 (page 24)
    'LVIS v0.5': (6.5, 7.7, 8.1, 8.5), 'VISOR': (6.3, 7.5, 8.3, 8.5), 'DRAM': (5.9, 6.8, 7.7, 8.0), 'IBD': (7.1, 7.9, 8.3, 8.4),
    'NDD20': (6.4, 8.2, 8.6, 8.9), 'OVIS': (6.1, 7.7, 7.2, 8.8), 'iShape': (4.9, 6.2, 7.1, 9.3)}


def ratings():
    W, H = 780, 360
    L, T = 110, 40
    x0, x1 = L, W - 30
    X = lambda v: x0 + (v - 4) / 6 * (x1 - x0)
    b = [text(20, 22, 'Mean rating from 1 (nonsense) to 10 (no visible error), one center point, 1,000 masks per dataset', 't-title')]
    for t in range(4, 11):
        b += [f'<line class="grid" x1="{X(t):.1f}" y1="{T}" x2="{X(t):.1f}" y2="{H - 50}"/>', text(X(t), H - 34, str(t), 't-tick', 'middle')]
    cls = [('RITM', 's2'), ('SAM, single output', 's1'), ('SAM', 's3'), ('masks drawn by people', 't-muted')]
    rh = (H - 50 - T) / len(RATINGS)
    for k, (ds, vals) in enumerate(RATINGS.items()):
        y = T + (k + 0.5) * rh
        b.append(text(L - 12, y + 4, ds, 't-tick', 'end'))
        b.append(f'<line class="edge-dim" x1="{X(min(vals)):.1f}" y1="{y:.1f}" x2="{X(max(vals)):.1f}" y2="{y:.1f}"/>')
        for (lab, c), v in zip(cls, vals):
            if c == 't-muted':
                b.append(f'<g class="mark"><title>{ds}, {lab}: {v}</title><rect class="box-ghost" x="{X(v) - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" rx="1"/></g>')
            else:
                b.append(f'<g class="mark"><title>{ds}, {lab}: {v}</title><circle class="{c} ring" cx="{X(v):.1f}" cy="{y:.1f}" r="6"/></g>')
    b += legend(L, H - 10, [('s2', 'RITM'), ('s1', 'SAM, single output'), ('s3', 'SAM')])
    b.append(f'<rect class="box-ghost" x="{L + 390}" y="{H - 19}" width="10" height="10" rx="1"/>')
    b.append(text(L + 406, H - 10, 'masks drawn by people', 't-tick'))
    return svg(W, H, 'Mean mask ratings from the human study of the paper, for 7 datasets. RITM gets between 4.9 and 7.1. SAM gets between 7.1 and 8.6. The masks drawn by people get between 8.0 and 9.3.', b)


F['p5_ratings'] = ratings()


# ---------------------------------------------------------------- 5. recall at each IoU threshold (our proposals)
def recall():
    P = R['proposals']
    xs = P['thresholds']
    W, H = 780, 300
    L, T, B = 64, 30, 50
    pw, ph = W - L - 200, H - T - B
    X = lambda v: L + 34 + (v - 0.5) / 0.45 * (pw - 50)
    Y = lambda v: T + ph - v * ph
    b = []
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        b += [f'<line class="grid" x1="{L}" y1="{Y(t):.1f}" x2="{L + pw}" y2="{Y(t):.1f}"/>', text(L - 8, Y(t) + 4, f'{t:.2f}', 't-tick', 'end')]
    bw = pw / len(xs) * 0.62
    for t, r in zip(xs, P['curve']):
        b.append(f'<g class="mark"><title>IoU threshold {t:.2f}: recall {r:.3f}</title>'
                 f'<rect class="s1" x="{X(t) - bw / 2:.1f}" y="{Y(r):.1f}" width="{bw:.1f}" height="{Y(0) - Y(r):.1f}" rx="3"/></g>')
        b.append(text(X(t), Y(0) + 18, f'{t:.2f}', 't-tick', 'middle'))
        b.append(text(X(t), Y(r) - 6, f'{r:.2f}', 't-tick', 'middle'))
    a = P['ar1000']
    b.append(f'<line class="l2" x1="{L - 6}" y1="{Y(a):.1f}" x2="{L + pw + 8}" y2="{Y(a):.1f}" style="stroke-dasharray:5 4"/>')
    b.append(text(L + pw + 14, Y(a) + 4, f'AR@1000 = mean = {a:.3f}', 't-s2'))
    b.append(text(L + pw / 2, H - 10, 'IoU threshold: how close a proposal must be to count as "found"', 't-tick', 'middle'))
    b.append(text(L - 50, T - 12, f'recall: share of the {P["objects"]} objects found', 't-tick'))
    return svg(W, H, f'Recall of our SAM proposals at each of the ten IoU thresholds, from 0.50 to 0.95. Average recall is the mean of the ten bars: {a:.3f}.', b)


F['p5_recall'] = recall()


# ---------------------------------------------------------------- 6. encoder size: ours vs the paper's ablation
def encoders():
    E = R['encoders']
    W, H = 780, 250
    b = [text(20, 22, 'Our 40 objects, one center point (mIoU)', 't-title')]
    names = [('base', 'ViT-B', '91M'), ('large', 'ViT-L', '308M'), ('huge', 'ViT-H', '636M')]
    X0, bw = 170, 520
    y = 44
    for key, nm, par in names:
        for k, (lab, cls) in enumerate((('1', 's2'), ('oracle', 's3'))):
            v = E[key][lab]
            yy = y + k * 22
            b += [text(X0 - 12, yy + 14, f'{nm} ({par})' if k == 0 else '', 't-tick', 'end'),
                  f'<g class="mark"><title>{nm} {"top-scored" if lab == "1" else "oracle"}: {v:.3f}</title>'
                  f'<rect class="{cls}" x="{X0}" y="{yy + 2}" width="{bw * v:.1f}" height="16" rx="3"/></g>',
                  text(X0 + bw * v + 6, yy + 15, f'{v:.3f}', 't-val')]
        y += 58
    b += legend(X0, y + 8, [('s2', 'top-scored mask'), ('s3', 'oracle: best of 3')])
    return svg(W, H, 'Our mini test with the three released image encoders. Top-scored mask and oracle mIoU for ViT-B, ViT-L and ViT-H on our 40 objects with one center point.', b)


F['p5_encoders'] = encoders()

json.dump(F, open(os.path.join(HERE, 'results', 'figs_part5.json'), 'w'))
print('figs_part5:', ', '.join(F))
