"""Figures for Part 3 (fine-tuning, higher resolution and the setup), drawn from results/part3_setup.json,
part3_posemb.json, part3_fewshot.json and part3_finetune.json. Writes results/figs_part3.json (key -> svg string).
CSS classes only, so every figure follows the dark, dim and light themes."""
import json, math, os
from figlib import svg, text, box, arrow, esc
from vitfig import token, line, hbars

S = json.load(open('results/part3_setup.json'))
load = lambda n: json.load(open(f'results/{n}.json')) if os.path.exists(f'results/{n}.json') else None
PE, FS, FT = load('part3_posemb'), load('part3_fewshot'), load('part3_finetune')
F = {}


def lines(x, y, ls, cls='t-muted', dy=16, anchor='start'):
    return [text(x, y + i * dy, s, cls, anchor) for i, s in enumerate(ls)]


# ---------------------------------------------------------------- 1. the head swap
def head_swap():
    b = [text(20, 24, 'Fine-tuning swaps the head: two pre-trained layers out, one zero layer in', 't-title')]
    # left: pre-trained
    b.append(text(170, 56, 'pre-trained on ImageNet-21k', 't-note', 'middle'))
    b += token(70, 270, 200, 'encoder: 12 layers, 197 × 768', 'box-1', h=34)
    b += [arrow(170, 268, 170, 240)]
    b += token(70, 206, 200, 'y = LN(z_L⁰): 768', 'box', h=32)
    b += [arrow(170, 204, 170, 176)]
    b += token(70, 142, 200, 'pre-logits 768×768, tanh', 'box-ghost', h=32)
    b += [arrow(170, 140, 170, 112)]
    b += token(70, 78, 200, '21,843-class layer', 'box-ghost', h=32)
    b += [line(64, 70, 276, 182, 'bad-line'), line(276, 70, 64, 182, 'bad-line'), text(170, 202 - 8, '', 't-muted')]
    b.append(text(282, 116, 'removed', 't-bad'))
    # right: fine-tune
    b.append(text(560, 56, 'fine-tuned on a task with K = 10 classes', 't-note', 'middle'))
    b += token(460, 270, 200, 'the same encoder', 'box-1', h=34)
    b += [arrow(560, 268, 560, 240)]
    b += token(460, 206, 200, 'y: 768', 'box', h=32)
    b += [arrow(560, 204, 560, 176)]
    b += token(460, 142, 200, 'W_head 768 × 10 = 0, b = 0', 'box-on', h=32)
    b += [arrow(560, 140, 560, 112)]
    b += token(460, 78, 200, '10 scores, all 0', 'box', h=32)
    s0 = FT['step0'] if FT else None
    if s0:
        b += lines(680, 90, ['step 0:', f'loss = ln 10 = {s0["loss"]:.4f}', f'head gradient: {s0["head_grad"]:.3f}',
                             f'body gradient: {s0["body_grad"]:.1f}'], 't-tick', 18)
    b += [arrow(300, 287, 452, 287)]
    b += lines(20, 336, ['The body keeps its pre-trained weights. At the first step the zero head gives every class the same score, so only the head',
                         'learns; the body starts to move at step 2, once the head is no longer zero.'])
    return svg(800, 362, 'The head swap: the ImageNet-21k checkpoint ends in a pre-logits layer and a 21,843-class layer, both removed; fine-tuning attaches one zero-initialised 768 by 10 layer. At step 0 the loss is ln 10 and only the head receives a gradient.', b)


F['p3_head_swap'] = head_swap()


# ---------------------------------------------------------------- 2. sequence length and compute
def seq_len():
    rows = S['variants']
    b = [text(20, 24, 'Forward compute of one picture (billions of operations), at 224 and 384 pixels', 't-title')]
    names = ['ViT-B/32', 'ViT-B/16', 'ViT-L/32', 'ViT-L/16', 'ViT-H/14']
    vmax = max(r['gflops'] for r in rows)
    x0, lw, bw = 20, 90, 520
    y = 50
    for n in names:
        for res, cls in [(224, 's1'), (384, 's2')]:
            r = next(q for q in rows if q['name'] == n and q['res'] == res)
            w = max(2, bw * r['gflops'] / vmax)
            if res == 224:
                b.append(text(x0 + lw - 8, y + 20, n, 't-tick t-strong', 'end'))
            b.append(f'<g class="mark"><title>{esc(n)} at {res}: {r["gflops"]} GFLOPs, sequence {r["seq"]}</title>'
                     f'<rect class="{cls}" x="{x0 + lw}" y="{y + 2}" width="{w:.1f}" height="14" rx="3"/></g>')
            b.append(text(x0 + lw + w + 6, y + 14, f'{r["gflops"]:,.1f}   ({r["seq"]} tokens)', 't-tick'))
            y += 18
        y += 10
    b += [f'<rect class="s1" x="560" y="40" width="12" height="12" rx="2"/>', text(578, 51, '224 × 224', 't-tick'),
          f'<rect class="s2" x="660" y="40" width="12" height="12" rx="2"/>', text(678, 51, '384 × 384', 't-tick')]
    b += lines(20, y + 14, ['Estimate from the shapes: 2 × (12 N D² + 2 N² D) per layer, N tokens of width D. Halving the patch side',
                            'gives 4 times the tokens; ViT-B/16 costs 4.06 times ViT-B/32 while having 1.9% fewer parameters.'])
    return svg(800, y + 48, 'Horizontal bars of the estimated forward compute for ViT-B/32, B/16, L/32, L/16 and H/14 at 224 and 384 pixels, with the sequence length of each; smaller patches and larger pictures both multiply the work.', b)


F['p3_seq_len'] = seq_len()


# ---------------------------------------------------------------- 3. tiny interpolation
def grid_vals(x, y, M, cell, title):
    b = [text(x + cell * len(M[0]) / 2, y - 10, title, 't-note', 'middle')]
    for i, r in enumerate(M):
        for j, v in enumerate(r):
            b.append(f'<rect class="cell" x="{x + j * cell + 1:.1f}" y="{y + i * cell + 1:.1f}" width="{cell - 2}" height="{cell - 2}" rx="4" style="fill-opacity:{0.1 + 0.6 * v / 3:.2f}"/>')
            b.append(text(x + j * cell + cell / 2, y + i * cell + cell / 2 + 5, f'{v:g}', 't-val', 'middle'))
    return b


def interp_tiny():
    T = PE['tiny'] if PE else {'g': [[0, 1], [2, 3]], 'z': [[0, .5, 1], [1, 1.5, 2], [2, 2.5, 3]]}
    b = [text(20, 24, 'Bilinear interpolation: stretch a 2 × 2 grid to 3 × 3', 't-title')]
    b += grid_vals(80, 70, T['g'], 66, 'old grid, 2 × 2')
    b += [arrow(240, 136, 330, 136), text(285, 126, 'zoom, order 1', 't-tick', 'middle')]
    b += grid_vals(360, 50, T['z'], 58, 'new grid, 3 × 3')
    b += lines(560, 80, ['corners keep their values', 'edge middles: average of 2', '  (0 + 1)/2 = 0.5', 'centre: average of 4',
                         '  (0 + 1 + 2 + 3)/4 = 1.5', '', 'In ViT each cell is a vector', 'of 768 numbers, and the grid', 'goes from 14 × 14 to 24 × 24.'], 't-tick', 18)
    return svg(800, 250, 'A 2 by 2 grid of the values 0, 1, 2 and 3 stretched to a 3 by 3 grid by bilinear interpolation: the corners keep their values, the edge middles are 0.5, 1, 2 and 2.5, and the centre is 1.5.', b)


F['p3_interp_tiny'] = interp_tiny()


# ---------------------------------------------------------------- 4. similarity maps before / after
def simgrid(x0, y0, M, size, mark, title):
    n = len(M); c = size / n
    lo, hi = -0.4, 1.0
    b = [text(x0 + size / 2, y0 - 10, title, 't-note', 'middle')]
    for i in range(n):
        for j in range(n):
            v = (M[i][j] - lo) / (hi - lo)
            cls = 'cell'
            b.append(f'<rect class="{cls}" x="{x0 + j * c:.2f}" y="{y0 + i * c:.2f}" width="{c - 0.6:.2f}" height="{c - 0.6:.2f}" style="fill-opacity:{max(0.03, min(1, v)):.3f}"/>')
    r, cc = mark
    b.append(f'<rect class="box-on" x="{x0 + cc * c:.2f}" y="{y0 + r * c:.2f}" width="{c:.2f}" height="{c:.2f}" style="fill-opacity:0"/>')
    return b


def posemb_sim():
    if not PE:
        return svg(10, 10, 'pending', [])
    b = [text(20, 24, 'Cosine similarity of one position row with every other row', 't-title')]
    b += simgrid(60, 60, PE['sim14'], 280, PE['sim_at'][0], f'14 × 14, as trained: row ({PE["sim_at"][0][0]}, {PE["sim_at"][0][1]})')
    b += [arrow(352, 200, 418, 200), text(385, 190, '2D stretch', 't-tick', 'middle')]
    b += simgrid(430, 60, PE['sim24'], 280, PE['sim_at'][1], f'24 × 24, after: the same place ({PE["sim_at"][1][0]}, {PE["sim_at"][1][1]})')
    b += lines(60, 364, ['Bright = similar (up to 1), faint = unlike (down to about −0.3). The neighbours of a row are similar to it, far rows are not;',
                         'the stretched grid keeps that map and only makes it smoother.'])
    return svg(800, 392, 'Two heatmaps of cosine similarity: one position row of the 14 by 14 grid of ViT-B/16 against all 196 rows, and the same place on the stretched 24 by 24 grid against all 576 rows; both are bright near the chosen place and faint far away.', b)


F['p3_posemb_sim'] = posemb_sim()


# ---------------------------------------------------------------- 5. accuracy at 384 with each position table
def res_acc():
    if not PE or len(PE.get('acc', [])) < 5:
        return svg(10, 10, 'pending', [])
    short = ['224: the training size', '384: 2D bilinear (official)', '384: 2D bicubic (library)', '384: 1D stretch of the list', '384: no position rows']
    items = [(s, 100 * a['acc1000']) for s, a in zip(short, PE['acc'])]
    b = hbars(20, 56, items, 400, row_h=34, cls='s1', label_w=220, fmt=lambda v: f'{v:.2f}%', vmax=100,
              title=f'ViT-B/16 (ImageNet fine-tuned) on {PE["n_val"]:,} Imagenette pictures, top-1 of all 1000 classes, no training')
    b += lines(20, 250, [f'Read at only the 10 Imagenette classes: ' + ', '.join(f'{100 * a["acc10"]:.2f}%' for a in PE['acc']) + ' (same order).'])
    return svg(800, 270, 'Bars of accuracy for five position tables: 224 pixels, 384 with 2D bilinear, 384 with 2D bicubic, 384 with a 1D stretch and 384 with no position rows; the two 2D versions keep the accuracy, the other two lose much of it.', b)


F['p3_res_acc'] = res_acc()


# ---------------------------------------------------------------- 6. datasets on a log scale
def datasets():
    items = [('Imagenette (ours)', 9469, '10 classes'), ('ImageNet', 1.3e6, '1k classes'), ('ImageNet-21k', 14e6, '21k classes'), ('JFT-300M', 303e6, '18k classes')]
    b = [text(20, 24, 'Training pictures, log scale', 't-title')]
    L, W = 170, 520
    X = lambda v: L + (math.log10(v) - 3) / (9 - 3) * W
    for t in range(3, 10):
        b += [f'<line class="grid" x1="{X(10 ** t):.1f}" y1="40" x2="{X(10 ** t):.1f}" y2="210"/>',
              text(X(10 ** t), 228, ['1k', '10k', '100k', '1M', '10M', '100M', '1B'][t - 3], 't-tick', 'middle')]
    for i, (n, v, c) in enumerate(items):
        y = 50 + i * 40
        cls = 's2' if 'ours' in n else 's1'
        b.append(text(L - 8, y + 18, n, 't-tick t-strong', 'end'))
        b.append(f'<g class="mark"><title>{esc(n)}: {v:,.0f} pictures</title><rect class="{cls}" x="{L}" y="{y + 4}" width="{X(v) - L:.1f}" height="20" rx="3"/></g>')
        lab = f'{v / 1e6:g}M, {c}' if v >= 1e6 else f'{v:,}, {c}'
        b.append(text(X(v) + 6, y + 19, lab, 't-tick'))
    return svg(800, 244, 'Bars on a log scale of the number of training pictures: Imagenette 9,469 (ours), ImageNet 1.3 million, ImageNet-21k 14 million and JFT-300M 303 million.', b)


F['p3_datasets'] = datasets()


# ---------------------------------------------------------------- 7. Batch Norm vs Group Norm
def bn_gn():
    G = S['bn_gn']
    b = [text(20, 24, 'Which numbers are averaged: Batch Norm against Group Norm', 't-title')]

    def batch(x0, title, hl):
        parts = [text(x0 + 150, 56, title, 't-note', 'middle')]
        for p, name in enumerate(['picture A', 'picture B']):
            parts.append(text(x0 + 70 + p * 160, 80, name, 't-tick', 'middle'))
            for ch in range(4):
                for px in range(2):
                    on = (p, ch) in hl
                    cls = 'box-on' if on else 'box'
                    parts.append(box(x0 + 30 + p * 160 + px * 40, 90 + ch * 34, 38, 30, cls, 4))
                parts.append(text(x0 + 20 + p * 160, 110 + ch * 34, f'c{ch + 1}', 't-tick', 'end'))
        return parts
    b += batch(20, 'Batch Norm: channel 1 across the batch', {(0, 0), (1, 0)})
    b += batch(420, 'Group Norm: channels 1 and 2 of one picture', {(0, 0), (0, 1)})
    f = lambda v: ', '.join(f'{x:.3f}' for x in v)
    b += lines(20, 250, [f'Tiny example, picture A channel 1 = {G["a"][0]}: with Batch Norm it becomes [{f(G["bn1"])}] with one partner and [{f(G["bn2"])}] with another.',
                         f'With Group Norm (group mean {G["gmean"]:.1f}, std {G["gstd"]:.3f}) it is always [{f(G["gn"])}]: the batch never enters.'], 't-muted', 18)
    return svg(800, 296, 'Two batches of two pictures with four channels of two pixels. Batch Norm highlights channel 1 in both pictures, averaged together; Group Norm highlights channels 1 and 2 of picture A only.', b)


F['p3_bn_gn'] = bn_gn()


# ---------------------------------------------------------------- 8. hybrid stages
def hybrid():
    st = S['resnet_stages']
    b = [text(20, 24, 'ResNet-50 on a 224 × 224 picture: each stage halves the grid', 't-title')]
    slots = [30, 290, 470, 640]
    for i, (c, h, w) in enumerate(st):
        x = slots[i]
        size = 160 * h / 56
        y = 60 + (160 - size) / 2
        cls = 'box-on' if i == 2 else ('box-4' if i == 3 else 'box-1')
        b.append(box(x, y, size, size, cls, 4))
        b.append(text(x + size / 2, 246, f'stage {i + 1}', 't-tick t-strong', 'middle'))
        b.append(text(x + size / 2, 264, f'{h} × {w} × {c}', 't-tick', 'middle'))
        b.append(text(x + size / 2, 282, f'{h * w:,} tokens', 't-tick', 'middle'))
        if i < 3:
            b.append(arrow(x + size + 8, 140, slots[i + 1] - 8, 140))
    b += lines(30, 316, ['Option (i) reads stage 4: 49 tokens of 2,048 numbers. Option (ii) moves stage 4\'s layers into stage 3 and reads its 14 × 14 grid:',
                         '196 tokens, 4 times longer, the same length as ViT-B/16. Each token becomes one "1 × 1 patch" for E.'])
    return svg(800, 344, 'The four stage outputs of ResNet-50 drawn as shrinking squares: 56 by 56, 28 by 28, 14 by 14 and 7 by 7, with 3,136, 784, 196 and 49 tokens; stage 3 and stage 4 are highlighted as the two hybrid options.', b)


F['p3_hybrid'] = hybrid()


# ---------------------------------------------------------------- 9. learning-rate schedules
def panel(x0, y0, w, h, pts, title, ymax, yfmt, xmax, xlab):
    X = lambda t: x0 + t / xmax * w
    Y = lambda v: y0 + h - v / ymax * h
    b = [text(x0, y0 - 12, title, 't-note')]
    for f in (0, 0.5, 1):
        b += [f'<line class="grid" x1="{x0}" y1="{Y(ymax * f):.1f}" x2="{x0 + w}" y2="{Y(ymax * f):.1f}"/>', text(x0 - 6, Y(ymax * f) + 4, yfmt(ymax * f), 't-tick', 'end')]
    b.append(f'<line class="axis" x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}"/>')
    b.append('<polyline class="l1" points="' + ' '.join(f'{X(t):.1f},{Y(v):.1f}' for t, v in pts) + '"/>')
    for t in (0, xmax / 2, xmax):
        b.append(text(X(t), y0 + h + 16, f'{t / 1000:,.0f}k', 't-tick', 'middle'))
    b.append(text(x0 + w / 2, y0 + h + 34, xlab, 't-muted', 'middle'))
    return b


def lr_schedules():
    b = [text(20, 24, 'Learning rate over training (Tables 3 and 4)', 't-title')]
    pre, im, ft = S['curve_pre'], S['curve_pre_im'], S['curve_ft']
    b += panel(70, 64, 190, 150, pre, 'ViT-B/16 on JFT-300M', 8e-4, lambda v: f'{v * 1e4:.0f}e-4' if v else '0', pre[-1][0], 'steps; warm-up 10k, linear decay')
    b += panel(340, 64, 190, 150, im, 'ViT on ImageNet', 3e-3, lambda v: f'{v * 1e3:.1f}e-3' if v else '0', im[-1][0], 'steps; warm-up 10k, cosine')
    b += panel(600, 64, 170, 150, ft, 'fine-tuning on ImageNet', 0.03, lambda v: f'{v:.3f}' if v else '0', 20000, 'steps; cosine, base 0.03')
    return svg(800, 270, 'Three learning-rate curves: pre-training on JFT-300M with a short linear warm-up and a long linear decay, ImageNet training with warm-up and cosine decay, and ImageNet fine-tuning with cosine decay from 0.03 to 0 over 20,000 steps.', b)


F['p3_lr_schedules'] = lr_schedules()


# ---------------------------------------------------------------- 10. tiny ridge example
def ridge_tiny():
    if not FS:
        return svg(10, 10, 'pending', [])
    T = FS['tiny']
    X, W = T['X'], T['W']
    b = [text(20, 24, 'The few-shot classifier of the tiny example', 't-title')]
    L, Tp, s = 80, 50, 60
    PX = lambda v: L + (v + 0.5) * s
    PY = lambda v: Tp + (3.5 - v) * s
    for v in range(0, 4):
        b += [f'<line class="grid" x1="{PX(v):.1f}" y1="{Tp}" x2="{PX(v):.1f}" y2="{PY(-0.5):.1f}"/>', text(PX(v), PY(-0.5) + 16, str(v), 't-tick', 'middle'),
              f'<line class="grid" x1="{L}" y1="{PY(v):.1f}" x2="{PX(3.5):.1f}" y2="{PY(v):.1f}"/>', text(L - 8, PY(v) + 4, str(v), 't-tick', 'end')]
    # decision line: x . (W[:,0] - W[:,1]) = 0
    d0, d1 = W[0][0] - W[0][1], W[1][0] - W[1][1]
    k = -d0 / d1                                   # the line x2 = k * x1
    lo, hi = max(-0.5, -0.5 / k), min(3.5, 3.5 / k)
    pts = [(x, k * x) for x in (lo, hi)]
    b.append(f'<line class="key-line" x1="{PX(pts[0][0]):.1f}" y1="{PY(pts[0][1]):.1f}" x2="{PX(pts[1][0]):.1f}" y2="{PY(pts[1][1]):.1f}"/>')
    for i, (x, y) in enumerate(X):
        cls = 's1' if i < 2 else 's3'
        b.append(f'<circle class="{cls} ring" cx="{PX(x):.1f}" cy="{PY(y):.1f}" r="8"/>')
        b.append(text(PX(x) + 12, PY(y) - 8, ['A1', 'A2', 'B1', 'B2'][i], 't-tick'))
    nx, ny = T['new']
    b += [f'<circle class="s4 ring" cx="{PX(nx):.1f}" cy="{PY(ny):.1f}" r="7"/>', text(PX(nx) + 12, PY(ny) + 16, 'new', 't-tick')]
    b += lines(400, 70, ['features x = (x₁, x₂); class A (blue), class B (green)', f'W = (XᵀX + I)⁻¹ XᵀY',
                         f'  = [[{W[0][0]:.4f}, {W[0][1]:.4f}],', f'     [{W[1][0]:.4f}, {W[1][1]:.4f}]]',
                         '', 'score A − score B = 0 on the orange line;', 'class A below it, class B above it.',
                         f'new picture (1.5, 0.5): scores {T["score"][0]:.3f} and {T["score"][1]:.3f}', '  → class A'], 't-tick', 19)
    return svg(800, 330, 'Four training points in two dimensions, two of class A and two of class B, the line where the two class scores of the ridge solution are equal, and a new point that falls on the class A side.', b)


F['p3_ridge_tiny'] = ridge_tiny()


# ---------------------------------------------------------------- 11. few-shot scores of one picture
def fewshot_scores():
    if not FS:
        return svg(10, 10, 'pending', [])
    sc, names = FS['example_scores'], FS['per_class']['names']
    b = [text(20, 24, 'The 10 scores x W of one validation picture (a tench), 5-shot, frozen ViT-B/16', 't-title')]
    x0, mid, half = 130, 540, 230
    X = lambda v: mid + v / 1.2 * half
    for t in (-1, 0, 1):
        b += [f'<line class="{"axis" if t == 0 else "grid"}" x1="{X(t):.1f}" y1="44" x2="{X(t):.1f}" y2="{50 + 10 * 26:.1f}"/>',
              text(X(t), 64 + 10 * 26, f'{t:+d}' if t else '0', 't-tick', 'middle')]
    for i, (n, v) in enumerate(zip(names, sc)):
        y = 50 + i * 26
        cls = 's3' if v > 0 else 's2'
        x, w = (mid, X(v) - mid) if v > 0 else (X(v), mid - X(v))
        b.append(text(x0 + 100, y + 16, n, 't-tick t-strong' if v > 0 else 't-tick', 'end'))
        b.append(f'<g class="mark"><title>{esc(n)}: {v:+.2f}</title><rect class="{cls}" x="{x:.1f}" y="{y + 4}" width="{w:.1f}" height="16" rx="3"/></g>')
        b.append(text(X(v) + (6 if v > 0 else -6), y + 17, f'{v:+.2f}', 't-tick', 'start' if v > 0 else 'end'))
    b += lines(20, 340, ['The layer was fitted to targets of +1 for the right class and \u22121 for the others, and it nearly hits them on a new picture:',
                         'tench +0.92, every other class close to \u22121. The largest score names the class.'])
    return svg(800, 366, 'Horizontal bars of the ten class scores that the 5-shot ridge layer gives one tench picture: plus 0.92 for tench and about minus 1 for each other class.', b)


F['p3_fewshot_scores'] = fewshot_scores()


# ---------------------------------------------------------------- 12. fine-tuning curve
def ft_curve():
    if not FT:
        return svg(10, 10, 'pending', [])
    H = FT['hist']; C = FT['curve']; T = FT['steps']
    b = [text(20, 24, f'Fine-tuning ViT-B/16 on Imagenette: {T} steps of {FT["bs"]} pictures', 't-title')]
    x0, y0, w, h = 70, 56, 300, 200
    X = lambda t: x0 + t / T * w
    lmax = max(hh['loss'] for hh in H)
    Y = lambda v: y0 + h - v / lmax * h
    b.append(text(x0, y0 - 10, 'training loss per step (and learning rate, dashed)', 't-note'))
    for t in (0, lmax / 2, lmax):
        b += [f'<line class="grid" x1="{x0}" y1="{Y(t):.1f}" x2="{x0 + w}" y2="{Y(t):.1f}"/>', text(x0 - 6, Y(t) + 4, f'{t:.1f}', 't-tick', 'end')]
    b.append('<polyline class="l1" points="' + ' '.join(f'{X(hh["step"]):.1f},{Y(hh["loss"]):.1f}' for hh in H) + '"/>')
    LR = FT['lr']
    b.append('<polyline class="edge-dim" points="' + ' '.join(f'{X(hh["step"]):.1f},{y0 + h - hh["lr"] / LR * h:.1f}' for hh in H) + '"/>')
    for t in (0, T // 2, T):
        b.append(text(X(t), y0 + h + 16, str(t), 't-tick', 'middle'))
    b.append(text(x0 + w / 2, y0 + h + 34, 'step', 't-muted', 'middle'))
    # accuracy
    x1 = 460
    X2 = lambda t: x1 + t / T * w
    Y2 = lambda v: y0 + h - v * h
    b.append(text(x1, y0 - 10, 'validation accuracy (1,000 pictures)', 't-note'))
    for t in (0, 0.5, 1):
        b += [f'<line class="grid" x1="{x1}" y1="{Y2(t):.1f}" x2="{x1 + w}" y2="{Y2(t):.1f}"/>', text(x1 - 6, Y2(t) + 4, f'{100 * t:.0f}%', 't-tick', 'end')]
    b.append('<polyline class="l3" points="' + ' '.join(f'{X2(s):.1f},{Y2(a):.1f}' for s, a in C) + '"/>')
    for s, a in C:
        b.append(f'<g class="mark"><title>step {s}: {100 * a:.1f}%</title><circle class="s3 ring" cx="{X2(s):.1f}" cy="{Y2(a):.1f}" r="4.5"/></g>')
    s5, a5 = C[1]
    b.append(text(X2(s5) + 10, Y2(a5) + 22, f'{100 * a5:.1f}% at step {s5}', 't-tick'))
    b.append(text(X2(0) + 10, Y2(C[0][1]) + 4, f'{100 * C[0][1]:.1f}% at step 0', 't-tick'))
    b.append(text(X2(C[-1][0]) - 4, Y2(C[-1][1]) - 10, f'{100 * C[-1][1]:.1f}%', 't-tick', 'end'))
    for t in (0, T // 2, T):
        b.append(text(X2(t), y0 + h + 16, str(t), 't-tick', 'middle'))
    b.append(text(x1 + w / 2, y0 + h + 34, 'step', 't-muted', 'middle'))
    return svg(800, 310, 'Left: the training loss of each step of our fine-tuning run, falling from ln 10 towards zero, with the learning-rate schedule dashed. Right: validation accuracy on 1,000 Imagenette pictures at a few steps, rising from 10 percent to near 100 percent.', b)


F['p3_finetune_curve'] = ft_curve()

json.dump(F, open('results/figs_part3.json', 'w'))
print(f'wrote {len(F)} figures:', ', '.join(F))
