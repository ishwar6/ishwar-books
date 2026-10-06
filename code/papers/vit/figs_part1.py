"""Figures for Part 1 (the big idea), drawn from results/part1.json and results/part1_math.json.
Writes results/figs_part1.json (key -> svg string). Colours come only from CSS classes."""
import json, math
from figlib import svg, text, box, arrow, lines_chart, esc
from vitfig import token, row, line, hbars

R = json.load(open('results/part1.json'))
M = json.load(open('results/part1_math.json'))
F = {}
SER = ['s1', 's2', 's3', 's4']
BOX = ['box-1', 'box-2', 'box-3', 'box-4']


# ---------------------------------------------------------------- small local helpers (no colours, only classes)
def strip(x, y, n, cls='s1', cell=11, gap=2, vals=None, vertical=False):
    """A vector as a strip of small squares; opacity follows |value| when values are given."""
    b = [box(x - 3, y - 3, (cell + 6 if vertical else n * (cell + gap) - gap + 6), (n * (cell + gap) - gap + 6 if vertical else cell + 6), 'box-ghost', 4)]
    m = max(1e-9, max(abs(v) for v in vals)) if vals else 1
    for i in range(n):
        xx, yy = (x, y + i * (cell + gap)) if vertical else (x + i * (cell + gap), y)
        op = 0.2 + 0.75 * abs(vals[i]) / m if vals else 0.35 + 0.5 * ((i * 7) % 5) / 4
        b.append(f'<rect class="{cls}" x="{xx:.1f}" y="{yy:.1f}" width="{cell}" height="{cell}" rx="2" style="fill-opacity:{op:.2f}"/>')
    return b


def grid_cells(x, y, n, m, cell, cls_fn=None, op_fn=None, rx=2, gap=1):
    """An n-row by m-column grid of squares."""
    b = []
    for i in range(n):
        for j in range(m):
            c = cls_fn(i, j) if cls_fn else 's1'
            op = op_fn(i, j) if op_fn else 0.45
            b.append(f'<rect class="{c}" x="{x + j * cell + gap / 2:.1f}" y="{y + i * cell + gap / 2:.1f}" width="{cell - gap:.1f}" height="{cell - gap:.1f}" rx="{rx}" style="fill-opacity:{op:.2f}"/>')
    return b


def signed_map(x, y, Mx, cell, title, tcls='t-tick'):
    """A small heatmap of a signed matrix: positive = cell class, negative = s2 class, opacity = |value| / max."""
    n, m = len(Mx), len(Mx[0])
    mx = max(1e-9, max(abs(v) for r in Mx for v in r))
    b = [box(x - 2, y - 2, m * cell + 4, n * cell + 4, 'box-ghost', 3), text(x + m * cell / 2, y - 10, title, tcls, 'middle')]
    for i in range(n):
        for j in range(m):
            v = Mx[i][j]
            if abs(v) < 1e-9:
                continue
            b.append(f'<g class="mark"><title>row {i}, column {j}: {v:.0f}</title>'
                     f'<rect class="{"cell" if v > 0 else "s2"}" x="{x + j * cell + 0.5:.1f}" y="{y + i * cell + 0.5:.1f}" width="{cell - 1:.1f}" height="{cell - 1:.1f}" rx="1.5" style="fill-opacity:{0.25 + 0.7 * abs(v) / mx:.2f}"/></g>')
    return b


def brace(x1, x2, y, label, up=False, tcls='t-tick'):
    d = -8 if up else 8
    return [f'<path class="edge" d="M{x1:.1f},{y - d:.1f} L{x1:.1f},{y:.1f} L{x2:.1f},{y:.1f} L{x2:.1f},{y - d:.1f}"/>',
            text((x1 + x2) / 2, y + (-12 if up else 20), label, tcls, 'middle')]


def logbars(x0, y0, items, bar_w, row_h=30, label_w=250, title=None):
    """Horizontal bars on a log10 scale: items = [(label, value, note, cls)]."""
    b = []
    if title:
        b.append(text(x0, y0 - 12, title, 't-title'))
    lo, hi = 3, math.log10(max(v for _, v, _, _ in items)) + 0.2
    for i, (lab, v, note, cls) in enumerate(items):
        y = y0 + i * row_h
        w = max(3, bar_w * (math.log10(v) - lo) / (hi - lo))
        b.append(text(x0 + label_w - 8, y + row_h / 2 + 4, lab, 't-tick', 'end'))
        b.append(f'<g class="mark"><title>{esc(lab)}: {v:,} weights</title><rect class="{cls}" x="{x0 + label_w:.1f}" y="{y + 4:.1f}" width="{w:.1f}" height="{row_h - 10:.1f}" rx="3"/></g>')
        b.append(text(x0 + label_w + w + 6, y + row_h / 2 + 4, f'{v:,}' + (f'   ({note})' if note else ''), 't-val'))
    for p in range(lo, int(hi) + 1):
        xx = x0 + label_w + bar_w * (p - lo) / (hi - lo)
        b += [line(xx, y0 - 2, xx, y0 + len(items) * row_h, 'grid'), text(xx, y0 + len(items) * row_h + 16, f'10^{p}', 't-muted', 'middle')]
    return b


# ---------------------------------------------------------------- 1. timeline
def timeline():
    ev = [('2012', 'AlexNet', ['deep CNN wins the', 'ImageNet contest'], 's1'),
          ('2015', 'ResNet', ['CNNs 100+ layers', 'deep; still the', 'vision standard'], 's1'),
          ('2017', 'Transformer', ['attention only;', 'built for', 'translation'], 's2'),
          ('2018', 'BERT', ['pre-train, then', 'fine-tune: the', 'NLP recipe'], 's2'),
          ('2020', 'ViT', ['the Transformer,', 'unchanged, on', 'image patches'], 's3')]
    xs = [90, 240, 390, 540, 690]
    b = [line(50, 96, 730, 96, 'axis', 'ah')]
    for (yr, name, lines, cls), x in zip(ev, xs):
        b += [f'<circle class="{cls}" cx="{x}" cy="96" r="8"/>', text(x, 40, yr, 't-title', 'middle'), text(x, 64, name, 't-note', 'middle')]
        for k, l in enumerate(lines):
            b.append(text(x, 124 + 17 * k, l, 't-muted', 'middle'))
    b += [f'<rect class="s1" x="40" y="196" width="10" height="10" rx="2"/>', text(56, 205, 'computer vision', 't-tick'),
          f'<rect class="s2" x="190" y="196" width="10" height="10" rx="2"/>', text(206, 205, 'language (NLP)', 't-tick'),
          f'<rect class="s3" x="330" y="196" width="10" height="10" rx="2"/>', text(346, 205, 'this paper: the two lines meet', 't-tick')]
    return svg(760, 220, 'Timeline: AlexNet 2012 and ResNet 2015 made convolutional networks the standard for vision; the Transformer 2017 and BERT 2018 took over language; ViT 2020 applies the language model design to images.', b)


F['p1_timeline'] = timeline()


# ---------------------------------------------------------------- 2. the analogy of the title
def analogy():
    b = [text(190, 24, 'Language: words → tokens → vectors', 't-title', 'middle'),
         text(570, 24, 'Vision: patches → tokens → vectors', 't-title', 'middle'), line(380, 14, 380, 300, 'edge-dim')]
    words = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    b += [box(30, 44, 320, 28, 'box', 6), text(190, 63, '"The cat sat on the mat"', 't-tick', 'middle'), arrow(190, 76, 190, 100)]
    parts, cx = row(31, 104, words, 48, 6.4)
    b += parts
    for x in cx:
        b += [arrow(x, 140, x, 164)] + strip(x - 6, 168, 7, 's1', 11, 2, vertical=True)
    b += [text(190, 276, '6 tokens, each a vector of 768 numbers', 't-muted', 'middle')]
    # right: a 3 x 3 picture
    gx, gy, c = 525, 40, 20
    b += grid_cells(gx, gy, 3, 3, c, lambda i, j: SER[(i * 3 + j) % 4], lambda i, j: 0.35 + 0.08 * ((i * 3 + j) % 5))
    b += [text(gx + 30, gy + 76, 'a picture, cut into 3 × 3', 't-tick', 'middle'), arrow(gx + 30, gy + 82, gx + 30, gy + 100)]
    labels = [f'p{i + 1}' for i in range(9)]
    classes = [BOX[i % 4] for i in range(9)]
    parts, cx = row(410, 144, labels, 30, 6, classes=classes, h=28)
    b += parts
    for i, x in enumerate(cx):
        b += [arrow(x, 176, x, 192)] + strip(x - 6, 196, 5, SER[i % 4], 11, 2, vertical=True)
    b += [text(570, 276, '9 patch tokens, each a vector of 768 numbers', 't-muted', 'middle')]
    return svg(760, 292, 'The analogy of the title: a sentence becomes a sequence of word tokens and then vectors; a picture cut into patches becomes a sequence of patch tokens and then vectors, read by the same Transformer.', b)


F['p1_analogy'] = analogy()


# ---------------------------------------------------------------- 3. a 3 x 3 grid flattened into a row
def grid_flatten():
    gx, gy, c = 30, 50, 50
    b = [text(gx + 75, 30, 'an image, cut into a 3 × 3 grid', 't-note', 'middle')]
    b += grid_cells(gx, gy, 3, 3, c, lambda i, j: SER[(i * 3 + j) % 4], lambda i, j: 0.3 + 0.07 * ((i * 3 + j) % 5), rx=4, gap=3)
    for i in range(3):
        for j in range(3):
            b.append(text(gx + j * c + c / 2, gy + i * c + c / 2 + 5, str(i * 3 + j + 1), 't-note', 'middle'))
    b += [arrow(gx + 160, gy + 75, gx + 210, gy + 75), text(gx + 185, gy + 62, 'flatten', 't-muted', 'middle'), text(gx + 185, gy + 100, 'row by row', 't-muted', 'middle')]
    parts, cx = row(250, 100, [str(i + 1) for i in range(9)], 48, 6, classes=[BOX[i % 4] for i in range(9)], h=50, tcls='t-note')
    b += parts
    b += brace(250, 250 + 9 * 54 - 6, 162, 'a sequence of 9 patch tokens, in reading order (left to right, top to bottom)')
    b += [text(250, 44, 'patch 1 is the top-left corner, patch 9 the bottom-right; the order is fixed', 't-muted'),
          text(250, 62, 'a position embedding (Part 2) tells the model where each patch came from', 't-muted')]
    return svg(760, 200, 'An image cut into a 3 by 3 grid of patches, then flattened row by row into a sequence of 9 patch tokens, numbered 1 to 9 in reading order.', b)


F['p1_grid_flatten'] = grid_flatten()


# ---------------------------------------------------------------- 4. 224 x 224 -> 14 x 14 = 196 patches, with the arithmetic
def arithmetic():
    v = R['vit']
    gx, gy, c = 40, 60, 16
    b = grid_cells(gx, gy, 14, 14, c, lambda i, j: 's1', lambda i, j: 0.22 + 0.18 * (((i * 5 + j * 3) % 7) / 6), rx=1, gap=1.5)
    hi_i, hi_j = 4, 9
    b.append(box(gx + hi_j * c, gy + hi_i * c, c, c, 'box-on', 2))
    b += brace(gx, gx + 14 * c, gy - 14, '224 pixels = 14 patches × 16 pixels', up=True)
    b += brace(gx, gx + 14 * c, gy + 14 * c + 10, '14 × 14 = 196 patches')
    # zoomed patch
    zx, zy, zc = 330, 60, 7
    b += [line(gx + (hi_j + 1) * c, gy + hi_i * c, zx, zy, 'edge-dim'), line(gx + (hi_j + 1) * c, gy + (hi_i + 1) * c, zx, zy + 16 * zc, 'edge-dim')]
    b += grid_cells(zx, zy, 16, 16, zc, lambda i, j: 's3', lambda i, j: 0.25 + 0.5 * (((i * 3 + j * 5) % 11) / 10), rx=1, gap=1)
    b.append(box(zx - 1, zy - 1, 16 * zc + 2, 16 * zc + 2, 'box-on', 2))
    b += brace(zx, zx + 16 * zc, zy - 14, 'one patch: 16 × 16 pixels', up=True)
    b += [text(zx + 56, zy + 16 * zc + 24, '× 3 colour channels', 't-tick', 'middle')]
    # arithmetic
    tx = 480
    rows = [('the image', f'3 × 224 × 224 = {v["numbers"]:,} numbers'), ('one patch', f'16 × 16 × 3 = {v["per_patch"]} numbers'),
            ('patches', f'(224 / 16)² = 14 × 14 = {v["n_patches"]}'), ('tokens', f'{v["n_patches"]} + 1 [class] = {v["seq_len"]}'),
            ('token matrix', f'{v["seq_len"]} × {v["hidden"]} numbers')]
    b.append(text(tx, 60, 'the arithmetic of ViT-B/16', 't-title'))
    for k, (a, s) in enumerate(rows):
        y = 90 + 40 * k
        b += [text(tx, y, a, 't-label'), text(tx, y + 18, s, 't-val')]
    return svg(760, 316, 'A 224 by 224 image becomes a 14 by 14 grid of 16 by 16 patches: 196 patches of 768 numbers each, plus one class token, so 197 tokens. The arithmetic is printed on the right.', b)


F['p1_arithmetic'] = arithmetic()


# ---------------------------------------------------------------- 5. local filter versus global attention
def local_vs_global():
    b = [text(190, 24, 'CNN: a small filter slides over the image', 't-title', 'middle'),
         text(570, 24, 'ViT: every patch attends to every patch', 't-title', 'middle'), line(380, 14, 380, 310, 'edge-dim')]
    gx, gy, c = 100, 64, 26
    b += grid_cells(gx, gy, 7, 7, c, lambda i, j: 's1', lambda i, j: 0.18, rx=2, gap=2)
    for (i, j), cls in [((1, 1), 'box-ghost'), ((3, 4), 'box-on'), ((5, 1), 'box-ghost')]:
        b.append(box(gx + j * c - c, gy + i * c - c, 3 * c, 3 * c, cls, 3))
    b += [arrow(gx, gy - 12, gx + 7 * c, gy - 12), text(gx + 3.5 * c, gy - 18, 'the window visits every position', 't-muted', 'middle')]
    b += [text(190, 264, '3 × 3 window: each output sees 9 neighbours (locality)', 't-tick', 'middle'),
          text(190, 282, 'the same 9 weights at every position (weight sharing)', 't-tick', 'middle'),
          text(190, 300, 'far-away pixels meet only after many layers', 't-muted', 'middle')]
    gx2 = 480
    b += grid_cells(gx2, gy, 7, 7, c, lambda i, j: 's3', lambda i, j: 0.18, rx=2, gap=2)
    qi, qj = 2, 3
    qx, qy = gx2 + qj * c + c / 2, gy + qi * c + c / 2
    for i in range(7):
        for j in range(7):
            if (i, j) != (qi, qj):
                b.append(line(qx, qy, gx2 + j * c + c / 2, gy + i * c + c / 2, 'edge-dim'))
    b.append(box(gx2 + qj * c, gy + qi * c, c, c, 'box-on', 3))
    b += [text(570, 264, 'one patch looks at all 49 patches, in the first layer', 't-tick', 'middle'),
          text(570, 282, 'how much to look at each is learned (attention weights)', 't-tick', 'middle'),
          text(570, 300, 'no built-in idea that neighbours matter more', 't-muted', 'middle')]
    return svg(760, 314, 'Left: a convolution slides a 3 by 3 filter over the image, so each output depends on 9 neighbours and the same weights are used everywhere. Right: in ViT one patch attends to all 49 patches at once, with learned weights and no built-in preference for neighbours.', b)


F['p1_local_vs_global'] = local_vs_global()


# ---------------------------------------------------------------- 6. translation equivariance, with the real toy numbers
def equivariance():
    e = M['equiv']
    c = 12
    b = [text(380, 22, 'the same 3 × 3 edge filter, applied to a square and to the square moved 2 pixels right', 't-title', 'middle')]
    x = 30
    b += signed_map(x, 60, e['img'], c, 'input (12 × 12)')
    b += [arrow(x + 12 * c + 6, 60 + 6 * c, x + 12 * c + 34, 60 + 6 * c), text(x + 12 * c + 20, 60 + 6 * c - 10, 'conv', 't-muted', 'middle')]
    x2 = x + 12 * c + 40
    b += signed_map(x2, 60 + c, e['out'], c, 'output (10 × 10)')
    x3 = x2 + 10 * c + 44
    b += signed_map(x3, 60, e['shifted_img'], c, 'input shifted 2 px →')
    b += [arrow(x3 + 12 * c + 6, 60 + 6 * c, x3 + 12 * c + 34, 60 + 6 * c), text(x3 + 12 * c + 20, 60 + 6 * c - 10, 'conv', 't-muted', 'middle')]
    x4 = x3 + 12 * c + 40
    b += signed_map(x4, 60 + c, e['out_shift'], c, 'output shifted 2 px →')
    b += [f'<rect class="cell" x="30" y="232" width="10" height="10" rx="2" style="fill-opacity:0.9"/>', text(46, 241, 'positive response (dark-to-bright edge)', 't-tick'),
          f'<rect class="s2" x="330" y="232" width="10" height="10" rx="2" style="fill-opacity:0.9"/>', text(346, 241, 'negative response (bright-to-dark edge)', 't-tick'),
          text(30, 266, f'max |shifted output moved back by 2 - original output| = {e["max_diff"]:.1f}: the feature moved with the object, nothing else changed', 't-note')]
    return svg(760, 280, 'Translation equivariance made concrete: a vertical-edge filter on a bright square finds its two edges; on the same square moved two pixels to the right it finds the same two edges, moved two pixels to the right. The maximum difference after shifting back is exactly zero.', b)


F['p1_equivariance'] = equivariance()


# ---------------------------------------------------------------- 7. three prior approaches versus ViT
def prior():
    cols = [('1. CNN + attention', ['Non-local nets (2018)', 'DETR (2020)'], 'cnn_attn'),
            ('2. attention inside a CNN', ['Stand-alone (2019)', 'Axial-DeepLab (2020)'], 'mixed'),
            ('3. pure local attention', ['"specialized attention', 'patterns", hard on TPUs'], 'local'),
            ('ViT: a standard Transformer', ['global attention,', 'patches as tokens'], 'vit')]
    b = []
    for k, (title, notes, kind) in enumerate(cols):
        x = 20 + k * 185
        b.append(text(x + 85, 26, title, 't-title' if kind != 'vit' else 't-on', 'middle'))
        for n_, s in enumerate(notes):
            b.append(text(x + 85, 46 + 16 * n_, s, 't-muted', 'middle'))
        y0 = 90
        if kind == 'cnn_attn':
            for i in range(4):
                b += [box(x + 25, y0 + 130 - i * 30, 120, 24, 'box-1', 5), text(x + 85, y0 + 146 - i * 30, 'conv layer', 't-tick', 'middle')]
            b += [box(x + 25, y0 + 10, 120, 24, 'box-2', 5), text(x + 85, y0 + 26, 'attention on top', 't-tick', 'middle')]
        elif kind == 'mixed':
            kinds = ['conv layer', 'local attention', 'conv layer', 'local attention', 'conv layer']
            for i, s in enumerate(kinds):
                b += [box(x + 25, y0 + 130 - i * 30, 120, 24, 'box-1' if s == 'conv layer' else 'box-2', 5), text(x + 85, y0 + 146 - i * 30, s, 't-tick', 'middle')]
        elif kind == 'local':
            for i in range(5):
                b += [box(x + 25, y0 + 130 - i * 30, 120, 24, 'box-2', 5), text(x + 85, y0 + 146 - i * 30, 'local attention', 't-tick', 'middle')]
        else:
            for i in range(5):
                b += [box(x + 25, y0 + 130 - i * 30, 120, 24, 'box-on', 5), text(x + 85, y0 + 146 - i * 30, 'Transformer layer', 't-tick', 'middle')]
        b += [box(x + 25, y0 + 170, 120, 24, 'box', 5), text(x + 85, y0 + 186, 'pixels' if kind != 'vit' else '16 × 16 patches', 't-tick', 'middle'),
              arrow(x + 85, y0 + 168, x + 85, y0 + 158)]
        b.append(text(x + 85, y0 + 220, {'cnn_attn': 'the CNN still does the work', 'mixed': 'the CNN skeleton stays',
                                           'local': 'efficient in theory, slow in practice', 'vit': '"fewest possible modifications"'}[kind], 't-tick', 'middle'))
    return svg(760, 330, 'The three earlier ways of using attention in vision (attention added on top of a CNN, attention replacing some CNN layers, and pure local attention with special patterns) next to ViT, which is a standard Transformer with global attention over patch tokens.', b)


F['p1_prior'] = prior()


# ---------------------------------------------------------------- 8. the quadratic cost, log scale
def quadratic():
    rows = M['cost']['rows']
    items = [(r['name'].replace(' + [class]', ''), r['nn'], r['mem'], 's1' if 'pixels' in r['name'] else 's3') for r in rows]
    items.sort(key=lambda t: t[1])
    b = logbars(20, 50, items, 300, row_h=32, label_w=250, title='attention weights per head per layer = n × n (log scale)')
    b.append(text(20, 252, f'pixels of a 224 × 224 image: {M["cost"]["ratio_n"]} × more tokens than ViT-B/16, so {M["cost"]["ratio_cost"]:,} × more attention weights', 't-note'))
    return svg(760, 268, 'Attention weights per head per layer on a log scale: 2,500 for ViT-B/32, 38,809 for ViT-B/16, about one million for 32 by 32 CIFAR pixels and 2.5 billion (10 GB in float32) for the 50,176 pixels of a 224 by 224 image.', b)


F['p1_quadratic'] = quadratic()


# ---------------------------------------------------------------- 9. the two-step recipe
def recipe():
    b = [text(20, 26, 'Step 1: pre-train once on a very large labelled image set', 't-title'),
         text(430, 26, 'Step 2: transfer to each small benchmark', 't-title')]
    for i in range(4):
        b.append(box(30 + i * 6, 52 + i * 6, 140, 90, 'box', 6))
    b += [text(106, 92, 'ImageNet-21k: 14M images', 't-tick', 'middle'), text(106, 110, 'or JFT-300M: 303M images', 't-tick', 'middle'),
          text(106, 130, 'with class labels', 't-muted', 'middle'), text(106, 180, 'thousands of TPU-core-days', 't-tick', 'middle'),
          arrow(188, 100, 232, 100), box(236, 70, 130, 64, 'box-on', 12), text(301, 98, 'ViT', 't-big', 'middle'), text(301, 118, 'learns to see', 't-tick', 'middle'),
          text(301, 160, 'all the weights', 't-tick', 'middle'), text(301, 178, 'are learned here', 't-tick', 'middle')]
    tasks = [('ImageNet', '1,000 classes, 1.3M images', 'box-1'), ('CIFAR-100', '100 classes, 50k tiny images', 'box-2'), ('VTAB', '19 tasks, 1,000 images each', 'box-3')]
    for i, (name, out, cls) in enumerate(tasks):
        y = 48 + i * 74
        b += [arrow(368, 102, 444, y + 26), box(448, y, 110, 52, 'box-on', 10), text(503, y + 23, 'ViT copy', 't-note', 'middle'),
              text(503, y + 41, 'starts pre-trained', 't-muted', 'middle'),
              box(558, y + 8, 26, 36, cls, 6), arrow(586, y + 26, 612, y + 26),
              text(618, y + 21, name, 't-note'), text(618, y + 39, out, 't-tick')]
    b.append(text(570, 278, '+ one new classification layer per benchmark (fine-tuning, Part 3)', 't-tick', 'middle'))
    return svg(810, 292, 'The recipe of the paper: pre-train ViT once on a very large labelled image set (ImageNet-21k or JFT-300M), then make a copy for each small benchmark and fine-tune it with one new output layer.', b)


F['p1_recipe'] = recipe()


# ---------------------------------------------------------------- 10. a sketch of data size versus accuracy (no numbers, a sketch)
def data_sketch():
    L, T, W, H = 70, 40, 520, 190
    b = [line(L, T + H, L + W, T + H, 'axis', 'ah'), line(L, T + H, L, T, 'axis', 'ah'),
         text(L + W / 2, T + H + 30, 'amount of pre-training data (more →)', 't-tick', 'middle'),
         text(L - 10, T - 12, 'accuracy on the target task', 't-tick')]
    res = [(0.0, 0.62), (0.25, 0.72), (0.5, 0.78), (0.75, 0.81), (1.0, 0.83)]
    vit = [(0.0, 0.45), (0.25, 0.62), (0.5, 0.76), (0.75, 0.86), (1.0, 0.92)]
    P = lambda pts: ' '.join(f'{L + x * W:.1f},{T + H - y * H:.1f}' for x, y in pts)
    b += [f'<polyline class="l1" points="{P(res)}"/>', f'<polyline class="l3" points="{P(vit)}"/>']
    for x, y in res:
        b.append(f'<circle class="s1 ring" cx="{L + x * W:.1f}" cy="{T + H - y * H:.1f}" r="4.5"/>')
    for x, y in vit:
        b.append(f'<circle class="s3 ring" cx="{L + x * W:.1f}" cy="{T + H - y * H:.1f}" r="4.5"/>')
    b += [text(L + W + 12, T + H - 0.83 * H + 4, 'ResNet (CNN)', 't-s1'), text(L + W + 12, T + H - 0.92 * H + 4, 'ViT', 't-s3'),
          text(L + 0.1 * W, T + H - 0.40 * H, 'small data: the CNN\'s built-in', 't-muted'), text(L + 0.1 * W, T + H - 0.40 * H + 16, 'assumptions help; ViT lags', 't-muted'),
          text(L + 0.62 * W, T + H - 0.62 * H + 44, 'large data: ViT learns the', 't-muted'), text(L + 0.62 * W, T + H - 0.62 * H + 60, 'assumptions, and more, from data', 't-muted'),
          text(L + 0.5 * W, T + 12, 'a sketch of the paper\'s claim, not measured data (the real numbers follow)', 't-muted', 'middle')]
    return svg(760, 280, 'A sketch of the paper\'s claim: with little pre-training data a convolutional network beats ViT because its built-in assumptions help; with a lot of data ViT overtakes it. This is a sketch of the idea, not measured data.', b)


F['p1_data_sketch'] = data_sketch()


# ---------------------------------------------------------------- 11. the real Table 5 ImageNet numbers for ViT-B/16 and ViT-L/16
F['p1_table5'] = lines_chart(760, 300, 'ImageNet top-1 accuracy of ViT-B/16 and ViT-L/16 after pre-training on ImageNet (1.3 million images), ImageNet-21k (14 million) and JFT-300M (303 million), from Table 5 of the paper. The larger model is worse with the smallest dataset and best with the largest.',
                             [1.3, 14, 303], [('ViT-B/16', [77.91, 83.97, 84.15], 'l1', 's1'), ('ViT-L/16', [76.53, 85.15, 87.12], 'l3', 's3')],
                             'pre-training images (millions, log scale): ImageNet 1.3M, ImageNet-21k 14M, JFT-300M 303M', 'ImageNet top-1 accuracy (%)', 74, 89,
                             [74, 77, 80, 83, 86, 89], xlog=True, fmt=lambda v: f'{v:.2f}%' if isinstance(v, float) and v != int(v) else f'{v}%')


# ---------------------------------------------------------------- 12. the four headline results
def headline():
    items = [('ImageNet', 88.55), ('ImageNet-ReaL', 90.72), ('CIFAR-100', 94.55), ('VTAB (19 tasks)', 77.63)]
    b = hbars(20, 50, items, 420, row_h=34, cls='s3', label_w=150, fmt=lambda v: f'{v:.2f}%', vmax=100, title='top-1 accuracy of the best model (ViT-H/14 pre-trained on JFT-300M)')
    b.append(text(20, 206, 'ImageNet-ReaL: ImageNet with cleaner labels. VTAB: the average over 19 small tasks with 1,000 training images each.', 't-muted'))
    return svg(760, 222, 'The four headline numbers of the introduction: 88.55 percent on ImageNet, 90.72 percent on ImageNet-ReaL, 94.55 percent on CIFAR-100 and 77.63 percent on the 19-task VTAB suite.', b)


F['p1_headline'] = headline()


# ---------------------------------------------------------------- 13. cosine similarity versus shift, from our run
def shift_chart():
    rows = M['shift']['rows']
    xs = [r['k'] for r in rows]
    cos = [r['mean_cos'] for r in rows]
    pv = [r['vit'][1] for r in rows]
    pr = [r['resnet'][1] for r in rows]
    return lines_chart(760, 320, 'Shifting the cat picture right by 1, 4, 8, 12 and 16 pixels: the mean cosine similarity between the original and shifted ViT-B/16 patch vectors drops to 0.457 at 12 pixels and returns to 1.000 at 16 pixels, a whole patch; meanwhile both ViT-B/16 and ResNet-50 keep their top-1 class, with probabilities that move only a little.',
                       xs, [('patch cosine, ViT', cos, 'l3', 's3'), ('P(Egyptian cat), ViT', pv, 'l1', 's1'), ('P(tiger cat), ResNet', pr, 'l2', 's2')],
                       'shift of the picture to the right, in pixels', 'cosine similarity / probability', 0.4, 1.0, [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                       fmt=lambda v: f'{v:.3f}', right=240)


F['p1_shift'] = shift_chart()


# ---------------------------------------------------------------- 14. ViT versus ResNet top-5 on the sample picture
def top5():
    v, r = R['vit']['top5'], R['resnet']['top5']
    b = [text(380, 24, 'the same picture (two tabby cats on a pink sofa) through two models', 't-title', 'middle')]
    b += hbars(20, 66, [(n, p) for n, p in v], 190, row_h=26, cls='s3', label_w=110, fmt=lambda p: f'{p:.4f}', vmax=1.0, title='ViT-B/16 (Transformer)', hi=v[0][0])
    b += hbars(400, 66, [(n, p) for n, p in r], 190, row_h=26, cls='s1', label_w=110, fmt=lambda p: f'{p:.4f}', vmax=1.0, title='ResNet-50 (CNN)', hi=r[0][0])
    b.append(text(380, 220, 'both say "cat" with about 94% confidence; they disagree about the breed (the picture has two cats and a remote control)', 't-muted', 'middle'))
    return svg(760, 236, 'Top-5 ImageNet classes for the sample picture: ViT-B/16 says Egyptian cat 0.937, tabby, tiger cat, lynx, Siamese cat; ResNet-50 says tiger cat 0.942, tabby, remote control, Egyptian cat, jinrikisha.', b)


F['p1_top5'] = top5()


# ---------------------------------------------------------------- 15. what ViT does and does not do in this paper
def scope():
    cols = [('image classification', 'one label for the whole picture', True, 'cls'),
            ('object detection', 'a box around every object', False, 'det'),
            ('segmentation', 'a class for every pixel', False, 'seg')]
    b = []
    for k, (title, note, yes, kind) in enumerate(cols):
        x = 30 + k * 245
        b += [box(x, 20, 210, 200, 'box-on' if yes else 'box-ghost', 12), text(x + 105, 46, title, 't-title', 'middle'), text(x + 105, 66, note, 't-muted', 'middle')]
        gx, gy, c = x + 45, 84, 10
        b += grid_cells(gx, gy, 8, 12, c, lambda i, j: 's1', lambda i, j: 0.15, rx=1, gap=1)
        if kind == 'cls':
            b += [box(gx + 30, gy + 92, 60, 22, 'box-3', 5), text(gx + 60, gy + 107, '"cat"', 't-tick', 'middle')]
        elif kind == 'det':
            b += [box(gx + 10, gy + 15, 45, 50, 'box-2', 3), box(gx + 65, gy + 25, 50, 45, 'box-4', 3), text(gx + 60, gy + 107, 'cat, remote', 't-tick', 'middle')]
        else:
            b += grid_cells(gx + 10, gy + 10, 5, 4, c, lambda i, j: 's2', lambda i, j: 0.7, rx=1, gap=1)
            b += grid_cells(gx + 70, gy + 20, 4, 5, c, lambda i, j: 's4', lambda i, j: 0.7, rx=1, gap=1)
            b.append(text(gx + 60, gy + 107, 'every pixel labelled', 't-tick', 'middle'))
        b.append(text(x + 105, 208, 'this paper' if yes else 'left to later work (Part 6)', 't-on' if yes else 't-muted', 'middle'))
    return svg(760, 236, 'What the paper does and does not do: it trains ViT for image classification (one label per picture); object detection and segmentation are left to later work.', b)


F['p1_scope'] = scope()


# ---------------------------------------------------------------- 16. one patch becomes one vector
def patch_vector():
    b = [text(20, 24, 'one patch, 16 × 16 pixels × 3 channels', 't-title')]
    px, py, c = 30, 44, 6
    for ch, cls in enumerate(['s1', 's2', 's3']):
        b += grid_cells(px + ch * 10, py + ch * 10, 16, 16, c, lambda i, j, cls=cls: cls, lambda i, j: 0.25 + 0.5 * (((i * 3 + j * 5) % 11) / 10), rx=1, gap=1)
    b += [text(px + 58, py + 140, 'red, green, blue', 't-muted', 'middle'), arrow(px + 128, py + 60, px + 160, py + 60), text(px + 144, py + 48, 'flatten', 't-muted', 'middle')]
    sx = 220
    b += strip(sx, py + 54, 32, 's1', 9, 2) + [text(sx + 32 * 11 + 6, py + 62, '…', 't-note')]
    b += brace(sx, sx + 32 * 11 - 2 + 20, py + 76, '768 numbers in a row (16 × 16 × 3)')
    b += [arrow(sx + 175, py + 110, sx + 175, py + 136), text(sx + 185, py + 128, 'multiply by a learned matrix E (768 × 768): the "linear embedding"', 't-tick')]
    b += strip(sx, py + 142, 32, 's3', 9, 2) + [text(sx + 32 * 11 + 6, py + 150, '…', 't-note')]
    b += brace(sx, sx + 32 * 11 - 2 + 20, py + 164, 'one patch token: a vector of D = 768 numbers, the same shape as a word vector in BERT')
    return svg(760, 250, 'One 16 by 16 patch with 3 colour channels is flattened into 768 numbers and multiplied by a learned matrix E to give a 768-number patch token, the same shape as a word vector in BERT.', b)


F['p1_patch_vector'] = patch_vector()


# ---------------------------------------------------------------- 17. parameter counts of Transformers in NLP, log scale
def param_growth():
    items = [('Transformer "big", 2017', 213_000_000, '213M', 's2'), ('BERT-large, 2018', 340_000_000, '340M', 's2'),
             ('GPT-3, 2020', 175_000_000_000, '175B', 's2'), ('GShard, 2020', 600_000_000_000, '600B', 's2')]
    b = [text(20, 28, 'parameters of NLP Transformers named in the first paragraph (log scale)', 't-title')]
    lo, hi = 8, 12
    for i, (lab, v, note, cls) in enumerate(items):
        y = 50 + i * 34
        w = 420 * (math.log10(v) - lo) / (hi - lo)
        b += [text(230, y + 20, lab, 't-tick', 'end'),
              f'<g class="mark"><title>{esc(lab)}: {v:,} parameters</title><rect class="{cls}" x="240" y="{y + 4}" width="{w:.1f}" height="24" rx="3"/></g>',
              text(246 + w, y + 20, note, 't-val')]
    for p in range(lo, hi + 1):
        xx = 240 + 420 * (p - lo) / (hi - lo)
        b += [line(xx, 46, xx, 50 + 4 * 34, 'grid'), text(xx, 50 + 4 * 34 + 16, {8: '100M', 9: '1B', 10: '10B', 11: '100B', 12: '1T'}[p], 't-muted', 'middle')]
    b.append(text(20, 228, 'ViT-B/16 has 86M parameters, ViT-L/16 307M, ViT-H/14 632M: the sizes of BERT, not of GPT-3', 't-muted'))
    return svg(760, 244, 'Parameter counts of the language Transformers the introduction points to, on a log scale: 213 million for the 2017 Transformer, 340 million for BERT-large, 175 billion for GPT-3 and 600 billion for GShard.', b)


F['p1_params'] = param_growth()

json.dump(F, open('results/figs_part1.json', 'w'))
print('figures:', len(F), ', '.join(F))
