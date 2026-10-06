"""Figures for Part 2 (the model: related work, Figure 1, patches, embeddings, the encoder, Appendix A), drawn from
results/part2_math.json and results/part2_attn.json. Writes results/figs_part2.json (key -> svg string)."""
import json, sys, math
import vitfig
sys.modules.setdefault('bertfig', vitfig)      # alammar.py imports `line` from bertfig; vitfig has the same helper
from figlib import svg, text, box, arrow, heatmap
from vitfig import token, row, line, hbars
from alammar import vec, vec_len, matrix, attn_grid, frame, brace

R = json.load(open('results/part2_math.json'))
T = json.load(open('results/part2_attn.json'))
F = {}
C = R['config']


def lines(x, y, ls, cls='t-tick', dy=15, anchor='start'):
    return [text(x, y + i * dy, s, cls, anchor) for i, s in enumerate(ls)]


def table(x0, y0, M, rows, cols, cw=58, ch=24, label_w=120, fmt='{:.3f}', cls='cell', title=None, signed=True):
    """A small table of numbers: cell shade = |value| / max, value printed in the cell."""
    b = []
    flat = [abs(v) for r in M for v in r]
    m = max(1e-9, max(flat))
    if title:
        b.append(text(x0, y0 - 10, title, 't-title'))
    for j, c in enumerate(cols):
        b.append(text(x0 + label_w + j * cw + cw / 2, y0 - 6 + (0 if title is None else 16), c, 't-tick', 'middle'))
    for i, r in enumerate(rows):
        y = y0 + i * ch + (0 if title is None else 20)
        b.append(text(x0 + label_w - 8, y + ch / 2 + 4, r, 't-tick', 'end'))
        for j, v in enumerate(M[i]):
            x = x0 + label_w + j * cw
            op = 0.08 + 0.6 * abs(v) / m
            c = cls if (v >= 0 or not signed) else 's2'
            b.append(f'<rect class="{c}" x="{x + 1:.1f}" y="{y + 1:.1f}" width="{cw - 2}" height="{ch - 2}" rx="3" style="fill-opacity:{op:.2f}"/>')
            b.append(text(x + cw / 2, y + ch / 2 + 4, fmt.format(v), 't-cell', 'middle'))
    return b


def grid_cells(x, y, n, cell, vals=None, cls='s1', gap=1):
    """An n x n grid of small squares; vals (n x n, 0..1) sets the opacity."""
    b = []
    for i in range(n):
        for j in range(n):
            v = vals[i][j] if vals else 0.35 + 0.3 * (((i * 5 + j * 3) % 7) / 6)
            b.append(f'<rect class="{cls}" x="{x + j * cell:.1f}" y="{y + i * cell:.1f}" width="{cell - gap}" height="{cell - gap}" rx="1.5" style="fill-opacity:{max(0.06, min(1, v)):.2f}"/>')
    return b


# ---------------------------------------------------------------- 1. map of the related work
def related_map():
    b = [box(150, 14, 460, 40, 'box', 10), text(380, 39, 'Full self-attention over pixels: cost grows with (number of pixels)²', 't-note', 'middle')]
    branches = [
        ('Local attention', ['Parmar 2018 (Image', 'Transformer): each', 'pixel attends to a', 'neighbourhood.', 'Hu, Ramachandran,', 'Zhao: replace convs']),
        ('Sparse and axial', ['Child 2019 (Sparse', 'Transformers);', 'Weissenborn 2019', '(blocks); Ho 2019,', 'Wang 2020a: one', 'axis at a time']),
        ('Small patches', ['Cordonnier 2020:', '2×2 patches, full', 'attention on top.', 'Small images only.', 'Closest to ViT.', '']),
        ('CNN + attention', ['Bello 2019; Hu 2018;', 'Carion 2020 (DETR);', 'Wang 2018; Sun 2019;', 'Wu 2020; Locatello', '2020; UNITER,', 'ViLBERT, VisualBERT']),
        ('Pixels as tokens', ['iGPT (Chen 2020a):', 'generative model', 'on down-scaled', 'pixels; linear', 'probe reaches 72%', 'on ImageNet']),
    ]
    for i, (t, ls) in enumerate(branches):
        x = 4 + i * 152
        cls = 'box-on' if i == 2 else 'box'
        b += [box(x, 90, 144, 128, cls, 10), text(x + 72, 110, t, 't-note', 'middle')] + lines(x + 8, 130, ls)
        b.append(arrow(380, 54, x + 72, 88))
    b += [box(100, 252, 560, 50, 'box-1', 10),
          text(380, 272, 'ViT: 16×16 patches, full (global) attention, the standard encoder unchanged,', 't-note', 'middle'),
          text(380, 290, 'and pre-training on 14M to 300M images', 't-note', 'middle'),
          arrow(380, 218, 380, 250, on=True),
          text(380, 326, 'the large-data line it joins: Mahajan 2018, Touvron 2019, Xie 2020, Sun 2017, Kolesnikov 2020 (BiT), Djolonga 2020', 't-muted', 'middle')]
    return svg(760, 340, 'Map of the related work in Section 2: five earlier ways of putting attention on images (local attention, sparse and axial attention, small 2 by 2 patches, CNN plus attention, iGPT on pixels) and ViT below them, which uses 16 by 16 patches, full attention and the standard encoder with large-scale pre-training.', b)


F['p2_related_map'] = related_map()


# ---------------------------------------------------------------- 2. Figure 1 redrawn as nine frames
def figure1_frames():
    b = []
    W, Hf, G = 244, 128, 8
    top5 = R['eq4']['top5'][0]

    def at(i):
        return 6 + (i % 3) * (W + G), 10 + (i // 3) * (Hf + 10)

    # 1 picture
    x, y = at(0)
    b += frame(x, y, W, Hf, 1, 'The picture')
    b += grid_cells(x + 24, y + 40, 3, 24, cls='s1', gap=1)
    b += lines(x + 110, y + 55, ['224 × 224 pixels,', '3 colour channels', 'x: H × W × C'], 't-tick', 16)
    # 2 patches
    x, y = at(1)
    b += frame(x, y, W, Hf, 2, 'Cut into patches')
    for i in range(3):
        for j in range(3):
            b.append(f'<rect class="s1" x="{x + 24 + j * 27}" y="{y + 40 + i * 27}" width="22" height="22" rx="2" style="fill-opacity:{0.35 + 0.3 * ((i * 5 + j * 3) % 7) / 6:.2f}"/>')
    b += lines(x + 115, y + 55, ['16 × 16 × 3 each', '224/16 = 14 per side', 'N = 14 × 14 = 196'], 't-tick', 16)
    # 3 flatten
    x, y = at(2)
    b += frame(x, y, W, Hf, 3, 'Flatten each patch')
    for i in range(3):
        b += vec(x + 24, y + 44 + i * 22, R['patch0_first48'][i * 12:(i + 1) * 12], 's1', cell=11, gap=2, outline=True)
    b += lines(x + 190, y + 55, ['768', 'numbers', 'per patch'], 't-tick', 16)
    # 4 linear projection
    x, y = at(3)
    b += frame(x, y, W, Hf, 4, 'Linear projection E')
    b += vec(x + 14, y + 70, 10, 's1', cell=9, gap=2)
    b.append(arrow(x + 128, y + 74, x + 142, y + 74))
    b += [box(x + 144, y + 56, 54, 36, 'box-4', 6), text(x + 171, y + 72, 'E', 't-math', 'middle'), text(x + 171, y + 86, '768×768', 't-cell', 'middle')]
    b.append(arrow(x + 200, y + 74, x + 214, y + 74))
    b += vec(x + 216, y + 70, 2, 's4', cell=9, gap=2, outline=False)
    b += [text(x + 14, y + 110, 'x_p E: 196 patch embeddings of 768', 't-tick')]
    # 5 class token
    x, y = at(4)
    b += frame(x, y, W, Hf, 5, 'Prepend [class]')
    parts, cx = row(x + 14, y + 52, ['[class]', 'p1', 'p2', 'p3', '…', 'p196'], 34, 4, classes=['box-2'] + ['box-1'] * 5, h=28, tcls='t-cell')
    b += parts + [text(x + 14, y + 104, '196 + 1 = 197 tokens; x_class is learned', 't-tick')]
    # 6 positions
    x, y = at(5)
    b += frame(x, y, W, Hf, 6, 'Add position embeddings')
    parts, cx = row(x + 14, y + 44, ['[class]', 'p1', 'p2', 'p3', '…', 'p196'], 34, 4, classes=['box-2'] + ['box-1'] * 5, h=24, tcls='t-cell')
    b += parts
    parts, cx = row(x + 14, y + 80, ['pos 0', 'pos 1', 'pos 2', 'pos 3', '…', 'pos 196'], 34, 4, cls='box-3', h=24, tcls='t-cell')
    b += parts + [text(x + 7, y + 73, '+', 't-note', 'middle'), text(x + 14, y + 120, 'E_pos: 197 learned rows (Eq. 1)', 't-tick')]
    # 7 encoder
    x, y = at(6)
    b += frame(x, y, W, Hf, 7, 'Transformer encoder')
    b += [box(x + 14, y + 40, 216, 50, 'box-1', 8), text(x + 122, y + 60, 'LN → MSA → + → LN → MLP → +', 't-tick', 'middle'),
          text(x + 122, y + 80, 'repeated L = 12 times (Eq. 2, 3)', 't-tick', 'middle'), text(x + 14, y + 112, '197 × 768 in, 197 × 768 out', 't-tick')]
    # 8 class token out
    x, y = at(7)
    b += frame(x, y, W, Hf, 8, 'Read the [class] output')
    parts, cx = row(x + 14, y + 52, ['z_L⁰', 'z_L¹', 'z_L²', '…', 'z_L¹⁹⁶'], 40, 4, classes=['box-on'] + ['box-ghost'] * 4, h=28, tcls='t-cell')
    b += parts + [text(x + 14, y + 104, 'y = LN(z_L⁰): 768 numbers (Eq. 4)', 't-tick')]
    # 9 head
    x, y = at(8)
    b += frame(x, y, W, Hf, 9, 'Classification head')
    b += [box(x + 14, y + 44, 110, 34, 'box-4', 6), text(x + 69, y + 65, 'Linear 768→1000', 't-cell', 'middle'),
          arrow(x + 126, y + 61, x + 140, y + 61), text(x + 144, y + 58, top5['label'], 't-note'), text(x + 144, y + 74, f'{top5["prob"]:.3f} (softmax)', 't-tick'),
          text(x + 14, y + 112, 'one linear layer when fine-tuned', 't-tick')]
    return svg(760, 426, 'Figure 1 of the paper redrawn as nine numbered frames: the picture, cutting into 16 by 16 patches, flattening, the linear projection E, prepending the class token, adding position embeddings, the 12-layer encoder, reading the class token output, and the classification head that answers Egyptian cat.', b)


F['p2_figure1_frames'] = figure1_frames()


# ---------------------------------------------------------------- 3. the reshape x -> x_p
def reshape():
    b = [text(20, 28, 'x ∈ R^(H×W×C): 224 × 224 × 3', 't-title')]
    cell = 11
    b += grid_cells(40, 50, 14, cell, cls='s1')
    b += brace(40, 40 + 14 * cell, 50 + 14 * cell + 6, '224 px = 14 patches × 16 px', 'edge')
    b += [text(30, 50 + 7 * cell + 4, '14', 't-tick', 'end')]
    b += [arrow(215, 125, 300, 125), text(257, 112, 'reshape', 't-note', 'middle')]
    b += lines(230, 160, ['N = HW / P²', '= 224 · 224 / 16²', '= 50,176 / 256', '= 196 patches'], 't-tick', 16)
    parts, Wm, Hm = matrix(330, 50, 14, 22, 's1', cell=9, gap=1.6, label='x_p', shape='196 rows × 768 columns  (N × P²·C)')
    b += parts
    xr = 330 + Wm + 14
    b += [text(xr, 60, 'row 0 = top-left patch', 't-tick'), text(xr, 50 + Hm, 'row 195 = bottom-right', 't-tick'),
          text(xr, 120, 'each row: the 768', 't-tick'), text(xr, 136, 'pixel numbers of', 't-tick'), text(xr, 152, 'one patch (16·16·3)', 't-tick'),
          text(20, 258, 'The grid is read row by row: patch n sits at row ⌊n/14⌋, column n mod 14.', 't-muted'),
          text(20, 275, 'The order is fixed, so position 1 is always the top-left corner.', 't-muted')]
    return svg(760, 288, 'The reshape of Section 3.1: the 224 by 224 by 3 picture, drawn as a 14 by 14 grid of 16-pixel patches, becomes the matrix x_p with 196 rows and 768 columns; N equals 224 times 224 over 16 squared equals 196.', b)


F['p2_reshape'] = reshape()


# ---------------------------------------------------------------- 4. one patch flattened to a strip of 768
def patch_flatten():
    b = [text(20, 28, 'One 16 × 16 × 3 patch becomes one row of 768 numbers', 't-title')]
    for i, (cls, lab) in enumerate([('s1', 'blue 16×16'), ('s3', 'green 16×16'), ('s2', 'red 16×16')]):
        off = (2 - i) * 12
        b.append(f'<rect class="{cls}" x="{30 + off}" y="{60 + off}" width="80" height="80" rx="4" style="fill-opacity:0.55"/>')
    b += [text(80, 170, 'patch 0 (top-left)', 't-tick', 'middle'), text(80, 186, 'C = 3 planes of 16×16', 't-tick', 'middle')]
    b.append(arrow(150, 100, 190, 100))
    vals = R['patch0_first48'][:40]
    b += vec(200, 94, vals, 's2', cell=11, gap=2, outline=True, title='x_p[0]')
    b += [text(200, 80, 'first 40 of the 768 numbers (red channel: pixel row 0, start of row 1):', 't-tick'),
          text(200, 128, ', '.join(f'{v:.3f}' for v in vals[:8]) + ', …', 't-code')]
    # schematic full strip
    y = 170
    segs = [('s2', '256 red: 16 rows × 16 pixels'), ('s3', '256 green'), ('s1', '256 blue')]
    for i, (cls, lab) in enumerate(segs):
        x = 200 + i * 180
        b.append(f'<rect class="{cls}" x="{x}" y="{y}" width="176" height="18" rx="3" style="fill-opacity:0.5"/>')
        b += brace(x, x + 176, y + 24, lab, 'edge')
    b += [text(200, y - 8, 'all 768 numbers, in the order (channel, row, column):', 't-tick'),
          text(20, 236, f'number 0 = red at pixel (0,0) = {R["patch0_rgb"][0]:.3f}; number 256 = green at (0,0) = {R["patch0_rgb"][1]:.3f}; number 512 = blue at (0,0) = {R["patch0_rgb"][2]:.3f}', 't-muted')]
    return svg(760, 250, 'One 16 by 16 by 3 patch flattened into a strip of 768 numbers: the 256 red values first, then 256 green, then 256 blue, with the real first 40 values of patch 0 drawn as shaded squares.', b)


F['p2_patch_flatten'] = patch_flatten()


# ---------------------------------------------------------------- 5. the matrix E
def e_matrix():
    b = [text(20, 28, 'The patch embedding is one matrix multiplication', 't-title')]
    p1, W1, H1 = matrix(40, 70, 14, 14, 's1', cell=9, gap=1.6, label='x_p', shape='196 × 768')
    b += p1 + [text(40 + W1 + 22, 70 + H1 / 2 + 6, '·', 't-big', 'middle')]
    p2, W2, H2 = matrix(40 + W1 + 44, 70, 14, 14, 's4', cell=9, gap=1.6, label='E', shape='768 × 768')
    b += p2 + [text(40 + W1 + 44 + W2 + 22, 70 + H2 / 2 + 6, '=', 't-big', 'middle')]
    p3, W3, H3 = matrix(40 + W1 + 44 + W2 + 44, 70, 14, 14, 's3', cell=9, gap=1.6, label='x_p E', shape='196 × 768')
    b += p3
    x = 40 + W1 + 44 + W2 + 44 + W3 + 30
    b += lines(x, 80, ['Each row of x_p (one', 'patch, 768 numbers)', 'times E gives one row', 'of 768 embedding', 'numbers.', '',
                       f'E: 768 × 768 = {768 * 768:,}', 'weights + 768 biases.'], 't-tick', 16)
    b += [text(20, 262, 'The embedding row for patch 0 starts ' + ', '.join(f'{v:.3f}' for v in R['patch0_emb_first6'][:4]) + ' …  (768 numbers in all).', 't-muted'),
          text(20, 279, 'D = 768 equals P²·C = 768 only by coincidence of ViT-B/16 at 224: for ViT-B/32 the rows are 32·32·3 = 3,072 long', 't-muted'),
          text(20, 296, 'and E is 3,072 × 768.', 't-muted')]
    return svg(760, 308, 'The patch embedding as a matrix product: x_p with shape 196 by 768 times E with shape 768 by 768 gives the patch embeddings with shape 196 by 768; E has 589,824 weights plus 768 biases.', b)


F['p2_E_matrix'] = e_matrix()


# ---------------------------------------------------------------- 6. conv equals linear
def conv_linear():
    b = []
    b += frame(6, 10, 366, 230, 1, 'Library: Conv2d, kernel 16, stride 16')
    b += grid_cells(30, 54, 7, 18, cls='s1')
    b.append(box(30 + 2 * 18, 54 + 2 * 18, 17, 17, 'box-on', 2))
    b += [text(160, 66, 'a 16×16×3 filter', 't-tick'), text(160, 82, 'placed on each patch', 't-tick'), text(160, 98, '(stride 16: no overlap)', 't-tick'),
          text(160, 124, '768 filters →', 't-tick'), text(160, 140, '768 numbers per patch', 't-tick'),
          text(30, 206, 'weight shape (768, 3, 16, 16)', 't-code'), text(30, 226, 'output (768, 14, 14) → (196, 768)', 't-code')]
    b += frame(388, 10, 366, 230, 2, 'Eq. 1: a Linear layer on x_p')
    b += vec(410, 70, 12, 's2', cell=9, gap=2)
    b += [text(410, 98, 'x_p row: 768 numbers', 't-tick'), arrow(546, 74, 566, 74)]
    b += [box(568, 56, 60, 36, 'box-4', 6), text(598, 72, 'E', 't-math', 'middle'), text(598, 86, '768×768', 't-cell', 'middle'), arrow(630, 74, 650, 74)]
    b += vec(652, 70, 7, 's4', cell=9, gap=2)
    b += [text(410, 140, 'E = conv.weight.reshape(768, 768).T', 't-code'), text(410, 160, 'x_p @ E + bias', 't-code'),
          text(410, 190, 'Same 589,824 weights, just reshaped.', 't-tick'),
          text(410, 210, f'max |difference|: {R["conv_vs_linear_maxdiff"]:.1e}', 't-note')]
    return svg(760, 250, 'Two views of the same patch embedding: on the left a convolution with a 16 by 16 kernel and stride 16 placed on every patch, on the right a linear layer applied to the flattened patch rows; the weights are the same numbers reshaped and the outputs agree to 5e-6.', b)


F['p2_conv_linear'] = conv_linear()


# ---------------------------------------------------------------- 7. [class] prepended and E_pos added
def class_pos():
    e = R['eq1']
    b = [text(20, 24, 'Eq. 1 with the real shapes and numbers', 't-title')]
    labels = ['x_class', 'x_p¹E', 'x_p²E', 'x_p³E', 'x_p⁴E', '…', 'x_p¹⁹⁶E']
    parts, cx = row(110, 44, labels, 60, 6, classes=['box-2'] + ['box-1'] * 6, h=28, tcls='t-cell')
    b += parts + [text(100, 62, '[ … ]', 't-tick', 'end'), text(110 + 7 * 66 + 4, 62, '197 × 768', 't-tick')]
    parts, cx = row(110, 92, ['E_pos[0]', 'E_pos[1]', 'E_pos[2]', 'E_pos[3]', 'E_pos[4]', '…', 'E_pos[196]'], 60, 6, cls='box-3', h=28, tcls='t-cell')
    b += parts + [text(100, 110, '+', 't-big', 'end'), text(110 + 7 * 66 + 4, 110, '197 × 768', 't-tick')]
    parts, cx = row(110, 140, ['z₀[0]', 'z₀[1]', 'z₀[2]', 'z₀[3]', 'z₀[4]', '…', 'z₀[196]'], 60, 6, cls='box', h=28, tcls='t-cell')
    b += parts + [text(100, 158, '=', 't-big', 'end'), text(110 + 7 * 66 + 4, 158, '197 × 768', 't-tick')]
    b.append(line(cx[0], 172, cx[0], 188, 'edge', 'ah'))
    f6 = lambda v: '[' + ', '.join(f'{t:6.3f}' for t in v) + ', …]'
    b += [text(20, 206, 'the [class] position, first 6 of 768 numbers:', 't-tick'),
          text(20, 226, 'x_class    ' + f6(e['x_class_first6']), 't-code'),
          text(20, 244, 'E_pos[0]   ' + f6(e['Epos0_first6']), 't-code'),
          text(20, 262, 'z_0[0]     ' + f6(e['z0_0_first6']), 't-code'),
          text(20, 288, f'the first patch: x_p¹E {f6(R["patch0_emb_first6"])}', 't-muted'),
          text(20, 304, f'+ E_pos[1] {f6(e["Epos1_first6"])}', 't-muted'),
          text(20, 320, f'= z_0[1] {f6(e["z0_1_first6"])}', 't-muted'),
          text(20, 336, f'Our z_0 matches the library embedding output to {e["maxdiff"]:.1e}.', 't-muted')]
    return svg(760, 348, 'Equation 1 drawn: the learned class embedding is prepended to the 196 patch embeddings, the 197 learned position rows are added, and the result is z_0 with shape 197 by 768; the real first six numbers of the class position are listed.', b)


F['p2_class_pos'] = class_pos()


# ---------------------------------------------------------------- 8. the encoder block, pre-norm
def encoder_block():
    b = [text(20, 24, 'One encoder layer: LayerNorm before each block, residual after it (Eq. 2, 3)', 't-title')]
    y = 110
    items = [('z_{l-1}', 'box', 62), ('LN', 'box-3', 48), ('MSA', 'box-1', 70), ('+', 'node', 0), ("z'_l", 'box', 50), ('LN', 'box-3', 48), ('MLP', 'box-4', 70), ('+', 'node', 0), ('z_l', 'box', 50)]
    x = 20
    centres = []
    for name, cls, w in items:
        if cls == 'node':
            b += [f'<circle class="node" cx="{x + 14}" cy="{y + 17}" r="13"/>', text(x + 14, y + 22, '+', 't-note', 'middle')]
            centres.append((x, x + 28))
            x += 28 + 22
        else:
            b += token(x, y, w, name, cls, h=34, tcls='t-tick')
            centres.append((x, x + w))
            x += w + 22
    for i in range(len(centres) - 1):
        b.append(arrow(centres[i][1] + 2, y + 17, centres[i + 1][0] - 2, y + 17))
    # residual arcs
    def arc(x1, x2, label):
        return [f'<path class="path" d="M{x1},{y} C{x1},{y - 60} {x2},{y - 60} {x2},{y + 2}" marker-end="url(#ah-on)"/>', text((x1 + x2) / 2, y - 48, label, 't-tick', 'middle')]
    b += arc((centres[0][0] + centres[0][1]) / 2, (centres[3][0] + centres[3][1]) / 2, 'residual: + z_{l-1}')
    b += arc((centres[4][0] + centres[4][1]) / 2, (centres[7][0] + centres[7][1]) / 2, "residual: + z'_l")
    b += brace(centres[0][0], centres[4][1], y + 50, "Eq. 2:  z'_l = MSA(LN(z_{l-1})) + z_{l-1}", 'edge')
    b += brace(centres[4][0], centres[8][1], y + 84, "Eq. 3:  z_l = MLP(LN(z'_l)) + z'_l", 'edge')
    b += [text(20, 236, 'Every box keeps the shape 197 × 768. LN and the MLP work on each token on its own;', 't-muted'),
          text(20, 253, 'only MSA mixes information between tokens.', 't-muted')]
    return svg(760, 266, 'One ViT encoder layer drawn left to right: the input passes through LayerNorm and multi-head self-attention and is added back to itself (Equation 2), then through LayerNorm and the MLP and is added back again (Equation 3); the two residual paths arc over the blocks.', b)


F['p2_encoder_block'] = encoder_block()


# ---------------------------------------------------------------- 9. the shapes inside MSA
def msa_shapes():
    s = R['layer1']['shapes']
    b = [text(20, 24, 'Inside MSA for ViT-B/16: 197 tokens, 12 heads of 64', 't-title')]
    p, W0, H0 = matrix(30, 70, 14, 10, 's1', cell=7, gap=1.4, label='LN(z)', shape='197 × 768')
    b += p
    x = 30 + W0 + 26
    b.append(arrow(x - 22, 70 + H0 / 2, x - 4, 70 + H0 / 2))
    # 12 heads as a stack of offset boxes
    for i in range(3):
        off = (2 - i) * 7
        b.append(box(x + off, 56 + off, 196, 118, 'box', 8))
    b += [text(x + 98, 76, 'head h (12 of these)', 't-note', 'middle'),
          text(x + 10, 96, 'q, k, v = LN(z) U_qkv', 't-tick'), text(x + 10, 112, 'each 197 × 64', 't-tick'),
          text(x + 10, 132, 'A = softmax(q kᵀ/√64)', 't-tick'), text(x + 10, 148, '197 × 197, rows sum to 1', 't-tick'),
          text(x + 10, 166, 'SA_h = A v: 197 × 64', 't-tick')]
    x2 = x + 196 + 26 + 10
    b.append(arrow(x + 196 + 16, 120, x2 - 4, 120))
    p, W1, H1 = matrix(x2, 70, 14, 10, 's3', cell=7, gap=1.4, label='concat', shape='197 × 768')
    b += p
    for j in range(1, 12):
        xx = x2 + j * (W1 / 12)
        b.append(line(xx, 68, xx, 70 + H1 + 2, 'edge-dim'))
    x3 = x2 + W1 + 26 + 8
    b.append(arrow(x2 + W1 + 14, 120, x3 - 4, 120))
    p, W2, H2 = matrix(x3, 70, 10, 10, 's4', cell=7, gap=1.4, label='U_msa', shape='768 × 768')
    b += p
    x4 = x3 + W2 + 26 + 8
    b.append(arrow(x3 + W2 + 14, 120, x4 - 4, 120))
    p, W3, H3 = matrix(x4, 70, 14, 10, 's2', cell=7, gap=1.4, label='MSA(z)', shape='197 × 768')
    b += p
    b += lines(20, 228, ['12 heads × 64 numbers = 768 = D, so the concatenation has the same width as the input.',
                         'Setting D_h = D/k keeps the parameter count the same whatever the number of heads k is (Appendix A).',
                         f'Attention parameters per layer: four 768 × 768 matrices (q, k, v, U_msa) and four biases = {R["params_B16"]["formula"]["attn_layer"]:,}.'], 't-muted', 17)
    return svg(760, 276, 'The shapes inside multi-head self-attention for ViT-B/16: the normalised input of 197 by 768 goes through 12 heads, each computing q, k, v of 197 by 64, an attention matrix of 197 by 197 and an output of 197 by 64; the heads are concatenated to 197 by 768 and multiplied by U_msa of 768 by 768.', b)


F['p2_msa_shapes'] = msa_shapes()


# ---------------------------------------------------------------- 10. where the [class] token looks: two heads of layer 1 as 14x14 heatmaps
def cls_attention():
    L1 = R['layer1']
    b = [text(20, 22, 'The [class] query in layer 1: its 196 patch weights on the 14 × 14 grid', 't-title')]
    rows = [str(i) for i in range(14)]
    cols = [str(j) for j in range(14)]
    for k, (G, ttl, x0) in enumerate([(L1['A_cls_grid'], f'head 1: {L1["A_cls_self"]:.2f} on [class] itself, largest patch weight {max(max(r) for r in L1["A_cls_grid"]):.4f}', 20),
                                        (L1['A_best_grid'], f'head {L1["best_head"]}: {L1["A_best_self"]:.3f} on itself, largest patch weight {max(max(r) for r in L1["A_best_grid"]):.4f}', 400)]):
        m = max(max(r) for r in G)
        M = [[v / m for v in r] for r in G]
        b.append(text(x0, 50, ttl, 't-tick'))
        b += heatmap(x0, 64, M, rows, cols, 17, label_w=34)
        b += [text(x0 + 34 + 7 * 17, 64 + 14 * 17 + 36, 'patch column 0 to 13', 't-muted', 'middle')]
    b += [text(20, 364, 'Rows are patch rows 0 to 13. Shade = weight divided by the largest weight in that head, so each head', 't-muted'),
          text(20, 381, 'is on its own scale. Hover a cell for the value.', 't-muted')]
    return svg(760, 392, 'Two 14 by 14 heatmaps of the class token attention weights in layer 1 of the real ViT-B/16 on the cat picture: head 1 puts 0.76 on the class token itself and spreads tiny weights over the patches; head 8 spreads its weight, mostly along the top row of patches.', b)


F['p2_cls_attention'] = cls_attention()


# ---------------------------------------------------------------- 11. the top-5 patches on the picture grid
def attn_on_grid():
    L1 = R['layer1']
    G = L1['A_best_grid']
    m = max(max(r) for r in G)
    cell = 16
    b = [text(20, 22, f'Head {L1["best_head"]} of layer 1: where the [class] token looks, on the patch grid of the picture', 't-title')]
    b += grid_cells(40, 44, 14, cell, vals=[[0.1 + 0.9 * v / m for v in r] for r in G], cls='s1', gap=1)
    for rank, t in enumerate(L1['best_top'][1:6], 1):
        x, y = 40 + t['col'] * cell, 44 + t['row'] * cell
        b += [box(x - 1, y - 1, cell, cell, 'box-on', 2), text(x + cell / 2, y + cell / 2 + 4, str(rank), 't-cell on', 'middle')]
    b += brace(40, 40 + 14 * cell, 44 + 14 * cell + 6, 'the picture, 14 × 14 patches', 'edge')
    x = 320
    b += [text(x, 56, 'rank   weight   patch (row, col)', 't-code')]
    b.append(text(x, 76, f'  -    {L1["A_best_self"]:.4f}   [class] itself', 't-code'))
    for rank, t in enumerate(L1['best_top'][1:6], 1):
        b.append(text(x, 76 + rank * 18, f'  {rank}    {t["weight"]:.4f}   patch {t["token"] - 1:3d}  (row {t["row"]:2d}, col {t["col"]:2d})', 't-code'))
    b += lines(x, 200, ['All 197 weights of this row add up to 1, so no single patch', 'can be large: the top patch has 0.015, and the top row of the', 'picture (the red sofa behind the cats) takes most of the weight.', 'In head 1 the same row puts 0.76 on [class] itself and at most', '0.0026 on any patch. Part 5 looks at what later layers attend to.'], 't-tick', 16)
    return svg(760, 300, 'The 14 by 14 patch grid of the cat picture shaded by the class token attention weights of head 8 in layer 1, with the five strongest patches outlined and numbered; all five sit in the top row of the picture and the largest weight is 0.015.', b)


F['p2_attn_on_grid'] = attn_on_grid()


# ---------------------------------------------------------------- 12. LayerNorm step by step
def layernorm_fig():
    ln = R['layernorm']
    mu, sd = ln['mean'], ln['std']
    xin = ln['input_first6']
    normed = ln['normed_first6']
    gam, bet, out = ln['gamma_first6'], ln['beta_first6'], ln['out_first6']
    rows = ['x (input)', 'x − μ', '(x − μ) / σ', 'γ', 'β', 'γ·(…) + β']
    M = [xin, [v - mu for v in xin], normed, gam, bet, out]
    b = [text(20, 24, 'LayerNorm on the [class] token entering layer 1, first 6 of its 768 numbers', 't-title')]
    b += table(20, 54, M, rows, [f'[{i}]' for i in range(6)], cw=70, ch=26, label_w=110, fmt='{:.3f}')
    x = 20 + 110 + 6 * 70 + 20
    b += lines(x, 70, [f'μ = mean of all 768 = {mu:.4f}', f'σ = √(variance + ε) = {sd:.4f}', 'ε = 1e-12', '', 'γ and β are learned,', 'one value per position,', 'the same for every token.', '', f'input length {ln["len_in"]:.2f},', f'output length {ln["len_out"]:.2f}.'], 't-tick', 16)
    b += [text(20, 246, f'Our hand computation matches the library LayerNorm to {ln["maxdiff"]:.0e}.', 't-muted'),
          text(20, 263, 'Rows 2 and 3 use the exact μ and σ, so they differ slightly from what the rounded row 1 would give.', 't-muted')]
    return svg(760, 276, 'LayerNorm step by step on the first six numbers of the class token: the input, minus the mean, divided by the standard deviation, then multiplied by the learned gamma and shifted by the learned beta, with the real values from ViT-B/16.', b)


F['p2_layernorm'] = layernorm_fig()


# ---------------------------------------------------------------- 13. the MLP and the GELU curve
def mlp_gelu():
    g = R['gelu']
    b = [text(20, 24, 'The MLP block: 768 → 3072 → 768 with GELU in between', 't-title')]
    b += token(60, 252, 110, 'LN(z′): 768', 'box', h=30)
    b += [arrow(115, 250, 115, 212), text(125, 236, 'W₁ (768 × 3072) + b₁', 't-tick')]
    b += token(20, 180, 190, 'h: 3072 numbers', 'box-4', h=30)
    b += [arrow(115, 178, 115, 140), text(125, 164, 'GELU, number by number', 't-tick')]
    b += token(20, 108, 190, 'GELU(h): 3072', 'box-4', h=30)
    b += [arrow(115, 106, 115, 68), text(125, 92, 'W₂ (3072 × 768) + b₂', 't-tick')]
    b += token(60, 36, 110, 'MLP(…): 768', 'box', h=30)
    # GELU plot
    L, Tt, pw, ph = 440, 44, 290, 210
    xmin, xmax, ymin, ymax = -4, 4, -0.6, 4
    X = lambda v: L + (v - xmin) / (xmax - xmin) * pw
    Y = lambda v: Tt + ph - (v - ymin) / (ymax - ymin) * ph
    b += [f'<line class="axis" x1="{L}" y1="{Y(0):.1f}" x2="{L + pw}" y2="{Y(0):.1f}"/>', f'<line class="axis" x1="{X(0):.1f}" y1="{Tt}" x2="{X(0):.1f}" y2="{Tt + ph}"/>']
    for t in [-4, -2, 2, 4]:
        b.append(text(X(t), Y(0) + 16, str(t), 't-tick', 'middle'))
    for t in [1, 2, 3, 4]:
        b += [f'<line class="grid" x1="{L}" y1="{Y(t):.1f}" x2="{L + pw}" y2="{Y(t):.1f}"/>', text(X(0) - 6, Y(t) + 4, str(t), 't-tick', 'end')]
    pts_relu = ' '.join(f'{X(x):.1f},{Y(y):.1f}' for x, y in zip(g['curve_x'], g['relu_y']))
    pts = ' '.join(f'{X(x):.1f},{Y(y):.1f}' for x, y in zip(g['curve_x'], g['curve_y']))
    b += [f'<polyline class="edge-dim" points="{pts_relu}"/>', f'<polyline class="l4" points="{pts}"/>']
    for x, y in zip(g['xs'], g['gelu']):
        b.append(f'<g class="mark"><title>GELU({x:.1f}) = {y:.4f}</title><circle class="s4 ring" cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4"/></g>')
    b += [text(L + 4, Tt + 12, 'GELU(x) = x · Φ(x)', 't-note'), text(L + 4, Tt + 28, 'dashed: ReLU = max(0, x)', 't-muted'),
          text(X(-3.9), Y(0.75), 'GELU(−1) = −0.159', 't-tick'), text(X(2) - 8, Y(1.955) + 4, 'GELU(2) = 1.955', 't-tick', 'end')]
    b += lines(20, 296, [f'In layer 1 only {100 * R["layer1"]["positive_share"]:.1f}% of the 3072 numbers of the [class] token are positive before GELU;',
                         'the negative ones come out small but not zero.',
                         f'MLP parameters per layer: 768·3072 + 3072 + 3072·768 + 768 = {R["params_B16"]["formula"]["mlp_layer"]:,} (8D² + 5D).'], 't-muted', 17)
    return svg(760, 340, 'Left, the MLP block of one ViT layer: 768 numbers widened to 3072 by W1, passed through GELU and narrowed back to 768 by W2. Right, the GELU curve x times Phi of x computed in code, with ReLU dashed for comparison; GELU of minus 1 is minus 0.159 and GELU of 2 is 1.955.', b)


F['p2_mlp_gelu'] = mlp_gelu()


# ---------------------------------------------------------------- 14. parameter breakdown
def params_fig():
    sh = R['params_B16']['shares']
    tot = R['params_B16']['counted']
    items = [('attention (12 layers)', sh['attention'] / 1e6), ('MLP (12 layers)', sh['mlp'] / 1e6), ('embeddings (E, [class], E_pos)', sh['embeddings'] / 1e6),
             ('head (768 × 1000 + 1000)', sh['head'] / 1e6), ('LayerNorms (25 of them)', sh['layernorms'] / 1e6)]
    b = hbars(20, 50, items, 380, row_h=34, cls='s1', label_w=220, fmt=lambda v: f'{v:.2f} M  ({100 * v * 1e6 / tot:.1f}%)', vmax=60,
              title=f'Where the {tot:,} parameters of ViT-B/16 (1000-class head) sit')
    return svg(760, 240, 'Horizontal bars of the parameter count of ViT-B/16: the MLP blocks hold 56.67 million (65.5 percent), attention 28.35 million (32.7 percent), the embeddings 0.74 million, the head 0.77 million and the LayerNorms 0.04 million.', b)


F['p2_params'] = params_fig()


# ---------------------------------------------------------------- 15. Table 1 as three columns of blocks
def table1_blocks():
    b = [text(20, 24, 'Table 1 drawn: one block per layer, block width ∝ hidden size D', 't-title')]
    V = R['params_variants']
    specs = [('ViT-Base', 12, 768, 3072, 12, '86M', R['params_B16']['formula']['without_head'], 150),
             ('ViT-Large', 24, 1024, 4096, 16, '307M', V['ViT-L/16']['formula']['without_head'], 380),
             ('ViT-Huge', 32, 1280, 5120, 16, '632M', V['ViT-H/14']['formula']['without_head'], 610)]
    for name, L, D, mlp, heads, paper, ours, cx in specs:
        w = D / 8
        bh, gap = 7, 1.6
        y0 = 320 - L * (bh + gap)
        for i in range(L):
            b.append(f'<rect class="s1" x="{cx - w / 2:.1f}" y="{y0 + i * (bh + gap):.1f}" width="{w:.1f}" height="{bh}" rx="1.5" style="fill-opacity:0.55"/>')
        b += [text(cx, y0 - 10, name, 't-note', 'middle'),
              text(cx, 338, f'L = {L} layers, D = {D}', 't-tick', 'middle'), text(cx, 354, f'MLP {mlp}, {heads} heads', 't-tick', 'middle'),
              text(cx, 370, f'paper: {paper}; ours: {ours / 1e6:.1f}M', 't-tick', 'middle')]
    b.append(text(20, 396, '"ours" counts the model without any classification head (the formula of this part); Part 3 reads Table 1 properly.', 't-muted'))
    return svg(760, 408, 'Table 1 of the paper as three stacks of blocks: ViT-Base with 12 layers of width 768, ViT-Large with 24 layers of width 1024 and ViT-Huge with 32 layers of width 1280, with the paper parameter counts of 86M, 307M and 632M and our counts without a head.', b)


F['p2_table1_blocks'] = table1_blocks()


# ---------------------------------------------------------------- 16. the hybrid architecture
def hybrid():
    b = [text(20, 24, 'Hybrid: the patches come from a CNN feature map instead of raw pixels', 't-title')]
    b += grid_cells(24, 60, 6, 12, cls='s2')
    b += [text(60, 148, 'picture', 't-tick', 'middle'), text(60, 164, '224 × 224 × 3', 't-tick', 'middle'), arrow(100, 96, 124, 96)]
    b += [box(126, 72, 110, 48, 'box-2', 8), text(181, 92, 'ResNet stages', 't-tick', 'middle'), text(181, 108, '(convolutions)', 't-tick', 'middle'), arrow(238, 96, 262, 96)]
    for i in range(3):
        off = (2 - i) * 6
        b.append(box(264 + off, 50 + off, 92, 92, 'box-ghost', 3))
    b += grid_cells(264, 50, 14, 6.6, cls='s3')
    b += [text(310, 162, 'feature map', 't-tick', 'middle'), text(310, 178, '14 × 14 × 1024', 't-tick', 'middle'), arrow(360, 96, 384, 96)]
    b += [box(386, 60, 120, 72, 'box', 8), text(446, 82, '1 × 1 "patches":', 't-tick', 'middle'), text(446, 98, '196 vectors', 't-tick', 'middle'), text(446, 114, 'of 1024 numbers', 't-tick', 'middle'), arrow(508, 96, 532, 96)]
    b += [box(534, 74, 56, 44, 'box-4', 6), text(562, 92, 'E', 't-math', 'middle'), text(562, 110, '1024 → 768', 't-cell', 'middle'), arrow(592, 96, 616, 96)]
    b += [box(618, 60, 130, 72, 'box-1', 8), text(683, 82, '[class] + 196 tokens', 't-tick', 'middle'), text(683, 98, '+ positions, then', 't-tick', 'middle'), text(683, 114, 'the same encoder', 't-tick', 'middle')]
    b += lines(20, 206, ['The paper (Section 4.1) takes the 7 × 7 output of stage 4 of a ResNet50, or the 14 × 14 output of a',
                         'stage 3 extended to replace stage 4, and feeds every position of the map as one token.',
                         'Everything after E is unchanged. The 14 × 14 map gives 196 tokens, as in ViT-B/16.'], 't-muted', 17)
    return svg(760, 258, 'The hybrid architecture: the picture goes through ResNet stages, the resulting 14 by 14 by 1024 feature map is read as 196 one by one patches of 1024 numbers, projected to 768 by E, and then handled exactly like the plain ViT sequence with a class token and position embeddings.', b)


F['p2_hybrid'] = hybrid()


# ---------------------------------------------------------------- 17. BERT versus ViT
def bert_vs_vit():
    b = [text(190, 24, 'BERT (text)', 't-title', 'middle'), text(570, 24, 'ViT (pictures)', 't-title', 'middle')]
    parts, cx = row(20, 44, ['[CLS]', 'the', 'kid', 'smiles', '[SEP]'], 60, 6, classes=['box-2', 'box', 'box', 'box', 'box'], h=28, tcls='t-cell')
    b += parts + [text(190, 92, 'WordPiece tokens', 't-tick', 'middle')]
    parts, cx2 = row(400, 44, ['[class]', 'p1', 'p2', '…', 'p196'], 60, 6, classes=['box-2', 'box-1', 'box-1', 'box-1', 'box-1'], h=28, tcls='t-cell')
    b += parts + [text(570, 92, '16 × 16 patches', 't-tick', 'middle')]
    for x0, lab in [(20, 'token table lookup + segment + position (then LayerNorm)'), (400, 'x_p E (linear projection) + position')]:
        b += [arrow(x0 + 164, 98, x0 + 164, 112), box(x0, 114, 328, 30, 'box-3', 8), text(x0 + 164, 133, lab, 't-tick', 'middle'), arrow(x0 + 164, 146, x0 + 164, 160)]
    for x0 in (20, 400):
        b += [box(x0, 162, 328, 50, 'box-1', 10), text(x0 + 164, 182, 'Transformer encoder', 't-note', 'middle'), text(x0 + 164, 200, '12 layers, width 768, 12 heads, MLP 3072', 't-tick', 'middle'),
              arrow(x0 + 164, 214, x0 + 164, 228)]
    b += [box(20, 230, 328, 30, 'box-on', 8), text(184, 249, 'C = output of [CLS] → task head (sentence label)', 't-tick', 'middle'),
          box(400, 230, 328, 30, 'box-on', 8), text(564, 249, 'y = LN(z_L⁰) → head (image class)', 't-tick', 'middle')]
    b += [text(374, 190, '=', 't-big', 'middle'), text(374, 134, '≈', 't-big', 'middle'),
          text(20, 288, 'Differences: BERT normalises after each block (post-norm) and looks tokens up in a table of 30,522 rows;', 't-muted'),
          text(20, 305, 'ViT normalises before each block (pre-norm) and projects every patch with one 768 × 768 matrix E.', 't-muted')]
    return svg(760, 316, 'BERT and ViT side by side: text tokens with a CLS token versus patch tokens with a class token, a lookup-plus-position embedding versus a linear projection plus position, the same 12-layer encoder of width 768, and the first output token feeding the task head in both.', b)


F['p2_bert_vs_vit'] = bert_vs_vit()


# ---------------------------------------------------------------- 18. where 2D structure enters (inductive bias)
def inductive():
    b = [text(20, 24, 'Where the 2D structure of the picture enters ViT: only at two points', 't-title')]
    steps = [('cut into 16 × 16 patches', 'box-on', ['2D used here:', 'neighbouring pixels', 'stay in one patch']),
             ('E + learned positions', 'box', ['no 2D at the start:', 'E_pos rows are random', 'until trained']),
             ('encoder: global MSA, per-token MLP', 'box', ['no 2D: every patch sees', 'every patch; the MLP', 'treats each token alike']),
             ('fine-tune at a new size: 2D interpolation of E_pos', 'box-on', ['2D used here:', 'position rows laid on', 'the grid and resized'])]
    x = 20
    widths = [140, 140, 200, 196]
    for (t, cls, ls), w in zip(steps, widths):
        b += [box(x, 50, w, 44, cls, 10)]
        words = t.split(': ') if ': ' in t else [t]
        if len(words) == 2:
            b += [text(x + w / 2, 68, words[0] + ':', 't-tick', 'middle'), text(x + w / 2, 84, words[1], 't-tick', 'middle')]
        else:
            b.append(text(x + w / 2, 77, t, 't-tick', 'middle'))
        b += lines(x + 6, 120, ls, 't-tick', 15)
        if x + w + 10 < 740:
            b.append(arrow(x + w + 2, 72, x + w + 16, 72))
        x += w + 16
    b += lines(20, 190, ['CNN: locality, 2D neighbourhoods and translation equivariance are built into every layer.',
                         'ViT: the MLP is local (one token at a time) and translation equivariant (the same weights for every token);',
                         'self-attention is global. Which patch is next to which has to be learned from data, which is why ViT',
                         'needs so much of it (Part 4).'], 't-muted', 17)
    return svg(760, 262, 'The two places where two-dimensional structure enters ViT, highlighted: cutting the picture into patches at the start, and the two-dimensional interpolation of position embeddings when fine-tuning at a new resolution; the embedding and the encoder in between carry no built-in 2D knowledge.', b)


F['p2_inductive_bias'] = inductive()


# ---------------------------------------------------------------- 19. the tiny attention example as a picture
def tiny_attention():
    h = T['heads'][0]
    names = ['t0', 't1', 't2', 't3']
    b = [text(20, 24, 'Head 1 of the tiny example: scaled scores, softmax weights and SA(z) = A v', 't-title')]
    b += table(20, 66, h['scaled'], names, names, cw=44, ch=24, label_w=30, fmt='{:.2f}', title='q kᵀ / √3')
    b += table(260, 66, h['A'], names, names, cw=44, ch=24, label_w=30, fmt='{:.2f}', title='A = softmax (rows sum to 1)', signed=False)
    v = [r[6:] for r in h['qkv']]
    b += table(500, 66, v, names, ['v₀', 'v₁', 'v₂'], cw=44, ch=24, label_w=30, fmt='{:.0f}', title='v (value rows)')
    b += table(500, 206, h['SA'], names, ['', '', ''], cw=44, ch=24, label_w=30, fmt='{:.2f}', title='SA(z) = A v')
    b += [text(20, 334, 'row t0 of SA(z) = 0.661·v(t0) + 0.208·v(t1) + 0.066·v(t2) + 0.066·v(t3)', 't-tick'),
          text(20, 352, f'its first number: 0.661·{v[0][0]:.0f} + 0.208·({v[1][0]:.0f}) + 0.066·({v[2][0]:.0f}) + 0.066·({v[3][0]:.0f}) = {h["SA"][0][0]:.3f}', 't-tick'),
          text(20, 370, 'Orange cells are negative numbers; the shade shows the size.', 't-muted'),
          arrow(236, 130, 254, 130), arrow(476, 130, 494, 130)]
    return svg(760, 392, 'The tiny worked example of Appendix A for head 1: the four by four scaled scores, the softmax weights whose rows sum to one, the four value rows of three numbers, and the output SA of z as the weighted sum of the value rows.', b)


F['p2_tiny_attention'] = tiny_attention()


# ---------------------------------------------------------------- 20. permutation equivariance
def permutation():
    P = T['perm']
    names = ['t0', 't1', 't2', 't3']
    perm = P['perm']
    b = [text(20, 24, 'Reorder the input rows: the output rows reorder the same way (no positions)', 't-title')]

    def column(x0, order, outs, title):
        parts = [text(x0 + 150, 52, title, 't-note', 'middle')]
        for i, idx in enumerate(order):
            y = 66 + i * 34
            parts += token(x0, y, 50, names[idx], 'box-1', h=26, tcls='t-cell')
            parts.append(arrow(x0 + 52, y + 13, x0 + 70, y + 13))
            parts += [box(x0 + 72, y, 60, 26, 'box', 6), text(x0 + 102, y + 17, 'MSA', 't-cell', 'middle'), arrow(x0 + 134, y + 13, x0 + 152, y + 13)]
            parts.append(text(x0 + 156, y + 17, '[' + ', '.join(f'{v:.2f}' for v in outs[i][:3]) + ', …]', 't-code'))
        return parts
    b += column(20, [0, 1, 2, 3], P['out'], 'original order')
    b += column(400, perm, P['out_perm'], 'shuffled: t2, t0, t3, t1')
    for i, idx in enumerate(perm):
        y1 = 66 + idx * 34 + 13
        y2 = 66 + i * 34 + 13
        b.append(f'<path class="edge-dim" d="M{20 + 300},{y1} C{20 + 340},{y1} {400 - 40},{y2} {400 - 4},{y2}"/>')
    b += [text(20, 216, 'Identical numbers, only the rows moved (max |difference| under 1e-6): attention only sees a set of tokens.', 't-muted'),
          text(20, 233, f'Add a position row to each token before attention and the shuffled run differs (max |difference| {P["maxdiff_with_pos"]:.3f}):', 't-muted'),
          text(20, 250, 'now the order matters.', 't-muted')]
    return svg(760, 262, 'Permutation equivariance of multi-head self-attention in the tiny example: the four input tokens in the original order and in a shuffled order each go through MSA, and the output rows are the same numbers in the shuffled order, joined by dashed lines; adding position embeddings breaks this.', b)


F['p2_permutation'] = permutation()

json.dump(F, open('results/figs_part2.json', 'w'))
print(f'wrote {len(F)} figures:', ', '.join(F))
