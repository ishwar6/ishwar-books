"""Figures for Part 3 (the model), drawn from results/part3.json.
Writes results/figs_part3.json (key -> svg string). Colours come only from CSS classes."""
import json, math, os
import numpy as np
from figlib import svg, text, box, arrow, esc, lines_chart
from samfig import token, line, hbars

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'part3.json')))
F = {}


def legend(x, y, items):
    b = []
    for cls, lab in items:
        b += [f'<rect class="{cls}" x="{x}" y="{y - 9}" width="10" height="10" rx="2"/>', text(x + 16, y, lab, 't-tick')]
        x += 16 + 7.2 * len(lab) + 22
    return b


def stage(x, y, w, h, cls, title, lines, tcls='t-note'):
    b = [box(x, y, w, h, cls, 8), text(x + w / 2, y + 20, title, tcls, 'middle')]
    for k, l in enumerate(lines):
        b.append(text(x + w / 2, y + 38 + 16 * k, l, 't-tick', 'middle'))
    return b


# ---------------------------------------------------------------- 1. the whole model with the real shapes
def overview():
    b = [text(20, 22, 'Image encoder: once per picture', 't-title')]
    y = 36
    b += stage(14, y, 112, 74, 'box', 'photo', ['640 × 480 × 3', 'resize + pad'])
    b += [arrow(128, y + 37, 144, y + 37)]
    b += stage(146, y, 112, 74, 'box', 'pixels', ['3 × 1024 × 1024', '3,145,728 numbers'])
    b += [arrow(260, y + 37, 276, y + 37)]
    b += stage(278, y, 120, 74, 'box-2', 'patch embedding', ['16 × 16 conv', '64 × 64 × 1280'])
    b += [arrow(400, y + 37, 416, y + 37)]
    # the 32 blocks as a strip
    x0 = 418
    b += [box(x0, y, 222, 74, 'box-2', 8), text(x0 + 111, y + 20, '32 transformer blocks', 't-note', 'middle')]
    glob = [g for g in R['configs']['huge']['global_blocks']]
    for i in range(32):
        cls = 's3' if i in glob else 's2'
        b.append(f'<rect class="{cls}" x="{x0 + 10 + i * 6.4:.1f}" y="{y + 30}" width="5" height="16" rx="1" style="fill-opacity:{0.95 if i in glob else 0.55}"/>')
    b.append(text(x0 + 111, y + 64, '28 windowed + 4 global', 't-tick', 'middle'))
    b += [arrow(642, y + 37, 656, y + 37)]
    b += stage(658, y, 108, 74, 'box-2', 'neck', ['1×1 conv, 3×3 conv', '256 × 64 × 64'])
    # the embedding
    b += [line(712, y + 76, 712, 150, 'edge'), line(712, 150, 120, 150, 'edge'), arrow(120, 150, 120, 172)]
    b.append(text(420, 144, 'image embedding 256 × 64 × 64 = 1,048,576 numbers', 't-tick', 'middle'))
    b.append(text(20, 172, 'Prompt encoder and mask decoder: once per prompt', 't-title'))
    y2 = 186
    b += stage(14, y2, 150, 88, 'box-1', 'click (275, 62)', ['Fourier encoding', '+ "foreground" vector', '→ 1 token × 256'])
    b += stage(14, y2 + 98, 150, 74, 'box-1', 'mask prompt (optional)', ['256 × 256 → 3 convs', '→ 256 × 64 × 64'])
    b += [arrow(166, y2 + 44, 196, y2 + 44), arrow(166, y2 + 135, 196, y2 + 100)]
    b += stage(198, y2, 160, 172, 'box-3', 'mask decoder', [])
    for k, l in enumerate(['7 tokens × 256:', '1 IoU + 4 mask + 2 prompt', '', 'image: 4,096 × 256', '(embedding + mask prompt)', '',
                           '2 two-way layers', '+ 1 final attention']):
        b.append(text(278, y2 + 42 + 15.5 * k, l, 't-tick', 'middle'))
    b += [arrow(360, y2 + 60, 390, y2 + 40), arrow(360, y2 + 130, 390, y2 + 130)]
    b += stage(392, y2, 180, 74, 'box-3', 'upscale ×4', ['2 transposed convs', '32 × 256 × 256'])
    b += stage(392, y2 + 98, 180, 74, 'box-3', 'token MLPs', ['each mask token → 32 numbers', 'IoU token → 4 scores'])
    b += [arrow(574, y2 + 37, 604, y2 + 70), arrow(574, y2 + 135, 604, y2 + 100)]
    b += stage(606, y2 + 40, 160, 92, 'box-on', 'dot product', ['32 · 32 at each pixel', '→ 3 masks 256 × 256', '+ 3 scores'])
    return svg(780, 372, 'SAM with the real shapes. The image encoder turns the 640 by 480 photo into 3 by 1024 by 1024 pixels, then 64 by 64 by 1280 patch vectors, then 32 blocks (28 windowed, 4 global) and a neck, giving the 256 by 64 by 64 image embedding. For each prompt, the prompt encoder makes tokens, and the mask decoder makes 3 masks of 256 by 256 and 3 scores.', b)


F['p3_overview'] = overview()


# ---------------------------------------------------------------- 2. windowed attention
def windows():
    W = R['windows']
    c = 4.2                                          # pixels per token cell
    x0, y0 = 20, 44
    n = W['padded']
    b = [text(20, 24, f'The 64 × 64 token grid, padded to {n} × {n}', 't-title')]
    b.append(f'<rect class="s4" x="{x0}" y="{y0}" width="{64 * c:.1f}" height="{64 * c:.1f}" style="fill-opacity:0.25"/>')
    b.append(f'<rect class="waste" x="{x0 + 64 * c:.1f}" y="{y0}" width="{(n - 64) * c:.1f}" height="{n * c:.1f}"/>')
    b.append(f'<rect class="waste" x="{x0}" y="{y0 + 64 * c:.1f}" width="{64 * c:.1f}" height="{(n - 64) * c:.1f}"/>')
    for k in range(6):
        b.append(line(x0 + k * 14 * c, y0, x0 + k * 14 * c, y0 + n * c, 'edge-on'))
        b.append(line(x0, y0 + k * 14 * c, x0 + n * c, y0 + k * 14 * c, 'edge-on'))
    b.append(f'<rect class="box-on" x="{x0 + 14 * c:.1f}" y="{y0 + 14 * c:.1f}" width="{14 * c:.1f}" height="{14 * c:.1f}" rx="0"/>')
    b.append(text(x0 + n * c / 2, y0 + n * c + 22, f'{W["n_windows"]} windows of 14 × 14 = {W["tokens_per_window"]} tokens; '
                  f'{W["pad_cells"]} padding cells', 't-muted', 'middle'))
    # cost bars
    X = 360
    b.append(text(X, 24, 'Attention scores per head, per block', 't-title'))
    mx = W['pairs_global']
    for i, (lab, v, cls) in enumerate([('global', W['pairs_global'], 's3'), ('windowed', W['pairs_window'], 's2')]):
        y = 50 + i * 44
        w = max(3, 240 * v / mx)
        b += [text(X + 70, y + 18, lab, 't-tick', 'end'),
              f'<g class="mark"><title>{lab}: {v:,}</title><rect class="{cls}" x="{X + 78}" y="{y + 4}" width="{w:.1f}" height="20" rx="3"/></g>',
              text(X + 84 + w, y + 19, f'{v:,}', 't-val')]
    b.append(text(X, 160, f'4,096 × 4,096 against 25 × 196 × 196: {W["ratio"]:.1f} times fewer', 't-muted'))
    # the 32 blocks
    b.append(text(X, 200, 'The 32 blocks of ViT-H', 't-title'))
    glob = R['configs']['huge']['global_blocks']
    for i in range(32):
        cls = 's3' if i in glob else 's2'
        b.append(f'<g class="mark"><title>block {i + 1}: {"global" if i in glob else "windowed"}</title>'
                 f'<rect class="{cls}" x="{X + i * 12:.1f}" y="214" width="10" height="26" rx="2" style="fill-opacity:{0.95 if i in glob else 0.5}"/></g>')
    for g in glob:
        b.append(text(X + g * 12 + 5, 256, str(g + 1), 't-tick', 'middle'))
    b += legend(X, 290, [('s2', 'windowed (14 × 14)'), ('s3', 'global (all 4,096)')])
    return svg(780, 360, 'Windowed attention. The 64 by 64 token grid is padded to 70 by 70 and cut into 25 windows of 14 by 14 tokens. Each window attends only inside itself: 960,400 scores per head instead of 16,777,216 for global attention. Blocks 8, 16, 24 and 32 of ViT-H use global attention.', b)


F['p3_windows'] = windows()


# ---------------------------------------------------------------- 3. the positional encoding: similarity map
def posenc():
    S = np.load(os.path.join(HERE, 'results', 'part3_pe_sim.npy'))[:48, :64]   # the photo covers 64 x 48 cells
    S2 = S.reshape(24, 2, 32, 2).mean(axis=(1, 3))
    c = 11
    x0, y0 = 20, 44
    P = R['pe_sim']
    b = [text(20, 24, 'Similarity of the click\'s encoding to the encoding of each place', 't-title')]
    for i in range(24):
        for j in range(32):
            v = float(S2[i, j])
            if v > 0.02:
                b.append(f'<rect class="cell" x="{x0 + j * c:.1f}" y="{y0 + i * c:.1f}" width="{c - 1:.1f}" height="{c - 1:.1f}" style="fill-opacity:{v:.2f}"/>')
    b.append(f'<rect class="box-ghost" x="{x0 - 1}" y="{y0 - 1}" width="{32 * c + 1}" height="{24 * c + 1}" rx="2"/>')
    cx, cy = (P['cell'][0] + 0.5) / 2, (P['cell'][1] + 0.5) / 2
    b.append(f'<circle class="s3 ring" cx="{x0 + cx * c:.1f}" cy="{y0 + cy * c:.1f}" r="5"/>')
    X = 400
    rows = [('at the click', P['at']), ('1 cell away (16 px)', P['d1']), ('4 cells away', P['d4']), ('16 cells away', P['d16']), ('far corner', P['corner'])]
    b.append(text(X, 60, 'cosine similarity, going right from the click', 't-tick'))
    for k, (lab, v) in enumerate(rows):
        y = 78 + k * 30
        w = max(2, 150 * max(v, 0))
        b += [text(X + 150, y + 15, lab, 't-tick', 'end'),
              f'<g class="mark"><title>{lab}: {v:.3f}</title><rect class="s1" x="{X + 158}" y="{y + 2}" width="{w:.1f}" height="18" rx="3"/></g>',
              text(X + 164 + w, y + 16, f'{v:.3f}', 't-val')]
    b.append(text(X, 250, 'The encoding is a fingerprint of the place.', 't-muted'))
    b.append(text(X, 268, 'Near places have similar fingerprints;', 't-muted'))
    b.append(text(X, 286, 'far places have unrelated ones (near 0).', 't-muted'))
    b.append(text(x0, y0 + 24 * c + 22, 'the photo as the 64 × 48 grid of the image embedding (shown in 2 × 2 groups)', 't-muted'))
    return svg(780, y0 + 24 * c + 36, 'Cosine similarity between the positional encoding of the click on the ear and the encoding of each cell of the image grid. The similarity is 0.999 at the click, 0.979 one cell away, 0.701 four cells away and near 0 sixteen cells away.', b)


F['p3_posenc'] = posenc()


# ---------------------------------------------------------------- 4. one two-way decoder layer
def two_way():
    b = [text(20, 22, 'One layer of the mask decoder (the decoder has two)', 't-title')]
    # two columns: tokens (left) and image (right)
    b += [box(20, 40, 150, 56, 'box-1', 8), text(95, 62, 'tokens', 't-note', 'middle'), text(95, 82, '7 × 256', 't-tick', 'middle')]
    b += [box(610, 40, 150, 56, 'box-2', 8), text(685, 62, 'image', 't-note', 'middle'), text(685, 82, '4,096 × 256', 't-tick', 'middle')]
    steps = [('1  self-attention', 'each token looks at the 7 tokens', 'tokens ask, tokens answer', 'box-1'),
             ('2  token-to-image attention', 'each token looks at the 4,096 image vectors', 'tokens ask, image answers', 'box-3'),
             ('3  MLP', 'each token alone: 256 → 2,048 → 256', 'no mixing', 'box-1'),
             ('4  image-to-token attention', 'each image vector looks at the 7 tokens', 'image asks, tokens answer', 'box-3')]
    for k, (t, d1, d2, cls) in enumerate(steps):
        y = 114 + k * 62
        b += [box(200, y, 380, 50, cls, 8), text(214, y + 21, t, 't-note'), text(214, y + 39, d1, 't-tick'), text(566, y + 21, d2, 't-muted', 'end')]
    # arrows: which side is updated
    b += [arrow(95, 98, 198, 139), arrow(95, 98, 198, 201), arrow(685, 98, 582, 201), arrow(95, 98, 198, 263)]
    b += [arrow(582, 325, 685, 360), arrow(198, 325, 95, 360)]
    b += [box(20, 362, 150, 50, 'box-1', 8), text(95, 383, 'new tokens', 't-note', 'middle'), text(95, 401, '7 × 256', 't-tick', 'middle')]
    b += [box(610, 362, 150, 50, 'box-2', 8), text(685, 383, 'new image', 't-note', 'middle'), text(685, 401, '4,096 × 256', 't-tick', 'middle')]
    b.append(text(390, 436, 'Steps 1 to 3 change the tokens. Step 4 changes the image. Each step: + residual, then layer norm.', 't-muted', 'middle'))
    return svg(780, 450, 'One two-way layer of the mask decoder. Step 1: the 7 tokens attend to each other. Step 2: the tokens attend to the 4,096 image vectors. Step 3: an MLP updates each token. Step 4: the image vectors attend to the tokens. Steps 1 to 3 change the tokens, step 4 changes the image.', b)


F['p3_two_way'] = two_way()


# ---------------------------------------------------------------- 5. the dynamic dot product
def dynamic():
    D = R['dot']
    b = [text(20, 22, 'A mask token becomes a 32-number filter; each pixel has 32 numbers', 't-title')]
    b += [box(20, 40, 170, 60, 'box-3', 8), text(105, 62, 'mask token (256)', 't-note', 'middle'), text(105, 82, '→ its own 3-layer MLP', 't-tick', 'middle'),
          arrow(192, 70, 226, 70), box(228, 40, 150, 60, 'box-on', 8), text(303, 62, 'w: 32 numbers', 't-note', 'middle'),
          text(303, 82, 'one per mask', 't-tick', 'middle')]
    b += [box(420, 40, 170, 60, 'box-2', 8), text(505, 62, 'upscaled image', 't-note', 'middle'), text(505, 82, '32 × 256 × 256', 't-tick', 'middle'),
          arrow(592, 70, 626, 70), box(628, 40, 140, 60, 'box-2', 8), text(698, 62, 'f: 32 numbers', 't-note', 'middle'),
          text(698, 82, 'at each pixel', 't-tick', 'middle')]
    b.append(text(390, 132, 'mask logit at a pixel = w · f = w₁f₁ + w₂f₂ + … + w₃₂f₃₂', 't-on', 'middle'))
    hdr = 160
    b += [text(30, hdr, 'pixel of the photo', 't-tick'), text(300, hdr, 'mask 1 (the ear)', 't-tick', 'middle'), text(560, hdr, 'mask 3 (the cat)', 't-tick', 'middle')]
    for k, name in enumerate(['ear', 'cat body', 'blanket']):
        y = hdr + 14 + k * 38
        d = D[name]
        b += [box(20, y, 750, 30, 'box', 6), text(30, y + 20, f'{name} {tuple(d["pixel"])}', 't-note')]
        for j, (mk, cx) in enumerate([('mask1', 300), ('mask3', 560)]):
            lg, pr = d[mk]['logit'], d[mk]['prob']
            b.append(text(cx, y + 20, f'logit {lg:+.2f}  →  p = {pr:.3f}', 't-on' if lg > 0 else 't-tick', 'middle'))
    b.append(text(20, hdr + 14 + 3 * 38 + 16, 'Positive logit: the pixel is in the mask. The same pixel vectors serve all masks; only w changes.', 't-muted'))
    return svg(790, hdr + 14 + 3 * 38 + 30, 'The dynamic dot product. Each mask token goes through a small MLP and becomes 32 numbers w. Each pixel of the upscaled image has 32 numbers f. The mask logit is the dot product. At the ear pixel both masks are positive; at the cat body only the cat mask is positive; at the blanket both are negative.', b)


F['p3_dynamic'] = dynamic()


# ---------------------------------------------------------------- 6. focal loss against cross-entropy
def focal_curve():
    ps = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    ce = [-math.log(p) for p in ps]
    fl = [-(1 - p) ** 2 * math.log(p) for p in ps]
    xs = [int(round(100 * p)) for p in ps]
    return lines_chart(780, 300, 'Loss of one pixel against p_t, the probability that the model gives to the correct answer. Cross-entropy is minus log p_t. Focal loss multiplies it by (1 minus p_t) squared, so pixels that the model already gets right cost almost nothing.',
                       xs, [('cross-entropy  −log p_t', ce, 'l2', 's2'), ('focal  −(1 − p_t)² log p_t', fl, 'l3', 's3')],
                       'p_t, the probability of the correct answer (%)', 'loss of one pixel', 0, 4, [0, 1, 2, 3, 4],
                       xticks=[2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99], fmt=lambda v: f'{v:.2f}' if isinstance(v, float) else f'{v}', right=230)


F['p3_focal'] = focal_curve()


# ---------------------------------------------------------------- 7. the min over three masks
def loss_bars():
    L = R['loss']
    b = [text(20, 22, 'Mask loss (20 × focal + dice) of each of the 3 masks, log scale', 't-title')]
    lo, hi = math.log10(0.01), math.log10(40)
    X0, BW = 200, 420
    X = lambda v: X0 + BW * (math.log10(max(v, 0.01)) - lo) / (hi - lo)
    for v in [0.01, 0.1, 1, 10]:
        b += [line(X(v), 40, X(v), 290, 'grid'), text(X(v), 306, f'{v:g}', 't-muted', 'middle')]
    y = 46
    names = ['mask 1 (427 px)', 'mask 2 (470 px)', 'mask 3 (49,520 px)']
    for tname in ['the ear', 'the cat']:
        b.append(text(20, y + 12, f'target: {tname}', 't-note'))
        y += 20
        best = L[tname]['best']
        for k in range(3):
            v = L[tname]['rows'][k]['mask_loss']
            on = k + 1 == best
            w = X(v) - X0
            b += [text(X0 - 10, y + 15, names[k], 't-tick t-strong' if on else 't-tick', 'end'),
                  f'<g class="mark"><title>{tname}, {names[k]}: {v:.4f}</title><rect class="{"s3" if on else "s4"}" x="{X0}" y="{y + 2}" width="{max(w, 3):.1f}" height="18" rx="3"/></g>',
                  text(X0 + max(w, 3) + 6, y + 16, f'{v:.4f}' + ('   ← the minimum: only this mask learns' if on else ''), 't-on' if on else 't-val')]
            y += 26
        y += 18
    return svg(780, 318, 'The mask loss of each of the three masks for the ear click, against two targets. If the target is the ear, mask 2 has the smallest loss, 0.0152. If the target is the cat, mask 3 has the smallest loss, 0.0471. Only the mask with the minimum loss gets a gradient.', b)


F['p3_loss'] = loss_bars()


# ---------------------------------------------------------------- 8. learning-rate schedule
def lr_fig():
    C = R['lr_curve']
    L, T, PW, PH = 80, 34, 640, 200
    lo, hi = math.log10(1e-6), math.log10(2e-3)
    X = lambda t: L + PW * t / 90_000
    Y = lambda v: T + PH - PH * (math.log10(v) - lo) / (hi - lo)
    b = [text(20, 20, 'Learning rate during training (log scale)', 't-title')]
    for v, lab in [(1e-5, '1e-5'), (1e-4, '1e-4'), (1e-3, '1e-3')]:
        b += [line(L, Y(v), L + PW, Y(v), 'grid'), text(L - 8, Y(v) + 4, lab, 't-tick', 'end')]
    for t in [0, 30_000, 60_000, 90_000]:
        b.append(text(X(t), T + PH + 20, f'{t // 1000}k', 't-tick', 'middle'))
    b.append(f'<line class="axis" x1="{L}" y1="{T + PH}" x2="{L + PW}" y2="{T + PH}"/>')
    pts = []
    for t, v in C:
        pts.append(f'{X(t):.1f},{Y(v):.1f}')
    b.append(f'<polyline class="l2" points="{" ".join(pts)}"/>')
    b += [text(X(30_000), Y(8e-4) - 8, '8e-4 after 250 warm-up steps', 't-note', 'middle'),
          text(X(73_000), Y(8e-5) - 8, '÷10 at 60k', 't-note', 'middle'), text(X(88_300), Y(8e-6) - 8, '÷10 at 86,666', 't-note', 'end')]
    b.append(text(L + PW / 2, T + PH + 42, 'training iteration (256 pictures each)', 't-tick', 'middle'))
    return svg(780, T + PH + 56, 'The learning rate of SAM training: a warm-up to 0.0008 over 250 iterations, then 0.0008 until iteration 60,000, 0.00008 until 86,666, and 0.000008 until the end at 90,000.', b)


F['p3_lr'] = lr_fig()


# ---------------------------------------------------------------- 9. where the compute and the time go
def compute():
    Fl, T = R['flops'], R['timing']['mps']
    b = [text(20, 22, 'Work for one picture and one click, SAM ViT-H', 't-title')]
    enc, dec = Fl['encoder'] / 1e9, Fl['decoder'] / 1e9
    lo, hi = math.log10(1), math.log10(10_000)
    X0, BW = 250, 380
    X = lambda v: X0 + BW * (math.log10(max(v, 1)) - lo) / (hi - lo)
    rows = [('image encoder', enc, f'{enc / 1000:.2f} TFLOPs', 's2'), ('prompt encoder + decoder', dec, f'{dec:.2f} GFLOPs', 's3')]
    for v, lab in [(1, '1'), (10, '10'), (100, '100'), (1000, '1,000'), (10000, '10,000')]:
        b += [line(X(v), 40, X(v), 116, 'grid'), text(X(v), 130, lab, 't-muted', 'middle')]
    for k, (lab, v, s, cls) in enumerate(rows):
        y = 46 + k * 34
        w = X(v) - X0
        b += [text(X0 - 10, y + 16, lab, 't-tick', 'end'),
              f'<g class="mark"><title>{lab}: {s}</title><rect class="{cls}" x="{X0}" y="{y + 2}" width="{max(w, 3):.1f}" height="20" rx="3"/></g>',
              text(X0 + max(w, 3) + 6, y + 17, s, 't-val')]
    b.append(text(X0 + BW / 2, 146, 'billions of floating-point operations (GFLOPs, log scale)', 't-muted', 'middle'))
    # time on the laptop GPU
    b.append(text(20, 186, 'Time on our laptop GPU (ms)', 't-title'))
    parts = [('28 windowed blocks', 28 * T['window_block'], 's2'), ('4 global blocks', 4 * T['global_block'], 's3'),
             ('patch embed + neck', T['patch'] + T['neck'], 's4'), ('prompt encoder + decoder', T['both'], 's1')]
    tot = sum(p[1] for p in parts)
    x = 20
    for lab, v, cls in parts:
        w = 740 * v / tot
        b.append(f'<g class="mark"><title>{lab}: {v:,.1f} ms</title><rect class="{cls}" x="{x:.1f}" y="200" width="{max(w, 1.5):.1f}" height="26"/></g>')
        x += w
    for k, (lab, v, cls) in enumerate(parts):
        s = f'{lab}: {v:,.0f} ms' if v >= 10 else f'{lab}: {v:.1f} ms'
        x, yy = 20 + (k % 2) * 380, 250 + (k // 2) * 22
        b += [f'<rect class="{cls}" x="{x}" y="{yy - 9}" width="10" height="10" rx="2"/>', text(x + 16, yy, s, 't-tick')]
    return svg(780, 288, 'The image encoder of ViT-H needs about 5.5 trillion operations for one picture; the prompt encoder and the mask decoder need about 3.6 billion for one click, less than 0.1 percent. On the laptop GPU, almost all the time goes to the 32 transformer blocks.', b)


F['p3_compute'] = compute()

json.dump(F, open(os.path.join(HERE, 'results', 'figs_part3.json'), 'w'))
print(f'{len(F)} figures:', ', '.join(F))
