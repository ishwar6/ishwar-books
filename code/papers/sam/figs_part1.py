"""Figures for Part 1 (the big idea), drawn from results/part1.json.
Writes results/figs_part1.json (key -> svg string). Colours come only from CSS classes."""
import json, math, os
from figlib import svg, text, box, arrow, esc
from samfig import token, line, hbars

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'part1.json')))
F = {}


def legend(x, y, items):
    b = []
    for cls, lab in items:
        b += [f'<rect class="{cls}" x="{x}" y="{y - 9}" width="10" height="10" rx="2"/>', text(x + 16, y, lab, 't-tick')]
        x += 16 + 7.2 * len(lab) + 22
    return b


def grid_cells(x, y, n, m, cell, fn, gap=1.5, rx=2):
    """fn(i, j) -> (class, opacity) or None for an empty cell."""
    b = []
    for i in range(n):
        for j in range(m):
            r = fn(i, j)
            if r is None:
                continue
            c, op = r
            b.append(f'<rect class="{c}" x="{x + j * cell + gap / 2:.1f}" y="{y + i * cell + gap / 2:.1f}" width="{cell - gap:.1f}" '
                     f'height="{cell - gap:.1f}" rx="{rx}" style="fill-opacity:{op:.2f}"/>')
    return b


# ---------------------------------------------------------------- 1. timeline
def timeline():
    ev = [('2015', 'FCN', ['a label for', 'every pixel'], 's1'),
          ('2017', 'Mask R-CNN', ['a mask for', 'every object'], 's1'),
          ('2020', 'GPT-3', ['one model, many', 'tasks, by prompt'], 's2'),
          ('2021', 'CLIP', ['pictures and text,', 'zero-shot'], 's2'),
          ('2023', 'SAM', ['one model, any', 'mask, by prompt'], 's3')]
    xs = [90, 240, 390, 540, 690]
    b = [line(50, 96, 730, 96, 'axis', 'ah')]
    for (yr, name, ls, cls), x in zip(ev, xs):
        b += [f'<circle class="{cls}" cx="{x}" cy="96" r="8"/>', text(x, 40, yr, 't-title', 'middle'), text(x, 64, name, 't-note', 'middle')]
        for k, l in enumerate(ls):
            b.append(text(x, 124 + 17 * k, l, 't-muted', 'middle'))
    b += legend(40, 190, [('s1', 'segmentation models'), ('s2', 'foundation models'), ('s3', 'this paper: the two lines meet')])
    return svg(760, 204, 'Timeline: FCN 2015 and Mask R-CNN 2017 made deep segmentation models; GPT-3 2020 and CLIP 2021 made prompted foundation models; SAM 2023 brings prompting to segmentation.', b)




# ---------------------------------------------------------------- 2. four kinds of output for one picture
def seg_kinds():
    """A toy 8 x 10 picture with two 'cats' and a 'sofa': classification, detection, semantic and instance segmentation."""
    n, m, c = 8, 10, 15
    cat1 = {(i, j) for i in range(2, 7) for j in range(1, 4)} | {(1, 2)}
    cat2 = {(i, j) for i in range(3, 7) for j in range(6, 9)} | {(2, 7)}
    titles = [('Classification', ['one label for the', 'whole picture: "cat"']),
              ('Detection', ['one box per', 'object']),
              ('Semantic segmentation', ['a class for every', 'pixel; cats not split']),
              ('Instance segmentation', ['a separate mask', 'for every object'])]
    b = []
    for k, (t, ls) in enumerate(titles):
        x0, y0 = 22 + k * 186, 40
        b.append(text(x0 + m * c / 2, 24, t, 't-title', 'middle'))

        def fn(i, j, k=k):
            if k == 0 or k == 1:
                return ('s4', 0.35) if (i, j) in cat1 or (i, j) in cat2 else ('box-ghost', 1)
            if k == 2:
                return ('s1', 0.8) if (i, j) in cat1 or (i, j) in cat2 else ('s4', 0.25)
            return ('s1', 0.85) if (i, j) in cat1 else ('s3', 0.85) if (i, j) in cat2 else ('s4', 0.25)
        b += grid_cells(x0, y0, n, m, c, fn)
        if k == 0:
            b.append(text(x0 + m * c / 2, y0 + n * c / 2 + 5, '"cat"', 't-note', 'middle'))
        if k == 1:
            b += [f'<rect class="edge-on" x="{x0 + 1 * c - 2}" y="{y0 + 1 * c - 2}" width="{3 * c + 4}" height="{6 * c + 4}" rx="2" style="fill:none"/>',
                  f'<rect class="edge-on" x="{x0 + 6 * c - 2}" y="{y0 + 2 * c - 2}" width="{3 * c + 4}" height="{5 * c + 4}" rx="2" style="fill:none"/>']
        for q, l in enumerate(ls):
            b.append(text(x0 + m * c / 2, y0 + n * c + 22 + 16 * q, l, 't-muted', 'middle'))
    return svg(760, 214, 'Four ways to describe one picture with two cats on a sofa: one label, boxes, a class per pixel, and one mask per cat.', b)


F['p1_seg_kinds'] = seg_kinds()


# ---------------------------------------------------------------- 3. a mask is a grid of 0s and 1s
def mask_grid():
    n, m, c = 9, 12, 22
    shape = {(i, j) for i in range(n) for j in range(m) if ((i - 4.2) / 3.6) ** 2 + ((j - 6) / 4.6) ** 2 <= 1}
    shape |= {(1, 3), (0, 3), (1, 8), (0, 8)}
    b = [text(20 + m * c / 2, 22, 'The picture area, cut into pixels', 't-title', 'middle')]
    b += grid_cells(20, 36, n, m, c, lambda i, j: ('s1', 0.85) if (i, j) in shape else ('s4', 0.22))
    x0 = 330
    b.append(text(x0 + m * c / 2 + 30, 22, 'The mask: one number per pixel', 't-title', 'middle'))
    for i in range(n):
        for j in range(m):
            on = (i, j) in shape
            b.append(text(x0 + 30 + j * c + c / 2, 36 + i * c + c / 2 + 5, '1' if on else '0', 't-mono t-on' if on else 't-mono t-muted', 'middle'))
    k = len(shape)
    b.append(text(20, 36 + n * c + 28, f'{k} of the {n * m} pixels belong to the object: area = {k}/{n * m} = {k / (n * m):.2f}', 't-note'))
    ear = R['clicks']['ear']['masks'][0]['pixels']
    b.append(text(20, 36 + n * c + 50, f'SAM\'s first mask for the ear click: {ear} ones among 640 × 480 = 307,200 numbers', 't-muted'))
    return svg(760, 36 + n * c + 64, 'A mask is a grid with one number per pixel: 1 where the object is and 0 elsewhere.', b)


F['p1_mask_grid'] = mask_grid()


# ---------------------------------------------------------------- 4. prompts: language model vs SAM
def prompt_analogy():
    b = [text(190, 24, 'Language model', 't-title', 'middle'), text(570, 24, 'SAM', 't-title', 'middle'), line(380, 12, 380, 260, 'edge-dim')]
    b += [box(40, 44, 300, 54, 'box-1', 8), text(190, 66, 'prompt (text):', 't-tick', 'middle'),
          text(190, 86, '"Translate to French: cat"', 't-note', 'middle'), arrow(190, 102, 190, 128),
          box(110, 132, 160, 40, 'box-on', 8), text(190, 157, 'model', 't-note', 'middle'), arrow(190, 176, 190, 202),
          box(40, 206, 300, 40, 'box-3', 8), text(190, 231, 'answer (text): "chat"', 't-note', 'middle')]
    kinds = [('point', 420), ('box', 500), ('mask', 580), ('text', 660)]
    for name, x in kinds:
        b.append(box(x, 44, 70, 54, 'box-1', 8))
        b.append(text(x + 35, 90, name, 't-tick', 'middle'))
    b += [f'<circle class="s3" cx="455" cy="66" r="6"/>', f'<rect class="edge-on" x="516" y="54" width="38" height="24" rx="2" style="fill:none"/>',
          f'<ellipse class="s1" cx="615" cy="66" rx="20" ry="11" style="fill-opacity:0.7"/>', text(695, 70, '"cat"', 't-tick', 'middle')]
    b += [arrow(570, 102, 570, 128), box(490, 132, 160, 40, 'box-on', 8), text(570, 157, 'SAM + the picture', 't-note', 'middle'),
          arrow(570, 176, 570, 202), box(400, 206, 340, 40, 'box-3', 8), text(570, 231, 'answer: a valid mask (and a score)', 't-note', 'middle')]
    return svg(760, 262, 'Left: a language model gets a text prompt and returns text. Right: SAM gets a point, box, mask or text prompt plus a picture, and returns a mask.', b)


F['p1_prompt_analogy'] = prompt_analogy()


# ---------------------------------------------------------------- 5. the three parts of the project
def three_parts():
    b = []
    items = [(30, 'box-1', '1. Task', ['promptable segmentation:', 'any prompt in,', 'a valid mask out'], '§2'),
             (280, 'box-2', '2. Model', ['SAM: image encoder,', 'prompt encoder,', 'mask decoder'], '§3'),
             (530, 'box-3', '3. Data', ['data engine: SAM helps', 'people label; the labels', 'train SAM; SA-1B'], '§4, §5')]
    for x, cls, t, ls, sec in items:
        b += [box(x, 30, 200, 130, cls, 12), text(x + 100, 56, t, 't-title', 'middle')]
        for k, l in enumerate(ls):
            b.append(text(x + 100, 80 + 18 * k, l, 't-note', 'middle'))
        b.append(text(x + 100, 150, f'paper {sec}', 't-muted', 'middle'))
    b += [arrow(234, 95, 276, 95), arrow(484, 95, 526, 95)]
    b.append(f'<path class="edge" d="M630,164 C630,206 380,206 380,166" marker-end="url(#ah)"/>')
    b.append(text(505, 226, 'more data makes a better model; a better model makes labelling faster', 't-muted', 'middle'))
    return svg(760, 240, 'The three linked parts of the project: the task defines what to learn, the model learns it, the data engine uses the model to collect more data, which trains a better model.', b)


F['p1_three_parts'] = three_parts()


# ---------------------------------------------------------------- 6. the pipeline with the real shapes
def pipeline():
    P = R['pipeline']
    b = []
    # stage boxes, left to right
    st = [(14, 'photo', [f'{P["orig"][0]} × {P["orig"][1]}', '× 3 colours'], 'box'),
          (150, 'resize + pad', [f'{P["resized"][0]} × {P["resized"][1]}', '→ 1024 × 1024'], 'box'),
          (286, 'image encoder', ['ViT-H/16', 'runs once'], 'box-2'),
          (422, 'image embedding', ['256 × 64 × 64', f'{P["embedding_numbers"]:,} numbers'], 'box-ghost')]
    for x, t, ls, cls in st:
        b += [box(x, 40, 120, 74, cls, 10), text(x + 60, 62, t, 't-note', 'middle')]
        for k, l in enumerate(ls):
            b.append(text(x + 60, 82 + 17 * k, l, 't-tick', 'middle'))
    for x in [134, 270, 406]:
        b.append(arrow(x + 1, 77, x + 15, 77))
    # prompt branch
    b += [box(422, 160, 120, 54, 'box-1', 10), text(482, 182, 'prompt', 't-note', 'middle'), text(482, 200, 'click (275, 62)', 't-tick', 'middle'),
          arrow(542, 187, 570, 187), arrow(542, 77, 570, 120)]
    b += [box(574, 96, 170, 116, 'box-3', 10), text(659, 120, 'prompt encoder', 't-note', 'middle'), text(659, 138, '+ mask decoder', 't-note', 'middle'),
          text(659, 160, 'runs per prompt', 't-tick', 'middle'), text(659, 178, '3 masks of 256 × 256', 't-tick', 'middle'),
          text(659, 196, '+ 3 scores', 't-tick', 'middle')]
    b += [arrow(659, 216, 659, 244), box(540, 248, 204, 46, 'box-on', 10), text(642, 268, 'resize back to 640 × 480', 't-note', 'middle'),
          text(642, 285, '3 masks, 3 scores', 't-tick', 'middle')]
    b += [text(14, 150, 'Heavy, once per picture', 't-title'), text(14, 170, 'ViT-H: 637M weights', 't-muted'),
          text(14, 268, 'Light, once per prompt', 't-title'), text(14, 288, 'decoder: 4.1M weights', 't-muted')]
    b.append(line(290, 140, 290, 300, 'edge-dim'))
    return svg(760, 310, 'SAM on our photo: the 640 by 480 photo is resized to 1024 by 768 and padded to 1024 by 1024, the image encoder makes a 256 by 64 by 64 embedding once, and the prompt encoder and mask decoder turn each prompt into three masks and three scores.', b)


F['p1_pipeline'] = pipeline()


# ---------------------------------------------------------------- 7. where the weights are
def params():
    rows = []
    for name, lab in [('facebook/sam-vit-base', 'SAM ViT-B'), ('facebook/sam-vit-large', 'SAM ViT-L'), ('facebook/sam-vit-huge', 'SAM ViT-H')]:
        p = R['params'][name]
        rows.append((lab, p['image_encoder'], p['mask_decoder'], p['prompt_encoder'], p['total']))
    L, top, bw, rh = 110, 50, 400, 48
    mx = max(r[4] for r in rows)
    b = [text(20, 24, 'Weights in each part of SAM (bar length = number of weights)', 't-title')]
    for i, (lab, e, d, pe, t) in enumerate(rows):
        y = top + i * rh
        we, wd = bw * e / mx, max(2.5, bw * d / mx)
        b += [text(L - 10, y + 20, lab, 't-tick', 'end'),
              f'<g class="mark"><title>{lab} image encoder: {e:,}</title><rect class="s2" x="{L}" y="{y + 6}" width="{we:.1f}" height="22" rx="3"/></g>',
              f'<g class="mark"><title>{lab} mask decoder: {d:,}</title><rect class="s3" x="{L + we + 1:.1f}" y="{y + 6}" width="{wd:.1f}" height="22" rx="3"/></g>',
              text(L + we + wd + 8, y + 22, f'{t / 1e6:.1f}M in total; encoder {100 * e / t:.1f}%', 't-val')]
    y = top + 3 * rh + 14
    b += legend(L, y, [('s2', 'image encoder'), ('s3', 'mask decoder (4.06M)'), ('s1', 'prompt encoder: 6,476 weights, too small to see')])
    return svg(760, y + 14, 'The image encoder holds 95.7 to 99.4 percent of the weights; the mask decoder has 4.06 million and the prompt encoder only 6,476.', b)


F['p1_params'] = params()


# ---------------------------------------------------------------- 8. amortized time per prompt
def amortized():
    T = R['timing']['facebook/sam-vit-huge']
    enc, dec = T['encoder_mps'] * 1000, T['decoder_cpu'] * 1000
    L, top, bw, rh = 170, 50, 470, 40
    rows = [(f'{a["k"]} prompt' + ('s' if a['k'] > 1 else ''), a['ms'], a['k']) for a in R['amortized']]
    mx = rows[0][1]
    b = [text(20, 24, 'Milliseconds per prompt, ViT-H (log scale)', 't-title')]
    lo, hi = math.log10(10), math.log10(mx * 1.3)
    X = lambda v: L + bw * (math.log10(v) - lo) / (hi - lo)
    for v in [10, 30, 100, 300, 1000, 3000]:
        b += [line(X(v), top - 6, X(v), top + 3 * rh, 'grid'), text(X(v), top + 3 * rh + 16, f'{v:,} ms', 't-muted', 'middle')]
    for i, (lab, ms, k) in enumerate(rows):
        y = top + i * rh
        we = X(min(ms, enc)) - L if k == 1 else X(enc / k + dec) - L
        b += [text(L - 10, y + 22, lab + ' per picture', 't-tick', 'end'),
              f'<g class="mark"><title>{lab}: {ms:.1f} ms per prompt</title><rect class="s2" x="{L}" y="{y + 8}" width="{we:.1f}" height="20" rx="3"/></g>',
              text(L + we + 8, y + 23, f'{ms:,.1f} ms', 't-val')]
    x50 = X(50)
    b += [line(x50, top - 14, x50, top + 3 * rh + 2, 'edge-on'), text(x50 + 6, top - 14, 'the paper\'s ~50 ms', 't-on')]
    y = top + 3 * rh + 42
    b.append(text(20, y, f'(encoder {enc:,.0f} ms on the GPU, once) / k  +  decoder {dec:.1f} ms on the CPU, per prompt', 't-muted'))
    return svg(760, y + 12, 'Time per prompt when one image embedding is reused for 1, 10 and 100 prompts: about 3,087, 324 and 48 milliseconds.', b)


F['p1_amortized'] = amortized()


# ---------------------------------------------------------------- 9. dataset scale
def data_scale():
    rows = [('SA-1B', 11, 1129, 's3'), ('Open Images', 1, 2.7, 's1'), ('LVIS v1', 0.120, 1.5, 's1'), ('COCO', 0.123, 0.9, 's1'), ('ADE20K', 0.028, 0.7, 's1')]
    L, top, bw, rh = 120, 54, 330, 34
    lo, hi = math.log10(0.1), math.log10(3000)
    X = lambda v: L + bw * (math.log10(v) - lo) / (hi - lo)
    b = [text(20, 24, 'Masks in each dataset, in millions (log scale)', 't-title')]
    for v, lab in [(0.1, '0.1M'), (1, '1M'), (10, '10M'), (100, '100M'), (1000, '1B')]:
        b += [line(X(v), top - 6, X(v), top + len(rows) * rh, 'grid'), text(X(v), top + len(rows) * rh + 16, lab, 't-muted', 'middle')]
    for i, (lab, imgs, masks, cls) in enumerate(rows):
        y = top + i * rh
        w = X(masks) - L
        b += [text(L - 10, y + 20, lab, 't-tick t-strong' if i == 0 else 't-tick', 'end'),
              f'<g class="mark"><title>{lab}: {masks}M masks on {imgs}M images</title><rect class="{cls}" x="{L}" y="{y + 6}" width="{w:.1f}" height="20" rx="3"/></g>',
              text(L + w + 8, y + 21, f'{masks:,}M masks · {imgs}M images · {masks / imgs:.0f} per image', 't-val')]
    return svg(760, top + len(rows) * rh + 30, 'SA-1B has 1,129 million masks on 11 million images, against 2.7 million masks for Open Images, the largest earlier set.', b)


F['p1_data_scale'] = data_scale()


# ---------------------------------------------------------------- 10. IoU
def iou_fig():
    E = R['ear_iou']
    b = [text(20, 24, 'Intersection over union (IoU)', 't-title')]
    b += [f'<circle class="s1" cx="130" cy="120" r="62" style="fill-opacity:0.45"/>',
          f'<circle class="s3" cx="190" cy="120" r="62" style="fill-opacity:0.45"/>',
          text(92, 124, 'A', 't-note', 'middle'), text(228, 124, 'B', 't-note', 'middle'),
          text(160, 206, 'two masks A and B', 't-muted', 'middle')]
    # intersection and union, schematic
    b += [text(400, 64, 'intersection A ∩ B', 't-note', 'middle'), text(400, 84, 'pixels in both', 't-muted', 'middle'),
          line(320, 104, 480, 104, 'edge'),
          text(400, 128, 'union A ∪ B', 't-note', 'middle'), text(400, 148, 'pixels in at least one', 't-muted', 'middle')]
    b += [text(560, 64, 'ear mask 1 vs ear mask 2', 't-tick'),
          text(560, 92, f'A = {E["a"]} pixels, B = {E["b"]} pixels', 't-tick'),
          text(560, 116, f'both: {E["inter"]} pixels', 't-tick'),
          text(560, 140, f'either: {E["union"]} pixels', 't-tick'),
          text(560, 168, f'IoU = {E["inter"]} / {E["union"]} = {E["iou"]:.3f}', 't-on')]
    b.append(text(20, 236, 'IoU = 1 means the two masks are identical. IoU = 0 means they share no pixel.', 't-muted'))
    return svg(760, 250, 'IoU is the number of pixels in both masks divided by the number of pixels in at least one mask. For the two ear masks it is 424 divided by 473, which is 0.896.', b)


F['p1_iou'] = iou_fig()


# ---------------------------------------------------------------- 11. zero-shot transfer
def zero_shot():
    b = [box(20, 40, 200, 110, 'box-2', 12), text(120, 66, 'Training data', 't-title', 'middle'),
         text(120, 92, 'SA-1B: 11M photos,', 't-note', 'middle'), text(120, 112, '1.1B masks', 't-note', 'middle'),
         arrow(224, 95, 280, 95), box(284, 60, 150, 70, 'box-on', 12), text(359, 92, 'SAM', 't-title', 'middle'),
         text(359, 112, 'no more training', 't-tick', 'middle')]
    tests = ['underwater', 'microscope cells', 'X-ray of bags', 'paintings', 'drone views', 'kitchen, first-person']
    for k, t in enumerate(tests):
        y = 26 + k * 26
        b += [arrow(438, 95, 500, y + 8), box(504, y - 6, 236, 22, 'box-1', 6), text(622, y + 9, t, 't-tick', 'middle')]
    b.append(text(20, 196, 'Zero-shot: the test pictures and tasks were never in the training data, and SAM gets no extra training for them.', 't-muted'))
    return svg(760, 210, 'Zero-shot transfer: SAM is trained once on SA-1B and then used without more training on new kinds of pictures such as underwater, microscope, X-ray and painting images.', b)


F['p1_zero_shot'] = zero_shot()

json.dump(F, open(os.path.join(HERE, 'results', 'figs_part1.json'), 'w'))
print(f'{len(F)} figures:', ', '.join(F))
