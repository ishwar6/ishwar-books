"""Figures for Part 2 (the task), drawn from results/part2.json.
Writes results/figs_part2.json (key -> svg string). Colours come only from CSS classes."""
import json, math, os
from figlib import svg, text, box, arrow, lines_chart
from samfig import line

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'part2.json')))
F = {}


def legend(x, y, items):
    b = []
    for cls, lab in items:
        b += [f'<rect class="{cls}" x="{x}" y="{y - 9}" width="10" height="10" rx="2"/>', text(x + 16, y, lab, 't-tick')]
        x += 16 + 7.2 * len(lab) + 22
    return b


def toy_cat(x, y, s=1.0, cls='s4', op=0.35):
    """A simple cat shape (body, head, two ears, tail) for the schematic figures."""
    return [f'<g transform="translate({x},{y}) scale({s})" style="fill-opacity:{op}">'
            f'<ellipse class="{cls}" cx="60" cy="70" rx="48" ry="26"/>'
            f'<circle class="{cls}" cx="112" cy="46" r="20"/>'
            f'<path class="{cls}" d="M98,32 L102,14 L112,28 Z"/><path class="{cls}" d="M114,28 L124,12 L128,32 Z"/>'
            f'<path class="{cls}" d="M14,66 C-4,60 -6,36 6,26 C10,38 14,52 22,58 Z"/></g>']


# ---------------------------------------------------------------- 1. the kinds of prompt
def prompt_kinds():
    b = []
    titles = [('foreground point', '"this is part of it"'), ('background point', '"this is not part of it"'),
              ('box', '"it is inside here"'), ('rough mask', '"about this shape"'), ('text', '"the cat"')]
    for k, (t, sub) in enumerate(titles):
        x0 = 10 + k * 150
        b += [box(x0, 10, 140, 150, 'box-ghost', 10)]
        b += toy_cat(x0 + 4, 30, 0.95)
        if k == 0:
            b.append(f'<circle class="s3 ring" cx="{x0 + 62}" cy="{30 + 0.95 * 70:.0f}" r="7"/>')
        if k == 1:
            b += [f'<circle class="s3 ring" cx="{x0 + 62}" cy="{30 + 0.95 * 70:.0f}" r="7"/>',
                  f'<circle class="s2 ring" cx="{x0 + 120}" cy="132" r="7"/>', text(x0 + 120, 136, '−', 't-tick', 'middle')]
        if k == 2:
            b.append(f'<rect class="edge-on" x="{x0 + 6}" y="36" width="{128}" height="{78}" rx="2" style="fill:none"/>')
        if k == 3:
            b.append(f'<ellipse class="s1" cx="{x0 + 70}" cy="{30 + 0.95 * 62:.0f}" rx="58" ry="32" style="fill-opacity:0.55"/>')
        if k == 4:
            b.append(text(x0 + 70, 138, '"the cat"', 't-note', 'middle'))
        b += [text(x0 + 70, 182, t, 't-note', 'middle'), text(x0 + 70, 200, sub, 't-muted', 'middle')]
    b.append(text(10, 232, 'Each prompt gives the model a different kind of hint. Points can be mixed: two foreground points and one background point are one prompt.', 't-muted'))
    return svg(760, 244, 'Five kinds of prompt: a foreground point, a background point, a box, a rough mask, and text.', b)


F['p2_prompt_kinds'] = prompt_kinds()


# ---------------------------------------------------------------- 2. nested masks of the tail click
def nested():
    m = R['clicks']['tail']['masks']
    ins = R['clicks']['tail']['inside']
    b = [text(20, 24, 'One click on the tail: three masks, each inside the next', 't-title')]
    # concentric areas as circles with area proportional to pixels
    cx, cy = 190, 160
    r3 = 110
    rs = [r3 * math.sqrt(mm['pixels'] / m[2]['pixels']) for mm in m]
    for k, cls in [(2, 's3'), (1, 's1'), (0, 's2')]:
        b.append(f'<g class="mark"><title>mask {k + 1}: {m[k]["pixels"]:,} pixels</title>'
                 f'<circle class="{cls}" cx="{cx + (r3 - rs[k]) * 0.55:.1f}" cy="{cy + (r3 - rs[k]) * 0.45:.1f}" r="{rs[k]:.1f}" style="fill-opacity:0.55"/></g>')
    b.append(f'<circle class="s4 ring" cx="{cx + (r3 - rs[0]) * 0.55:.1f}" cy="{cy + (r3 - rs[0]) * 0.45:.1f}" r="4"/>')
    rows = [('subpart', 'mask 1: tail', m[0]), ('part', 'mask 2: tail + back leg', m[1]), ('whole', 'mask 3: whole cat', m[2])]
    for k, (lev, lab, mm) in enumerate(rows):
        y = 70 + k * 58
        b += [f'<rect class="{["s2", "s1", "s3"][k]}" x="380" y="{y - 11}" width="12" height="12" rx="2"/>',
              text(400, y, f'{lev}: {lab}', 't-note'),
              text(400, y + 20, f'{mm["pixels"]:,} pixels, predicted IoU {mm["score"]:.3f}', 't-tick')]
    b += [text(380, 252, f'{100 * ins[0]:.1f}% of mask 1 is inside mask 2; {100 * ins[1]:.1f}% of mask 2 is inside mask 3', 't-muted'),
          text(380, 272, 'circle area is in proportion to the number of pixels', 't-muted')]
    return svg(760, 290, 'The three masks of the tail click are nested: the tail inside the tail and back leg inside the whole cat, with 2,347, 12,024 and 50,290 pixels.', b)


F['p2_nested'] = nested()


# ---------------------------------------------------------------- 4. why one output averages: the loss of one pixel
def averaging():
    W, H, L, T, B = 760, 316, 70, 58, 54
    pw, ph = 420, H - T - B
    X = lambda p: L + pw * p
    Y = lambda v: T + ph - ph * min(v, 3.0) / 3.0
    b = [text(20, 24, 'Loss of one pixel when people disagree: L(p) = q·(−ln p) + (1 − q)·(−ln(1 − p))', 't-title')]
    for v in [0, 1, 2, 3]:
        b += [line(L, Y(v), L + pw, Y(v), 'grid'), text(L - 8, Y(v) + 4, f'{v}', 't-tick', 'end')]
    for p in [0, 0.25, 0.5, 0.75, 1]:
        b.append(text(X(p), T + ph + 18, f'{p:g}', 't-tick', 'middle'))
    b.append(f'<line class="axis" x1="{L}" y1="{T + ph}" x2="{L + pw}" y2="{T + ph}"/>')
    for q, lc, dc in [(0.5, 'l1', 's1'), (0.8, 'l3', 's3')]:
        ps = [i / 200 for i in range(2, 199)]
        pts = ' '.join(f'{X(p):.1f},{Y(-(q * math.log(p) + (1 - q) * math.log(1 - p))):.1f}' for p in ps)
        b.append(f'<polyline class="{lc}" points="{pts}" style="fill:none"/>')
        Lq = -(q * math.log(q) + (1 - q) * math.log(1 - q))
        b.append(f'<g class="mark"><title>q = {q}: smallest loss {Lq:.3f} at p = {q}</title><circle class="{dc} ring" cx="{X(q):.1f}" cy="{Y(Lq):.1f}" r="6"/></g>')
    b += [text(L + pw / 2, H - 12, 'p: the probability that the single output gives to "this pixel is part of the mask"', 't-tick', 'middle'),
          text(L - 50, T - 10, 'loss', 't-tick')]
    tx = L + pw + 30
    T0 = R['toy']
    b += [text(tx, 70, 'A pixel on the cat, not on the tail.', 't-note'),
          text(tx, 90, 'q = fraction of people who want the cat.', 't-muted'),
          f'<rect class="s1" x="{tx}" y="{113}" width="10" height="10" rx="2"/>', text(tx + 16, 122, f'q = 0.5: best p = {T0[0]["best_p"]:.2f}', 't-note'),
          f'<rect class="s3" x="{tx}" y="{139}" width="10" height="10" rx="2"/>', text(tx + 16, 148, f'q = 0.8: best p = {T0[1]["best_p"]:.2f}', 't-note'),
          text(tx, 186, 'The best single answer is the', 't-muted'), text(tx, 204, 'average of the answers, not one', 't-muted'),
          text(tx, 222, 'of them. At p = 0.5 the pixel sits', 't-muted'), text(tx, 240, 'on the threshold: a blurred mask.', 't-muted')]
    return svg(W, H, 'The loss of one pixel for a single output, when a fraction q of people want the cat and the rest want only the tail. The smallest loss is at p = q: 0.5 for q = 0.5 and 0.8 for q = 0.8.', b)


F['p2_averaging'] = averaging()


# ---------------------------------------------------------------- 5. the training loop
def sim_loop():
    b = [box(20, 40, 150, 64, 'box-1', 10), text(95, 66, 'first prompt', 't-note', 'middle'), text(95, 86, 'point or noisy box', 't-tick', 'middle'),
         arrow(174, 72, 216, 72),
         box(220, 40, 140, 64, 'box-on', 10), text(290, 66, 'SAM', 't-title', 'middle'), text(290, 86, 'decoder only', 't-tick', 'middle'),
         arrow(364, 72, 406, 72),
         box(410, 40, 150, 64, 'box-2', 10), text(485, 66, 'predicted mask', 't-note', 'middle'), text(485, 86, 'best score if 3', 't-tick', 'middle'),
         arrow(564, 72, 600, 72),
         box(604, 40, 140, 64, 'box-3', 10), text(674, 66, 'compare with', 't-note', 'middle'), text(674, 86, 'ground truth', 't-note', 'middle')]
    b += [arrow(674, 108, 674, 150),
          box(560, 154, 184, 78, 'box-ghost', 10), text(652, 176, 'error region', 't-note', 'middle'),
          text(652, 196, 'missed pixels → + point', 't-tick', 'middle'), text(652, 214, 'extra pixels → − point', 't-tick', 'middle'),
          arrow(556, 193, 440, 193),
          box(220, 160, 216, 66, 'box-1', 10), text(328, 184, 'next prompt', 't-note', 'middle'),
          text(328, 204, 'all points so far', 't-tick', 'middle'), text(328, 220, '+ previous mask logits', 't-tick', 'middle'),
          f'<path class="edge" d="M290,156 L290,108" marker-end="url(#ah)"/>']
    b += [text(20, 264, '11 rounds for each mask: the first prompt, 8 new points, and 2 rounds with no new point (only the previous mask).', 't-muted'),
          text(20, 284, 'The loss is computed after each round. The image encoder runs once; only the light decoder runs 11 times.', 't-muted')]
    return svg(760, 296, 'The training loop of the paper: a first prompt goes to SAM, the mask is compared with the ground truth, a new point is taken from the error region, and the points and the previous mask logits go back in as the next prompt.', b)


F['p2_sim_loop'] = sim_loop()


# ---------------------------------------------------------------- 6. IoU after each round
def sim_curve():
    S = R['sim']
    det = [r['iou'] for r in R['sim_detail']]
    xs = list(range(1, 12))
    return lines_chart(760, 300, 'IoU with the target after each of the 11 rounds. The run that starts on the tail goes from 0.027 to 0.972 after one more point. The mean of 20 runs goes from 0.974 to 0.979 with the mask prompt and to 0.967 without it.',
                       xs, [('run from the tail click', det, 'l2', 's2'), ('mean of 20 runs, with mask prompt', S['mean'], 'l1', 's1'),
                            ('mean of 20 runs, no mask prompt', S['mean_nomask'], 'l3', 's3')],
                       'round', 'IoU with the target', 0, 1, [0, 0.25, 0.5, 0.75, 1], fmt=lambda v: f'{v:.2f}', right=250)


F['p2_sim_curve'] = sim_curve()


# ---------------------------------------------------------------- 7. noisy boxes
def box_noise():
    rows = R['boxes']['rows']
    L, top, bw, rh = 250, 50, 330, 44
    labels = ['10%, max 20 px (training)', '10%, no limit', '20%, no limit', '40%, no limit']
    b = [text(20, 24, 'Noisy boxes around the right cat: mean IoU with the tight-box mask (30 boxes each)', 't-title')]
    for v in [0, 0.5, 1.0]:
        b += [line(L + bw * v, top - 6, L + bw * v, top + 4 * rh - 6, 'grid'), text(L + bw * v, top + 4 * rh + 10, f'{v:.1f}', 't-muted', 'middle')]
    for i, (lab, r) in enumerate(zip(labels, rows)):
        y = top + i * rh
        w = bw * r['mean']
        cls = 's3' if i == 0 else 's1'
        b += [text(L - 10, y + 15, lab, 't-tick t-strong' if i == 0 else 't-tick', 'end'),
              text(L - 10, y + 31, f'corners move {r["shift"]:.0f} px on average', 't-muted', 'end'),
              f'<g class="mark"><title>{lab}: mean IoU {r["mean"]:.3f}, worst {r["min"]:.3f}</title><rect class="{cls}" x="{L}" y="{y + 4}" width="{w:.1f}" height="22" rx="3"/></g>',
              text(L + w + 8, y + 20, f'{r["mean"]:.3f}  ({r["above90"]}/30 above 0.9)', 't-val')]
    return svg(760, top + 4 * rh + 24, 'Mean IoU with the tight-box mask for noisy boxes: 0.995 with the training noise, 0.889, 0.514 and 0.225 with larger noise.', b)


F['p2_box_noise'] = box_noise()


# ---------------------------------------------------------------- 9. multi-task vs task generalization by composition
def multitask():
    b = [text(20, 24, 'Multi-task model', 't-title'), text(400, 24, 'Promptable model, used as a component', 't-title')]
    tasks = ['semantic', 'instance', 'panoptic']
    for i, t in enumerate(tasks):
        y = 44 + i * 40
        b += [box(20, y, 110, 30, 'box-1', 6), text(75, y + 20, t, 't-tick', 'middle'), arrow(134, y + 15, 170, 100)]
        b += [arrow(290, 100, 326, y + 15), box(330, y, 50, 30, 'box-1', 6), text(355, y + 20, t[:4] + '.', 't-tick', 'middle')]
    b += [box(174, 70, 112, 60, 'box-on', 8), text(230, 96, 'one model', 't-note', 'middle'), text(230, 114, '3 fixed tasks', 't-tick', 'middle'),
          text(20, 184, 'train: these 3 tasks', 't-muted'), text(250, 184, 'test: the same 3', 't-muted')]
    comps = [('detector', 'boxes', 'instance masks'), ('grid of points', 'points', 'object proposals'),
             ('grid + edge filter', 'points', 'edges'), ('a person', 'clicks', 'interactive masks')]
    for i, (src, p, out) in enumerate(comps):
        y = 44 + i * 36
        b += [box(400, y, 124, 28, 'box-2', 6), text(462, y + 19, src, 't-tick', 'middle'), arrow(528, y + 14, 566, y + 14),
              text(547, y + 5, p, 't-muted', 'middle'),
              box(570, y, 54, 28, 'box-on', 6), text(597, y + 19, 'SAM', 't-tick', 'middle'), arrow(628, y + 14, 650, y + 14),
              text(654, y + 19, out, 't-tick')]
    b += [text(400, 202, 'train: one task (prompt → valid mask)', 't-muted'), text(400, 220, 'test: new tasks, by what you connect to it', 't-muted')]
    return svg(760, 234, 'Left: a multi-task model learns a fixed set of tasks and is tested on the same tasks. Right: one promptable model becomes a part of different systems: a detector gives boxes for instance masks, a grid of points gives object proposals and edges, a person gives clicks.', b)


F['p2_multitask'] = multitask()


# ---------------------------------------------------------------- 10. composition: CLIP in DALL-E, SAM in a system
def composition():
    b = [text(20, 24, 'Composition: a foundation model as one block of a larger system', 't-title')]
    # DALL-E row
    b += [box(20, 46, 140, 46, 'box-1', 8), text(90, 74, 'text', 't-note', 'middle'), arrow(164, 69, 196, 69),
          box(200, 40, 330, 58, 'box-ghost', 10), text(212, 58, 'DALL·E (an image generator)', 't-tick'),
          box(300, 62, 90, 30, 'box-on', 6), text(345, 82, 'CLIP', 't-note', 'middle'),
          text(400, 82, 'ranks the pictures', 't-muted'),
          arrow(534, 69, 566, 69), box(570, 46, 170, 46, 'box-3', 8), text(655, 74, 'pictures', 't-note', 'middle')]
    # SAM row
    b += [box(20, 136, 140, 46, 'box-1', 8), text(90, 164, 'photo', 't-note', 'middle'), arrow(164, 159, 196, 159),
          box(200, 124, 330, 70, 'box-ghost', 10), text(212, 142, 'a cat instance segmenter', 't-tick'),
          box(212, 152, 110, 30, 'box-2', 6), text(267, 172, 'cat detector', 't-tick', 'middle'), arrow(326, 167, 360, 167),
          text(343, 157, 'box', 't-muted', 'middle'),
          box(364, 152, 70, 30, 'box-on', 6), text(399, 172, 'SAM', 't-note', 'middle'),
          arrow(534, 159, 566, 159), box(570, 136, 170, 46, 'box-3', 8), text(655, 164, 'one mask per cat', 't-note', 'middle')]
    b.append(text(20, 222, 'Neither block was trained for the whole system. The prompt is the connection between them.', 't-muted'))
    return svg(760, 236, 'Two examples of composition. CLIP is one block inside the DALL·E image generator. SAM is one block inside a cat instance segmenter, after a cat detector that gives the boxes.', b)


F['p2_composition'] = composition()

json.dump(F, open(os.path.join(HERE, 'results', 'figs_part2.json'), 'w'))
print(f'{len(F)} figures:', ', '.join(F))
