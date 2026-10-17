"""Figures for Part 4 (the data engine and SA-1B), drawn from results/part4.json and from the paper's own numbers.
Writes results/figs_part4.json (key -> svg string). Colours come only from CSS classes."""
import json, math, os
import numpy as np
from figlib import svg, text, box, arrow, esc
from samfig import line

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'part4.json')))
F = {}


def legend(x, y, items):
    b = []
    for cls, lab in items:
        b += [f'<rect class="{cls}" x="{x}" y="{y - 9}" width="10" height="10" rx="2"/>', text(x + 16, y, lab, 't-tick')]
        x += 16 + 7.0 * len(lab) + 22
    return b


# ---------------------------------------------------------------- 1. the three stages of the data engine (numbers of §4)
def engine():
    st = [('1  assisted-manual', 'box-1', ['annotators click, SAM draws,', 'brush and eraser fix it'],
           [('images', '120k'), ('new masks', '4.3M'), ('time per mask', '34 s → 14 s'), ('masks per image', '20 → 44'),
            ('model retrained', '6 times')]),
          ('2  semi-automatic', 'box-4', ['SAM pre-fills confident masks,', 'people add the missed objects'],
           [('images', '180k'), ('new masks', '5.9M'), ('time per mask', 'back to 34 s'), ('masks per image', '44 → 72'),
            ('model retrained', '5 times')]),
          ('3  fully automatic', 'box-3', ['SAM alone: 32 × 32 points,', 'filters, NMS, crops'],
           [('images', '11M'), ('new masks', '1.1B'), ('time per mask', 'no person'), ('masks per image', 'about 100'),
            ('model retrained', 'no more')])]
    b = [text(20, 22, 'The data engine: three stages, one model that improves on the way', 't-title')]
    w, gap = 236, 16
    for i, (t, cls, d, rows) in enumerate(st):
        x = 20 + i * (w + gap)
        b += [box(x, 38, w, 262, cls, 10), text(x + 14, 62, t, 't-note')]
        b += [text(x + 14, 84, d[0], 't-muted'), text(x + 14, 100, d[1], 't-muted')]
        for k, (lab, v) in enumerate(rows):
            y = 132 + k * 34
            b += [line(x + 12, y - 16, x + w - 12, y - 16, 'grid'), text(x + 14, y, lab, 't-tick'), text(x + w - 14, y, v, 't-val', 'end')]
        if i < 2:
            b.append(arrow(x + w + 2, 170, x + w + gap - 2, 170))
    b.append(text(20, 326, 'Stages 1 and 2 give 10.2M masks from people; stage 3 gives 1.1B. SA-1B keeps only the masks of stage 3 (99.1% of all).', 't-muted'))
    return svg(780, 340, 'The three stages of the data engine. Stage 1, assisted-manual: 120 thousand images, 4.3 million masks, 34 to 14 seconds per mask, 20 to 44 masks per image, 6 retrains. Stage 2, semi-automatic: 180 thousand images, 5.9 million new masks, 34 seconds per mask, 44 to 72 masks per image, 5 retrains. Stage 3, fully automatic: 11 million images, 1.1 billion masks, about 100 per image.', b)


F['p4_engine'] = engine()


# ---------------------------------------------------------------- 2. the stability score on 10 pixels
def stab_toy():
    T = R['toy']
    v = T['logits']
    X0, Y0, BW, H = 70, 40, 60, 210
    lo, hi = -4.5, 4.0
    Y = lambda u: Y0 + H * (hi - u) / (hi - lo)
    b = [text(20, 22, 'Ten pixels of one mask: the logit of each pixel, and three cut-offs', 't-title')]
    for c, lab, cls in [(1, 'cut at +1', 'edge-2'), (0, 'cut at 0: the mask', 'edge-dim'), (-1, 'cut at −1', 'edge-1')]:
        b += [line(X0 - 10, Y(c), X0 + 10 * BW + 10, Y(c), cls), text(X0 + 10 * BW + 16, Y(c) + 4, lab, 't-tick')]
    b.append(line(X0 - 10, Y(0), X0 - 10, Y(0), 'axis'))
    for i, u in enumerate(v):
        x = X0 + i * BW + 8
        y0, y1 = (Y(u), Y(0)) if u >= 0 else (Y(0), Y(u))
        cls = 's2' if u > 1 else ('s1' if u > -1 else 's4')
        op = '' if u > -1 else ' style="fill-opacity:0.35"'
        b.append(f'<g class="mark"><title>pixel {i + 1}: logit {u:+.1f}</title><rect class="{cls}" x="{x}" y="{y0:.1f}" width="{BW - 16}" height="{max(1.5, y1 - y0):.1f}" rx="3"{op}/></g>')
        b.append(text(x + (BW - 16) / 2, Y(u) + (-6 if u >= 0 else 16), f'{u:+.1f}', 't-tick', 'middle'))
    b.append(text(X0 - 14, Y(hi) + 4, 'logit', 't-muted', 'end'))
    yb = Y0 + H + 34
    b += [text(20, yb, f'above +1: {T["hi"]} pixels (orange)     above 0: {T["zero"]} pixels     above −1: {T["lo"]} pixels (orange + blue)', 't-note')]
    b += [text(20, yb + 24, f'stability = {T["hi"]} / {T["lo"]} = {T["stab"]:.3f}: under 0.95, so this mask would be rejected', 't-on')]
    return svg(800, yb + 40, 'Ten pixels of one mask with their logits. Three pixels are above plus 1, five are above 0 and seven are above minus 1. The stability score is 3 divided by 7, 0.429.', b)


F['p4_stab_toy'] = stab_toy()


# ---------------------------------------------------------------- 3. predicted IoU against stability for the 3,072 candidates
def scatter():
    P = np.array(R['scatter'])
    xs_lo, xs_hi, ys_lo, ys_hi = 0.4, 1.05, 0.0, 1.0
    nx, ny = 26, 20
    H2, _, _ = np.histogram2d(P[:, 0].clip(xs_lo, xs_hi - 1e-6), P[:, 1].clip(ys_lo, ys_hi - 1e-6), bins=[nx, ny],
                              range=[[xs_lo, xs_hi], [ys_lo, ys_hi]])
    L, T, PW, PH = 80, 50, 520, 300
    cw, ch = PW / nx, PH / ny
    X = lambda u: L + PW * (u - xs_lo) / (xs_hi - xs_lo)
    Y = lambda u: T + PH - PH * (u - ys_lo) / (ys_hi - ys_lo)
    mx = H2.max()
    b = [text(20, 22, f'The {len(P):,} candidate masks of the cat photo: predicted IoU against stability', 't-title')]
    for i in range(nx):
        for j in range(ny):
            n = H2[i, j]
            if n:
                op = 0.12 + 0.88 * math.sqrt(n / mx)
                b.append(f'<g class="mark"><title>{int(n)} masks</title><rect class="cell" x="{L + i * cw + 0.5:.1f}" y="{T + PH - (j + 1) * ch + 0.5:.1f}" '
                         f'width="{cw - 1:.1f}" height="{ch - 1:.1f}" style="fill-opacity:{op:.2f}"/></g>')
    b += [f'<rect class="box-on" x="{X(0.88):.1f}" y="{Y(1.0):.1f}" width="{X(xs_hi) - X(0.88):.1f}" height="{Y(0.95) - Y(1.0):.1f}" rx="2" style="fill-opacity:0.12"/>']
    b += [line(X(0.88), T, X(0.88), T + PH, 'edge-2'), line(L, Y(0.95), L + PW, Y(0.95), 'edge-2')]
    for t_ in [0.4, 0.6, 0.8, 1.0]:
        b.append(text(X(t_), T + PH + 18, f'{t_:.1f}', 't-tick', 'middle'))
    for t_ in [0, 0.25, 0.5, 0.75, 1.0]:
        b.append(text(L - 8, Y(t_) + 4, f'{t_:.2f}', 't-tick', 'end'))
    b.append(f'<line class="axis" x1="{L}" y1="{T + PH}" x2="{L + PW}" y2="{T + PH}"/>')
    b.append(text(L + PW / 2, T + PH + 40, 'predicted IoU (the IoU head)', 't-tick', 'middle'))
    b.append(text(L + 6, T - 10, 'stability', 't-tick'))
    c = R['cats']
    xx = L + PW + 24
    b += [text(xx, 70, 'kept by both filters', 't-on'), text(xx, 88, f'(top right box): {c["pass_iou_stab"]}', 't-tick'),
          text(xx, 124, 'orange lines:', 't-note'), text(xx, 142, 'predicted IoU 0.88', 't-tick'), text(xx, 160, 'stability 0.95', 't-tick'),
          text(xx, 196, 'darker cell = more masks', 't-muted'), text(xx, 214, 'IoU under 0.4 drawn at 0.4', 't-muted')]
    return svg(800, T + PH + 52, f'A 2D histogram of the {len(P)} candidate masks of the cat photo, predicted IoU on the x axis and stability on the y axis. Only the top right corner, predicted IoU above 0.88 and stability at least 0.95, is kept: {c["pass_iou_stab"]} masks.', b)


F['p4_scatter'] = scatter()


# ---------------------------------------------------------------- 4. the funnel: how many masks survive each step
def funnel():
    rows_c = R['cats']
    rows_k = R['car_crops']
    steps = [('candidates (points × 3)', 'candidates'), ('predicted IoU > 0.88', 'pass_iou'), ('stability ≥ 0.95', 'pass_iou_stab'),
             ('covers < 95% of the picture', 'pass_big'), ('not at an inner crop edge', 'pass_edge'), ('NMS inside each crop', 'after_nms_in_crops'),
             ('NMS across crops', 'after_nms_across'), ('clean-up, NMS again: final', 'final')]
    b = [text(20, 22, 'How many masks survive each step', 't-title')]
    cols = [('cat photo, full picture only', rows_c, 's1', 300), ('car photo, full picture + 20 crops', rows_k, 's3', 560)]
    for t_, rows, cls, x0 in cols:
        b.append(text(x0, 48, t_, 't-note'))
    for k, (lab, key) in enumerate(steps):
        y = 64 + k * 34
        b.append(text(280, y + 17, lab, 't-tick', 'end'))
        for t_, rows, cls, x0 in cols:
            v = rows[key]
            w = max(2, 190 * math.sqrt(v / rows['candidates']))
            b.append(f'<g class="mark"><title>{esc(t_)}, {esc(lab)}: {v:,}</title><rect class="{cls}" x="{x0}" y="{y + 4}" width="{w:.1f}" height="20" rx="3"/></g>')
            b.append(text(x0 + w + 6, y + 19, f'{v:,}', 't-val'))
    y = 64 + len(steps) * 34 + 10
    b.append(text(20, y, 'Bar length is the square root of the count, so the small final numbers stay visible. The cat photo has no crops,', 't-muted'))
    b.append(text(20, y + 16, 'so its crop steps change nothing.', 't-muted'))
    return svg(800, y + 28, f'The number of masks after each step of the fully automatic stage. Cat photo: {rows_c["candidates"]} candidates, {rows_c["final"]} final masks. Car photo with crops: {rows_k["candidates"]} candidates, {rows_k["final"]} final masks.', b)


F['p4_funnel'] = funnel()


# ---------------------------------------------------------------- 5. relative mask size and concavity on our sample
def sizes():
    rel = R['stats']['cats']['rel'] + R['stats']['car']['rel']
    con = R['stats']['cats']['con'] + R['stats']['car']['con']
    edges = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    hr = np.histogram(rel, bins=edges)[0]
    hc = np.histogram(con, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.6, 1.0])[0]
    b = [text(20, 22, f'Our {len(rel)} final masks (two photos): a tiny sample, not SA-1B', 't-title')]

    def bars(x0, title, labs, vals, cls):
        out = [text(x0, 52, title, 't-note')]
        mx = max(vals)
        for i, (lab, v) in enumerate(zip(labs, vals)):
            x = x0 + i * 50
            h = 170 * v / mx
            out.append(f'<g class="mark"><title>{esc(lab)}: {v}</title><rect class="{cls}" x="{x}" y="{240 - h:.1f}" width="40" height="{max(h, 1.5):.1f}" rx="3"/></g>')
            out.append(text(x + 20, 234 - h, str(v), 't-val', 'middle'))
            out.append(text(x + 20, 256, lab, 't-tick', 'middle'))
        out.append(f'<line class="axis" x1="{x0 - 4}" y1="240" x2="{x0 + 50 * len(vals)}" y2="240"/>')
        return out

    b += bars(20, 'relative size = √(mask area / picture area)', ['<.05', '.05-.1', '.1-.2', '.2-.3', '.3-.5', '.5-.75', '>.75'], hr.tolist(), 's1')
    b += bars(420, 'concavity = 1 − area / convex hull area', ['<.1', '.1-.2', '.2-.3', '.3-.4', '.4-.6', '>.6'], hc.tolist(), 's3')
    b.append(text(20, 284, 'Most masks are small and nearly convex, the same tendency as the SA-1B curves of Figure 6.', 't-muted'))
    return svg(780, 296, 'Histograms of our final masks. Left: relative size, the square root of the mask area divided by the picture area; most masks are under 0.2. Right: concavity; most masks are under 0.1.', b)


F['p4_sizes'] = sizes()


# ---------------------------------------------------------------- 6. Table 1 as bars
def table1():
    T = R['table1'][:5]
    b = [text(20, 22, 'Share of images from each region (Table 1 of the paper)', 't-title')]
    L, W = 190, 520
    mx = 50
    for t_ in [0, 10, 20, 30, 40, 50]:
        x = L + W * t_ / mx
        b += [line(x, 40, x, 40 + len(T) * 64, 'grid'), text(x, 40 + len(T) * 64 + 16, f'{t_}%', 't-tick', 'middle')]
    for i, row in enumerate(T):
        y = 44 + i * 64
        b.append(text(L - 10, y + 30, row[0], 't-tick', 'end'))
        for k, (v, cls, nm) in enumerate([(row[4], 's1', 'SA-1B'), (row[5], 's2', 'COCO'), (row[6], 's4', 'Open Images')]):
            w = W * v / mx
            b.append(f'<g class="mark"><title>{esc(row[0])}, {nm}: {v}%</title><rect class="{cls}" x="{L}" y="{y + k * 18}" width="{max(w, 2):.1f}" height="15" rx="3"/></g>')
            b.append(text(L + max(w, 2) + 6, y + k * 18 + 12, f'{v}%', 't-tick'))
    b += legend(L, 40 + len(T) * 64 + 42, [('s1', 'SA-1B'), ('s2', 'COCO'), ('s4', 'Open Images')])
    return svg(780, 40 + len(T) * 64 + 56, 'Share of images by region. SA-1B: Africa 2.8, Asia and Oceania 36.2, Europe 49.8, Latin America 3.5, North America 7.7 percent. COCO and Open Images have 48.3 and 42.8 percent from North America.', b)


F['p4_table1'] = table1()


# ---------------------------------------------------------------- 7. Tables 2 and 6: confidence intervals
def fairness():
    T = R['table2']
    rows = [('people', g) for g in T['people']] + [('clothing', g) for g in T['clothing']]
    L, W, lo, hi = 230, 480, 45, 90
    X = lambda v: L + W * (v - lo) / (hi - lo)
    b = [text(20, 22, 'mIoU with 1 point and its 95% confidence interval (Tables 2 and 6)', 't-title')]
    y = 52
    for t_ in range(45, 91, 5):
        b.append(text(X(t_), 52 + 16 * 22 + 50, str(t_), 't-tick', 'middle'))
    for t_ in range(45, 91, 5):
        b.append(line(X(t_), 40, X(t_), 52 + 16 * 22 + 34, 'grid'))
    last = None
    for tab, (nm, m, e, _, _) in rows:
        if tab != last:
            b.append(text(20, y + 4, 'segmenting people' if tab == 'people' else 'segmenting clothing (Appendix C)', 't-note'))
            y += 22
            last = tab
        hl = (tab == 'clothing' and nm in ('feminine', 'masculine')) or (tab == 'people' and nm in ('older', 'middle'))
        b.append(text(L - 12, y + 4, nm, 't-tick t-strong' if hl else 't-tick', 'end'))
        cls = 'edge-2' if hl else 'edge-1'
        b += [line(X(m - e), y, X(m + e), y, cls), line(X(m - e), y - 5, X(m - e), y + 5, cls), line(X(m + e), y - 5, X(m + e), y + 5, cls),
              f'<g class="mark"><title>{tab}, {nm}: {m} ± {e}</title><circle class="{"s2" if hl else "s1"} ring" cx="{X(m):.1f}" cy="{y}" r="5"/></g>',
              text(X(m + e) + 8, y + 4, f'{m} ± {e}', 't-tick')]
        y += 22
    b.append(text(L + W / 2, 52 + 16 * 22 + 70, 'mIoU (%) with one random point; orange: the pairs whose intervals do not overlap', 't-muted', 'middle'))
    return svg(780, 52 + 16 * 22 + 84, 'Mean IoU with one point and 95 percent confidence intervals, from Tables 2 and 6. For people, only older (62.9 plus or minus 6.7) and middle age (54.5 plus or minus 1.3) do not overlap. For clothing, feminine (76.3 plus or minus 1.1) and masculine (81.0 plus or minus 1.2) do not overlap.', b)


F['p4_fairness'] = fairness()

json.dump(F, open(os.path.join(HERE, 'results', 'figs_part4.json'), 'w'))
print(f'{len(F)} figures:', ', '.join(F))
