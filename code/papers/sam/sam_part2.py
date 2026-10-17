"""Part 2: the promptable segmentation task (paper Section 2 and Figure 3), on the cat photo of Part 1.
- one click, three nested masks (subpart, part, whole), as in Figure 3: areas, scores, how much each mask sits inside the next
- "valid": the three masks of one click compared with two reference objects; one of the three is not valid
- one output for an ambiguous click (multimask_output=False), and a toy pixel that shows why one output averages
- a second point: positive or negative
- a box and a point on the same head
- the pre-training simulation of Appendix A: a first point or box, 8 points from the error region, the previous
  mask logits fed back as a mask prompt, 2 rounds with no new point: 11 rounds, IoU against a reference after each
- noisy boxes, with the noise of the paper's training (10% of the side, at most 20 pixels), and larger noise
Writes results/part2.json, results/part2_stdout.txt, results/part2_*.npy and the pictures public/img/papers/sam/p2-*.png."""
import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset
from transformers import SamModel, SamProcessor
from common import Log, save

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
IMG = os.path.join(HERE, '..', '..', '..', 'public', 'img', 'papers', 'sam')
os.makedirs(IMG, exist_ok=True)
log = Log('part2')
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
R = {}

image = load_dataset('huggingface/cats-image')['test'][0]['image'].convert('RGB')
W0, H0 = image.size
N_PIX = W0 * H0
proc = SamProcessor.from_pretrained('facebook/sam-vit-huge')
sam = SamModel.from_pretrained('facebook/sam-vit-huge').to(DEV).eval()
inp = proc(images=image, return_tensors='pt')
with torch.no_grad():
    emb = sam.get_image_embeddings(inp['pixel_values'].to(DEV))
log(f'picture: {W0} x {H0} = {N_PIX:,} pixels; SAM ViT-H; image embedding {tuple(emb.shape)} computed once')


def predict(points=None, labels=None, box=None, mask_logits=None, multimask=True):
    """One call of the prompt encoder + mask decoder. Returns full-size masks (bool), scores, low-res logits (k, 256, 256)."""
    kw = {}
    if points is not None:
        kw['input_points'] = [[points]]
        kw['input_labels'] = [[labels]]
    if box is not None:
        kw['input_boxes'] = [[box]]
    p = proc(images=image, return_tensors='pt', **kw)
    args = {k: p[k].float().to(DEV) if k != 'input_labels' else p[k].to(DEV) for k in ('input_points', 'input_labels', 'input_boxes') if k in p}
    if mask_logits is not None:
        args['input_masks'] = torch.as_tensor(mask_logits, dtype=torch.float32).reshape(1, 1, 256, 256).to(DEV)
    with torch.no_grad():
        out = sam(image_embeddings=emb, multimask_output=multimask, **args)
    low = out.pred_masks[0, 0].cpu()
    masks = proc.image_processor.post_process_masks(out.pred_masks.cpu(), p['original_sizes'], p['reshaped_input_sizes'])[0][0]
    return masks.numpy(), out.iou_scores[0, 0].cpu().numpy(), low.numpy()


def iou(a, b):
    return float((a & b).sum() / max(1, (a | b).sum()))


def inside(a, b):
    """Fraction of mask a that is also in mask b."""
    return float((a & b).sum() / max(1, a.sum()))


def pct(n):
    return 100 * n / N_PIX


# ------------------------------------------------------------------ 1. one click, three nested masks (Figure 3)
log('\n== 1. one click, three masks (our Figure 3) ==')
CLICKS = [('tail', (180, 305), 'the dark tail of the left cat', ['tail', 'tail + back leg', 'whole cat']),
          ('paw', (565, 290), 'a back paw of the right cat', ['paw', 'paw + lower leg', 'whole cat']),
          ('remote', (352, 120), 'the right remote control', ['a few buttons', 'part of the remote', 'whole remote'])]
R['clicks'] = {}
for key, (x, y), what, names in CLICKS:
    m, s, _ = predict([[x, y]], [1])
    log(f'click at ({x}, {y}) on {what}:')
    rows = []
    for k in range(3):
        n = int(m[k].sum())
        rows.append(dict(name=names[k], pixels=n, percent=pct(n), score=float(s[k])))
        log(f'  mask {k + 1} ({names[k]:<18}) {n:>7,} pixels = {pct(n):5.2f}%   predicted IoU {s[k]:.3f}')
    nest = [inside(m[0], m[1]), inside(m[1], m[2])]
    pair = {'1-2': iou(m[0], m[1]), '1-3': iou(m[0], m[2]), '2-3': iou(m[1], m[2])}
    log(f'  nested: {100 * nest[0]:.1f}% of mask 1 is inside mask 2, {100 * nest[1]:.1f}% of mask 2 is inside mask 3')
    log('  IoU between the masks: ' + ', '.join(f'{k}: {v:.3f}' for k, v in pair.items()))
    R['clicks'][key] = dict(point=[x, y], masks=rows, inside=nest, pair_iou=pair, best=int(np.argmax(s)) + 1)
    np.save(os.path.join(RES, f'part2_{key}_masks.npy'), m)
t = R['clicks']['tail']['masks']
log(f'tail click: the masks grow {t[1]["pixels"] / t[0]["pixels"]:.1f}x and then {t[2]["pixels"] / t[1]["pixels"]:.1f}x')

# ------------------------------------------------------------------ 2. valid or not valid
log('\n== 2. valid or not: a click on the right cat\'s head ==')
HEAD_PT = (400, 325)
HEAD_BOX = [350, 262, 475, 362]
mh, sh, low_head = predict([list(HEAD_PT)], [1])
rb, s_rb, _ = predict(box=HEAD_BOX, multimask=True)
ref_head, s_rh = rb[int(np.argmax(s_rb))], s_rb[int(np.argmax(s_rb))]
ref_cat = np.load(os.path.join(RES, 'part1_box_mask.npy'))       # Part 1: box (255, 20)-(640, 360) around the right cat
log(f'reference "head": box {HEAD_BOX}, the best of its 3 masks, {int(ref_head.sum()):,} pixels (predicted IoU {s_rh:.3f})')
log(f'reference "cat":  Part 1 box mask of the right cat, {int(ref_cat.sum()):,} pixels')
log(f'click at {HEAD_PT}:')
R['valid'] = dict(point=list(HEAD_PT), head_box=HEAD_BOX, ref_head=int(ref_head.sum()), ref_cat=int(ref_cat.sum()), masks=[])
for k in range(3):
    a, b = iou(mh[k], ref_head), iou(mh[k], ref_cat)
    best = max(a, b)
    R['valid']['masks'].append(dict(pixels=int(mh[k].sum()), score=float(sh[k]), iou_head=a, iou_cat=b, best=best))
    log(f'  mask {k + 1}: {int(mh[k].sum()):>7,} pixels, predicted IoU {sh[k]:.3f}   IoU with head {a:.3f}, with cat {b:.3f}   '
        f'-> best {best:.3f} ({"head" if a > b else "cat"})')
m3 = mh[2]
log(f'  mask 3 covers {pct(int(m3.sum())):.1f}% of the picture; {100 * inside(ref_cat, m3):.1f}% of the cat is inside it, '
    f'and {int((m3 & ~ref_cat).sum()):,} of its pixels are not the cat')
np.save(os.path.join(RES, 'part2_head_masks.npy'), mh)
np.save(os.path.join(RES, 'part2_ref_head.npy'), ref_head)

# ------------------------------------------------------------------ 3. one output for an ambiguous click
log('\n== 3. one output (multimask_output=False) for the same clicks ==')
R['single'] = {}
for key, (x, y), what, names in CLICKS + [('head', HEAD_PT, 'the right cat\'s head', ['head', 'whole cat', 'cat + blanket'])]:
    m1, s1, low1 = predict([[x, y]], [1], multimask=False)
    ms = np.load(os.path.join(RES, f'part2_{key}_masks.npy'))
    ious = [iou(m1[0], ms[k]) for k in range(3)]
    p = 1 / (1 + np.exp(-np.clip(low1[0], -60, 60)))
    unsure = int(((p > 0.25) & (p < 0.75)).sum())
    R['single'][key] = dict(pixels=int(m1[0].sum()), score=float(s1[0]), iou_with_three=ious, unsure_lowres=unsure)
    log(f'{key:<7} 1 mask: {int(m1[0].sum()):>7,} pixels, predicted IoU {s1[0]:.3f}; IoU with masks 1/2/3: '
        + ' / '.join(f'{v:.3f}' for v in ious) + f'; low-res pixels with probability 0.25-0.75: {unsure:,} of 65,536')
for key in ['tail', 'paw', 'remote']:
    _, _, low3 = predict([list(R['clicks'][key]['point'])], [1])
    p3 = 1 / (1 + np.exp(-np.clip(low3, -60, 60)))
    R['clicks'][key]['unsure_lowres'] = [int(((q > 0.25) & (q < 0.75)).sum()) for q in p3]
    log(f'{key:<7} 3 masks: low-res pixels with probability 0.25-0.75: ' + ' / '.join(f'{v:,}' for v in R['clicks'][key]['unsure_lowres']))

# a toy pixel: on the cat, but not on the tail. Half of the people wanted the tail (answer 0), half the cat (answer 1)
log('\ntoy pixel on the cat but not on the tail; the right answer is 1 for a fraction q of the people (they want the cat), else 0')
log('expected loss of one output with probability p:  L(p) = q * (-ln p) + (1 - q) * (-ln(1 - p))')
log(f'  the two penalties: -ln(0.99) = {-np.log(0.99):.3f}, -ln(0.01) = {-np.log(0.01):.3f}, -ln(0.5) = {-np.log(0.5):.3f}')
R['toy'] = []
for q in [0.5, 0.8]:
    row = {}
    for p in [0.01, 0.5, q, 0.99]:
        L = -(q * np.log(p) + (1 - q) * np.log(1 - p))
        row[str(p)] = float(L)
    best_p = min(np.linspace(0.01, 0.99, 99), key=lambda p: -(q * np.log(p) + (1 - q) * np.log(1 - p)))
    R['toy'].append(dict(q=q, loss=row, best_p=float(best_p)))
    log(f'  q = {q}: ' + ', '.join(f'L({p}) = {v:.3f}' for p, v in row.items()) + f'; the smallest loss is at p = {best_p:.2f}')

# ------------------------------------------------------------------ 4. a second point: positive or negative
log('\n== 4. a second point after the tail click ==')
BODY = (100, 200)
R['second'] = {}
for lab, word in [(1, 'positive (foreground)'), (0, 'negative (background)')]:
    m2, s2, _ = predict([[180, 305], list(BODY)], [1, lab], multimask=False)
    tm = np.load(os.path.join(RES, 'part2_tail_masks.npy'))
    ious = [iou(m2[0], tm[k]) for k in range(3)]
    R['second'][str(lab)] = dict(pixels=int(m2[0].sum()), score=float(s2[0]), iou_with_three=ious)
    log(f'tail (180, 305) + {word} point at {BODY}: 1 mask, {int(m2[0].sum()):,} pixels, predicted IoU {s2[0]:.3f}; '
        f'IoU with tail mask 1 / 2 / 3: ' + ' / '.join(f'{v:.3f}' for v in ious))
    np.save(os.path.join(RES, f'part2_second_{lab}.npy'), m2[0])

# ------------------------------------------------------------------ 5. a point and a box on the same tail
log('\n== 5. a point and a box on the tail ==')
TAIL_BOX = [140, 285, 220, 345]
tm = np.load(os.path.join(RES, 'part2_tail_masks.npy'))
ts = [r['score'] for r in R['clicks']['tail']['masks']]
mb3, sb3, _ = predict(box=TAIL_BOX, multimask=True)
mb1, sb1, _ = predict(box=TAIL_BOX, multimask=False)
log('point (180, 305), 3 masks: ' + ', '.join(f'{int(tm[k].sum()):,} px ({ts[k]:.3f})' for k in range(3)))
log(f'box {TAIL_BOX}, 3 masks: ' + ', '.join(f'{int(mb3[k].sum()):,} px ({sb3[k]:.3f})' for k in range(3)))
spread_pt = max(int(m.sum()) for m in tm) / min(int(m.sum()) for m in tm)
spread_bx = max(int(m.sum()) for m in mb3) / min(int(m.sum()) for m in mb3)
log(f'largest / smallest mask: point {spread_pt:.1f}x, box {spread_bx:.2f}x; IoU of box masks 1-2 {iou(mb3[0], mb3[1]):.3f}, '
    f'1-3 {iou(mb3[0], mb3[2]):.3f}, 2-3 {iou(mb3[1], mb3[2]):.3f}')
log(f'box, one mask: {int(mb1[0].sum()):,} px ({sb1[0]:.3f}); IoU with the tail mask of the point: {iou(mb1[0], tm[0]):.3f}')
R['point_box'] = dict(box_xyxy=TAIL_BOX, point=[dict(pixels=int(tm[k].sum()), score=ts[k]) for k in range(3)],
                      box=[dict(pixels=int(mb3[k].sum()), score=float(sb3[k])) for k in range(3)],
                      spread_point=spread_pt, spread_box=spread_bx, box_pair_iou=[iou(mb3[0], mb3[1]), iou(mb3[0], mb3[2]), iou(mb3[1], mb3[2])],
                      box_single=int(mb1[0].sum()), box_single_score=float(sb1[0]), box_single_iou_tail=iou(mb1[0], tm[0]))

# ------------------------------------------------------------------ 6. the pre-training simulation (Appendix A)
log('\n== 6. the training simulation of the paper, with the left cat as the target ==')
log('round 1: a point (or a box) from the target; rounds 2-9: one new point each, from the error region;')
log('one extra round with no new point at a random place, and one at the end; the previous logits go back in as a mask prompt')
TARGET = np.load(os.path.join(RES, 'part1_ear_masks.npy'))[2]          # Part 1: the whole left cat (ear click, mask 3)
TB = np.argwhere(TARGET)
tbox = [int(TB[:, 1].min()), int(TB[:, 0].min()), int(TB[:, 1].max()), int(TB[:, 0].max())]


def noisy_box(b, rng, frac=0.1, cap=20):
    w, h = b[2] - b[0], b[3] - b[1]
    sd = np.array([w, h, w, h]) * frac
    noise = np.clip(rng.normal(0, sd), -cap, cap) if cap else rng.normal(0, sd)
    return [float(v) for v in np.array(b) + noise]


def simulate(seed, first=None, use_mask=True, verbose=False, keep=None, start=None):
    rng = np.random.default_rng(seed)
    kind = first or ('point' if rng.random() < 0.5 else 'box')
    pts, labs, box = [], [], None
    if kind == 'point':
        y, x = TB[rng.integers(len(TB))]
        pts, labs = [[int(x), int(y)]], [1]
        if start is not None:
            pts = [list(start)]
    else:
        box = noisy_box(tbox, rng)
    empty_at = int(rng.integers(2, 10))                # one no-new-point round among the 8 point rounds
    plan = ['first'] + ['point'] * 8
    plan.insert(empty_at, 'none')
    plan.append('none')
    logits, rows = None, []
    for r, step in enumerate(plan, 1):
        info = ''
        if step == 'point':
            fn, fp = TARGET & ~mask, mask & ~TARGET
            err = np.argwhere(fn | fp)
            y, x = err[rng.integers(len(err))]
            lab = 1 if fn[y, x] else 0
            pts.append([int(x), int(y)]); labs.append(lab)
            info = (f'{"+" if lab else "-"} point ({x}, {y}); error region {int(fn.sum()):,} missed + {int(fp.sum()):,} extra, '
                    f'P(+) = {fn.sum() / (fn.sum() + fp.sum()):.3f}')
        elif step == 'none':
            info = 'no new point, only the previous mask'
        else:
            info = f'first prompt: {kind} ' + (f'({pts[0][0]}, {pts[0][1]})' if kind == 'point' else '(' + ', '.join(f'{v:.0f}' for v in box) + ')')
        n_prompts = len(pts) + (1 if box is not None else 0)
        multi = n_prompts == 1                          # one prompt: 3 masks, keep the best score; more prompts: 1 mask
        m, s, low = predict(pts or None, labs or None, box, logits if (use_mask and r > 1) else None, multimask=multi)
        k = int(np.argmax(s)) if multi else 0
        mask, logits = m[k], low[k]
        rows.append(dict(round=r, step=step, iou=iou(mask, TARGET), score=float(s[k]), pixels=int(mask.sum()),
                         points=len(pts), info=info))
        if keep is not None:
            keep.append((mask.copy(), [q[:] for q in pts], labs[:]))
        if verbose:
            log(f'  round {r:>2}: {info:<80} -> {int(mask.sum()):>7,} px, IoU {rows[-1]["iou"]:.3f} (score {s[k]:.3f})')
    return rows


log(f'target: the whole left cat (Part 1, ear click, mask 3), {int(TARGET.sum()):,} pixels; its tight box {tbox}')
log('one run, first prompt the tail click of section 1 (seed 0 for the later points):')
SIM_MASKS = []
detail = simulate(0, first='point', verbose=True, keep=SIM_MASKS, start=(180, 305))
R['sim_detail'] = detail
SEEDS = range(20)
runs = [simulate(s) for s in SEEDS]
runs_nomask = [simulate(s, use_mask=False) for s in SEEDS]
mean = [float(np.mean([r[i]['iou'] for r in runs])) for i in range(11)]
mean_nm = [float(np.mean([r[i]['iou'] for r in runs_nomask])) for i in range(11)]
low_ = [float(np.min([r[i]['iou'] for r in runs])) for i in range(11)]
n_box = sum(1 for s in SEEDS if np.random.default_rng(s).random() >= 0.5)
log(f'20 runs (seeds 0-19; {20 - n_box} start with a point, {n_box} with a noisy box). Mean IoU after each round:')
log('  round:            ' + ''.join(f'{i:>7}' for i in range(1, 12)))
log('  with mask prompt: ' + ''.join(f'{v:7.3f}' for v in mean))
log('  without:          ' + ''.join(f'{v:7.3f}' for v in mean_nm))
log('  worst run (with): ' + ''.join(f'{v:7.3f}' for v in low_))
first_pt = [r[0]['iou'] for r, s in zip(runs, SEEDS) if np.random.default_rng(s).random() < 0.5]
first_bx = [r[0]['iou'] for r, s in zip(runs, SEEDS) if np.random.default_rng(s).random() >= 0.5]
log(f'round 1 alone: mean IoU {np.mean(first_pt):.3f} when it is a point, {np.mean(first_bx):.3f} when it is a box')
R['sim'] = dict(target_pixels=int(TARGET.sum()), tight_box=tbox, mean=mean, mean_nomask=mean_nm, worst=low_,
                n_box=n_box, first_point=float(np.mean(first_pt)), first_box=float(np.mean(first_bx)))


# ------------------------------------------------------------------ 7. noisy boxes: the detector box as a prompt
log('\n== 7. noisy boxes around the right cat ==')
CB = np.argwhere(ref_cat)
cbox = [int(CB[:, 1].min()), int(CB[:, 0].min()), int(CB[:, 1].max()), int(CB[:, 0].max())]
mt, st, _ = predict(box=cbox, multimask=False)
ref_tight = mt[0]
log(f'tight box of the Part 1 cat mask: {cbox} (width {cbox[2] - cbox[0]}, height {cbox[3] - cbox[1]}); '
    f'its mask: {int(ref_tight.sum()):,} px, score {st[0]:.3f}, IoU with the Part 1 mask {iou(ref_tight, ref_cat):.3f}')
R['boxes'] = dict(tight=cbox, tight_pixels=int(ref_tight.sum()), rows=[])
for frac, cap, name in [(0.1, 20, '10% of the side, at most 20 px (training)'), (0.1, None, '10% of the side, no limit'),
                        (0.2, None, '20% of the side, no limit'), (0.4, None, '40% of the side, no limit')]:
    rng = np.random.default_rng(0)
    vals, shifts, worst = [], [], None
    for _ in range(30):
        b = noisy_box(cbox, rng, frac, cap)
        shifts.append(float(np.mean(np.abs(np.array(b) - cbox))))
        mb, sb, _ = predict(box=b, multimask=False)
        vals.append(iou(mb[0], ref_tight))
        if worst is None or vals[-1] < worst['iou']:
            worst = dict(box=[round(v, 1) for v in b], iou=vals[-1], pixels=int(mb[0].sum()), score=float(sb[0]))
    row = dict(frac=frac, cap=cap, name=name, mean=float(np.mean(vals)), min=float(np.min(vals)),
               above90=int(np.sum(np.array(vals) > 0.9)), shift=float(np.mean(shifts)))
    row['worst'] = worst
    R['boxes']['rows'].append(row)
    log(f'  noise sd {name:<44} mean corner shift {row["shift"]:5.1f} px: mean IoU {row["mean"]:.3f}, worst {row["min"]:.3f}, '
        f'{row["above90"]}/30 above 0.9')
    log(f'      worst box ({", ".join(f"{v:.0f}" for v in worst["box"])}): {worst["pixels"]:,} px, score {worst["score"]:.3f}')
w, h = cbox[2] - cbox[0], cbox[3] - cbox[1]
log(f'worked example: sd for x = 0.1 x {w} = {0.1 * w:.1f} px, for y = 0.1 x {h} = {0.1 * h:.1f} px; both are above the 20 px limit')
save('part2', R)


# ------------------------------------------------------------------ 8. pictures for the article
def font(size):
    for f in ['/System/Library/Fonts/Helvetica.ttc', '/System/Library/Fonts/Menlo.ttc']:
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


COLS = [(255, 196, 0), (64, 200, 255), (60, 90, 255)]


def overlay(mask, col, a=0.55, base=None):
    arr = np.array(base or image).astype(float)
    arr[mask] = (1 - a) * arr[mask] + a * np.array(col)
    e = arr.astype('uint8')
    edge = mask ^ np.pad(mask, 1, mode='edge')[:-2, 1:-1] | mask ^ np.pad(mask, 1, mode='edge')[1:-1, :-2]
    e[edge] = (255, 255, 255)
    return Image.fromarray(e)


def dot(im, x, y, r=7, pos=True):
    d = ImageDraw.Draw(im)
    d.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2], fill=(255, 255, 255))
    d.ellipse([x - r, y - r, x + r, y + r], fill=(40, 220, 90) if pos else (235, 50, 50))
    return im


def grid(tiles, labels, path, cols, scale=1.0, cap=58):
    w, h = tiles[0].size
    w, h = int(w * scale), int(h * scale)
    rows = (len(tiles) + cols - 1) // cols
    out = Image.new('RGB', (cols * w + (cols - 1) * 12, rows * (h + cap) + (rows - 1) * 10), (24, 25, 33))
    d = ImageDraw.Draw(out)
    for i, (t, (l1, l2)) in enumerate(zip(tiles, labels)):
        x, y = (i % cols) * (w + 12), (i // cols) * (h + cap + 10)
        out.paste(t.resize((w, h), Image.LANCZOS), (x, y))
        d.text((x + 10, y + h + 7), l1, font=font(20), fill=(235, 236, 245))
        d.text((x + 10, y + h + 32), l2, font=font(17), fill=(160, 162, 180))
    out.save(path, optimize=True)
    print('wrote', path, out.size)


# Figure 3 on our photo: one column per click, one row per mask (crops around each click)
CROPS = {'tail': (20, 150, 340, 480), 'paw': (330, 20, 640, 360), 'remote': (250, 40, 450, 240)}
tiles, labels = [], []
for k in range(3):
    for key, (x, y), what, names in CLICKS:
        ms = np.load(os.path.join(RES, f'part2_{key}_masks.npy'))
        c = CROPS[key]
        t = dot(overlay(ms[k], COLS[k]), x, y, 6).crop(c).resize((300, 300 * (c[3] - c[1]) // (c[2] - c[0])))
        t = t.resize((300, 300))
        r = R['clicks'][key]['masks'][k]
        tiles.append(t)
        labels.append((f'mask {k + 1}: {names[k]}', f'score {r["score"]:.3f}  ·  {r["pixels"]:,} px'))
grid(tiles, labels, os.path.join(IMG, 'p2-fig3-ours.png'), 3)

# valid or not: the head click
tiles = [dot(overlay(mh[k], COLS[k]), *HEAD_PT, 6) for k in range(3)]
v = R['valid']['masks']
grid(tiles, [(f'mask {k + 1}: {["head", "whole cat", "cat + blanket + sofa"][k]}',
              f'score {v[k]["score"]:.3f}  ·  best IoU {v[k]["best"]:.2f}') for k in range(3)],
     os.path.join(IMG, 'p2-valid.png'), 3, scale=0.6)

# a second point: positive or negative
tiles, labels = [], []
tm = np.load(os.path.join(RES, 'part2_tail_masks.npy'))
s1 = R['single']['tail']
tiles.append(dot(overlay(np.zeros_like(tm[0]), COLS[0]), 180, 305))
labels.append(('one click on the tail', 'three masks (tail, part, cat)'))
for lab, ttl in [(1, 'tail + positive point'), (0, 'tail + negative point')]:
    m2 = np.load(os.path.join(RES, f'part2_second_{lab}.npy'))
    t = dot(dot(overlay(m2, COLS[2] if lab else COLS[0]), 180, 305), *BODY, pos=bool(lab))
    tiles.append(t)
    r = R['second'][str(lab)]
    labels.append((ttl, f'1 mask: {r["pixels"]:,} px  ·  score {r["score"]:.3f}'))
grid([t.crop((0, 60, 360, 480)) for t in tiles], labels, os.path.join(IMG, 'p2-second-point.png'), 3)

# point vs box on the tail
tiles, labels = [], []
CT = (60, 200, 300, 400)
for k in range(3):
    tiles.append(dot(overlay(tm[k], COLS[k]), 180, 305, 6).crop(CT))
    labels.append((f'point, mask {k + 1}', f'{int(tm[k].sum()):,} px  ·  score {ts[k]:.3f}'))
for k in range(3):
    t = overlay(mb3[k], COLS[k])
    ImageDraw.Draw(t).rectangle(TAIL_BOX, outline=(40, 220, 90), width=3)
    tiles.append(t.crop(CT))
    labels.append((f'box, mask {k + 1}', f'{int(mb3[k].sum()):,} px  ·  score {sb3[k]:.3f}'))
grid(tiles, labels, os.path.join(IMG, 'p2-point-box.png'), 3, scale=1.25)

# the simulation, rounds 1, 2, 3 and 11 of the detailed run
show = {1, 2, 3, 11}
tiles, labels = [], []
sm = SIM_MASKS
for r in sorted(show):
    mask, pts, labs = sm[r - 1]
    t = overlay(mask, COLS[1])
    tgt_edge = TARGET ^ np.pad(TARGET, 1, mode='edge')[:-2, 1:-1] | TARGET ^ np.pad(TARGET, 1, mode='edge')[1:-1, :-2]
    a = np.array(t)
    a[tgt_edge | np.roll(tgt_edge, 1, 0) | np.roll(tgt_edge, 1, 1)] = (255, 230, 0)
    t = Image.fromarray(a)
    for (x, y), l in zip(pts, labs):
        dot(t, x, y, 5, bool(l))
    tiles.append(t.crop((0, 40, 340, 480)))
    d = detail[r - 1]
    labels.append((f'round {r}: {d["points"]} point' + ('s' if d['points'] > 1 else ''), f'IoU with the target {d["iou"]:.3f}'))
grid(tiles, labels, os.path.join(IMG, 'p2-sim-rounds.png'), 2)
