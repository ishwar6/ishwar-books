"""Part 4: the data engine and SA-1B.
- the fully automatic stage of the data engine (Section 4 and Appendix B), rebuilt by hand with the released SAM ViT-H:
  a 32 x 32 grid of points, three masks for each point, the predicted-IoU filter, the stability score (computed by hand
  from the mask logits at -1, 0 and +1), the filter for masks that cover the whole picture, greedy box NMS (written by
  hand and checked against torchvision), the 20 zoomed-in crops, and the clean-up of small regions and holes
- the same run with the transformers pipeline("mask-generation"), to check our version mask by mask
- what each filter removes, as pictures
- a tiny "dataset statistics" sample in the style of Section 5 (masks per image, relative mask size, concavity),
  on our two photos only (SA-1B itself is not downloaded: it is very large and needs a licence)
- the arithmetic behind the numbers of Sections 4, 5 and 6 (Tables 1, 2 and 6 of the paper)
Writes results/part4.json, results/part4_stdout.txt and pictures public/img/papers/sam/p4-*.png."""
import math, os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torchvision.ops import batched_nms, box_iou
from transformers import SamModel, SamProcessor, pipeline
from common import Log, save

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, '..', '..', '..', 'public', 'img', 'papers', 'sam')
os.makedirs(IMG, exist_ok=True)
log = Log('part4')
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.manual_seed(0)
R = {}

# the paper's settings for the fully automatic stage (Appendix B)
IOU_T, STAB_T, OFFSET, BIG_T, NMS_T, MIN_AREA = 0.88, 0.95, 1.0, 0.95, 0.7, 100
BATCH = 32                                       # grid points per decoder call (memory only; it changes no result)

cats = load_dataset('huggingface/cats-image')['test'][0]['image'].convert('RGB')                     # 640 x 480
car_full = Image.open(hf_hub_download('ybelkada/segment-anything', 'assets/car.png')).convert('RGB')    # 2646 x 1764
car = car_full.resize((1500, 1000), Image.LANCZOS)
proc = SamProcessor.from_pretrained('facebook/sam-vit-huge')
sam = SamModel.from_pretrained('facebook/sam-vit-huge').to(DEV).eval()
log(f'model: facebook/sam-vit-huge (the released SAM ViT-H) on {DEV}; photos: cats {cats.size[0]} x {cats.size[1]}, '
    f'car {car_full.size[0]} x {car_full.size[1]} resized to {car.size[0]} x {car.size[1]}')


# ------------------------------------------------------------------ the building blocks
def grid(n):
    """n x n points at the centres of an n x n chessboard over the unit square (as in the released code)."""
    off = 1 / (2 * n)
    s = np.linspace(off, 1 - off, n)
    return np.stack(np.meshgrid(s, s), -1).reshape(-1, 2)          # (x, y), row by row


def stability(logits, offset=OFFSET):
    """IoU of the mask cut at +offset with the mask cut at -offset. The first is inside the second,
    so the IoU is just the ratio of their sizes."""
    hi = (logits > offset).flatten(-2).sum(-1).float()
    lo = (logits > -offset).flatten(-2).sum(-1).float()
    return torch.where(lo > 0, hi / lo.clamp(min=1), torch.zeros_like(lo)), hi, lo


def mask_box(m):
    """Bounding box x0, y0, x1, y1 of each boolean mask (inclusive pixel coordinates)."""
    rows, cols = m.any(-1), m.any(-2)
    H, W = m.shape[-2:]
    ar_h = torch.arange(H, device=m.device)
    ar_w = torch.arange(W, device=m.device)
    y1 = torch.where(rows, ar_h, -1).max(-1).values
    y0 = torch.where(rows, ar_h, H).min(-1).values
    x1 = torch.where(cols, ar_w, -1).max(-1).values
    x0 = torch.where(cols, ar_w, W).min(-1).values
    return torch.stack([x0, y0, x1, y1], -1)


def greedy_nms(boxes, scores, thr):
    """Greedy NMS by hand: take the best box, delete every box that overlaps it by more than thr, repeat."""
    order = scores.argsort(descending=True).tolist()
    keep, gone = [], set()
    iou = box_iou(boxes, boxes)
    for i in order:
        if i in gone:
            continue
        keep.append(i)
        for j in order:
            if j not in gone and j != i and iou[i, j] > thr:
                gone.add(j)
    return keep


def crop_boxes(W, H, n_layers=2, overlap_ratio=512 / 1500):
    """The full picture, then 2 x 2 and 4 x 4 partly overlapping crops (the layout of the released code)."""
    boxes, layers = [[0, 0, W, H]], [0]
    short = min(W, H)
    for layer in range(1, n_layers + 1):
        n = 2 ** layer
        ov = int(overlap_ratio * short * (2 / n))
        cw, ch = math.ceil((ov * (n - 1) + W) / n), math.ceil((ov * (n - 1) + H) / n)
        for x0 in [int((cw - ov) * i) for i in range(n)]:
            for y0 in [int((ch - ov) * i) for i in range(n)]:
                boxes.append([x0, y0, min(x0 + cw, W), min(y0 + ch, H)])
                layers.append(layer)
    return boxes, layers


@torch.no_grad()
def run_crop(img, box, n_side, keep_logits_for=()):
    """Prompt SAM with an n_side x n_side grid on one crop. Returns one record per candidate mask, and the boolean
    masks (in crop coordinates) of the candidates that pass the three filters."""
    x0, y0, x1, y1 = box
    crop = img.crop(box)
    w, h = crop.size
    pts = grid(n_side) * np.array([w, h])
    q = proc(images=crop, input_points=[[[list(map(float, p))] for p in pts]], return_tensors='pt')
    emb = sam.get_image_embeddings(q['pixel_values'].to(DEV))
    P = q['input_points'].float().to(DEV)                          # (1, n_points, 1, 2) in the 1024 frame
    nh, nw = [int(v) for v in q['reshaped_input_sizes'][0]]
    recs, keep, extra = [], {}, {}
    full = img.size[0] * img.size[1]
    for i in range(0, len(pts), BATCH):
        p = P[:, i:i + BATCH]
        o = sam(image_embeddings=emb, input_points=p, input_labels=torch.ones(p.shape[:3], dtype=torch.long, device=DEV),
                multimask_output=True)
        lg = o.pred_masks[0]                                       # (B, 3, 256, 256) logits
        B = lg.shape[0]
        lg = F.interpolate(lg.flatten(0, 1)[:, None], (1024, 1024), mode='bilinear', align_corners=False)
        lg = F.interpolate(lg[..., :nh, :nw], (h, w), mode='bilinear', align_corners=False)[:, 0]   # (B*3, h, w)
        st, hi, lo = stability(lg)
        m = lg > 0
        area = m.flatten(1).sum(-1)
        bx = mask_box(m)
        sc = o.iou_scores[0].flatten()
        # "removed masks that touch the inner boundaries of the crops": near a crop edge that is not a picture edge
        gb = bx.float() + torch.tensor([x0, y0, x0, y0], device=DEV)
        near_crop = torch.isclose(gb, torch.tensor([x0, y0, x1 - 1, y1 - 1], device=DEV, dtype=torch.float), atol=20)
        near_img = torch.isclose(gb, torch.tensor([0, 0, img.size[0] - 1, img.size[1] - 1], device=DEV, dtype=torch.float), atol=20)
        edge = (near_crop & ~near_img).any(-1)
        for j in range(B * 3):
            pi, k = i + j // 3, j % 3
            r = dict(point=pi, k=k, xy=[float(pts[pi][0] + x0), float(pts[pi][1] + y0)], iou=float(sc[j]),
                     stab=float(st[j]), hi=int(hi[j]), lo=int(lo[j]), area=int(area[j]),
                     box=[int(v) for v in gb[j].tolist()], edge=bool(edge[j]) and box != [0, 0, img.size[0], img.size[1]])
            r['big'] = r['area'] >= BIG_T * full
            r['pass_iou'], r['pass_stab'] = r['iou'] > IOU_T, r['stab'] >= STAB_T
            r['pass'] = r['pass_iou'] and r['pass_stab'] and not r['big'] and not r['edge'] and r['area'] > 0
            recs.append(r)
            if r['pass']:
                keep[len(recs) - 1] = m[j].cpu()
            if (pi, k) in keep_logits_for:
                extra[(pi, k)] = lg[j].cpu()
    return recs, keep, extra


def paste(mask, box, W, H):
    out = torch.zeros((H, W), dtype=torch.bool)
    out[box[1]:box[3], box[0]:box[2]] = mask
    return out


def clean(mask):
    """Appendix B post-processing: fill holes under 100 pixels, remove pieces under 100 pixels.
    A mask whose largest piece is under 100 pixels is removed (returns None)."""
    m = mask.numpy().astype(np.uint8)
    changed = False
    n, lab, stats, _ = cv2.connectedComponentsWithStats(1 - m, connectivity=8)          # holes = pieces of the inverse
    for c in range(1, n):
        x, y, w_, h_, a = stats[c]
        touches = x == 0 or y == 0 or x + w_ == m.shape[1] or y + h_ == m.shape[0]
        if a < MIN_AREA and not touches:
            m[lab == c] = 1
            changed = True
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    sizes = stats[1:, -1]
    if len(sizes) == 0 or sizes.max() < MIN_AREA:
        return None, True
    for c in range(1, n):
        if stats[c, -1] < MIN_AREA:
            m[lab == c] = 0
            changed = True
    return torch.from_numpy(m.astype(bool)), changed


def generate(img, name, n_layers=0, keep_logits_for=()):
    """The whole fully automatic stage on one picture. Returns the final masks and the counts at each step."""
    W, H = img.size
    boxes, layers = crop_boxes(W, H, n_layers)
    all_masks, all_rec, counts, extra = [], [], dict(), {}
    per_crop_after_nms = []
    for ci, (b, layer) in enumerate(zip(boxes, layers)):
        n_side = 32 // (2 ** layer)
        recs, keep, ex = run_crop(img, b, n_side, keep_logits_for if ci == 0 else ())
        extra.update(ex)
        idx = sorted(keep)
        for r in recs:
            r['crop'] = ci
        if idx:
            bx = torch.tensor([recs[i]['box'] for i in idx], dtype=torch.float)
            sc = torch.tensor([recs[i]['iou'] for i in idx])
            k1 = batched_nms(bx, sc, torch.zeros(len(idx)), NMS_T).tolist()
            k2 = greedy_nms(bx, sc, NMS_T)
            assert sorted(k1) == sorted(k2), 'our greedy NMS differs from torchvision'
            for k in k1:
                recs[idx[k]]['nms_crop'] = True
                all_masks.append((paste(keep[idx[k]], b, W, H), recs[idx[k]], ci, layer))
            per_crop_after_nms.append(len(k1))
        else:
            per_crop_after_nms.append(0)
        all_rec += recs
    c = dict(points=sum(len({(r['crop'], r['point']) for r in all_rec if r['crop'] == ci}) for ci in range(len(boxes))),
             candidates=len(all_rec),
             pass_iou=sum(r['pass_iou'] for r in all_rec),
             pass_iou_stab=sum(r['pass_iou'] and r['pass_stab'] for r in all_rec),
             pass_big=sum(r['pass_iou'] and r['pass_stab'] and not r['big'] for r in all_rec),
             pass_edge=sum(r['pass'] for r in all_rec),
             after_nms_in_crops=len(all_masks), crops=len(boxes), per_crop=per_crop_after_nms)
    # NMS across crops: the most zoomed-in crop ranks first (score = 1 / crop area)
    if len(boxes) > 1 and all_masks:
        bx = torch.stack([mask_box(m[0][None])[0] for m in all_masks]).float()
        sc = torch.tensor([1.0 / ((boxes[m[2]][2] - boxes[m[2]][0]) * (boxes[m[2]][3] - boxes[m[2]][1])) for m in all_masks])
        kk = batched_nms(bx, sc, torch.zeros(len(all_masks)), NMS_T).tolist()
        all_masks = [all_masks[k] for k in sorted(kk)]
    c['after_nms_across'] = len(all_masks)
    extra['_pre'] = list(all_masks)
    # clean small pieces and holes, then NMS once more, preferring the masks that did not change
    cleaned, n_changed, n_removed = [], 0, 0
    for m, r, ci, layer in all_masks:
        m2, ch = clean(m)
        if m2 is None:
            n_removed += 1
            continue
        n_changed += ch
        cleaned.append((m2, r, ci, layer, ch))
    c['cleaned_changed'], c['cleaned_removed'] = n_changed, n_removed
    if cleaned:
        bx = torch.stack([mask_box(m[0][None])[0] for m in cleaned]).float()
        sc = torch.tensor([0.0 if m[4] else 1.0 for m in cleaned])
        kk = batched_nms(bx, sc, torch.zeros(len(cleaned)), NMS_T).tolist()
        cleaned = [cleaned[k] for k in sorted(kk)]
    c['final'] = len(cleaned)
    return cleaned, all_rec, c, extra


def iou(a, b):
    return float((a & b).sum()) / max(1, float((a | b).sum()))


# ------------------------------------------------------------------ 1. the stability score on a tiny example
log('\n== 1. the stability score, by hand ==')
toy = torch.tensor([[-4.0, -1.5, -0.6, 0.3, 1.8, 3.2, 2.5, 0.9, -0.4, -2.0]])
h_, l_, z_ = int((toy > 1).sum()), int((toy > -1).sum()), int((toy > 0).sum())
log('one row of 10 mask logits:', ', '.join(f'{v:+.1f}' for v in toy[0].tolist()))
log(f'  cut at +1: {h_} pixels;  cut at 0 (the mask): {z_} pixels;  cut at -1: {l_} pixels;  stability = {h_} / {l_} = {h_ / l_:.3f}')
log(f'  logit +1 = probability {torch.sigmoid(torch.tensor(1.0)):.3f};  logit -1 = probability {torch.sigmoid(torch.tensor(-1.0)):.3f}'
    f'  -> in the words of Section 4: 0.5 - delta and 0.5 + delta with delta = {torch.sigmoid(torch.tensor(1.0)) - 0.5:.3f}')
R['toy'] = dict(logits=toy[0].tolist(), hi=h_, zero=z_, lo=l_, stab=h_ / l_)

# ------------------------------------------------------------------ 2. the fully automatic stage on the cat photo
log('\n== 2. the fully automatic stage on the cat photo: 32 x 32 points, full picture only ==')
G = grid(32)
log(f'grid: 32 x 32 = {len(G)} points; first point at ({G[0][0] * cats.size[0]:.1f}, {G[0][1] * cats.size[1]:.1f}) px, '
    f'spacing {cats.size[0] / 32:.1f} px across and {cats.size[1] / 32:.1f} px down')
SHOW_PT = 32 * 6 + 13                                             # a grid point on the head of the left cat
probe = [(SHOW_PT, 0), (SHOW_PT, 1), (SHOW_PT, 2)]
masks_c, rec_c, cnt_c, ex_c = generate(cats, 'cats', 0, keep_logits_for=set(probe))
R['cats'] = cnt_c
log(f'candidates: {cnt_c["points"]} points x 3 masks = {cnt_c["candidates"]}')
log(f'  predicted IoU > {IOU_T}:                    {cnt_c["pass_iou"]:5d}  (removed {cnt_c["candidates"] - cnt_c["pass_iou"]})')
log(f'  and stability >= {STAB_T}:                  {cnt_c["pass_iou_stab"]:5d}  (removed {cnt_c["pass_iou"] - cnt_c["pass_iou_stab"]})')
log(f'  and covers < 95% of the picture:          {cnt_c["pass_big"]:5d}  (removed {cnt_c["pass_iou_stab"] - cnt_c["pass_big"]})')
log(f'  box NMS at 0.7 (duplicates):              {cnt_c["after_nms_in_crops"]:5d}  (removed {cnt_c["pass_big"] - cnt_c["after_nms_in_crops"]})')
log(f'  pieces and holes under 100 px: {cnt_c["cleaned_changed"]} masks changed, {cnt_c["cleaned_removed"]} removed; NMS again: {cnt_c["final"]} final masks')
st_all = np.array([r['stab'] for r in rec_c])
io_all = np.array([r['iou'] for r in rec_c])
log(f'stability only (ignoring the IoU filter): {int((st_all >= STAB_T).sum())} of {len(st_all)} pass;  both fail: '
    f'{int(((st_all < STAB_T) & (io_all <= IOU_T)).sum())}')
log(f'candidates with a stability of exactly 0.95 (where > and >= differ): {int((st_all == 0.95).sum())}')
slots = [sum(1 for m in masks_c if m[1]['k'] == k) for k in range(3)]
log(f'final masks by output slot (mask 1, 2, 3 of the point): {slots[0]}, {slots[1]}, {slots[2]}')
R['cats']['slots'] = slots
R['hist'] = dict(iou=np.histogram(io_all, bins=20, range=(0.5, 1.05))[0].tolist(),
                 stab=np.histogram(st_all, bins=20, range=(0.0, 1.0))[0].tolist())
R['scatter'] = [[round(r['iou'], 3), round(r['stab'], 3)] for r in rec_c]

# the three masks of one grid point
log(f'\nthe three masks of grid point {SHOW_PT} at ({rec_c[3 * SHOW_PT]["xy"][0]:.0f}, {rec_c[3 * SHOW_PT]["xy"][1]:.0f}):')
R['probe'] = []
for k in range(3):
    r = rec_c[3 * SHOW_PT + k]
    log(f'  mask {k + 1}: {r["area"]:6,} px  predicted IoU {r["iou"]:.3f}  pixels > +1: {r["hi"]:6,}  pixels > -1: {r["lo"]:6,}  '
        f'stability {r["hi"]:,} / {r["lo"]:,} = {r["stab"]:.3f}  ->  {"kept" if r["pass"] else "rejected"}')
    R['probe'].append({k_: r[k_] for k_ in ('area', 'iou', 'hi', 'lo', 'stab', 'pass', 'xy')})

# how many grid points found each final mask
boxes_pass = torch.tensor([r['box'] for r in rec_c if r['pass']], dtype=torch.float)
final_boxes = torch.stack([mask_box(m[0][None])[0] for m in masks_c]).float()
hits = (box_iou(final_boxes, boxes_pass) > NMS_T).sum(-1)
order = hits.argsort(descending=True)
R['dupes'] = []
log('\nduplicates: candidates (after the filters) whose box overlaps a final mask by more than 0.7')
for t in order[:3].tolist():
    log(f'  final mask of {int(masks_c[t][0].sum()):,} px: {int(hits[t])} candidates from {len({(r["point"]) for r in rec_c if r["pass"] and box_iou(torch.tensor([r["box"]], dtype=torch.float), final_boxes[t:t + 1])[0, 0] > NMS_T})} grid points')
    R['dupes'].append(dict(pixels=int(masks_c[t][0].sum()), candidates=int(hits[t])))

# ------------------------------------------------------------------ 3. the same with the transformers pipeline
log('\n== 3. the same run with transformers pipeline("mask-generation") ==')
gen = pipeline('mask-generation', model=sam, image_processor=proc.image_processor)
out = gen(cats, points_per_batch=64, pred_iou_thresh=IOU_T, stability_score_thresh=STAB_T, crops_nms_thresh=NMS_T)
pm = [torch.as_tensor(np.array(m)).bool() for m in out['masks']]
# our list before the clean-up, for a fair comparison (the pipeline does no clean-up and no 95% filter)
ours_raw = ex_c['_pre']
best = [max(iou(m[0], p) for p in pm) for m in ours_raw]
back = [max(iou(p, m[0]) for m in ours_raw) for p in pm]
log(f'pipeline: {len(pm)} masks;  ours before the clean-up: {len(ours_raw)} masks (after it: {len(masks_c)})')
log(f'each of our masks against its best match in the pipeline output: IoU >= 0.99 for {sum(b >= 0.99 for b in best)} of {len(best)}, '
    f'lowest {min(best):.3f};  pipeline masks with a match >= 0.99 in ours: {sum(b >= 0.99 for b in back)} of {len(back)}')
R['pipeline'] = dict(n=len(pm), match99=sum(b >= 0.99 for b in best), lowest=min(best), ours=len(ours_raw), back99=sum(b >= 0.99 for b in back))
del gen

# ------------------------------------------------------------------ 4. the car photo, with and without the zoomed-in crops
log('\n== 4. the car photo (1500 x 1000): full picture only, then with the 20 zoomed-in crops ==')
masks_a, rec_a, cnt_a, _ = generate(car, 'car', 0)
log(f'full picture only: {cnt_a["candidates"]} candidates -> IoU {cnt_a["pass_iou"]} -> stability {cnt_a["pass_iou_stab"]} -> '
    f'< 95% {cnt_a["pass_big"]} -> NMS {cnt_a["after_nms_in_crops"]} -> final {cnt_a["final"]}')
cb, cl = crop_boxes(*car.size, 2)
log(f'crops: {len(cb)} = 1 full picture + {cl.count(1)} (2 x 2) + {cl.count(2)} (4 x 4); sizes '
    f'{cb[1][2] - cb[1][0]} x {cb[1][3] - cb[1][1]} and {cb[5][2] - cb[5][0]} x {cb[5][3] - cb[5][1]} px; points 32 x 32, 16 x 16 and 8 x 8')
masks_b, rec_b, cnt_b, _ = generate(car, 'car-crops', 2)
log(f'with crops: {cnt_b["points"]} points, {cnt_b["candidates"]} candidates -> IoU {cnt_b["pass_iou"]} -> stability {cnt_b["pass_iou_stab"]} -> '
    f'< 95% {cnt_b["pass_big"]} -> not at an inner crop edge {cnt_b["pass_edge"]} -> NMS in each crop {cnt_b["after_nms_in_crops"]} -> '
    f'NMS across crops {cnt_b["after_nms_across"]} -> final {cnt_b["final"]}')
by_layer = [sum(1 for m in masks_b if m[3] == L) for L in range(3)]
log(f'final masks by source: full picture {by_layer[0]}, 2 x 2 crops {by_layer[1]}, 4 x 4 crops {by_layer[2]}')
R['car'], R['car_crops'], R['car_by_layer'] = cnt_a, cnt_b, by_layer
R['crop_boxes'] = cb
log('the same with crops on the cat photo:')
masks_cc, rec_cc, cnt_cc, _ = generate(cats, 'cats-crops', 2)
by_layer_c = [sum(1 for m in masks_cc if m[3] == L) for L in range(3)]
log(f'  {cnt_cc["candidates"]} candidates -> final {cnt_cc["final"]} (full picture {by_layer_c[0]}, 2 x 2 {by_layer_c[1]}, 4 x 4 {by_layer_c[2]})')
R['cats_crops'], R['cats_by_layer'] = cnt_cc, by_layer_c

# ------------------------------------------------------------------ 5. dataset statistics on our tiny sample
log('\n== 5. Section 5 style statistics on our two photos (a tiny sample, not SA-1B) ==')


def concavity(m):
    mm = m.numpy().astype(np.uint8)
    cs, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(np.concatenate(cs))
    hm = np.zeros_like(mm)
    cv2.fillPoly(hm, [hull], 1)
    return max(0.0, 1 - mm.sum() / max(1, (hm | mm).sum()))


sq = np.array([[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]], dtype=np.float32)
a_sq = cv2.contourArea(sq)
a_hull = cv2.contourArea(cv2.convexHull(sq))
log(f'tiny example: a 4 x 4 square with a 2 x 2 bite: area {a_sq:.0f}, convex hull area {a_hull:.0f}, concavity 1 - {a_sq:.0f}/{a_hull:.0f} = {1 - a_sq / a_hull:.3f}')
R['toy_concavity'] = dict(area=a_sq, hull=a_hull, c=1 - a_sq / a_hull)
stats = {}
for nm, ms, im in [('cats', masks_cc, cats), ('car', masks_b, car)]:
    A = im.size[0] * im.size[1]
    rel = [math.sqrt(float(m[0].sum()) / A) for m in ms]
    con = [concavity(m[0]) for m in ms]
    ctr = []
    for m in ms:
        ys, xs = np.nonzero(m[0].numpy())
        ctr.append([float(xs.mean() / im.size[0]), float(ys.mean() / im.size[1])])
    stats[nm] = dict(n=len(ms), rel=rel, con=con, ctr=ctr)
    small = sum(r < 0.1 for r in rel)
    log(f'{nm}: {len(ms)} masks;  relative size sqrt(area / picture): median {np.median(rel):.3f}, '
        f'under 0.1: {small} ({100 * small / len(rel):.0f}%), 0.1 to 0.3: {sum(0.1 <= r < 0.3 for r in rel)}, 0.3 and above: {sum(r >= 0.3 for r in rel)}')
    log(f'      concavity: median {np.median(con):.3f}, mean {np.mean(con):.3f}, max {max(con):.3f};  under 0.1: {sum(c_ < 0.1 for c_ in con)}')
R['stats'] = stats
rel_all = stats['cats']['rel'] + stats['car']['rel']
R['rel_hist'] = np.histogram(rel_all, bins=[0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])[0].tolist()
log(f'both photos: {len(rel_all)} masks, {len(rel_all) / 2:.1f} per photo (SA-1B: 1.1B / 11M = 100 per image); '
    f'relative size bins 0-0.05-0.1-0.2-0.3-0.5-0.75-1: {R["rel_hist"]}')

# ------------------------------------------------------------------ 6. the arithmetic of Sections 4, 5 and 6
log('\n== 6. the numbers of the paper, worked out ==')
s1m, s1i, s2m, s2i = 4.3e6, 120e3, 5.9e6, 180e3
log(f'stage 1: 4.3M masks / 120k images = {s1m / s1i:.1f} masks per image (it grew from 20 to 44 during the stage)')
log(f'stage 2: 5.9M new masks / 180k images = {s2m / s2i:.1f} new masks per image (72 per image with the automatic ones)')
log(f'stages 1 + 2: {(s1m + s2m) / 1e6:.1f}M masks; retrains: 6 + 5 = 11')
log(f'stage 3: 1.1B masks / 11M images = {1.1e9 / 11e6:.0f} masks per image; share made fully automatically: '
    f'(1129M - 10.2M) / 1129M = {100 * (1129 - 10.2) / 1129:.1f}%')
log(f'time per mask: 14 s x 6.5 = {14 * 6.5:.0f} s for a COCO mask; 14 s / 2 = {14 / 2:.0f} s for a box with extreme points')
yrs = 1.1e9 * 14 / (3600 * 24 * 365)
log(f'1.1B masks by hand at 14 s each: {1.1e9 * 14 / 3600 / 1e6:.2f} million hours = {yrs:.0f} years of work without a pause')
log(f'SA-1B pictures: 3300 x 4950 = {3300 * 4950 / 1e6:.1f}M pixels = {3300 * 4950 / (480 * 640):.0f}x a COCO picture (480 x 640); '
    f'released at 1500 x {4950 * 1500 / 3300:.0f} = {1500 * 2250 / 1e6:.2f}M pixels = {1500 * 2250 / (480 * 640):.0f}x COCO')
log(f'grid spacing on an average SA-1B picture: {3300 / 32:.0f} x {4950 / 32:.0f} px; in the 1024 frame that SAM reads: {1024 / 32:.0f} px')
log(f'scale that SAM applies: full car photo 1024 / 1500 = {1024 / 1500:.2f}; a 4 x 4 crop 1024 / 503 = {1024 / 503:.2f} '
    f'-> an object looks {(1024 / 503) / (1024 / 1500):.1f}x larger in the crop')
log(f'mask quality study: 500 images, ~50k masks -> ~{50e3 / 500:.0f} masks per image; 94% of ~50k = ~{0.94 * 50e3:,.0f} pairs above 0.90 IoU')
log(f'dataset sizes (Figure 6 legend): masks per image: SA-1B {1129 / 11:.0f}, Open Images {2.7 / 1:.1f}, ADE20K {0.7 / 0.028:.0f}, '
    f'COCO {0.9 / 0.123:.1f}, LVIS {1.5 / 0.120:.1f}')
log(f'  SA-1B / Open Images: images {11 / 1:.0f}x, masks {1129 / 2.7:.0f}x, masks per image {(1129 / 11) / 2.7:.0f}x; '
    f'SA-1B / ADE20K masks per image {(1129 / 11) / (0.7 / 0.028):.1f}x (the paper: 3.5x, from exact counts)')
T1 = [('Africa', 54, 0.30, 28, 2.8, 3.0, 1.7), ('Asia & Oceania', 70, 3.9, 423, 36.2, 11.4, 14.3), ('Europe', 47, 5.4, 540, 49.8, 34.2, 36.2),
      ('Latin America & Carib.', 42, 0.38, 36, 3.5, 3.1, 5.0), ('North America', 4, 0.83, 80, 7.7, 48.3, 42.8),
      ('high income', 81, 5.8, 598, 54.0, 89.1, 87.5), ('middle income', 108, 4.9, 499, 45.0, 10.5, 12.0), ('low income', 28, 0.1, 9.4, 0.9, 0.4, 0.5)]
log('Table 1: masks per image in each group = #masks / #images')
for nm, nc, im_, mk, a, b, c_ in T1:
    log(f'  {nm:<24} {mk:>5}M / {im_:>4}M = {mk / im_:6.1f};  share of SA-1B images {a:4.1f}%  COCO {b:4.1f}%  Open Images {c_:4.1f}%')
log(f'  regions add up to {sum(r[4] for r in T1[:5]):.1f}% (SA-1B), {sum(r[5] for r in T1[:5]):.1f}% (COCO), {sum(r[6] for r in T1[:5]):.1f}% (Open Images); '
    f'countries {sum(r[1] for r in T1[:5])} by region, {sum(r[1] for r in T1[5:])} by income')
log(f'  Africa in SA-1B: 28M masks = {28 / 2.7:.0f}x all the masks of Open Images (2.7M)')
R['table1'] = T1
T2 = dict(people=[('feminine', 54.4, 1.7, 90.4, 0.6), ('masculine', 55.7, 1.7, 90.1, 0.6), ('older', 62.9, 6.7, 92.6, 1.3),
                  ('middle', 54.5, 1.3, 90.2, 0.5), ('young', 54.2, 2.2, 91.2, 0.7)] +
          [(f'skin {i + 1}', a, b, c_, d) for i, (a, b, c_, d) in enumerate([(52.9, 2.2, 91.0, 0.9), (51.5, 1.4, 91.1, 0.5), (52.2, 1.9, 91.4, 0.7),
                                                                            (51.5, 2.7, 91.7, 1.0), (52.4, 4.2, 92.5, 1.4), (56.7, 6.3, 91.2, 2.4)])],
          clothing=[('feminine', 76.3, 1.1, 90.7, 0.5), ('masculine', 81.0, 1.2, 92.3, 0.4), ('older', 81.9, 3.8, 92.8, 1.6),
                    ('middle', 78.2, 0.8, 91.3, 0.3), ('young', 77.3, 2.7, 91.5, 0.9)])
R['table2'] = T2


def interval(m, e):
    return m - e, m + e


log('Tables 2 and 6: 95% confidence intervals (mean +- half-width), 1 point')
for tab, a_, b_ in [('people', 'older', 'middle'), ('people', 'feminine', 'masculine'), ('clothing', 'feminine', 'masculine'),
                    ('clothing', 'older', 'middle')]:
    ra = next(r for r in T2[tab] if r[0] == a_)
    rb = next(r for r in T2[tab] if r[0] == b_)
    ia, ib = interval(ra[1], ra[2]), interval(rb[1], rb[2])
    ov = ia[0] <= ib[1] and ib[0] <= ia[1]
    log(f'  {tab:<8} {a_:<9} [{ia[0]:.1f}, {ia[1]:.1f}]  {b_:<9} [{ib[0]:.1f}, {ib[1]:.1f}]  -> {"overlap" if ov else "disjoint"}')
ia, ib = interval(76.3, 1.1), interval(81.0, 1.2)
log(f'  clothing, gender: gap 81.0 - 76.3 = {81.0 - 76.3:.1f} points at 1 point; 92.3 - 90.7 = {92.3 - 90.7:.1f} at 3 points; '
    f'3-point intervals [{90.7 - 0.5:.1f}, {90.7 + 0.5:.1f}] and [{92.3 - 0.4:.1f}, {92.3 + 0.4:.1f}]')

save('part4', R)

# ------------------------------------------------------------------ 7. pictures


def font(size):
    for f_ in ['/System/Library/Fonts/Helvetica.ttc', '/System/Library/Fonts/Menlo.ttc']:
        if os.path.exists(f_):
            return ImageFont.truetype(f_, size)
    return ImageFont.load_default()


COLS = [(255, 196, 0), (64, 200, 255), (60, 90, 255), (255, 120, 60)]


def overlay(mask, col, a=0.55, base=None, img=cats):
    arr = np.array(base if base is not None else img).astype(float)
    m = mask.numpy() if torch.is_tensor(mask) else mask
    arr[m] = (1 - a) * arr[m] + a * np.array(col)
    e = arr.astype('uint8')
    edge = m ^ np.pad(m, 1, mode='edge')[:-2, 1:-1] | m ^ np.pad(m, 1, mode='edge')[1:-1, :-2]
    e[edge] = (255, 255, 255)
    return Image.fromarray(e)


def dot(im, x, y, r=7, col=(40, 220, 90)):
    d = ImageDraw.Draw(im)
    d.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2], fill=(255, 255, 255))
    d.ellipse([x - r, y - r, x + r, y + r], fill=col)
    return im


def panels(tiles, labels, path, scale=1.0, cols=None):
    w, h = tiles[0].size
    w, h = int(w * scale), int(h * scale)
    cap = 64
    cols = cols or len(tiles)
    rows = math.ceil(len(tiles) / cols)
    out = Image.new('RGB', (cols * w + (cols - 1) * 12, rows * (h + cap) + (rows - 1) * 8), (24, 25, 33))
    d = ImageDraw.Draw(out)
    for i, (t, (l1, l2)) in enumerate(zip(tiles, labels)):
        x, y = (i % cols) * (w + 12), (i // cols) * (h + cap + 8)
        out.paste(t.resize((w, h), Image.LANCZOS), (x, y))
        d.text((x + 10, y + h + 9), l1, font=font(22), fill=(235, 236, 245))
        d.text((x + 10, y + h + 36), l2, font=font(18), fill=(160, 162, 180))
    out.save(path, optimize=True)
    print('wrote', path, out.size)


def all_masks(ms, img, seed=3):
    rng = np.random.default_rng(seed)
    arr = np.array(img).astype(float)
    edge_all = np.zeros(arr.shape[:2], bool)
    for m in sorted(ms, key=lambda t: -int(t[0].sum())):
        mm = m[0].numpy()
        col = rng.integers(40, 255, 3)
        arr[mm] = 0.45 * arr[mm] + 0.55 * col
        edge_all |= mm ^ np.pad(mm, 1, mode='edge')[:-2, 1:-1] | mm ^ np.pad(mm, 1, mode='edge')[1:-1, :-2]
    e = arr.astype('uint8')
    e[edge_all] = (255, 255, 255)
    return Image.fromarray(e)


# the grid and the three masks of one point
gi = cats.copy()
d = ImageDraw.Draw(gi)
for x, y in G * np.array(cats.size):
    d.ellipse([x - 2.2, y - 2.2, x + 2.2, y + 2.2], fill=(40, 220, 90), outline=(255, 255, 255))
px, py = R['probe'][0]['xy']
tiles, labs = [dot(gi, px, py, 6, (255, 70, 70))], [('the 32 x 32 grid', '1,024 points, one prompt each')]
for k in range(3):
    lg = ex_c[(SHOW_PT, k)]
    pr = R['probe'][k]
    tiles.append(dot(overlay(lg > 0, COLS[k]), px, py, 6, (255, 70, 70)))
    labs.append((f'mask {k + 1}: ' + ('kept' if pr['pass'] else 'rejected'), f'pred. IoU {pr["iou"]:.3f} · stability {pr["stab"]:.3f}'))
panels(tiles, labs, os.path.join(IMG, 'p4-grid.png'), scale=0.75)

# the final masks
panels([cats, all_masks(masks_c, cats)], [('the photo', f'{cats.size[0]} x {cats.size[1]} px'),
                                         (f'{len(masks_c)} masks', '32 x 32 points, full picture only')],
       os.path.join(IMG, 'p4-amg-cats.png'))
panels([car, all_masks(masks_a, car), all_masks(masks_b, car)],
       [('the photo', '1500 x 1000 px'), (f'{len(masks_a)} masks', 'full picture only'), (f'{len(masks_b)} masks', 'with the 20 zoomed-in crops')],
       os.path.join(IMG, 'p4-amg-car.png'), scale=0.42)

# the crops and the masks that only the crops found
ci_ = car.copy()
d = ImageDraw.Draw(ci_)
for b, L in zip(cb[1:], cl[1:]):
    d.rectangle(b, outline=(255, 196, 0) if L == 1 else (64, 200, 255), width=6 if L == 1 else 3)
only = [m for m in masks_b if m[3] > 0]
panels([ci_, all_masks(only, car)], [('the 20 crops', '4 crops of 2 x 2 (yellow), 16 crops of 4 x 4 (blue)'),
                                     (f'{len(only)} masks from the crops', 'kept after NMS across crops')],
       os.path.join(IMG, 'p4-crops.png'), scale=0.42)

# what the filters remove (cat photo)
full_logits = {}
low_iou = sorted([r for r in rec_c if r['pass_stab'] and not r['pass_iou'] and r['area'] > 2000], key=lambda r: r['iou'])
low_stab = sorted([r for r in rec_c if r['pass_iou'] and not r['pass_stab'] and r['area'] > 2000], key=lambda r: r['stab'])
pick_iou = [low_iou[0], low_iou[len(low_iou) // 3], low_iou[2 * len(low_iou) // 3]]
pick_stab = [low_stab[0], low_stab[len(low_stab) // 3], low_stab[2 * len(low_stab) // 3]]
want = {(r['point'], r['k']) for r in pick_iou + pick_stab}
_, _, ex2 = run_crop(cats, [0, 0, cats.size[0], cats.size[1]], 32, want)
tiles, labs = [], []
for r in pick_iou:
    lg = ex2[(r['point'], r['k'])]
    tiles.append(dot(overlay(lg > 0, COLS[3]), *r['xy'], 6))
    labs.append((f'pred. IoU {r["iou"]:.3f} (< 0.88)', f'stability {r["stab"]:.3f} · {r["area"]:,} px'))
panels(tiles, labs, os.path.join(IMG, 'p4-reject-iou.png'), scale=0.75)
tiles, labs = [], []
for r in pick_stab:
    lg = ex2[(r['point'], r['k'])]
    core, band = (lg > 1).numpy(), ((lg > -1) & (lg <= 1)).numpy()
    arr = np.array(cats).astype(float)
    arr[core] = 0.45 * arr[core] + 0.55 * np.array(COLS[1])
    arr[band] = 0.25 * arr[band] + 0.75 * np.array((255, 60, 60))
    tiles.append(dot(Image.fromarray(arr.astype('uint8')), *r['xy'], 6))
    labs.append((f'stability {r["stab"]:.3f} (< 0.95)', f'pred. IoU {r["iou"]:.3f} · red: {r["lo"] - r["hi"]:,} unsure px'))
panels(tiles, labs, os.path.join(IMG, 'p4-reject-stab.png'), scale=0.75)
R['reject_examples'] = dict(iou=[{k: r[k] for k in ('iou', 'stab', 'area', 'hi', 'lo')} for r in pick_iou],
                            stab=[{k: r[k] for k in ('iou', 'stab', 'area', 'hi', 'lo')} for r in pick_stab])
for r in pick_stab:
    log(f'unstable example: stability {r["hi"]:,} / {r["lo"]:,} = {r["stab"]:.3f}, predicted IoU {r["iou"]:.3f}, {r["lo"] - r["hi"]:,} pixels between -1 and +1')
for r in pick_iou:
    log(f'low-confidence example: predicted IoU {r["iou"]:.3f}, stability {r["stab"]:.3f}, {r["area"]:,} px')

# NMS: one kept mask and two of the duplicates it removed
t = order[0].item()
kept_box = final_boxes[t:t + 1]
dup = [r for r in rec_c if r['pass'] and not r.get('nms_crop')
       and box_iou(torch.tensor([r['box']], dtype=torch.float), kept_box)[0, 0] > NMS_T]
dup = sorted(dup, key=lambda r: -r['iou'])
pk = [dup[0], dup[len(dup) // 2]]
_, _, ex3 = run_crop(cats, [0, 0, cats.size[0], cats.size[1]], 32, {(r['point'], r['k']) for r in pk})
km = masks_c[t]
tiles = [dot(overlay(km[0], COLS[2]), *km[1]['xy'], 6)]
labs = [('kept: highest predicted IoU', f'pred. IoU {km[1]["iou"]:.3f} · {int(km[0].sum()):,} px')]
R['nms_example'] = dict(kept=km[1]['iou'], dups=[])
for r in pk:
    m = ex3[(r['point'], r['k'])] > 0
    bi = float(box_iou(torch.tensor([r['box']], dtype=torch.float), kept_box)[0, 0])
    tiles.append(dot(overlay(m, COLS[0]), *r['xy'], 6))
    labs.append((f'removed: box IoU {bi:.3f} > 0.7', f'pred. IoU {r["iou"]:.3f} · mask IoU {iou(m, km[0]):.3f}'))
    R['nms_example']['dups'].append(dict(box_iou=bi, iou=r['iou'], mask_iou=iou(m, km[0])))
    log(f'NMS example: duplicate with box IoU {bi:.3f}, predicted IoU {r["iou"]:.3f} (kept mask: {km[1]["iou"]:.3f}), mask IoU {iou(m, km[0]):.3f}')
panels(tiles, labs, os.path.join(IMG, 'p4-nms.png'), scale=0.75)
log(f'the kept mask had {len(dup)} duplicates among the filtered candidates')
R['nms_example']['n_dups'] = len(dup)
save('part4', R)
