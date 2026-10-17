"""Part 5: the zero-shot experiments of §7, in small, on real masks drawn by people.
- a mini §7.1: one "center" point per object (the point farthest from the boundary, by distance transform),
  SAM ViT-H, mIoU of the top-scored mask vs the "oracle" best of 3; then 2, 3, 5 and 9 points placed by the
  error-region rule of RITM; also a random first point (Figure 9d)
- the same mini test with ViT-B and ViT-L (the encoder-size ablation of Figure 13, right)
- a mini §7.4: the ground-truth box as the prompt, with and without the extra refinement pass of §D.4
- a mini §7.3: object proposals from a grid of points, and average recall (AR) against the people's masks
- a mini §7.2: edge maps from automatic masks (Sobel on the mask probability maps, then thinning), next to
  the Sobel filter on the photo itself. No benchmark number: we have no BSDS500 here.
Our sample: the first images of the COCO panoptic val2017 set on Hugging Face (nielsr/coco-panoptic-val2017),
read without login. COCO images are also the images of LVIS, one of the paper's 23 datasets. It is a tiny sample,
not the paper's benchmark.
Writes results/part5.json, results/part5_stdout.txt and pictures in public/img/papers/sam/p5-*.png."""
import io, os, random
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from torchvision.ops import nms
from transformers import SamModel, SamProcessor
from common import Log, save

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, '..', '..', '..', 'public', 'img', 'papers', 'sam')
CACHE = os.path.expanduser('~/.cache/papers/sam_part5_coco_sample.npz')
os.makedirs(IMG, exist_ok=True)
log = Log('part5')
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
R = {}
N_OBJECTS, PER_IMAGE, MIN_AREA = 40, 4, 24 * 24
POINTS = [1, 2, 3, 5, 9]

COCO = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat',
        10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
        18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
        28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball',
        38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
        44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
        54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake',
        62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop',
        74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
        82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
        90: 'toothbrush'}


# ------------------------------------------------------------------ 0. the sample: COCO panoptic val2017, first images
def load_sample(n_images=16):
    """The first n images of the first parquet row group, with the masks of their objects ("things")."""
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return list(z['images']), list(z['labels']), list(z['infos']), list(z['ids'])
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq
    fs = HfFileSystem()
    f = fs.open('datasets/nielsr/coco-panoptic-val2017/data/train-00000-of-00002-ac9a9b049ea19ce7.parquet')
    rows = pq.ParquetFile(f).read_row_group(0).slice(0, n_images).to_pylist()
    images, labels, infos, ids = [], [], [], []
    for r in rows:
        images.append(np.array(Image.open(io.BytesIO(r['image']['bytes'])).convert('RGB')))
        lab = np.array(Image.open(io.BytesIO(r['label']['bytes'])).convert('RGB')).astype(np.int64)
        labels.append(lab[..., 0] + 256 * lab[..., 1] + 256 * 256 * lab[..., 2])     # panoptic id per pixel
        infos.append(r['segments_info'])
        ids.append(r['image_id'])
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, images=np.array(images, dtype=object), labels=np.array(labels, dtype=object),
                        infos=np.array(infos, dtype=object), ids=np.array(ids))
    return images, labels, infos, ids


images, labels, infos, image_ids = load_sample()
rng = random.Random(0)
objects = []            # (image index, gt mask, category, box xyxy)
all_things = []         # every non-crowd object of every image, for the proposal recall
for i, (lab, inf) in enumerate(zip(labels, infos)):
    things = [s for s in inf if s['category_id'] in COCO and not s['iscrowd']]
    for s in things:
        x, y, w, h = s['bbox']
        all_things.append((i, lab == s['id'], COCO[s['category_id']], [x, y, x + w, y + h]))
    big = [s for s in things if s['area'] >= MIN_AREA]
    rng.shuffle(big)
    for s in big[:PER_IMAGE]:
        if len(objects) < N_OBJECTS:
            x, y, w, h = s['bbox']
            objects.append((i, lab == s['id'], COCO[s['category_id']], [x, y, x + w, y + h]))
used_images = sorted({o[0] for o in objects})
areas = np.array([o[1].sum() for o in objects])
log('== 0. our sample ==')
log(f'COCO panoptic val2017 (Hugging Face nielsr/coco-panoptic-val2017), the first {len(images)} images')
log(f'objects: {len(objects)} from {len(used_images)} images (up to {PER_IMAGE} per image, chosen at random, area >= {MIN_AREA} px = 24 x 24)')
log(f'object size: smallest {areas.min():,} px, median {int(np.median(areas)):,} px, largest {areas.max():,} px')
cats = {}
for o in objects:
    cats[o[2]] = cats.get(o[2], 0) + 1
log('categories: ' + ', '.join(f'{k} {v}' for k, v in sorted(cats.items(), key=lambda kv: -kv[1])))
R['sample'] = dict(images=len(used_images), objects=len(objects), min_area=int(areas.min()), median_area=int(np.median(areas)),
                   max_area=int(areas.max()), categories=cats, image_ids=[int(image_ids[i]) for i in used_images])


def iou(a, b):
    return float((a & b).sum() / max(1, (a | b).sum()))


def farthest(region):
    """The pixel farthest from the boundary of a region (the image border counts as boundary, as in RITM)."""
    d = ndimage.distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1]
    y, x = np.unravel_index(np.argmax(d), d.shape)
    return int(x), int(y), float(d[y, x])


def next_point(gt, pred):
    """The error-region rule: the point farthest from the boundary of the larger error. Foreground if missed pixels."""
    fn, fp = gt & ~pred, ~gt & pred
    xn, yn, dn = farthest(fn)
    xp, yp, dp = farthest(fp)
    return (xn, yn, 1) if dn >= dp else (xp, yp, 0)


class Sam:
    def __init__(self, name):
        self.proc = SamProcessor.from_pretrained(f'facebook/sam-vit-{name}')
        self.model = SamModel.from_pretrained(f'facebook/sam-vit-{name}').to(DEV).eval()
        self.cache = {}

    def embed(self, i):
        if i not in self.cache:
            p = self.proc(images=Image.fromarray(images[i]), return_tensors='pt')
            with torch.no_grad():
                self.cache = {i: (self.model.get_image_embeddings(p['pixel_values'].to(DEV)), p)}
        return self.cache[i]

    def predict(self, i, points=None, labels_=None, box=None, mask_logits=None, multimask=True):
        emb, p0 = self.embed(i)
        kw = {}
        if points is not None:
            kw['input_points'], kw['input_labels'] = [[points]], [[labels_]]
        if box is not None:
            kw['input_boxes'] = [[box]]
        p = self.proc(images=Image.fromarray(images[i]), return_tensors='pt', **kw) if kw else p0
        args = {k: p[k].float().to(DEV) if k != 'input_labels' else p[k].to(DEV)
                for k in ('input_points', 'input_labels', 'input_boxes') if k in p}
        if mask_logits is not None:
            args['input_masks'] = torch.as_tensor(mask_logits, dtype=torch.float32).reshape(1, 1, 256, 256).to(DEV)
        with torch.no_grad():
            out = self.model(image_embeddings=emb, multimask_output=multimask, **args)
        masks = self.proc.image_processor.post_process_masks(out.pred_masks.cpu(), p0['original_sizes'], p0['reshaped_input_sizes'])[0][0]
        return masks.numpy(), out.iou_scores[0, 0].cpu().numpy(), out.pred_masks[0, 0].cpu().numpy()


def point_curve(sam, first='center', seed=0, use_mask=False):
    """For each object: the IoU after 1, 2, 3, 5 and 9 points (top-scored mask at 1 point), and the 1-point oracle."""
    rr = np.random.default_rng(seed)
    per = []
    for k, (i, gt, cat, box) in enumerate(objects):
        if first == 'center':
            x, y, _ = farthest(gt)
        else:
            ys, xs = np.nonzero(gt)
            j = rr.integers(len(xs))
            x, y = int(xs[j]), int(ys[j])
        pts, labs = [[x, y]], [1]
        ms, sc, low = sam.predict(i, pts, labs, multimask=True)
        ious3 = [iou(m, gt) for m in ms]
        top = int(np.argmax(sc))
        rec = dict(obj=k, cat=cat, area=int(gt.sum()), first=[x, y], scores=[round(float(s), 3) for s in sc],
                   ious3=[round(v, 4) for v in ious3], top=top, oracle=int(np.argmax(ious3)), iou={1: ious3[top]})
        pred, logit = ms[top], low[top]
        for n in range(2, POINTS[-1] + 1):
            px, py, lab = next_point(gt, pred)
            pts.append([px, py]); labs.append(lab)
            m1, _, l1 = sam.predict(i, pts, labs, mask_logits=logit if use_mask else None, multimask=False)
            pred, logit = m1[0], l1[0]
            if n in POINTS:
                rec['iou'][n] = iou(pred, gt)
        rec['points'] = pts
        rec['labels'] = labs
        per.append(rec)
    return per


def summary(per):
    out = {f'{n}': float(np.mean([r['iou'][n] for r in per])) for n in POINTS}
    out['oracle'] = float(np.mean([max(r['ious3']) for r in per]))
    return out


# ------------------------------------------------------------------ 1. the mini §7.1 with ViT-H
log('\n== 1. one center point per object, SAM ViT-H (a mini version of §7.1) ==')
sam = Sam('huge')
per_h = point_curve(sam)
S = summary(per_h)
R['vit_h_center'] = S
R['per_object'] = per_h
log(f'{"#":>3} {"object":<14}{"area":>8}  {"center":>11}  {"IoU mask 1":>10}{"mask 2":>8}{"mask 3":>8}  {"top":>4}{"oracle":>7}  {"1 pt":>6}{"2 pts":>7}{"3 pts":>7}{"5 pts":>7}{"9 pts":>7}')
for r in per_h:
    i3 = r['ious3']
    log(f'{r["obj"]:>3} {r["cat"]:<14}{r["area"]:>8,}  {str(tuple(r["first"])):>11}  {i3[0]:>10.3f}{i3[1]:>8.3f}{i3[2]:>8.3f}  '
        f'{r["top"] + 1:>4}{r["oracle"] + 1:>7}  ' + ''.join(f'{r["iou"][n]:>{6 if n == 1 else 7}.3f}' for n in POINTS))
same = sum(r['top'] == r['oracle'] for r in per_h)
log(f'mIoU over {len(per_h)} objects, 1 center point: top-scored mask {S["1"]:.3f}, oracle (best of 3) {S["oracle"]:.3f}')
log(f'the top-scored mask was also the best of the 3 for {same} of {len(per_h)} objects')
log('mIoU after N points (error-region rule): ' + ', '.join(f'{n} -> {S[str(n)]:.3f}' for n in POINTS))
gap = [(max(r['ious3']) - r['iou'][1], r) for r in per_h]
gap.sort(key=lambda t: -t[0])
log('largest gaps between the oracle and the top-scored mask:')
for g, r in gap[:5]:
    log(f'  #{r["obj"]} {r["cat"]}: top mask {r["top"] + 1} IoU {r["iou"][1]:.3f}, oracle mask {r["oracle"] + 1} IoU {max(r["ious3"]):.3f} (+{g:.3f}); scores {r["scores"]}')
low1 = sum(r['iou'][1] < 0.5 for r in per_h)
log(f'objects with IoU below 0.5 after 1 point: {low1}; after 3 points: {sum(r["iou"][3] < 0.5 for r in per_h)}')
R['same_top_oracle'] = same
R['below_half'] = {1: low1, 3: sum(r['iou'][3] < 0.5 for r in per_h)}

# a worked example of the error-region rule, for the text
ex = gap[0][1]
log(f'\nerror-region rule on #{ex["obj"]} {ex["cat"]}: points ' + ', '.join(
    f'({x}, {y}) {"fg" if l else "bg"}' for (x, y), l in zip(ex['points'][:3], ex['labels'][:3])))

S_mask = summary(point_curve(sam, use_mask=True))
log('variant: the previous mask logits also go back in as a mask prompt: ' + ', '.join(f'{n} -> {S_mask[str(n)]:.3f}' for n in POINTS))
R['vit_h_center_with_mask'] = S_mask

per_rand = point_curve(sam, first='random', seed=0)
S_rand = summary(per_rand)
log('random first point (Figure 9d): ' + ', '.join(f'{n} -> {S_rand[str(n)]:.3f}' for n in POINTS) + f', oracle at 1 point {S_rand["oracle"]:.3f}')
R['vit_h_random'] = S_rand

# ------------------------------------------------------------------ 2. box prompts (a mini §7.4)
log('\n== 2. the ground-truth box as the prompt (a mini version of §7.4, with the refinement pass of §D.4) ==')
box_iou, box_ref = [], []
box_rows = []
for k, (i, gt, cat, box) in enumerate(objects):
    ms, sc, low = sam.predict(i, box=box, multimask=False)
    a = iou(ms[0], gt)
    m2, s2, _ = sam.predict(i, box=box, mask_logits=low[0], multimask=False)
    b = iou(m2[0], gt)
    box_iou.append(a); box_ref.append(b)
    box_rows.append(dict(obj=k, cat=cat, iou=a, refined=b, score=float(sc[0])))
log(f'mIoU with the box prompt: {np.mean(box_iou):.3f}; after one refinement pass (mask + box back in): {np.mean(box_ref):.3f}')
log(f'refinement helped {sum(b > a + 1e-4 for a, b in zip(box_iou, box_ref))} objects, hurt {sum(b < a - 1e-4 for a, b in zip(box_iou, box_ref))}')
worst = sorted(box_rows, key=lambda r: r['refined'])[:4]
log('lowest box IoUs: ' + '; '.join(f'#{r["obj"]} {r["cat"]} {r["refined"]:.3f}' for r in worst))
R['box'] = dict(miou=float(np.mean(box_iou)), refined=float(np.mean(box_ref)), rows=box_rows)

# ------------------------------------------------------------------ 3. object proposals and average recall (a mini §7.3)
log('\n== 3. object proposals from a 32 x 32 grid of points, and average recall (a mini version of §7.3) ==')
GRID, NMS_T, TOPK = 32, 0.9, 1000
THR = np.round(np.arange(0.5, 0.951, 0.05), 2)


def grid_masks(sam, i, grid, batch=16, keep_logits=False):
    """Prompt SAM with a regular grid of single foreground points. Returns 3 masks per point: full-size boolean masks
    (as packed bits), boxes, predicted IoUs, stability scores and (optionally) full-size probability maps."""
    emb, p0 = sam.embed(i)
    H, W = images[i].shape[:2]
    xs = (np.arange(grid) + 0.5) / grid * W
    ys = (np.arange(grid) + 0.5) / grid * H
    pts = np.array([[x, y] for y in ys for x in xs])
    packed, boxes, scores, stab, probs = [], [], [], [], []
    scale = 1024 / max(H, W)
    for s in range(0, len(pts), batch):
        P = torch.tensor(pts[s:s + batch] * scale, dtype=torch.float32).reshape(-1, 1, 1, 2).to(DEV)
        L = torch.ones(P.shape[0], 1, 1, dtype=torch.long).to(DEV)
        with torch.no_grad():
            out = sam.model(image_embeddings=emb.expand(1, -1, -1, -1), input_points=P.reshape(1, -1, 1, 2),
                            input_labels=L.reshape(1, -1, 1), multimask_output=True)
        low = out.pred_masks[0]                                      # (points, 3, 256, 256)
        up = F.interpolate(low.reshape(-1, 1, 256, 256), (1024, 1024), mode='bilinear', align_corners=False)
        h2, w2 = int(round(H * scale)), int(round(W * scale))
        up = F.interpolate(up[..., :h2, :w2], (H, W), mode='bilinear', align_corners=False)[:, 0]   # logits, full size
        m = up > 0
        hi, lo = (up > 1.0).flatten(1).sum(1).float(), (up > -1.0).flatten(1).sum(1).float()
        stab.append((hi / lo.clamp(min=1)).cpu().numpy())
        scores.append(out.iou_scores[0].reshape(-1).cpu().numpy())
        mc = m.cpu().numpy()
        packed.append(np.packbits(mc.reshape(mc.shape[0], -1), axis=1))
        for mm in mc:
            yy, xx = np.nonzero(mm.any(1))[0], np.nonzero(mm.any(0))[0]
            boxes.append([xx[0], yy[0], xx[-1] + 1, yy[-1] + 1] if len(xx) else [0, 0, 0, 0])
        if keep_logits:
            probs.append(torch.sigmoid(up).half().cpu().numpy())
    out = dict(packed=np.concatenate(packed), boxes=np.array(boxes, dtype=np.float32), scores=np.concatenate(scores),
               stab=np.concatenate(stab), shape=(H, W), points=pts)
    if keep_logits:
        out['probs'] = np.concatenate(probs)
    return out


def unpack(g, idx):
    H, W = g['shape']
    return np.unpackbits(g['packed'][idx], axis=1)[:, :H * W].reshape(len(idx), H, W).astype(bool)


best_iou = []              # per ground-truth object: (best IoU among the top-1000 proposals, best among top 100, area)
props_per_image = []
example = None
for i in sorted({t[0] for t in all_things}):
    g = grid_masks(sam, i, GRID)
    ok = g['boxes'][:, 2] > g['boxes'][:, 0]
    idx = np.nonzero(ok)[0]
    keep = nms(torch.tensor(g['boxes'][idx]), torch.tensor(g['scores'][idx]), NMS_T).numpy()
    idx = idx[keep]
    rank = np.argsort(-(g['scores'][idx] + g['stab'][idx]) / 2)     # average of confidence and stability (§D.3)
    idx = idx[rank][:TOPK]
    props_per_image.append(len(idx))
    M = torch.tensor(unpack(g, idx).reshape(len(idx), -1), dtype=torch.float32)
    gts = [t for t in all_things if t[0] == i]
    G = torch.tensor(np.stack([t[1].reshape(-1) for t in gts]), dtype=torch.float32)
    inter = G @ M.T
    union = G.sum(1, keepdim=True) + M.sum(1)[None] - inter
    I = (inter / union.clamp(min=1)).numpy()                          # (ground truths, proposals)
    for gi, t in enumerate(gts):
        best_iou.append((float(I[gi].max()), float(I[gi, :100].max()), int(t[1].sum()), t[2]))
    if example is None and len(gts) >= 6:
        example = dict(i=i, best=[float(I[gi].max()) for gi in range(len(gts))], arg=[int(I[gi].argmax()) for gi in range(len(gts))])
        example['masks'] = unpack(g, idx[example['arg']])
    log(f'  image {image_ids[i]}: {GRID * GRID * 3:,} masks -> {len(idx):,} after box NMS {NMS_T}; {len(gts)} objects drawn by people')
B = np.array([b[0] for b in best_iou]); B100 = np.array([b[1] for b in best_iou]); A = np.array([b[2] for b in best_iou])


def ar(best, sel=None):
    best = best if sel is None else best[sel]
    return float(np.mean([(best >= t).mean() for t in THR])), [float((best >= t).mean()) for t in THR]


ar_all, curve = ar(B)
ar100, curve100 = ar(B100)
small, med, large = A < 32 ** 2, (A >= 32 ** 2) & (A < 96 ** 2), A >= 96 ** 2
log(f'ground-truth objects: {len(B)} (small {small.sum()}, medium {med.sum()}, large {large.sum()}); proposals per image: mean {np.mean(props_per_image):.0f}')
log('recall at each IoU threshold (top 1000): ' + ', '.join(f'{t:.2f} {r:.3f}' for t, r in zip(THR, curve)))
log(f'AR@1000 = mean of the 10 recalls = {ar_all:.3f}   (AR@100 = {ar100:.3f})')
log(f'AR@1000 by size: small {ar(B, small)[0]:.3f}, medium {ar(B, med)[0]:.3f}, large {ar(B, large)[0]:.3f}')
R['proposals'] = dict(grid=GRID, nms=NMS_T, objects=len(B), small=int(small.sum()), medium=int(med.sum()), large=int(large.sum()),
                      per_image=props_per_image, ar1000=ar_all, ar100=ar100, curve=curve, curve100=curve100,
                      ar_small=ar(B, small)[0], ar_medium=ar(B, med)[0], ar_large=ar(B, large)[0], thresholds=THR.tolist())

# ------------------------------------------------------------------ 4. edges from masks (a mini §7.2), on one photo
log('\n== 4. edge maps from SAM masks (a mini version of §7.2) ==')
from datasets import load_dataset
cats_img = np.array(load_dataset('huggingface/cats-image')['test'][0]['image'].convert('RGB'))
images.append(cats_img)
CI = len(images) - 1
g = grid_masks(sam, CI, 16, keep_logits=True)
idx = np.nonzero(g['boxes'][:, 2] > g['boxes'][:, 0])[0]
keep = nms(torch.tensor(g['boxes'][idx]), torch.tensor(g['scores'][idx]), 0.7).numpy()
idx = idx[keep]
log(f'16 x 16 grid -> {16 * 16 * 3} masks (3 per point), no filter by predicted IoU or stability; {len(idx)} left after box NMS 0.7')
H, W = cats_img.shape[:2]
E = np.zeros((H, W), np.float32)
k3 = np.ones((3, 3), np.uint8)
for j in idx:
    p = g['probs'][j].astype(np.float32)
    sx, sy = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(sx, sy)
    m = (p > 0.5).astype(np.uint8)
    border = (cv2.dilate(m, k3) - cv2.erode(m, k3)) > 0                 # the boundary pixels of the mask
    E = np.maximum(E, np.where(border, mag, 0))
E = (E - E.min()) / max(1e-6, E.max() - E.min())


def edge_nms(E):
    """Thin an edge map: keep a pixel only if it is the maximum across the edge (along the normal direction)."""
    Es = cv2.GaussianBlur(E, (0, 0), 1.0)
    oy, ox = np.gradient(Es)
    oxy, oxx = np.gradient(ox)
    oyy, _ = np.gradient(oy)
    O = np.mod(np.arctan2(oyy * np.sign(-oxy), oxx + 1e-9), np.pi)
    yy, xx = np.mgrid[0:E.shape[0], 0:E.shape[1]].astype(np.float32)
    keep = np.ones_like(E, bool)
    for s in (1, -1):
        mx = (xx + s * np.cos(O)).astype(np.float32)
        my = (yy + s * np.sin(O)).astype(np.float32)
        keep &= E >= cv2.remap(E, mx, my, cv2.INTER_LINEAR) - 1e-6
    return np.where(keep, E, 0)


Et = edge_nms(E)
gray = cv2.cvtColor(cats_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
sob = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
sob = sob / sob.max()
sob_t = edge_nms(sob)
log(f'SAM edge map: {int((Et > 0.1).sum()):,} edge pixels above 0.1 after thinning; Sobel on the photo: {int((sob_t > 0.1).sum()):,}')
R['edges'] = dict(masks=16 * 16 * 3, after_nms=int(len(idx)), sam_px=int((Et > 0.1).sum()), sobel_px=int((sob_t > 0.1).sum()))

# ------------------------------------------------------------------ 6. pictures
def font(size):
    for f in ['/System/Library/Fonts/Helvetica.ttc', '/System/Library/Fonts/Menlo.ttc']:
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def outline(mask):
    m = mask.astype(np.uint8)
    return (cv2.dilate(m, k3) - m) > 0


def overlay(img, mask, col, a=0.5, gt=None):
    arr = img.astype(float).copy()
    arr[mask] = (1 - a) * arr[mask] + a * np.array(col)
    out = arr.astype('uint8')
    out[outline(mask)] = (255, 255, 255)
    if gt is not None:
        g = outline(gt)
        g = cv2.dilate(g.astype(np.uint8), k3) > 0
        out[g] = (255, 70, 70)
    return Image.fromarray(out)


def dot(im, x, y, r=6, pos=True):
    d = ImageDraw.Draw(im)
    d.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2], fill=(255, 255, 255))
    d.ellipse([x - r, y - r, x + r, y + r], fill=(40, 220, 90) if pos else (235, 50, 50))
    return im


def grid_img(tiles, labels_, path, cols, cap=58):
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    out = Image.new('RGB', (cols * w + (cols - 1) * 12, rows * (h + cap) + (rows - 1) * 10), (24, 25, 33))
    d = ImageDraw.Draw(out)
    for n, (t, (l1, l2)) in enumerate(zip(tiles, labels_)):
        x, y = (n % cols) * (w + 12), (n // cols) * (h + cap + 10)
        out.paste(t.resize((w, h), Image.LANCZOS), (x, y))
        d.text((x + 10, y + h + 7), l1, font=font(20), fill=(235, 236, 245))
        d.text((x + 10, y + h + 32), l2, font=font(17), fill=(160, 162, 180))
    out.save(path, optimize=True)
    log(f'wrote {os.path.basename(path)} {out.size}')


def crop_around(gt, pad=0.35, size=(360, 300)):
    ys, xs = np.nonzero(gt)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cw, ch = x1 - x0, y1 - y0
    # grow to the tile's aspect ratio, with some context
    cw2, ch2 = max(cw * (1 + 2 * pad), 120), max(ch * (1 + 2 * pad), 100)
    r = size[0] / size[1]
    if cw2 / ch2 < r:
        cw2 = ch2 * r
    else:
        ch2 = cw2 / r
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return int(cx - cw2 / 2), int(cy - ch2 / 2), int(cx + cw2 / 2), int(cy + ch2 / 2)


def tile(img, box, size=(360, 300)):
    x0, y0, x1, y1 = box
    H, W = img.shape[:2] if isinstance(img, np.ndarray) else (img.size[1], img.size[0])
    pil = img if isinstance(img, Image.Image) else Image.fromarray(img)
    canvas = Image.new('RGB', (x1 - x0, y1 - y0), (24, 25, 33))
    canvas.paste(pil.crop((max(0, x0), max(0, y0), min(W, x1), min(H, y1))), (max(0, -x0), max(0, -y0)))
    return canvas.resize(size, Image.LANCZOS), (x1 - x0) / size[0]


def tile_xy(x, y, box, scale):
    return (x - box[0]) / scale, (y - box[1]) / scale


# (a) one point: the people's mask and the center point, SAM's top-scored mask, the oracle mask
COL = (64, 200, 255)
R["pictures"] = {}
easy = max((r for r in per_h if r['area'] > 5000), key=lambda r: r['iou'][1])
show = [gap[0][1], gap[1][1], easy]
tiles, labs_ = [], []
for r in show:
    i, gt, cat = objects[r['obj']][:3]
    ms, sc, _ = sam.predict(i, [r['first']], [1], multimask=True)
    bx = crop_around(gt)
    x, y = r['first']
    t, s = tile(overlay(images[i], gt, (255, 196, 0)), bx)
    tiles.append(dot(t, *tile_xy(x, y, bx, s)))
    labs_.append((f'people\'s mask: {cat}', f'{r["area"]:,} px, center point ({x}, {y})'))
    for which, name in ((r['top'], 'top-scored'), (r['oracle'], 'oracle')):
        t, s = tile(overlay(images[i], ms[which], COL, gt=gt), bx)
        tiles.append(dot(t, *tile_xy(x, y, bx, s)))
        labs_.append((f'{name}: mask {which + 1} of 3', f'score {sc[which]:.3f}, IoU {r["ious3"][which]:.3f}'))
grid_img(tiles, labs_, os.path.join(IMG, 'p5-one-point.png'), 3)
R['pictures']['one_point'] = [r['obj'] for r in show]

# (b) more points: the object whose IoU grows most from 1 to 3 points
grow = max(per_h, key=lambda r: r['iou'][3] - r['iou'][1])
i, gt, cat = objects[grow['obj']][:3]
bx = crop_around(gt)
tiles, labs_ = [], []
for n in (1, 2, 3):
    pts, labs = grow['points'][:n], grow['labels'][:n]
    if n == 1:
        ms, sc, _ = sam.predict(i, pts, labs, multimask=True)
        m = ms[grow['top']]
    else:
        ms, sc, _ = sam.predict(i, pts, labs, multimask=False)
        m = ms[0]
    t, s = tile(overlay(images[i], m, COL, gt=gt), bx)
    for (x, y), l in zip(pts, labs):
        dot(t, *tile_xy(x, y, bx, s), pos=bool(l))
    tiles.append(t)
    labs_.append((f'{n} point{"s" if n > 1 else ""}: IoU {iou(m, gt):.3f}', 'green = on the object, red = not on it'))
grid_img(tiles, labs_, os.path.join(IMG, 'p5-more-points.png'), 3)
log(f'more points example: #{grow["obj"]} {cat}: ' + ', '.join(f'{n} pt IoU {grow["iou"][n]:.3f}' for n in (1, 2, 3)) +
    '; points ' + ', '.join(f'({x}, {y}) {"fg" if l else "bg"}' for (x, y), l in zip(grow['points'][:3], grow['labels'][:3])))
R['pictures']['more_points'] = grow['obj']

# (c) box prompts: the two lowest IoUs and the median
srt = sorted(box_rows, key=lambda r: r['refined'])
pick = [srt[0], srt[1], srt[len(srt) // 2]]
tiles, labs_ = [], []
for r in pick:
    i, gt, cat, box = objects[r['obj']]
    ms, sc, low = sam.predict(i, box=box, multimask=False)
    m2, _, _ = sam.predict(i, box=box, mask_logits=low[0], multimask=False)
    bx = crop_around(gt)
    t, s = tile(overlay(images[i], m2[0], COL, gt=gt), bx)
    d = ImageDraw.Draw(t)
    x0, y0 = tile_xy(box[0], box[1], bx, s)
    x1, y1 = tile_xy(box[2], box[3], bx, s)
    d.rectangle([x0, y0, x1, y1], outline=(40, 220, 90), width=3)
    tiles.append(t)
    labs_.append((f'{cat}: IoU {r["refined"]:.3f}', 'blue = SAM, red line = COCO mask'))
grid_img(tiles, labs_, os.path.join(IMG, 'p5-boxes.png'), 3)
R['pictures']['boxes'] = [r['obj'] for r in pick]

# (d) proposals: the objects of one picture and the best proposal for each
i = example['i']
gts = [t for t in all_things if t[0] == i]
arr = images[i].astype(float).copy()
palette = [(255, 196, 0), (64, 200, 255), (60, 220, 120), (240, 90, 200), (255, 120, 60), (150, 120, 255)]
lab_im = images[i].copy()
for gi, t in enumerate(gts):
    lab_im[outline(t[1])] = palette[gi % len(palette)]
prop = images[i].astype(float).copy()
for gi, m in enumerate(example['masks']):
    prop[m] = 0.45 * prop[m] + 0.55 * np.array(palette[gi % len(palette)])
prop = prop.astype('uint8')
for m in example['masks']:
    prop[outline(m)] = (255, 255, 255)
P = Image.fromarray(prop)
d = ImageDraw.Draw(P)
for gi, t in enumerate(gts):
    ys, xs = np.nonzero(t[1])
    d.text((xs.mean() - 16, ys.mean() - 10), f'{example["best"][gi]:.2f}', font=font(18), fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
grid_img([Image.fromarray(lab_im), P], [(f'the {len(gts)} objects that people drew', f'COCO image {image_ids[i]}'),
                                         ('the best of SAM\'s proposals for each', 'number = IoU with the people\'s mask')],
         os.path.join(IMG, 'p5-proposals.png'), 2)
log(f'proposal example: image {image_ids[i]}, best IoU per object: ' + ', '.join(f'{t[2]} {b:.2f}' for t, b in zip(gts, example['best'])))
R['pictures']['proposals'] = dict(image=int(image_ids[i]), best=example['best'], cats=[t[2] for t in gts])

# (e) edges: the photo, Sobel on the photo, SAM's edges (dark lines on white, as in BSDS500)
def ink(E, gamma=1.0):
    v = 255 - (np.clip(E, 0, 1) ** gamma * 255)
    return Image.fromarray(np.stack([v] * 3, -1).astype('uint8'))


grid_img([Image.fromarray(cats_img), ink(sob_t), ink(Et)],
         [('the photo', '640 x 480'), ('Sobel filter on the photo', 'thinned'), ('SAM: edges of its masks', f'{len(idx)} masks, thinned')],
         os.path.join(IMG, 'p5-edges.png'), 3)

# ------------------------------------------------------------------ 5. encoder size (Figure 13, right)
log('\n== 5. the same mini test with the three image encoders (Figure 13, right) ==')
R['encoders'] = {}
del sam
for name in ['base', 'large', 'huge']:
    if name == 'huge':
        s = S
    else:
        m = Sam(name)
        s = summary(point_curve(m))
        del m
        if DEV == 'mps':
            torch.mps.empty_cache()
    R['encoders'][name] = s
    log(f'ViT-{name[0].upper()}: 1 point {s["1"]:.3f}, oracle {s["oracle"]:.3f}, 3 points {s["3"]:.3f}, 5 points {s["5"]:.3f}')

save('part5', R)
np.save(os.path.join(HERE, 'results', 'part5_edges.npy'), np.stack([Et, sob_t]).astype(np.float16))
log('\nsaved results/part5.json')
