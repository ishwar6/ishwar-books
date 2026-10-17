"""Part 6: the limitations of Section 8, measured on the released SAM models.
- fine structures: a real bicycle photo (COCO val2017 174482) with a box prompt and a point on the rear wheel,
  and a drawn wheel with 24 spokes of known width (1 to 16 pixels), where the correct mask is known exactly:
  spoke recall (spoke pixels inside the mask) and gap fill (empty pixels between the spokes inside the mask)
- small disconnected components: a 16 x 16 grid of single-point prompts on two photos; for each prompt the mask
  with the highest predicted IoU; connected components counted with scipy (8-connectivity)
- boundary crispness: the 256 x 256 low-resolution logits against the upscaled mask, the width of the soft band
  where the probability is between 0.1 and 0.9, and a drawn edge with teeth of known width (1 to 32 pixels)
- ViT-B against ViT-H on all of the above, and the agreement of their masks (the way MobileSAM measures itself)
Writes results/part6.json, results/part6_stdout.txt and the pictures public/img/papers/sam/p6-*.png."""
import os, urllib.request
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from datasets import load_dataset
from transformers import SamModel, SamProcessor
from common import Log, save

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
IMG = os.path.join(HERE, '..', '..', '..', 'public', 'img', 'papers', 'sam')
os.makedirs(IMG, exist_ok=True)
log = Log('part6')
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
R = {}
MODELS = {'ViT-H': 'facebook/sam-vit-huge', 'ViT-B': 'facebook/sam-vit-base'}

# ------------------------------------------------------------------ the pictures
BIKE_URL = 'http://images.cocodataset.org/val2017/000000174482.jpg'
bike_path = os.path.join(RES, 'part6_bike.jpg')
if not os.path.exists(bike_path):
    req = urllib.request.Request(BIKE_URL, headers={'User-Agent': 'Mozilla/5.0'})
    open(bike_path, 'wb').write(urllib.request.urlopen(req).read())
bike = Image.open(bike_path).convert('RGB')
cats = load_dataset('huggingface/cats-image')['test'][0]['image'].convert('RGB')
log(f'bike photo: COCO val2017 174482, {bike.size[0]} x {bike.size[1]} pixels; cats photo: {cats.size[0]} x {cats.size[1]} pixels')
log(f'device: {DEV}')


class Sam:
    def __init__(self, name):
        self.proc = SamProcessor.from_pretrained(name)
        self.m = SamModel.from_pretrained(name).to(DEV).eval()
        self.params = sum(p.numel() for p in self.m.parameters())
        self.enc_params = sum(p.numel() for p in self.m.vision_encoder.parameters())
        self.dec_params = sum(p.numel() for p in self.m.mask_decoder.parameters()) + sum(p.numel() for p in self.m.prompt_encoder.parameters())
        self.cache = {}

    def embed(self, image):
        key = id(image)
        if key not in self.cache:
            p = self.proc(images=image, return_tensors='pt')
            with torch.no_grad():
                emb = self.m.get_image_embeddings(p['pixel_values'].to(DEV))
            self.cache[key] = (emb, p['original_sizes'], p['reshaped_input_sizes'])
        return self.cache[key]

    def run(self, image, points=None, box=None, multimask=True):
        """points: list of (x, y), one single-point prompt each. box: one box. Returns logits (N, K, 256, 256),
        full-size masks (N, K, H, W) as bool, full-size probabilities, scores (N, K)."""
        emb, osz, rsz = self.embed(image)
        s = 1024 / max(image.size)
        kw = {}
        if points is not None:
            kw['input_points'] = torch.tensor([[[[x * s, y * s]] for x, y in points]], dtype=torch.float32, device=DEV)
            kw['input_labels'] = torch.ones((1, len(points), 1), dtype=torch.long, device=DEV)
        if box is not None:
            kw['input_boxes'] = torch.tensor([[[v * s for v in box]]], dtype=torch.float32, device=DEV)
        with torch.no_grad():
            out = self.m(image_embeddings=emb, multimask_output=multimask, **kw)
        low = out.pred_masks.cpu()
        up = self.proc.image_processor.post_process_masks(low, osz, rsz, binarize=False)[0]
        return low[0].numpy(), (up > 0).numpy(), torch.sigmoid(up).numpy(), out.iou_scores[0].cpu().numpy()


def overlay(image, mask, col=(230, 60, 230), a=0.55):
    arr = np.array(image).astype(float)
    arr[mask] = (1 - a) * arr[mask] + a * np.array(col)
    return Image.fromarray(arr.astype('uint8'))


def font(size):
    for f in ['/System/Library/Fonts/Helvetica.ttc', '/System/Library/Fonts/Menlo.ttc']:
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def grid(tiles, labels, path, cols, cap=58):
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    out = Image.new('RGB', (cols * w + (cols - 1) * 12, rows * (h + cap) + (rows - 1) * 10), (24, 25, 33))
    d = ImageDraw.Draw(out)
    for i, (t, (l1, l2)) in enumerate(zip(tiles, labels)):
        x, y = (i % cols) * (w + 12), (i // cols) * (h + cap + 10)
        out.paste(t, (x, y))
        d.text((x + 10, y + h + 7), l1, font=font(20), fill=(235, 236, 245))
        d.text((x + 10, y + h + 32), l2, font=font(17), fill=(160, 162, 180))
    out.save(path, optimize=True)
    print('wrote', os.path.relpath(path, HERE), out.size)


sams = {k: Sam(v) for k, v in MODELS.items()}
log('\n== the two models ==')
R['models'] = {}
for k, s in sams.items():
    log(f'{k}: {s.params:,} weights, of which image encoder {s.enc_params:,}, prompt encoder + mask decoder {s.dec_params:,}')
    R['models'][k] = dict(params=s.params, encoder=s.enc_params, decoder=s.dec_params)
log(f'ViT-H has {sams["ViT-H"].enc_params / sams["ViT-B"].enc_params:.1f} x the encoder weights of ViT-B')

# ------------------------------------------------------------------ 1. fine structures: the real bicycle
log('\n== 1. fine structures: the bicycle photo ==')
BOX = (195, 8, 497, 388)
WHEEL_PT = (482, 270)
R['bike'] = {}
bike_tiles = []
for k, s in sams.items():
    _, mb, _, sb = s.run(bike, box=BOX, multimask=False)
    _, mw, _, sw = s.run(bike, points=[WHEEL_PT])
    best = 1                                 # mask 2: the wheel level (mask 3 is the whole bicycle)
    wheel = mw[0, best]
    filled = ndi.binary_fill_holes(wheel)
    holes = int(filled.sum() - wheel.sum())
    n_h = int(ndi.label(~wheel)[1]) - 1      # background components other than the outside = holes
    log(f'{k}: box {list(BOX)} -> 1 mask, predicted IoU {sb[0, 0]:.3f}, {int(mb[0, 0].sum()):,} pixels')
    log(f'{k}: point {WHEEL_PT} on the rear tyre -> mask {best + 1} (the wheel), predicted IoU {sw[0, best]:.3f}, '
        f'{int(wheel.sum()):,} pixels; holes inside it: {n_h} ({holes} pixels)')
    R['bike'][k] = dict(box_score=float(sb[0, 0]), box_pixels=int(mb[0, 0].sum()), wheel_best=best + 1,
                        wheel_score=float(sw[0, best]), wheel_pixels=int(wheel.sum()), wheel_holes=n_h, wheel_hole_pixels=holes)
    if k == 'ViT-H':
        np.save(os.path.join(RES, 'part6_bike_box.npy'), mb[0, 0])
        bike_tiles = [overlay(bike, mb[0, 0]), overlay(bike, wheel)]
        d = ImageDraw.Draw(bike_tiles[1])
        x, y = WHEEL_PT
        d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=(255, 255, 255)); d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(40, 220, 90))
        d = ImageDraw.Draw(bike_tiles[0])
        d.rectangle(BOX, outline=(255, 255, 255), width=3)
grid(bike_tiles, [('ViT-H, box around the bicycle', f'predicted IoU {R["bike"]["ViT-H"]["box_score"]:.3f}  ·  the wheels are solid discs'),
                  ('ViT-H, one point on the rear tyre', f'mask {R["bike"]["ViT-H"]["wheel_best"]}, predicted IoU {R["bike"]["ViT-H"]["wheel_score"]:.3f}  ·  '
                   f'{R["bike"]["ViT-H"]["wheel_holes"]} holes')],
     os.path.join(IMG, 'p6-bike.png'), 2)

# ------------------------------------------------------------------ 2. fine structures: a drawn wheel with known spokes
log('\n== 2. a drawn wheel: 24 spokes of known width, box prompt, 1024 x 1024 pixels ==')
N, C, R_OUT, R_IN, HUB = 1024, 512, 440, 410, 40
yy, xx = np.mgrid[:N, :N]
rr = np.hypot(xx - C, yy - C)
INNER = (rr < R_IN - 1) & (rr > HUB + 1)       # the area between hub and rim: spokes and empty gaps


def wheel_image(w):
    im = Image.new('L', (N, N), 215)
    d = ImageDraw.Draw(im)
    d.ellipse([C - R_OUT, C - R_OUT, C + R_OUT, C + R_OUT], fill=30)
    d.ellipse([C - R_IN, C - R_IN, C + R_IN, C + R_IN], fill=215)
    for k in range(24):
        a = 2 * np.pi * k / 24
        d.line([C, C, C + (R_IN + 2) * np.cos(a), C + (R_IN + 2) * np.sin(a)], fill=30, width=w)
    d.ellipse([C - HUB, C - HUB, C + HUB, C + HUB], fill=30)
    return im.convert('RGB'), np.array(im) < 120


WIDTHS = [1, 2, 3, 4, 6, 8, 12, 16]
WBOX = (C - R_OUT - 5, C - R_OUT - 5, C + R_OUT + 5, C + R_OUT + 5)
log('the image is 1024 x 1024, so the 256 x 256 low-resolution mask has one cell for each 4 x 4 pixels')
log(f'{"model":<7}{"spoke width":>12}{"spoke px":>10}{"score":>8}{"spoke recall":>14}{"gap fill":>10}{"IoU":>7}{"best of 3 IoU":>15}')
R['wheel'] = {k: [] for k in sams}
wheel_tiles, wheel_labels = [], []
for k, s in sams.items():
    for w in WIDTHS:
        im, gt = wheel_image(w)
        _, m1, _, s1 = s.run(im, box=WBOX, multimask=False)
        _, m3, _, _ = s.run(im, box=WBOX, multimask=True)
        s.cache.clear()
        mk = m1[0, 0]
        sp, gap = gt & INNER, (~gt) & INNER
        rec, fill = (mk & sp).sum() / sp.sum(), (mk & gap).sum() / gap.sum()
        iou = (mk & gt).sum() / (mk | gt).sum()
        best3 = max((m3[0, j] & gt).sum() / (m3[0, j] | gt).sum() for j in range(3))
        log(f'{k:<7}{w:>9} px{int(sp.sum()):>10,}{s1[0, 0]:>8.3f}{rec:>14.3f}{fill:>10.3f}{iou:>7.3f}{best3:>15.3f}')
        R['wheel'][k].append(dict(width=w, spoke_px=int(sp.sum()), score=float(s1[0, 0]), recall=float(rec), gap_fill=float(fill),
                                  iou=float(iou), best3_iou=float(best3)))
        if w in (1, 2, 3, 4, 8):
            t = overlay(im, mk, a=0.6).crop((C - R_OUT - 20, C - R_OUT - 20, C + R_OUT + 20, C + R_OUT + 20)).resize((220, 220), Image.LANCZOS)
            wheel_tiles.append(t)
            wheel_labels.append((f'{k}, spokes {w} px', f'recall {rec:.2f} · gap fill {fill:.2f}'))
grid(wheel_tiles, wheel_labels, os.path.join(IMG, 'p6-wheels.png'), 5)
for k in sams:
    ok = [r for r in R['wheel'][k] if r['recall'] > 0.9 and r['gap_fill'] < 0.1]
    log(f'{k}: widths with spoke recall > 0.9 and gap fill < 0.1 (the correct shape): {[r["width"] for r in ok] or "none"}')

# ------------------------------------------------------------------ 3. small disconnected components
log('\n== 3. small disconnected components: 16 x 16 grid of single-point prompts ==')
EIGHT = np.ones((3, 3), int)
R['components'] = {}
speck_examples = []
for iname, image in [('bike', bike), ('cats', cats)]:
    W0, H0 = image.size
    xs = (np.arange(16) + 0.5) * W0 / 16
    ys = (np.arange(16) + 0.5) * H0 / 16
    pts = [(float(x), float(y)) for y in ys for x in xs]
    for k, s in sams.items():
        stats = []
        for i in range(0, len(pts), 64):
            _, mk, pr, sc = s.run(image, points=pts[i:i + 64])
            for j in range(mk.shape[0]):
                b = int(np.argmax(sc[j]))
                m = mk[j, b]
                hi, lo = pr[j, b] > 0.7311, pr[j, b] > 0.2689     # logits > +1 and > -1 (sigmoid(1) = 0.7311)
                stab = hi.sum() / max(1, lo.sum())
                lab, n = ndi.label(m, structure=EIGHT)
                sizes = np.bincount(lab.ravel())[1:] if n else np.array([])
                main = int(sizes.max()) if n else 0
                small = [int(v) for v in sizes if v < 100 and v != main]
                stats.append(dict(pt=pts[i + j], score=float(sc[j, b]), stab=float(stab), area=int(m.sum()), n=int(n), small=small))
                if k == 'ViT-H' and len(small) >= 3 and sc[j, b] > 0.88:
                    speck_examples.append((iname, image, m, lab, sizes, pts[i + j], float(sc[j, b]), len(small)))
        nz = [t for t in stats if t['n'] > 0]
        multi = [t for t in nz if t['n'] > 1]
        with_small = [t for t in nz if t['small']]
        good = [t for t in nz if t['score'] > 0.88]
        good_small = [t for t in good if t['small']]
        stable = [t for t in good if t['stab'] >= 0.95]
        stable_small = [t for t in stable if t['small']]
        all_small = [v for t in nz for v in t['small']]
        log(f'{iname} {k}: {len(stats)} prompts, {len(multi)} masks with more than one piece ({100 * len(multi) / len(nz):.0f}%), '
            f'{len(with_small)} with stray pieces under 100 pixels ({100 * len(with_small) / len(nz):.0f}%), '
            f'{len(all_small)} such pieces in total, median size {int(np.median(all_small)) if all_small else 0} pixels')
        log(f'   only masks with predicted IoU > 0.88: {len(good)} masks, {len(good_small)} with stray pieces under 100 pixels '
            f'({100 * len(good_small) / max(1, len(good)):.0f}%)')
        log(f'   and also stable (IoU of the masks at logit +1 and -1 >= 0.95): {len(stable)} masks, {len(stable_small)} with stray pieces '
            f'under 100 pixels ({100 * len(stable_small) / max(1, len(stable)):.0f}%)')
        R['components'].setdefault(iname, {})[k] = dict(prompts=len(stats), nonempty=len(nz), multi=len(multi), with_small=len(with_small),
                                                         small_total=len(all_small), small_median=int(np.median(all_small)) if all_small else 0,
                                                         good=len(good), good_small=len(good_small), stable=len(stable), stable_small=len(stable_small),
                                                         pct_multi=100 * len(multi) / len(nz), pct_small=100 * len(with_small) / len(nz))
# one example picture: the confident mask with the most stray pieces
speck_examples.sort(key=lambda e: -e[7])
tiles, labels = [], []
used = set()
for iname, image, m, lab, sizes, pt, sc, ns in speck_examples:
    if iname in used:
        continue
    used.add(iname)
    main_id = int(np.argmax(sizes)) + 1
    t = overlay(image, m)
    d = ImageDraw.Draw(t)
    for cid, v in enumerate(sizes, start=1):
        if cid != main_id and v < 100:
            yy_, xx_ = np.nonzero(lab == cid)
            cx, cy = xx_.mean(), yy_.mean()
            d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], outline=(255, 230, 0), width=2)
    x, y = pt
    d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(255, 255, 255)); d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(40, 220, 90))
    t = t.resize((int(t.width * 0.9), int(t.height * 0.9)), Image.LANCZOS)
    tiles.append(t)
    labels.append((f'{iname} photo, ViT-H, click ({x:.0f}, {y:.0f})', f'predicted IoU {sc:.3f}  ·  {ns} stray pieces under 100 px (yellow rings)'))
    log(f'example {iname}: click ({x:.0f}, {y:.0f}), predicted IoU {sc:.3f}, main piece {int(sizes.max()):,} px, '
        f'{ns} stray pieces under 100 px: sizes {sorted(int(v) for i_, v in enumerate(sizes) if i_ + 1 != main_id and v < 100)}')
if tiles:
    hh = min(t.height for t in tiles)
    tiles = [t.crop((0, 0, t.width, hh)) for t in tiles]
    ww = max(t.width for t in tiles)
    tiles = [t if t.width == ww else t.resize((ww, int(t.height * ww / t.width))) for t in tiles]
    hh = min(t.height for t in tiles)
    tiles = [t.crop((0, 0, ww, hh)) for t in tiles]
    grid(tiles, labels, os.path.join(IMG, 'p6-specks.png'), 1)

# ------------------------------------------------------------------ 4. boundary crispness
log('\n== 4. boundary crispness: the low-resolution logits and the soft band ==')
R['boundary'] = {}
for k, s in sams.items():
    low, mk, prob, sc = s.run(bike, points=[WHEEL_PT])
    b = 1                                              # the wheel mask, as in section 1
    m, p = mk[0, b], prob[0, b]
    edge = m ^ ndi.binary_erosion(m)
    band = (p > 0.1) & (p < 0.9)
    width = band.sum() / edge.sum()
    scale = max(bike.size) / 256
    log(f'{k}: low-resolution logits {tuple(low.shape[-2:])}; one cell covers {scale:.2f} x {scale:.2f} photo pixels')
    log(f'{k}: wheel mask edge {int(edge.sum()):,} pixels long; pixels with probability between 0.1 and 0.9: {int(band.sum()):,}; '
        f'mean width of the soft band {width:.2f} pixels')
    R['boundary'][k] = dict(cell=scale, edge=int(edge.sum()), band=int(band.sum()), width=float(width))
    if k == 'ViT-H':
        # the zoom picture: photo, the 256 x 256 cells (nearest), the final mask, all on the same crop
        cx0, cy0, cs = 440, 200, 64
        crop = bike.crop((cx0, cy0, cx0 + cs, cy0 + cs))
        Z = 5
        t1 = crop.resize((cs * Z, cs * Z), Image.NEAREST)
        lowm = low[0, b] > 0                                   # 256 x 256, covers the padded 1024 x 1024 square
        cells = np.kron(lowm, np.ones((4, 4), bool))[:int(round(bike.size[1] * 1024 / 640)), :1024]
        cells = np.array(Image.fromarray(cells).resize(bike.size, Image.NEAREST))
        t2 = overlay(bike, cells, a=0.6).crop((cx0, cy0, cx0 + cs, cy0 + cs)).resize((cs * Z, cs * Z), Image.NEAREST)
        t3 = overlay(bike, m, a=0.6).crop((cx0, cy0, cx0 + cs, cy0 + cs)).resize((cs * Z, cs * Z), Image.NEAREST)
        pc = (np.clip(p, 0, 1) * 255).astype('uint8')
        t4 = Image.fromarray(pc).convert('RGB').crop((cx0, cy0, cx0 + cs, cy0 + cs)).resize((cs * Z, cs * Z), Image.NEAREST)
        grid([t1, t2, t3, t4],
             [('the photo', f'64 x 64 pixels around the tyre edge'),
              ('the 256 x 256 logits > 0', f'one cell = {scale:.1f} x {scale:.1f} pixels'),
              ('the final mask', 'logits upscaled, then > 0'),
              ('the probability', f'white = 1, black = 0; band {width:.1f} px')],
             os.path.join(IMG, 'p6-boundary.png'), 4)
        R['zoom'] = dict(x0=cx0, y0=cy0, size=cs)
        diff = (cells != m).sum()
        log(f'{k}: the final mask and the 256 x 256 cell mask (blown up without interpolation) differ in {int(diff):,} pixels, '
            f'{100 * diff / m.sum():.1f}% of the mask area')

log('\n== 4b. a drawn edge with teeth of known width (1024 x 1024, box prompt): how fine can the edge be? ==')
TOP, H_T = 520, 16


def comb_image(t):
    a = np.full((N, N), 215, np.uint8)
    a[TOP:900, 112:912] = 30
    if t:
        for x in range(112, 912):
            if ((x - 112) // t) % 2 == 0:
                a[TOP - H_T:TOP, x] = 30
    return Image.fromarray(a).convert('RGB'), a < 120


log(f'teeth {H_T} pixels high on the top edge of a dark block; tooth width = gap width = t')
log(f'{"model":<7}{"t":>6}{"tooth recall":>14}{"gap fill":>10}{"edge error (no teeth)":>24}')
R['comb'] = {k: [] for k in sams}
comb_tiles, comb_labels = [], []
for k, s in sams.items():
    for t in [0, 1, 2, 4, 8, 16, 32]:
        im, gt = comb_image(t)
        _, m1, _, _ = s.run(im, box=(100, TOP - H_T - 12, 924, 912), multimask=False)
        s.cache.clear()
        mk = m1[0, 0]
        band = np.zeros_like(gt); band[TOP - H_T:TOP, 112:912] = True
        if t == 0:
            col = mk[:, 112:912]
            top = np.array([np.argmax(c) if c.any() else N for c in col.T])
            err = float(np.abs(top - TOP).mean())
            log(f'{k:<7}{"none":>6}{"":>14}{"":>10}{err:>21.2f} px')
            R['comb'][k].append(dict(t=0, edge_err=err))
            continue
        teeth, gaps = gt & band, (~gt) & band
        rec, fill = (mk & teeth).sum() / teeth.sum(), (mk & gaps).sum() / gaps.sum()
        log(f'{k:<7}{t:>4} px{rec:>14.3f}{fill:>10.3f}')
        R['comb'][k].append(dict(t=t, recall=float(rec), gap_fill=float(fill)))
        if k == 'ViT-H' and t in (2, 4, 8, 16):
            c = overlay(im, mk, a=0.6).crop((300, TOP - 40, 364, TOP + 24)).resize((256, 256), Image.NEAREST)
            comb_tiles.append(c)
            comb_labels.append((f'ViT-H, teeth {t} px', f'recall {rec:.2f} · gap fill {fill:.2f}'))
grid(comb_tiles, comb_labels, os.path.join(IMG, 'p6-teeth.png'), 4)

# ------------------------------------------------------------------ 5. ViT-B against ViT-H: do they agree?
log('\n== 5. ViT-B against ViT-H on the same prompts (masks of ViT-H taken as the reference) ==')
R['agree'] = {}
for iname, image in [('bike', bike), ('cats', cats)]:
    W0, H0 = image.size
    xs = (np.arange(16) + 0.5) * W0 / 16
    ys = (np.arange(16) + 0.5) * H0 / 16
    pts = [(float(x), float(y)) for y in ys for x in xs]
    own, close = [], []
    for i in range(0, len(pts), 64):
        _, mh, _, sh = sams['ViT-H'].run(image, points=pts[i:i + 64])
        _, mb, _, sb = sams['ViT-B'].run(image, points=pts[i:i + 64])
        for j in range(mh.shape[0]):
            a = mh[j, int(np.argmax(sh[j]))]                   # what a user of ViT-H gets
            if not a.any():
                continue
            ious = [(a & mb[j, q]).sum() / max(1, (a | mb[j, q]).sum()) for q in range(3)]
            own.append(ious[int(np.argmax(sb[j]))])            # what a user of ViT-B gets
            close.append(max(ious))                            # the closest of ViT-B's three masks
    for nm, v in [('ViT-B top mask', np.array(own)), ('closest of 3 ViT-B masks', np.array(close))]:
        log(f'{iname}, {nm}: {len(v)} prompts, mean IoU with the ViT-H top mask {v.mean():.3f}, median {np.median(v):.3f}, '
            f'{100 * (v > 0.9).mean():.0f}% above 0.9, {100 * (v < 0.5).mean():.0f}% below 0.5')
        R['agree'].setdefault(iname, {})[nm] = dict(n=len(v), mean=float(v.mean()), median=float(np.median(v)),
                                                   above90=float((v > 0.9).mean()), below50=float((v < 0.5).mean()))

save('part6', R)
log('\nsaved results/part6.json')
