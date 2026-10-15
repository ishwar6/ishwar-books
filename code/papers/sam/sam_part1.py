"""Part 1: SAM on one picture.
- the input pipeline: 640x480 photo -> 1024x768 -> padded 1024x1024 -> 256x64x64 image embedding
- parameter counts of the three released models (ViT-B, ViT-L, ViT-H), split into image encoder, prompt encoder, mask decoder
- one click on a cat's ear and one click on a remote control: three masks each, with SAM's own scores
- one box prompt
- timing: the image encoder once, then the prompt encoder + mask decoder per prompt (the "amortized real-time" claim)
Writes results/part1.json, results/part1_stdout.txt and the overlay pictures in public/img/papers/sam/."""
import os, time
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset
from transformers import SamModel, SamProcessor
from common import Log, save

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, '..', '..', '..', 'public', 'img', 'papers', 'sam')
os.makedirs(IMG, exist_ok=True)
log = Log('part1')
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
R = {}

image = load_dataset('huggingface/cats-image')['test'][0]['image'].convert('RGB')
W0, H0 = image.size
log(f'picture: {W0} x {H0} pixels (width x height)')


# ------------------------------------------------------------------ 1. parameter counts of the three models
def count(mod):
    return sum(p.numel() for p in mod.parameters())


log('\n== parameters of the three released SAM models ==')
log(f'{"model":<20}{"image encoder":>16}{"prompt encoder":>16}{"mask decoder":>14}{"total":>16}')
R['params'] = {}
for name in ['facebook/sam-vit-base', 'facebook/sam-vit-large', 'facebook/sam-vit-huge']:
    m = SamModel.from_pretrained(name)
    enc, pe, dec = count(m.vision_encoder), count(m.prompt_encoder), count(m.mask_decoder)
    tot = count(m)
    assert enc + pe + dec == tot
    R['params'][name] = dict(image_encoder=enc, prompt_encoder=pe, mask_decoder=dec, total=tot,
                             layers=m.config.vision_config.num_hidden_layers, width=m.config.vision_config.hidden_size,
                             heads=m.config.vision_config.num_attention_heads,
                             global_layers=list(m.config.vision_config.global_attn_indexes))
    log(f'{name:<20}{enc:>16,}{pe:>16,}{dec:>14,}{tot:>16,}')
    del m
for name, p in R['params'].items():
    log(f'  {name}: {p["layers"]} layers, width {p["width"]}, {p["heads"]} heads, global attention in layers {p["global_layers"]}; '
        f'encoder = {100 * p["image_encoder"] / p["total"]:.1f}% of all weights')

# ------------------------------------------------------------------ 2. the input pipeline, with ViT-H (the paper's default)
log('\n== the input pipeline (facebook/sam-vit-huge) ==')
proc = SamProcessor.from_pretrained('facebook/sam-vit-huge')
sam = SamModel.from_pretrained('facebook/sam-vit-huge').to(DEV).eval()
inp = proc(images=image, return_tensors='pt')
rh, rw = inp['reshaped_input_sizes'][0].tolist()
px = inp['pixel_values']
log(f'resize: longest side {max(W0, H0)} -> 1024, scale {1024 / max(W0, H0):.4f}; picture becomes {rw} x {rh}')
log(f'pad: {rw} x {rh} -> {px.shape[-1]} x {px.shape[-2]} (zeros added below: {px.shape[-2] - rh} rows)')
log(f'pixel tensor: {tuple(px.shape)} = {px[0].numel():,} numbers')
with torch.no_grad():
    emb = sam.get_image_embeddings(px.to(DEV))
log(f'image embedding: {tuple(emb.shape)} = {emb[0].numel():,} numbers (each of 64 x 64 = 4,096 cells covers 16 x 16 pixels)')
R['pipeline'] = dict(orig=[W0, H0], resized=[rw, rh], padded=list(px.shape[-2:]), pixel_numbers=px[0].numel(),
                     embedding=list(emb.shape), embedding_numbers=emb[0].numel())


# ------------------------------------------------------------------ 3. point prompts: three masks per click
def predict(points=None, labels=None, boxes=None, multimask=True):
    kw = {}
    if points is not None:
        kw['input_points'] = [[points]]
        kw['input_labels'] = [[labels]]
    if boxes is not None:
        kw['input_boxes'] = [[boxes]]
    p = proc(images=image, return_tensors='pt', **kw)
    args = {k: p[k].float().to(DEV) if k != 'input_labels' else p[k].to(DEV) for k in ('input_points', 'input_labels', 'input_boxes') if k in p}
    with torch.no_grad():
        out = sam(image_embeddings=emb, multimask_output=multimask, **args)
    masks = proc.image_processor.post_process_masks(out.pred_masks.cpu(), p['original_sizes'], p['reshaped_input_sizes'])[0][0]
    return masks.numpy(), out.iou_scores[0, 0].cpu().numpy(), out.pred_masks.shape


def iou(a, b):
    return (a & b).sum() / (a | b).sum()


R['clicks'] = {}
for key, (x, y), what in [('ear', (275, 62), 'the left cat\'s ear'), ('remote', (80, 90), 'the buttons of the left remote control')]:
    masks, scores, shape = predict([[x, y]], [1])
    log(f'\n== one click at ({x}, {y}) on {what} ==')
    log(f'low-resolution mask logits from the decoder: {tuple(shape)}  (batch, prompts, 3 masks, 256, 256)')
    rows = []
    for k in range(3):
        n = int(masks[k].sum())
        rows.append(dict(score=float(scores[k]), pixels=n, percent=100 * n / (W0 * H0)))
        log(f'  mask {k + 1}: predicted IoU {scores[k]:.3f}   {n:>7,} pixels = {100 * n / (W0 * H0):6.2f}% of the picture')
    best = int(np.argmax(scores))
    log(f'  SAM ranks mask {best + 1} first (highest predicted IoU)')
    pair = {f'{i + 1}-{j + 1}': float(iou(masks[i], masks[j])) for i in range(3) for j in range(i + 1, 3)}
    log('  IoU between the masks: ' + ', '.join(f'{k}: {v:.3f}' for k, v in pair.items()))
    R['clicks'][key] = dict(point=[x, y], masks=rows, best=best + 1, pair_iou=pair)
    np.save(os.path.join(HERE, 'results', f'part1_{key}_masks.npy'), masks)

# the worked IoU example: masks 1 and 2 of the ear click
m = np.load(os.path.join(HERE, 'results', 'part1_ear_masks.npy'))
inter, union = int((m[0] & m[1]).sum()), int((m[0] | m[1]).sum())
log(f'\nworked IoU, ear mask 1 vs ear mask 2: intersection {inter} pixels, union {union} pixels, IoU = {inter}/{union} = {inter / union:.3f}')
R['ear_iou'] = dict(a=int(m[0].sum()), b=int(m[1].sum()), inter=inter, union=union, iou=inter / union)

# one click, one mask: what the ambiguity-unaware output looks like
masks1, score1, _ = predict([[275, 62]], [1], multimask=False)
log(f'same ear click with multimask_output=False: 1 mask, predicted IoU {score1[0]:.3f}, {int(masks1[0].sum()):,} pixels')
R['ear_single'] = dict(score=float(score1[0]), pixels=int(masks1[0].sum()))

# two clicks remove the ambiguity: ear (foreground) + body (foreground)
masks2, score2, _ = predict([[275, 62], [150, 250]], [1, 1], multimask=False)
log(f'two clicks (ear + body), one mask: predicted IoU {score2[0]:.3f}, {int(masks2[0].sum()):,} pixels')
R['two_clicks'] = dict(score=float(score2[0]), pixels=int(masks2[0].sum()))

# ------------------------------------------------------------------ 4. a box prompt
box = [255, 20, 640, 360]            # x0, y0, x1, y1 around the right cat
mb, sb, _ = predict(boxes=box, multimask=False)
log(f'\n== a box prompt {box} around the right cat ==')
log(f'  1 mask, predicted IoU {sb[0]:.3f}, {int(mb[0].sum()):,} pixels = {100 * mb[0].mean():.2f}% of the picture')
R['box'] = dict(box=box, score=float(sb[0]), pixels=int(mb[0].sum()))
np.save(os.path.join(HERE, 'results', 'part1_box_mask.npy'), mb[0])


# ------------------------------------------------------------------ 5. timing: encoder once, decoder per prompt
def timeit(fn, n, warm=2):
    for _ in range(warm):
        fn()
    if DEV == 'mps':
        torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    if DEV == 'mps':
        torch.mps.synchronize()
    return (time.perf_counter() - t) / n


log('\n== timing on this computer (Apple M5 Pro) ==')
R['timing'] = {}
pts = torch.tensor([[[[275.0, 62.0]]]]) * 1024 / max(W0, H0)
lab = torch.tensor([[[1]]])
for name in ['facebook/sam-vit-base', 'facebook/sam-vit-huge']:
    mm = SamModel.from_pretrained(name).eval()
    T = {}
    for dev in ['mps', 'cpu']:
        mm.to(dev)
        x = px.to(dev)
        with torch.no_grad():
            e = mm.get_image_embeddings(x)
            T[f'encoder_{dev}'] = timeit(lambda: mm.get_image_embeddings(x), 3 if dev == 'cpu' else 10, 1)
            p, l = pts.to(dev), lab.to(dev)
            T[f'decoder_{dev}'] = timeit(lambda: mm(image_embeddings=e, input_points=p, input_labels=l, multimask_output=True), 50, 5)
    R['timing'][name] = T
    log(f'{name}: image encoder {T["encoder_mps"] * 1000:7.1f} ms (GPU)  {T["encoder_cpu"] * 1000:8.1f} ms (CPU)   '
        f'prompt encoder + mask decoder {T["decoder_mps"] * 1000:5.1f} ms (GPU)  {T["decoder_cpu"] * 1000:5.1f} ms (CPU)')
    del mm
T = R['timing']['facebook/sam-vit-huge']
log('\namortized time per prompt, ViT-H, encoder on the GPU and decoder on the CPU:  (T_enc + k * T_dec) / k')
R['amortized'] = []
for k in [1, 10, 100]:
    a = (T['encoder_mps'] + k * T['decoder_cpu']) / k
    R['amortized'].append(dict(k=k, ms=a * 1000))
    log(f'  k = {k:>3} prompts: ({T["encoder_mps"] * 1000:.0f} + {k} x {T["decoder_cpu"] * 1000:.1f}) / {k} = {a * 1000:7.1f} ms per prompt')

# ------------------------------------------------------------------ 6. dataset arithmetic from the paper
log('\n== the dataset numbers of the paper ==')
log(f'SA-1B: 1,129M masks / 11M images = {1129 / 11:.0f} masks per image')
log(f'Open Images: 2.7M masks / 1M images = {2.7 / 1:.1f} masks per image; SA-1B / Open Images = {1129 / 2.7:.0f}x masks, {11 / 1:.0f}x images, '
    f'{(1129 / 11) / 2.7:.0f}x masks per image')
save('part1', R)


# ------------------------------------------------------------------ 7. pictures for the article
def font(size):
    for f in ['/System/Library/Fonts/Helvetica.ttc', '/System/Library/Fonts/Menlo.ttc']:
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


COLS = [(255, 196, 0), (64, 200, 255), (60, 90, 255)]


def overlay(mask, col, a=0.55):
    arr = np.array(image).astype(float)
    arr[mask] = (1 - a) * arr[mask] + a * np.array(col)
    im = Image.fromarray(arr.astype('uint8'))
    # outline the mask so the edge is easy to see
    edge = mask ^ np.pad(mask, 1, mode='edge')[:-2, 1:-1] | mask ^ np.pad(mask, 1, mode='edge')[1:-1, :-2]
    e = np.array(im)
    e[edge] = (255, 255, 255)
    return Image.fromarray(e)


def dot(im, x, y, r=7, col=(40, 220, 90)):
    d = ImageDraw.Draw(im)
    d.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2], fill=(255, 255, 255))
    d.ellipse([x - r, y - r, x + r, y + r], fill=col)
    return im


def panels(tiles, labels, path, scale=1.0):
    """Tiles side by side, each with a two-line caption on a dark strip under it."""
    w, h = tiles[0].size
    w, h = int(w * scale), int(h * scale)
    cap = 64
    out = Image.new('RGB', (len(tiles) * w + (len(tiles) - 1) * 12, h + cap), (24, 25, 33))
    d = ImageDraw.Draw(out)
    for i, (t, (l1, l2)) in enumerate(zip(tiles, labels)):
        x = i * (w + 12)
        out.paste(t.resize((w, h), Image.LANCZOS), (x, 0))
        d.text((x + 10, h + 9), l1, font=font(22), fill=(235, 236, 245))
        d.text((x + 10, h + 36), l2, font=font(18), fill=(160, 162, 180))
    out.save(path, optimize=True)
    print('wrote', path, out.size)


# the picture with the two clicks marked
im = dot(dot(image.copy(), 275, 62), 80, 90)
im.save(os.path.join(IMG, 'p1-sample.png'))

names = {'ear': ['ear (small)', 'ear (larger)', 'whole cat'], 'remote': ['buttons', 'whole remote', 'sofa back, with holes']}
for key, crop in [('ear', (165, 2, 385, 162)), ('remote', (0, 30, 220, 190))]:
    ms = np.load(os.path.join(HERE, 'results', f'part1_{key}_masks.npy'))
    x, y = R['clicks'][key]['point']
    tiles, labels = [], []
    for k in range(3):
        t = dot(overlay(ms[k], COLS[k]), x, y, 5).crop(crop)
        tiles.append(t)
        r = R['clicks'][key]['masks'][k]
        labels.append((f'mask {k + 1}: {names[key][k]}', f'score {r["score"]:.3f}  ·  {r["pixels"]:,} pixels'))
    panels(tiles, labels, os.path.join(IMG, f'p1-{key}-masks.png'), scale=2.0)
    # the same three masks on the full picture
    tiles = [dot(overlay(ms[k], COLS[k]), x, y) for k in range(3)]
    panels(tiles, [(f'mask {k + 1}: {names[key][k]}', f'score {R["clicks"][key]["masks"][k]["score"]:.3f}') for k in range(3)],
           os.path.join(IMG, f'p1-{key}-masks-full.png'))

# box prompt
t = overlay(np.load(os.path.join(HERE, 'results', 'part1_box_mask.npy')), COLS[1])
d = ImageDraw.Draw(t)
d.rectangle(box, outline=(40, 220, 90), width=4)
panels([t], [('box prompt: one mask', f'score {R["box"]["score"]:.3f}  ·  {R["box"]["pixels"]:,} pixels')],
       os.path.join(IMG, 'p1-box-mask.png'))
