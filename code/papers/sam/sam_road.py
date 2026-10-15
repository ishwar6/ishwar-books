"""Background page (the road to SAM): YOLO and SAM on the same picture.
- YOLO11n-seg (Ultralytics): one pass over the picture gives boxes, class names, scores and masks for its 80 COCO classes
- SAM ViT-H: the YOLO boxes become box prompts, and SAM gives one mask for each box (no class names)
- the agreement (IoU) between the YOLO mask and the SAM mask of each object, and the time of each model
- what each model cannot do: YOLO has no class "ear"; SAM gives no class names
Writes results/road.json, results/road_stdout.txt and public/img/papers/sam/road-yolo-sam.png."""
import os, time
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset
from transformers import SamModel, SamProcessor
from ultralytics import YOLO
from common import Log, save

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, '..', '..', '..', 'public', 'img', 'papers', 'sam')
log = Log('road')
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
R = {}

image = load_dataset('huggingface/cats-image')['test'][0]['image'].convert('RGB')
W0, H0 = image.size
arr = np.array(image)
log(f'picture: {W0} x {H0} pixels')

# ------------------------------------------------------------------ 1. YOLO: one pass, fixed list of classes
yolo = YOLO('yolo11n-seg.pt')
n_par = sum(p.numel() for p in yolo.model.parameters())
names = yolo.model.names
log(f'\n== YOLO11n-seg ==\nparameters: {n_par:,}   classes it knows: {len(names)}')
log('the first 20 classes: ' + ', '.join(names[i] for i in range(20)))
log(f'is "ear" a class? {"ear" in names.values()}   is "button" a class? {"button" in names.values()}')
res = yolo.predict(image, conf=0.25, retina_masks=True, verbose=False, device=DEV)[0]
dets = []
for k in range(len(res.boxes)):
    cls = names[int(res.boxes.cls[k])]
    conf = float(res.boxes.conf[k])
    box = [round(float(v)) for v in res.boxes.xyxy[k].tolist()]
    m = res.masks.data[k].cpu().numpy() > 0.5
    dets.append(dict(cls=cls, conf=round(conf, 3), box=box, yolo_area=int(m.sum())))
    dets[-1]['_mask'] = m
log(f'{"object":<10}{"score":>7}   {"box (x1, y1, x2, y2)":<24}{"mask pixels":>12}')
for d in dets:
    log(f'{d["cls"]:<10}{d["conf"]:>7.3f}   {str(tuple(d["box"])):<24}{d["yolo_area"]:>12,}')


def bench(fn, n=20):
    fn(); fn()
    t = time.time()
    for _ in range(n):
        fn()
    return 1000 * (time.time() - t) / n


t_yolo = {dev: bench(lambda: yolo.predict(image, verbose=False, device=dev)) for dev in [DEV, 'cpu']}
log(f'time for the whole picture (all objects): {t_yolo[DEV]:.1f} ms ({DEV})   {t_yolo["cpu"]:.1f} ms (cpu)')

# ------------------------------------------------------------------ 2. SAM: the YOLO boxes become prompts
proc = SamProcessor.from_pretrained('facebook/sam-vit-huge')
sam = SamModel.from_pretrained('facebook/sam-vit-huge').to(DEV).eval()
sam_par = sum(p.numel() for p in sam.parameters())
inp = proc(images=image, return_tensors='pt')
with torch.no_grad():
    emb = sam.get_image_embeddings(inp['pixel_values'].to(DEV))
log(f'\n== SAM ViT-H, prompted with the YOLO boxes ==\nparameters: {sam_par:,} ({sam_par / n_par:.0f} times YOLO11n-seg)')
log(f'{"object":<10}{"SAM score":>10}{"SAM pixels":>12}{"YOLO pixels":>13}{"IoU(YOLO, SAM)":>16}')
for d in dets:
    p = proc(images=image, input_boxes=[[d['box']]], return_tensors='pt')
    with torch.no_grad():
        out = sam(image_embeddings=emb, input_boxes=p['input_boxes'].float().to(DEV), multimask_output=False)
    m = proc.image_processor.post_process_masks(out.pred_masks.cpu(), p['original_sizes'], p['reshaped_input_sizes'])[0][0, 0].numpy()
    a, b = d['_mask'], m
    iou = (a & b).sum() / (a | b).sum()
    d.update(sam_score=round(float(out.iou_scores[0, 0, 0]), 3), sam_area=int(m.sum()), iou=round(float(iou), 3), _sam=m)
    log(f'{d["cls"]:<10}{d["sam_score"]:>10.3f}{d["sam_area"]:>12,}{d["yolo_area"]:>13,}{d["iou"]:>16.3f}')


def enc():
    with torch.no_grad():
        sam.get_image_embeddings(inp['pixel_values'].to(DEV))
    if DEV == 'mps':
        torch.mps.synchronize()


t_enc = bench(enc, n=5)
log(f'time: SAM image encoder {t_enc:.1f} ms ({DEV}), then about 6 ms for each box (Part 1)')
log(f'YOLO is {t_enc / t_yolo[DEV]:.0f} times faster on this picture, and it also names the objects')

R = dict(yolo=dict(params=n_par, classes=len(names), ms=t_yolo), sam=dict(params=sam_par, enc_ms=t_enc),
         dets=[{k: v for k, v in d.items() if not k.startswith('_')} for d in dets])
save('road', R)

# ------------------------------------------------------------------ 3. the picture: YOLO left, SAM right
COL = [(255, 196, 0), (64, 200, 255), (255, 90, 120), (120, 230, 120), (190, 120, 255), (255, 150, 60)]
try:
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 15)
except OSError:
    font = ImageFont.load_default()


def panel(key, label):
    out = arr.astype(float).copy()
    for i, d in enumerate(dets):
        m = d[key]
        out[m] = 0.45 * out[m] + 0.55 * np.array(COL[i % len(COL)])
    im = Image.fromarray(out.astype(np.uint8))
    dr = ImageDraw.Draw(im)
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = d['box']
        dr.rectangle([x1, y1, x2 - 1, y2 - 1], outline=COL[i % len(COL)], width=3)
        if label:
            t = f'{d["cls"]} {d["conf"]:.2f}'
            tw = dr.textlength(t, font=font)
            ty = y1 + 2 if y1 < 22 else y1 - 21
            dr.rectangle([x1, ty, x1 + tw + 8, ty + 19], fill=COL[i % len(COL)])
            dr.text((x1 + 4, ty + 1), t, fill=(0, 0, 0), font=font)
    return im


a, b = panel('_mask', True), panel('_sam', False)
gap = 16
both = Image.new('RGB', (2 * W0 + gap, H0), (255, 255, 255))
both.paste(a, (0, 0)); both.paste(b, (W0 + gap, 0))
both.save(os.path.join(IMG, 'road-yolo-sam.png'), optimize=True)
log('\nsaved road-yolo-sam.png')
