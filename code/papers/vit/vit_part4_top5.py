"""Part 4: what "top-1" and "top-5" accuracy mean, and why ImageNet needed new labels (ImageNet ReaL).

Table 2 reports ImageNet top-1 accuracy, Appendix D.9 reports ObjectNet top-1 and top-5. This script runs three
released ImageNet classifiers on the 3,925 validation pictures of Imagenette (a 10-class subset of ImageNet's
validation pictures, here at 160 px on the short side) and scores their 1000-way predictions:
  - top-1: the highest-scoring class is the true one; top-5: the true class is among the five highest;
  - then it saves a picture of pictures where ViT-L/16's top-1 answer is "wrong" but its top-5 contains the label:
    many show a second object, which is the problem ImageNet ReaL relabels.
Models: google/vit-base-patch16-224 and google/vit-large-patch16-224 (ImageNet-21k, then fine-tuned on ImageNet)
and microsoft/resnet-50 (ImageNet only), each with its own released preprocessing. What it cannot show: 10 easy classes are not the 1000 of ImageNet, so the accuracies are far
above the paper's ImageNet numbers; and these pictures may have been seen in pre-training (the ImageNet-21k set
contains ImageNet's classes), so this is a lesson on the metrics, not a benchmark.
Writes results/part4_top5.json, results/part4_top5_stdout.txt and public/img/papers/vit/p4-top5-misses.png."""
import glob, os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModelForImageClassification
from common import Log, save

log = Log('part4_top5')
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
ROOT = os.path.expanduser('~/.cache/papers/imagenette/imagenette2-160/val')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'public', 'img', 'papers', 'vit')
SYN = {'n01440764': 0, 'n02102040': 217, 'n02979186': 482, 'n03000684': 491, 'n03028079': 497,
       'n03394916': 566, 'n03417042': 569, 'n03425413': 571, 'n03445777': 574, 'n03888257': 701}
MODELS = [('ViT-B/16', 'google/vit-base-patch16-224'), ('ViT-L/16', 'google/vit-large-patch16-224'),
          ('ResNet-50', 'microsoft/resnet-50')]
files = sorted(glob.glob(os.path.join(ROOT, '*', '*.JPEG')))
labels = np.array([SYN[f.split('/')[-2]] for f in files])
R = {}


@torch.no_grad()
def run(mid, bs=32):
    proc = AutoImageProcessor.from_pretrained(mid)
    model = AutoModelForImageClassification.from_pretrained(mid).to(DEV).eval()
    names = model.config.id2label
    top = []
    for i in range(0, len(files), bs):
        ims = [Image.open(f).convert('RGB') for f in files[i:i + bs]]
        logits = model(pixel_values=proc(images=ims, return_tensors='pt')['pixel_values'].to(DEV)).logits.float()
        top.append(logits.topk(5, dim=1).indices.cpu())
    del model
    return torch.cat(top).numpy(), names


log(f'== top-1 and top-5 accuracy over all 1000 ImageNet classes, on the {len(files):,} Imagenette validation pictures ==')
log(f'  the 10 classes: ' + ', '.join(f'{i}' for i in SYN.values()))
first = None
for name, mid in MODELS:
    top, names = run(mid)
    t1 = (top[:, 0] == labels).mean() * 100
    t5 = (top == labels[:, None]).any(1).mean() * 100
    miss = np.flatnonzero((top[:, 0] != labels) & (top == labels[:, None]).any(1))
    R[name] = dict(top1=round(t1, 2), top5=round(t5, 2), n_top1_wrong=int((top[:, 0] != labels).sum()), n_saved_by_top5=int(len(miss)))
    log(f'  {name:<12} top-1 {t1:5.2f}%   top-5 {t5:5.2f}%   top-1 wrong on {R[name]["n_top1_wrong"]} pictures, '
        f'of which {len(miss)} have the label in the top 5')
    if name == 'ViT-L/16':
        first = (top, names, miss)

top, names, miss = first
short = lambda i: names[int(i)].split(',')[0]
log('')
log('  ViT-L/16, pictures whose top-1 is not the label but whose top-5 contains it (8 picked at random, seed 3):')
picked = []
rng = np.random.default_rng(3)
for i in sorted(rng.choice(miss, size=min(8, len(miss)), replace=False)):
    rank = int(np.flatnonzero(top[i] == labels[i])[0]) + 1
    picked.append(dict(file=os.path.basename(files[i]), label=short(labels[i]), top1=short(top[i, 0]), rank=rank))
    log(f'    {os.path.basename(files[i]):<28} label: {short(labels[i]):<16} top-1: {short(top[i, 0]):<22} label at rank {rank}')
R['misses'] = picked

# a picture of the eight
cell, pad, cap = 220, 10, 46
W = 4 * cell + 5 * pad
H = 2 * (cell + cap) + 3 * pad
im = Image.new('RGB', (W, H), (255, 255, 255))
d = ImageDraw.Draw(im)
font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 15)
for n, p in enumerate(picked):
    r, c = divmod(n, 4)
    x, y = pad + c * (cell + pad), pad + r * (cell + cap + pad)
    ph = Image.open(os.path.join(ROOT, [f for f in files if f.endswith(p['file'])][0].split('/')[-2], p['file'])).convert('RGB')
    s = min(ph.size)
    ph = ph.crop(((ph.width - s) // 2, (ph.height - s) // 2, (ph.width - s) // 2 + s, (ph.height - s) // 2 + s)).resize((cell, cell))
    im.paste(ph, (x, y))
    d.text((x, y + cell + 4), f'label: {p["label"]}', fill=(30, 30, 30), font=font)
    d.text((x, y + cell + 24), f'top-1: {p["top1"]}', fill=(190, 60, 20), font=font)
im.save(os.path.join(OUT, 'p4-top5-misses.png'), optimize=True)
log(f'  saved p4-top5-misses.png ({W}x{H})')
save('part4_top5', R)
