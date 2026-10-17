"""Part 4: Section 4.3 in miniature: train a small ViT and a small ResNet from scratch on growing subsets of CIFAR-10.

Section 4.3 trains ViTs and ResNets on random subsets of JFT-300M (9M, 30M, 90M, 300M images) with the same
hyper-parameters for every size, early stopping and "the best validation accuracy achieved during training".
We cannot pre-train on JFT, so this script asks the same question at a scale a laptop can afford:
  - data: random class-balanced subsets of the 50,000 CIFAR-10 training images (1k, 5k, 50k), each
    subset inside the next one; validation = the first 5,000 CIFAR-10 test images;
  - models of similar compute: a ViT (the transformers ViT code, 32x32 input, 4x4 patches, D = 128, 6 layers,
    4 heads, MLP 512) and a ResNet (CIFAR-style ResNet-18 layout with a narrow width), parameters and
    multiply-adds counted;
  - one recipe for both and for every size: AdamW, learning rate 1e-3 with linear warm-up and cosine decay,
    weight decay 0.05, batch 128, the same number of steps, random crop and flip; validation every 250 steps,
    best accuracy kept (early stopping).
What it cannot show: 32x32 pictures and at most 50,000 of them (JFT has 300 million); one run per point (no
error bars); tiny models. It shows the direction of the effect, not the paper's cross-over point.
Writes results/part4_scratch.json and results/part4_scratch_stdout.txt."""
import glob, io, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import pyarrow.parquet as pq
from PIL import Image
from transformers import ViTConfig, ViTForImageClassification
from common import Log, save

log = Log('part4_scratch')
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
HUB = os.path.expanduser('~/.cache/huggingface/hub')
SIZES = [1000, 5000, 50000]
STEPS, WARM, BS, LR, WD, EVAL = int(os.environ.get('STEPS', 2500)), 250, 128, 1e-3, 0.05, 250
torch.manual_seed(0)


def load(split):
    f = glob.glob(os.path.join(HUB, f'datasets--uoft-cs--cifar10/snapshots/*/*/{split}-*.parquet'))[0]
    df = pq.read_table(f).to_pandas()
    x = np.stack([np.array(Image.open(io.BytesIO(b['bytes'])).convert('RGB')) for b in df['img']])
    return torch.tensor(x).permute(0, 3, 1, 2).contiguous(), torch.tensor(df['label'].to_numpy())


xtr, ytr = load('train')
xva, yva = load('test')
xva, yva = xva[:5000], yva[:5000]
MEAN = xtr.float().mean((0, 2, 3), keepdim=True) / 255
STD = (xtr.float() / 255).std((0, 2, 3), keepdim=True)
rng = np.random.default_rng(0)
order = np.concatenate([rng.permutation(np.flatnonzero(ytr.numpy() == c)) for c in range(10)]).reshape(10, -1)
SUBSETS = {n: torch.tensor(order[:, :n // 10].reshape(-1)) for n in SIZES}   # n/10 per class; each subset inside the next


# ---------------------------------------------------------------- the two models
class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout)
        self.c2, self.b2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout)
        self.skip = nn.Sequential() if stride == 1 and cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        h = Fn.relu(self.b1(self.c1(x)))
        return Fn.relu(self.b2(self.c2(h)) + self.skip(x))


class ResNet(nn.Module):
    """CIFAR-style ResNet-18 layout (2-2-2-2 basic blocks, a 3x3 stem, no max-pool) with a chosen width."""
    def __init__(self, w=24):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU())
        ch, layers, cin = [w, 2 * w, 4 * w, 8 * w], [], w
        for i, c in enumerate(ch):
            layers += [Block(cin, c, 1 if i == 0 else 2), Block(c, c, 1)]
            cin = c
        self.layers, self.head = nn.Sequential(*layers), nn.Linear(cin, 10)

    def forward(self, x):
        return self.head(self.layers(self.stem(x)).mean((2, 3)))


class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        c = ViTConfig(image_size=32, patch_size=4, num_channels=3, hidden_size=128, num_hidden_layers=6, num_attention_heads=4,
                      intermediate_size=512, num_labels=10, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
        self.m = ViTForImageClassification(c)

    def forward(self, x):
        return self.m(pixel_values=x).logits


def count(model):
    from torch.utils.flop_counter import FlopCounterMode
    model = model.cpu().eval()
    with FlopCounterMode(display=False) as fc:
        model(torch.zeros(1, 3, 32, 32))
    return sum(p.numel() for p in model.parameters()), fc.get_total_flops() / 2   # multiply-adds per image


# ---------------------------------------------------------------- training
def augment(x):
    """Random crop with 4 pixels of padding, and a random horizontal flip, on the GPU."""
    n, ar = x.shape[0], torch.arange(32, device=x.device)
    xp = Fn.pad(x, (4, 4, 4, 4), mode='reflect')
    i, j = torch.randint(0, 9, (2, n), device=x.device)          # a different crop for every picture
    x = xp[torch.arange(n, device=x.device)[:, None, None], :, (i[:, None] + ar)[:, :, None], (j[:, None] + ar)[:, None, :]].permute(0, 3, 1, 2)
    flip = torch.rand(n, device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], x.flip(3), x)


def norm(xb):
    return (xb.float() / 255 - MEAN.to(xb.device)) / STD.to(xb.device)


@torch.no_grad()
def evaluate(model):
    model.eval()
    correct = 0
    for i in range(0, len(xva), 500):
        xb = norm(xva[i:i + 500].to(DEV))
        correct += (model(xb).argmax(1).cpu() == yva[i:i + 500]).sum().item()
    model.train()
    return 100 * correct / len(xva)


def train(make, n, seed=0):
    torch.manual_seed(seed)
    model = make().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / WARM if s < WARM else 0.5 * (1 + math.cos(math.pi * (s - WARM) / (STEPS - WARM))))
    idx = SUBSETS[n]
    xs, ys = xtr[idx].to(DEV), ytr[idx].to(DEV)
    best, curve, last_loss = 0.0, [], 0.0
    g = torch.Generator().manual_seed(seed)
    for step in range(1, STEPS + 1):
        b = torch.randint(0, n, (BS,), generator=g)
        xb = augment(norm(xs[b]))
        loss = Fn.cross_entropy(model(xb), ys[b])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        last_loss = 0.98 * last_loss + 0.02 * loss.item() if step > 1 else loss.item()
        if step % EVAL == 0:
            acc = evaluate(model)
            best = max(best, acc)
            curve.append([step, round(acc, 2), round(last_loss, 3)])
    del model
    if DEV == 'mps':
        torch.mps.empty_cache()
    return best, curve


p_vit, f_vit = count(ViT())
p_res, f_res = count(ResNet())
log('== Section 4.3 in miniature: CIFAR-10 from scratch, same recipe for both models and every data size ==')
log(f'  ViT    (4x4 patches -> 64 tokens + [class], D 128, 6 layers, 4 heads, MLP 512): {p_vit / 1e6:.2f}M parameters, {f_vit / 1e6:.0f}M multiply-adds per image')
log(f'  ResNet (ResNet-18 layout, widths 24-48-96-192, BatchNorm):                     {p_res / 1e6:.2f}M parameters, {f_res / 1e6:.0f}M multiply-adds per image')
log(f'  recipe: AdamW lr {LR:g}, warm-up {WARM} steps then cosine, weight decay {WD}, batch {BS}, {STEPS} steps, crop + flip;')
log(f'          validation on 5,000 CIFAR-10 test images every {EVAL} steps, best value kept (early stopping)')
log(f'  {"images":>8} {"per class":>10} {"epochs":>7} {"ViT best":>10} {"ResNet best":>12} {"ResNet - ViT":>13}')
R = dict(params=dict(vit=p_vit, resnet=p_res), madds=dict(vit=f_vit, resnet=f_res), steps=STEPS, runs={})
for n in SIZES:
    t0 = time.time()
    bv, cv = train(ViT, n)
    br, cr = train(ResNet, n)
    R['runs'][n] = dict(vit=bv, resnet=br, vit_curve=cv, resnet_curve=cr)
    log(f'  {n:>8,} {n // 10:>10,} {STEPS * BS / n:>7.0f} {bv:>9.1f}% {br:>11.1f}% {br - bv:>+12.1f}')
    print(f'    ({time.time() - t0:.0f} s)', flush=True)
    save('part4_scratch', R)
r = R['runs']
log(f'  going from 1,000 to 50,000 images: ViT {r[1000]["vit"]:.1f}% -> {r[50000]["vit"]:.1f}% ({r[50000]["vit"] - r[1000]["vit"]:+.1f}), '
    f'ResNet {r[1000]["resnet"]:.1f}% -> {r[50000]["resnet"]:.1f}% ({r[50000]["resnet"] - r[1000]["resnet"]:+.1f})')
save('part4_scratch', R)
