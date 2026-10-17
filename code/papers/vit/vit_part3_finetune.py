"""Part 3, fine-tuning (Section 3.2, Section 4.1, Table 4, Appendix B.1.1) on a small scale.
Start from google/vit-base-patch16-224-in21k (ViT-B/16 pre-trained on ImageNet-21k), drop the pre-trained head
(the pre-logits layer and the 21k-class layer), attach a zero-initialised 768 x 10 layer and fine-tune on Imagenette
(10 ImageNet classes) with the paper's recipe pieces: SGD with momentum 0.9, no weight decay, cosine decay
(after a short linear warm-up, our choice), gradient clipping at global norm 1, and an exponential (Polyak) average
of the weights. Our batch is 32, not 512, and the run is 300 steps at 224 pixels, not 20,000 at 384.
1. Step 0 by hand: with a zero head every score is 0, the loss is ln 10, and the body gets no gradient at all.
   The same check with a randomly initialised head.
2. The run: loss, learning rate, gradient norm and clipping at every step; validation accuracy along the way.
Writes results/part3_finetune.json and results/part3_finetune_stdout.txt."""
import copy, glob, math, os, random, time
import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image
from transformers import ViTForImageClassification
from common import Log, save

torch.manual_seed(0)
random.seed(0)
log = Log('part3_finetune')
R = {}
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
IMNET = os.path.expanduser('~/.cache/papers/imagenette/imagenette2-160')
WNID = ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079', 'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257']
STEPS, BS, LR, WARM, CLIP, EMA = 300, 32, 0.003, 30, 1.0, 0.99


def listing(split):
    fs, ys = [], []
    for k, w in enumerate(WNID):
        f = sorted(glob.glob(f'{IMNET}/{split}/{w}/*.JPEG'))
        fs += f; ys += [k] * len(f)
    return fs, ys


train, train_y = listing('train')
val, val_y = listing('val')
sub = [i for k in range(10) for i in [j for j, y in enumerate(val_y) if y == k][:100]]     # 100 per class for the curve


def pixels(f, flip=False):
    im = Image.open(f).convert('RGB').resize((224, 224), Image.BILINEAR)
    if flip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    return (torch.from_numpy(np.asarray(im).copy()).float().permute(2, 0, 1) / 255 - 0.5) / 0.5


def evaluate(m, idx, bs=50):
    m.eval(); right = 0
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            x = torch.stack([pixels(val[j]) for j in b]).to(DEV)
            right += (m(pixel_values=x).logits.argmax(1).cpu() == torch.tensor([val_y[j] for j in b])).sum().item()
    return right / len(idx)


log(f'== 0. data: Imagenette, {len(train)} training and {len(val)} validation pictures, 10 classes; device {DEV} ==')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=10)
log('  loaded google/vit-base-patch16-224-in21k into a 10-class classifier')
log(f'  the pre-logits layer (768 x 768 + tanh) is dropped: model.vit.pooler is {model.vit.pooler}')
torch.nn.init.zeros_(model.classifier.weight); torch.nn.init.zeros_(model.classifier.bias)
log(f'  new head: {model.classifier} with every weight and bias set to 0 ({768 * 10 + 10:,} numbers)')
model.to(DEV)

# ---------------------------------------------------------------- 1. the first step by hand
log('')
log('== 1. step 0 with the zero head ==')
rng = random.Random(1)
b = rng.sample(range(len(train)), BS)
x = torch.stack([pixels(train[j]) for j in b]).to(DEV); y = torch.tensor([train_y[j] for j in b]).to(DEV)
model.train()
out = model(pixel_values=x, output_hidden_states=True)
feats = model.vit.layernorm(out.hidden_states[-1][:, 0])
loss = Fn.cross_entropy(out.logits, y)
loss.backward()
log(f'  logits of the first picture: {out.logits[0].detach().cpu().tolist()}  -> softmax gives 0.1 to each of the 10 classes')
log(f'  loss = -ln(0.1) = ln 10 = {math.log(10):.6f};  measured {loss.item():.6f}')
body = math.sqrt(sum(p.grad.pow(2).sum().item() for n, p in model.named_parameters() if 'classifier' not in n and p.grad is not None))
head = model.classifier.weight.grad.norm().item()
P = torch.softmax(out.logits, 1).detach(); Y1 = Fn.one_hot(y, 10).float()
gW = ((P - Y1).T @ feats.detach()) / BS
log(f'  gradient of the head weight by hand, (softmax - one_hot)^T y / batch: norm {gW.norm().item():.4f}; autograd: {head:.4f}; '
    f'max |difference| {(gW - model.classifier.weight.grad).abs().max().item():.1e}')
log(f'  gradient reaching the body (all 85.8M other numbers): norm {body:.1f}  (W_head^T (softmax - one_hot) = 0 because W_head = 0)')
R['step0'] = dict(loss=loss.item(), head_grad=head, body_grad=body)
model.zero_grad()
rnd = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=10).to(DEV).train()
torch.manual_seed(0); torch.nn.init.normal_(rnd.classifier.weight, std=0.02); torch.nn.init.zeros_(rnd.classifier.bias)
o2 = rnd(pixel_values=x); l2 = Fn.cross_entropy(o2.logits, y); l2.backward()
body2 = math.sqrt(sum(p.grad.pow(2).sum().item() for n, p in rnd.named_parameters() if 'classifier' not in n and p.grad is not None))
log(f'  for comparison, a random head (normal, std 0.02): loss {l2.item():.4f}, gradient reaching the body: norm {body2:.3f}, '
    f'logits of picture 1 range {o2.logits[0].min().item():.2f} to {o2.logits[0].max().item():.2f}')
R['step0_random'] = dict(loss=l2.item(), body_grad=body2)
del rnd, o2, l2
if DEV == 'mps':
    torch.mps.empty_cache()

# ---------------------------------------------------------------- 2. the run
log('')
log(f'== 2. fine-tuning: {STEPS} steps, batch {BS}, SGD momentum 0.9, no weight decay, base LR {LR}, linear warm-up {WARM} steps, '
    f'cosine decay to 0, clip at global norm {CLIP}, weight average factor {EMA}, random horizontal flips ==')
opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.0)
lr_at = lambda t: LR * t / WARM if t < WARM else LR * 0.5 * (1 + math.cos(math.pi * (t - WARM) / (STEPS - WARM)))
ema = copy.deepcopy(model).eval()
for p_ in ema.parameters():
    p_.requires_grad_(False)
acc0 = evaluate(model, sub)
log(f'  step   0: validation accuracy (1,000 pictures, 100 per class) {100 * acc0:.1f}%  (zero head: every class ties, the first wins)')
hist, curve = [], [[0, acc0]]
order = []
while len(order) < STEPS * BS:                      # a new shuffle for each pass over the training set
    o = list(range(len(train))); random.shuffle(o); order += o
t0 = time.time()
for step in range(1, STEPS + 1):
    model.train()
    b = order[(step - 1) * BS: step * BS]
    x = torch.stack([pixels(train[j], flip=random.random() < 0.5) for j in b]).to(DEV)
    y = torch.tensor([train_y[j] for j in b]).to(DEV)
    for g in opt.param_groups:
        g['lr'] = lr_at(step - 1)
    loss = Fn.cross_entropy(model(pixel_values=x).logits, y)
    opt.zero_grad(); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP).item()
    opt.step()
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.mul_(EMA).add_(pm.detach(), alpha=1 - EMA)
    hist.append(dict(step=step, loss=loss.item(), lr=lr_at(step - 1), gnorm=gn))
    if step in (5, 10, 25, 50, 100, 150, 200, 250, 300):
        a = evaluate(model, sub); curve.append([step, a])
        log(f'  step {step:3d}: lr {lr_at(step - 1):.5f}  loss {loss.item():.4f}  gradient norm before clipping {gn:7.3f}  '
            f'validation {100 * a:5.1f}%   [{time.time() - t0:.0f} s]')
R['hist'], R['curve'] = hist, curve
clipped = sum(h['gnorm'] > CLIP for h in hist)
log(f'  clipping was active in {clipped} of {STEPS} steps; median gradient norm: first 50 steps {np.median([h["gnorm"] for h in hist[:50]]):.3f}, '
    f'last 50 steps {np.median([h["gnorm"] for h in hist[-50:]]):.3f}')
log(f'  mean training loss: steps 1 to 10 {np.mean([h["loss"] for h in hist[:10]]):.3f}, steps 291 to 300 {np.mean([h["loss"] for h in hist[-10:]]):.3f}; '
    f'pictures seen {STEPS * BS:,} = {STEPS * BS / len(train):.2f} passes over the training set')
full = evaluate(model, list(range(len(val))))
full_ema = evaluate(ema, list(range(len(val))))
log(f'  final, all {len(val)} validation pictures: last weights {100 * full:.2f}%   averaged weights (factor {EMA}) {100 * full_ema:.2f}%')
R.update(clipped=clipped, final=full, final_ema=full_ema, steps=STEPS, bs=BS, lr=LR, warm=WARM, ema=EMA, n_train=len(train), n_val=len(val))
save('part3_finetune', R)
log('saved results/part3_finetune.json')
