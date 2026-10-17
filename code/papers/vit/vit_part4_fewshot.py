"""Part 4: a small, honest analogue of Table 2 and Figures 3 and 4: few-shot linear evaluation of frozen models.

The paper's "few-shot linear accuracy" freezes a pre-trained model, takes the representation of k labelled images per
class, and solves a regularised least-squares problem that maps them to {-1, 1}^K target vectors, in closed form.
This script:
  1. works the closed form by hand on 4 images with 2 features and 2 classes, and checks it against torch;
  2. extracts frozen representations on two datasets of Table 2, CIFAR-100 (100 classes) and Oxford-IIIT Pets
     (37 classes), for the five ViT sizes of Figure 3 (B/32, B/16, L/32, L/16, H/14, all released after
     ImageNet-21k pre-training) and two ResNet-50s (google/bit-50, a BiT R50x1 with an ImageNet-1k head, and microsoft/resnet-50);
  3. solves the few-shot problem for k = 1, 2, 5, 10 images per class, 5 random draws each, and prints the
     mean and spread of the test accuracy.
What it cannot show: no JFT-300M, no fine-tuning, small test sets (1,000 CIFAR-100 and 1,110 Pets images), and the
two ResNets were pre-trained on other data (bit-50 ends in an ImageNet-1k head; its card does not say whether it is BiT-S or BiT-M; resnet-50 on ImageNet
only), so they are reference points, not the paper's matched baselines. Representations: the ViT "pre-logits"
(pooler: dense + tanh on the [class] token, the layer before the pre-training head) and the ResNets' pooled features.
Writes results/part4_fewshot.json and results/part4_fewshot_stdout.txt; features are cached in
~/.cache/papers/vit_part4_feats/."""
import glob, io, os, time
import numpy as np
import torch
import pyarrow.parquet as pq
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from common import Log, save

log = Log('part4_fewshot')
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
HUB = os.path.expanduser('~/.cache/huggingface/hub')
CACHE = os.path.expanduser('~/.cache/papers/vit_part4_feats')
os.makedirs(CACHE, exist_ok=True)
SHOTS = [1, 2, 5, 10]
DRAWS = 5
POOL, HOLD = 20, 5           # per class: the shots are drawn from 20 images, 5 more choose lambda
LAMBDAS = [10.0 ** e for e in range(-3, 6)]
MODELS = [('ViT-B/32', 'google/vit-base-patch32-224-in21k'), ('ViT-B/16', 'google/vit-base-patch16-224-in21k'),
          ('ViT-L/32', 'google/vit-large-patch32-224-in21k'), ('ViT-L/16', 'google/vit-large-patch16-224-in21k'),
          ('ViT-H/14', 'google/vit-huge-patch14-224-in21k'),
          ('BiT R50x1', 'google/bit-50'), ('ResNet-50 (ImageNet-1k)', 'microsoft/resnet-50')]
R = {}


# ---------------------------------------------------------------- 1. the closed form by hand
def ridge(X, Y, lam):
    """W = (X^T X + lam I)^-1 X^T Y: the regularised least-squares map from features to {-1, 1} targets."""
    d = X.shape[1]
    return torch.linalg.solve(X.T @ X + lam * torch.eye(d, dtype=X.dtype), X.T @ Y)


log('== 1. the few-shot closed form on a tiny example: 4 images, 2 features, 2 classes, lambda = 1 ==')
X = torch.tensor([[1., 0.], [2., 1.], [0., 2.], [1., 3.]])
y = [0, 0, 1, 1]
Y = -torch.ones(4, 2)
Y[torch.arange(4), torch.tensor(y)] = 1
log('  X (one row per image) =', X.tolist(), '  classes', y)
log('  Y (targets in {-1, 1}) =', Y.tolist())
XtX = X.T @ X
log('  X^T X =', XtX.tolist(), '  + lambda I =', (XtX + torch.eye(2)).tolist())
log('  X^T Y =', (X.T @ Y).tolist())
W = ridge(X, Y, 1.0)
log('  W = (X^T X + I)^-1 X^T Y =', [[round(v, 4) for v in r] for r in W.tolist()])
xn = torch.tensor([1., 2.])
s = xn @ W
log(f'  new image x = [1, 2]: scores x W = {[round(v, 4) for v in s.tolist()]} -> class {int(s.argmax())}')
R['tiny'] = dict(W=W.tolist(), scores=s.tolist())


# ---------------------------------------------------------------- 2. data
def parquet(pattern):
    f = glob.glob(os.path.join(HUB, pattern))[0]
    return pq.read_table(f).to_pandas()


def load(name):
    if name == 'CIFAR-100':
        tr, te = parquet('datasets--uoft-cs--cifar100/snapshots/*/cifar100/train-*.parquet'), parquet('datasets--uoft-cs--cifar100/snapshots/*/cifar100/test-*.parquet')
        col, lab = 'img', 'fine_label'
    else:
        tr, te = parquet('datasets--timm--oxford-iiit-pet/snapshots/*/data/train-*.parquet'), parquet('datasets--timm--oxford-iiit-pet/snapshots/*/data/test-*.parquet')
        col, lab = 'image', 'label'
    rng = np.random.default_rng(0)
    ytr = tr[lab].to_numpy()
    K = len(set(ytr))
    pool, hold = [], []
    for c in range(K):                       # per class: a shuffled list; the first 20 are the shot pool, the next 5 choose lambda
        where = np.flatnonzero(ytr == c)
        rng.shuffle(where)
        pool.append(where[:POOL])
        hold.append(where[POOL:POOL + HOLD])
    yte = te[lab].to_numpy()
    per = 10 if name == 'CIFAR-100' else 30  # test images per class
    te_idx = np.concatenate([np.flatnonzero(yte == c)[:per] for c in range(K)])
    imgs = lambda df, ii: [df[col].iloc[i]['bytes'] for i in ii]
    pool_i, hold_i = np.concatenate(pool), np.concatenate(hold)
    return dict(K=K, train=imgs(tr, pool_i), ytrain=ytr[pool_i], hold=imgs(tr, hold_i), yhold=ytr[hold_i],
                test=imgs(te, te_idx), ytest=yte[te_idx])


DATA = {n: load(n) for n in ['CIFAR-100', 'Oxford-IIIT Pets']}
for n, d in DATA.items():
    log(f'  {n}: {d["K"]} classes; shot pool {len(d["train"])} images ({POOL} per class), {len(d["hold"])} held out to choose lambda, '
        f'{len(d["test"])} test images')


# ---------------------------------------------------------------- 3. frozen features
@torch.no_grad()
def features(mid, blobs, bs=32):
    proc = AutoImageProcessor.from_pretrained(mid)
    kw = dict(size={'shortest_edge': 224}, crop_size={'height': 224, 'width': 224}) if 'bit' in mid else {}   # 224 px for every model
    model = AutoModel.from_pretrained(mid).to(DEV).eval().half()   # float16 on the GPU: a shared machine, so we save time
    out = []
    for i in range(0, len(blobs), bs):
        ims = [Image.open(io.BytesIO(b)).convert('RGB') for b in blobs[i:i + bs]]
        px = proc(images=ims, return_tensors='pt', **kw)['pixel_values'].to(DEV).half()
        o = model(pixel_values=px)
        f = o.pooler_output
        out.append(f.reshape(f.shape[0], -1).float().cpu())
    del model
    if DEV == 'mps':
        torch.mps.empty_cache()
    return torch.cat(out).numpy()


def feats(name, mid, dname):
    path = os.path.join(CACHE, f'{mid.replace("/", "_")}_{dname.replace(" ", "_")}.npz')
    if not os.path.exists(path):
        d = DATA[dname]
        t0 = time.time()
        F = {k: features(mid, d[k]) for k in ['train', 'hold', 'test']}
        np.savez(path, **F)
        print(f'    ({name} on {dname}: features in {time.time() - t0:.0f} s)')
    z = np.load(path)
    return {k: torch.tensor(z[k], dtype=torch.float64) for k in ['train', 'hold', 'test']}


# ---------------------------------------------------------------- 4. few-shot linear evaluation
def onehot_pm1(y, K):
    Y = -torch.ones(len(y), K, dtype=torch.float64)
    Y[torch.arange(len(y)), torch.tensor(y)] = 1
    return Y


def fewshot(F, d, k, lam, rng):
    K = d['K']
    pick_i = np.concatenate([np.arange(c * POOL, (c + 1) * POOL)[rng.permutation(POOL)[:k]] for c in range(K)])
    Xtr, ytr = F['train'][pick_i], d['ytrain'][pick_i]
    mu = Xtr.mean(0)                           # centre on the shots (this plays the role of a bias term)
    W = ridge(Xtr - mu, onehot_pm1(ytr, K), lam)
    acc = lambda X, y: float((((X - mu) @ W).argmax(1).numpy() == y).mean())
    return acc, W


log('')
log(f'== 2. frozen features, then {DRAWS} random draws of k images per class; test accuracy in % (mean and std over draws) ==')
log(f'   lambda is chosen once per model and dataset from {[f"{l:g}" for l in LAMBDAS]}, at k = 10, on held-out training images (never test images)')
for dname, d in DATA.items():
    log('')
    log(f'-- {dname} ({d["K"]} classes, chance = {100 / d["K"]:.1f}%) --')
    log(f'  {"model":<24}{"dim":>6}{"lambda":>8}' + ''.join(f'{f"{k}-shot":>14}' for k in SHOTS))
    R.setdefault(dname, {})
    for name, mid in MODELS:
        F = feats(name, mid, dname)
        F = {k: v / v.norm(dim=1, keepdim=True) for k, v in F.items()}   # unit length, so one lambda grid fits every model
        best = None
        for lam in LAMBDAS:                     # choose lambda on the held-out images, at k = 10
            accs = []
            for s in range(3):
                acc, _ = fewshot(F, d, 10, lam, np.random.default_rng(100 + s))
                accs.append(acc(F['hold'], d['yhold']))
            if best is None or np.mean(accs) > best[0]:
                best = (np.mean(accs), lam)
        lam = best[1]
        res = {}
        for k in SHOTS:
            accs = []
            for s in range(DRAWS):
                acc, _ = fewshot(F, d, k, lam, np.random.default_rng(s))
                accs.append(100 * acc(F['test'], d['ytest']))
            res[k] = (float(np.mean(accs)), float(np.std(accs)))
        R[dname][name] = dict(dim=int(F['train'].shape[1]), lam=lam, acc={str(k): [round(m, 2), round(s, 2)] for k, (m, s) in res.items()})
        log(f'  {name:<24}{F["train"].shape[1]:>6}{lam:>8g}' + ''.join(f'{m:>8.1f} ±{s:>4.1f}' for k, (m, s) in res.items()))

save('part4_fewshot', R)


# ---------------------------------------------------------------- 5. two checks on the results
log('')
log('== 3. which Oxford-IIIT Pets breeds are also ImageNet-1k classes (name match against the 1,000 class names) ==')
from transformers import AutoConfig
in1k = [n.lower() for n in AutoConfig.from_pretrained('microsoft/resnet-50').id2label.values()]
syn = [[s.strip() for s in n.split(',')] for n in in1k]
PETS = ['abyssinian', 'american bulldog', 'american pit bull terrier', 'basset hound', 'beagle', 'bengal', 'birman', 'bombay',
        'boxer', 'british shorthair', 'chihuahua', 'egyptian mau', 'english cocker spaniel', 'english setter', 'german shorthaired',
        'great pyrenees', 'havanese', 'japanese chin', 'keeshond', 'leonberger', 'maine coon', 'miniature pinscher', 'newfoundland',
        'persian', 'pomeranian', 'pug', 'ragdoll', 'russian blue', 'saint bernard', 'samoyed', 'scottish terrier', 'shiba inu',
        'siamese', 'sphynx', 'staffordshire bull terrier', 'wheaten terrier', 'yorkshire terrier']
hits = [p for p in PETS if any(p in (s, s.removesuffix(' cat'), s.removesuffix(' dog')) for ss in syn for s in ss)]
log(f'  {len(hits)} of 37 breeds have an ImageNet-1k class of the same name: {", ".join(hits)}')
R['pets_in_imagenet1k'] = hits

log('')
log('== 4. does float16 change the features? ViT-H/14 on 4 CIFAR-100 test pictures, float16 on the GPU against float32 on the CPU ==')
mid = 'google/vit-huge-patch14-224-in21k'
proc = AutoImageProcessor.from_pretrained(mid)
ims = [Image.open(io.BytesIO(b)).convert('RGB') for b in DATA['CIFAR-100']['test'][:4]]
px = proc(images=ims, return_tensors='pt')['pixel_values']
m = AutoModel.from_pretrained(mid).eval()
with torch.no_grad():
    f32 = m(pixel_values=px).pooler_output
    f16 = m.to(DEV).half()(pixel_values=px.to(DEV).half()).pooler_output.float().cpu()
cos = torch.nn.functional.cosine_similarity(f32, f16)
log(f'  cosine similarity float16 vs float32, per picture: {", ".join(f"{c:.6f}" for c in cos.tolist())}')
R['fp16_cos'] = cos.tolist()
save('part4_fewshot', R)
