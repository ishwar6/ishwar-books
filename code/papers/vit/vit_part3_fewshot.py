"""Part 3, the few-shot metric of Section 4.1: "a regularized least-squares regression problem that maps the
(frozen) representation of a subset of training images to {-1, 1}^K target vectors", solved in closed form.
1. A tiny example by hand: four pictures with two features each, two classes, lambda = 1.
2. The real thing on frozen google/vit-base-patch16-224-in21k (ViT-B/16 pre-trained on ImageNet-21k, never
   trained on Imagenette): 1-, 5- and 10-shot on the 10 Imagenette classes, 5 random draws each, scored on the
   3,925 validation pictures, for a few values of lambda and two choices of the representation.
Writes results/part3_fewshot.json and results/part3_fewshot_stdout.txt."""
import glob, os, random
import numpy as np
import torch
from PIL import Image
from transformers import ViTModel
from common import Log, save

log = Log('part3_fewshot')
R = {}
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
IMNET = os.path.expanduser('~/.cache/papers/imagenette/imagenette2-160')
WNID = ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079', 'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257']
NAMES = ['tench', 'springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']
np.set_printoptions(precision=3, suppress=True)


def ridge(X, Y, lam):
    """W = (X^T X + lam I)^-1 X^T Y: the exact minimiser of ||X W - Y||^2 + lam ||W||^2."""
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)


# ---------------------------------------------------------------- 1. tiny example
log('== 1. the closed form by hand: 4 pictures, 2 features, K = 2 classes, lambda = 1 ==')
X = np.array([[2.0, 1.0], [1.0, 0.0], [0.0, 2.0], [1.0, 3.0]])
Y = np.array([[1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, 1.0]])
log(f'  X (rows = pictures A1, A2, B1, B2) = {X.tolist()}')
log(f'  Y (targets in {{-1, 1}}: class A = [1, -1], class B = [-1, 1]) = {Y.tolist()}')
XtX = X.T @ X
A = XtX + np.eye(2)
XtY = X.T @ Y
W = np.linalg.solve(A, XtY)
log(f'  X^T X = {XtX.tolist()}   + lambda I = {A.tolist()}   det = {np.linalg.det(A):.0f}')
log(f'  (X^T X + I)^-1 = {np.round(np.linalg.inv(A), 4).tolist()}')
log(f'  X^T Y = {XtY.tolist()}')
log(f'  W = (X^T X + I)^-1 X^T Y = {np.round(W, 4).tolist()}')
new = np.array([1.5, 0.5])
log(f'  new picture x = {new.tolist()}: scores x W = {np.round(new @ W, 4).tolist()} -> class {"AB"[int(np.argmax(new @ W))]}')
log(f'  training scores X W = {np.round(X @ W, 3).tolist()} (all on the right side of 0)')
R['tiny'] = dict(X=X.tolist(), Y=Y.tolist(), XtX=XtX.tolist(), inv=np.linalg.inv(A).tolist(), XtY=XtY.tolist(), W=W.tolist(),
                 new=new.tolist(), score=(new @ W).tolist())
log('')

# ---------------------------------------------------------------- 2. frozen features
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').eval().to(DEV)


def pixels(f):
    im = Image.open(f).convert('RGB').resize((224, 224), Image.BILINEAR)
    return (torch.from_numpy(np.asarray(im).copy()).float().permute(2, 0, 1) / 255 - 0.5) / 0.5


def features(files, bs=48):
    pre, cls = [], []
    with torch.no_grad():
        for i in range(0, len(files), bs):
            o = model(pixel_values=torch.stack([pixels(f) for f in files[i:i + bs]]).to(DEV))
            pre.append(o.pooler_output.float().cpu()); cls.append(o.last_hidden_state[:, 0].float().cpu())
    return torch.cat(pre).numpy(), torch.cat(cls).numpy()


rng = random.Random(0)
pool, pool_y = [], []
for k, w in enumerate(WNID):
    fs = sorted(glob.glob(f'{IMNET}/train/{w}/*.JPEG'))
    pick = rng.sample(fs, 50)
    pool += pick; pool_y += [k] * 50
val, val_y = [], []
for k, w in enumerate(WNID):
    fs = sorted(glob.glob(f'{IMNET}/val/{w}/*.JPEG'))
    val += fs; val_y += [k] * len(fs)
pool_y, val_y = np.array(pool_y), np.array(val_y)
log(f'== 2. frozen ViT-B/16 (ImageNet-21k weights), device {DEV}: features of {len(pool)} training pictures (a pool of 50 per class) '
    f'and {len(val)} validation pictures ==')
Ptr, Ctr = features(pool)
Pva, Cva = features(val)
log(f'  representation "pre-logits": the tanh layer on the [class] token, {Ptr.shape[1]} numbers;  "class token": the final LayerNorm output (Eq. 4)')


def run(Ftr, Fva, shots, lam, seeds=5):
    accs = []
    for s in range(seeds):
        r = np.random.default_rng(s)
        idx = np.concatenate([r.choice(np.where(pool_y == k)[0], shots, replace=False) for k in range(10)])
        Xs = np.hstack([Ftr[idx], np.ones((len(idx), 1))])            # a bias column
        Ys = -np.ones((len(idx), 10)); Ys[np.arange(len(idx)), pool_y[idx]] = 1
        Wr = ridge(Xs, Ys, lam)
        pred = (np.hstack([Fva, np.ones((len(Fva), 1))]) @ Wr).argmax(1)
        accs.append((pred == val_y).mean())
    return float(np.mean(accs)), float(np.std(accs)), accs


grid = {}
log('  accuracy on the 3,925 validation pictures, mean +- std over 5 random draws of the training pictures')
for rep, (Ftr, Fva) in [('pre-logits', (Ptr, Pva)), ('class token', (Ctr, Cva))]:
    for lam in [0.01, 1.0, 100.0]:
        for shots in [1, 5, 10]:
            m, sd, accs = run(Ftr, Fva, shots, lam)
            grid[f'{rep}|{lam}|{shots}'] = dict(mean=m, std=sd, accs=accs)
        log(f'  {rep:<11} lambda {lam:>6g}:  1-shot {100 * grid[f"{rep}|{lam}|1"]["mean"]:5.1f} +- {100 * grid[f"{rep}|{lam}|1"]["std"]:4.1f}%   '
            f'5-shot {100 * grid[f"{rep}|{lam}|5"]["mean"]:5.1f} +- {100 * grid[f"{rep}|{lam}|5"]["std"]:4.1f}%   '
            f'10-shot {100 * grid[f"{rep}|{lam}|10"]["mean"]:5.1f} +- {100 * grid[f"{rep}|{lam}|10"]["std"]:4.1f}%')
R['grid'] = grid
log('  chance is 10%; 1-shot means 10 training pictures in total, one per class')
m, sd, accs = run(Ptr, Pva, 5, 1.0)
r = np.random.default_rng(0)
idx = np.concatenate([r.choice(np.where(pool_y == k)[0], 5, replace=False) for k in range(10)])
Xs = np.hstack([Ptr[idx], np.ones((len(idx), 1))]); Ys = -np.ones((50, 10)); Ys[np.arange(50), pool_y[idx]] = 1
Wr = ridge(Xs, Ys, 1.0)
log(f'  5-shot, draw 0, pre-logits, lambda 1: X is {Xs.shape} (50 pictures x 768 features + bias), Y is {Ys.shape}, W is {Wr.shape}; '
    f'one solve of a {Xs.shape[1]}x{Xs.shape[1]} system')
S = np.hstack([Pva, np.ones((len(Pva), 1))]) @ Wr
conf = np.zeros((10, 10), int)
for t, p in zip(val_y, S.argmax(1)):
    conf[t, p] += 1
per = conf.diagonal() / conf.sum(1)
log('  per-class accuracy (5-shot, draw 0): ' + ', '.join(f'{n} {100 * a:.0f}%' for n, a in zip(NAMES, per)))
ex = S[0]
log(f'  scores of the first validation picture (a {NAMES[val_y[0]]}): ' + ' '.join(f'{v:+.2f}' for v in ex) + f'  -> class {NAMES[ex.argmax()]}')
R['per_class'] = dict(names=NAMES, acc=per.tolist())
R['example_scores'] = ex.tolist()
save('part3_fewshot', R)
log('saved results/part3_fewshot.json')
