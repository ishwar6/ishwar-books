"""Part 3, higher resolution: the 2D interpolation of the position embeddings (Section 3.2), by hand and on the
released google/vit-base-patch16-224 (pre-trained on ImageNet-21k, fine-tuned on ImageNet at 224).
1. A 2x2 grid of numbers resized to 3x3 with the official code's method (scipy.ndimage.zoom, order 1), by hand.
2. The real 14x14 grid of 768-number position rows resized to 24x24 by our own bilinear code, checked against
   scipy.ndimage.zoom, and compared with the library's bicubic resize (interpolate_pos_encoding=True).
3. Cosine-similarity maps of one position row against all others, before (14x14) and after (24x24).
4. Accuracy on the Imagenette validation set (3,925 pictures of 10 ImageNet classes) at 224 and at 384 with
   2D bilinear, 2D bicubic, a wrong 1D resize of the position list, and no position rows at all. No training:
   the ImageNet head is read at the 10 Imagenette classes.
Writes results/part3_posemb.json and results/part3_posemb_stdout.txt."""
import glob, os, time
import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as Fn
from PIL import Image
from transformers import ViTForImageClassification
from common import Log, save

torch.manual_seed(0)
log = Log('part3_posemb')
R = {}
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
IMNET = os.path.expanduser('~/.cache/papers/imagenette/imagenette2-160')
WNID = ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079', 'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257']
IDX = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]      # the ImageNet class of each Imagenette folder

# ---------------------------------------------------------------- 1. the tiny example
log('== 1. resize a 2x2 grid to 3x3 the way the official code does (scipy.ndimage.zoom, order=1 = linear) ==')
g = np.array([[0.0, 1.0], [2.0, 3.0]])
z = scipy.ndimage.zoom(g, 1.5, order=1)
log(f'  grid 2x2: {g.tolist()}')
for r in z:
    log('    ' + '  '.join(f'{v:4.2f}' for v in r))
log('  corners stay, the new middle row and column are averages of their neighbours: (0+1)/2 = 0.5, (0+2)/2 = 1, (0+1+2+3)/4 = 1.5')
R['tiny'] = dict(g=g.tolist(), z=z.tolist())
log('')

# ---------------------------------------------------------------- 2. the real position table
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').eval()
pos = model.vit.embeddings.position_embeddings.detach()[0]          # (197, 768)
cls_pos, grid = pos[:1], pos[1:].reshape(14, 14, 768)
log('== 2. the position table of google/vit-base-patch16-224 ==')
log(f'  E_pos {tuple(pos.shape)} = 1 row for [class] + 196 rows = a 14x14 grid of 768 numbers')
log('  at 384 pixels: 384/16 = 24 patches per side, N = 576, so 577 rows are needed: 380 rows do not exist')


def bilinear_by_hand(G, n):
    """Corner-aligned linear interpolation in 2D, the same rule as scipy.ndimage.zoom(order=1):
    new row i sits at old position i*(old-1)/(new-1); mix the two nearest old rows, then the same for columns."""
    old = G.shape[0]
    out = np.zeros((n, n, G.shape[2]))
    for i in range(n):
        y = i * (old - 1) / (n - 1); y0 = int(np.floor(y)); y1 = min(y0 + 1, old - 1); dy = y - y0
        for j in range(n):
            x = j * (old - 1) / (n - 1); x0 = int(np.floor(x)); x1 = min(x0 + 1, old - 1); dx = x - x0
            out[i, j] = ((1 - dy) * (1 - dx) * G[y0, x0] + (1 - dy) * dx * G[y0, x1]
                         + dy * (1 - dx) * G[y1, x0] + dy * dx * G[y1, x1])
    return out


G = grid.numpy().astype(np.float64)
ours = bilinear_by_hand(G, 24)
ref = scipy.ndimage.zoom(G, (24 / 14, 24 / 14, 1), order=1)
log(f'  our bilinear 14x14 -> 24x24: {ours.shape}; max |difference| vs scipy.ndimage.zoom(order=1): {np.abs(ours - ref).max():.1e}')
y = 5 * 13 / 23
log(f'  new row 5 sits at old position 5 x 13/23 = {y:.4f}: {1 - (y - int(y)):.4f} of old row {int(y)} + {y - int(y):.4f} of old row {int(y) + 1}')
bic = Fn.interpolate(grid.permute(2, 0, 1)[None], size=(24, 24), mode='bicubic', align_corners=False)[0].permute(1, 2, 0)
with torch.no_grad():
    lib = model.vit.embeddings.interpolate_pos_encoding(torch.zeros(1, 577, 768), 384, 384)[0, 1:].reshape(24, 24, 768)
log(f'  library interpolate_pos_encoding: bicubic, align_corners=False; our bicubic call matches it: max |difference| {(bic - lib).abs().max():.1e}')
log(f'  bilinear (official) vs bicubic (library) position rows: mean |difference| {np.abs(ours - bic.numpy()).mean():.4f}, '
    f'mean |value| {np.abs(ours).mean():.4f}')
corner = ours[0, 0] - G[0, 0]
log(f'  corner row (0,0) kept by bilinear: max |difference| {np.abs(corner).max():.1e}')
R['interp'] = dict(maxdiff_scipy=float(np.abs(ours - ref).max()), maxdiff_lib=float((bic - lib).abs().max()),
                   bil_vs_bic=float(np.abs(ours - bic.numpy()).mean()), mean_abs=float(np.abs(ours).mean()))
log('')

# ---------------------------------------------------------------- 3. similarity maps
log('== 3. cosine similarity of one position row with all the others ==')


def simmap(Gr, r, c):
    v = Gr[r, c] / np.linalg.norm(Gr[r, c])
    n = Gr / np.linalg.norm(Gr, axis=2, keepdims=True)
    return (n @ v)


s14 = simmap(G, 3, 10)
s24 = simmap(ours, round(3 * 23 / 13), round(10 * 23 / 13))
log(f'  14x14: row (3, 10) vs all 196: max {s14.max():.3f} (itself), mean {s14.mean():.3f}; nearest neighbours: '
    f'(3, 9) {s14[3, 9]:.3f}, (3, 11) {s14[3, 11]:.3f}, (2, 10) {s14[2, 10]:.3f}, (4, 10) {s14[4, 10]:.3f}; far corner (13, 0) {s14[13, 0]:.3f}')
log(f'  24x24: row ({round(3 * 23 / 13)}, {round(10 * 23 / 13)}) (the same place) vs all 576: neighbours '
    f'{s24[5, 17]:.3f}, {s24[5, 19]:.3f}; far corner (23, 0) {s24[23, 0]:.3f}')
R['sim14'] = s14.tolist(); R['sim24'] = s24.tolist(); R['sim_at'] = [[3, 10], [round(3 * 23 / 13), round(10 * 23 / 13)]]
log('')

# ---------------------------------------------------------------- 4. accuracy on Imagenette
files, labels = [], []
for k, w in enumerate(WNID):
    fs = sorted(glob.glob(f'{IMNET}/val/{w}/*.JPEG'))
    files += fs; labels += [k] * len(fs)
labels = torch.tensor(labels)


def pixels(f, S):
    im = Image.open(f).convert('RGB').resize((S, S), Image.BILINEAR)
    return (torch.from_numpy(np.asarray(im).copy()).float().permute(2, 0, 1) / 255 - 0.5) / 0.5


def evaluate(m, S, posrows=None, bs=48):
    m = m.to(DEV)
    emb = m.vit.embeddings
    if posrows is not None:
        emb.position_embeddings = torch.nn.Parameter(posrows[None].float().to(DEV))
        emb.image_size = (S, S); emb.patch_embeddings.image_size = (S, S); emb.patch_embeddings.num_patches = (S // 16) ** 2
    right10 = right1000 = 0
    with torch.no_grad():
        for i in range(0, len(files), bs):
            x = torch.stack([pixels(f, S) for f in files[i:i + bs]]).to(DEV)
            lo = m(pixel_values=x).logits.float().cpu()
            y = labels[i:i + bs]
            right10 += (lo[:, IDX].argmax(1) == y).sum().item()
            right1000 += (lo.argmax(1) == torch.tensor(IDX)[y]).sum().item()
    return right10 / len(files), right1000 / len(files)


def fresh():
    return ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').eval()


log(f'== 4. accuracy on the Imagenette validation set: {len(files)} pictures (160 px on the short side), device {DEV} ==')
log('  the ImageNet head of the checkpoint is read at the 10 Imagenette classes; no training; "of 1000" = the top class of all 1000 must be right')
one_d = Fn.interpolate(grid.reshape(1, 196, 768).permute(0, 2, 1), size=576, mode='linear', align_corners=True)[0].T   # wrong: a 1D list
conds = [('224, the training resolution (196 patches)', 224, None),
         ('384, 2D bilinear (the official code)', 384, torch.cat([cls_pos, torch.from_numpy(ours).float().reshape(576, 768)])),
         ('384, 2D bicubic (the library)', 384, torch.cat([cls_pos, bic.reshape(576, 768)])),
         ('384, 1D: the 196 rows stretched as one list', 384, torch.cat([cls_pos, one_d])),
         ('384, no position rows at all (zeros)', 384, torch.zeros(577, 768))]
acc = []
for name, S, rows in conds:
    t = time.time()
    a10, a1000 = evaluate(fresh(), S, rows)
    acc.append(dict(name=name, res=S, acc10=a10, acc1000=a1000))
    log(f'  {name:<46} accuracy {100 * a10:5.2f}% (of 10)   {100 * a1000:5.2f}% (of 1000)   [{time.time() - t:.0f} s]')
R['acc'] = acc
R['n_val'] = len(files)
save('part3_posemb', R)
log('saved results/part3_posemb.json')
