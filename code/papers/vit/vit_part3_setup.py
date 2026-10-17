"""Part 3, the setup by hand: the arithmetic behind Section 4.1, Tables 1, 3 and 4 and Appendix B.1 of the ViT paper.
1. Table 1 and the "ViT-L/16" notation: sequence length, attention pairs and forward compute for each patch size and
   resolution, and the parameter count of each released checkpoint read from its weight file (ViT-H/14 on the meta
   device, no weights needed): the patch size changes the compute, almost not the parameters.
2. The hybrid's two options: the real feature-map shapes of microsoft/resnet-50 (stage 4 gives 7x7 = 49 tokens,
   stage 3 gives 14x14 = 196, the "4x longer sequence").
3. The BiT baseline: Batch Norm against Group Norm on a tiny batch, and Weight Standardization, by hand.
4. The schedules of Tables 3 and 4 turned into steps, epochs and learning-rate curves.
5. The optimizer steps by hand: Adam (pre-training), SGD with momentum (fine-tuning), clipping at global norm 1,
   and Polyak averaging, each on a few numbers you can check with a pencil.
Runs on the CPU in under a minute. Writes results/part3_setup.json and results/part3_setup_stdout.txt."""
import glob, math, os
import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image
from safetensors import safe_open
from transformers import ViTConfig, ViTModel, ResNetModel, AutoImageProcessor
from common import Log, save

torch.manual_seed(0)
log = Log('part3_setup')
R = {}
HUB = os.path.expanduser('~/.cache/huggingface/hub')
IMNET = os.path.expanduser('~/.cache/papers/imagenette/imagenette2-160')

# ---------------------------------------------------------------- 1. Table 1, patch sizes, sequence lengths, compute
log('== 1. Table 1 and the patch-size notation: ViT-<size>/<patch> ==')
SIZES = {'B': (768, 12, 3072, 12), 'L': (1024, 24, 4096, 16), 'H': (1280, 32, 5120, 16)}
PAPER_PARAMS = {'B': '86M', 'L': '307M', 'H': '632M'}
VARIANTS = [('B', 32), ('B', 16), ('L', 32), ('L', 16), ('H', 14)]


def gflops(D, L, mlp, N):
    """Forward multiply-adds x 2 for one picture: per layer q, k, v and the output matrix (4 N D^2), the MLP
    (2 N D mlp), the scores q k^T and the weighted sum A v (2 N^2 D). The patch embedding is small and left out."""
    per_layer = 2 * (4 * N * D * D + 2 * N * D * mlp + 2 * N * N * D)
    return L * per_layer / 1e9


rows = []
log('  N = (resolution / patch)^2 patches, plus 1 [class] token; attention compares every pair: (N+1)^2 pairs per head')
for size, P in VARIANTS:
    D, L, mlp, heads = SIZES[size]
    for res in (224, 384):
        n = (res // P) ** 2
        rows.append(dict(name=f'ViT-{size}/{P}', res=res, grid=res // P, N=n, seq=n + 1, pairs=(n + 1) ** 2,
                         gflops=round(gflops(D, L, mlp, n + 1), 1)))
        log(f'  ViT-{size}/{P:<2} at {res}: grid {res // P:>2}x{res // P:<2} N = {n:>4}  sequence {n + 1:>4}  '
            f'pairs per head {(n + 1) ** 2:>9,}  forward compute about {gflops(D, L, mlp, n + 1):6.1f} GFLOPs')
R['variants'] = rows
b16 = next(r for r in rows if r['name'] == 'ViT-B/16' and r['res'] == 224)
b32 = next(r for r in rows if r['name'] == 'ViT-B/32' and r['res'] == 224)
log(f'  ViT-B/16 vs ViT-B/32 at 224: {b16["N"]} vs {b32["N"]} patches = x{b16["N"] / b32["N"]:.0f} '
    f'(patch side halved, so (1/2)^-2 = 4); compute x{b16["gflops"] / b32["gflops"]:.2f}')
for name, P, res in [('ViT-L/16', 16, 512), ('ViT-H/14', 14, 518)]:
    n = (res // P) ** 2
    log(f'  Table 2 fine-tuning resolution: {name} at {res}: grid {res // P}x{res // P}, N = {n}, sequence {n + 1}')
    R[f'{name}@{res}'] = dict(grid=res // P, N=n)
log('')

log('-- parameters of the released checkpoints, read from the weight files (no head, no pre-logits layer) --')


def count_file(repo):
    """Count the numbers in a checkpoint file without building the model. Returns (backbone, pooler)."""
    files = glob.glob(f'{HUB}/models--google--{repo}/snapshots/*/model.safetensors')
    if files:
        tot = pool = 0
        with safe_open(files[0], 'pt') as f:
            for k in f.keys():
                n = int(np.prod(f.get_slice(k).get_shape()))
                if 'classifier' in k:
                    continue
                if 'pooler' in k:
                    pool += n
                else:
                    tot += n
        return tot, pool, 'weight file'
    return None


params = {}
for size, P, repo in [('B', 32, 'vit-base-patch32-224-in21k'), ('B', 16, 'vit-base-patch16-224-in21k'),
                      ('L', 32, 'vit-large-patch32-224-in21k'), ('L', 16, 'vit-large-patch16-224'), ('H', 14, 'vit-huge-patch14-224-in21k')]:
    got = count_file(repo)
    if got is None:                    # ViT-H/14 weights are not downloaded: build it on the meta device from its config
        D, L, mlp, heads = SIZES[size]
        c = ViTConfig(hidden_size=D, num_hidden_layers=L, num_attention_heads=heads, intermediate_size=mlp, patch_size=P)
        with torch.device('meta'):
            m = ViTModel(c, add_pooling_layer=True)
        pool = sum(p.numel() for n_, p in m.named_parameters() if 'pooler' in n_)
        got = (sum(p.numel() for p in m.parameters()) - pool, pool, 'meta device')
    tot, pool, src = got
    params[f'ViT-{size}/{P}'] = dict(backbone=tot, pooler=pool, source=src, repo=repo)
    log(f'  ViT-{size}/{P:<2} ({repo}, {src}): backbone {tot:>11,}  = {tot / 1e6:6.1f}M   pre-logits layer {pool:>9,}   Table 1: {PAPER_PARAMS[size]}')
d16, d32 = params['ViT-B/16']['backbone'], params['ViT-B/32']['backbone']
log(f'  ViT-B/32 minus ViT-B/16: {d32 - d16:,} parameters ({100 * (d32 - d16) / d16:.1f}% more), '
    f'because E grows from 768x768 to 3072x768 (+{(3072 - 768) * 768:,}) and the position table shrinks from 197 to 50 rows ({(50 - 197) * 768:,})')
R['params'] = params
l32 = params['ViT-L/32']
log(f'  ViT-L/32 with its pre-logits layer: {(l32["backbone"] + l32["pooler"]) / 1e6:.1f}M; ViT-L/16 has {params["ViT-L/16"]["backbone"] / 1e6:.1f}M without one')
log('')

# ---------------------------------------------------------------- 2. hybrids: the ResNet-50 stages
log('== 2. the hybrid: which ResNet-50 stage feeds ViT ==')
proc = AutoImageProcessor.from_pretrained('microsoft/resnet-50')
rn = ResNetModel.from_pretrained('microsoft/resnet-50').eval()
img = Image.open(sorted(glob.glob(f'{IMNET}/val/n03888257/*.JPEG'))[0]).convert('RGB')   # a parachute
with torch.no_grad():
    out = rn(**proc(img.resize((224, 224)), return_tensors='pt', do_resize=False, do_center_crop=False), output_hidden_states=True)
stages = [tuple(h.shape[1:]) for h in out.hidden_states[1:]]
for i, s in enumerate(stages):
    log(f'  stage {i + 1} output: {s[0]:>4} channels x {s[1]:>2} x {s[2]:<2} -> as 1x1 patches: {s[1] * s[2]:>5} tokens of {s[0]} numbers')
log(f'  option (i), stage 4: {stages[3][1] * stages[3][2]} tokens; option (ii), extended stage 3: {stages[2][1] * stages[2][2]} tokens '
    f'= x{stages[2][1] * stages[2][2] // (stages[3][1] * stages[3][2])} (the "4x longer sequence")')
nbn = sum(1 for m in rn.modules() if isinstance(m, torch.nn.BatchNorm2d))
log(f'  microsoft/resnet-50 is a standard ResNet: it has {nbn} BatchNorm2d layers (BiT replaces each with Group Norm)')
R['resnet_stages'] = stages
R['resnet_bn_layers'] = nbn
log('')

# ---------------------------------------------------------------- 3. Batch Norm vs Group Norm, Weight Standardization
log('== 3. Batch Norm vs Group Norm on a tiny batch (4 channels, 2 pixels per channel, 2 groups) ==')
a = torch.tensor([[1., 3.], [2., 2.], [0., 4.], [5., 7.]])        # picture A: 4 channels x 2 pixels
b1 = torch.tensor([[9., 9.], [0., 0.], [1., 1.], [2., 2.]])       # partner picture 1
b2 = torch.tensor([[-5., -3.], [6., 8.], [2., 0.], [3., 3.]])     # partner picture 2


def bn(batch):        # normalise each channel over the batch and the pixels
    m = batch.mean(dim=(0, 2), keepdim=True); v = batch.var(dim=(0, 2), unbiased=False, keepdim=True)
    return (batch - m) / torch.sqrt(v + 1e-5)


def gn(x, groups=2):  # normalise each group of channels inside one picture
    g = x.reshape(groups, -1)
    return ((g - g.mean(1, keepdim=True)) / torch.sqrt(g.var(1, unbiased=False, keepdim=True) + 1e-5)).reshape(x.shape)


fm = lambda t: '[' + ', '.join(f'{v:6.3f}' for v in t.flatten().tolist()) + ']'
bn1, bn2 = bn(torch.stack([a, b1]))[0], bn(torch.stack([a, b2]))[0]
gn1, gn2 = gn(a), gn(a)
log(f'  picture A, channel 1 = {a[0].tolist()}')
log(f'  Batch Norm, A batched with partner 1: A channel 1 -> {fm(bn1[0])}')
log(f'  Batch Norm, A batched with partner 2: A channel 1 -> {fm(bn2[0])}   (same picture, different output)')
gm = a[:2].mean().item(); gs = a[:2].var(unbiased=False).sqrt().item()
log(f'  Group Norm, group 1 = channels 1 and 2 of A = {a[:2].flatten().tolist()}: mean {gm:.3f}, std {gs:.3f}')
log(f'  Group Norm: A channel 1 -> {fm(gn1[0])} with any partner (it never looks at the batch); torch GroupNorm agrees: '
    f'{torch.allclose(Fn.group_norm(a.unsqueeze(0), 2, eps=1e-5)[0], gn1, atol=1e-6)}')
R['bn_gn'] = dict(a=a.tolist(), bn1=bn1[0].tolist(), bn2=bn2[0].tolist(), gn=gn1[0].tolist(), gmean=gm, gstd=gs)
w = torch.tensor([[0.2, -0.1, 0.5, 0.0], [1.0, 2.0, 3.0, 2.0]])     # two filters, 4 weights each
ws = (w - w.mean(1, keepdim=True)) / w.std(1, unbiased=False, keepdim=True)
log(f'  Weight Standardization: filter 2 = {w[1].tolist()}: mean {w[1].mean():.2f}, std {w[1].std(unbiased=False):.4f} -> {fm(ws[1])}')
log(f'  filter 1 = {[round(x, 2) for x in w[0].tolist()]} -> {fm(ws[0])}   (every filter now has mean 0 and std 1 before the convolution uses it)')
R['ws'] = dict(w=w.tolist(), ws=ws.tolist())
log('')

# ---------------------------------------------------------------- 4. the schedules of Tables 3 and 4
log('== 4. Tables 3 and 4 as steps and epochs ==')
JFT, I21K, I1K = 303e6, 14e6, 1.28e6     # images (Section 4.1; ImageNet-1k train set 1.28M)
sched = {}
for name, images, epochs, lr, decay in [('ViT-B/16 on JFT-300M', JFT, 7, 8e-4, 'linear'), ('ViT-H/14 on JFT-300M', JFT, 14, 3e-4, 'linear'),
                                        ('ViT-B/16 on ImageNet-21k', I21K, 90, 1e-3, 'linear'), ('ViT on ImageNet', I1K, 300, 3e-3, 'cosine')]:
    steps = epochs * images / 4096
    sched[name] = dict(steps=round(steps), epochs=epochs, base_lr=lr, decay=decay, warmup=10000)
    log(f'  pre-training {name:<26}: {epochs:>3} epochs x {images / 1e6:6.2f}M images / batch 4096 = {steps:>9,.0f} steps; '
        f'warm-up 10,000 = {100 * 10000 / steps:4.1f}% of them; base LR {lr:g}, {decay} decay')
FT = [('ImageNet', 20000, 1.28e6), ('CIFAR-100', 10000, 50000), ('CIFAR-10', 10000, 50000), ('Oxford-IIIT Pets', 500, 3680),
      ('Oxford Flowers-102', 500, 2040), ('VTAB task', 2500, 1000)]
for name, steps, n in FT:
    seen = steps * 512
    sched['ft ' + name] = dict(steps=steps, train=n, epochs=round(seen / n, 1))
    log(f'  fine-tuning {name:<19}: {steps:>6,} steps x batch 512 = {seen / 1e6:6.2f}M pictures seen = {seen / n:7.1f} passes over {n:,.0f} training pictures')
R['schedules'] = sched
for name, n in [('ImageNet', 1.3e6), ('ImageNet-21k', 14e6), ('JFT-300M', 303e6)]:
    log(f'  {name:<13} {n / 1e6:6.1f}M pictures = {n / 9469:9,.0f} times the 9,469 training pictures of Imagenette')


def lr_pre(t, total, base, warm=10000, kind='linear'):
    if t < warm:
        return base * t / warm
    p = (t - warm) / (total - warm)
    return base * (1 - p) if kind == 'linear' else base * 0.5 * (1 + math.cos(math.pi * p))


tot = sched['ViT-B/16 on JFT-300M']['steps']
R['curve_pre'] = [[t, lr_pre(t, tot, 8e-4)] for t in np.linspace(0, tot, 121).astype(int).tolist()]
R['curve_pre_im'] = [[t, lr_pre(t, sched['ViT on ImageNet']['steps'], 3e-3, kind='cosine')]
                     for t in np.linspace(0, sched['ViT on ImageNet']['steps'], 121).astype(int).tolist()]
R['curve_ft'] = [[t, 0.03 * 0.5 * (1 + math.cos(math.pi * t / 20000))] for t in np.linspace(0, 20000, 121).astype(int).tolist()]
log(f'  ViT-B/16 on JFT: LR at step 5,000 = {lr_pre(5000, tot, 8e-4):.1e} (half way up the warm-up), at 10,000 = {lr_pre(10000, tot, 8e-4):.1e}, '
    f'at the middle of the decay = {lr_pre(10000 + (tot - 10000) // 2, tot, 8e-4):.1e}')
log(f'  ImageNet fine-tuning at base LR 0.03, cosine: step 0 -> 0.03, step 10,000 -> {0.03 * 0.5 * (1 + math.cos(math.pi / 2)):.3f}, '
    f'step 15,000 -> {0.03 * 0.5 * (1 + math.cos(math.pi * 0.75)):.4f}, step 20,000 -> 0')
log('')

# ---------------------------------------------------------------- 5. optimizer steps by hand
log('== 5. one weight, three gradients: Adam, SGD with momentum, clipping, Polyak averaging ==')
gs3 = [0.5, -0.1, 0.3]
w_, m, v = 1.0, 0.0, 0.0
lr, b1_, b2_ = 1e-3, 0.9, 0.999
adam = []
for t, g in enumerate(gs3, 1):
    m = b1_ * m + (1 - b1_) * g
    v = b2_ * v + (1 - b2_) * g * g
    mh, vh = m / (1 - b1_ ** t), v / (1 - b2_ ** t)
    step = lr * mh / (math.sqrt(vh) + 1e-8)
    w_ -= step
    adam.append(dict(t=t, g=g, m=m, v=v, mh=mh, vh=vh, step=step, w=w_))
    log(f'  Adam t={t}: g={g:5.2f}  m={m:.4f}  v={v:.6f}  m_hat={mh:.4f}  v_hat={vh:.4f}  step = lr*m_hat/sqrt(v_hat) = {step:.6f}  w={w_:.6f}')
p = torch.nn.Parameter(torch.tensor([1.0]))
opt = torch.optim.Adam([p], lr=lr, betas=(b1_, b2_), eps=1e-8)
for g in gs3:
    opt.zero_grad(); p.grad = torch.tensor([g]); opt.step()
log(f'  torch.optim.Adam after the same three gradients: w={p.item():.6f}  (ours {w_:.6f})')
w_, buf, lr_s = 1.0, 0.0, 0.01
sgd = []
for t, g in enumerate(gs3, 1):
    buf = 0.9 * buf + g
    w_ -= lr_s * buf
    sgd.append(dict(t=t, g=g, buf=buf, w=w_))
    log(f'  SGD momentum 0.9, lr 0.01, t={t}: g={g:5.2f}  velocity = 0.9*velocity + g = {buf:.4f}  w = w - lr*velocity = {w_:.5f}')
p = torch.nn.Parameter(torch.tensor([1.0]))
opt = torch.optim.SGD([p], lr=lr_s, momentum=0.9)
for g in gs3:
    opt.zero_grad(); p.grad = torch.tensor([g]); opt.step()
log(f'  torch.optim.SGD(momentum=0.9) after the same gradients: w={p.item():.5f}  (ours {w_:.5f})')
R['adam'], R['sgd'] = adam, sgd
ga, gb = torch.tensor([3.0, 0.0]), torch.tensor([4.0])
norm = torch.sqrt((ga ** 2).sum() + (gb ** 2).sum()).item()
log(f'  clipping at global norm 1: gradients of two tensors [3, 0] and [4]; global norm sqrt(9 + 0 + 16) = {norm:.1f} > 1, '
    f'so every gradient is multiplied by 1/{norm:.0f}: [{3 / norm:.1f}, 0.0] and [{4 / norm:.1f}]; direction kept, length 1')
pa, pb = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(1))
pa.grad, pb.grad = ga.clone(), gb.clone()
torch.nn.utils.clip_grad_norm_([pa, pb], 1.0)
log(f'  torch.nn.utils.clip_grad_norm_: {[round(x, 4) for x in pa.grad.tolist()]} and {[round(x, 4) for x in pb.grad.tolist()]}')
ws_seq = [1.00, 1.20, 0.90, 1.10, 1.05]
avg, avgs = ws_seq[0], []
for x in ws_seq:
    avg = 0.9 * avg + 0.1 * x
    avgs.append(avg)
log(f'  Polyak (exponential) averaging, factor 0.9: weights {ws_seq} -> averages {[round(x, 4) for x in avgs]}')
log(f'  a factor of 0.9999 averages over roughly the last 1/(1 - 0.9999) = {1 / (1 - 0.9999):,.0f} steps')
R['polyak'] = dict(ws=ws_seq, avg=avgs)
save('part3_setup', R)
log('saved results/part3_setup.json')
