"""Part 2, the model by hand: every step of Section 3.1 and Appendix A of the ViT paper (Eq. 1 to 8) recomputed
in plain torch on the released google/vit-base-patch16-224 checkpoint with one real 224x224 picture, and compared
with the library at each step: patch cutting and flattening, the conv-equals-linear patch embedding, the [class]
token and position embeddings (Eq. 1), LayerNorm by hand, one attention head of layer 1, the 12-head concat and
output projection, the two residuals (Eq. 2, 3), all 12 layers in our own loop, the final LayerNorm and the
classifier (Eq. 4), the single-linear-layer head versus the in21k pooler, and the parameter-count formula for
ViT-B/16, ViT-L/16 and ViT-H/14 (the last two on the meta device). Also saves the sample picture to
public/img/papers/vit/p2-sample.png. Runs on the CPU in about a minute.
Writes results/part2_math.json and results/part2_math_stdout.txt."""
import math, os
import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image, ImageDraw
from transformers import ViTForImageClassification, ViTModel, ViTConfig
from datasets import load_dataset
from common import Log, save

torch.manual_seed(0)
torch.set_printoptions(precision=3, sci_mode=False)
log = Log('part2_math')
R = {}
fmt = lambda v, n=6: '[' + ', '.join(f'{x:7.3f}' for x in list(v)[:n]) + ']'
IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'public', 'img', 'papers', 'vit')
os.makedirs(IMG, exist_ok=True)

# ---------------------------------------------------------------- 0. the model and one real picture
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', attn_implementation='eager').eval()
cfg = model.config
D, L, k, P, C = cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.patch_size, cfg.num_channels
Dh = D // k
H = W = cfg.image_size
N = (H // P) * (W // P)
pic = load_dataset('huggingface/cats-image', split='test')[0]['image'].convert('RGB')
im224 = pic.resize((224, 224), Image.BILINEAR)
im224.resize((448, 448), Image.NEAREST).save(os.path.join(IMG, 'p2-sample.png'), optimize=True)
grid = im224.resize((448, 448), Image.NEAREST).convert('RGB')
d = ImageDraw.Draw(grid)
for i in range(1, 14):
    d.line([(i * 32, 0), (i * 32, 448)], fill=(255, 255, 255), width=1)
    d.line([(0, i * 32), (448, i * 32)], fill=(255, 255, 255), width=1)
grid.save(os.path.join(IMG, 'p2-sample-grid.png'), optimize=True)
# the released preprocessing: resize to 224x224, scale to [0, 1], then (x - 0.5) / 0.5 per channel
px = torch.from_numpy(np.asarray(im224).copy()).float().permute(2, 0, 1) / 255.0   # (3, 224, 224) in [0, 1]
x = (px - 0.5) / 0.5                                      # x in R^{C x H x W}: (3, 224, 224)
pixel_values = x.unsqueeze(0)                             # a batch of one picture: (1, 3, 224, 224)

log('== 0. the model and the picture ==')
log(f'  checkpoint: google/vit-base-patch16-224   D={D} L={L} heads={k} D_h={Dh} MLP={cfg.intermediate_size} P={P} C={C}')
log(f'  picture: {pic.size[0]}x{pic.size[1]} photo of two cats, resized to {H}x{W}; x shape {tuple(x.shape)} (C, H, W), values in [{x.min():.2f}, {x.max():.2f}]')
log(f'  N = HW / P^2 = {H}*{W} / {P}^2 = {H * W} / {P * P} = {N} patches, each P^2*C = {P}*{P}*{C} = {P * P * C} numbers')
R['config'] = dict(D=D, L=L, heads=k, Dh=Dh, mlp=cfg.intermediate_size, P=P, C=C, H=H, W=W, N=N, eps=cfg.layer_norm_eps,
                   picture=list(pic.size))
log('')

# ---------------------------------------------------------------- 1. cut the picture into patches and flatten them
log('== 1. cut x (3, 224, 224) into 196 patches of 16x16x3 and flatten each to 768 numbers ==')
# x[:, i*16:(i+1)*16, j*16:(j+1)*16] is patch (row i, col j). unfold gives (C, 14, 14, 16, 16); we order the
# 768 numbers as (channel, row in patch, col in patch), the same order as the conv weight below.
patches = x.unfold(1, P, P).unfold(2, P, P)              # (3, 14, 14, 16, 16)
patches = patches.permute(1, 2, 0, 3, 4).reshape(N, C * P * P)   # (196, 768): row n = patch n, row-major over the 14x14 grid
log(f'  x.unfold -> {tuple(x.unfold(1, P, P).unfold(2, P, P).shape)}  (channels, 14 rows of patches, 14 columns, 16, 16)')
log(f'  x_p = flattened patches: {tuple(patches.shape)}   (N = {N} patches, P^2*C = {C * P * P} numbers each)')
log(f'  patch 0 (top-left corner), first 8 of its 768 numbers: {fmt(patches[0], 8)}')
log(f'  patch 0, number 0 is red channel, pixel (0,0): x[0,0,0] = {x[0, 0, 0]:.3f}; number 256 is green (0,0): {x[1, 0, 0]:.3f}; number 512 is blue (0,0): {x[2, 0, 0]:.3f}')
log(f'  check: patch 17 (row 1, col 3) == x[:, 16:32, 48:64] flattened: {torch.equal(patches[17], x[:, 16:32, 48:64].reshape(-1))}')
R['patch0_first8'] = [round(v, 4) for v in patches[0][:8].tolist()]
R['patch0_first48'] = [round(v, 4) for v in patches[0][:48].tolist()]
R['patch0_green_first16'] = [round(v, 4) for v in patches[0][256:272].tolist()]
R['patch0_blue_first16'] = [round(v, 4) for v in patches[0][512:528].tolist()]
R['patch_shape'] = list(patches.shape)
R['patch0_rgb'] = [round(x[c, 0, 0].item(), 4) for c in range(3)]
log('')

# ---------------------------------------------------------------- 2. the library's Conv2d patch embedding is a Linear(768, 768)
log('== 2. the patch embedding E: the library\'s Conv2d(3, 768, kernel 16, stride 16) equals Linear(768 -> 768) on x_p ==')
emb = model.vit.embeddings
conv = emb.patch_embeddings.projection
Wc, bc = conv.weight.detach(), conv.bias.detach()        # (768, 3, 16, 16), (768,)
E = Wc.reshape(D, C * P * P).T                           # E in R^{(P^2 C) x D} = (768, 768): column d = filter d, flattened
with torch.no_grad():
    lib_patch_emb = conv(pixel_values).flatten(2).transpose(1, 2)[0]   # what the library computes: (196, 768)
    our_patch_emb = patches @ E + bc                                     # x_p E (+ the conv bias): (196, 768)
log(f'  conv weight {tuple(Wc.shape)} reshaped to E {tuple(E.shape)} (rows = the 768 pixel numbers of a patch, columns = D = {D})')
log(f'  x_p E: ({N} x {C * P * P}) . ({C * P * P} x {D}) = {tuple(our_patch_emb.shape)}')
log(f'  max |difference|, our x_p E + b vs the library conv: {(our_patch_emb - lib_patch_emb).abs().max():.2e}')
log(f'  patch 0 embedding, first 6 of 768: {fmt(our_patch_emb[0])}')
R['conv_vs_linear_maxdiff'] = (our_patch_emb - lib_patch_emb).abs().max().item()
R['E_shape'] = list(E.shape)
R['patch0_emb_first6'] = [round(v, 4) for v in our_patch_emb[0][:6].tolist()]
log('')

# ---------------------------------------------------------------- 3. Eq. 1: prepend [class], add position embeddings
log('== 3. Eq. 1: z_0 = [x_class; x_p^1 E; ...; x_p^N E] + E_pos ==')
x_class = emb.cls_token.detach()[0]                      # (1, 768): the learnable [class] embedding
E_pos = emb.position_embeddings.detach()[0]              # (197, 768): one learnable row per position, 0 = [class]
z0 = torch.cat([x_class, our_patch_emb], 0) + E_pos      # (197, 768)
with torch.no_grad():
    lib_z0 = emb(pixel_values)[0]
log(f'  x_class {tuple(x_class.shape)}   [x_class; patches] {tuple(torch.cat([x_class, our_patch_emb], 0).shape)}   E_pos {tuple(E_pos.shape)}   z_0 {tuple(z0.shape)}')
log(f'  x_class, first 6:        {fmt(x_class[0])}')
log(f'  E_pos[0] (for [class]):  {fmt(E_pos[0])}')
log(f'  z_0[0] = x_class + E_pos[0]: {fmt(z0[0])}')
log(f'  E_pos[1] (for patch 1):  {fmt(E_pos[1])}')
log(f'  z_0[1] = x_p^1 E + E_pos[1]: {fmt(z0[1])}')
log(f'  max |difference|, our z_0 vs the library embedding output: {(z0 - lib_z0).abs().max():.2e}')
R['eq1'] = dict(x_class_first6=[round(v, 4) for v in x_class[0][:6].tolist()], Epos0_first6=[round(v, 4) for v in E_pos[0][:6].tolist()],
                z0_0_first6=[round(v, 4) for v in z0[0][:6].tolist()], Epos1_first6=[round(v, 4) for v in E_pos[1][:6].tolist()],
                z0_1_first6=[round(v, 4) for v in z0[1][:6].tolist()], maxdiff=(z0 - lib_z0).abs().max().item(),
                shapes=dict(x_class=list(x_class.shape), E_pos=list(E_pos.shape), z0=list(z0.shape)))
log('')

# ---------------------------------------------------------------- the library's reference values for every layer
with torch.no_grad():
    out = model(pixel_values, output_hidden_states=True, output_attentions=True)
hs = out.hidden_states                                   # 13 tensors: z_0, z_1, ..., z_12 (before the final LayerNorm)
lib_attn = out.attentions                                # 12 tensors of (1, 12, 197, 197)


def layernorm_by_hand(v, ln, eps):
    mu = v.mean()
    var = ((v - mu) ** 2).mean()                         # biased variance, as torch.nn.LayerNorm uses
    return ln.weight.detach() * (v - mu) / torch.sqrt(var + eps) + ln.bias.detach(), mu, var


def gelu_exact(t):
    return 0.5 * t * (1.0 + torch.erf(t / math.sqrt(2.0)))   # x * Phi(x), Phi = standard normal CDF


def layer_by_hand(z, lay, want_details=False):
    """One encoder layer, Eq. 2 and 3, in plain torch. z: (197, 768). Returns z_next (and details for the article)."""
    eps = cfg.layer_norm_eps
    g1, b1 = lay.layernorm_before.weight.detach(), lay.layernorm_before.bias.detach()
    mu = z.mean(-1, keepdim=True)
    var = ((z - mu) ** 2).mean(-1, keepdim=True)
    ln1 = g1 * (z - mu) / torch.sqrt(var + eps) + b1                    # LN(z_{l-1}), per token
    at = lay.attention
    q = ln1 @ at.q_proj.weight.detach().T + at.q_proj.bias.detach()     # (197, 768) = 12 heads x 64
    kk = ln1 @ at.k_proj.weight.detach().T + at.k_proj.bias.detach()
    v = ln1 @ at.v_proj.weight.detach().T + at.v_proj.bias.detach()
    qh, kh, vh = (t.view(-1, k, Dh).transpose(0, 1) for t in (q, kk, v))   # (12, 197, 64) each
    scores = qh @ kh.transpose(1, 2) / math.sqrt(Dh)                   # (12, 197, 197)   Eq. 6 inside the softmax
    A = torch.softmax(scores, -1)                                       # Eq. 6
    SA = A @ vh                                                         # (12, 197, 64)    Eq. 7, one per head
    concat = SA.transpose(0, 1).reshape(-1, k * Dh)                     # (197, 768)       [SA_1; ...; SA_12]
    msa = concat @ at.o_proj.weight.detach().T + at.o_proj.bias.detach()   # Eq. 8: ... U_msa
    z_prime = msa + z                                                   # Eq. 2
    g2, b2 = lay.layernorm_after.weight.detach(), lay.layernorm_after.bias.detach()
    mu2 = z_prime.mean(-1, keepdim=True)
    var2 = ((z_prime - mu2) ** 2).mean(-1, keepdim=True)
    ln2 = g2 * (z_prime - mu2) / torch.sqrt(var2 + eps) + b2
    h = ln2 @ lay.mlp.fc1.weight.detach().T + lay.mlp.fc1.bias.detach()  # (197, 3072)
    hg = gelu_exact(h)
    mlp = hg @ lay.mlp.fc2.weight.detach().T + lay.mlp.fc2.bias.detach()  # (197, 768)
    z_next = mlp + z_prime                                              # Eq. 3
    if not want_details:
        return z_next
    return z_next, dict(ln1=ln1, q=q, k=kk, v=v, qh=qh, kh=kh, vh=vh, scores=scores, A=A, SA=SA, concat=concat, msa=msa,
                        z_prime=z_prime, ln2=ln2, h=h, hg=hg, mlp=mlp)


# ---------------------------------------------------------------- 4. layer 1 by hand, with the numbers
log('== 4. layer 1 by hand: LayerNorm, q k v, attention of head 1 for the [class] token, concat, U_msa, residual, LN, MLP, residual ==')
lay = model.vit.layers[0]
eps = cfg.layer_norm_eps
v0 = z0[0]                                               # the [class] token's 768 numbers entering layer 1
ln_hand, mu, var = layernorm_by_hand(v0, lay.layernorm_before, eps)
with torch.no_grad():
    ln_lib = lay.layernorm_before(v0)
log(f'  LayerNorm of the [class] token (768 numbers), eps = {eps:g}:')
log(f'    input, first 6:            {fmt(v0)}')
log(f'    mean mu = {mu:.4f}   variance = {var:.4f}   std = sqrt(var + eps) = {torch.sqrt(var + eps):.4f}')
log(f'    (x - mu)/std, first 6:     {fmt((v0 - mu) / torch.sqrt(var + eps))}')
log(f'    gamma, first 6:            {fmt(lay.layernorm_before.weight.detach())}')
log(f'    beta, first 6:             {fmt(lay.layernorm_before.bias.detach())}')
log(f'    gamma*(..)+beta, first 6:  {fmt(ln_hand)}')
log(f'    library LayerNorm:         {fmt(ln_lib)}   max |difference| {(ln_hand - ln_lib).abs().max():.1e}')
R['layernorm'] = dict(input_first6=[round(t, 4) for t in v0[:6].tolist()], mean=mu.item(), var=var.item(), std=torch.sqrt(var + eps).item(),
                      normed_first6=[round(t, 4) for t in ((v0 - mu) / torch.sqrt(var + eps))[:6].tolist()],
                      gamma_first6=[round(t, 4) for t in lay.layernorm_before.weight.detach()[:6].tolist()],
                      beta_first6=[round(t, 4) for t in lay.layernorm_before.bias.detach()[:6].tolist()],
                      out_first6=[round(t, 4) for t in ln_hand[:6].tolist()], maxdiff=(ln_hand - ln_lib).abs().max().item(),
                      len_in=v0.norm().item(), len_out=ln_hand.norm().item())

z1, dt = layer_by_hand(z0, lay, want_details=True)
log(f'  q = LN(z_0) W_q + b_q: {tuple(dt["q"].shape)}  (the same for k and v); split into {k} heads of {Dh}: {tuple(dt["qh"].shape)}')
log(f'  head 1: q_class . k_j / sqrt({Dh}) for all 197 tokens j: {tuple(dt["scores"][0, 0].shape)}; first 6 scores: {fmt(dt["scores"][0, 0])}')
A_cls = dt['A'][0, 0]                                    # head 1 (index 0), query = [class] (row 0): 197 weights
log(f'  softmax -> weights, first 6: {fmt(A_cls)}   sum of all 197 = {A_cls.sum():.4f}   max {A_cls.max():.4f}   min {A_cls.min():.4f}')
top = torch.topk(A_cls, 6)
log('  the 6 largest weights of the [class] query in head 1 of layer 1:')
for w_, i in zip(top.values.tolist(), top.indices.tolist()):
    where = '[class] itself' if i == 0 else f'patch {i - 1:3d} = grid row {(i - 1) // 14:2d}, col {(i - 1) % 14:2d}'
    log(f'    token {i:3d}  weight {w_:.4f}   {where}')
log(f'  weight on [class] itself: {A_cls[0]:.4f};  on all 196 patches together: {A_cls[1:].sum():.4f}')
self_w = dt['A'][:, 0, 0]
log('  weight of the [class] query on itself in each of the 12 heads of layer 1: ' + ' '.join(f'{t:.2f}' for t in self_w.tolist()))
hbest = int(self_w.argmin())
A_best = dt['A'][hbest, 0]
topb = torch.topk(A_best, 5)
log(f'  head {hbest + 1} spreads the [class] query most (self weight {self_w[hbest]:.3f}); its top-5 patches:')
for w_, i in zip(topb.values.tolist(), topb.indices.tolist()):
    where = '[class] itself' if i == 0 else f'patch {i - 1:3d} = grid row {(i - 1) // 14:2d}, col {(i - 1) % 14:2d}'
    log(f'    token {i:3d}  weight {w_:.4f}   {where}')
log(f'  SA_1 for [class] = sum_j A_0j v_j: {tuple(dt["SA"][0, 0].shape)}, first 6: {fmt(dt["SA"][0, 0])}')
log(f'  concat of 12 heads: {tuple(dt["concat"].shape)};  times U_msa {tuple(lay.attention.o_proj.weight.shape)} + bias: {tuple(dt["msa"].shape)}')
log(f'  Eq. 2: z\'_1 = MSA + z_0, [class] first 6: {fmt(dt["z_prime"][0])}')
log(f'  LN(z\'_1) [class] first 6:   {fmt(dt["ln2"][0])}')
log(f'  MLP: LN(z\') W_1 + b_1 -> {tuple(dt["h"].shape)}; GELU; W_2 + b_2 -> {tuple(dt["mlp"].shape)}')
log(f'  [class] before GELU, first 6: {fmt(dt["h"][0])}')
log(f'  [class] after GELU, first 6:  {fmt(dt["hg"][0])}')
log(f'  share of the 3072 numbers that are positive before GELU ([class]): {(dt["h"][0] > 0).float().mean():.3f}')
log(f'  Eq. 3: z_1 = MLP + z\'_1, [class] first 6: {fmt(z1[0])}')
with torch.no_grad():
    lib_z1 = hs[1][0]
log(f'  max |difference|, our z_1 vs the library hidden_states[1]: {(z1 - lib_z1).abs().max():.2e}')
log(f'  max |difference|, our head-1 weights vs the library attentions[0] (all 12 heads): {(dt["A"] - lib_attn[0][0]).abs().max():.2e}')
# GELU at a few values
xs = torch.tensor([-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0])
log('  GELU(x) = x * Phi(x) at a few values, against torch.nn.functional.gelu:')
log('    x:               ' + ' '.join(f'{t:7.2f}' for t in xs.tolist()))
log('    Phi(x):          ' + ' '.join(f'{t:7.4f}' for t in (0.5 * (1 + torch.erf(xs / math.sqrt(2)))).tolist()))
log('    x*Phi(x):        ' + ' '.join(f'{t:7.4f}' for t in gelu_exact(xs).tolist()))
log('    F.gelu(x):       ' + ' '.join(f'{t:7.4f}' for t in Fn.gelu(xs).tolist()))
log('    relu(x):         ' + ' '.join(f'{t:7.4f}' for t in Fn.relu(xs).tolist()))
gx = torch.linspace(-4, 4, 81)
R['gelu'] = dict(xs=xs.tolist(), phi=(0.5 * (1 + torch.erf(xs / math.sqrt(2)))).tolist(), gelu=gelu_exact(xs).tolist(),
                 curve_x=gx.tolist(), curve_y=gelu_exact(gx).tolist(), relu_y=Fn.relu(gx).tolist(),
                 maxdiff_vs_torch=(gelu_exact(gx) - Fn.gelu(gx)).abs().max().item())
A_grid = A_cls[1:].view(14, 14)
R['layer1'] = dict(
    scores_cls_first6=[round(t, 4) for t in dt['scores'][0, 0][:6].tolist()],
    A_cls_first6=[round(t, 4) for t in A_cls[:6].tolist()], A_cls_sum=A_cls.sum().item(), A_cls_self=A_cls[0].item(),
    A_cls_max=A_cls.max().item(), A_cls_min=A_cls.min().item(),
    top=[dict(token=i, weight=w_, row=(i - 1) // 14 if i else None, col=(i - 1) % 14 if i else None) for w_, i in zip(top.values.tolist(), top.indices.tolist())],
    A_cls_grid=[[round(t, 5) for t in r] for r in A_grid.tolist()],
    A_cls_full=[round(t, 6) for t in A_cls.tolist()],
    A_cls_grid_allheads=[[round(t, 5) for t in r] for r in dt['A'][:, 0, 1:].mean(0).view(14, 14).tolist()],
    A_cls_heads_self=[round(t, 4) for t in dt['A'][:, 0, 0].tolist()],
    best_head=hbest + 1, A_best_grid=[[round(t, 5) for t in r] for r in A_best[1:].view(14, 14).tolist()], A_best_self=A_best[0].item(),
    best_top=[dict(token=i, weight=w_, row=(i - 1) // 14 if i else None, col=(i - 1) % 14 if i else None) for w_, i in zip(topb.values.tolist(), topb.indices.tolist())],
    sa_cls_first6=[round(t, 4) for t in dt['SA'][0, 0][:6].tolist()],
    zprime_cls_first6=[round(t, 4) for t in dt['z_prime'][0][:6].tolist()], ln2_cls_first6=[round(t, 4) for t in dt['ln2'][0][:6].tolist()],
    h_cls_first6=[round(t, 4) for t in dt['h'][0][:6].tolist()], hg_cls_first6=[round(t, 4) for t in dt['hg'][0][:6].tolist()],
    positive_share=(dt['h'][0] > 0).float().mean().item(),
    z1_cls_first6=[round(t, 4) for t in z1[0][:6].tolist()], maxdiff_z1=(z1 - lib_z1).abs().max().item(),
    maxdiff_attn=(dt['A'] - lib_attn[0][0]).abs().max().item(),
    shapes=dict(q=list(dt['q'].shape), qh=list(dt['qh'].shape), scores=list(dt['scores'].shape), SA=list(dt['SA'].shape),
                concat=list(dt['concat'].shape), Umsa=list(lay.attention.o_proj.weight.shape), h=list(dt['h'].shape)))
log('')

# ---------------------------------------------------------------- 5. all 12 layers in our own loop
log('== 5. all 12 layers, our loop vs the library ==')
z = z0
diffs, rel = [], []
for l in range(L):
    z = layer_by_hand(z, model.vit.layers[l])
    diffs.append((z - hs[l + 1][0]).abs().max().item())
    rel.append(diffs[-1] / hs[l + 1][0].abs().max().item())
    log(f'  layer {l + 1:2d}: z_{l + 1} {tuple(z.shape)}   max |difference| vs hidden_states[{l + 1}]: {diffs[-1]:.2e}   '
        f'(largest value in z_{l + 1}: {hs[l + 1][0].abs().max():7.1f}, so relative {rel[-1]:.1e})')
zL = z
log(f'  the final logits from our z_12 vs the library differ by at most {(model.classifier(model.vit.layernorm(zL)[0]) - out.logits[0]).abs().max():.2e} (see step 6)')
R['layers_maxdiff'] = diffs
R['layers_reldiff'] = rel
R['layers_maxabs'] = [hs[l + 1][0].abs().max().item() for l in range(L)]
log('')

# ---------------------------------------------------------------- 6. Eq. 4 and the classification head
log('== 6. Eq. 4: y = LN(z_L^0), then the classification head ==')
y_hand, mu_L, var_L = layernorm_by_hand(zL[0], model.vit.layernorm, eps)
with torch.no_grad():
    y_lib = model.vit.layernorm(hs[-1])[0, 0]
    last = model.vit(pixel_values).last_hidden_state[0, 0]
log(f'  z_L^0 (the [class] token after 12 layers), first 6: {fmt(zL[0])}   mean {mu_L:.3f} var {var_L:.3f}')
log(f'  y = LN(z_L^0), first 6: {fmt(y_hand)}   max |difference| vs the library: {(y_hand - last).abs().max():.2e}')
head = model.classifier
log(f'  head of the fine-tuned checkpoint: {head}   (one linear layer, as Section 3.1 says for fine-tuning)')
logits = y_hand @ head.weight.detach().T + head.bias.detach()
with torch.no_grad():
    lib_logits = out.logits[0]
log(f'  logits = y W_head + b: {tuple(logits.shape)}   max |difference| vs model.logits: {(logits - lib_logits).abs().max():.2e}')
probs = torch.softmax(logits, -1)
top5 = torch.topk(probs, 5)
log('  top-5 ImageNet classes:')
top5_list = []
for p_, i in zip(top5.values.tolist(), top5.indices.tolist()):
    log(f'    {p_:6.3f}  class {i:3d}  {cfg.id2label[i]}')
    top5_list.append(dict(prob=p_, idx=i, label=cfg.id2label[i]))
in21k = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').eval()
log(f'  the pre-trained-only checkpoint google/vit-base-patch16-224-in21k ends in: {in21k.pooler}')
log(f'  fine-tuned checkpoint has a pooler: {model.vit.pooler is not None};  in21k checkpoint has a pooler: {in21k.pooler is not None}')
log(f'  (the paper: an MLP with one hidden layer at pre-training time, a single linear layer at fine-tuning time)')
R['eq4'] = dict(zL_cls_first6=[round(t, 4) for t in zL[0][:6].tolist()], y_first6=[round(t, 4) for t in y_hand[:6].tolist()],
                maxdiff_y=(y_hand - last).abs().max().item(), maxdiff_logits=(logits - lib_logits).abs().max().item(),
                head=str(head), top5=top5_list, in21k_pooler=str(in21k.pooler),
                ft_has_pooler=model.vit.pooler is not None, in21k_has_pooler=in21k.pooler is not None)
log('')


# ---------------------------------------------------------------- 7. parameter counting
def formula(D, L, P, C, N, K, mlp=None):
    mlp = mlp or 4 * D
    patch_emb = P * P * C * D + D
    cls = D
    pos = (N + 1) * D
    attn_layer = 4 * (D * D + D)
    mlp_layer = D * mlp + mlp + mlp * D + D
    ln_layer = 4 * D
    final_ln = 2 * D
    head = D * K + K
    total = patch_emb + cls + pos + L * (attn_layer + mlp_layer + ln_layer) + final_ln + head
    return dict(patch_emb=patch_emb, cls=cls, pos=pos, attn_layer=attn_layer, mlp_layer=mlp_layer, ln_layer=ln_layer,
                per_layer=attn_layer + mlp_layer + ln_layer, final_ln=final_ln, head=head, total=total, without_head=total - head,
                embeddings=patch_emb + cls + pos, attention=L * attn_layer, mlp_total=L * mlp_layer, layernorms=L * ln_layer + final_ln)


def count(m):
    return sum(p.numel() for p in m.parameters())


log('== 7. counting the parameters ==')
log('  formula: (P^2 C D + D) + D + (N+1) D + L (12 D^2 + 13 D) + 2 D + (D K + K)')
log('           patch emb       cls  position  L layers            LN   head')
K = 1000
f = formula(D, L, P, C, N, K)
real = count(model)
log(f'  ViT-B/16 at 224, K = 1000:')
log(f'    patch embedding {f["patch_emb"]:>12,}   [class] {f["cls"]:>6,}   positions {f["pos"]:>9,}')
log(f'    one layer: attention 4(D^2 + D) = {f["attn_layer"]:,}   MLP 8D^2 + 5D = {f["mlp_layer"]:,}   2 LayerNorms 4D = {f["ln_layer"]:,}   -> {f["per_layer"]:,}   x {L} = {L * f["per_layer"]:,}')
log(f'    final LayerNorm {f["final_ln"]:,}   head D K + K = {f["head"]:,}')
log(f'    formula total {f["total"]:,}   counted in model.parameters() {real:,}   same: {f["total"] == real}')
log(f'    without the 1000-class head: {f["without_head"]:,}   (paper, Table 1: 86M)')
tot = f['total']
shares = dict(attention=f['attention'], mlp=f['mlp_total'], embeddings=f['embeddings'], layernorms=f['layernorms'], head=f['head'])
log('    where they sit: ' + '   '.join(f'{n} {v / 1e6:.2f}M ({100 * v / tot:.1f}%)' for n, v in shares.items()))
R['params_B16'] = dict(formula=f, counted=real, same=f['total'] == real, shares=shares)

variants = {
    'ViT-L/16': dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096, patch_size=16, paper=307),
    'ViT-H/14': dict(hidden_size=1280, num_hidden_layers=32, num_attention_heads=16, intermediate_size=5120, patch_size=14, paper=632),
}
R['params_variants'] = {}
for name, vcfg in variants.items():
    paper = vcfg.pop('paper')
    c = ViTConfig(**vcfg, num_labels=K)
    with torch.device('meta'):
        m = ViTForImageClassification(c)
    n_meta = count(m)
    Nv = (224 // c.patch_size) ** 2
    fv = formula(c.hidden_size, c.num_hidden_layers, c.patch_size, 3, Nv, K, c.intermediate_size)
    log(f'  {name} at 224 (meta device, no weights): D={c.hidden_size} L={c.num_hidden_layers} heads={c.num_attention_heads} MLP={c.intermediate_size} N={Nv}')
    log(f'    formula {fv["total"]:,}   counted {n_meta:,}   same: {fv["total"] == n_meta}')
    log(f'    with the 1000-class head {fv["total"] / 1e6:.1f}M, without it {fv["without_head"] / 1e6:.1f}M   (paper: {paper}M)')
    R['params_variants'][name] = dict(formula=fv, counted=n_meta, same=fv['total'] == n_meta, paper=paper, cfg=vcfg, N=Nv)
log('')
log('done.')
save('part2_math', R)
