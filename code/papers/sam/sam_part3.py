"""Part 3: the model, opened up.
- the configuration of the three image encoders (layers, width, heads, window size, global-attention blocks)
- the weights of every piece: patch embedding, blocks, neck; prompt encoder; mask decoder
- the shape of the data at each step of the image encoder (ViT-H), and the cost of windowed vs global attention
- the prompt encoder: the random Fourier positional encoding of one click, number by number; boxes; masks
- the mask decoder, re-computed by hand step by step (two-way transformer, upscaling, dynamic dot product),
  checked against the model's own output
- the four output tokens, the IoU head, and where the mask tokens look in the picture
- compute (FLOPs) and time of each part on this computer
- the training losses by hand (focal, dice, IoU MSE, min over three masks) on real SAM outputs
- the simulated interactive training game: 11 rounds of prompts, played with the released model
- the training recipe arithmetic (learning-rate schedule, epochs)
Writes results/part3.json, results/part3_stdout.txt, results/part3_*.npy and pictures in public/img/papers/sam/."""
import math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset
from transformers import SamModel, SamProcessor
from common import Log, save

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, '..', '..', '..', 'public', 'img', 'papers', 'sam')
os.makedirs(IMG, exist_ok=True)
log = Log('part3')
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.manual_seed(0)
R = {}

image = load_dataset('huggingface/cats-image')['test'][0]['image'].convert('RGB')
W0, H0 = image.size
EAR, BODY, SOFA = (275, 62), (120, 220), (250, 400)        # pixels of the 640 x 480 photo
CAT_BOX = [5, 53, 317, 472]                                # a box around the left cat


def count(mod):
    return sum(p.numel() for p in mod.parameters())


# ------------------------------------------------------------------ 1. the three image encoders
log('== 1. the image encoders of the three released models ==')
log(f'{"model":<8}{"blocks":>7}{"width":>7}{"heads":>7}{"head dim":>9}{"MLP":>7}{"window":>8}  global-attention blocks (0-based)')
R['configs'] = {}
for name in ['base', 'large', 'huge']:
    m = SamModel.from_pretrained(f'facebook/sam-vit-{name}')
    v = m.config.vision_config
    enc = m.vision_encoder
    pe = count(enc.patch_embed)
    pos = enc.pos_embed.numel()
    win_blocks = [l for l in enc.layers if l.window_size > 0]
    glob_blocks = [l for l in enc.layers if l.window_size == 0]
    neck = count(enc.neck)
    R['configs'][name] = dict(blocks=v.num_hidden_layers, width=v.hidden_size, heads=v.num_attention_heads,
                              head_dim=v.hidden_size // v.num_attention_heads, mlp=v.mlp_dim, window=v.window_size,
                              global_blocks=list(v.global_attn_indexes), patch=v.patch_size,
                              patch_embed=pe, pos_embed=pos, window_block=count(win_blocks[0]), global_block=count(glob_blocks[0]),
                              neck=neck, encoder=count(enc))
    log(f'{"ViT-" + name[0].upper():<8}{v.num_hidden_layers:>7}{v.hidden_size:>7}{v.num_attention_heads:>7}'
        f'{v.hidden_size // v.num_attention_heads:>9}{v.mlp_dim:>7}{v.window_size:>8}  {list(v.global_attn_indexes)}')
    del m
log('\nweights inside each image encoder:')
log(f'{"model":<8}{"patch embed":>13}{"pos. embed":>12}{"window block":>14}{"global block":>14}{"neck":>10}{"total":>14}')
for name, c in R['configs'].items():
    log(f'{"ViT-" + name[0].upper():<8}{c["patch_embed"]:>13,}{c["pos_embed"]:>12,}{c["window_block"]:>14,}{c["global_block"]:>14,}'
        f'{c["neck"]:>10,}{c["encoder"]:>14,}')
c = R['configs']['huge']
nb, ng = c['blocks'] - len(c['global_blocks']), len(c['global_blocks'])
chk = c['patch_embed'] + c['pos_embed'] + nb * c['window_block'] + ng * c['global_block'] + c['neck']
log(f'check ViT-H: {c["patch_embed"]:,} + {c["pos_embed"]:,} + {nb} x {c["window_block"]:,} + {ng} x {c["global_block"]:,} + {c["neck"]:,} = {chk:,}')
assert chk == c['encoder']
blk = nb * c['window_block'] + ng * c['global_block']
log(f'the 32 blocks hold {blk:,} weights = {100 * blk / c["encoder"]:.1f}% of the encoder; one windowed block = '
    f'{c["window_block"] / 6476:,.0f} x the whole prompt encoder (6,476)')
log(f'pixels: 1024 x 1024 / (224 x 224) = {1024 ** 2 / 224 ** 2:.1f} times the input of a standard ViT')
log(f'a global block has {c["global_block"] - c["window_block"]:,} more weights than a windowed block: its relative-position '
    f'tables cover 2 x 64 - 1 = 127 offsets, not 2 x 14 - 1 = 27')

# ------------------------------------------------------------------ 2. ViT-H: shapes through the encoder
log('\n== 2. ViT-H: the shape of the data at each step of the image encoder ==')
proc = SamProcessor.from_pretrained('facebook/sam-vit-huge')
sam = SamModel.from_pretrained('facebook/sam-vit-huge').to(DEV).eval().requires_grad_(False)
enc = sam.vision_encoder
inp = proc(images=image, return_tensors='pt')
px = inp['pixel_values'].to(DEV)
shapes = []
with torch.no_grad():
    h = enc.patch_embed(px)
    shapes.append(('pixel values', tuple(px.shape)))
    shapes.append(('patch embedding (16 x 16 conv, stride 16)', tuple(h.shape)))
    h = h + enc.pos_embed
    shapes.append(('+ absolute position embedding', tuple(h.shape)))
    lw = enc.layers[0]
    hn = lw.layer_norm1(h)
    win, pad = lw.window_partition(hn, lw.window_size)
    shapes.append(('windowed block: pad 64 -> 70, cut into windows', tuple(win.shape)))
    for l in enc.layers:
        h = l(h)
    shapes.append(('after 32 blocks', tuple(h.shape)))
    n1 = enc.neck.layer_norm1(enc.neck.conv1(h.permute(0, 3, 1, 2)))
    shapes.append(('neck: 1 x 1 conv 1280 -> 256 + layer norm', tuple(n1.shape)))
    emb = enc.neck.layer_norm2(enc.neck.conv2(n1))
    shapes.append(('neck: 3 x 3 conv 256 -> 256 + layer norm', tuple(emb.shape)))
    ref = sam.get_image_embeddings(px)
for s, t in shapes:
    log(f'  {s:<48} {str(t):<24} {math.prod(t):>12,} numbers')
log(f'  our step-by-step embedding vs get_image_embeddings: max difference {float((emb - ref).abs().max()):.2e}')
R['encoder_shapes'] = [[s, list(t)] for s, t in shapes]

# windowed vs global attention
G, w = 64, 14
padded = math.ceil(G / w) * w
nwin = (padded // w) ** 2
pairs_win, pairs_glob = nwin * (w * w) ** 2, (G * G) ** 2
log(f'\nwindowed attention: the 64 x 64 grid is padded to {padded} x {padded} ({padded * padded - G * G} padding cells), '
    f'then cut into {padded // w} x {padded // w} = {nwin} windows of {w} x {w} = {w * w} tokens')
log(f'  score pairs per head: windowed {nwin} x {w * w}^2 = {pairs_win:,};  global 4096^2 = {pairs_glob:,};  ratio {pairs_glob / pairs_win:.1f}')
R['windows'] = dict(grid=G, window=w, padded=padded, pad_cells=padded * padded - G * G, n_windows=nwin,
                    tokens_per_window=w * w, pairs_window=pairs_win, pairs_global=pairs_glob, ratio=pairs_glob / pairs_win)
del ref, h, hn, win, n1

# ------------------------------------------------------------------ 3. the prompt encoder: one click, number by number
log('\n== 3. the prompt encoder: the click on the ear ==')
pe_mod = sam.prompt_encoder
Gm = pe_mod.shared_embedding.positional_embedding.detach().cpu()          # 2 x 128 random Gaussian matrix
log(f'random matrix G: shape {tuple(Gm.shape)}, mean {Gm.mean():.3f}, std {Gm.std():.3f}  (drawn once at random, then kept fixed)')
SC = 1024 / max(W0, H0)
x1024, y1024 = EAR[0] * SC, EAR[1] * SC
u = ((x1024 + 0.5) / 1024, (y1024 + 0.5) / 1024)
cpt = (2 * u[0] - 1, 2 * u[1] - 1)
log(f'click {EAR} in the photo -> x {SC:.1f} -> ({x1024:.1f}, {y1024:.1f}) in the 1024 frame -> +0.5 (pixel centre) / 1024 -> '
    f'({u[0]:.4f}, {u[1]:.4f}) -> 2u - 1 -> ({cpt[0]:.4f}, {cpt[1]:.4f})')
cvec = torch.tensor(cpt, dtype=torch.float32)
proj = cvec @ Gm                                                            # 128 numbers
ang = 2 * math.pi * proj
penc = torch.cat([torch.sin(ang), torch.cos(ang)])
for j in range(3):
    log(f'  column {j}: g = ({Gm[0, j]:+.3f}, {Gm[1, j]:+.3f});  c . g = {cpt[0]:+.4f} x {Gm[0, j]:+.3f} + {cpt[1]:+.4f} x {Gm[1, j]:+.3f} = {proj[j]:+.4f};'
        f'  angle 2 pi x {proj[j]:+.4f} = {ang[j]:+.3f};  sin {math.sin(ang[j]):+.3f}, cos {math.cos(ang[j]):+.3f}')
log(f'positional encoding: 128 sines + 128 cosines = {penc.numel()} numbers; first four: '
    + ', '.join(f'{v:+.3f}' for v in penc[:4].tolist()))
norms = Gm.norm(dim=0)
periods = 512 / norms
log(f'stripe width (period) of each of the 128 frequencies, in pixels of the 1024 frame: '
    f'shortest {periods.min():.1f}, median {periods.median():.1f}, longest {periods.max():.0f}')
R['posenc'] = dict(click=list(EAR), frame=[x1024, y1024], unit=list(u), centred=list(cpt),
                   cols=[[float(Gm[0, j]), float(Gm[1, j]), float(proj[j]), float(ang[j]), math.sin(ang[j]), math.cos(ang[j])] for j in range(3)],
                   first4=penc[:4].tolist(), period_min=float(periods.min()), period_median=float(periods.median()),
                   period_max=float(periods.max()), G_std=float(Gm.std()))

# compare with the model's own sparse embedding
p = proc(images=image, input_points=[[[list(EAR)]]], input_labels=[[[1]]], return_tensors='pt')
pts, labs = p['input_points'].float().to(DEV), p['input_labels'].to(DEV)
with torch.no_grad():
    sparse, dense = sam.get_prompt_embeddings(input_points=pts, input_labels=labs)
fg = pe_mod.point_embed[1].weight[0].detach().cpu()
mine = penc + fg
log(f'sparse prompt embedding from the model: {tuple(sparse.shape)}  (image, prompts, tokens, 256): the click + one padding token')
log(f'  token 1 = positional encoding + learned "foreground point" vector: max difference to ours {float((sparse[0, 0, 0].cpu() - mine).abs().max()):.2e}')
nap = pe_mod.not_a_point_embed.weight[0].detach().cpu()
log(f'  token 2 = the learned "not a point" vector (padding): max difference {float((sparse[0, 0, 1].cpu() - nap).abs().max()):.2e}')
log(f'dense prompt embedding (no mask given): {tuple(dense.shape)}: one learned "no mask" vector copied to all 64 x 64 cells')
types = {'foreground point': pe_mod.point_embed[1].weight[0], 'background point': pe_mod.point_embed[0].weight[0],
         'box top-left corner': pe_mod.point_embed[2].weight[0], 'box bottom-right corner': pe_mod.point_embed[3].weight[0],
         'not a point': pe_mod.not_a_point_embed.weight[0]}
log('the learned type vectors (256 numbers each): length, and its ratio to the length of a positional encoding (' + f'{penc.norm():.2f})')
R['type_vectors'] = {}
for k, v in types.items():
    n = float(v.detach().norm())
    R['type_vectors'][k] = n
    log(f'  {k:<24} length {n:6.2f}   ratio {n / float(penc.norm()):.2f}')
R['penc_norm'] = float(penc.norm())

# similarity of the encoding of the click with the encoding of every cell of the 64 x 64 grid
with torch.no_grad():
    grid_pe = sam.get_image_wide_positional_embeddings()[0].cpu()           # 256 x 64 x 64
sim = torch.einsum('c,chw->hw', penc, grid_pe) / (penc.norm() * grid_pe.norm(dim=0))
cx, cy = int(x1024 // 16), int(y1024 // 16)
log(f'cosine similarity of the click encoding with the grid encodings: at the click cell ({cx}, {cy}) {sim[cy, cx]:.3f}; '
    f'1 cell away {sim[cy, cx + 1]:.3f}; 4 cells away {sim[cy, cx + 4]:.3f}; 16 cells away {sim[cy, cx + 16]:.3f}; '
    f'far corner {sim[63, 0]:.3f}')
R['pe_sim'] = dict(cell=[cx, cy], at=float(sim[cy, cx]), d1=float(sim[cy, cx + 1]), d4=float(sim[cy, cx + 4]), d16=float(sim[cy, cx + 16]),
                   corner=float(sim[63, 0]))
np.save(os.path.join(HERE, 'results', 'part3_pe_sim.npy'), sim.numpy())

# a box and a mask as prompts
pb = proc(images=image, input_boxes=[[CAT_BOX]], return_tensors='pt')
with torch.no_grad():
    sb, db = sam.get_prompt_embeddings(input_boxes=pb['input_boxes'].float().to(DEV))
    mprompt = torch.zeros(1, 1, 256, 256, device=DEV)
    dm = pe_mod.mask_embed(mprompt)
    c1 = pe_mod.mask_embed.conv1(mprompt)
    c2 = pe_mod.mask_embed.conv2(pe_mod.mask_embed.activation(pe_mod.mask_embed.layer_norm1(c1)))
log(f'box {CAT_BOX}: sparse embedding {tuple(sb.shape)}: two tokens, top-left corner and bottom-right corner')
log(f'mask prompt 256 x 256: conv 2x2 stride 2 -> {tuple(c1.shape)}; conv 2x2 stride 2 -> {tuple(c2.shape)}; conv 1x1 -> {tuple(dm.shape)}; '
    f'then added to the image embedding cell by cell')
pe_parts = {'random matrix G (2 x 128)': Gm.numel(), 'point and corner types (4 x 256)': 4 * 256, '"not a point" (256)': 256,
            '"no mask" (256)': 256, 'mask convolutions': count(pe_mod.mask_embed)}
log('weights of the prompt encoder: ' + ', '.join(f'{k} {v:,}' for k, v in pe_parts.items()) + f'; total {sum(pe_parts.values()):,}')
assert sum(pe_parts.values()) == count(pe_mod)
R['prompt_encoder'] = dict(parts=pe_parts, total=count(pe_mod), box_sparse=list(sb.shape), mask_steps=[list(c1.shape), list(c2.shape), list(dm.shape)])

# ------------------------------------------------------------------ 4. the mask decoder, by hand
log('\n== 4. the mask decoder, step by step (ViT-H, the click on the ear) ==')
dec = sam.mask_decoder
tr = dec.transformer
dparts = {'output tokens (1 IoU + 4 mask, x 256)': count(dec.iou_token) + count(dec.mask_tokens),
          'two-way layer 1': count(tr.layers[0]), 'two-way layer 2': count(tr.layers[1]),
          'final token-to-image attention + norm': count(tr.final_attn_token_to_image) + count(tr.layer_norm_final_attn),
          'upscaling (2 transposed convs + norm)': count(dec.upscale_conv1) + count(dec.upscale_conv2) + count(dec.upscale_layer_norm),
          '4 mask MLPs (256 -> 256 -> 256 -> 32)': count(dec.output_hypernetworks_mlps), 'IoU head (256 -> 256 -> 256 -> 4)': count(dec.iou_prediction_head)}
L1 = tr.layers[0]
sub = {'self-attention': count(L1.self_attn), 'token-to-image attention': count(L1.cross_attn_token_to_image), 'MLP 256 -> 2048 -> 256': count(L1.mlp),
       'image-to-token attention': count(L1.cross_attn_image_to_token),
       '4 layer norms': count(L1.layer_norm1) + count(L1.layer_norm2) + count(L1.layer_norm3) + count(L1.layer_norm4)}
log('weights of the mask decoder:')
for k, v in dparts.items():
    log(f'  {k:<42}{v:>11,}')
log(f'  {"total":<42}{sum(dparts.values()):>11,}')
assert sum(dparts.values()) == count(dec)
log('  inside one two-way layer: ' + ', '.join(f'{k} {v:,}' for k, v in sub.items()))
R['decoder_params'] = dict(parts=dparts, layer=sub, total=count(dec))
log(f'attention widths: self-attention q, k, v = {L1.self_attn.internal_dim} (8 heads of {L1.self_attn.internal_dim // 8}); '
    f'cross-attention q, k, v = {L1.cross_attn_token_to_image.internal_dim} (8 heads of {L1.cross_attn_token_to_image.internal_dim // 8})')


def attend(mod, q, k, v, keep=False):
    """SAM's attention, written out: project, split into 8 heads, softmax(q k^T / sqrt(d)) v, join, project."""
    Q, K, V = mod.q_proj(q), mod.k_proj(k), mod.v_proj(v)
    nh, d = 8, Q.shape[-1] // 8
    sp = lambda t: t.reshape(t.shape[0], -1, nh, d).transpose(1, 2)
    A = torch.softmax(sp(Q) @ sp(K).transpose(-1, -2) / math.sqrt(d), dim=-1)
    out = (A @ sp(V)).transpose(1, 2).reshape(Q.shape[0], -1, nh * d)
    return (mod.out_proj(out), A) if keep else mod.out_proj(out)


with torch.no_grad():
    img_pe = sam.get_image_wide_positional_embeddings()                    # 1 x 256 x 64 x 64
    out_tok = torch.cat([dec.iou_token.weight, dec.mask_tokens.weight], 0)[None]   # 1 x 5 x 256
    prompt_tok = sparse[:, 0]                                               # 1 x 2 x 256
    tokens = torch.cat([out_tok, prompt_tok], 1)                            # 1 x 7 x 256
    keys = (emb + dense).flatten(2).transpose(1, 2)                         # 1 x 4096 x 256
    kpe = img_pe.flatten(2).transpose(1, 2)
    log(f'tokens: {out_tok.shape[1]} output tokens (1 IoU + 4 mask) + {prompt_tok.shape[1]} prompt tokens = {tuple(tokens.shape)}')
    log(f'image: embedding + dense prompt, flattened: {tuple(keys.shape)}  (the 64 x 64 grid as 4,096 vectors)')
    q = tokens
    trace = []
    for i, L in enumerate(tr.layers):
        if i == 0:
            q = L.layer_norm1(attend(L.self_attn, q, q, q))                 # first layer: no positional re-add, no residual
        else:
            q = L.layer_norm1(q + attend(L.self_attn, q + tokens, q + tokens, q))
        trace.append((f'layer {i + 1}, step 1: self-attention among the 7 tokens', tuple(q.shape)))
        q = L.layer_norm2(q + attend(L.cross_attn_token_to_image, q + tokens, keys + kpe, keys))
        trace.append((f'layer {i + 1}, step 2: tokens attend to the 4,096 image vectors', tuple(q.shape)))
        q = L.layer_norm3(q + L.mlp(q))
        trace.append((f'layer {i + 1}, step 3: MLP on each token', tuple(q.shape)))
        keys = L.layer_norm4(keys + attend(L.cross_attn_image_to_token, keys + kpe, q + tokens, q))
        trace.append((f'layer {i + 1}, step 4: image vectors attend to the tokens', tuple(keys.shape)))
    fin, Afin = attend(tr.final_attn_token_to_image, q + tokens, keys + kpe, keys, keep=True)
    q = tr.layer_norm_final_attn(q + fin)
    trace.append(('final: tokens attend to the image once more', tuple(q.shape)))
    img = keys.transpose(1, 2).reshape(1, 256, 64, 64)
    up1 = dec.upscale_conv1(img)
    up1a = dec.activation(dec.upscale_layer_norm(up1))
    up2 = dec.activation(dec.upscale_conv2(up1a))
    trace.append(('upscale: transposed conv 2x2, stride 2', tuple(up1.shape)))
    trace.append(('upscale: transposed conv 2x2, stride 2', tuple(up2.shape)))
    hyper = torch.stack([dec.output_hypernetworks_mlps[i](q[:, 1 + i]) for i in range(4)], 1)   # 1 x 4 x 32
    trace.append(('4 mask tokens through their MLPs', tuple(hyper.shape)))
    masks = (hyper @ up2.flatten(2)).reshape(1, 4, 256, 256)
    trace.append(('dot product with every pixel vector -> 4 masks', tuple(masks.shape)))
    iou = dec.iou_prediction_head(q[:, 0])
    trace.append(('IoU token through the IoU head -> 4 scores', tuple(iou.shape)))
    model_out = sam(image_embeddings=emb, input_points=pts, input_labels=labs, multimask_output=True)
for s, t in trace:
    log(f'  {s:<58} {str(t)}')
d_mask = float((masks[0, 1:].cpu() - model_out.pred_masks[0, 0].cpu()).abs().max())
d_iou = float((iou[0, 1:].cpu() - model_out.iou_scores[0, 0].cpu()).abs().max())
log(f'our hand-written decoder vs the model: masks max difference {d_mask:.2e}, scores max difference {d_iou:.2e}')
R['decoder_trace'] = [[s, list(t)] for s, t in trace]
R['decoder_check'] = dict(mask=d_mask, iou=d_iou)

# the dynamic dot product at three pixels
lowres = lambda xy: (int(xy[0] * SC / 4), int(xy[1] * SC / 4))
log('\nthe dynamic dot product: mask logit at pixel = (32 numbers from the mask token) . (32 numbers of the upscaled pixel)')
R['dot'] = {}
for name, xy in [('ear', EAR), ('cat body', BODY), ('blanket', SOFA)]:
    gx, gy = lowres(xy)
    f = up2[0, :, gy, gx].cpu()
    row = {}
    for k in [1, 3]:
        wv = hyper[0, k].cpu()
        prod = wv * f
        logit = float(prod.sum())
        row[f'mask{k}'] = dict(logit=logit, prob=1 / (1 + math.exp(-logit)))
        if name == 'ear' and k == 1:
            top = torch.argsort(prod.abs(), descending=True)[:3].tolist()
            log(f'  mask token 1, first 4 numbers w = ' + ', '.join(f'{v:+.3f}' for v in wv[:4].tolist())
                + f';  pixel at the ear, first 4 numbers f = ' + ', '.join(f'{v:+.3f}' for v in f[:4].tolist()))
            log('  the 3 largest products w_i f_i: ' + ', '.join(f'i={i}: {wv[i]:+.3f} x {f[i]:+.3f} = {prod[i]:+.3f}' for i in top))
            R['dot']['example'] = dict(w4=wv[:4].tolist(), f4=f[:4].tolist(), top=[[i, float(wv[i]), float(f[i]), float(prod[i])] for i in top])
    R['dot'][name] = dict(pixel=list(xy), cell=[gx, gy], **row)
    log(f'  {name:<9} photo {str(xy):<11} -> 256 grid ({gx:>3}, {gy:>3}):  mask 1 (ear) logit {row["mask1"]["logit"]:+7.2f} -> p {row["mask1"]["prob"]:.3f};'
        f'   mask 3 (cat) logit {row["mask3"]["logit"]:+7.2f} -> p {row["mask3"]["prob"]:.3f}')

# the four output tokens and their scores
iou4 = iou[0].cpu().numpy()
post = lambda m: proc.image_processor.post_process_masks(m.cpu(), p['original_sizes'], p['reshaped_input_sizes'], binarize=False)[0][0]
full4 = post(masks[:, None])                                                 # 4 x 480 x 640 logits
areas = [(full4[k] > 0).sum().item() for k in range(4)]
log('\nthe four mask tokens for the single ear click (token 0 is the single-mask output for several prompts):')
for k in range(4):
    log(f'  token {k}: predicted IoU {iou4[k]:.3f}   {areas[k]:>7,} pixels')
R['tokens4'] = [dict(token=k, score=float(iou4[k]), pixels=int(areas[k])) for k in range(4)]
np.save(os.path.join(HERE, 'results', 'part3_tokens4.npy'), (full4 > 0).numpy())

# where the mask tokens look: final token-to-image attention, mean over the 8 heads
att = Afin[0].mean(0).cpu()                                                  # 7 x 4096
R['attention'] = {}
for k in [1, 2, 3]:
    a = att[1 + k].reshape(64, 64)
    top = torch.topk(a.flatten(), 1).indices.item()
    R['attention'][f'mask{k}'] = dict(max=float(a.max()), argmax=[top % 64, top // 64])
    log(f'  mask token {k} attention: strongest cell ({top % 64}, {top // 64}) = pixel about ({(top % 64) * 16 / SC:.0f}, {(top // 64) * 16 / SC:.0f}) of the photo, weight {a.max():.4f}')
np.save(os.path.join(HERE, 'results', 'part3_attention.npy'), att.numpy())

# ------------------------------------------------------------------ 5. compute and time
log('\n== 5. compute (FLOPs) and time of each part (time: the fastest of several runs) ==')
from torch.utils.flop_counter import FlopCounterMode
# plain ('eager') attention, so that the counter sees every matrix product of the attention
sam_cpu = SamModel.from_pretrained('facebook/sam-vit-huge', attn_implementation='eager').eval().requires_grad_(False)
pxc, ptsc, labc = px.cpu(), pts.cpu(), labs.cpu()
with torch.no_grad():
    embc = sam_cpu.get_image_embeddings(pxc)


def flops(fn):
    fc = FlopCounterMode(display=False)
    with fc:
        fn()
    return fc.get_total_flops()


with torch.no_grad():
    f_enc = flops(lambda: sam_cpu.get_image_embeddings(pxc))
    f_dec = flops(lambda: sam_cpu(image_embeddings=embc, input_points=ptsc, input_labels=labc, multimask_output=True))
    hh = sam_cpu.vision_encoder.patch_embed(pxc) + sam_cpu.vision_encoder.pos_embed
    f_win = flops(lambda: sam_cpu.vision_encoder.layers[0](hh))
    f_glob = flops(lambda: sam_cpu.vision_encoder.layers[7](hh))
log(f'image encoder ViT-H: {f_enc / 1e12:.2f} TFLOPs per picture')
log(f'prompt encoder + mask decoder: {f_dec / 1e9:.2f} GFLOPs per prompt = {100 * f_dec / f_enc:.2f}% of the encoder')
log(f'encoder / (prompt encoder + decoder) = {f_enc / f_dec:,.0f}')
log(f'one windowed block {f_win / 1e9:.1f} GFLOPs, one global block {f_glob / 1e9:.1f} GFLOPs ({f_glob / f_win:.2f} x)')
att_g, att_w = 4 * 4096 ** 2 * 1280, 4 * nwin * 196 ** 2 * 1280
log(f'  of which attention scores and mixing (2 matrix products, 2 FLOPs per multiply-add): global 4 x 4096^2 x 1280 = {att_g / 1e9:.1f} GFLOPs, '
    f'windowed 4 x 25 x 196^2 x 1280 = {att_w / 1e9:.1f} GFLOPs')
R['flops_attention'] = dict(global_=att_g, window=att_w)
log(f'if all 32 blocks were global: {(32 * f_glob + (f_enc - 28 * f_win - 4 * f_glob)) / 1e12:.2f} TFLOPs ({(32 * f_glob + (f_enc - 28 * f_win - 4 * f_glob)) / f_enc:.2f} x)')
R['flops'] = dict(encoder=f_enc, decoder=f_dec, ratio=f_dec / f_enc, window_block=f_win, global_block=f_glob,
                  all_global=32 * f_glob + (f_enc - 28 * f_win - 4 * f_glob))


def timeit(fn, n, warm=2, dev='cpu'):
    """The fastest of n runs, in ms (the fastest run is the one least disturbed by other programs)."""
    for _ in range(warm):
        fn()
    best = float('inf')
    for _ in range(n):
        if dev == 'mps':
            torch.mps.synchronize()
        t = time.perf_counter()
        fn()
        if dev == 'mps':
            torch.mps.synchronize()
        best = min(best, time.perf_counter() - t)
    return 1000 * best


del sam_cpu
R['timing'] = {}
for dev in ['mps', 'cpu']:
    sam.to(dev)
    ve = sam.vision_encoder
    x = px.to(dev)
    with torch.no_grad():
        e = sam.get_image_embeddings(x)
        hh = ve.patch_embed(x) + ve.pos_embed
        P, Lb = pts.to(dev), labs.to(dev)
        sp_, de_ = sam.get_prompt_embeddings(input_points=P, input_labels=Lb)
        ipe = sam.get_image_wide_positional_embeddings()
        nrep = 3 if dev == 'cpu' else 10
        T = dict(encoder=timeit(lambda: sam.get_image_embeddings(x), nrep, 1, dev),
                 patch=timeit(lambda: ve.patch_embed(x), 20, 2, dev),
                 window_block=timeit(lambda: ve.layers[0](hh), 10, 2, dev),
                 global_block=timeit(lambda: ve.layers[7](hh), 10, 2, dev),
                 neck=timeit(lambda: ve.neck(hh), 20, 2, dev),
                 prompt=timeit(lambda: sam.get_prompt_embeddings(input_points=P, input_labels=Lb), 100, 5, dev),
                 decoder=timeit(lambda: sam.mask_decoder(e, ipe, sp_, de_, True), 100, 5, dev),
                 both=timeit(lambda: sam(image_embeddings=e, input_points=P, input_labels=Lb, multimask_output=True), 100, 5, dev))
    R['timing'][dev] = T
    log(f'{dev.upper():<4} image encoder {T["encoder"]:7.1f} ms  (patch embed {T["patch"]:.1f}, windowed block {T["window_block"]:.1f}, '
        f'global block {T["global_block"]:.1f}, neck {T["neck"]:.1f})   prompt encoder {T["prompt"]:.2f} ms   mask decoder {T["decoder"]:.2f} ms   '
        f'both {T["both"]:.2f} ms')
sam.to(DEV)

# ------------------------------------------------------------------ 6. the losses by hand
log('\n== 6. the training loss by hand, on real SAM outputs ==')


def predict(points=None, labels=None, box=None, mask_in=None, multimask=True):
    kw = {}
    if points is not None:
        kw.update(input_points=[[points]], input_labels=[[labels]])
    if box is not None:
        kw['input_boxes'] = [[box]]
    q_ = proc(images=image, return_tensors='pt', **kw)
    a = {k: (q_[k].float() if k != 'input_labels' else q_[k]).to(DEV) for k in ('input_points', 'input_labels', 'input_boxes') if k in q_}
    if mask_in is not None:
        a['input_masks'] = mask_in.to(DEV)
    with torch.no_grad():
        o = sam(image_embeddings=emb, multimask_output=multimask, **a)
    full = proc.image_processor.post_process_masks(o.pred_masks.cpu(), q_['original_sizes'], q_['reshaped_input_sizes'], binarize=False)[0][0]
    return full, o.iou_scores[0, 0].cpu(), o.pred_masks[0, 0].cpu()


ear_logits, ear_scores, _ = predict([list(EAR)], [1])                         # 3 x 480 x 640
cat_ref = (predict(box=CAT_BOX, multimask=False)[0][0] > 0)
ys, xs = np.where(ear_logits[0].numpy() > 0)
ear_box = [int(xs.min()) - 4, int(ys.min()) - 4, int(xs.max()) + 4, int(ys.max()) + 4]
ear_ref = (predict(box=ear_box, multimask=False)[0][0] > 0)
log(f'stand-in references (we have no hand-drawn mask): "the cat" = SAM\'s mask for the box {CAT_BOX}: {int(cat_ref.sum()):,} pixels;')
log(f'  "the ear" = SAM\'s mask for the box {ear_box}: {int(ear_ref.sum()):,} pixels')
R['refs'] = dict(cat_box=CAT_BOX, cat_pixels=int(cat_ref.sum()), ear_box=ear_box, ear_pixels=int(ear_ref.sum()))
ALPHA, GAMMA = 0.25, 2.0


def focal(logit, y):
    pr = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(logit, y, reduction='none')
    pt = pr * y + (1 - pr) * (1 - y)
    at = ALPHA * y + (1 - ALPHA) * (1 - y)
    return float((at * (1 - pt) ** GAMMA * ce).mean()), float(ce.mean())


def dice(logit, y):
    pr = torch.sigmoid(logit)
    return 1 - float((2 * (pr * y).sum() + 1) / (pr.sum() + y.sum() + 1))


def hard_iou(logit, y):
    m = logit > 0
    yb = y > 0.5
    return float((m & yb).sum() / (m | yb).sum())


R['loss'] = {}
for tname, ref in [('the ear', ear_ref), ('the cat', cat_ref)]:
    y = ref.float()
    rows = []
    log(f'target: {tname}')
    log(f'  {"mask":<6}{"pixels":>8}{"focal":>9}{"20 x focal":>12}{"dice":>8}{"mask loss":>11}{"true IoU":>10}{"pred. IoU":>11}{"(pred - true)^2":>17}{"CE (for comparison)":>21}')
    for k in range(3):
        fl, ce = focal(ear_logits[k], y)
        dl = dice(ear_logits[k], y)
        ti = hard_iou(ear_logits[k], y)
        pi = float(ear_scores[k])
        rows.append(dict(pixels=int((ear_logits[k] > 0).sum()), focal=fl, dice=dl, mask_loss=20 * fl + dl, true_iou=ti, pred_iou=pi,
                         iou_se=(pi - ti) ** 2, ce=ce))
        r = rows[-1]
        log(f'  {k + 1:<6}{r["pixels"]:>8,}{fl:>9.4f}{20 * fl:>12.4f}{dl:>8.4f}{r["mask_loss"]:>11.4f}{ti:>10.3f}{pi:>11.3f}{r["iou_se"]:>17.4f}{ce:>21.4f}')
    best = int(np.argmin([r['mask_loss'] for r in rows]))
    iou_term = float(np.mean([r['iou_se'] for r in rows]))
    total = rows[best]['mask_loss'] + iou_term
    log(f'  the minimum mask loss is mask {best + 1}: {rows[best]["mask_loss"]:.4f}. Only mask {best + 1} gets a gradient from the mask loss.')
    log(f'  IoU term (mean over the 3 masks): {iou_term:.4f};  total = {rows[best]["mask_loss"]:.4f} + 1.0 x {iou_term:.4f} = {total:.4f}')
    R['loss'][tname] = dict(rows=rows, best=best + 1, iou_term=iou_term, total=total)

# what a single output would learn: the average of the two answers
p_avg = 0.5 * torch.sigmoid(ear_logits[0]) + 0.5 * torch.sigmoid(ear_logits[2])
lg_avg = torch.log(p_avg.clamp(1e-6, 1 - 1e-6) / (1 - p_avg.clamp(1e-6, 1 - 1e-6)))
R['average'] = {}
log('one output that averages "ear" and "cat" (probability 0.5 on the cat body):')
for tname, ref in [('the ear', ear_ref), ('the cat', cat_ref)]:
    y = ref.float()
    ml = 20 * focal(lg_avg, y)[0] + dice(lg_avg, y)
    R['average'][tname] = ml
    log(f'  against {tname}: mask loss {ml:.4f}  (best of the three masks: {R["loss"][tname]["rows"][R["loss"][tname]["best"] - 1]["mask_loss"]:.4f})')
log(f'  ratio to the best mask: ear {R["average"]["the ear"] / R["loss"]["the ear"]["rows"][R["loss"]["the ear"]["best"] - 1]["mask_loss"]:.0f} x, '
    f'cat {R["average"]["the cat"] / R["loss"]["the cat"]["rows"][R["loss"]["the cat"]["best"] - 1]["mask_loss"]:.1f} x')
half = ((p_avg > 0.4) & (p_avg < 0.6)).sum().item()
log(f'  pixels with a probability between 0.4 and 0.6: {half:,} (a blurred, unsure mask)')
R['average']['unsure_pixels'] = half
np.save(os.path.join(HERE, 'results', 'part3_avg_prob.npy'), p_avg.numpy())
np.save(os.path.join(HERE, 'results', 'part3_refs.npy'), np.stack([ear_ref.numpy(), cat_ref.numpy()]))

# tiny worked examples for the article
log('\ntiny worked examples:')
for pr_, y_ in [(0.9, 1), (0.3, 1), (0.6, 0)]:
    pt_ = pr_ if y_ == 1 else 1 - pr_
    ce_ = -math.log(pt_)
    a_ = ALPHA if y_ == 1 else 1 - ALPHA
    log(f'  p = {pr_}, y = {y_}: p_t = {pt_:.1f}; CE = -log {pt_:.1f} = {ce_:.4f}; (1 - p_t)^2 = {(1 - pt_) ** 2:.4f}; '
        f'focal (alpha_t = {a_}) = {a_} x {(1 - pt_) ** 2:.4f} x {ce_:.4f} = {a_ * (1 - pt_) ** 2 * ce_:.5f}')
ce_e, ce_h = -math.log(0.9), -math.log(0.3)
fl_e, fl_h = 0.25 * 0.01 * ce_e, 0.25 * 0.49 * ce_h
log(f'  hard (p = 0.3) / easy (p = 0.9): cross-entropy {ce_h / ce_e:.1f} x, focal {fl_h / fl_e:.0f} x')
pp, yy = [0.9, 0.8, 0.3, 0.1], [1, 1, 1, 0]
num, den = 2 * sum(a * b for a, b in zip(pp, yy)), sum(pp) + sum(yy)
log(f'  dice on 4 pixels p = {pp}, y = {yy}: 2 x sum(p y) = {num:.1f}; sum(p) + sum(y) = {sum(pp):.1f} + {sum(yy)} = {den:.1f}; '
    f'dice = {num / den:.4f}; loss = {1 - num / den:.4f}')
R['tiny'] = dict(dice_num=num, dice_den=den, dice=num / den)
log(f'  IoU head: predicted 0.90, true 0.80 -> (0.90 - 0.80)^2 = {(0.9 - 0.8) ** 2:.4f}')
# attention with one query, two keys, d = 2
qv, k1, k2, v1, v2 = np.array([1., 0.]), np.array([2., 0.]), np.array([0., 2.]), np.array([10., 0.]), np.array([0., 10.])
sc_ = np.array([qv @ k1, qv @ k2]) / math.sqrt(2)
wt_ = np.exp(sc_) / np.exp(sc_).sum()
log(f'  attention: q = (1, 0), k1 = (2, 0), k2 = (0, 2): scores q.k / sqrt(2) = {sc_[0]:.3f}, {sc_[1]:.3f}; '
    f'softmax = e^{sc_[0]:.3f} / (e^{sc_[0]:.3f} + e^0) = {wt_[0]:.3f}, {wt_[1]:.3f}; output = {wt_[0]:.3f} x (10, 0) + {wt_[1]:.3f} x (0, 10) = '
    f'({(wt_[0] * v1 + wt_[1] * v2)[0]:.2f}, {(wt_[0] * v1 + wt_[1] * v2)[1]:.2f})')
R['tiny_attention'] = dict(scores=sc_.tolist(), weights=wt_.tolist())
# a one-frequency Fourier encoding
for xx in [0.10, 0.12, 0.30]:
    a_ = 2 * math.pi * 3 * xx
    log(f'  Fourier, g = 3: x = {xx:.2f} -> angle 2 pi x 3 x {xx:.2f} = {a_:.3f} -> (sin, cos) = ({math.sin(a_):+.3f}, {math.cos(a_):+.3f})')
# a dot product with 3 numbers
for f_ in [[0.3, 0.2, 0.4], [-0.1, 0.5, 0.0]]:
    lg_ = 2 * f_[0] - 1 * f_[1] + 0.5 * f_[2]
    log(f'  dot product: w = (2, -1, 0.5), f = {tuple(f_)}: logit = {lg_:+.2f} -> sigmoid = {1 / (1 + math.exp(-lg_)):.3f}')

# ------------------------------------------------------------------ 7. the interactive training game, 11 rounds
log('\n== 7. the simulated interactive game of training, played with the released model ==')
rng = np.random.default_rng(3)
target = cat_ref.numpy()
pts_l, lab_l, prev, rounds = [], [], None, []
no_new = {rng.integers(2, 10), 11}                                           # one round without a point among 2..9, one at the end
log(f'target: the left cat ({int(target.sum()):,} pixels). Rounds without a new point: {sorted(int(r) for r in no_new)}')
for r in range(1, 12):
    if r == 1:
        ys, xs = np.where(target)
        i = rng.integers(len(xs))
        pts_l, lab_l = [[int(xs[i]), int(ys[i])]], [1]
        what = f'first point ({xs[i]}, {ys[i]}), foreground'
    elif r in no_new:
        what = 'no new point: only the previous mask'
    else:
        fn_ = target & ~cur
        fp_ = cur & ~target
        err = fn_ | fp_
        ys, xs = np.where(err)
        i = rng.integers(len(xs))
        fg_ = bool(fn_[ys[i], xs[i]])
        pts_l.append([int(xs[i]), int(ys[i])]); lab_l.append(1 if fg_ else 0)
        what = f'point ({xs[i]}, {ys[i]}) from the error region ({int(err.sum()):,} px), {"foreground (missed pixel)" if fg_ else "background (extra pixel)"}'
    multi = len(pts_l) == 1
    full, sc, low = predict(pts_l, lab_l, mask_in=prev, multimask=multi)
    k = int(torch.argmax(sc)) if multi else 0
    cur = (full[k] > 0).numpy()
    prev = low[k][None, None]                                                # the unthresholded logits go to the next round
    iu = float((cur & target).sum() / (cur | target).sum())
    rounds.append(dict(round=r, what=what, points=len(pts_l), iou=iu, score=float(sc[k]), pixels=int(cur.sum())))
    log(f'  round {r:>2}: {what:<78} points {len(pts_l)}  IoU {iu:.3f}  predicted {float(sc[k]):.3f}')
    if r in (1, 2, 11):
        np.save(os.path.join(HERE, 'results', f'part3_round{r}.npy'), cur)
R['rounds'] = rounds
R['round_points'] = [pts_l, lab_l]

# box prompt with the training noise: std 10% of the side, at most 20 pixels
bw, bh = CAT_BOX[2] - CAT_BOX[0], CAT_BOX[3] - CAT_BOX[1]
sd = [min(0.1 * bw, 20), min(0.1 * bh, 20)]
noise = rng.normal(0, 1, 4) * np.array([sd[0], sd[1], sd[0], sd[1]])
nbox = [int(round(a + b)) for a, b in zip(CAT_BOX, noise)]
full, sc, _ = predict(box=nbox, multimask=False)
iu = float(((full[0] > 0).numpy() & target).sum() / ((full[0] > 0).numpy() | target).sum())
log(f'noisy box for training: sides {bw} x {bh} px -> noise std {sd[0]:.1f}, {sd[1]:.1f} px (10% of the side, at most 20); '
    f'box {CAT_BOX} -> {nbox}; IoU with the target {iu:.3f}')
R['noisy_box'] = dict(box=CAT_BOX, sd=sd, noisy=nbox, iou=iu)

# ------------------------------------------------------------------ 8. the training recipe in numbers
log('\n== 8. the training recipe in numbers (from Appendix A) ==')
it, bs, imgs = 90_000, 256, 11_000_000
log(f'{it:,} iterations x {bs} images = {it * bs:,} pictures seen = {it * bs / imgs:.2f} passes over 11M pictures')
log(f'masks: up to 64 per GPU, 1 picture per GPU (256 pictures on 256 GPUs) -> up to {64 * 11} decoder runs per picture per iteration (64 masks x 11 rounds)')


def lr_at(t):
    base = 8e-4
    if t < 250:
        return base * (t + 1) / 250
    return base * (0.1 if t >= 60_000 else 1) * (0.1 if t >= 86_666 else 1)


sched = [(t, lr_at(t)) for t in [0, 125, 249, 1000, 59_999, 60_000, 86_665, 86_666, 89_999]]
log('learning rate: ' + '; '.join(f'it {t:,}: {v:.1e}' for t, v in sched))
R['lr'] = [[t, v] for t, v in sched]
R['lr_curve'] = [[t, lr_at(t)] for t in list(range(0, 300, 10)) + list(range(300, 90_001, 300))]
ld = 0.8
log(f'layer-wise decay {ld}: with the usual rule (rate x 0.8^(33 - layer), layer 1..32 = blocks, 0 = patch embedding):')
R['layer_decay'] = []
for name_, layer in [('mask decoder and neck', 33), ('block 32 (last)', 32), ('block 16', 16), ('block 1 (first)', 1), ('patch embedding', 0)]:
    v = 8e-4 * ld ** (33 - layer)
    R['layer_decay'].append([name_, v])
    log(f'  {name_:<22} 8e-4 x 0.8^{33 - layer:<2} = {v:.2e}')
log(f'  decoder rate / first-block rate = {1 / ld ** 32:,.0f}')

save('part3', R)

# ------------------------------------------------------------------ 9. pictures for the article


def font(size):
    for f_ in ['/System/Library/Fonts/Helvetica.ttc', '/System/Library/Fonts/Menlo.ttc']:
        if os.path.exists(f_):
            return ImageFont.truetype(f_, size)
    return ImageFont.load_default()


COLS = [(255, 196, 0), (64, 200, 255), (60, 90, 255), (255, 120, 60)]


def overlay(mask, col, a=0.55, base=None):
    arr = np.array(base if base is not None else image).astype(float)
    arr[mask] = (1 - a) * arr[mask] + a * np.array(col)
    e = arr.astype('uint8')
    edge = mask ^ np.pad(mask, 1, mode='edge')[:-2, 1:-1] | mask ^ np.pad(mask, 1, mode='edge')[1:-1, :-2]
    e[edge] = (255, 255, 255)
    return Image.fromarray(e)


def dot(im, x, y, r=7, col=(40, 220, 90)):
    d = ImageDraw.Draw(im)
    d.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2], fill=(255, 255, 255))
    d.ellipse([x - r, y - r, x + r, y + r], fill=col)
    return im


def panels(tiles, labels, path, scale=1.0):
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


# the four output tokens
m4 = np.load(os.path.join(HERE, 'results', 'part3_tokens4.npy'))
nm = ['token 0 (single-mask output)', 'token 1', 'token 2', 'token 3']
panels([dot(overlay(m4[k], COLS[k]), *EAR, 6) for k in range(4)],
       [(nm[k], f'score {R["tokens4"][k]["score"]:.3f}  ·  {R["tokens4"][k]["pixels"]:,} px') for k in range(4)],
       os.path.join(IMG, 'p3-four-tokens.png'), scale=0.75)

# attention of the three mask tokens, as a heat map over the photo
A = np.load(os.path.join(HERE, 'results', 'part3_attention.npy'))
tiles = []
for k in [1, 2, 3]:
    a = A[1 + k].reshape(64, 64)[:48, :64]                                   # the photo covers 64 x 48 cells; the rest is padding
    a = np.sqrt(a / a.max())                                                # square root: so the weak, wide attention shows
    heat = np.array(Image.fromarray((a * 255).astype('uint8')).resize((W0, H0), Image.BILINEAR)).astype(float) / 255
    base = np.array(image).astype(float) * 0.35
    col = np.array([255, 210, 60])
    arr = base + heat[..., None] * col * 0.85
    tiles.append(dot(Image.fromarray(arr.clip(0, 255).astype('uint8')), *EAR, 6))
panels(tiles, [(f'mask token {k}', f'mask: {R["tokens4"][k]["pixels"]:,} px') for k in [1, 2, 3]],
       os.path.join(IMG, 'p3-token-attention.png'))

# the averaged answer that one output would learn
pa = np.load(os.path.join(HERE, 'results', 'part3_avg_prob.npy'))
arr = np.array(image).astype(float) * 0.35 + pa[..., None] * np.array([90, 170, 255]) * 0.8
refs = np.load(os.path.join(HERE, 'results', 'part3_refs.npy'))
panels([dot(overlay(refs[0], COLS[0]), *EAR, 6), dot(overlay(refs[1], COLS[2]), *EAR, 6), dot(Image.fromarray(arr.clip(0, 255).astype('uint8')), *EAR, 6)],
       [('valid answer 1: the ear', f'{R["refs"]["ear_pixels"]:,} px'), ('valid answer 2: the cat', f'{R["refs"]["cat_pixels"]:,} px'),
        ('one output: the average', f'{R["average"]["unsure_pixels"]:,} px near p = 0.5')],
       os.path.join(IMG, 'p3-average.png'))

# rounds 1, 2 and 11 of the game
tiles, labs_ = [], []
for r in (1, 2, 11):
    cm = np.load(os.path.join(HERE, 'results', f'part3_round{r}.npy'))
    t = overlay(cm, COLS[1])
    n = R['rounds'][r - 1]['points']
    for (x, y), l in zip(R['round_points'][0][:n], R['round_points'][1][:n]):
        dot(t, x, y, 6, (40, 220, 90) if l == 1 else (240, 70, 70))
    tiles.append(t)
    labs_.append((f'round {r}: {n} point' + ('s' if n > 1 else ''), f'IoU with the target {R["rounds"][r - 1]["iou"]:.3f}'))
panels(tiles, labs_, os.path.join(IMG, 'p3-rounds.png'))
