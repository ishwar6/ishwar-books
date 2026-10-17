"""Precompute what the demo shows for the curated pictures, from the float32 PyTorch model (vitwrap.ViTSteps), and the
per-model files the browser needs. Run pictures.py first.  Usage: python precompute.py [model ...]  (default vit-b16 deit-ti16)

public/demos/vit/<model>/
  meta.json        D, heads, layers, the 1,000 ImageNet labels, the [class] token's own numbers
  possim.bin       float16 (196, 196): cosine similarity of the learned position embeddings of every pair of patches (Fig. 7 centre)
  pos.bin          float16 (197, D): the position embeddings E_pos (to show z_0 = [x_class; x_p E] + E_pos)
  (ViT-B/16 only, the curated pictures:)
  <slug>.json      top-5 for four runs (normal, patches shuffled, no position embeddings, both), mean attention distance
                   per layer and head (Fig. 7 right), attention rollout (Fig. 6), the shuffle order used
  <slug>.bin       float16: patch embeddings (196, D), then the [class] row of every attention map (L, H, 197)
  <slug>.att.bin   gzip-compressed (named .bin so no server marks it Content-Encoding: gzip): for every attention row,
                   float16 weight on [class] (L, H, 197), float16 largest weight on a patch (L, H, 197), then uint8
                   (L, H, 197, 196) = round(63 * weight on each patch / that largest weight). Every patch weight is
                   within 1/126 of its row's largest patch weight; 2 MB per picture instead of 22 MB.
public/demos/vit/pictures.json   the curated pictures, their source and licence
Also writes results/precompute_<model>.json with the top-5 lists (the numbers the page shows)."""
import gzip, json, os, sys
import numpy as np
import torch
from PIL import Image
from vitwrap import ViTSteps
from pictures import PICTURES, LICENCES

HERE = os.path.dirname(os.path.abspath(__file__))
PUB = os.path.join(HERE, '..', '..', '..', '..', 'public', 'demos', 'vit')
MODELS = {'vit-b16': 'google/vit-base-patch16-224', 'deit-ti16': 'facebook/deit-tiny-patch16-224'}
G, P = 14, 16                                    # 14 x 14 patches of 16 x 16 pixels
LEVELS = 63
SHUFFLE = np.random.default_rng(0).permutation(G * G)


def picture(slug):
    a = np.asarray(Image.open(os.path.join(PUB, 'img', f'{slug}.png')).convert('RGB'), dtype=np.float32)
    return a.transpose(2, 0, 1)[None].copy()


def shuffle_tiles(x, perm):
    """Slot k of the new picture gets patch perm[k] of the old one."""
    t = x.reshape(1, 3, G, P, G, P).transpose(0, 1, 2, 4, 3, 5).reshape(1, 3, G * G, P, P)
    t = t[:, :, perm]
    return t.reshape(1, 3, G, G, P, P).transpose(0, 1, 2, 4, 3, 5).reshape(1, 3, G * P, G * P).copy()


def centres():
    r, c = np.divmod(np.arange(G * G), G)
    return np.stack([r, c], 1) * P + P / 2       # patch centres in pixels


DIST = np.linalg.norm(centres()[:, None] - centres()[None], axis=-1)   # (196, 196) pixels


def mean_distance(att):
    """att (L, H, 197, 197). For each query patch: the attention-weighted mean distance in pixels to the key patches
    (the [class] token is left out and the weights over the 196 patches are renormalised to sum to 1); then the
    mean over the 196 query patches. Gives (L, H)."""
    a = att[:, :, 1:, 1:]
    a = a / a.sum(-1, keepdims=True)
    return (a * DIST).sum(-1).mean(-1)


def rollout(att):
    """Attention rollout (Abnar and Zuidema 2020), as the paper's Figure 6: average the heads, add the identity for the
    residual connection, renormalise the rows, multiply the layers; the [class] row over the 196 patches."""
    n = att.shape[-1]
    R = np.eye(n)
    for A in att.mean(1):
        A = 0.5 * A + 0.5 * np.eye(n)
        R = A @ R
    v = R[0, 1:]
    return v / v.sum()


def top5(logits, labels):
    p = torch.softmax(torch.from_numpy(logits), -1).numpy()
    i = np.argsort(-p)[:5]
    return [[int(k), labels[k], round(float(p[k]), 4)] for k in i]


def run(m, x, pos=1.0):
    with torch.no_grad():
        e, z0, att, lg = m(torch.from_numpy(x), torch.tensor([pos]))
    return e[0].numpy(), z0[0].numpy(), att.numpy().astype(np.float64), lg[0].numpy()


if __name__ == '__main__':
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    for key in sys.argv[1:] or list(MODELS):
        m = ViTSteps(MODELS[key]).eval()
        out = os.path.join(PUB, key)
        os.makedirs(out, exist_ok=True)
        labels = [m.m.config.id2label[i] for i in range(1000)]
        pos = m.m.vit.embeddings.position_embeddings[0].detach().numpy()          # (197, D)
        pp = pos[1:] / np.linalg.norm(pos[1:], axis=-1, keepdims=True)
        pos.astype(np.float16).tofile(os.path.join(out, 'pos.bin'))
        (pp @ pp.T).astype(np.float16).tofile(os.path.join(out, 'possim.bin'))
        cls = m.m.vit.embeddings.cls_token[0, 0].detach().numpy()
        json.dump(dict(model=MODELS[key], D=m.D, heads=m.H, layers=m.L, patch=P, grid=G,
                       params=sum(p.numel() for p in m.m.parameters()), labels=labels,
                       cls=[round(float(v), 5) for v in cls], att_levels=LEVELS), open(os.path.join(out, 'meta.json'), 'w'))
        R = {}
        for slug, iid, lic, flickr, name in (PICTURES if key == 'vit-b16' else []):   # the live model needs only meta and positions
            x = picture(slug)
            e, z0, att, lg = run(m, x)
            res = dict(slug=slug, name=name, coco=iid, licence=LICENCES[lic][0], flickr=flickr,
                       top5=top5(lg, labels),
                       shuffled=top5(run(m, shuffle_tiles(x, SHUFFLE))[3], labels),
                       nopos=top5(run(m, x, 0.0)[3], labels),
                       nopos_shuffled=top5(run(m, shuffle_tiles(x, SHUFFLE), 0.0)[3], labels),
                       shuffle=SHUFFLE.tolist(),
                       distance=np.round(mean_distance(att), 2).tolist(),
                       rollout=np.round(rollout(att), 6).tolist())
            lg_ns = run(m, x, 0.0)[3]
            lg_nss = run(m, shuffle_tiles(x, SHUFFLE), 0.0)[3]
            res['nopos_shuffle_max_abs_diff'] = float(np.abs(lg_ns - lg_nss).max())
            json.dump(res, open(os.path.join(out, f'{slug}.json'), 'w'))
            np.concatenate([e.reshape(-1), att[:, :, 0, :].reshape(-1)]).astype(np.float16).tofile(os.path.join(out, f'{slug}.bin'))
            cw = att[..., 0]                                                 # weight on the [class] token
            mx = att[..., 1:].max(-1)                                         # largest weight on a patch
            q = np.round(LEVELS * att[..., 1:] / mx[..., None]).astype(np.uint8)
            with gzip.open(os.path.join(out, f'{slug}.att.bin'), 'wb', compresslevel=9) as f:
                f.write(cw.astype(np.float16).tobytes() + mx.astype(np.float16).tobytes() + q.tobytes())
            R[slug] = {k: res[k] for k in ('top5', 'shuffled', 'nopos', 'nopos_shuffled', 'nopos_shuffle_max_abs_diff')}
            print(f'{key} {slug:11s} {res["top5"][0][1][:24]:24s} {res["top5"][0][2]:.3f} | shuffled: {res["shuffled"][0][1][:20]:20s} {res["shuffled"][0][2]:.3f}'
                  f' | no pos: {res["nopos"][0][1][:20]:20s} {res["nopos"][0][2]:.3f} | no pos + shuffled differs by {res["nopos_shuffle_max_abs_diff"]:.1e}'
                  f' | att.bin {os.path.getsize(os.path.join(out, f"{slug}.att.bin")) / 1e6:.2f} MB')
        if R:
            json.dump([dict(slug=s, name=n, coco=i, licence=LICENCES[l][0], licence_url=LICENCES[l][1], flickr=f)
                       for s, i, l, f, n in PICTURES], open(os.path.join(PUB, 'pictures.json'), 'w'), indent=1)
            json.dump(R, open(os.path.join(HERE, 'results', f'precompute_{key}.json'), 'w'), indent=1)
