"""Python reference masks for the browser check of the demo (/papers/sam/demo).

For a fixed list of prompts on three photos, run transformers' SamModel (facebook/sam-vit-huge) on the fp32 image
embedding, upscale with the processor's post_process_masks (as in Parts 1 to 4), and save the three masks (run-length
encoded), the predicted IoUs and SAM's first choice. check_browser.mjs replays the same prompts in the browser
(onnxruntime-web, float16 embedding) and reports the mask IoU and the score difference of each case.
It also measures what the float16 storage of the embedding changes, in PyTorch.

Writes code/papers/sam/demo/results/reference.json and reference_stdout.txt."""
import os, json
import numpy as np
import torch
from transformers import SamModel, SamProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
PUB = os.path.join(HERE, '..', '..', '..', '..', 'public', 'demos', 'sam')
RES = os.path.join(HERE, 'results')
CACHE = os.path.expanduser('~/.cache/papers/sam_demo')

# each case: photo, list of rounds; a round is points [[x, y, label]], an optional box [x0, y0, x1, y1] in photo pixels,
# and whether the previous round's first-ranked low-res logits go in as the mask prompt
CASES = [
    ('cats', 'one click on the ear', [dict(pts=[[275, 62, 1]])]),
    ('cats', 'ear, body, and a background click on the sofa', [dict(pts=[[275, 62, 1], [120, 220, 1], [500, 420, 0]])]),
    ('cats', 'a box around the right cat', [dict(box=[255, 20, 640, 360])]),
    ('cats', 'ear, then body with the previous mask as prompt', [dict(pts=[[275, 62, 1]]),
                                                               dict(pts=[[275, 62, 1], [120, 220, 1]], prev=True)]),
    ('car', 'one click on the hub of the back wheel', [dict(pts=[[320, 420, 1]])]),
    ('car', 'a box around the front wheel', [dict(box=[795, 315, 930, 445])]),
    ('bike', 'one click on the front tyre', [dict(pts=[[470, 250, 1]])]),
    ('bike', 'one click on the frame, one background click on the tree', [dict(pts=[[300, 250, 1], [140, 120, 0]])]),
]


def rle(mask):
    flat = mask.reshape(-1).astype(np.uint8)
    change = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    runs = np.diff(bounds).tolist()
    return runs if flat[0] == 0 else [0] + runs


def main():
    proc = SamProcessor.from_pretrained('facebook/sam-vit-huge')
    sam = SamModel.from_pretrained('facebook/sam-vit-huge').eval()
    meta = {p['key']: p for p in json.load(open(os.path.join(PUB, 'photos.json')))['photos']}
    out, lines = [], []
    log = lambda s: (print(s), lines.append(s))
    log('reference: transformers SamModel facebook/sam-vit-huge, CPU, fp32 embedding; masks = logits > 0 after '
        'post_process_masks')
    worst16 = (1.0, 0.0)
    for key, name, rounds in CASES:
        w, h = meta[key]['width'], meta[key]['height']
        s = 1024 / max(w, h)
        emb32 = torch.from_numpy(np.load(os.path.join(CACHE, f'emb_{key}_fp32.npy')))
        emb16 = torch.from_numpy(np.fromfile(os.path.join(PUB, 'emb', f'{key}.bin'), '<f2').astype(np.float32)
                                 .reshape(1, 256, 64, 64))
        res = {}
        for tag, emb in (('fp32', emb32), ('fp16', emb16)):
            prev = None
            for r in rounds:
                kw = {}
                if 'pts' in r:
                    kw['input_points'] = torch.tensor([[[[x * s, y * s] for x, y, _ in r['pts']]]], dtype=torch.float32)
                    kw['input_labels'] = torch.tensor([[[l for _, _, l in r['pts']]]])
                if 'box' in r:
                    kw['input_boxes'] = torch.tensor([[[c * s for c in r['box']]]], dtype=torch.float32)
                if r.get('prev'):
                    kw['input_masks'] = prev
                with torch.no_grad():
                    o = sam(image_embeddings=emb, multimask_output=True, **kw)
                scores = o.iou_scores[0, 0]
                prev = o.pred_masks[0, 0, scores.argmax()][None, None]
            masks = proc.image_processor.post_process_masks(o.pred_masks, [[h, w]], [[round(h * s), round(w * s)]])[0][0]
            res[tag] = (masks.numpy(), scores.numpy())
        m32, s32 = res['fp32']
        m16, s16 = res['fp16']
        ious16 = [float((a & b).sum() / max((a | b).sum(), 1)) for a, b in zip(m32, m16)]
        worst16 = (min(worst16[0], min(ious16)), max(worst16[1], float(np.abs(s32 - s16).max())))
        top = int(s32.argmax())
        log(f'{key:<6} {name:<55} scores {", ".join(f"{v:.3f}" for v in s32)}  first: mask {top + 1}  '
            f'pixels {", ".join(f"{int(m.sum()):,}" for m in m32)}')
        log(f'       fp16 embedding vs fp32: mask IoU {", ".join(f"{v:.4f}" for v in ious16)}; '
            f'score diff {np.abs(s32 - s16).max():.1e}')
        out.append(dict(photo=key, name=name, rounds=rounds, scores=[float(v) for v in s32], top=top,
                        width=w, height=h, masks=[rle(m) for m in m32], pixels=[int(m.sum()) for m in m32]))
    log(f'float16 storage of the embedding, all cases: lowest mask IoU {worst16[0]:.4f}, largest score diff {worst16[1]:.1e}')
    json.dump(out, open(os.path.join(RES, 'reference.json'), 'w'))
    open(os.path.join(RES, 'reference_stdout.txt'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
