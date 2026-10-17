"""Compare the browser's live run (DeiT-Ti/16, float16 weights, ONNX Runtime Web) with the float32 PyTorch model on the
same 224 x 224 pictures: max |logit difference|, top-1 and top-5, attention rollout and mean attention distance.
The browser numbers come from a JSON that the page exposes (window.__vitDemo) and a playwright script saves.
Usage: python check_live.py live.json      Writes results/check_live.json."""
import json, os, sys
import numpy as np
import torch
from vitwrap import ViTSteps
from precompute import picture, rollout, mean_distance

HERE = os.path.dirname(os.path.abspath(__file__))
B = json.load(open(sys.argv[1]))
m = ViTSteps('facebook/deit-tiny-patch16-224').eval()
rows = []
for slug, r in B['res'].items():
    with torch.no_grad():
        _, _, att, lg = m(torch.from_numpy(picture(slug)), torch.ones(1))
    a, b = lg[0].numpy(), np.array(r['logits'], dtype=np.float32)
    att = att.numpy().astype(np.float64)
    ro, rb = rollout(att), np.array(r['rollout'])
    d, db = mean_distance(att), np.array(r['distance'])
    rows.append(dict(picture=slug, logit_max_abs_diff=float(np.abs(a - b).max()), same_top1=bool(a.argmax() == b.argmax()),
                     same_top5=list(np.argsort(-a)[:5]) == list(np.argsort(-b)[:5]),
                     top1=m.m.config.id2label[int(a.argmax())].split(',')[0], top1_prob=float(torch.softmax(lg[0], -1).max()),
                     rollout_max_abs_diff=float(np.abs(ro - rb).max()), rollout_max=float(ro.max()),
                     rollout_corr=float(np.corrcoef(ro, rb)[0, 1]),
                     distance_max_abs_diff_px=float(np.abs(d - db).max())))
    x = rows[-1]
    print(f'{slug:11s} top-1 {x["top1"][:22]:22s} {x["top1_prob"]:.3f}  logits max |diff| {x["logit_max_abs_diff"]:.4f}  same top-1 {x["same_top1"]}'
          f'  same top-5 {x["same_top5"]}  rollout max |diff| {x["rollout_max_abs_diff"]:.2e} (max {x["rollout_max"]:.4f})'
          f'  distance max |diff| {x["distance_max_abs_diff_px"]:.3f} px')
out = dict(pictures=rows, timing=B.get('timing'), click_to_first_live_ms=B.get('clickToFirstLive'),
           max_logit_diff=max(x['logit_max_abs_diff'] for x in rows), top1=sum(x['same_top1'] for x in rows),
           top5=sum(x['same_top5'] for x in rows), n=len(rows),
           max_rollout_diff=max(x['rollout_max_abs_diff'] for x in rows), max_distance_diff=max(x['distance_max_abs_diff_px'] for x in rows))
json.dump(out, open(os.path.join(HERE, 'results', 'check_live.json'), 'w'), indent=1)
print(f'all: max |logit diff| {out["max_logit_diff"]:.4f}, top-1 {out["top1"]}/{out["n"]}, top-5 {out["top5"]}/{out["n"]}, '
      f'rollout max |diff| {out["max_rollout_diff"]:.2e}, distance max |diff| {out["max_distance_diff"]:.3f} px')
