"""Part 4: read the training curves that vit_part4_scratch.py saved (results/part4_scratch.json) and print what they
show: for each data size and model, the smoothed training loss and the validation accuracy at a few points of training,
the step where validation was best, and the training loss at the end. Runs no model.
Writes results/part4_scratch_curves_stdout.txt."""
import json, os
from common import Log

log = Log('part4_scratch_curves')
R = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'part4_scratch.json')))
log('== the training curves of vit_part4_scratch.py: validation accuracy (%) and smoothed training loss ==')
for n, r in R['runs'].items():
    for m, lab in [('vit', 'ViT'), ('resnet', 'ResNet')]:
        c = r[f'{m}_curve']
        best = max(c, key=lambda x: x[1])
        pts = '  '.join(f'step {s}: {a:.1f}% / loss {l:.3f}' for s, a, l in c if s in (500, 1500, 2500))
        log(f'  {int(n):>6,} images  {lab:<6} {pts}   best {best[1]:.1f}% at step {best[0]}')
