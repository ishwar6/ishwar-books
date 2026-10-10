"""Chapter 4: fit a Kaplan-style power law L(N) = (Nc / N)^alpha to the final losses of the five laptop runs."""
import json, math
import numpy as np
from common import Log, save

log = Log('ch4_fit')
runs = json.load(open('results/ch4_scaling_mini.json'))
N = np.array([r['N'] for r in runs], float)
L = np.array([r['curve'][-1]['val'] for r in runs])
slope, icpt = np.polyfit(np.log(N), np.log(L), 1)          # log L = -alpha log N + alpha log Nc
alpha = -slope
Nc = math.exp(icpt / alpha)
log(f'fit on {len(N)} runs (6.6M tokens each): log L = {icpt:.3f} {slope:+.4f} log N')
log(f'  -> L(N) = (Nc / N)^alpha with alpha = {alpha:.3f}, Nc = {Nc:.3g}')
for n, l in zip(N, L):
    log(f'  N = {n:>10,.0f}: measured {l:.3f}, fit {(Nc / n) ** alpha:.3f}')
log(f'  every 10x more parameters multiplies the loss by {10 ** -alpha:.3f} here (Kaplan et al.: 0.839)')
log(f'  extrapolation to N = 1e8 (not measured): {(Nc / 1e8) ** alpha:.3f}  (a warning: the data, 14.9M tokens, would run out first)')
save('ch4_fit', {'alpha': alpha, 'Nc': Nc, 'N': N.tolist(), 'L': L.tolist(), 'per10x': 10 ** -alpha})
