"""Part 6 experiment 2: why unconstrained hyper-connections blow up and why the mHC constraint stops it (CPU, seconds).

A hyper-connection keeps n residual streams instead of one and mixes them at every layer with an n x n matrix H.
Across many layers the streams are multiplied by the PRODUCT of all those matrices. This script compares:

  1. HC  : H = I + noise          (unconstrained, learned near the identity; small errors compound)
  2. mHC : H = Sinkhorn-Knopp(exp(raw))   (projected to a doubly stochastic matrix: rows and columns sum to 1)

and measures the "Amax gain" the mHC paper uses (arXiv 2512.24880, page 6): the largest absolute row sum of the
composite mapping (forward signal) and the largest absolute column sum (backward gradient).
It also runs the tiny 2 x 2 Sinkhorn example from the text and checks our NumPy Sinkhorn against a PyTorch version.
Run:  ~/.venvs/attn/bin/python part6_mhc_toy.py
"""
import numpy as np
import torch
from common import Log, save

log = Log('part6_mhc_toy')


def sinkhorn(raw, iters=20):
    """Method 1 (NumPy). exp() makes every entry positive; then alternate column and row normalisation.
    Same order as the paper's Eq. 8/9: M(t) = T_r(T_c(M(t-1)))."""
    M = np.exp(raw)
    for _ in range(iters):
        M = M / M.sum(axis=0, keepdims=True)     # T_c: every column sums to 1
        M = M / M.sum(axis=1, keepdims=True)     # T_r: every row sums to 1
    return M


def sinkhorn_torch(raw, iters=20):
    """Method 2 (PyTorch, log domain, the numerically safe form used in practice)."""
    logM = torch.as_tensor(raw, dtype=torch.float64)
    for _ in range(iters):
        logM = logM - torch.logsumexp(logM, dim=0, keepdim=True)
        logM = logM - torch.logsumexp(logM, dim=1, keepdim=True)
    return logM.exp().numpy()


def gains(P):
    """Amax gain: worst-case growth of a forward signal (row sums) and of a backward gradient (column sums)."""
    return np.abs(P).sum(axis=1).max(), np.abs(P).sum(axis=0).max()


# ---- 1. the tiny example in the text --------------------------------------------------------------------------
log('1) A 2 x 2 matrix that is a little too big: H = [[1.1, 0.1], [0.1, 1.1]], every row sums to 1.2')
H = np.array([[1.1, 0.1], [0.1, 1.1]])
x = np.array([1.0, 1.0])
for depth in (1, 10, 20, 40):
    y = np.linalg.matrix_power(H, depth) @ x
    log(f'   after {depth:2d} layers the signal [1, 1] becomes [{y[0]:,.1f}, {y[1]:,.1f}]')

log('\n2) Sinkhorn-Knopp on raw = [[0, ln 3], [0, 0]], so exp(raw) = [[1, 3], [1, 1]]')
M = np.exp(np.array([[0.0, np.log(3)], [0.0, 0.0]]))
for t in range(1, 4):
    M = M / M.sum(axis=0, keepdims=True)
    log(f'   step {t} columns: [[{M[0,0]:.3f}, {M[0,1]:.3f}], [{M[1,0]:.3f}, {M[1,1]:.3f}]]  row sums {M.sum(1).round(3).tolist()}')
    M = M / M.sum(axis=1, keepdims=True)
    log(f'   step {t} rows   : [[{M[0,0]:.3f}, {M[0,1]:.3f}], [{M[1,0]:.3f}, {M[1,1]:.3f}]]  column sums {M.sum(0).round(3).tolist()}')
M20 = sinkhorn(np.array([[0.0, np.log(3)], [0.0, 0.0]]))
log(f'   after 20 steps: [[{M20[0,0]:.4f}, {M20[0,1]:.4f}], [{M20[1,0]:.4f}, {M20[1,1]:.4f}]], '
    f'rows {M20.sum(1).round(6).tolist()}, columns {M20.sum(0).round(6).tolist()}')
y = M20 @ np.array([4.0, 2.0])
log(f'   mixing the streams [4, 2]: [{y[0]:.3f}, {y[1]:.3f}], mean before 3.000, mean after {y.mean():.3f}')

# ---- 2. deep stacks with n = 4 streams (the V4 setting hc_mult = 4) --------------------------------------------
n, depth, eps, seeds = 4, 120, 0.2, 5
log(f'\n3) n = {n} streams, {depth} mixing layers (a 60-block model: attention + MoE per block), noise scale {eps}, {seeds} seeds')
curves = {'HC': [], 'mHC': [], 'Identity': []}
spec_hc, spec_mhc = [], []
agree = 0.0
for seed in range(seeds):
    rng = np.random.default_rng(seed)
    P_hc, P_m = np.eye(n), np.eye(n)
    g_hc, g_m = [], []
    for l in range(depth):
        G = rng.standard_normal((n, n))
        H_hc = np.eye(n) + eps * G                        # unconstrained, near the identity
        raw = 2.0 * np.eye(n) + G                         # raw logits, biased toward keeping each stream
        H_m = sinkhorn(raw)
        agree = max(agree, np.abs(H_m - sinkhorn_torch(raw)).max())
        spec_hc.append(np.linalg.norm(H_hc, 2)); spec_mhc.append(np.linalg.norm(H_m, 2))
        P_hc, P_m = H_hc @ P_hc, H_m @ P_m                # composite mapping from layer 0 to layer l
        g_hc.append(max(gains(P_hc))); g_m.append(max(gains(P_m)))
    curves['HC'].append(g_hc); curves['mHC'].append(g_m)
curves['Identity'] = [[1.0] * depth]
med = {k: np.median(np.array(v), axis=0) for k, v in curves.items()}
mx = {k: np.max(np.array(v), axis=0) for k, v in curves.items()}
log('   layer   HC gain (median / max of seeds)   mHC gain (median / max)')
for l in (1, 10, 30, 60, 90, 120):
    log(f'   {l:5d}   {med["HC"][l-1]:12,.1f} / {mx["HC"][l-1]:12,.1f}        {med["mHC"][l-1]:6.3f} / {mx["mHC"][l-1]:6.3f}')
log(f'   largest single-layer spectral norm: HC {max(spec_hc):.3f}, mHC {max(spec_mhc):.6f}')
log(f'   NumPy Sinkhorn vs PyTorch log-domain Sinkhorn: largest difference {agree:.1e}')

# what happens to an actual signal: the stream mean is kept, the differences between streams fade
rng = np.random.default_rng(0)
x0 = np.array([3.0, -1.0, 2.0, 0.0])
x_hc, x_m = x0.copy(), x0.copy()
for l in range(depth):
    G = rng.standard_normal((n, n))
    x_hc = (np.eye(n) + eps * G) @ x_hc
    x_m = sinkhorn(2.0 * np.eye(n) + G) @ x_m
log(f'\n4) one signal, streams {x0.tolist()} (mean {x0.mean():.2f}, norm {np.linalg.norm(x0):.2f}), after {depth} layers:')
log(f'   HC : streams [{", ".join(f"{v:,.1f}" for v in x_hc)}], mean {x_hc.mean():,.2f}, norm {np.linalg.norm(x_hc):,.1f}')
log(f'   mHC: streams [{", ".join(f"{v:.3f}" for v in x_m)}], mean {x_m.mean():.3f}, norm {np.linalg.norm(x_m):.3f}')

save('part6_mhc_toy', dict(depth=depth, n=n, eps=eps, hc_median=med['HC'].tolist(), hc_max=mx['HC'].tolist(),
                           mhc_median=med['mHC'].tolist(), mhc_max=mx['mHC'].tolist(),
                           sinkhorn2x2=M20.tolist()))
