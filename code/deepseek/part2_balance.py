"""Part 2: a simulation of DeepSeek-V3's auxiliary-loss-free load balancing (CPU, NumPy, no real model).

Router of V3's shape: 256 experts, each token picks 8 (no node limit here, to keep the effect clear).
Each step a fresh batch of 4,096 random tokens arrives. Some experts are "popular": their affinities get a fixed
offset, as a trained router's would. We then:
  A. route with no balancing at all;
  B. route with V3's bias rule: after each step, b_i -= gamma if expert i was overloaded, b_i += gamma if it was
     underloaded (gamma = 0.001, the paper's value);
  C. add a toy "rich get richer" feedback (a picked expert gets better, so it is picked more): routing collapse;
  D. the same feedback, with the bias rule switched on.
We report MaxVio = (max load - mean load) / mean load, the measure used by Wang et al. (2024), and the
sequence-wise balance loss of Eq. 17 to 20 (alpha = 0.0001) before and after.
"""
import numpy as np
from common import Log, save

log = Log('part2_balance')
E, K, T, D, STEPS, GAMMA = 256, 8, 4096, 32, 400, 0.001
rng = np.random.default_rng(0)
centroids = rng.normal(0, 1, (E, D)) / np.sqrt(D)
popular = rng.normal(0, 0.35, E)                         # fixed preference: some experts score higher for every token


def step(offset, bias):
    u = rng.normal(0, 1, (T, D))
    s = 1 / (1 + np.exp(-(u @ centroids.T + offset)))    # affinities s_i,t (sigmoid)
    ids = np.argpartition(-(s + bias), K, axis=1)[:, :K]  # top-8 of s + b
    load = np.bincount(ids.ravel(), minlength=E)         # tokens each expert received
    return s, ids, load


def maxvio(load):
    return (load.max() - load.mean()) / load.mean()


def bal_loss(s, ids, alpha=1e-4):
    """Eq. 17 to 20: L_Bal = alpha * sum_i f_i P_i over the T tokens of one sequence."""
    T_ = s.shape[0]
    f = E / (K * T_) * np.bincount(ids.ravel(), minlength=E)    # Eq. 18: load, scaled so 1.0 = fair share
    P = (s / s.sum(1, keepdims=True)).mean(0)                    # Eq. 19-20: mean normalised affinity
    return alpha * (f * P).sum(), f, P


# tiny worked example of Eq. 17-20 (4 experts, K = 1, 4 tokens), numbers you can check by hand
log('== Worked example of the balance loss: 4 experts, each token picks 1, 4 tokens ==')
s_toy = np.array([[0.9, 0.1, 0.1, 0.1], [0.8, 0.2, 0.1, 0.1], [0.7, 0.1, 0.6, 0.2], [0.1, 0.2, 0.3, 0.9]])
for name, ids_toy in [('collapsed (picks 0,0,0,3)', np.array([[0], [0], [0], [3]])),
                      ('balanced  (picks 0,1,2,3)', np.array([[0], [1], [2], [3]]))]:
    Eo, Ko, To = 4, 1, 4
    f = Eo / (Ko * To) * np.bincount(ids_toy.ravel(), minlength=Eo)
    P = (s_toy / s_toy.sum(1, keepdims=True)).mean(0)
    log(f'  {name}: f = {np.round(f, 2).tolist()}  P = {np.round(P, 3).tolist()}  sum f*P = {(f * P).sum():.3f}')

runs = {}
for name, feedback, balance in [('A no balancing', 0.0, False), ('B bias rule', 0.0, True),
                                ('C feedback, no balancing', 0.004, False), ('D feedback + bias rule', 0.004, True)]:
    offset, bias = popular.copy(), np.zeros(E)
    hist, first, last = [], None, None
    for t in range(STEPS):
        s, ids, load = step(offset, bias)
        hist.append(maxvio(load))
        if t == 0:
            first = (load.copy(), bal_loss(s, ids)[0])
        if balance:                                       # V3's rule, applied at the end of every step
            bias -= GAMMA * np.sign(load - load.mean())
        if feedback:                                      # toy: experts that get more tokens get better
            offset += feedback * (load / load.mean() - 1)
        last = (load.copy(), bal_loss(s, ids)[0], s, ids)
    runs[name] = dict(maxvio=hist, load_first=first[0].tolist(), load_last=last[0].tolist(),
                      loss_first=first[1], loss_last=last[1], bias=bias.tolist())
    top = np.sort(last[0])[::-1]
    log(f'\n== {name} ==')
    log(f'  MaxVio: step 1 {hist[0]:.2f}, step 100 {hist[99]:.2f}, step {STEPS} {hist[-1]:.2f}')
    log(f'  tokens per expert at the last step: fair share {T * K // E}, busiest {top[0]}, idlest {top[-1]}, '
        f'experts with no token {int((last[0] == 0).sum())}')
    log(f'  busiest 8 experts take {100 * top[:8].sum() / (T * K):.1f}% of all assignments (fair: {100 * 8 / E:.1f}%)')
    log(f'  balance loss L_Bal (alpha = 1e-4): first step {first[1]:.3e}, last step {last[1]:.3e}')
    if balance:
        log(f'  bias range after {STEPS} steps: {bias.min():+.3f} to {bias.max():+.3f}; '
            f'correlation with the popularity offset: {np.corrcoef(bias, popular)[0, 1]:+.2f}')

# the gate values still come from s, not s + b: check on one batch of run B
s, ids, _ = step(popular, np.array(runs['B bias rule']['bias']))
g = np.take_along_axis(s, ids, 1)
log('\n== Gates use the original affinities ==')
log(f'  one token: chosen experts {sorted(ids[0].tolist())}')
log(f'  gates from s (normalised): {np.round(g[0] / g[0].sum(), 3).tolist()}')

save('part2_balance', dict(E=E, K=K, T=T, steps=STEPS, gamma=GAMMA, runs=runs, popular=popular.tolist()))
