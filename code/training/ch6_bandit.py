"""Chapter 6: REINFORCE on a four-armed bandit, with and without a baseline.
A "prompt" with four possible "replies" (arms). Reply k earns a noisy star rating with mean MU[k]. The policy is a softmax over
four logits. We (1) work one REINFORCE step by hand, (2) check by Monte Carlo that the estimator is unbiased with and without a
baseline and measure its variance, (3) sweep the baseline value, (4) train 200 seeds with three baselines and compare."""
import numpy as np
from common import Log, save

log = Log('ch6_bandit')
MU = np.array([5.0, 6.0, 7.0, 8.0])     # mean rating of each reply; reply 4 (index 3) is best
SD = 1.0                                 # rating noise
K = len(MU)
R = {'mu': MU.tolist(), 'sd': SD}


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


# ---------------------------------------------------------------- 1. one step by hand
theta = np.zeros(K)
pi = softmax(theta)
rng = np.random.default_rng(0)
a = 2                                     # we pick reply 3 to make the example concrete
r = 7.3
score = -pi.copy(); score[a] += 1         # d log pi(a) / d theta = onehot(a) - pi
log('== 1. one REINFORCE step by hand, theta = 0')
log('pi            =', np.round(pi, 4))
log(f'action a = {a + 1}, reward r = {r}')
log('score  = onehot(a) - pi =', np.round(score, 4))
log('grad (no baseline) = r * score       =', np.round(r * score, 4))
b = float(pi @ MU)
log(f'baseline b = V = sum_k pi_k mu_k = {b}')
log('grad (baseline)    = (r - b) * score =', np.round((r - b) * score, 4))
true_grad = pi * (MU - pi @ MU)
log('exact gradient of J = pi_k (mu_k - J)  =', np.round(true_grad, 4))
lr = 0.1
log('new pi after one step, lr 0.1, no baseline:', np.round(softmax(theta + lr * r * score), 4))
log('new pi after one step, lr 0.1, baseline   :', np.round(softmax(theta + lr * (r - b) * score), 4))
R['step'] = dict(pi=pi.tolist(), a=a, r=r, score=score.tolist(), g0=(r * score).tolist(), b=b,
                 gb=((r - b) * score).tolist(), true=true_grad.tolist(),
                 pi_new0=softmax(theta + lr * r * score).tolist(), pi_newb=softmax(theta + lr * (r - b) * score).tolist())

# ---------------------------------------------------------------- 2. unbiased? and how noisy?
def estimates(theta, b, n, rng):
    pi = softmax(theta)
    acts = rng.choice(K, size=n, p=pi)
    rew = MU[acts] + SD * rng.standard_normal(n)
    S = -np.tile(pi, (n, 1)); S[np.arange(n), acts] += 1
    return (rew - b)[:, None] * S, S, rew


N = 200_000
log(f'\n== 2. Monte Carlo check at theta = 0 with {N:,} samples')
G0, S, rew = estimates(theta, 0.0, N, np.random.default_rng(1))
Gb, _, _ = estimates(theta, b, N, np.random.default_rng(1))
log('mean score vector E[grad log pi]     =', np.round(S.mean(0), 4), '(should be 0)')
log('mean estimate, no baseline           =', np.round(G0.mean(0), 4))
log('mean estimate, baseline b = 6.5      =', np.round(Gb.mean(0), 4))
log('exact gradient                       =', np.round(true_grad, 4))
v0, vb = G0.var(0).sum(), Gb.var(0).sum()
log(f'total variance (sum over 4 components): no baseline {v0:.3f}, baseline {vb:.3f}, ratio {v0 / vb:.1f}x')
R['mc'] = dict(N=N, mean_score=S.mean(0).tolist(), mean0=G0.mean(0).tolist(), meanb=Gb.mean(0).tolist(), var0=v0, varb=vb)

# ---------------------------------------------------------------- 3. sweep the baseline
log('\n== 3. variance of the estimate as a function of a constant baseline b (theta = 0)')
sq = (S ** 2).sum(1)
b_star = float((rew * sq).sum() / sq.sum())
sweep = []
for bb in np.arange(0, 12.01, 0.5):
    g = (rew - bb)[:, None] * S
    sweep.append([float(bb), float(g.var(0).sum())])
    if bb in (0, 3, 5, 6, 6.5, 7, 8, 10, 12):
        log(f'  b = {bb:5.1f}: total variance {sweep[-1][1]:8.3f},  mean estimate {np.round(g.mean(0), 3)}')
g = (rew - b_star)[:, None] * S
log(f'variance-minimising constant b* = E[r |score|^2] / E[|score|^2] = {b_star:.3f}, variance {g.var(0).sum():.3f}')
R['sweep'] = sweep
R['b_star'] = [b_star, float(g.var(0).sum())]

# second snapshot: a policy that already prefers reply 4
theta2 = np.array([0.0, 0.0, 0.0, 2.0])
pi2 = softmax(theta2)
G0, S2, rew2 = estimates(theta2, 0.0, N, np.random.default_rng(2))
V2 = float(pi2 @ MU)
Gb2, _, _ = estimates(theta2, V2, N, np.random.default_rng(2))
log(f'\nat theta = {theta2.tolist()}: pi = {np.round(pi2, 3)}, V = {V2:.3f}')
log(f'  total variance: no baseline {G0.var(0).sum():.3f}, baseline V {Gb2.var(0).sum():.3f}, ratio {G0.var(0).sum() / Gb2.var(0).sum():.1f}x')
R['mc2'] = dict(pi=pi2.tolist(), V=V2, var0=float(G0.var(0).sum()), varb=float(Gb2.var(0).sum()))

# ---------------------------------------------------------------- 4. training, 200 seeds
SEEDS, STEPS, LR = 200, 400, 0.05
log(f'\n== 4. training: {SEEDS} seeds x {STEPS} steps, one sample per step, learning rate {LR}')
runs = {}
for name in ['none', 'running', 'value']:
    J = np.zeros((SEEDS, STEPS + 1))
    pbest = np.zeros((SEEDS, STEPS + 1))
    winner = np.zeros(SEEDS, int)
    for s in range(SEEDS):
        rng = np.random.default_rng(1000 + s)
        th = np.zeros(K)
        avg, n = 0.0, 0
        for t in range(STEPS + 1):
            p = softmax(th)
            J[s, t], pbest[s, t] = p @ MU, p[3]
            if t == STEPS:
                winner[s] = int(np.argmax(p)) + 1
                break
            a = rng.choice(K, p=p)
            r = MU[a] + SD * rng.standard_normal()
            base = {'none': 0.0, 'running': avg, 'value': p @ MU}[name]
            sc = -p; sc[a] += 1
            th = th + LR * (r - base) * sc
            n += 1; avg += (r - avg) / n
    final_best = (pbest[:, -1] > 0.5).mean()
    wrong = (pbest[:, -1] < 0.1).mean()
    log(f'  baseline {name:8s}: mean J at step 100/200/400 = {J[:, 100].mean():.3f} / {J[:, 200].mean():.3f} / {J[:, -1].mean():.3f};'
        f'  spread (10th-90th pct) at 400: {np.percentile(J[:, -1], 10):.3f} to {np.percentile(J[:, -1], 90):.3f};'
        f'  seeds with P(best) > 0.5: {final_best:.0%};  seeds with P(best) < 0.1: {wrong:.0%}')
    log(f'      most likely reply at the end, count over seeds: { {int(k): int(v) for k, v in zip(*np.unique(winner, return_counts=True))} }')
    steps = list(range(0, STEPS + 1, 10))
    runs[name] = dict(steps=steps, mean=J.mean(0)[steps].tolist(), p10=np.percentile(J, 10, 0)[steps].tolist(),
                      p90=np.percentile(J, 90, 0)[steps].tolist(), pbest_mean=pbest.mean(0)[steps].tolist(),
                      final_best=float(final_best), final_wrong=float(wrong), winners=np.bincount(winner, minlength=5)[1:].tolist(), final_J=J[:, -1].tolist(),
                      examples=J[:6][:, steps].tolist())
R['train'] = dict(seeds=SEEDS, steps=STEPS, lr=LR, runs=runs)
save('ch6_bandit', R)
