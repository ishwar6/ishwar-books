"""Chapter 6: returns, value functions, TD errors and GAE on a tiny "token" world where everything can be computed exactly.
The policy writes a reply of T = 6 tokens, each A or B. The state is (position t, number of A's so far). The only reward
comes at the end: 1 if the reply has at least four A's, else 0 (like a checker that marks an answer right or wrong).
(1) returns with and without discount, (2) exact V, Q, A for a fixed policy, (3) one episode worked by hand: TD errors and
GAE, (4) bias and variance of GAE against the exact advantage for many lambdas and three value-function qualities,
(5) actor-critic training with several lambdas against plain REINFORCE, over seeds."""
import numpy as np
from math import comb
from common import Log, save

log = Log('ch6_gae')
T, NEED, P_A = 6, 4, 0.6
R = {'T': T, 'need': NEED, 'pA': P_A}


def V_exact(t, k, p=P_A):
    """Probability of ending with >= NEED A's from state (t, k) when every later token is A with probability p."""
    left = T - t
    return sum(comb(left, j) * p ** j * (1 - p) ** (left - j) for j in range(left + 1) if k + j >= NEED)


V = np.zeros((T + 1, T + 1))
for t in range(T + 1):
    for k in range(t + 1):
        V[t, k] = V_exact(t, k)
Q = np.zeros((T, T + 1, 2))                       # action 0 = A, 1 = B
for t in range(T):
    for k in range(t + 1):
        Q[t, k, 0], Q[t, k, 1] = V[t + 1, k + 1], V[t + 1, k]
A = Q - V[:T, :, None]

# ---------------------------------------------------------------- 1. returns and discount
log('== 1. return of the episode rewards [0, 0, 0, 0, 0, 1]')
rew = [0, 0, 0, 0, 0, 1]
for g in [1.0, 0.9]:
    G = [sum(g ** (j - t) * rew[j] for j in range(t, T)) for t in range(T)]
    log(f'  gamma = {g}: G_t for t = 0..5 = {[round(x, 4) for x in G]}')
R['returns'] = {str(g): [sum(g ** (j - t) * rew[j] for j in range(t, T)) for t in range(T)] for g in [1.0, 0.9]}

# ---------------------------------------------------------------- 2. exact values
log(f'\n== 2. exact values for the fixed policy P(A) = {P_A}')
log(f'  V(start) = probability of success = {V[0, 0]:.4f}')
for t in range(T + 1):
    log(f'  t = {t}: V(t, k) for k = 0..{t}: {[round(float(V[t, k]), 3) for k in range(t + 1)]}')
log(f'  at the start: Q(A) = {Q[0, 0, 0]:.4f}, Q(B) = {Q[0, 0, 1]:.4f}, advantage A(A) = {A[0, 0, 0]:+.4f}, A(B) = {A[0, 0, 1]:+.4f}')
R['V'] = V.tolist()

# ---------------------------------------------------------------- 3. one episode by hand
rng = np.random.default_rng(3)
noise = rng.normal(0, 0.15, V.shape)
Vhat_rough = V + noise
Vhat_rough[T] = V[T]                               # at the end the value is just the reward (known)
ep_rng = np.random.default_rng(11)
while True:                                        # pick an episode that succeeds, for a friendlier example
    acts = (ep_rng.random(T) >= P_A).astype(int)
    if (acts == 0).sum() >= NEED and acts[0] == 0 and acts[1] == 1:
        break
ks = np.concatenate([[0], np.cumsum(acts == 0)])
lam, gam = 0.95, 1.0
log(f'\n== 3. one episode: tokens {"".join("AB"[a] for a in acts)}, reward at the end = 1; lambda = {lam}, gamma = {gam}')
log('  using a rough value estimate (exact V plus noise of sd 0.15)')
rows = []
for t in range(T):
    r_t = 1.0 if t == T - 1 and ks[T] >= NEED else 0.0
    v, vn = Vhat_rough[t, ks[t]], (0.0 if t == T - 1 else Vhat_rough[t + 1, ks[t + 1]])
    # terminal: the episode ends after token 6, so V(s_T) = 0 and the reward carries the outcome
    delta = r_t + gam * vn - v
    rows.append([t, int(ks[t]), 'AB'[acts[t]], r_t, float(v), float(vn), float(delta), float(A[t, ks[t], acts[t]])])
adv = [0.0] * T
run = 0.0
for t in reversed(range(T)):
    run = rows[t][6] + gam * lam * run
    adv[t] = run
for t in range(T):
    _, k, tok, r_t, v, vn, d, a_true = rows[t]
    log(f'  t={t} state (t={t}, k={k}) token {tok}: r={r_t:.0f}  V(s_t)={v:.3f}  V(s_t+1)={vn:.3f}  delta={d:+.3f}  '
        f'GAE={adv[t]:+.3f}  exact A={a_true:+.3f}')
mc = [rows[-1][3] - rows[t][4] for t in range(T)]
adv = [float(x) for x in adv]
log('  lambda = 1 (Monte Carlo minus V):', [round(float(x), 3) for x in mc])
log('  lambda = 0 (one TD error):       ', [round(float(r[6]), 3) for r in rows])
R['episode'] = dict(tokens=''.join('AB'[a] for a in acts), rows=rows, gae=adv, lam=lam, mc=mc)

# ---------------------------------------------------------------- 4. bias and variance of GAE
N = 100_000
LAMS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]
log(f'\n== 4. GAE against the exact advantage, {N:,} episodes, gamma = 1')
acts = (np.random.default_rng(5).random((N, T)) >= P_A).astype(int)
ks = np.concatenate([np.zeros((N, 1), int), np.cumsum(acts == 0, 1)], 1)
final = (ks[:, T] >= NEED).astype(float)
R['gae'] = {}
for vname, sd in [('exact V', 0.0), ('V + noise 0.05', 0.05), ('V + noise 0.15', 0.15)]:
    Vh = V + np.random.default_rng(3).normal(0, 1, V.shape) * sd
    vals = np.stack([Vh[t, ks[:, t]] for t in range(T)], 1)                      # V(s_t)
    nxt = np.concatenate([vals[:, 1:], np.zeros((N, 1))], 1)                       # V(s_t+1), 0 after the last token
    r = np.zeros((N, T)); r[:, -1] = final
    delta = r + nxt - vals
    true = np.stack([A[t, ks[:, t], acts[:, t]] for t in range(T)], 1)
    key = ks[:, :T] * 2 + acts                                                     # (k, a) within each t
    out = []
    for lam in LAMS:
        adv = np.zeros((N, T)); run = np.zeros(N)
        for t in reversed(range(T)):
            run = delta[:, t] + lam * run
            adv[:, t] = run
        b2 = var = 0.0
        for t in range(T):
            for kk in np.unique(key[:, t]):
                m = key[:, t] == kk
                w = m.mean()
                b2 += w * (adv[m, t].mean() - true[m, t].mean()) ** 2 / T
                var += w * adv[m, t].var() / T
        out.append([lam, b2, var, b2 + var])
        log(f'  {vname:15s} lambda {lam:4.2f}: bias^2 {b2:.4f}  variance {var:.4f}  total error {b2 + var:.4f}')
    R['gae'][vname] = out

# ---------------------------------------------------------------- 5. actor-critic training with GAE
SEEDS, ITERS, BATCH, LR_PI, LR_V = 100, 150, 16, 0.5, 0.2
log(f'\n== 5. training from P(A) = 0.5 everywhere: {SEEDS} seeds, {ITERS} updates of {BATCH} episodes each')


def train(method, lam, seed, offset=0.0):
    rng = np.random.default_rng(seed)
    th = np.zeros((T, T + 1))                       # logit of A at state (t, k); P(A) = sigmoid
    Vh = np.zeros((T + 1, T + 1))                   # critic starts knowing nothing
    curve = []
    for it in range(ITERS):
        pA = 1 / (1 + np.exp(-th))
        # success probability of the current policy, computed exactly by dynamic programming
        dist = np.zeros(T + 1); dist[0] = 1
        for t in range(T):
            new = np.zeros(T + 1)
            for k in range(t + 1):
                new[k + 1] += dist[k] * pA[t, k]; new[k] += dist[k] * (1 - pA[t, k])
            dist = new
        curve.append(dist[NEED:].sum())
        acts = np.zeros((BATCH, T), int); ks = np.zeros((BATCH, T + 1), int)
        for t in range(T):
            acts[:, t] = (rng.random(BATCH) >= pA[t, ks[:, t]]).astype(int)
            ks[:, t + 1] = ks[:, t] + (acts[:, t] == 0)
        final = (ks[:, T] >= NEED).astype(float) + offset
        vals = np.stack([Vh[t, ks[:, t]] for t in range(T)], 1)
        if method == 'reinforce':
            adv = np.repeat(final[:, None], T, 1)   # every token gets the whole return, no baseline
        elif method == 'batchmean':
            loo = (final.sum() - final) / (BATCH - 1)   # leave-one-out mean of the other episodes' returns
            adv = np.repeat((final - loo)[:, None], T, 1)
        else:
            nxt = np.concatenate([vals[:, 1:], np.zeros((BATCH, 1))], 1)
            r = np.zeros((BATCH, T)); r[:, -1] = final
            delta = r + nxt - vals
            adv = np.zeros((BATCH, T)); run = np.zeros(BATCH)
            for t in reversed(range(T)):
                run = delta[:, t] + lam * run
                adv[:, t] = run
            target = adv + vals                        # the lambda-return, the critic's regression target
            g = np.zeros_like(Vh); c = np.zeros_like(Vh)
            for t in range(T):
                np.add.at(g[t], ks[:, t], target[:, t] - vals[:, t]); np.add.at(c[t], ks[:, t], 1)
            Vh += LR_V * g / np.maximum(c, 1)
        grad = np.zeros_like(th)
        for t in range(T):
            p = pA[t, ks[:, t]]
            dlog = np.where(acts[:, t] == 0, 1 - p, -p)  # d log pi(a) / d logit
            np.add.at(grad[t], ks[:, t], adv[:, t] * dlog)
        th += LR_PI * grad / BATCH
    return curve


R['train'] = {}
for name, method, lam in [('REINFORCE, no baseline', 'reinforce', None), ('REINFORCE, leave-one-out baseline', 'batchmean', None), ('GAE lambda 0', 'ac', 0.0),
                          ('GAE lambda 0.5', 'ac', 0.5), ('GAE lambda 0.95', 'ac', 0.95), ('GAE lambda 1', 'ac', 1.0)]:
    C = np.array([train(method, lam, 100 + s) for s in range(SEEDS)])
    hit = [(np.argmax(c >= 0.9) if (c >= 0.9).any() else ITERS) for c in C]
    log(f'  {name:34s}: P(success) after 10/25/50/150 updates = {C[:, 10].mean():.3f} / {C[:, 25].mean():.3f} / '
        f'{C[:, 50].mean():.3f} / {C[:, -1].mean():.3f};  median updates to reach 0.9: {int(np.median(hit))};  sd over seeds at 25: {C[:, 25].std():.3f}')
    R['train'][name] = dict(mean=C.mean(0).tolist(), p10=np.percentile(C, 10, 0).tolist(), p90=np.percentile(C, 90, 0).tolist(),
                            median_hit=float(np.median(hit)), sd25=float(C[:, 25].std()))
OFF = 5.0
log(f'\n   same, but every episode also gets +{OFF} (reward 5 for failure, 6 for success)')
R['train_offset'] = {}
for name, method, lam in [('REINFORCE, no baseline', 'reinforce', None), ('REINFORCE, leave-one-out baseline', 'batchmean', None),
                          ('GAE lambda 0.95', 'ac', 0.95)]:
    C = np.array([train(method, lam, 100 + s, OFF) for s in range(SEEDS)])
    hit = [(np.argmax(c >= 0.9) if (c >= 0.9).any() else ITERS) for c in C]
    log(f'  {name:34s}: P(success) after 10/25/50/150 updates = {C[:, 10].mean():.3f} / {C[:, 25].mean():.3f} / '
        f'{C[:, 50].mean():.3f} / {C[:, -1].mean():.3f};  median to 0.9: {int(np.median(hit))};  sd at 25: {C[:, 25].std():.3f};  '
        f'seeds below 0.5 at the end: {(C[:, -1] < 0.5).mean():.0%}')
    R['train_offset'][name] = dict(mean=C.mean(0).tolist(), p10=np.percentile(C, 10, 0).tolist(), p90=np.percentile(C, 90, 0).tolist(),
                                   median_hit=float(np.median(hit)), sd25=float(C[:, 25].std()), below=float((C[:, -1] < 0.5).mean()))
save('ch6_gae', R)
