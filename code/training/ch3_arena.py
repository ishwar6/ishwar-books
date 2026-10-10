"""Chapter 3, experiment 6: ratings from pairwise votes, as in Chatbot Arena (a simulation; no real vote data).
Six imaginary models have hidden "true" ratings on the Elo scale. We simulate votes between random pairs with the
Bradley-Terry probability, then try to recover the ratings in two ways:
  online Elo     update two ratings after every vote (K = 4); the result depends on the order of the votes
  Bradley-Terry  fit all votes at once by maximum likelihood (logistic regression); order does not matter
and attach 95% confidence intervals by bootstrap (resample the votes with replacement, refit, take percentiles).
Also: a worked Elo update, and how the confidence interval of a win rate shrinks with the number of votes."""
import math
import numpy as np
from common import Log, save

log = Log('ch3_arena')
rng = np.random.default_rng(0)
NAMES = ['model-A', 'model-B', 'model-C', 'model-D', 'model-E', 'model-F']
TRUE = np.array([1250., 1210., 1190., 1120., 1060., 1000.])
M = len(NAMES)


def p_win(ra, rb):
    """Elo / Bradley-Terry on the chess scale: 400 points = 10-to-1 odds."""
    return 1 / (1 + 10 ** ((rb - ra) / 400))


# ---------------------------------------------------------------- worked Elo update
log('== one Elo update, by hand ==')
ra, rb, K = 1200, 1100, 32
e = p_win(ra, rb)
log(f'A has 1200, B has 1100.  expected score of A: 1 / (1 + 10^((1100 - 1200)/400)) = {e:.4f}')
log(f'A wins (score 1), K = {K}:  A -> 1200 + {K} x (1 - {e:.4f}) = {ra + K * (1 - e):.1f}   B -> 1100 - {K} x (1 - {e:.4f}) = {rb - K * (1 - e):.1f}')
log(f'B wins instead (an upset):  A -> 1200 - {K} x {e:.4f} = {ra - K * e:.1f}   B -> 1100 + {K} x {e:.4f} = {rb + K * e:.1f}')
log('')


def simulate(n):
    a = rng.integers(0, M, n)
    b = (a + rng.integers(1, M, n)) % M                       # a different model
    y = (rng.random(n) < p_win(TRUE[a], TRUE[b])).astype(float)   # 1 if a won
    return a, b, y


def elo(a, b, y, k=4, init=1000.):
    r = np.full(M, init)
    for i, j, s in zip(a, b, y):
        e = p_win(r[i], r[j])
        r[i] += k * (s - e); r[j] -= k * (s - e)
    return r


def bt(a, b, y, iters=2000, lr=0.5):
    """Maximum-likelihood Bradley-Terry: minimise the binary cross-entropy of sigmoid(xi_a - xi_b) against y.
    The votes are first counted into a table W[i, j] = wins of i over j, so each step is a 6 x 6 computation.
    Plain gradient descent on the natural-log scale, then converted to the Elo scale (x 400 / ln 10)."""
    W = np.zeros((M, M))
    np.add.at(W, (a, b), y); np.add.at(W, (b, a), 1 - y)
    Nn = W + W.T                                              # games played between i and j
    xi = np.zeros(M)
    for _ in range(iters):
        P = 1 / (1 + np.exp(-(xi[:, None] - xi[None, :])))   # P[i, j] = chance i beats j
        g = (Nn * P - W).sum(1)                               # gradient of the total cross-entropy
        xi -= lr * g / Nn.sum(1)
        xi -= xi.mean()
    return xi * 400 / math.log(10)


def anchor(r):
    return r - r[-1] + 1000                                   # model-F fixed at 1000, like Arena's anchoring


a, b, y = simulate(6000)
log(f'== {len(y)} simulated votes between random pairs ==')
true = anchor(TRUE)
r_elo = anchor(elo(a, b, y))
r_bt = anchor(bt(a, b, y))
orders = [anchor(elo(a[p], b[p], y[p])) for p in (rng.permutation(len(y)) for _ in range(100))]
spread = np.percentile(orders, [2.5, 97.5], axis=0)
boots = []
for _ in range(300):
    idx = rng.integers(0, len(y), len(y))
    boots.append(anchor(bt(a[idx], b[idx], y[idx])))
ci = np.percentile(boots, [2.5, 97.5], axis=0)
log(f'{"model":9s} {"true":>6s} {"online Elo":>11s} {"Elo over 100 orders":>22s} {"Bradley-Terry":>14s} {"95% bootstrap CI":>18s}')
for i in range(M):
    log(f'{NAMES[i]:9s} {true[i]:6.0f} {r_elo[i]:11.0f} {spread[0, i]:10.0f} to {spread[1, i]:5.0f} {r_bt[i]:14.0f} {ci[0, i]:9.0f} to {ci[1, i]:5.0f}')
res = {'true': true.tolist(), 'elo': r_elo.tolist(), 'elo_order_lo': spread[0].tolist(), 'elo_order_hi': spread[1].tolist(),
       'bt': r_bt.tolist(), 'bt_lo': ci[0].tolist(), 'bt_hi': ci[1].tolist(), 'names': NAMES, 'votes': len(y)}
overlap = ci[0, 1] <= ci[1, 2]
log(f'model-B and model-C are {true[1] - true[2]:.0f} points apart; their intervals {"overlap: the data cannot separate them" if overlap else "do not overlap"}')

# how the BT rating error shrinks with more votes
log('')
log('== more votes, narrower intervals (Bradley-Terry, 95% bootstrap CI width for model-A) ==')
res['by_n'] = []
for n in [300, 1000, 3000, 10000]:
    a2, b2, y2 = simulate(n)
    bs = [anchor(bt(a2[i], b2[i], y2[i]))[0] for i in (rng.integers(0, n, n) for _ in range(200))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    res['by_n'].append(dict(n=n, lo=lo, hi=hi, fit=float(anchor(bt(a2, b2, y2))[0])))
    log(f'{n:6d} votes:  model-A = {res["by_n"][-1]["fit"]:5.0f}, 95% CI {lo:5.0f} to {hi:5.0f}  (width {hi - lo:4.0f})')

# win rate of one pair with a confidence interval
log('')
log('== a win rate and its 95% confidence interval (normal approximation and bootstrap) ==')
res['winrate'] = []
pt = p_win(TRUE[0], TRUE[1])
for n in [20, 100, 400, 1600]:
    w = (rng.random(n) < pt).astype(float)
    p = w.mean()
    se = math.sqrt(p * (1 - p) / n)
    bs = [rng.choice(w, n).mean() for _ in range(2000)]
    blo, bhi = np.percentile(bs, [2.5, 97.5])
    res['winrate'].append(dict(n=n, p=p, lo=p - 1.96 * se, hi=p + 1.96 * se, blo=blo, bhi=bhi))
    log(f'{n:5d} votes: model-A beats model-B {p:.3f}   normal CI {p - 1.96 * se:.3f} to {p + 1.96 * se:.3f}   bootstrap CI {blo:.3f} to {bhi:.3f}')
log(f'(true probability: {pt:.3f})')
res['p_true_AB'] = pt
save('ch3_arena', res)
