"""Chapter 7: the Bradley-Terry reward model loss by hand, plus two small simulations.
  1. one pair: probability, loss and gradient for real numbers
  2. shifting every reward by a constant changes nothing (why reward models get normalised)
  3. a ranking of K answers becomes K(K-1)/2 pairs (InstructGPT)
  4. the Elo link: one unit of reward = 400 / ln 10 Elo points
  5. label noise (Christiano et al.): with a 10% chance of a random answer, the loss of a flipped label is capped
  6. ratings or comparisons? 20 simulated raters with personal offsets and scales
  7. a process reward: the score of a solution from per-step probabilities (Lightman et al. 2023)"""
import itertools, math
import numpy as np
import torch
import torch.nn.functional as F
from common import Log, save

log = Log('ch7_bt')
sig = lambda z: 1 / (1 + math.exp(-z))
R = {}

log('== 1. one pair ==')
rc, rr = 1.3, 0.4
d = rc - rr
p = sig(d)
log(f'r(chosen) = {rc}, r(rejected) = {rr}, gap = {d:.1f}')
log(f'P(chosen beats rejected) = sigmoid({d:.1f}) = 1 / (1 + e^-{d:.1f}) = 1 / (1 + {math.exp(-d):.4f}) = {p:.4f}')
log(f'loss = -log({p:.4f}) = {-math.log(p):.4f}')
log(f'if the model had them the wrong way round (gap -{d:.1f}): P = {sig(-d):.4f}, loss = {-math.log(sig(-d)):.4f}')
# gradient with autograd
t = torch.tensor([rc, rr], requires_grad=True)
loss = -F.logsigmoid(t[0] - t[1])
loss.backward()
log(f'autograd: loss {loss.item():.4f}, d loss / d r(chosen) = {t.grad[0].item():.4f}, d loss / d r(rejected) = {t.grad[1].item():.4f}')
log(f'check: -(1 - P) = {-(1 - p):.4f}')
R['pair'] = dict(rc=rc, rr=rr, p=p, loss=-math.log(p), loss_wrong=-math.log(sig(-d)), grad=-(1 - p))
gaps = np.linspace(-4, 4, 81)
R['curve'] = dict(gap=gaps.tolist(), loss=[-math.log(sig(g)) for g in gaps], grad=[-(1 - sig(g)) for g in gaps])
for g in [-2, -1, 0, 1, 2, 4]:
    log(f'  gap {g:+d}: P {sig(g):.3f}  loss {-math.log(sig(g)):.3f}  gradient on r(chosen) {-(1 - sig(g)):+.3f}')

log('\n== 2. adding a constant to every reward ==')
for c in [0, 10, -7.5]:
    log(f'  shift {c:+6.1f}: rewards ({rc + c:.1f}, {rr + c:.1f}) -> loss {-math.log(sig((rc + c) - (rr + c))):.4f}')

log('\n== 3. a ranking of K answers ==')
for K in [2, 4, 9]:
    log(f'  K = {K}: {K * (K - 1) // 2} pairs')
ranked = {'A': 2.1, 'B': 1.2, 'C': 0.9, 'D': -0.5}   # labeler ranked A > B > C > D; these are the model's rewards
pairs = list(itertools.combinations(ranked, 2))
losses = []
for w, l in pairs:
    li = -math.log(sig(ranked[w] - ranked[l]))
    losses.append(li)
    log(f'  {w} > {l}: gap {ranked[w] - ranked[l]:+.1f}, loss {li:.4f}')
log(f'  mean over the {len(pairs)} pairs (the 1/C(K,2) in InstructGPT\'s equation): {np.mean(losses):.4f}')
R['rank'] = dict(rewards=ranked, pairs=[f'{w}>{l}' for w, l in pairs], losses=losses, mean=float(np.mean(losses)))

log('\n== 4. the Elo link ==')
k = 400 / math.log(10)
log(f'Elo: P = 1 / (1 + 10^(-(R_A - R_B) / 400)) = sigmoid((R_A - R_B) * ln(10) / 400)')
log(f'so a reward gap of 1 equals {k:.1f} Elo points; our gap of {d:.1f} = {d * k:.0f} Elo points; 100 Elo points = gap {100 / k:.3f} -> P {sig(100 / k):.3f}')
R['elo'] = dict(per_unit=k, gap_elo=d * k)

log('\n== 5. label noise (Christiano et al. 2017: 10% chance of a random answer) ==')
for g in [1, 3, 6, 10]:
    pc = 0.1 * 0.5 + 0.9 * sig(g)
    log(f'  gap {g:2d}: plain P {sig(g):.5f}, loss if the label is flipped {-math.log(1 - sig(g)):6.2f} | '
        f'noisy P {pc:.4f}, loss if flipped {-math.log(1 - pc):.2f}')
log(f'  the flipped-label loss can never exceed -log(0.05) = {-math.log(0.05):.2f}')

log('\n== 6. ratings or comparisons? ==')
rng = np.random.default_rng(0)
n_items, n_raters, per_rater = 200, 20, 60          # each rater sees 60 items -> 1,200 ratings, or 600 comparisons
rank = lambda v: np.argsort(np.argsort(v))
sp = lambda a, b: np.corrcoef(rank(a), rank(b))[0, 1]


def simulate(drift):
    """drift = 0: each rater has a fixed offset and scale. drift > 0: the rater's offset also wanders from item to item
    (a random walk), the way absolute judgements are pulled by whatever was seen just before."""
    q = rng.normal(size=n_items)                     # the true quality of each answer
    off = rng.normal(0, 1.0, n_raters)               # rater j is harsh or generous ...
    scl = rng.uniform(0.3, 2.0, n_raters)            # ... and uses a narrow or a wide part of the scale
    sums, cnt = np.zeros(n_items), np.zeros(n_items)
    W, L = [], []
    for j in range(n_raters):
        items = rng.choice(n_items, per_rater, replace=False)
        walk = off[j] + np.cumsum(rng.normal(0, drift, per_rater))
        walk[1::2] = walk[0::2]                      # the two answers of a side-by-side pair are judged at the same moment
        seen = scl[j] * q[items] + walk + rng.normal(0, 0.7, per_rater)          # what rater j perceives
        rating = np.clip(np.round(5.5 + 1.5 * seen), 1, 10)                      # written down on a 1..10 scale
        sums[items] += rating; cnt[items] += 1
        for a, b in zip(range(0, per_rater, 2), range(1, per_rater, 2)):        # the same 60 items as 30 comparisons
            w, l = (a, b) if seen[a] > seen[b] else (b, a)
            W.append(items[w]); L.append(items[l])
    mean_rating = np.where(cnt > 0, sums / np.maximum(cnt, 1), sums.sum() / cnt.sum())
    s = torch.zeros(n_items, requires_grad=True)    # Bradley-Terry: one strength per answer, fitted to the comparisons
    opt = torch.optim.Adam([s], lr=0.05)
    Wt, Lt = torch.tensor(np.array(W)), torch.tensor(np.array(L))
    for _ in range(300):
        opt.zero_grad(); l = -F.logsigmoid(s[Wt] - s[Lt]).mean() + 1e-3 * (s ** 2).mean(); l.backward(); opt.step()
    return sp(mean_rating, q), sp(s.detach().numpy(), q)


log(f'{n_items} answers, {n_raters} raters (own offset and scale), each sees {per_rater} answers: '
    f'{n_raters * per_rater} ratings or {n_raters * per_rater // 2} comparisons; 50 repeats')
R['ratings_vs_comparisons'] = {}
for drift in [0.0, 0.3, 0.6]:
    res = np.array([simulate(drift) for _ in range(50)])
    log(f'  drift {drift:.1f}: rank correlation with the truth | mean rating {res[:, 0].mean():.3f} (sd {res[:, 0].std():.3f}) | '
        f'Bradley-Terry {res[:, 1].mean():.3f} (sd {res[:, 1].std():.3f}) | comparisons better in {int((res[:, 1] > res[:, 0]).sum())}/50')
    R['ratings_vs_comparisons'][str(drift)] = dict(rating=res[:, 0].tolist(), bt=res[:, 1].tolist())
log('\n== 7. process reward: scoring a solution from its steps (illustrative step probabilities) ==')
sols = {'right answer': [0.98, 0.96, 0.97, 0.95, 0.94], 'one wrong step': [0.98, 0.95, 0.97, 0.30, 0.90]}
R['prm'] = {}
for nm, ps in sols.items():
    prod = float(np.prod(ps))
    log(f'  {nm:15s} steps {ps}: product {prod:.3f}, minimum {min(ps):.2f}')
    R['prm'][nm] = dict(steps=ps, product=prod, minimum=min(ps))
save('ch7_bt', R)
