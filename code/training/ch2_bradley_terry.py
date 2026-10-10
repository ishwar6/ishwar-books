"""Chapter 2, experiment 1: fit a tiny reward model to pairwise preferences with the Bradley-Terry loss,
the way Christiano et al. (2017) and every RLHF paper since fit a reward model.

The setting is a toy on purpose, so we can check the answer. Each "response" is described by 4 numbers
(is it correct, how detailed, how polite, how long). A simulated labeler has a hidden true reward and, shown two
responses, prefers A with probability sigmoid(r_A - r_B). The reward model only ever sees which one was preferred.
Question: can it recover the hidden reward from the choices alone?"""
import numpy as np
import torch
from common import Log, save

log = Log('ch2_bradley_terry')
rng = np.random.default_rng(0)
torch.manual_seed(0)

FEATURES = ['correct', 'detail', 'polite', 'length']
TRUE_W = np.array([3.0, 1.5, 1.0, -0.5])      # the labeler's hidden taste: correctness matters most, length a little bad


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ---- 1. the Bradley-Terry formula on two numbers
log('== 1. Bradley-Terry on two responses ==')
rA, rB = 1.2, 0.4
log(f'r_A = {rA}, r_B = {rB}  ->  P(A preferred) = sigmoid({rA} - {rB}) = sigmoid({rA - rB:.1f}) = {sigmoid(rA - rB):.4f}')
log(f'add 10 to both: sigmoid({rA + 10} - {rB + 10}) = {sigmoid((rA + 10) - (rB + 10)):.4f}   (only the difference matters)')
for d in [0, 0.5, 1, 2, 3, 5]:
    log(f'  reward gap {d:>3}:  P(better one preferred) = {sigmoid(d):.3f}')
gap_table = {d: float(sigmoid(d)) for d in [0, 0.5, 1, 2, 3, 5]}

# ---- 2. make a dataset of responses and noisy human-like comparisons
N_RESP = 200
X = np.column_stack([rng.integers(0, 2, N_RESP),          # correct: 0 or 1
                     rng.random(N_RESP),                   # detail 0..1
                     rng.random(N_RESP),                   # polite 0..1
                     rng.random(N_RESP)])                  # length 0..1
r_true = X @ TRUE_W


def make_pairs(n):
    i, j = rng.integers(0, N_RESP, n), rng.integers(0, N_RESP, n)
    keep = i != j
    i, j = i[keep], j[keep]
    a_wins = rng.random(len(i)) < sigmoid(r_true[i] - r_true[j])     # the labeler is noisy, like real people
    winner = np.where(a_wins, i, j)
    loser = np.where(a_wins, j, i)
    return winner, loser


w_tr, l_tr = make_pairs(1000)
w_te, l_te = make_pairs(2000)
log('\n== 2. data ==')
log(f'{N_RESP} responses, {len(w_tr)} training comparisons, {len(w_te)} test comparisons')
ex = w_tr[0], l_tr[0]
desc = lambda k: ', '.join(f'{f}={X[k][n]:.2f}' for n, f in enumerate(FEATURES)) + f'  (true reward {r_true[k]:.2f})'
log(f'example comparison: chosen  {desc(ex[0])}')
log(f'                    rejected {desc(ex[1])}')
# the best any model can do: predict "the truly better one wins". Labels are noisy, so this is below 100%.
ceiling = np.mean(r_true[w_te] > r_true[l_te])
log(f'test accuracy of the TRUE reward (the ceiling, because labels are noisy): {ceiling:.3f}')

# ---- 3. the reward model: a linear score r(x) = w . x, trained with the Bradley-Terry loss
Xt = torch.tensor(X, dtype=torch.float32)
w = torch.zeros(4, requires_grad=True)                     # starts knowing nothing: every response scores 0
opt = torch.optim.Adam([w], lr=0.05)
W_tr, L_tr = torch.tensor(w_tr), torch.tensor(l_tr)


def loss_fn(win, lose):
    r = Xt @ w
    return -torch.nn.functional.logsigmoid(r[win] - r[lose]).mean()   # -log sigmoid(r_chosen - r_rejected)


def accuracy(win, lose):
    r = (Xt @ w).detach().numpy()
    return float(np.mean((r[win] > r[lose]) + 0.5 * (r[win] == r[lose])))   # a tie counts as half right


log('\n== 3. training: loss = -log sigmoid(r_chosen - r_rejected), averaged over pairs ==')
curve = []
for step in range(401):
    loss = loss_fn(W_tr, L_tr)
    if step % 50 == 0:
        acc = accuracy(w_te, l_te)
        curve.append(dict(step=step, loss=float(loss), test_acc=acc, w=w.detach().numpy().round(3).tolist()))
        log(f'step {step:>3}  loss {loss.item():.4f}  test acc {acc:.3f}  w = {np.round(w.detach().numpy(), 2)}')
    opt.zero_grad()
    loss.backward()
    opt.step()

w_hat = w.detach().numpy()
log('\n== 4. what it learned ==')
for f, t, h in zip(FEATURES, TRUE_W, w_hat):
    log(f'{f:>8}: true weight {t:+.2f}   learned {h:+.2f}')
corr = np.corrcoef(X @ w_hat, r_true)[0, 1]
log(f'correlation between learned and true reward over the 200 responses: {corr:.4f}')
log(f'start loss ln 2 = {np.log(2):.4f} (every pair a coin flip); final loss {curve[-1]["loss"]:.4f}')

# ---- 5. how many comparisons are needed?
log('\n== 5. accuracy vs number of comparisons ==')
by_n = []
for n in [25, 50, 100, 250, 500, 1000, 2000]:
    wn, ln = make_pairs(n)
    v = torch.zeros(4, requires_grad=True)
    o = torch.optim.Adam([v], lr=0.05)
    for _ in range(400):
        r = Xt @ v
        l = -torch.nn.functional.logsigmoid(r[torch.tensor(wn)] - r[torch.tensor(ln)]).mean()
        o.zero_grad(); l.backward(); o.step()
    rr = (Xt @ v).detach().numpy()
    a = float(np.mean(rr[w_te] > rr[l_te]))
    vw = v.detach().numpy()
    by_n.append(dict(n=len(wn), test_acc=a, w=vw.round(3).tolist()))
    log(f'{len(wn):>5} comparisons -> test accuracy {a:.3f}   learned w = {np.round(vw, 2)}')

save('ch2_bradley_terry', dict(gap_table=gap_table, curve=curve, true_w=TRUE_W.tolist(), learned_w=w_hat.tolist(),
                               ceiling=float(ceiling), corr=float(corr), by_n=by_n, rA=rA, rB=rB))
