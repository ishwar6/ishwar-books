"""Background page, experiments (c) and (d): the reinforcement-learning pieces that RLHF and R1 stand on.

1. REINFORCE on a tiny "language model" that picks one of 4 answers (a bandit): the gradient estimate by hand in
   numpy (Method 1) against torch autograd (Method 2); its mean and its variance with and without a baseline;
   then training runs over many seeds.
2. The reward-model loss of RLHF (Bradley-Terry on a pair): by hand vs F.logsigmoid and
   F.binary_cross_entropy_with_logits; and four answers ranked by a labeler -> six pairs -> learned scores.
3. The per-token KL penalty of RLHF on three tokens.
4. The PPO clipped objective on a few ratios: by hand vs torch, with the gradient that autograd sees.
5. A memory estimate: why a critic (value model) costs a second big model.
Output: results/r1_partb_rl_stdout.txt and results/r1_partb_rl.json. CPU only, no model is loaded."""
import math
import numpy as np
import torch
import torch.nn.functional as F
import common

log = common.Log('r1_partb_rl')
R = {}

# ====================================================================== 1. REINFORCE on a 4-answer bandit
log('== 1. REINFORCE: a policy picks one of 4 answers; only answer 2 is right (reward 1, else 0)')
theta = np.array([0.5, 0.0, -0.5, 0.0])              # (4,) the logits: the "model" before training
reward = np.array([0.0, 0.0, 1.0, 0.0])               # (4,) reward of each answer


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def grad_log_pi(theta, a):
    """Method 1: d log pi(a) / d theta for a softmax policy = one_hot(a) - pi.   (4,)"""
    g = -softmax(theta)
    g[a] += 1.0
    return g


pi = softmax(theta)
J = (pi * reward).sum()                                # expected reward = P(right answer)
exact = pi * (reward - J)                              # the true gradient of J, worked out by hand
log(f'  pi = {np.round(pi, 4).tolist()}   expected reward J = {J:.4f}')
log(f'  exact gradient dJ/dtheta = pi * (r - J) = {np.round(exact, 4).tolist()}')

# Method 2: autograd for one sampled answer and for the exact gradient
th = torch.tensor(theta, requires_grad=True)
lp = torch.log_softmax(th, -1)[2]
lp.backward()
log(f'  grad log pi(a=2): ours {np.round(grad_log_pi(theta, 2), 4).tolist()}   autograd {np.round(th.grad.numpy(), 4).tolist()}')
th.grad = None
(torch.softmax(th, -1) * torch.tensor(reward)).sum().backward()
log(f'  exact gradient by autograd           {np.round(th.grad.numpy(), 4).tolist()}')

rng = np.random.default_rng(0)
S = 200_000
acts = rng.choice(4, size=S, p=pi)
glp = np.eye(4)[acts] - pi                             # (S, 4): grad log pi for every sample
stats = {}
for shift in [0.0, 5.0]:
    r = reward[acts] + shift                           # the same game; with shift 5 every reward is 5 or 6
    Jsh = J + shift
    for name, b in [('no baseline', 0.0), ('baseline b = J', Jsh)]:
        g = (r - b)[:, None] * glp                      # (S, 4): one REINFORCE estimate per sample
        mean = g.mean(0)
        var = g.var(0).sum()                            # total variance over the 4 entries
        stats[f'{shift}/{name}'] = dict(mean=mean.tolist(), var=float(var))
        log(f'  rewards +{shift:.0f}, {name:15}: mean of {S:,} estimates {np.round(mean, 4).tolist()}  variance {var:.4f}')
R['reinforce'] = dict(pi=pi.tolist(), J=J, exact=exact.tolist(), stats=stats)


def train(baseline, shift, seed, steps=300, lr=0.1):
    """Plain REINFORCE, one sample per step. baseline: use a running mean of the reward."""
    rng = np.random.default_rng(seed)
    th, avg, hist = theta.copy(), 0.0, []
    for t in range(steps):
        p = softmax(th)
        a = rng.choice(4, p=p)
        r = reward[a] + shift
        b = avg if baseline else 0.0
        th += lr * (r - b) * grad_log_pi(th, a)         # gradient ascent on the reward
        avg += 0.1 * (r - avg)                          # running mean of the reward
        hist.append(softmax(th)[2])
    return hist


log('  training: 300 steps, one sample per step, learning rate 0.1, 200 seeds; P(right answer) after the step')
curves = {}
for shift in [0.0, 5.0]:
    for bl in [False, True]:
        H = np.array([train(bl, shift, s) for s in range(200)])     # (200 seeds, 300 steps)
        key = f'rewards +{shift:.0f}, ' + ('baseline' if bl else 'no baseline')
        curves[key] = dict(mean=H.mean(0)[::10].tolist(), p10=np.percentile(H, 10, 0)[::10].tolist(),
                           p90=np.percentile(H, 90, 0)[::10].tolist())
        log(f'  {key:28}: step 50 mean {H[:, 49].mean():.3f}, step 300 mean {H[:, -1].mean():.3f}, '
            f'worst seed {H[:, -1].min():.3f}, seeds below 0.5 at the end {int((H[:, -1] < 0.5).sum())}/200')
R['curves'] = curves

# ====================================================================== 2. reward model loss
log('')
log('== 2. Reward model (Bradley-Terry): loss = -log sigmoid(r(x, y_w) - r(x, y_l))')
for rw, rl in [(1.2, 0.4), (0.4, 1.2), (3.0, 0.0)]:
    ours = -math.log(1 / (1 + math.exp(-(rw - rl))))                               # Method 1
    lib = -F.logsigmoid(torch.tensor(rw - rl)).item()                              # Method 2
    lib2 = F.binary_cross_entropy_with_logits(torch.tensor(rw - rl), torch.tensor(1.0)).item()
    log(f'  r_w = {rw:.1f}, r_l = {rl:.1f}: P(w beats l) = {1 / (1 + math.exp(-(rw - rl))):.3f}, '
        f'loss ours {ours:.4f}  logsigmoid {lib:.4f}  BCE-with-logits {lib2:.4f}')

# A labeler ranks K = 4 answers: D > B > A > C. That gives K(K-1)/2 = 6 pairs. Learn one score per answer.
names = ['A', 'B', 'C', 'D']
rank = ['D', 'B', 'A', 'C']
pairs = [(rank[i], rank[j]) for i in range(4) for j in range(i + 1, 4)]           # (winner, loser)
r = torch.zeros(4, requires_grad=True)
opt = torch.optim.SGD([r], lr=0.5)
for step in range(300):
    w = torch.tensor([names.index(a) for a, _ in pairs])
    l = torch.tensor([names.index(b) for _, b in pairs])
    loss = -F.logsigmoid(r[w] - r[l]).mean()
    opt.zero_grad(); loss.backward(); opt.step()
log(f'  ranking {" > ".join(rank)} -> {len(pairs)} pairs: {pairs}')
log(f'  learned scores {dict(zip(names, [round(v, 2) for v in r.tolist()]))}, final loss {loss.item():.3f}')
rr = r.detach()
log(f'  add 10 to every score: loss {(-F.logsigmoid((rr + 10)[w] - (rr + 10)[l]).mean()).item():.3f} (only differences count)')
R['rm'] = dict(scores=dict(zip(names, rr.tolist())))

# ====================================================================== 3. KL penalty per token
log('')
log('== 3. RLHF reward with a per-token KL penalty: r_t = -beta * log(pi(t) / pi_ref(t)), plus r_RM at the end')
beta, r_rm = 0.02, 1.5
p_pol = [0.60, 0.90, 0.30]
p_ref = [0.30, 0.90, 0.40]
pen = [-beta * math.log(a / b) for a, b in zip(p_pol, p_ref)]
lib = (-beta * (torch.log(torch.tensor(p_pol)) - torch.log(torch.tensor(p_ref)))).tolist()
log(f'  beta = {beta} (InstructGPT, App. C.4), reward model score {r_rm} on the last token')
for t in range(3):
    log(f'  token {t + 1}: pi {p_pol[t]:.2f}  pi_ref {p_ref[t]:.2f}  penalty ours {pen[t]:+.4f}  torch {lib[t]:+.4f}')
log(f'  total reward = {r_rm} + ({" + ".join(f"{v:+.4f}" for v in pen)}) = {r_rm + sum(pen):.4f}')

# ====================================================================== 4. PPO clipped objective
log('')
log('== 4. PPO clipped objective, epsilon = 0.2: min(ratio * A, clip(ratio, 0.8, 1.2) * A)')
eps = 0.2


def ppo_term(ratio, A, eps=0.2):
    """Method 1: one term of L_CLIP by hand."""
    clipped = min(max(ratio, 1 - eps), 1 + eps)
    return min(ratio * A, clipped * A)


ratios = [0.5, 0.9, 1.0, 1.1, 1.5]
rows = []
for A in [1.0, -1.0]:
    rt = torch.tensor(ratios, requires_grad=True)
    lib = torch.min(rt * A, torch.clamp(rt, 1 - eps, 1 + eps) * A)               # Method 2
    lib.sum().backward()
    for k, q in enumerate(ratios):
        ours = ppo_term(q, A)
        rows.append(dict(A=A, ratio=q, ours=ours, lib=lib[k].item(), grad=rt.grad[k].item()))
        log(f'  A = {A:+.0f}, ratio {q:.1f}: unclipped {q * A:+.2f}  ours {ours:+.2f}  torch {lib[k].item():+.2f}  '
            f'gradient {rt.grad[k].item() + 0.0:+.0f}')
R['ppo'] = rows

# ====================================================================== 5. memory of the four models
log('')
log('== 5. Memory estimate for RLHF with PPO on a 7B policy (rule of thumb, not a measurement)')
P = 7e9
train_b = 16      # bytes per trained parameter: bf16 weights 2 + bf16 grads 2 + fp32 master and Adam moments 12
infer_b = 2       # bytes per frozen parameter: bf16 weights only
mem = dict(policy=P * train_b, critic=P * train_b, reference=P * infer_b, reward=P * infer_b)
for k, v in mem.items():
    log(f'  {k:10}: {v / 1e9:5.0f} GB  ({"trained" if k in ("policy", "critic") else "frozen"})')
tot = sum(mem.values())
log(f'  total {tot / 1e9:.0f} GB; without the critic {(tot - mem["critic"]) / 1e9:.0f} GB '
    f'({100 * mem["critic"] / tot:.0f}% less), before activations and the KV cache of generation')
R['mem'] = {k: v / 1e9 for k, v in mem.items()}

common.save('r1_partb_rl', R)
