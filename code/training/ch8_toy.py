"""Chapter 8, small exact experiments behind the PPO sections.

1. The clipped objective for one token, four cases, with the gradient from autograd.
2. Re-using one batch for many gradient steps: the REINFORCE loss, the unclipped ratio objective (TRPO's surrogate
   without its constraint) and PPO's clipped objective, on a six-token bandit with noisy rewards, 500 seeds.
3. Memory arithmetic for the four models of RLHF."""
import math
import numpy as np
import torch
from common import Log, save

log = Log('ch8_toy')
out = {}

# ------------------------------------------------------------------ 1. one token, four cases
log('== 1. the clipped objective for one token (epsilon = 0.2)')
cases = [('A > 0, ratio went up a lot', -2.303, -2.040, +1.5),
         ('A > 0, ratio went down', -2.303, -2.408, +1.5),
         ('A < 0, ratio went down a lot', -2.303, -2.660, -0.8),
         ('A < 0, ratio went up', -2.303, -1.897, -0.8)]
rows = []
for name, old, new, A in cases:
    lp = torch.tensor(new, requires_grad=True)
    r = torch.exp(lp - old)
    unclipped = r * A
    clipped = torch.clamp(r, 0.8, 1.2) * A
    obj = torch.min(unclipped, clipped)
    obj.backward()
    row = dict(case=name, old_lp=old, new_lp=new, p_old=math.exp(old), p_new=math.exp(new), A=A, ratio=r.item(),
               unclipped=unclipped.item(), clipped=clipped.item(), objective=obj.item(), grad=lp.grad.item())
    rows.append(row)
    log(f'  {name:30s} p_old {row["p_old"]:.3f} p_new {row["p_new"]:.3f} ratio {row["ratio"]:.3f}  A {A:+.1f}  '
        f'r*A {row["unclipped"]:+.3f}  clip(r)*A {row["clipped"]:+.3f}  min {row["objective"]:+.3f}  '
        f'd objective / d log p {row["grad"]:+.3f}')
out['cases'] = rows

# ------------------------------------------------------------------ 2. many steps on one batch
log('\n== 2. one batch, many gradient steps: six-token bandit, true mean rewards below, noise sd 1.0')
MU = np.array([0.3, 0.2, 0.1, 0.0, -0.1, -0.2])
logits0 = np.array([0.0, 0.2, 0.4, 0.4, 0.2, 0.0])       # the old policy slightly prefers the middling tokens
p0 = np.exp(logits0) / np.exp(logits0).sum()
log(f'  true means {MU.tolist()}  old policy {np.round(p0, 3).tolist()}  expected reward {p0 @ MU:+.4f}')
B, K, LR, SEEDS = 32, 60, 0.1, 500


def run(kind, seed):
    g = np.random.default_rng(seed)
    a = g.choice(6, size=B, p=p0)
    r = MU[a] + g.normal(0, 1.0, B)
    A = torch.tensor(r - r.mean(), dtype=torch.float32)
    at = torch.tensor(a)
    th = torch.tensor(logits0, dtype=torch.float32, requires_grad=True)
    old_lp = torch.log_softmax(torch.tensor(logits0, dtype=torch.float32), 0)[at]
    opt = torch.optim.Adam([th], lr=LR)
    tr = []
    for k in range(K + 1):
        p = torch.softmax(th, 0)
        lp = torch.log(p)[at]
        ratio = torch.exp(lp - old_lp)
        with torch.no_grad():
            pn = p.numpy()
            tr.append((float(pn @ MU), float((p0 * np.log(p0 / pn)).sum()), float(((ratio - 1).abs() > 0.2).float().mean())))
        if k == K:
            break
        if kind == 'reinforce':
            loss = -(lp * A).mean()
        elif kind == 'ratio':
            loss = -(ratio * A).mean()
        else:
            loss = -torch.min(ratio * A, torch.clamp(ratio, 0.8, 1.2) * A).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return np.array(tr)


res = {}
for kind in ['reinforce', 'ratio', 'clip']:
    T = np.stack([run(kind, s) for s in range(SEEDS)])     # seeds x steps x (reward, kl, clipped share)
    res[kind] = dict(reward=T[:, :, 0].mean(0).tolist(), kl=T[:, :, 1].mean(0).tolist(),
                     clipped=T[:, :, 2].mean(0).tolist(), worse=(T[:, :, 0] < p0 @ MU).mean(0).tolist(),
                     reward_p10=np.percentile(T[:, :, 0], 10, axis=0).tolist())
    for k in [1, 5, 10, 20, 40, 60]:
        log(f'  {kind:9s} after {k:2d} steps: true expected reward {res[kind]["reward"][k]:+.3f} '
            f'(10th percentile {res[kind]["reward_p10"][k]:+.3f}), KL(old||new) {res[kind]["kl"][k]:.3f}, '
            f'worse than before in {res[kind]["worse"][k]:.0%} of seeds, samples outside [0.8, 1.2] {res[kind]["clipped"][k]:.0%}')
out['bandit'] = dict(mu=MU.tolist(), p0=p0.tolist(), B=B, K=K, lr=LR, seeds=SEEDS, start=float(p0 @ MU), runs=res)

# ------------------------------------------------------------------ 2b. the full loop: sample, update K times, repeat
ITERS, KK = 40, 10
log(f'\n== 2b. the loop: {ITERS} iterations, each samples a fresh batch of {B} from the current policy and takes K updates on it ({SEEDS} seeds)')


def loop(kind, K, seed):
    g = np.random.default_rng(10_000 + seed)
    th = torch.tensor(logits0, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([th], lr=LR)
    tr = []
    for it in range(ITERS + 1):
        with torch.no_grad():
            p = torch.softmax(th, 0).numpy().astype(np.float64); p /= p.sum()
        tr.append((float(p @ MU), float(p[0]), float(p.max())))
        if it == ITERS:
            break
        a = g.choice(6, size=B, p=p)
        r = MU[a] + g.normal(0, 1.0, B)
        A = torch.tensor(r - r.mean(), dtype=torch.float32)
        at = torch.tensor(a)
        old_lp = torch.log_softmax(th.detach(), 0)[at]
        for k in range(K):
            lp = torch.log_softmax(th, 0)[at]
            ratio = torch.exp(lp - old_lp)
            if kind == 'reinforce':
                loss = -(lp * A).mean()
            elif kind == 'ratio':
                loss = -(ratio * A).mean()
            else:
                loss = -torch.min(ratio * A, torch.clamp(ratio, 0.8, 1.2) * A).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return np.array(tr)


res2 = {}
for kind, K in [('reinforce', 1), ('reinforce', KK), ('ratio', KK), ('clip', KK)]:
    key = f'{kind}_K{K}'
    T = np.stack([loop(kind, K, s) for s in range(SEEDS)])
    stuck = ((T[:, -1, 2] > 0.95) & (T[:, -1, 1] < 0.5)).mean()      # near-deterministic on a token that is not the best
    res2[key] = dict(reward=T[:, :, 0].mean(0).tolist(), p10=np.percentile(T[:, :, 0], 10, axis=0).tolist(),
                     p_best=T[:, :, 1].mean(0).tolist(), stuck=float(stuck))
    log(f'  {key:13s} expected reward after 10/20/40 iterations: ' +
        ' / '.join(f'{res2[key]["reward"][i]:+.3f}' for i in (10, 20, 40)) +
        f'   10th percentile at 40: {res2[key]["p10"][40]:+.3f}   P(best token) at 40: {res2[key]["p_best"][40]:.2f}'
        f'   stuck on a worse token: {stuck:.0%}')
out['loop'] = dict(iters=ITERS, K=KK, runs=res2)

# ------------------------------------------------------------------ 3. memory
log('\n== 3. memory for the four models (weights and optimizer state only; no activations, no KV cache)')
log('  trainable models (policy, value): 16 bytes/parameter = bf16 weights 2 + bf16 grads 2 + fp32 master copy 4 + Adam m 4 + Adam v 4')
log('  frozen models (reference, reward): 2 bytes/parameter (bf16 weights)')
mem = []
for name, n in [('0.5B (our Qwen2.5-0.5B)', 0.494e9), ('7B', 7e9), ('70B', 70e9)]:
    pol = val = 16 * n
    ref = rm = 2 * n
    tot = pol + val + ref + rm
    mem.append(dict(name=name, params=n, policy=pol, value=val, ref=ref, rm=rm, total=tot, sft=16 * n))
    log(f'  {name:24s} policy {pol / 1e9:7.1f} GB  value {val / 1e9:7.1f} GB  reference {ref / 1e9:6.1f} GB  reward {rm / 1e9:6.1f} GB'
        f'  total {tot / 1e9:7.1f} GB  (SFT of the same model: {16 * n / 1e9:.1f} GB, so RLHF needs {tot / (16 * n):.2f}x)')
out['memory'] = mem

save('ch8_toy', out)
