"""Part 2 (GRPO), the maths without a model: Equations 1 to 3 on small numbers you can check by hand.
  1. Equation 3, group-normalised advantages, on 4 answers (and the case where all rewards are equal).
  2. The ratio, the clip and the min of Equation 1, for A > 0 and A < 0, with the gradient of each.
  3. Equation 2, the k3 estimator of the KL: never negative, and its average matches the true KL.
  4. Method 1 (our functions) against Method 2 (TRL's GRPOTrainer code) on the same tensors.
  5. Why the ratio is taken per token and not for a whole answer.
Writes results/r1_part2_math_stdout.txt and results/r1_part2_math.json."""
import math
import torch
from common import Log, save
from r1_part2_grpo import group_advantages, ratio, clipped_surrogate, k3_kl, grpo_loss
from r1_part2_trl import trl_loss, trl_advantages

log = Log('r1_part2_math')
R = {}
torch.manual_seed(0)
f = lambda t: '[' + ', '.join(f'{v:.3f}' for v in t.tolist()) + ']'

log('== 1. Equation 3: advantages inside one group of G = 4 answers')
for rw in ([1., 0., 0., 1.], [1., 0., 0., 0.]):
    r = torch.tensor(rw)
    pop = group_advantages(r, 4, unbiased=False, eps=0.0)
    smp = group_advantages(r, 4, unbiased=True, eps=0.0)
    log(f'rewards {rw}: mean {r.mean():.3f}, std (divide by G) {r.std(unbiased=False):.3f}, '
        f'std (divide by G-1) {r.std():.3f}')
    log(f'   A, std divides by G:   {f(pop)}   sum {pop.sum():.3f}')
    log(f'   A, std divides by G-1: {f(smp)}   sum {smp.sum():.3f}')
    R[f'adv_{rw}'] = dict(pop=pop.tolist(), sample=smp.tolist())
for rw in ([1., 1., 1., 1.], [0., 0., 0., 0.]):
    r = torch.tensor(rw)
    log(f'rewards {rw}: std {r.std():.3f}; no eps: {f(group_advantages(r, 4, eps=0.0))}; '
        f'eps 1e-4: {f(group_advantages(r, 4, eps=1e-4))}')

log('')
log('== 2. One token of Equation 1: min(rho*A, clip(rho, 0.8, 1.2)*A), epsilon = 0.2')
log('  rho  |  A=+1: term   d/d(rho) |  A=-1: term   d/d(rho)')
rows = []
for rv in [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5]:
    out = []
    for a in (1.0, -1.0):
        rho = torch.tensor([[rv]], requires_grad=True)
        term = clipped_surrogate(rho, torch.tensor([a]), 0.2)
        term.sum().backward()
        out += [term.item(), rho.grad.item()]
    rows.append([rv] + out)
    log(f'  {rv:.1f}  |  {out[0]:+.2f}       {out[1]:+.1f}     |  {out[2]:+.2f}       {out[3]:+.1f}')
R['clip_table'] = rows

log('')
log('== 3. Equation 2, the k3 estimator: x - log x - 1 with x = pi_ref / pi_theta')
log('   x      k1 = -log x    k3 = x - log x - 1')
for x in [0.25, 0.5, 0.9, 1.0, 1.1, 2.0, 4.0]:
    log(f'  {x:4.2f}     {-math.log(x) + 0.0:+.4f}         {x - math.log(x) - 1:.4f}')
p_theta = torch.tensor([0.50, 0.20, 0.15, 0.10, 0.05])   # the policy's next-token probabilities, 5 tokens
p_ref = torch.tensor([0.40, 0.30, 0.10, 0.10, 0.10])     # the reference model's probabilities
true_kl = (p_theta * (p_theta / p_ref).log()).sum()
xs = p_ref / p_theta
exact_k3 = (p_theta * (xs - xs.log() - 1)).sum()
log(f'pi_theta = {f(p_theta)}')
log(f'pi_ref   = {f(p_ref)}')
log(f'true KL(pi_theta || pi_ref) = sum pi_theta log(pi_theta/pi_ref) = {true_kl:.5f}')
log(f'exact average of k3 under pi_theta                         = {exact_k3:.5f}')
g = torch.Generator().manual_seed(0)
idx = torch.multinomial(p_theta, 100_000, replacement=True, generator=g)
lt, lr = p_theta.log()[idx], p_ref.log()[idx]
k1 = lt - lr                                               # -log x
k3 = k3_kl(lt, lr)
log(f'100,000 tokens sampled from pi_theta: mean k1 {k1.mean():.5f} (std {k1.std():.4f}, '
    f'{(k1 < 0).float().mean() * 100:.1f}% of samples below 0)')
log(f'                                      mean k3 {k3.mean():.5f} (std {k3.std():.4f}, smallest {k3.min():.4f})')
R['kl'] = dict(true=true_kl.item(), exact_k3=exact_k3.item(), mc_k1=k1.mean().item(), mc_k3=k3.mean().item(),
               std_k1=k1.std().item(), std_k3=k3.std().item())

log('')
log('== 4. Method 1 (from scratch) vs Method 2 (TRL GRPOTrainer._compute_loss), same tensors')
G, T = 4, 6
rewards = torch.tensor([1., 0., 0., 1., 0., 0., 0., 1.])   # 2 questions x G = 4 answers
mask = torch.ones(8, T)
for i, n in enumerate([6, 4, 5, 3, 6, 2, 5, 4]):           # answers of different lengths: pad the rest
    mask[i, n:] = 0
logp_old = -torch.rand(8, T) * 3                            # (8, 6): log-probs under the sampling policy
logp_ref = logp_old + 0.2 * torch.randn(8, T)               # (8, 6): the reference policy, a little different
logp1 = (logp_old + 0.3 * torch.randn(8, T)).requires_grad_()   # (8, 6): the policy after some updates
logp2 = logp1.detach().clone().requires_grad_()
A1 = group_advantages(rewards, G)
A2 = trl_advantages(rewards, G)
log(f'advantages, ours: {f(A1)}')
log(f'advantages, TRL:  {f(A2)}')
log(f'max |difference| of advantages: {(A1 - A2).abs().max():.2e}')
L1 = grpo_loss(logp1, logp_old, logp_ref, A1, mask, epsilon=0.2, beta=0.04)
L2, metrics = trl_loss(logp2, logp_old, logp_ref, A2, mask, epsilon=0.2, beta=0.04)
L1.backward(); L2.backward()
log(f'loss, ours: {L1.item():.8f}')
log(f'loss, TRL:  {L2.item():.8f}')
log(f'|difference| of the loss: {abs(L1.item() - L2.item()):.2e}; max |difference| of the gradients: '
    f'{(logp1.grad - logp2.grad).abs().max():.2e}')
rho = ratio(logp1.detach(), logp_old)
Au = A1.unsqueeze(1)
ours_clip = ((((rho < 0.8) & (Au < 0)) | ((rho > 1.2) & (Au > 0))).float() * mask).sum() / mask.sum()
log(f'share of tokens in the flat (clipped) region: ours {ours_clip:.4f}, TRL metric {metrics["clip_ratio/region_mean"][0]:.4f}')
log(f'agree to 1e-6: {abs(L1.item() - L2.item()) < 1e-6 and (logp1.grad - logp2.grad).abs().max().item() < 1e-6}')
R['agree'] = dict(loss_ours=L1.item(), loss_trl=L2.item(), grad_diff=(logp1.grad - logp2.grad).abs().max().item())

log('')
log('== 5. One ratio for a whole answer, or one per token')
for per_tok, n in [(1.01, 300), (0.99, 300), (1.01, 1000)]:
    log(f'each of {n} tokens has ratio {per_tok}: ratio of the whole answer = {per_tok}^{n} = {per_tok ** n:.4g}')
save('r1_part2_math', R)
