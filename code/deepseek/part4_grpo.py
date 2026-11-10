"""Part 4: GRPO by hand and with PyTorch, on toy numbers you can check.

What it shows (every printed line is saved to results/part4_grpo_stdout.txt, numbers to results/part4_grpo.json):
  1. Rule-based rewards (accuracy + format, as in DeepSeek-R1-Zero) and the language consistency reward (R1).
  2. Group-relative advantages for a group of 4 answers: Method 1 from scratch, Method 2 vectorised torch.
  3. Special groups: all right, one right, three right; population vs sample std; V3.2's mean-only advantage.
  4. The clipped surrogate for one token, for A > 0 and A < 0.
  5. The k3 KL estimator of DeepSeekMath: always >= 0, and its average equals the true KL (Monte Carlo check).
  6. The whole GRPO loss on a toy batch with padding: loops vs torch, same number.
  7. A toy RL run: a 4-way choice trained with GRPO; the probability of the right, well-formatted answer rises.
  8. Cost arithmetic from the R1 paper (Nature version / arXiv v2, Table 7).
CPU only, a few seconds. Run: ~/.venvs/attn/bin/python part4_grpo.py"""
import math, random, re
import torch
from common import Log, save

log = Log('part4_grpo')
out = {}
torch.manual_seed(0)
random.seed(0)

# ---------------------------------------------------------------- 1. rule-based rewards
log('== 1. Rule-based rewards (accuracy + format) ==')
QUESTION = 'What is 7 x 8? Put the final answer in \\boxed{}.'
TRUTH = '56'
group = [
    '<think>7 x 8 = 7 x 4 x 2 = 28 x 2 = 56</think> <answer>\\boxed{56}</answer>',
    '<think>7 x 8 = 54, I think</think> <answer>\\boxed{54}</answer>',
    '<think>7 x 9 = 63</think> <answer>\\boxed{63}</answer>',
    'The answer is 56.',
]


def accuracy_reward(text, truth):
    """1 if the last \\boxed{...} holds the right answer, else 0. No box = no checkable answer = 0."""
    boxes = re.findall(r'\\boxed\{([^}]*)\}', text)
    return 1.0 if boxes and boxes[-1].strip() == truth else 0.0


def format_reward(text):
    """1 if the reasoning sits inside <think>...</think> and the answer inside <answer>...</answer>."""
    return 1.0 if re.fullmatch(r'\s*<think>.+?</think>\s*<answer>.+?</answer>\s*', text, flags=re.S) else 0.0


rewards = []
for i, o in enumerate(group, 1):
    acc, fmt = accuracy_reward(o, TRUTH), format_reward(o)
    rewards.append(acc + fmt)                       # R1-Zero: the two rewards are added with the same weight
    log(f'o{i}: accuracy {acc:.0f} + format {fmt:.0f} = reward {acc + fmt:.0f}   | {o}')
out['rewards'] = rewards


def language_consistency(cot, target='en'):
    """R1's language consistency reward: the share of words in the chain of thought that are in the target language.
    Toy rule: a word counts as English if it has Latin letters, as Chinese if it has CJK characters."""
    words = re.findall(r'[A-Za-z]+|[\u4e00-\u9fff]+', cot)
    eng = sum(1 for w in words if re.match(r'[A-Za-z]', w))
    return eng / len(words)


cot_mixed = 'First square both sides 然后 we get a quartic 方程 so check each root'
cot_clean = 'First square both sides then we get a quartic equation so check each root'
lc_mixed, lc_clean = language_consistency(cot_mixed), language_consistency(cot_clean)
log(f'language consistency, mixed CoT: {lc_mixed:.3f}   clean CoT: {lc_clean:.3f}')
out['lc'] = {'mixed': lc_mixed, 'clean': lc_clean}

# ---------------------------------------------------------------- 2. group-relative advantages
log('\n== 2. Group-relative advantage, A_i = (r_i - mean(r)) / std(r) ==')


def advantages_scratch(r, eps=1e-6):
    """Method 1: plain Python. Population std (divide by G), as in the formula's usual reading."""
    G = len(r)
    mean = sum(r) / G
    std = math.sqrt(sum((x - mean) ** 2 for x in r) / G)
    return mean, std, [(x - mean) / (std + eps) for x in r]


mean, std, A1 = advantages_scratch(rewards)
log(f'Method 1 (from scratch): rewards {rewards}  mean {mean:.4f}  std {std:.4f}')
log('  advantages: ' + ', '.join(f'{a:+.4f}' for a in A1))


def advantages_torch(R, eps=1e-6, unbiased=False):
    """Method 2: a whole batch of groups at once. R has shape (B, G): B questions, G answers each."""
    mean = R.mean(dim=1, keepdim=True)                       # (B, 1)
    std = R.std(dim=1, keepdim=True, unbiased=unbiased)      # (B, 1)
    return (R - mean) / (std + eps)                          # (B, G)


R = torch.tensor([rewards,
                  [1., 0., 0., 0.],       # one right answer of four
                  [1., 1., 1., 0.],       # three right of four
                  [1., 1., 1., 1.]])      # all right: no signal
A2 = advantages_torch(R)
log(f'Method 2 (torch, batch of {R.shape[0]} groups), first row: ' + ', '.join(f'{a:+.4f}' for a in A2[0].tolist()))
log(f'max |Method 1 - Method 2| = {max(abs(a - b) for a, b in zip(A1, A2[0].tolist())):.2e}')
out['adv_group4'] = A1

log('\n== 3. Special groups ==')
names = ['our group [2,1,1,0]', 'one right [1,0,0,0]', 'three right [1,1,1,0]', 'all right [1,1,1,1]']
out['special'] = {}
for n, row, arow in zip(names, R.tolist(), A2.tolist()):
    log(f'{n:24s} -> ' + ', '.join(f'{a:+.3f}' for a in arow))
    out['special'][n] = arow
A_sample = advantages_torch(R[:1], unbiased=True)[0].tolist()
log("torch's default std divides by G-1 (sample std). Our group then gets: " + ', '.join(f'{a:+.4f}' for a in A_sample))
out['adv_group4_sample_std'] = A_sample
A_v32 = (R - R.mean(dim=1, keepdim=True))
log('V3.2 mean-only advantage (no division by std), our group: ' + ', '.join(f'{a:+.3f}' for a in A_v32[0].tolist()))
log('V3.2 mean-only, one right of four:                         ' + ', '.join(f'{a:+.3f}' for a in A_v32[1].tolist()))
out['adv_v32'] = A_v32.tolist()

# ---------------------------------------------------------------- 4. the clipped surrogate
log('\n== 4. Clipped surrogate for one token: min(ratio*A, clip(ratio, 1-eps, 1+eps)*A), eps = 0.2 ==')
EPS = 0.2
ratios = [0.5, 0.8, 1.0, 1.1, 1.2, 1.5, 2.0]


def surrogate(ratio, A, eps=EPS):
    return min(ratio * A, max(1 - eps, min(ratio, 1 + eps)) * A)


log('ratio   A=+1.414   A=-1.414')
out['clip'] = []
for q in ratios:
    sp, sn = surrogate(q, A1[0]), surrogate(q, A1[3])
    out['clip'].append([q, sp, sn])
    log(f'{q:5.2f}   {sp:+8.3f}   {sn:+8.3f}')
# dense curve for the figure
out['clip_curve'] = [[x / 100, surrogate(x / 100, 1.0), surrogate(x / 100, -1.0)] for x in range(30, 201, 2)]

# ---------------------------------------------------------------- 5. the k3 KL estimator
log('\n== 5. KL estimator k3 = pi_ref/pi - log(pi_ref/pi) - 1 ==')


def k3(p_theta, p_ref):
    x = p_ref / p_theta
    return x - math.log(x) - 1


for pt, pr in [(0.5, 0.5), (0.5, 0.4), (0.5, 0.6), (0.9, 0.1), (0.1, 0.9)]:
    log(f'pi_theta {pt:.1f}  pi_ref {pr:.1f}  ->  k3 = {k3(pt, pr):.4f}')
# Monte Carlo: average k3 over tokens sampled from pi_theta equals the exact KL(pi_theta || pi_ref)
p = torch.tensor([0.50, 0.30, 0.15, 0.05])     # the trained policy over a 4-token vocabulary
q = torch.tensor([0.40, 0.40, 0.10, 0.10])     # the reference policy
exact = float((p * (p / q).log()).sum())
idx = torch.multinomial(p, 200_000, replacement=True, generator=torch.Generator().manual_seed(1))
x = q[idx] / p[idx]
mc = float((x - x.log() - 1).mean())
log(f'exact KL(pi_theta || pi_ref) = {exact:.5f}; mean of k3 over 200,000 samples = {mc:.5f}')
out['kl'] = {'exact': exact, 'mc': mc}

# ---------------------------------------------------------------- 6. the whole GRPO loss on a toy batch
log('\n== 6. The GRPO loss on a toy group (G = 4, answers of 5, 3, 4, 2 tokens) ==')
BETA = 0.04                                     # DeepSeekMath's KL coefficient
lens = [5, 3, 4, 2]
T = max(lens)
g = torch.Generator().manual_seed(7)
logp_old = -torch.rand(4, T, generator=g) * 2            # log-probs of the sampled tokens, old policy
logp_new = logp_old + 0.3 * torch.randn(4, T, generator=g)  # after a few updates the policy moved a little
logp_ref = logp_old + 0.2 * torch.randn(4, T, generator=g)
mask = torch.zeros(4, T)
for i, n in enumerate(lens):
    mask[i, :n] = 1.0                                    # 1 for real tokens, 0 for padding
A = torch.tensor(A1)


def grpo_loss_scratch():
    """Method 1: loops, exactly the order of equation (3): 1/G sum_i 1/|o_i| sum_t (surrogate - beta * KL)."""
    total = 0.0
    for i in range(4):
        s = 0.0
        for t in range(lens[i]):
            ratio = math.exp(logp_new[i, t] - logp_old[i, t])
            surr = min(ratio * A1[i], max(1 - EPS, min(ratio, 1 + EPS)) * A1[i])
            kl = k3(math.exp(logp_new[i, t]), math.exp(logp_ref[i, t]))
            s += surr - BETA * kl
        total += s / lens[i]
    return -total / 4                                    # we minimise the negative of the objective


def grpo_loss_torch(logp_new, logp_old, logp_ref, A, mask, eps=EPS, beta=BETA):
    """Method 2: the same, vectorised. Shapes: logp_* and mask (G, T), A (G,)."""
    ratio = (logp_new - logp_old).exp()                                  # (G, T)
    surr = torch.minimum(ratio * A[:, None], ratio.clamp(1 - eps, 1 + eps) * A[:, None])
    x = (logp_ref - logp_new).exp()                                      # pi_ref / pi_theta
    kl = x - (logp_ref - logp_new) - 1                                   # k3, always >= 0
    per_token = (surr - beta * kl) * mask
    per_answer = per_token.sum(dim=1) / mask.sum(dim=1)                  # 1/|o_i| sum_t
    return -per_answer.mean()                                            # 1/G sum_i, negated


l1 = grpo_loss_scratch()
lp = logp_new.clone().requires_grad_(True)
l2 = grpo_loss_torch(lp, logp_old, logp_ref, A, mask)
l2.backward()
log(f'Method 1 (loops) loss = {l1:.6f}')
log(f'Method 2 (torch) loss = {l2.item():.6f}   difference {abs(l1 - l2.item()):.1e}')
gsum = (lp.grad * mask).sum(dim=1)
log('gradient of the loss summed over each answer (negative = that answer is pushed up): '
    + ', '.join(f'{v:+.3f}' for v in gsum.tolist()))
out['loss'] = {'scratch': l1, 'torch': l2.item(), 'grad_sum': gsum.tolist()}

# ---------------------------------------------------------------- 7. a toy RL run
log('\n== 7. A toy RL run: one question, four possible answers, GRPO with G = 8 ==')
# The "policy" is a softmax over the 4 answers of section 1. A language model chooses tokens; this toy chooses
# whole answers, so each answer is one "token" and the ratio is a per-answer ratio, as in R1's sequence-level form.
logits = torch.tensor([0.0, 0.6, 0.6, 1.2], requires_grad=True)   # start: the right, formatted answer is unlikely
reward_of = torch.tensor(rewards)                                   # [2, 1, 1, 0]
opt = torch.optim.SGD([logits], lr=0.5)
curve = []
for step in range(41):
    probs = logits.softmax(0)
    curve.append([step] + [round(v, 4) for v in probs.tolist()])
    if step % 5 == 0:
        log(f'step {step:2d}: p(o1..o4) = ' + ', '.join(f'{v:.3f}' for v in probs.tolist())
            + f'   expected reward {float((probs.detach() * reward_of).sum()):.3f}')
    with torch.no_grad():
        old_logp = logits.log_softmax(0).clone()
        picks = torch.multinomial(probs, 8, replacement=True, generator=torch.Generator().manual_seed(100 + step))
        r = reward_of[picks]
        adv = (r - r.mean()) / (r.std(unbiased=False) + 1e-6)
    for _ in range(2):                              # two small updates on the same samples (old policy fixed)
        opt.zero_grad()
        new_logp = logits.log_softmax(0)[picks]
        ratio = (new_logp - old_logp[picks]).exp()
        surr = torch.minimum(ratio * adv, ratio.clamp(1 - EPS, 1 + EPS) * adv)
        (-surr.mean()).backward()
        opt.step()
out['toy_run'] = curve

# ---------------------------------------------------------------- 8. cost arithmetic
log('\n== 8. Cost arithmetic, R1 paper (Nature version / arXiv v2), Supplementary B.4.4 and Table 7 ==')
gpus = 64 * 8
zero_h, r1_h, sft_h = gpus * 198, gpus * 80, 5_000
log(f'R1-Zero: {gpus} H800 x 198 h = {zero_h:,} GPU hours (table: 101K)')
log(f'R1:      {gpus} H800 x  80 h = {r1_h:,} GPU hours (table: 41K)')
log(f'SFT data creation: {sft_h:,} GPU hours (table: 5K)')
tot = 101_000 + 5_000 + 41_000
log(f'total as printed: {tot:,} GPU hours x $2 = ${tot * 2:,}')
v3 = 2_788_000
log(f'V3 base model (V3 report, Table 1): {v3:,} GPU hours = ${v3 * 2:,}')
log(f'R1 on top of V3, as a share of the V3 figure: {tot / v3:.1%}  (our arithmetic)')
log(f'10% of V3 pre-training alone (2,664K GPU hours) = {0.1 * 2_664_000:,.0f} GPU hours (our arithmetic; V3.2 does not say which base)')
out['cost'] = {'zero_h': zero_h, 'r1_h': r1_h, 'total_h': tot, 'share_of_v3': tot / v3}

save('part4_grpo', out)
log('\nsaved results/part4_grpo.json')
