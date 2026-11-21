"""GRPO from scratch (Method 1 of Part 2): Equations 1 to 3 of DeepSeek-R1 as five short PyTorch functions.

Shapes:  B = number of answers in the batch (questions x G), T = longest answer in tokens.
Every per-token tensor is (B, T). The mask is 1 on a real answer token and 0 on padding.
The paper writes the ratio for a whole answer; like DeepSeekMath and TRL, we apply it to each token
and average over the tokens of an answer (the 1/|o_i| of DeepSeekMath's Equation 3)."""
import torch


def group_advantages(rewards, G, unbiased=True, eps=1e-4):
    """Equation 3: A_i = (r_i - mean(r)) / std(r), inside each group of G answers to one question.
    rewards: (B,) with B = n_questions * G, the G answers of one question next to each other.
    unbiased=True divides the variance by G - 1 (as torch.std and TRL do); False divides by G.
    eps keeps the division finite when all G rewards are equal (std = 0). Returns (B,)."""
    r = rewards.view(-1, G)                                   # (n_questions, G): one row per question
    mean = r.mean(dim=1, keepdim=True)                        # (n_questions, 1): the group's average reward
    std = r.std(dim=1, keepdim=True, unbiased=unbiased)       # (n_questions, 1): the group's spread
    return ((r - mean) / (std + eps)).view(-1)                # (B,): better than the group > 0, worse < 0


def ratio(logp_new, logp_old):
    """pi_theta(o_t) / pi_theta_old(o_t) for every token, from log-probabilities: exp(log a - log b) = a / b.
    logp_new, logp_old: (B, T). Returns (B, T). At the first step of a batch both are equal, so the ratio is 1."""
    return torch.exp(logp_new - logp_old)


def clipped_surrogate(rho, A, epsilon):
    """The min(...) of Equation 1 for every token: min(rho * A, clip(rho, 1 - eps, 1 + eps) * A).
    rho: (B, T) ratios; A: (B,) one advantage per answer, the same for all its tokens. Returns (B, T)."""
    A = A.unsqueeze(1)                                        # (B, 1): broadcast over the tokens of each answer
    unclipped = rho * A                                       # (B, T): the plain policy-gradient term
    clipped = torch.clamp(rho, 1 - epsilon, 1 + epsilon) * A  # (B, T): the ratio held inside [1 - eps, 1 + eps]
    return torch.minimum(unclipped, clipped)                  # (B, T): the smaller, more careful of the two


def k3_kl(logp_theta, logp_ref):
    """Equation 2 for every token: pi_ref/pi_theta - log(pi_ref/pi_theta) - 1, never below 0.
    logp_theta, logp_ref: (B, T). Returns (B, T)."""
    log_r = logp_ref - logp_theta                             # (B, T): log of pi_ref / pi_theta
    return torch.exp(log_r) - log_r - 1                       # (B, T): x - log x - 1 >= 0 for every x > 0


def grpo_loss(logp, logp_old, logp_ref, A, mask, epsilon=0.2, beta=0.04):
    """Equation 1 as a loss to minimise (the objective J with a minus sign), per token and masked.
    logp: (B, T) log-probs of the answer tokens under the policy being trained (carries the gradient).
    logp_old: (B, T) the same under the policy that sampled the answers (no gradient).
    logp_ref: (B, T) the same under the frozen reference policy.  A: (B,).  mask: (B, T) of 0/1.
    Returns a scalar: the mean over answers of the mean over each answer's tokens."""
    rho = ratio(logp, logp_old)                               # (B, T)
    per_token = clipped_surrogate(rho, A, epsilon) - beta * k3_kl(logp, logp_ref)   # (B, T): J per token
    per_answer = (per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)       # (B,): 1/|o_i| sum over t
    return -per_answer.mean()                                 # scalar: 1/G sum over i, with a minus to minimise
