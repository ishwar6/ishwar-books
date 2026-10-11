---
description: "RLHF with PPO from the inside: why plain REINFORCE wastes samples and takes unsafe steps, trust regions in plain words, the probability ratio and PPO's clipped objective worked by hand, the value loss and the entropy bonus, GAE on a real reply, the four models of RLHF and what they cost in memory, the implementation details that make PPO work for language models, and a from-scratch PPO run on Qwen2.5-0.5B-Instruct against the Chapter 7 reward model, checked by a stronger gold judge."
---
# Chapter 8 · RLHF with PPO: the full loop

> **Goal:** by the end of this chapter you can explain why RLHF uses PPO instead of plain REINFORCE, write down the clipped objective and compute it by hand for any token, say what the value model, the reference model and the reward model each do in the loop and what they cost, list the implementation details that decide whether a PPO run works, and read the dashboard of a real run: reward, KL, length, clip fraction, value loss. You will also have run PPO yourself on Qwen2.5-0.5B-Instruct with the reward model from Chapter 7, and checked with an independent judge how much of the reward gain was real.

---

## 8.1 Where we are, and what is still missing

[Chapter 6](/books/how-models-are-trained/06-reinforcement-learning-basics) built reinforcement learning for text from nothing: the policy gradient, REINFORCE, baselines, the advantage, the value function, GAE, and the per-token KL penalty to a reference model. It ended with a working method, REINFORCE with a leave-one-out baseline, that turned GPT-2 into a writer of positive movie reviews, and with a list of three reasons why most RLHF systems from 2019 to 2023 used something more complicated. [Chapter 7](/books/how-models-are-trained/07-reward-models) built the judge: a reward model on Qwen2.5-0.5B-Instruct that agrees with GPT-4's preferences on 69.5% of held-out UltraFeedback pairs, likes length a little too much, and over-promises when you optimise against it.

This chapter puts the two together with the algorithm that the classic RLHF papers used: **PPO**, Proximal Policy Optimization (Schulman et al., 2017). Ziegler et al. (2019), Stiennon et al. (2020), InstructGPT (Ouyang et al., 2022) and Anthropic's assistant (Bai et al., 2022) all trained their policies with it. Chapter 2 told that story; here we take the method apart.

We start from what REINFORCE gets wrong for a language model. There are three problems.

1. **Each batch is used once.** Generating replies is the expensive part of a step. In Chapter 6, sampling 64 replies of 24 tokens took most of each step's time; with replies of a hundred tokens or more and a bigger model, generation dominates even more. REINFORCE takes one gradient step and throws the batch away, because after the step the replies are no longer samples from the current policy, and the policy gradient formula assumes they are.
2. **Nothing limits the size of a step.** The policy gradient tells us a *direction*. How far to go is left to the learning rate. Too small and training is slow; too large and one unlucky batch can push the policy somewhere bad, from which the next batches (sampled from that bad policy) may not rescue it.
3. **Credit per token is coarse.** With one reward at the end and a baseline per reply, every token of a reply shares the same advantage, up to the KL pieces. A learned value function can do better, as Section 6.10 showed with GAE.

PPO answers all three: a **probability ratio** lets it reuse a batch for several gradient steps, **clipping** that ratio keeps each update small, and a **value model** with GAE gives each token its own advantage.

> [!DEFINITION] PPO (Proximal Policy Optimization)
> A policy gradient method that samples a batch with the current policy, then takes several gradient steps on that same batch while stopping the new policy from moving too far from the one that sampled it. "Proximal" means "staying close". It does this by clipping the probability ratio between the new and the old policy (Section 8.4).

{{FIG:ch8_reuse|The first difference. REINFORCE takes one gradient step per batch of samples. PPO takes several: our run in Section 8.9 makes 2 passes over each batch in 2 minibatches, so 4 gradient steps for each round of sampling.}}

## 8.2 Re-using a batch: the probability ratio

Why can't REINFORCE simply take ten steps on the same batch? Recall its loss from Section 6.5, averaged over all sampled tokens:

$$L^{\text{PG}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N} \hat A_i \,\log \pi_\theta(a_i \mid s_i)$$

where:

- $$i$$ runs over the $$N$$ sampled tokens in the batch (every token of every reply),
- $$s_i$$ is the state (prompt plus the reply so far) and $$a_i$$ the token that was sampled there,
- $$\pi_\theta(a_i \mid s_i)$$ is the probability the current policy gives that token,
- $$\hat A_i$$ is the advantage estimate for that token, a fixed number computed before the update.

Its gradient is the policy gradient only at the parameters that *produced* the samples. After one step the parameters have changed, and the batch is now a sample from an older policy. Keep minimising $$L^{\text{PG}}$$ on it and you are no longer following the gradient of anything meaningful: tokens with positive advantage get pushed up without limit, because the loss keeps rewarding a higher $$\log \pi$$ no matter how high it already is. The PPO paper says this directly:

> [!PAPER] Schulman et al. (2017), Proximal Policy Optimization Algorithms · Section 2.1 to 2.2 · page 2
> [![A passage of the PPO paper: while it is appealing to perform multiple steps of optimization on this loss L PG using the same trajectory, doing so is not well-justified, and empirically it often leads to destructively large policy updates; then section 2.2 Trust Region Methods: in TRPO an objective function, the surrogate objective, is maximized subject to a constraint on the size of the policy update, with equations 3 and 4: maximize the expected ratio of new to old probability times the advantage, subject to the expected KL between old and new policy being at most delta](/img/training/ch8-ppo-destructive.png)](/img/training/ch8-ppo-destructive.png)
>
> **Context:** the background section, just before the paper introduces its own objective.
>
> **What it says:** performing "multiple steps of optimization on this loss $$L^{PG}$$ using the same trajectory" is "not well-justified" and "often leads to destructively large policy updates". TRPO instead maximises a "surrogate" objective "subject to a constraint on the size of the policy update".
>
> **Why it matters:** this is the whole motivation in two sentences. The rest of the paper is about getting TRPO's safety with an objective that ordinary gradient descent can optimise.

The fix for the first half of the problem, "the samples come from an older policy", is a standard statistical tool.

> [!DEFINITION] Importance sampling
> A way to compute an average under one distribution using samples from another. Each sample is weighted by the ratio of its probability under the distribution you care about to its probability under the one that produced it.

Let $$\pi_{\theta_{\text{old}}}$$ be the policy that sampled the batch and $$\pi_\theta$$ the policy being updated. For each sampled token define the **probability ratio**:

$$r_i(\theta) = \frac{\pi_\theta(a_i \mid s_i)}{\pi_{\theta_{\text{old}}}(a_i \mid s_i)} = \exp\big(\log \pi_\theta(a_i \mid s_i) - \log \pi_{\theta_{\text{old}}}(a_i \mid s_i)\big)$$

where:

- $$\pi_{\theta_{\text{old}}}(a_i \mid s_i)$$ is the probability of the sampled token under the policy that sampled it, stored when the batch was generated and never changed afterwards,
- $$\pi_\theta(a_i \mid s_i)$$ is its probability under the policy as it is now, during the update,
- the right-hand form is how code computes it: from two log-probabilities, which is numerically safer than dividing small numbers.

At the start of the update the two policies are the same, so every ratio is exactly 1. A ratio of 1.3 means the updated policy now gives this token 30% more probability than the policy that sampled it; 0.7 means 30% less.

The **surrogate objective** of TRPO weights each advantage by its ratio:

$$L^{\text{CPI}}(\theta) = \frac{1}{N}\sum_{i=1}^{N} r_i(\theta)\, \hat A_i$$

where CPI stands for "conservative policy iteration" (Kakade and Langford, 2002), where this objective first appeared, and the other symbols are as above.

Two facts make it the right thing to optimise. First, by importance sampling, it estimates how much better than the old policy the new one would do, using only the old samples. Second, its gradient at $$\theta = \theta_{\text{old}}$$ is exactly the policy gradient. To see this, use $$\nabla_\theta r = r \,\nabla_\theta \log \pi_\theta$$ (the log-derivative trick of Section 6.4 again):

$$\nabla_\theta L^{\text{CPI}} = \frac{1}{N}\sum_i \hat A_i \, r_i(\theta)\, \nabla_\theta \log \pi_\theta(a_i \mid s_i) \;\xrightarrow{\;\theta = \theta_{\text{old}},\ r_i = 1\;}\; \frac{1}{N}\sum_i \hat A_i\, \nabla_\theta \log \pi_\theta(a_i \mid s_i)$$

where the arrow means "evaluated at the old parameters, where every ratio is 1", and the right-hand side is minus the gradient of $$L^{\text{PG}}$$: the REINFORCE gradient. So the first step of optimising $$L^{\text{CPI}}$$ is a REINFORCE step. Later steps differ, because the factor $$r_i$$ in front of each term records how much the policy has already moved on that token.

**Worked example.** A token had probability 0.100 under the old policy and advantage $$+1.5$$. After a few gradient steps the policy gives it 0.130. Its ratio is $$0.130 / 0.100 = 1.30$$ and its term in $$L^{\text{CPI}}$$ is $$1.30 \times 1.5 = 1.95$$. The gradient of that term with respect to the token's log-probability is $$r \hat A = 1.95$$: the objective still wants to push this token up, and even harder than at the start (1.5), because the ratio multiplies the advantage. Nothing in $$L^{\text{CPI}}$$ says "you have moved enough". That is the second half of the problem.

## 8.3 Trust regions: only trust the batch near where it was sampled

A batch of 32 replies is a small, noisy sample. It tells us something reliable about the policy *close* to the one that produced it, and almost nothing about policies far away. TRPO (Schulman et al., 2015) turned this intuition into an algorithm: maximise the surrogate, but only inside a region around the old policy where the surrogate can be trusted.

> [!DEFINITION] Trust region
> A neighbourhood of the current parameters (or, here, of the current policy) inside which a cheap approximation of the true objective is believed to be accurate. Optimisation takes the best step *within* the region instead of the best step overall.

> [!PAPER] Schulman et al. (2015), Trust Region Policy Optimization · Section 4 · page 3
> [![A paragraph of the TRPO paper: in practice, if we used the penalty coefficient C recommended by the theory above, the step sizes would be very small; one way to take larger steps in a robust way is to use a constraint on the KL divergence between the new policy and the old policy, i.e. a trust region constraint: maximize L theta old of theta subject to D KL max of theta old and theta at most delta](/img/training/ch8-trpo-trust.png)](/img/training/ch8-trpo-trust.png)
>
> **Context:** after proving that the true objective is at least the surrogate minus a KL penalty, the paper turns the bound into an algorithm.
>
> **What it says:** the penalty the theory recommends would make "the step sizes ... very small", so instead TRPO uses "a constraint on the KL divergence between the new policy and the old policy, i.e., a trust region constraint".
>
> **Why it matters:** the KL between consecutive policies becomes the yardstick for "how far did this update move?". PPO keeps the yardstick and replaces the constraint by something cheaper.

In plain words, TRPO's rule is: *improve the surrogate as much as you can, as long as the average KL divergence between the old and new policy, over the states in the batch, stays below a small number $$\delta$$.* Solving that constrained problem needs second-order machinery (the conjugate gradient method and a line search), which is awkward for networks with billions of parameters and for code that is otherwise plain Adam. PPO's question was: can a first-order method, ordinary minibatch gradient descent, get the same protection?

Before answering, it is worth seeing the danger with numbers. `ch8_toy.py` builds a six-token "bandit": one state, six possible tokens with true mean rewards $$0.3, 0.2, 0.1, 0.0, -0.1, -0.2$$, each observed with noise of standard deviation 1.0. That noise is large compared with the gaps, as it is for a reward model scoring replies. The old policy slightly prefers the middling tokens (expected reward $$+0.050$$). We sample one batch of 32 tokens, compute advantages as reward minus the batch mean, and then take 60 Adam steps on that one batch with three different objectives: the REINFORCE loss reused, the ratio objective $$L^{\text{CPI}}$$, and PPO's clipped objective (Section 8.4). Repeated for 500 different batches:

```text
  reinforce after 60 steps: true expected reward +0.098 (10th percentile +0.010), KL(old||new) 5.222, worse than before in 22% of seeds
  ratio     after 60 steps: true expected reward +0.147 (10th percentile -0.098), KL(old||new) 3.549, worse than before in 25% of seeds
  clip      after 60 steps: true expected reward +0.063 (10th percentile +0.041), KL(old||new) 0.042, worse than before in 24% of seeds
```

{{FIG:ch8_bandit_one|Sixty gradient steps on one batch of 32, averaged over 500 batches. Left: without clipping, the policy keeps moving away from the one that sampled the batch (KL above 3.5 nats); with clipping it stops near 0.04. Right: the worst 10% of batches. Without clipping they end well below where they started; with clipping they stay above it.}}

Read the three lines as a gamble. With no limit, the policy moves far (a KL of 3.5 to 5.2 nats means it is now nearly deterministic) towards whichever token *looked* best in these 32 noisy samples. On average that is a better token, so the mean goes up. But in the worst tenth of batches, the ratio objective ends at $$-0.098$$, far below the start: it bet everything on a token that only looked good by chance. The clipped objective moves about a hundred times less (KL 0.042), gains less from one batch, and its worst tenth still ends above the start.

One batch is not the whole story, though. In real training the next batch is sampled from the *new* policy. A policy that has collapsed onto one token samples only that token, every advantage becomes zero, and learning stops. The second experiment runs the full loop: 40 iterations, each sampling a fresh batch of 32 from the current policy and taking either 1 or 10 gradient steps on it.

```text
  reinforce_K1  expected reward after 10/20/40 iterations: +0.097 / +0.144 / +0.215   10th percentile at 40: +0.161   P(best token) at 40: 0.52   stuck on a worse token: 0%
  reinforce_K10 expected reward after 10/20/40 iterations: +0.220 / +0.234 / +0.238   10th percentile at 40: +0.100   P(best token) at 40: 0.56   stuck on a worse token: 41%
  ratio_K10     expected reward after 10/20/40 iterations: +0.224 / +0.239 / +0.244   10th percentile at 40: +0.101   P(best token) at 40: 0.61   stuck on a worse token: 35%
  clip_K10      expected reward after 10/20/40 iterations: +0.191 / +0.243 / +0.271   10th percentile at 40: +0.199   P(best token) at 40: 0.77   stuck on a worse token: 11%
```

{{FIG:ch8_bandit_loop|The full loop over 500 seeds. One REINFORCE step per batch is safe but slow. Ten unclipped steps per batch are fast at first, then stall: about 4 in 10 seeds lock onto a worse token and never leave. Ten clipped steps end with the highest reward and the most seeds on the best token.}}

This is the trade PPO makes, measured:

- **One step per batch** never gets stuck, but after 40 batches it has only reached $$+0.215$$.
- **Ten unclipped steps per batch** use each batch more, so they are ahead after 10 iterations ($$+0.224$$ against $$+0.097$$). But 35% to 41% of the seeds end almost deterministic on a token that is not the best, and from there no batch can teach them anything.
- **Ten clipped steps per batch** get most of the speed (ahead of one-step REINFORCE at every point) without most of the risk: the best final reward ($$+0.271$$), the best worst-case ($$+0.199$$), and the best token holds 77% of the probability on average.

The toy is small and its numbers belong to it, but the shape of the result is the one the PPO paper reported on robot-control tasks (Section 8.4 shows their Table 1).

## 8.4 The clipped objective

PPO's main objective, in the paper's notation:

> [!PAPER] Schulman et al. (2017), PPO · Section 3, equations 6 and 7 · page 3
> [![The clipped surrogate objective section of the PPO paper: L CLIP of theta equals the expected minimum of r t times A t and clip of r t between 1 minus epsilon and 1 plus epsilon times A t, with epsilon say 0.2; highlighted: clipping the probability ratio removes the incentive for moving r t outside of the interval; the final objective is a lower bound, i.e. a pessimistic bound, on the unclipped objective; we only ignore the change in probability ratio when it would make the objective improve, and we include it when it makes the objective worse](/img/training/ch8-ppo-clip.png)](/img/training/ch8-ppo-clip.png)
>
> **Context:** the core of the paper, right after the ratio and $$L^{CPI}$$ are defined.
>
> **What it says:** clipping the ratio "removes the incentive for moving $$r_t$$ outside of the interval $$[1-\epsilon, 1+\epsilon]$$", and taking the minimum makes the objective "a lower bound (i.e., a pessimistic bound) on the unclipped objective". In one sentence: "we only ignore the change in probability ratio when it would make the objective improve, and we include it when it makes the objective worse."
>
> **Why it matters:** this is the entire trick. No constraint, no second-order method: one `clamp` and one `min` in the loss.

$$L^{\text{CLIP}}(\theta) = \frac{1}{N}\sum_{i=1}^{N} \min\Big(r_i(\theta)\,\hat A_i,\;\; \mathrm{clip}\big(r_i(\theta),\, 1-\epsilon,\, 1+\epsilon\big)\,\hat A_i\Big)$$

where:

- $$r_i(\theta)$$ is the probability ratio of token $$i$$ (Section 8.2),
- $$\hat A_i$$ is its advantage, fixed during the update,
- $$\mathrm{clip}(r, 1-\epsilon, 1+\epsilon)$$ cuts $$r$$ to the interval $$[1-\epsilon, 1+\epsilon]$$: values below $$1-\epsilon$$ become $$1-\epsilon$$, values above $$1+\epsilon$$ become $$1+\epsilon$$,
- $$\epsilon$$ (epsilon) is the clip range, almost always 0.2, so the interval is $$[0.8, 1.2]$$,
- $$\min$$ takes the smaller of the unclipped and the clipped term, for each token separately,
- the policy maximises $$L^{\text{CLIP}}$$; code minimises its negative.

The best way to understand it is token by token. There are only four situations, depending on the sign of the advantage and on which way the ratio has moved.

{{FIG:ch8_clip_shape|One token's term in the objective as a function of its ratio. Left, a good token (A = +1.5): the objective rises with the ratio until 1.2, then goes flat, so pushing further earns nothing. Right, a bad token (A = -0.8): the objective improves as the ratio falls, until 0.8, then goes flat. On the other side of 1 the clipped and unclipped lines coincide: mistakes are always fully counted.}}

**Worked example: the four cases.** Each token had probability 0.100 under the old policy ($$\log \pi_{\text{old}} = -2.303$$). After some gradient steps its probability has changed. `ch8_toy.py` computes each term and, with autograd, the gradient of the term with respect to the token's new log-probability:

```text
== 1. the clipped objective for one token (epsilon = 0.2)
  A > 0, ratio went up a lot     p_old 0.100 p_new 0.130 ratio 1.301  A +1.5  r*A +1.951  clip(r)*A +1.800  min +1.800  d objective / d log p +0.000
  A > 0, ratio went down         p_old 0.100 p_new 0.090 ratio 0.900  A +1.5  r*A +1.350  clip(r)*A +1.350  min +1.350  d objective / d log p +1.350
  A < 0, ratio went down a lot   p_old 0.100 p_new 0.070 ratio 0.700  A -0.8  r*A -0.560  clip(r)*A -0.640  min -0.640  d objective / d log p +0.000
  A < 0, ratio went up           p_old 0.100 p_new 0.150 ratio 1.501  A -0.8  r*A -1.201  clip(r)*A -0.960  min -1.201  d objective / d log p -1.201
```

1. **Good token, already pushed up a lot.** $$r = 0.130 / 0.100 = 1.30$$. Unclipped: $$1.30 \times 1.5 = 1.95$$. Clipped: $$1.2 \times 1.5 = 1.80$$. The minimum is 1.80, the clipped term, which does not depend on $$\theta$$ at all, so the gradient is **0**. This token has had its share of this batch's credit; PPO stops pushing it.
2. **Good token whose probability went down** (other updates in the minibatch pulled it down to 0.090). $$r = 0.90$$, inside the interval, so both terms are $$0.90 \times 1.5 = 1.35$$, and the gradient is $$r \hat A = 1.35$$: keep pushing it up. Even if $$r$$ had fallen below 0.8, the minimum would pick the unclipped term ($$r \hat A$$ is then smaller than $$0.8 \hat A$$), so the gradient would stay. A good token losing probability is a mistake, and mistakes are always corrected.
3. **Bad token, already pushed down a lot.** $$r = 0.70$$. Unclipped: $$0.70 \times (-0.8) = -0.56$$. Clipped: $$0.8 \times (-0.8) = -0.64$$. The minimum is $$-0.64$$, the clipped, constant term: gradient **0**. The bad token has been pushed down enough for one batch.
4. **Bad token whose probability went up.** $$r = 1.50$$. Unclipped: $$1.50 \times (-0.8) = -1.20$$. Clipped: $$1.2 \times (-0.8) = -0.96$$. The minimum is the unclipped $$-1.20$$, so the gradient is $$r \hat A = -1.20$$: push it back down, at full strength.

The pattern: the objective is clipped only on the side where the policy has already moved *in the direction the advantage wanted*. In the two "wrong-way" cases the minimum chooses the unclipped term and the gradient is the full $$r \hat A$$. That is why the objective is a pessimistic bound: for every token it takes the less flattering of the two estimates.

> [!NOTE]
> Clipping does not stop the ratio from leaving $$[0.8, 1.2]$$. A token with zero gradient can still move, because it shares parameters with tokens that do have gradients. What clipping removes is the *incentive* to move further. In practice the ratios of a well-tuned run stay close to the interval, and the share of tokens outside it, the **clip fraction**, is one of the numbers to watch (Section 8.11).

The loss in code (simplified from `ch8_ppo.py`):

```python
new_lp = token_logprobs(policy, ids, att, temp)          # log pi_theta of every sampled token, with gradient
ratio = torch.exp(new_lp - old_lp)                        # old_lp was stored when the batch was sampled
pg1 = -A * ratio                                          # minus the unclipped term
pg2 = -A * torch.clamp(ratio, 1 - 0.2, 1 + 0.2)           # minus the clipped term
pg_loss = (torch.max(pg1, pg2) * mask).sum() / mask.sum() # max of the negatives = min of the objectives
```

Line by line: the log-probabilities of the sampled tokens under the current policy; the ratio from the difference of logs; the two terms with a minus sign (we minimise); `torch.max` of the negatives, which is the minimum of the objectives; and an average over the reply tokens only (`mask` is 1 on tokens the policy chose and 0 on the prompt and padding).

Was the minimum worth it? The PPO paper compared it with the alternatives on seven simulated robot-control tasks:

> [!PAPER] Schulman et al. (2017), PPO · Table 1 · page 6
> [![Table 1 of the PPO paper: average normalized score over 21 runs on 7 environments; no clipping or penalty minus 0.39; clipping epsilon 0.1: 0.76; clipping epsilon 0.2: 0.82, in bold; clipping epsilon 0.3: 0.70; adaptive KL with three targets: 0.68, 0.74, 0.71; fixed KL with beta 0.3, 1, 3, 10: 0.62, 0.71, 0.72, 0.69](/img/training/ch8-ppo-table1.png)](/img/training/ch8-ppo-table1.png)
>
> **Context:** the first experiment, scores rescaled so that a random policy gets 0 and the best result 1.
>
> **What it says:** with "no clipping or penalty" the average is $$-0.39$$, worse than random, because one task (half cheetah) diverges. Clipping with $$\epsilon = 0.2$$ scores best, 0.82. A KL penalty between the old and new policy, fixed or adaptive, also works (0.62 to 0.74) but less well.
>
> **Why it matters:** the same lesson as our toy: reusing samples without a limit is destructive; a limit makes reuse pay. And $$\epsilon = 0.2$$, chosen here in 2017, is still the default in RLHF code.

Two different KL penalties appear in this chapter, and they are easy to confuse. The one in Table 1 is between the **old and the new** policy of one update: it is PPO's step-size control, the job clipping does in our run. The one from Chapter 6 is between the **policy and the reference model**: it keeps the whole run close to where it started, so that the reward model is not exploited. RLHF uses clipping for the first and the per-token penalty for the second.

## 8.5 The rest of the loss: value and entropy

PPO trains two things at once: the policy, and a value function used for the advantages. The paper writes one combined objective:

> [!PAPER] Schulman et al. (2017), PPO · Section 5, equation 9 · page 5
> [![Equation 9 of the PPO paper: L CLIP plus VF plus S equals the expectation of L CLIP minus c 1 times L VF plus c 2 times S of pi theta at s t, where c1 and c2 are coefficients, S denotes an entropy bonus, and L VF is a squared-error loss V theta of s t minus V target squared](/img/training/ch8-ppo-alg.png)](/img/training/ch8-ppo-alg.png)
>
> **Context:** the algorithm section, for networks that share parameters between the policy and the value function.
>
> **What it says:** maximise the clipped objective, minus $$c_1$$ times the value function's squared error, plus $$c_2$$ times an "entropy bonus".
>
> **Why it matters:** these are the three parts of every PPO loss. In RLHF the first dominates, the second runs on a separate network, and the third is usually switched off.

**The value loss.** The value model $$V_\phi$$ (parameters $$\phi$$) is trained by regression towards the returns computed with GAE. Most RLHF code, following the original OpenAI implementation, also clips the value update:

$$L^{\text{VF}}(\phi) = \frac{1}{2N}\sum_{i=1}^{N} \max\Big(\big(V_\phi(s_i) - \hat R_i\big)^2,\; \big(V_{\text{old}}(s_i) + \mathrm{clip}(V_\phi(s_i) - V_{\text{old}}(s_i), -\epsilon_v, \epsilon_v) - \hat R_i\big)^2\Big)$$

where:

- $$V_\phi(s_i)$$ is the value model's current estimate for state $$s_i$$,
- $$V_{\text{old}}(s_i)$$ is its estimate when the batch was collected (stored, like $$\log \pi_{\text{old}}$$),
- $$\hat R_i = \hat A_i + V_{\text{old}}(s_i)$$ is the return target from GAE (the "$$\lambda$$-return" of Section 6.10), computed before advantage whitening,
- $$\epsilon_v$$ is the value clip range (0.2 in our run and in the N+ paper),
- the $$\max$$ takes the larger, more pessimistic error, the same idea as the policy's $$\min$$.

**Worked example.** A state had $$V_{\text{old}} = 0.10$$ and its return target is $$\hat R = 0.90$$. After a gradient step the value model says $$V_\phi = 0.50$$. The unclipped error is $$(0.50 - 0.90)^2 = 0.16$$. The clipped prediction is $$0.10 + \mathrm{clip}(0.40, -0.2, 0.2) = 0.30$$, with error $$(0.30 - 0.90)^2 = 0.36$$. The maximum is 0.36, the clipped term, which has no gradient: the value model has moved 0.2 towards its target in this batch and must wait for the next one. Value clipping is controversial: in a large study of PPO on control tasks, Andrychowicz et al. (2021) found that this kind of value clipping "hurts the performance regardless of the clipping threshold". It is in our code because it is in the RLHF reference implementations we follow, and the fraction of clipped values is logged so we can see whether it matters.

**The coefficient $$c_1$$.** When the policy and the value function share one network, $$c_1$$ balances their gradients. In RLHF they are usually separate networks (Section 8.7), and then $$c_1$$ only rescales the value model's learning rate. We use separate optimisers and no $$c_1$$.

**The entropy bonus.** $$S[\pi_\theta](s)$$ is the entropy of the policy's next-token distribution. Adding it rewards keeping options open. Control tasks sometimes need it; RLHF usually sets $$c_2 = 0$$, because the KL penalty to the reference model already keeps the distribution wide, and Chapter 6 showed (Ziegler et al.'s Table 10) that an entropy bonus alone does not prevent degenerate text. We still *measure* entropy: a sharp fall is a sign of collapse.

## 8.6 Advantages for a reply: GAE with a KL penalty at every token

Section 6.10 derived GAE and its $$\lambda$$ trade-off on a toy. In RLHF the only new thing is what the per-token rewards look like. With a KL coefficient $$\beta$$, Section 6.12's shaped reward is:

$$r_t = -\beta\big(\log \pi_{\text{old}}(y_t \mid s_t) - \log \pi_{\text{ref}}(y_t \mid s_t)\big) + \mathbb{1}[t = T-1]\; \tilde r(x, y)$$

where:

- $$y_t$$ is reply token $$t$$, $$s_t$$ the prompt plus the earlier reply tokens, and $$T$$ the number of reply tokens (for a finished reply the last one is `<|im_end|>`),
- $$\log \pi_{\text{old}}$$ and $$\log \pi_{\text{ref}}$$ are the log-probabilities under the policy that sampled the reply and under the frozen reference model,
- $$\tilde r(x, y)$$ is the reward model's score after normalisation and the EOS rule of Section 8.8, given at the last token only.

Then the TD errors and the GAE recursion are exactly as in Chapter 6, with $$\gamma = 1$$ and $$\lambda = 0.95$$:

$$\delta_t = r_t + V(s_{t+1}) - V(s_t), \qquad \hat A_t = \delta_t + \lambda \hat A_{t+1}, \qquad \hat R_t = \hat A_t + V(s_t)$$

where $$V(s_T) = 0$$ after the last token (the episode is over), the recursion runs backwards from the last token, and $$\hat R_t$$ is the value target.

**Worked example on a real reply.** In the first batch of the $$\beta = 0.05$$ run, one prompt asked: *Determine the topic of the question. Question: "what is the source of geothermal energy?" Topic:* The policy answered "The topic of this question is Geothermal Energy." in 11 tokens, counting the closing `<|im_end|>`. `ch8_ppo.py` saved every number for this reply:

```text
 t  token          log pi_old  log pi_ref   V(s_t)     r_t    A_t (GAE)  return
 0  'The'            -0.095      -0.095     -1.652   +0.000    +0.350    -1.302
 1  ' topic'         -0.000      -0.000     -7.545   +0.000    +6.572    -0.973
 3  ' this'          -6.890      -6.890     -4.920   +0.000    +4.262    -0.658
 6  ' Ge'            -0.896      -0.896     -4.706   +0.000    +4.606    -0.100
 8  ' Energy'        -0.011      -0.011     -6.637   +0.000    +7.155    +0.518
 9  '.'              -0.017      -0.017     -3.944   +0.000    +4.696    +0.753
10  '<|im_end|>'     -0.019      -0.019     -0.964   +0.843    +1.807    +0.843
```

(Selected rows; the full table is in `results/ch8_ppo_beta005.json`.)

{{FIG:ch8_token_gae|The same reply, all 11 tokens. In the first iteration the policy equals the reference, so every KL reward is 0 and the only reward is the score at the last token. The values come from a value model that is still a copy of the reward model, and they are far too low everywhere except at the end.}}

- **The rewards.** At the first iteration the policy *is* the reference, so $$\log \pi_{\text{old}} = \log \pi_{\text{ref}}$$ on every token and every KL reward is exactly 0. The reward model gives the reply a raw score of 6.037; normalised, $$(6.037 - 4.496) / 1.828 = +0.843$$, which lands on the last token. (The token " this" has $$\log \pi = -6.890$$, a probability of 0.001: sampling at temperature 0.7 still picks unlikely tokens now and then.)
- **The last token.** $$\delta_{10} = r_{10} + V(s_{11}) - V(s_{10}) = 0.843 + 0 - (-0.964) = 1.807$$. Nothing follows, so $$\hat A_{10} = 1.807$$ and the return is $$\hat A_{10} + V(s_{10}) = 1.807 - 0.964 = 0.843$$: the score itself.
- **One step back.** $$\delta_9 = 0 + V(s_{10}) - V(s_9) = -0.964 - (-3.944) = 2.980$$, and $$\hat A_9 = 2.980 + 0.95 \times 1.807 = 2.980 + 1.717 = 4.697$$, which the script prints as 4.696 from unrounded values.
- **Two steps back.** $$\delta_8 = -3.944 - (-6.637) = 2.693$$ and $$\hat A_8 = 2.693 + 0.95 \times 4.696 = 7.154$$.

Look at the values: $$-7.5$$ for the state after "The", $$-6.6$$ before "Energy", and close to the real score only at `<|im_end|>`. The value model is still a copy of the reward model, and a reward model was trained only to read the *last* token; at every other position its head produces numbers with no meaning. The N+ paper shows the same picture for its reward model (their Detail 13: "most values in the reward logits are non-valid and negative"). As a result every advantage in this reply is large and positive. That is less harmful than it looks, because advantages are whitened over the whole batch: what matters is how this reply's advantages compare with those of the other 31 replies, which suffer from the same offset. Within a few iterations the value model learns real per-token values (Section 8.10 shows its explained variance rising), and from then on the advantages mean what Section 6.8 said they should.

## 8.7 RLHF as a loop: four models

Put the pieces together and one PPO iteration for a language model looks like this:

{{FIG:ch8_loop|One iteration of RLHF with PPO, as implemented in ch8_ppo.py. Steps 1 to 4 need no gradients: they produce, for every reply token, its old log-probability, its reference log-probability, its value, its reward and its advantage. Step 5 is the only one that trains, and it is repeated for several epochs on the same batch.}}

Four networks take part. Stiennon et al. describe the setup that everyone after them copied:

> [!PAPER] Stiennon et al. (2020), Learning to summarize from human feedback · Section 3.4 · page 6
> [![A passage of the paper: the full reward R of x and y equals r theta of x and y minus beta times the log of pi RL over pi SFT; highlighted: this KL term acts as an entropy bonus; for the PPO value function we use a Transformer with completely separate parameters from the policy, which prevents updates to the value function from partially destroying the pretrained policy early in training; we initialize the value function to the parameters of the reward model](/img/training/ch8-stiennon-value.png)](/img/training/ch8-stiennon-value.png)
>
> **Context:** the description of the "human feedback policies".
>
> **What it says:** the reward is the reward model's score minus $$\beta$$ times the log-ratio to the supervised model. The KL term "acts as an entropy bonus" and keeps outputs close to what the reward model saw in training. The value function uses "completely separate parameters from the policy", initialised "to the parameters of the reward model".
>
> **Why it matters:** this paragraph fixes the shape of RLHF with PPO: a policy and a frozen copy of it, a reward model, and a value model that starts as a copy of the reward model.

{{FIG:ch8_four_models|The four models. Two are trained (policy and value), two are frozen (reference and reward). In our run the policy and the reference share one copy of Qwen2.5-0.5B-Instruct (the reference is the policy with its LoRA adapter switched off), and the value model starts as a copy of the Chapter 7 reward model.}}

> [!DEFINITION] Value model (critic)
> A network that reads a prompt and a partial reply and predicts the return still to come: here, roughly, the reward model score the finished reply will get, minus the KL penalties along the way. In RLHF it is usually a language model with a one-number head, like the reward model, read at every token instead of only the last.

- **Policy** $$\pi_\theta$$: the model being trained, initialised from the SFT model. It generates the replies and receives the PPO gradient.
- **Reference** $$\pi_{\text{ref}}$$: a frozen copy of the starting policy. It is only used to compute $$\log \pi_{\text{ref}}$$ for the KL penalty.
- **Reward model** $$r_\psi$$: frozen. It scores each finished reply once.
- **Value model** $$V_\phi$$: trained by the value loss. Initialised from the reward model, because the reward model already "knows" what a good reply looks like; at the start its per-token values are not good predictions of the return, but it learns them faster than a value model starting from scratch.

InstructGPT used the same four, at a larger scale: a 6B reward model and a 6B value model for policies of all sizes.

> [!PAPER] Ouyang et al. (2022), InstructGPT · Appendix C.4 · page 42
> [![Appendix C.4 of InstructGPT: RL policies are initialized from the supervised models; KL reward with beta 0.02; 256k episodes, about 31k unique prompts; batch 512 split into 8 minibatches of 64 and trained for a single inner epoch; constant learning rate with warmup; exponential moving average of weights; no discount; clip ratio 0.2; sampling temperature 1; a 6B reward model and a 6B value function, the latter initialized from the former; pretraining gradients mixed in with coefficient gamma 27.8](/img/training/ch8-instructgpt-c4.png)](/img/training/ch8-instructgpt-c4.png)
>
> **Context:** the appendix with the RL hyperparameters.
>
> **What it says:** $$\beta = 0.02$$; batches of 512 replies split into 8 minibatches and trained for "a single inner epoch"; clip ratio 0.2; no discount; the value function "is initialized from" the reward model. For PPO-ptx, pretraining gradients are added with coefficient $$\gamma = 27.8$$.
>
> **Why it matters:** these settings are a useful anchor. Note one epoch, not four: with batches that large, each batch already gives 8 gradient steps.

**PPO-ptx.** InstructGPT noticed that RLHF made the model worse on some public NLP benchmarks (SQuAD, DROP), a cost they called the "alignment tax". Their fix was to mix ordinary next-token prediction on pretraining data into the RL loss: the objective of Chapter 2's Equation 2, reward minus $$\beta$$ times KL plus $$\gamma$$ times the log-likelihood of pretraining text. With the 27.8 coefficient and 8 pretraining examples per RL episode, the language-modelling gradient is a substantial part of every step. Our run does not do this; it is a way to protect abilities that the reward model does not measure.

**Adaptive KL.** Ziegler et al.'s controller (Section 6.12 worked an example) adjusts $$\beta$$ to hit a target KL instead of fixing it. InstructGPT and the N+ reproduction used a fixed $$\beta$$; libraries offer both. We use fixed values, so that the two runs differ in one number only.

### What four models cost

RLHF is expensive in memory as much as in compute. A rough count for full fine-tuning with Adam in mixed precision: a trained model needs about 16 bytes per parameter (2 for the bf16 weights, 2 for the gradients, 4 for an fp32 master copy of the weights, 4 and 4 for Adam's two moving averages); a frozen model in bf16 needs 2. That ignores activations and the cache used while generating, which add a lot for long replies.

```text
== 3. memory for the four models (weights and optimizer state only; no activations, no KV cache)
  0.5B (our Qwen2.5-0.5B)  policy     7.9 GB  value     7.9 GB  reference    1.0 GB  reward    1.0 GB  total    17.8 GB  (SFT of the same model: 7.9 GB, so RLHF needs 2.25x)
  7B                       policy   112.0 GB  value   112.0 GB  reference   14.0 GB  reward   14.0 GB  total   252.0 GB  (SFT of the same model: 112.0 GB, so RLHF needs 2.25x)
  70B                      policy  1120.0 GB  value  1120.0 GB  reference  140.0 GB  reward  140.0 GB  total  2520.0 GB  (SFT of the same model: 1120.0 GB, so RLHF needs 2.25x)
```

{{FIG:ch8_memory|Weights and optimizer state for the four models with full fine-tuning, from ch8_toy.py. The two trained models dominate. RLHF with a same-size value model needs 2.25 times the memory of supervised fine-tuning, before counting activations and generation.}}

**Worked example, 7B.** Policy: $$16 \times 7 \times 10^9 = 112$$ GB. Value model, same size: 112 GB. Reference and reward model: $$2 \times 7 \times 10^9 = 14$$ GB each. Total $$112 + 112 + 14 + 14 = 252$$ GB, against 112 GB to fine-tune the same model with SFT: $$252 / 112 = 2.25$$ times. That is more than three 80 GB GPUs before a single activation is stored, which is why RLHF code is full of tricks: sharing a backbone between policy and value (Stiennon et al. found separate networks work better), offloading the frozen models, LoRA, and the critic-free methods that drop the value model altogether.

We fit the four models of a 0.5B policy into far less with two tricks. The **policy is LoRA** (rank 16 on all seven linear layers of every block, 8.8 million trainable parameters), so the **reference is free**: it is the same network with the adapter switched off (`with policy.disable_adapter():`). The **value model** is a second copy of the Chapter 7 reward model whose LoRA adapter and score head are trained. Everything runs in fp32 on an Apple M5 Pro (64 GB); after the first iteration the process held 39.4 GB of GPU memory, most of it activations and the 151,936-wide logits of 16 sequences at a time, not weights.

## 8.8 The details that decide whether it works

The PPO paper gives an algorithm; making it work for language models took years of folklore. Huang et al. (2024) reproduced OpenAI's summarisation results (Stiennon et al., 2020) from scratch and wrote down every detail they found that mattered, more than 20 of them. Their hyperparameters are close to Stiennon's:

> [!PAPER] Huang et al. (2024), The N+ Implementation Details of RLHF with PPO · Table 7 · page 14
> [![Table 7 of the N+ paper, PPO hyperparameters: 1,000,000 episodes, about 8.56 epochs; AdamW with epsilon 1e-5 and learning rate 3e-6; linear scheduler; batch size 512; beta, the KL penalty coefficient, 0.05; gamma 1.0; lambda for GAE 0.95; 1 minibatch; K, the number of PPO update iterations per epoch, 4; policy clipping coefficient 0.2; value clipping coefficient 0.2](/img/training/ch8-nplus-table7.png)](/img/training/ch8-nplus-table7.png)
>
> **Context:** the PPO section of the paper, after the SFT and reward model sections.
>
> **What it says:** $$\beta = 0.05$$, $$\gamma = 1$$, $$\lambda = 0.95$$, 4 PPO epochs per batch, clip 0.2 for both the policy and the value, AdamW with $$\epsilon = 10^{-5}$$, batches of 512, a million episodes.
>
> **Why it matters:** our run copies these settings except the scale: batches of 32 and a few thousand episodes, a LoRA policy, and a higher learning rate to match (Section 8.9).

The details that matter most for RLHF, with what our code does for each:

**1. Normalise the reward model's score before RL.** A Bradley-Terry reward model's scale and offset are arbitrary (Section 7.4). InstructGPT and the N+ paper shift it so that the reference answers score 0 on average. We take 192 replies of the starting policy to training prompts, score them, and use their mean and standard deviation: every score is turned into $$(r - \mu) / \sigma$$, so 0 means "as good as an average reply of the starting model" and 1 means one standard deviation better. This also makes $$\beta$$ meaningful: a KL of 10 nats at $$\beta = 0.05$$ costs half a standard deviation of reward.

**2. Initialise the value model from the reward model.**

> [!PAPER] Huang et al. (2024) · Detail 22 · page 14
> [![Detail 22 of the N+ paper: value model initializes from the reward model; trained value model looks like a per-token RM; similar to Stiennon et al., we initialize the value network based on the reward model; this warm-starting of the value network can greatly improve initial gradients to the policy and reduce drift or alignment tax over training; in RL training the value function would aim to predict the end-of-episode return at each token, effectively acting as a per-token RM](/img/training/ch8-nplus-value.png)](/img/training/ch8-nplus-value.png)
>
> **Context:** one of the PPO training details.
>
> **What it says:** "This warm-starting of the value network can greatly improve initial gradients to the policy and reduce drift / alignment tax over training". After training, the value model is "effectively acting as a per-token RM".
>
> **Why it matters:** a value model that starts from random predictions makes the first advantages mostly noise. Starting from the reward model gives it a head start, and we apply the same normalisation $$(v - \mu)/\sigma$$ to its outputs so that values and rewards are on one scale.

**3. The EOS trick: a fixed penalty for replies that never end.** A reward model is trained to read its score at the token that closes the answer (Section 7.5). A reply cut off by the length limit has no such token, so its score is not well defined.

> [!PAPER] Huang et al. (2024) · Detail 23 · page 15
> [![The EOS trick in the N+ paper: the EOS trick serves a couple of purposes for RL; first, defined reward scores: it guarantees that the PPO model receives a defined reward score, because the RM only backpropagates loss on the EOS token during training; without an EOS token the completion's reward is undefined; the EOS trick assigns a constant -1 reward in these cases; second, constraining completion length: longer completions that lack an EOS token are penalized with a -1 reward](/img/training/ch8-nplus-eos.png)](/img/training/ch8-nplus-eos.png)
>
> **Context:** a detail reverse-engineered from the behaviour of OpenAI's original checkpoints.
>
> **What it says:** "Without an EOS token, the completion's reward is undefined. The EOS trick assigns a constant -1 reward in these cases." It also constrains length: "longer completions that lack an EOS token are penalized".
>
> **Why it matters:** our reward model likes longer answers (Section 7.9), and our replies are capped at 128 tokens. Without this rule, the policy could drift towards long answers that get cut off and still collect a reward for them.

We give every reply that has not emitted `<|im_end|>` within 128 tokens a normalised score of $$-1$$: one standard deviation below the starting model's average, whatever the reward model would have said.

**4. Sample with a temperature, and use the same temperature in the loss.** N+ samples at temperature 0.7 (Table 7). A detail that is easy to miss: the logits must then be divided by 0.7 also when computing log-probabilities for the ratio and the KL, as the reference implementations do. Otherwise $$\pi_{\text{old}}$$ is not the distribution the tokens actually came from, and the ratio at the first gradient step is not 1. We do the same.

**5. Turn off dropout.**

> [!PAPER] Huang et al. (2024) · Detail 7 · page 7
> [![Detail 7 of the N+ paper: disable dropout to ensure PPO's ratio calculation still works; with dropout activated, the log probabilities of tokens will not be reproducible, making calculating the KL penalty unreliable while also causing the ratios of the PPO to be not 1s during the first epoch](/img/training/ch8-nplus-dropout.png)](/img/training/ch8-nplus-dropout.png)
>
> **Context:** a model detail that applies to SFT, reward model and PPO alike.
>
> **What it says:** with dropout on, log-probabilities are random, so the ratios are "not 1s during the first epoch".
>
> **Why it matters:** the same principle as the temperature: everything in PPO assumes that recomputing a log-probability gives the same number. Qwen2.5 has no dropout, and our LoRA dropout is 0.

**6. Whiten the advantages.** After GAE, subtract the batch mean of the advantages and divide by their standard deviation, over all reply tokens (Detail 25 of N+). This keeps the size of the policy gradient stable as the reward scale drifts during training. We do it.

**7. Whitening the rewards is optional.** Ziegler et al.'s code could also whiten the per-token rewards before GAE (Detail 24). N+ treats it as optional; we do not use it.

**8. Average the loss over tokens, mask the prompt.** The loss is averaged over every reply token in the minibatch; prompt and padding tokens are masked out of the policy loss, the value loss and the KL. Implementations differ here (per-token mean, per-reply mean, or a constant divisor), and the choice changes how much each token of a long reply counts.

**9. Clip the gradient norm, use Adam with a small epsilon.** Gradients are clipped to norm 1 for both models, Adam's $$\epsilon$$ is $$10^{-5}$$ as in N+, and there is no weight decay.

**10. Reshuffle prompts when they run out.** Training draws prompts without replacement and reshuffles when they are used up (Detail 21).

The same list, for PPO in general and not only RLHF, was compiled by Huang et al. (2022) in "The 37 Implementation Details of Proximal Policy Optimization" (see References), and Engstrom et al. (2020) showed that on control tasks such "code-level optimizations" explain much of PPO's advantage over TRPO. For RLHF, Zheng et al. (2023) ran ablations of a similar list on 7B models and named their combination PPO-max. The common message: most of these details are small on their own, and a run that ignores several of them often fails in ways that look like a problem with the method.

## 8.9 Hands-on: PPO on Qwen2.5-0.5B-Instruct

Everything above now goes into one script, `ch8_ppo.py`: about 230 lines of plain PyTorch on top of the helpers in `ch8_common.py`, no RL library. It trains Qwen2.5-0.5B-Instruct, the same model Chapter 7 sampled for best-of-n, against the Chapter 7 reward model.

**The prompts.** UltraFeedback prompts of at most 128 tokens, none of which the reward model was trained or tested on. Qwen2.5-0.5B-Instruct likes to write long answers (in Chapter 7, only 27.5% of its samples ended within 256 tokens), and long replies make every step slow. So `ch8_prompts.py` caps replies at 128 tokens and keeps the prompts for which one sample of the starting model ended within that cap: 258 of 1,200 training candidates and 39 of 200 test candidates. This choice favours prompts with short answers (the kept replies averaged 45 tokens); it makes the runs fast and makes "did the reply end?" a real constraint the policy can break.

**The settings**, following Table 7 of the N+ paper except where noted:

| setting | value | note |
|---|---|---|
| policy | Qwen2.5-0.5B-Instruct + LoRA, rank 16, alpha 32, all linear layers | 8.8 million trainable parameters |
| reward model | Chapter 7, frozen | scores normalised with mean 4.496, sd 1.828 of 192 starting replies |
| value model | copy of the reward model, LoRA + head trained | values on the same normalised scale |
| replies | 32 per iteration, one per prompt, temperature 0.7, at most 128 tokens | N+: 512 per batch |
| EOS rule | normalised score $$-1$$ if no `<|im_end|>` | Detail 23 |
| KL | per-token $$k_1$$ penalty, $$\beta = 0.05$$ or $$\beta = 0$$ | N+: 0.05 |
| GAE | $$\gamma = 1$$, $$\lambda = 0.95$$, advantages whitened | as N+ |
| PPO | 2 epochs x 2 minibatches of 16, clip 0.2, value clip 0.2 | N+: 4 epochs x 1 minibatch |
| optimiser | AdamW, learning rate $$3 \times 10^{-5}$$ for both models, $$\epsilon = 10^{-5}$$, gradient norm 1 | N+: $$3 \times 10^{-6}$$ for full fine-tuning; LoRA needs a larger rate |
| length | 20 iterations, 640 replies per run, one seed | N+: a million episodes |

The learning rate came from one short trial: at $$5 \times 10^{-5}$$ with 4 epochs, the first iteration already had an approximate KL between old and new policy of 0.098 per token and a clip fraction of 0.22, far above the usual range (Section 8.11), so we lowered the rate and the number of epochs. An iteration of the $$\beta = 0.05$$ run took 30 to 75 seconds on the M5 Pro, of which generation was only 6 to 9 seconds: the forward and backward passes, with a softmax over the 151,936-entry vocabulary at every token, dominate. That is the main reason the runs are short.

### The code

The heart of `ch8_ppo.py`, simplified (the full script also logs, saves checkpoints and handles memory):

```python
S = generate(policy, prompts, max_new=128, temp=0.7)              # 1. rollout: 32 replies
ids, att, act = batch_tensors(S)                                   #    act = 1 on reply tokens (incl. <|im_end|>)
with torch.no_grad():                                              # 2. score, no gradients
    old_lp = token_logprobs(policy, ids, att, temp)                #    log pi_old, logits / 0.7
    with policy.disable_adapter():
        ref_lp = token_logprobs(policy, ids, att, temp)            #    log pi_ref: same weights, adapter off
    old_v = values_of(ids, att)                                    #    value at every token, normalised
    raw = score_ids(rm, rm_texts(S))                               #    reward model, one score per reply
    score = (raw - MU) / SD
    score[~finished] = -1.0                                        #    the EOS trick
    rew = -beta * (old_lp - ref_lp) * act                          # 3. per-token KL penalty ...
    rew[rows, last_token] += score                                 #    ... plus the score at the last token
    adv, ret = gae(rew, old_v * act, act, gamma=1.0, lam=0.95)     # 4. advantages and value targets
    adv = whiten(adv, act)
for epoch in range(2):                                             # 5. update, 2 epochs x 2 minibatches
    for mb in minibatches(32, size=16):
        ratio = torch.exp(token_logprobs(policy, ids[mb], att[mb], temp) - old_lp[mb])
        pg = torch.max(-adv[mb] * ratio, -adv[mb] * ratio.clamp(0.8, 1.2))
        v = values_of(ids[mb], att[mb])
        v_clip = old_v[mb] + (v - old_v[mb]).clamp(-0.2, 0.2)
        vl = 0.5 * torch.max((v - ret[mb]) ** 2, (v_clip - ret[mb]) ** 2)
        loss = masked_mean(pg, act[mb]) + masked_mean(vl, act[mb])
        loss.backward(); clip_grad_norm_(..., 1.0); opt_p.step(); opt_v.step()
```

Block by block:

1. **Rollout.** One reply per prompt, sampled with Hugging Face's `generate` from the current policy. `batch_tensors` builds prompt-plus-reply sequences (with the closing `<|im_end|>` for replies that finished) and a mask `act` that is 1 exactly on the tokens the policy chose.
2. **Score.** Three forward passes without gradients: the policy (its log-probabilities become $$\log \pi_{\text{old}}$$, fixed for this iteration), the same network with the adapter disabled ($$\log \pi_{\text{ref}}$$), and the value model. `values_of` runs the value model's transformer and applies its score head to *every* position, not only the last, which is what turns a reward model into a value model. Then the reward model scores each reply as in Chapter 7, the score is normalised, and replies that never ended get $$-1$$.
3. **Rewards.** The per-token reward of Section 8.6: $$-\beta$$ times the log-ratio on every reply token, plus the score on the reply's last token.
4. **Advantages.** The backward GAE recursion of Section 6.10 (`gae` handles right-padded rows: the value after a reply's last token is 0), then whitening over all reply tokens of the batch.
5. **Update.** The clipped policy loss of Section 8.4 and the clipped value loss of Section 8.5, averaged over reply tokens, two passes over the batch in minibatches of 16. The policy's LoRA adapter and the value model have separate AdamW optimisers.

The value model needs one more line of explanation. In Chapter 7 the reward model read the hidden state of the *last* token. Here, `values_of` reads every position:

```python
def values_of(ids, att):
    inner = value.base_model.model                     # Qwen2ForSequenceClassification with LoRA layers inside
    h = inner.model(input_ids=ids, attention_mask=att).last_hidden_state
    v = inner.score(h)[..., 0].float()                 # the 896 -> 1 head, applied at every position
    return ((v - MU) / SD)[:, :-1]                     # position t = the state before choosing token t+1
```

The hidden state at position $$t$$ has seen the prompt and the reply up to token $$t$$, which is exactly the state in which the policy chose token $$t+1$$. Hence the shift by one at the end, the same shift that aligns logits with the tokens they predict (Chapter 1).

### What one iteration measures

Every iteration logs a line like this one (the first iteration of the $$\beta = 0.05$$ run):

```text
step   0  score -0.125  rm_raw +4.178  kl   0.00  len  71.6  fin 0.66  ent 0.830  clipfrac 0.062  approx_kl 0.0132  v_loss 4.057  ev -0.55  (54s, gen 6s)
```

- **score**: mean normalised reward model score of the 32 replies, after the EOS rule. The starting model's replies average about 0 by construction; this batch is slightly below.
- **rm_raw**: the same scores before normalisation, on the reward model's own scale.
- **kl**: the sum of $$\log \pi_{\text{old}} - \log \pi_{\text{ref}}$$ over each reply's tokens, averaged over replies: the $$k_1$$ estimate of Section 6.12, in nats per reply. Zero at the start, since the policy *is* the reference.
- **len** and **fin**: mean reply length in tokens and the share of replies that ended.
- **ent**: the policy's mean entropy per reply token, in nats.
- **clipfrac**: during the update, the share of reply tokens whose ratio was outside $$[0.8, 1.2]$$.
- **approx_kl**: during the update, the mean over tokens of $$(r - 1) - \log r$$, Schulman's $$k_3$$ estimate of $$\mathrm{KL}(\pi_{\text{old}} \Vert \pi_\theta)$$: how far each update moved the policy from the one that sampled the batch.
- **v_loss** and **ev**: the value loss and the explained variance of the returns by the old values. An $$ev$$ of $$-0.55$$ means the freshly copied reward model is a worse predictor of the returns than their plain average, as expected for a reward model read at tokens it was never trained on.

## 8.10 Results

Two runs, identical except for $$\beta$$: the same seed, the same prompts in the same order, the same starting adapter. One uses the N+ value $$\beta = 0.05$$, the other no KL penalty at all. Each ran for 20 iterations (640 replies). One seed per run, so differences smaller than the batch-to-batch noise should not be read as real.

### What the training batches show

{{FIG:ch8_curves|Training batches of the two runs, 5-iteration running means. The proxy score rises a little and noisily. The clearest changes are elsewhere: replies get shorter (from about 53 to 31 tokens) and almost all of them end, so fewer collect the -1 of the EOS rule. Without a KL penalty the policy drifts further from the reference.}}

Averages over the first five and the last five iterations of each run, from the `hist` entries in `results/ch8_ppo_<run>.json`:

| | $$\beta = 0.05$$ | no KL |
|---|---|---|
| proxy score (EOS rule), iterations 0 to 4 | $$-0.018$$ | $$-0.012$$ |
| proxy score (EOS rule), iterations 15 to 19 | $$+0.095$$ | $$+0.129$$ |
| reply length, tokens | 52.6 to 31.0 | 52.9 to 30.9 |
| share of replies that end | 0.81 to 0.91 | 0.81 to 0.92 |
| KL to reference, iterations 15 to 19 (nats per reply) | 3.49 | 5.26 |
| entropy per token (nats) | 0.817 to 0.743 | 0.822 to 0.733 |

Three things happened, and only one of them is what you might expect from "RLHF makes answers better".

1. **The policy learned to finish.** The share of replies that end within 128 tokens went from about 81% to about 91%, and the average reply got 40% shorter. With the EOS rule, a reply that is cut off scores $$-1$$ whatever it says, while the starting model's finished replies average about 0. The cheapest way to raise the reward was to stop getting cut off. This is the "constraining completion length" effect of the N+ paper, and in our setup it outweighs the reward model's own slight preference for length.
2. **The proxy score rose, a little.** About $$+0.1$$ standard deviations over 20 iterations. The batch-to-batch noise (32 different prompts each time) is about as large, so the curve is jagged.
3. **Without the penalty, the policy moved further.** Both runs started at the same place and saw the same prompts; by the end the unpenalised policy was at 5.3 nats from the reference against 3.5 for $$\beta = 0.05$$. At $$\beta = 0.05$$, 3.5 nats cost $$0.05 \times 3.5 = 0.17$$ in reward, comparable to the reward gained, which is why the penalty bites even this early.

### Is it real? The gold judge

The training reward is measured on the training prompts by the reward model being optimised. `ch8_eval.py` checks the checkpoints after 10 and 20 iterations on the 39 held-out prompts, two replies each (78 replies per checkpoint, the same prompts and the same sampling seed for every checkpoint), with two judges: our reward model (the proxy) and Skywork-Reward-V2-Qwen3-0.6B (the gold), the stronger reward model Chapter 7 used. Both scores are in standard-deviation units of the starting model. `ch8_analysis.py` pairs each reply with the starting model's reply to the same prompt and gives 95% bootstrap intervals:

```text
78 held-out replies per checkpoint (same prompts, same sampling seed)
beta005  step 10: proxy change -0.008 [-0.124, +0.108]  gold change +0.129 [-0.076, +0.333]  proxy with EOS rule +0.140 (start +0.071)  finished 0.94 (start 0.79)  tokens 30.9 (start 59.3)  KL 2.69
beta005  step 20: proxy change +0.074 [-0.053, +0.197]  gold change -0.044 [-0.232, +0.172]  proxy with EOS rule +0.203 (start +0.071)  finished 0.90 (start 0.79)  tokens 35.9 (start 59.3)  KL 3.20
nokl     step 10: proxy change +0.012 [-0.104, +0.132]  gold change +0.113 [-0.085, +0.313]  proxy with EOS rule +0.185 (start +0.071)  finished 0.94 (start 0.79)  tokens 30.7 (start 59.3)  KL 2.86
nokl     step 20: proxy change +0.018 [-0.103, +0.131]  gold change +0.042 [-0.138, +0.230]  proxy with EOS rule +0.185 (start +0.071)  finished 0.94 (start 0.79)  tokens 28.8 (start 59.3)  KL 5.65
```

{{FIG:ch8_gold|Held-out replies at the start and after 10 and 20 iterations. Dashed: our reward model (no EOS rule). Solid: the gold reward model. Changes are small and every interval includes zero; the gold peaks at iteration 10 in both runs and falls back by iteration 20.}}

{{FIG:ch8_gold_kl|The same checkpoints plotted against their KL from the starting model, Gao-style. Solid: gold; dashed: proxy. The no-KL run reaches iteration 20 at almost twice the KL of the penalised run without a better gold score.}}

Read honestly, the held-out numbers say:

- **The objective PPO optimised did go up.** The proxy *with the EOS rule*, which is what the policy is paid, rose from $$+0.071$$ to between $$+0.14$$ and $$+0.20$$ on prompts it never saw, mostly because the share of finished replies rose from 0.79 to 0.90 to 0.94.
- **The reward model's own score barely moved.** Without the EOS rule the proxy changed by $$-0.008$$ to $$+0.074$$, all within the noise.
- **The gold went up, then came back.** At iteration 10 both runs are about $$+0.12$$ above the start on gold, larger than their proxy change; shorter, finished answers are something Skywork likes. By iteration 20 the gold is back near zero ($$-0.044$$ and $$+0.042$$), while the proxy with the EOS rule held or rose. No interval excludes zero, so this is a hint, not a result. But it is the shape Gao et al. describe: gold first rises with KL, then falls while the proxy keeps going.
- **The penalty bought distance, not quality.** At iteration 20 the unpenalised policy is 5.65 nats from the start, the penalised one 3.20, with no measurable difference in gold. That is Gao et al.'s "akin to early stopping" result in miniature: the penalty slows the drift; it does not make the reward model more trustworthy.

What we did *not* see in 20 iterations: no length explosion (lengths fell), no collapse into repeated phrases (distinct-2 stayed between 0.88 and 0.90 at every checkpoint), no entropy crash (0.82 to 0.74 nats per token). With a reward model as weak as ours, longer runs would very likely find its holes, as best-of-n did in Chapter 7; our budget stopped well before that.

### Before and after

The same held-out prompts, the same sampling seed, replies from the starting model and from both runs after 20 iterations (from `results/ch8_analysis_stdout.txt`; scores are proxy and gold in starting-model standard deviations):

```text
PROMPT: 'Act as a Salesforce administrator expert. If okay, just say "Okay".'
  [start    ] proxy -0.47 gold -0.18 tokens  18 "Okay. I'm here to help! What would you like to know about Salesforce administrators?"
  [beta 0.05] proxy +0.52 gold +0.54 tokens   2 'Okay.'
  [no KL    ] proxy +0.52 gold +0.54 tokens   2 'Okay.'

PROMPT: 'Detailed Instructions: ... classify given tweet text into two categories: 1) positive, and 2) negative ...'
  [start    ] proxy +1.18 gold +1.32 tokens  45 'Based on the content of the tweet provided, it appears to be expressing excitement about moving to California for summer vacation, which is likely seen as a positive experience. Therefore, I would classify this tweet as having a positive classification.'
  [beta 0.05] proxy +1.28 gold +0.38 tokens  13 'Based on the content provided, this tweet appears to be:\nPositive'
  [no KL    ] proxy +0.30 gold -1.18 tokens  28 'Based on the provided tweet text, this appears to be an informative and enthusiastic statement about living in Phoenix for the time being moving to California.'

PROMPT: 'Could you paraphrase the instruction for adding potatoes and mushrooms? ...'
  [start    ] proxy +0.22 gold -0.29 tokens 111 "Sure! Here's a paraphrased version of the instruction: ..."
  [beta 0.05] proxy +0.36 gold -0.04 tokens  77 'For the mashed potato and mushroom dish, combine all remaining ingredients except potatoes and mushrooms ...'
  [no KL    ] proxy +0.10 gold -1.26 tokens  72 'For this recipe, follow these steps:\n1. Add potatoes and mushrooms to an ovenproof quart dish. ...'
```

- **"Just say Okay."** The starting model said more than it was asked; both trained policies say exactly "Okay.", and both judges approve. This is the good side of learning to be brief.
- **The tweet.** The penalised policy answers with the label, as asked. The unpenalised one has drifted into describing the tweet without classifying it; the gold model drops it from $$+1.32$$ to $$-1.18$$, our reward model only to $$+0.30$$.
- **The recipe.** The unpenalised reply tells you to add the potatoes and mushrooms first, the opposite of the instruction; the gold judge marks it down to $$-1.26$$, our reward model barely notices.

Three prompts prove nothing, but they show what the averages hide: the same small average change is made of real improvements and real regressions, and the weaker judge misses more of the regressions.


## 8.11 What breaks, and what to watch

PPO for language models fails in a handful of recognisable ways. Each has a signature on the dashboard.

**Reward hacking.** The policy finds what the reward model pays for that people would not. Chapter 6 saw it with a sentiment classifier ("beautifully captures this superb masterpiece"), Chapter 7 measured its best-of-n version, and Gao et al. (2022) mapped it for PPO: the gold reward first rises with KL, then falls, while the proxy keeps rising. Their result on the KL penalty is worth knowing because it is not what most people expect:

> [!PAPER] Gao et al. (2022), Scaling Laws for Reward Model Overoptimization · Section 3.6 · page 7
> [![Section 3.6 of the paper, Effect of KL Penalty: when varying the KL penalty for RL, the gold RM scores depend only on the KL distance of the policy; the KL penalty only causes the gold RM score to converge earlier, but does not affect the KL-gold reward frontier, and so the effect of the penalty on the gold score is akin to early stopping; PPO's surrogate objective incorporates an implicit penalty on the KL between the old and the new policy](/img/training/ch8-gao-kl.png)](/img/training/ch8-gao-kl.png)
>
> **Context:** the RL half of the overoptimisation study, with a 1.2B policy and a 1.2B proxy reward model trained on a 6B "gold" model's labels.
>
> **What it says:** "the gold RM scores depend only on the KL distance of the policy". The penalty makes the gold score "converge earlier, but does not affect the KL-gold reward frontier, and so the effect of the penalty on the gold score is akin to early stopping". They add that the result "could be particularly sensitive to hyperparameters".
>
> **Why it matters:** the KL penalty is a leash, not a better optimiser. It decides *how far* the policy goes; how much gold reward you get at a given distance is set by the reward model. That is exactly the comparison our two runs allow.

**Length exploitation.** Reward models tend to prefer longer answers (Section 7.9), so PPO tends to make answers longer:

> [!PAPER] Singhal et al. (2023), A Long Way to Go · Figure 2 · page 3
> [![Figure 2 of Singhal et al.: histograms of output length for the SFT model before PPO in blue and after standard PPO in red, on three datasets, WebGPT, Stack and RLCD; the red distributions sit clearly to the right; caption: across all settings, PPO leads to large length increases](/img/training/ch8-singhal-fig2.png)](/img/training/ch8-singhal-fig2.png)
>
> **Context:** the paper's first analysis, on three preference datasets with LLaMA-7B policies.
>
> **What it says:** "Across all settings, PPO leads to large length increases." Later sections show that much of the reward gain can be matched by optimising length alone.
>
> **Why it matters:** always plot length next to reward. A reward curve that rises together with length needs a second judge before it is believed.

In our runs the effect went the other way: replies got shorter, because the EOS rule made cut-off replies the most expensive thing the policy could produce (Section 8.10). Within finished held-out replies, the correlation between our reward model's score and length was $$+0.24$$ for the starting model and $$-0.10$$ after 20 iterations of $$\beta = 0.05$$ (`results/ch8_analysis_stdout.txt`). Length bias is a property of the whole set-up, reward model plus length limit plus EOS rule, not of the reward model alone.

**Mode collapse.** The policy concentrates on a few patterns: the same opening, the same structure, the same phrases, whatever the prompt. Entropy falls, diversity metrics fall, and the KL to the reference rises. Zheng et al. (2023) describe it from 7B runs:

> [!PAPER] Zheng et al. (2023), Secrets of RLHF in Large Language Models Part I: PPO · Section 5.2 · page 12
> [![A passage of the paper: we observed stable convergence processes in training loss, but higher rewards do not reflect better policy behaviors from the perspective of human and GPT-4 evaluation; indicative training metrics include perplexity, KL divergence between the policy and reference models, and the average length of generation responses; the collapsed model uniformly delivers longer responses and exhibits lower perplexity for such generative patterns](/img/training/ch8-secrets-collapse.png)](/img/training/ch8-secrets-collapse.png)
>
> **Context:** the section on metrics for monitoring PPO training, after a run of "vanilla" PPO that went wrong.
>
> **What it says:** "higher rewards do not reflect better policy behaviors"; the useful signals were "perplexity, KL divergence between the policy and reference models, and the average length of generation responses", and "the collapsed model uniformly delivers longer responses and exhibits lower perplexity".
>
> **Why it matters:** the reward and the loss both looked fine. The failure was visible only in KL, length and perplexity, and in reading the outputs.

**Value-model problems.** A value model that predicts badly gives noisy or biased advantages (Section 6.10). Its signatures: explained variance near zero or negative, a value loss that does not fall, and many clipped value updates. A value model initialised from the reward model starts badly on every token except the last (Detail 13 of N+ shows the same picture), so a negative explained variance in the first iterations is expected; one that stays negative is not.

{{FIG:ch8_ppo_stats|PPO's own diagnostics for both runs. Clip fraction and approximate KL are highest in the first iterations, then stay small: each update moves the policy only a little. The value model's explained variance climbs from about -0.7 (a reward model read at every token is a poor critic) to about 0.8. Entropy is noisy and drifts down a little.}}

**The dashboard.** Putting it together, these are the numbers to log every iteration, with what our $$\beta = 0.05$$ run showed for each:

| metric | what it measures | healthy | our $$\beta = 0.05$$ run |
|---|---|---|---|
| reward (proxy) | what the policy is paid | rises, then flattens | $$-0.02$$ to $$+0.10$$ (5-iteration means) |
| gold or human score | what you actually want | rises with the proxy | $$+0.13$$ at iteration 10, $$-0.04$$ at 20 (noise $$\pm 0.2$$) |
| KL to reference | total drift from the start | grows slowly, then levels off | 0 to about 3.5 nats per reply |
| reply length, share finished | the cheapest hack | stable or explained | 53 to 31 tokens, 81% to 91% finished |
| entropy | diversity of each next-token choice | falls slowly | 0.82 to 0.74 nats per token |
| clip fraction | share of tokens with ratio outside $$[0.8, 1.2]$$ | about 0.01 to 0.2 | 0.016 on average |
| approx KL (old to new) | size of each update | about 0.001 to 0.02 per token | 0.0025 on average |
| value loss, explained variance | quality of the critic | loss falls, EV rises above 0 | EV from $$-0.70$$ to $$+0.81$$ |
| value clip fraction | how often the value update hits the clip | small | 0.36: the value model starts far off |

The "healthy" ranges are rough rules of thumb from practice, not laws. The first trial of our setup (learning rate $$5 \times 10^{-5}$$, 4 epochs) had an approx KL of 0.098 and a clip fraction of 0.22 in its first iteration, a sign that each update was too large, and we changed the settings because of it. The value clip fraction of 0.36 is the cost of initialising the value model from a reward model whose per-token outputs are far from the returns (Section 8.6): for many tokens, the value model wanted to move more than 0.2 per update. Its explained variance still rose from negative to 0.81 within 20 iterations.

## 8.12 What we did not do

To keep the run small enough to repeat in about an hour on one machine, we left out several things that real systems do: an SFT stage of our own (we start from an instruction-tuned model), a learning-rate schedule, batches of hundreds of replies, many seeds, a pretraining loss (PPO-ptx), an adaptive KL controller, and an evaluation by people. Our gold judge is another reward model, which is a stronger and independent reward model but not the truth (Section 7.10 discussed this). Each of these would change the numbers. The mechanics, and the warning signs, are the same.

> [!TAKEAWAYS]
> - REINFORCE uses each batch once and has no limit on the size of a step. PPO reuses each batch for several gradient steps through the probability ratio $$r = \pi_\theta / \pi_{\text{old}}$$, and limits each update by clipping it.
> - The ratio objective $$\mathbb{E}[r \hat A]$$ has the REINFORCE gradient at the start of the update, and keeps pushing after that. TRPO limits it with a KL constraint between old and new policy; PPO replaces the constraint by $$\min(r \hat A, \mathrm{clip}(r, 0.8, 1.2)\hat A)$$, which has zero gradient once a token has moved 20% in the direction its advantage wanted, and full gradient when it moved the wrong way.
> - In a six-token bandit with noisy rewards, ten clipped steps per batch beat both one REINFORCE step per batch (reward $$+0.271$$ against $$+0.215$$ after 40 batches) and ten unclipped steps, which left about 40% of seeds stuck on a worse token.
> - The full PPO loss adds a value loss (often clipped) and an entropy bonus (usually zero in RLHF). Advantages come from GAE with $$\gamma = 1$$, $$\lambda = 0.95$$, on per-token rewards made of KL pieces plus the reward model score at the last token.
> - RLHF with PPO needs four models: policy (trained), reference (frozen), reward model (frozen) and value model (trained, initialised from the reward model). With full fine-tuning that is about 2.25 times the memory of SFT: 252 GB of weights and optimizer state for a 7B model.
> - The details decide the outcome: normalise the reward, initialise the value model from the reward model, penalise replies that never end (the EOS trick), use the sampling temperature in the loss, turn off dropout, whiten advantages, mask the prompt.
> - Our 20-iteration PPO runs on Qwen2.5-0.5B-Instruct (one seed each) mainly taught the policy to finish its replies: finished share 0.79 to 0.94 and length 59 to 29 to 36 tokens on held-out prompts. The training objective rose; the reward model's own score and the gold score changed by less than the noise, with the gold peaking at iteration 10.
> - Without a KL penalty the policy drifted almost twice as far (5.65 against 3.20 nats) with no gold benefit: the penalty sets how far the policy goes, which is how Gao et al. describe it.
> - Watch KL, length, the share of finished replies, entropy, clip fraction, approx KL and the value model's explained variance, and check the reward with a second judge and by reading outputs. A rising reward on its own proves nothing.

## What comes next

PPO needs a reward model, a value model, a reference model and a sampling loop, and every one of them has to be tuned. The next chapter asks whether all of that is necessary. Direct Preference Optimization (DPO) uses the closed form of the KL-regularised optimum from Section 6.12 to turn the preference data itself into a loss on the policy: no reward model to train, no sampling during training, no value model, and a loss that looks like the reward model loss of Chapter 7. We will derive it, train it on the same UltraFeedback pairs, and compare it with the PPO run of this chapter.

## References

### Papers

- Schulman, J., Levine, S., Moritz, P., Jordan, M. I., Abbeel, P. (2015). Trust Region Policy Optimization. ICML 2015. [arXiv:1502.05477](https://arxiv.org/abs/1502.05477)
- Schulman, J., Moritz, P., Levine, S., Jordan, M., Abbeel, P. (2015). High-Dimensional Continuous Control Using Generalized Advantage Estimation. [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). Proximal Policy Optimization Algorithms. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Radford, A., Amodei, D., Christiano, P., Irving, G. (2019). Fine-Tuning Language Models from Human Preferences. [arXiv:1909.08593](https://arxiv.org/abs/1909.08593)
- Stiennon, N., Ouyang, L., Wu, J., Ziegler, D. M., Lowe, R., Voss, C., Radford, A., Amodei, D., Christiano, P. (2020). Learning to summarize from human feedback. NeurIPS 2020. [arXiv:2009.01325](https://arxiv.org/abs/2009.01325)
- Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training language models to follow instructions with human feedback (InstructGPT). [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- Bai, Y., Jones, A., Ndousse, K., et al. (2022), Anthropic. Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback. [arXiv:2204.05862](https://arxiv.org/abs/2204.05862)
- Huang, S., Noukhovitch, M., Hosseini, A., Rasul, K., Wang, W., Tunstall, L. (2024). The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization. [arXiv:2403.17031](https://arxiv.org/abs/2403.17031)
- Engstrom, L., Ilyas, A., Santurkar, S., Tsipras, D., Janoos, F., Rudolph, L., Madry, A. (2020). Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO. ICLR 2020. [arXiv:2005.12729](https://arxiv.org/abs/2005.12729)
- Andrychowicz, M., Raichuk, A., Stańczyk, P., et al. (2021). What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study. ICLR 2021. [arXiv:2006.05990](https://arxiv.org/abs/2006.05990)
- Zheng, R., Dou, S., Gao, S., et al. (2023). Secrets of RLHF in Large Language Models Part I: PPO. [arXiv:2307.04964](https://arxiv.org/abs/2307.04964)
- Gao, L., Schulman, J., Hilton, J. (2022). Scaling Laws for Reward Model Overoptimization. [arXiv:2210.10760](https://arxiv.org/abs/2210.10760)
- Singhal, P., Goyal, T., Xu, J., Durrett, G. (2023). A Long Way to Go: Investigating Length Correlations in RLHF. [arXiv:2310.03716](https://arxiv.org/abs/2310.03716)
- Ahmadian, A., Cremer, C., Gallé, M., et al. (2024). Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs. [arXiv:2402.14740](https://arxiv.org/abs/2402.14740)

### Other sources

- Huang, S., Dossa, R. F. J., Raffin, A., Kanervisto, A., Wang, W. (2022). The 37 Implementation Details of Proximal Policy Optimization. ICLR Blog Track. [iclr-blog-track.github.io](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- Huang, S., Liu, T., von Werra, L. (2023). The N Implementation Details of RLHF with PPO. Hugging Face blog. [huggingface.co/blog](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo)
- Code for the N+ paper: [github.com/vwxyzjn/summarize_from_feedback_details](https://github.com/vwxyzjn/summarize_from_feedback_details)
- OpenAI Spinning Up: Proximal Policy Optimization. [spinningup.openai.com](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- Schulman, J. Approximating KL Divergence (the $$k_1$$ and $$k_3$$ estimators). [joschu.net](http://joschu.net/blog/kl-approx.html)
- TRL documentation, PPO Trainer. [huggingface.co/docs/trl](https://huggingface.co/docs/trl/main/en/ppo_trainer)
- Models and data: [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [Skywork-Reward-V2-Qwen3-0.6B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B), [UltraFeedback binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized).
