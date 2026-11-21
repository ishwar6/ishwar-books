---
title: "GRPO: Learning from a Group of Answers"
description: "The learning rule behind DeepSeek-R1, term by term: sample a group of answers, score them, and push the model toward the better ones without a critic. We write GRPO from scratch, match it with TRL, and run one real step on a small model."
part: 2
covers: "§2.1, §2.2, §2.2.1, Equations 1 to 3"
date: 2026-11-21
tags: [deepseek, grpo, reinforcement-learning, ppo, reasoning]
---

DeepSeek-R1-Zero learned to reason from one short rule: try a question many times, then make the better tries more likely. This page reads the three equations that turn that rule into training, slowly, with numbers you can check by hand and code that runs.

{{FIG:p2_here|You are here. GRPO is the reinforcement learning step of the paper. It turns DeepSeek-V3-Base into R1-Zero, and it runs again in two of the four stages that make DeepSeek-R1.}}

## Three things the paper does {§2.1}

> [!PAPER] · §2.1 · page 5
> [![Section 2.1 of the paper, with heavily relied on large amounts of supervised data, even without using supervised, small amount of cold-start data, applies RL directly to the base model without any SFT data and Distill the reasoning capability highlighted](/img/papers/deepseek-r1/p2-overview.png)](/img/papers/deepseek-r1/p2-overview.png)
>
> **What it says:** earlier work "heavily relied on large amounts of supervised data". This paper improves reasoning with reinforcement learning "even without using supervised fine-tuning (SFT) as a cold start". Then it lists three results: R1-Zero, which "applies RL directly to the base model without any SFT data"; R1, which starts from "a small amount of cold-start data"; and models that "distill the reasoning capability" of R1.
>
> **What it means:**
>
> The usual way to teach a model a skill is to show it many worked examples. People write a good answer for each question, and the model learns to copy them. That is supervised fine-tuning. For long reasoning, such examples are slow and costly to write, and a person's solution is not always the best path for a model.
>
> This section says the authors tried something else first. They gave the base model only questions and a way to check the final answer. No worked solutions at all. The result is R1-Zero. Then they added a few thousand written examples at the start, to make the output readable, and got R1. Last, they copied R1's skill into small models.
>
> All three results rest on one learning rule. It is the subject of this page. [Part 4](part-4-r1-zero-and-the-aha-moment.md) shows what R1-Zero learned with it, [Part 5](part-5-the-r1-pipeline.md) the R1 recipe, and [Part 6](part-6-distillation.md) the small models.

## Learning without worked examples {§2.2}

> [!PAPER] · §2.2 · page 5
> [![The first paragraph of Section 2.2, with heavily depended on supervised data, which are time-intensive to gather, without any supervised data and pure reinforcement learning process highlighted](/img/papers/deepseek-r1/p2-pure-rl.png)](/img/papers/deepseek-r1/p2-pure-rl.png)
>
> **What it says:** their earlier reinforcement learning work "heavily depended on supervised data, which are time-intensive to gather". Here they test what a model can learn "without any supervised data", through "a pure reinforcement learning process".
>
> **What it means:**
>
> The "previous works" are DeepSeek's own. DeepSeekMath (Shao et al., 2024) first fine-tuned its model on 776,000 worked maths examples (its §3.1), and only then ran reinforcement learning on top.
>
> The new test removes the supervised step. The model gets no supervised fine-tuning on worked reasoning before RL. It has read a great deal of text in pretraining, maths included, but nobody tunes it to copy step-by-step solutions.
>
> It must find good reasoning by itself, guided only by a score at the end. Think of a student who has read many textbooks, but on the homework gets only "right" or "wrong" for each answer, never a model solution. If the student keeps trying, and keeps what worked, the student can still improve. The question is whether a language model can do the same at scale.

To learn from scores, the model needs a rule that turns a score into a change of its weights. Reinforcement learning has a name for the model in this role.

> [!DEFINITION] Policy
> In reinforcement learning, the **policy** is the thing that chooses actions. Here it is the language model itself. At each step it gives a probability to every possible next token, and sampling from them writes the answer. We write it $$\pi_\theta$$, where $$\theta$$ are the weights. $$\pi_\theta(o \mid q)$$ is the probability that the model writes the whole answer $$o$$ to the question $$q$$.

## Group Relative Policy Optimization {§2.2.1}

> [!PAPER] · §2.2.1 · page 5
> [![The start of Section 2.2.1, with save the training costs of RL, foregoes the critic model, typically the same size as the policy model, estimates the baseline from group scores instead and samples a group of outputs highlighted](/img/papers/deepseek-r1/p2-grpo-intro.png)](/img/papers/deepseek-r1/p2-grpo-intro.png)
>
> **What it says:** "to save the training costs of RL", they use GRPO, which "foregoes the critic model that is typically the same size as the policy model, and estimates the baseline from group scores instead". For each question, GRPO "samples a group of outputs".
>
> **What it means:**
>
> This one sentence carries three words that need background: critic, baseline and group. The rest of this section gives that background, one piece at a time. The short version is a cost argument. The usual algorithm, PPO, trains a second large network next to the language model. GRPO removes that second network and uses a group of answers to the same question in its place.
>
> The authors chose GRPO because it is cheap, not because it is the only rule that works. The later long version of the paper compares it with a well-tuned PPO and finds PPO can come close. GRPO is simpler and needs less memory.

### Background: the problem every policy-gradient method solves

Start with the simplest rule. The model writes an answer. If the reward is high, make every token of that answer a little more likely. If it is low, make them a little less likely. This is the **policy gradient**, and its basic form is "reward times the gradient of the log-probability of the answer".

A small example shows the weakness. Say rewards are 1 for a right answer and 0 for a wrong one. Then wrong answers get a push of 0: the model learns nothing from its mistakes. Now say a different task gives rewards between 8 and 10. Every answer gets a large positive push, even the worst one. The model learns slowly, because the useful signal (9.8 is better than 8.1) hides in a large shared number.

The fix is to subtract a **baseline**: a guess of the usual reward for this question. Then only "better than usual" gets a positive push and "worse than usual" gets a negative one. The difference has its own name.

> [!DEFINITION] Advantage
> The **advantage** of an answer is how much better it did than expected: its reward minus a baseline. A positive advantage means "do this more", a negative one means "do this less". In GRPO the advantage of answer $$i$$ is written $$A_i$$, and the baseline is the average reward of the group (Equation 3).

The hard part is the baseline. A question that the model nearly always solves should have a high baseline, and a hard question a low one. Something must estimate it for each question. PPO trains a network for that.

[The Road to R1](background-the-road-to-r1.md) tells how PPO came to language models through human feedback. Here we need only its structure.

### PPO and its critic

> [!PAPER] Shao et al. (2024), DeepSeekMath · §4.1.1 · page 11
> [![Section 4.1.1 of DeepSeekMath with the PPO objective, Equation 1, and actor-critic RL algorithm and Generalized Advantage Estimation (GAE) highlighted](/img/papers/deepseek-r1/p2-dsm-ppo.png)](/img/papers/deepseek-r1/p2-dsm-ppo.png)
>
> **What it says:** PPO "is an actor-critic RL algorithm", widely used to fine-tune language models. It maximises an objective with a ratio of new to old probabilities, a clip with $$\varepsilon$$, and an advantage $$A_t$$ "computed by applying Generalized Advantage Estimation (GAE)".
>
> **What it means:**
>
> "Actor-critic" means two networks. The actor is the policy: the language model that writes. The critic is a second network that watches each partial answer and predicts the reward that will come at the end. Its prediction is the baseline. GAE is a recipe that combines those predictions, token by token, into an advantage for every token.
>
> Look at the shape of PPO's objective. It already has the ratio, the clip and the min. GRPO keeps all of that. What GRPO changes is only where $$A$$ comes from.

> [!DEFINITION] Value model (critic)
> A network that reads the question and the first $$t$$ tokens of an answer and predicts the final reward. It is trained next to the policy. In language-model training it is usually a language model of the same size with a one-number output head.

> [!PAPER] Shao et al. (2024), DeepSeekMath · §4.1.1 · page 13
> [![A paragraph of DeepSeekMath with another model of comparable size as, substantial memory and computational burden, only the last token is assigned a reward score and uses the average reward of multiple sampled outputs highlighted](/img/papers/deepseek-r1/p2-dsm-burden.png)](/img/papers/deepseek-r1/p2-dsm-burden.png)
>
> **What it says:** the value function in PPO is "another model of comparable size as the policy model", which "brings a substantial memory and computational burden". Also, "only the last token is assigned a reward score", which makes a per-token value function hard to train. GRPO instead "uses the average reward of multiple sampled outputs" to the same question as the baseline.
>
> **What it means:**
>
> The first problem is size: a 7-billion-parameter policy needs a critic of about 7 billion parameters. The critic is trained, so it also needs gradients and optimizer state. The second problem is the signal. The reward arrives once, after the last token. The critic must still guess it after token 5, token 50 and token 500. Those middle guesses get very little direct feedback.
>
> The fix avoids the guessing. To know the usual reward for a question, just answer it several times and average. That average is a real measurement, not a prediction, and it costs no extra network.

The later long version of the R1 paper adds a reason that matters for long reasoning.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1, later long version · Supplementary A.3 · page 15
> [![A paragraph of the long version of the R1 paper, with inherently difficult, especially when only the final outcome reward is available and reflection and revision highlighted](/img/papers/deepseek-r1/p2-long-value.png)](/img/papers/deepseek-r1/p2-long-value.png)
>
> **What it says:** predicting the final reward from a partial answer is "inherently difficult, especially when only the final outcome reward is available". With long chains of thought, the model does "reflection and revision", so early content "may later be revised or contradicted".
>
> **What it means:**
>
> Picture a critic reading a reasoning model halfway through, just after a wrong intermediate result. Is the answer doomed? Not if the model later writes "Wait, let me check" and fixes it. R1-Zero learned exactly this habit ([Part 4](part-4-r1-zero-and-the-aha-moment.md)). A critic that judges partial answers would often be wrong about such answers. GRPO judges only complete answers, so this problem does not occur.

How big is the saving? Our script counts the memory for the weights and the optimizer state only. A trained model with mixed-precision Adam keeps about 16 bytes per parameter (bf16 weights and gradients, fp32 master weights and two fp32 Adam moments). A frozen model that only runs forward needs 2 bytes per parameter. We assume the critic and the reward model have the same size as the policy.

```text
== 7B policy: 7B parameters per model
PPO                  policy 112 + reference 14 + reward model 14 + value model 112 = 252 GB
GRPO, reward model   policy 112 + reference 14 + reward model 14 = 140 GB
GRPO, rule rewards   policy 112 + reference 14 = 126 GB
GRPO with rule rewards needs 50% of the memory of PPO; it saves 126 GB
80 GB GPUs just to hold these (rounded up): PPO 4, GRPO with rule rewards 2
the value model alone: 112 GB, 2 GPUs of 80 GB (rounded up)

== DeepSeek-V3 size: 671B parameters per model
PPO                  policy 10,736 + reference 1,342 + reward model 1,342 + value model 10,736 = 24,156 GB
GRPO, reward model   policy 10,736 + reference 1,342 + reward model 1,342 = 13,420 GB
GRPO, rule rewards   policy 10,736 + reference 1,342 = 12,078 GB
GRPO with rule rewards needs 50% of the memory of PPO; it saves 12,078 GB
80 GB GPUs just to hold these (rounded up): PPO 302, GRPO with rule rewards 151
the value model alone: 10,736 GB, 135 GPUs of 80 GB (rounded up)
```

For a model the size of DeepSeek-V3 (671 billion parameters, see [the V3 architecture](/writings/deepseek-2-the-v3-architecture/)), the critic alone would take 10,736 GB in this count. That is 135 GPUs of 80 GB only to hold one network. R1-Zero also has no reward model: its rewards are rules in a program ([Part 3](part-3-rewards-and-the-template.md)). So GRPO in R1-Zero trains one large network and keeps one frozen copy. These numbers are a rough count. Real training adds activations and the memory for sampling, and DeepSeek uses FP8 and other savings.

{{FIG:p2_ppo_grpo|PPO and GRPO side by side. PPO keeps four models and trains two of them, the policy and the value model. GRPO as used for R1-Zero trains only the policy, keeps a frozen reference, scores answers with a program, and has no critic. The memory figures come from our script.}}

### Where GRPO came from

GRPO was not new in R1. DeepSeek introduced it a year earlier, in the DeepSeekMath paper, to train a 7B maths model.

> [!PAPER] Shao et al. (2024), DeepSeekMath · Figure 4 · page 13
> [![Figure 4 of DeepSeekMath. Top: PPO, where the policy's output goes to a reference model, a reward model and a value model; the reward and the value feed GAE, which gives A. Bottom: GRPO, where the policy writes outputs o1 to oG, a reward model gives r1 to rG, and a group computation gives A1 to AG. Trained models are yellow, frozen models blue. GRPO foregoes the value model and estimating the baseline from group scores are highlighted](/img/papers/deepseek-r1/p2-dsm-figure4.png)](/img/papers/deepseek-r1/p2-dsm-figure4.png)
>
> **What it says:** "GRPO foregoes the value model, instead estimating the baseline from group scores, significantly reducing training resources."
>
> **What it means:**
>
> Read the two rows from left to right. In the PPO row, one output $$o$$ goes to three models, and the yellow value model is trained. In the GRPO row, the policy writes $$G$$ outputs to the same question $$q$$. Each gets a reward. The box "Group Computation" turns the $$G$$ rewards into $$G$$ advantages. The yellow value model is gone.
>
> Notice where the KL arrow points. In PPO it joins the reward, before GAE. In GRPO it goes straight back to the policy. The next boxes explain that difference.

> [!PAPER] Shao et al. (2024), DeepSeekMath · Equation 3 · page 13
> [![Equation 3 of DeepSeekMath, the GRPO objective with a sum over the G outputs, 1 over the length of output i, a sum over its tokens t, and the min of the ratio times A hat i t and the clipped ratio times A hat i t, minus beta times the KL; with instead of adding KL and directly adding the KL divergence highlighted](/img/papers/deepseek-r1/p2-dsm-eq3.png)](/img/papers/deepseek-r1/p2-dsm-eq3.png)
>
> **What it says:** the GRPO objective averages over the $$G$$ outputs, and inside each output it averages over its tokens: $$\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}$$. The ratio is per token, $$\pi_\theta(o_{i,t} \mid q, o_{i,<t})$$. And "instead of adding KL penalty in the reward, GRPO regularizes by directly adding the KL divergence" to the loss.
>
> **What it means:**
>
> This is GRPO in its original, more detailed form. R1's Equation 1 is the same idea written in a shorter way. Two details matter here and below. First, everything is per token: each token has its own ratio, and an answer's score is the average over its tokens. Second, the KL term sits in the loss, outside the advantage, so it does not mix into the group comparison.

> [!PAPER] Shao et al. (2024), DeepSeekMath · §4.1.2 · page 14
> [![Section 4.1.2 of DeepSeekMath, with subtracting the group average and dividing by the group standard deviation and sets the advantages highlighted](/img/papers/deepseek-r1/p2-dsm-outcome.png)](/img/papers/deepseek-r1/p2-dsm-outcome.png)
>
> **What it says:** the $$G$$ rewards are normalised "by subtracting the group average and dividing by the group standard deviation". With a single reward at the end, this "sets the advantages" of all tokens of an output to the same normalised reward.
>
> **What it means:**
>
> This is the "group relative" in the name. Each answer is graded on a curve, against the other answers to the same question. And each token of an answer gets the grade of the whole answer. The method does not try to say which token was the good one. Over many questions, tokens that often appear in good answers get pushed up more often than they get pushed down.

> [!PAPER] Shao et al. (2024), DeepSeekMath · §4.2 · page 15
> [![The training settings of DeepSeekMath-RL, with The KL coefficient is 0.04, we sample 64 outputs and single update following each exploration stage highlighted](/img/papers/deepseek-r1/p2-dsm-settings.png)](/img/papers/deepseek-r1/p2-dsm-settings.png)
>
> **What it says:** "The KL coefficient is 0.04. For each question, we sample 64 outputs." The policy has "a single update following each exploration stage".
>
> **What it means:**
>
> These are the first published GRPO settings: $$\beta = 0.04$$ and a group of $$G = 64$$. Keep them in mind. R1 v1 does not give its own values; the later version does, and they differ (below).

Here is the whole loop on a real group from our run. The model is Qwen2.5-0.5B-Instruct, the question comes from GSM8K, and the details are in the last section.

{{FIG:p2_loop|The GRPO loop on our real group. One GSM8K question, eight answers sampled from the model, a reward of 1 for a right final answer and 0 for a wrong one, the group-normalised advantages of Equation 3, and an update that raises the answers with a positive advantage.}}

## Equation 1, term by term {§2.2.1}

> [!PAPER] · Equation 1 · page 5
> [![Equation 1 of the paper, highlighted: J GRPO of theta equals the expectation over a question q and a group of G outputs from pi theta old, of 1 over G times the sum over i of the min of the ratio times A i and the clipped ratio times A i, minus beta times the KL divergence between pi theta and pi ref](/img/papers/deepseek-r1/p2-eq1.png)](/img/papers/deepseek-r1/p2-eq1.png)
>
> **What it says:** the policy is trained by maximising $$\mathcal{J}_{GRPO}(\theta)$$, an average over a group of $$G$$ outputs of a clipped, ratio-weighted advantage, minus $$\beta$$ times a KL divergence to a reference policy.
>
> **What it means:**
>
> In words: sample a group, then make the good answers more likely and the bad ones less likely. Do it in small steps, and do not drift far from the start. Each part of the equation does one of these jobs. We read them one at a time below.

Here is Equation 1 again, the same as the paper:

$$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\Big[q \sim P(Q),\ \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{old}}(O \mid q)\Big]
\frac{1}{G}\sum_{i=1}^{G}\left( \min\!\left( \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{old}}(o_i \mid q)} A_i,\ \text{clip}\!\left(\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{old}}(o_i \mid q)}, 1-\varepsilon, 1+\varepsilon\right) A_i \right) - \beta\, \mathbb{D}_{KL}\!\left(\pi_\theta \,\|\, \pi_{ref}\right) \right)
$$

### The expectation: where the data comes from

$$\mathbb{E}[\dots]$$ means "on average over". The bracket says what we average over. $$q \sim P(Q)$$ picks a question from the training set. $$\{o_i\}_{i=1}^{G} \sim \pi_{\theta_{old}}(O \mid q)$$ samples $$G$$ answers to that question from $$\pi_{\theta_{old}}$$. In code, the expectation is just a batch: many questions, $$G$$ answers each.

Why $$\pi_{\theta_{old}}$$ and not $$\pi_\theta$$? Sampling long answers is the slow part of training. So the trainer samples a big batch once, with a frozen copy of the weights. Then it makes several gradient steps on that batch. During those steps the weights $$\theta$$ move, but the answers were written by the old weights $$\theta_{old}$$. The equation must correct for that, and the ratio does it.

> [!DEFINITION] Old policy
> $$\pi_{\theta_{old}}$$ is the model as it was when the current batch of answers was sampled. It is the same network as $$\pi_\theta$$, frozen at an earlier moment, so it costs no extra memory: its log-probabilities are computed once and stored.

### The ratio: how much more likely is this answer now

$$
\rho_i = \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{old}}(o_i \mid q)}
$$

The ratio compares the probability of the same answer under the new and the old weights. Take one token. The old model gave it probability 0.20. After a few steps, the model gives it 0.25. The ratio is $$0.25 / 0.20 = 1.25$$: the token is now 25% more likely. A ratio of 1 means no change.

In code we never divide two small probabilities. We store log-probabilities and use $$\rho = \exp(\log \pi_\theta - \log \pi_{\theta_{old}})$$. The two forms are equal, and the second one does not underflow.

At the first step on a new batch, $$\theta = \theta_{old}$$, so every ratio is exactly 1. The reward part of the objective (the clipped term, or surrogate) is then $$\frac{1}{G}\sum_i A_i$$. That is 0, because the advantages of a group sum to zero.

The full objective also subtracts $$\beta$$ times the KL to the reference. That term is 0 only when the current model also equals the reference. This is true at the very start of training, and in our real run below, where the reference is a copy of the current weights. Later in training the two have drifted apart, so the KL term is not 0 even at the first step on a batch.

Either way, the gradient of the reward part is not 0. The gradient of $$\rho_i A_i$$ at $$\rho_i = 1$$ is $$A_i \nabla_\theta \log \pi_\theta(o_i \mid q)$$. That is the plain policy gradient with the group as baseline. Our real run shows both facts.

### The clip and the min: small, safe steps

A step can be too large. One lucky batch could push the probability of some answer from 0.2 to 0.9, and the model would forget other useful behaviour. PPO's clip removes the reward for moving each probability too far from the old model in one batch.

> [!PAPER] Schulman et al. (2017), PPO · §3 · page 3
> [![Section 3 of the PPO paper, with the clipped objective L CLIP, Equation 7, and removes the incentive for moving r t outside of the interval and pessimistic bound highlighted](/img/papers/deepseek-r1/p2-ppo-clip.png)](/img/papers/deepseek-r1/p2-ppo-clip.png)
>
> **What it says:** clipping the ratio "removes the incentive for moving $$r_t$$ outside of the interval $$[1-\epsilon, 1+\epsilon]$$". Taking the minimum of the clipped and unclipped terms makes the objective "a lower bound (i.e., a pessimistic bound)". The suggested value is $$\epsilon = 0.2$$.
>
> **What it means:**
>
> $$\text{clip}(\rho, 0.8, 1.2)$$ holds the ratio inside $$[0.8, 1.2]$$: a ratio of 1.5 becomes 1.2, and 0.5 becomes 0.8. Once a good token is already 20% more likely than in the old model, the objective stops paying for more. The gradient of that token's term becomes zero, so this term stops pushing it. This is a missing incentive, not a hard limit. Every token shares the same weights. So the push from other tokens, the KL term or Adam's momentum can still move it past 1.2.
>
> The min makes the clip one-sided, in the careful direction. Say a step went the wrong way and a good answer became less likely. Then the min keeps the unclipped term, so the error is still fully penalised. Only "too much of a good thing" is cut off. That is why the authors call it pessimistic.

The picture shows one token's term as a function of the ratio. The dashed line is $$\rho A$$ without the clip; the solid line is what Equation 1 uses.

{{FIG:p2_clip|One token of Equation 1 as a function of the ratio, with epsilon 0.2. Left, a good answer (A = +1): the term rises with the ratio and goes flat above 1.2. Right, a bad answer (A = -1): the term falls with the ratio and goes flat below 0.8. The dot is the start, ratio 1. The shaded band is the interval from 0.8 to 1.2.}}

Our script computes the term and its slope (the gradient with respect to $$\rho$$) for a few ratios. A slope of 0 means the token gets no push from this batch.

```text
== 2. One token of Equation 1: min(rho*A, clip(rho, 0.8, 1.2)*A), epsilon = 0.2
  rho  |  A=+1: term   d/d(rho) |  A=-1: term   d/d(rho)
  0.5  |  +0.50       +1.0     |  -0.80       +0.0
  0.7  |  +0.70       +1.0     |  -0.80       +0.0
  0.9  |  +0.90       +1.0     |  -0.90       -1.0
  1.0  |  +1.00       +1.0     |  -1.00       -1.0
  1.1  |  +1.10       +1.0     |  -1.10       -1.0
  1.3  |  +1.20       +0.0     |  -1.30       -1.0
  1.5  |  +1.20       +0.0     |  -1.50       -1.0
```

Read the left half. For a good answer, ratios below 1.2 keep a slope of +1: keep raising it. At 1.3 and 1.5 the slope is 0: it has risen enough for this batch. Now the right half. For a bad answer at ratio 0.5 or 0.7 the slope is 0: it has already dropped enough. At 1.3 the bad answer became *more* likely by mistake, and the slope stays at -1, so the error is corrected.

### The average over the group: 1/G

$$\frac{1}{G}\sum_{i=1}^{G}$$ averages the $$G$$ answers of one question, so a question counts the same whether $$G$$ is 8 or 64. Inside each answer, the paper's short form hides a second average, over tokens. We come back to it in the per-token section.

### The KL penalty: stay near the reference

$$-\beta\, \mathbb{D}_{KL}(\pi_\theta \,\|\, \pi_{ref})$$ subtracts a cost for moving away from a reference model. The clip limits one batch. The KL term limits the whole run: after thousands of steps, the model should still be close enough to a sensible starting point. $$\beta$$ sets how strong that pull is.

> [!DEFINITION] Reference policy
> $$\pi_{ref}$$ is a frozen copy of the model, usually the model before reinforcement learning started. It only runs forward passes. The later long version of the paper replaces it with the latest policy every 400 steps. The pull is then toward a recent model, not the first one.

> [!DEFINITION] KL divergence
> A measure of how different two probability distributions are: $$\mathbb{D}_{KL}(p \,\|\, r) = \sum_x p(x) \log \frac{p(x)}{r(x)}$$. It is 0 when $$p = r$$ and positive otherwise. It is not symmetric: here the sum is weighted by the policy's own probabilities $$p = \pi_\theta$$.

### The paper's ratio is for a whole answer; code uses tokens

R1's Equation 1 writes $$\pi_\theta(o_i \mid q)$$, the probability of the whole answer. A language model writes the answer token by token, so this is a product: $$\pi_\theta(o_i \mid q) = \prod_t \pi_\theta(o_{i,t} \mid q, o_{i,<t})$$. The ratio of the whole answer is then the product of the ratios of its tokens.

A product of many numbers near 1 is not near 1. Our script shows it:

```text
== 5. One ratio for a whole answer, or one per token
each of 300 tokens has ratio 1.01: ratio of the whole answer = 1.01^300 = 19.79
each of 300 tokens has ratio 0.99: ratio of the whole answer = 0.99^300 = 0.04904
each of 1000 tokens has ratio 1.01: ratio of the whole answer = 1.01^1000 = 2.096e+04
```

A change of 1% per token makes the whole-answer ratio 19.79 after 300 tokens. A clip at 1.2 would then stop almost every long answer. R1's answers run to thousands of tokens. So in practice the ratio and the clip apply to each token, and an answer's score is the mean over its tokens. That is DeepSeekMath's Equation 3 with its $$\frac{1}{|o_i|}\sum_t$$, which TRL's `GRPOTrainer` and our code compute. And the R1 paper's short form is best read as notation for this per-token version.

## Equation 2: a KL that is never negative {§2.2.1}

> [!PAPER] · Equation 2 · page 5
> [![Equation 2 of the paper, highlighted: D KL of pi theta and pi ref equals pi ref of o i given q over pi theta of o i given q, minus the log of the same ratio, minus 1](/img/papers/deepseek-r1/p2-eq2.png)](/img/papers/deepseek-r1/p2-eq2.png)
>
> **What it says:** $$\mathbb{D}_{KL}(\pi_\theta \,\|\, \pi_{ref}) = \frac{\pi_{ref}(o_i \mid q)}{\pi_\theta(o_i \mid q)} - \log\frac{\pi_{ref}(o_i \mid q)}{\pi_\theta(o_i \mid q)} - 1$$.
>
> **What it means:**
>
> The true KL is a sum over every possible answer, and there are far too many answers to sum. So the paper uses an estimate from the answers it already sampled. This particular formula is known as the "k3" estimator, from a 2020 note by John Schulman. Its two useful properties are shown below with numbers.

> [!PAPER] Shao et al. (2024), DeepSeekMath · §4.1.1 · page 14
> [![The KL estimator in DeepSeekMath, Equation 4, with unbiased estimator (Schulman, 2020) and guaranteed to be positive highlighted](/img/papers/deepseek-r1/p2-dsm-kl.png)](/img/papers/deepseek-r1/p2-dsm-kl.png)
>
> **What it says:** they estimate the KL "with the following unbiased estimator (Schulman, 2020)", per token, "which is guaranteed to be positive".
>
> **What it means:**
>
> "Unbiased" means that on average the estimate equals the true KL. "Guaranteed to be positive" means that each single estimate is at least 0 (it is 0 exactly when the two models agree). The simple estimate $$\log \frac{\pi_\theta}{\pi_{ref}}$$ is also unbiased, but a single value can be negative. That is not an error. Single values of a fair estimate land on both sides of the truth, and only their average has to equal the KL. The cost is noise: the positive and negative values cancel only over many samples.

Write $$x = \pi_{ref}(o) / \pi_\theta(o)$$ for one sampled token. Equation 2 is then $$k_3 = x - \log x - 1$$.

**Why it is never negative.** The curve $$\log x$$ always lies below its tangent line at $$x = 1$$, which is $$x - 1$$. So $$\log x \le x - 1$$ for every $$x > 0$$, and $$x - \log x - 1 \ge 0$$. The two are equal only at $$x = 1$$, where the two models agree.

**Why its average is the true KL.** Average over tokens sampled from $$\pi_\theta$$. The first part averages to $$\sum_o \pi_\theta(o) \frac{\pi_{ref}(o)}{\pi_\theta(o)} = \sum_o \pi_{ref}(o) = 1$$. So the $$x$$ and the $$-1$$ cancel on average, and what is left is the average of $$-\log x = \log \frac{\pi_\theta}{\pi_{ref}}$$. That average is exactly the definition of $$\mathbb{D}_{KL}(\pi_\theta \,\|\, \pi_{ref})$$.

Now with numbers. The first table is $$k_1 = -\log x$$ (the simple estimate) and $$k_3$$ for a few ratios. Then two small distributions over 5 tokens, the exact KL, and 100,000 sampled tokens:

```text
== 3. Equation 2, the k3 estimator: x - log x - 1 with x = pi_ref / pi_theta
   x      k1 = -log x    k3 = x - log x - 1
  0.25     +1.3863         0.6363
  0.50     +0.6931         0.1931
  0.90     +0.1054         0.0054
  1.00     +0.0000         0.0000
  1.10     -0.0953         0.0047
  2.00     -0.6931         0.3069
  4.00     -1.3863         1.6137
pi_theta = [0.500, 0.200, 0.150, 0.100, 0.050]
pi_ref   = [0.400, 0.300, 0.100, 0.100, 0.100]
true KL(pi_theta || pi_ref) = sum pi_theta log(pi_theta/pi_ref) = 0.05664
exact average of k3 under pi_theta                         = 0.05664
100,000 tokens sampled from pi_theta: mean k1 0.05613 (std 0.3213, 25.0% of samples below 0)
                                      mean k3 0.05662 (std 0.0659, smallest 0.0000)
```

Check one row by hand. For $$x = 2$$: $$2 - \log 2 - 1 = 2 - 0.6931 - 1 = 0.3069$$. Both estimators average to about the true KL of 0.0566. But a single $$k_1$$ is below zero for 25% of the sampled tokens. Its spread (0.3213) is also almost five times the spread of $$k_3$$ (0.0659).

Both are fair on average, so the gain from $$k_3$$ is not that it is "more right". The gain is a smaller spread for the same average, so fewer samples give a steady estimate.

Note also that "never negative" is a fact about each value, not about its gradient. The gradient is a separate question. One token's $$k_3$$ has gradient $$(1 - x)\nabla_\theta \log \pi_\theta(o)$$. Averaged over tokens sampled from $$\pi_\theta$$, that is $$-\sum_o \pi_{ref}(o)\nabla_\theta \log \pi_\theta(o)$$, the gradient of $$\mathbb{D}_{KL}(\pi_{ref} \,\|\, \pi_\theta)$$: the KL with its two sides swapped. Close to the reference the two directions of the KL nearly agree, so in practice the term still keeps the model near the reference.

{{FIG:p2_k3|The two per-token KL estimates as functions of x, the ratio of reference to policy probability. The plain estimate k1 = -log x goes below zero when x is above 1. Equation 2, k3 = x - log x - 1, stays at or above zero and touches zero only at x = 1.}}

There is one more choice in Equation 1: the KL goes straight into the loss. PPO for language models usually subtracts a KL penalty from the reward at every token. The later long version of the paper says that in PPO this penalises the cumulative KL, which "may implicitly penalize the length of the response". R1-Zero's answers had to grow long, so a penalty that grows with length would work against it.

## Equation 3: the group is the baseline {§2.2.1}

> [!PAPER] · Equation 3 · page 5
> [![The text after Equation 2 and Equation 3, with are hyper-parameters and computed using a group of rewards highlighted, and Equation 3 marked: A i equals r i minus the mean of r 1 to r G, divided by the std of r 1 to r G](/img/papers/deepseek-r1/p2-eq3.png)](/img/papers/deepseek-r1/p2-eq3.png)
>
> **What it says:** $$\varepsilon$$ and $$\beta$$ "are hyper-parameters", and the advantage $$A_i$$ is "computed using a group of rewards": $$A_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\})}$$.
>
> **What it means:**
>
> This is the baseline without a critic. Subtract the group's mean reward: answers above the mean get a positive advantage, answers below get a negative one. Divide by the group's standard deviation: every question then gives advantages of about the same size, whether its rewards spread a lot or a little.
>
> Note what the paper does not say: the values of $$\varepsilon$$, $$\beta$$ and $$G$$. Version 1 gives none of them. We come back to this below.

### Four answers, by hand

One question, $$G = 4$$ answers. Answers 1 and 4 are right (reward 1), answers 2 and 3 are wrong (reward 0).

The mean is $$(1 + 0 + 0 + 1)/4 = 0.5$$. Each answer is 0.5 away from the mean, so the standard deviation is 0.5. Then:

$$
A_1 = \frac{1 - 0.5}{0.5} = +1, \qquad A_2 = \frac{0 - 0.5}{0.5} = -1, \qquad A_3 = -1, \qquad A_4 = +1
$$

The right answers are pushed up, the wrong ones down, with equal force.

There is one detail the paper leaves open: the standard deviation can divide by $$G$$ or by $$G - 1$$. PyTorch's `torch.std` and TRL divide by $$G - 1$$ (Bessel's correction), which makes the standard deviation 0.577 here and the advantages $$\pm 0.866$$. The direction is the same. Only the size changes, by a fixed factor that the learning rate absorbs. Our code uses $$G - 1$$, like TRL, so the two methods match.

Now a harder question: only answer 1 is right.

```text
rewards [1.0, 0.0, 0.0, 0.0]: mean 0.250, std (divide by G) 0.433, std (divide by G-1) 0.500
   A, std divides by G:   [1.732, -0.577, -0.577, -0.577]   sum 0.000
   A, std divides by G-1: [1.500, -0.500, -0.500, -0.500]   sum 0.000
```

The one right answer gets a large push (+1.732), and each wrong one a small push down (-0.577). This is the useful part of the rule. A rare success on a hard question is exactly what the model should learn from, and the curve makes it count most. In every case the advantages of a group sum to zero: the group pushes up as much as it pushes down.

### When every answer gets the same reward

If all four answers are right, the mean is 1 and every $$r_i - \text{mean}$$ is 0. The standard deviation is also 0, so Equation 3 is $$0/0$$. Code needs a small number in the denominator, and TRL adds $$10^{-4}$$:

```text
rewards [1.0, 1.0, 1.0, 1.0]: std 0.000; no eps: [nan, nan, nan, nan]; eps 1e-4: [0.000, 0.000, 0.000, 0.000]
rewards [0.0, 0.0, 0.0, 0.0]: std 0.000; no eps: [nan, nan, nan, nan]; eps 1e-4: [0.000, 0.000, 0.000, 0.000]
```

Without the small number, every advantage is "not a number", and one such value spoils the whole gradient. With it, every advantage is 0, and the reward part of the loss gives no signal for this question. Only the KL term can still move the model, by pulling it toward the reference. In our real run below that term is 0 as well, because the reference is a copy of the current weights.

Giving no reward signal is correct: if all answers are equally good, there is nothing to prefer, but it has a cost. A question that is too easy (all right) or too hard (all wrong) wastes all its $$G$$ samples. GRPO learns only from questions at the edge of what the model can do.

### The settings v1 does not give

R1 v1 names $$\varepsilon$$ and $$\beta$$ but gives no values, and it does not give $$G$$. The later long version of the paper (published with the Nature article) does.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1, later long version · §2.1 · page 3
> [![The R1-Zero training settings in the long version, with learning rate to 3e-6, the KL coefficient to 0.001, sample 16 outputs, replace the reference model with the latest policy model and single inner epoch highlighted](/img/papers/deepseek-r1/p2-long-settings.png)](/img/papers/deepseek-r1/p2-long-settings.png)
>
> **What it says:** for R1-Zero, the "learning rate to 3e-6, the KL coefficient to 0.001", and "we sample 16 outputs" per question. Each step uses 32 questions, so a batch of 512 answers. "Every 400 steps, we replace the reference model with the latest policy model." Each rollout of 8,192 outputs is split into 16 mini-batches and trained for "a single inner epoch".
>
> **What it means:**
>
> So $$G = 16$$, and $$\beta = 0.001$$, which is 40 times smaller than DeepSeekMath's 0.04. The KL pull in R1-Zero is very weak, and the reference moves every 400 steps. The authors wanted the model to wander far from the base model, and it did.
>
> The 16 mini-batches explain why the ratio matters. The answers are sampled once. The first mini-batch trains with ratio 1. By the sixteenth, the weights have moved fifteen times, so the ratios are no longer 1 and the clip can act.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1, later long version · §3.2.1 · pages 7 and 8
> [![The first RL stage settings of DeepSeek-R1 in the long version, with GRPO clip ratio epsilon to 10 highlighted](/img/papers/deepseek-r1/p2-long-clip.png)](/img/papers/deepseek-r1/p2-long-clip.png)
> [![A note in the long version, with truncation and instability during training highlighted](/img/papers/deepseek-r1/p2-long-clip-note.png)](/img/papers/deepseek-r1/p2-long-clip-note.png)
>
> **What it says:** for the first RL stage of R1, they set "the GRPO clip ratio $$\varepsilon$$ to 10". They add that "the clip ratio plays a crucial role": a lower value can lead to "truncation of gradients for a significant number of tokens", and a higher value "may cause instability during training".
>
> **What it means:**
>
> With $$\varepsilon = 10$$ the clip interval is $$[-9, 11]$$. A ratio is never negative, so the lower clip never acts, and only a token that became 11 times more likely gets cut. That is far from PPO's usual 0.2. The note explains why: with 0.2, many tokens hit the flat part of the clip and stop learning. Rare tokens, such as "Wait" at the start of a self-check, can change their probability a lot in one step. A tight clip would cut exactly those.
>
> For our code and run we use $$\varepsilon = 0.2$$ (PPO's value and TRL's default), so that the clip is visible, and $$\beta = 0.001$$ from the long version.

## GRPO in code, two ways

### Method 1: from scratch

Five short functions, one per idea. Shapes are in the comments: $$B$$ is the number of answers in the batch (questions times $$G$$) and $$T$$ the longest answer in tokens.

```python
def group_advantages(rewards, G, unbiased=True, eps=1e-4):
    r = rewards.view(-1, G)                                   # (n_questions, G): one row per question
    mean = r.mean(dim=1, keepdim=True)                        # (n_questions, 1): the group's average reward
    std = r.std(dim=1, keepdim=True, unbiased=unbiased)       # (n_questions, 1): the group's spread
    return ((r - mean) / (std + eps)).view(-1)                # (B,): better than the group > 0, worse < 0


def ratio(logp_new, logp_old):
    return torch.exp(logp_new - logp_old)                     # (B, T): pi_theta / pi_theta_old per token


def clipped_surrogate(rho, A, epsilon):
    A = A.unsqueeze(1)                                        # (B, 1): broadcast over the tokens of each answer
    unclipped = rho * A                                       # (B, T): the plain policy-gradient term
    clipped = torch.clamp(rho, 1 - epsilon, 1 + epsilon) * A  # (B, T): the ratio held inside [1 - eps, 1 + eps]
    return torch.minimum(unclipped, clipped)                  # (B, T): the smaller, more careful of the two


def k3_kl(logp_theta, logp_ref):
    log_r = logp_ref - logp_theta                             # (B, T): log of pi_ref / pi_theta
    return torch.exp(log_r) - log_r - 1                       # (B, T): x - log x - 1 >= 0 for every x > 0


def grpo_loss(logp, logp_old, logp_ref, A, mask, epsilon=0.2, beta=0.04):
    rho = ratio(logp, logp_old)                               # (B, T)
    per_token = clipped_surrogate(rho, A, epsilon) - beta * k3_kl(logp, logp_ref)   # (B, T): J per token
    per_answer = (per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)       # (B,): 1/|o_i| sum over t
    return -per_answer.mean()                                 # scalar: 1/G sum over i, with a minus to minimise
```

The last function is Equation 1 with the per-token practice. The `mask` is 1 on real answer tokens and 0 on padding, because the answers in a batch have different lengths. The minus sign at the end turns "maximise $$\mathcal{J}$$" into "minimise the loss", which is what optimisers do.

### Method 2: TRL's GRPOTrainer

TRL is the Hugging Face library for training language models with reinforcement learning. Its `GRPOTrainer` computes the loss in a method called `_compute_loss`. Here are its core lines, from trl 1.18 (shortened; the other loss types and the logging are left out):

```python
log_ratio = per_token_logps - old_per_token_logps
coef_1 = torch.exp(log_importance_weights)          # the ratio, per token
per_token_kl = (
    torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
)                                                   # Equation 2, k3
coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
per_token_loss1 = coef_1 * advantages
per_token_loss2 = coef_2 * advantages
per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
per_token_loss = per_token_loss + self.beta * per_token_kl
loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()   # loss_type="grpo"
```

It is the same equation, written by other people. We do not copy these lines; we call TRL's real method, which normally runs inside a full trainer with a model and a dataset. Our file `r1_part2_trl.py` gives it a small stand-in object with only the settings it reads (`loss_type='grpo'`, `beta`, `epsilon`). A stand-in model call returns our log-probabilities. Every line of the loss then runs as TRL wrote it.

TRL computes the advantages inside the method that also samples text. So we repeat those lines exactly: a group mean, a group standard deviation with $$G - 1$$, and $$+10^{-4}$$.

We feed both methods the same tensors: 2 questions with 4 answers each, 6 tokens per answer with some padding. The log-probabilities of the policy, the old policy and the reference all differ, so the ratios are not 1 and some tokens are clipped.

```text
== 4. Method 1 (from scratch) vs Method 2 (TRL GRPOTrainer._compute_loss), same tensors
advantages, ours: [0.866, -0.866, -0.866, 0.866, -0.500, -0.500, -0.500, 1.500]
advantages, TRL:  [0.866, -0.866, -0.866, 0.866, -0.500, -0.500, -0.500, 1.500]
max |difference| of advantages: 0.00e+00
loss, ours: 0.09479159
loss, TRL:  0.09479159
|difference| of the loss: 0.00e+00; max |difference| of the gradients: 0.00e+00
share of tokens in the flat (clipped) region: ours 0.1429, TRL metric 0.1429
agree to 1e-6: True
```

The advantages, the loss and the gradients agree to the last bit. In both, 14.29% of the real tokens sit in the flat part of the clip. The first group is our four-answer example (1, 0, 0, 1), with the $$G - 1$$ standard deviation: $$\pm 0.866$$. The second group has one right answer out of four: +1.5 for it, -0.5 for the others.

[![Terminal output of r1_part2_math.py with the advantages, the clip table, the k3 numbers and the agreement check](/img/papers/deepseek-r1/r1_part2_math-run.png)](/img/papers/deepseek-r1/r1_part2_math-run.png)

## One real GRPO step on a small model

Now the same loss on a real model. The script `r1_part2_run.py` loads Qwen2.5-0.5B-Instruct on a Mac (MPS, float32). For each GSM8K test question, in order, it samples $$G = 8$$ answers at temperature 1, with at most 512 new tokens. A check gives reward 1 when the whole expression in the last `\boxed{}` equals the reference answer, and 0 otherwise. With no box, it uses the last number in the text.

The check compares the two with math-verify, so a boxed $$3/2$$ counts as 1.5, not as 3. The script keeps the first question whose group has both right and wrong answers. We use $$\varepsilon = 0.2$$ and $$\beta = 0.001$$.

One setting needs care. The model's own configuration file asks for a repetition penalty of 1.1, temperature 0.7, top-p 0.8 and top-k 20 whenever it generates. Each of these changes the probabilities the answers are drawn from. Our loss computes log-probabilities from the plain softmax of the logits. If the two differ, $$\pi_{\theta_{old}}$$ in the ratio is not the policy that really sampled the answers, and the run is no longer on-policy. So the script switches all of them off:

```text
model Qwen/Qwen2.5-0.5B-Instruct on mps, float32; G = 8, temperature 1.0, max_new_tokens 512
checkpoint defaults: repetition_penalty 1.1, temperature 0.7, top_p 0.8, top_k 20
sampling with: temperature 1.0, top_p 1.0, top_k 0, repetition_penalty 1.0 (plain softmax, the same distribution the log-probs use)

== 1. Sample a group of 8 answers per question, reward 1 if the final answer is right
question 0: gold 18; rewards [0, 0, 0, 0, 1, 1, 0, 0]; advantages [-0.54, -0.54, -0.54, -0.54, +1.62, +1.62, -0.54, -0.54]
```

The very first question already has both right and wrong answers: it is the duck-egg question. Janet's ducks lay 16 eggs a day. She eats 3 for breakfast, bakes muffins with 4, and sells the rest at 2 dollars an egg. How much does she make each day? The answer is $$(16 - 3 - 4) \times 2 = 18$$, and two of the eight answers get it right.

```text
answer 0: 177 tokens, final answer       26, reward 0, A -0.540
answer 1: 316 tokens, final answer       14, reward 0, A -0.540
answer 2: 336 tokens, final answer       41, reward 0, A -0.540
answer 3: 296 tokens, final answer       34, reward 0, A -0.540
answer 4: 347 tokens, final answer       18, reward 1, A +1.620
answer 5: 280 tokens, final answer       18, reward 1, A +1.620
answer 6: 371 tokens, final answer       14, reward 0, A -0.540
answer 7: 430 tokens, final answer       16, reward 0, A -0.540
```

Answer 0 is a typical mistake. It subtracts the 3 eggs for breakfast, forgets the 4 for muffins, and sells 13 eggs for 26 dollars.

Check the advantages by hand. The mean reward is $$2/8 = 0.25$$. The standard deviation with $$G - 1$$ is $$\sqrt{(2 \times 0.75^2 + 6 \times 0.25^2)/7} = \sqrt{1.5/7} = 0.463$$. So each right answer gets $$0.75 / 0.463 = +1.620$$ and each wrong one $$-0.25 / 0.463 = -0.540$$. The two pushes up balance the six pushes down: $$2 \times 1.620 = 6 \times 0.540$$.

### Log-probabilities, ratios and the loss at the first step

```text
== 3. Per-token log-probs, ratios and KL at the first step
shapes: ids (8, 523), per-token log-probs (8, 430), mask (8, 430)
answer 4, first 10 tokens and log pi_theta:
  To:-0.10  _determine:-0.09  _how:-0.00  _much:-0.01  _Janet:-0.64  _makes:-0.03  _in:-2.26  _dollars:-0.04  _every:-0.09  _day:-0.00
log pi_theta(o_i | q) of each whole answer (sum over its tokens): [-50.2, -96.9, -190.4, -162.6, -80.3, -80.3, -72.7, -137.9]
ratios pi_theta / pi_theta_old on real tokens: min 1.0000, max 1.0000
max |log pi_theta - log pi_ref| 7.58e-05; k3 KL mean over real tokens -4.79e-09 (0 up to float rounding)
loss, ours +0.00000004; TRL +0.00000003; |difference| 7.5e-09
mean of the advantages +3.0e-08: at ratio 1 the loss is -(1/G) sum A_i = 0, but its gradient is not 0
```

The batch is 8 rows of 523 tokens: the prompt, then up to 430 answer tokens. The mask marks the real answer tokens. Each token has a log-probability: "To" at the start of answer 4 has $$-0.10$$, so the model gave it probability $$e^{-0.10} = 0.90$$. Summed over the 347 tokens of answer 4, the log-probability of the whole answer is $$-80.3$$.

Every ratio is exactly 1.0000. The old policy is the model that just sampled, and no step has happened yet, so $$\pi_\theta = \pi_{\theta_{old}}$$. TRL does the same: with one update per batch, it reuses the current log-probabilities, detached, as the old ones.

In this toy run the reference is also a copy of the same weights. A second forward pass on the GPU differs only by $$7.58 \times 10^{-5}$$ of rounding, so the KL term is 0 too. In real training, after many steps, the reference would differ and the KL term would not be 0. As predicted, the loss is 0 (up to rounding) in both methods, and they agree to $$7.5 \times 10^{-9}$$.

### One gradient step

The gradient of that zero loss has a norm of 5.020. We take one AdamW step from the same weights twice: once with the later version's learning rate, $$3 \times 10^{-6}$$, and once with $$10^{-7}$$.

```text
-- learning rate 3e-06
right answers: mean change +28.388; wrong answers: mean change -42.036
answer 4 (right): probability of the whole answer x1.5e+11
-- learning rate 1e-07
answer  reward      A     log pi before   log pi after   change
    0       0   -0.540         -50.23         -51.66   -1.436
    1       0   -0.540         -96.85         -97.39   -0.536
    2       0   -0.540        -190.40        -192.30   -1.894
    3       0   -0.540        -162.58        -163.85   -1.271
    4       1   +1.620         -80.34         -78.57   +1.768
    5       1   +1.620         -80.29         -78.32   +1.976
    6       0   -0.540         -72.73         -73.09   -0.367
    7       0   -0.540        -137.88        -138.45   -0.567
right answers: mean change +1.872; wrong answers: mean change -1.012
answer 4 (right): probability of the whole answer x5.861
```

The direction is right at both sizes: both right answers go up and all six wrong ones go down. The size needs care. Adam's first step moves every weight by about the learning rate, whatever the size of its gradient. Over 494 million weights that adds up. With $$3 \times 10^{-6}$$ on one tiny batch, right answer 4 became $$1.5 \times 10^{11}$$ times more likely, a huge jump.

In real training a step averages over 512 answers, and Adam's statistics settle over many steps. So for a readable picture we keep $$10^{-7}$$. Right answer 4 becomes 5.861 times more likely, and the wrong answers lose about 1 in log-probability on average.

{{FIG:p2_step|The change of each answer's log-probability after one AdamW step with learning rate 1e-7 on our group. The two right answers (A = +1.62) went up by 1.77 and 1.98; the six wrong answers (A = -0.54) went down by between 0.37 and 1.89.}}

### The second step: the ratios and the clip wake up

```text
== 5. The same batch after the step with learning rate 1e-07: ratios move away from 1
per-token ratios: min 0.6318, max 1.2368, outside [0.8, 1.2]: 0.12% of 2553 tokens
answer 4: whole-answer ratio exp(sum of log ratios) = 5.861
k3 KL to the reference, mean over real tokens: 2.10e-04
loss at the second step, ours -0.00387105; TRL -0.00387105; |difference| 0.0e+00
share of tokens in the flat (clipped) region, TRL metric: 0.12%
(the loss went from +0.000000 to -0.003871)
```

If we trained a second time on the same batch, as R1 does with its 16 mini-batches, the ratios would no longer be 1. Per token they now run from 0.632 to 1.237, and 0.12% of the 2,553 real tokens are outside $$[0.8, 1.2]$$. The ratio of the whole of answer 4 is 5.861, far outside any clip. That is the per-token point in practice: each token moved a little, and the whole answer moved a lot. The KL to the reference is no longer 0 but small, $$2.10 \times 10^{-4}$$ per token. The loss went down from 0 to $$-0.003871$$, and our code and TRL print the same value to every digit shown (difference 0.0e+00).

[![Terminal output of r1_part2_run.py: the sampled groups, rewards, advantages, log-probabilities, ratios, the loss and the effect of one step](/img/papers/deepseek-r1/r1_part2_run-run.png)](/img/papers/deepseek-r1/r1_part2_run-run.png)

This is one step on one question. [Part 8](part-8-train-a-reasoner-from-scratch.md) runs the loop for many steps and trains a small model to reason. First, [Part 3](part-3-rewards-and-the-template.md) looks at where the rewards come from.

> [!TAKEAWAYS] Key takeaways
> - The paper's three results (R1-Zero, R1 and the distilled models) all rest on one learning rule, GRPO.
> - PPO needs a critic about as large as the policy and trained next to it. For a 7B policy our count gives 252 GB for PPO and 126 GB for GRPO with rule rewards, half.
> - GRPO replaces the critic with a measurement: sample $$G$$ answers to the same question and use the group's mean reward as the baseline (Equation 3).
> - The advantage $$(r_i - \text{mean}) / \text{std}$$ sums to zero in each group. A rare right answer gets a big push: +1.732 when 1 of 4 is right.
> - When every answer in a group gets the same reward, every advantage is 0. The reward part of the loss then learns nothing from the question; only the KL term can still act. Code adds $$10^{-4}$$ to the standard deviation to avoid $$0/0$$.
> - The clip with $$\varepsilon$$ removes the incentive to move a token's probability far from the old model in one batch, but does not forbid it. The min keeps every mistake fully penalised.
> - R1's Equation 1 writes the ratio for a whole answer. In practice (DeepSeekMath, TRL, our code) each token has its own ratio, averaged over the answer. A 1% change per token is a factor of 19.79 over 300 tokens.
> - Equation 2 is the k3 estimator $$x - \log x - 1$$: never negative, and its average is the true KL (0.05664 in our example). It goes into the loss, not into the reward.
> - Version 1 gives no values for $$\varepsilon$$, $$\beta$$ or $$G$$. The later long version gives $$G = 16$$, $$\beta = 0.001$$ and, for R1, a clip ratio of 10.
> - Our from-scratch GRPO loss matches TRL's `GRPOTrainer` exactly on toy tensors, and to $$7.5 \times 10^{-9}$$ or better on a real model. On Qwen2.5-0.5B, one step made a right answer (2 of 8 were right) 5.861 times more likely.

## The paper and the code

[Paper (arXiv 2501.12948)](https://arxiv.org/abs/2501.12948) · [Official repository](https://github.com/deepseek-ai/DeepSeek-R1) · [Our scripts for this series](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/deepseek-r1)
