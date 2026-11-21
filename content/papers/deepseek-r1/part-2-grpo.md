---
title: "GRPO: Learning from a Group of Answers"
description: "The learning rule behind DeepSeek-R1, term by term: sample a group of answers, score them, and push the model toward the better ones without a critic. We write GRPO from scratch, match it with TRL, and run one real step on a small model."
part: 2
covers: "§2.1, §2.2, §2.2.1, Equations 1 to 3"
date: 2026-11-21
tags: [deepseek, grpo, reinforcement-learning, ppo, reasoning]
---

DeepSeek-R1-Zero learned to reason from one short rule: try a question many times, then make the better tries more likely. This page reads the three equations that turn that rule into training, slowly, with numbers you can check by hand and code that runs.

<figure class="fig"><svg viewBox="0 0 680 200" role="img" aria-label="You are here: GRPO is the reinforcement learning step that turns DeepSeek-V3-Base into R1-Zero, and it runs again in two stages of the R1 pipeline."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">DeepSeek-R1-Zero (§2.2)</text><rect class="box" x="20.0" y="38.0" width="180.0" height="46.0" rx="8"/><text class="t-note" x="110.0" y="58.0" text-anchor="middle">DeepSeek-V3-Base</text><text class="t-muted" x="110.0" y="74.0" text-anchor="middle">pre-trained</text><rect class="box-on" x="235.0" y="38.0" width="180.0" height="46.0" rx="8"/><text class="t-note" x="325.0" y="58.0" text-anchor="middle">RL with GRPO</text><text class="t-muted" x="325.0" y="74.0" text-anchor="middle">this part</text><line class="edge" x1="200.0" y1="61.0" x2="233.0" y2="61.0" marker-end="url(#ah)"/><rect class="box" x="450.0" y="38.0" width="180.0" height="46.0" rx="8"/><text class="t-note" x="540.0" y="58.0" text-anchor="middle">DeepSeek-R1-Zero</text><text class="t-muted" x="540.0" y="74.0" text-anchor="middle">the result</text><line class="edge" x1="415.0" y1="61.0" x2="448.0" y2="61.0" marker-end="url(#ah)"/><text class="t-title" x="20.0" y="126.0" text-anchor="start">DeepSeek-R1 (§2.3): GRPO runs in two of the four stages</text><rect class="box" x="20.0" y="140.0" width="112.0" height="42.0" rx="8"/><text class="t-note" x="76.0" y="165.0" text-anchor="middle">cold-start SFT</text><rect class="box-on" x="152.0" y="140.0" width="112.0" height="42.0" rx="8"/><text class="t-note" x="208.0" y="165.0" text-anchor="middle">RL with GRPO</text><line class="edge" x1="132.0" y1="161.0" x2="150.0" y2="161.0" marker-end="url(#ah)"/><rect class="box" x="284.0" y="140.0" width="112.0" height="42.0" rx="8"/><text class="t-note" x="340.0" y="165.0" text-anchor="middle">sampling + SFT</text><line class="edge" x1="264.0" y1="161.0" x2="282.0" y2="161.0" marker-end="url(#ah)"/><rect class="box-on" x="416.0" y="140.0" width="112.0" height="42.0" rx="8"/><text class="t-note" x="472.0" y="165.0" text-anchor="middle">RL with GRPO</text><line class="edge" x1="396.0" y1="161.0" x2="414.0" y2="161.0" marker-end="url(#ah)"/><rect class="box" x="548.0" y="140.0" width="112.0" height="42.0" rx="8"/><text class="t-note" x="604.0" y="165.0" text-anchor="middle">DeepSeek-R1</text><line class="edge" x1="528.0" y1="161.0" x2="546.0" y2="161.0" marker-end="url(#ah)"/></svg><figcaption>You are here. GRPO is the reinforcement learning step of the paper. It turns DeepSeek-V3-Base into R1-Zero, and it runs again in two of the four stages that make DeepSeek-R1.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 680 346" role="img" aria-label="PPO keeps four models, two of them trained. GRPO as used for R1-Zero keeps a trained policy and a frozen reference; the reward is a program and there is no critic."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="170.0" y="24.0" text-anchor="middle">PPO</text><rect class="box-on" x="20.0" y="40.0" width="300.0" height="46.0" rx="8"/><text class="t-note" x="170.0" y="60.0" text-anchor="middle">policy</text><text class="t-muted" x="170.0" y="76.0" text-anchor="middle">trained</text><rect class="box" x="20.0" y="98.0" width="300.0" height="46.0" rx="8"/><text class="t-note" x="170.0" y="118.0" text-anchor="middle">reference</text><text class="t-muted" x="170.0" y="134.0" text-anchor="middle">frozen</text><rect class="box" x="20.0" y="156.0" width="300.0" height="46.0" rx="8"/><text class="t-note" x="170.0" y="176.0" text-anchor="middle">reward model</text><text class="t-muted" x="170.0" y="192.0" text-anchor="middle">frozen</text><rect class="box-on" x="20.0" y="214.0" width="300.0" height="46.0" rx="8"/><text class="t-note" x="170.0" y="234.0" text-anchor="middle">value model (critic)</text><text class="t-muted" x="170.0" y="250.0" text-anchor="middle">trained, same size</text><text class="t-strong t-note" x="170.0" y="288.0" text-anchor="middle">7B policy: 252 GB of weights and optimizer state</text><text class="t-title" x="510.0" y="24.0" text-anchor="middle">GRPO in R1-Zero</text><rect class="box-on" x="360.0" y="40.0" width="300.0" height="46.0" rx="8"/><text class="t-note" x="510.0" y="60.0" text-anchor="middle">policy</text><text class="t-muted" x="510.0" y="76.0" text-anchor="middle">trained</text><rect class="box" x="360.0" y="98.0" width="300.0" height="46.0" rx="8"/><text class="t-note" x="510.0" y="118.0" text-anchor="middle">reference</text><text class="t-muted" x="510.0" y="134.0" text-anchor="middle">frozen</text><rect class="box-ghost" x="360.0" y="156.0" width="300.0" height="46.0" rx="8"/><text class="t-note" x="510.0" y="176.0" text-anchor="middle">reward: rules</text><text class="t-muted" x="510.0" y="192.0" text-anchor="middle">a program, no weights</text><rect class="box-ghost" x="360.0" y="214.0" width="300.0" height="46.0" rx="8"/><text class="t-note" x="510.0" y="234.0" text-anchor="middle">no critic</text><text class="t-muted" x="510.0" y="250.0" text-anchor="middle">the group is the baseline</text><text class="t-strong t-note" x="510.0" y="288.0" text-anchor="middle">7B policy: 126 GB of weights and optimizer state</text><line class="edge-dim" x1="340.0" y1="14.0" x2="340.0" y2="296.0"/><text class="t-muted" x="20.0" y="316.0" text-anchor="start">Filled boxes are trained (weights, gradients and Adam state: 16 bytes per parameter).</text><text class="t-muted" x="20.0" y="334.0" text-anchor="start">Plain boxes only run forward passes (2 bytes per parameter). Dashed boxes need no memory.</text></svg><figcaption>PPO and GRPO side by side. PPO keeps four models and trains two of them, the policy and the value model. GRPO as used for R1-Zero trains only the policy, keeps a frozen reference, scores answers with a program, and has no critic. The memory figures come from our script.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 680 400" role="img" aria-label="The GRPO loop on our real group: one question, 8 sampled answers, a 0 or 1 reward for each, the group-normalised advantages, and the update."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><rect class="box" x="20.0" y="150.0" width="110.0" height="50.0" rx="8"/><text class="t-note" x="75.0" y="172.0" text-anchor="middle">question q</text><text class="t-muted" x="75.0" y="188.0" text-anchor="middle">GSM8K</text><text class="t-title" x="250.0" y="24.0" text-anchor="middle">8 answers from pi_old</text><text class="t-title" x="410.0" y="24.0" text-anchor="middle">reward r_i</text><text class="t-title" x="560.0" y="24.0" text-anchor="middle">advantage A_i</text><rect class="box" x="190.0" y="38.0" width="120.0" height="28.0" rx="8"/><text class="t-tick" x="250.0" y="56.0" text-anchor="middle">answer 0</text><line class="edge" x1="130.0" y1="175.0" x2="188.0" y2="52.0"/><rect class="box" x="380.0" y="38.0" width="60.0" height="28.0" rx="8"/><text class="t-tick" x="410.0" y="56.0" text-anchor="middle">0</text><line class="edge" x1="312.0" y1="52.0" x2="378.0" y2="52.0" marker-end="url(#ah)"/><rect class="box" x="520.0" y="38.0" width="80.0" height="28.0" rx="8"/><text class="t-tick" x="560.0" y="56.0" text-anchor="middle">-0.54</text><line class="edge" x1="442.0" y1="52.0" x2="518.0" y2="52.0" marker-end="url(#ah)"/><rect class="box" x="190.0" y="74.0" width="120.0" height="28.0" rx="8"/><text class="t-tick" x="250.0" y="92.0" text-anchor="middle">answer 1</text><line class="edge" x1="130.0" y1="175.0" x2="188.0" y2="88.0"/><rect class="box" x="380.0" y="74.0" width="60.0" height="28.0" rx="8"/><text class="t-tick" x="410.0" y="92.0" text-anchor="middle">0</text><line class="edge" x1="312.0" y1="88.0" x2="378.0" y2="88.0" marker-end="url(#ah)"/><rect class="box" x="520.0" y="74.0" width="80.0" height="28.0" rx="8"/><text class="t-tick" x="560.0" y="92.0" text-anchor="middle">-0.54</text><line class="edge" x1="442.0" y1="88.0" x2="518.0" y2="88.0" marker-end="url(#ah)"/><rect class="box" x="190.0" y="110.0" width="120.0" height="28.0" rx="8"/><text class="t-tick" x="250.0" y="128.0" text-anchor="middle">answer 2</text><line class="edge" x1="130.0" y1="175.0" x2="188.0" y2="124.0"/><rect class="box" x="380.0" y="110.0" width="60.0" height="28.0" rx="8"/><text class="t-tick" x="410.0" y="128.0" text-anchor="middle">0</text><line class="edge" x1="312.0" y1="124.0" x2="378.0" y2="124.0" marker-end="url(#ah)"/><rect class="box" x="520.0" y="110.0" width="80.0" height="28.0" rx="8"/><text class="t-tick" x="560.0" y="128.0" text-anchor="middle">-0.54</text><line class="edge" x1="442.0" y1="124.0" x2="518.0" y2="124.0" marker-end="url(#ah)"/><rect class="box" x="190.0" y="146.0" width="120.0" height="28.0" rx="8"/><text class="t-tick" x="250.0" y="164.0" text-anchor="middle">answer 3</text><line class="edge" x1="130.0" y1="175.0" x2="188.0" y2="160.0"/><rect class="box" x="380.0" y="146.0" width="60.0" height="28.0" rx="8"/><text class="t-tick" x="410.0" y="164.0" text-anchor="middle">0</text><line class="edge" x1="312.0" y1="160.0" x2="378.0" y2="160.0" marker-end="url(#ah)"/><rect class="box" x="520.0" y="146.0" width="80.0" height="28.0" rx="8"/><text class="t-tick" x="560.0" y="164.0" text-anchor="middle">-0.54</text><line class="edge" x1="442.0" y1="160.0" x2="518.0" y2="160.0" marker-end="url(#ah)"/><rect class="box" x="190.0" y="182.0" width="120.0" height="28.0" rx="8"/><text class="t-tick" x="250.0" y="200.0" text-anchor="middle">answer 4</text><line class="edge" x1="130.0" y1="175.0" x2="188.0" y2="196.0"/><rect class="box-1" x="380.0" y="182.0" width="60.0" height="28.0" rx="8"/><text class="t-tick" x="410.0" y="200.0" text-anchor="middle">1</text><line class="edge" x1="312.0" y1="196.0" x2="378.0" y2="196.0" marker-end="url(#ah)"/><rect class="box-on" x="520.0" y="182.0" width="80.0" height="28.0" rx="8"/><text class="t-tick" x="560.0" y="200.0" text-anchor="middle">+1.62</text><line class="edge" x1="442.0" y1="196.0" x2="518.0" y2="196.0" marker-end="url(#ah)"/><rect class="box" x="190.0" y="218.0" width="120.0" height="28.0" rx="8"/><text class="t-tick" x="250.0" y="236.0" text-anchor="middle">answer 5</text><line class="edge" x1="130.0" y1="175.0" x2="188.0" y2="232.0"/><rect class="box-1" x="380.0" y="218.0" width="60.0" height="28.0" rx="8"/><text class="t-tick" x="410.0" y="236.0" text-anchor="middle">1</text><line class="edge" x1="312.0" y1="232.0" x2="378.0" y2="232.0" marker-end="url(#ah)"/><rect class="box-on" x="520.0" y="218.0" width="80.0" height="28.0" rx="8"/><text class="t-tick" x="560.0" y="236.0" text-anchor="middle">+1.62</text><line class="edge" x1="442.0" y1="232.0" x2="518.0" y2="232.0" marker-end="url(#ah)"/><rect class="box" x="190.0" y="254.0" width="120.0" height="28.0" rx="8"/><text class="t-tick" x="250.0" y="272.0" text-anchor="middle">answer 6</text><line class="edge" x1="130.0" y1="175.0" x2="188.0" y2="268.0"/><rect class="box" x="380.0" y="254.0" width="60.0" height="28.0" rx="8"/><text class="t-tick" x="410.0" y="272.0" text-anchor="middle">0</text><line class="edge" x1="312.0" y1="268.0" x2="378.0" y2="268.0" marker-end="url(#ah)"/><rect class="box" x="520.0" y="254.0" width="80.0" height="28.0" rx="8"/><text class="t-tick" x="560.0" y="272.0" text-anchor="middle">-0.54</text><line class="edge" x1="442.0" y1="268.0" x2="518.0" y2="268.0" marker-end="url(#ah)"/><rect class="box" x="190.0" y="290.0" width="120.0" height="28.0" rx="8"/><text class="t-tick" x="250.0" y="308.0" text-anchor="middle">answer 7</text><line class="edge" x1="130.0" y1="175.0" x2="188.0" y2="304.0"/><rect class="box" x="380.0" y="290.0" width="60.0" height="28.0" rx="8"/><text class="t-tick" x="410.0" y="308.0" text-anchor="middle">0</text><line class="edge" x1="312.0" y1="304.0" x2="378.0" y2="304.0" marker-end="url(#ah)"/><rect class="box" x="520.0" y="290.0" width="80.0" height="28.0" rx="8"/><text class="t-tick" x="560.0" y="308.0" text-anchor="middle">-0.54</text><line class="edge" x1="442.0" y1="304.0" x2="518.0" y2="304.0" marker-end="url(#ah)"/><text class="t-muted" x="560.0" y="340.0" text-anchor="middle">Equation 3</text><text class="t-muted" x="410.0" y="340.0" text-anchor="middle">is it right?</text><path class="edge" d="M600,310 C660,360 120,380 75,202" marker-end="url(#ah)"/><text class="t-note" x="340.0" y="386.0" text-anchor="middle">update the policy with Equation 1: raise the answers with A &gt; 0, lower the ones with A &lt; 0</text></svg><figcaption>The GRPO loop on our real group. One GSM8K question, eight answers sampled from the model, a reward of 1 for a right final answer and 0 for a wrong one, the group-normalised advantages of Equation 3, and an update that raises the answers with a positive advantage.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 680 232" role="img" aria-label="One token of Equation 1 as a function of the ratio, with epsilon 0.2. For a good answer the term stops growing above 1.2; for a bad answer it stops falling below 0.8. The dashed line is the term without the clip."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="160.0" y="18.0" text-anchor="middle">A = +1 (a good answer)</text><rect class="plane" x="150.0" y="30" width="50.0" height="160"/><line class="axis" x1="50.0" y1="180.6" x2="300.0" y2="180.6"/><text class="t-tick" x="50.0" y="208.0" text-anchor="middle">0</text><text class="t-tick" x="150.0" y="208.0" text-anchor="middle">0.8</text><text class="t-tick" x="175.0" y="208.0" text-anchor="middle">1</text><text class="t-tick" x="200.0" y="208.0" text-anchor="middle">1.2</text><text class="t-tick" x="300.0" y="208.0" text-anchor="middle">2</text><polyline class="base-line" points="50.0,180.6 51.2,179.6 52.5,178.7 53.8,177.8 55.0,176.8 56.2,175.9 57.5,174.9 58.8,174.0 60.0,173.1 61.2,172.1 62.5,171.2 63.8,170.2 65.0,169.3 66.2,168.4 67.5,167.4 68.8,166.5 70.0,165.5 71.2,164.6 72.5,163.6 73.8,162.7 75.0,161.8 76.2,160.8 77.5,159.9 78.8,158.9 80.0,158.0 81.2,157.1 82.5,156.1 83.8,155.2 85.0,154.2 86.2,153.3 87.5,152.4 88.8,151.4 90.0,150.5 91.2,149.5 92.5,148.6 93.8,147.6 95.0,146.7 96.2,145.8 97.5,144.8 98.8,143.9 100.0,142.9 101.2,142.0 102.5,141.1 103.8,140.1 105.0,139.2 106.2,138.2 107.5,137.3 108.8,136.4 110.0,135.4 111.2,134.5 112.5,133.5 113.8,132.6 115.0,131.6 116.2,130.7 117.5,129.8 118.8,128.8 120.0,127.9 121.2,126.9 122.5,126.0 123.8,125.1 125.0,124.1 126.2,123.2 127.5,122.2 128.8,121.3 130.0,120.4 131.2,119.4 132.5,118.5 133.8,117.5 135.0,116.6 136.2,115.6 137.5,114.7 138.8,113.8 140.0,112.8 141.2,111.9 142.5,110.9 143.8,110.0 145.0,109.1 146.2,108.1 147.5,107.2 148.8,106.2 150.0,105.3 151.2,104.4 152.5,103.4 153.8,102.5 155.0,101.5 156.2,100.6 157.5,99.6 158.8,98.7 160.0,97.8 161.2,96.8 162.5,95.9 163.8,94.9 165.0,94.0 166.2,93.1 167.5,92.1 168.8,91.2 170.0,90.2 171.2,89.3 172.5,88.4 173.8,87.4 175.0,86.5 176.2,85.5 177.5,84.6 178.8,83.6 180.0,82.7 181.2,81.8 182.5,80.8 183.8,79.9 185.0,78.9 186.2,78.0 187.5,77.1 188.8,76.1 190.0,75.2 191.2,74.2 192.5,73.3 193.8,72.4 195.0,71.4 196.2,70.5 197.5,69.5 198.8,68.6 200.0,67.6 201.2,66.7 202.5,65.8 203.8,64.8 205.0,63.9 206.2,62.9 207.5,62.0 208.8,61.1 210.0,60.1 211.2,59.2 212.5,58.2 213.8,57.3 215.0,56.4 216.2,55.4 217.5,54.5 218.8,53.5 220.0,52.6 221.2,51.6 222.5,50.7 223.8,49.8 225.0,48.8 226.2,47.9 227.5,46.9 228.8,46.0 230.0,45.1 231.2,44.1 232.5,43.2 233.8,42.2 235.0,41.3 236.2,40.4 237.5,39.4 238.8,38.5 240.0,37.5 241.2,36.6 242.5,35.6 243.8,34.7 245.0,33.8 246.2,32.8 247.5,31.9 248.8,30.9 250.0,30.0"/><polyline class="l1" style="stroke-width:3" points="50.0,180.6 51.2,179.6 52.5,178.7 53.8,177.8 55.0,176.8 56.2,175.9 57.5,174.9 58.8,174.0 60.0,173.1 61.2,172.1 62.5,171.2 63.8,170.2 65.0,169.3 66.2,168.4 67.5,167.4 68.8,166.5 70.0,165.5 71.2,164.6 72.5,163.6 73.8,162.7 75.0,161.8 76.2,160.8 77.5,159.9 78.8,158.9 80.0,158.0 81.2,157.1 82.5,156.1 83.8,155.2 85.0,154.2 86.2,153.3 87.5,152.4 88.8,151.4 90.0,150.5 91.2,149.5 92.5,148.6 93.8,147.6 95.0,146.7 96.2,145.8 97.5,144.8 98.8,143.9 100.0,142.9 101.2,142.0 102.5,141.1 103.8,140.1 105.0,139.2 106.2,138.2 107.5,137.3 108.8,136.4 110.0,135.4 111.2,134.5 112.5,133.5 113.8,132.6 115.0,131.6 116.2,130.7 117.5,129.8 118.8,128.8 120.0,127.9 121.2,126.9 122.5,126.0 123.8,125.1 125.0,124.1 126.2,123.2 127.5,122.2 128.8,121.3 130.0,120.4 131.2,119.4 132.5,118.5 133.8,117.5 135.0,116.6 136.2,115.6 137.5,114.7 138.8,113.8 140.0,112.8 141.2,111.9 142.5,110.9 143.8,110.0 145.0,109.1 146.2,108.1 147.5,107.2 148.8,106.2 150.0,105.3 151.2,104.4 152.5,103.4 153.8,102.5 155.0,101.5 156.2,100.6 157.5,99.6 158.8,98.7 160.0,97.8 161.2,96.8 162.5,95.9 163.8,94.9 165.0,94.0 166.2,93.1 167.5,92.1 168.8,91.2 170.0,90.2 171.2,89.3 172.5,88.4 173.8,87.4 175.0,86.5 176.2,85.5 177.5,84.6 178.8,83.6 180.0,82.7 181.2,81.8 182.5,80.8 183.8,79.9 185.0,78.9 186.2,78.0 187.5,77.1 188.8,76.1 190.0,75.2 191.2,74.2 192.5,73.3 193.8,72.4 195.0,71.4 196.2,70.5 197.5,69.5 198.8,68.6 200.0,67.6 201.2,67.6 202.5,67.6 203.8,67.6 205.0,67.6 206.2,67.6 207.5,67.6 208.8,67.6 210.0,67.6 211.2,67.6 212.5,67.6 213.8,67.6 215.0,67.6 216.2,67.6 217.5,67.6 218.8,67.6 220.0,67.6 221.2,67.6 222.5,67.6 223.8,67.6 225.0,67.6 226.2,67.6 227.5,67.6 228.8,67.6 230.0,67.6 231.2,67.6 232.5,67.6 233.8,67.6 235.0,67.6 236.2,67.6 237.5,67.6 238.8,67.6 240.0,67.6 241.2,67.6 242.5,67.6 243.8,67.6 245.0,67.6 246.2,67.6 247.5,67.6 248.8,67.6 250.0,67.6 251.2,67.6 252.5,67.6 253.8,67.6 255.0,67.6 256.2,67.6 257.5,67.6 258.8,67.6 260.0,67.6 261.2,67.6 262.5,67.6 263.8,67.6 265.0,67.6 266.2,67.6 267.5,67.6 268.8,67.6 270.0,67.6 271.2,67.6 272.5,67.6 273.8,67.6 275.0,67.6 276.2,67.6 277.5,67.6 278.8,67.6 280.0,67.6 281.2,67.6 282.5,67.6 283.8,67.6 285.0,67.6 286.2,67.6 287.5,67.6 288.8,67.6 290.0,67.6 291.2,67.6 292.5,67.6 293.8,67.6 295.0,67.6 296.2,67.6 297.5,67.6 298.8,67.6 300.0,67.6"/><circle class="s2" cx="175.0" cy="86.5" r="5"/><text class="t-note" x="252.5" y="87.6" text-anchor="middle">flat: this term stops pushing</text><text class="t-muted" x="102.5" y="121.5" text-anchor="middle">slope A</text><text class="t-tick" x="160.0" y="224.0" text-anchor="middle">ratio rho = pi_theta / pi_old</text><text class="t-title" x="510.0" y="18.0" text-anchor="middle">A = -1 (a bad answer)</text><rect class="plane" x="500.0" y="30" width="50.0" height="160"/><line class="axis" x1="400.0" y1="37.3" x2="650.0" y2="37.3"/><text class="t-tick" x="400.0" y="208.0" text-anchor="middle">0</text><text class="t-tick" x="500.0" y="208.0" text-anchor="middle">0.8</text><text class="t-tick" x="525.0" y="208.0" text-anchor="middle">1</text><text class="t-tick" x="550.0" y="208.0" text-anchor="middle">1.2</text><text class="t-tick" x="650.0" y="208.0" text-anchor="middle">2</text><polyline class="base-line" points="400.0,37.3 401.2,38.0 402.5,38.7 403.8,39.5 405.0,40.2 406.2,40.9 407.5,41.6 408.8,42.4 410.0,43.1 411.2,43.8 412.5,44.5 413.8,45.3 415.0,46.0 416.2,46.7 417.5,47.5 418.8,48.2 420.0,48.9 421.2,49.6 422.5,50.4 423.8,51.1 425.0,51.8 426.2,52.5 427.5,53.3 428.8,54.0 430.0,54.7 431.2,55.5 432.5,56.2 433.8,56.9 435.0,57.6 436.2,58.4 437.5,59.1 438.8,59.8 440.0,60.5 441.2,61.3 442.5,62.0 443.8,62.7 445.0,63.5 446.2,64.2 447.5,64.9 448.8,65.6 450.0,66.4 451.2,67.1 452.5,67.8 453.8,68.5 455.0,69.3 456.2,70.0 457.5,70.7 458.8,71.5 460.0,72.2 461.2,72.9 462.5,73.6 463.8,74.4 465.0,75.1 466.2,75.8 467.5,76.5 468.8,77.3 470.0,78.0 471.2,78.7 472.5,79.5 473.8,80.2 475.0,80.9 476.2,81.6 477.5,82.4 478.8,83.1 480.0,83.8 481.2,84.5 482.5,85.3 483.8,86.0 485.0,86.7 486.2,87.5 487.5,88.2 488.8,88.9 490.0,89.6 491.2,90.4 492.5,91.1 493.8,91.8 495.0,92.5 496.2,93.3 497.5,94.0 498.8,94.7 500.0,95.5 501.2,96.2 502.5,96.9 503.8,97.6 505.0,98.4 506.2,99.1 507.5,99.8 508.8,100.5 510.0,101.3 511.2,102.0 512.5,102.7 513.8,103.5 515.0,104.2 516.2,104.9 517.5,105.6 518.8,106.4 520.0,107.1 521.2,107.8 522.5,108.5 523.8,109.3 525.0,110.0 526.2,110.7 527.5,111.5 528.8,112.2 530.0,112.9 531.2,113.6 532.5,114.4 533.8,115.1 535.0,115.8 536.2,116.5 537.5,117.3 538.8,118.0 540.0,118.7 541.2,119.5 542.5,120.2 543.8,120.9 545.0,121.6 546.2,122.4 547.5,123.1 548.8,123.8 550.0,124.5 551.2,125.3 552.5,126.0 553.8,126.7 555.0,127.5 556.2,128.2 557.5,128.9 558.8,129.6 560.0,130.4 561.2,131.1 562.5,131.8 563.8,132.5 565.0,133.3 566.2,134.0 567.5,134.7 568.8,135.5 570.0,136.2 571.2,136.9 572.5,137.6 573.8,138.4 575.0,139.1 576.2,139.8 577.5,140.5 578.8,141.3 580.0,142.0 581.2,142.7 582.5,143.5 583.8,144.2 585.0,144.9 586.2,145.6 587.5,146.4 588.8,147.1 590.0,147.8 591.2,148.5 592.5,149.3 593.8,150.0 595.0,150.7 596.2,151.5 597.5,152.2 598.8,152.9 600.0,153.6 601.2,154.4 602.5,155.1 603.8,155.8 605.0,156.5 606.2,157.3 607.5,158.0 608.8,158.7 610.0,159.5 611.2,160.2 612.5,160.9 613.8,161.6 615.0,162.4 616.2,163.1 617.5,163.8 618.8,164.5 620.0,165.3 621.2,166.0 622.5,166.7 623.8,167.5 625.0,168.2 626.2,168.9 627.5,169.6 628.8,170.4 630.0,171.1 631.2,171.8 632.5,172.5 633.8,173.3 635.0,174.0 636.2,174.7 637.5,175.5 638.8,176.2 640.0,176.9 641.2,177.6 642.5,178.4 643.8,179.1 645.0,179.8 646.2,180.5 647.5,181.3 648.8,182.0 650.0,182.7"/><polyline class="l1" style="stroke-width:3" points="400.0,95.5 401.2,95.5 402.5,95.5 403.8,95.5 405.0,95.5 406.2,95.5 407.5,95.5 408.8,95.5 410.0,95.5 411.2,95.5 412.5,95.5 413.8,95.5 415.0,95.5 416.2,95.5 417.5,95.5 418.8,95.5 420.0,95.5 421.2,95.5 422.5,95.5 423.8,95.5 425.0,95.5 426.2,95.5 427.5,95.5 428.8,95.5 430.0,95.5 431.2,95.5 432.5,95.5 433.8,95.5 435.0,95.5 436.2,95.5 437.5,95.5 438.8,95.5 440.0,95.5 441.2,95.5 442.5,95.5 443.8,95.5 445.0,95.5 446.2,95.5 447.5,95.5 448.8,95.5 450.0,95.5 451.2,95.5 452.5,95.5 453.8,95.5 455.0,95.5 456.2,95.5 457.5,95.5 458.8,95.5 460.0,95.5 461.2,95.5 462.5,95.5 463.8,95.5 465.0,95.5 466.2,95.5 467.5,95.5 468.8,95.5 470.0,95.5 471.2,95.5 472.5,95.5 473.8,95.5 475.0,95.5 476.2,95.5 477.5,95.5 478.8,95.5 480.0,95.5 481.2,95.5 482.5,95.5 483.8,95.5 485.0,95.5 486.2,95.5 487.5,95.5 488.8,95.5 490.0,95.5 491.2,95.5 492.5,95.5 493.8,95.5 495.0,95.5 496.2,95.5 497.5,95.5 498.8,95.5 500.0,95.5 501.2,96.2 502.5,96.9 503.8,97.6 505.0,98.4 506.2,99.1 507.5,99.8 508.8,100.5 510.0,101.3 511.2,102.0 512.5,102.7 513.8,103.5 515.0,104.2 516.2,104.9 517.5,105.6 518.8,106.4 520.0,107.1 521.2,107.8 522.5,108.5 523.8,109.3 525.0,110.0 526.2,110.7 527.5,111.5 528.8,112.2 530.0,112.9 531.2,113.6 532.5,114.4 533.8,115.1 535.0,115.8 536.2,116.5 537.5,117.3 538.8,118.0 540.0,118.7 541.2,119.5 542.5,120.2 543.8,120.9 545.0,121.6 546.2,122.4 547.5,123.1 548.8,123.8 550.0,124.5 551.2,125.3 552.5,126.0 553.8,126.7 555.0,127.5 556.2,128.2 557.5,128.9 558.8,129.6 560.0,130.4 561.2,131.1 562.5,131.8 563.8,132.5 565.0,133.3 566.2,134.0 567.5,134.7 568.8,135.5 570.0,136.2 571.2,136.9 572.5,137.6 573.8,138.4 575.0,139.1 576.2,139.8 577.5,140.5 578.8,141.3 580.0,142.0 581.2,142.7 582.5,143.5 583.8,144.2 585.0,144.9 586.2,145.6 587.5,146.4 588.8,147.1 590.0,147.8 591.2,148.5 592.5,149.3 593.8,150.0 595.0,150.7 596.2,151.5 597.5,152.2 598.8,152.9 600.0,153.6 601.2,154.4 602.5,155.1 603.8,155.8 605.0,156.5 606.2,157.3 607.5,158.0 608.8,158.7 610.0,159.5 611.2,160.2 612.5,160.9 613.8,161.6 615.0,162.4 616.2,163.1 617.5,163.8 618.8,164.5 620.0,165.3 621.2,166.0 622.5,166.7 623.8,167.5 625.0,168.2 626.2,168.9 627.5,169.6 628.8,170.4 630.0,171.1 631.2,171.8 632.5,172.5 633.8,173.3 635.0,174.0 636.2,174.7 637.5,175.5 638.8,176.2 640.0,176.9 641.2,177.6 642.5,178.4 643.8,179.1 645.0,179.8 646.2,180.5 647.5,181.3 648.8,182.0 650.0,182.7"/><circle class="s2" cx="525.0" cy="110.0" r="5"/><text class="t-note" x="447.5" y="83.5" text-anchor="middle">flat: this term stops pushing</text><text class="t-muted" x="581.2" y="159.6" text-anchor="middle">slope A: still penalised</text><text class="t-tick" x="510.0" y="224.0" text-anchor="middle">ratio rho = pi_theta / pi_old</text></svg><figcaption>One token of Equation 1 as a function of the ratio, with epsilon 0.2. Left, a good answer (A = +1): the term rises with the ratio and goes flat above 1.2. Right, a bad answer (A = -1): the term falls with the ratio and goes flat below 0.8. The dot is the start, ratio 1. The shaded band is the interval from 0.8 to 1.2.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 640 250" role="img" aria-label="Two per-token estimates of the KL as functions of the probability ratio x. The plain log ratio k1 is negative for x above 1; Equation 2, k3, never goes below zero and touches zero only at x = 1."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="60" y1="199.8" x2="490.0" y2="199.8"/><text class="t-tick" x="52.0" y="203.8" text-anchor="end">-1</text><line class="grid" x1="60" y1="148.7" x2="490.0" y2="148.7"/><text class="t-tick" x="52.0" y="152.7" text-anchor="end">0</text><line class="grid" x1="60" y1="97.6" x2="490.0" y2="97.6"/><text class="t-tick" x="52.0" y="101.6" text-anchor="end">1</text><line class="grid" x1="60" y1="46.4" x2="490.0" y2="46.4"/><text class="t-tick" x="52.0" y="50.4" text-anchor="end">2</text><text class="t-tick" x="119.3" y="230.0" text-anchor="middle">0.5</text><text class="t-tick" x="193.4" y="230.0" text-anchor="middle">1</text><text class="t-tick" x="341.7" y="230.0" text-anchor="middle">2</text><text class="t-tick" x="490.0" y="230.0" text-anchor="middle">3</text><polyline class="l2" points="60.0,31.0 62.1,37.9 64.3,44.0 66.5,49.4 68.6,54.4 70.8,58.8 72.9,63.0 75.0,66.8 77.2,70.3 79.3,73.7 81.5,76.8 83.7,79.7 85.8,82.5 87.9,85.1 90.1,87.6 92.2,90.0 94.4,92.3 96.6,94.5 98.7,96.6 100.9,98.6 103.0,100.5 105.2,102.4 107.3,104.2 109.5,105.9 111.6,107.6 113.8,109.3 115.9,110.8 118.0,112.4 120.2,113.8 122.3,115.3 124.5,116.7 126.7,118.1 128.8,119.4 130.9,120.7 133.1,122.0 135.2,123.2 137.4,124.4 139.6,125.6 141.7,126.7 143.9,127.9 146.0,129.0 148.1,130.0 150.3,131.1 152.5,132.1 154.6,133.1 156.8,134.1 158.9,135.1 161.0,136.1 163.2,137.0 165.4,137.9 167.5,138.8 169.7,139.7 171.8,140.6 173.9,141.5 176.1,142.3 178.2,143.1 180.4,144.0 182.5,144.8 184.7,145.6 186.8,146.3 189.0,147.1 191.2,147.9 193.3,148.6 195.5,149.4 197.6,150.1 199.8,150.8 201.9,151.5 204.0,152.2 206.2,152.9 208.3,153.6 210.5,154.2 212.7,154.9 214.8,155.5 216.9,156.2 219.1,156.8 221.2,157.5 223.4,158.1 225.5,158.7 227.7,159.3 229.8,159.9 232.0,160.5 234.2,161.1 236.3,161.6 238.5,162.2 240.6,162.8 242.8,163.3 244.9,163.9 247.0,164.4 249.2,165.0 251.3,165.5 253.5,166.0 255.6,166.6 257.8,167.1 260.0,167.6 262.1,168.1 264.2,168.6 266.4,169.1 268.6,169.6 270.7,170.1 272.8,170.6 275.0,171.1 277.1,171.5 279.3,172.0 281.5,172.5 283.6,172.9 285.8,173.4 287.9,173.9 290.1,174.3 292.2,174.8 294.3,175.2 296.5,175.6 298.6,176.1 300.8,176.5 302.9,176.9 305.1,177.4 307.2,177.8 309.4,178.2 311.6,178.6 313.7,179.0 315.9,179.4 318.0,179.8 320.1,180.2 322.3,180.6 324.4,181.0 326.6,181.4 328.8,181.8 330.9,182.2 333.1,182.6 335.2,183.0 337.3,183.3 339.5,183.7 341.7,184.1 343.8,184.5 345.9,184.8 348.1,185.2 350.2,185.5 352.4,185.9 354.5,186.3 356.7,186.6 358.8,187.0 361.0,187.3 363.1,187.7 365.3,188.0 367.5,188.3 369.6,188.7 371.8,189.0 373.9,189.4 376.1,189.7 378.2,190.0 380.4,190.4 382.5,190.7 384.6,191.0 386.8,191.3 389.0,191.6 391.1,192.0 393.2,192.3 395.4,192.6 397.6,192.9 399.7,193.2 401.8,193.5 404.0,193.8 406.1,194.1 408.3,194.4 410.5,194.7 412.6,195.0 414.8,195.3 416.9,195.6 419.1,195.9 421.2,196.2 423.3,196.5 425.5,196.8 427.6,197.1 429.8,197.4 431.9,197.7 434.1,198.0 436.3,198.2 438.4,198.5 440.5,198.8 442.7,199.1 444.9,199.3 447.0,199.6 449.1,199.9 451.3,200.2 453.4,200.4 455.6,200.7 457.8,201.0 459.9,201.2 462.0,201.5 464.2,201.8 466.4,202.0 468.5,202.3 470.6,202.5 472.8,202.8 474.9,203.1 477.1,203.3 479.3,203.6 481.4,203.8 483.5,204.1 485.7,204.3 487.9,204.6 490.0,204.8"/><polyline class="l1" style="stroke-width:3" points="60.0,77.0 62.1,83.2 64.3,88.5 66.5,93.2 68.6,97.4 70.8,101.1 72.9,104.5 75.0,107.6 77.2,110.4 79.3,113.0 81.5,115.4 83.7,117.6 85.8,119.6 87.9,121.5 90.1,123.3 92.2,124.9 94.4,126.5 96.6,127.9 98.7,129.3 100.9,130.5 103.0,131.7 105.2,132.8 107.3,133.9 109.5,134.9 111.6,135.8 113.8,136.7 115.9,137.6 118.0,138.4 120.2,139.1 122.3,139.8 124.5,140.5 126.7,141.1 128.8,141.7 130.9,142.2 133.1,142.8 135.2,143.3 137.4,143.7 139.6,144.2 141.7,144.6 143.9,145.0 146.0,145.3 148.1,145.6 150.3,146.0 152.5,146.3 154.6,146.5 156.8,146.8 158.9,147.0 161.0,147.2 163.2,147.4 165.4,147.6 167.5,147.8 169.7,147.9 171.8,148.1 173.9,148.2 176.1,148.3 178.2,148.4 180.4,148.5 182.5,148.5 184.7,148.6 186.8,148.6 189.0,148.6 191.2,148.7 193.3,148.7 195.5,148.7 197.6,148.6 199.8,148.6 201.9,148.6 204.0,148.5 206.2,148.5 208.3,148.4 210.5,148.4 212.7,148.3 214.8,148.2 216.9,148.1 219.1,148.0 221.2,147.9 223.4,147.7 225.5,147.6 227.7,147.5 229.8,147.3 232.0,147.2 234.2,147.0 236.3,146.9 238.5,146.7 240.6,146.5 242.8,146.3 244.9,146.2 247.0,146.0 249.2,145.8 251.3,145.6 253.5,145.3 255.6,145.1 257.8,144.9 260.0,144.7 262.1,144.4 264.2,144.2 266.4,144.0 268.6,143.7 270.7,143.5 272.8,143.2 275.0,143.0 277.1,142.7 279.3,142.4 281.5,142.1 283.6,141.9 285.8,141.6 287.9,141.3 290.1,141.0 292.2,140.7 294.3,140.4 296.5,140.1 298.6,139.8 300.8,139.5 302.9,139.2 305.1,138.9 307.2,138.6 309.4,138.2 311.6,137.9 313.7,137.6 315.9,137.2 318.0,136.9 320.1,136.6 322.3,136.2 324.4,135.9 326.6,135.5 328.8,135.2 330.9,134.8 333.1,134.5 335.2,134.1 337.3,133.7 339.5,133.4 341.7,133.0 343.8,132.6 345.9,132.2 348.1,131.9 350.2,131.5 352.4,131.1 354.5,130.7 356.7,130.3 358.8,129.9 361.0,129.6 363.1,129.2 365.3,128.8 367.5,128.4 369.6,128.0 371.8,127.6 373.9,127.2 376.1,126.7 378.2,126.3 380.4,125.9 382.5,125.5 384.6,125.1 386.8,124.7 389.0,124.3 391.1,123.8 393.2,123.4 395.4,123.0 397.6,122.6 399.7,122.1 401.8,121.7 404.0,121.3 406.1,120.8 408.3,120.4 410.5,119.9 412.6,119.5 414.8,119.1 416.9,118.6 419.1,118.2 421.2,117.7 423.3,117.3 425.5,116.8 427.6,116.4 429.8,115.9 431.9,115.5 434.1,115.0 436.3,114.5 438.4,114.1 440.5,113.6 442.7,113.2 444.9,112.7 447.0,112.2 449.1,111.8 451.3,111.3 453.4,110.8 455.6,110.3 457.8,109.9 459.9,109.4 462.0,108.9 464.2,108.4 466.4,108.0 468.5,107.5 470.6,107.0 472.8,106.5 474.9,106.0 477.1,105.5 479.3,105.1 481.4,104.6 483.5,104.1 485.7,103.6 487.9,103.1 490.0,102.6"/><circle class="s1" cx="193.4" cy="148.7" r="5"/><text class="t-s2" x="500.0" y="208.8" text-anchor="start">k1 = -log x: can be &lt; 0</text><text class="t-s1" x="500.0" y="106.6" text-anchor="start">k3 = x - log x - 1</text><text class="t-muted" x="193.4" y="170.7" text-anchor="middle">both 0 at x = 1</text><text class="t-tick" x="275.0" y="248.0" text-anchor="middle">x = pi_ref(o) / pi_theta(o)</text></svg><figcaption>The two per-token KL estimates as functions of x, the ratio of reference to policy probability. The plain estimate k1 = -log x goes below zero when x is above 1. Equation 2, k3 = x - log x - 1, stays at or above zero and touches zero only at x = 1.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 680 332" role="img" aria-label="After one GRPO step on our group, the log-probability of every right answer went up and every wrong answer went down."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-tick" x="20.0" y="22.0" text-anchor="start">answer</text><text class="t-tick" x="110.0" y="22.0" text-anchor="start">reward</text><text class="t-tick" x="190.0" y="22.0" text-anchor="start">A</text><text class="t-tick" x="400.0" y="22.0" text-anchor="middle">change of log pi_theta(o_i | q) after one step</text><line class="axis" x1="400.0" y1="30.0" x2="400.0" y2="312.0"/><text class="t-tick" x="20.0" y="52.0" text-anchor="start">0</text><text class="t-tick" x="110.0" y="52.0" text-anchor="start">0</text><text class="t-tick" x="190.0" y="52.0" text-anchor="start">-0.54</text><g class="mark"><title>answer 0: -50.23 to -51.66</title><rect class="s2" x="254.7" y="40" width="145.3" height="20" rx="3"/></g><text class="t-val" x="244.7" y="54.0" text-anchor="end">-1.436</text><text class="t-tick" x="20.0" y="86.0" text-anchor="start">1</text><text class="t-tick" x="110.0" y="86.0" text-anchor="start">0</text><text class="t-tick" x="190.0" y="86.0" text-anchor="start">-0.54</text><g class="mark"><title>answer 1: -96.85 to -97.39</title><rect class="s2" x="345.8" y="74" width="54.2" height="20" rx="3"/></g><text class="t-val" x="335.8" y="88.0" text-anchor="end">-0.536</text><text class="t-tick" x="20.0" y="120.0" text-anchor="start">2</text><text class="t-tick" x="110.0" y="120.0" text-anchor="start">0</text><text class="t-tick" x="190.0" y="120.0" text-anchor="start">-0.54</text><g class="mark"><title>answer 2: -190.40 to -192.30</title><rect class="s2" x="208.4" y="108" width="191.6" height="20" rx="3"/></g><text class="t-val" x="198.4" y="122.0" text-anchor="end">-1.894</text><text class="t-tick" x="20.0" y="154.0" text-anchor="start">3</text><text class="t-tick" x="110.0" y="154.0" text-anchor="start">0</text><text class="t-tick" x="190.0" y="154.0" text-anchor="start">-0.54</text><g class="mark"><title>answer 3: -162.58 to -163.85</title><rect class="s2" x="271.3" y="142" width="128.7" height="20" rx="3"/></g><text class="t-val" x="261.3" y="156.0" text-anchor="end">-1.271</text><text class="t-tick" x="20.0" y="188.0" text-anchor="start">4</text><text class="t-tick" x="110.0" y="188.0" text-anchor="start">1</text><text class="t-tick" x="190.0" y="188.0" text-anchor="start">+1.62</text><g class="mark"><title>answer 4: -80.34 to -78.57</title><rect class="s1" x="400.0" y="176" width="179.0" height="20" rx="3"/></g><text class="t-val" x="587.0" y="190.0" text-anchor="start">+1.768</text><text class="t-tick" x="20.0" y="222.0" text-anchor="start">5</text><text class="t-tick" x="110.0" y="222.0" text-anchor="start">1</text><text class="t-tick" x="190.0" y="222.0" text-anchor="start">+1.62</text><g class="mark"><title>answer 5: -80.29 to -78.32</title><rect class="s1" x="400.0" y="210" width="200.0" height="20" rx="3"/></g><text class="t-val" x="608.0" y="224.0" text-anchor="start">+1.976</text><text class="t-tick" x="20.0" y="256.0" text-anchor="start">6</text><text class="t-tick" x="110.0" y="256.0" text-anchor="start">0</text><text class="t-tick" x="190.0" y="256.0" text-anchor="start">-0.54</text><g class="mark"><title>answer 6: -72.73 to -73.09</title><rect class="s2" x="362.9" y="244" width="37.1" height="20" rx="3"/></g><text class="t-val" x="352.9" y="258.0" text-anchor="end">-0.367</text><text class="t-tick" x="20.0" y="290.0" text-anchor="start">7</text><text class="t-tick" x="110.0" y="290.0" text-anchor="start">0</text><text class="t-tick" x="190.0" y="290.0" text-anchor="start">-0.54</text><g class="mark"><title>answer 7: -137.88 to -138.45</title><rect class="s2" x="342.6" y="278" width="57.4" height="20" rx="3"/></g><text class="t-val" x="332.6" y="292.0" text-anchor="end">-0.567</text></svg><figcaption>The change of each answer's log-probability after one AdamW step with learning rate 1e-7 on our group. The two right answers (A = +1.62) went up by 1.77 and 1.98; the six wrong answers (A = -0.54) went down by between 0.37 and 1.89.</figcaption></figure>

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
