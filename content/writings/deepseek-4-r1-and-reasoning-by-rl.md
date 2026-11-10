---
title: "DeepSeek-R1 and Reasoning by Reinforcement Learning"
description: "How DeepSeek taught models to reason with rewards a program can check: GRPO by hand, R1-Zero, the four-stage R1 recipe, distillation, costs, and what came after."
date: 2026-11-10
tags: [deepseek, reinforcement-learning, grpo, reasoning, llm]
series: "DeepSeek, From the Ground Up"
series_part: 4
motif: graph
accent: "#7fd4a8"
---

Give a model a hard maths problem and a pencil. Do not show it how to solve the problem. Tell it only, at the end, whether its final answer was right. Then let it try again, thousands of times, on thousands of problems.

In January 2025, DeepSeek showed what happens when you do this to a large model. Their model DeepSeek-R1-Zero started at 15.6% on AIME 2024, a hard competition for school students. After pure reinforcement learning, with no worked examples from people, it scored 77.9% (the figure in the Nature version of the paper). Along the way, its answers grew from a few hundred tokens to many thousands. It started to check its own work, and to write "Wait" when it found a mistake.

[Part 3](deepseek-3-training-v3.md) showed how DeepSeek pre-trained V3, the base model under all of this. This part shows what happened after pre-training. It builds the method, GRPO, from a group of four answers you can score by hand. Then it follows R1 from the first experiment to the model people use, and on to V3.2, V4 and V4.1.

> [!TIP] The whole idea in one minute
> Reinforcement learning lets a model learn from a score instead of from examples. For maths and code, a program can give the score: is the boxed answer right, do the tests pass? **GRPO** makes this cheap. For each question it samples a group of answers and compares each one with the group's average. No second large network has to predict scores.
>
> DeepSeek used it on V3-Base with no other help and got **R1-Zero**, which learned long, self-checking reasoning on its own. Its text was hard to read, so a four-stage recipe made **R1**. Small models learned R1's skill faster by copying its answers than by their own RL. Later models spent much more compute on RL, then moved to distilling many specialist teachers into one student.

## From predicting the next token to earning a reward

A pre-trained model has one skill. It continues text. Give it the start of a maths solution and it writes a likely next line, because it saw billions of similar lines. Nothing in that training rewards a right answer over a fluent wrong one.

The usual next step is **supervised fine-tuning (SFT)**. People write good answers, and the model learns to copy them. SFT teaches a format and a style. It cannot teach the model to do better than the people who wrote the examples. It also cannot show the model the hundred wrong paths that lead to one right answer.

Reinforcement learning works the other way round. The model writes its own answers. A **reward** scores each answer. Training then makes high-scoring answers more likely and low-scoring answers less likely. Nobody has to write the right path. The model has to find it, and the reward tells it when it did.

> [!DEFINITION] Policy, rollout and reward
> In reinforcement learning (RL) the model being trained is the **policy**: it maps a prompt to a probability for each next token. A **rollout** is one full answer that the policy samples. The **reward** is one number for that answer, for example 1 if it is right and 0 if not.

This is how the assistants of 2022 to 2024 were trained, with a reward model that learned people's preferences and an algorithm called PPO. The page [How models are trained today](how-models-are-trained-today.md) tells that history. This part starts where DeepSeek changed it.

## PPO and its value model

Picture one answer of 600 tokens that gets a reward of 1 at the end. Which of the 600 tokens deserve the credit? Maybe token 212 was the clever step. Maybe token 540 fixed an error. RL needs some way to give each token its share.

**PPO**, the standard algorithm, answers this with a second network, the **value model**. At every token it predicts the reward that the answer will get in the end. If the prediction jumps up after token 212, token 212 was a good move. The difference between what happened and what the value model expected is the **advantage**.

> [!DEFINITION] Advantage
> How much better an action turned out than expected. A positive advantage means "do this more often". A negative one means "do this less often". The expectation that it is compared against is called the **baseline**.

> [!DEFINITION] PPO and the value model
> Proximal Policy Optimization (Schulman et al., 2017) is the RL algorithm that InstructGPT made standard for language models. It needs a **value model** (also called a critic): a network that predicts, at each token, the reward the answer will get. Its estimates turn one end-of-answer reward into an advantage for every token.

That design works, but for a large language model it is expensive in two ways. DeepSeek said so plainly when they proposed the alternative.

> [!PAPER] Shao et al. (2024), DeepSeekMath · §4.1.1 · page 13
> [![The DeepSeekMath paragraph explaining that PPO's value function is another model of comparable size and that usually only the last token gets a reward, with those phrases highlighted](/img/deepseek/papers/p4-math-value.png)](/img/deepseek/papers/p4-math-value.png)
>
> **What it says:** the value function is "another model of comparable size as the policy model", which brings "a substantial memory and computational burden". And "usually only the last token is assigned a reward score", which makes a per-token value function hard to train. GRPO instead "uses the average reward of multiple sampled outputs, produced in response to the same question, as the baseline."
>
> **What it means:**
>
> The first cost is memory. A value model for a 7B policy is about another 7B network, with its own gradients and optimizer state. For a 671B model like V3, that is a second 671B network in training. The GPUs hold two giants instead of one.
>
> The second cost is quality. The value model must look at half an answer and guess its final score. For a long chain of thought this is close to hopeless. The model may write a wrong line at token 300, notice it at token 900 and fix it. A value model that judged token 300 as "bad" was wrong about the outcome. The Nature version of the R1 paper makes this exact point about long reasoning: early content "may later be revised or contradicted".

<figure class="fig"><svg viewBox="0 0 680 250" role="img" aria-label="PPO keeps a policy, a reference, a reward model and a trained value model. GRPO drops the value model and uses the group of answers as the baseline."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="170.0" y="24.0" text-anchor="middle">PPO: four models</text><text class="t-title" x="510.0" y="24.0" text-anchor="middle">GRPO: three, or two plus rules</text><rect class="box-on" x="30.0" y="44.0" width="135.0" height="52.0" rx="8"/><text class="t-note" x="97.5" y="67.0" text-anchor="middle">policy</text><text class="t-muted" x="97.5" y="84.0" text-anchor="middle">trained</text><rect class="box" x="180.0" y="44.0" width="135.0" height="52.0" rx="8"/><text class="t-note" x="247.5" y="67.0" text-anchor="middle">reference</text><text class="t-muted" x="247.5" y="84.0" text-anchor="middle">frozen</text><rect class="box" x="30.0" y="110.0" width="135.0" height="52.0" rx="8"/><text class="t-note" x="97.5" y="133.0" text-anchor="middle">reward model</text><text class="t-muted" x="97.5" y="150.0" text-anchor="middle">frozen</text><rect class="box-on" x="180.0" y="110.0" width="135.0" height="52.0" rx="8"/><text class="t-note" x="247.5" y="133.0" text-anchor="middle">value model</text><text class="t-muted" x="247.5" y="150.0" text-anchor="middle">trained, same size</text><rect class="box-on" x="365.0" y="44.0" width="135.0" height="52.0" rx="8"/><text class="t-note" x="432.5" y="67.0" text-anchor="middle">policy</text><text class="t-muted" x="432.5" y="84.0" text-anchor="middle">trained</text><rect class="box" x="515.0" y="44.0" width="135.0" height="52.0" rx="8"/><text class="t-note" x="582.5" y="67.0" text-anchor="middle">reference</text><text class="t-muted" x="582.5" y="84.0" text-anchor="middle">frozen</text><rect class="box" x="365.0" y="110.0" width="285.0" height="52.0" rx="8"/><text class="t-note" x="507.5" y="133.0" text-anchor="middle">reward: model or rules</text><text class="t-muted" x="507.5" y="150.0" text-anchor="middle">R1: answer checker + format check</text><line class="edge-dim" x1="340.0" y1="12" x2="340.0" y2="250"/><text class="t-muted" x="30.0" y="190.0" text-anchor="start">Advantage of each token comes from the</text><text class="t-muted" x="30.0" y="207.0" text-anchor="start">value model (GAE). The value model must</text><text class="t-muted" x="30.0" y="224.0" text-anchor="start">guess the final reward from a half answer.</text><text class="t-muted" x="365.0" y="190.0" text-anchor="start">Advantage of each answer comes from its</text><text class="t-muted" x="365.0" y="207.0" text-anchor="start">group: sample G answers to one question,</text><text class="t-muted" x="365.0" y="224.0" text-anchor="start">compare each reward with the group mean.</text></svg><figcaption>PPO trains two large networks, the policy and a value model of about the same size. GRPO keeps only the trained policy, a frozen reference copy, and a reward, which can be a set of rules.</figcaption></figure>

## GRPO: let the group be the baseline

Think of a golf course. A score of 5 on a hole means nothing by itself. On a short, flat hole it is a bad score. On a long hole into the wind, where most players took 6, it is a good one. You do not need a model of the hole to judge the 5. You need the scores of the other players on the same hole, on the same day.

GRPO uses the same idea. For one question, it samples a group of answers from the current model. It scores all of them. Each answer is then judged against the others: better than the group's average is good, worse is bad. The group's average replaces the value model.

> [!DEFINITION] GRPO
> Group Relative Policy Optimization, from the DeepSeekMath paper (February 2024). For each question it samples $$G$$ answers and scores them. Each answer's distance from the group mean, divided by the group's standard deviation, becomes the advantage of every token in that answer. It has no value model.

> [!PAPER] Shao et al. (2024), DeepSeekMath · §4.1.1 · page 13
> [![Figure 4 of DeepSeekMath: PPO passes the policy's output through a reference model, a reward model and a trained value model into GAE; GRPO samples outputs o1 to oG, scores them with a reward model, and a group computation gives advantages A1 to AG](/img/deepseek/papers/p4-math-figure.png)](/img/deepseek/papers/p4-math-figure.png)
>
> **What it says:** "GRPO foregoes the value model, instead estimating the baseline from group scores, significantly reducing training resources."
>
> **What it means:**
>
> Compare the two rows. In the top row, one output $$o$$ goes to three models. The value model's output $$v$$ joins the reward inside **GAE** (generalised advantage estimation, PPO's way of spreading credit over tokens). The value model is yellow, so it is trained alongside the policy.
>
> In the bottom row, the policy writes $$G$$ outputs for the same question $$q$$. The reward model gives $$G$$ numbers. A small "Group Computation" box turns them into $$G$$ advantages. Only one yellow box is left.
>
> The price is more sampling: $$G$$ answers per question instead of one. DeepSeekMath used 64 per question. R1-Zero used 16.

### The advantage, by hand

The advantage of answer $$i$$ in a group is its reward, minus the group's mean reward, divided by the group's standard deviation:

$$
\hat{A}_i = \frac{r_i - \operatorname{mean}(r_1, \ldots, r_G)}{\operatorname{std}(r_1, \ldots, r_G)}
$$

Read it left to right. $$\hat{A}_i$$ is the advantage of answer $$i$$. $$r_i$$ is its reward. The mean is the group's average reward, so the top of the fraction says "how much better than the group". The standard deviation is the typical spread of rewards in the group.

Dividing by it puts every group on the same scale, whether its rewards spread over 0 to 1 or 0 to 10. In outcome-only RL, every token of answer $$i$$ gets the same $$\hat{A}_i$$.

Now a group of four. The question is "What is 7 x 8?". The reward follows R1-Zero. An answer gets 1 point if the boxed answer is right. It gets 1 more point if the reasoning sits inside think tags and the answer inside answer tags. Here are four sampled answers and their scores.

<figure class="fig"><svg viewBox="0 0 680 300" role="img" aria-label="Four answers to 7 x 8 get rewards 2, 1, 1, 0 and advantages +1.414, 0, 0, -1.414."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Question: What is 7 x 8? Four sampled answers, scored by rules</text><text class="t-tick" x="250.0" y="52.0" text-anchor="middle">accuracy</text><text class="t-tick" x="330.0" y="52.0" text-anchor="middle">format</text><text class="t-tick" x="410.0" y="52.0" text-anchor="middle">reward</text><text class="t-tick" x="560.0" y="52.0" text-anchor="middle">advantage</text><rect class="box-1" x="20.0" y="64.0" width="190.0" height="36.0" rx="6"/><text class="t-note" x="32.0" y="87.0" text-anchor="start">o1</text><text class="t-muted" x="62.0" y="87.0" text-anchor="start">think tags, boxed 56</text><text class="t-val" x="250.0" y="87.0" text-anchor="middle">1</text><text class="t-muted" x="290.0" y="87.0" text-anchor="middle">+</text><text class="t-val" x="330.0" y="87.0" text-anchor="middle">1</text><text class="t-muted" x="370.0" y="87.0" text-anchor="middle">=</text><text class="t-val" x="410.0" y="87.0" text-anchor="middle">2</text><path class="s1" d="M560,72H612.5684624950369Q616.5684624950369,72 616.5684624950369,76V88Q616.5684624950369,92 612.5684624950369,92H560Z"/><text class="t-s1" x="622.6" y="87.0" text-anchor="start">+1.414</text><rect class="box" x="20.0" y="110.0" width="190.0" height="36.0" rx="6"/><text class="t-note" x="32.0" y="133.0" text-anchor="start">o2</text><text class="t-muted" x="62.0" y="133.0" text-anchor="start">think tags, boxed 54</text><text class="t-val" x="250.0" y="133.0" text-anchor="middle">0</text><text class="t-muted" x="290.0" y="133.0" text-anchor="middle">+</text><text class="t-val" x="330.0" y="133.0" text-anchor="middle">1</text><text class="t-muted" x="370.0" y="133.0" text-anchor="middle">=</text><text class="t-val" x="410.0" y="133.0" text-anchor="middle">1</text><text class="t-muted" x="566.0" y="133.0" text-anchor="start">0.000</text><rect class="box" x="20.0" y="156.0" width="190.0" height="36.0" rx="6"/><text class="t-note" x="32.0" y="179.0" text-anchor="start">o3</text><text class="t-muted" x="62.0" y="179.0" text-anchor="start">think tags, boxed 63</text><text class="t-val" x="250.0" y="179.0" text-anchor="middle">0</text><text class="t-muted" x="290.0" y="179.0" text-anchor="middle">+</text><text class="t-val" x="330.0" y="179.0" text-anchor="middle">1</text><text class="t-muted" x="370.0" y="179.0" text-anchor="middle">=</text><text class="t-val" x="410.0" y="179.0" text-anchor="middle">1</text><text class="t-muted" x="566.0" y="179.0" text-anchor="start">0.000</text><rect class="box-2" x="20.0" y="202.0" width="190.0" height="36.0" rx="6"/><text class="t-note" x="32.0" y="225.0" text-anchor="start">o4</text><text class="t-muted" x="62.0" y="225.0" text-anchor="start">"The answer is 56."</text><text class="t-val" x="250.0" y="225.0" text-anchor="middle">0</text><text class="t-muted" x="290.0" y="225.0" text-anchor="middle">+</text><text class="t-val" x="330.0" y="225.0" text-anchor="middle">0</text><text class="t-muted" x="370.0" y="225.0" text-anchor="middle">=</text><text class="t-val" x="410.0" y="225.0" text-anchor="middle">0</text><path class="s2" d="M503.43153750496305,210H556.0Q560.0,210 560.0,214V226Q560.0,230 556.0,230H503.43153750496305Z" transform="rotate(180 531.7157687524815 220)"/><text class="t-bad" x="497.4" y="225.0" text-anchor="end">-1.414</text><line class="axis" x1="560" y1="60" x2="560" y2="244"/><text class="t-note" x="20.0" y="270.0" text-anchor="start">mean = 1, std = 0.707 (divide by G = 4).  A = (reward - mean) / std</text><text class="t-muted" x="20.0" y="290.0" text-anchor="start">o4 holds the right number but gets the lowest score: no box, no tags, nothing a rule can check.</text></svg><figcaption>Our group of four. The rewards are 2, 1, 1 and 0, so the mean is 1. The right, well-formatted answer gets a positive advantage, the two wrong ones get zero, and the unformatted one gets a negative advantage.</figcaption></figure>

Check it with a pencil. The mean is $$(2 + 1 + 1 + 0) / 4 = 1$$. The squared distances from the mean are 1, 0, 0 and 1, which average to 0.5. The standard deviation is $$\sqrt{0.5} = 0.707$$. So answer 1 gets $$(2 - 1) / 0.707 = +1.414$$ and answer 4 gets $$-1.414$$. Answers 2 and 3 sit exactly at the mean and get 0.

Answer 4 holds the right number, 56, and still gets the worst score. It wrote no box and no tags, so no rule can check it. This is not a bug in the toy. A rule-based reward pays only for answers it can verify, and the model learns that format quickly.

Here is the same computation in code, two ways. Method 1 is plain Python for one group. Method 2 is PyTorch for a whole batch of groups at once, which is how a real trainer does it.

```python
def advantages_scratch(r, eps=1e-6):
    """Method 1: plain Python. Population std (divide by G), as in the formula's usual reading."""
    G = len(r)
    mean = sum(r) / G
    std = math.sqrt(sum((x - mean) ** 2 for x in r) / G)
    return mean, std, [(x - mean) / (std + eps) for x in r]


def advantages_torch(R, eps=1e-6, unbiased=False):
    """Method 2: a whole batch of groups at once. R has shape (B, G): B questions, G answers each."""
    mean = R.mean(dim=1, keepdim=True)                       # (B, 1)
    std = R.std(dim=1, keepdim=True, unbiased=unbiased)      # (B, 1)
    return (R - mean) / (std + eps)                          # (B, G)
```

The output of `part4_grpo.py` (CPU, PyTorch, a few seconds):

```text
Method 1 (from scratch): rewards [2.0, 1.0, 1.0, 0.0]  mean 1.0000  std 0.7071
  advantages: +1.4142, +0.0000, +0.0000, -1.4142
Method 2 (torch, batch of 4 groups), first row: +1.4142, +0.0000, +0.0000, -1.4142
max |Method 1 - Method 2| = 5.08e-08
```

The small `eps` stops a division by zero. When does that happen? When every answer in the group gets the same reward. The script runs three more groups to show what the advantage does in each case:

```text
our group [2,1,1,0]      -> +1.414, +0.000, +0.000, -1.414
one right [1,0,0,0]      -> +1.732, -0.577, -0.577, -0.577
three right [1,1,1,0]    -> +0.577, +0.577, +0.577, -1.732
all right [1,1,1,1]      -> +0.000, +0.000, +0.000, +0.000
```

Look at the second and third rows. When one answer in four is right, that rare success gets a big push, +1.732. When three are right, the one failure gets a big push down. The method spends its effort on the surprise.

The last row is the important limit: a group where every answer is right (or every answer is wrong) has zero advantage everywhere. That question teaches the model nothing in this step. Questions must be hard enough that the model sometimes fails, and within reach often enough that it sometimes succeeds.

> [!WARNING] Which standard deviation?
> The formula says "std" but does not say whether it divides by $$G$$ or by $$G - 1$$. PyTorch's default divides by $$G - 1$$. With it, our group gets ±1.2247 instead of ±1.4142. The direction is the same, but the step size changes. Check this when you compare two GRPO implementations.

> [!PAPER] Shao et al. (2024), DeepSeekMath · §4.1.2 · page 14
> [![The KL estimator equation (4) with which is guaranteed to be positive highlighted, then section 4.1.2 Outcome Supervision RL with GRPO with normalized by subtracting the group average and dividing by the group standard deviation highlighted](/img/deepseek/papers/p4-math-adv.png)](/img/deepseek/papers/p4-math-adv.png)
>
> **What it says:** the rewards "are normalized by subtracting the group average and dividing by the group standard deviation", and outcome supervision "sets the advantages" of all tokens in the output to that normalised reward. Above it, the KL term uses an "unbiased estimator", "which is guaranteed to be positive."
>
> **What it means:**
>
> The second paragraph is our pencil computation, in the paper's notation. Note the words "of all tokens". With an outcome reward, GRPO gives up on judging individual tokens. Every token of a good answer is pushed up by the same amount, even the useless ones. Over thousands of questions, tokens that often appear in good answers rise, and tokens that often appear in bad ones fall. Credit gets sorted out by averaging, not by a value model.
>
> The paper also tried process supervision, with a reward at the end of each reasoning step. R1 later dropped that idea, for reasons we will see near the end.

## The full objective: a clip and a leash

The advantage says which answers to make more likely. The objective says how far to move in one update. GRPO keeps two safety devices from PPO. A **clip** limits how much any token's probability may change per update. A **KL penalty** keeps the model near a reference copy of itself.

> [!DEFINITION] Probability ratio
> For a sampled token: its probability under the model being trained, divided by its probability under the model that sampled it (the "old" policy). A ratio of 1.3 means the update made that token 30% more likely.

> [!DEFINITION] KL divergence
> Kullback-Leibler divergence, a measure of how different two probability distributions are. It is 0 when they are the same and grows as they move apart. Here it measures how far the trained model has drifted from a frozen **reference model**.

> [!PAPER] Shao et al. (2024), DeepSeekMath · §4.1.1 · page 13
> [![Equation 3 of DeepSeekMath, the GRPO objective, with by maximizing the following objective highlighted](/img/deepseek/papers/p4-math-eq.png)](/img/deepseek/papers/p4-math-eq.png)
>
> **What it says:** GRPO samples a group of outputs "from the old policy" and optimizes the policy model "by maximizing the following objective", equation (3). $$\varepsilon$$ and $$\beta$$ are hyper-parameters.
>
> **What it means:**
>
> It looks heavy, but it has only three parts, and you have met two. The fraction inside min is the probability ratio of one token. $$\hat{A}_{i,t}$$ is the group advantage from the pencil example. The new piece is the min with a clip, and the KL term at the end.
>
> The two sums are bookkeeping. The inner sum runs over the tokens of answer $$i$$ and divides by its length $$|o_i|$$, so it is an average per token. The outer sum averages over the $$G$$ answers of the group.

Here is the objective, written out:

$$
\mathcal{J}(\theta) = \frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\Big\{\min\big[\rho_{i,t}\hat{A}_{i,t},\ \operatorname{clip}(\rho_{i,t},1-\varepsilon,1+\varepsilon)\,\hat{A}_{i,t}\big]-\beta\,\mathbb{D}_{KL}\big[\pi_\theta\,\|\,\pi_{ref}\big]\Big\}
$$

$$
\rho_{i,t} = \frac{\pi_\theta(o_{i,t}\mid q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}\mid q, o_{i,<t})}
$$

Read the symbols. $$\theta$$ are the weights being trained, and $$\pi_\theta$$ is the model with those weights. $$q$$ is the question and $$o_i$$ the $$i$$-th answer; $$o_{i,t}$$ is its token number $$t$$, and $$o_{i,<t}$$ the tokens before it.

$$\pi_{\theta_{old}}$$ is the model that sampled the group, and $$\pi_{ref}$$ the frozen reference. $$\rho_{i,t}$$ (rho) is the probability ratio. $$\varepsilon$$ (epsilon) sets the clip range, and $$\beta$$ (beta) the strength of the KL penalty. Training changes $$\theta$$ to make $$\mathcal{J}$$ larger.

### The clip

Take one token of answer 1, with advantage +1.414, and $$\varepsilon = 0.2$$. If the update makes it 10% more likely (ratio 1.1), it adds $$1.1 \times 1.414 = 1.556$$. At ratio 1.2 it adds 1.697. At 1.5 the clip holds the ratio at 1.2, so it still adds only 1.697. Pushing that token further earns nothing, so the gradient stops pushing it. Our script prints the full table:

```text
ratio   A=+1.414   A=-1.414
 0.50     +0.707     -1.131
 0.80     +1.131     -1.131
 1.00     +1.414     -1.414
 1.10     +1.556     -1.556
 1.20     +1.697     -1.697
 1.50     +1.697     -2.121
 2.00     +1.697     -2.828
```

<figure class="fig"><svg viewBox="0 0 680 290" role="img" aria-label="The clipped surrogate: for a positive advantage the gain stops growing above ratio 1.2; for a negative advantage the gain stops below 0.8 but the loss keeps growing above 1.2."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="22.0" text-anchor="start">What one token adds to the objective, as its probability ratio moves (eps = 0.2)</text><line class="grid" x1="70" y1="230.9" x2="590" y2="230.9"/><text class="t-tick" x="62.0" y="234.9" text-anchor="end">-2</text><line class="grid" x1="70" y1="185.5" x2="590" y2="185.5"/><text class="t-tick" x="62.0" y="189.5" text-anchor="end">-1</text><line class="grid" x1="70" y1="140.0" x2="590" y2="140.0"/><text class="t-tick" x="62.0" y="144.0" text-anchor="end">0</text><line class="grid" x1="70" y1="94.5" x2="590" y2="94.5"/><text class="t-tick" x="62.0" y="98.5" text-anchor="end">+1</text><line class="grid" x1="70" y1="49.1" x2="590" y2="49.1"/><text class="t-tick" x="62.0" y="53.1" text-anchor="end">+2</text><text class="t-tick" x="131.2" y="258.0" text-anchor="middle">0.5</text><text class="t-tick" x="222.9" y="258.0" text-anchor="middle">0.8</text><text class="t-tick" x="284.1" y="258.0" text-anchor="middle">1.0</text><text class="t-tick" x="345.3" y="258.0" text-anchor="middle">1.2</text><text class="t-tick" x="437.1" y="258.0" text-anchor="middle">1.5</text><text class="t-tick" x="590.0" y="258.0" text-anchor="middle">2.0</text><rect class="box-ghost" x="222.9" y="40" width="122.4" height="200"/><text class="t-muted" x="284.1" y="54.0" text-anchor="middle">trust region</text><polyline class="l1" points="70.0,126.4 76.1,125.5 82.2,124.5 88.4,123.6 94.5,122.7 100.6,121.8 106.7,120.9 112.8,120.0 118.9,119.1 125.1,118.2 131.2,117.3 137.3,116.4 143.4,115.5 149.5,114.5 155.6,113.6 161.8,112.7 167.9,111.8 174.0,110.9 180.1,110.0 186.2,109.1 192.4,108.2 198.5,107.3 204.6,106.4 210.7,105.5 216.8,104.5 222.9,103.6 229.1,102.7 235.2,101.8 241.3,100.9 247.4,100.0 253.5,99.1 259.6,98.2 265.8,97.3 271.9,96.4 278.0,95.5 284.1,94.5 290.2,93.6 296.4,92.7 302.5,91.8 308.6,90.9 314.7,90.0 320.8,89.1 326.9,88.2 333.1,87.3 339.2,86.4 345.3,85.5 351.4,85.5 357.5,85.5 363.6,85.5 369.8,85.5 375.9,85.5 382.0,85.5 388.1,85.5 394.2,85.5 400.4,85.5 406.5,85.5 412.6,85.5 418.7,85.5 424.8,85.5 430.9,85.5 437.1,85.5 443.2,85.5 449.3,85.5 455.4,85.5 461.5,85.5 467.6,85.5 473.8,85.5 479.9,85.5 486.0,85.5 492.1,85.5 498.2,85.5 504.4,85.5 510.5,85.5 516.6,85.5 522.7,85.5 528.8,85.5 534.9,85.5 541.1,85.5 547.2,85.5 553.3,85.5 559.4,85.5 565.5,85.5 571.6,85.5 577.8,85.5 583.9,85.5 590.0,85.5"/><polyline class="l2" points="70.0,176.4 76.1,176.4 82.2,176.4 88.4,176.4 94.5,176.4 100.6,176.4 106.7,176.4 112.8,176.4 118.9,176.4 125.1,176.4 131.2,176.4 137.3,176.4 143.4,176.4 149.5,176.4 155.6,176.4 161.8,176.4 167.9,176.4 174.0,176.4 180.1,176.4 186.2,176.4 192.4,176.4 198.5,176.4 204.6,176.4 210.7,176.4 216.8,176.4 222.9,176.4 229.1,177.3 235.2,178.2 241.3,179.1 247.4,180.0 253.5,180.9 259.6,181.8 265.8,182.7 271.9,183.6 278.0,184.5 284.1,185.5 290.2,186.4 296.4,187.3 302.5,188.2 308.6,189.1 314.7,190.0 320.8,190.9 326.9,191.8 333.1,192.7 339.2,193.6 345.3,194.5 351.4,195.5 357.5,196.4 363.6,197.3 369.8,198.2 375.9,199.1 382.0,200.0 388.1,200.9 394.2,201.8 400.4,202.7 406.5,203.6 412.6,204.5 418.7,205.5 424.8,206.4 430.9,207.3 437.1,208.2 443.2,209.1 449.3,210.0 455.4,210.9 461.5,211.8 467.6,212.7 473.8,213.6 479.9,214.5 486.0,215.5 492.1,216.4 498.2,217.3 504.4,218.2 510.5,219.1 516.6,220.0 522.7,220.9 528.8,221.8 534.9,222.7 541.1,223.6 547.2,224.5 553.3,225.5 559.4,226.4 565.5,227.3 571.6,228.2 577.8,229.1 583.9,230.0 590.0,230.9"/><text class="t-s1" x="596.0" y="89.5" text-anchor="start">A = +1: flat</text><text class="t-s1" x="596.0" y="105.5" text-anchor="start">above 1.2</text><text class="t-bad" x="76.1" y="210.5" text-anchor="start">A = -1: keeps falling</text><text class="t-bad" x="76.1" y="227.5" text-anchor="start">as the ratio grows past 1.2</text><text class="t-tick" x="330.0" y="280.0" text-anchor="middle">ratio = new probability / old probability of the sampled token</text></svg><figcaption>The clipped term for one token. With a positive advantage the gain stops at ratio 1.2. With a negative advantage the gain stops at 0.8, but the penalty keeps growing if a bad token became more likely.</figcaption></figure>

The min makes the clip one-sided in a careful way. A good token gets no extra credit for moving more than 20%. A bad token gets no extra credit for being pushed down more than 20%. But if an update made a bad token *more* likely (ratio 1.5 with a negative advantage), the min keeps the full penalty, −2.121. The objective is always the more pessimistic of the two readings. This keeps each update small, so the samples from the old model stay a fair guide to the new one.

### The KL leash

The second device is the KL term. GRPO estimates it per token with a formula that needs only the probabilities of the sampled token:

$$
\mathbb{D}_{KL}\big[\pi_\theta\,\|\,\pi_{ref}\big] \approx \frac{\pi_{ref}(o_{i,t})}{\pi_\theta(o_{i,t})} - \log\frac{\pi_{ref}(o_{i,t})}{\pi_\theta(o_{i,t})} - 1
$$

Call the ratio $$x = \pi_{ref}/\pi_\theta$$. The formula is $$x - \log x - 1$$. It is 0 when $$x = 1$$ (the two models agree on that token) and positive for every other $$x$$, because $$\log x \le x - 1$$ always. This estimator is often called **k3**, from John Schulman's note on KL approximations. A pencil check: if the trained model gives a token 0.5 and the reference gave it 0.4, then $$x = 0.8$$ and the term is $$0.8 + 0.223 - 1 = 0.023$$.

The script checks the claim that matters: averaged over tokens sampled from the trained model, k3 equals the true KL.

```text
pi_theta 0.5  pi_ref 0.4  ->  k3 = 0.0231
pi_theta 0.9  pi_ref 0.1  ->  k3 = 1.3083
pi_theta 0.1  pi_ref 0.9  ->  k3 = 5.8028
exact KL(pi_theta || pi_ref) = 0.05143; mean of k3 over 200,000 samples = 0.05156
```

The last line is the point. One token's k3 can be large or small, but the average over 200,000 samples lands within 0.0002 of the exact value. Note the third line too: a token the trained model almost never picks (0.1) but the reference liked (0.9) gives a large penalty. V3.2 later found this case caused trouble, as we will see.

One more design choice. PPO for language models usually subtracts the KL from the reward at every token. GRPO puts it straight into the loss instead, so the advantage stays a clean group comparison. The Nature version of the R1 paper adds a reason that matters for long reasoning.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1, Nature version (arXiv v2) · Supplementary A.3 · page 16
> [![The R1 paper explaining that PPO's per-token KL penalty may implicitly penalize the length of the response, that the reference policy is periodically updated, and that PPO with lambda 0.95 performs considerably worse than GRPO](/img/deepseek/papers/p4-r1-kl-length.png)](/img/deepseek/papers/p4-r1-kl-length.png)
>
> **What it says:** PPO's per-token KL penalty "may implicitly penalize the length of the response". DeepSeek therefore "periodically update the reference policy". In their test on DeepSeek-Coder-V2-Lite, "When λ is set to 0.95" (the common default), "PPO performs considerably worse than GRPO"; with λ = 1.0 it comes close.
>
> **What it means:**
>
> If every token pays a small KL fee inside the reward, a 10,000-token answer pays 20 times the fee of a 500-token answer. The model learns that thinking longer is expensive. That is exactly the wrong lesson for a reasoning model. GRPO's average per token does not add up the fee that way.
>
> The second point is a leash that moves. R1-Zero ran for thousands of steps. A reference frozen at step 0 would pull the model back toward the base model forever. Resetting the reference to the current policy (every 400 steps in R1-Zero) lets the model travel far, while each stretch stays controlled.

### The whole loss in code

Now the full objective on a toy group: four answers of 5, 3, 4 and 2 tokens, padded to 5. Method 1 follows equation (3) with loops. Method 2 is the vectorised form, with a mask for padding.

```python
def grpo_loss_torch(logp_new, logp_old, logp_ref, A, mask, eps=EPS, beta=BETA):
    """Method 2: the same, vectorised. Shapes: logp_* and mask (G, T), A (G,)."""
    ratio = (logp_new - logp_old).exp()                                  # (G, T)
    surr = torch.minimum(ratio * A[:, None], ratio.clamp(1 - eps, 1 + eps) * A[:, None])
    x = (logp_ref - logp_new).exp()                                      # pi_ref / pi_theta
    kl = x - (logp_ref - logp_new) - 1                                   # k3, always >= 0
    per_token = (surr - beta * kl) * mask
    per_answer = per_token.sum(dim=1) / mask.sum(dim=1)                  # 1/|o_i| sum_t
    return -per_answer.mean()                                            # 1/G sum_i, negated
```

```text
Method 1 (loops) loss = -0.054389
Method 2 (torch) loss = -0.054389   difference 2.2e-09
gradient of the loss summed over each answer (negative = that answer is pushed up): -0.199, +0.003, -0.004, +0.171
```

The two methods agree to nine decimal places. The last line shows the direction of learning. The optimizer moves against the gradient, so answer 1 (advantage +1.414) is pushed up and answer 4 (−1.414) is pushed down. Answers 2 and 3 have zero advantage. Their tiny gradients come only from the KL term.

## Rewards a program can check

Every RL method depends on its reward. For a chat assistant, the reward is usually a **reward model**: a network trained on people's choices between two answers. It can judge anything, but it can also be fooled.

> [!DEFINITION] Reward hacking
> When a policy finds answers that score high with the reward but are not what the reward was meant to measure. A reward model that likes confident, long answers can be "hacked" by answers that are confident and long, but wrong.

For maths and code there is a better judge: a program. A maths answer can be compared with the known answer. Code can be run against tests. Such a reward cannot be talked into a high score. This approach is now often called RL with verifiable rewards.

> [!DEFINITION] Rule-based (verifiable) reward
> A reward computed by fixed code instead of a learned network: an answer matcher, a compiler with test cases, a format checker. It is exact on the cases it can check, and it cannot be fooled by style.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1, arXiv v1 · §2.2.2 · page 6
> [![The Reward Modeling section of R1 v1: a rule-based reward system with accuracy rewards and format rewards, and the sentence explaining that neural reward models may suffer from reward hacking](/img/deepseek/papers/p4-r1-rewards.png)](/img/deepseek/papers/p4-r1-rewards.png)
>
> **What it says:** R1-Zero uses "a rule-based reward system" with two parts. "Accuracy rewards" check the final answer, given "within a box" for maths, or run test cases for code. "Format rewards" require the thinking to sit between think tags. DeepSeek did not use a neural reward model "because we find that the neural reward model may suffer from reward hacking", and "retraining the reward model needs additional training resources".
>
> **What it means:**
>
> This is a choice to train only on questions that a program can grade. That rules out essays and open questions, but it buys an honest signal. Over thousands of RL steps, a learned judge would face millions of attempts to find its weak spots. A string comparison has no weak spots to find.
>
> The format reward is small but important. It makes the model separate its scratch work from its answer, so the accuracy check knows where to look. In the Nature version, the two rewards are added "with the same weight". Our toy group used exactly that rule.

Here are our two reward functions, and what they gave the four answers:

```python
def accuracy_reward(text, truth):
    """1 if the last \\boxed{...} holds the right answer, else 0. No box = no checkable answer = 0."""
    boxes = re.findall(r'\\boxed\{([^}]*)\}', text)
    return 1.0 if boxes and boxes[-1].strip() == truth else 0.0


def format_reward(text):
    """1 if the reasoning sits inside <think>...</think> and the answer inside <answer>...</answer>."""
    return 1.0 if re.fullmatch(r'\s*<think>.+?</think>\s*<answer>.+?</answer>\s*', text, flags=re.S) else 0.0
```

```text
o1: accuracy 1 + format 1 = reward 2   | <think>7 x 8 = 7 x 4 x 2 = 28 x 2 = 56</think> <answer>\boxed{56}</answer>
o2: accuracy 0 + format 1 = reward 1   | <think>7 x 8 = 54, I think</think> <answer>\boxed{54}</answer>
o3: accuracy 0 + format 1 = reward 1   | <think>7 x 9 = 63</think> <answer>\boxed{63}</answer>
o4: accuracy 0 + format 0 = reward 0   | The answer is 56.
```

The R1 paper's checkers are more careful than our regular expressions. Its RL infrastructure runs a code executor, an answer matcher and a format checker as separate services (Nature version, Supplementary B.1). The idea is the same.

### Watching it learn

Before the real thing, a toy run. Our "policy" here is not a language model. It is a choice between the four answers above, with a probability for each. At the start, it prefers the unformatted answer, as a base model with no training might. Each step samples 8 answers, scores them with the rules, computes group advantages and takes two clipped updates.

```text
step  0: p(o1..o4) = 0.126, 0.229, 0.229, 0.417   expected reward 0.709
step  5: p(o1..o4) = 0.373, 0.270, 0.270, 0.087   expected reward 1.286
step 10: p(o1..o4) = 0.757, 0.117, 0.090, 0.036   expected reward 1.721
step 20: p(o1..o4) = 0.952, 0.023, 0.017, 0.008   expected reward 1.944
step 30: p(o1..o4) = 0.970, 0.014, 0.010, 0.006   expected reward 1.964
step 40: p(o1..o4) = 0.970, 0.014, 0.010, 0.006   expected reward 1.964
```

<figure class="fig"><svg viewBox="0 0 680 290" role="img" aria-label="In our toy run, the right and well-formatted answer rises from 0.126 to 0.970 in 30 steps, then the curve stops moving."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="22.0" text-anchor="start">Toy GRPO run: probability of each answer to 7 x 8 (our run)</text><line class="grid" x1="60" y1="240.0" x2="500" y2="240.0"/><text class="t-tick" x="52.0" y="244.0" text-anchor="end">0.00</text><line class="grid" x1="60" y1="190.0" x2="500" y2="190.0"/><text class="t-tick" x="52.0" y="194.0" text-anchor="end">0.25</text><line class="grid" x1="60" y1="140.0" x2="500" y2="140.0"/><text class="t-tick" x="52.0" y="144.0" text-anchor="end">0.50</text><line class="grid" x1="60" y1="90.0" x2="500" y2="90.0"/><text class="t-tick" x="52.0" y="94.0" text-anchor="end">0.75</text><line class="grid" x1="60" y1="40.0" x2="500" y2="40.0"/><text class="t-tick" x="52.0" y="44.0" text-anchor="end">1.00</text><text class="t-tick" x="60.0" y="258.0" text-anchor="middle">0</text><text class="t-tick" x="170.0" y="258.0" text-anchor="middle">10</text><text class="t-tick" x="280.0" y="258.0" text-anchor="middle">20</text><text class="t-tick" x="390.0" y="258.0" text-anchor="middle">30</text><text class="t-tick" x="500.0" y="258.0" text-anchor="middle">40</text><polyline class="l1" points="60.0,214.9 71.0,206.9 82.0,197.1 93.0,186.9 104.0,178.6 115.0,165.4 126.0,158.4 137.0,133.6 148.0,110.0 159.0,110.0 170.0,88.5 181.0,76.4 192.0,69.0 203.0,64.0 214.0,59.1 225.0,59.1 236.0,59.1 247.0,55.1 258.0,52.0 269.0,49.6 280.0,49.6 291.0,49.6 302.0,49.6 313.0,49.6 324.0,49.6 335.0,47.7 346.0,47.7 357.0,47.7 368.0,46.0 379.0,46.0 390.0,46.0 401.0,46.0 412.0,46.0 423.0,46.0 434.0,46.0 445.0,46.0 456.0,46.0 467.0,46.0 478.0,46.0 489.0,46.0 500.0,46.0"/><polyline class="l3" points="60.0,194.2 71.0,187.4 82.0,184.3 93.0,181.9 104.0,182.3 115.0,186.0 126.0,182.4 137.0,197.1 148.0,205.1 159.0,205.1 170.0,216.6 181.0,223.0 192.0,227.7 203.0,229.6 214.0,231.2 225.0,231.2 236.0,231.2 247.0,233.6 258.0,234.6 269.0,235.4 280.0,235.4 291.0,235.4 302.0,235.4 313.0,235.4 324.0,235.4 335.0,236.1 346.0,236.1 357.0,236.1 368.0,237.2 379.0,237.2 390.0,237.2 401.0,237.2 412.0,237.2 423.0,237.2 434.0,237.2 445.0,237.2 456.0,237.2 467.0,237.2 478.0,237.2 489.0,237.2 500.0,237.2"/><polyline class="l4" points="60.0,194.2 71.0,187.4 82.0,182.8 93.0,181.9 104.0,182.3 115.0,186.0 126.0,192.2 137.0,201.5 148.0,213.8 159.0,213.8 170.0,222.1 181.0,225.6 192.0,227.7 203.0,229.6 214.0,232.5 225.0,232.5 236.0,232.5 247.0,233.6 258.0,235.4 269.0,236.7 280.0,236.7 291.0,236.7 302.0,236.7 313.0,236.7 324.0,236.7 335.0,237.6 346.0,237.6 357.0,237.6 368.0,238.0 379.0,238.0 390.0,238.0 401.0,238.0 412.0,238.0 423.0,238.0 434.0,238.0 445.0,238.0 456.0,238.0 467.0,238.0 478.0,238.0 489.0,238.0 500.0,238.0"/><polyline class="l2" points="60.0,156.6 71.0,178.3 82.0,195.7 93.0,209.3 104.0,216.7 115.0,222.6 126.0,227.0 137.0,227.8 148.0,231.0 159.0,231.0 170.0,232.8 181.0,235.0 192.0,235.7 203.0,236.9 214.0,237.3 225.0,237.3 236.0,237.3 247.0,237.7 258.0,238.1 269.0,238.3 280.0,238.3 291.0,238.3 302.0,238.3 313.0,238.3 324.0,238.3 335.0,238.6 346.0,238.6 357.0,238.6 368.0,238.8 379.0,238.8 390.0,238.8 401.0,238.8 412.0,238.8 423.0,238.8 434.0,238.8 445.0,238.8 456.0,238.8 467.0,238.8 478.0,238.8 489.0,238.8 500.0,238.8"/><rect class="s1" x="512.0" y="41.0" width="10" height="10" rx="2"/><text class="t-tick" x="528.0" y="50.0" text-anchor="start">o1 right + format: 0.970</text><rect class="s3" x="512.0" y="232.2" width="10" height="10" rx="2"/><text class="t-tick" x="528.0" y="241.2" text-anchor="start">o2 wrong (54): 0.014</text><rect class="s4" x="512.0" y="249.2" width="10" height="10" rx="2"/><text class="t-tick" x="528.0" y="258.2" text-anchor="start">o3 wrong (63): 0.010</text><rect class="s2" x="512.0" y="266.2" width="10" height="10" rx="2"/><text class="t-tick" x="528.0" y="275.2" text-anchor="start">o4 right, no format: 0.006</text><text class="t-tick" x="280.0" y="280.0" text-anchor="middle">GRPO step (8 samples per step, eps 0.2)</text></svg><figcaption>Our toy GRPO run. The unformatted answer falls first, the right and formatted answer rises, and after step 30 nothing changes.</figcaption></figure>

Two things to notice. First, the unformatted answer o4 falls fastest, even though its number was right. The format lesson comes first, just as in real runs.

Second, the curve stops at 0.970. From step 30, every group of 8 samples was eight copies of o1. All rewards equal, all advantages zero, no update. This is the "all right" row from before, in action. Real training keeps a stream of harder questions for exactly this reason.

## R1-Zero: RL straight from the base model

Now the real experiment. DeepSeek took V3-Base, the 671B pre-trained model from Part 3, and applied GRPO with rule-based rewards. There was no SFT first. The only instruction was a template: think inside think tags, then answer inside answer tags.

<figure class="fig"><svg viewBox="0 0 680 310" role="img" aria-label="R1-Zero training loop: 32 questions, 16 answers each, rule-based checks, group advantages, a GRPO update."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="22.0" text-anchor="start">One step of DeepSeek-R1-Zero (numbers from the Nature version)</text><rect class="box-2" x="20.0" y="60.0" width="130.0" height="60.0" rx="8"/><text class="t-note" x="85.0" y="87.0" text-anchor="middle">32 questions</text><text class="t-muted" x="85.0" y="104.0" text-anchor="middle">maths, code, logic</text><rect class="box-on" x="190.0" y="60.0" width="140.0" height="60.0" rx="8"/><text class="t-note" x="260.0" y="87.0" text-anchor="middle">policy</text><text class="t-muted" x="260.0" y="104.0" text-anchor="middle">V3-Base at step 0</text><rect class="box" x="370.0" y="50.0" width="130.0" height="80.0" rx="8"/><text class="t-note" x="435.0" y="87.0" text-anchor="middle">16 answers each</text><text class="t-muted" x="435.0" y="104.0" text-anchor="middle">512 in total</text><rect class="box-1" x="540.0" y="40.0" width="120.0" height="44.0" rx="8"/><text class="t-note" x="600.0" y="67.0" text-anchor="middle">answer checker</text><rect class="box-1" x="540.0" y="96.0" width="120.0" height="44.0" rx="8"/><text class="t-note" x="600.0" y="123.0" text-anchor="middle">format checker</text><line class="edge" x1="150.0" y1="90.0" x2="188.0" y2="90.0" marker-end="url(#ah)"/><line class="edge" x1="330.0" y1="90.0" x2="368.0" y2="90.0" marker-end="url(#ah)"/><line class="edge" x1="500.0" y1="74.0" x2="538.0" y2="62.0" marker-end="url(#ah)"/><line class="edge" x1="500.0" y1="106.0" x2="538.0" y2="118.0" marker-end="url(#ah)"/><rect class="box" x="370.0" y="190.0" width="290.0" height="56.0" rx="8"/><text class="t-note" x="515.0" y="215.0" text-anchor="middle">rewards -&gt; group advantages</text><text class="t-muted" x="515.0" y="232.0" text-anchor="middle">reward = accuracy + format, A = (r - mean)/std</text><rect class="box-on" x="20.0" y="190.0" width="310.0" height="56.0" rx="8"/><text class="t-note" x="175.0" y="215.0" text-anchor="middle">GRPO update of the policy</text><text class="t-muted" x="175.0" y="232.0" text-anchor="middle">lr 3e-6, KL weight 0.001, clip ratio</text><line class="edge" x1="600.0" y1="140.0" x2="600.0" y2="188.0" marker-end="url(#ah)"/><line class="edge" x1="368.0" y1="218.0" x2="332.0" y2="218.0" marker-end="url(#ah)"/><line class="path" x1="260.0" y1="190.0" x2="260.0" y2="124.0" marker-end="url(#ah-on)"/><text class="t-muted" x="20.0" y="280.0" text-anchor="start">Repeat 10,400 steps. Every 400 steps the reference model becomes a copy of the policy.</text><text class="t-muted" x="20.0" y="298.0" text-anchor="start">Answer length cap: 32,768 tokens, raised to 65,536 at step 8,200.</text></svg><figcaption>One step of R1-Zero training: 32 questions, 16 sampled answers each, checks by rules, group advantages, and a GRPO update of the policy.</figcaption></figure>

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1, Nature version (arXiv v2) · §2.1 · page 3
> [![The R1-Zero training settings: 16 outputs with a maximum length of 32,768 tokens before the 8.2k step and 65,536 afterward, a significant jump at the 8.2k step, 10,400 steps, batch size 512, reference model replaced every 400 steps](/img/deepseek/papers/p4-r1-hparams.png)](/img/deepseek/papers/p4-r1-hparams.png)
>
> **What it says:** learning rate 3e-6, KL coefficient 0.001, temperature 1. For each question, "16 outputs with a maximum length of 32,768 tokens before the 8.2k step and 65,536 tokens afterward". Performance and length "exhibit a significant jump at the 8.2k step". Training ran "10,400 steps", with a "batch size of 512", and "Every 400 steps, we replace the reference model with the latest policy model."
>
> **What it means:**
>
> Turn the numbers into a picture. Each step takes 32 questions and samples 16 answers to each, so 512 answers, some of them 30,000 tokens long. The Nature version adds that each rollout produces 8,192 answers (16 steps' worth), split into 16 mini-batches for updates. That sampling is where most of the compute goes.
>
> The KL coefficient of 0.001 is tiny next to DeepSeekMath's 0.04. With the reference reset every 400 steps, the leash is long and keeps moving. The length cap matters too: when DeepSeek doubled it at step 8,200, scores jumped. The model had been cut off mid-thought.

### The score, and which number to quote

The headline result has two versions, and the press mixed them up. Here is what each one says.

<figure class="fig"><svg viewBox="0 0 680 260" role="img" aria-label="R1-Zero AIME 2024 accuracy: 15.6% at the start, 71.0% in arXiv v1, 77.9% in the Nature version, 86.7% with majority voting."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="22.0" text-anchor="start">R1-Zero on AIME 2024: which number, from which version</text><text class="t-note" x="20.0" y="64.0" text-anchor="start">start of RL</text><text class="t-muted" x="20.0" y="81.0" text-anchor="start">both versions</text><path class="s3" d="M270,50H317.48Q321.48,50 321.48,54V74Q321.48,78 317.48,78H270Z"/><text class="t-val" x="329.5" y="70.0" text-anchor="start">15.6%</text><text class="t-note" x="20.0" y="116.0" text-anchor="start">R1-Zero, arXiv v1 (Jan 2025)</text><text class="t-muted" x="20.0" y="133.0" text-anchor="start">pass@1</text><path class="s2" d="M270,102H500.29999999999995Q504.29999999999995,102 504.29999999999995,106V126Q504.29999999999995,130 500.29999999999995,130H270Z"/><text class="t-val" x="512.3" y="122.0" text-anchor="start">71.0%</text><text class="t-note" x="20.0" y="168.0" text-anchor="start">R1-Zero, Nature / arXiv v2</text><text class="t-muted" x="20.0" y="185.0" text-anchor="start">pass@1</text><path class="s1" d="M270,154H523.0699999999999Q527.0699999999999,154 527.0699999999999,158V178Q527.0699999999999,182 523.0699999999999,182H270Z"/><text class="t-val" x="535.1" y="174.0" text-anchor="start">77.9%</text><text class="t-note" x="20.0" y="220.0" text-anchor="start">R1-Zero, majority vote</text><text class="t-muted" x="20.0" y="237.0" text-anchor="start">cons@16 (v2), cons@64 (v1)</text><path class="s4" d="M270,206H552.11Q556.11,206 556.11,210V230Q556.11,234 552.11,234H270Z"/><text class="t-val" x="564.1" y="226.0" text-anchor="start">86.7%</text></svg><figcaption>R1-Zero on AIME 2024. Both versions start at 15.6%. arXiv v1 reports 71.0% pass@1, the Nature version reports 77.9%, and majority voting gives 86.7% in both.</figcaption></figure>

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1, arXiv v1 · §2.2.4 · page 6
> [![R1 v1 text: the average pass@1 score on AIME 2024 jumping from an initial 15.6% to an impressive 71.0%](/img/deepseek/papers/p4-r1-aime-v1.png)](/img/deepseek/papers/p4-r1-aime-v1.png)
>
> **What it says:** the January 2025 preprint reports AIME 2024 pass@1 "jumping from an initial 15.6% to an impressive 71.0%", comparable to OpenAI-o1-0912.
>
> **What it means:**
>
> **Pass@1** here is the average accuracy of single attempts: DeepSeek samples 16 answers per question and averages their accuracy. So 71.0% means a single R1-Zero answer was right about 7 times in 10. Starting from 15.6%, that is more than four times better, from rewards alone.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1, Nature version (arXiv v2) · §2.3 · page 4
> [![Nature version text: the average pass@1 score on AIME 2024 jumping from an initial 15.6% to 77.9%](/img/deepseek/papers/p4-r1-aime-v2.png)](/img/deepseek/papers/p4-r1-aime-v2.png)
>
> **What it says:** the peer-reviewed version, published in Nature on 17 September 2025, reports "jumping from an initial 15.6% to 77.9%". With self-consistency decoding (majority voting) it reaches 86.7%.
>
> **What it means:**
>
> The two versions report different final numbers for the same named model. The Nature text also describes the length cap rising from 32,768 to 65,536 tokens at step 8,200, with a jump in performance. It is our reading, not DeepSeek's statement, that this longer run explains 77.9% versus 71.0%. When you quote R1-Zero, name the version: 71.0% (arXiv v1) or 77.9% (Nature, arXiv v2).

### Longer answers, and the word "wait"

Nobody told R1-Zero to think longer. The reward did not pay for length. Yet the length of its answers grew almost steadily through training.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1, arXiv v1 · §2.2.4 · page 8
> [![Figure 3 of R1 v1: the average response length of DeepSeek-R1-Zero on the training set grows from a few hundred tokens to nearly 10,000 over about 8,000 RL steps](/img/deepseek/papers/p4-r1-length.png)](/img/deepseek/papers/p4-r1-length.png)
>
> **What it says:** Figure 3 shows "The average response length of DeepSeek-R1-Zero on the training set during the RL process". The model "naturally learns to solve reasoning tasks with more thinking time."
>
> **What it means:**
>
> Read the axes. Over about 8,000 steps, the average answer grows from a few hundred tokens to nearly 10,000. The pale band is the noise from step to step; the dark line is the trend. Its slope does not flatten.
>
> Why would rewards for right answers make answers longer? Because on hard problems, longer attempts were more often right. Answers that tried a second method, or checked a result, won their groups more often. GRPO pushed up every token of those winners, including the extra checking. Length is a side effect of what wins, not a goal.

The checking turned into a recognisable habit. DeepSeek called it the "aha moment".

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1, arXiv v1 · §2.2.4 · page 9
> [![Table 3 of R1 v1: R1-Zero squares both sides of an equation, then writes Wait, wait. Wait. That's an aha moment I can flag here, and re-evaluates step by step](/img/deepseek/papers/p4-r1-aha.png)](/img/deepseek/papers/p4-r1-aha.png)
>
> **What it says:** an intermediate R1-Zero, solving an equation with nested square roots, stops and writes "Wait, wait. Wait. That’s an aha moment I can flag here." The caption says the model learns "to rethink using an anthropomorphic tone".
>
> **What it means:**
>
> Look at what happens around the line. The model squares both sides, rearranges, reaches a quartic equation, and then stops itself. It goes back to the start and redoes the steps more carefully. No one wrote that behaviour into the data. It survived because answers that went back and checked won more of their groups.
>
> Be careful with the word "aha". It is a human word for a pattern of tokens. The model learned that writing "wait" and re-checking pays off. That is impressive, but it is not evidence of a feeling.

The Nature version made the moment measurable instead of anecdotal.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1, Nature version (arXiv v2) · §2.3 · page 5
> [![Nature version text: the aha moment characterized by a sudden increase in the use of the word wait during reflections](/img/deepseek/papers/p4-r1-wait.png)](/img/deepseek/papers/p4-r1-wait.png)
>
> **What it says:** the aha moment is "characterized by a sudden increase in the use of the word “wait” during reflections", with details in Supplementary C.2.
>
> **What it means:**
>
> In that supplement, DeepSeek counts reflective words such as "wait", "mistake", "verify" and "check" over training. Together they rise 5 to 7 times from the start. "Wait" alone was almost absent early on, appeared now and then between steps 4,000 and 7,000, and then spiked after step 8,000. The habit did not grow smoothly. It switched on, which is why it looked like a moment.

There was a price. R1-Zero's thinking was often hard to read. It mixed languages, "occasionally combining English and Chinese within a single chain-of-thought response" (Nature version). The supplement suggests why: V3-Base was trained mostly on Chinese and English, and nothing in the reward asked for one language. The reward paid only for the final box, so the scratch work drifted into whatever helped.

### What RL adds, and what it does not

Is the model learning new skills, or getting more reliable at skills it had? DeepSeekMath had already asked this in 2024.

> [!DEFINITION] Pass@k and Maj@k
> **Pass@k**: a question counts as solved if any of $$k$$ sampled answers is right. It measures what the model *can* do. **Maj@k** (also cons@k): the most common answer among $$k$$ samples is taken as the answer. It measures what the model does *reliably*.

> [!PAPER] Shao et al. (2024), DeepSeekMath · §5.2.2 · page 21
> [![DeepSeekMath text: RL enhances Maj@K's performance but not Pass@K, rendering the output distribution more robust, boosting the correct response from TopK rather than the enhancement of fundamental capabilities](/img/deepseek/papers/p4-math-majk.png)](/img/deepseek/papers/p4-math-majk.png)
>
> **What it says:** "RL enhances Maj@K’s performance but not Pass@K." The improvement seems to come from "boosting the correct response from TopK rather than the enhancement of fundamental capabilities."
>
> **What it means:**
>
> Picture a student who can solve a problem one time in five. RL does not make the student smarter. It makes the one-in-five path the habit. That is what our toy run did: o1 was always possible, and GRPO made it the default.
>
> This was a 7B model with short answers, so it does not settle the question for R1-Zero. But it explains why the base model matters so much. RL can only make a good path common if the model sometimes finds it. R1's own supplement says the same from the other side: RL on a 7B dense and a 16B MoE base "consistently failed to yield meaningful improvements".

## From R1-Zero to R1: four stages

R1-Zero proved a point. It was not a product. Its thinking mixed languages and lacked structure, and it trained only on questions with checkable answers. DeepSeek built R1 in four stages to keep the reasoning and fix the rest.

> [!DEFINITION] Rejection sampling
> Sample many answers per prompt, keep only the good ones (checked by rules or a judge), and throw the rest away. The kept answers become SFT data.

<figure class="fig"><svg viewBox="0 0 680 350" role="img" aria-label="The four stages of DeepSeek-R1: cold-start SFT, reasoning RL, rejection sampling with SFT on 800K samples, and RL for all scenarios."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="22.0" text-anchor="start">DeepSeek-R1: four stages, each starting from a checkpoint</text><rect class="box-1" x="20.0" y="44.0" width="300.0" height="52.0" rx="8"/><text class="t-note" x="170.0" y="67.0" text-anchor="middle">1. cold start SFT</text><text class="t-muted" x="170.0" y="84.0" text-anchor="middle">thousands of long, readable CoT</text><text class="t-note" x="340.0" y="75.0" text-anchor="start">V3-Base -&gt; Dev1</text><line class="edge" x1="170.0" y1="96.0" x2="170.0" y2="108.0" marker-end="url(#ah)"/><rect class="box-on" x="20.0" y="110.0" width="300.0" height="52.0" rx="8"/><text class="t-note" x="170.0" y="133.0" text-anchor="middle">2. reasoning RL</text><text class="t-muted" x="170.0" y="150.0" text-anchor="middle">rules + language consistency</text><text class="t-note" x="340.0" y="141.0" text-anchor="start">Dev1 -&gt; Dev2</text><line class="edge" x1="170.0" y1="162.0" x2="170.0" y2="174.0" marker-end="url(#ah)"/><rect class="box-1" x="20.0" y="176.0" width="300.0" height="52.0" rx="8"/><text class="t-note" x="170.0" y="199.0" text-anchor="middle">3. rejection sampling + SFT</text><text class="t-muted" x="170.0" y="216.0" text-anchor="middle">600K reasoning + 200K other</text><text class="t-note" x="340.0" y="207.0" text-anchor="start">fresh V3-Base -&gt; Dev3</text><line class="edge" x1="170.0" y1="228.0" x2="170.0" y2="240.0" marker-end="url(#ah)"/><rect class="box-on" x="20.0" y="242.0" width="300.0" height="52.0" rx="8"/><text class="t-note" x="170.0" y="265.0" text-anchor="middle">4. RL for all scenarios</text><text class="t-muted" x="170.0" y="282.0" text-anchor="middle">rules + reward models</text><text class="t-note" x="340.0" y="273.0" text-anchor="start">Dev3 -&gt; R1</text><path class="edge" d="M 470 136 C 530 136, 530 202, 470 202" marker-end="url(#ah)"/><text class="t-muted" x="540.0" y="166.0" text-anchor="start">Dev2 writes the</text><text class="t-muted" x="540.0" y="182.0" text-anchor="start">600K reasoning</text><text class="t-muted" x="540.0" y="198.0" text-anchor="start">samples</text><text class="t-muted" x="20.0" y="320.0" text-anchor="start">Stage 3 trains a fresh copy of V3-Base for two epochs on about 800K samples.</text><text class="t-muted" x="20.0" y="338.0" text-anchor="start">The same 800K samples later train the distilled Qwen and Llama models.</text></svg><figcaption>The four stages of DeepSeek-R1, and the checkpoint each one produces. Stage 3 starts again from V3-Base, with data written by the Dev2 checkpoint.</figcaption></figure>

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1, Nature version (arXiv v2) · §3 · page 6
> [![Figure 2 of the Nature version: R1-Zero samples reasoning data that is filtered and refined; SFT on cold-start long CoT gives Dev-1; RL with rule-based reward and language consistency gives Dev-2; SFT on reasoning and non-reasoning data from V3-Base gives Dev-3; RL with rule-based and preference rewards gives R1](/img/deepseek/papers/p4-r1-pipeline.png)](/img/deepseek/papers/p4-r1-pipeline.png)
>
> **What it says:** "The multi-stage pipeline of DeepSeek-R1". Dev1, Dev2 and Dev3 "represent intermediate checkpoints within this pipeline."
>
> **What it means:**
>
> Follow the arrows from the left. R1-Zero writes reasoning, which is filtered and cleaned with help from V3 and people. That becomes the cold-start data, and SFT on it gives Dev1.
>
> RL on reasoning prompts gives Dev2. Dev2 then writes a large set of reasoning answers, which join non-reasoning data from V3. A fresh V3-Base is fine-tuned on all of it, giving Dev3. A last RL stage gives R1.
>
> The pattern is "RL to find good answers, SFT to teach them cleanly". RL is good at discovering; SFT is good at installing. The pipeline uses each where it is strong.

The four stages, with their numbers (arXiv v1 unless noted):

1. **Cold start.** "Thousands" of long, readable chains of thought fine-tune V3-Base. The format ends each answer with a summary. The Nature version's supplement says the motivation "is primarily product-driven".
2. **Reasoning RL.** The R1-Zero recipe, plus a language consistency reward. In the Nature version this stage uses a GRPO clip ratio $$\varepsilon$$ of 10, very large next to the usual 0.2. The paper says only that the value "plays a crucial role in training".
3. **Rejection sampling and SFT.** Dev2 writes answers; only correct ones are kept, and mixed-language or messy ones are filtered out. That gives about 600K reasoning samples. With about 200K non-reasoning samples (writing, factual questions, translation), V3-Base is fine-tuned for two epochs on about 800K samples.
4. **RL for all scenarios.** Rule rewards for reasoning, reward models for helpfulness and harmlessness. The helpfulness model judges only the final summary; the harmlessness model judges the whole response, thinking included.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1, arXiv v1 · §2.3.2 · page 10
> [![R1 v1 text: a language consistency reward calculated as the proportion of target language words in the CoT, which results in a slight degradation in the model's performance](/img/deepseek/papers/p4-r1-langmix.png)](/img/deepseek/papers/p4-r1-langmix.png)
>
> **What it says:** "we introduce a language consistency reward during RL training, which is calculated as the proportion of target language words in the CoT." The ablation shows "a slight degradation in the model’s performance", but the result is more readable.
>
> **What it means:**
>
> Our script scores two versions of one thought. "First square both sides 然后 we get a quartic 方程 so check each root" has 12 English words and 2 Chinese ones, so it scores 12 / 14 = 0.857. The clean English version scores 1.000.
>
> The interesting part is the cost. A reward for readability lowered scores a little. So the mixing was not random noise; some of it helped the model think. DeepSeek chose readers over a few points. The Nature supplement reports the trade in detail: maths stayed about the same, coding dropped slightly.

The last stage shows where reward models still cause trouble.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1, Nature version (arXiv v2) · §3.2 · page 8
> [![Nature version text: the clip ratio plays a crucial role in training; the second RL stage has 1,700 training steps, with general instruction data and preference rewards only in the final 400 steps, because more steps may lead to reward hacking](/img/deepseek/papers/p4-r1-stage2.png)](/img/deepseek/papers/p4-r1-stage2.png)
>
> **What it says:** the second RL stage "comprises a total of 1,700 training steps", and preference-based rewards are used "exclusively in the final 400 steps". "More training steps with the model based preference reward signal may lead to reward hacking". The reward is the sum of a reasoning reward, a general reward and a language reward.
>
> **What it means:**
>
> Here DeepSeek meets the problem that R1-Zero avoided. The helpfulness reward model is a network, and a network can be gamed. Their supplement shows a run where the reward score climbs while real Codeforces performance falls. Their fix is plain: use the learned reward for only 400 steps, not long enough for the policy to find its weak spots.
>
> So the line between the two kinds of reward is also a line in time. Rule rewards can run for 10,000 steps. Learned rewards get a short, final pass.

### What each stage buys

The Nature version measured every checkpoint. This is the clearest picture of the recipe.

<figure class="fig"><svg viewBox="0 0 680 300" role="img" aria-label="Scores of R1-Zero, Dev1, Dev2, Dev3 and R1 on AIME 2024, IF-Eval and AlpacaEval 2.0."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="22.0" text-anchor="start">What each stage buys (Nature version, Table 3)</text><line class="grid" x1="60" y1="240.0" x2="500" y2="240.0"/><text class="t-tick" x="52.0" y="244.0" text-anchor="end">0</text><line class="grid" x1="60" y1="190.0" x2="500" y2="190.0"/><text class="t-tick" x="52.0" y="194.0" text-anchor="end">25</text><line class="grid" x1="60" y1="140.0" x2="500" y2="140.0"/><text class="t-tick" x="52.0" y="144.0" text-anchor="end">50</text><line class="grid" x1="60" y1="90.0" x2="500" y2="90.0"/><text class="t-tick" x="52.0" y="94.0" text-anchor="end">75</text><line class="grid" x1="60" y1="40.0" x2="500" y2="40.0"/><text class="t-tick" x="52.0" y="44.0" text-anchor="end">100</text><text class="t-tick" x="60.0" y="260.0" text-anchor="middle">R1-Zero</text><text class="t-tick" x="170.0" y="260.0" text-anchor="middle">Dev1</text><text class="t-tick" x="280.0" y="260.0" text-anchor="middle">Dev2</text><text class="t-tick" x="390.0" y="260.0" text-anchor="middle">Dev3</text><text class="t-tick" x="500.0" y="260.0" text-anchor="middle">R1</text><polyline class="l1" points="60.0,84.2 170.0,122.0 280.0,92.0 390.0,83.8 500.0,80.4"/><g class="mark"><title>AIME 2024, R1-Zero: 77.9</title><circle class="s1 ring" cx="60.0" cy="84.2" r="5"/></g><g class="mark"><title>AIME 2024, Dev1: 59.0</title><circle class="s1 ring" cx="170.0" cy="122.0" r="5"/></g><g class="mark"><title>AIME 2024, Dev2: 74.0</title><circle class="s1 ring" cx="280.0" cy="92.0" r="5"/></g><g class="mark"><title>AIME 2024, Dev3: 78.1</title><circle class="s1 ring" cx="390.0" cy="83.8" r="5"/></g><g class="mark"><title>AIME 2024, R1: 79.8</title><circle class="s1 ring" cx="500.0" cy="80.4" r="5"/></g><polyline class="l3" points="60.0,146.8 170.0,96.6 280.0,96.0 390.0,83.8 500.0,73.4"/><g class="mark"><title>IF-Eval, R1-Zero: 46.6</title><circle class="s3 ring" cx="60.0" cy="146.8" r="5"/></g><g class="mark"><title>IF-Eval, Dev1: 71.7</title><circle class="s3 ring" cx="170.0" cy="96.6" r="5"/></g><g class="mark"><title>IF-Eval, Dev2: 72.0</title><circle class="s3 ring" cx="280.0" cy="96.0" r="5"/></g><g class="mark"><title>IF-Eval, Dev3: 78.1</title><circle class="s3 ring" cx="390.0" cy="83.8" r="5"/></g><g class="mark"><title>IF-Eval, R1: 83.3</title><circle class="s3 ring" cx="500.0" cy="73.4" r="5"/></g><polyline class="l2" points="60.0,190.6 170.0,139.8 280.0,128.4 390.0,115.8 500.0,64.8"/><g class="mark"><title>AlpacaEval 2.0, R1-Zero: 24.7</title><circle class="s2 ring" cx="60.0" cy="190.6" r="5"/></g><g class="mark"><title>AlpacaEval 2.0, Dev1: 50.1</title><circle class="s2 ring" cx="170.0" cy="139.8" r="5"/></g><g class="mark"><title>AlpacaEval 2.0, Dev2: 55.8</title><circle class="s2 ring" cx="280.0" cy="128.4" r="5"/></g><g class="mark"><title>AlpacaEval 2.0, Dev3: 62.1</title><circle class="s2 ring" cx="390.0" cy="115.8" r="5"/></g><g class="mark"><title>AlpacaEval 2.0, R1: 87.6</title><circle class="s2 ring" cx="500.0" cy="64.8" r="5"/></g><rect class="s2" x="516.0" y="59.8" width="10" height="10" rx="2"/><text class="t-note" x="532.0" y="68.8" text-anchor="start">AlpacaEval 2.0: 87.6</text><rect class="s3" x="516.0" y="77.8" width="10" height="10" rx="2"/><text class="t-note" x="532.0" y="86.8" text-anchor="start">IF-Eval: 83.3</text><rect class="s1" x="516.0" y="95.8" width="10" height="10" rx="2"/><text class="t-note" x="532.0" y="104.8" text-anchor="start">AIME 2024: 79.8</text><text class="t-muted" x="60.0" y="284.0" text-anchor="start">The cold start (Dev1) costs AIME points; reasoning RL (Dev2) wins them back.</text></svg><figcaption>Benchmark scores at each checkpoint (Nature version, Table 3). The cold start lifts instruction following but costs AIME points. Reasoning RL recovers them. The final RL stage mostly lifts the general chat benchmark.</figcaption></figure>

| Benchmark | R1-Zero | Dev1 | Dev2 | Dev3 | R1 |
|---|---|---|---|---|---|
| AIME 2024 (pass@1) | 77.9 | 59.0 | 74.0 | 78.1 | 79.8 |
| MATH-500 | 95.9 | 94.2 | 95.9 | 95.4 | 97.3 |
| GPQA Diamond | 75.8 | 66.1 | 70.7 | 71.2 | 71.5 |
| LiveCodeBench | 50.0 | 57.5 | 63.5 | 64.6 | 65.9 |
| IF-Eval (prompt strict) | 46.6 | 71.7 | 72.0 | 78.1 | 83.3 |
| AlpacaEval 2.0 (LC win rate) | 24.7 | 50.1 | 55.8 | 62.1 | 87.6 |
| ArenaHard | 53.6 | 77.0 | 73.2 | 75.6 | 92.3 |

Source: DeepSeek-R1, Nature version (arXiv v2), Table 3, page 9.

Read down the AIME row. The cold start drops it from 77.9 to 59.0: a few thousand human-friendly examples pulled the model away from its own long style. Reasoning RL brings it back to 74.0. Note GPQA too: R1-Zero's 75.8 is never reached again. R1 is not better than R1-Zero at everything. It is a model people can use.

## Distillation: small models learn faster by copying

The 800K samples from stage 3 had a second use. DeepSeek fine-tuned six open models on them, with SFT only and no RL: Qwen2.5-Math-1.5B and 7B, Qwen2.5-14B and 32B, Llama-3.1-8B and Llama-3.3-70B-Instruct.

| Model | AIME 2024 pass@1 | MATH-500 | GPQA Diamond | LiveCodeBench | Codeforces rating |
|---|---|---|---|---|---|
| R1-Distill-Qwen-1.5B | 28.9 | 83.9 | 33.8 | 16.9 | 954 |
| R1-Distill-Qwen-7B | 55.5 | 92.8 | 49.1 | 37.6 | 1189 |
| R1-Distill-Llama-8B | 50.4 | 89.1 | 49.0 | 39.6 | 1205 |
| R1-Distill-Qwen-14B | 69.7 | 93.9 | 59.1 | 53.1 | 1481 |
| R1-Distill-Qwen-32B | 72.6 | 94.3 | 62.1 | 57.2 | 1691 |
| R1-Distill-Llama-70B | 70.0 | 94.5 | 65.2 | 57.5 | 1633 |
| GPT-4o-0513 (for scale) | 9.3 | 74.6 | 49.9 | 32.9 | 759 |

Source: R1 arXiv v1, Table 5 (identical in the Nature version, Table 15).

A 1.5B model at 28.9% on AIME, three times GPT-4o's 9.3%, from SFT on a teacher's answers. But the more important experiment asked a sharper question. Could a 32B model reach the same level with its own large-scale RL, like R1-Zero did?

<figure class="fig"><svg viewBox="0 0 680 322" role="img" aria-label="On Qwen2.5-32B, distillation from R1 beats large-scale RL on all four benchmarks."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="22.0" text-anchor="start">Same Qwen2.5-32B base, two ways to teach it to reason</text><rect class="s3" x="20" y="36" width="12" height="12" rx="2"/><text class="t-note" x="38.0" y="47.0" text-anchor="start">large-scale RL, over 10K steps</text><rect class="s1" x="300" y="36" width="12" height="12" rx="2"/><text class="t-note" x="318.0" y="47.0" text-anchor="start">SFT on 800K R1 samples (distilled)</text><text class="t-note" x="20.0" y="98.0" text-anchor="start">AIME 2024</text><path class="s3" d="M160,70H334.6Q338.6,70 338.6,74V84Q338.6,88 334.6,88H160Z"/><text class="t-val" x="344.6" y="84.0" text-anchor="start">47.0</text><path class="s1" d="M160,92H431.88Q435.88,92 435.88,96V106Q435.88,110 431.88,110H160Z"/><text class="t-val" x="441.9" y="106.0" text-anchor="start">72.6</text><text class="t-note" x="20.0" y="156.0" text-anchor="start">MATH-500</text><path class="s3" d="M160,128H504.08Q508.08,128 508.08,132V142Q508.08,146 504.08,146H160Z"/><text class="t-val" x="514.1" y="142.0" text-anchor="start">91.6</text><path class="s1" d="M160,150H514.3399999999999Q518.3399999999999,150 518.3399999999999,154V164Q518.3399999999999,168 514.3399999999999,168H160Z"/><text class="t-val" x="524.3" y="164.0" text-anchor="start">94.3</text><text class="t-note" x="20.0" y="214.0" text-anchor="start">GPQA Diamond</text><path class="s3" d="M160,186H365.0Q369.0,186 369.0,190V200Q369.0,204 365.0,204H160Z"/><text class="t-val" x="375.0" y="200.0" text-anchor="start">55.0</text><path class="s1" d="M160,208H391.98Q395.98,208 395.98,212V222Q395.98,226 391.98,226H160Z"/><text class="t-val" x="402.0" y="222.0" text-anchor="start">62.1</text><text class="t-note" x="20.0" y="272.0" text-anchor="start">LiveCodeBench</text><path class="s3" d="M160,244H308.76Q312.76,244 312.76,248V258Q312.76,262 308.76,262H160Z"/><text class="t-val" x="318.8" y="258.0" text-anchor="start">40.2</text><path class="s1" d="M160,266H373.36Q377.36,266 377.36,270V280Q377.36,284 373.36,284H160Z"/><text class="t-val" x="383.4" y="280.0" text-anchor="start">57.2</text><text class="t-muted" x="20.0" y="312.0" text-anchor="start">Pass@1 in %, arXiv v1 Table 6 (same numbers in the Nature version, Table 16).</text></svg><figcaption>The same Qwen2.5-32B base, trained two ways. Distillation from R1 wins on all four benchmarks.</figcaption></figure>

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1, arXiv v1 · §4.1 · page 15
> [![Table 6 of R1 v1: QwQ-32B-Preview, DeepSeek-R1-Zero-Qwen-32B and DeepSeek-R1-Distill-Qwen-32B on AIME, MATH-500, GPQA and LiveCodeBench, followed by the two conclusions about distillation](/img/deepseek/papers/p4-r1-distill.png)](/img/deepseek/papers/p4-r1-distill.png)
>
> **What it says:** RL on Qwen2.5-32B for over 10K steps reaches about QwQ-32B-Preview (AIME 47.0), while the distilled model gets 72.6. So "distilling more powerful models into smaller ones yields excellent results", while small models doing RL "may not even achieve the performance of distillation". But going further "may still require more powerful base models and larger-scale reinforcement learning."
>
> **What it means:**
>
> Recall the Maj@k lesson. RL makes good paths common, but only the paths the model can already find. A 32B model finds fewer of R1's paths on its own. Distillation hands it the paths directly, written out token by token.
>
> The two conclusions fit together. To *move the frontier*, you need a big base and big RL. To *spread* the result, copying is cheaper and better. This is the logic that V4 later builds its whole post-training around.

The Nature version adds one more test. DeepSeek ran RL on Qwen2-Math-7B, a base released in August 2024, before OpenAI's o1 and its long reasoning outputs were public. After about 10,000 steps it reached 22.3% on AIME 2024, against 7.9% for its instruct version. The point is that RL alone can grow reasoning, even in a base that never saw reasoning traces from another model.

## What did not work

The R1 paper has a short section that many papers leave out: two ideas that failed.

> [!DEFINITION] Process reward model (PRM)
> A reward model that scores each step of a solution instead of only the final answer. It promises finer credit for each step, the same goal as PPO's value model.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1, arXiv v1 · §4.2 · page 15
> [![R1 v1 text on the Process Reward Model: three main limitations, the difficulty of defining a fine-grain step, and that a model-based PRM inevitably leads to reward hacking](/img/deepseek/papers/p4-r1-prm.png)](/img/deepseek/papers/p4-r1-prm.png)
>
> **What it says:** a PRM has "three main limitations": it is "challenging to explicitly define a fine-grain step"; it is hard to judge whether a step is correct; and once a model-based PRM is used, "it inevitably leads to reward hacking".
>
> **What it means:**
>
> What is one step in a 10,000-token thought? A line, a paragraph, an equation? R1-Zero's thinking has no clean steps. It wanders, backtracks and says "wait". A step judge trained on tidy textbook solutions would punish exactly the messy checking that made R1-Zero good.
>
> The third point is the deeper one. A PRM is a learned network judging every step, so the policy gets thousands of chances per answer to fool it. The paper adds that PRMs still help to rerank the best of N answers at inference time. They failed as the training signal, not as a tool.

> [!DEFINITION] Monte Carlo tree search (MCTS)
> A search that explores a tree of possible moves, guided by a value model that rates positions. It powered AlphaGo and AlphaZero.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1, arXiv v1 · §4.2 · page 16
> [![R1 v1 text on MCTS: an exponentially larger search space, getting stuck in local optima, a fine-grained value model is inherently difficult to train, and iteratively boosting performance through self-search remains a significant challenge](/img/deepseek/papers/p4-r1-mcts.png)](/img/deepseek/papers/p4-r1-mcts.png)
>
> **What it says:** token generation has an "exponentially larger search space" than chess. Limits on the search lead to getting "stuck in local optima". "Training a fine-grained value model is inherently difficult". So "iteratively boosting model performance through self-search remains a significant challenge."
>
> **What it means:**
>
> A Go board has at most a few hundred legal moves. A language model chooses among about 128,000 tokens at every position, for thousands of positions. Even with steps grouped into larger moves, the tree is too wide to search well.
>
> Both failures come back to one thing: judging a half-finished answer. PPO's value model, the PRM and MCTS all need it. GRPO with outcome rewards never judges partial work. It waits for the end, checks the answer, and lets averaging over many groups do the rest.

## The Nature version: cost, data and safety

The arXiv preprint of January 2025 said nothing about cost. The Nature paper of September 2025, with its supplement, did. These numbers are the most quoted, and the most misquoted, part of the R1 story.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1, Nature version (arXiv v2) · Supplementary B.4.4 · page 35
> [![The Training Cost section: A100 GPUs for experiments with a smaller 30B model, 64*8 H800 GPUs for R1-Zero over about 198 hours, the same GPUs for R1 over about 80 hours, and 5K GPU hours for the SFT datasets](/img/deepseek/papers/p4-r1-cost-text.png)](/img/deepseek/papers/p4-r1-cost-text.png)
>
> **What it says:** preparation used "A100 GPUs" with a "smaller model (30B parameters)". R1-Zero used "64*8 H800 GPUs" for "approximately 198 hours". R1 used the same GPUs for "about 4 days, or roughly 80 hours". Creating the SFT data took "5K GPU hours".
>
> **What it means:**
>
> 64 x 8 is 512 GPUs, not the 648 that some news reports printed. 512 GPUs for 198 hours is 101,376 GPU hours, which the table rounds to 101K. R1's 80 hours give 40,960, rounded to 41K. Our script does this arithmetic so you can check it.
>
> Notice the first sentence too. The 30B experiments on A100s came first, and their cost is not in the total. Neither are the failed PRM and MCTS attempts.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1, Nature version (arXiv v2) · Supplementary Table 7 · page 37
> [![Table 7: training costs of DeepSeek-R1 at an H800 rental price of 2 dollars per GPU hour: R1-Zero 101K hours or 202K dollars, SFT data creation 5K or 10K dollars, R1 41K or 82K dollars, total 147K hours or 294K dollars](/img/deepseek/papers/p4-r1-cost-table.png)](/img/deepseek/papers/p4-r1-cost-table.png)
>
> **What it says:** "Training costs of DeepSeek-R1, assuming the rental price of H800 is $2 per GPU hour": 101K + 5K + 41K = 147K H800 GPU hours, or $294K.
>
> **What it means:**
>
> Like V3's $5.576M in Part 3, this is a rental-equivalent price for GPUs DeepSeek owns, not money that left a bank account. And it is the cost *on top of* V3-Base. Without the base model, there is nothing to run RL on.

<figure class="fig"><svg viewBox="0 0 680 292" role="img" aria-label="R1 cost 147K GPU hours, or $294K: 101K for R1-Zero, 5K for SFT data, 41K for R1. The V3 base model cost 2,788K GPU hours."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="22.0" text-anchor="start">GPU hours (thousands of H800 hours), at the papers' $2 per hour</text><text class="t-note" x="20.0" y="64.0" text-anchor="start">R1-Zero RL</text><path class="s1" d="M190,48H197.954806312769Q201.954806312769,48 201.954806312769,52V68Q201.954806312769,72 197.954806312769,72H190Z"/><text class="t-val" x="210.0" y="65.0" text-anchor="start">101K = $202K</text><text class="t-note" x="20.0" y="108.0" text-anchor="start">SFT data creation</text><path class="s3" d="M190,92H190Q192,92 192,94V114Q192,116 190,116H190Z"/><text class="t-val" x="200.0" y="109.0" text-anchor="start">5K = $10K</text><text class="t-note" x="20.0" y="152.0" text-anchor="start">R1 (all its stages)</text><path class="s1" d="M190,136H190.85294117647058Q194.85294117647058,136 194.85294117647058,140V156Q194.85294117647058,160 190.85294117647058,160H190Z"/><text class="t-val" x="202.9" y="153.0" text-anchor="start">41K = $82K</text><text class="t-note" x="20.0" y="196.0" text-anchor="start">R1 total</text><path class="s2" d="M190,180H203.39956958393114Q207.39956958393114,180 207.39956958393114,184V200Q207.39956958393114,204 203.39956958393114,204H190Z"/><text class="t-val" x="215.4" y="197.0" text-anchor="start">147K = $294K</text><text class="t-note" x="20.0" y="240.0" text-anchor="start">V3 base model, for scale</text><path class="s4" d="M190,224H516.0Q520.0,224 520.0,228V244Q520.0,248 516.0,248H190Z"/><text class="t-val" x="528.0" y="241.0" text-anchor="start">2,788K = $5.576M</text><text class="t-muted" x="20.0" y="280.0" text-anchor="start">R1 rows: Nature version, Supplementary Table 7. V3 row: V3 report, Table 1. Neither counts research runs.</text></svg><figcaption>R1's reinforcement learning and SFT data cost 147K GPU hours. The V3 base model under it cost 2,788K. Neither figure counts research runs.</figcaption></figure>

```text
R1-Zero: 512 H800 x 198 h = 101,376 GPU hours (table: 101K)
R1:      512 H800 x  80 h = 40,960 GPU hours (table: 41K)
SFT data creation: 5,000 GPU hours (table: 5K)
total as printed: 147,000 GPU hours x $2 = $294,000
V3 base model (V3 report, Table 1): 2,788,000 GPU hours = $5,576,000
R1 on top of V3, as a share of the V3 figure: 5.3%  (our arithmetic)
```

> [!WARNING] "R1 cost $294K"
> That headline is true only as written in the paper. It means 147K H800 GPU hours for R1's own RL and SFT data, priced at $2 per hour. It excludes the V3 base ($5.576M on the same terms, itself excluding research), the 30B preparation runs and all failed experiments. The 5.3% share above is our arithmetic, not DeepSeek's.

The Nature version also disclosed things the preprint did not. V3-Base was trained on "plain web pages and e-books, without incorporating any synthetic data", though DeepSeek notes that some web pages hold OpenAI-generated answers. The RL data is described by domain. It holds 26K maths questions, 17K programming problems plus 8K bug fixes, 22K STEM multiple-choice questions and 15K logic problems. General helpfulness and harmlessness prompts complete it.

Maths decontamination removed about six million pre-training texts. A safety section compares R1 with other models and describes the extra risk-control system that DeepSeek runs in its service.

## After R1: thinking modes, bigger RL, and distillation again

R1 was one model with one mode: always think. The models that followed changed three things. They merged thinking and non-thinking into one model. They spent far more compute on RL. And then, in V4, they replaced the final RL stage with distillation.

<figure class="fig"><svg viewBox="0 0 680 340" role="img" aria-label="Timeline: GRPO in DeepSeekMath, R1, V3.1 hybrid thinking, DeepSeekMath-V2, V3.2, V4 on-policy distillation, V4.1 data and environments."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="22.0" text-anchor="start">How DeepSeek post-training changed, 2024 to 2026</text><line class="axis" x1="110" y1="40" x2="110" y2="320"/><circle class="s1 ring" cx="110" cy="56" r="6"/><text class="t-tick" x="98.0" y="60.0" text-anchor="end">Feb 2024</text><text class="t-note" x="126.0" y="60.0" text-anchor="start">DeepSeekMath</text><text class="t-muted" x="310.0" y="60.0" text-anchor="start">GRPO: no value model</text><circle class="s1 ring" cx="110" cy="96" r="6"/><text class="t-tick" x="98.0" y="100.0" text-anchor="end">Jan 2025</text><text class="t-note" x="126.0" y="100.0" text-anchor="start">R1-Zero and R1</text><text class="t-muted" x="310.0" y="100.0" text-anchor="start">pure RL; four-stage pipeline</text><circle class="s1 ring" cx="110" cy="136" r="6"/><text class="t-tick" x="98.0" y="140.0" text-anchor="end">Aug 2025</text><text class="t-note" x="126.0" y="140.0" text-anchor="start">V3.1</text><text class="t-muted" x="310.0" y="140.0" text-anchor="start">thinking + non-thinking in one model</text><circle class="s1 ring" cx="110" cy="176" r="6"/><text class="t-tick" x="98.0" y="180.0" text-anchor="end">Nov 2025</text><text class="t-note" x="126.0" y="180.0" text-anchor="start">DeepSeekMath-V2</text><text class="t-muted" x="310.0" y="180.0" text-anchor="start">a verifier as the reward</text><circle class="s1 ring" cx="110" cy="216" r="6"/><text class="t-tick" x="98.0" y="220.0" text-anchor="end">Dec 2025</text><text class="t-note" x="126.0" y="220.0" text-anchor="start">V3.2, V3.2-Speciale</text><text class="t-muted" x="310.0" y="220.0" text-anchor="start">RL &gt; 10% of pre-training cost</text><circle class="s1 ring" cx="110" cy="256" r="6"/><text class="t-tick" x="98.0" y="260.0" text-anchor="end">Apr 2026</text><text class="t-note" x="126.0" y="260.0" text-anchor="start">V4</text><text class="t-muted" x="310.0" y="260.0" text-anchor="start">mixed RL replaced by on-policy distillation</text><circle class="s1 ring" cx="110" cy="296" r="6"/><text class="t-tick" x="98.0" y="300.0" text-anchor="end">Sep 2026</text><text class="t-note" x="126.0" y="300.0" text-anchor="start">V4.1-Flash</text><text class="t-muted" x="310.0" y="300.0" text-anchor="start">same algorithms; better data and environments</text></svg><figcaption>Two and a half years of DeepSeek post-training, from GRPO in a 7B maths model to V4.1-Flash.</figcaption></figure>

### R1-0528 and V3.1: one model, two modes

In May 2025, **R1-0528** updated R1 with "increased computational resources" and "algorithmic optimization mechanisms" during post-training, by its model card. AIME 2025 rose from 70.0% to 87.5%. There was no paper.

In August 2025, **V3.1** put both behaviours into one set of weights. Its card says: "One model supports both thinking mode and non-thinking mode by changing the chat template." The thinking mode reached answer quality "comparable" to R1-0528 "while responding more quickly". The card gives no RL details, so the recipe behind hybrid thinking is not public.

> [!DEFINITION] Hybrid thinking
> One model that can answer directly or first write a chain of thought, chosen by the chat template. It replaces a separate chat model and reasoning model.

### V3.2: RL at more than 10% of pre-training

The V3.2 paper (December 2025) is mostly about sparse attention, which [Part 5](deepseek-5-sparse-attention-and-v3-2.md) covers. Its RL sections are just as important.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-V3.2 · §1 · page 2
> [![V3.2 introduction: a stable and scalable RL protocol that allocates a post-training computational budget exceeding 10% of the pre-training cost, and agentic task synthesis with over 1,800 environments and 85,000 prompts](/img/deepseek/papers/p4-v32-budget.png)](/img/deepseek/papers/p4-v32-budget.png)
>
> **What it says:** the RL framework "allocates a post-training computational budget exceeding 10% of the pre-training cost, unlocking advanced capabilities." Later the paper adds that performance kept improving "with extended RL training budget".
>
> **What it means:**
>
> Compare with V3, whose post-training was 5K GPU hours, under 0.2% of its 2,788K total. Even R1's 147K hours were about 5% of the V3 figure. V3.2 put more than 10% of pre-training into post-training. RL went from a finishing step to a major phase.
>
> The paper gives no GPU hours and does not say which pre-training it measures against. If it were V3's 2,664K pre-training hours, 10% would be about 266K GPU hours. That is our arithmetic on an assumption, not a DeepSeek number.

V3.2 also changed the shape of the pipeline. First, specialist models for maths, programming, logic, agents, agentic coding and agentic search are each trained with RL. Their outputs become training data for one model. Then a single RL stage trains reasoning, agents and human alignment together.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-V3.2 · §2.2 · page 6
> [![V3.2 text: merge reasoning, agent and human alignment training into one RL stage; V3.2-Speciale trained exclusively on reasoning data with a reduced length penalty, with the dataset and reward method from DeepSeekMath-V2](/img/deepseek/papers/p4-v32-speciale.png)](/img/deepseek/papers/p4-v32-speciale.png)
>
> **What it says:** "we merge reasoning, agent, and human alignment training into one RL stage", to avoid forgetting between stages. Rewards are rule-based outcomes, a length penalty and language consistency, plus a generative reward model with per-prompt rubrics for general tasks. **V3.2-Speciale** "was trained exclusively on reasoning data with a reduced length penalty during RL", using the "dataset and reward method from DeepSeekMath-V2".
>
> **What it means:**
>
> R1 had four stages because each fixed a problem of the one before. V3.2 folds the RL into one stage, so learning to be helpful cannot erase what reasoning RL taught. A **length penalty** now appears: once models learned to think long, they had to be taught to stop.
>
> Speciale is the experiment of loosening that penalty. The paper's Table 3 shows the cost: about 23K output tokens per AIME 2025 answer for 96.0%. It also reached gold-level scores at IMO 2025 (35 of 42) and IOI 2025 (492 of 600), by the paper's Table 4.

DeepSeekMath-V2 (November 2025) supplied that proof reward. It trains a verifier that grades whole proofs and a second check on the verifier itself, then rewards the prover with the verifier's score. It is the first DeepSeek reward for answers that are not a single number. [Part 8](deepseek-8-the-specialised-models.md) covers it with the other specialised models.

### Four changes that keep big RL stable

Scaling RL ten times broke things that small runs never showed. V3.2 changes the advantage and adds four fixes, all built on GRPO: an unbiased KL estimate, off-policy sequence masking, Keep Routing and Keep Sampling Mask.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-V3.2 · §3.1 · page 7
> [![V3.2 section 3.1: the advantage calculated by subtracting the average reward of the group; the Unbiased KL Estimate with the importance-sampling ratio; weak KL penalty or omitting it entirely can improve performance for mathematics](/img/deepseek/papers/p4-v32-grpo.png)](/img/deepseek/papers/p4-v32-grpo.png)
>
> **What it says:** the advantage is now "subtracting the average reward of the group", with no division by the standard deviation. The "Unbiased KL Estimate" multiplies the k3 term by the ratio $$\pi_\theta / \pi_{old}$$. For some domains, "such as mathematics, applying a relatively weak KL penalty or even omitting it entirely can yield improved performance."
>
> **What it means:**
>
> Run our groups through the new advantage. Our group [2, 1, 1, 0] becomes +1, 0, 0, −1 instead of ±1.414. The "one right" group becomes +0.75 and −0.25, instead of +1.732 and −0.577. Without the division, a group with a small spread no longer gets its advantages blown up. Rare surprises still win, but less dramatically.
>
> The KL fix targets the case our k3 table showed: a token the trained model rarely picks but the reference liked. The paper says k3's gradient gives such tokens "disproportionately large, unbounded weights". The tokens were sampled by the old model, not the current one, so a sampling correction is needed. And for maths, the best leash may be no leash.

```text
V3.2 mean-only advantage (no division by std), our group: +1.000, +0.000, +0.000, -1.000
V3.2 mean-only, one right of four:                         +0.750, -0.250, -0.250, -0.250
```

The other three fixes deal with a quieter problem. The samples come from a fast inference engine, while the updates run in a training framework. The two never compute exactly the same numbers.

> [!DEFINITION] Off-policy data
> Samples produced by a model that is not exactly the one being trained. In large RL runs this happens for two reasons. One big batch of rollouts feeds several updates. And the inference engine and the trainer differ in small numeric details.

The first of them, **Off-Policy Sequence Masking**, drops answers that have a negative advantage *and* have drifted far from the current model, measured by KL. The paper's reasoning: a model "benefits the most by learning from its own mistakes", but mistakes made by a noticeably different model can mislead it. Answers with positive advantage are never masked.

<figure class="fig"><svg viewBox="0 0 680 312" role="img" aria-label="In sampling the router picks experts 2 and 5. Recomputing the route during training may pick 2 and 6. Keep Routing replays the recorded route."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="22.0" text-anchor="start">Keep Routing: train the same experts that wrote the answer</text><text class="t-note" x="20.0" y="76.0" text-anchor="start">sampling engine</text><rect class="box" x="220.0" y="56.0" width="70.0" height="32.0" rx="8"/><text class="t-note" x="255.0" y="77.0" text-anchor="middle">token</text><rect class="box" x="330.0" y="56.0" width="32.0" height="32.0" rx="5"/><text class="t-tick" x="346.0" y="77.0" text-anchor="middle">0</text><rect class="box" x="370.0" y="56.0" width="32.0" height="32.0" rx="5"/><text class="t-tick" x="386.0" y="77.0" text-anchor="middle">1</text><rect class="box-on" x="410.0" y="56.0" width="32.0" height="32.0" rx="5"/><text class="t-tick" x="426.0" y="77.0" text-anchor="middle">2</text><rect class="box" x="450.0" y="56.0" width="32.0" height="32.0" rx="5"/><text class="t-tick" x="466.0" y="77.0" text-anchor="middle">3</text><rect class="box" x="490.0" y="56.0" width="32.0" height="32.0" rx="5"/><text class="t-tick" x="506.0" y="77.0" text-anchor="middle">4</text><rect class="box-on" x="530.0" y="56.0" width="32.0" height="32.0" rx="5"/><text class="t-tick" x="546.0" y="77.0" text-anchor="middle">5</text><rect class="box" x="570.0" y="56.0" width="32.0" height="32.0" rx="5"/><text class="t-tick" x="586.0" y="77.0" text-anchor="middle">6</text><rect class="box" x="610.0" y="56.0" width="32.0" height="32.0" rx="5"/><text class="t-tick" x="626.0" y="77.0" text-anchor="middle">7</text><line class="edge" x1="290.0" y1="72.0" x2="326.0" y2="72.0" marker-end="url(#ah)"/><text class="t-note" x="20.0" y="166.0" text-anchor="start">training, routes recomputed</text><rect class="box" x="220.0" y="146.0" width="70.0" height="32.0" rx="8"/><text class="t-note" x="255.0" y="167.0" text-anchor="middle">token</text><rect class="box" x="330.0" y="146.0" width="32.0" height="32.0" rx="5"/><text class="t-tick" x="346.0" y="167.0" text-anchor="middle">0</text><rect class="box" x="370.0" y="146.0" width="32.0" height="32.0" rx="5"/><text class="t-tick" x="386.0" y="167.0" text-anchor="middle">1</text><rect class="box-on" x="410.0" y="146.0" width="32.0" height="32.0" rx="5"/><text class="t-tick" x="426.0" y="167.0" text-anchor="middle">2</text><rect class="box" x="450.0" y="146.0" width="32.0" height="32.0" rx="5"/><text class="t-tick" x="466.0" y="167.0" text-anchor="middle">3</text><rect class="box" x="490.0" y="146.0" width="32.0" height="32.0" rx="5"/><text class="t-tick" x="506.0" y="167.0" text-anchor="middle">4</text><rect class="box" x="530.0" y="146.0" width="32.0" height="32.0" rx="5"/><text class="t-tick" x="546.0" y="167.0" text-anchor="middle">5</text><rect class="box-on" x="570.0" y="146.0" width="32.0" height="32.0" rx="5"/><text class="t-tick" x="586.0" y="167.0" text-anchor="middle">6</text><rect class="box" x="610.0" y="146.0" width="32.0" height="32.0" rx="5"/><text class="t-tick" x="626.0" y="167.0" text-anchor="middle">7</text><line class="edge" x1="290.0" y1="162.0" x2="326.0" y2="162.0" marker-end="url(#ah)"/><text class="t-note" x="20.0" y="256.0" text-anchor="start">training with Keep Routing</text><rect class="box" x="220.0" y="236.0" width="70.0" height="32.0" rx="8"/><text class="t-note" x="255.0" y="257.0" text-anchor="middle">token</text><rect class="box" x="330.0" y="236.0" width="32.0" height="32.0" rx="5"/><text class="t-tick" x="346.0" y="257.0" text-anchor="middle">0</text><rect class="box" x="370.0" y="236.0" width="32.0" height="32.0" rx="5"/><text class="t-tick" x="386.0" y="257.0" text-anchor="middle">1</text><rect class="box-on" x="410.0" y="236.0" width="32.0" height="32.0" rx="5"/><text class="t-tick" x="426.0" y="257.0" text-anchor="middle">2</text><rect class="box" x="450.0" y="236.0" width="32.0" height="32.0" rx="5"/><text class="t-tick" x="466.0" y="257.0" text-anchor="middle">3</text><rect class="box" x="490.0" y="236.0" width="32.0" height="32.0" rx="5"/><text class="t-tick" x="506.0" y="257.0" text-anchor="middle">4</text><rect class="box-on" x="530.0" y="236.0" width="32.0" height="32.0" rx="5"/><text class="t-tick" x="546.0" y="257.0" text-anchor="middle">5</text><rect class="box" x="570.0" y="236.0" width="32.0" height="32.0" rx="5"/><text class="t-tick" x="586.0" y="257.0" text-anchor="middle">6</text><rect class="box" x="610.0" y="236.0" width="32.0" height="32.0" rx="5"/><text class="t-tick" x="626.0" y="257.0" text-anchor="middle">7</text><line class="edge" x1="290.0" y1="252.0" x2="326.0" y2="252.0" marker-end="url(#ah)"/><text class="t-muted" x="330.0" y="122.0" text-anchor="start">expert 5 picked: its weights produced the token</text><text class="t-bad" x="330.0" y="212.0" text-anchor="start">tiny number differences pick expert 6 instead</text><text class="t-muted" x="330.0" y="302.0" text-anchor="start">the recorded route is replayed: same experts</text></svg><figcaption>Keep Routing. The router in the sampling engine picked experts 2 and 5. Recomputing the route in training can pick a different expert. Keep Routing replays the route that was used.</figcaption></figure>

> [!PAPER] DeepSeek-AI (2025), DeepSeek-V3.2 · §3.1 · page 8
> [![V3.2 text on Keep Routing: preserve the expert routing paths used during sampling, found crucial for RL training stability of MoE models, adopted since DeepSeek-V3-0324; and Keep Sampling Mask: preserve the truncation masks](/img/deepseek/papers/p4-v32-routing.png)](/img/deepseek/papers/p4-v32-routing.png)
>
> **What it says:** DeepSeek "preserve the expert routing paths used during sampling" and enforce them in training. This was "crucial for RL training stability of MoE models", and has been used "since DeepSeek-V3-0324". **Keep Sampling Mask** does the same for top-p and top-k: they "preserve the truncation masks" from sampling and apply them in training.
>
> **What it means:**
>
> Recall from [Part 2](deepseek-2-the-v3-architecture.md) that each token in V3 visits 8 of 256 routed experts, picked by a router score. Two scores that differ in the fifth decimal can pick different experts. Then the trainer computes gradients for an expert that never produced the token. The probability ratio compares two different networks, and the clip can no longer keep updates small.
>
> Sampling masks have the same problem. If sampling removed rare tokens with top-p, the trainer must use the same reduced set of choices, or old and new probabilities mean different things. Both fixes make the trainer see exactly what the sampler did. Note the date: Keep Routing goes back to V3-0324, so it was in use months before the paper described it.

### Agentic tasks: building the exam

R1 trained on maths, code and logic, where answers can be checked. V3.2 wanted agents that use tools. That needs tasks with tools, environments, and a way to check the result. There were not enough of them, so DeepSeek made them.

> [!DEFINITION] Environment
> In agent RL, the world the model acts in. It has tools the model can call (search, a code sandbox, a database) and the state those tools change. It also has a check that says whether the task was done.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-V3.2 · §3.2.3 · page 11
> [![V3.2 text on General Agent: an environment-synthesis agent that synthesizes 1,827 task-oriented environments, hard to solve but quick to verify; tasks keep only instances with non-zero pass@100, 4,417 in total](/img/deepseek/papers/p4-v32-agent.png)](/img/deepseek/papers/p4-v32-agent.png)
>
> **What it says:** an agent "synthesizes 1,827 task-oriented environments". The tasks are hard to solve and quick to verify. The agent builds a database, writes task-specific tools, proposes a task with a solution and a Python verifier, then makes the task harder step by step. Only tasks with "non-zero pass@100" were kept: "4,417 in total".
>
> **What it means:**
>
> This is the R1 idea, turned into a factory. R1 needed answers a program could check. Here a model writes the program that checks. The paper's example is trip planning. Finding a plan that meets every constraint is hard. Testing one candidate plan against the constraints is a few lines of code.
>
> The filter "non-zero pass@100" is the "all wrong" row from our GRPO table, applied up front. A task that 100 attempts never solve gives a group with zero advantage everywhere. It costs compute and teaches nothing, so it is thrown out before training.

| Task type | Tasks | Environment | Prompts |
|---|---|---|---|
| Code agent | 24,667 | real | extracted |
| Search agent | 50,275 | real | synthesized |
| General agent | 4,417 | synthesized | synthesized |
| Code interpreter | 5,908 | real | extracted |

Source: DeepSeek-V3.2, Table 1, page 10. The four rows add up to 85,267, the paper's "85,000 complex prompts".

### V4: on-policy distillation from many teachers

V4 (April 2026) kept most of V3.2's pipeline and changed its last step.

> [!DEFINITION] On-policy distillation (OPD)
> Distillation where the student writes its own answers and the teacher scores every token of them with its full probability distribution. Ordinary distillation trains on the teacher's answers. OPD trains on the student's, so the student learns to recover from its own typical mistakes.

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4 · §5.1 · page 28
> [![V4 text: the mixed RL stage was entirely replaced by On-Policy Distillation; each specialist uses fine-tuning then GRPO, maintaining hyper-parameters closely aligned with prior research](/img/deepseek/papers/p4-v4-opd.png)](/img/deepseek/papers/p4-v4-opd.png)
>
> **What it says:** "the mixed Reinforcement Learning (RL) stage was entirely replaced by On-Policy Distillation". Each domain specialist still gets fine-tuning and then RL with "Group Relative Policy Optimization (GRPO)", "maintaining hyper-parameters closely aligned with our prior research".
>
> **What it means:**
>
> GRPO did not disappear. It moved. It now trains the specialists, one per domain, where a clear reward exists. Merging them into one model is no longer RL with a mixed reward. It is the student learning to match the right teacher on each kind of task.
>
> This is the R1 distillation result, scaled up. R1 showed that copying a strong teacher beats a student's own RL. V4 makes the final model a student of its own specialists.

<figure class="fig"><svg viewBox="0 0 680 300" role="img" aria-label="On-policy distillation in V4: the student writes answers; more than ten specialist teachers give their full token distributions; the student minimises a weighted reverse KL."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="22.0" text-anchor="start">V4: many specialists teach one student, on the student's own answers</text><rect class="box-1" x="20.0" y="44.0" width="116.0" height="40.0" rx="8"/><text class="t-note" x="78.0" y="69.0" text-anchor="middle">maths</text><rect class="box-1" x="150.0" y="44.0" width="116.0" height="40.0" rx="8"/><text class="t-note" x="208.0" y="69.0" text-anchor="middle">code</text><rect class="box-1" x="280.0" y="44.0" width="116.0" height="40.0" rx="8"/><text class="t-note" x="338.0" y="69.0" text-anchor="middle">agents</text><rect class="box-1" x="410.0" y="44.0" width="116.0" height="40.0" rx="8"/><text class="t-note" x="468.0" y="69.0" text-anchor="middle">instructions</text><rect class="box-1" x="540.0" y="44.0" width="116.0" height="40.0" rx="8"/><text class="t-note" x="598.0" y="69.0" text-anchor="middle">...</text><text class="t-muted" x="20.0" y="102.0" text-anchor="start">each specialist: SFT, then GRPO on its own domain (more than ten teachers in total)</text><rect class="box-on" x="220.0" y="140.0" width="240.0" height="52.0" rx="8"/><text class="t-note" x="340.0" y="163.0" text-anchor="middle">student</text><text class="t-muted" x="340.0" y="180.0" text-anchor="middle">writes its own answer</text><rect class="box" x="170.0" y="236.0" width="340.0" height="52.0" rx="8"/><text class="t-note" x="340.0" y="259.0" text-anchor="middle">loss: sum of w_i x KL(student || teacher_i)</text><text class="t-muted" x="340.0" y="276.0" text-anchor="middle">full vocabulary, every token</text><line class="edge" x1="340.0" y1="192.0" x2="340.0" y2="234.0" marker-end="url(#ah)"/><line class="edge" x1="340.0" y1="112.0" x2="340.0" y2="138.0" marker-end="url(#ah)"/><text class="t-muted" x="480.0" y="158.0" text-anchor="start">teachers score every token</text><text class="t-muted" x="480.0" y="176.0" text-anchor="start">of the student's answer</text></svg><figcaption>V4's post-training: specialists trained with SFT and GRPO become teachers. The student writes its own answers, and the loss pulls its token distribution toward each teacher's.</figcaption></figure>

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4 · §5.1.2 · page 33
> [![V4 text: more than ten teacher models distill a single student; prior works use a token-level KL estimate, which leads to high variance in gradient estimation, so V4 adopts full-vocabulary logit distillation](/img/deepseek/papers/p4-v4-teachers.png)](/img/deepseek/papers/p4-v4-teachers.png)
>
> **What it says:** "more than ten teacher models covering various domains are employed to distill a single student model." Earlier work used a per-token KL estimate as an RL-style advantage, which leads to "high variance in gradient estimation". V4 uses "full-vocabulary logit distillation" instead.
>
> **What it means:**
>
> The objective is a weighted sum of reverse KL terms, $$\sum_i w_i \, \mathbb{D}_{KL}(\pi_\theta \,\|\, \pi_{E_i})$$, one per teacher $$E_i$$. "Full vocabulary" means that at each position of the student's answer, the student compares its whole distribution over 128K tokens with the teacher's. It does not compare only the probability of the one token it sampled.
>
> Compare this with GRPO. A group of 16 answers gives one number per answer. A teacher's full distribution gives about 128,000 numbers per token. That is a far richer signal, which is why the estimate is so much less noisy. The price is running more than ten teachers and moving their logits around, which the V4 paper spends pages of engineering on.

### V4.1: the algorithm is not the point

The V4.1-Flash paper (September 2026) makes a statement that sums up the whole arc.

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4.1-Flash · §5.1 · page 25
> [![V4.1 text: we refrain from introducing novel post-training algorithms; under a fixed and unremarkable optimization procedure, improvements in synthesized data and environments account for essentially all of the observed gains](/img/deepseek/papers/p4-v41-data.png)](/img/deepseek/papers/p4-v41-data.png)
>
> **What it says:** "we refrain from introducing novel post-training algorithms." The recipe is SFT, then RL, then OPD. "Under a fixed and unremarkable optimization procedure", better scale, diversity and verifiability of synthesized data and environments "account for essentially all of the observed gains".
>
> **What it means:**
>
> In 2024, DeepSeek's contribution to post-training was an algorithm. By late 2026, their own view is that the algorithm is settled, and the work is in the exams: tasks, environments and checkers. The paper even trains the model to build its own tasks, scored for difficulty and correctness.
>
> It is the same lesson as R1-Zero, one level up. R1-Zero showed that a checkable reward is enough to grow reasoning. V4.1 says that making more, and better, checkable tasks is where the progress now comes from.

## Benchmarks across the line

Scores from different papers use different settings, so compare within a row only with care. Every number below comes from the source named in its column.

| Benchmark | V3 (V3 report) | R1-Zero (R1 Nature, Table 3) | R1 (R1 v1, Table 4) | R1-0528 (card) | V3.1 Thinking (card) | V3.2 Thinking (V3.2, Table 2) | V3.2-Speciale (V3.2, Table 3) |
|---|---|---|---|---|---|---|---|
| AIME 2024 | 39.2 | 77.9 | 79.8 | 91.4 | 93.1 | - | - |
| AIME 2025 | - | - | - | 87.5 | 88.4 | 93.1 | 96.0 |
| GPQA Diamond | 59.1 | 75.8 | 71.5 | 81.0 | 80.1 | 82.4 | 85.7 |
| LiveCodeBench | 40.5 | 50.0 | 65.9 | 73.3 | 74.8 | 83.3 | 88.7 |
| Codeforces rating | - | 1444 | 2029 | 1930 | 2091 | 2386 | 2701 |
| SWE Verified | 42.0 | 43.2 | 49.2 | 57.6 | - | 73.1 | - |

LiveCodeBench uses different date windows in different reports, and the R1-0528 and V3.1 Codeforces numbers are for Div1. V3's report limits output to 8K tokens; R1 allowed 32K and R1-0528 64K. The trend is still clear: each generation spent more on post-training and got more from it.

<details>
<summary>Research notes: GRPO's sequence form in the Nature version, and other details</summary>

- **Sequence-level ratio.** DeepSeekMath writes GRPO per token: a ratio and a clip for each token $$o_{i,t}$$, averaged over $$1/|o_i|$$. The Nature version of R1 writes it per answer: $$\pi_\theta(o_i \mid q) / \pi_{\theta_{old}}(o_i \mid q)$$, with no sum over tokens. It does not discuss the difference. V3.2 goes back to the per-token form. Our toy run, where each "answer" is one choice, is the sequence form by construction.
- **Clip ratio 10.** The Nature version sets "the GRPO clip ratio ε to 10" for R1's first RL stage. With ε = 10 the clip range is [−9, 11], so it almost never binds on the low side. We quote the number as printed and do not interpret it further.
- **Reward models in R1.** The helpfulness reward model was trained on 66,000 preference pairs, each judged by DeepSeek-V3 four times with the order swapped. The safety model was trained point-wise on 106,000 prompts labelled safe or unsafe. Both were trained for one epoch at learning rate 6e-6, batch 256.
- **RL infrastructure.** The Nature supplement describes four modules: rollout on vLLM workers with expert parallelism and MTP for self-speculative decoding, an inference module for the reward and reference models, rule-based reward services run asynchronously, and training.
- **R1-Zero's AIME curve.** The 86.7% majority-vote figure is cons@64 in arXiv v1 (Table 2) and is described as self-consistency in the Nature text; the Nature Figure 1 legend says cons@16.
- **Not public.** The RL recipes of R1-0528 and V3.1, and the GPU hours of any post-training after R1.

</details>

> [!TAKEAWAYS]
> - GRPO (DeepSeekMath, 2024) removes PPO's value model. The advantage of each answer is its reward minus the group mean, divided by the group's standard deviation.
> - Groups where every answer scores the same give zero advantage and teach nothing. Training needs questions the model sometimes solves.
> - GRPO keeps PPO's clip and adds a k3 KL estimate straight into the loss. Our script shows k3 averages to the true KL.
> - R1-Zero applied GRPO with rule rewards (accuracy + format) directly to V3-Base. AIME 2024 rose from 15.6% to 71.0% in arXiv v1 and to 77.9% in the Nature version.
> - Long answers and self-checking ("wait") emerged without being rewarded directly; readability and language mixing were the price.
> - R1 used four stages: cold-start SFT, reasoning RL, rejection sampling with SFT on about 800K samples, and RL for all scenarios. Learned rewards ran only in the last 400 steps.
> - Distilling R1 into a 32B model beat RL on the same base (AIME 72.6 vs 47.0). PRMs and MCTS did not work for training.
> - The Nature version priced R1's RL at 147K H800 GPU hours on 512 GPUs, or $294K at $2 per hour. That is on top of the V3 base.
> - V3.2 spent more than 10% of pre-training cost on RL, with mean-only advantages, an unbiased KL, off-policy masking, Keep Routing and synthesized agent environments.
> - V4 replaced the final mixed RL with on-policy distillation from more than ten specialist teachers; V4.1 says data and environments, not algorithms, gave essentially all its gains.

**Next:** [Part 5, sparse attention and V3.2](deepseek-5-sparse-attention-and-v3-2.md): how DeepSeek made long contexts cheap enough for this much reasoning. A lightning indexer picks 2,048 tokens out of 128K for each query.

## References

1. Z. Shao et al. [*DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*](https://arxiv.org/abs/2402.03300). 2024. GRPO, §4.1.
2. DeepSeek-AI. [*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*](https://arxiv.org/abs/2501.12948). arXiv v1 (January 2025) and v2 (January 2026, the Nature text with supplement).
3. D. Guo et al. [*DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning*](https://www.nature.com/articles/s41586-025-09422-z). Nature 645, 633-638 (2025).
4. DeepSeek-AI. [*DeepSeek-V3.2*](https://arxiv.org/abs/2512.02556). 2025. §3 on scaling GRPO and agentic tasks.
5. DeepSeek-AI. [*DeepSeek-V4*](https://arxiv.org/abs/2606.19348). 2026. §5 on specialists and on-policy distillation.
6. DeepSeek-AI. [*DeepSeek-V4.1-Flash*](https://arxiv.org/abs/2609.19969). 2026. §5.1.
7. Z. Shao et al. [*DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning*](https://arxiv.org/abs/2511.22570). 2025.
8. J. Schulman et al. [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347). 2017.
9. J. Schulman. [*Approximating KL Divergence*](http://joschu.net/blog/kl-approx.html). 2020. The k3 estimator.
10. Model cards: [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528), [DeepSeek-V3.1](https://huggingface.co/deepseek-ai/DeepSeek-V3.1), [DeepSeek-V3.2-Speciale](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Speciale).
11. Our scripts: `code/deepseek/part4_grpo.py` (GRPO by hand and in PyTorch), `figs_part4.py`, `shots_part4.py`.
