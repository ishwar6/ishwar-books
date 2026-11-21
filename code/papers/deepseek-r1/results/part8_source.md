---
title: "Train a Reasoner from Scratch: R1-Zero's Recipe in 330 Lines"
description: "We write the whole DeepSeek-R1-Zero method in a small Python package: the template, the two reward rules and GRPO. We check it against TRL and train a 0.5B model on a laptop."
part: 8
covers: "§2.2 (the whole R1-Zero recipe)"
date: 2026-11-21
tags: [deepseek, grpo, reinforcement-learning, from-scratch, pytorch, trl]
---

Parts 2 and 3 took R1-Zero's recipe apart, one equation and one rule at a time. Here we put it back together in about 330 lines of Python. The code takes a model, asks it maths questions, and makes it better at them with only two rule checks. Is the final answer right? Is it in the requested shape?

The proof has three steps. First, our loss gives the same number as the loss code of TRL, a widely used library, on the same tensors. Second, our accuracy rule gives the same verdicts as math_verify, a library built to compare maths answers. Our format rule agrees with an independent second check. Third, we run the recipe on a laptop and report what happened, including what did not.

## The recipe in one paragraph {§2.2}

> [!PAPER] · §2.2 · page 5
> [![Section 2.2 of the paper: the authors explore the potential of LLMs to develop reasoning capabilities without any supervised data, focusing on their self-evolution through a pure reinforcement learning process](/img/papers/deepseek-r1/p8-pure-rl.png)](/img/papers/deepseek-r1/p8-pure-rl.png)
>
> **What it says:** "we explore the potential of LLMs to develop reasoning capabilities without any supervised data, focusing on their self-evolution through a pure reinforcement learning process."
>
> **What it means:**
>
> No one shows R1-Zero a worked solution. It gets a question and writes an answer. Then it learns only from two checks: was the final answer right, and was it in the requested shape? Over thousands of rounds, answers that ended well become more likely and answers that ended badly become less likely.
>
> That loop has four parts, and each part is one file of our package. A template turns a question into a prompt. A sampler writes several answers. Two rules score them. GRPO turns the scores into a change of the weights. Everything else is bookkeeping.

The package lives in [code/papers/deepseek-r1/minir1](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/deepseek-r1/minir1). It uses PyTorch and the `transformers` library to load the model and to sample text. Every step of the method itself is our own code.

{{FIG:p8_map|The minir1 package as one loop. A question goes through the template, the sampler writes G answers, the rules score them, GRPO turns the scores into a loss, and train.py repeats this. eval.py measures held-out questions before and after.}}

| File | Lines | What it holds | Paper |
|---|---|---|---|
| `template.py` | @@COUNT minir1/template.py@@ | The prompt of Table 1 | §2.2.3 |
| `rewards.py` | @@COUNT minir1/rewards.py@@ | The accuracy rule and the format rule | §2.2.2 |
| `grpo.py` | @@COUNT minir1/grpo.py@@ | Sampling, log-probabilities, Eq. 3, Eq. 1 and Eq. 2 | §2.2.1 |
| `train.py` | @@COUNT minir1/train.py@@ | The data, the loop and the log | §2.2 |
| `eval.py` | @@COUNT minir1/eval.py@@ | Held-out questions, scored with the same rules | §2.2.4 |

The page walks through the files in the order a question meets them. For each piece you see the lines of the paper, the code, and a check.

## The template {§2.2.3}

{{FIG:p8_map_template|You are here: template.py turns a question into the text the model reads.}}

> [!PAPER] · Table 1 · page 6
> [![Table 1 of the paper, the template for DeepSeek-R1-Zero: a conversation between User and Assistant, the reasoning process and answer enclosed within think and answer tags, then User: prompt. Assistant:](/img/papers/deepseek-r1/p8-table1.png)](/img/papers/deepseek-r1/p8-table1.png)
>
> **What it says:** "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively", ending with "User: prompt. Assistant:".
>
> **What it means:**
>
> This one paragraph is the whole instruction. The model reads it, then the question, then "Assistant:", and it continues the text. A base model was trained only to continue text, so the template frames the question as a conversation that it can continue.
>
> The two tag pairs are what the rules check later. The thinking goes between the think tags, and the final answer between the answer tags. [Part 3](part-3-rewards-and-the-template.md) explains the template in full.

> [!PAPER] · §2.2.3 · page 6
> [![Section 2.2.3: the authors intentionally limit their constraints to this structural format, to accurately observe the model's natural progression during the RL process](/img/papers/deepseek-r1/p8-limit.png)](/img/papers/deepseek-r1/p8-limit.png)
>
> **What it says:** "We intentionally limit our constraints to this structural format", so that they can "accurately observe the model's natural progression during the RL process."
>
> **What it means:**
>
> The template says where the thinking goes, not how to think. It does not ask the model to check its work or to try another method. Any such habit that appears later was learned from the reward, not copied from the prompt.
>
> We keep this rule. Our code changes the content of the template in no way.

Here is the whole file:

@@FILE minir1/template.py@@

The constant is Table 1 word for word. We made one addition, and the code says so: the prompt ends with the opening tag ` <think>`. The model then starts inside the thinking block.

> [!DEFINITION] Prefill
> Text that we put at the start of the model's answer before it writes anything. The model continues from there. Here the prefill is the opening tag ` <think>`, so every answer starts in the thinking block.

Why the prefill? Without it, our 0.5B models rarely passed the format check. We sampled 4 answers to each of 24 training questions, at temperature 1, with at most 300 new tokens:

@@LOG r1_part8_probe samples@@

With the plain template, the base model passed the format check on 2 of 96 answers (0.021) and the instruction-tuned model on none. With the prefill, both start to pass it:

@@LOG r1_part8_probe_prefill samples@@

The base model now passes on 11 of 96 answers (0.115) and the Instruct model on 41 of 96 (0.427). GRPO's reward signal comes only from differences inside a group. If every answer in a group gets the same reward, the rewards push nothing up or down (we show why in the GRPO section). A start of 2% leaves almost every group with equal rewards. A start of 10% to 40% gives many groups a mix of good and bad answers.

The prefill does not tell the model how to reason, so it keeps the spirit of §2.2.3. The rewards still see the full text: `full_completion` puts the tag back in front of what the model wrote.

## The rewards {§2.2.2}

{{FIG:p8_map_rewards|You are here: rewards.py turns each answer into a number.}}

> [!PAPER] · §2.2.2 · page 6
> [![Section 2.2.2, accuracy rewards: the accuracy reward model evaluates whether the response is correct; for math problems the model must provide the final answer in a specified format, enabling reliable rule-based verification of correctness](/img/papers/deepseek-r1/p8-accuracy.png)](/img/papers/deepseek-r1/p8-accuracy.png)
>
> **What it says:** "The accuracy reward model evaluates whether the response is correct." The model must "provide the final answer in a specified format", which enables "rule-based verification of correctness."
>
> **What it means:**
>
> A maths question has one right answer, so a program can check it. The trick is to know where the answer is. A free text answer can hold ten numbers, and the last one is not always the result.
>
> So the paper asks for the answer in a fixed place, for example inside a box. Our place is the answer tags. The rule reads the text between them and compares it with the known answer. No model judges anything.

> [!PAPER] · §2.2.2 · page 6
> [![Section 2.2.2, format rewards: a format reward model enforces the model to put its thinking process between think and close think tags](/img/papers/deepseek-r1/p8-format.png)](/img/papers/deepseek-r1/p8-format.png)
>
> **What it says:** a format reward "enforces the model to put its thinking process between '<think>' and '</think>' tags."
>
> **What it means:**
>
> The second rule ignores whether the answer is right. It checks only the shape of the text: thinking in the think tags, then the answer in the answer tags. A regular expression is enough for that.
>
> The format rule is the easier one to earn, so it is often learned first. It also makes the accuracy rule usable, because the accuracy rule needs the answer tags to find the answer.

> [!PAPER] · §2.2.2 · page 6
> [![Section 2.2.2: the authors do not apply the outcome or process neural reward model, because the neural reward model may suffer from reward hacking in large-scale reinforcement learning](/img/papers/deepseek-r1/p8-no-neural.png)](/img/papers/deepseek-r1/p8-no-neural.png)
>
> **What it says:** "We do not apply the outcome or process neural reward model", because "the neural reward model may suffer from reward hacking".
>
> **What it means:**
>
> A neural reward model is a network that guesses how good an answer is. A policy that trains against it for long enough finds answers that fool the guess. That is reward hacking. A rule has no learned guess to fool, but it can still have holes. A careless number parser can pay for a wrong answer, and a loose format check can pay for an empty block. [Part 3](part-3-rewards-and-the-template.md) shows such holes, and our first version of this package had some of them (see below).
>
> This choice also keeps our package small. The whole reward is a short file of plain Python.

> [!DEFINITION] Regular expression
> A small pattern language for text. `<answer>(.*?)</answer>` means "the opening tag, then the shortest run of any characters, then the closing tag". Python's `re` module finds every place where the pattern fits.

Here is the whole file:

@@FILE minir1/rewards.py@@

`extract_answer` takes the last answer pair, so an answer that changes its mind is graded on its final word.

`to_number` reads the final value of the answer. If there is a `\boxed{...}`, it takes what is inside the last box, counting braces. It keeps only what comes after the last `=`, so "3 + 4 = 7" gives 7. It removes dollar signs and the commas of thousands, so a dollar price of 1,250.00 written with its sign gives 1250. A fraction counts as a fraction and `1e3` as a thousand. A sum or a power that was never worked out, such as "10 - 2", gives no number at all, and so no reward.

`format_reward` is the tag parser of [Part 3](part-3-rewards-and-the-template.md). It finds the four tags and asks for exactly `<think>`, `</think>`, `<answer>`, `</answer>`, once each and in this order. Both blocks must hold some text, and nothing may stand before, between or after them.

> [!NOTE] Our first version had two holes
> The first version of this file used the last number of the answer, and one regular expression for the format. Both paid for things they should not.
>
> `1/2` earned the accuracy point against a gold answer of 2, because the last number in "1/2" is 2. `1e3` earned it against a gold answer of 3. And `<think> </think><answer> </answer>` earned the format point with two empty blocks, as did an answer with two think blocks. Saved training answers had such shapes and got the point. The run on this page uses the fixed rules.

Here are the fixed rules on those hand-made cases:

@@LOG r1_part8_check ^accuracy, answer|^format [01]:@@

Version 1 of the paper names the two rewards but does not say how they combine. The later, longer version adds them with the same weight (its Eq. 4). We do the same, so one answer earns 0, 1 or 2.

**Method 2, with libraries.** The math_verify library parses two maths expressions and checks if they are equal. It knows fractions, LaTeX and units.

We ran it and our accuracy rule on every answer of the held-out evaluation. Both read the text inside the last answer tags and compare it with the GSM8K answer. For the format rule we wrote a second, independent check. It is one regular expression: neither block may contain a tag, and each block must hold a non-space character.

@@LOG r1_part8_train_instruct agree@@

Both checks agree on all 400 answers of the Instruct run (200 before training, 200 after). They also agree on the 400 answers of the base run.

Read these numbers for what they are. The math_verify check covers only the accuracy rule, and both sides read the same text: what `extract_answer` found inside the last answer tags. It does not test the extraction itself, and it says nothing about format. The format check is the separate second line. GSM8K answers are also whole numbers, which keeps the comparison easy: the fraction and power cases above barely occur here. For MATH or AIME problems, use math_verify, or a parser like it, as the accuracy rule.

## Group Relative Policy Optimization {§2.2.1}

Now the heart of the method. GRPO comes from an earlier DeepSeek paper on maths, and [Part 2](part-2-grpo.md) explains each equation term by term. Here we write each equation as one short function.

> [!PAPER] · §2.2.1 · page 5
> [![Section 2.2.1: GRPO foregoes the critic model that is typically the same size as the policy model, and estimates the baseline from group scores instead; for each question it samples a group of outputs from the old policy](/img/papers/deepseek-r1/p8-grpo-intro.png)](/img/papers/deepseek-r1/p8-grpo-intro.png)
>
> **What it says:** GRPO "foregoes the critic model" and "estimates the baseline from group scores instead". For each question it "samples a group of outputs" from the old policy.
>
> **What it means:**
>
> To learn, the model must know whether an answer was better or worse than expected. PPO, the usual method, trains a second network of the same size, the critic, to predict what to expect. GRPO drops the critic. It asks the same question G times and uses the group's own average as the expectation.
>
> Think of a teacher who grades on a curve. Your score counts only against the scores of your classmates on the same test. A hard test lowers everyone's score, but the best answers still stand out.

### Step 1: sample a group

{{FIG:p8_map_grpo_sample|You are here: sample_group writes G answers to each question with the current model.}}

> [!DEFINITION] Policy
> In reinforcement learning, the policy is the thing that chooses actions. Here the policy is the language model: each token it writes is an action. $$\pi_\theta(o \mid q)$$ is the probability that the model with weights $$\theta$$ writes the answer $$o$$ to the question $$q$$.

@@FUNC minir1/grpo.py sample_group@@

The function repeats each prompt G times, so the G answers to one question sit side by side. Padding on the left lines up the ends of the prompts, which is where generation starts. We sample at temperature 1 with no top-p or top-k filter, so the answers come from the model's real distribution. The later version of the paper also uses temperature 1.

Generation stops at the string `</answer>`, or at an end-of-text token, or at the length limit. When one answer stops early, `generate` fills the rest of its row with padding. For Qwen the padding token is the same token as end-of-text, so the ids alone cannot tell a sampled end from a filler.

`completion_lengths` decides from how the row stopped. An answer that stopped on `</answer>` ends with the token that completes the string. An answer that stopped on a sampled end token keeps that token, because the model chose it. An answer that hit the limit keeps everything.

> [!NOTE] A bug we fixed
> Our first version marked everything up to and including the first end-or-padding token. After a stop on `</answer>`, that token is padding the model never chose. It still entered the loss, the KL and the length. The fixed mask, on three hand-made rows:

@@LOG r1_part8_check ^mask,@@

The mask is 1 for the tokens the model really wrote and 0 after. Every later step uses it, so padding never counts. The evaluation uses the same function for its lengths.

### Step 2: the log-probability of each token

To change the probability of an answer, we need the probability of each of its tokens. One forward pass over prompt plus answer gives them all.

@@FUNC minir1/grpo.py per_token_logps@@

The logits at position $$t$$ predict the token at position $$t+1$$. So for $$T$$ answer tokens we keep the last $$T+1$$ positions and drop the very last one. `gather` picks the logit of the token that was actually written. Subtracting `logsumexp` over the vocabulary turns that logit into a log-probability, without building the full table of 151,936 probabilities per position a second time.

The paper writes $$\pi_\theta(o_i \mid q)$$ for a whole answer. The probability of an answer is the product of the probabilities of its tokens, so its logarithm is a sum over tokens. The code keeps one number per token, because Eq. 1 works on each token, as the next box shows.

### Step 3: the group advantage (Eq. 3)

{{FIG:p8_map_grpo_update|You are here: the update half of grpo.py, from rewards to a loss.}}

> [!PAPER] · §2.2.1 · page 5
> [![Equation 3 of the paper: the advantage A_i is the reward r_i minus the mean of the group's rewards, divided by the standard deviation of the group's rewards](/img/papers/deepseek-r1/p8-eq3.png)](/img/papers/deepseek-r1/p8-eq3.png)
>
> **What it says:** $$A_i$$ is "the advantage, computed using a group of rewards": $$A_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})}$$.
>
> **What it means:**
>
> Subtract the group average, and answers above average get a positive number while answers below get a negative one. Divide by the spread, and the numbers have the same size whether the rewards differ by 0.1 or by 2.
>
> If all G answers get the same reward, every advantage is 0. Then the group gives no reward signal, right or wrong: nothing pushes its answers up or down. Only the KL term of the loss still acts on it. This is why a good group size and questions of the right difficulty matter so much.

$$
A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}
$$

Read it left to right. The advantage of answer $$i$$ is its reward $$r_i$$, minus the mean reward of its group. That difference is divided by the standard deviation of the group's rewards. The paper does not say which standard deviation. We use the sample one (divide by $$G-1$$), as TRL does. We also add $$10^{-4}$$ below the line, so a group of equal rewards does not divide by zero.

@@FUNC minir1/grpo.py group_advantages@@

A worked example with $$G = 4$$. The rewards are 2 (right and well formed), 0, 1 (well formed, wrong) and 0. The mean is 0.75.

The deviations are 1.25, -0.75, 0.25 and -0.75. Their squares add up to 2.75, and $$\sqrt{2.75/3} = 0.957$$. So the advantages are $$1.25/0.957 = 1.31$$, then -0.78, 0.26 and -0.78. Our function and Python's `statistics` module agree:

@@LOG r1_part8_check ^Eq. 3|mean|advantages \(|same reward@@

The last line is the case of equal rewards: four zero advantages. Such a group gives no reward signal. It is not completely silent, though. The KL term of the next steps still pulls each token toward the reference model. With $$A = 0$$, $$\pi_\theta = 0.6$$, $$\pi_{ref} = 0.5$$ and $$\beta = 0.04$$, our loss still has a gradient:

@@LOG r1_part8_check ^A = 0@@

That is $$\beta(1 - \pi_{ref}/\pi_\theta) = 0.04 \times (1 - 0.5/0.6)$$: a small pull back toward the reference.

> [!PAPER] Shao et al. (2024), DeepSeekMath · §4.1.2 · page 14
> [![DeepSeekMath, section 4.1.2: outcome supervision sets the advantages of all tokens in the output as the normalized reward](/img/papers/deepseek-r1/p8-math-outcome.png)](/img/papers/deepseek-r1/p8-math-outcome.png)
>
> **What it says:** outcome supervision "sets the advantages $$\hat{A}_{i,t}$$ of all tokens in the output as the normalized reward".
>
> **What it means:**
>
> The reward comes once, at the end of an answer. GRPO does not try to say which token earned it. Every token of answer $$i$$ gets the same advantage $$A_i$$.
>
> That is crude but fair on average. A token that appears in many good answers is pushed up many times. A token that appears in good and bad answers alike is pushed up and down, and those pushes cancel.

### Step 4: the clipped ratio (Eq. 1)

> [!PAPER] · §2.2.1 · page 5
> [![Equation 1 of the paper, the GRPO objective: the mean over the G outputs of the minimum of the probability ratio times the advantage and the clipped ratio times the advantage, minus beta times the KL divergence to the reference policy](/img/papers/deepseek-r1/p8-eq1.png)](/img/papers/deepseek-r1/p8-eq1.png)
>
> **What it says:** maximise $$\frac{1}{G}\sum_{i=1}^{G}\left(\min\left(\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{old}}(o_i \mid q)} A_i, \text{clip}\left(\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{old}}(o_i \mid q)}, 1-\varepsilon, 1+\varepsilon\right) A_i\right) - \beta \mathbb{D}_{KL}(\pi_\theta \| \pi_{ref})\right)$$.
>
> **What it means:**
>
> The ratio compares the model now with the model that wrote the answers. A ratio of 1.3 means the answer has become 30% more likely since it was sampled. Multiplying by the advantage makes good answers more likely and bad ones less likely.
>
> The clip works like a speed limit on the incentive. Once the ratio has moved more than $$\varepsilon$$ in the right direction, the min picks the clipped value. That value is flat, so its gradient is zero, and this token stops pulling the model further.
>
> It is not a hard bound on how much the model can change. Other tokens, the KL term and the optimiser's momentum still move the weights, and a moved weight changes the probability of a clipped token too. The clip removes one reason to move far; it does not forbid it.

> [!PAPER] Shao et al. (2024), DeepSeekMath · §4.1 · page 13
> [![DeepSeekMath's Equation 3: the same GRPO objective written per token, with a mean over the tokens of each output, 1 over the length of o_i times the sum over t](/img/papers/deepseek-r1/p8-math-eq3.png)](/img/papers/deepseek-r1/p8-math-eq3.png)
>
> **What it says:** the same objective, per token: $$\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min\left[\frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} \mid q, o_{i,<t})}\hat{A}_{i,t}, \ldots\right]$$.
>
> **What it means:**
>
> The R1 paper writes the ratio for a whole answer. The paper GRPO comes from writes it for each token, then averages over the $$|o_i|$$ tokens of the answer. Code follows this per-token form, and so do TRL and our package.
>
> The per-token form is also safer with numbers. The probability of a 300-token answer is a product of 300 small numbers. A ratio of two such products can be huge or tiny, while a ratio for one token stays close to 1.

$$
L = -\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left[\min\left(\rho_{i,t}A_i,\ \text{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon)A_i\right) - \beta\, D_{i,t}\right], \qquad \rho_{i,t} = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} \mid q, o_{i,<t})}
$$

This is the loss our code minimises. $$\rho_{i,t}$$ is the ratio for token $$t$$ of answer $$i$$. $$D_{i,t}$$ is the KL term of the next step, for the same token. The minus sign in front turns "maximise the objective" into "minimise the loss", which is what an optimiser does.

Four tokens show what the min and the clip do, with $$\varepsilon = 0.2$$:

@@LOG r1_part8_check ^Eq. 1 term@@

With a positive advantage, a ratio of 1.5 is cut to 1.2: the token has already gained enough, so this batch stops pushing it. A ratio of 0.7 is kept: the token lost ground, and it may climb back. With a negative advantage the mirror holds. A ratio of 0.5 counts as 0.8, flat, so the push down stops. A ratio of 1.3 keeps its full value: the bad token became more likely, and the loss keeps pushing it down.

So a token counts as clipped only in two cases: ratio below $$1 - \varepsilon$$ with a negative advantage, or above $$1 + \varepsilon$$ with a positive one. Only then does the min pick the flat branch. Our loss counts clipped tokens this way, the same way as TRL's `clip_ratio/region_mean`:

@@LOG r1_part8_check clipped\? A@@

The ratios 0.7 and 1.3 lie outside the interval, but they are not clipped. Our first version counted every ratio outside the interval, so it reported these two as clipped too. That made the clipped share in our first logs too high.

> [!DEFINITION] Old policy and mini-batch
> The old policy $$\pi_{\theta_{old}}$$ is the model at the moment it wrote the answers. A mini-batch is a slice of those answers used for one optimiser step. When one set of answers is split into several mini-batches, the model changes after the first step. The later steps then see a ratio different from 1.

The ratio is not 1 only when the model changed since sampling. With one optimiser step per batch of answers, it is always exactly 1 and the clip never acts. The later version of the paper splits each batch into 16 mini-batches. We split ours into 2, so the second half of each step meets a real ratio.

### Step 5: the KL leash (Eq. 2)

> [!PAPER] · §2.2.1 · page 5
> [![Equation 2 of the paper: the KL divergence term equals pi_ref over pi_theta, minus the log of pi_ref over pi_theta, minus 1](/img/papers/deepseek-r1/p8-eq2.png)](/img/papers/deepseek-r1/p8-eq2.png)
>
> **What it says:** $$\mathbb{D}_{KL}(\pi_\theta \| \pi_{ref}) = \frac{\pi_{ref}(o_i \mid q)}{\pi_\theta(o_i \mid q)} - \log\frac{\pi_{ref}(o_i \mid q)}{\pi_\theta(o_i \mid q)} - 1$$.
>
> **What it means:**
>
> The reference model is a frozen copy of the model from before training. This term grows when the model drifts away from it, and the loss pays $$\beta$$ times that amount. It is a leash: the model can move, but every step away costs a little.
>
> The form $$x - \ln x - 1$$ is never negative and is 0 only at $$x = 1$$. It estimates the KL divergence from samples alone, without a sum over the whole vocabulary. [Part 2](part-2-grpo.md) shows why the estimate is right on average, when the tokens are sampled from the current model $$\pi_\theta$$.
>
> In our loop that condition holds only for the first mini-batch of a step. After the first optimiser step, the model has changed, but the tokens were sampled by the old model. The KL we log for later mini-batches is then an estimate on old-policy samples, with no correction. It is close when the model moves little, but it is not exact. TRL computes the same thing, unless its `use_bias_correction_kl` setting is on. That setting multiplies each token's KL term by the ratio, which corrects for exactly this.

With $$x = \pi_{ref}/\pi_\theta$$ for one token:

@@LOG r1_part8_check ^Eq. 2@@

When the two models agree, the cost is 0. Moving a probability from 0.5 to 0.6 costs 0.016. Moving it from 0.5 to 0.9 costs 0.143, nine times more for four times the move. Small drifts are cheap and big ones are expensive.

### The whole loss, in one function

@@FUNC minir1/grpo.py grpo_loss@@

Each line is one piece of the equations. `ratio` is $$\rho_{i,t}$$. `torch.min(ratio * A, clipped * A)` is the min of Eq. 1. `kl` is Eq. 2 per token. Then the mask removes padding, the inner mean is the $$1/|o_i|$$ sum, and the outer mean is the $$1/G$$ sum.

For the log, the function returns sums, not means: the total KL, the number of clipped tokens and the number of real tokens. The loop adds them over all slices of a step and divides once. Slices hold different numbers of tokens, so an average of slice means would weight them wrongly, and our first version did exactly that.

**Method 2, with TRL.** TRL's `GRPOTrainer` has a method, `_compute_loss`, that takes the same tensors: prompt and answer tokens, masks, advantages, old and reference log-probabilities. We built a trainer around the same model and called that method directly, so this is TRL's own code.

The batch is 8 real answers of Qwen2.5-0.5B-Instruct to 2 GSM8K questions. Real rewards of 8 short answers are often all equal, and then every advantage is 0. Fixed rewards turn every term on. In training, the three policies differ, so we took two optimiser steps first. The reference is the start, the old policy comes after one step, and the current one after two.

@@LOG r1_part8_check batch:|fixed rewards|advantages:|^epsilon|tokens clipped, ours|mean KL per token|loss|difference@@

[![Terminal output of r1_part8_check.py: the equations on small numbers, then our loss against TRL's loss](/img/papers/deepseek-r1/r1_part8_check-run.png)](/img/papers/deepseek-r1/r1_part8_check-run.png)

Read the first two settings, which use the KL of Eq. 2. Our share of clipped tokens equals TRL's `clip_ratio/region_mean`: 0.0202 at $$\varepsilon = 0.2$$ and 0.1254 at $$\varepsilon = 0.02$$. Our mean KL per token equals TRL's `kl`: 0.273503.

The two losses differ by $$3.81 \times 10^{-5}$$ on a loss of about 64.7, a relative difference of $$5.9 \times 10^{-7}$$. Where does that come from? Not from the formula.

TRL computes the log-probabilities itself, through its own path to the output layer, and they differ from ours by at most $$1.3 \times 10^{-6}$$: float32 rounding. The ratio is an exponential of a log-probability difference, and a few ratios here are huge, so a tiny rounding difference becomes a visible one. The last line of each setting settles it. Fed with the log-probabilities TRL computed, our `grpo_loss` gives exactly TRL's loss: the difference is 0.00e+00.

The loss of about 64.7 looks large, and it is real. A few tokens were sampled with a very low probability, and two optimiser steps made them much more likely: their ratio is huge. Where the advantage is negative, the min of Eq. 1 keeps the huge value, as in the last case of the four-token example. Those few tokens dominate the mean.

The third line is a surprise worth knowing. TRL 1.18 has a setting, `use_bias_correction_kl`, that is on by default. It multiplies each token's KL term by the ratio, a change that comes from the later DeepSeek-V3.2 paper. With it on, TRL's loss is 65.019 and ours is 64.698, and TRL's logged KL is 9.148457 instead of 0.273503. That is not Eq. 2 of the R1 paper, so for R1 we turn it off. A library default can quietly change the method you think you are running.

## The loop {§2.2}

{{FIG:p8_map_train|You are here: train.py repeats sample, score, compare and update, and writes one log line per step.}}

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1, later version · §2.1 · page 3
> [![The later version of the paper: to train DeepSeek-R1-Zero, the learning rate is 3e-6, the KL coefficient 0.001, the sampling temperature 1; 16 outputs per question; each rollout of 8,192 outputs is randomly split into 16 mini-batches](/img/papers/deepseek-r1/p8-long-hparams.png)](/img/papers/deepseek-r1/p8-long-hparams.png)
>
> **What it says:** "learning rate to 3e-6, the KL coefficient to 0.001, and the sampling temperature to 1"; "we sample 16 outputs" for each question, with a maximum length of "32,768 tokens before the 8.2k step and 65,536 tokens afterward"; "Every 400 steps, we replace the reference model"; each rollout of 8,192 outputs is "randomly split into 16 mini-batches".
>
> **What it means:**
>
> Version 1 of the paper gives no settings for GRPO. The later, longer version does, and these are the numbers above. R1-Zero trained for 10,400 steps. Answers could be up to 32,768 tokens long until step 8,200 and up to 65,536 tokens after it. One training step is 32 questions with 16 answers each, 512 answers. One rollout writes 8,192 answers, 16 steps' worth, and they are used in 16 mini-batches of 512, one optimiser step each.
>
> We copy what fits on a laptop: the learning rate of 3e-6, temperature 1, and the split into mini-batches. Everything else is smaller. One rollout of ours is 4 questions with 8 answers each, at most 300 new tokens, split into 2 mini-batches of 16. Our KL weight $$\beta$$ is 0.04, TRL's old default and 40 times the paper's value. Our reference model stays frozen for the whole run. That is our choice for a short run; the later paper replaces its reference with the current policy every 400 steps.

The settings of our run, as the script printed them:

@@LOG r1_part8_train_instruct ^config@@

Here is the loop. It is the five steps above, in order:

@@FUNC minir1/train.py train@@

Before the first step, `copy.deepcopy` makes the reference model, and `requires_grad_(False)` freezes it. Each step then does the following:

1. Pick 4 random training questions and sample 8 answers to each with the current model (one rollout).
2. Score the 32 answers with the two rules, and turn the scores into advantages, group by group.
3. Compute the log-probabilities of the old policy and of the reference, once, without gradients. They do not change during the step.
4. Shuffle the 32 answers into 2 mini-batches of 16. For each, compute the loss in slices of 8 to save memory and add up the gradients. Then clip their total size to 1 and take one AdamW step.
5. Write one line of JSON with the mean rewards, the mean length, the KL, the share of clipped tokens and the memory the process uses. The KL and the clipped share are averaged over every real token of the step. Then free the memory the step left behind.

Every 25 steps it also saves the 8 answers to the first question, so we can read what the model writes.

> [!DEFINITION] Gradient clipping
> `clip_grad_norm_` scales all gradients down together when their total size is above a limit (here 1). One unusual batch then cannot make a huge step. It is a different thing from the clip of Eq. 1, which acts on the ratio of each token.

## The evaluation

{{FIG:p8_map_eval|You are here: eval.py answers 200 held-out questions and scores them with the same rules.}}

@@FILE minir1/eval.py@@

The evaluation uses the first 200 questions of the GSM8K test split. The model never trains on them. It answers each one greedily, the most likely token every time, with the same limit of 300 new tokens. We set `repetition_penalty=1.0` on purpose.

The Instruct model's own generation settings add a repetition penalty of 1.1, which would quietly change "the most likely token" into "the most likely token after a penalty". Our first evaluation had that penalty; this one does not. The paper reports pass@1 from several samples at temperature 0.6. One greedy answer is cheaper, and it gives the same result every time.

## The run: first try, the base model

R1-Zero starts from a base model, so we did too: Qwen2.5-0.5B, a model of 0.5 billion parameters that was never tuned to follow instructions. Before training, it answered 10% of the 200 held-out questions correctly, and 35.5% of its greedy answers passed the format parser:

@@LOG r1_part8_train_base ^BEFORE|^before@@

We gave this run 20 minutes, to keep most of our time budget for the second model. It made 35 steps. In windows of 10 steps:

@@BLOCK r1_part8_stats 6 ^== base model@@

{{FIG:p8_reward_base|The base model run, 10-step averages. The format reward climbs to about 0.9 within 15 steps. The accuracy reward stays near 0.04 the whole time.}}

{{FIG:p8_length_base|Answer length in the base model run, 10-step averages. Answers get shorter, not longer: they drop from about 140 tokens at step 1 to between about 30 and 60 tokens.}}

The format reward was 0.762 in the first 10 steps and 0.944 in the last 5. The accuracy reward stayed between 0.031 and 0.050. And the answers got much shorter, from 139.9 tokens at step 1 to 34.1 tokens on average in steps 31 to 35. The held-out test after training shows the same thing on new questions:

@@LOG r1_part8_train_base ^AFTER|^after@@

Format went from 35.5% to 98.5%, but accuracy fell from 10% to 4%, and the mean answer shrank from 98.9 to 37.3 tokens. On this model and in this short run, the rules taught the shape and nothing else.

The samples show what the model writes. Here are three answers at step 2 and three at step 25:

@@BLOCK r1_part8_samples 4 ^== base model, training step (2|25),@@

At step 2 one answer copies "reasoning process here", the placeholder text of Table 1, into the think tags. The other two put a few unrelated words in the think tags and an answer with no number at all. By step 25 the answers are short and tidy: a sentence of thinking, then an expression or a number. The format rule cannot tell real thinking from a copied placeholder, so a short well-formed answer is the cheapest way to earn one point.

This is the problem behind "equal-reward groups" in the table. Suppose all 8 answers to a question have the right shape and all 8 are wrong. Then they all get a reward of 1, and every advantage is 0. The group gives no reward signal; only the KL term still acts on it. In steps 31 to 35, half of the groups were like that. The model had learned the shape, and it almost never solved a problem, so there was little left for the rewards to teach.

The table has one more surprise: a KL of 24.257 at step 1, against a median of 0.204. After a single optimiser step, the second mini-batch already met a few tokens whose probability had fallen far below the reference. We do not know which tokens they were. The KL was back at 0.143 by step 2.

> [!PAPER] · §4.1 · page 15
> [![Section 4.1 of the paper: smaller models relying on the large-scale RL require enormous computational power and may not even achieve the performance of distillation; advancing further may require more powerful base models and larger-scale reinforcement learning](/img/papers/deepseek-r1/p8-small.png)](/img/papers/deepseek-r1/p8-small.png)
>
> **What it says:** "smaller models relying on the large-scale RL mentioned in this paper require enormous computational power and may not even achieve the performance of distillation". Going further "may still require more powerful base models and larger-scale reinforcement learning."
>
> **What it means:**
>
> The paper compares two ways to make a small model reason: RL on the small model itself, and distillation from a big one. For small models, it says, RL costs a lot and may still fall short. It does not say why, and it does not claim that RL can never find new behaviour.
>
> One plausible reason is easy to see in code. GRPO learns from the answers that score better than their group. If a model almost never writes a right answer, there is almost nothing to push up. Whether RL mostly sharpens what a model can already do, or also finds new things, is an open question ([Part 4](part-4-r1-zero-and-the-aha-moment.md) weighs the evidence). [Part 6](part-6-distillation.md) shows the paper's own test on a 32B model.

Our plan was to try the base model first. If it earned almost no accuracy reward, we would switch to the instruction-tuned model. It earned almost none, and its held-out accuracy fell.

## The run: second try, the Instruct model

Qwen2.5-0.5B-Instruct has the same size, but it was fine-tuned to follow instructions and to explain its steps. That is a long way from R1-Zero's pure start, and we say so. It is closer to DeepSeek-R1, which starts RL from a fine-tuned model ([Part 5](part-5-the-r1-pipeline.md)).

Before training, its greedy answers were right 16.5% of the time, and 55% had the right shape:

@@LOG r1_part8_train_instruct ^BEFORE|^before@@

Then 122 steps of our GRPO, the whole 80-minute budget. Here are the first steps and the last steps of the log:

@@LOG r1_part8_train_instruct ^step +([1-5]|11[89]|12[0-2]) @@

And the 10-step windows:

@@BLOCK r1_part8_stats 14 ^== instruct model@@

[![Terminal output of the Instruct run: evaluation before, 122 GRPO steps, evaluation after](/img/papers/deepseek-r1/r1_part8_train_instruct-run.png)](/img/papers/deepseek-r1/r1_part8_train_instruct-run.png)

{{FIG:p8_reward|The Instruct model run, 10-step averages. The format reward rises from below 0.5 to above 0.9 within about 15 steps. The accuracy reward rises from about 0.05 to about 0.3, and it moves a lot from step to step.}}

Three things happened. The format was learned first: the format reward was 0.656 in the first 10 steps and 0.966 in the next 10. The accuracy reward rose more slowly. It was 0.041 in the first 10 steps, 0.338 in steps 41 to 50, and 0.359 in steps 111 to 120. On the way it dipped as low as 0.122.

With only 4 questions per step, one step can show 0.000 and two steps later 0.250, so read the trend, not single steps. And the answers did not collapse. They dipped to about 79 tokens on average in steps 11 to 20, then grew back to about 140.

The held-out test confirms the trend on questions the model never trained on:

@@LOG r1_part8_train_instruct ^AFTER|^after@@

{{FIG:p8_before_after|The 200 held-out GSM8K questions, greedy answers, before and after 122 steps of GRPO. Accuracy goes from 0.165 to 0.390, and the share of well-formed answers from 0.550 to 0.980.}}

Accuracy went from 16.5% to 39.0%: 33 right answers before, 78 after. 196 of the 200 answers now have the right shape. On 200 questions a gap of 22.5 points is far larger than the noise of a single evaluation. But it is one run with one seed, so we do not claim a precise size.

> [!NOTE] Numbers from our first run
> An earlier version of this page reported a shorter run made with the bugs listed in the notes above. They were the extra padding token in the mask, the loose format and number rules, and the wrong clip count and KL average. The evaluation also had a repetition penalty of 1.1. Even the score of the untrained model changed, because the evaluation itself changed. Those numbers are not comparable with the ones here, and this page no longer uses them.

Here are four held-out questions, before and after. The text is cut at 500 characters:

@@BLOCK r1_part8_samples 3 ^== held-out question@@

Training did not fix every answer, and it broke some. Question 2 went from wrong to right: the trained model now works out the cost, the new value and the profit step by step. Question 0 went the other way: right before, wrong after. Question 3 is wrong both times, with the same mistake (it multiplies only two of the three numbers). A held-out accuracy is a count of such wins and losses, and on these 200 questions the wins were many more.

### The KL stayed small

The windows table shows a mean KL between 0.088 and 0.217 in every window. The largest KL of a single step was at step 1:

@@LOG r1_part8_stats ^instruct model, largest KL@@

Eq. 2 shows how one step could jump much higher. For one token, the term is $$x - \ln x - 1$$ with $$x = \pi_{ref}/\pi_\theta$$. If the model has made a token very unlikely that the reference still likes, $$x$$ is huge, and the term grows like $$x$$ itself. A token with $$\pi_\theta$$ 100 times smaller than $$\pi_{ref}$$ costs $$100 - \ln 100 - 1 = 94.4$$. A few such tokens are enough to lift the mean of a whole step.

Our first Instruct run had such a step, with a mean KL far above every other step. The fixed Instruct run has none, but the fixed base run has one at step 1, as we saw above. So such spikes are not only a product of the old padding bug, and we did not find their cause.

### What did not happen: no Figure 3

> [!PAPER] · Figure 3 · page 8
> [![Figure 3 of the paper: the average length per response of DeepSeek-R1-Zero during training rises from a few hundred tokens to about 10,000 tokens over 8,000 steps](/img/papers/deepseek-r1/p8-fig3.png)](/img/papers/deepseek-r1/p8-fig3.png)
>
> **What it says:** "DeepSeek-R1-Zero naturally learns to solve reasoning tasks with more thinking time."
>
> **What it means:**
>
> Over about 8,000 steps, R1-Zero's answers grow from a few hundred tokens to nearly 10,000. Nobody asked for longer answers. Longer thinking simply led to more right answers, so GRPO made it more likely.
>
> This is the curve everyone wants to reproduce. [Part 4](part-4-r1-zero-and-the-aha-moment.md) reads it in detail.

> [!PAPER] · §2.2.4 · page 8
> [![Section 2.2.4: the computation ranges from generating hundreds to thousands of reasoning tokens, allowing the model to explore and refine its thought processes in greater depth](/img/papers/deepseek-r1/p8-thinking.png)](/img/papers/deepseek-r1/p8-thinking.png)
>
> **What it says:** the thinking "ranges from generating hundreds to thousands of reasoning tokens".
>
> **What it means:**
>
> The longer answers are not padding. They hold checks, second attempts and a change of plan when the first plan fails. That is the behaviour the paper calls self-evolution.
>
> Why would length grow at all? Here is a likely story. The model sometimes solves harder problems by thinking longer. The length limit leaves room for that. And training runs long enough for a slow drift to add up. The paper shows the curve; it does not test which of these conditions are needed.

{{FIG:p8_length|Answer length in the Instruct model run, 10-step averages. It dips to about 80 tokens around step 15, then climbs back to between about 125 and 150 tokens.}}

Our answers did get somewhat longer. On the held-out questions the mean went from 115 tokens before training to 160 after. During training, the 10-step mean was 115 in the first window. After a dip in steps 11 to 20, it stayed between 138 and 148 from step 81 to step 120.

That is a rise of about 40%, not the twenty-fold growth of Figure 3. We cannot tell from one run why it happened. Longer answers may simply carry more of the arithmetic written out.

Our setting is far from R1-Zero's in every way the box above lists, so we did not expect a Figure 3. Our limit is 300 new tokens, about 3% of R1-Zero's final mean length, so a long answer is cut off and earns nothing. We ran 122 steps, not 10,400. And GSM8K problems need only a few arithmetic steps. These are differences, not a tested explanation: we did not run the experiment that would separate them.

We did not see an aha moment either. The 0.5B model writes simple, linear arithmetic. It does not stop to say "Wait" and check itself. We did reproduce the first part of the story. Rules alone, with no example of good reasoning, taught the model the format in about 15 steps. They also raised its accuracy on new questions.

### What the laptop could and could not do

The run used an Apple laptop with 64 GB of memory, through PyTorch's `mps` device. The policy, the frozen reference and the optimiser state are all full float32 copies of the model.

Our first run did not get far. After about 30 steps its steps slowed down sharply, because the process kept growing until the machine started to swap. Every step writes answers of a different length, so the `mps` allocator keeps freed blocks of many sizes and the cache grows.

The fix is two lines in `free_memory`: drop the step's tensors, collect garbage and call `torch.mps.empty_cache()` after every step. The resident size that `psutil` reports does not see `mps` buffers: it stayed near 1.5 GB. So a small script, `r1_part8_memwatch.py`, read the process footprint once a minute. That is the number Activity Monitor shows:

@@LOG r1_part8_memory_instruct ^minute +(0\.0|1\.0|2\.0|40\.|77\.|78\.|79\.)@@

Over the 80 readings of the fixed run, the footprint stayed between 20 and 34 GB, with no upward trend. It moves from reading to reading because a reading can land anywhere inside a step. The logits of 8 answers over a vocabulary of 151,936 tokens take a lot of memory while they exist. The step time stayed flat too, as the seconds-per-step column of the windows table shows. So the run used its whole 80 minutes for 122 steps instead of stalling.

Generation is the slow part of RL. Each of our rollouts writes 32 answers of up to 300 tokens before the model can learn anything, and then takes 2 optimiser steps. In the later paper, each R1-Zero rollout wrote 8,192 answers of up to 32,768 tokens (65,536 after step 8,200). Then it took 16 optimiser steps.

Compare the most tokens one rollout may write: $$8{,}192 \times 32{,}768$$ against $$32 \times 300$$, almost 28,000 times more. That is a ratio of maximum rollout budgets, not of tokens actually written per optimiser step. Our run had 122 rollouts and 244 optimiser steps. The method is the same; the scale is not.

## Method 2: the same recipe with TRL

TRL's `GRPOTrainer` does every step of our loop: sampling, rewards, advantages, the loss and the optimiser. We give it the same template and the same two rules. The learning rate, $$\varepsilon$$, $$\beta$$ and G are the same too. We run it for 4 steps only, to show the route. It is a demonstration, not a copy of our run. Each TRL step here uses 2 questions, not 4, and takes one optimiser step, not two.

@@LINES r1_part8_trl.py 16 45@@

TRL passes the extra columns of the dataset, here `gold`, to every reward function as keyword arguments. The reward functions put the prefilled `<think>` back in front, as `sample_group` does.

Three settings matter. `lr_scheduler_type='constant'` keeps the learning rate fixed, as in our loop. `use_bias_correction_kl=False` gives the KL of Eq. 2, as the loss check above showed. And generation must stop at `</answer>`, as ours does. TRL passes `generation_kwargs` to `transformers`, but a stop string also needs the tokenizer, which TRL does not pass. So we bind the tokenizer to `model.generate` with `functools.partial`.

Without the stop string, TRL would keep sampling after `</answer>`. Those extra tokens would enter its loss, its KL and its lengths, while the rewards might see only the cut text. That is a different objective.

With the stop string, one small difference is left. After a stop on `</answer>`, TRL's mask keeps the first padding token. For Qwen, padding is the end-of-text token, and TRL reads it as a sampled end. Our `completion_lengths` leaves it out. So TRL's `completions/mean_length` counts one token more for each answer that stopped on `</answer>`.

TRL logs the same quantities as our loop:

@@LOG r1_part8_trl ^step|steps in@@

Four steps of 2 questions prove nothing about learning, and we do not read the rewards as a trend. One column is worth a look. `clip_ratio/region_mean` is 0 at every step. Here TRL takes one optimiser step per batch of answers, so the ratio is exactly 1, as Step 4 predicts.

The library is the right tool for real work: it adds vLLM for fast sampling, many GPUs and many loss variants. Our package is the right tool to see what happens inside. The check above shows that, on the same tensors, they compute the same loss.

## Run it yourself

The code needs Python, `torch`, `transformers` and `datasets`, plus `trl` and `math-verify` for the checks. The training run needs a machine with about 64 GB of memory, or a GPU.

```bash
git clone https://github.com/ishwar6/ishwar-books
cd ishwar-books/code/papers/deepseek-r1
python r1_part8_probe.py prefill      # do the models earn any reward before training?
python r1_part8_check.py              # the equations on small numbers, and our loss against TRL
python r1_part8_train.py Qwen/Qwen2.5-0.5B-Instruct 400 80   # before eval, 80 minutes of GRPO, after eval
python r1_part8_train.py Qwen/Qwen2.5-0.5B 400 20            # the base model, 20 minutes
python r1_part8_memwatch.py <pid> instruct   # optional, in a second terminal: memory once a minute
python r1_part8_trl.py                # Method 2: 4 steps of TRL's GRPOTrainer
python r1_part8_stats.py              # the 10-step windows of the logs
```

Some good next experiments:

- Give the format reward a weight of 0.5, so a right answer counts more than a tidy one. Does the base model learn more?
- Raise the limit to 1,000 tokens and train on harder questions, such as MATH, to give length a reason to grow.
- Set `minibatches` to 1 and check that the share of clipped tokens drops to 0, as Step 4 predicts.

> [!TAKEAWAYS] Key takeaways
> - The whole R1-Zero method fits in five short files, about 330 lines of Python: the template, two reward rules, GRPO and a training loop.
> - Small details decide what the code really trains on. The mask must cover exactly the tokens the model wrote. The format rule must reject empty and repeated blocks. The number rule must not pay for "1/2" when the answer is 2. Our first version got all three wrong.
> - On 400 held-out answers of each run, our accuracy rule agrees with math_verify on the extracted answer. Our format parser agrees with an independent regular expression. Neither check tests the extraction itself.
> - GRPO needs no critic. Each answer is compared with its own group: $$A_i = (r_i - \text{mean})/\text{std}$$. A group of equal rewards gives no reward signal; only the KL term still acts on it.
> - The clip of Eq. 1 only acts when the model has changed since sampling, which is why the answers are split into mini-batches. It removes an incentive; it is not a hard bound on the change.
> - On the same log-probabilities, our loss equals TRL 1.18's `GRPOTrainer._compute_loss` exactly. Our clip share and KL equal TRL's logged ones. This holds once TRL's default `use_bias_correction_kl=True` (a later DeepSeek-V3.2 change) is turned off.
> - In 35 steps, the 0.5B base model learned the format: 35.5% to 98.5% on held-out questions. But its accuracy fell from 10% to 4%, as its answers shrank to short, tidy guesses.
> - In 122 steps and 80 minutes on a laptop, the 0.5B Instruct model went from 16.5% to 39.0% on 200 held-out GSM8K questions. Its well-formed answers went from 55% to 98%. This is one run with one seed.
> - Answers grew by about 40%, not twenty-fold, and there was no aha moment. With a 300-token limit, 122 steps and easy problems, this run cannot tell us what Figure 3 needs.

## The paper and the code

[Paper (arXiv 2501.12948)](https://arxiv.org/abs/2501.12948) · [Official repository](https://github.com/deepseek-ai/DeepSeek-R1) · [Our scripts and the minir1 package](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/deepseek-r1)

