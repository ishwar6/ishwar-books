---
title: "The Road to R1"
label: Background
description: "How a language model learns, how it was taught to show its steps, and how reinforcement learning with human feedback works. The ideas the R1 paper assumes, with small runs you can check."
covers: "Radford et al. 2018, Wei et al. 2022, Wang et al. 2022, Ouyang et al. 2022, Schulman et al. 2017, Lightman et al. 2023, Shao et al. 2024, DeepSeek-V3 2024"
date: 2026-11-21
tags: [deepseek, reasoning, reinforcement-learning, rlhf, ppo, chain-of-thought, history]
---

The R1 paper fits its method into two pages because it assumes that you know eight years of work before it. This page tells that story in class order, with the one equation that each idea stands on and a small run that you can check.

The story has one question. A language model can write fluent text. How do you make it *think* before it answers, and how do you train that skill when nobody can show it good thinking?

{{FIG:pb_timeline|From 2017 to 2025. Training methods (PPO, GPT-1, InstructGPT) give the machinery. Chain of thought, voting, process rewards and o1 show that thinking helps. DeepSeek's own pieces (GRPO, V3) come last, and R1 puts them together.}}

## How a language model learns

Every model on this page starts the same way. It reads a huge amount of text and learns one skill: guess the next token. GPT-1 wrote this down in 2018, and the equation has not changed since.

> [!PAPER] Radford et al. (2018), GPT-1 · §3.1 · page 3
> [![The GPT-1 pre-training objective: the sum over positions of the log probability of each token given the k tokens before it](/img/papers/deepseek-r1/pb-gpt1-loss.png)](/img/papers/deepseek-r1/pb-gpt1-loss.png)
>
> **What it says:** "we use a standard language modeling objective to maximize the following likelihood": $$L_1(\mathcal{U}) = \sum_i \log P(u_i \mid u_{i-k}, \dots, u_{i-1}; \Theta)$$, "where k is the size of the context window".
>
> **What it means:**
>
> Cover a sentence with a card and slide the card one word to the right at a time. At each stop, the model says how likely each possible next word is. Then you show the real word, and the model scores the log of the probability that it gave to that word. Training changes the weights $$\Theta$$ to raise the total score.
>
> Nobody labels anything: the text is its own answer key, so you can train on trillions of tokens. The result continues text very well, but nothing in this loss asks it to follow instructions.

We usually write the same idea as a loss that goes *down*, the **cross-entropy**. For a text of $$T$$ tokens:

$$
\mathcal{L} = -\frac{1}{T-1}\sum_{t=1}^{T-1} \log p_\theta(x_{t+1} \mid x_1, \dots, x_t)
$$

Read it from the right. $$p_\theta(x_{t+1} \mid x_1, \dots, x_t)$$ is the probability that the model, with weights $$\theta$$, gives to the token that really came next. The log of a probability is negative, so the minus sign makes each term positive, and we take the mean. A perfect guess costs $$-\log 1 = 0$$.

A tiny example by hand. Say the model gives the three right tokens the probabilities 0.5, 0.25 and 0.9. The costs are $$-\log 0.5 = 0.693$$, $$-\log 0.25 = 1.386$$ and $$-\log 0.9 = 0.105$$. The loss is their mean, 0.728. The unsure middle guess costs the most.

> [!DEFINITION] Base model
> A model after pre-training only. It continues any text, but it was not taught to answer questions. R1 starts from DeepSeek-V3-Base, a base model.

### The loss in code

Our **Method 1** writes the loss out: shift by one position, take a stable log-softmax by hand, and pick the probability of the right token. **Method 2** is PyTorch's `F.cross_entropy`, and the model's own `loss`. We run both on Qwen2.5-0.5B, a small base model.

```python
def next_token_loss(logits, ids):
    """Method 1: the next-token cross-entropy written out.
    logits: (T, V) scores at each position; ids: (T,) the tokens. Position t predicts token t + 1."""
    pred = logits[:-1]                                   # (T-1, V): the guesses for tokens 1..T-1
    target = ids[1:]                                     # (T-1,):   the tokens that really came next
    m = pred.max(dim=-1, keepdim=True).values            # subtract the max so exp() cannot overflow
    log_z = m.squeeze(-1) + torch.log(torch.exp(pred - m).sum(-1))   # (T-1,): log of the softmax denominator
    log_p = pred.gather(-1, target[:, None]).squeeze(-1) - log_z     # (T-1,): log p(right token)
    return -log_p, -log_p.mean()                         # per-token losses and their mean
```

```python
    per_tok, ours = next_token_loss(logits, ids[0])
    lib = F.cross_entropy(logits[:-1], ids[0, 1:])                    # Method 2
```

```text
== 1. Next-token loss of Qwen2.5-0.5B (base model) on one sentence
  sentence: 'The capital of France is Paris, and the capital of Italy is Rome.'
  tokens T = 15, vocabulary V = 151,936, logits shape (15, 151936)
  next token        p(next)   -log p
  ' capital'         0.0001    9.317
  ' of'              0.5409    0.614
  ' France'          0.0416    3.179
  ' is'              0.7210    0.327
  ' Paris'           0.3156    1.153
  ','                0.2210    1.509
  ' and'             0.1819    1.705
  ' the'             0.2121    1.551
  ' capital'         0.1571    1.851
  ' of'              0.9671    0.033
  ' Italy'           0.0438    3.128
  ' is'              0.9876    0.013
  ' Rome'            0.9232    0.080
  '.'                0.6554    0.423
  mean loss, ours (Method 1)              : 1.7773
  mean loss, F.cross_entropy (Method 2)    : 1.7773
  mean loss, model(ids, labels=ids).loss   : 1.7773
  max |ours - library| per token: 9.5e-07
  perplexity exp(loss) = 5.91;  a uniform guess would give loss ln(V) = 11.93
```

The three numbers agree, and each token differs by less than one millionth. Look at the column of probabilities. After "The", the word "capital" is a surprise, so it costs 9.3. But " Rome" gets 0.92, because by then the model has seen the pattern "the capital of X is Y" once. A loss of 1.78 means that the model is about as unsure as a choice among 6 tokens, not 151,936.

### Supervised fine-tuning: from base model to chat model

To get a model that answers, people write example answers and train on them. This is **supervised fine-tuning** (SFT). It was step 1 of the recipe that made the first instruction-following GPT models.

> [!PAPER] Ouyang et al. (2022), InstructGPT · §3.1 · page 6
> [![The three steps of InstructGPT: collect demonstrations and train a supervised policy, collect comparisons and train a reward model, optimize a policy against the reward model with PPO](/img/papers/deepseek-r1/pb-igpt-steps.png)](/img/papers/deepseek-r1/pb-igpt-steps.png)
>
> **What it says:** Step 1, "Collect demonstration data, and train a supervised policy." Step 2, "Collect comparison data, and train a reward model." Step 3, "Optimize a policy against the reward model using PPO."
>
> **What it means:**
>
> This is the recipe of almost every chat model from 2022 to 2024. Step 1 is SFT: people write good answers to prompts, and the model learns to copy them. Steps 2 and 3 are reinforcement learning from human feedback (RLHF), which we reach below.
>
> Note the word **policy**. It is what reinforcement learning calls "the thing that acts", here the language model. R1 uses the same word.

SFT uses the same loss as pre-training, with one change. The data is a prompt and a response, and the loss counts only the response tokens. The model should learn to *answer*, not to write questions. In code, the prompt tokens get the label $$-100$$, and both methods skip them:

```python
def sft_loss(logits, labels):
    """Method 1 for SFT: the same loss, but only where labels != -100 (the response tokens)."""
    per_tok, _ = next_token_loss(logits, labels.clamp(min=0))
    keep = (labels[1:] != -100).float()                  # (T-1,): 1 on response tokens, 0 on prompt tokens
    return (per_tok * keep).sum() / keep.sum()
```

```text
== 2. SFT loss: the same loss, only on the response tokens
  prompt 13 tokens (label -100), response 18 tokens
  SFT loss, ours (Method 1)                       : 0.5647
  SFT loss, F.cross_entropy(ignore_index=-100)     : 0.5647
  SFT loss, model(ids, labels=labels).loss         : 0.5647
  loss on every token (prompt too), for comparison  : 1.1737
```

All three give 0.5647. The loss on every token is twice as high, because the question itself is hard to predict. SFT ignores that part.

> [!DEFINITION] Supervised fine-tuning (SFT)
> Training on prompt and response pairs with the next-token loss, counted on the response only. The result is a chat model, or instruct model. R1's pipeline has two SFT stages ([Part 5](part-5-the-r1-pipeline.md)), and its small models are made with SFT alone ([Part 6](part-6-distillation.md)).

{{FIG:pb_stages|You are here: the four stages of the classic recipe. Pre-training makes a base model, SFT makes a chat model, and a reward model plus RL with PPO make up RLHF. R1-Zero will skip SFT and the learned reward model.}}

SFT has a cost that matters for R1. It needs good answers to copy, and for hard maths, thousands of long, correct, step-by-step solutions are slow and costly to write. The examples also shape what the model learns: it is trained to imitate them, not to find methods they do not show. A student can still end up better than the examples, but SFT alone gives no direct push in that direction.

## Chain of thought: let the model write its steps

In early 2022, a Google team found that you can get much better answers with no training at all. You only show the model a few examples that write out their steps.

> [!PAPER] Wei et al. (2022), Chain-of-Thought Prompting · Abstract and Figure 1 · page 1
> [![The chain-of-thought abstract: a chain of thought is a series of intermediate reasoning steps, and the ability emerges in large models](/img/papers/deepseek-r1/pb-cot-abstract.png)](/img/papers/deepseek-r1/pb-cot-abstract.png)
> [![Figure 1 of the chain-of-thought paper: standard prompting answers 27 and is wrong; chain-of-thought prompting writes 23 - 20 = 3, 3 + 6 = 9 and is right](/img/papers/deepseek-r1/pb-cot-figure1.png)](/img/papers/deepseek-r1/pb-cot-figure1.png)
>
> **What it says:** A chain of thought is "a series of intermediate reasoning steps", and the skill appears "naturally in sufficiently large language models" when a few worked examples are in the prompt.
>
> **What it means:**
>
> Look at the figure. On the left, the model jumps from the question to a number, 27, and is wrong. On the right, the example in the prompt shows its working. So the model writes its working too: 23 minus 20 is 3, plus 6 is 9.
>
> Why does writing help? A model spends the same fixed amount of computation on each token that it writes. A one-token answer gets one step of thought. A written chain gets one step per token. Each result that it writes, such as "3", stays in the text, where the next steps can read it. The text becomes the model's scratch paper.
>
> The word "naturally" matters for R1. The skill is already in a big base model; a prompt brings it out. R1 asks whether a reward can make it grow.

A second idea came two months later. If one chain of thought can go wrong, write many and take a vote.

> [!PAPER] Wang et al. (2022), Self-Consistency · Abstract · page 1
> [![The self-consistency abstract: sample a diverse set of reasoning paths, select the most consistent answer, plus 17.9 percent on GSM8K](/img/papers/deepseek-r1/pb-sc-abstract.png)](/img/papers/deepseek-r1/pb-sc-abstract.png)
>
> **What it says:** Self-consistency "samples a diverse set of reasoning paths" and "selects the most consistent answer", which improves GSM8K by "+17.9%".
>
> **What it means:**
>
> Ask five students to solve the same problem on their own. Wrong solutions are wrong in different ways, so they spread over many answers. Right solutions all land on the same number. So the most common answer is right more often than any one student.
>
> This works only when answers can be compared, like numbers, and $$k$$ samples cost $$k$$ times the compute of one. The R1 paper reports such votes too: R1-Zero goes from 71.0% with one sample to 86.7% with a vote on AIME 2024 (§1, page 3).

The vote is one line of maths. Given $$k$$ sampled answers $$a_1, \dots, a_k$$:

$$
\hat{a} = \arg\max_{a} \sum_{i=1}^{k} \mathbb{1}[a_i = a]
$$

$$\mathbb{1}[a_i = a]$$ is 1 when sample $$i$$ gave the answer $$a$$, and 0 otherwise. The sum counts the votes for $$a$$, and $$\arg\max$$ picks the answer with the most votes. With five samples 18, 18, 16, 18, 20, the counts are 3, 1 and 1, so the vote says 18.

> [!DEFINITION] GSM8K
> Grade-school maths word problems with a number as the answer (Cobbe et al., 2021). The test set has 1,319 problems.

### Thinking and voting on a small model

We tried both ideas on Qwen2.5-0.5B-Instruct, a chat model of half a billion parameters, with the first 30 GSM8K test questions. Method 1 reads the answer with our own pattern and counts the votes by hand. Method 2 uses the math-verify library to check answers and `collections.Counter` to count votes.

```python
def vote(answers):
    """Method 1: majority vote by hand. Count each answer; the first one to reach the top count wins ties."""
    counts, best = {}, None
    for a in answers:
        if a is None:
            continue
        counts[a] = counts.get(a, 0) + 1
        if best is None or counts[a] > counts[best]:
            best = a
    return best


def vote_lib(answers):
    """Method 2: the same vote with collections.Counter."""
    c = Counter(a for a in answers if a is not None)
    return c.most_common(1)[0][0] if c else None
```

```text
== Chain of thought and majority vote: Qwen/Qwen2.5-0.5B-Instruct, first 30 GSM8K test questions, device mps
   direct: greedy, reply starts with 'The answer is ', max 16 new tokens; CoT: greedy, max 512; samples: 16 per question, temperature 0.7, max 512

                              our regex   math-verify   mean new tokens
  direct answer (greedy)        1/30         1/30            7.9
  step by step (greedy)        12/30        11/30          314.0
  one sample (mean of 16)         7.8/30       7.2/30        342.1
  majority vote of  1 samples    7/30
  majority vote of  2 samples    7/30
  majority vote of  4 samples   10/30
  majority vote of  8 samples   11/30
  majority vote of 16 samples   14/30
  at least one of 16 right       21/30
  vote by hand vs Counter: 0 differences over 5 values of k x 30 questions
  regex vs math-verify: same verdict on 524 of 540 replies
  example, question 17: gold 230; greedy CoT said 115; 16 samples said ['2', '2', '320', '294', '230', '460', '230', '230', '450', '240', '170', '185', '230', '150', '230', '80']; vote 230
  generation time 540 s
```

[![The terminal output of the chain-of-thought run](/img/papers/deepseek-r1/r1_partb_cot-run.png)](/img/papers/deepseek-r1/r1_partb_cot-run.png)

{{FIG:pb_cot|Our run on 30 GSM8K questions with Qwen2.5-0.5B-Instruct. A direct answer is right once. One greedy chain of thought is right 12 times. A vote over 16 sampled chains is right 14 times.}}

Read the table from the top. When the reply must start with "The answer is", the model is right on 1 question of 30. When it writes its steps first, it is right on 12.

One sampled chain (temperature 0.7) is right on 7.8 questions on average, fewer than the greedy chain. Sampling adds variety and also adds mistakes. But the vote turns that variety into accuracy: 10 with 4 samples, 11 with 8, and 14 with 16. Question 17 shows how. The greedy chain said 115; the 16 samples said many things, but 230, the right answer, came up 5 times and won.

Now look at the line "at least one of 16 right": 21 questions. The model *can* solve them, but not reliably. One way to read RL is as a method that moves the single answer toward the best sample.

Take these numbers as a demonstration, not a benchmark. 30 questions is a small sample, so one question moves the score by 3.3 points. The model is tiny, we cut chains at 512 tokens, and our answer reader and math-verify disagree on 16 of 540 replies. Our hand-written vote and `Counter` agree on all 150 votes.

## Reinforcement learning, in the words of a language model

SFT needs an example of the right answer. **Reinforcement learning** (RL) needs only a score after the model acts. The model tries, a judge scores the try, and high-scoring tries become more likely. For maths, the judge can be a rule: is the final number right?

Here are the parts in the words of a language model. The **policy** $$\pi_\theta$$ is the model itself, with weights $$\theta$$. A prompt $$x$$ is the situation. Each token that the model writes is one **action**. The full response $$y$$ is one episode, and the **reward** $$r(x, y)$$ comes once, at the end. The goal is to make the average reward high:

$$
J(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)}\big[\, r(x, y) \,\big]
$$

We cannot take the gradient of a rule like "is the number right". But in 1992, Williams showed a way around it, the **REINFORCE** estimate. Sample a response, then push its log-probability up in proportion to its reward:

$$
\nabla_\theta J = \mathbb{E}\big[\, (r(x, y) - b)\, \nabla_\theta \log \pi_\theta(y \mid x) \,\big],
\qquad
\log \pi_\theta(y \mid x) = \sum_t \log \pi_\theta(y_t \mid x, y_{<t})
$$

Read the left formula. $$\nabla_\theta \log \pi_\theta(y \mid x)$$ is the direction that makes this response more likely. $$r - b$$ says how far to go, and in which direction. $$b$$ is a **baseline**, a number that does not depend on the response. The right formula says that the log-probability of a response is the sum over its tokens. So every token of the response gets the same push.

> [!PAPER] Schulman et al. (2017), PPO · §2.1 · page 2
> [![The policy gradient estimator of the PPO paper: the expected gradient of the log policy times the advantage estimate](/img/papers/deepseek-r1/pb-ppo-pg.png)](/img/papers/deepseek-r1/pb-ppo-pg.png)
>
> **What it says:** "The most commonly used gradient estimator has the form" $$\hat{g} = \hat{\mathbb{E}}_t[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\hat{A}_t]$$, where $$\hat{A}_t$$ "is an estimator of the advantage function at timestep t".
>
> **What it means:**
>
> This is REINFORCE with the baseline built in. $$\hat{A}_t$$, the **advantage**, is reward minus baseline: how much better this action was than expected. A positive advantage makes the action more likely. A negative one makes it less likely.
>
> For a language model, the state $$s_t$$ is the prompt plus the tokens so far, and the action $$a_t$$ is the next token. GRPO in [Part 2](part-2-grpo.md) is, at its heart, a new way to compute $$\hat{A}_t$$.

### Why a baseline helps

Our tiny "model" picks one of four answers with probabilities 0.3875, 0.235, 0.1425 and 0.235. Only answer 2 is right, with reward 1. For a softmax policy, $$\nabla \log \pi(a)$$ is "one-hot of $$a$$ minus $$\pi$$". So a sample of answer 2 gives the update $$[-0.3875, -0.235, 0.8575, -0.235]$$: up for answer 2, down for the rest. A wrong sample has reward 0 and changes nothing.

Now add 5 to every reward. The true gradient does not change, because only differences matter. But a wrong answer now gets pushed *up* with weight 5, and only on average do the right answers win. A baseline near the mean reward takes the 5 away again. Here is the check, with 200,000 samples:

```python
def grad_log_pi(theta, a):
    """Method 1: d log pi(a) / d theta for a softmax policy = one_hot(a) - pi.   (4,)"""
    g = -softmax(theta)
    g[a] += 1.0
    return g
```

```python
        g = (r - b)[:, None] * glp                      # (S, 4): one REINFORCE estimate per sample
        mean = g.mean(0)
        var = g.var(0).sum()                            # total variance over the 4 entries
```

```text
== 1. REINFORCE: a policy picks one of 4 answers; only answer 2 is right (reward 1, else 0)
  pi = [0.3875, 0.235, 0.1425, 0.235]   expected reward J = 0.1425
  exact gradient dJ/dtheta = pi * (r - J) = [-0.0552, -0.0335, 0.1222, -0.0335]
  grad log pi(a=2): ours [-0.3875, -0.235, 0.8575, -0.235]   autograd [-0.3875, -0.235, 0.8575, -0.235]
  exact gradient by autograd           [-0.0552, -0.0335, 0.1222, -0.0335]
  rewards +0, no baseline    : mean of 200,000 estimates [-0.0554, -0.0336, 0.1227, -0.0336]  variance 0.1221
  rewards +0, baseline b = J : mean of 200,000 estimates [-0.0554, -0.0338, 0.1226, -0.0334]  variance 0.0961
  rewards +5, no baseline    : mean of 200,000 estimates [-0.0554, -0.0272, 0.1254, -0.0428]  variance 19.5260
  rewards +5, baseline b = J : mean of 200,000 estimates [-0.0554, -0.0338, 0.1226, -0.0334]  variance 0.0961
```

Method 2, PyTorch's autograd, gives the same $$\nabla \log \pi$$ and the same exact gradient as our formulas. All four estimates have the right mean, up to sampling noise, because $$\nabla \log \pi$$ is zero on average. But the variance falls from 0.1221 to 0.0961 with 0/1 rewards, and from 19.5 to 0.0961 with shifted rewards.

Variance decides whether training works. We trained the same tiny policy for 300 steps, one sample per step, with 200 random seeds:

```text
  rewards +0, no baseline     : step 50 mean 0.332, step 300 mean 0.960, worst seed 0.950, seeds below 0.5 at the end 0/200
  rewards +0, baseline        : step 50 mean 0.332, step 300 mean 0.959, worst seed 0.927, seeds below 0.5 at the end 0/200
  rewards +5, no baseline     : step 50 mean 0.270, step 300 mean 0.428, worst seed 0.001, seeds below 0.5 at the end 112/200
  rewards +5, baseline        : step 50 mean 0.333, step 300 mean 0.953, worst seed 0.872, seeds below 0.5 at the end 0/200
```

{{FIG:pb_reinforce|Probability of the right answer while training, mean of 200 seeds. With 0/1 rewards, and with shifted rewards plus a baseline, the policy reaches about 0.95. With shifted rewards and no baseline, 112 of 200 seeds end below 0.5, and one seed ends at 0.001.}}

With a baseline, every run learns. A baseline is not required: REINFORCE is correct without one, as the unshifted run shows. But it can cut the noise a great deal: from 19.5 to 0.0961 with shifted rewards. So most RL methods for language models use one, and the question is where it comes from. PPO learns it with a second network. GRPO takes it from a group of answers.

## RLHF: learning from human rankings

For most prompts, no rule can say if an answer is good. Is this poem nice? Is this email polite? Step 2 of InstructGPT trains a model to give that score. People see several answers to one prompt and rank them, and a **reward model** learns to agree with them.

> [!PAPER] Ouyang et al. (2022), InstructGPT · §3.5 · page 8
> [![The reward model loss of InstructGPT: minus the expected log sigmoid of the reward difference between the preferred and the other completion](/img/papers/deepseek-r1/pb-igpt-rm.png)](/img/papers/deepseek-r1/pb-igpt-rm.png)
>
> **What it says:** $$\text{loss}(\theta) = -\frac{1}{\binom{K}{2}} E_{(x, y_w, y_l) \sim D}[\log(\sigma(r_\theta(x, y_w) - r_\theta(x, y_l)))]$$, where $$r_\theta(x, y)$$ "is the scalar output of the reward model" and $$y_w$$ "is the preferred completion out of the pair".
>
> **What it means:**
>
> People are bad at giving an answer a score from 1 to 10, but they are good at saying which of two answers is better. So the data is pairs: a winner $$y_w$$ and a loser $$y_l$$. The model gives each a number. The difference goes through a sigmoid, $$\sigma(z) = 1/(1 + e^{-z})$$, which turns it into a probability that the winner wins. The loss is minus the log of that probability.
>
> This is the Bradley-Terry model of 1952, a classic way to rank players from their wins. A labeler who ranks $$K$$ answers gives $$\binom{K}{2}$$ pairs. The reward model is the SFT model with a last layer that outputs one number.

A tiny example. The reward model gives the winner 1.2 and the loser 0.4. The difference is 0.8, $$\sigma(0.8) = 0.690$$, and the loss is $$-\log 0.690 = 0.371$$. If the scores were the other way round, the loss would be 1.171, so training pushes them apart. Method 1 is the formula; Method 2 is PyTorch's `F.logsigmoid` and `F.binary_cross_entropy_with_logits`:

```text
== 2. Reward model (Bradley-Terry): loss = -log sigmoid(r(x, y_w) - r(x, y_l))
  r_w = 1.2, r_l = 0.4: P(w beats l) = 0.690, loss ours 0.3711  logsigmoid 0.3711  BCE-with-logits 0.3711
  r_w = 0.4, r_l = 1.2: P(w beats l) = 0.310, loss ours 1.1711  logsigmoid 1.1711  BCE-with-logits 1.1711
  r_w = 3.0, r_l = 0.0: P(w beats l) = 0.953, loss ours 0.0486  logsigmoid 0.0486  BCE-with-logits 0.0486
  ranking D > B > A > C -> 6 pairs: [('D', 'B'), ('D', 'A'), ('D', 'C'), ('B', 'A'), ('B', 'C'), ('A', 'C')]
  learned scores {'A': -1.35, 'B': 1.35, 'C': -4.33, 'D': 4.33}, final loss 0.029
  add 10 to every score: loss 0.029 (only differences count)
```

The last three lines learn four scores from one ranking, D > B > A > C, and they come out in that order. Adding 10 to every score leaves the loss the same. So a reward model has no natural zero, and InstructGPT shifts its scores before RL.

### Step 3: raise the score, but stay close

Now the policy writes answers, the reward model scores them, and RL raises the score. But the reward model is only a guess of what people like. Far from familiar text, the policy can find odd answers that score high and that people dislike.

> [!PAPER] Ouyang et al. (2022), InstructGPT · §3.5 · page 9
> [![The RL setup of InstructGPT: the reward model gives a reward at the end of the episode, a per-token KL penalty from the SFT model, and a value function initialized from the reward model](/img/papers/deepseek-r1/pb-igpt-rl.png)](/img/papers/deepseek-r1/pb-igpt-rl.png)
>
> **What it says:** The environment "produces a reward determined by the reward model and ends the episode". The authors "add a per-token KL penalty from the SFT model at each token", and "The value function is initialized from the RM."
>
> **What it means:**
>
> Three ideas sit in four lines. First, one episode is one response, and the reward comes once, at the end. Second, the policy pays a small price at every token where it moves away from the SFT model. Think of a dog on a leash: it may explore, but not too far from its owner.
>
> Third, there is a value function, the **critic**. It is a separate network that predicts the final reward from a partial response: the baseline of the section above, learned. It starts from the reward model, so it has the same size.

The reward that RL sees, from the paper's Equation 2 (we drop the term that mixes in pre-training data), is:

$$
R(x, y) = r_\theta(x, y) - \beta \log \frac{\pi^{RL}(y \mid x)}{\pi^{SFT}(y \mid x)}
$$

$$r_\theta(x, y)$$ is the reward model's score. The log ratio is large when the policy makes this response much more likely than the SFT model did; this is the **KL penalty**. $$\beta$$ sets the length of the leash: InstructGPT uses $$\beta = 0.02$$ (Appendix C.4). Spread over tokens, each token pays $$-\beta \log(\pi / \pi^{SFT})$$ for its own probability:

```text
== 3. RLHF reward with a per-token KL penalty: r_t = -beta * log(pi(t) / pi_ref(t)), plus r_RM at the end
  beta = 0.02 (InstructGPT, App. C.4), reward model score 1.5 on the last token
  token 1: pi 0.60  pi_ref 0.30  penalty ours -0.0139  torch -0.0139
  token 2: pi 0.90  pi_ref 0.90  penalty ours -0.0000  torch -0.0000
  token 3: pi 0.30  pi_ref 0.40  penalty ours +0.0058  torch +0.0058
  total reward = 1.5 + (-0.0139 + -0.0000 + +0.0058) = 1.4919
```

Token 1 doubled its probability compared with the reference, so it pays 0.0139. Token 2 did not move, so it pays nothing. Token 3 gets a small bonus, but on average the cost is never below zero: the mean log ratio is the KL divergence. The frozen SFT model that gives $$\pi^{SFT}$$ is called the **reference model**.

> [!DEFINITION] KL divergence
> A measure of how different two probability distributions are. It is zero when they are equal and grows as they move apart. In RLHF and in R1, a KL term keeps the trained policy near a frozen reference model.

### PPO: small, safe steps

The last piece is the update rule. Plain REINFORCE uses each batch of samples once, because a large step on old samples can wreck the policy. **Proximal Policy Optimization** (PPO, 2017) lets you reuse a batch for several steps, and it stops each step from going too far.

> [!PAPER] Schulman et al. (2017), PPO · §3 · page 3
> [![The clipped surrogate objective of PPO: the minimum of the ratio times the advantage and the clipped ratio times the advantage, with epsilon 0.2](/img/papers/deepseek-r1/pb-ppo-clip.png)](/img/papers/deepseek-r1/pb-ppo-clip.png)
>
> **What it says:** $$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon)\hat{A}_t)]$$, "say, ϵ = 0.2". Clipping "removes the incentive for moving $$r_t$$ outside of the interval $$[1 - \epsilon, 1 + \epsilon]$$", and "we take the minimum of the clipped and unclipped objective".
>
> **What it means:**
>
> Careful: here $$r_t(\theta)$$ is not a reward. It is a ratio, the new probability of the token divided by the probability when the sample was made. A ratio of 1.3 means that the update made this token 30% more likely.
>
> The clip takes away the reward for speeding; it is not a hard speed limit. For a good token (advantage above 0), the gain stops growing once the ratio passes 1.2, so this term no longer pushes the token further. For a bad token, it stops once the ratio falls below 0.8. The ratio can still go past these points, because every token shares the same weights and the updates for other tokens move it too. The minimum makes sure that the clip can only take gain away, never add it. GRPO keeps this exact clip, as Equation 1 of R1 shows.

Our Method 1 is the formula with Python's `min` and `max`. Method 2 is `torch.min` and `torch.clamp`, and autograd tells us which terms still push:

```python
def ppo_term(ratio, A, eps=0.2):
    """Method 1: one term of L_CLIP by hand."""
    clipped = min(max(ratio, 1 - eps), 1 + eps)
    return min(ratio * A, clipped * A)
```

```python
    lib = torch.min(rt * A, torch.clamp(rt, 1 - eps, 1 + eps) * A)               # Method 2
```

```text
== 4. PPO clipped objective, epsilon = 0.2: min(ratio * A, clip(ratio, 0.8, 1.2) * A)
  A = +1, ratio 0.5: unclipped +0.50  ours +0.50  torch +0.50  gradient +1
  A = +1, ratio 0.9: unclipped +0.90  ours +0.90  torch +0.90  gradient +1
  A = +1, ratio 1.0: unclipped +1.00  ours +1.00  torch +1.00  gradient +1
  A = +1, ratio 1.1: unclipped +1.10  ours +1.10  torch +1.10  gradient +1
  A = +1, ratio 1.5: unclipped +1.50  ours +1.20  torch +1.20  gradient +0
  A = -1, ratio 0.5: unclipped -0.50  ours -0.80  torch -0.80  gradient +0
  A = -1, ratio 0.9: unclipped -0.90  ours -0.90  torch -0.90  gradient -1
  A = -1, ratio 1.0: unclipped -1.00  ours -1.00  torch -1.00  gradient -1
  A = -1, ratio 1.1: unclipped -1.10  ours -1.10  torch -1.10  gradient -1
  A = -1, ratio 1.5: unclipped -1.50  ours -1.50  torch -1.50  gradient -1
```

Read the gradient column. A good token at ratio 1.5 has already moved too far, so its gradient is 0. A bad token at ratio 0.5 has already dropped enough, so its gradient is 0 too. But a bad token at ratio 1.5, one that the update made *more* likely by mistake, keeps its full gradient. The clip stops greed, not repair.

{{FIG:pb_ppo_clip|One term of the PPO objective against the probability ratio. Left: for a good answer, the gain is flat after 1.2. Right: for a bad answer, it is flat below 0.8. The dot is the start, ratio 1. The dashed line is the plain product without the clip.}}

### The cost: four models in memory

Now count the models that RLHF with PPO keeps in memory. The policy is trained, and the reference model is a frozen copy of SFT. The reward model is frozen too, and the critic is trained. The critic is not small: InstructGPT used a 6B reward model and a 6B critic started from it (Appendix C.4).

> [!PAPER] Shao et al. (2024), DeepSeekMath · §4.1.1 · page 13
> [![DeepSeekMath on PPO: the value function is another model of comparable size to the policy, a memory and compute burden, and only the last token gets a reward](/img/papers/deepseek-r1/pb-dsm-critic.png)](/img/papers/deepseek-r1/pb-dsm-critic.png)
>
> **What it says:** The value function in PPO is "another model of comparable size as the policy model", which brings "substantial memory and computational burden", and "only the last token is assigned a reward score by the reward model".
>
> **What it means:**
>
> These are DeepSeek's own words, about a year before R1, and they name two problems. The first is memory: a second model of the policy's size, with its own gradients and optimizer state. The second is subtler. The critic must predict, at every token, a reward that arrives only at the end, and a poor critic gives a poor baseline.

How much memory is "substantial"? A common rule says that a model trained with Adam in mixed precision needs about 16 bytes per parameter. A frozen one needs 2. For a 7-billion-parameter policy, this is our estimate, not a measurement:

```text
== 5. Memory estimate for RLHF with PPO on a 7B policy (rule of thumb, not a measurement)
  policy    :   112 GB  (trained)
  critic    :   112 GB  (trained)
  reference :    14 GB  (frozen)
  reward    :    14 GB  (frozen)
  total 252 GB; without the critic 140 GB (44% less), before activations and the KV cache of generation
```

{{FIG:pb_rlhf|RLHF with PPO and its four models. The policy writes a response. The frozen reference model gives the KL penalty, the frozen reward model gives the score, and the trained critic gives the baseline. PPO combines them into one update of the policy. Two of the four models are trained.}}

Drop the critic and the bill falls by 44% in this estimate. Hold that thought. It is exactly the model that GRPO removes, and [Part 2](part-2-grpo.md) shows how.

## Outcome rewards, process rewards, and reward hacking

For maths, there is a choice about *what* to score. You can score only the final answer, or you can score every step of the working.

> [!PAPER] Lightman et al. (2023), Let's Verify Step by Step · Abstract · page 1
> [![The abstract of Let's Verify Step by Step: outcome supervision gives feedback for a final result, process supervision for each step, and the process-supervised model solves 78 percent of a MATH subset](/img/papers/deepseek-r1/pb-prm-abstract.png)](/img/papers/deepseek-r1/pb-prm-abstract.png)
>
> **What it says:** "outcome supervision, which provides feedback for a final result, or process supervision, which provides feedback for each intermediate reasoning step". The process-supervised model "solves 78% of" a subset of the MATH test set.
>
> **What it means:**
>
> A teacher who marks only the final answer is an **outcome reward model** (ORM). A teacher who marks every line is a **process reward model** (PRM). The PRM gives more signal: it can say *where* a solution went wrong. A lucky guess after wrong steps also gets no credit.
>
> But look at the cost. This paper used 800,000 step-level labels from people, the PRM800K dataset. The R1 authors tried PRMs and gave up on them for large-scale RL. [Part 7](part-7-results-failures-and-limits.md) reads their reasons in §4.2. A "step" is hard to define, step labels are hard to get, and a learned judge invites reward hacking.

**Reward hacking** is a policy that gets a high score without doing the task. A learned reward model is only an imperfect copy of the real goal, and RL is very good at finding the gaps.

> [!PAPER] Gao et al. (2022), Reward Model Overoptimization · §1 · page 1
> [![Goodhart's law: when a measure becomes a target, it ceases to be a good measure; optimizing too much against a reward model hinders the true objective](/img/papers/deepseek-r1/pb-gao-goodhart.png)](/img/papers/deepseek-r1/pb-gao-goodhart.png)
>
> **What it says:** "When a measure becomes a target, it ceases to be a good measure." "Optimizing too much against such a model eventually hinders the true objective".
>
> **What it means:**
>
> Pay students for every page of homework, and you get long homework, not good homework. A reward model that slightly prefers long, confident answers will, after enough RL, get very long, very confident answers.
>
> The KL leash slows this, but it does not stop it. The R1 paper takes a blunter path for maths and code. It drops the learned reward model and uses rules: compare the final number with the known answer, or run the code on tests. A rule that checks the final number is much harder to fool. [Part 3](part-3-rewards-and-the-template.md) covers those rules.

## Thinking longer at answer time

Chain of thought and voting share one idea: you can buy better answers with more computation *at answer time*. This is called **test-time compute**. Our small run showed the cost. A chain of thought used about 314 tokens per answer, against 8 for a direct answer, and a vote of 16 costs 16 chains.

In September 2024, OpenAI released o1, a model that writes a long hidden chain of thought before each answer. It was much stronger on competition maths and code. OpenAI's announcement showed accuracy rising with more thinking time, but it did not publish how it trained the model.

> [!PAPER] · §1 · page 3
> [![The R1 introduction on o1: o1 first introduced inference-time scaling by increasing the length of the chain of thought; none of the prior methods reached o1's general reasoning](/img/papers/deepseek-r1/pb-r1-o1.png)](/img/papers/deepseek-r1/pb-r1-o1.png)
>
> **What it says:** OpenAI's o1 models "were the first to introduce inference-time scaling by increasing the length of the Chain-of-Thought reasoning process", but of the methods that others tried, "none of these methods has achieved general reasoning performance comparable to OpenAI's o1 series models".
>
> **What it means:**
>
> This is the situation in late 2024. The field knew that long thinking works, because o1 showed it. Nobody outside OpenAI knew how to train it. The list of attempts is the list of this page: process reward models, RL, and search over many paths.
>
> "Inference-time scaling" is a new axis. Before o1, you made a model bigger or trained it longer. Now you could also let it think longer.

### DeepSeek's two pieces

Two earlier DeepSeek papers supply the parts that R1 is built from: an RL algorithm and a base model.

> [!PAPER] Shao et al. (2024), DeepSeekMath · Abstract · page 1
> [![The DeepSeekMath abstract: Group Relative Policy Optimization, a variant of PPO that optimizes the memory usage of PPO](/img/papers/deepseek-r1/pb-dsm-abstract.png)](/img/papers/deepseek-r1/pb-dsm-abstract.png)
>
> **What it says:** "we introduce Group Relative Policy Optimization (GRPO), a variant of Proximal Policy Optimization (PPO), that enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO."
>
> **What it means:**
>
> In February 2024, DeepSeek published a 7-billion-parameter maths model. Its RL step used a new variant of PPO that keeps the clip and the KL leash but drops the critic. The memory we counted above is the reason.
>
> There, GRPO was a final polish on a model already trained with SFT on maths. R1 asks a bolder question with the same tool. [Part 2](part-2-grpo.md) explains it line by line.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 · Abstract · page 1
> [![The DeepSeek-V3 abstract: a Mixture-of-Experts model with 671B total parameters, 37B activated for each token, pre-trained on 14.8 trillion tokens](/img/papers/deepseek-r1/pb-v3-abstract.png)](/img/papers/deepseek-r1/pb-v3-abstract.png)
>
> **What it says:** DeepSeek-V3 is a Mixture-of-Experts model "with 671B total parameters with 37B activated for each token", pre-trained on "14.8 trillion" tokens, "followed by Supervised Fine-Tuning and Reinforcement Learning stages".
>
> **What it means:**
>
> V3 came out one month before R1. A Mixture-of-Experts model has many expert blocks and uses only a few of them for each token. So it holds 671 billion parameters but computes with 37 billion per token. The [DeepSeek-V3 article](/writings/deepseek-2-the-v3-architecture/) explains how.
>
> R1 does not start from the V3 chat model. It starts from **DeepSeek-V3-Base**, before any SFT. That choice is the whole experiment.

## The question R1 answers {§1}

Put the pieces side by side. A base model already holds the pieces of reasoning, and chain of thought shows that writing steps helps. RL can raise any score that you can compute. PPO makes the steps safe, but its critic doubles the trained models. Learned reward models invite hacking. And o1 showed that long thinking at answer time pays off, without saying how.

> [!PAPER] · §1 · page 3
> [![The R1 introduction: the first step toward improving reasoning using pure RL, without any supervised data, with DeepSeek-V3-Base and GRPO](/img/papers/deepseek-r1/pb-r1-pure.png)](/img/papers/deepseek-r1/pb-r1-pure.png)
>
> **What it says:** "we take the first step toward improving language model reasoning capabilities using pure reinforcement learning (RL)", "without any supervised data", with "DeepSeek-V3-Base as the base model" and "GRPO (Shao et al., 2024) as the RL framework".
>
> **What it means:**
>
> Here is the question in one line. Take a base model, with no SFT and no human-written chains of thought. Give it only RL with GRPO and a reward that a rule can check. Does it learn to think?
>
> If the answer is yes, then good reasoning does not have to be shown to a model. It can be rewarded into it. [Part 1](part-1-the-big-idea.md) starts with the paper's answer.

> [!TAKEAWAYS] Key takeaways
> - **Pre-training is next-token cross-entropy.** On one sentence, Qwen2.5-0.5B had a loss of 1.7773, and our hand-written loss matched `F.cross_entropy` to 9.5e-07.
> - **SFT is the same loss on the response only.** It can only copy the answers that people write.
> - **Chain of thought gives the model scratch paper**, and a majority vote over samples fixes many single mistakes. On 30 GSM8K questions our 0.5B model went from 1 right (direct) to 12 (steps) to 14 (vote of 16).
> - **REINFORCE pushes up the log-probability of a response in proportion to reward minus a baseline.** With all rewards shifted by 5, the baseline cut our variance from 19.5 to 0.0961.
> - **RLHF learns a reward model from ranked pairs** with the loss $$-\log \sigma(r_w - r_l)$$. RL then adds a KL penalty to a frozen reference model ($$\beta = 0.02$$ in InstructGPT).
> - **PPO clips the probability ratio to [0.8, 1.2].** Outside that range, a term that would gain more has a gradient of zero.
> - **PPO's critic is a second big trained model.** In our 7B estimate it is 112 of 252 GB; GRPO removes it.
> - **Process rewards give more signal but cost labels and invite reward hacking.** R1 uses rule-based rewards instead.
> - **o1 showed that thinking longer helps but not how to train it.** R1 asks whether pure RL on a base model is enough.

All the code of this page is in three scripts. `r1_partb_loss.py` has the losses, and `r1_partb_cot.py` has the thinking and voting run. `r1_partb_rl.py` has REINFORCE, the reward model, the KL penalty, PPO and the memory estimate.

## The paper and the code

[Paper (arXiv 2501.12948)](https://arxiv.org/abs/2501.12948) · [Official repository](https://github.com/deepseek-ai/DeepSeek-R1) · [Our code for this series](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/deepseek-r1)
