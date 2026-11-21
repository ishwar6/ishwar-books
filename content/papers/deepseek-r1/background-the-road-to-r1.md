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

<figure class="fig"><svg viewBox="0 0 770 304" role="img" aria-label="Timeline: PPO 2017, GPT-1 2018, chain of thought, self-consistency and InstructGPT 2022, process reward models 2023, DeepSeekMath with GRPO, OpenAI o1 and DeepSeek-V3 in 2024, and DeepSeek-R1 in January 2025."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="axis" x1="24.0" y1="150.0" x2="746.0" y2="150.0" marker-end="url(#ah)"/><circle class="s1" cx="50" cy="150" r="7"/><line class="grid" x1="50.0" y1="141.0" x2="50.0" y2="124.0"/><text class="t-title" x="50.0" y="52.0" text-anchor="middle">2017</text><text class="t-note" x="50.0" y="72.0" text-anchor="middle">PPO</text><text class="t-muted" x="50.0" y="92.0" text-anchor="middle">clipped policy</text><text class="t-muted" x="50.0" y="107.0" text-anchor="middle">updates</text><circle class="s1" cx="124" cy="150" r="7"/><line class="grid" x1="124.0" y1="159.0" x2="124.0" y2="176.0"/><text class="t-title" x="124.0" y="194.0" text-anchor="middle">2018</text><text class="t-note" x="124.0" y="214.0" text-anchor="middle">GPT-1</text><text class="t-muted" x="124.0" y="234.0" text-anchor="middle">pre-train, then</text><text class="t-muted" x="124.0" y="249.0" text-anchor="middle">fine-tune</text><circle class="s2" cx="198" cy="150" r="7"/><line class="grid" x1="198.0" y1="141.0" x2="198.0" y2="124.0"/><text class="t-title" x="198.0" y="52.0" text-anchor="middle">2022</text><text class="t-note" x="198.0" y="72.0" text-anchor="middle">Chain of thought</text><text class="t-muted" x="198.0" y="92.0" text-anchor="middle">write the steps</text><text class="t-muted" x="198.0" y="107.0" text-anchor="middle">before the answer</text><circle class="s2" cx="272" cy="150" r="7"/><line class="grid" x1="272.0" y1="159.0" x2="272.0" y2="176.0"/><text class="t-title" x="272.0" y="194.0" text-anchor="middle">2022</text><text class="t-note" x="272.0" y="214.0" text-anchor="middle">Self-consistency</text><text class="t-muted" x="272.0" y="234.0" text-anchor="middle">sample many,</text><text class="t-muted" x="272.0" y="249.0" text-anchor="middle">take the vote</text><circle class="s1" cx="346" cy="150" r="7"/><line class="grid" x1="346.0" y1="141.0" x2="346.0" y2="124.0"/><text class="t-title" x="346.0" y="52.0" text-anchor="middle">2022</text><text class="t-note" x="346.0" y="72.0" text-anchor="middle">InstructGPT</text><text class="t-muted" x="346.0" y="92.0" text-anchor="middle">SFT, reward model,</text><text class="t-muted" x="346.0" y="107.0" text-anchor="middle">PPO (RLHF)</text><circle class="s2" cx="420" cy="150" r="7"/><line class="grid" x1="420.0" y1="159.0" x2="420.0" y2="176.0"/><text class="t-title" x="420.0" y="194.0" text-anchor="middle">2023</text><text class="t-note" x="420.0" y="214.0" text-anchor="middle">Process rewards</text><text class="t-muted" x="420.0" y="234.0" text-anchor="middle">a score for</text><text class="t-muted" x="420.0" y="249.0" text-anchor="middle">every step</text><circle class="s3" cx="494" cy="150" r="7"/><line class="grid" x1="494.0" y1="141.0" x2="494.0" y2="124.0"/><text class="t-title" x="494.0" y="52.0" text-anchor="middle">Feb 2024</text><text class="t-note" x="494.0" y="72.0" text-anchor="middle">DeepSeekMath</text><text class="t-muted" x="494.0" y="92.0" text-anchor="middle">GRPO: PPO</text><text class="t-muted" x="494.0" y="107.0" text-anchor="middle">without a critic</text><circle class="s2" cx="568" cy="150" r="7"/><line class="grid" x1="568.0" y1="159.0" x2="568.0" y2="176.0"/><text class="t-title" x="568.0" y="194.0" text-anchor="middle">Sep 2024</text><text class="t-note" x="568.0" y="214.0" text-anchor="middle">OpenAI o1</text><text class="t-muted" x="568.0" y="234.0" text-anchor="middle">long thinking,</text><text class="t-muted" x="568.0" y="249.0" text-anchor="middle">method not public</text><circle class="s3" cx="642" cy="150" r="7"/><line class="grid" x1="642.0" y1="141.0" x2="642.0" y2="124.0"/><text class="t-title" x="642.0" y="52.0" text-anchor="middle">Dec 2024</text><text class="t-note" x="642.0" y="72.0" text-anchor="middle">DeepSeek-V3</text><text class="t-muted" x="642.0" y="92.0" text-anchor="middle">the base model</text><text class="t-muted" x="642.0" y="107.0" text-anchor="middle">for R1</text><circle class="s3" cx="716" cy="150" r="7"/><line class="grid" x1="716.0" y1="159.0" x2="716.0" y2="176.0"/><text class="t-title" x="716.0" y="194.0" text-anchor="middle">Jan 2025</text><text class="t-note" x="716.0" y="214.0" text-anchor="middle">DeepSeek-R1</text><text class="t-muted" x="716.0" y="234.0" text-anchor="middle">reasoning from</text><text class="t-muted" x="716.0" y="249.0" text-anchor="middle">RL with rules</text><rect class="s1" x="24" y="281" width="10" height="10" rx="2"/><text class="t-tick" x="40.0" y="290.0" text-anchor="start">training methods</text><rect class="s2" x="177.2" y="281" width="10" height="10" rx="2"/><text class="t-tick" x="193.2" y="290.0" text-anchor="start">reasoning and test time</text><rect class="s3" x="380.79999999999995" y="281" width="10" height="10" rx="2"/><text class="t-tick" x="396.8" y="290.0" text-anchor="start">DeepSeek</text></svg><figcaption>From 2017 to 2025. Training methods (PPO, GPT-1, InstructGPT) give the machinery. Chain of thought, voting, process rewards and o1 show that thinking helps. DeepSeek's own pieces (GRPO, V3) come last, and R1 puts them together.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 760 152" role="img" aria-label="Four stages: pre-training gives a base model, SFT gives a chat model, then a reward model and RL with PPO make up RLHF."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><rect class="box-1" x="20.0" y="30.0" width="160.0" height="84.0" rx="10"/><text class="t-title" x="100.0" y="56.0" text-anchor="middle">Pre-training</text><text class="t-muted" x="100.0" y="80.0" text-anchor="middle">next token on</text><text class="t-muted" x="100.0" y="98.0" text-anchor="middle">trillions of tokens</text><rect class="box-1" x="205.0" y="30.0" width="160.0" height="84.0" rx="10"/><text class="t-title" x="285.0" y="56.0" text-anchor="middle">SFT</text><text class="t-muted" x="285.0" y="80.0" text-anchor="middle">copy written</text><text class="t-muted" x="285.0" y="98.0" text-anchor="middle">answers</text><line class="edge" x1="181.0" y1="72.0" x2="203.0" y2="72.0" marker-end="url(#ah)"/><rect class="box-2" x="390.0" y="30.0" width="160.0" height="84.0" rx="10"/><text class="t-title" x="470.0" y="56.0" text-anchor="middle">Reward model</text><text class="t-muted" x="470.0" y="80.0" text-anchor="middle">learn from</text><text class="t-muted" x="470.0" y="98.0" text-anchor="middle">human rankings</text><line class="edge" x1="366.0" y1="72.0" x2="388.0" y2="72.0" marker-end="url(#ah)"/><rect class="box-2" x="575.0" y="30.0" width="160.0" height="84.0" rx="10"/><text class="t-title" x="655.0" y="56.0" text-anchor="middle">RL with PPO</text><text class="t-muted" x="655.0" y="80.0" text-anchor="middle">raise the score,</text><text class="t-muted" x="655.0" y="98.0" text-anchor="middle">stay near SFT</text><line class="edge" x1="551.0" y1="72.0" x2="573.0" y2="72.0" marker-end="url(#ah)"/><text class="t-tick" x="100.0" y="140.0" text-anchor="middle">base model</text><text class="t-tick" x="285.0" y="140.0" text-anchor="middle">chat model</text><text class="t-tick" x="470.5" y="140.0" text-anchor="middle">RLHF (InstructGPT, 2022)</text></svg><figcaption>You are here: the four stages of the classic recipe. Pre-training makes a base model, SFT makes a chat model, and a reward model plus RL with PPO make up RLHF. R1-Zero will skip SFT and the learned reward model.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 760 260" role="img" aria-label="Accuracy of Qwen2.5-0.5B-Instruct on 30 GSM8K questions: a direct answer, one greedy chain of thought, one sampled chain, and majority votes over 4, 8 and 16 sampled chains."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="30.0" text-anchor="start">Qwen2.5-0.5B-Instruct, first 30 GSM8K test questions</text><text class="t-tick" x="142.0" y="61.0" text-anchor="end">direct answer</text><g class="mark"><title>direct answer: 3%</title><rect class="s2" x="150.0" y="43.0" width="15.7" height="26.0" rx="3"/></g><text class="t-val" x="171.7" y="61.0" text-anchor="start">3%</text><text class="t-tick" x="142.0" y="95.0" text-anchor="end">step by step</text><g class="mark"><title>step by step: 40%</title><rect class="s2" x="150.0" y="77.0" width="188.0" height="26.0" rx="3"/></g><text class="t-val" x="344.0" y="95.0" text-anchor="start">40%</text><text class="t-tick" x="142.0" y="129.0" text-anchor="end">one sample</text><g class="mark"><title>one sample: 26%</title><rect class="s2" x="150.0" y="111.0" width="122.4" height="26.0" rx="3"/></g><text class="t-val" x="278.4" y="129.0" text-anchor="start">26%</text><text class="t-tick" x="142.0" y="163.0" text-anchor="end">vote of 4</text><g class="mark"><title>vote of 4: 33%</title><rect class="s2" x="150.0" y="145.0" width="156.7" height="26.0" rx="3"/></g><text class="t-val" x="312.7" y="163.0" text-anchor="start">33%</text><text class="t-tick" x="142.0" y="197.0" text-anchor="end">vote of 8</text><g class="mark"><title>vote of 8: 37%</title><rect class="s2" x="150.0" y="179.0" width="172.3" height="26.0" rx="3"/></g><text class="t-val" x="328.3" y="197.0" text-anchor="start">37%</text><text class="t-tick t-strong" x="142.0" y="231.0" text-anchor="end">vote of 16</text><g class="mark"><title>vote of 16: 47%</title><rect class="s2" x="150.0" y="213.0" width="219.3" height="26.0" rx="3"/></g><text class="t-val" x="375.3" y="231.0" text-anchor="start">47%</text></svg><figcaption>Our run on 30 GSM8K questions with Qwen2.5-0.5B-Instruct. A direct answer is right once. One greedy chain of thought is right 12 times. A vote over 16 sampled chains is right 14 times.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 760 300" role="img" aria-label="Probability of the right answer during REINFORCE training, average of 200 seeds. With every reward shifted up by 5, training without a baseline stalls; with a baseline it learns as fast as with 0/1 rewards."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="250.0" x2="560" y2="250.0"/><text class="t-tick" x="56.0" y="254.0" text-anchor="end">0.00</text><line class="grid" x1="64" y1="195.0" x2="560" y2="195.0"/><text class="t-tick" x="56.0" y="199.0" text-anchor="end">0.25</text><line class="grid" x1="64" y1="140.0" x2="560" y2="140.0"/><text class="t-tick" x="56.0" y="144.0" text-anchor="end">0.50</text><line class="grid" x1="64" y1="85.0" x2="560" y2="85.0"/><text class="t-tick" x="56.0" y="89.0" text-anchor="end">0.75</text><line class="grid" x1="64" y1="30.0" x2="560" y2="30.0"/><text class="t-tick" x="56.0" y="34.0" text-anchor="end">1.00</text><text class="t-tick" x="64.0" y="270.0" text-anchor="middle">1</text><text class="t-tick" x="149.5" y="270.0" text-anchor="middle">51</text><text class="t-tick" x="235.0" y="270.0" text-anchor="middle">101</text><text class="t-tick" x="320.6" y="270.0" text-anchor="middle">151</text><text class="t-tick" x="406.1" y="270.0" text-anchor="middle">201</text><text class="t-tick" x="491.6" y="270.0" text-anchor="middle">251</text><text class="t-tick" x="560.0" y="270.0" text-anchor="middle">291</text><line class="axis" x1="64" y1="250" x2="560" y2="250"/><polyline class="l1" points="64.0,217.7 81.1,208.2 98.2,204.7 115.3,199.3 132.4,193.6 149.5,189.5 166.6,182.0 183.7,180.2 200.8,176.7 217.9,171.5 235.0,168.3 252.1,168.0 269.2,166.9 286.3,165.2 303.4,166.2 320.6,168.3 337.7,165.2 354.8,163.3 371.9,161.3 389.0,158.1 406.1,157.8 423.2,157.4 440.3,157.5 457.4,157.1 474.5,155.4 491.6,155.6 508.7,157.2 525.8,156.8 542.9,156.9 560.0,157.0"/><g class="mark"><title>+5, no baseline, 1: 0.15</title><circle class="s1 ring" cx="64.0" cy="217.7" r="5"/></g><g class="mark"><title>+5, no baseline, 11: 0.19</title><circle class="s1 ring" cx="81.1" cy="208.2" r="5"/></g><g class="mark"><title>+5, no baseline, 21: 0.21</title><circle class="s1 ring" cx="98.2" cy="204.7" r="5"/></g><g class="mark"><title>+5, no baseline, 31: 0.23</title><circle class="s1 ring" cx="115.3" cy="199.3" r="5"/></g><g class="mark"><title>+5, no baseline, 41: 0.26</title><circle class="s1 ring" cx="132.4" cy="193.6" r="5"/></g><g class="mark"><title>+5, no baseline, 51: 0.28</title><circle class="s1 ring" cx="149.5" cy="189.5" r="5"/></g><g class="mark"><title>+5, no baseline, 61: 0.31</title><circle class="s1 ring" cx="166.6" cy="182.0" r="5"/></g><g class="mark"><title>+5, no baseline, 71: 0.32</title><circle class="s1 ring" cx="183.7" cy="180.2" r="5"/></g><g class="mark"><title>+5, no baseline, 81: 0.33</title><circle class="s1 ring" cx="200.8" cy="176.7" r="5"/></g><g class="mark"><title>+5, no baseline, 91: 0.36</title><circle class="s1 ring" cx="217.9" cy="171.5" r="5"/></g><g class="mark"><title>+5, no baseline, 101: 0.37</title><circle class="s1 ring" cx="235.0" cy="168.3" r="5"/></g><g class="mark"><title>+5, no baseline, 111: 0.37</title><circle class="s1 ring" cx="252.1" cy="168.0" r="5"/></g><g class="mark"><title>+5, no baseline, 121: 0.38</title><circle class="s1 ring" cx="269.2" cy="166.9" r="5"/></g><g class="mark"><title>+5, no baseline, 131: 0.39</title><circle class="s1 ring" cx="286.3" cy="165.2" r="5"/></g><g class="mark"><title>+5, no baseline, 141: 0.38</title><circle class="s1 ring" cx="303.4" cy="166.2" r="5"/></g><g class="mark"><title>+5, no baseline, 151: 0.37</title><circle class="s1 ring" cx="320.6" cy="168.3" r="5"/></g><g class="mark"><title>+5, no baseline, 161: 0.39</title><circle class="s1 ring" cx="337.7" cy="165.2" r="5"/></g><g class="mark"><title>+5, no baseline, 171: 0.39</title><circle class="s1 ring" cx="354.8" cy="163.3" r="5"/></g><g class="mark"><title>+5, no baseline, 181: 0.40</title><circle class="s1 ring" cx="371.9" cy="161.3" r="5"/></g><g class="mark"><title>+5, no baseline, 191: 0.42</title><circle class="s1 ring" cx="389.0" cy="158.1" r="5"/></g><g class="mark"><title>+5, no baseline, 201: 0.42</title><circle class="s1 ring" cx="406.1" cy="157.8" r="5"/></g><g class="mark"><title>+5, no baseline, 211: 0.42</title><circle class="s1 ring" cx="423.2" cy="157.4" r="5"/></g><g class="mark"><title>+5, no baseline, 221: 0.42</title><circle class="s1 ring" cx="440.3" cy="157.5" r="5"/></g><g class="mark"><title>+5, no baseline, 231: 0.42</title><circle class="s1 ring" cx="457.4" cy="157.1" r="5"/></g><g class="mark"><title>+5, no baseline, 241: 0.43</title><circle class="s1 ring" cx="474.5" cy="155.4" r="5"/></g><g class="mark"><title>+5, no baseline, 251: 0.43</title><circle class="s1 ring" cx="491.6" cy="155.6" r="5"/></g><g class="mark"><title>+5, no baseline, 261: 0.42</title><circle class="s1 ring" cx="508.7" cy="157.2" r="5"/></g><g class="mark"><title>+5, no baseline, 271: 0.42</title><circle class="s1 ring" cx="525.8" cy="156.8" r="5"/></g><g class="mark"><title>+5, no baseline, 281: 0.42</title><circle class="s1 ring" cx="542.9" cy="156.9" r="5"/></g><g class="mark"><title>+5, no baseline, 291: 0.42</title><circle class="s1 ring" cx="560.0" cy="157.0" r="5"/></g><polyline class="l2" points="64.0,217.7 81.1,209.6 98.2,202.9 115.3,195.0 132.4,185.5 149.5,175.6 166.6,165.1 183.7,153.8 200.8,142.4 217.9,131.0 235.0,120.3 252.1,110.5 269.2,100.8 286.3,92.8 303.4,84.8 320.6,78.1 337.7,72.1 354.8,66.9 371.9,62.4 389.0,58.5 406.1,55.2 423.2,52.4 440.3,50.0 457.4,48.0 474.5,46.2 491.6,44.8 508.7,43.5 525.8,42.6 542.9,41.7 560.0,41.0"/><g class="mark"><title>+5, baseline, 1: 0.15</title><circle class="s2 ring" cx="64.0" cy="217.7" r="5"/></g><g class="mark"><title>+5, baseline, 11: 0.18</title><circle class="s2 ring" cx="81.1" cy="209.6" r="5"/></g><g class="mark"><title>+5, baseline, 21: 0.21</title><circle class="s2 ring" cx="98.2" cy="202.9" r="5"/></g><g class="mark"><title>+5, baseline, 31: 0.25</title><circle class="s2 ring" cx="115.3" cy="195.0" r="5"/></g><g class="mark"><title>+5, baseline, 41: 0.29</title><circle class="s2 ring" cx="132.4" cy="185.5" r="5"/></g><g class="mark"><title>+5, baseline, 51: 0.34</title><circle class="s2 ring" cx="149.5" cy="175.6" r="5"/></g><g class="mark"><title>+5, baseline, 61: 0.39</title><circle class="s2 ring" cx="166.6" cy="165.1" r="5"/></g><g class="mark"><title>+5, baseline, 71: 0.44</title><circle class="s2 ring" cx="183.7" cy="153.8" r="5"/></g><g class="mark"><title>+5, baseline, 81: 0.49</title><circle class="s2 ring" cx="200.8" cy="142.4" r="5"/></g><g class="mark"><title>+5, baseline, 91: 0.54</title><circle class="s2 ring" cx="217.9" cy="131.0" r="5"/></g><g class="mark"><title>+5, baseline, 101: 0.59</title><circle class="s2 ring" cx="235.0" cy="120.3" r="5"/></g><g class="mark"><title>+5, baseline, 111: 0.63</title><circle class="s2 ring" cx="252.1" cy="110.5" r="5"/></g><g class="mark"><title>+5, baseline, 121: 0.68</title><circle class="s2 ring" cx="269.2" cy="100.8" r="5"/></g><g class="mark"><title>+5, baseline, 131: 0.71</title><circle class="s2 ring" cx="286.3" cy="92.8" r="5"/></g><g class="mark"><title>+5, baseline, 141: 0.75</title><circle class="s2 ring" cx="303.4" cy="84.8" r="5"/></g><g class="mark"><title>+5, baseline, 151: 0.78</title><circle class="s2 ring" cx="320.6" cy="78.1" r="5"/></g><g class="mark"><title>+5, baseline, 161: 0.81</title><circle class="s2 ring" cx="337.7" cy="72.1" r="5"/></g><g class="mark"><title>+5, baseline, 171: 0.83</title><circle class="s2 ring" cx="354.8" cy="66.9" r="5"/></g><g class="mark"><title>+5, baseline, 181: 0.85</title><circle class="s2 ring" cx="371.9" cy="62.4" r="5"/></g><g class="mark"><title>+5, baseline, 191: 0.87</title><circle class="s2 ring" cx="389.0" cy="58.5" r="5"/></g><g class="mark"><title>+5, baseline, 201: 0.89</title><circle class="s2 ring" cx="406.1" cy="55.2" r="5"/></g><g class="mark"><title>+5, baseline, 211: 0.90</title><circle class="s2 ring" cx="423.2" cy="52.4" r="5"/></g><g class="mark"><title>+5, baseline, 221: 0.91</title><circle class="s2 ring" cx="440.3" cy="50.0" r="5"/></g><g class="mark"><title>+5, baseline, 231: 0.92</title><circle class="s2 ring" cx="457.4" cy="48.0" r="5"/></g><g class="mark"><title>+5, baseline, 241: 0.93</title><circle class="s2 ring" cx="474.5" cy="46.2" r="5"/></g><g class="mark"><title>+5, baseline, 251: 0.93</title><circle class="s2 ring" cx="491.6" cy="44.8" r="5"/></g><g class="mark"><title>+5, baseline, 261: 0.94</title><circle class="s2 ring" cx="508.7" cy="43.5" r="5"/></g><g class="mark"><title>+5, baseline, 271: 0.94</title><circle class="s2 ring" cx="525.8" cy="42.6" r="5"/></g><g class="mark"><title>+5, baseline, 281: 0.95</title><circle class="s2 ring" cx="542.9" cy="41.7" r="5"/></g><g class="mark"><title>+5, baseline, 291: 0.95</title><circle class="s2 ring" cx="560.0" cy="41.0" r="5"/></g><polyline class="l3" points="64.0,218.2 81.1,213.5 98.2,206.8 115.3,198.7 132.4,188.3 149.5,175.4 166.6,160.6 183.7,143.7 200.8,126.8 217.9,110.5 235.0,96.6 252.1,85.3 269.2,76.3 286.3,69.0 303.4,63.4 320.6,59.0 337.7,55.4 354.8,52.6 371.9,50.3 389.0,48.3 406.1,46.7 423.2,45.4 440.3,44.2 457.4,43.2 474.5,42.3 491.6,41.5 508.7,40.8 525.8,40.2 542.9,39.7 560.0,39.2"/><g class="mark"><title>0/1, no baseline, 1: 0.14</title><circle class="s3 ring" cx="64.0" cy="218.2" r="5"/></g><g class="mark"><title>0/1, no baseline, 11: 0.17</title><circle class="s3 ring" cx="81.1" cy="213.5" r="5"/></g><g class="mark"><title>0/1, no baseline, 21: 0.20</title><circle class="s3 ring" cx="98.2" cy="206.8" r="5"/></g><g class="mark"><title>0/1, no baseline, 31: 0.23</title><circle class="s3 ring" cx="115.3" cy="198.7" r="5"/></g><g class="mark"><title>0/1, no baseline, 41: 0.28</title><circle class="s3 ring" cx="132.4" cy="188.3" r="5"/></g><g class="mark"><title>0/1, no baseline, 51: 0.34</title><circle class="s3 ring" cx="149.5" cy="175.4" r="5"/></g><g class="mark"><title>0/1, no baseline, 61: 0.41</title><circle class="s3 ring" cx="166.6" cy="160.6" r="5"/></g><g class="mark"><title>0/1, no baseline, 71: 0.48</title><circle class="s3 ring" cx="183.7" cy="143.7" r="5"/></g><g class="mark"><title>0/1, no baseline, 81: 0.56</title><circle class="s3 ring" cx="200.8" cy="126.8" r="5"/></g><g class="mark"><title>0/1, no baseline, 91: 0.63</title><circle class="s3 ring" cx="217.9" cy="110.5" r="5"/></g><g class="mark"><title>0/1, no baseline, 101: 0.70</title><circle class="s3 ring" cx="235.0" cy="96.6" r="5"/></g><g class="mark"><title>0/1, no baseline, 111: 0.75</title><circle class="s3 ring" cx="252.1" cy="85.3" r="5"/></g><g class="mark"><title>0/1, no baseline, 121: 0.79</title><circle class="s3 ring" cx="269.2" cy="76.3" r="5"/></g><g class="mark"><title>0/1, no baseline, 131: 0.82</title><circle class="s3 ring" cx="286.3" cy="69.0" r="5"/></g><g class="mark"><title>0/1, no baseline, 141: 0.85</title><circle class="s3 ring" cx="303.4" cy="63.4" r="5"/></g><g class="mark"><title>0/1, no baseline, 151: 0.87</title><circle class="s3 ring" cx="320.6" cy="59.0" r="5"/></g><g class="mark"><title>0/1, no baseline, 161: 0.88</title><circle class="s3 ring" cx="337.7" cy="55.4" r="5"/></g><g class="mark"><title>0/1, no baseline, 171: 0.90</title><circle class="s3 ring" cx="354.8" cy="52.6" r="5"/></g><g class="mark"><title>0/1, no baseline, 181: 0.91</title><circle class="s3 ring" cx="371.9" cy="50.3" r="5"/></g><g class="mark"><title>0/1, no baseline, 191: 0.92</title><circle class="s3 ring" cx="389.0" cy="48.3" r="5"/></g><g class="mark"><title>0/1, no baseline, 201: 0.92</title><circle class="s3 ring" cx="406.1" cy="46.7" r="5"/></g><g class="mark"><title>0/1, no baseline, 211: 0.93</title><circle class="s3 ring" cx="423.2" cy="45.4" r="5"/></g><g class="mark"><title>0/1, no baseline, 221: 0.94</title><circle class="s3 ring" cx="440.3" cy="44.2" r="5"/></g><g class="mark"><title>0/1, no baseline, 231: 0.94</title><circle class="s3 ring" cx="457.4" cy="43.2" r="5"/></g><g class="mark"><title>0/1, no baseline, 241: 0.94</title><circle class="s3 ring" cx="474.5" cy="42.3" r="5"/></g><g class="mark"><title>0/1, no baseline, 251: 0.95</title><circle class="s3 ring" cx="491.6" cy="41.5" r="5"/></g><g class="mark"><title>0/1, no baseline, 261: 0.95</title><circle class="s3 ring" cx="508.7" cy="40.8" r="5"/></g><g class="mark"><title>0/1, no baseline, 271: 0.95</title><circle class="s3 ring" cx="525.8" cy="40.2" r="5"/></g><g class="mark"><title>0/1, no baseline, 281: 0.96</title><circle class="s3 ring" cx="542.9" cy="39.7" r="5"/></g><g class="mark"><title>0/1, no baseline, 291: 0.96</title><circle class="s3 ring" cx="560.0" cy="39.2" r="5"/></g><line class="grid" x1="566.0" y1="39.2" x2="576.0" y2="39.2"/><rect class="s3" x="580.0" y="34.2" width="10" height="10" rx="2"/><text class="t-note" x="596.0" y="43.2" text-anchor="start">0/1, no baseline: 0.96</text><line class="grid" x1="566.0" y1="41.0" x2="576.0" y2="56.2"/><rect class="s2" x="580.0" y="51.2" width="10" height="10" rx="2"/><text class="t-note" x="596.0" y="60.2" text-anchor="start">+5, baseline: 0.95</text><line class="grid" x1="566.0" y1="157.0" x2="576.0" y2="157.0"/><rect class="s1" x="580.0" y="152.0" width="10" height="10" rx="2"/><text class="t-note" x="596.0" y="161.0" text-anchor="start">+5, no baseline: 0.42</text><text class="t-tick" x="312.0" y="290.0" text-anchor="middle">training step (one sample per step)</text><text class="t-tick" x="14.0" y="18.0" text-anchor="start">P(right answer), mean of 200 seeds</text></svg><figcaption>Probability of the right answer while training, mean of 200 seeds. With 0/1 rewards, and with shifted rewards plus a baseline, the policy reaches about 0.95. With shifted rewards and no baseline, 112 of 200 seeds end below 0.5, and one seed ends at 0.001.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 760 312" role="img" aria-label="One term of the PPO clipped objective against the probability ratio. For a good answer the gain stops at ratio 1.2; for a bad answer it stops at 0.8. The dashed line is the unclipped product."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="200.0" y="22.0" text-anchor="middle">advantage A = +1 (a good answer)</text><line class="axis" x1="60.0" y1="140.0" x2="340.0" y2="140.0"/><line class="grid" x1="146.2" y1="40.0" x2="146.2" y2="240.0"/><text class="t-tick" x="146.2" y="256.0" text-anchor="middle">0.8</text><line class="grid" x1="189.2" y1="40.0" x2="189.2" y2="240.0"/><text class="t-tick" x="189.2" y="256.0" text-anchor="middle">1.0</text><line class="grid" x1="232.3" y1="40.0" x2="232.3" y2="240.0"/><text class="t-tick" x="232.3" y="256.0" text-anchor="middle">1.2</text><polyline class="l3" style="stroke-dasharray:5 4" points="60.0,117.8 62.2,117.2 64.3,116.7 66.5,116.1 68.6,115.6 70.8,115.0 72.9,114.4 75.1,113.9 77.2,113.3 79.4,112.8 81.5,112.2 83.7,111.7 85.8,111.1 88.0,110.6 90.2,110.0 92.3,109.4 94.5,108.9 96.6,108.3 98.8,107.8 100.9,107.2 103.1,106.7 105.2,106.1 107.4,105.6 109.5,105.0 111.7,104.4 113.8,103.9 116.0,103.3 118.2,102.8 120.3,102.2 122.5,101.7 124.6,101.1 126.8,100.6 128.9,100.0 131.1,99.4 133.2,98.9 135.4,98.3 137.5,97.8 139.7,97.2 141.8,96.7 144.0,96.1 146.2,95.6 148.3,95.0 150.5,94.4 152.6,93.9 154.8,93.3 156.9,92.8 159.1,92.2 161.2,91.7 163.4,91.1 165.5,90.6 167.7,90.0 169.8,89.4 172.0,88.9 174.2,88.3 176.3,87.8 178.5,87.2 180.6,86.7 182.8,86.1 184.9,85.6 187.1,85.0 189.2,84.4 191.4,83.9 193.5,83.3 195.7,82.8 197.8,82.2 200.0,81.7 202.2,81.1 204.3,80.6 206.5,80.0 208.6,79.4 210.8,78.9 212.9,78.3 215.1,77.8 217.2,77.2 219.4,76.7 221.5,76.1 223.7,75.6 225.8,75.0 228.0,74.4 230.2,73.9 232.3,73.3 234.5,72.8 236.6,72.2 238.8,71.7 240.9,71.1 243.1,70.6 245.2,70.0 247.4,69.4 249.5,68.9 251.7,68.3 253.8,67.8 256.0,67.2 258.2,66.7 260.3,66.1 262.5,65.6 264.6,65.0 266.8,64.4 268.9,63.9 271.1,63.3 273.2,62.8 275.4,62.2 277.5,61.7 279.7,61.1 281.8,60.6 284.0,60.0 286.2,59.4 288.3,58.9 290.5,58.3 292.6,57.8 294.8,57.2 296.9,56.7 299.1,56.1 301.2,55.6 303.4,55.0 305.5,54.4 307.7,53.9 309.8,53.3 312.0,52.8 314.2,52.2 316.3,51.7 318.5,51.1 320.6,50.6 322.8,50.0 324.9,49.4 327.1,48.9 329.2,48.3 331.4,47.8 333.5,47.2 335.7,46.7 337.8,46.1 340.0,45.6"/><polyline class="l1" points="60.0,117.8 62.2,117.2 64.3,116.7 66.5,116.1 68.6,115.6 70.8,115.0 72.9,114.4 75.1,113.9 77.2,113.3 79.4,112.8 81.5,112.2 83.7,111.7 85.8,111.1 88.0,110.6 90.2,110.0 92.3,109.4 94.5,108.9 96.6,108.3 98.8,107.8 100.9,107.2 103.1,106.7 105.2,106.1 107.4,105.6 109.5,105.0 111.7,104.4 113.8,103.9 116.0,103.3 118.2,102.8 120.3,102.2 122.5,101.7 124.6,101.1 126.8,100.6 128.9,100.0 131.1,99.4 133.2,98.9 135.4,98.3 137.5,97.8 139.7,97.2 141.8,96.7 144.0,96.1 146.2,95.6 148.3,95.0 150.5,94.4 152.6,93.9 154.8,93.3 156.9,92.8 159.1,92.2 161.2,91.7 163.4,91.1 165.5,90.6 167.7,90.0 169.8,89.4 172.0,88.9 174.2,88.3 176.3,87.8 178.5,87.2 180.6,86.7 182.8,86.1 184.9,85.6 187.1,85.0 189.2,84.4 191.4,83.9 193.5,83.3 195.7,82.8 197.8,82.2 200.0,81.7 202.2,81.1 204.3,80.6 206.5,80.0 208.6,79.4 210.8,78.9 212.9,78.3 215.1,77.8 217.2,77.2 219.4,76.7 221.5,76.1 223.7,75.6 225.8,75.0 228.0,74.4 230.2,73.9 232.3,73.3 234.5,73.3 236.6,73.3 238.8,73.3 240.9,73.3 243.1,73.3 245.2,73.3 247.4,73.3 249.5,73.3 251.7,73.3 253.8,73.3 256.0,73.3 258.2,73.3 260.3,73.3 262.5,73.3 264.6,73.3 266.8,73.3 268.9,73.3 271.1,73.3 273.2,73.3 275.4,73.3 277.5,73.3 279.7,73.3 281.8,73.3 284.0,73.3 286.2,73.3 288.3,73.3 290.5,73.3 292.6,73.3 294.8,73.3 296.9,73.3 299.1,73.3 301.2,73.3 303.4,73.3 305.5,73.3 307.7,73.3 309.8,73.3 312.0,73.3 314.2,73.3 316.3,73.3 318.5,73.3 320.6,73.3 322.8,73.3 324.9,73.3 327.1,73.3 329.2,73.3 331.4,73.3 333.5,73.3 335.7,73.3 337.8,73.3 340.0,73.3"/><circle class="s1" cx="189.2" cy="84.4" r="5"/><text class="t-tick" x="200.0" y="276.0" text-anchor="middle">ratio = new probability / old probability</text><text class="t-note" x="286.2" y="63.3" text-anchor="middle">flat: no gain past 1.2</text><text class="t-title" x="570.0" y="22.0" text-anchor="middle">advantage A = -1 (a bad answer)</text><line class="axis" x1="430.0" y1="140.0" x2="710.0" y2="140.0"/><line class="grid" x1="516.2" y1="40.0" x2="516.2" y2="240.0"/><text class="t-tick" x="516.2" y="256.0" text-anchor="middle">0.8</text><line class="grid" x1="559.2" y1="40.0" x2="559.2" y2="240.0"/><text class="t-tick" x="559.2" y="256.0" text-anchor="middle">1.0</text><line class="grid" x1="602.3" y1="40.0" x2="602.3" y2="240.0"/><text class="t-tick" x="602.3" y="256.0" text-anchor="middle">1.2</text><polyline class="l3" style="stroke-dasharray:5 4" points="430.0,162.2 432.2,162.8 434.3,163.3 436.5,163.9 438.6,164.4 440.8,165.0 442.9,165.6 445.1,166.1 447.2,166.7 449.4,167.2 451.5,167.8 453.7,168.3 455.8,168.9 458.0,169.4 460.2,170.0 462.3,170.6 464.5,171.1 466.6,171.7 468.8,172.2 470.9,172.8 473.1,173.3 475.2,173.9 477.4,174.4 479.5,175.0 481.7,175.6 483.8,176.1 486.0,176.7 488.2,177.2 490.3,177.8 492.5,178.3 494.6,178.9 496.8,179.4 498.9,180.0 501.1,180.6 503.2,181.1 505.4,181.7 507.5,182.2 509.7,182.8 511.8,183.3 514.0,183.9 516.2,184.4 518.3,185.0 520.5,185.6 522.6,186.1 524.8,186.7 526.9,187.2 529.1,187.8 531.2,188.3 533.4,188.9 535.5,189.4 537.7,190.0 539.8,190.6 542.0,191.1 544.2,191.7 546.3,192.2 548.5,192.8 550.6,193.3 552.8,193.9 554.9,194.4 557.1,195.0 559.2,195.6 561.4,196.1 563.5,196.7 565.7,197.2 567.8,197.8 570.0,198.3 572.2,198.9 574.3,199.4 576.5,200.0 578.6,200.6 580.8,201.1 582.9,201.7 585.1,202.2 587.2,202.8 589.4,203.3 591.5,203.9 593.7,204.4 595.8,205.0 598.0,205.6 600.2,206.1 602.3,206.7 604.5,207.2 606.6,207.8 608.8,208.3 610.9,208.9 613.1,209.4 615.2,210.0 617.4,210.6 619.5,211.1 621.7,211.7 623.8,212.2 626.0,212.8 628.2,213.3 630.3,213.9 632.5,214.4 634.6,215.0 636.8,215.6 638.9,216.1 641.1,216.7 643.2,217.2 645.4,217.8 647.5,218.3 649.7,218.9 651.8,219.4 654.0,220.0 656.2,220.6 658.3,221.1 660.5,221.7 662.6,222.2 664.8,222.8 666.9,223.3 669.1,223.9 671.2,224.4 673.4,225.0 675.5,225.6 677.7,226.1 679.8,226.7 682.0,227.2 684.2,227.8 686.3,228.3 688.5,228.9 690.6,229.4 692.8,230.0 694.9,230.6 697.1,231.1 699.2,231.7 701.4,232.2 703.5,232.8 705.7,233.3 707.8,233.9 710.0,234.4"/><polyline class="l1" points="430.0,184.4 432.2,184.4 434.3,184.4 436.5,184.4 438.6,184.4 440.8,184.4 442.9,184.4 445.1,184.4 447.2,184.4 449.4,184.4 451.5,184.4 453.7,184.4 455.8,184.4 458.0,184.4 460.2,184.4 462.3,184.4 464.5,184.4 466.6,184.4 468.8,184.4 470.9,184.4 473.1,184.4 475.2,184.4 477.4,184.4 479.5,184.4 481.7,184.4 483.8,184.4 486.0,184.4 488.2,184.4 490.3,184.4 492.5,184.4 494.6,184.4 496.8,184.4 498.9,184.4 501.1,184.4 503.2,184.4 505.4,184.4 507.5,184.4 509.7,184.4 511.8,184.4 514.0,184.4 516.2,184.4 518.3,185.0 520.5,185.6 522.6,186.1 524.8,186.7 526.9,187.2 529.1,187.8 531.2,188.3 533.4,188.9 535.5,189.4 537.7,190.0 539.8,190.6 542.0,191.1 544.2,191.7 546.3,192.2 548.5,192.8 550.6,193.3 552.8,193.9 554.9,194.4 557.1,195.0 559.2,195.6 561.4,196.1 563.5,196.7 565.7,197.2 567.8,197.8 570.0,198.3 572.2,198.9 574.3,199.4 576.5,200.0 578.6,200.6 580.8,201.1 582.9,201.7 585.1,202.2 587.2,202.8 589.4,203.3 591.5,203.9 593.7,204.4 595.8,205.0 598.0,205.6 600.2,206.1 602.3,206.7 604.5,207.2 606.6,207.8 608.8,208.3 610.9,208.9 613.1,209.4 615.2,210.0 617.4,210.6 619.5,211.1 621.7,211.7 623.8,212.2 626.0,212.8 628.2,213.3 630.3,213.9 632.5,214.4 634.6,215.0 636.8,215.6 638.9,216.1 641.1,216.7 643.2,217.2 645.4,217.8 647.5,218.3 649.7,218.9 651.8,219.4 654.0,220.0 656.2,220.6 658.3,221.1 660.5,221.7 662.6,222.2 664.8,222.8 666.9,223.3 669.1,223.9 671.2,224.4 673.4,225.0 675.5,225.6 677.7,226.1 679.8,226.7 682.0,227.2 684.2,227.8 686.3,228.3 688.5,228.9 690.6,229.4 692.8,230.0 694.9,230.6 697.1,231.1 699.2,231.7 701.4,232.2 703.5,232.8 705.7,233.3 707.8,233.9 710.0,234.4"/><circle class="s1" cx="559.2" cy="195.6" r="5"/><text class="t-tick" x="570.0" y="276.0" text-anchor="middle">ratio = new probability / old probability</text><text class="t-note" x="477.4" y="216.7" text-anchor="middle">flat: no gain below 0.8</text><rect class="s1" x="60" y="291" width="10" height="10" rx="2"/><text class="t-tick" x="76.0" y="300.0" text-anchor="start">PPO clipped objective</text><rect class="s3" x="249.20000000000002" y="291" width="10" height="10" rx="2"/><text class="t-tick" x="265.2" y="300.0" text-anchor="start">unclipped ratio x A</text></svg><figcaption>One term of the PPO objective against the probability ratio. Left: for a good answer, the gain is flat after 1.2. Right: for a bad answer, it is flat below 0.8. The dot is the start, ratio 1. The dashed line is the plain product without the clip.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 760 350" role="img" aria-label="RLHF with PPO. The policy writes a response. A frozen reference model gives a KL penalty, a frozen reward model gives a score, and a trained critic gives a baseline. PPO uses them to update the policy."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><rect class="box" x="20.0" y="128.0" width="90.0" height="34.0" rx="6"/><text class="t-note" x="65.0" y="149.5" text-anchor="middle">prompt x</text><rect class="box-on" x="160.0" y="110.0" width="150.0" height="70.0" rx="10"/><text class="t-title" x="235.0" y="138.0" text-anchor="middle">Policy</text><text class="t-muted" x="235.0" y="160.0" text-anchor="middle">trained (the LLM)</text><line class="path" x1="112.0" y1="145.0" x2="158.0" y2="145.0" marker-end="url(#ah-on)"/><rect class="box" x="360.0" y="128.0" width="110.0" height="34.0" rx="6"/><text class="t-note" x="415.0" y="149.5" text-anchor="middle">response y</text><line class="path" x1="312.0" y1="145.0" x2="358.0" y2="145.0" marker-end="url(#ah-on)"/><rect class="box-1" x="20.0" y="250.0" width="160.0" height="66.0" rx="10"/><text class="t-title" x="100.0" y="276.0" text-anchor="middle">Reference</text><text class="t-muted" x="100.0" y="298.0" text-anchor="middle">frozen copy of SFT</text><text class="t-note" x="100.0" y="336.0" text-anchor="middle">KL penalty</text><line class="edge" x1="415.0" y1="164.0" x2="100.0" y2="248.0" marker-end="url(#ah)"/><rect class="box-1" x="300.0" y="250.0" width="160.0" height="66.0" rx="10"/><text class="t-title" x="380.0" y="276.0" text-anchor="middle">Reward model</text><text class="t-muted" x="380.0" y="298.0" text-anchor="middle">frozen, from rankings</text><text class="t-note" x="380.0" y="336.0" text-anchor="middle">score r(x, y)</text><line class="edge" x1="415.0" y1="164.0" x2="380.0" y2="248.0" marker-end="url(#ah)"/><rect class="box-on" x="580.0" y="250.0" width="160.0" height="66.0" rx="10"/><text class="t-title" x="660.0" y="276.0" text-anchor="middle">Critic (value)</text><text class="t-muted" x="660.0" y="298.0" text-anchor="middle">trained, same size</text><text class="t-note" x="660.0" y="336.0" text-anchor="middle">baseline V</text><line class="edge" x1="415.0" y1="164.0" x2="660.0" y2="248.0" marker-end="url(#ah)"/><rect class="box-2" x="560.0" y="108.0" width="180.0" height="74.0" rx="10"/><text class="t-title" x="650.0" y="134.0" text-anchor="middle">PPO update</text><text class="t-muted" x="650.0" y="154.0" text-anchor="middle">advantage = reward</text><text class="t-muted" x="650.0" y="170.0" text-anchor="middle">minus baseline, clipped</text><line class="edge" x1="472.0" y1="145.0" x2="558.0" y2="145.0" marker-end="url(#ah)"/><line class="path" x1="650.0" y1="106.0" x2="650.0" y2="70.0"/><line class="path" x1="650.0" y1="70.0" x2="235.0" y2="70.0"/><line class="path" x1="235.0" y1="70.0" x2="235.0" y2="108.0" marker-end="url(#ah-on)"/><text class="t-tick" x="440.0" y="62.0" text-anchor="middle">new weights for the policy</text><rect class="box-on" x="20" y="21" width="10" height="10" rx="2"/><text class="t-tick" x="36.0" y="30.0" text-anchor="start">trained (two big models)</text><rect class="box-1" x="230.8" y="21" width="10" height="10" rx="2"/><text class="t-tick" x="246.8" y="30.0" text-anchor="start">frozen (two more models)</text></svg><figcaption>RLHF with PPO and its four models. The policy writes a response. The frozen reference model gives the KL penalty, the frozen reward model gives the score, and the trained critic gives the baseline. PPO combines them into one update of the policy. Two of the four models are trained.</figcaption></figure>

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
