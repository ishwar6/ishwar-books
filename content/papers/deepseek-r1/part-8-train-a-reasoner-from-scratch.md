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

<figure class="fig"><svg viewBox="0 0 700 280" role="img" aria-label="The minir1 package as a loop. template.py turns a question into the Table 1 prompt; sample_group in grpo.py samples G answers; rewards.py scores them; the update part of grpo.py turns rewards into advantages and a loss; train.py repeats this; eval.py measures held-out questions. Highlighted: none."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><rect class="box" x="20.0" y="30.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="120.0" y="60.0" text-anchor="middle">template.py</text><text class="t-tick" x="120.0" y="82.0" text-anchor="middle">Table 1: the prompt</text><text class="t-tick" x="120.0" y="100.0" text-anchor="middle">question → text</text><rect class="box" x="250.0" y="30.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="350.0" y="60.0" text-anchor="middle">grpo.py: sample_group</text><text class="t-tick" x="350.0" y="82.0" text-anchor="middle">G answers per question</text><text class="t-tick" x="350.0" y="100.0" text-anchor="middle">from π_old</text><rect class="box" x="480.0" y="30.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="580.0" y="60.0" text-anchor="middle">rewards.py</text><text class="t-tick" x="580.0" y="82.0" text-anchor="middle">accuracy + format</text><text class="t-tick" x="580.0" y="100.0" text-anchor="middle">rules, no network</text><rect class="box" x="480.0" y="170.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="580.0" y="200.0" text-anchor="middle">grpo.py: Eq. 3, 1, 2</text><text class="t-tick" x="580.0" y="222.0" text-anchor="middle">advantages, clipped ratio,</text><text class="t-tick" x="580.0" y="240.0" text-anchor="middle">KL to π_ref → loss</text><rect class="box" x="250.0" y="170.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="350.0" y="200.0" text-anchor="middle">train.py</text><text class="t-tick" x="350.0" y="222.0" text-anchor="middle">the loop, the optimiser,</text><text class="t-tick" x="350.0" y="240.0" text-anchor="middle">one log line per step</text><rect class="box" x="20.0" y="170.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="120.0" y="200.0" text-anchor="middle">eval.py</text><text class="t-tick" x="120.0" y="222.0" text-anchor="middle">200 held-out questions</text><text class="t-tick" x="120.0" y="240.0" text-anchor="middle">before and after</text><line class="edge" x1="220.0" y1="75.0" x2="248.0" y2="75.0" marker-end="url(#ah)"/><line class="edge" x1="450.0" y1="75.0" x2="478.0" y2="75.0" marker-end="url(#ah)"/><line class="edge" x1="580.0" y1="120.0" x2="580.0" y2="168.0" marker-end="url(#ah)"/><line class="edge" x1="480.0" y1="215.0" x2="452.0" y2="215.0" marker-end="url(#ah)"/><line class="path" x1="350.0" y1="170.0" x2="350.0" y2="122.0" marker-end="url(#ah-on)"/><line class="edge" x1="250.0" y1="215.0" x2="222.0" y2="215.0" marker-end="url(#ah)"/><text class="t-tick" x="358.0" y="150.0" text-anchor="start">next step</text></svg><figcaption>The minir1 package as one loop. A question goes through the template, the sampler writes G answers, the rules score them, GRPO turns the scores into a loss, and train.py repeats this. eval.py measures held-out questions before and after.</figcaption></figure>

| File | Lines | What it holds | Paper |
|---|---|---|---|
| `template.py` | 22 | The prompt of Table 1 | §2.2.3 |
| `rewards.py` | 81 | The accuracy rule and the format rule | §2.2.2 |
| `grpo.py` | 87 | Sampling, log-probabilities, Eq. 3, Eq. 1 and Eq. 2 | §2.2.1 |
| `train.py` | 109 | The data, the loop and the log | §2.2 |
| `eval.py` | 28 | Held-out questions, scored with the same rules | §2.2.4 |

The page walks through the files in the order a question meets them. For each piece you see the lines of the paper, the code, and a check.

## The template {§2.2.3}

<figure class="fig"><svg viewBox="0 0 700 280" role="img" aria-label="The minir1 package as a loop. template.py turns a question into the Table 1 prompt; sample_group in grpo.py samples G answers; rewards.py scores them; the update part of grpo.py turns rewards into advantages and a loss; train.py repeats this; eval.py measures held-out questions. Highlighted: template."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><rect class="box-on" x="20.0" y="30.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="120.0" y="60.0" text-anchor="middle">template.py</text><text class="t-tick" x="120.0" y="82.0" text-anchor="middle">Table 1: the prompt</text><text class="t-tick" x="120.0" y="100.0" text-anchor="middle">question → text</text><text class="t-note" x="120.0" y="22.0" text-anchor="middle">you are here</text><rect class="box" x="250.0" y="30.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="350.0" y="60.0" text-anchor="middle">grpo.py: sample_group</text><text class="t-tick" x="350.0" y="82.0" text-anchor="middle">G answers per question</text><text class="t-tick" x="350.0" y="100.0" text-anchor="middle">from π_old</text><rect class="box" x="480.0" y="30.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="580.0" y="60.0" text-anchor="middle">rewards.py</text><text class="t-tick" x="580.0" y="82.0" text-anchor="middle">accuracy + format</text><text class="t-tick" x="580.0" y="100.0" text-anchor="middle">rules, no network</text><rect class="box" x="480.0" y="170.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="580.0" y="200.0" text-anchor="middle">grpo.py: Eq. 3, 1, 2</text><text class="t-tick" x="580.0" y="222.0" text-anchor="middle">advantages, clipped ratio,</text><text class="t-tick" x="580.0" y="240.0" text-anchor="middle">KL to π_ref → loss</text><rect class="box" x="250.0" y="170.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="350.0" y="200.0" text-anchor="middle">train.py</text><text class="t-tick" x="350.0" y="222.0" text-anchor="middle">the loop, the optimiser,</text><text class="t-tick" x="350.0" y="240.0" text-anchor="middle">one log line per step</text><rect class="box" x="20.0" y="170.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="120.0" y="200.0" text-anchor="middle">eval.py</text><text class="t-tick" x="120.0" y="222.0" text-anchor="middle">200 held-out questions</text><text class="t-tick" x="120.0" y="240.0" text-anchor="middle">before and after</text><line class="edge" x1="220.0" y1="75.0" x2="248.0" y2="75.0" marker-end="url(#ah)"/><line class="edge" x1="450.0" y1="75.0" x2="478.0" y2="75.0" marker-end="url(#ah)"/><line class="edge" x1="580.0" y1="120.0" x2="580.0" y2="168.0" marker-end="url(#ah)"/><line class="edge" x1="480.0" y1="215.0" x2="452.0" y2="215.0" marker-end="url(#ah)"/><line class="path" x1="350.0" y1="170.0" x2="350.0" y2="122.0" marker-end="url(#ah-on)"/><line class="edge" x1="250.0" y1="215.0" x2="222.0" y2="215.0" marker-end="url(#ah)"/><text class="t-tick" x="358.0" y="150.0" text-anchor="start">next step</text></svg><figcaption>You are here: template.py turns a question into the text the model reads.</figcaption></figure>

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

```python
"""Table 1 of the paper: the only instruction DeepSeek-R1-Zero gets. It fixes the shape of the answer
(<think> ... </think> <answer> ... </answer>) and says nothing about how to reason."""

TEMPLATE = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, "
    "respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. "
    "User: {prompt}. Assistant:"
)
# Our one addition: the prompt ends with the opening tag, so a 0.5B model starts inside <think> (see the text).
PREFILL = ' <think>'


def make_prompt(question):
    """Put one question into the template. The text is fed to the model as is: no chat template, no examples."""
    return TEMPLATE.format(prompt=question.strip().rstrip('.')) + PREFILL


def full_completion(generated):
    """What the rewards see: the prefilled tag plus the generated text."""
    return '<think>' + generated
```

The constant is Table 1 word for word. We made one addition, and the code says so: the prompt ends with the opening tag ` <think>`. The model then starts inside the thinking block.

> [!DEFINITION] Prefill
> Text that we put at the start of the model's answer before it writes anything. The model continues from there. Here the prefill is the opening tag ` <think>`, so every answer starts in the thinking block.

Why the prefill? Without it, our 0.5B models rarely passed the format check. We sampled 4 answers to each of 24 training questions, at temperature 1, with at most 300 new tokens:

```text
Qwen/Qwen2.5-0.5B: 96 samples, accuracy reward 0.010, format reward 0.021, mean length 198 tokens, 45 s
Qwen/Qwen2.5-0.5B-Instruct: 96 samples, accuracy reward 0.031, format reward 0.000, mean length 215 tokens, 44 s
```

With the plain template, the base model passed the format check on 2 of 96 answers (0.021) and the instruction-tuned model on none. With the prefill, both start to pass it:

```text
Qwen/Qwen2.5-0.5B: 96 samples, accuracy reward 0.021, format reward 0.115, mean length 137 tokens, 45 s
Qwen/Qwen2.5-0.5B-Instruct: 96 samples, accuracy reward 0.052, format reward 0.427, mean length 141 tokens, 43 s
```

The base model now passes on 11 of 96 answers (0.115) and the Instruct model on 41 of 96 (0.427). GRPO's reward signal comes only from differences inside a group. If every answer in a group gets the same reward, the rewards push nothing up or down (we show why in the GRPO section). A start of 2% leaves almost every group with equal rewards. A start of 10% to 40% gives many groups a mix of good and bad answers.

The prefill does not tell the model how to reason, so it keeps the spirit of §2.2.3. The rewards still see the full text: `full_completion` puts the tag back in front of what the model wrote.

## The rewards {§2.2.2}

<figure class="fig"><svg viewBox="0 0 700 280" role="img" aria-label="The minir1 package as a loop. template.py turns a question into the Table 1 prompt; sample_group in grpo.py samples G answers; rewards.py scores them; the update part of grpo.py turns rewards into advantages and a loss; train.py repeats this; eval.py measures held-out questions. Highlighted: rewards."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><rect class="box" x="20.0" y="30.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="120.0" y="60.0" text-anchor="middle">template.py</text><text class="t-tick" x="120.0" y="82.0" text-anchor="middle">Table 1: the prompt</text><text class="t-tick" x="120.0" y="100.0" text-anchor="middle">question → text</text><rect class="box" x="250.0" y="30.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="350.0" y="60.0" text-anchor="middle">grpo.py: sample_group</text><text class="t-tick" x="350.0" y="82.0" text-anchor="middle">G answers per question</text><text class="t-tick" x="350.0" y="100.0" text-anchor="middle">from π_old</text><rect class="box-on" x="480.0" y="30.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="580.0" y="60.0" text-anchor="middle">rewards.py</text><text class="t-tick" x="580.0" y="82.0" text-anchor="middle">accuracy + format</text><text class="t-tick" x="580.0" y="100.0" text-anchor="middle">rules, no network</text><text class="t-note" x="580.0" y="22.0" text-anchor="middle">you are here</text><rect class="box" x="480.0" y="170.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="580.0" y="200.0" text-anchor="middle">grpo.py: Eq. 3, 1, 2</text><text class="t-tick" x="580.0" y="222.0" text-anchor="middle">advantages, clipped ratio,</text><text class="t-tick" x="580.0" y="240.0" text-anchor="middle">KL to π_ref → loss</text><rect class="box" x="250.0" y="170.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="350.0" y="200.0" text-anchor="middle">train.py</text><text class="t-tick" x="350.0" y="222.0" text-anchor="middle">the loop, the optimiser,</text><text class="t-tick" x="350.0" y="240.0" text-anchor="middle">one log line per step</text><rect class="box" x="20.0" y="170.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="120.0" y="200.0" text-anchor="middle">eval.py</text><text class="t-tick" x="120.0" y="222.0" text-anchor="middle">200 held-out questions</text><text class="t-tick" x="120.0" y="240.0" text-anchor="middle">before and after</text><line class="edge" x1="220.0" y1="75.0" x2="248.0" y2="75.0" marker-end="url(#ah)"/><line class="edge" x1="450.0" y1="75.0" x2="478.0" y2="75.0" marker-end="url(#ah)"/><line class="edge" x1="580.0" y1="120.0" x2="580.0" y2="168.0" marker-end="url(#ah)"/><line class="edge" x1="480.0" y1="215.0" x2="452.0" y2="215.0" marker-end="url(#ah)"/><line class="path" x1="350.0" y1="170.0" x2="350.0" y2="122.0" marker-end="url(#ah-on)"/><line class="edge" x1="250.0" y1="215.0" x2="222.0" y2="215.0" marker-end="url(#ah)"/><text class="t-tick" x="358.0" y="150.0" text-anchor="start">next step</text></svg><figcaption>You are here: rewards.py turns each answer into a number.</figcaption></figure>

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

```python
"""The two rule-based rewards of §2.2.2. No neural reward model: a few lines of Python decide each reward."""
import re
from fractions import Fraction

ANSWER_RE = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
TAG_RE = re.compile(r'</?(think|answer)>')
# One number: '-3', '12.5', '1/2', '-7/4', '1e3'. The lookarounds stop it from starting or ending inside a word.
NUMBER_RE = re.compile(r'(?<![\w.])(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?:\s*/\s*(\d+(?:\.\d+)?))?(?!\w|\.\d)')
OPERATOR_RE = re.compile(r'\d\s*(?:[-+*×÷^])\s*\(?-?\d')     # '10 - 2', '3 * 4', '2^3': an expression, not a number


def extract_answer(completion):
    """The text inside the last <answer> ... </answer> pair, or None when there is no such pair."""
    found = ANSWER_RE.findall(completion)
    return found[-1].strip() if found else None


def extract_boxed(text):
    """What is inside the last \\boxed{...}, counting braces so that '\\boxed{\\frac{1}{2}}' works. None if unclosed."""
    start = text.rfind('\\boxed')
    i = text.find('{', start) if start >= 0 else -1
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        depth += {'{': 1, '}': -1}.get(text[j], 0)
        if depth == 0:                                   # the brace that closes the box
            return text[i + 1:j]
    return None


def to_number(text):
    """The final value of an answer as an exact Fraction, or None when it is not one number.
    '$1,250.00' -> 1250, 'x = -3' -> -3, '3 + 4 = 7' -> 7, '1/2' -> 1/2, '\\boxed{36/2}' -> 18, '1e3' -> 1000,
    '10 - 2' -> None (an unworked expression), '2^3' -> None, 'none' -> None."""
    if '\\boxed' in text:
        text = extract_boxed(text)                       # the box is the final answer
        if text is None:
            return None
    text = text.split('=')[-1]                           # the final expression: what comes after the last '='
    text = re.sub(r'\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}', r'\1/\2', text)     # \frac{a}{b} -> a/b
    text = re.sub(r'\\text\{[^{}]*\}|\\[$%,!]|[$%]', ' ', text)               # unit text, dollar and per cent signs
    while re.search(r'\d,\d{3}(?!\d)', text):
        text = re.sub(r'(\d),(\d{3})(?!\d)', r'\1\2', text)                    # 1,250,000 -> 1250000
    if OPERATOR_RE.search(text):
        return None                                      # a sum or a power left unworked: no single number
    found = NUMBER_RE.findall(text)
    if not found:
        return None
    a, b = found[-1]                                     # the last number of the final expression
    value = Fraction(float(a)) if 'e' in a.lower() else Fraction(a)
    return None if b and float(b) == 0 else (value / Fraction(b) if b else value)


def accuracy_reward(completion, gold):
    """1.0 when the number in the <answer> tags equals the known answer, else 0.0 (the 'accuracy reward')."""
    answer = extract_answer(completion)
    if answer is None:
        return 0.0                                   # no answer tags: nothing to check
    value = to_number(answer)
    return float(value is not None and value == Fraction(gold))


def format_reward(completion):
    """1.0 when the completion is exactly <think> ... </think> <answer> ... </answer>: each tag once, in this order,
    both blocks non-empty, nothing before, between or after them (the 'format reward'). Part 3's tag parser."""
    tags = list(TAG_RE.finditer(completion))
    if [t.group(0) for t in tags] != ['<think>', '</think>', '<answer>', '</answer>']:
        return 0.0                                   # a tag is missing, repeated or out of order
    t1, t2, a1, a2 = tags
    if completion[:t1.start()].strip() or completion[t2.end():a1.start()].strip() or completion[a2.end():].strip():
        return 0.0                                   # text before, between or after the two blocks
    if not completion[t1.end():t2.start()].strip() or not completion[a1.end():a2.start()].strip():
        return 0.0                                   # an empty block
    return 1.0


def total_reward(completion, gold):
    """The reward GRPO sees. Version 1 of the paper names the two rules but not how they combine; the later version
    adds them with the same weight (its Eq. 4). So one answer earns 0, 1 or 2."""
    return accuracy_reward(completion, gold) + format_reward(completion)
```

`extract_answer` takes the last answer pair, so an answer that changes its mind is graded on its final word.

`to_number` reads the final value of the answer. If there is a `\boxed{...}`, it takes what is inside the last box, counting braces. It keeps only what comes after the last `=`, so "3 + 4 = 7" gives 7. It removes dollar signs and the commas of thousands, so a dollar price of 1,250.00 written with its sign gives 1250. A fraction counts as a fraction and `1e3` as a thousand. A sum or a power that was never worked out, such as "10 - 2", gives no number at all, and so no reward.

`format_reward` is the tag parser of [Part 3](part-3-rewards-and-the-template.md). It finds the four tags and asks for exactly `<think>`, `</think>`, `<answer>`, `</answer>`, once each and in this order. Both blocks must hold some text, and nothing may stand before, between or after them.

> [!NOTE] Our first version had two holes
> The first version of this file used the last number of the answer, and one regular expression for the format. Both paid for things they should not.
>
> `1/2` earned the accuracy point against a gold answer of 2, because the last number in "1/2" is 2. `1e3` earned it against a gold answer of 3. And `<think> </think><answer> </answer>` earned the format point with two empty blocks, as did an answer with two think blocks. Saved training answers had such shapes and got the point. The run on this page uses the fixed rules.

Here are the fixed rules on those hand-made cases:

```text
accuracy, answer             '18' gold   18: value    18  reward 1
accuracy, answer      '$1,250.00' gold 1250: value  1250  reward 1
accuracy, answer      '3 + 4 = 7' gold    7: value     7  reward 1
accuracy, answer            '1/2' gold    2: value   1/2  reward 0
accuracy, answer            '1e3' gold    3: value  1000  reward 0
accuracy, answer  '\\boxed{36/2}' gold   18: value    18  reward 1
accuracy, answer         '10 - 2' gold    8: value  None  reward 0
accuracy, answer            '2^3' gold    8: value  None  reward 0
format 1: '<think> 2 + 3 = 5 </think> <answer> 5 </answer>'
format 0: '<think> </think><answer> </answer>'
format 0: '<think> a </think><think> b </think><answer> 5 </answer>'
format 0: '<think> a </think><answer> 5 </answer> more'
format 0: '<think> a <answer> 5 </answer>'
```

Version 1 of the paper names the two rewards but does not say how they combine. The later, longer version adds them with the same weight (its Eq. 4). We do the same, so one answer earns 0, 1 or 2.

**Method 2, with libraries.** The math_verify library parses two maths expressions and checks if they are equal. It knows fractions, LaTeX and units.

We ran it and our accuracy rule on every answer of the held-out evaluation. Both read the text inside the last answer tags and compare it with the GSM8K answer. For the format rule we wrote a second, independent check. It is one regular expression: neither block may contain a tag, and each block must hold a non-space character.

```text
before: our accuracy rule and math_verify agree on 200 of 200 answers (both read the text inside the last answer tags)
before: our format parser and an independent regular expression agree on 200 of 200 answers
after: our accuracy rule and math_verify agree on 200 of 200 answers (both read the text inside the last answer tags)
after: our format parser and an independent regular expression agree on 200 of 200 answers
```

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

<figure class="fig"><svg viewBox="0 0 700 280" role="img" aria-label="The minir1 package as a loop. template.py turns a question into the Table 1 prompt; sample_group in grpo.py samples G answers; rewards.py scores them; the update part of grpo.py turns rewards into advantages and a loss; train.py repeats this; eval.py measures held-out questions. Highlighted: grpo_sample."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><rect class="box" x="20.0" y="30.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="120.0" y="60.0" text-anchor="middle">template.py</text><text class="t-tick" x="120.0" y="82.0" text-anchor="middle">Table 1: the prompt</text><text class="t-tick" x="120.0" y="100.0" text-anchor="middle">question → text</text><rect class="box-on" x="250.0" y="30.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="350.0" y="60.0" text-anchor="middle">grpo.py: sample_group</text><text class="t-tick" x="350.0" y="82.0" text-anchor="middle">G answers per question</text><text class="t-tick" x="350.0" y="100.0" text-anchor="middle">from π_old</text><text class="t-note" x="350.0" y="22.0" text-anchor="middle">you are here</text><rect class="box" x="480.0" y="30.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="580.0" y="60.0" text-anchor="middle">rewards.py</text><text class="t-tick" x="580.0" y="82.0" text-anchor="middle">accuracy + format</text><text class="t-tick" x="580.0" y="100.0" text-anchor="middle">rules, no network</text><rect class="box" x="480.0" y="170.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="580.0" y="200.0" text-anchor="middle">grpo.py: Eq. 3, 1, 2</text><text class="t-tick" x="580.0" y="222.0" text-anchor="middle">advantages, clipped ratio,</text><text class="t-tick" x="580.0" y="240.0" text-anchor="middle">KL to π_ref → loss</text><rect class="box" x="250.0" y="170.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="350.0" y="200.0" text-anchor="middle">train.py</text><text class="t-tick" x="350.0" y="222.0" text-anchor="middle">the loop, the optimiser,</text><text class="t-tick" x="350.0" y="240.0" text-anchor="middle">one log line per step</text><rect class="box" x="20.0" y="170.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="120.0" y="200.0" text-anchor="middle">eval.py</text><text class="t-tick" x="120.0" y="222.0" text-anchor="middle">200 held-out questions</text><text class="t-tick" x="120.0" y="240.0" text-anchor="middle">before and after</text><line class="edge" x1="220.0" y1="75.0" x2="248.0" y2="75.0" marker-end="url(#ah)"/><line class="edge" x1="450.0" y1="75.0" x2="478.0" y2="75.0" marker-end="url(#ah)"/><line class="edge" x1="580.0" y1="120.0" x2="580.0" y2="168.0" marker-end="url(#ah)"/><line class="edge" x1="480.0" y1="215.0" x2="452.0" y2="215.0" marker-end="url(#ah)"/><line class="path" x1="350.0" y1="170.0" x2="350.0" y2="122.0" marker-end="url(#ah-on)"/><line class="edge" x1="250.0" y1="215.0" x2="222.0" y2="215.0" marker-end="url(#ah)"/><text class="t-tick" x="358.0" y="150.0" text-anchor="start">next step</text></svg><figcaption>You are here: sample_group writes G answers to each question with the current model.</figcaption></figure>

> [!DEFINITION] Policy
> In reinforcement learning, the policy is the thing that chooses actions. Here the policy is the language model: each token it writes is an action. $$\pi_\theta(o \mid q)$$ is the probability that the model with weights $$\theta$$ writes the answer $$o$$ to the question $$q$$.

```python
@torch.no_grad()
def sample_group(model, tok, prompts, G, max_new_tokens, temperature=1.0):
    """Sample G completions for each prompt from the current policy (the 'old policy' of Eq. 1)."""
    texts = [p for p in prompts for _ in range(G)]               # B = len(prompts) * G, groups side by side
    enc = tok(texts, return_tensors='pt', padding=True, padding_side='left').to(model.device)
    out = model.generate(**enc, do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
                         repetition_penalty=1.0, max_new_tokens=max_new_tokens,
                         stop_strings=['</answer>'], tokenizer=tok, pad_token_id=tok.pad_token_id)
    prompt_ids, completion_ids = enc.input_ids, out[:, enc.input_ids.size(1):]          # (B, P), (B, T)
    lengths = completion_lengths(completion_ids.tolist(), tok, end_token_ids(model, tok))
    T = completion_ids.size(1)
    completion_mask = (torch.arange(T)[None, :] < torch.tensor(lengths)[:, None]).long().to(model.device)  # (B, T)
    texts = [full_completion(tok.decode(ids[:n], skip_special_tokens=True))            # only the real tokens
             for ids, n in zip(completion_ids.tolist(), lengths)]
    return dict(prompt_ids=prompt_ids, prompt_mask=enc.attention_mask,
                completion_ids=completion_ids, completion_mask=completion_mask, texts=texts)
```

The function repeats each prompt G times, so the G answers to one question sit side by side. Padding on the left lines up the ends of the prompts, which is where generation starts. We sample at temperature 1 with no top-p or top-k filter, so the answers come from the model's real distribution. The later version of the paper also uses temperature 1.

Generation stops at the string `</answer>`, or at an end-of-text token, or at the length limit. When one answer stops early, `generate` fills the rest of its row with padding. For Qwen the padding token is the same token as end-of-text, so the ids alone cannot tell a sampled end from a filler.

`completion_lengths` decides from how the row stopped. An answer that stopped on `</answer>` ends with the token that completes the string. An answer that stopped on a sampled end token keeps that token, because the model chose it. An answer that hit the limit keeps everything.

> [!NOTE] A bug we fixed
> Our first version marked everything up to and including the first end-or-padding token. After a stop on `</answer>`, that token is padding the model never chose. It still entered the loss, the KL and the length. The fixed mask, on three hand-made rows:

```text
mask, stopped on </answer>, then 2 padding tokens: tokens [' ', '5', ' </', 'answer', '>', '<|endoftext|>', '<|endoftext|>'] -> [1, 1, 1, 1, 1, 0, 0]
mask, stopped on a sampled <|im_end|>, then 1 padding token: tokens [' ', '5', '<|im_end|>', '<|endoftext|>'] -> [1, 1, 1, 0]
mask, hit the length limit: tokens [' ', '5', ' is', ' the', ' answer'] -> [1, 1, 1, 1, 1]
```

The mask is 1 for the tokens the model really wrote and 0 after. Every later step uses it, so padding never counts. The evaluation uses the same function for its lengths.

### Step 2: the log-probability of each token

To change the probability of an answer, we need the probability of each of its tokens. One forward pass over prompt plus answer gives them all.

```python
def per_token_logps(model, batch, temperature=1.0):
    """log pi(o_t | q, o_<t) for every completion token: the numbers that go into the ratio of Eq. 1."""
    ids = torch.cat([batch['prompt_ids'], batch['completion_ids']], 1)                  # (B, P + T)
    mask = torch.cat([batch['prompt_mask'], batch['completion_mask']], 1)               # (B, P + T)
    T = batch['completion_ids'].size(1)
    # The logits at position t predict token t + 1, so keep the T + 1 last positions and drop the very last one.
    logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=T + 1).logits[:, :-1].float()   # (B, T, V)
    logits = logits / temperature
    chosen = logits.gather(-1, batch['completion_ids'].unsqueeze(-1)).squeeze(-1)      # (B, T): score of the token
    return chosen - torch.logsumexp(logits, -1)                                         # (B, T): log softmax
```

The logits at position $$t$$ predict the token at position $$t+1$$. So for $$T$$ answer tokens we keep the last $$T+1$$ positions and drop the very last one. `gather` picks the logit of the token that was actually written. Subtracting `logsumexp` over the vocabulary turns that logit into a log-probability, without building the full table of 151,936 probabilities per position a second time.

The paper writes $$\pi_\theta(o_i \mid q)$$ for a whole answer. The probability of an answer is the product of the probabilities of its tokens, so its logarithm is a sum over tokens. The code keeps one number per token, because Eq. 1 works on each token, as the next box shows.

### Step 3: the group advantage (Eq. 3)

<figure class="fig"><svg viewBox="0 0 700 280" role="img" aria-label="The minir1 package as a loop. template.py turns a question into the Table 1 prompt; sample_group in grpo.py samples G answers; rewards.py scores them; the update part of grpo.py turns rewards into advantages and a loss; train.py repeats this; eval.py measures held-out questions. Highlighted: grpo_update."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><rect class="box" x="20.0" y="30.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="120.0" y="60.0" text-anchor="middle">template.py</text><text class="t-tick" x="120.0" y="82.0" text-anchor="middle">Table 1: the prompt</text><text class="t-tick" x="120.0" y="100.0" text-anchor="middle">question → text</text><rect class="box" x="250.0" y="30.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="350.0" y="60.0" text-anchor="middle">grpo.py: sample_group</text><text class="t-tick" x="350.0" y="82.0" text-anchor="middle">G answers per question</text><text class="t-tick" x="350.0" y="100.0" text-anchor="middle">from π_old</text><rect class="box" x="480.0" y="30.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="580.0" y="60.0" text-anchor="middle">rewards.py</text><text class="t-tick" x="580.0" y="82.0" text-anchor="middle">accuracy + format</text><text class="t-tick" x="580.0" y="100.0" text-anchor="middle">rules, no network</text><rect class="box-on" x="480.0" y="170.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="580.0" y="200.0" text-anchor="middle">grpo.py: Eq. 3, 1, 2</text><text class="t-tick" x="580.0" y="222.0" text-anchor="middle">advantages, clipped ratio,</text><text class="t-tick" x="580.0" y="240.0" text-anchor="middle">KL to π_ref → loss</text><text class="t-note" x="580.0" y="162.0" text-anchor="middle">you are here</text><rect class="box" x="250.0" y="170.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="350.0" y="200.0" text-anchor="middle">train.py</text><text class="t-tick" x="350.0" y="222.0" text-anchor="middle">the loop, the optimiser,</text><text class="t-tick" x="350.0" y="240.0" text-anchor="middle">one log line per step</text><rect class="box" x="20.0" y="170.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="120.0" y="200.0" text-anchor="middle">eval.py</text><text class="t-tick" x="120.0" y="222.0" text-anchor="middle">200 held-out questions</text><text class="t-tick" x="120.0" y="240.0" text-anchor="middle">before and after</text><line class="edge" x1="220.0" y1="75.0" x2="248.0" y2="75.0" marker-end="url(#ah)"/><line class="edge" x1="450.0" y1="75.0" x2="478.0" y2="75.0" marker-end="url(#ah)"/><line class="edge" x1="580.0" y1="120.0" x2="580.0" y2="168.0" marker-end="url(#ah)"/><line class="edge" x1="480.0" y1="215.0" x2="452.0" y2="215.0" marker-end="url(#ah)"/><line class="path" x1="350.0" y1="170.0" x2="350.0" y2="122.0" marker-end="url(#ah-on)"/><line class="edge" x1="250.0" y1="215.0" x2="222.0" y2="215.0" marker-end="url(#ah)"/><text class="t-tick" x="358.0" y="150.0" text-anchor="start">next step</text></svg><figcaption>You are here: the update half of grpo.py, from rewards to a loss.</figcaption></figure>

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

```python
def group_advantages(rewards, G):
    """Eq. 3: A_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G), each answer against its own group."""
    r = rewards.view(-1, G)                                                             # (questions, G)
    mean, std = r.mean(1, keepdim=True), r.std(1, keepdim=True)                         # std with n - 1, as TRL
    return ((r - mean) / (std + 1e-4)).view(-1)                                         # (B,): 0 when all equal
```

A worked example with $$G = 4$$. The rewards are 2 (right and well formed), 0, 1 (well formed, wrong) and 0. The mean is 0.75.

The deviations are 1.25, -0.75, 0.25 and -0.75. Their squares add up to 2.75, and $$\sqrt{2.75/3} = 0.957$$. So the advantages are $$1.25/0.957 = 1.31$$, then -0.78, 0.26 and -0.78. Our function and Python's `statistics` module agree:

```text
Eq. 3, rewards of one group: [2.0, 0.0, 1.0, 0.0]
  mean 0.75  std (n - 1) 0.9574
  advantages (ours)      : [1.3054, -0.7833, 0.2611, -0.7833]
  advantages (statistics): [1.3054, -0.7833, 0.2611, -0.7833]
  all four the same reward: [0.0, 0.0, 0.0, 0.0]
  tokens clipped, ours 0.0202, TRL clip_ratio/region_mean 0.0202
  mean KL per token, ours 0.273503, TRL kl 0.273503
  tokens clipped, ours 0.1254, TRL clip_ratio/region_mean 0.1254
  mean KL per token, ours 0.273503, TRL kl 0.273503
  tokens clipped, ours 0.0202, TRL clip_ratio/region_mean 0.0202
  mean KL per token, ours 0.273503, TRL kl 9.148457
```

The last line is the case of equal rewards: four zero advantages. Such a group gives no reward signal. It is not completely silent, though. The KL term of the next steps still pulls each token toward the reference model. With $$A = 0$$, $$\pi_\theta = 0.6$$, $$\pi_{ref} = 0.5$$ and $$\beta = 0.04$$, our loss still has a gradient:

```text
A = 0, pi_theta 0.6, pi_ref 0.5, beta 0.04: gradient of the loss on log pi_theta = 0.006667
```

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

```text
Eq. 1 term, A = +1, ratio 1.5: min(+1.50, +1.20) = +1.20
Eq. 1 term, A = +1, ratio 0.7: min(+0.70, +0.80) = +0.70
Eq. 1 term, A = -1, ratio 0.5: min(-0.50, -0.80) = -0.80
Eq. 1 term, A = -1, ratio 1.3: min(-1.30, -1.20) = -1.30
```

With a positive advantage, a ratio of 1.5 is cut to 1.2: the token has already gained enough, so this batch stops pushing it. A ratio of 0.7 is kept: the token lost ground, and it may climb back. With a negative advantage the mirror holds. A ratio of 0.5 counts as 0.8, flat, so the push down stops. A ratio of 1.3 keeps its full value: the bad token became more likely, and the loss keeps pushing it down.

So a token counts as clipped only in two cases: ratio below $$1 - \varepsilon$$ with a negative advantage, or above $$1 + \varepsilon$$ with a positive one. Only then does the min pick the flat branch. Our loss counts clipped tokens this way, the same way as TRL's `clip_ratio/region_mean`:

```text
  clipped? A = +1, ratio 1.5: yes
  clipped? A = +1, ratio 0.7: no
  clipped? A = -1, ratio 0.5: yes
  clipped? A = -1, ratio 1.3: no
```

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

```text
Eq. 2, pi_ref 0.5, pi_theta 0.5: 1.0000 - ln 1.0000 - 1 = 0.00000
Eq. 2, pi_ref 0.5, pi_theta 0.6: 0.8333 - ln 0.8333 - 1 = 0.01565
Eq. 2, pi_ref 0.5, pi_theta 0.4: 1.2500 - ln 1.2500 - 1 = 0.02686
Eq. 2, pi_ref 0.5, pi_theta 0.9: 0.5556 - ln 0.5556 - 1 = 0.14334
```

When the two models agree, the cost is 0. Moving a probability from 0.5 to 0.6 costs 0.016. Moving it from 0.5 to 0.9 costs 0.143, nine times more for four times the move. Small drifts are cheap and big ones are expensive.

### The whole loss, in one function

```python
def grpo_loss(logps, old_logps, ref_logps, advantages, mask, eps=0.2, beta=0.04):
    """Eq. 1 with Eq. 2, per token, as a loss to minimise (the paper maximises J, so the sign flips)."""
    ratio = torch.exp(logps - old_logps)                         # pi_theta / pi_theta_old, (B, T)
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps)               # clip(ratio, 1 - eps, 1 + eps)
    A = advantages.unsqueeze(1)                                  # (B, 1): one advantage for every token of answer i
    surrogate = torch.min(ratio * A, clipped * A)                # the pessimistic one of the two
    log_r = ref_logps - logps                                    # log(pi_ref / pi_theta)
    kl = torch.exp(log_r) - log_r - 1                            # Eq. 2, always >= 0
    per_token = -(surrogate - beta * kl)                         # (B, T)
    m = mask.float()
    per_answer = (per_token * m).sum(1) / m.sum(1).clamp(min=1)  # 1/|o_i| sum over the tokens of answer i
    loss = per_answer.mean()                                     # 1/G sum over answers (and over questions)
    with torch.no_grad():   # sums and a token count, so the loop can take one token-weighted mean over a whole step
        # A token is clipped when the min of Eq. 1 picks the flat branch: ratio < 1 - eps with A < 0, or
        # ratio > 1 + eps with A > 0 (TRL's clip_ratio/region_mean counts the same tokens).
        is_clipped = ((ratio < 1 - eps) & (A < 0)) | ((ratio > 1 + eps) & (A > 0))
        stats = dict(kl_sum=(kl * m).sum().item(), clipped_sum=(is_clipped.float() * m).sum().item(),
                     tokens=m.sum().item())
    return loss, stats
```

Each line is one piece of the equations. `ratio` is $$\rho_{i,t}$$. `torch.min(ratio * A, clipped * A)` is the min of Eq. 1. `kl` is Eq. 2 per token. Then the mask removes padding, the inner mean is the $$1/|o_i|$$ sum, and the outer mean is the $$1/G$$ sum.

For the log, the function returns sums, not means: the total KL, the number of clipped tokens and the number of real tokens. The loop adds them over all slices of a step and divides once. Slices hold different numbers of tokens, so an average of slice means would weight them wrongly, and our first version did exactly that.

**Method 2, with TRL.** TRL's `GRPOTrainer` has a method, `_compute_loss`, that takes the same tensors: prompt and answer tokens, masks, advantages, old and reference log-probabilities. We built a trainer around the same model and called that method directly, so this is TRL's own code.

The batch is 8 real answers of Qwen2.5-0.5B-Instruct to 2 GSM8K questions. Real rewards of 8 short answers are often all equal, and then every advantage is 0. Fixed rewards turn every term on. In training, the three policies differ, so we took two optimiser steps first. The reference is the start, the old policy comes after one step, and the current one after two.

```text
A = 0, pi_theta 0.6, pi_ref 0.5, beta 0.04: gradient of the loss on log pi_theta = 0.006667
== (b) our loss against TRL on one fixed batch
batch: 2 questions x 4 completions, completion tensor (8, 96), 694 real completion tokens
fixed rewards used for the check: [2.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 2.0]
advantages: [1.3054, -0.7833, 0.2611, -0.7833, 0.0, 0.0, -1.2246, 1.2246]
epsilon 0.2, beta 0.04, TRL use_bias_correction_kl=False:
  tokens clipped, ours 0.0202, TRL clip_ratio/region_mean 0.0202
  mean KL per token, ours 0.273503, TRL kl 0.273503
  loss, ours: +64.69831085
  loss, TRL 1.18.2 GRPOTrainer._compute_loss: +64.69834900
  |difference|: loss 3.81e-05, per-token log-probs 1.30e-06, gradient of lm_head 1.29e-02
  relative difference of the two losses: 5.9e-07
  our loss on the log-probs TRL computed: +64.69834900, |difference| from TRL 0.00e+00
epsilon 0.02, beta 0.04, TRL use_bias_correction_kl=False:
  tokens clipped, ours 0.1254, TRL clip_ratio/region_mean 0.1254
  mean KL per token, ours 0.273503, TRL kl 0.273503
  loss, ours: +64.70719147
  loss, TRL 1.18.2 GRPOTrainer._compute_loss: +64.70722961
  |difference|: loss 3.81e-05, per-token log-probs 1.30e-06, gradient of lm_head 8.20e-03
  relative difference of the two losses: 5.9e-07
  our loss on the log-probs TRL computed: +64.70722961, |difference| from TRL 0.00e+00
epsilon 0.2, beta 0.04, TRL use_bias_correction_kl=True:
  tokens clipped, ours 0.0202, TRL clip_ratio/region_mean 0.0202
  mean KL per token, ours 0.273503, TRL kl 9.148457
  loss, ours: +64.69831085
  loss, TRL 1.18.2 GRPOTrainer._compute_loss: +65.01915741
  |difference|: loss 3.21e-01, per-token log-probs 1.30e-06, gradient of lm_head 1.50e+02
  relative difference of the two losses: 4.9e-03
  our loss on the log-probs TRL computed: +64.69834900, |difference| from TRL 3.21e-01
```

[![Terminal output of r1_part8_check.py: the equations on small numbers, then our loss against TRL's loss](/img/papers/deepseek-r1/r1_part8_check-run.png)](/img/papers/deepseek-r1/r1_part8_check-run.png)

Read the first two settings, which use the KL of Eq. 2. Our share of clipped tokens equals TRL's `clip_ratio/region_mean`: 0.0202 at $$\varepsilon = 0.2$$ and 0.1254 at $$\varepsilon = 0.02$$. Our mean KL per token equals TRL's `kl`: 0.273503.

The two losses differ by $$3.81 \times 10^{-5}$$ on a loss of about 64.7, a relative difference of $$5.9 \times 10^{-7}$$. Where does that come from? Not from the formula.

TRL computes the log-probabilities itself, through its own path to the output layer, and they differ from ours by at most $$1.3 \times 10^{-6}$$: float32 rounding. The ratio is an exponential of a log-probability difference, and a few ratios here are huge, so a tiny rounding difference becomes a visible one. The last line of each setting settles it. Fed with the log-probabilities TRL computed, our `grpo_loss` gives exactly TRL's loss: the difference is 0.00e+00.

The loss of about 64.7 looks large, and it is real. A few tokens were sampled with a very low probability, and two optimiser steps made them much more likely: their ratio is huge. Where the advantage is negative, the min of Eq. 1 keeps the huge value, as in the last case of the four-token example. Those few tokens dominate the mean.

The third line is a surprise worth knowing. TRL 1.18 has a setting, `use_bias_correction_kl`, that is on by default. It multiplies each token's KL term by the ratio, a change that comes from the later DeepSeek-V3.2 paper. With it on, TRL's loss is 65.019 and ours is 64.698, and TRL's logged KL is 9.148457 instead of 0.273503. That is not Eq. 2 of the R1 paper, so for R1 we turn it off. A library default can quietly change the method you think you are running.

## The loop {§2.2}

<figure class="fig"><svg viewBox="0 0 700 280" role="img" aria-label="The minir1 package as a loop. template.py turns a question into the Table 1 prompt; sample_group in grpo.py samples G answers; rewards.py scores them; the update part of grpo.py turns rewards into advantages and a loss; train.py repeats this; eval.py measures held-out questions. Highlighted: train."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><rect class="box" x="20.0" y="30.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="120.0" y="60.0" text-anchor="middle">template.py</text><text class="t-tick" x="120.0" y="82.0" text-anchor="middle">Table 1: the prompt</text><text class="t-tick" x="120.0" y="100.0" text-anchor="middle">question → text</text><rect class="box" x="250.0" y="30.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="350.0" y="60.0" text-anchor="middle">grpo.py: sample_group</text><text class="t-tick" x="350.0" y="82.0" text-anchor="middle">G answers per question</text><text class="t-tick" x="350.0" y="100.0" text-anchor="middle">from π_old</text><rect class="box" x="480.0" y="30.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="580.0" y="60.0" text-anchor="middle">rewards.py</text><text class="t-tick" x="580.0" y="82.0" text-anchor="middle">accuracy + format</text><text class="t-tick" x="580.0" y="100.0" text-anchor="middle">rules, no network</text><rect class="box" x="480.0" y="170.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="580.0" y="200.0" text-anchor="middle">grpo.py: Eq. 3, 1, 2</text><text class="t-tick" x="580.0" y="222.0" text-anchor="middle">advantages, clipped ratio,</text><text class="t-tick" x="580.0" y="240.0" text-anchor="middle">KL to π_ref → loss</text><rect class="box-on" x="250.0" y="170.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="350.0" y="200.0" text-anchor="middle">train.py</text><text class="t-tick" x="350.0" y="222.0" text-anchor="middle">the loop, the optimiser,</text><text class="t-tick" x="350.0" y="240.0" text-anchor="middle">one log line per step</text><text class="t-note" x="350.0" y="162.0" text-anchor="middle">you are here</text><rect class="box" x="20.0" y="170.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="120.0" y="200.0" text-anchor="middle">eval.py</text><text class="t-tick" x="120.0" y="222.0" text-anchor="middle">200 held-out questions</text><text class="t-tick" x="120.0" y="240.0" text-anchor="middle">before and after</text><line class="edge" x1="220.0" y1="75.0" x2="248.0" y2="75.0" marker-end="url(#ah)"/><line class="edge" x1="450.0" y1="75.0" x2="478.0" y2="75.0" marker-end="url(#ah)"/><line class="edge" x1="580.0" y1="120.0" x2="580.0" y2="168.0" marker-end="url(#ah)"/><line class="edge" x1="480.0" y1="215.0" x2="452.0" y2="215.0" marker-end="url(#ah)"/><line class="path" x1="350.0" y1="170.0" x2="350.0" y2="122.0" marker-end="url(#ah-on)"/><line class="edge" x1="250.0" y1="215.0" x2="222.0" y2="215.0" marker-end="url(#ah)"/><text class="t-tick" x="358.0" y="150.0" text-anchor="start">next step</text></svg><figcaption>You are here: train.py repeats sample, score, compare and update, and writes one log line per step.</figcaption></figure>

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

```text
config: {"model": "Qwen/Qwen2.5-0.5B-Instruct", "steps": 400, "minutes": 80.0, "questions": 4, "G": 8, "max_new_tokens": 300, "temperature": 1.0, "lr": 3e-06, "eps": 0.2, "beta": 0.04, "minibatches": 2, "micro": 8, "seed": 0}
```

Here is the loop. It is the five steps above, in order:

```python
def train(cfg, policy, tok, data, log, jsonl_path, samples_path):
    random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    ref = copy.deepcopy(policy).eval().requires_grad_(False)           # pi_ref: frozen copy of the start
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg['lr'], betas=(0.9, 0.99), weight_decay=0.0)
    G, start, samples = cfg['G'], time.time(), []
    open(jsonl_path, 'w').close()
    log(f'memory at the start: {memory_gb():.1f} GB')
    for step in range(1, cfg['steps'] + 1):
        qs = random.sample(data, cfg['questions'])
        # 1. Sample a group of G answers per question from the current policy (pi_old).
        policy.eval()
        batch = sample_group(policy, tok, [make_prompt(q['question']) for q in qs], G,
                             cfg['max_new_tokens'], cfg['temperature'])
        # 2. Score every answer with the two rules, then compare it with its own group (Eq. 3).
        gold = [q['gold'] for q in qs for _ in range(G)]
        acc = torch.tensor([accuracy_reward(t, g) for t, g in zip(batch['texts'], gold)])
        fmt = torch.tensor([format_reward(t) for t in batch['texts']])
        rewards = acc + fmt                                              # (B,)
        adv = group_advantages(rewards, G).to(policy.device)           # (B,)
        # 3. Log-probs of pi_old and pi_ref, once, before any update of this step.
        B = rewards.numel()
        with torch.no_grad():
            old = torch.cat([per_token_logps(policy, rows(batch, slice(i, i + cfg['micro'])), cfg['temperature'])
                             for i in range(0, B, cfg['micro'])])
            refl = torch.cat([per_token_logps(ref, rows(batch, slice(i, i + cfg['micro'])), cfg['temperature'])
                              for i in range(0, B, cfg['micro'])])
        # 4. Split the answers into mini-batches; one optimiser step each. From the second one on, pi_theta != pi_old.
        policy.train()
        order, stats = torch.randperm(B), []
        for mb in order.chunk(cfg['minibatches']):
            opt.zero_grad()
            for idx in mb.split(cfg['micro']):
                sub = rows(batch, idx.to(policy.device))
                logps = per_token_logps(policy, sub, cfg['temperature'])
                loss, st = grpo_loss(logps, old[idx], refl[idx], adv[idx], sub['completion_mask'],
                                     cfg['eps'], cfg['beta'])
                (loss * len(idx) / len(mb)).backward()                  # mean over the mini-batch
                stats.append(st)                                        # plain floats, no tensors kept
                del sub, logps, loss
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
        # 5. Log the step.
        # KL and clipped share: one mean over every real token of the step (each slice has a different token count).
        lengths = batch['completion_mask'].sum(1).float()
        tokens = sum(s['tokens'] for s in stats)
        row = dict(step=step, reward=rewards.mean().item(), acc=acc.mean().item(), fmt=fmt.mean().item(),
                   length=lengths.mean().item(), kl=sum(s['kl_sum'] for s in stats) / tokens,
                   clipped=sum(s['clipped_sum'] for s in stats) / tokens,
                   zero_std_groups=(rewards.view(-1, G).std(1) == 0).float().mean().item(),
                   seconds=round(time.time() - start, 1))
        texts = batch['texts']
        del batch, old, refl, adv
        free_memory(policy.device)
        row['rss_gb'] = round(memory_gb(), 2)
        with open(jsonl_path, 'a') as f:
            f.write(json.dumps(row) + '\n')
        log(f"step {step:4d}  reward {row['reward']:.3f}  acc {row['acc']:.3f}  format {row['fmt']:.3f}  "
            f"length {row['length']:6.1f}  kl {row['kl']:.5f}  clipped {row['clipped']:.4f}  "
            f"memory {row['rss_gb']:.1f} GB  {row['seconds']:.0f} s")
        if step in (1, 2, 3) or step % 25 == 0:
            samples.append(dict(step=step, question=qs[0]['question'], gold=qs[0]['gold'],
                                answers=[dict(text=t, acc=a.item(), fmt=f.item())
                                         for t, a, f in zip(texts[:G], acc[:G], fmt[:G])]))
            with open(samples_path, 'w') as f:
                json.dump(samples, f, indent=1)
        if time.time() - start > cfg['minutes'] * 60:
            log(f'time budget of {cfg["minutes"]} minutes reached after step {step}')
            break
    return policy
```

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

<figure class="fig"><svg viewBox="0 0 700 280" role="img" aria-label="The minir1 package as a loop. template.py turns a question into the Table 1 prompt; sample_group in grpo.py samples G answers; rewards.py scores them; the update part of grpo.py turns rewards into advantages and a loss; train.py repeats this; eval.py measures held-out questions. Highlighted: eval."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><rect class="box" x="20.0" y="30.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="120.0" y="60.0" text-anchor="middle">template.py</text><text class="t-tick" x="120.0" y="82.0" text-anchor="middle">Table 1: the prompt</text><text class="t-tick" x="120.0" y="100.0" text-anchor="middle">question → text</text><rect class="box" x="250.0" y="30.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="350.0" y="60.0" text-anchor="middle">grpo.py: sample_group</text><text class="t-tick" x="350.0" y="82.0" text-anchor="middle">G answers per question</text><text class="t-tick" x="350.0" y="100.0" text-anchor="middle">from π_old</text><rect class="box" x="480.0" y="30.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="580.0" y="60.0" text-anchor="middle">rewards.py</text><text class="t-tick" x="580.0" y="82.0" text-anchor="middle">accuracy + format</text><text class="t-tick" x="580.0" y="100.0" text-anchor="middle">rules, no network</text><rect class="box" x="480.0" y="170.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="580.0" y="200.0" text-anchor="middle">grpo.py: Eq. 3, 1, 2</text><text class="t-tick" x="580.0" y="222.0" text-anchor="middle">advantages, clipped ratio,</text><text class="t-tick" x="580.0" y="240.0" text-anchor="middle">KL to π_ref → loss</text><rect class="box" x="250.0" y="170.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="350.0" y="200.0" text-anchor="middle">train.py</text><text class="t-tick" x="350.0" y="222.0" text-anchor="middle">the loop, the optimiser,</text><text class="t-tick" x="350.0" y="240.0" text-anchor="middle">one log line per step</text><rect class="box-on" x="20.0" y="170.0" width="200.0" height="90.0" rx="10"/><text class="t-title" x="120.0" y="200.0" text-anchor="middle">eval.py</text><text class="t-tick" x="120.0" y="222.0" text-anchor="middle">200 held-out questions</text><text class="t-tick" x="120.0" y="240.0" text-anchor="middle">before and after</text><text class="t-note" x="120.0" y="162.0" text-anchor="middle">you are here</text><line class="edge" x1="220.0" y1="75.0" x2="248.0" y2="75.0" marker-end="url(#ah)"/><line class="edge" x1="450.0" y1="75.0" x2="478.0" y2="75.0" marker-end="url(#ah)"/><line class="edge" x1="580.0" y1="120.0" x2="580.0" y2="168.0" marker-end="url(#ah)"/><line class="edge" x1="480.0" y1="215.0" x2="452.0" y2="215.0" marker-end="url(#ah)"/><line class="path" x1="350.0" y1="170.0" x2="350.0" y2="122.0" marker-end="url(#ah-on)"/><line class="edge" x1="250.0" y1="215.0" x2="222.0" y2="215.0" marker-end="url(#ah)"/><text class="t-tick" x="358.0" y="150.0" text-anchor="start">next step</text></svg><figcaption>You are here: eval.py answers 200 held-out questions and scores them with the same rules.</figcaption></figure>

```python
"""Held-out check: greedy answers to GSM8K test questions, scored with the same two rules as in training."""
import torch
from .template import make_prompt, full_completion
from .rewards import accuracy_reward, format_reward
from .grpo import end_token_ids, completion_lengths


@torch.no_grad()
def evaluate(model, tok, problems, max_new_tokens=300, batch=32):
    model.eval()
    end_ids = end_token_ids(model, tok)
    out = []
    for i in range(0, len(problems), batch):
        chunk = problems[i:i + batch]
        enc = tok([make_prompt(p['question']) for p in chunk], return_tensors='pt', padding=True,
                  padding_side='left').to(model.device)
        # Greedy, with no repetition penalty: the Instruct model's own generation config would add one of 1.1.
        gen = model.generate(**enc, do_sample=False, repetition_penalty=1.0, max_new_tokens=max_new_tokens,
                             stop_strings=['</answer>'], tokenizer=tok, pad_token_id=tok.pad_token_id)
        rows = gen[:, enc.input_ids.size(1):].tolist()
        for p, ids, n in zip(chunk, rows, completion_lengths(rows, tok, end_ids)):   # n: tokens really written
            text = full_completion(tok.decode(ids[:n], skip_special_tokens=True))
            out.append(dict(question=p['question'], gold=p['gold'], text=text, length=n,
                            acc=accuracy_reward(text, p['gold']), fmt=format_reward(text)))
    n = len(out)
    summary = dict(n=n, acc=sum(o['acc'] for o in out) / n, fmt=sum(o['fmt'] for o in out) / n,
                   length=sum(o['length'] for o in out) / n)
    return summary, out
```

The evaluation uses the first 200 questions of the GSM8K test split. The model never trains on them. It answers each one greedily, the most likely token every time, with the same limit of 300 new tokens. We set `repetition_penalty=1.0` on purpose.

The Instruct model's own generation settings add a repetition penalty of 1.1, which would quietly change "the most likely token" into "the most likely token after a penalty". Our first evaluation had that penalty; this one does not. The paper reports pass@1 from several samples at temperature 0.6. One greedy answer is cheaper, and it gives the same result every time.

## The run: first try, the base model

R1-Zero starts from a base model, so we did too: Qwen2.5-0.5B, a model of 0.5 billion parameters that was never tuned to follow instructions. Before training, it answered 10% of the 200 held-out questions correctly, and 35.5% of its greedy answers passed the format parser:

```text
BEFORE training: accuracy 0.100  format 0.355  mean length 98.9 tokens  (102 s)
before: our accuracy rule and math_verify agree on 200 of 200 answers (both read the text inside the last answer tags)
before: our format parser and an independent regular expression agree on 200 of 200 answers
```

We gave this run 20 minutes, to keep most of our time budget for the second model. It made 35 steps. In windows of 10 steps:

```text
== base model: 35 steps in 20 minutes
steps    accuracy  format  length  KL      clipped  equal-reward groups  seconds per step  psutil RSS GB
  1- 10     0.044   0.762    47.5   2.654   0.0104                 0.35              34.3            1.5
 11- 20     0.050   0.894    54.2   0.209   0.0014                 0.42              36.8            1.3
 21- 30     0.041   0.872    43.7   0.224   0.0006                 0.28              37.4            1.3
 31- 35     0.031   0.944    34.1   0.241   0.0016                 0.50              27.9            1.3
base model, largest KL of one step: 24.257 at step 1; median KL 0.204
```

<figure class="fig"><svg viewBox="0 0 800 330" role="img" aria-label="Training rewards of the base model run, averaged over 10 steps. Format reward goes from 0.16 to 0.93; accuracy reward from 0.00 to 0.03, over 35 steps."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="280.0" x2="570" y2="280.0"/><text class="t-tick" x="56.0" y="284.0" text-anchor="end">0</text><line class="grid" x1="64" y1="217.5" x2="570" y2="217.5"/><text class="t-tick" x="56.0" y="221.5" text-anchor="end">0.25</text><line class="grid" x1="64" y1="155.0" x2="570" y2="155.0"/><text class="t-tick" x="56.0" y="159.0" text-anchor="end">0.50</text><line class="grid" x1="64" y1="92.5" x2="570" y2="92.5"/><text class="t-tick" x="56.0" y="96.5" text-anchor="end">0.75</text><line class="grid" x1="64" y1="30.0" x2="570" y2="30.0"/><text class="t-tick" x="56.0" y="34.0" text-anchor="end">1.00</text><text class="t-tick" x="64.0" y="300.0" text-anchor="middle">1</text><text class="t-tick" x="197.9" y="300.0" text-anchor="middle">10</text><text class="t-tick" x="346.8" y="300.0" text-anchor="middle">20</text><text class="t-tick" x="495.6" y="300.0" text-anchor="middle">30</text><line class="axis" x1="64" y1="280" x2="570" y2="280"/><polyline class="l2" points="64.0,240.9 78.9,190.2 93.8,165.4 108.6,147.2 123.5,131.6 138.4,117.2 153.3,104.8 168.2,95.4 183.1,92.5 197.9,89.4 212.8,70.6 227.7,62.8 242.6,57.3 257.5,51.9 272.4,49.5 287.2,53.4 302.1,55.0 317.0,55.8 331.9,55.8 346.8,56.6 361.6,62.0 376.5,65.9 391.4,65.9 406.3,68.3 421.2,68.3 436.1,64.4 450.9,65.2 465.8,66.7 480.7,64.4 495.6,62.0 510.5,56.6 525.4,50.3 540.2,48.0 555.1,45.6 570.0,46.4"/><g class="mark"><title>format reward, 1: 0.16</title><circle class="s2 ring" cx="64.0" cy="240.9" r="5"/></g><g class="mark"><title>format reward, 2: 0.36</title><circle class="s2 ring" cx="78.9" cy="190.2" r="5"/></g><g class="mark"><title>format reward, 3: 0.46</title><circle class="s2 ring" cx="93.8" cy="165.4" r="5"/></g><g class="mark"><title>format reward, 4: 0.53</title><circle class="s2 ring" cx="108.6" cy="147.2" r="5"/></g><g class="mark"><title>format reward, 5: 0.59</title><circle class="s2 ring" cx="123.5" cy="131.6" r="5"/></g><g class="mark"><title>format reward, 6: 0.65</title><circle class="s2 ring" cx="138.4" cy="117.2" r="5"/></g><g class="mark"><title>format reward, 7: 0.70</title><circle class="s2 ring" cx="153.3" cy="104.8" r="5"/></g><g class="mark"><title>format reward, 8: 0.74</title><circle class="s2 ring" cx="168.2" cy="95.4" r="5"/></g><g class="mark"><title>format reward, 9: 0.75</title><circle class="s2 ring" cx="183.1" cy="92.5" r="5"/></g><g class="mark"><title>format reward, 10: 0.76</title><circle class="s2 ring" cx="197.9" cy="89.4" r="5"/></g><g class="mark"><title>format reward, 11: 0.84</title><circle class="s2 ring" cx="212.8" cy="70.6" r="5"/></g><g class="mark"><title>format reward, 12: 0.87</title><circle class="s2 ring" cx="227.7" cy="62.8" r="5"/></g><g class="mark"><title>format reward, 13: 0.89</title><circle class="s2 ring" cx="242.6" cy="57.3" r="5"/></g><g class="mark"><title>format reward, 14: 0.91</title><circle class="s2 ring" cx="257.5" cy="51.9" r="5"/></g><g class="mark"><title>format reward, 15: 0.92</title><circle class="s2 ring" cx="272.4" cy="49.5" r="5"/></g><g class="mark"><title>format reward, 16: 0.91</title><circle class="s2 ring" cx="287.2" cy="53.4" r="5"/></g><g class="mark"><title>format reward, 17: 0.90</title><circle class="s2 ring" cx="302.1" cy="55.0" r="5"/></g><g class="mark"><title>format reward, 18: 0.90</title><circle class="s2 ring" cx="317.0" cy="55.8" r="5"/></g><g class="mark"><title>format reward, 19: 0.90</title><circle class="s2 ring" cx="331.9" cy="55.8" r="5"/></g><g class="mark"><title>format reward, 20: 0.89</title><circle class="s2 ring" cx="346.8" cy="56.6" r="5"/></g><g class="mark"><title>format reward, 21: 0.87</title><circle class="s2 ring" cx="361.6" cy="62.0" r="5"/></g><g class="mark"><title>format reward, 22: 0.86</title><circle class="s2 ring" cx="376.5" cy="65.9" r="5"/></g><g class="mark"><title>format reward, 23: 0.86</title><circle class="s2 ring" cx="391.4" cy="65.9" r="5"/></g><g class="mark"><title>format reward, 24: 0.85</title><circle class="s2 ring" cx="406.3" cy="68.3" r="5"/></g><g class="mark"><title>format reward, 25: 0.85</title><circle class="s2 ring" cx="421.2" cy="68.3" r="5"/></g><g class="mark"><title>format reward, 26: 0.86</title><circle class="s2 ring" cx="436.1" cy="64.4" r="5"/></g><g class="mark"><title>format reward, 27: 0.86</title><circle class="s2 ring" cx="450.9" cy="65.2" r="5"/></g><g class="mark"><title>format reward, 28: 0.85</title><circle class="s2 ring" cx="465.8" cy="66.7" r="5"/></g><g class="mark"><title>format reward, 29: 0.86</title><circle class="s2 ring" cx="480.7" cy="64.4" r="5"/></g><g class="mark"><title>format reward, 30: 0.87</title><circle class="s2 ring" cx="495.6" cy="62.0" r="5"/></g><g class="mark"><title>format reward, 31: 0.89</title><circle class="s2 ring" cx="510.5" cy="56.6" r="5"/></g><g class="mark"><title>format reward, 32: 0.92</title><circle class="s2 ring" cx="525.4" cy="50.3" r="5"/></g><g class="mark"><title>format reward, 33: 0.93</title><circle class="s2 ring" cx="540.2" cy="48.0" r="5"/></g><g class="mark"><title>format reward, 34: 0.94</title><circle class="s2 ring" cx="555.1" cy="45.6" r="5"/></g><g class="mark"><title>format reward, 35: 0.93</title><circle class="s2 ring" cx="570.0" cy="46.4" r="5"/></g><polyline class="l1" points="64.0,280.0 78.9,260.5 93.8,267.0 108.6,268.3 123.5,270.6 138.4,269.6 153.3,271.1 168.2,272.2 183.1,268.7 197.9,269.1 212.8,267.5 227.7,271.4 242.6,269.1 257.5,268.3 272.4,265.9 287.2,266.7 302.1,266.7 317.0,264.4 331.9,268.3 346.8,267.5 361.6,267.5 376.5,265.9 391.4,267.5 406.3,268.3 421.2,269.1 436.1,269.8 450.9,269.8 465.8,271.4 480.7,270.6 495.6,269.8 510.5,271.4 525.4,272.2 540.2,271.4 555.1,270.6 570.0,272.2"/><g class="mark"><title>accuracy reward, 1: 0.00</title><circle class="s1 ring" cx="64.0" cy="280.0" r="5"/></g><g class="mark"><title>accuracy reward, 2: 0.08</title><circle class="s1 ring" cx="78.9" cy="260.5" r="5"/></g><g class="mark"><title>accuracy reward, 3: 0.05</title><circle class="s1 ring" cx="93.8" cy="267.0" r="5"/></g><g class="mark"><title>accuracy reward, 4: 0.05</title><circle class="s1 ring" cx="108.6" cy="268.3" r="5"/></g><g class="mark"><title>accuracy reward, 5: 0.04</title><circle class="s1 ring" cx="123.5" cy="270.6" r="5"/></g><g class="mark"><title>accuracy reward, 6: 0.04</title><circle class="s1 ring" cx="138.4" cy="269.6" r="5"/></g><g class="mark"><title>accuracy reward, 7: 0.04</title><circle class="s1 ring" cx="153.3" cy="271.1" r="5"/></g><g class="mark"><title>accuracy reward, 8: 0.03</title><circle class="s1 ring" cx="168.2" cy="272.2" r="5"/></g><g class="mark"><title>accuracy reward, 9: 0.05</title><circle class="s1 ring" cx="183.1" cy="268.7" r="5"/></g><g class="mark"><title>accuracy reward, 10: 0.04</title><circle class="s1 ring" cx="197.9" cy="269.1" r="5"/></g><g class="mark"><title>accuracy reward, 11: 0.05</title><circle class="s1 ring" cx="212.8" cy="267.5" r="5"/></g><g class="mark"><title>accuracy reward, 12: 0.03</title><circle class="s1 ring" cx="227.7" cy="271.4" r="5"/></g><g class="mark"><title>accuracy reward, 13: 0.04</title><circle class="s1 ring" cx="242.6" cy="269.1" r="5"/></g><g class="mark"><title>accuracy reward, 14: 0.05</title><circle class="s1 ring" cx="257.5" cy="268.3" r="5"/></g><g class="mark"><title>accuracy reward, 15: 0.06</title><circle class="s1 ring" cx="272.4" cy="265.9" r="5"/></g><g class="mark"><title>accuracy reward, 16: 0.05</title><circle class="s1 ring" cx="287.2" cy="266.7" r="5"/></g><g class="mark"><title>accuracy reward, 17: 0.05</title><circle class="s1 ring" cx="302.1" cy="266.7" r="5"/></g><g class="mark"><title>accuracy reward, 18: 0.06</title><circle class="s1 ring" cx="317.0" cy="264.4" r="5"/></g><g class="mark"><title>accuracy reward, 19: 0.05</title><circle class="s1 ring" cx="331.9" cy="268.3" r="5"/></g><g class="mark"><title>accuracy reward, 20: 0.05</title><circle class="s1 ring" cx="346.8" cy="267.5" r="5"/></g><g class="mark"><title>accuracy reward, 21: 0.05</title><circle class="s1 ring" cx="361.6" cy="267.5" r="5"/></g><g class="mark"><title>accuracy reward, 22: 0.06</title><circle class="s1 ring" cx="376.5" cy="265.9" r="5"/></g><g class="mark"><title>accuracy reward, 23: 0.05</title><circle class="s1 ring" cx="391.4" cy="267.5" r="5"/></g><g class="mark"><title>accuracy reward, 24: 0.05</title><circle class="s1 ring" cx="406.3" cy="268.3" r="5"/></g><g class="mark"><title>accuracy reward, 25: 0.04</title><circle class="s1 ring" cx="421.2" cy="269.1" r="5"/></g><g class="mark"><title>accuracy reward, 26: 0.04</title><circle class="s1 ring" cx="436.1" cy="269.8" r="5"/></g><g class="mark"><title>accuracy reward, 27: 0.04</title><circle class="s1 ring" cx="450.9" cy="269.8" r="5"/></g><g class="mark"><title>accuracy reward, 28: 0.03</title><circle class="s1 ring" cx="465.8" cy="271.4" r="5"/></g><g class="mark"><title>accuracy reward, 29: 0.04</title><circle class="s1 ring" cx="480.7" cy="270.6" r="5"/></g><g class="mark"><title>accuracy reward, 30: 0.04</title><circle class="s1 ring" cx="495.6" cy="269.8" r="5"/></g><g class="mark"><title>accuracy reward, 31: 0.03</title><circle class="s1 ring" cx="510.5" cy="271.4" r="5"/></g><g class="mark"><title>accuracy reward, 32: 0.03</title><circle class="s1 ring" cx="525.4" cy="272.2" r="5"/></g><g class="mark"><title>accuracy reward, 33: 0.03</title><circle class="s1 ring" cx="540.2" cy="271.4" r="5"/></g><g class="mark"><title>accuracy reward, 34: 0.04</title><circle class="s1 ring" cx="555.1" cy="270.6" r="5"/></g><g class="mark"><title>accuracy reward, 35: 0.03</title><circle class="s1 ring" cx="570.0" cy="272.2" r="5"/></g><line class="grid" x1="576.0" y1="46.4" x2="586.0" y2="46.4"/><rect class="s2" x="590.0" y="41.4" width="10" height="10" rx="2"/><text class="t-note" x="606.0" y="50.4" text-anchor="start">format reward: 0.93</text><line class="grid" x1="576.0" y1="272.2" x2="586.0" y2="272.2"/><rect class="s1" x="590.0" y="267.2" width="10" height="10" rx="2"/><text class="t-note" x="606.0" y="276.2" text-anchor="start">accuracy reward: 0.03</text><text class="t-tick" x="317.0" y="320.0" text-anchor="middle">training step</text><text class="t-tick" x="14.0" y="18.0" text-anchor="start">mean reward (10-step average)</text></svg><figcaption>The base model run, 10-step averages. The format reward climbs to about 0.9 within 15 steps. The accuracy reward stays near 0.04 the whole time.</figcaption></figure>

<figure class="fig"><svg viewBox="0 0 800 300" role="img" aria-label="Mean answer length in tokens of the base model run, averaged over 10 steps: 140 at the start and 35 at step 35."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="250.0" x2="580" y2="250.0"/><text class="t-tick" x="56.0" y="254.0" text-anchor="end">0</text><line class="grid" x1="64" y1="140.0" x2="580" y2="140.0"/><text class="t-tick" x="56.0" y="144.0" text-anchor="end">80</text><line class="grid" x1="64" y1="30.0" x2="580" y2="30.0"/><text class="t-tick" x="56.0" y="34.0" text-anchor="end">160</text><text class="t-tick" x="64.0" y="270.0" text-anchor="middle">1</text><text class="t-tick" x="200.6" y="270.0" text-anchor="middle">10</text><text class="t-tick" x="352.4" y="270.0" text-anchor="middle">20</text><text class="t-tick" x="504.1" y="270.0" text-anchor="middle">30</text><line class="axis" x1="64" y1="250" x2="580" y2="250"/><polyline class="l3" points="64.0,57.7 79.2,121.9 94.4,142.9 109.5,162.1 124.7,172.3 139.9,181.0 155.1,187.0 170.2,190.3 185.4,189.5 200.6,184.7 215.8,194.8 230.9,192.3 246.1,190.5 261.3,186.5 276.5,184.3 291.6,180.7 306.8,176.5 322.0,174.0 337.2,171.5 352.4,175.5 367.5,174.4 382.7,175.9 397.9,178.3 413.1,178.1 428.2,179.1 443.4,181.0 458.6,182.7 473.8,183.2 488.9,187.8 504.1,190.0 519.3,196.3 534.5,200.4 549.6,202.3 564.8,203.3 580.0,202.1"/><g class="mark"><title>mean length (tokens), 1: 140</title><circle class="s3 ring" cx="64.0" cy="57.7" r="5"/></g><g class="mark"><title>mean length (tokens), 2: 93</title><circle class="s3 ring" cx="79.2" cy="121.9" r="5"/></g><g class="mark"><title>mean length (tokens), 3: 78</title><circle class="s3 ring" cx="94.4" cy="142.9" r="5"/></g><g class="mark"><title>mean length (tokens), 4: 64</title><circle class="s3 ring" cx="109.5" cy="162.1" r="5"/></g><g class="mark"><title>mean length (tokens), 5: 56</title><circle class="s3 ring" cx="124.7" cy="172.3" r="5"/></g><g class="mark"><title>mean length (tokens), 6: 50</title><circle class="s3 ring" cx="139.9" cy="181.0" r="5"/></g><g class="mark"><title>mean length (tokens), 7: 46</title><circle class="s3 ring" cx="155.1" cy="187.0" r="5"/></g><g class="mark"><title>mean length (tokens), 8: 43</title><circle class="s3 ring" cx="170.2" cy="190.3" r="5"/></g><g class="mark"><title>mean length (tokens), 9: 44</title><circle class="s3 ring" cx="185.4" cy="189.5" r="5"/></g><g class="mark"><title>mean length (tokens), 10: 48</title><circle class="s3 ring" cx="200.6" cy="184.7" r="5"/></g><g class="mark"><title>mean length (tokens), 11: 40</title><circle class="s3 ring" cx="215.8" cy="194.8" r="5"/></g><g class="mark"><title>mean length (tokens), 12: 42</title><circle class="s3 ring" cx="230.9" cy="192.3" r="5"/></g><g class="mark"><title>mean length (tokens), 13: 43</title><circle class="s3 ring" cx="246.1" cy="190.5" r="5"/></g><g class="mark"><title>mean length (tokens), 14: 46</title><circle class="s3 ring" cx="261.3" cy="186.5" r="5"/></g><g class="mark"><title>mean length (tokens), 15: 48</title><circle class="s3 ring" cx="276.5" cy="184.3" r="5"/></g><g class="mark"><title>mean length (tokens), 16: 50</title><circle class="s3 ring" cx="291.6" cy="180.7" r="5"/></g><g class="mark"><title>mean length (tokens), 17: 53</title><circle class="s3 ring" cx="306.8" cy="176.5" r="5"/></g><g class="mark"><title>mean length (tokens), 18: 55</title><circle class="s3 ring" cx="322.0" cy="174.0" r="5"/></g><g class="mark"><title>mean length (tokens), 19: 57</title><circle class="s3 ring" cx="337.2" cy="171.5" r="5"/></g><g class="mark"><title>mean length (tokens), 20: 54</title><circle class="s3 ring" cx="352.4" cy="175.5" r="5"/></g><g class="mark"><title>mean length (tokens), 21: 55</title><circle class="s3 ring" cx="367.5" cy="174.4" r="5"/></g><g class="mark"><title>mean length (tokens), 22: 54</title><circle class="s3 ring" cx="382.7" cy="175.9" r="5"/></g><g class="mark"><title>mean length (tokens), 23: 52</title><circle class="s3 ring" cx="397.9" cy="178.3" r="5"/></g><g class="mark"><title>mean length (tokens), 24: 52</title><circle class="s3 ring" cx="413.1" cy="178.1" r="5"/></g><g class="mark"><title>mean length (tokens), 25: 52</title><circle class="s3 ring" cx="428.2" cy="179.1" r="5"/></g><g class="mark"><title>mean length (tokens), 26: 50</title><circle class="s3 ring" cx="443.4" cy="181.0" r="5"/></g><g class="mark"><title>mean length (tokens), 27: 49</title><circle class="s3 ring" cx="458.6" cy="182.7" r="5"/></g><g class="mark"><title>mean length (tokens), 28: 49</title><circle class="s3 ring" cx="473.8" cy="183.2" r="5"/></g><g class="mark"><title>mean length (tokens), 29: 45</title><circle class="s3 ring" cx="488.9" cy="187.8" r="5"/></g><g class="mark"><title>mean length (tokens), 30: 44</title><circle class="s3 ring" cx="504.1" cy="190.0" r="5"/></g><g class="mark"><title>mean length (tokens), 31: 39</title><circle class="s3 ring" cx="519.3" cy="196.3" r="5"/></g><g class="mark"><title>mean length (tokens), 32: 36</title><circle class="s3 ring" cx="534.5" cy="200.4" r="5"/></g><g class="mark"><title>mean length (tokens), 33: 35</title><circle class="s3 ring" cx="549.6" cy="202.3" r="5"/></g><g class="mark"><title>mean length (tokens), 34: 34</title><circle class="s3 ring" cx="564.8" cy="203.3" r="5"/></g><g class="mark"><title>mean length (tokens), 35: 35</title><circle class="s3 ring" cx="580.0" cy="202.1" r="5"/></g><line class="grid" x1="586.0" y1="202.1" x2="596.0" y2="202.1"/><rect class="s3" x="600.0" y="197.1" width="10" height="10" rx="2"/><text class="t-note" x="616.0" y="206.1" text-anchor="start">mean length (tokens): 35</text><text class="t-tick" x="322.0" y="290.0" text-anchor="middle">training step</text><text class="t-tick" x="14.0" y="18.0" text-anchor="start">tokens per answer (10-step average)</text></svg><figcaption>Answer length in the base model run, 10-step averages. Answers get shorter, not longer: they drop from about 140 tokens at step 1 to between about 30 and 60 tokens.</figcaption></figure>

The format reward was 0.762 in the first 10 steps and 0.944 in the last 5. The accuracy reward stayed between 0.031 and 0.050. And the answers got much shorter, from 139.9 tokens at step 1 to 34.1 tokens on average in steps 31 to 35. The held-out test after training shows the same thing on new questions:

```text
AFTER training: accuracy 0.040  format 0.985  mean length 37.3 tokens  (51 s)
after: our accuracy rule and math_verify agree on 200 of 200 answers (both read the text inside the last answer tags)
after: our format parser and an independent regular expression agree on 200 of 200 answers
```

Format went from 35.5% to 98.5%, but accuracy fell from 10% to 4%, and the mean answer shrank from 98.9 to 37.3 tokens. On this model and in this short run, the rules taught the shape and nothing else.

The samples show what the model writes. Here are three answers at step 2 and three at step 25:

```text
== base model, training step 2, gold answer 10
Q: At the burger hut, you can buy a burger for $5, french fries for $3, and a soft drink for $3.  If you order a special burger meal, you get all 3 of th...
  [accuracy 0, format 1] '<think> reasoning process here </think>\n<answer> Boone has eaten all but 1 of his burger meals </answer>'
  [accuracy 0, format 0] '<think> Burger-Broblems </think>. <answer> Joe eats at an awesome burger shack </answer>'
  [accuracy 0, format 0] "<think> calculating with burger marketing </think> <answer> (burger meals relate relationship 1 )<item> 1'* burger meals are 2 *$ 5 = $10 </item> <item> 1*\\$3 + 1*\\$3 = $6 </item> <item> 1*burger meals total = $12 </item> <think> assuming 1 * burger meals are equal to 1 * food items </think> <item> 6 * $2 = $12 </item> <item> 10 + $12 = $22 </item> <answer> savings = food items - burger meals = $12 - $10 = $2 </answe"
== base model, training step 25, gold answer 8000
Q: Mr. Wells has a garden of flowers with 50 rows. If each row has 400 flowers and Mr. Wells cuts 60% of the flowers, how many flowers are remaining in t...
  [accuracy 0, format 1] "<think> First, let's figure out the total number of flowers Mr. Wells has. </think> <answer> 50 x 400 = 20,000 flowers </answer>"
  [accuracy 0, format 1] '<think> Subtract 60% of total flowers </think> <answer> 40*50 - 0.6*50 = 1600 </answer>'
  [accuracy 0, format 1] '<think> Call us!</think> <answer> 400 </answer>'
```

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

```text
BEFORE training: accuracy 0.165  format 0.550  mean length 115.1 tokens  (90 s)
before: our accuracy rule and math_verify agree on 200 of 200 answers (both read the text inside the last answer tags)
before: our format parser and an independent regular expression agree on 200 of 200 answers
```

Then 122 steps of our GRPO, the whole 80-minute budget. Here are the first steps and the last steps of the log:

```text
step    1  reward 0.500  acc 0.062  format 0.438  length  132.2  kl 1.16192  clipped 0.0395  memory 1.4 GB  43 s
step    2  reward 0.344  acc 0.062  format 0.281  length  158.5  kl 0.09710  clipped 0.0079  memory 1.4 GB  90 s
step    3  reward 0.406  acc 0.062  format 0.344  length  123.7  kl 0.13512  clipped 0.0134  memory 1.4 GB  134 s
step    4  reward 0.469  acc 0.000  format 0.469  length  183.0  kl 0.07868  clipped 0.0024  memory 1.4 GB  177 s
step    5  reward 0.625  acc 0.031  format 0.594  length  144.8  kl 0.08281  clipped 0.0026  memory 1.4 GB  223 s
step  118  reward 1.188  acc 0.281  format 0.906  length  165.5  kl 0.11107  clipped 0.0002  memory 1.1 GB  4672 s
step  119  reward 1.375  acc 0.406  format 0.969  length  141.3  kl 0.10981  clipped 0.0007  memory 1.1 GB  4705 s
step  120  reward 1.312  acc 0.344  format 0.969  length  158.0  kl 0.09827  clipped 0.0006  memory 1.1 GB  4745 s
step  121  reward 1.188  acc 0.281  format 0.906  length  158.6  kl 0.09319  clipped 0.0008  memory 1.1 GB  4787 s
step  122  reward 1.188  acc 0.344  format 0.844  length  187.2  kl 0.10839  clipped 0.0005  memory 1.1 GB  4829 s
```

And the 10-step windows:

```text
== instruct model: 122 steps in 80 minutes
steps    accuracy  format  length  KL      clipped  equal-reward groups  seconds per step  psutil RSS GB
  1- 10     0.041   0.656   114.7   0.217   0.0077                 0.23              40.2            1.4
 11- 20     0.075   0.966    78.9   0.120   0.0021                 0.42              36.4            1.4
 21- 30     0.131   0.931   125.3   0.088   0.0003                 0.20              40.4            1.5
 31- 40     0.113   0.944   117.3   0.106   0.0007                 0.42              35.7            1.5
 41- 50     0.338   0.953   133.5   0.109   0.0012                 0.33              39.6            1.5
 51- 60     0.206   0.950   140.2   0.103   0.0006                 0.33              40.3            1.4
 61- 70     0.122   0.963   108.3   0.143   0.0005                 0.47              41.5            1.2
 71- 80     0.259   0.972   124.6   0.121   0.0006                 0.28              38.1            1.1
 81- 90     0.297   0.959   148.0   0.122   0.0006                 0.28              41.6            1.1
 91-100     0.303   0.953   138.0   0.137   0.0008                 0.12              42.0            1.1
101-110     0.250   0.978   143.4   0.122   0.0002                 0.30              40.7            1.2
111-120     0.359   0.959   147.1   0.111   0.0004                 0.12              38.1            1.2
121-122     0.312   0.875   172.9   0.101   0.0006                 0.00              42.0            1.1
```

[![Terminal output of the Instruct run: evaluation before, 122 GRPO steps, evaluation after](/img/papers/deepseek-r1/r1_part8_train_instruct-run.png)](/img/papers/deepseek-r1/r1_part8_train_instruct-run.png)

<figure class="fig"><svg viewBox="0 0 800 330" role="img" aria-label="Training rewards of the Instruct model run, averaged over 10 steps. Format reward goes from 0.44 to 0.94; accuracy reward from 0.06 to 0.35, over 122 steps."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="280.0" x2="570" y2="280.0"/><text class="t-tick" x="56.0" y="284.0" text-anchor="end">0</text><line class="grid" x1="64" y1="217.5" x2="570" y2="217.5"/><text class="t-tick" x="56.0" y="221.5" text-anchor="end">0.25</text><line class="grid" x1="64" y1="155.0" x2="570" y2="155.0"/><text class="t-tick" x="56.0" y="159.0" text-anchor="end">0.50</text><line class="grid" x1="64" y1="92.5" x2="570" y2="92.5"/><text class="t-tick" x="56.0" y="96.5" text-anchor="end">0.75</text><line class="grid" x1="64" y1="30.0" x2="570" y2="30.0"/><text class="t-tick" x="56.0" y="34.0" text-anchor="end">1.00</text><text class="t-tick" x="64.0" y="300.0" text-anchor="middle">1</text><text class="t-tick" x="164.4" y="300.0" text-anchor="middle">25</text><text class="t-tick" x="268.9" y="300.0" text-anchor="middle">50</text><text class="t-tick" x="373.5" y="300.0" text-anchor="middle">75</text><text class="t-tick" x="478.0" y="300.0" text-anchor="middle">100</text><line class="axis" x1="64" y1="280" x2="570" y2="280"/><polyline class="l2" points="64.0,170.6 80.7,173.8 97.5,121.1 114.2,73.0 130.9,45.6 147.6,35.5 164.4,40.2 181.1,46.4 197.8,46.4 214.5,41.7 231.3,42.5 248.0,41.7 264.7,41.7 281.5,46.4 298.2,47.2 314.9,42.5 331.6,40.2 348.4,39.4 365.1,34.7 381.8,31.6 398.5,37.0 415.3,42.5 432.0,37.0 448.7,42.5 465.5,40.2 482.2,41.7 498.9,37.8 515.6,33.9 532.4,33.9 549.1,39.4 565.8,42.5 570.0,45.6"/><g class="mark"><title>format reward, 1: 0.44</title><circle class="s2 ring" cx="64.0" cy="170.6" r="5"/></g><g class="mark"><title>format reward, 5: 0.42</title><circle class="s2 ring" cx="80.7" cy="173.8" r="5"/></g><g class="mark"><title>format reward, 9: 0.64</title><circle class="s2 ring" cx="97.5" cy="121.1" r="5"/></g><g class="mark"><title>format reward, 13: 0.83</title><circle class="s2 ring" cx="114.2" cy="73.0" r="5"/></g><g class="mark"><title>format reward, 17: 0.94</title><circle class="s2 ring" cx="130.9" cy="45.6" r="5"/></g><g class="mark"><title>format reward, 21: 0.98</title><circle class="s2 ring" cx="147.6" cy="35.5" r="5"/></g><g class="mark"><title>format reward, 25: 0.96</title><circle class="s2 ring" cx="164.4" cy="40.2" r="5"/></g><g class="mark"><title>format reward, 29: 0.93</title><circle class="s2 ring" cx="181.1" cy="46.4" r="5"/></g><g class="mark"><title>format reward, 33: 0.93</title><circle class="s2 ring" cx="197.8" cy="46.4" r="5"/></g><g class="mark"><title>format reward, 37: 0.95</title><circle class="s2 ring" cx="214.5" cy="41.7" r="5"/></g><g class="mark"><title>format reward, 41: 0.95</title><circle class="s2 ring" cx="231.3" cy="42.5" r="5"/></g><g class="mark"><title>format reward, 45: 0.95</title><circle class="s2 ring" cx="248.0" cy="41.7" r="5"/></g><g class="mark"><title>format reward, 49: 0.95</title><circle class="s2 ring" cx="264.7" cy="41.7" r="5"/></g><g class="mark"><title>format reward, 53: 0.93</title><circle class="s2 ring" cx="281.5" cy="46.4" r="5"/></g><g class="mark"><title>format reward, 57: 0.93</title><circle class="s2 ring" cx="298.2" cy="47.2" r="5"/></g><g class="mark"><title>format reward, 61: 0.95</title><circle class="s2 ring" cx="314.9" cy="42.5" r="5"/></g><g class="mark"><title>format reward, 65: 0.96</title><circle class="s2 ring" cx="331.6" cy="40.2" r="5"/></g><g class="mark"><title>format reward, 69: 0.96</title><circle class="s2 ring" cx="348.4" cy="39.4" r="5"/></g><g class="mark"><title>format reward, 73: 0.98</title><circle class="s2 ring" cx="365.1" cy="34.7" r="5"/></g><g class="mark"><title>format reward, 77: 0.99</title><circle class="s2 ring" cx="381.8" cy="31.6" r="5"/></g><g class="mark"><title>format reward, 81: 0.97</title><circle class="s2 ring" cx="398.5" cy="37.0" r="5"/></g><g class="mark"><title>format reward, 85: 0.95</title><circle class="s2 ring" cx="415.3" cy="42.5" r="5"/></g><g class="mark"><title>format reward, 89: 0.97</title><circle class="s2 ring" cx="432.0" cy="37.0" r="5"/></g><g class="mark"><title>format reward, 93: 0.95</title><circle class="s2 ring" cx="448.7" cy="42.5" r="5"/></g><g class="mark"><title>format reward, 97: 0.96</title><circle class="s2 ring" cx="465.5" cy="40.2" r="5"/></g><g class="mark"><title>format reward, 101: 0.95</title><circle class="s2 ring" cx="482.2" cy="41.7" r="5"/></g><g class="mark"><title>format reward, 105: 0.97</title><circle class="s2 ring" cx="498.9" cy="37.8" r="5"/></g><g class="mark"><title>format reward, 109: 0.98</title><circle class="s2 ring" cx="515.6" cy="33.9" r="5"/></g><g class="mark"><title>format reward, 113: 0.98</title><circle class="s2 ring" cx="532.4" cy="33.9" r="5"/></g><g class="mark"><title>format reward, 117: 0.96</title><circle class="s2 ring" cx="549.1" cy="39.4" r="5"/></g><g class="mark"><title>format reward, 121: 0.95</title><circle class="s2 ring" cx="565.8" cy="42.5" r="5"/></g><g class="mark"><title>format reward, 122: 0.94</title><circle class="s2 ring" cx="570.0" cy="45.6" r="5"/></g><polyline class="l1" points="64.0,264.4 80.7,269.1 97.5,270.5 114.2,273.0 130.9,267.5 147.6,256.6 164.4,250.3 181.1,242.5 197.8,251.1 214.5,257.3 231.3,248.8 248.0,220.6 264.7,198.0 281.5,205.8 298.2,222.2 314.9,226.1 331.6,234.7 348.4,246.4 365.1,238.6 381.8,221.4 398.5,215.9 415.3,215.9 432.0,203.4 448.7,201.1 465.5,208.1 482.2,213.6 498.9,223.0 515.6,220.6 532.4,202.7 549.1,198.0 565.8,190.9 570.0,191.7"/><g class="mark"><title>accuracy reward, 1: 0.06</title><circle class="s1 ring" cx="64.0" cy="264.4" r="5"/></g><g class="mark"><title>accuracy reward, 5: 0.04</title><circle class="s1 ring" cx="80.7" cy="269.1" r="5"/></g><g class="mark"><title>accuracy reward, 9: 0.04</title><circle class="s1 ring" cx="97.5" cy="270.5" r="5"/></g><g class="mark"><title>accuracy reward, 13: 0.03</title><circle class="s1 ring" cx="114.2" cy="273.0" r="5"/></g><g class="mark"><title>accuracy reward, 17: 0.05</title><circle class="s1 ring" cx="130.9" cy="267.5" r="5"/></g><g class="mark"><title>accuracy reward, 21: 0.09</title><circle class="s1 ring" cx="147.6" cy="256.6" r="5"/></g><g class="mark"><title>accuracy reward, 25: 0.12</title><circle class="s1 ring" cx="164.4" cy="250.3" r="5"/></g><g class="mark"><title>accuracy reward, 29: 0.15</title><circle class="s1 ring" cx="181.1" cy="242.5" r="5"/></g><g class="mark"><title>accuracy reward, 33: 0.12</title><circle class="s1 ring" cx="197.8" cy="251.1" r="5"/></g><g class="mark"><title>accuracy reward, 37: 0.09</title><circle class="s1 ring" cx="214.5" cy="257.3" r="5"/></g><g class="mark"><title>accuracy reward, 41: 0.12</title><circle class="s1 ring" cx="231.3" cy="248.8" r="5"/></g><g class="mark"><title>accuracy reward, 45: 0.24</title><circle class="s1 ring" cx="248.0" cy="220.6" r="5"/></g><g class="mark"><title>accuracy reward, 49: 0.33</title><circle class="s1 ring" cx="264.7" cy="198.0" r="5"/></g><g class="mark"><title>accuracy reward, 53: 0.30</title><circle class="s1 ring" cx="281.5" cy="205.8" r="5"/></g><g class="mark"><title>accuracy reward, 57: 0.23</title><circle class="s1 ring" cx="298.2" cy="222.2" r="5"/></g><g class="mark"><title>accuracy reward, 61: 0.22</title><circle class="s1 ring" cx="314.9" cy="226.1" r="5"/></g><g class="mark"><title>accuracy reward, 65: 0.18</title><circle class="s1 ring" cx="331.6" cy="234.7" r="5"/></g><g class="mark"><title>accuracy reward, 69: 0.13</title><circle class="s1 ring" cx="348.4" cy="246.4" r="5"/></g><g class="mark"><title>accuracy reward, 73: 0.17</title><circle class="s1 ring" cx="365.1" cy="238.6" r="5"/></g><g class="mark"><title>accuracy reward, 77: 0.23</title><circle class="s1 ring" cx="381.8" cy="221.4" r="5"/></g><g class="mark"><title>accuracy reward, 81: 0.26</title><circle class="s1 ring" cx="398.5" cy="215.9" r="5"/></g><g class="mark"><title>accuracy reward, 85: 0.26</title><circle class="s1 ring" cx="415.3" cy="215.9" r="5"/></g><g class="mark"><title>accuracy reward, 89: 0.31</title><circle class="s1 ring" cx="432.0" cy="203.4" r="5"/></g><g class="mark"><title>accuracy reward, 93: 0.32</title><circle class="s1 ring" cx="448.7" cy="201.1" r="5"/></g><g class="mark"><title>accuracy reward, 97: 0.29</title><circle class="s1 ring" cx="465.5" cy="208.1" r="5"/></g><g class="mark"><title>accuracy reward, 101: 0.27</title><circle class="s1 ring" cx="482.2" cy="213.6" r="5"/></g><g class="mark"><title>accuracy reward, 105: 0.23</title><circle class="s1 ring" cx="498.9" cy="223.0" r="5"/></g><g class="mark"><title>accuracy reward, 109: 0.24</title><circle class="s1 ring" cx="515.6" cy="220.6" r="5"/></g><g class="mark"><title>accuracy reward, 113: 0.31</title><circle class="s1 ring" cx="532.4" cy="202.7" r="5"/></g><g class="mark"><title>accuracy reward, 117: 0.33</title><circle class="s1 ring" cx="549.1" cy="198.0" r="5"/></g><g class="mark"><title>accuracy reward, 121: 0.36</title><circle class="s1 ring" cx="565.8" cy="190.9" r="5"/></g><g class="mark"><title>accuracy reward, 122: 0.35</title><circle class="s1 ring" cx="570.0" cy="191.7" r="5"/></g><line class="grid" x1="576.0" y1="45.6" x2="586.0" y2="45.6"/><rect class="s2" x="590.0" y="40.6" width="10" height="10" rx="2"/><text class="t-note" x="606.0" y="49.6" text-anchor="start">format reward: 0.94</text><line class="grid" x1="576.0" y1="191.7" x2="586.0" y2="191.7"/><rect class="s1" x="590.0" y="186.7" width="10" height="10" rx="2"/><text class="t-note" x="606.0" y="195.7" text-anchor="start">accuracy reward: 0.35</text><text class="t-tick" x="317.0" y="320.0" text-anchor="middle">training step</text><text class="t-tick" x="14.0" y="18.0" text-anchor="start">mean reward (10-step average)</text></svg><figcaption>The Instruct model run, 10-step averages. The format reward rises from below 0.5 to above 0.9 within about 15 steps. The accuracy reward rises from about 0.05 to about 0.3, and it moves a lot from step to step.</figcaption></figure>

Three things happened. The format was learned first: the format reward was 0.656 in the first 10 steps and 0.966 in the next 10. The accuracy reward rose more slowly. It was 0.041 in the first 10 steps, 0.338 in steps 41 to 50, and 0.359 in steps 111 to 120. On the way it dipped as low as 0.122.

With only 4 questions per step, one step can show 0.000 and two steps later 0.250, so read the trend, not single steps. And the answers did not collapse. They dipped to about 79 tokens on average in steps 11 to 20, then grew back to about 140.

The held-out test confirms the trend on questions the model never trained on:

```text
AFTER training: accuracy 0.390  format 0.980  mean length 159.8 tokens  (92 s)
after: our accuracy rule and math_verify agree on 200 of 200 answers (both read the text inside the last answer tags)
after: our format parser and an independent regular expression agree on 200 of 200 answers
```

<figure class="fig"><svg viewBox="0 0 680 300" role="img" aria-label="Held-out GSM8K, 200 questions, greedy: accuracy 0.165 before and 0.390 after training; format 0.550 before and 0.980 after."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><rect class="box-2" x="120.0" y="218.7" width="70.0" height="31.4" rx="4"/><text class="t-note" x="155.0" y="210.7" text-anchor="middle">0.165</text><text class="t-tick" x="155.0" y="268.0" text-anchor="middle">before</text><rect class="box-1" x="210.0" y="175.9" width="70.0" height="74.1" rx="4"/><text class="t-note" x="245.0" y="167.9" text-anchor="middle">0.390</text><text class="t-tick" x="245.0" y="268.0" text-anchor="middle">after</text><text class="t-title" x="200.0" y="290.0" text-anchor="middle">accuracy</text><rect class="box-2" x="380.0" y="145.5" width="70.0" height="104.5" rx="4"/><text class="t-note" x="415.0" y="137.5" text-anchor="middle">0.550</text><text class="t-tick" x="415.0" y="268.0" text-anchor="middle">before</text><rect class="box-1" x="470.0" y="63.8" width="70.0" height="186.2" rx="4"/><text class="t-note" x="505.0" y="55.8" text-anchor="middle">0.980</text><text class="t-tick" x="505.0" y="268.0" text-anchor="middle">after</text><text class="t-title" x="460.0" y="290.0" text-anchor="middle">format</text><line class="axis" x1="100" y1="250" x2="620" y2="250"/></svg><figcaption>The 200 held-out GSM8K questions, greedy answers, before and after 122 steps of GRPO. Accuracy goes from 0.165 to 0.390, and the share of well-formed answers from 0.550 to 0.980.</figcaption></figure>

Accuracy went from 16.5% to 39.0%: 33 right answers before, 78 after. 196 of the 200 answers now have the right shape. On 200 questions a gap of 22.5 points is far larger than the noise of a single evaluation. But it is one run with one seed, so we do not claim a precise size.

> [!NOTE] Numbers from our first run
> An earlier version of this page reported a shorter run made with the bugs listed in the notes above. They were the extra padding token in the mask, the loose format and number rules, and the wrong clip count and KL average. The evaluation also had a repetition penalty of 1.1. Even the score of the untrained model changed, because the evaluation itself changed. Those numbers are not comparable with the ones here, and this page no longer uses them.

Here are four held-out questions, before and after. The text is cut at 500 characters:

```text
== held-out question 0, gold answer 18
Q: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the rem...
  before [accuracy 1, format 1, 128 tokens] "<think> First, we need to calculate the total number of eggs laid by the ducks per day. Janet's ducks lay 16 eggs per day. She eats 3 eggs for breakfast every morning, and she bakes muffins for her friends every day, which means she eats 3 + 4 = 7 eggs for her friends. So, the total number of eggs laid per day is 16 - 7 = 9 eggs. </think> <answer> Janet makes $2 per fresh duck egg, so she makes 9 x $2 = $18 every day at the farmers' market. </answer>"
  after  [accuracy 0, format 1, 183 tokens] "<think> First, we need to calculate the total number of eggs laid by Janet's ducks per day. Since Janet's ducks lay 16 eggs per day, and she eats 3 eggs for breakfast every morning, she will eat 3 eggs every day. She also bakes 4 muffins every day, which means she will bake 4 eggs every day. So, the total number of eggs laid by her ducks every day is 3 + 4 = 7 eggs. \n\nNext, we need to calculate how much money Janet makes every day from selling the eggs. Since she sells the eggs at $2 per fresh d"
== held-out question 1, gold answer 3
Q: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?
  before [accuracy 1, format 1, 83 tokens] '<think> To find the total number of bolts needed, we need to add the number of blue and white bolts together. We know that the number of blue bolts is 2 and the number of white bolts is half of that, which is 2/2 = 1. So, the total number of bolts needed is 2 + 1 = 3. </think> <answer> 3 </answer>'
  after  [accuracy 1, format 1, 76 tokens] '<think> First, we need to identify the amount of blue and white fibers that each robe takes. According to the question, a robe takes 2 bolts of blue fiber and half that much white fiber. So, each robe takes 2 blue fibers and 2/2 = 1 white fiber. </think> <answer> 2 + 1 = 3 </answer>'
== held-out question 2, gold answer 70000
Q: Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%...
  before [accuracy 0, format 1, 36 tokens] '<think> To calculate the profit, we need to subtract the initial cost from the final value of the house. </think> <answer> $30,000 </answer>'
  after  [accuracy 1, format 1, 234 tokens] '<think> First, we need to calculate the total cost of the house. Josh spent $80,000 on the house and $50,000 on repairs, so the total cost was $80,000 + $50,000 = $130,000. Next, we need to calculate the increase in the value of the house. The increase in value was 150% of the original value, which means the increase was $80,000 * 1.5 = $120,000. The increase in value is $120,000, so the total increase in value is $120,000 + $80,000 = $200,000. The profit is the increase in value minus the cost '
== held-out question 3, gold answer 540
Q: James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?
  before [accuracy 0, format 1, 76 tokens] '<think> To calculate the total distance James runs in a week, we need to multiply the number of sprints he runs each week by the distance he runs each sprint. So, 3 sprints * 60 meters per sprint = 180 meters per week. </think> <answer> James runs a total of 180 meters a week. </answer>'
  after  [accuracy 0, format 1, 63 tokens] '<think> First, we need to calculate the total distance James runs each sprint. Since he runs 60 meters each sprint and he runs 3 times a week, he runs 60 * 3 = 180 meters each week. </think> <answer> 180 meters </answer>'
```

Training did not fix every answer, and it broke some. Question 2 went from wrong to right: the trained model now works out the cost, the new value and the profit step by step. Question 0 went the other way: right before, wrong after. Question 3 is wrong both times, with the same mistake (it multiplies only two of the three numbers). A held-out accuracy is a count of such wins and losses, and on these 200 questions the wins were many more.

### The KL stayed small

The windows table shows a mean KL between 0.088 and 0.217 in every window. The largest KL of a single step was at step 1:

```text
instruct model, largest KL of one step: 1.162 at step 1; median KL 0.112
```

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

<figure class="fig"><svg viewBox="0 0 800 300" role="img" aria-label="Mean answer length in tokens of the Instruct model run, averaged over 10 steps: 132 at the start and 156 at step 122."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="250.0" x2="580" y2="250.0"/><text class="t-tick" x="56.0" y="254.0" text-anchor="end">0</text><line class="grid" x1="64" y1="140.0" x2="580" y2="140.0"/><text class="t-tick" x="56.0" y="144.0" text-anchor="end">90</text><line class="grid" x1="64" y1="30.0" x2="580" y2="30.0"/><text class="t-tick" x="56.0" y="34.0" text-anchor="end">180</text><text class="t-tick" x="64.0" y="270.0" text-anchor="middle">1</text><text class="t-tick" x="166.3" y="270.0" text-anchor="middle">25</text><text class="t-tick" x="273.0" y="270.0" text-anchor="middle">50</text><text class="t-tick" x="379.6" y="270.0" text-anchor="middle">75</text><text class="t-tick" x="486.2" y="270.0" text-anchor="middle">100</text><line class="axis" x1="64" y1="250" x2="580" y2="250"/><polyline class="l3" points="64.0,88.4 81.1,68.6 98.1,109.1 115.2,130.7 132.2,161.8 149.3,154.1 166.3,119.4 183.4,98.4 200.5,97.9 217.5,110.8 234.6,106.4 251.6,90.4 268.7,85.4 285.8,79.1 302.8,72.9 319.9,83.2 336.9,100.3 354.0,113.9 371.0,118.3 388.1,118.6 405.2,95.1 422.2,80.0 439.3,74.2 456.3,70.9 473.4,75.5 490.4,80.5 507.5,76.1 524.6,75.9 541.6,81.8 558.7,71.2 575.7,67.5 580.0,59.7"/><g class="mark"><title>mean length (tokens), 1: 132</title><circle class="s3 ring" cx="64.0" cy="88.4" r="5"/></g><g class="mark"><title>mean length (tokens), 5: 148</title><circle class="s3 ring" cx="81.1" cy="68.6" r="5"/></g><g class="mark"><title>mean length (tokens), 9: 115</title><circle class="s3 ring" cx="98.1" cy="109.1" r="5"/></g><g class="mark"><title>mean length (tokens), 13: 98</title><circle class="s3 ring" cx="115.2" cy="130.7" r="5"/></g><g class="mark"><title>mean length (tokens), 17: 72</title><circle class="s3 ring" cx="132.2" cy="161.8" r="5"/></g><g class="mark"><title>mean length (tokens), 21: 78</title><circle class="s3 ring" cx="149.3" cy="154.1" r="5"/></g><g class="mark"><title>mean length (tokens), 25: 107</title><circle class="s3 ring" cx="166.3" cy="119.4" r="5"/></g><g class="mark"><title>mean length (tokens), 29: 124</title><circle class="s3 ring" cx="183.4" cy="98.4" r="5"/></g><g class="mark"><title>mean length (tokens), 33: 124</title><circle class="s3 ring" cx="200.5" cy="97.9" r="5"/></g><g class="mark"><title>mean length (tokens), 37: 114</title><circle class="s3 ring" cx="217.5" cy="110.8" r="5"/></g><g class="mark"><title>mean length (tokens), 41: 117</title><circle class="s3 ring" cx="234.6" cy="106.4" r="5"/></g><g class="mark"><title>mean length (tokens), 45: 131</title><circle class="s3 ring" cx="251.6" cy="90.4" r="5"/></g><g class="mark"><title>mean length (tokens), 49: 135</title><circle class="s3 ring" cx="268.7" cy="85.4" r="5"/></g><g class="mark"><title>mean length (tokens), 53: 140</title><circle class="s3 ring" cx="285.8" cy="79.1" r="5"/></g><g class="mark"><title>mean length (tokens), 57: 145</title><circle class="s3 ring" cx="302.8" cy="72.9" r="5"/></g><g class="mark"><title>mean length (tokens), 61: 136</title><circle class="s3 ring" cx="319.9" cy="83.2" r="5"/></g><g class="mark"><title>mean length (tokens), 65: 122</title><circle class="s3 ring" cx="336.9" cy="100.3" r="5"/></g><g class="mark"><title>mean length (tokens), 69: 111</title><circle class="s3 ring" cx="354.0" cy="113.9" r="5"/></g><g class="mark"><title>mean length (tokens), 73: 108</title><circle class="s3 ring" cx="371.0" cy="118.3" r="5"/></g><g class="mark"><title>mean length (tokens), 77: 108</title><circle class="s3 ring" cx="388.1" cy="118.6" r="5"/></g><g class="mark"><title>mean length (tokens), 81: 127</title><circle class="s3 ring" cx="405.2" cy="95.1" r="5"/></g><g class="mark"><title>mean length (tokens), 85: 139</title><circle class="s3 ring" cx="422.2" cy="80.0" r="5"/></g><g class="mark"><title>mean length (tokens), 89: 144</title><circle class="s3 ring" cx="439.3" cy="74.2" r="5"/></g><g class="mark"><title>mean length (tokens), 93: 147</title><circle class="s3 ring" cx="456.3" cy="70.9" r="5"/></g><g class="mark"><title>mean length (tokens), 97: 143</title><circle class="s3 ring" cx="473.4" cy="75.5" r="5"/></g><g class="mark"><title>mean length (tokens), 101: 139</title><circle class="s3 ring" cx="490.4" cy="80.5" r="5"/></g><g class="mark"><title>mean length (tokens), 105: 142</title><circle class="s3 ring" cx="507.5" cy="76.1" r="5"/></g><g class="mark"><title>mean length (tokens), 109: 142</title><circle class="s3 ring" cx="524.6" cy="75.9" r="5"/></g><g class="mark"><title>mean length (tokens), 113: 138</title><circle class="s3 ring" cx="541.6" cy="81.8" r="5"/></g><g class="mark"><title>mean length (tokens), 117: 146</title><circle class="s3 ring" cx="558.7" cy="71.2" r="5"/></g><g class="mark"><title>mean length (tokens), 121: 149</title><circle class="s3 ring" cx="575.7" cy="67.5" r="5"/></g><g class="mark"><title>mean length (tokens), 122: 156</title><circle class="s3 ring" cx="580.0" cy="59.7" r="5"/></g><line class="grid" x1="586.0" y1="59.7" x2="596.0" y2="59.7"/><rect class="s3" x="600.0" y="54.7" width="10" height="10" rx="2"/><text class="t-note" x="616.0" y="63.7" text-anchor="start">mean length (tokens): 156</text><text class="t-tick" x="322.0" y="290.0" text-anchor="middle">training step</text><text class="t-tick" x="14.0" y="18.0" text-anchor="start">tokens per answer (10-step average)</text></svg><figcaption>Answer length in the Instruct model run, 10-step averages. It dips to about 80 tokens around step 15, then climbs back to between about 125 and 150 tokens.</figcaption></figure>

Our answers did get somewhat longer. On the held-out questions the mean went from 115 tokens before training to 160 after. During training, the 10-step mean was 115 in the first window. After a dip in steps 11 to 20, it stayed between 138 and 148 from step 81 to step 120.

That is a rise of about 40%, not the twenty-fold growth of Figure 3. We cannot tell from one run why it happened. Longer answers may simply carry more of the arithmetic written out.

Our setting is far from R1-Zero's in every way the box above lists, so we did not expect a Figure 3. Our limit is 300 new tokens, about 3% of R1-Zero's final mean length, so a long answer is cut off and earns nothing. We ran 122 steps, not 10,400. And GSM8K problems need only a few arithmetic steps. These are differences, not a tested explanation: we did not run the experiment that would separate them.

We did not see an aha moment either. The 0.5B model writes simple, linear arithmetic. It does not stop to say "Wait" and check itself. We did reproduce the first part of the story. Rules alone, with no example of good reasoning, taught the model the format in about 15 steps. They also raised its accuracy on new questions.

### What the laptop could and could not do

The run used an Apple laptop with 64 GB of memory, through PyTorch's `mps` device. The policy, the frozen reference and the optimiser state are all full float32 copies of the model.

Our first run did not get far. After about 30 steps its steps slowed down sharply, because the process kept growing until the machine started to swap. Every step writes answers of a different length, so the `mps` allocator keeps freed blocks of many sizes and the cache grows.

The fix is two lines in `free_memory`: drop the step's tensors, collect garbage and call `torch.mps.empty_cache()` after every step. The resident size that `psutil` reports does not see `mps` buffers: it stayed near 1.5 GB. So a small script, `r1_part8_memwatch.py`, read the process footprint once a minute. That is the number Activity Monitor shows:

```text
minute   0.0  after step    2  footprint  30.0 GB
minute   1.0  after step    4  footprint  22.0 GB
minute   2.0  after step    5  footprint  27.0 GB
minute  40.1  after step   64  footprint  27.0 GB
minute  77.1  after step  120  footprint  26.0 GB
minute  78.1  after step  121  footprint  29.0 GB
minute  79.1  after step  122  footprint  29.0 GB
```

Over the 80 readings of the fixed run, the footprint stayed between 20 and 34 GB, with no upward trend. It moves from reading to reading because a reading can land anywhere inside a step. The logits of 8 answers over a vocabulary of 151,936 tokens take a lot of memory while they exist. The step time stayed flat too, as the seconds-per-step column of the windows table shows. So the run used its whole 80 minutes for 122 steps instead of stalling.

Generation is the slow part of RL. Each of our rollouts writes 32 answers of up to 300 tokens before the model can learn anything, and then takes 2 optimiser steps. In the later paper, each R1-Zero rollout wrote 8,192 answers of up to 32,768 tokens (65,536 after step 8,200). Then it took 16 optimiser steps.

Compare the most tokens one rollout may write: $$8{,}192 \times 32{,}768$$ against $$32 \times 300$$, almost 28,000 times more. That is a ratio of maximum rollout budgets, not of tokens actually written per optimiser step. Our run had 122 rollouts and 244 optimiser steps. The method is the same; the scale is not.

## Method 2: the same recipe with TRL

TRL's `GRPOTrainer` does every step of our loop: sampling, rewards, advantages, the loss and the optimiser. We give it the same template and the same two rules. The learning rate, $$\varepsilon$$, $$\beta$$ and G are the same too. We run it for 4 steps only, to show the route. It is a demonstration, not a copy of our run. Each TRL step here uses 2 questions, not 4, and takes one optimiser step, not two.

```python
def accuracy(completions, gold, **kw):               # TRL passes the dataset columns (here: gold) as keyword arguments
    return [accuracy_reward(full_completion(c), g) for c, g in zip(completions, gold)]   # put the prefilled tag back


def fmt(completions, **kw):
    return [format_reward(full_completion(c)) for c in completions]


with common.heavy('part8'):
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    name = 'Qwen/Qwen2.5-0.5B-Instruct'
    train, _ = load_gsm8k()
    ds = Dataset.from_list([dict(prompt=make_prompt(q['question']), gold=q['gold']) for q in train[:64]])
    args = GRPOConfig(
        output_dir=tempfile.mkdtemp(), max_steps=4, learning_rate=3e-6, lr_scheduler_type='constant', loss_type='grpo', beta=0.04, epsilon=0.2,
        use_bias_correction_kl=False,                           # the KL of R1's Eq. 2, as in our package
        num_generations=8, per_device_train_batch_size=16,      # 16 completions per step = 2 questions x G = 8
        max_completion_length=300, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
        generation_kwargs=dict(stop_strings=['</answer>']),      # stop where our sampler stops
        logging_steps=1, report_to='none',
        bf16=False, save_strategy='no', seed=0)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(name)
    model.generate = functools.partial(model.generate, tokenizer=tok)   # generate() needs the tokenizer for a stop string
    trainer = GRPOTrainer(model=model, reward_funcs=[accuracy, fmt], args=args, train_dataset=ds,
                          processing_class=tok)
    t = common.timer()
    trainer.train()
```

TRL passes the extra columns of the dataset, here `gold`, to every reward function as keyword arguments. The reward functions put the prefilled `<think>` back in front, as `sample_group` does.

Three settings matter. `lr_scheduler_type='constant'` keeps the learning rate fixed, as in our loop. `use_bias_correction_kl=False` gives the KL of Eq. 2, as the loss check above showed. And generation must stop at `</answer>`, as ours does. TRL passes `generation_kwargs` to `transformers`, but a stop string also needs the tokenizer, which TRL does not pass. So we bind the tokenizer to `model.generate` with `functools.partial`.

Without the stop string, TRL would keep sampling after `</answer>`. Those extra tokens would enter its loss, its KL and its lengths, while the rewards might see only the cut text. That is a different objective.

With the stop string, one small difference is left. After a stop on `</answer>`, TRL's mask keeps the first padding token. For Qwen, padding is the end-of-text token, and TRL reads it as a sampled end. Our `completion_lengths` leaves it out. So TRL's `completions/mean_length` counts one token more for each answer that stopped on `</answer>`.

TRL logs the same quantities as our loop:

```text
step 1: reward 0.8125  rewards/accuracy/mean 0.0625  rewards/fmt/mean 0.75  completions/mean_length 109.7  kl -8.389e-09  clip_ratio/region_mean 0  frac_reward_zero_std 0
step 2: reward 0.6875  rewards/accuracy/mean 0.0625  rewards/fmt/mean 0.625  completions/mean_length 175.8  kl 0.01768  clip_ratio/region_mean 0  frac_reward_zero_std 0
step 3: reward 1  rewards/accuracy/mean 0.0625  rewards/fmt/mean 0.9375  completions/mean_length 138.1  kl 0.05312  clip_ratio/region_mean 0  frac_reward_zero_std 0
step 4: reward 1.438  rewards/accuracy/mean 0.5  rewards/fmt/mean 0.9375  completions/mean_length 142.3  kl 0.05238  clip_ratio/region_mean 0  frac_reward_zero_std 0
4 steps in 96 s on mps
```

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

