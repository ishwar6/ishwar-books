---
title: "R1-Zero and the Aha Moment"
description: "What pure RL gave DeepSeek-R1-Zero, why its answers grew from hundreds to thousands of tokens, and the moment it wrote Wait. We test the released 1.5B distill."
part: 4
covers: "§2.2.4, Table 2, Figure 2, Figure 3, Table 3"
date: 2026-11-21
tags: [deepseek, reasoning, reinforcement-learning, grpo, aha-moment]
---

Parts 2 and 3 built the machine: GRPO, two rule-based rewards and a one-paragraph template. This part switches it on. Out comes a model that scores close to OpenAI's o1. Its answers grow from hundreds to thousands of tokens. And one day it stops in the middle of its algebra to write "Wait, wait. Wait."

<figure class="fig"><svg viewBox="0 0 760 220" role="img" aria-label="Pipeline: DeepSeek-V3-Base goes through RL with GRPO, rule rewards and the template, and becomes DeepSeek-R1-Zero, the subject of this part. Its readability problems lead to DeepSeek-R1 in Part 5."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><rect class="box" x="20.0" y="40.0" width="210.0" height="70.0" rx="10"/><text class="t-title" x="125.0" y="70.0" text-anchor="middle">DeepSeek-V3-Base</text><text class="t-note" x="125.0" y="92.0" text-anchor="middle">pre-trained, no reasoning SFT</text><line class="path" x1="232.0" y1="75.0" x2="272.0" y2="75.0" marker-end="url(#ah-on)"/><rect class="box" x="275.0" y="40.0" width="210.0" height="70.0" rx="10"/><text class="t-title" x="380.0" y="70.0" text-anchor="middle">RL with GRPO</text><text class="t-note" x="380.0" y="92.0" text-anchor="middle">rule rewards + template (Parts 2, 3)</text><line class="path" x1="487.0" y1="75.0" x2="527.0" y2="75.0" marker-end="url(#ah-on)"/><rect class="box-on" x="530.0" y="40.0" width="210.0" height="70.0" rx="10"/><text class="t-title" x="635.0" y="70.0" text-anchor="middle">DeepSeek-R1-Zero</text><text class="t-note" x="635.0" y="92.0" text-anchor="middle">this part: what came out</text><rect class="box-ghost" x="530.0" y="150.0" width="210.0" height="56.0" rx="10"/><text class="t-note" x="635.0" y="174.0" text-anchor="middle">DeepSeek-R1 (Part 5)</text><text class="t-muted" x="635.0" y="193.0" text-anchor="middle">fixes readability</text><line class="edge" x1="635.0" y1="112.0" x2="635.0" y2="148.0" marker-end="url(#ah)"/><text class="t-tick" x="20.0" y="24.0" text-anchor="start">You are here: the result of pure RL on the base model</text></svg><figcaption>Where this part sits. DeepSeek-V3-Base goes through RL with GRPO, the two rule-based rewards and the template of Parts 2 and 3. What comes out is DeepSeek-R1-Zero. Its problems with readability lead to DeepSeek-R1 in Part 5.</figcaption></figure>

## Performance of DeepSeek-R1-Zero {§2.2.4}

> [!PAPER] · §2.2.4 · page 6
> [![The first paragraph of Performance of DeepSeek-R1-Zero, with steady and consistent enhancement, from an initial 15.6% to an impressive 71.0% and comparable to OpenAI-o1-0912 highlighted](/img/papers/deepseek-r1/p4-performance.png)](/img/papers/deepseek-r1/p4-performance.png)
>
> **What it says:** on AIME 2024, R1-Zero shows "a steady and consistent enhancement in performance". The average pass@1 score goes "from an initial 15.6% to an impressive 71.0%", "comparable to OpenAI-o1-0912".
>
> **What it means:**
>
> The first number is the starting point. At the start of RL, DeepSeek-V3-Base with the template of Table 1 already solves 15.6% of the AIME problems. AIME 2024 has 30 problems, so that is between 4 and 5 problems ([Part 1](part-1-the-big-idea.md) introduced the benchmarks). The base model is not empty. It has read a great deal of mathematics during pre-training.
>
> The second number is the end point: 71.0%, about 21 of the 30 problems. Between the two points there is only RL. Nobody showed the model a worked solution. The only signal was a 1 or a 0 for each final answer, plus the format check.
>
> The comparison is with o1-0912, the September 2024 version of OpenAI's reasoning model (the 0912 in its name). Table 2 gives it 74.4%. So "comparable" here means 3.4 points below it with one answer, as we see in a moment.

### How the scores are counted {§3}

The 71.0% is not one run of the model. It is an average over many answers. The rule is in the evaluation section of the paper.

> [!PAPER] · §3 · page 12
> [![The Evaluation Setup paragraph, with greedy decoding, higher repetition rates, sampling temperature of 0.6, correctness of the i-th response and cons@64 highlighted](/img/papers/deepseek-r1/p4-eval-setup.png)](/img/papers/deepseek-r1/p4-eval-setup.png)
>
> **What it says:** "using greedy decoding to evaluate long-output reasoning models results in higher repetition rates". So the authors sample with "a sampling temperature of 0.6 and a top-p value of 0.95", generate $$k$$ answers per question, and report $$\text{pass@1} = \frac{1}{k}\sum_{i=1}^{k} p_i$$, where $$p_i$$ is "the correctness of the $$i$$-th response". For AIME they also report a majority vote over 64 samples, "cons@64".
>
> **What it means:**
>
> Greedy decoding always picks the most likely next token. For a short answer that is fine. For an answer of thousands of tokens it often falls into a loop: the same sentence comes back again and again. After one repeat, the repeat is the most likely thing to say. So the authors sample, with a little randomness.
>
> Sampling has a price. Run the same question twice and you can get one right answer and one wrong one. A single run is a coin toss. The fix is to ask $$k$$ times and take the mean of the 1s and 0s. That mean is an estimate of the chance that one answer is right, which is what "pass@1" means here.
>
> The vote is a different number. It asks: if we sample many answers and keep the most common final answer, is that one right? [Part 1](part-1-the-big-idea.md) gave both definitions. Here we compute them.

A tiny example with two questions and $$k = 4$$ answers each makes the two numbers concrete.

| question | right answer | the 4 final answers | $$p_1 \dots p_4$$ | pass@1 | vote | vote right? |
|---|---|---|---|---|---|---|
| A | 42 | 42, 42, 17, 42 | 1, 1, 0, 1 | 3/4 = 0.75 | 42 | yes |
| B | 5 | 5, 7, 9, 7 | 1, 0, 0, 0 | 1/4 = 0.25 | 7 | no |
| mean | | | | 0.50 | | 1/2 = 0.50 |

Question B shows that a vote can also lose. The model found the right answer once, but its wrong answers agreed with each other, and the wrong value won. A vote helps when the right answer is the most common one, even if it is not the majority of all answers.

> [!IMPORTANT] What our run can show, and what it cannot
> We cannot run R1-Zero. It is a 671-billion-parameter model, and its training run is not public. We run the released **DeepSeek-R1-Distill-Qwen-1.5B**, a small model fine-tuned on R1's answers ([Part 6](part-6-distillation.md)). Next to it we run its base model **Qwen2.5-Math-1.5B**, prompted with R1-Zero's template from Table 1. The base model is our picture of "step 0", and the distill our picture of a model that reasons at length.
>
> Neither is R1-Zero, and we train nothing here. So our run shows the behaviour, not the training curve. Our limits on a laptop (Apple M-series GPU): 20 MATH-500 problems picked at random (seed 0) and 8 samples each. We sample with temperature 0.6 and top-p 0.95 as in the paper. Each answer gets at most 2,048 new tokens, where the paper allows 32,768.

**Method 1: from scratch.** The paper's formula, written as a loop, and a vote over our own normalised answer strings:

```python
def pass_at_1_m1(groups):
    """The paper's formula, page 12: pass@1 = (1/k) * sum_i p_i for each question, then the mean over questions."""
    per_q = []
    for g in groups:                              # g: the k correctness values (0 or 1) of one question
        per_q.append(sum(g) / len(g))
    return sum(per_q) / len(per_q)


def cons_m1(answers, gold):
    """Majority vote: the most common normalised answer wins (ties: the one seen first). None votes do not count."""
    votes = [normalise(a) for a in answers if a is not None]
    if not votes:
        return False, None
    counts = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts, key=lambda v: (counts[v], -votes.index(v)))
    return best == normalise(gold), best
```

`pass_at_1_m1` takes, for each question, the list of $$k$$ correctness values and averages it. Then it averages over the questions. `cons_m1` counts how often each final answer occurs and keeps the most common one. `normalise` removes spaces, `\left`, `\dfrac` and similar, so that `\dfrac{1}{4}` and `\frac14` count as one answer.

**Method 2: with libraries.** NumPy does the mean in one line. For the vote, two answers join one class when the **math-verify** library says they are equal as mathematics. It parses LaTeX with SymPy, so `0.25` and `\frac{1}{4}` vote together:

```python
def pass_at_1_m2(groups):
    return float(np.mean(np.array(groups, dtype=float)))      # [questions, k] -> one number


def cons_m2(answers, gold):
    """Majority vote where two answers are one class if math-verify says they are equal."""
    classes = []                                  # [representative, count]
    for a in answers:
        if a is None:
            continue
        for c in classes:
            if same_m2(c[0], a):
                c[1] += 1
                break
        else:
            classes.append([a, 1])
    if not classes:
        return False, None
    best = max(classes, key=lambda c: c[1])[0]     # max keeps the first class on a tie
    return correct_m2(best, gold), best
```

The grading of each answer is also done twice: our own string compare, and math-verify. One policy decides what we grade, and Part 6 uses the same one.

An answer that hits the 2,048-token limit has no final answer, so it counts as wrong. For the distill, we grade only the text after `</think>`: the summary it writes once it has finished thinking. A distill answer that never closes `</think>` counts as wrong. In a finished answer, the final answer is the last `\boxed{}`. Here is what the run printed (the full log is `results/r1_part4_stdout.txt`):

```text
20 MATH-500 problems x 8 samples, temperature 0.6, top-p 0.95, max_new_tokens 2048

== Qwen2.5-Math-1.5B (base, R1-Zero template): 160 answers
grading: ours and math-verify agree on 153/160; ours says 94 correct, math-verify 101 (we use math-verify)
   differ: problem 388 gold '\\frac{11}{2}' pred '5.5' ours False math-verify True
   differ: problem 388 gold '\\frac{11}{2}' pred '5.5' ours False math-verify True
   differ: problem 388 gold '\\frac{11}{2}' pred '5.5' ours False math-verify True
   differ: problem 388 gold '\\frac{11}{2}' pred '5.5' ours False math-verify True
   differ: problem 388 gold '\\frac{11}{2}' pred '5.5' ours False math-verify True
   differ: problem 456 gold '-2,1' pred '1, -2' ours False math-verify True
   differ: problem 456 gold '-2,1' pred '1, -2' ours False math-verify True
truncated at 2048 tokens: 12/160 (scored as wrong); finished but no final answer found: 11/160
answers that finished inside 2048 tokens: 148/160, of which correct: 101/148 = 0.682
pass@1 (mean of 8 samples per problem): Method 1 0.6312, Method 2 numpy 0.6312, agree: True
cons@8 (majority vote): Method 1 0.7000 (14/20), Method 2 0.7500 (15/20)
pass@8 (at least one of 8 correct): 0.8000 (16/20)
correct answers out of 8, per problem: 8 6 8 6 6 7 6 7 1 8 6 6 0 7 0 8 5 0 0 6

== DeepSeek-R1-Distill-Qwen-1.5B: 160 answers
grading: ours and math-verify agree on 160/160; ours says 61 correct, math-verify 61 (we use math-verify)
truncated at 2048 tokens: 96/160 (scored as wrong); finished but no final answer found: 0/160
answers that finished inside 2048 tokens: 64/160, of which correct: 61/64 = 0.953
pass@1 (mean of 8 samples per problem): Method 1 0.3812, Method 2 numpy 0.3812, agree: True
cons@8 (majority vote): Method 1 0.6000 (12/20), Method 2 0.6000 (12/20)
pass@8 (at least one of 8 correct): 0.6000 (12/20)
correct answers out of 8, per problem: 5 7 3 0 0 8 3 2 0 8 2 6 0 8 0 5 0 0 0 4
```

![Terminal output of r1_part4_analyze.py with the grading, pass@1 and vote lines of both models](/img/papers/deepseek-r1/r1_part4-run.png)

Start with the checks. Pass@1 from the loop and from NumPy agree to the last digit for both models. The two graders agree on 153 of 160 base answers and on all 160 distill answers.

Every difference is a case our strings miss and math-verify gets right: `5.5` against the reference `\frac{11}{2}`, and `1, -2` against `-2,1`. The same split explains why the two votes differ for the base model (14 against 15 problems). Our string vote treats `5.5` and `\frac{11}{2}` as two different answers. We keep the math-verify grades.

Now the surprise. The base model scores 0.631 and the reasoning model only 0.381. The reason is our length limit. The distill hit 2,048 tokens on 96 of its 160 answers, and a cut-off answer has no final answer, so it counts as wrong. Of the 64 distill answers that did finish, 61 were right: 95.3%. The base model writes short answers (median 518 tokens) and finished 148 of 160.

The paper gives this same distill 83.9% pass@1 on MATH-500 (Table 5, page 14), with a limit of 32,768 tokens. So our low number is not a fault of the model. It is a lesson about reasoning models: they need room to think. Cutting one off at 2,048 tokens is like taking the exam paper away halfway through. It is also why the paper sets its limit so high.

The vote helps both models. For the distill, pass@1 is 0.381 but the vote over 8 answers is right on 12 of 20 problems (0.600). When the distill can solve a problem, it usually solves it in a few of its 8 tries. The right answer then wins the vote. Keep the sample size in mind: with 20 problems, one problem moves a score by 5 points.

### Table 2, column by column {§Table 2}

> [!PAPER] · Table 2 · page 7
> [![Table 2 of the paper: AIME 2024 pass@1 and cons@64, MATH-500, GPQA Diamond, LiveCodeBench and Codeforces rating for OpenAI-o1-mini, OpenAI-o1-0912 and DeepSeek-R1-Zero, with the R1-Zero row highlighted](/img/papers/deepseek-r1/p4-table2.png)](/img/papers/deepseek-r1/p4-table2.png)
>
> **What it says:** "Comparison of DeepSeek-R1-Zero and OpenAI o1 models on reasoning-related benchmarks." R1-Zero: 71.0, 86.7, 95.9, 73.3, 50.0 and a rating of 1444.
>
> **What it means:**
>
> Read the table one column at a time, and always against the o1-0912 row, the stronger of the two OpenAI models. R1-Zero wins two columns and loses four. Both wins are mathematics. The two large losses are both programming.
>
> This is a fair summary of what pure RL bought. It is not a model that beats o1 everywhere. It is a model that got close to o1 on reasoning, from a base model, with no human examples of reasoning at all.

<figure class="fig"><svg viewBox="0 0 750 492" role="img" aria-label="Six bar panels from Table 2. AIME pass@1: o1-mini 63.6, o1-0912 74.4, R1-Zero 71.0. AIME cons@64: 80.0, 83.3, 86.7. MATH-500: 90.0, 94.8, 95.9. GPQA Diamond: 60.0, 77.3, 73.3. LiveCodeBench: 53.8, 63.4, 50.0. Codeforces rating: 1820, 1843, 1444."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="50.0" text-anchor="start">AIME pass@1</text><line class="axis" x1="20" y1="226" x2="240" y2="226"/><g class="mark"><title>o1-mini, AIME pass@1: 63.6</title><path class="s3" d="M32,226.0V134.6Q32,130.6 36,130.6H80Q84,130.6 84,134.6V226.0Z"/></g><text class="t-val" x="58.0" y="124.6" text-anchor="middle">63.6</text><text class="t-tick" x="58.0" y="242.0" text-anchor="middle">o1-mini</text><g class="mark"><title>o1-0912, AIME pass@1: 74.4</title><path class="s2" d="M100,226.0V118.4Q100,114.4 104,114.4H148Q152,114.4 152,118.4V226.0Z"/></g><text class="t-val t-strong" x="126.0" y="108.4" text-anchor="middle">74.4</text><text class="t-tick" x="126.0" y="242.0" text-anchor="middle">o1-0912</text><g class="mark"><title>R1-Zero, AIME pass@1: 71.0</title><path class="s1" d="M168,226.0V123.5Q168,119.5 172,119.5H216Q220,119.5 220,123.5V226.0Z"/></g><text class="t-val" x="194.0" y="113.5" text-anchor="middle">71.0</text><text class="t-tick" x="194.0" y="242.0" text-anchor="middle">R1-Zero</text><text class="t-title" x="270.0" y="50.0" text-anchor="start">AIME cons@64</text><line class="axis" x1="270" y1="226" x2="490" y2="226"/><g class="mark"><title>o1-mini, AIME cons@64: 80.0</title><path class="s3" d="M282,226.0V110.0Q282,106.0 286,106.0H330Q334,106.0 334,110.0V226.0Z"/></g><text class="t-val" x="308.0" y="100.0" text-anchor="middle">80.0</text><text class="t-tick" x="308.0" y="242.0" text-anchor="middle">o1-mini</text><g class="mark"><title>o1-0912, AIME cons@64: 83.3</title><path class="s2" d="M350,226.0V105.05Q350,101.05 354,101.05H398Q402,101.05 402,105.05V226.0Z"/></g><text class="t-val" x="376.0" y="95.0" text-anchor="middle">83.3</text><text class="t-tick" x="376.0" y="242.0" text-anchor="middle">o1-0912</text><g class="mark"><title>R1-Zero, AIME cons@64: 86.7</title><path class="s1" d="M418,226.0V99.94999999999999Q418,95.94999999999999 422,95.94999999999999H466Q470,95.94999999999999 470,99.94999999999999V226.0Z"/></g><text class="t-val t-strong" x="444.0" y="89.9" text-anchor="middle">86.7</text><text class="t-tick" x="444.0" y="242.0" text-anchor="middle">R1-Zero</text><text class="t-title" x="520.0" y="50.0" text-anchor="start">MATH-500 pass@1</text><line class="axis" x1="520" y1="226" x2="740" y2="226"/><g class="mark"><title>o1-mini, MATH-500 pass@1: 90.0</title><path class="s3" d="M532,226.0V95.0Q532,91.0 536,91.0H580Q584,91.0 584,95.0V226.0Z"/></g><text class="t-val" x="558.0" y="85.0" text-anchor="middle">90.0</text><text class="t-tick" x="558.0" y="242.0" text-anchor="middle">o1-mini</text><g class="mark"><title>o1-0912, MATH-500 pass@1: 94.8</title><path class="s2" d="M600,226.0V87.80000000000001Q600,83.80000000000001 604,83.80000000000001H648Q652,83.80000000000001 652,87.80000000000001V226.0Z"/></g><text class="t-val" x="626.0" y="77.8" text-anchor="middle">94.8</text><text class="t-tick" x="626.0" y="242.0" text-anchor="middle">o1-0912</text><g class="mark"><title>R1-Zero, MATH-500 pass@1: 95.9</title><path class="s1" d="M668,226.0V86.15Q668,82.15 672,82.15H716Q720,82.15 720,86.15V226.0Z"/></g><text class="t-val t-strong" x="694.0" y="76.2" text-anchor="middle">95.9</text><text class="t-tick" x="694.0" y="242.0" text-anchor="middle">R1-Zero</text><text class="t-title" x="20.0" y="286.0" text-anchor="start">GPQA Diamond</text><line class="axis" x1="20" y1="462" x2="240" y2="462"/><g class="mark"><title>o1-mini, GPQA Diamond: 60.0</title><path class="s3" d="M32,462.0V376.0Q32,372.0 36,372.0H80Q84,372.0 84,376.0V462.0Z"/></g><text class="t-val" x="58.0" y="366.0" text-anchor="middle">60.0</text><text class="t-tick" x="58.0" y="478.0" text-anchor="middle">o1-mini</text><g class="mark"><title>o1-0912, GPQA Diamond: 77.3</title><path class="s2" d="M100,462.0V350.05Q100,346.05 104,346.05H148Q152,346.05 152,350.05V462.0Z"/></g><text class="t-val t-strong" x="126.0" y="340.1" text-anchor="middle">77.3</text><text class="t-tick" x="126.0" y="478.0" text-anchor="middle">o1-0912</text><g class="mark"><title>R1-Zero, GPQA Diamond: 73.3</title><path class="s1" d="M168,462.0V356.05Q168,352.05 172,352.05H216Q220,352.05 220,356.05V462.0Z"/></g><text class="t-val" x="194.0" y="346.1" text-anchor="middle">73.3</text><text class="t-tick" x="194.0" y="478.0" text-anchor="middle">R1-Zero</text><text class="t-title" x="270.0" y="286.0" text-anchor="start">LiveCodeBench</text><line class="axis" x1="270" y1="462" x2="490" y2="462"/><g class="mark"><title>o1-mini, LiveCodeBench: 53.8</title><path class="s3" d="M282,462.0V385.3Q282,381.3 286,381.3H330Q334,381.3 334,385.3V462.0Z"/></g><text class="t-val" x="308.0" y="375.3" text-anchor="middle">53.8</text><text class="t-tick" x="308.0" y="478.0" text-anchor="middle">o1-mini</text><g class="mark"><title>o1-0912, LiveCodeBench: 63.4</title><path class="s2" d="M350,462.0V370.9Q350,366.9 354,366.9H398Q402,366.9 402,370.9V462.0Z"/></g><text class="t-val t-strong" x="376.0" y="360.9" text-anchor="middle">63.4</text><text class="t-tick" x="376.0" y="478.0" text-anchor="middle">o1-0912</text><g class="mark"><title>R1-Zero, LiveCodeBench: 50.0</title><path class="s1" d="M418,462.0V391.0Q418,387.0 422,387.0H466Q470,387.0 470,391.0V462.0Z"/></g><text class="t-val" x="444.0" y="381.0" text-anchor="middle">50.0</text><text class="t-tick" x="444.0" y="478.0" text-anchor="middle">R1-Zero</text><text class="t-title" x="520.0" y="286.0" text-anchor="start">Codeforces rating</text><line class="axis" x1="520" y1="462" x2="740" y2="462"/><g class="mark"><title>o1-mini, Codeforces rating: 1820</title><path class="s3" d="M532,462.0V329.5Q532,325.5 536,325.5H580Q584,325.5 584,329.5V462.0Z"/></g><text class="t-val" x="558.0" y="319.5" text-anchor="middle">1820</text><text class="t-tick" x="558.0" y="478.0" text-anchor="middle">o1-mini</text><g class="mark"><title>o1-0912, Codeforces rating: 1843</title><path class="s2" d="M600,462.0V327.775Q600,323.775 604,323.775H648Q652,323.775 652,327.775V462.0Z"/></g><text class="t-val t-strong" x="626.0" y="317.8" text-anchor="middle">1843</text><text class="t-tick" x="626.0" y="478.0" text-anchor="middle">o1-0912</text><g class="mark"><title>R1-Zero, Codeforces rating: 1444</title><path class="s1" d="M668,462.0V357.7Q668,353.7 672,353.7H716Q720,353.7 720,357.7V462.0Z"/></g><text class="t-val" x="694.0" y="347.7" text-anchor="middle">1444</text><text class="t-tick" x="694.0" y="478.0" text-anchor="middle">R1-Zero</text><text class="t-tick" x="20.0" y="22.0" text-anchor="start">Table 2 redrawn: higher is better in every panel; the best value of each panel is bold</text></svg><figcaption>Table 2 redrawn as bars. R1-Zero (blue) is best in two panels: AIME with a vote of 64 (86.7) and MATH-500 (95.9). It is a little below o1-0912 on AIME pass@1 and GPQA Diamond, and far below both o1 models on LiveCodeBench (50.0) and the Codeforces rating (1444).</figcaption></figure>

Two of the columns are new in this series.

> [!DEFINITION] LiveCodeBench and the Codeforces rating
> **LiveCodeBench** is a set of programming problems collected from contests over time. The paper uses problems from August 2024 to January 2025, so they are newer than most training data. A program counts as right only if it passes the hidden tests. The **Codeforces rating** is the score system of the Codeforces contest site. Human players gain points when they beat stronger players and lose points when they lose. The paper turns the model's results on 10 contests into the rating a human with the same results would have.

Now the six columns, against o1-0912:

- **AIME pass@1: 71.0 against 74.4.** R1-Zero is 3.4 points below. It is above o1-mini (63.6).
- **AIME cons@64: 86.7 against 83.3.** With a vote over 64 answers, R1-Zero is 3.4 points above. Both models use the same vote here, so this is a fair comparison.
- **MATH-500: 95.9 against 94.8.** A small win, on easier problems where both models are close to the top.
- **GPQA Diamond: 73.3 against 77.3.** Science questions that need knowledge as well as reasoning. R1-Zero is 4.0 points below, but far above o1-mini (60.0).
- **LiveCodeBench: 50.0 against 63.4.** The largest gap: 13.4 points. R1-Zero is also below o1-mini (53.8).
- **Codeforces: 1444 against 1843.** About 400 rating points below, and 376 below o1-mini (1820).

The v1 paper does not explain the weak code scores of R1-Zero. We can only point to two facts. The finished DeepSeek-R1 does much better on the same two tests: 65.9 on LiveCodeBench and a rating of 2029 (Table 4, page 13). And the conclusion of the paper says that RL was not yet used much for software engineering tasks, because testing code is slow (page 16). Both facts fit the reading that R1-Zero's RL mostly sharpened mathematics, but the paper does not state it.

> [!PAPER] · §2.2.4 · page 7
> [![The paragraph after Table 2, with without the need for any supervised fine-tuning data, majority voting and from 71.0% to 86.7% highlighted](/img/papers/deepseek-r1/p4-majority.png)](/img/papers/deepseek-r1/p4-majority.png)
>
> **What it says:** RL gives R1-Zero "robust reasoning capabilities without the need for any supervised fine-tuning data". With "majority voting" on AIME, its score "escalates from 71.0% to 86.7%, thereby exceeding the performance of OpenAI-o1-0912".
>
> **What it means:**
>
> The first sentence is the real claim of the paper. Every earlier recipe for a reasoning model started with supervised fine-tuning on worked solutions. R1-Zero skips that step. The reward only says "right" or "wrong" at the end.
>
> The second sentence needs care. A vote of 64 answers costs 64 times the compute of one answer. It shows how much right knowledge the model holds. On most AIME problems its most common answer is right, even when a single sample is not. But it is not the same as being right in one try.
>
> Why does the vote help so much here, from 71.0 to 86.7? AIME answers are whole numbers from 0 to 999. A wrong answer can be any of a thousand values, so wrong answers rarely agree. A right answer is one value. On a problem where the model is right 40% of the time and its wrong answers scatter, the right value still wins the vote.

### Figure 2: the training curve {§Figure 2}

> [!PAPER] · Figure 2 · page 7
> [![Figure 2: AIME accuracy of DeepSeek-R1-Zero during training, a blue pass@1 curve and a red cons@16 curve over about 8,000 steps, with dashed lines for o1-0912 pass@1 and cons@64. The caption phrase about 16 responses is highlighted](/img/papers/deepseek-r1/p4-figure2.png)](/img/papers/deepseek-r1/p4-figure2.png)
>
> **What it says:** "AIME accuracy of DeepSeek-R1-Zero during training. For each question, we sample 16 responses and calculate the overall average accuracy to ensure a stable evaluation."
>
> **What it means:**
>
> The horizontal axis is RL steps; one step is one update of the model. Every few hundred steps the authors stop, sample 16 answers to each of the 30 AIME problems, and score them. The blue curve is pass@1, the mean over all 16 answers. The red curve is the vote over the same 16 answers (cons@16). The two dashed lines are o1-0912's scores from Table 2.
>
> Look at the legend. The red curve is a vote of 16, not 64. The 86.7% of Table 2 is a vote of 64. The red curve reaches about the same height at its last point, but the two numbers come from different votes.
>
> The blue curve rises fast in the first 3,000 steps and then more slowly. It also goes down at times. "Steady and consistent" describes the trend, not every point.

Why 16 answers per problem? With 30 problems, one answer per problem gives a very noisy score. Our script prints how noisy:

```text
AIME decoding noise on 30 fixed questions (each question p(1-p) <= 0.25): 1 sample per question, standard error at most 9.1 points; 16 independent samples per question, at most 2.3 points (an upper bound; it leaves out the noise of which 30 questions were chosen)
```

A score that moves 9 points by chance cannot show a curve. The 30 questions are fixed, and the 16 answers to one question are independent draws from the model. Each question has some true rate of success $$p$$, and one draw has variance $$p(1-p)$$, which is never more than 0.25. So with 16 answers per question, the noise from sampling is at most 2.3 points, four times less than with one answer.

This number covers only the dice rolls of decoding. It says nothing about a second kind of noise: a different set of 30 questions would give a different score. More answers per question do not reduce that noise.

<figure class="fig"><svg viewBox="0 0 710 310" role="img" aria-label="Sketch of Figure 2 from the numbers in the text: pass@1 on AIME 2024 rises from 15.6 percent to 71.0 percent during RL; majority voting reaches 86.7 percent at the end; dashed reference lines at 74.4 and 83.3 percent for OpenAI o1-0912. The straight line only joins two points; the real curve is in the screenshot."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="70" y1="270.0" x2="540" y2="270.0"/><text class="t-tick" x="62.0" y="274.0" text-anchor="end">0%</text><line class="grid" x1="70" y1="224.0" x2="540" y2="224.0"/><text class="t-tick" x="62.0" y="228.0" text-anchor="end">20%</text><line class="grid" x1="70" y1="178.0" x2="540" y2="178.0"/><text class="t-tick" x="62.0" y="182.0" text-anchor="end">40%</text><line class="grid" x1="70" y1="132.0" x2="540" y2="132.0"/><text class="t-tick" x="62.0" y="136.0" text-anchor="end">60%</text><line class="grid" x1="70" y1="86.0" x2="540" y2="86.0"/><text class="t-tick" x="62.0" y="90.0" text-anchor="end">80%</text><line class="grid" x1="70" y1="40.0" x2="540" y2="40.0"/><text class="t-tick" x="62.0" y="44.0" text-anchor="end">100%</text><line class="axis" x1="70" y1="270" x2="540" y2="270"/><line class="base-line" x1="70" y1="98.9" x2="540" y2="98.9"/><line class="base-line" x1="70" y1="78.4" x2="540" y2="78.4"/><line class="l1" stroke-dasharray="6 5" x1="80" y1="234.1" x2="530" y2="106.7"/><circle class="s1 ring" cx="80" cy="234.1" r="6"/><circle class="s1 ring" cx="530" cy="106.7" r="6"/><circle class="s2 ring" cx="530" cy="70.6" r="6"/><text class="t-note" x="90.0" y="238.1" text-anchor="start">pass@1 at the start: 15.6%</text><text class="t-note" x="520.0" y="136.7" text-anchor="end">pass@1 at the end: 71.0%</text><text class="t-note" x="520.0" y="60.6" text-anchor="end">majority vote at the end: 86.7%</text><text class="t-tick" x="546.0" y="102.9" text-anchor="start">o1-0912 pass@1 74.4%</text><text class="t-tick" x="546.0" y="82.4" text-anchor="start">o1-0912 cons@64 83.3%</text><text class="t-tick" x="305.0" y="294.0" text-anchor="middle">RL training steps (start to end, thousands of steps)</text><text class="t-tick" x="10.0" y="24.0" text-anchor="start">AIME 2024 accuracy</text></svg><figcaption>The shape of Figure 2, drawn only from numbers in the text: pass@1 starts at 15.6% and ends at 71.0%; the vote ends at 86.7%; the dashed lines are o1-0912 at 74.4% and 83.3% from Table 2. The straight line only joins the two stated points. The real, bumpy curve is in the screenshot above.</figcaption></figure>

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1, later long version · §2.3 · pages 4 and 5
> [![Figure 1 of the later long version: on the left the AIME accuracy curve continued to 10,000 steps with a human participants line, on the right the response length curve rising to about 13,000 tokens with a jump after step 8,000](/img/papers/deepseek-r1/p4-later-fig1.png)](/img/papers/deepseek-r1/p4-later-fig1.png)
> [![The sentence of the later version: pass@1 on AIME 2024 jumping from an initial 15.6% to 77.9%](/img/papers/deepseek-r1/p4-later-779.png)](/img/papers/deepseek-r1/p4-later-779.png)
>
> **What it says:** in the later version (the one that became the Nature article), the same curve runs to 10,400 steps, and pass@1 goes "from an initial 15.6% to 77.9%". With the vote, "the model's performance can be further improved, achieving an accuracy of 86.7%".
>
> **What it means:**
>
> The v1 paper stops its curve a little after step 8,000 at 71.0%. The later version shows the same run continued for about 2,000 more steps. Pass@1 climbs to 77.9%, above o1-0912's 74.4%. The Nature article gives the same 77.9%.
>
> The dashed line is also new: the average score of human AIME participants. R1-Zero passes it early in training. So the 71.0% of v1 was a point on a curve that was still going up, not a ceiling.

## The self-evolution process {§2.2.4}

> [!PAPER] · §2.2.4 · page 7
> [![The Self-evolution Process paragraph, with initiating RL directly from the base model and without the influence of the supervised fine-tuning stage highlighted](/img/papers/deepseek-r1/p4-self-evolution.png)](/img/papers/deepseek-r1/p4-self-evolution.png)
>
> **What it says:** "By initiating RL directly from the base model, we can closely monitor the model's progression without the influence of the supervised fine-tuning stage."
>
> **What it means:**
>
> This is the scientific reason for R1-Zero. If you fine-tune on worked solutions first, the model copies their style: their length, their habits, their phrases. Then you cannot tell what RL added and what the examples added. Starting from the base model removes the examples. Whatever new habits appear during RL, the reward caused them.
>
> There is one gap in this argument, and we come back to it below. The base model was pre-trained on a large part of the internet. A habit can appear during RL because RL made it more frequent, not because RL invented it.

> [!PAPER] · Figure 3 · page 8
> [![Figure 3: the average response length of DeepSeek-R1-Zero on the training set, rising from a few hundred tokens at step 0 to almost 10,000 tokens after 8,000 steps, with a pale band around the curve. The caption phrase naturally learns to solve reasoning tasks with more thinking time is highlighted](/img/papers/deepseek-r1/p4-figure3.png)](/img/papers/deepseek-r1/p4-figure3.png)
>
> **What it says:** "The average response length of DeepSeek-R1-Zero on the training set during the RL process. DeepSeek-R1-Zero naturally learns to solve reasoning tasks with more thinking time."
>
> **What it means:**
>
> This is the most famous curve of the paper. The vertical axis is the mean number of tokens in an answer, on the training questions. At the start the answers are a few hundred tokens long. By the end of the plot the mean is close to 10,000. The pale band shows how much the length moves from step to step.
>
> The growth is almost a straight line. It does not stop or flatten in this plot. Nothing in the reward asks for length; the template asks only for the two pairs of tags.

> [!PAPER] · §2.2.4 · page 8
> [![The sentences after Figure 3, with not the result of external adjustments and hundreds to thousands of reasoning tokens highlighted](/img/papers/deepseek-r1/p4-thinking-time.png)](/img/papers/deepseek-r1/p4-thinking-time.png)
>
> **What it says:** the growth of thinking time "is not the result of external adjustments but rather an intrinsic development within the model". The computation "ranges from generating hundreds to thousands of reasoning tokens".
>
> **What it means:**
>
> "External adjustments" would be things like a reward for long answers, a rule "think for at least 2,000 tokens", or examples of long solutions. The authors used none. The model was free to write 50 tokens or 5,000 tokens. It chose to write more, step after step.
>
> "Test-time computation" (the words in the paper) is the compute spent while answering, as opposed to training. Each extra token is one more pass through the network. A longer answer is, quite literally, more computing spent on the problem.

<figure class="fig"><svg viewBox="0 0 570 280" role="img" aria-label="Sketch of the shape of Figure 3: the average response length grows during RL from hundreds of tokens to thousands of tokens, with no length reward. The shape is drawn by hand; the real curve is in the screenshot."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="axis" x1="70" y1="240" x2="540" y2="240"/><line class="axis" x1="70" y1="40" x2="70" y2="240"/><polyline class="l1" points="80.0,226.0 91.2,224.4 102.5,222.2 113.8,219.7 125.0,217.0 136.2,214.1 147.5,211.1 158.8,207.9 170.0,204.6 181.2,201.2 192.5,197.7 203.8,194.1 215.0,190.5 226.2,186.7 237.5,182.9 248.8,179.0 260.0,175.1 271.2,171.1 282.5,167.0 293.8,162.9 305.0,158.7 316.2,154.5 327.5,150.2 338.8,145.9 350.0,141.5 361.2,137.1 372.5,132.6 383.8,128.1 395.0,123.6 406.2,119.0 417.5,114.3 428.8,109.7 440.0,104.9 451.2,100.2 462.5,95.4 473.8,90.6 485.0,85.7 496.2,80.9 507.5,75.9 518.8,71.0 530.0,66.0"/><text class="t-note" x="88.0" y="180.0" text-anchor="start">hundreds of tokens</text><text class="t-note" x="526.0" y="58.0" text-anchor="end">thousands of tokens</text><text class="t-tick" x="305.0" y="264.0" text-anchor="middle">RL training steps</text><text class="t-tick" x="10.0" y="24.0" text-anchor="start">average length of a response</text><text class="t-muted" x="526.0" y="220.0" text-anchor="end">no length reward, no instruction to think longer</text></svg><figcaption>The shape of Figure 3 in the words of the text: from hundreds of tokens to thousands of tokens, with no reward for length. The curve is drawn by hand; the screenshot above has the real one.</figcaption></figure>

### Why a correctness reward makes answers longer

Nobody rewarded length, yet length grew. Why? Think of a student whose tests are graded only on the final answer, with no marks for working. One week she checks her sums before she hands in. Her grades go up. Nobody told her to write more, but checking is now a habit, and her papers get longer.

The paper's reading is that RL does the same thing, with numbers. GRPO samples a group of answers to each question ([Part 2](part-2-grpo.md)). Some answers check a step or try a second method, and so they are longer. On a hard question, those are more often right. GRPO gives each answer an advantage: its reward minus the group mean, divided by the group's standard deviation (equation 3).

<figure class="fig"><svg viewBox="0 0 840 270" role="img" aria-label="A group of four sampled answers to one question: two short answers of 300 and 450 tokens are wrong and get advantage minus 1; two long answers of 1,200 and 1,500 tokens are right and get advantage plus 1, so RL makes the long, careful answers more likely."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-tick" x="110.0" y="70.0" text-anchor="end">answer 1</text><g class="mark"><title>answer 1: 300 tokens, reward 0</title><path class="s3" d="M120,56H176.0Q180.0,56 180.0,60V74Q180.0,78 176.0,78H120Z"/></g><text class="t-val" x="188.0" y="72.0" text-anchor="start">300 tokens</text><text class="t-note" x="520.0" y="72.0" text-anchor="start">reward 0</text><text class="t-bad" x="620.0" y="72.0" text-anchor="start">A = -1</text><text class="t-note" x="690.0" y="72.0" text-anchor="start">made less likely</text><text class="t-tick" x="110.0" y="114.0" text-anchor="end">answer 2</text><g class="mark"><title>answer 2: 450 tokens, reward 0</title><path class="s3" d="M120,100H206.0Q210.0,100 210.0,104V118Q210.0,122 206.0,122H120Z"/></g><text class="t-val" x="218.0" y="116.0" text-anchor="start">450 tokens</text><text class="t-note" x="520.0" y="116.0" text-anchor="start">reward 0</text><text class="t-bad" x="620.0" y="116.0" text-anchor="start">A = -1</text><text class="t-note" x="690.0" y="116.0" text-anchor="start">made less likely</text><text class="t-tick" x="110.0" y="158.0" text-anchor="end">answer 3</text><g class="mark"><title>answer 3: 1200 tokens, reward 1</title><path class="s1" d="M120,144H356.0Q360.0,144 360.0,148V162Q360.0,166 356.0,166H120Z"/></g><text class="t-val" x="368.0" y="160.0" text-anchor="start">1200 tokens</text><text class="t-note" x="520.0" y="160.0" text-anchor="start">reward 1</text><text class="t-on" x="620.0" y="160.0" text-anchor="start">A = +1</text><text class="t-note" x="690.0" y="160.0" text-anchor="start">made more likely</text><text class="t-tick" x="110.0" y="202.0" text-anchor="end">answer 4</text><g class="mark"><title>answer 4: 1500 tokens, reward 1</title><path class="s1" d="M120,188H416.0Q420.0,188 420.0,192V206Q420.0,210 416.0,210H120Z"/></g><text class="t-val" x="428.0" y="204.0" text-anchor="start">1500 tokens</text><text class="t-note" x="520.0" y="204.0" text-anchor="start">reward 1</text><text class="t-on" x="620.0" y="204.0" text-anchor="start">A = +1</text><text class="t-note" x="690.0" y="204.0" text-anchor="start">made more likely</text><text class="t-title" x="120.0" y="30.0" text-anchor="start">One question, four sampled answers (made-up lengths)</text><text class="t-muted" x="120.0" y="252.0" text-anchor="start">mean reward 0.5, standard deviation 0.5, so A = (r - 0.5) / 0.5 = +1 or -1 (equation 3)</text></svg><figcaption>One question, four sampled answers with made-up lengths. The two short answers are wrong (reward 0) and the two long ones are right (reward 1). The mean reward is 0.5 and the standard deviation 0.5 (dividing by G), so the advantages are -1, -1, +1, +1. The code in Part 2 divides by G-1, as PyTorch does by default, which gives -0.866 and +0.866; the direction of the update is the same. The update makes the long, careful answers more likely and the short guesses less likely.</figcaption></figure>

Work the tiny example by hand. The rewards are 0, 0, 1, 1, with mean 0.5. Each reward is 0.5 away from the mean, so the standard deviation is 0.5.

The advantages are $$(0 - 0.5)/0.5 = -1$$ for the two wrong answers and $$(1 - 0.5)/0.5 = +1$$ for the two right ones. The update raises the probability of every token of the right answers. That includes the tokens "let me verify this" and the second method. It lowers the probability of the quick guesses.

One step changes the model very little. But the change has a direction: toward the kind of answer that checks. After thousands of steps the typical answer checks a lot, and it is long. Nothing pushes the other way. The reward has no cost per token. The only limit is the maximum length of an answer.

Three honest notes complete the picture. The first one comes from DeepSeek itself.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1, later long version · §2.1 · page 3
> [![The training details of R1-Zero in the later version, with sample 16 outputs with a maximum length of 32,768 tokens, before the 8.2k step and 65,536 tokens afterward, exhibit a significant jump at the 8.2k step and 10,400 steps highlighted](/img/papers/deepseek-r1/p4-later-limits.png)](/img/papers/deepseek-r1/p4-later-limits.png)
>
> **What it says:** for each question R1-Zero sampled "16 outputs with a maximum length of 32,768 tokens before the 8.2k step and 65,536 tokens afterward. As a result, both the performance and response length of DeepSeek-R1-Zero exhibit a significant jump at the 8.2k step". Training ran for "10,400 steps".
>
> **What it means:**
>
> The v1 paper does not give these settings; the later version does. They matter for Figure 3. An answer can only grow up to the length limit. At step 8,200 the authors doubled the limit, and both the length and the score jumped.
>
> So one external adjustment did happen: the room to think was made larger. The steady climb before step 8,200 is the model's own. The jump at step 8,200 is partly the new room. The v1 figure ends at about step 8,400, so it shows only the start of that jump.

The second note is about the optimizer. In the GRPO of DeepSeekMath, the loss of each answer is averaged over its own tokens. A later study found that this average has a side effect.

> [!PAPER] Liu et al. (2025), Understanding R1-Zero-Like Training · §3.1 · page 6
> [![The response-level length bias bullet, with longer responses are penalized less due to their larger |o_i| and lengthier responses highlighted](/img/papers/deepseek-r1/p4-liu-length.png)](/img/papers/deepseek-r1/p4-liu-length.png)
>
> **What it says:** dividing the loss by the answer length $$|o_i|$$ creates a "response-level length bias". For a wrong answer (negative advantage), "longer responses are penalized less due to their larger $$|o_i|$$, causing the policy to prefer lengthier responses among incorrect ones."
>
> **What it means:**
>
> Take two wrong answers, one of 100 tokens and one of 1,000 tokens, both with advantage $$-1$$. The penalty of each answer is spread over its own tokens. Each token of the short answer gets one hundredth of it; each token of the long answer gets one thousandth. So the long wrong answer is pushed down ten times more gently per token.
>
> Over many steps, this lets long wrong answers survive better than short wrong ones. So part of the growth in Figure 3 may come from this bias, not only from "thinking more helps". Liu et al. remove the division and call the result Dr. GRPO. They report the same accuracy with shorter wrong answers. This does not cancel the paper's story; the length curve mixes two causes.

The third note is our own data. We can ask the question of the tiny example on real answers. Inside one group, are the right answers longer or shorter than the wrong ones?

```text
== Qwen2.5-Math-1.5B (base, R1-Zero template)
length (new tokens): mean 687, median 518; correct answers mean 502 (n=101), wrong answers mean 1005 (n=59), wrong and finished mean 739 (n=47)
length histogram, bins of 256 tokens: correct [23, 38, 18, 14, 7, 1, 0, 0], wrong [8, 11, 7, 9, 5, 4, 0, 15]; numpy agrees: True
GRPO view, groups with mixed results: positive-advantage answers mean length 609 (n=69), negative-advantage answers 785 (n=27)

== DeepSeek-R1-Distill-Qwen-1.5B
length (new tokens): mean 1834, median 2048; correct answers mean 1497 (n=61), wrong answers mean 2042 (n=99), wrong and finished mean 1865 (n=3)
length histogram, bins of 256 tokens: correct [0, 0, 0, 3, 13, 20, 10, 15], wrong [0, 0, 0, 0, 0, 0, 1, 98]; numpy agrees: True
GRPO view, groups with mixed results: positive-advantage answers mean length 1605 (n=37), negative-advantage answers 2048 (n=35)
```

Our data does not show "longer is better" inside a group. It shows the opposite.

In both models the right answers are shorter on average than the wrong ones. The base model: 502 against 1,005 tokens. The distill: 1,497 against 2,042. In the groups with mixed results, the answers that GRPO would reward are shorter too (609 against 785, and 1,605 against 2,048). One GRPO step on these groups would push toward shorter answers, not longer.

Why does this not contradict Figure 3? Look at where the wrong answers are. Many of them are long because the model got lost: it went round in circles until our limit cut it off. Even the base model's wrong answers that did finish are longer than its right ones (739 against 502 tokens). A model that does not know the way talks the longest. These long wrong answers are exactly the ones that the length bias of Liu et al. punishes too gently.

So our snapshot cannot tell which way a training run would move. It shows that length and correctness are tangled: careful checking makes an answer longer, and so does being lost. The paper reports the result of thousands of steps on harder problems, with a much higher limit. The story "checking pays, and checking takes tokens" is the authors' reading, and it is plausible. But a laptop run of two fixed models cannot prove it.

<figure class="fig"><svg viewBox="0 0 640 292" role="img" aria-label="Histogram of answer lengths for DeepSeek-R1-Distill-Qwen-1.5B, 160 answers: correct answers [0, 0, 0, 3, 13, 20, 10, 15], wrong answers [0, 0, 0, 0, 0, 0, 1, 98], in bins of 256 tokens from 0 to 2,048."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="60" y1="246.0" x2="620" y2="246.0"/><text class="t-tick" x="52.0" y="250.0" text-anchor="end">0</text><line class="grid" x1="60" y1="206.0" x2="620" y2="206.0"/><text class="t-tick" x="52.0" y="210.0" text-anchor="end">20</text><line class="grid" x1="60" y1="166.0" x2="620" y2="166.0"/><text class="t-tick" x="52.0" y="170.0" text-anchor="end">40</text><line class="grid" x1="60" y1="126.0" x2="620" y2="126.0"/><text class="t-tick" x="52.0" y="130.0" text-anchor="end">60</text><line class="grid" x1="60" y1="86.0" x2="620" y2="86.0"/><text class="t-tick" x="52.0" y="90.0" text-anchor="end">80</text><line class="grid" x1="60" y1="46.0" x2="620" y2="46.0"/><text class="t-tick" x="52.0" y="50.0" text-anchor="end">100</text><g class="mark"><title>0 to 256 tokens, correct: 0</title><path class="s1" d="M66.0,246.01V246.01Q66.0,246.0 66.01,246.0H92.99Q93.0,246.0 93.0,246.01V246.01Z"/></g><g class="mark"><title>0 to 256 tokens, wrong: 0</title><path class="s3" d="M95.0,246.01V246.01Q95.0,246.0 95.01,246.0H121.99Q122.0,246.0 122.0,246.01V246.01Z"/></g><g class="mark"><title>256 to 512 tokens, correct: 0</title><path class="s1" d="M136.0,246.01V246.01Q136.0,246.0 136.01,246.0H162.99Q163.0,246.0 163.0,246.01V246.01Z"/></g><g class="mark"><title>256 to 512 tokens, wrong: 0</title><path class="s3" d="M165.0,246.01V246.01Q165.0,246.0 165.01,246.0H191.99Q192.0,246.0 192.0,246.01V246.01Z"/></g><g class="mark"><title>512 to 768 tokens, correct: 0</title><path class="s1" d="M206.0,246.01V246.01Q206.0,246.0 206.01,246.0H232.99Q233.0,246.0 233.0,246.01V246.01Z"/></g><g class="mark"><title>512 to 768 tokens, wrong: 0</title><path class="s3" d="M235.0,246.01V246.01Q235.0,246.0 235.01,246.0H261.99Q262.0,246.0 262.0,246.01V246.01Z"/></g><g class="mark"><title>768 to 1024 tokens, correct: 3</title><path class="s1" d="M276.0,246.0V244.0Q276.0,240.0 280.0,240.0H299.0Q303.0,240.0 303.0,244.0V246.0Z"/></g><text class="t-val" x="289.5" y="236.0" text-anchor="middle">3</text><g class="mark"><title>768 to 1024 tokens, wrong: 0</title><path class="s3" d="M305.0,246.01V246.01Q305.0,246.0 305.01,246.0H331.99Q332.0,246.0 332.0,246.01V246.01Z"/></g><g class="mark"><title>1024 to 1280 tokens, correct: 13</title><path class="s1" d="M346.0,246.0V224.0Q346.0,220.0 350.0,220.0H369.0Q373.0,220.0 373.0,224.0V246.0Z"/></g><text class="t-val" x="359.5" y="216.0" text-anchor="middle">13</text><g class="mark"><title>1024 to 1280 tokens, wrong: 0</title><path class="s3" d="M375.0,246.01V246.01Q375.0,246.0 375.01,246.0H401.99Q402.0,246.0 402.0,246.01V246.01Z"/></g><g class="mark"><title>1280 to 1536 tokens, correct: 20</title><path class="s1" d="M416.0,246.0V210.0Q416.0,206.0 420.0,206.0H439.0Q443.0,206.0 443.0,210.0V246.0Z"/></g><text class="t-val" x="429.5" y="202.0" text-anchor="middle">20</text><g class="mark"><title>1280 to 1536 tokens, wrong: 0</title><path class="s3" d="M445.0,246.01V246.01Q445.0,246.0 445.01,246.0H471.99Q472.0,246.0 472.0,246.01V246.01Z"/></g><g class="mark"><title>1536 to 1792 tokens, correct: 10</title><path class="s1" d="M486.0,246.0V230.0Q486.0,226.0 490.0,226.0H509.0Q513.0,226.0 513.0,230.0V246.0Z"/></g><text class="t-val" x="499.5" y="222.0" text-anchor="middle">10</text><g class="mark"><title>1536 to 1792 tokens, wrong: 1</title><path class="s3" d="M515.0,246.0V246.0Q515.0,244.0 517.0,244.0H540.0Q542.0,244.0 542.0,246.0V246.0Z"/></g><text class="t-val" x="528.5" y="240.0" text-anchor="middle">1</text><g class="mark"><title>1792 to 2048 tokens, correct: 15</title><path class="s1" d="M556.0,246.0V220.0Q556.0,216.0 560.0,216.0H579.0Q583.0,216.0 583.0,220.0V246.0Z"/></g><text class="t-val" x="569.5" y="212.0" text-anchor="middle">15</text><g class="mark"><title>1792 to 2048 tokens, wrong: 98</title><path class="s3" d="M585.0,246.0V54.0Q585.0,50.0 589.0,50.0H608.0Q612.0,50.0 612.0,54.0V246.0Z"/></g><text class="t-val" x="598.5" y="46.0" text-anchor="middle">98</text><text class="t-tick" x="60.0" y="262.0" text-anchor="middle">0</text><text class="t-tick" x="130.0" y="262.0" text-anchor="middle">256</text><text class="t-tick" x="200.0" y="262.0" text-anchor="middle">512</text><text class="t-tick" x="270.0" y="262.0" text-anchor="middle">768</text><text class="t-tick" x="340.0" y="262.0" text-anchor="middle">1,024</text><text class="t-tick" x="410.0" y="262.0" text-anchor="middle">1,280</text><text class="t-tick" x="480.0" y="262.0" text-anchor="middle">1,536</text><text class="t-tick" x="550.0" y="262.0" text-anchor="middle">1,792</text><text class="t-tick" x="620.0" y="262.0" text-anchor="middle">2,048</text><line class="axis" x1="60" y1="246" x2="620" y2="246"/><text class="t-tick" x="340.0" y="282.0" text-anchor="middle">answer length in new tokens (the last bin ends at our 2,048 limit)</text><text class="t-title" x="20.0" y="20.0" text-anchor="start">DeepSeek-R1-Distill-Qwen-1.5B, 160 answers</text><rect class="s1" x="430" y="24" width="10" height="10" rx="2"/><text class="t-note" x="446.0" y="33.0" text-anchor="start">correct (61)</text><rect class="s3" x="530" y="24" width="10" height="10" rx="2"/><text class="t-note" x="546.0" y="33.0" text-anchor="start">wrong (99)</text></svg><figcaption>Lengths of the distill's 160 answers in bins of 256 tokens. 98 of the 99 wrong answers sit in the last bin, and 96 of them were cut off by our 2,048-token limit. The 61 right answers spread from 768 tokens to the limit.</figcaption></figure>

<figure class="fig"><svg viewBox="0 0 640 292" role="img" aria-label="Histogram of answer lengths for Qwen2.5-Math-1.5B with the R1-Zero template, 160 answers: correct answers [23, 38, 18, 14, 7, 1, 0, 0], wrong answers [8, 11, 7, 9, 5, 4, 0, 15], in bins of 256 tokens from 0 to 2,048."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="60" y1="246.0" x2="620" y2="246.0"/><text class="t-tick" x="52.0" y="250.0" text-anchor="end">0</text><line class="grid" x1="60" y1="196.0" x2="620" y2="196.0"/><text class="t-tick" x="52.0" y="200.0" text-anchor="end">10</text><line class="grid" x1="60" y1="146.0" x2="620" y2="146.0"/><text class="t-tick" x="52.0" y="150.0" text-anchor="end">20</text><line class="grid" x1="60" y1="96.0" x2="620" y2="96.0"/><text class="t-tick" x="52.0" y="100.0" text-anchor="end">30</text><line class="grid" x1="60" y1="46.0" x2="620" y2="46.0"/><text class="t-tick" x="52.0" y="50.0" text-anchor="end">40</text><g class="mark"><title>0 to 256 tokens, correct: 23</title><path class="s1" d="M66.0,246.0V135.0Q66.0,131.0 70.0,131.0H89.0Q93.0,131.0 93.0,135.0V246.0Z"/></g><text class="t-val" x="79.5" y="127.0" text-anchor="middle">23</text><g class="mark"><title>0 to 256 tokens, wrong: 8</title><path class="s3" d="M95.0,246.0V210.0Q95.0,206.0 99.0,206.0H118.0Q122.0,206.0 122.0,210.0V246.0Z"/></g><text class="t-val" x="108.5" y="202.0" text-anchor="middle">8</text><g class="mark"><title>256 to 512 tokens, correct: 38</title><path class="s1" d="M136.0,246.0V60.0Q136.0,56.0 140.0,56.0H159.0Q163.0,56.0 163.0,60.0V246.0Z"/></g><text class="t-val" x="149.5" y="52.0" text-anchor="middle">38</text><g class="mark"><title>256 to 512 tokens, wrong: 11</title><path class="s3" d="M165.0,246.0V195.0Q165.0,191.0 169.0,191.0H188.0Q192.0,191.0 192.0,195.0V246.0Z"/></g><text class="t-val" x="178.5" y="187.0" text-anchor="middle">11</text><g class="mark"><title>512 to 768 tokens, correct: 18</title><path class="s1" d="M206.0,246.0V160.0Q206.0,156.0 210.0,156.0H229.0Q233.0,156.0 233.0,160.0V246.0Z"/></g><text class="t-val" x="219.5" y="152.0" text-anchor="middle">18</text><g class="mark"><title>512 to 768 tokens, wrong: 7</title><path class="s3" d="M235.0,246.0V215.0Q235.0,211.0 239.0,211.0H258.0Q262.0,211.0 262.0,215.0V246.0Z"/></g><text class="t-val" x="248.5" y="207.0" text-anchor="middle">7</text><g class="mark"><title>768 to 1024 tokens, correct: 14</title><path class="s1" d="M276.0,246.0V180.0Q276.0,176.0 280.0,176.0H299.0Q303.0,176.0 303.0,180.0V246.0Z"/></g><text class="t-val" x="289.5" y="172.0" text-anchor="middle">14</text><g class="mark"><title>768 to 1024 tokens, wrong: 9</title><path class="s3" d="M305.0,246.0V205.0Q305.0,201.0 309.0,201.0H328.0Q332.0,201.0 332.0,205.0V246.0Z"/></g><text class="t-val" x="318.5" y="197.0" text-anchor="middle">9</text><g class="mark"><title>1024 to 1280 tokens, correct: 7</title><path class="s1" d="M346.0,246.0V215.0Q346.0,211.0 350.0,211.0H369.0Q373.0,211.0 373.0,215.0V246.0Z"/></g><text class="t-val" x="359.5" y="207.0" text-anchor="middle">7</text><g class="mark"><title>1024 to 1280 tokens, wrong: 5</title><path class="s3" d="M375.0,246.0V225.0Q375.0,221.0 379.0,221.0H398.0Q402.0,221.0 402.0,225.0V246.0Z"/></g><text class="t-val" x="388.5" y="217.0" text-anchor="middle">5</text><g class="mark"><title>1280 to 1536 tokens, correct: 1</title><path class="s1" d="M416.0,246.0V245.0Q416.0,241.0 420.0,241.0H439.0Q443.0,241.0 443.0,245.0V246.0Z"/></g><text class="t-val" x="429.5" y="237.0" text-anchor="middle">1</text><g class="mark"><title>1280 to 1536 tokens, wrong: 4</title><path class="s3" d="M445.0,246.0V230.0Q445.0,226.0 449.0,226.0H468.0Q472.0,226.0 472.0,230.0V246.0Z"/></g><text class="t-val" x="458.5" y="222.0" text-anchor="middle">4</text><g class="mark"><title>1536 to 1792 tokens, correct: 0</title><path class="s1" d="M486.0,246.01V246.01Q486.0,246.0 486.01,246.0H512.99Q513.0,246.0 513.0,246.01V246.01Z"/></g><g class="mark"><title>1536 to 1792 tokens, wrong: 0</title><path class="s3" d="M515.0,246.01V246.01Q515.0,246.0 515.01,246.0H541.99Q542.0,246.0 542.0,246.01V246.01Z"/></g><g class="mark"><title>1792 to 2048 tokens, correct: 0</title><path class="s1" d="M556.0,246.01V246.01Q556.0,246.0 556.01,246.0H582.99Q583.0,246.0 583.0,246.01V246.01Z"/></g><g class="mark"><title>1792 to 2048 tokens, wrong: 15</title><path class="s3" d="M585.0,246.0V175.0Q585.0,171.0 589.0,171.0H608.0Q612.0,171.0 612.0,175.0V246.0Z"/></g><text class="t-val" x="598.5" y="167.0" text-anchor="middle">15</text><text class="t-tick" x="60.0" y="262.0" text-anchor="middle">0</text><text class="t-tick" x="130.0" y="262.0" text-anchor="middle">256</text><text class="t-tick" x="200.0" y="262.0" text-anchor="middle">512</text><text class="t-tick" x="270.0" y="262.0" text-anchor="middle">768</text><text class="t-tick" x="340.0" y="262.0" text-anchor="middle">1,024</text><text class="t-tick" x="410.0" y="262.0" text-anchor="middle">1,280</text><text class="t-tick" x="480.0" y="262.0" text-anchor="middle">1,536</text><text class="t-tick" x="550.0" y="262.0" text-anchor="middle">1,792</text><text class="t-tick" x="620.0" y="262.0" text-anchor="middle">2,048</text><line class="axis" x1="60" y1="246" x2="620" y2="246"/><text class="t-tick" x="340.0" y="282.0" text-anchor="middle">answer length in new tokens (the last bin ends at our 2,048 limit)</text><text class="t-title" x="20.0" y="20.0" text-anchor="start">Qwen2.5-Math-1.5B with the R1-Zero template, 160 answers</text><rect class="s1" x="430" y="24" width="10" height="10" rx="2"/><text class="t-note" x="446.0" y="33.0" text-anchor="start">correct (101)</text><rect class="s3" x="530" y="24" width="10" height="10" rx="2"/><text class="t-note" x="546.0" y="33.0" text-anchor="start">wrong (59)</text></svg><figcaption>Lengths of the base model's 160 answers with the R1-Zero template. 61 of the 101 right answers are shorter than 512 tokens. The 15 wrong answers in the last bin include the 12 that our limit cut off.</figcaption></figure>

## Reflection and new approaches {§2.2.4}

> [!PAPER] · §2.2.4 · page 8
> [![The paragraph on sophisticated behaviors, with revisits and reevaluates its previous steps, exploration of alternative approaches and not explicitly programmed highlighted](/img/papers/deepseek-r1/p4-reflection.png)](/img/papers/deepseek-r1/p4-reflection.png)
>
> **What it says:** behaviours such as reflection, "where the model revisits and reevaluates its previous steps", and "the exploration of alternative approaches to problem-solving arise spontaneously". They are "not explicitly programmed".
>
> **What it means:**
>
> Reflection is the model reading its own work and asking whether it is right. In text it looks like "Wait, let me check that step" or "That gives a negative length, which is impossible". Exploring an alternative looks like "Alternatively, we could use the quadratic formula". Both are what a good student does on an exam.
>
> "Not explicitly programmed" is true in a narrow sense: the template of Table 1 does not ask for them, and no reward counts them ([Part 3](part-3-rewards-and-the-template.md)). The paper's earlier sentence says the authors avoided "mandating reflective reasoning" on purpose, so that they could watch what appears.

The v1 paper describes these behaviours but does not count them. The later version does.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1, later long version · Appendix C.2 · pages 38 and 39
> [![The paragraphs of Appendix C.2, with rises 5- to 7-fold, nearly absent during early training and significant spikes after step 8000 highlighted](/img/papers/deepseek-r1/p4-later-reflection.png)](/img/papers/deepseek-r1/p4-later-reflection.png)
> [![Figure 9 of the later version: on the left the frequency of reflection words rising from about 1,500 to between 5,000 and 14,000 over 10,000 steps; on the right the frequency of the word wait, near zero until step 4,000 and spiking above 1,400 after step 8,000](/img/papers/deepseek-r1/p4-later-fig9.png)](/img/papers/deepseek-r1/p4-later-fig9.png)
>
> **What it says:** the authors counted ten reflective words chosen by three people ("wait", "mistake", "however", "but", "retry", "error", "verify", "wrong", "evaluate", "check"). Their count "rises 5- to 7-fold compared to the start of training". The word "wait" "was nearly absent during early training, showed occasional usage between steps 4000-7000, and then exhibited significant spikes after step 8000."
>
> **What it means:**
>
> This is the evidence that the v1 paper lacked. Reflection was not zero at the start: the left plot begins at about 1,500 words, not at 0. RL multiplied it by 5 to 7.
>
> The word "wait" is a sharper story. It is almost never used for 4,000 steps, then rarely, then very often after step 8,000. Note that step 8,200 is also where the length limit doubled. The published curves cannot separate the two events.

Our run counts four phrases in every answer of both small models. Method 1 uses regular expressions; Method 2 splits the text into words and counts them with `collections.Counter`:

```python
REFLECT = {                                   # the four phrases we count; lower case, word boundaries
    'wait': r'\bwait\b',
    'let me check': r'\blet me check\b',
    'alternatively': r'\balternatively\b',
    're-evaluate': r'\bre-?evaluat\w*',
}


def reflect_m1(text):
    t = text.lower()
    return {k: len(re.findall(p, t)) for k, p in REFLECT.items()}


def reflect_m2(text):
    """Split into lower-case word tokens and count with Counter; two-word phrases from pairs of tokens."""
    toks = re.findall(r"[a-z]+(?:-[a-z]+)?", text.lower())
    c1 = Counter(toks)
    c2 = Counter(zip(toks, toks[1:], toks[2:]))
    return {'wait': c1['wait'], 'let me check': c2[('let', 'me', 'check')], 'alternatively': c1['alternatively'],
            're-evaluate': sum(n for w, n in c1.items() if w.startswith(('reevaluat', 're-evaluat')))}
```

`\b` in a regular expression means "word boundary", so `\bwait\b` counts "Wait," and "wait." but not "awaiting". Method 2 gets the same effect a different way: it cuts the text into words first, so "awaiting" is one word and never equals "wait". The two methods must agree on every total.

```text
== Qwen2.5-Math-1.5B (base, R1-Zero template)
reflection words, total over 160 answers: Method 1 {'wait': 0, 'let me check': 0, 'alternatively': 0, 're-evaluate': 0}
                                              Method 2 {'wait': 0, 'let me check': 0, 'alternatively': 0, 're-evaluate': 0}, agree: True
per answer: wait 0.00, let me check 0.00, alternatively 0.00, re-evaluate 0.00
per 1,000 tokens: wait 0.00, let me check 0.00, alternatively 0.00, re-evaluate 0.00
answers with at least one "wait": 0/160
"wait" per answer: correct answers 0.00, wrong answers 0.00

== DeepSeek-R1-Distill-Qwen-1.5B
reflection words, total over 160 answers: Method 1 {'wait': 719, 'let me check': 77, 'alternatively': 208, 're-evaluate': 0}
                                              Method 2 {'wait': 719, 'let me check': 77, 'alternatively': 208, 're-evaluate': 0}, agree: True
per answer: wait 4.49, let me check 0.48, alternatively 1.30, re-evaluate 0.00
per 1,000 tokens: wait 2.45, let me check 0.26, alternatively 0.71, re-evaluate 0.00
answers with at least one "wait": 159/160
"wait" per answer: correct answers 3.00, wrong answers 5.41
```

<figure class="fig"><svg viewBox="0 0 680 300" role="img" aria-label="Reflection phrases per answer. wait: base 0.00, distill 4.49; let me check: base 0.00, distill 0.48; alternatively: base 0.00, distill 1.30; re-evaluate: base 0.00, distill 0.00."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-tick" x="120.0" y="74.0" text-anchor="end">"wait"</text><g class="mark"><title>base, wait: 0.00 per answer</title><path class="s3" d="M130,54H130Q132,54 132,56V70Q132,72 130,72H130Z"/></g><text class="t-val" x="138.0" y="68.0" text-anchor="start">0.00</text><g class="mark"><title>distill, wait: 4.49 per answer</title><path class="s1" d="M130,76H596.0Q600.0,76 600.0,80V90Q600.0,94 596.0,94H130Z"/></g><text class="t-val" x="606.0" y="90.0" text-anchor="start">4.49</text><text class="t-tick" x="120.0" y="130.0" text-anchor="end">"let me check"</text><g class="mark"><title>base, let me check: 0.00 per answer</title><path class="s3" d="M130,110H130Q132,110 132,112V126Q132,128 130,128H130Z"/></g><text class="t-val" x="138.0" y="124.0" text-anchor="start">0.00</text><g class="mark"><title>distill, let me check: 0.48 per answer</title><path class="s1" d="M130,132H176.3337969401947Q180.3337969401947,132 180.3337969401947,136V146Q180.3337969401947,150 176.3337969401947,150H130Z"/></g><text class="t-val" x="186.3" y="146.0" text-anchor="start">0.48</text><text class="t-tick" x="120.0" y="186.0" text-anchor="end">"alternatively"</text><g class="mark"><title>base, alternatively: 0.00 per answer</title><path class="s3" d="M130,166H130Q132,166 132,168V182Q132,184 130,184H130Z"/></g><text class="t-val" x="138.0" y="180.0" text-anchor="start">0.00</text><g class="mark"><title>distill, alternatively: 1.30 per answer</title><path class="s1" d="M130,188H261.96662030598054Q265.96662030598054,188 265.96662030598054,192V202Q265.96662030598054,206 261.96662030598054,206H130Z"/></g><text class="t-val" x="272.0" y="202.0" text-anchor="start">1.30</text><text class="t-tick" x="120.0" y="242.0" text-anchor="end">"re-evaluate"</text><g class="mark"><title>base, re-evaluate: 0.00 per answer</title><path class="s3" d="M130,222H130Q132,222 132,224V238Q132,240 130,240H130Z"/></g><text class="t-val" x="138.0" y="236.0" text-anchor="start">0.00</text><g class="mark"><title>distill, re-evaluate: 0.00 per answer</title><path class="s1" d="M130,244H130Q132,244 132,246V260Q132,262 130,262H130Z"/></g><text class="t-val" x="138.0" y="258.0" text-anchor="start">0.00</text><rect class="s3" x="130" y="20" width="10" height="10" rx="2"/><text class="t-note" x="146.0" y="29.0" text-anchor="start">Qwen2.5-Math-1.5B (base)</text><rect class="s1" x="350" y="20" width="10" height="10" rx="2"/><text class="t-note" x="366.0" y="29.0" text-anchor="start">R1-Distill-Qwen-1.5B</text><text class="t-muted" x="130.0" y="288.0" text-anchor="start">mean count per answer, same 20 problems x 8 samples for each model</text></svg><figcaption>Reflection phrases per answer on the same 160 prompts. The base model with the R1-Zero template wrote none of the four phrases. The distill wrote "wait" 4.49 times per answer and "alternatively" 1.30 times. Neither model wrote "re-evaluate".</figcaption></figure>

The two methods agree on every total. The contrast is total. In 160 answers, the base model never wrote "wait", "let me check" or "alternatively". The distill wrote "wait" 719 times, and 159 of its 160 answers contain at least one. Per 1,000 tokens that is 2.45 "wait"s, so the count is not only a side effect of longer answers.

The distill learned this habit by copying R1's answers, and R1 inherited it from the RL of R1-Zero. So this is the behaviour of Table 3, carried down into a model small enough for a laptop. It is not the training curve: we see the end point, not the climb.

One more line is worth a look. The distill's wrong answers contain more "wait"s than its right ones: 5.41 against 3.00 per answer. A model says "wait" most when it struggles, and the struggling answers are also the long ones that run out of room. More reflection does not mean more right answers here, which matches the finding of Liu et al. in the next section.

## The aha moment {§2.2.4, Table 3}

> [!PAPER] · §2.2.4 · page 8
> [![The Aha Moment paragraph, with intermediate version of the model and allocate more thinking time to a problem by reevaluating its initial approach highlighted](/img/papers/deepseek-r1/p4-aha.png)](/img/papers/deepseek-r1/p4-aha.png)
>
> **What it says:** an "aha moment" occurs in "an intermediate version of the model". In this phase, R1-Zero "learns to allocate more thinking time to a problem by reevaluating its initial approach".
>
> **What it means:**
>
> "Intermediate version" means a checkpoint from the middle of training, not the final model. The authors read its answers and found one where the model stops, says it has found a problem, and starts again.
>
> The key words are "reevaluating its initial approach". Many answers check a number. This one doubts its whole plan. Here is the answer itself.

> [!PAPER] · Table 3 · page 9
> [![Table 3: the question If a > 1, then the sum of the real solutions of sqrt(a - sqrt(a + x)) = x is equal to, and the response that squares both sides, reaches a quartic, then writes Wait, wait. Wait. That's an aha moment I can flag here, and restarts step by step. The aha sentence and reevaluate this step-by-step are highlighted](/img/papers/deepseek-r1/p4-table3.png)](/img/papers/deepseek-r1/p4-table3.png)
>
> **What it says:** the question is "If $$a > 1$$, then the sum of the real solutions of $$\sqrt{a - \sqrt{a + x}} = x$$ is equal to". The model squares twice and reaches $$x^4 - 2ax^2 - x + (a^2 - a) = 0$$. Then: "Wait, wait. Wait. That's an aha moment I can flag here. Let's reevaluate this step-by-step to identify if the correct sum can be ..."
>
> **What it means:**
>
> Read the response as a story. The model starts with the standard move: square both sides to remove a square root. It moves the inner root to one side and squares again. Now it has a polynomial of degree four, a quartic. The dots hide some lines. Then it stops, and the tone changes: "Wait, wait. Wait." It even names what is happening, in words a person would use.
>
> The caption calls this "an anthropomorphic tone": the model talks like a person. Nobody wrote "Wait" into a training example for R1-Zero; it had no supervised reasoning fine-tuning before RL. Its pretraining text surely held some written-out maths, but no one fine-tuned it on reasoning traces. It produced these tokens, they led to a right answer often enough, and the reward made them more likely.
>
> The table cuts the answer with "..." at the most interesting moment. It never shows the final answer. So, what was the model working on, and what is the right answer? We can solve it ourselves.

### Solve it yourself

The model's first path hides a trap, and it helps to see it. Here are the steps, with the reason for each.

<figure class="fig"><svg viewBox="0 0 780 290" role="img" aria-label="Six steps to solve the square root equation: square, square again, get a quartic whose four roots sum to zero, factor it into two quadratics, check the four candidates in the first equation, and keep the one root x equals square root of 4a minus 3, minus 1, over 2."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><rect class="box" x="20.0" y="40.0" width="230.0" height="90.0" rx="10"/><text class="t-title" x="34.0" y="64.0" text-anchor="start">1. square</text><text class="t-mono" x="34.0" y="90.0" text-anchor="start">a - √(a+x) = x²</text><text class="t-note" x="34.0" y="114.0" text-anchor="start">needs x ≥ 0</text><line class="edge" x1="252.0" y1="85.0" x2="268.0" y2="85.0" marker-end="url(#ah)"/><rect class="box" x="270.0" y="40.0" width="230.0" height="90.0" rx="10"/><text class="t-title" x="284.0" y="64.0" text-anchor="start">2. square again</text><text class="t-mono" x="284.0" y="90.0" text-anchor="start">a + x = (a - x²)²</text><text class="t-note" x="284.0" y="114.0" text-anchor="start">needs a - x² ≥ 0</text><line class="edge" x1="502.0" y1="85.0" x2="518.0" y2="85.0" marker-end="url(#ah)"/><rect class="box-2" x="520.0" y="40.0" width="230.0" height="90.0" rx="10"/><text class="t-title" x="534.0" y="64.0" text-anchor="start">3. a quartic</text><text class="t-mono" x="534.0" y="90.0" text-anchor="start">x⁴-2ax²-x+a²-a = 0</text><text class="t-note" x="534.0" y="114.0" text-anchor="start">no x³ term: 4 roots sum to 0</text><rect class="box" x="20.0" y="160.0" width="230.0" height="90.0" rx="10"/><text class="t-title" x="34.0" y="184.0" text-anchor="start">4. factor</text><text class="t-mono" x="34.0" y="210.0" text-anchor="start">(x²+x+1-a)(x²-x-a) = 0</text><text class="t-note" x="34.0" y="234.0" text-anchor="start">four candidates</text><line class="edge" x1="252.0" y1="205.0" x2="268.0" y2="205.0" marker-end="url(#ah)"/><rect class="box" x="270.0" y="160.0" width="230.0" height="90.0" rx="10"/><text class="t-title" x="284.0" y="184.0" text-anchor="start">5. check each</text><text class="t-mono" x="284.0" y="210.0" text-anchor="start">in the first equation</text><text class="t-note" x="284.0" y="234.0" text-anchor="start">only one survives</text><line class="edge" x1="502.0" y1="205.0" x2="518.0" y2="205.0" marker-end="url(#ah)"/><rect class="box-on" x="520.0" y="160.0" width="230.0" height="90.0" rx="10"/><text class="t-title" x="534.0" y="184.0" text-anchor="start">6. answer</text><text class="t-mono" x="534.0" y="210.0" text-anchor="start">x = (√(4a-3) - 1)/2</text><text class="t-note" x="534.0" y="234.0" text-anchor="start">the sum is this one root</text><line class="edge" x1="635.0" y1="132.0" x2="135.0" y2="158.0" marker-end="url(#ah)"/></svg><figcaption>The six steps: square, square again, get a quartic with no x-cubed term, factor it into two quadratics, test the four candidates in the first equation, and keep the one that works: x = (sqrt(4a - 3) - 1) / 2.</figcaption></figure>

**Step 1: square once.** A square root is never negative, so the right side $$x$$ must be at least 0. Squaring gives

$$
a - \sqrt{a + x} = x^2 \quad\Longrightarrow\quad \sqrt{a + x} = a - x^2 .
$$

The left side is again a square root, so we also need $$a - x^2 \ge 0$$.

**Step 2: square again.**

$$
a + x = (a - x^2)^2 = a^2 - 2a x^2 + x^4 \quad\Longrightarrow\quad x^4 - 2a x^2 - x + (a^2 - a) = 0 .
$$

This is the quartic of Table 3. The model got this far.

**Step 3: the trap.** The quartic has no $$x^3$$ term. By Vieta's rule, the four roots of $$x^4 + c_3 x^3 + \dots = 0$$ add up to $$-c_3$$. Here $$c_3 = 0$$, so the four roots add up to 0. A hurried solver writes "the sum is 0" and stops. That answer is wrong, because squaring adds false roots.

> [!DEFINITION] Extraneous root
> A value that solves an equation you got by squaring, but not the equation you started with. Squaring forgets signs: $$x = -2$$ does not solve $$x = 2$$, but it solves $$x^2 = 4$$. Every root found after squaring must be checked in the first equation.

**Step 4: factor.** Read the quartic as a quadratic in $$a$$, not in $$x$$:

$$
a^2 - (2x^2 + 1)\,a + (x^4 - x) = 0 .
$$

Its discriminant is $$(2x^2 + 1)^2 - 4(x^4 - x) = 4x^2 + 4x + 1 = (2x + 1)^2$$, a perfect square. So $$a = x^2 + x + 1$$ or $$a = x^2 - x$$, and the quartic splits into two quadratics in $$x$$:

$$
(x^2 + x + 1 - a)\,(x^2 - x - a) = 0 .
$$

**Step 5: check the four candidates.** The second factor gives $$x^2 = x + a$$, so $$a - x^2 = -x$$. Step 1 needs $$a - x^2 \ge 0$$, so $$x \le 0$$, and also $$x \ge 0$$. Only $$x = 0$$ is left, and then $$a = 0$$, which breaks $$a > 1$$. Both roots of the second factor are false.

The first factor gives $$x = \frac{-1 \pm \sqrt{4a - 3}}{2}$$. The minus sign gives a negative $$x$$, which is false. The plus sign gives a positive $$x$$, because $$4a - 3 > 1$$ when $$a > 1$$.

**Step 6: the answer.** There is exactly one real solution, so the "sum" is that one root:

$$
x = \frac{\sqrt{4a - 3} - 1}{2} .
$$

Check it with $$a = 3$$: $$x = (\sqrt{9} - 1)/2 = 1$$, and $$\sqrt{3 - \sqrt{3 + 1}} = \sqrt{3 - 2} = 1$$. It works.

**Method 1: by hand in Python.** Build the four candidates from the two quadratics, then keep only those that pass the first equation:

```python
def candidates(a):
    """The quartic is a quadratic in a:  a^2 - (2x^2 + 1) a + (x^4 - x) = 0.
    Its discriminant is (2x^2 + 1)^2 - 4(x^4 - x) = 4x^2 + 4x + 1 = (2x + 1)^2, a perfect square, so
    a = x^2 + x + 1  or  a = x^2 - x.  Each one is a quadratic in x with two roots: four candidates in all."""
    out = []
    d1 = 4 * a - 3                                   # x^2 + x + (1 - a) = 0
    if d1 >= 0:
        out += [(-1 + math.sqrt(d1)) / 2, (-1 - math.sqrt(d1)) / 2]
    d2 = 4 * a + 1                                   # x^2 - x - a = 0
    if d2 >= 0:
        out += [(1 + math.sqrt(d2)) / 2, (1 - math.sqrt(d2)) / 2]
    return out


def satisfies_original(a, x, tol=1e-9):
    """Plug x into sqrt(a - sqrt(a + x)) = x. Both square roots must exist and the result must equal x."""
    if a + x < 0:
        return False
    inner = a - math.sqrt(a + x)
    if inner < 0:
        return False
    return abs(math.sqrt(inner) - x) < tol
```

**Method 2: with SymPy and NumPy.** SymPy expands and factors the quartic for any $$a$$, and solves the first equation for $$a = 3$$. `numpy.roots` finds the four roots of the quartic numerically. Then SymPy checks the answer for every $$a > 1$$ by substitution. We write $$a$$ through $$u = \sqrt{4a - 3} - 1 > 0$$, so that every square root becomes a plain expression.

```text
equation: sqrt(a - sqrt(a + x)) = x, with a > 1
after squaring twice (sympy): Eq(a**2 - 2*a*x**2 - a + x**4 - x, 0)
factored (sympy):             Eq((-a + x**2 - x)*(-a + x**2 + x + 1), 0)
sum of the 4 quartic roots (Vieta, minus the x^3 coefficient): 0  <- the tempting wrong answer
a = 3: Method 1 candidates [1.0, -2.0, 2.3028, -1.3028], kept [1.0]
        numpy.roots [2.3028, 1.0, -1.3028, -2.0], sum +0.0000; formula (sqrt(4a-3)-1)/2 = 1.0000
a = 7: Method 1 candidates [2.0, -3.0, 3.1926, -2.1926], kept [2.0]
        numpy.roots [3.1926, 2.0, -2.1926, -3.0], sum -0.0000; formula (sqrt(4a-3)-1)/2 = 2.0000
a = 13: Method 1 candidates [3.0, -4.0, 4.1401, -3.1401], kept [3.0]
        numpy.roots [4.1401, 3.0, -3.1401, -4.0], sum +0.0000; formula (sqrt(4a-3)-1)/2 = 3.0000
a = 2.5: Method 1 candidates [0.8229, -1.8229, 2.1583, -1.1583], kept [0.8229]
        numpy.roots [2.1583, 0.8229, -1.1583, -1.8229], sum -0.0000; formula (sqrt(4a-3)-1)/2 = 0.8229
sympy solve for a = 3: [1]
sympy: a + X = (u + 2)**2/4  so sqrt(a + X) = u/2 + 1 ; a - sqrt(a + X) = u**2/4  so the left side = u/2  and X = u/2
sympy check: left side - X = 0  for every a > 1
answer: the only real solution is x = (sqrt(4a - 3) - 1)/2, so the sum is (sqrt(4a - 3) - 1)/2
```

![Terminal output of r1_part4_aha_math.py: the quartic, its factors, the candidates and the one kept root](/img/papers/deepseek-r1/r1_part4_aha_math-run.png)

Read the lines for $$a = 3$$. The quartic has four real roots: 2.3028, 1, -1.3028 and -2. NumPy says they add up to 0, exactly as Vieta says. Only 1 passes the first equation. The same happens for $$a = 7$$ (kept: 2) and $$a = 13$$ (kept: 3), as the formula says.

The last SymPy line shows that the left side minus our answer is 0 for every $$a > 1$$. So the right answer is $$\frac{\sqrt{4a - 3} - 1}{2}$$, and "0" was the trap.

This trap may be what the model sensed, but the paper does not say so. The excerpt in Table 3 stops at the quartic. It does not show the model using Vieta, writing a sum of 0, or finding the false roots. All we can say is this: squaring leads to a tidy quartic with an easy wrong sum. At that point, the model chose to go back and check its steps.

We do not know how the answer ended, because the table cuts it. Doubt at that point was a good idea.

### A real "Wait" from our run

Here is one real "Wait" from the distill on our MATH-500 problems, copied from its output:

```text
== a real "Wait" (distill, problem 20: 'Evaluate $(1+2i)6-3i$.', sample 2, 1579 tokens, final answer '6 + 9i', correct True)
Another thought: if I were to write this out in terms of real and imaginary parts, the expression is (1*6 - 2i*6) - 3i, which is (6 - 12i) - 3i. Wait, hold on, that's a different approach. Is that correct?

Wait, no, that's not right. Because when you have (1 + 2i)*6, it's 6*(1) + 6*(2i) = 6 + 12i. Then subtracting 3i, so 6 + 12i - 3i. So, that's 6 + (12i - 3i) = 6 + 9i. But if I distribute the multiplication first, maybe I can see another way. Let's think about it: 1*6 is 6, 2i*6 is 12i, so that's 6 + 12i. Then subtract 3i, so 6 + 12i - 3i is 6 + 9i.
```

The problem is one line: evaluate $$(1+2i)6 - 3i$$. The model already had the right answer, $$6 + 9i$$, a few paragraphs earlier. Here it tries "another thought" and makes a sign error: it writes $$(1 \cdot 6 - 2i \cdot 6)$$ where a plus belongs. The next sentence doubts the step. The one after rejects it ("that's not right") and redoes the product.

That is reflection in miniature: an error, a doubt, a repair. It also shows the cost. This answer used 1,579 tokens for a sum a person does in one line. The habit that saves the model on hard problems also runs on easy ones.

> [!PAPER] · §2.2.4 · page 8
> [![The second Aha Moment paragraph, with we simply provide it with the right incentives highlighted](/img/papers/deepseek-r1/p4-aha-incentives.png)](/img/papers/deepseek-r1/p4-aha-incentives.png)
>
> **What it says:** "rather than explicitly teaching the model on how to solve a problem, we simply provide it with the right incentives, and it autonomously develops advanced problem-solving strategies."
>
> **What it means:**
>
> This is the lesson the authors draw. Supervised learning tells the model what to write. RL only tells it whether the end result was right, and lets it find the way. If a habit helps, the habit grows.
>
> The sentence is strong, and the next section asks how strong. "Develops" can mean two things. It can mean the model invented the habit. Or it can mean the habit was already there, rarely used, and the reward made it common. The evidence after the paper points more to the second.

### Was the aha moment new?

In the months after the paper, several groups studied R1-Zero-style training in the open. The most direct test came from Sea AI Lab.

> [!PAPER] Liu et al. (2025), Understanding R1-Zero-Like Training · §2.3 · page 4
> [![Section 2.3 of Liu et al., Aha Moment Already Appears in Base Models Including DeepSeek-V3-Base, with already exhibit self-reflection keywords, DeepSeek-V3-Base also generates a decent amount of self-reflections and not positively correlated with higher accuracy highlighted](/img/papers/deepseek-r1/p4-liu-aha.png)](/img/papers/deepseek-r1/p4-liu-aha.png)
>
> **What it says:** earlier studies suggested "there may not be Aha moment" in open replications, because the base models "already exhibit self-reflection keywords". Liu et al. ran DeepSeek-V3-Base itself on 500 MATH questions with the R1 template: "DeepSeek-V3-Base also generates a decent amount of self-reflections". In R1-Zero, self-reflection is more frequent, but "not positively correlated with higher accuracy".
>
> **What it means:**
>
> This is the base model of R1-Zero, before any RL. It already writes "wait" and even "aha" sometimes. So RL did not invent reflection. It found it in the base model and made it much more common, which matches the 5- to 7-fold rise of the later DeepSeek figure.
>
> The second finding is more surprising. In R1-Zero's answers, more reflection did not mean more right answers. One reading: a model reflects most on the problems it finds hard, and those are the ones it gets wrong more often. Liu et al. report the correlation, not a cause.
>
> So the fair statement is this. Pure RL with a correctness reward made reflection and long thinking common, and the scores rose with them. Whether RL created the ability, or only brought out an ability the base model had, is not settled by the paper. The evidence so far points to the second.

Our small run adds one data point. With the R1-Zero template, temperature 0.6 and 2,048 tokens, the base model Qwen2.5-Math-1.5B wrote "wait" in none of 160 answers. The distill wrote it in 159. Liu et al. report some self-reflection for Qwen2.5-Math-1.5B too, counted over 500 questions and several temperatures, with keywords and a language-model check. So the habit can exist in the base model and still be rare enough to miss in 160 samples. Either way, after training it is everywhere.

## Drawback of DeepSeek-R1-Zero {§2.2.4}

> [!PAPER] · §2.2.4 · page 9
> [![The Drawback of DeepSeek-R1-Zero paragraph, with poor readability and language mixing highlighted](/img/papers/deepseek-r1/p4-drawback.png)](/img/papers/deepseek-r1/p4-drawback.png)
>
> **What it says:** R1-Zero "struggles with challenges like poor readability, and language mixing". To make the reasoning readable, the authors build DeepSeek-R1, which uses "RL with human-friendly cold-start data".
>
> **What it means:**
>
> The reward checks two things: the final answer and the tags. It does not read the thinking in between. So the thinking can be messy and still earn full reward. Nothing pushes it to be clear, short or in one language.
>
> Think of scratch paper in an exam. Nobody grades it, so you write half sentences, arrows and crossed-out sums. R1-Zero's thinking is scratch paper written by a model that saw Chinese and English in pre-training. If a Chinese word is the likely next token in the middle of an English sentence, nothing stops it.

> [!DEFINITION] Language mixing
> Switching between languages inside one answer, for example an English question and a chain of thought that moves between English and Chinese. The final answer can still be right. The problem is for the person who reads the thinking.

> [!PAPER] · §2.3.1 · page 10
> [![The Readability bullet of section 2.3.1, with often not suitable, mix multiple languages and lack markdown formatting highlighted](/img/papers/deepseek-r1/p4-readability.png)](/img/papers/deepseek-r1/p4-readability.png)
>
> **What it says:** "A key limitation of DeepSeek-R1-Zero is that its content is often not suitable for reading. Responses may mix multiple languages or lack markdown formatting to highlight answers for users."
>
> **What it means:**
>
> Readability is the reason DeepSeek-R1 exists. A model that thinks well but cannot be read is a research result, not a product. Users want to follow the reasoning, check it and trust it.
>
> The fix comes in two parts, both in [Part 5](part-5-the-r1-pipeline.md). First, a few thousand readable long answers are used to fine-tune the base model before RL (the "cold start"). Second, a language consistency reward is added during RL: the share of words in the target language.

We looked for language mixing in our run and found none: 0 of 160 answers from either model contain a Chinese, Japanese or Korean character. That is expected. The distill learned from R1's answers, and for those the authors "filtered out chain-of-thought with mixed languages" (page 10). The base model had no RL, so it had no chance to drift. R1-Zero's mixing came from RL with a reward that never reads the thinking, and our run cannot reproduce that.

> [!TAKEAWAYS]
> - Pure RL took DeepSeek-V3-Base from 15.6% to 71.0% pass@1 on AIME 2024, with no supervised examples of reasoning. A vote of 64 answers gives 86.7%.
> - Pass@1 here is the mean correctness of $$k$$ sampled answers (temperature 0.6, top-p 0.95), because greedy decoding of long answers repeats itself.
> - In Table 2, R1-Zero beats o1-0912 on AIME cons@64 and MATH-500 and is a little below on AIME pass@1 and GPQA Diamond. It is far below on LiveCodeBench (50.0 against 63.4) and Codeforces (1444 against 1843).
> - Answers grew from hundreds to thousands of tokens with no length reward. A correctness reward favours answers that check their work, and checking takes tokens.
> - Two other causes are in the length curve too. The length limit was doubled at step 8,200 (later version). And GRPO's per-token average penalises long wrong answers less (Liu et al.).
> - Reflection words rose 5 to 7 times during training, and "wait" appeared late (later version, Figure 9). The base model already reflected sometimes.
> - The Table 3 question holds a real trap, though the excerpt does not show the model falling into it. The quartic's four roots sum to 0, but only one root, $$(\sqrt{4a - 3} - 1)/2$$, solves the first equation.
> - On a laptop, the released 1.5B distill wrote "wait" in 159 of 160 answers and its base model in none. With our 2,048-token limit, 96 distill answers were cut off, so its pass@1 (0.381) fell below the base model's (0.631); 61 of its 64 finished answers were right.
> - R1-Zero's thinking is hard to read and mixes languages, because the reward never reads the thinking. DeepSeek-R1 fixes this in Part 5.

## The paper and the code

[Paper (arXiv 2501.12948)](https://arxiv.org/abs/2501.12948) · [Official repository](https://github.com/deepseek-ai/DeepSeek-R1) · [Our scripts for this part](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/deepseek-r1)
