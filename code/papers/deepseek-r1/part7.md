---
title: "Results, Failures and Limits"
description: "How R1 is scored, where it wins and where it loses, the two ideas that failed, and what R1 still cannot do. We write pass@k and majority voting in Python, and test greedy decoding and few-shot prompts on the 1.5B model."
part: 7
covers: "§3, §3.1, Table 4, §4.2, §5"
date: 2026-11-21
tags: [deepseek, reasoning, evaluation, pass-at-k, mcts, llm]
---

A score of 79.8 on a maths test sounds like one clean number. Behind it are choices: which test, which prompt, how many tries, how hot the sampling, and what counts as right. This part reads those choices first. Then it reads the results, the two ideas that did not work, and the four limits the authors list.

{{FIG:p7_eval|Where this part sits, and how §3 scores one question. R1 writes several answers with sampling. pass@1 is the share of right answers; cons@k grades only the most common answer.}}

## Benchmarks {§3}

> [!PAPER] · §3 · page 11
> [![The first lines of the Benchmarks paragraph of Section 3, with MMLU, GPQA Diamond, SimpleQA and SWE-Bench Verified highlighted](/img/papers/deepseek-r1/p7-benchmarks.png)](/img/papers/deepseek-r1/p7-benchmarks.png)
>
> **What it says:** "We evaluate models on MMLU, MMLU-Redux, MMLU-Pro, C-Eval, and CMMLU, IFEval, FRAMES, GPQA Diamond, SimpleQA, C-SimpleQA, SWE-Bench Verified" and more.
>
> **What it means:**
>
> A benchmark is a fixed set of questions with known answers, so two models can sit the same exam. The list is long because no single exam tests everything. A model can be strong at contest maths and weak at facts, and the paper wants to show both sides.
>
> The names look like a random pile, but they fall into a few groups. Each group asks a different question about the model. The table below sorts them the way Table 4 of the paper does.

| Group | Benchmarks | What a question looks like |
|---|---|---|
| Knowledge | MMLU, MMLU-Redux, MMLU-Pro, GPQA Diamond | A multiple-choice question from school or university subjects. GPQA Diamond is graduate-level science, written so that a web search does not help. |
| Facts | SimpleQA, C-SimpleQA (Chinese) | A short question with one fact as the answer, such as a date or a name. |
| Reading | DROP, FRAMES | Answer from a long text or several documents: find the numbers, then compute. |
| Following instructions | IF-Eval | "Write 3 paragraphs, no commas." A program checks the format. |
| Open-ended writing | AlpacaEval 2.0, ArenaHard | A free request, like "plan a trip". Another model judges the answer. |
| Code | LiveCodeBench, Codeforces, SWE Verified, Aider-Polyglot | Contest problems with hidden tests, or a real bug to fix in a real repository. |
| Maths | AIME 2024, MATH-500, CNMO 2024 | A competition problem with one exact answer. |
| Chinese | CLUEWSC, C-Eval, C-SimpleQA | Chinese versions of reading, knowledge and facts. |

> [!PAPER] · §3 · page 12
> [![The rest of the Benchmarks paragraph, with LiveCodeBench, Codeforces, AIME 2024, using LLMs as judges and we only feed the final summary to evaluation to avoid the length bias highlighted](/img/papers/deepseek-r1/p7-judges.png)](/img/papers/deepseek-r1/p7-judges.png)
>
> **What it says:** "we also evaluate our models on open-ended generation tasks using LLMs as judges [...] Here, we only feed the final summary to evaluation to avoid the length bias."
>
> **What it means:**
>
> Maths has one right answer, so a program can grade it. "Plan a trip to Kyoto" has no single right answer. AlpacaEval 2.0 and ArenaHard solve this with a referee: a strong model reads two answers and says which one is better. Here the referee is GPT-4-Turbo-1106.
>
> Model referees have a known weakness: they tend to prefer the longer answer, even when it is not better. R1 makes this worse, because its full output contains thousands of tokens of thinking. If the referee saw all of it, R1 could win on length alone. So the authors show the referee only the final summary, the part a user reads after the thinking. The last RL stage (§2.3.4, [Part 5](part-5-the-r1-pipeline.md)) also judged helpfulness on the summary only.

> [!DEFINITION] LLM as a judge, and length bias
> An **LLM judge** is a language model that grades the answers of another model, usually by comparing two answers to the same request. **Length bias** is its habit of preferring the longer answer. AlpacaEval 2.0 reports a "length-controlled" win rate (LC-winrate) that corrects for it.

## Evaluation prompts {§3}

> [!PAPER] · §3 · page 12
> [![The Evaluation Prompts paragraph, with in a zero-shot setting, we slightly modify the prompt to the zero-shot setting and The CoT in few-shot may hurt the performance of DeepSeek-R1 highlighted](/img/papers/deepseek-r1/p7-prompts.png)](/img/papers/deepseek-r1/p7-prompts.png)
>
> **What it says:** "since the original prompts are few-shot, we slightly modify the prompt to the zero-shot setting. The CoT in few-shot may hurt the performance of DeepSeek-R1."
>
> **What it means:**
>
> A **few-shot** prompt shows the model a few solved examples before the real question. A **zero-shot** prompt asks the question with no examples. For older models, a few examples helped a lot: they showed the format and the style of the answer. Many benchmarks were built in that time, so their official prompts are few-shot.
>
> For R1 the examples get in the way. The examples in such prompts carry short worked solutions. R1 was trained to write long reasoning, then a summary. A prompt full of short solutions pulls it toward the short style it learned to leave behind. So the authors rewrite those prompts as zero-shot. They come back to this in §5 as a limit, and we test it on the small model there.

> [!PAPER] · §3 · page 12
> [![The rest of the Evaluation Prompts paragraph, with 10 Div.2 contests, expected ratings and percentages of competitors, agentless framework and capped at a maximum of 32,768 tokens highlighted](/img/papers/deepseek-r1/p7-code-setup.png)](/img/papers/deepseek-r1/p7-code-setup.png)
>
> **What it says:** Codeforces uses "problems from 10 Div.2 contests", SWE-Bench Verified uses "the agentless framework", and "DeepSeek-R1 outputs are capped at a maximum of 32,768 tokens for each benchmark."
>
> **What it means:**
>
> Each code benchmark has its own rules. For Codeforces, the model solves the problems of ten past contests, and expert-written tests check its code. From the problems it solves, the authors compute the rating and the percentile it would have earned. [Part 1](part-1-the-big-idea.md) explained both numbers.
>
> SWE-Bench Verified gives the model a real bug in a real Python project. "Agentless" means a fixed pipeline: find the file, write a patch, run the tests. The model does not explore the repository on its own. Aider asks for edits in a "diff" format: the model writes only the changes, not the whole file.
>
> The last sentence matters most for a reasoning model. 32,768 tokens is the budget for thinking plus the answer. An answer that has not finished at that point is cut, and a cut answer has no final result, so it counts as wrong.

## Baselines {§3}

> [!PAPER] · §3 · page 12
> [![The Baselines paragraph, with challenging in mainland China and based on official reports highlighted](/img/papers/deepseek-r1/p7-baselines.png)](/img/papers/deepseek-r1/p7-baselines.png)
>
> **What it says:** "Since accessing the OpenAI-o1-1217 API is challenging in mainland China, we report its performance based on official reports."
>
> **What it means:**
>
> Five models stand next to R1: DeepSeek-V3 (the chat model from the same base), Anthropic's Sonnet 3.5, GPT-4o, o1-mini and o1-1217. The authors ran four of them. They could not run o1-1217, so they copied its numbers from OpenAI's own reports.
>
> This has two consequences for Table 4. First, o1-1217 has a "-" on every benchmark that OpenAI did not report. Second, its numbers come from a different setup: other prompts, other sampling, maybe other grading. When R1 and o1-1217 differ by one point, the two setups may explain the gap as well as the two models.

## Evaluation setup {§3}

> [!PAPER] · §3 · page 12
> [![The Evaluation Setup paragraph, with greedy decoding to evaluate long-output reasoning models results in higher repetition rates and significant variability across different checkpoints, sampling temperature of 0.6 and typically between 4 and 64 highlighted](/img/papers/deepseek-r1/p7-setup.png)](/img/papers/deepseek-r1/p7-setup.png)
>
> **What it says:** "using greedy decoding to evaluate long-output reasoning models results in higher repetition rates and significant variability across different checkpoints. Therefore, we default to pass@k evaluation [...] we use a sampling temperature of 0.6 and a top-p value of 0.95 to generate k responses (typically between 4 and 64, depending on the test set size) for each question."
>
> **What it means:**
>
> A model gives a probability for every possible next token. **Greedy decoding** always takes the most likely one. It is simple and gives the same answer every time. For a reasoning model it has two problems, and the paper names both.
>
> The first is repetition. Greedy decoding has no randomness, so once the text enters a cycle it cannot leave it. Suppose the model writes "Let me check again." and the most likely next words lead back to the same sentence. Then it writes that sentence again, and again, until the token limit cuts it. A long output gives more chances to fall into such a cycle.
>
> The second is variance across checkpoints. Greedy decoding is not random: for a fixed model and question it gives the same answer every time. But it hangs on one path, the most likely token at every step. A small change to the weights can flip one early token and send the whole answer down a different path. So two checkpoints that are about equally good can get quite different greedy scores.
>
> Sampling several answers and averaging them measures something smoother: how often the model is right under temperature 0.6 and top-p 0.95. That number moves less when one token flips.

> [!DEFINITION] Temperature and top-p
> Two settings for sampling the next token. **Temperature** $$T$$ divides the model's scores (logits) before the softmax. $$T < 1$$ makes the likely tokens more likely, $$T = 1$$ keeps the model's own probabilities, and $$T \to 0$$ is greedy decoding. **Top-p** (nucleus sampling) keeps only the most likely tokens whose probabilities add up to $$p$$, and samples among them. With $$p = 0.95$$, the rare tail that holds the last 5% of the probability is never picked.

> [!PAPER] · §3 · page 12
> [![The pass@1 formula and the next lines, with correctness of the i-th response and cons@64 highlighted](/img/papers/deepseek-r1/p7-pass1.png)](/img/papers/deepseek-r1/p7-pass1.png)
>
> **What it says:** "$$\text{pass@1} = \frac{1}{k}\sum_{i=1}^{k} p_i$$, where $$p_i$$ denotes the correctness of the i-th response. [...] For AIME 2024, we also report consensus (majority vote) results (Wang et al., 2022) using 64 samples, denoted as cons@64."
>
> **What it means:**
>
> pass@1 asks: if you ask the model once, how likely is a right answer? The paper estimates this by asking $$k$$ times and counting. Each answer gets $$p_i = 1$$ if it is right and $$p_i = 0$$ if it is wrong. The mean of these is the share of right answers. The benchmark score is that share, averaged over all questions.
>
> cons@64 asks a different question: if the model may think 64 times and then vote, is the vote right? [Part 1](part-1-the-big-idea.md) defined majority voting. It needs only the most common answer to be right, not most of the answers.

$$
\text{pass@1} = \frac{1}{k}\sum_{i=1}^{k} p_i
$$

Read it from left to right. $$k$$ is the number of sampled answers to one question. $$p_i$$ is 1 or 0 for the $$i$$-th answer. The sum counts the right answers, and $$\frac{1}{k}$$ turns the count into a share. A tiny example: the model answers one question 8 times and gets $$p = (1, 0, 1, 1, 0, 1, 1, 0)$$. Five are right, so pass@1 is $$5/8 = 0.625$$.

Why is this better than one greedy answer? First, the two measure different things. Greedy accuracy scores one fixed rule: always take the most likely token. Sampled pass@1 scores the chance that one answer, sampled at temperature 0.6 and top-p 0.95, is right. The two numbers need not be equal.

Second, the sampled score can be measured precisely. For one question, one sample can only say 0 or 1; eight samples can say 0.625, and 64 samples can say it more precisely still. That is why the paper uses more samples on small test sets such as AIME, where each question moves the score a lot.

### pass@k, the general form

The paper reports only pass@1, but it cites pass@k from Chen et al. (2021). With $$k$$ tries, how likely is at least one right answer? That chance is pass@k. This is the natural score for code, where you can run the tests and keep any answer that passes.

> [!PAPER] Chen et al. (2021), Evaluating Large Language Models Trained on Code · §2.1 · page 3
> [![The pass@k definition in Chen et al. 2021, with count the number of correct, unbiased estimator and it is biased highlighted, and Equation 1](/img/papers/deepseek-r1/p7-chen.png)](/img/papers/deepseek-r1/p7-chen.png)
>
> **What it says:** "we generate $$n \ge k$$ samples per task [...] count the number of correct samples $$c \le n$$ which pass unit tests, and calculate the unbiased estimator" $$\text{pass@}k := \mathbb{E}_{\text{Problems}}\left[1 - \binom{n-c}{k} / \binom{n}{k}\right]$$. The shortcut $$1 - (1 - \hat{p})^k$$ "is biased".
>
> **What it means:**
>
> The direct way to measure pass@10 is to take 10 samples and check if any is right. That one check is again a coin flip, so the score jumps around. Chen et al. take more samples than they need, say $$n = 200$$, and ask a smarter question.
>
> Among the 200 samples, $$c$$ are right. Imagine you pick $$k$$ of the 200 at random. What is the chance that every one you picked is wrong? Count the ways to pick $$k$$ from the $$n - c$$ wrong ones. Divide by the ways to pick $$k$$ from all $$n$$.
>
> One minus that is the chance of at least one right answer. The formula averages this over every possible pick at once, so it uses all 200 samples and has far less noise.
>
> The tempting shortcut is to estimate the chance of a right answer as $$\hat{p} = c/n$$, then compute $$1 - (1 - \hat{p})^k$$. The paper warns that this is biased: on average it does not give the true pass@k. We check that with a simulation below.

> [!DEFINITION] pass@k
> The chance that at least one of $$k$$ answers to a question is right. The benchmark score averages it over all questions. pass@1 is the special case $$k = 1$$: the chance that a single answer is right. More tries never lower the score, so pass@k grows with $$k$$.

$$
\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}
$$

Read it for one question. $$n$$ is the number of samples we drew, $$c$$ how many of them are right, and $$k$$ the number of tries we allow. $$\binom{n}{k}$$, read "n choose k", is the number of ways to pick $$k$$ items from $$n$$. The fraction is the share of picks that hold only wrong answers. The score of a benchmark is the mean of this over all questions.

A tiny example you can check by hand: $$n = 5$$ samples, $$c = 2$$ right, $$k = 2$$ tries. Call the right samples R1 and R2 and the wrong ones W1, W2, W3. There are $$\binom{5}{2} = 10$$ pairs. The pairs with no right sample are W1-W2, W1-W3 and W2-W3: $$\binom{3}{2} = 3$$ of them. So pass@2 is $$1 - 3/10 = 0.7$$. Seven of the ten pairs hold a right answer.

With $$k = 1$$ the formula becomes $$1 - \frac{n-c}{n} = \frac{c}{n}$$, the share of right samples. That is exactly the paper's pass@1. So the paper's formula is the $$k = 1$$ case of Chen et al.'s.

The binomials get huge fast. $$\binom{200}{100}$$ has 59 digits. Chen et al. avoid them with a product. Write out both binomials and cancel the common factors:

$$
\frac{\binom{n-c}{k}}{\binom{n}{k}} = \prod_{i=n-c+1}^{n} \left(1 - \frac{k}{i}\right)
$$

The product has $$c$$ factors, one per right sample, and each is a number between 0 and 1. No big numbers appear. For the tiny example, $$n = 5$$ and $$c = 2$$ give the factors $$i = 4$$ and $$i = 5$$: $$(1 - 2/4)(1 - 2/5) = 0.5 \times 0.6 = 0.3$$. One minus 0.3 is 0.7 again.

### pass@k in code

**Method 1: our own estimator, the product form.** One loop over the $$c$$ factors.

```python
def pass_at_k_product(n, c, k):
    """Method 1, our own. n samples, c correct, budget k.
    P(all k picks wrong) = C(n-c, k) / C(n, k) = prod_{i=n-c+1}^{n} (1 - k/i)  (no big numbers)."""
    if n - c < k:                               # fewer than k wrong answers: every k-subset holds a correct one
        return 1.0
    all_wrong = 1.0
    for i in range(n - c + 1, n + 1):           # c factors, one per correct sample
        all_wrong *= 1 - k / i
    return 1 - all_wrong
```

The first test handles a corner case. If there are fewer than $$k$$ wrong samples, you cannot pick $$k$$ wrong ones, so the chance of at least one right answer is 1. Then the loop multiplies the $$c$$ factors $$(1 - k/i)$$, which is the chance that all $$k$$ picks are wrong.

**Method 2: the formula as written, with exact integers**, and the numpy one-liner that Chen et al. print in their Figure 3. A brute-force check lists every subset.

```python
def pass_at_k_comb(n, c, k):
    """Method 2a: the formula as written, with exact integers from math.comb."""
    return 1 - math.comb(n - c, k) / math.comb(n, k)   # math.comb(a, b) = 0 when b > a


def pass_at_k_numpy(n, c, k):
    """Method 2b: the numpy script of Chen et al., Figure 3."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def pass_at_k_brute(n, c, k):
    """Check: list every way to pick k of the n samples and count the picks that hold a correct one."""
    samples = [1] * c + [0] * (n - c)
    picks = list(itertools.combinations(range(n), k))
    good = sum(any(samples[j] for j in pick) for pick in picks)
    return good / len(picks), good, len(picks)
```

> [!PAPER] Chen et al. (2021), Evaluating Large Language Models Trained on Code · Figure 3 · page 3
> [![The numpy function pass_at_k from Figure 3 of Chen et al. 2021, with the line if n minus c is less than k return 1.0 and unbiased estimate of pass@k highlighted](/img/papers/deepseek-r1/p7-chen-code.png)](/img/papers/deepseek-r1/p7-chen-code.png)
>
> **What it says:** "A numerically stable script for calculating an unbiased estimate of pass@k."
>
> **What it means:**
>
> This is the same product as our Method 1, written in one line of numpy. `np.arange(n - c + 1, n + 1)` makes the $$c$$ values of $$i$$, `1.0 - k / ...` makes the factors, and `np.prod` multiplies them. "Numerically stable" means that no step makes a number so large or so small that the computer loses its digits.

We ran the three formulas on every $$n$$ up to 40, every $$c$$ and every $$k$$, and the brute force up to $$n = 10$$:

```text
== 2. the unbiased pass@k estimator (Chen et al. 2021)
worked example: n=5 samples, c=2 correct, k=2
  all 10 pairs listed, 7 hold a correct answer  ->  7/10 = 0.700
  formula 1 - C(3,2)/C(5,2) = 1 - 3/10 = 0.700
  product 1 - (1-2/4)(1-2/5) = 1 - 0.50*0.60 = 0.700
all n <= 40, every c and k: max |product - comb| or |product - numpy| = 2.2e-16 (brute force for n <= 10)
Chen et al. use n=200: C(200,100) has 59 digits; pass@100 with c=3: comb 0.876884, product 0.876884

n = 16 samples per question, c correct: pass@k
   c | pass@1   pass@2   pass@4   pass@8   pass@16 
   0 |    0.000    0.000    0.000    0.000    0.000
   1 |    0.062    0.125    0.250    0.500    1.000
   2 |    0.125    0.242    0.450    0.767    1.000
   4 |    0.250    0.450    0.728    0.962    1.000
   8 |    0.500    0.767    0.962    1.000    1.000
  12 |    0.750    0.950    0.999    1.000    1.000
  16 |    1.000    1.000    1.000    1.000    1.000

== 3. the shortcut 1 - (1 - c/n)^k is biased: simulation
true success rate per sample q = 0.1, n = 16, k = 8, 200,000 simulated questions
  true pass@8 = 1 - (1-q)^8 = 0.5695
  mean of unbiased estimator     = 0.5685
  mean of 1 - (1 - c/n)^k        = 0.4898

== 4. cons@k: majority vote over k sampled answers
  plurality right        pass@1 0.500  cons (tie=fraction) 1.00  cons (tie=first) 1   own == lib: True
  plurality wrong        pass@1 0.250  cons (tie=fraction) 0.00  cons (tie=first) 0   own == lib: True
  right but only 3 of 8  pass@1 0.375  cons (tie=fraction) 1.00  cons (tie=first) 1   own == lib: True
  tie 2-2                pass@1 0.500  cons (tie=fraction) 0.50  cons (tie=first) 1   own == lib: True
  tie 2-2, wrong first   pass@1 0.500  cons (tie=fraction) 0.50  cons (tie=first) 0   own == lib: True
  unfinished answers     pass@1 0.250  cons (tie=fraction) 1.00  cons (tie=first) 1   own == lib: True
```

The ten pairs of the tiny example give 0.700 by listing, by the formula and by the product. Over all the cases, the methods differ by at most $$2.2 \cdot 10^{-16}$$, the rounding error of a float. The table is the useful part. With 16 samples of which only 2 are right, pass@1 is 0.125, but pass@8 is already 0.767.

{{FIG:p7_passk|pass@k from our estimator, for 16 samples of which c are right. Even one right sample in 16 gives pass@8 = 0.5: a model that rarely finds the answer still finds it often when it may try many times.}}

The last two blocks of the output test two more claims. The simulation gives each of 200,000 questions a true success rate of 0.1 per sample, draws 16 samples, and estimates pass@8. The true pass@8 is $$1 - 0.9^8 = 0.5695$$. Our estimator averages 0.5685, close to the truth. The shortcut $$1 - (1 - c/n)^k$$ averages 0.4898, far below. This is the bias Chen et al. warn about.

The second test is majority voting. The paper does not say how it breaks a tie, so we wrote two rules. "Fraction" gives half a point when the right answer ties with one wrong answer. "First" lets the answer seen first win.

The third case is the reason cons@64 can beat pass@1 by a lot. The right answer "12" appears in only 3 of 8 samples, so pass@1 is 0.375. But the wrong answers are all different, so "12" still wins the vote and cons@8 is 1.

```python
def cons_at_k_own(answers, gold, ties='fraction'):
    """Method 1, our own. answers: k final answers (None = no answer). Count each answer, take the most frequent.
    ties='fraction': a tie between t answers gives 1/t credit if gold is among them; 'first': the earliest wins."""
    counts, first_seen = {}, {}
    for i, a in enumerate(answers):
        if a is None:
            continue                            # an answer that never finished does not vote
        counts[a] = counts.get(a, 0) + 1
        first_seen.setdefault(a, i)
    if not counts:
        return 0.0
    top = max(counts.values())
    winners = [a for a in counts if counts[a] == top]
    if ties == 'first':
        return float(min(winners, key=lambda a: first_seen[a]) == gold)
    return (gold in winners) / len(winners)
```

Method 2 does the same with `collections.Counter(votes).most_common(1)` for the "first" rule, and `numpy.unique(..., return_counts=True)` for the "fraction" rule. The two methods agreed on all six cases. This is how R1-Zero went from 71.0 pass@1 to 86.7 cons@64 on AIME ([Part 4](part-4-r1-zero-and-the-aha-moment.md)). Its wrong answers scatter, and its right answers agree.

<figure><img src="/img/papers/deepseek-r1/r1_part7_passk-run.png" alt="Terminal output of r1_part7_passk.py: pass@1 of eight answers, the worked pass@k example, the agreement check, the table for 16 samples, the bias simulation and six majority-vote cases" loading="lazy" /><figcaption>The output of r1_part7_passk.py, as it ran for this article.</figcaption></figure>

### Greedy against sampling, on the 1.5B model

The paper's claim about greedy decoding is easy to test on the small model. We took the first 8 problems of MATH-500 that have no drawing. For each one, the released DeepSeek-R1-Distill-Qwen-1.5B wrote one greedy answer and 4 sampled answers at the paper's settings. The prompt is the one the model card recommends. It is the problem, then "Please reason step by step, and put your final answer within \boxed{}". The run used a Mac (mps, bfloat16) and a limit of 2,048 new tokens per answer, not the paper's 32,768.

The only difference between the two modes is one line:

```python
    kw = dict(do_sample=True, temperature=TEMP, top_p=TOP_P) if sample else dict(do_sample=False, temperature=None, top_p=None)
```

math-verify grades the text after `</think>`. An answer that never closes its thinking has no final answer and counts as wrong. We measure repetition three ways. The repeated 4-gram rate is the share of 4-token windows that already appeared earlier in the same answer. A loop is a tail of 512 tokens that repeats with a fixed period. The last measure is the longest block of tokens that an answer writes twice.

```python
def rep_rate(ids, n=4):
    """Repeated n-gram rate: share of the n-token windows that already appeared earlier in the same answer."""
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def loop_period(ids, tail=512, need=0.95):
    """Smallest p <= 256 such that the last `tail` tokens repeat with period p (95% of positions agree)."""
    t = ids[-tail:]
    if len(t) < tail:
        return None
    for p in range(1, 257):
        same = sum(t[i] == t[i - p] for i in range(p, len(t))) / (len(t) - p)
        if same >= need:
            return p
    return None
```

The output, with the few-shot rows left for §5:

```text
model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B, device mps, bfloat16
8 MATH-500 problems (the first 8 without a drawing), max_new_tokens 2048
greedy: 1 answer each; sampling: 4 answers each at T=0.6, top-p=0.95

== greedy
  problem              lvl  gold            tokens  hit-limit  rep4   tail-rep4  loop  correct
  test/precalculus/807   2  \left( 3, \f...   850    0/1      0.13   0.11       0/1   1/1
  test/intermediate_al   5  p - q            2048    1/1      0.43   0.28       0/1   0/1
  test/algebra/2584.js   3  \frac{14}{3}     1444    0/1      0.32   0.35       0/1   1/1
  test/number_theory/5   3  9                1874    0/1      0.23   0.11       0/1   1/1
  test/prealgebra/1622   2  42                891    0/1      0.22   0.16       0/1   1/1
  test/number_theory/5   3  27               2048    1/1      0.34   0.18       0/1   0/1
  test/precalculus/927   4  90^\circ         2048    1/1      0.35   0.15       0/1   0/1
  test/algebra/2036.js   3  3\sqrt{13}       1059    0/1      0.15   0.14       0/1   1/1
  total: 8 answers, correct 5/8 = 0.625, hit the 2048-token limit 3/8, loop at the end 0/8, mean tokens 1533
  mean repeated 4-gram rate 0.272, in the last 512 tokens 0.183; correct among finished answers 5/5

== sample
  problem              lvl  gold            tokens  hit-limit  rep4   tail-rep4  loop  correct
  test/precalculus/807   2  \left( 3, \f...   869    0/4      0.15   0.12       0/4   4/4
  test/intermediate_al   5  p - q            2048    4/4      0.42   0.33       0/4   0/4
  test/algebra/2584.js   3  \frac{14}{3}     1635    1/4      0.33   0.28       0/4   3/4
  test/number_theory/5   3  9                1639    0/4      0.21   0.16       0/4   4/4
  test/prealgebra/1622   2  42                914    0/4      0.25   0.19       0/4   4/4
  test/number_theory/5   3  27               2048    4/4      0.30   0.17       0/4   0/4
  test/precalculus/927   4  90^\circ         2048    2/4      0.26   0.15       0/4   2/4
  test/algebra/2036.js   3  3\sqrt{13}       1841    0/4      0.17   0.14       0/4   4/4
  total: 32 answers, correct 21/32 = 0.656, hit the 2048-token limit 11/32, loop at the end 0/32, mean tokens 1608
  mean repeated 4-gram rate 0.261, in the last 512 tokens 0.192; correct among finished answers 21/21

== sampled answers: pass@1, pass@k (unbiased, n = 4) and cons@4 (fraction credit on ties)
  sample   pass@1 0.656  pass@2 0.729  pass@4 0.750  cons@4 0.750
  fewshot  pass@1 0.469  pass@2 0.583  pass@4 0.625  cons@4 0.625
  greedy   accuracy 0.625 (one answer per problem)

== longest block of tokens written twice in one answer
  greedy   mean   29.4 tokens, max   52, answers with a block of 50+ tokens repeated: 1/8
  sample   mean   23.9 tokens, max   48, answers with a block of 50+ tokens repeated: 0/32
  fewshot  mean   22.7 tokens, max   52, answers with a block of 50+ tokens repeated: 1/32
  greedy, problem test/precalculus/927.json: a block of 52 tokens at token 811 comes again at token 912:
  | - \(12 \times \frac{1}{6} = 2\)
  | - \(12 \times (-1) = -12\)
  | - \(12 \times (-\frac{1}{4}) = -3\)
```

{{FIG:p7_greedy|Our run on 8 MATH-500 problems with a 2,048-token limit. Greedy and zero-shot sampling score about the same, and every wrong answer was one cut by the limit. The few-shot prompt is tested in §5 below.}}

The honest result first: **we did not see the repetition the paper describes.** No greedy answer ended in a loop. The repeated 4-gram rate is 0.272 for greedy and 0.261 for sampling, almost the same. The longest block written twice is 52 tokens, and it is not a loop. The model recomputes a short list of products while it checks its work.

Our test is small. The paper's claim is about outputs of up to 32,768 tokens from a far larger model, and a loop needs room to form. A 2,048-token limit cuts most long answers before that could happen.

What we did see is the cut itself. Greedy hit the limit on 3 of 8 problems, sampling on 11 of 32 answers. Every answer that finished was right: 5 of 5 for greedy, 21 of 21 for sampling. So on these problems the limit, not the reasoning, decides the score. That is why the paper allows 32,768 tokens.

We also saw the variance argument at work. On the problem with gold answer $$90^\circ$$, greedy gives 0, but 2 of 4 samples are right, so pass@1 for that problem is 0.5. On the $$\frac{14}{3}$$ problem, greedy gives 1 and sampling gives 3 of 4. Greedy rounds each problem to 0 or 1; sampling keeps the fraction. Over 8 problems, greedy scores 0.625 and the sampled pass@1 is 0.656.

## DeepSeek-R1 evaluation {§3.1}

> [!PAPER] · Table 4 · page 13
> [![Table 4 of the paper: English, code, maths and Chinese benchmarks for Anthropic's Sonnet 3.5, GPT-4o, DeepSeek-V3, o1-mini, o1-1217 and DeepSeek-R1, with IF-Eval, SimpleQA, Aider-Polyglot and C-SimpleQA highlighted](/img/papers/deepseek-r1/p7-table4.png)](/img/papers/deepseek-r1/p7-table4.png)
>
> **What it says:** "Comparison between DeepSeek-R1 and other representative models." Bold marks the best score in each row.
>
> **What it means:**
>
> Read the table one column pair at a time. R1 against V3 is the cleanest comparison, because both come from the same base model, DeepSeek-V3-Base, and both have 37B active of 671B total parameters. Any gap between them comes from post-training: R1's reasoning RL against V3's ordinary chat training.
>
> R1 against o1-1217 is the headline comparison, but it is less clean. Many o1-1217 cells are "-", and its numbers come from OpenAI's reports, not from the authors' runs. On the rows that both have, R1 is within about four points of o1-1217, except on SimpleQA and Aider.
>
> The four highlighted rows are where R1 loses to a model it should beat. They teach the most, so we read them below with the paragraphs that explain them.

{{FIG:p7_table4|Table 4 of the paper as a dot chart, by group. Each row joins the lowest and highest of the three scores. R1 (blue) is far right of V3 (green) on maths and competition code, and close to o1-1217 (orange) where both are reported. Bold names are the rows where R1 is behind.}}

The dot chart shows the pattern at a glance. The longest lines are in maths and code. AIME goes from 39.2 for V3 to 79.8 for R1, and CNMO from 43.2 to 78.8. The Codeforces percentile goes from 58.7 to 96.3.

These are the tasks that reasoning RL trained on, with rewards a program can check. The English knowledge rows move less: MMLU from 88.5 to 90.8. The model already knew those facts; reasoning helps it use them a little better.

> [!PAPER] · §3.1 · page 13
> [![The paragraph after Table 4, with enhanced accuracy in STEM-related questions, excels on FRAMES, its tendency to refuse answering certain queries after safety RL and accuracy of over 70% highlighted](/img/papers/deepseek-r1/p7-knowledge.png)](/img/papers/deepseek-r1/p7-knowledge.png)
>
> **What it says:** the gains on MMLU, MMLU-Pro and GPQA Diamond come from "enhanced accuracy in STEM-related questions". On Chinese SimpleQA, R1 is worse than V3 "primarily due to its tendency to refuse answering certain queries after safety RL. Without safety RL, DeepSeek-R1 could achieve an accuracy of over 70%."
>
> **What it means:**
>
> Knowledge tests mix subjects. A history question needs a fact; a physics question needs a fact and a few steps of computation. The authors say the gain comes from the second kind. That fits: RL trained on maths and science, so it improved the steps, not the facts.
>
> SimpleQA tests facts alone, and here R1 (30.1) beats V3 (24.9) but stays far below o1-1217 (47.0). Reasoning cannot invent a fact the model never stored. The paper does not explain the gap to o1-1217.
>
> C-SimpleQA is the odd one: R1 gets 63.7 and V3 gets 68.0. The reason is not knowledge but refusals. The last RL stage ([Part 5](part-5-the-r1-pipeline.md)) trained R1 to be harmless, and it learned to refuse some questions. A refusal scores zero on a fact test. The authors say that without safety RL the score would pass 70. This is a common trade: safety training costs some points on benchmarks that never needed safety.

> [!PAPER] · §3.1 · page 13
> [![The IF-Eval and open-ended paragraph, with instruction-following data during the final stages, generalization benefits of large-scale and average of 689 tokens on ArenaHard and 2,218 characters on AlpacaEval 2.0 highlighted](/img/papers/deepseek-r1/p7-ifeval.png)](/img/papers/deepseek-r1/p7-ifeval.png)
>
> **What it says:** "DeepSeek-R1 also delivers impressive results on IF-Eval [...] the summary lengths generated by DeepSeek-R1 are concise, with an average of 689 tokens on ArenaHard and 2,218 characters on AlpacaEval 2.0. This indicates that DeepSeek-R1 avoids introducing length bias during GPT-based evaluations".
>
> **What it means:**
>
> Read the text next to the table. The text calls IF-Eval "impressive", but Table 4 gives R1 83.3, below V3 (86.1) and Sonnet 3.5 (86.5). So R1 is good at following format rules, but it is not the best in the table. The text credits the instruction-following data added in the last SFT and RL stages. The Nature version later showed how much those stages mattered: R1-Zero scored only 46.6 on IF-Eval.
>
> The open-ended rows are R1's biggest surprise. AlpacaEval 2.0 jumps from 70.0 for V3 to 87.6, and ArenaHard from 85.5 to 92.3. These are writing tasks, not maths. The authors read it as a sign that large-scale RL helps beyond the tasks it trained on.
>
> Then they answer the obvious doubt: did R1 win by writing longer? The judge saw only the summary, and those summaries are short: 689 tokens on average on ArenaHard, 2,218 characters on AlpacaEval 2.0. The paper does not give the lengths of the other models, so this is evidence against length bias, not proof.

> [!PAPER] · §3.1 · page 14
> [![The maths and code paragraph, with outperforms DeepSeek-R1 on Aider but achieves comparable performance on SWE Verified and related RL training data currently remains very limited highlighted](/img/papers/deepseek-r1/p7-math-code.png)](/img/papers/deepseek-r1/p7-math-code.png)
>
> **What it says:** "On engineering-oriented coding tasks, OpenAI-o1-1217 outperforms DeepSeek-R1 on Aider but achieves comparable performance on SWE Verified. We believe the engineering performance of DeepSeek-R1 will improve in the next version, as the amount of related RL training data currently remains very limited."
>
> **What it means:**
>
> The code rows split in two. Contest code (LiveCodeBench, Codeforces) is short programs for puzzle-like problems with hidden tests. That is close to maths, and R1 is at o1's level: 65.9 against 63.4 on LiveCodeBench. Engineering code (SWE Verified, Aider) is editing real projects. There R1 gets 49.2 on SWE Verified, only 7.2 points above V3, and 53.3 on Aider against o1-1217's 61.7.
>
> The reason is the cost of a reward. A contest answer is checked in seconds. A repository patch needs the project's whole test suite, which can take minutes. RL needs millions of such checks, so the authors trained very little on this kind of task. §5 lists it as a limit.

The Codeforces rows show a second effect: a rating and a percentile measure the same skill on different scales. R1's rating is 2,029 and o1-1217's is 2,061, a gap of 32 points. Their percentiles are 96.3 and 96.6, a gap of only 0.3. V3's rating of 1,134 gives 58.7. Few human players have high ratings, so near the top a large step in rating moves the percentile only a little. Compare ratings when two strong models are close; the percentile hides their gap.

## Unsuccessful attempts {§4.2}

> [!PAPER] · §4.2 · page 15
> [![The opening of Section 4.2, with this does not imply that these approaches are incapable highlighted](/img/papers/deepseek-r1/p7-unsuccessful.png)](/img/papers/deepseek-r1/p7-unsuccessful.png)
>
> **What it says:** "We share our failure experiences here to provide insights, but this does not imply that these approaches are incapable of developing effective reasoning models."
>
> **What it means:**
>
> Few papers report what did not work, and this section is short. It matters because in 2024 the two ideas below were the favourite guesses for how o1 worked. Many labs were building them. The authors say: we tried both, and at our scale they did not pay off.
>
> Note the careful wording. "Did not work for us, in our setup" is a weaker claim than "cannot work". Both ideas are still used for other jobs, as the paragraphs themselves say.

### Process reward models

> [!DEFINITION] Process reward model (PRM)
> A neural network that scores each step of a solution, not only the final answer. It is trained on solutions where people (or models) marked each step right or wrong. An **outcome reward** scores only the end result. Lightman et al. (2023) trained a PRM on 800,000 step labels and found it picked good solutions better than an outcome model.

A PRM is an attractive idea. With an outcome reward, a long solution that goes wrong in step 9 of 10 gets the same zero as nonsense. A teacher who marks each step can say "the first eight steps were good". That is a richer signal, and it should make learning faster.

> [!PAPER] · §4.2 · page 15
> [![The PRM paragraph, with define a fine-grain step in general reasoning, determining whether the current intermediate step is correct, inevitably leads to reward and rerank the top-N responses highlighted](/img/papers/deepseek-r1/p7-prm.png)](/img/papers/deepseek-r1/p7-prm.png)
>
> **What it says:** PRM has "three main limitations". "First, it is challenging to explicitly define a fine-grain step in general reasoning. Second, determining whether the current intermediate step is correct is a challenging task. [...] Third, once a model-based PRM is introduced, it inevitably leads to reward hacking".
>
> **What it means:**
>
> **What is a step?** In a school proof, a step is a line. In R1's thinking, the text runs on: "so x is 3, wait, that ignores the negative root, let me redo the square". Is that one step or three? A PRM needs the text cut into steps before it can score them, and no rule cuts free reasoning cleanly.
>
> **Is the step right?** Some steps are wrong but useful. "Assume x is positive" may be a guess the model later drops. A step can also be right but useless. Labelling needs a judge who understands the whole solution. Labels from a model are not reliable enough, and labels from people do not scale to millions of RL samples.
>
> **Reward hacking.** A PRM is a neural network, and RL is very good at finding the inputs where a network gives a high score by mistake. [Part 3](part-3-rewards-and-the-template.md) explained why the authors avoided neural rewards for R1-Zero for this reason. The rule "is the boxed answer equal to the true answer" cannot be fooled in this way. The authors still grant PRMs two uses: reranking the top N answers a model wrote, and assisting guided search, as in Snell et al. (2024). Their negative verdict is narrower: in their large-scale RL experiments, the gain was small for the extra compute.

### Monte Carlo Tree Search

The second failed idea comes from game-playing programs. To see why it was tempting and why it failed, we need the background first.

**Where it comes from.** In October 2015, DeepMind's AlphaGo beat the European Go champion by 5 games to 0, the first win of a program over a human professional. The paper appeared in 2016.

Go was the hard case of board games, because the number of possible games is far too large to search. AlphaGo combined two neural networks with a search method called Monte Carlo Tree Search (MCTS). A **policy network** suggests promising moves. A **value network** looks at a board and estimates who will win.

AlphaZero (2017) then learned chess, shogi and Go from self-play alone, with no human games.

**How MCTS works, in plain words.** Picture a tree. The root is the current board, each branch is a move, and each node is the board after that move. The program cannot grow the whole tree, so it grows it where it looks promising. Each round has four steps, shown below. After many rounds, the move from the root with the most visits is played.

{{FIG:p7_mcts|One round of Monte Carlo Tree Search. Select walks down the tree along promising children; expand adds one new position; the value model scores it; back-up adds the score to every node on the path, so good branches get more visits next round.}}

> [!DEFINITION] Value model
> A network that looks at a partial state, such as a board or the first half of a solution. It predicts the final outcome: the chance to win, or the chance that the finished solution will be right. In MCTS it replaces playing every game to the end.

The appeal for language models is clear. Treat the reasoning as a game. A node is a partial solution, a move is the next piece of text, and a win is a right final answer. Search the tree at answer time, and use the best paths found by search as training data. The model then gets better, its search gets better, and the loop repeats, as in AlphaZero.

> [!PAPER] · §4.2 · page 15
> [![The first MCTS paragraph, with breaking answers into smaller parts, multiple tags, guided by a pre-trained value and unlike chess, where the search space is relatively well-defined highlighted](/img/papers/deepseek-r1/p7-mcts-a.png)](/img/papers/deepseek-r1/p7-mcts-a.png)
>
> **What it says:** "This approach involves breaking answers into smaller parts to allow the model to explore the solution space systematically. [...] we first use collected prompts to find answers via MCTS guided by a pre-trained value model. Subsequently, we use the resulting question-answer pairs to train both the actor model and the value model". "First, unlike chess, where the search space is relatively well-defined, token generation presents an exponentially larger search space."
>
> **What it means:**
>
> The authors built exactly the AlphaZero loop. The model marks its reasoning steps with tags, so that each step is a move. MCTS, guided by a value model, searches for good answers. The answers then train both the model that writes (the "actor") and the value model, and the loop repeats.
>
> The first problem is the size of the tree. On a Go board the moves are fixed by the rules. In text, the next piece can be almost anything. We count how big that gets below.

> [!PAPER] Silver et al. (2016), AlphaGo · page 484
> [![A passage from the AlphaGo paper, with chess (b about 35, d about 80), Go (b about 250, d about 150) and exhaustive search is highlighted](/img/papers/deepseek-r1/p7-alphago.png)](/img/papers/deepseek-r1/p7-alphago.png)
>
> **What it says:** a game tree holds about $$b^d$$ sequences of moves, "where b is the game's breadth (number of legal moves per position) and d is its depth (game length). In large games, such as chess (b≈35, d≈80) and especially Go (b≈250, d≈150), exhaustive search is infeasible".
>
> **What it means:**
>
> $$b^d$$ is the number of different games: $$b$$ choices at each of $$d$$ moves. For Go, $$250^{150}$$ is a number with 360 digits. Nobody can search that. MCTS works anyway. The policy network keeps only a few good moves at each node. The value network stops the search long before the end of the game.
>
> Now replace the board with text. A model with a vocabulary of 151,665 tokens has 151,665 legal "moves" at every step. A reasoning answer is not 150 moves long but thousands of tokens. Even if a good policy keeps only a few tokens per step, the depth alone makes the tree far larger than Go's.

How large is the tree for a real model? A full vocabulary is the wrong count, because no policy considers every token. A better count asks the model itself. At each step of an answer, how many tokens does top-p sampling at the paper's settings actually allow? That gives the branching factor at each step of the one path the model actually took.

We ran one forward pass over each of the 32 zero-shot sampled answers. At every step we sort the next-token probabilities and count how many tokens it takes to reach 0.95:

```python
                    for T, store in ((TEMP, s_a), (1.0, s1_a)):
                        pr = torch.softmax(logits / T, -1)
                        srt = pr.sort(-1, descending=True).values.cumsum(-1)
                        store += ((srt < TOP_P).sum(-1) + 1).clamp(max=pr.shape[-1]).tolist()   # tokens to reach 0.95
```

Then we multiply the nucleus sizes along one answer, two ways. Method 1 adds their base-10 logarithms. Method 2 multiplies them as exact Python integers and counts the digits:

```python
m1 = sum(math.log10(s) for s in one['sizes'])
prod = 1
for s in one['sizes']:
    prod *= s
digits = str(prod)
```

```text
== branching factor of a search over tokens (51,452 steps of 32 sampled answers)
  vocabulary: 151,665 tokens in the tokenizer (151,936 rows in the output layer)
  top-p 0.95 nucleus at T=0.6: median 1, mean 1.59, 90th pct 3, 99th pct 7, max 22, size 1 at 71.9% of steps
  top-p 0.95 nucleus at T=1.0: median 1, mean 2.64, 90th pct 6, 99th pct 21, max 78, size 1 at 60.9% of steps
  mean entropy at T=0.6: 0.233 nats

== how many different answers could top-p sampling write? one answer (test/precalculus/927.json), 1997 tokens
  Method 1 (sum of log10 of nucleus sizes):  10^182.1
  Method 2 (exact product, count the digits): 183 digits = 10^182.1
  Go, b = 250, d = 150:   250^150 = 10^359.7  (exact: 360 digits)
  chess, b = 35, d = 80:  35^80 = 10^123.5  (exact: 124 digits)
  whole vocabulary, 1997 tokens: 151,665^1997 = 10^10346
  a tree of depth 150 with our median nucleus at T=0.6: 1^150 = 10^0.0
```

{{FIG:p7_nucleus|Nucleus sizes over 51,452 steps of our sampled answers. At the paper's temperature of 0.6, 71.9% of steps allow only one token. The rest allow a few, and the largest nucleus we saw had 22 tokens.}}

The numbers hold a surprise. At $$T = 0.6$$ the median nucleus has one token, and 71.9% of the steps allow no choice at all. The model is sure of most tokens: the end of a word, the next symbol of an equation. Choices appear at a few places, like which method to try or whether to write "Wait". So a model-guided tree is narrow, not wide.

It is also deep. Take one finished answer of 1,997 tokens and multiply its nucleus sizes. The product is $$10^{182.1}$$, and both methods agree to the digit count of 183. Read this number with care.

It is the size of a hypothetical tree in which every branch copies the branching schedule of this one answer. Step 1 is as wide as our step 1, step 2 as wide as our step 2, and so on. It is not a count of the real tree. A different early token changes every later distribution. So the other branches have their own nucleus sizes and their own lengths. The script's heading ("how many different answers could top-p sampling write?") is only a rough guide to that question.

As an order of magnitude, the number still makes the point. A tree with this schedule is smaller than Go's $$10^{359.7}$$, but no computer can search $$10^{182}$$ paths either. The unrestricted space, all 151,665 tokens at all 1,997 steps, has $$10^{10346}$$ paths. That is the "exponentially larger search space" of the paper.

{{FIG:p7_tree|The size of four search trees, as the number of digits. Chess and Go come from the AlphaGo paper. The two blue bars are for one of our 1,997-token answers: a hypothetical tree that repeats this answer's nucleus size at every step, and the paths through the whole vocabulary.}}

The count also shows the deeper trouble. In Go, every move is a real decision. In text, the decisions hide among long stretches of forced tokens, and nothing marks where one step ends. That is why the authors made the model write tags at its reasoning steps, and why they capped the children per node. The search still needs a value model that can judge half a solution, and that is the second problem below.

> [!PAPER] · §4.2 · page 16
> [![The second MCTS paragraph, with maximum extension limit for each, stuck in local optima, Training a fine-grained value model is inherently difficult and iteratively boosting model performance through self-search remains a highlighted](/img/papers/deepseek-r1/p7-mcts-b.png)](/img/papers/deepseek-r1/p7-mcts-b.png)
>
> **What it says:** "we set a maximum extension limit for each node, but this can lead to the model getting stuck in local optima. Second, the value model directly influences the quality of generation [...] Training a fine-grained value model is inherently difficult". "while MCTS can improve performance during inference when paired with a pre-trained value model, iteratively boosting model performance through self-search remains a significant challenge."
>
> **What it means:**
>
> To keep the tree small, the authors let each node grow only a few children. That is a cap on exploration. If the right idea is not among the first few continuations of a step, the search never sees it. It then keeps improving inside a poor region of the tree: a local optimum.
>
> The second problem is deeper. In Go, the value network learns from millions of finished games, and every game ends in a clear win or loss. A value model for text must look at half a solution and guess whether it will end right. Half a proof often looks fine until the last line. A weak value model sends the search the wrong way, the search produces poor training data, and the loop cannot lift itself.
>
> Compare this with what worked. GRPO ([Part 2](part-2-grpo.md)) needs no value model and no tree. It samples whole answers, checks only the final result, and compares answers within a group. The authors' verdict is that MCTS helps at answer time with a good value model, but not as a training loop.

## Conclusion, limitations, and future work {§5}

> [!PAPER] · §5 · page 16
> [![The first two paragraphs of Section 5, with comparable to OpenAI-o1-1217 on a range of tasks, 800K training samples and 28.9% on AIME and 83.9% on MATH highlighted](/img/papers/deepseek-r1/p7-conclusion.png)](/img/papers/deepseek-r1/p7-conclusion.png)
>
> **What it says:** "DeepSeek-R1 achieves performance comparable to OpenAI-o1-1217 on a range of tasks." The distilled 1.5B model "outperforms GPT-4o and [Anthropic's Sonnet 3.5] on math benchmarks with 28.9% on AIME and 83.9% on MATH."
>
> **What it means:**
>
> The summary keeps the claim modest: "comparable on a range of tasks", not "better". Table 4 supports that. R1 leads on maths and some code, and trails on facts, Aider and some knowledge tests.
>
> The second paragraph repeats the result of [Part 6](part-6-distillation.md): 800,000 samples from R1 turn small open models into strong reasoners. The 1.5B model we run in this series is one of them. Its 83.9 on MATH-500 is for the full test set with long outputs; our 8-problem run with 2,048 tokens is a much rougher check.

> [!PAPER] · §5 · page 16
> [![The first two limits of Section 5, with function calling, multi-turn, complex role-playing, and JSON output, optimized for Chinese and English and might use English for reasoning and responses highlighted](/img/papers/deepseek-r1/p7-limits-a.png)](/img/papers/deepseek-r1/p7-limits-a.png)
>
> **What it says:** "the capabilities of DeepSeek-R1 fall short of DeepSeek-V3 in tasks such as function calling, multi-turn, complex role-playing, and JSON output." And: "DeepSeek-R1 is currently optimized for Chinese and English [...] DeepSeek-R1 might use English for reasoning and responses, even if the query is in a language other than English or Chinese."
>
> **What it means:**
>
> **General capability.** Function calling and JSON output need a strict format that a program will read. A long, free chain of thought does not help there, and it can hurt: the model may think its way out of the required shape. Multi-turn chats and role-play need the model to keep a persona and a history. R1's training data was mostly single questions with checkable answers, so these skills got less practice than in V3.
>
> **Language mixing.** [Part 5](part-5-the-r1-pipeline.md) added a reward for writing the thinking in the language of the question: the share of target-language words. It reduced the mixing of Chinese and English. It does not make the model fluent in other languages. Ask in French or Hindi, and R1 may reason, and even answer, in English. The Nature version adds a likely cause: the base model, DeepSeek-V3-Base, saw mainly Chinese and English.

> [!PAPER] · §5 · page 16
> [![The last two limits of Section 5, with Few-shot prompting consistently degrades its performance, zero-shot setting, long evaluation times, rejection sampling on software engineering data and asynchronous evaluations highlighted](/img/papers/deepseek-r1/p7-limits-b.png)](/img/papers/deepseek-r1/p7-limits-b.png)
>
> **What it says:** "we observe that it is sensitive to prompts. Few-shot prompting consistently degrades its performance. Therefore, we recommend users directly describe the problem and specify the output format using a zero-shot setting". For software engineering, "Due to the long evaluation times, which impact the efficiency of the RL process, large-scale RL has not been applied extensively".
>
> **What it means:**
>
> **Prompt sensitivity.** This is the claim of the evaluation-prompts paragraph, now stated as advice for users. Do not show R1 examples. State the problem and the format you want, and let it reason. We tested this on the 1.5B model; the result is below.
>
> **Software engineering.** This is the reason behind the weak Aider and SWE rows. RL learns from many tries. If each try needs a long test run, the GPUs that generate answers sit idle while the tests run. The authors name two fixes.
>
> Rejection sampling keeps only good patches as fine-tuning data, so it needs no tests during RL. Asynchronous evaluation lets generation continue while earlier answers are still being tested.

We tested the advice on the same 8 problems. The few-shot prompt puts two short solved examples before the problem: the sum of the first 10 integers, and a linear equation. Everything else stays the same: 4 samples per problem at $$T = 0.6$$, top-p 0.95, 2,048 new tokens.

```text
== fewshot
  problem              lvl  gold            tokens  hit-limit  rep4   tail-rep4  loop  correct
  test/precalculus/807   2  \left( 3, \f...   884    0/4      0.13   0.11       0/4   4/4
  test/intermediate_al   5  p - q            2048    4/4      0.37   0.34       0/4   0/4
  test/algebra/2584.js   3  \frac{14}{3}     2048    2/4      0.35   0.25       0/4   2/4
  test/number_theory/5   3  9                1376    2/4      0.24   0.19       0/4   2/4
  test/prealgebra/1622   2  42                946    1/4      0.25   0.20       0/4   3/4
  test/number_theory/5   3  27               2048    4/4      0.29   0.16       0/4   0/4
  test/precalculus/927   4  90^\circ         2048    4/4      0.27   0.15       0/4   0/4
  test/algebra/2036.js   3  3\sqrt{13}       1227    0/4      0.17   0.11       0/4   4/4
  total: 32 answers, correct 15/32 = 0.469, hit the 2048-token limit 17/32, loop at the end 0/32, mean tokens 1649
  mean repeated 4-gram rate 0.259, in the last 512 tokens 0.188; correct among finished answers 15/15

== sampled answers: pass@1, pass@k (unbiased, n = 4) and cons@4 (fraction credit on ties)
  sample   pass@1 0.656  pass@2 0.729  pass@4 0.750  cons@4 0.750
  fewshot  pass@1 0.469  pass@2 0.583  pass@4 0.625  cons@4 0.625
  greedy   accuracy 0.625 (one answer per problem)
```

The few-shot prompt lowered pass@1 from 0.656 to 0.469 and cons@4 from 0.750 to 0.625. The direction matches the paper's "consistently degrades". The size of the drop means little: it is 6 answers out of 32 on 8 problems.

Look at where the drop comes from. Every finished few-shot answer was right too, 15 of 15. The few-shot answers simply hit the limit more often, 17 of 32 against 11 of 32. Our run cannot tell why. It shows only that the examples did not make the answers shorter or better, and on these problems they made them worse.

<figure><img src="/img/papers/deepseek-r1/r1_part7_gen-run.png" alt="Terminal output of r1_part7_gen.py: greedy, zero-shot and few-shot runs on 8 MATH-500 problems, pass@k and cons@4, the longest repeated blocks, nucleus sizes and the size of the search tree" loading="lazy" /><figcaption>The output of r1_part7_gen.py, as it ran for this article (the first 60 lines).</figcaption></figure>

## What came after

The paper was later revised and published in Nature, received on 14 February 2025 and published online on 17 September 2025. That version keeps the method and the main tables, and adds detail. Its Table 2 scores every stage of the pipeline.

For example, IF-Eval rose from 46.6 for R1-Zero to 83.3 for R1, and AlpacaEval 2.0 from 24.7 to 87.6. The last stages bought the general skills. Its supplement gives the cost: 147,000 H800 GPU hours in all. At an assumed rental price of 2 US dollars per GPU hour, that is 294,000 US dollars.

> [!PAPER] Nature version (2025), DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning · page 636
> [![The Nature version's limits on structure output and tool use and on token efficiency, with cannot make use of tools, Token efficiency and overthinking highlighted](/img/papers/deepseek-r1/p7-nature-limits.png)](/img/papers/deepseek-r1/p7-nature-limits.png)
>
> **What it says:** "DeepSeek-R1 cannot make use of tools, such as search engines and calculators". And under "Token efficiency": there are still "instances of excessive reasoning", which the authors call "overthinking", "in response to simpler questions."
>
> **What it means:**
>
> The list of limits grew. Two new ones are about how R1 thinks, not what it knows. It cannot call a calculator or a search engine while it thinks. So every fact and every sum must come from its own weights. It also sometimes thinks for thousands of tokens about a question that needs one line.
>
> The Nature version also names reward hacking as a limit of the whole method. For tasks such as writing, no rule can check the answer, so pure RL is hard to scale there.

The rest of the story is in [DeepSeek-R1 and reasoning by reinforcement learning](/writings/deepseek-4-r1-and-reasoning-by-rl/), part 4 of our DeepSeek series. It follows the line from R1 to R1-0528, V3.1 with its thinking and non-thinking modes, V3.2, which trained agents that call tools, and V4. Several limits of §5 became the goals of these later models.

> [!TAKEAWAYS]
> - **Scores depend on the setup.** R1 is scored zero-shot, with up to 32,768 tokens, by sampling 4 to 64 answers at temperature 0.6 and top-p 0.95. An LLM judge sees only the final summary.
> - **pass@1 is the share of right answers.** It is the $$k = 1$$ case of the unbiased pass@k of Chen et al., $$1 - \binom{n-c}{k} / \binom{n}{k}$$. Our product form, math.comb and numpy agree to $$2.2 \cdot 10^{-16}$$.
> - **The shortcut is biased.** With a true rate of 0.1 per sample, the true pass@8 is 0.5695. The unbiased estimator averages 0.5685; $$1 - (1 - c/n)^k$$ averages 0.4898.
> - **Majority voting rewards agreement.** A right answer in 3 of 8 samples still wins the vote when the wrong answers scatter. That is how cons@64 beats pass@1.
> - **R1 wins where the reward is a program.** AIME rises from 39.2 (V3) to 79.8, CNMO from 43.2 to 78.8, Codeforces from the 58.7th to the 96.3rd percentile.
> - **R1 loses in four places.** IF-Eval (83.3 against V3's 86.1), SimpleQA (30.1 against o1-1217's 47.0), C-SimpleQA (63.7 against 68.0, from safety refusals) and Aider (53.3 against 61.7).
> - **Two ideas failed at scale.** A process reward model needs clear steps, reliable step labels and a judge that RL cannot fool. MCTS over tokens needs a value model that can judge half a solution.
> - **A model-guided token tree is narrow but deep.** At $$T = 0.6$$, 71.9% of steps allow one token, yet a hypothetical tree with the branching schedule of one 1,997-token answer still has $$10^{182.1}$$ paths.
> - **Our small run did not show greedy loops.** On 8 problems with 2,048 tokens, greedy scored 0.625 and sampling 0.656. Every wrong answer was cut by the token limit.
> - **Few-shot hurt here too.** pass@1 fell from 0.656 to 0.469 with two worked examples, mostly because more answers hit the limit. The paper advises a zero-shot prompt.

## The paper and the code

[Paper (arXiv 2501.12948)](https://arxiv.org/abs/2501.12948) · [official repository](https://github.com/deepseek-ai/DeepSeek-R1) · [our scripts for this series](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/deepseek-r1)
