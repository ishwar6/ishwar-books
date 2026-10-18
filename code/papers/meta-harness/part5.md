---
title: "Limits, and What Came After"
description: "Where Meta-Harness can fool itself, what a search costs, and why it rests on one strong proposer. Then five later papers that changed the loop, each read from its own PDF, and a small run on a laptop that shows how a search score can look better than it is."
part: 5
covers: "§5, §4.3, footnote 1, Figure 7, Appendix A.2, Appendix D"
date: 2026-10-18
tags: [agents, harness-engineering, overfitting, evaluation, self-improvement]
---

A search that picks the best of many tries always looks a little better than it is. This part asks how much that matters for Meta-Harness, what a run costs, and what the papers that came after it changed. Each piece below shows the exact lines of a paper, then explains them, and then checks the idea with code you can run.

## What the authors claim in the Discussion {§5}

> [!PAPER] · §5
> [![The first lines of Section 5, with generalize to out-of-distribution classification datasets, unseen base models in the math setting, a few hours and readable, transferable strategies highlighted](/img/papers/meta-harness/p5-discussion.png)](/img/papers/meta-harness/p5-discussion.png)
>
> **What it says:** the discovered harnesses "generalize to out-of-distribution classification datasets" (Table 5) and "to unseen base models in the math setting" (Table 6). A search run "completes in a few hours of wall-clock time" and produces "readable, transferable strategies".
>
> **What it means:**
>
> The authors make three claims here. The harnesses work on data the search never saw. They work with models the search never used. And a person can read them, because they are short Python programs, not a list of numbers.
>
> Each claim has a limit that the rest of this page looks at. "Generalize" was tested on text classification and math, but not on TerminalBench-2, where the search and the final score used the same tasks. "A few hours" is wall-clock time, not money, and the paper gives no price. "Readable" helps a human find overfitting, but someone must still do the reading.

> [!DEFINITION] Overfitting
> A result that is good on the data used to choose it, and worse on new data. A student who learns the answers to last year's exam by heart can score 100 on that exam and fail a new one. In harness search, the "exam" is the set of tasks that scores each candidate.

## Overfitting in code space {§5, Figure 7}

> [!PAPER] · §5
> [![Lines of Section 5, with overfitting in code space is also more inspectable, brittle if-chains or hard-coded class mappings, and weight-space overfitting is not highlighted](/img/papers/meta-harness/p5-overfit-code.png)](/img/papers/meta-harness/p5-overfit-code.png)
>
> **What it says:** "Overfitting in code space is also more inspectable: brittle if-chains or hard-coded class mappings are visible on inspection in a way that weight-space overfitting is not."
>
> **What it means:**
>
> When a neural network overfits, the damage hides in millions of weights. Nobody can point at one weight and say "this one memorised question 17". A harness is different. If the proposer writes `if "chess" in task: install_stockfish()`, a person can see that line. Then they can ask: is it a general rule, or a trick for one known task?
>
> Note the word "inspectable". It means overfitting can be found, not that it cannot happen. The search still picks the candidate with the best score on a fixed set of tasks. Any choice of "best of many" on a fixed set can reward luck. The next sections show how large that effect is.

> [!DEFINITION] Search set and test set
> The **search set** is the set of tasks that scores each candidate harness during the search. The proposer sees these scores and traces. The **test set** is a separate set of tasks that is used once, at the end, to report the result. [Part 2](part-2-the-method.md) shows where each one enters the loop.

> [!PAPER] · Figure 7
> [![Figure 7 of the paper: three scatter plots for LawBench, Symptom2Disease and USPTO. Each pink dot is a discovered harness, placed by its search-set accuracy on the horizontal axis and its test accuracy on the vertical axis. Baselines such as zero-shot, few-shot and ACE are labelled. A dashed diagonal shows y equal to x](/img/papers/meta-harness/p5-figure7.png)](/img/papers/meta-harness/p5-figure7.png)
>
> **What it says:** "Search-set vs. test accuracy per dataset for discovered text-classification strategies. Each pink dot is a discovered strategy; baselines are labeled. The dashed diagonal is $$y=x$$."
>
> **What it means:**
>
> Each dot is one harness. Its place left to right is its score on the search set. Its place bottom to top is its score on the test set. A dot on the dashed line scored the same on both. A dot below the line did worse on the test set than on the search set, which is what overfitting looks like.
>
> Read the three panels by eye, because the paper gives no numbers for this figure. On LawBench the dots climb with the line: a better search score mostly means a better test score. On Symptom2Disease most dots sit above the line, in a flat band near 85 to 90 percent test accuracy. On USPTO the dots with the best search scores, near 30 percent, have test scores of only about 16 to 19 percent.
>
> Two cautions help here. First, Appendix D says the search set should be hard for the baseline, so search scores can sit lower than test scores for every harness. What matters is the order: does a higher search score predict a higher test score? Second, the selected harness has 48.6 percent average test accuracy (Table 2), and the best search-set accuracy is 56.7 (Table 4). The paper does not say these two numbers belong to one harness, so their difference is not a measured overfitting gap.

> [!PAPER] · Appendix D
> [![A bullet of Appendix D, with filtering for examples that the baseline gets wrong, the search has little to optimize if the baseline already saturates the evaluation, and roughly 50 full evaluations per run highlighted](/img/papers/meta-harness/p5-appd-set-a.png)](/img/papers/meta-harness/p5-appd-set-a.png)
> [![The continuation on the next page, with 88 problems for math retrieval and a fast, discriminative eval is more valuable than a large one highlighted](/img/papers/meta-harness/p5-appd-set-b.png)](/img/papers/meta-harness/p5-appd-set-b.png)
>
> **What it says:** build the search set by "filtering for examples that the baseline gets wrong" or by picking difficult ones. Keep it small, "for roughly 50 full evaluations per run (50–100 examples in our classification experiments, 88 problems for math retrieval)". "A fast, discriminative eval is more valuable than a large one."
>
> **What it means:**
>
> This is good engineering advice, and it also explains Figure 7. If the search set holds the cases the baseline fails, every harness scores lower there than on a normal test set. So the dots of Figure 7 can sit above the diagonal without any magic.
>
> The advice has a price. A search set of 50 to 100 examples is small, and the loop compares dozens of candidates on it. The smaller the set, the more one lucky example moves a score. The code below measures that effect.

### How much luck can a search buy? (code)

Think of 40 students who all know exactly 40 percent of the material. Each one takes a 100-question test. Their scores will not all be 40: some get lucky questions. If you pick the top scorer and call them the best student, you are mostly picking luck. Statisticians call this the **winner's curse**.

> [!DEFINITION] Winner's curse
> When you choose the best of many noisy scores, the chosen score is too high on average. The reason is the choice itself: the winner is often the candidate whose noise happened to be largest. A new, independent measurement of the winner falls back towards its true value.

Here is the same idea as code. **Method 1** simulates it: 40 candidates, each truly 40 percent accurate, each scored on 100 random examples, and we keep the best score. We repeat this 2,000 times and average.

```python
def best_of_n_sim(true_acc, n_cand, n_search, trials, seed=0):
    """Method 1: simulate. Every candidate has the same true accuracy; we keep the best search score."""
    rng = random.Random(seed)
    best = []
    for _ in range(trials):
        scores = [sum(rng.random() < true_acc for _ in range(n_search)) for _ in range(n_cand)]
        best.append(max(scores) / n_search)
    return 100 * sum(best) / trials
```

**Method 2** gets the exact answer with the binomial distribution of the `scipy` library. One candidate's number of correct answers out of $$n$$ follows a binomial distribution with cumulative probability $$F(k)$$. All $$N$$ candidates are at most $$k$$ only if each one is, so the best score is at most $$k$$ with probability $$F(k)^N$$. The expected best score is the sum of the chances that it reaches each value:

$$
\mathbb{E}[\max] = \sum_{k=1}^{n} P(\max \ge k) = \sum_{k=1}^{n} \left(1 - F(k-1)^N\right)
$$

Read it from left to right. $$\mathbb{E}[\max]$$ is the average best number of correct answers. $$P(\max \ge k)$$ is the chance that the best candidate gets at least $$k$$ right. $$F(k-1)^N$$ is the chance that all $$N$$ candidates get $$k-1$$ or fewer right. A tiny check: with $$N = 1$$ the formula gives the plain average $$n p$$, because one candidate has no one to beat.

```python
def best_of_n_exact(true_acc, n_cand, n_search):
    """Method 2: exact. P(max >= k) = 1 - F(k-1)^N, and E[max] = sum over k of P(max >= k)."""
    F = stats.binom(n_search, true_acc).cdf(np.arange(n_search))      # F(0) ... F(n-1)
    return 100 * np.sum(1 - F ** n_cand) / n_search
```

The two methods agree, and the answer is large:

```text
40 candidates, all truly 40% accurate, search set of 100 examples
expected best search score: simulation 50.70%, exact 50.68%, agree within 0.3: True
the winner looks 10.7 points better than it is
inflation of the best search score (points), exact:
  candidates  n=50     n=88     n=100    n=250    
           1  0.0      0.0      0.0      0.0      
           4  7.1      5.4      5.1      3.2      
          10  10.7     8.1      7.6      4.8      
          40  15.1     11.4     10.7     6.7      
         109  17.8     13.4     12.6     7.9      
```

The sizes in the table come from the paper. Text classification had 40 candidates and math retrieval 109. The search sets held 50 to 100, 88 and 250 examples. With 40 equal candidates and 100 examples, the winner looks 10.7 points better than it is. More candidates make the effect larger. A larger search set makes it smaller.

{{FIG:p5_curse|How many points the best search score is above the true 40 percent, when every candidate is equally good. Each line is one search-set size. The effect grows with the number of candidates and shrinks as the search set grows. Exact values from the binomial distribution.}}

This model is a worst case, because every candidate here is equally good, so all of the spread is luck. Real candidates differ. When one harness is truly better by 10 points, the search will mostly find it, and only a part of its lead is luck. The table tells you how large a lead you need before you trust it.

### Our own small Figure 7 (code)

Let us make the same picture with a real model. Our task model is Qwen2.5-0.5B-Instruct, a small open model that runs on a laptop. The task is SST-2: decide whether a short movie review is positive or negative. We split 420 reviews into three disjoint sets: 60 for search, 60 held out, and 300 for the test.

A "harness" here is the smallest thing that deserves the name: a function that turns a review into a prompt. Two templates, each with no examples or with four examples drawn by one of five seeds, give 12 harnesses.

```python
def make_harness(template, seed):
    """A 'harness' here is a function from an input sentence to a prompt: a template plus 0 or 4 examples."""
    ex = [] if seed is None else shots(seed)
    head = ''.join(TEMPLATES[template].format(s=r['sentence'].strip()) + LABELS[r['label']] + '\n\n' for r in ex)
    return lambda s: head + TEMPLATES[template].format(s=s.strip())
```

To score a prompt, we ask the model which word is more likely next: " positive" or " negative". **Method 1** reads that from one forward pass: the log-probability of each label's token at the last position. **Method 2** asks the `transformers` library instead. When we give the model the label token as a training target, its built-in loss is minus the log-probability of that label.

```python
@torch.no_grad()
def scores_method1(prompts, bs=16):
    """Method 1: one forward pass; the log-probability of each label's token at the last position."""
    res = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors='pt', padding=True).to(dev)
        logits = model(**enc).logits[:, -1, :]                         # (batch, vocab)
        logp = torch.log_softmax(logits.float(), dim=-1)[:, LABEL_IDS]  # (batch, 2)
        res.append(logp.cpu())
    return torch.cat(res)


@torch.no_grad()
def logprob_method2(prompt, label_word):
    """Method 2: the library's own loss. With labels on the label token only, loss = -log P(label | prompt)."""
    ids_p = tok(prompt, return_tensors='pt').input_ids
    ids_l = tok(label_word, return_tensors='pt').input_ids
    ids = torch.cat([ids_p, ids_l], dim=1).to(dev)
    labels = ids.clone()
    labels[:, :ids_p.shape[1]] = -100                                  # ignore the prompt positions
    return -model(input_ids=ids, labels=labels).loss.item()
```

The two scorers agree to about three parts in a hundred thousand. Then every harness is scored on all three sets:

```text
label scoring check on 5 prompts: max difference ours vs library loss = 2.96e-05, agree: True
harness         search  held-out   test
A-zero           83.3%     86.7%  87.3%
A-4shot-s1       91.7%     95.0%  91.3%
A-4shot-s2       90.0%     95.0%  92.0%
A-4shot-s3       90.0%     85.0%  90.0%
A-4shot-s4       88.3%     90.0%  89.7%
A-4shot-s5       93.3%     93.3%  90.0%
B-zero           66.7%     68.3%  72.7%
B-4shot-s1       80.0%     86.7%  84.3%
B-4shot-s2       80.0%     81.7%  81.7%
B-4shot-s3       83.3%     85.0%  86.0%
B-4shot-s4       83.3%     78.3%  79.3%
B-4shot-s5       76.7%     73.3%  73.7%
chosen by search score: A-4shot-s5: search 93.3%, test 90.0%, gap +3.3 points
best on test (unknown during search): A-4shot-s2 at 92.0%; mean test of all 12: 84.8%
rank correlation of search and test over 12 harnesses: ours 0.912, scipy 0.912, agree: True
standard error of one accuracy on 60 examples near 93%: 3.2 points
```

{{FIG:p5_scatter|Our 12 prompt harnesses around Qwen2.5-0.5B-Instruct on SST-2. Left to right is the accuracy on 60 search reviews, bottom to top the accuracy on 300 test reviews. The dashed line is y = x. The harness with the best search score (93.3 percent) gets 90.0 percent on the test set; a different harness is best on the test set at 92.0 percent.}}

Both lessons of Figure 7 show up. The search worked: it chose a good harness, and the search order and the test order agree well. The **rank correlation** is 0.912. But the chosen harness lost 3.3 points between search and test, and it was not the best one on the test set. With only 60 search reviews, one review is 1.7 points, and the standard error of one score is about 3 points. The winner's lead over the next harness was 1.6 points, one review, well within that noise.

> [!DEFINITION] Rank correlation (Spearman)
> Sort the harnesses by search score and give them ranks 1, 2, 3 and so on; do the same by test score. The Spearman correlation is the ordinary correlation of the two lists of ranks. It is 1 when both orders are the same, 0 when they are unrelated, and -1 when one order is the reverse of the other.

We wrote the rank correlation from scratch (`spearman` in the script: rank, then Pearson correlation) and checked it against `scipy.stats.spearmanr`. Both give 0.912.

What our run cannot show: our harnesses are 12 fixed prompts, not programs that a proposer wrote after reading traces. Our task model has 0.5 billion parameters, while the paper's classifier is GPT-OSS-120B. So our numbers show the selection effect, not the size of the paper's gains. [Part 6](part-6-build-it-from-scratch.md) builds a full search loop and measures the same gap there.

## TerminalBench-2: search and test on the same tasks {§4.3}

> [!PAPER] · §4.3
> [![A paragraph of Section 4.3, with we perform search and final evaluation on the same 89-task benchmark, discovery problem, and regex-based audits for task-specific string leakage highlighted](/img/papers/meta-harness/p5-tbench-same.png)](/img/papers/meta-harness/p5-tbench-same.png)
>
> **What it says:** "we perform search and final evaluation on the same 89-task benchmark". The authors treat it as a "discovery problem". They say this is common practice, and that a separate split "would materially weaken the search signal". They check for overfitting "by manual inspection and regex-based audits for task-specific string leakage".
>
> **What it means:**
>
> On text classification and math, the search set and the test set are different. On TerminalBench-2 they are the same 89 tasks. So the scores in Table 7 (76.4 percent with Opus 4.6, 37.6 with Haiku 4.5) come from the exam the search practised on. The winner's curse of the last section applies to them in full.
>
> The authors are open about this, and their reasons are real. The benchmark is small, each run is expensive, and other teams also tune their agents on the same public tasks. A "discovery problem" means the goal is to find one good configuration for this benchmark, not to predict how it does on new tasks.
>
> The audits catch one kind of overfitting: a harness that mentions a task by name. They cannot catch the quieter kind, where a general-looking change happens to help these 89 tasks more than others. Later work tested exactly this, and the answer is in the last section of this page.

> [!DEFINITION] Pass rate (pass@1)
> The share of tasks an agent solves in one attempt each. On TerminalBench-2 a task passes when its own tests pass. A pass rate of 37.6 percent on 89 tasks is about 33 to 34 tasks, depending on how many runs are averaged.

### One task is 1.12 points (code)

A benchmark with 89 tasks moves in steps of 100 / 89 = 1.12 points. The **standard error** tells us how much a pass rate jumps between two runs of the same harness, when only luck changes.

> [!DEFINITION] Standard error
> The typical size of the random error in a measured average. For a pass rate $$p$$ on $$n$$ independent tasks it is $$\sqrt{p(1-p)/n}$$. A rough 95 percent range is the measured value plus or minus 1.96 standard errors.

For $$p = 0.644$$ and $$n = 89$$: $$0.644 \times 0.356 = 0.229$$, divided by 89 is 0.00258, and the square root is 0.0508, or 5.08 points. **Method 1** is that formula; **Method 2** asks `scipy` for the spread of the binomial distribution.

```python
def pass_rate_se(p, n):
    """Method 1: standard error of a pass rate p measured on n independent tasks, in points."""
    return 100 * math.sqrt(p * (1 - p) / n)


p0 = 0.644                                         # the Terminus-KIRA baseline in the Appendix A.2 search log
se1 = pass_rate_se(p0, N_TASKS)
se2 = 100 * stats.binom(N_TASKS, p0).std() / N_TASKS   # Method 2: the binomial distribution of scipy
```

```text
one task = 100 / 89 = 1.12 points
standard error at 64.4% on 89 tasks: from scratch 5.08 points, scipy 5.08 points, same: True
rough 95% range for one run: 64.4 +- 9.9 points
  evo_marker_fix       -5.6 points = -5.0 tasks if each task ran once
  evo_single_confirm   -6.7 points = -6.0 tasks if each task ran once
  evo_strip_only       -1.1 points = -1.0 tasks if each task ran once
runs per task (1 to 10) that fit 64.4, 58.9, 57.8 and 63.3 exactly: [9, 10]
```

Keep these numbers in mind for the search log in the next sections. A drop of 1.1 points is one task. If each task ran once, even the drops of 5.6 and 6.7 points are about one standard error. The last line is a small puzzle. With 1 to 8 runs per task, the four percentages of the log are never whole numbers of passes. So they are not plain pass counts over a fixed number of runs, and the paper does not say how they were averaged.

## What a search costs {§3, §1, Appendix D}

> [!PAPER] · §3
> [![A line of Section 3, with a typical run evaluates roughly 60 harnesses over 20 iterations highlighted](/img/papers/meta-harness/p5-cost-run.png)](/img/papers/meta-harness/p5-cost-run.png)
> [![A line of Section 1, with a single evaluation can produce up to 10,000,000 tokens of diagnostic information highlighted](/img/papers/meta-harness/p5-cost-tokens.png)](/img/papers/meta-harness/p5-cost-tokens.png)
>
> **What it says:** "a typical run evaluates roughly 60 harnesses over 20 iterations". And "a single evaluation can produce up to 10,000,000 tokens of diagnostic information".
>
> **What it means:**
>
> A run has two bills. The first is the **evaluations**: every candidate harness runs the task model on the whole search set. For example, 60 harnesses on 50 to 100 examples are 3,000 to 6,000 harness runs. On TerminalBench-2, each run is a long agent session in a sandbox. The second is the **proposer**. At every iteration, a frontier coding agent reads from a store that grows by up to 10 million tokens per evaluation.
>
> The paper reports counts and time, not money. Section 5 says a run takes "a few hours of wall-clock time". The acknowledgements thank a company for API credit. A reader who wants to copy the method must budget for both bills. The proposer reads a lot: a median of 82 files per iteration (Appendix A.1, covered in [Part 2](part-2-the-method.md)).

> [!PAPER] · Appendix D
> [![The bullet Write a good skill of Appendix D, with iterating on the skill text had a larger effect on search quality than changing iteration count or population size, and a few short evolution runs of 3 to 5 iterations each highlighted](/img/papers/meta-harness/p5-appd-skill.png)](/img/papers/meta-harness/p5-appd-skill.png)
>
> **What it says:** "iterating on the skill text had a larger effect on search quality than changing iteration count or population size". Expect to run "a few short evolution runs (3–5 iterations each)" to debug the skill "before committing to a full run".
>
> **What it means:**
>
> The skill is the instruction file that tells the proposer its job, where files live and what it may not touch. The authors say this file matters more than the size of the search. That is a hidden cost. Before the automatic search starts, a person writes and rewrites the skill, and each rewrite needs a few short runs to test it.
>
> So the human work moves; it does not disappear. Writing harnesses by hand becomes writing a good skill, a hard search set and a clean evaluator. The authors present these tips honestly as "engineering lessons", not as scientific claims.

## One strong proposer {§5, footnote 1}

> [!PAPER] · §5
> [![The last lines of Section 5, with one particularly strong coding-agent proposer and how the effect varies across proposer agents remains for future work highlighted](/img/papers/meta-harness/p5-one-proposer.png)](/img/papers/meta-harness/p5-one-proposer.png)
> [![Footnote 1 on page 4, with only became practical recently and around early 2026 highlighted](/img/papers/meta-harness/p5-footnote.png)](/img/papers/meta-harness/p5-footnote.png)
>
> **What it says:** the experiments show that harness search "can work with one particularly strong coding-agent proposer", which is Anthropic's coding agent with Opus 4.6. How the effect "varies across proposer agents remains for future work". Footnote 1 adds that the workflow "only became practical recently", after coding agents improved "around early 2026".
>
> **What it means:**
>
> Every result in the paper comes from one proposer. So we do not know which part of the gain comes from the loop. Another part may come from that agent's skill with logs and code. A weaker proposer might wander, break the interface, or miss the confound that the next section describes.
>
> The footnote says the same from the other side: the authors tried earlier and it did not work well enough. That is good news and bad news. The method can only get better as coding agents improve. But today it needs a frontier agent, which costs money and is not open. The papers in the last section of this page attack exactly this point.

## Inside a search run {Appendix A.2}

Appendix A follows one TerminalBench-2 search with Opus 4.6 for 10 iterations. [Part 4](part-4-math-and-terminalbench.md) summarises it; here we read the log with the noise numbers above in mind.

> [!PAPER] · Appendix A.2
> [![The start of Iterations 1 to 2 in Appendix A.2, with both regress sharply from the 64.4% Terminus-KIRA baseline highlighted](/img/papers/meta-harness/p5-a2-start.png)](/img/papers/meta-harness/p5-a2-start.png)
> [![The Iteration 3 paragraph and log box, with Root cause of regressions, the structural bugfixes were confounded with harmful prompt changes, and 63.3%, -1.1pp highlighted](/img/papers/meta-harness/p5-a2-iter3.png)](/img/papers/meta-harness/p5-a2-iter3.png)
>
> **What it says:** the first two candidates "both regress sharply from the 64.4% Terminus-KIRA baseline" (58.9 and 57.8 percent). Both bundled a code fix with a new prompt. At iteration 3 the proposer writes: "The structural bugfixes were confounded with harmful prompt changes." It keeps the fixes, restores the old prompt, and gets 63.3 percent, "-1.1pp".
>
> **What it means:**
>
> Iterations 1 and 2 each changed two things at once: a different code fix, and the same new prompt. Both got worse. The proposer noticed that the one thing they shared was the prompt, so it tested the fixes without it. That is how a careful engineer debugs, and it is the best evidence in the paper that reading full traces helps.
>
> Now apply the noise numbers. The 63.3 percent is about one task below the baseline, well inside one standard error. So iteration 3 shows that the fixes did not hurt much; it cannot show that they helped. The proposer's story is plausible, but one run on 89 tasks cannot prove it.

> [!DEFINITION] Confound
> A second change that moves together with the change you study, so you cannot tell which one caused the effect. If you take a new medicine and also start sleeping more, and you feel better, the sleep is a confound. The cure is to change one thing at a time.

> [!DEFINITION] Regression
> In software, a change that makes something that used to work stop working. In this log, a candidate "regresses" when its pass rate falls below the baseline.

> [!PAPER] · Appendix A.2
> [![The Iteration 7 paragraph and log box, with purely additive, without risking regression on already-passing tasks, and this candidate is the best result so far highlighted](/img/papers/meta-harness/p5-a2-iter7.png)](/img/papers/meta-harness/p5-a2-iter7.png)
>
> **What it says:** after six regressions, the proposer stops editing the control loop and tries a change that is "purely additive". It runs one shell command before the first model call and adds an environment snapshot to the prompt, "without risking regression on already-passing tasks". "This candidate is the best result so far."
>
> **What it means:**
>
> The proposer drew a good rule for any engineer. When edits to a working system keep breaking it, add information instead of changing behaviour. Telling the agent which tools are installed cannot make an easy task harder. It can only save the turns the agent would spend finding out.
>
> This is also the change that became the released harness (Figure 9, see [Part 4](part-4-math-and-terminalbench.md)). Note what the log does not give: the exact score of iteration 7 inside the search. The final 76.4 percent comes from the evaluation in Table 7, on the same 89 tasks.

> [!PAPER] · Appendix A.2
> [![The Iteration 10 paragraph and log box, with worth +18pp and crashed before evaluation highlighted](/img/papers/meta-harness/p5-a2-iter10.png)](/img/papers/meta-harness/p5-a2-iter10.png)
>
> **What it says:** at iteration 10 the proposer quotes a separate, earlier search run: "don't cleanup service artifacts" was "worth +18pp". Iteration 9 tried the same idea "but crashed before evaluation".
>
> **What it means:**
>
> Two facts hide here. First, the filesystem can hold more than one run, and the proposer uses lessons from an earlier run. That is useful. But it also means a run's history is wider than its 10 iterations, and so is the chance to fit the same 89 tasks.
>
> Second, candidates crash. Algorithm 1 checks each new harness with an interface test, but a harness can pass that test and still fail during the long evaluation. A crashed candidate costs a full iteration and gives no score. The appendix does not count how often this happened.

## Co-evolving the harness and the weights {§5}

> [!PAPER] · §5
> [![Lines of Section 5, with stronger general-purpose agents can outperform hand-engineered and co-evolve the harness and the model weights highlighted](/img/papers/meta-harness/p5-coevolve.png)](/img/papers/meta-harness/p5-coevolve.png)
>
> **What it says:** "once a search space becomes accessible, stronger general-purpose agents can outperform hand-engineered solutions". A next step is "to co-evolve the harness and the model weights, letting the strategy shape what the model learns and vice versa".
>
> **What it means:**
>
> The first sentence cites "The Bitter Lesson": in machine learning, general methods that use more computation tend to beat methods built on human knowledge. The authors place harness design in that story. A hand-written harness encodes what one engineer knows; a search with a strong agent can explore far more designs.
>
> The second sentence points ahead. A harness changes the traces the model produces, and traces can become training data. So a better harness could teach the next model, and a better model could need a different harness. Two of the later papers below follow exactly this thread.

## What came after

Within six months, several groups published follow-up work. Each one fixes a different limit from this page. We read each from its own PDF.

| Paper (arXiv) | Who proposes the new harness | What it changes |
|---|---|---|
| Agentic Harness Engineering, 2604.25850 | agents on the same base model as the task agent | structured evidence, and a prediction for every edit |
| Self-Harness, 2606.09498 | the task model itself | a held-out regression gate before any edit is kept |
| Recursive Harness Self-Improvement, 2607.15524 | an optimizer model, from pairwise judgments | one comparison per step instead of a population |
| Rethinking the Evaluation of Harness Evolution, 2607.12227 | (an evaluation study) | matched budgets and a held-out task split |
| Harness-Zero, 2609.24974 | an evolved harness, then fine-tuning | moves the harness gains into the model weights |

### Agentic Harness Engineering (April 2026)

> [!PAPER] Lin et al. (2026), Agentic Harness Engineering · Abstract
> [![The abstract of Agentic Harness Engineering, with component observability, experience observability distills millions of raw trajectory tokens, decision observability pairs every edit with a self-declared prediction, and from 69.7% to 77.0% highlighted](/img/papers/meta-harness/p5-ahe-abstract.png)](/img/papers/meta-harness/p5-ahe-abstract.png)
>
> **What it says:** AHE is a closed loop with three kinds of "observability". Every editable part of the harness is a file. Millions of trace tokens are condensed into layered reports. Each edit carries "a self-declared prediction" that the next round checks. Ten iterations lift pass@1 on TerminalBench-2 "from 69.7% to 77.0%".
>
> **What it means:**
>
> Meta-Harness gives the proposer raw files and lets it decide what to read. AHE adds structure around the same idea. An "Agent Debugger" first reads the traces and writes a report per task, so the proposer starts from root causes instead of 10 million raw tokens. And every edit must state which tasks it should fix and which it might break.
>
> AHE also cites Meta-Harness as one of the few methods that evolve a whole harness, not only its prompt. Its task agent, debugger and evolve agent all use one model, GPT-5.4, so the gain cannot come from a stronger helper. Like Meta-Harness on TerminalBench-2, AHE evolves on all 89 tasks and reports on them. It then tests the frozen harness on SWE-bench Verified and on other models.

> [!PAPER] Lin et al. (2026), Agentic Harness Engineering · §4.4.2
> [![A paragraph of the AHE paper, with regression precision of 11.8% and regression-recall of 11.1% and most upcoming regressions go unforeseen highlighted](/img/papers/meta-harness/p5-ahe-blind.png)](/img/papers/meta-harness/p5-ahe-blind.png)
>
> **What it says:** when the evolve agent predicts which tasks an edit will break, its "precision of 11.8% and regression-recall of 11.1%" are only about twice the random baselines of 5.6 and 5.4 percent. So "most upcoming regressions go unforeseen".
>
> **What it means:**
>
> Because AHE records a prediction for every edit, it can grade the agent's foresight, which Meta-Harness never measured. For fixes the agent is good: its fix predictions are right 33.7 percent of the time, against 6.5 percent for random guesses. For regressions it is almost blind.
>
> This matches the Meta-Harness log. In Appendix A.2, the proposer explained each failure well after it happened, but six of its first seven candidates still regressed. Explaining the past is easier than predicting what an edit will break.

> [!DEFINITION] Precision and recall
> Say an agent predicts that an edit breaks 10 tasks, and 2 of them really break, out of 18 that broke in total. **Precision** is the share of its predictions that came true: 2 / 10 = 20 percent. **Recall** is the share of real breaks it predicted: 2 / 18 = 11 percent.

### Self-Harness (June 2026)

> [!PAPER] Zhang et al. (2026), Self-Harness · Abstract
> [![The abstract of Self-Harness, with an LLM-based agent improves its own operating harness, without relying on human engineers or stronger external agents, and the three stages Weakness Mining, Harness Proposal and Proposal Validation highlighted](/img/papers/meta-harness/p5-self-abstract.png)](/img/papers/meta-harness/p5-self-abstract.png)
>
> **What it says:** in Self-Harness, "an LLM-based agent improves its own operating harness, without relying on human engineers or stronger external agents". The loop has three stages: "Weakness Mining", "Harness Proposal" and "Proposal Validation".
>
> **What it means:**
>
> This is a direct answer to the one-proposer limit. In Meta-Harness, a frontier coding agent writes harnesses for another model. In Self-Harness, the same model that does the tasks also proposes changes to its own harness. The paper uses three open-weight models: MiniMax M2.5, Qwen3.5-35B-A3B and GLM-5.
>
> The price of a weaker proposer is a smaller, more careful loop. Failures are first grouped by cause, edits must be "minimal", and the proposer sees grouped evidence instead of raw logs. On TerminalBench-2, held-out pass rates rise from 40.5 to 61.9 percent (MiniMax M2.5), 23.8 to 38.1 (Qwen3.5) and 42.9 to 57.1 (GLM-5). These use a 64-task subset, so they are not directly comparable with Table 7.

> [!PAPER] Zhang et al. (2026), Self-Harness · §3.4
> [![The acceptance rule of Self-Harness, with a candidate is accepted only if it improves at least one split without degrading the other, the rule with three conditions, and proposals that only trade off one split against the other are rejected highlighted](/img/papers/meta-harness/p5-self-rule.png)](/img/papers/meta-harness/p5-self-rule.png)
>
> **What it says:** "A candidate is accepted only if it improves at least one split without degrading the other." Candidates that "only trade off one split against the other are rejected", even if the total goes up.
>
> **What it means:**
>
> The tasks are split in two. The **held-in** split gives the traces that the proposer reads. The **held-out** split is never shown to the proposer; it only acts as a gate. An edit that helps held-in tasks but hurts held-out ones is probably fitting the tasks it studied, so it is thrown away.
>
> This is the guard against overfitting that Meta-Harness's TerminalBench-2 run did not have. One caution remains: the gate uses held-out scores to choose, so the held-out numbers also feel some winner's curse. The paper reports no third, untouched split.

The rule in symbols, with $$\Delta_{\text{in}}$$ and $$\Delta_{\text{ho}}$$ the change in the number of passed tasks on each split:

$$
\text{accept} \iff \Delta_{\text{in}} \ge 0, \quad \Delta_{\text{ho}} \ge 0, \quad \max(\Delta_{\text{in}}, \Delta_{\text{ho}}) > 0
$$

Read it as three tests. Neither split may get worse. At least one must get better. A candidate with $$(+3, 0)$$ passes. A candidate with $$(+5, -1)$$ fails, even though it gains 4 tasks in total. A candidate with $$(0, 0)$$ fails, because it changes nothing.

### The acceptance rule in code

**Method 1** is the rule for one candidate. **Method 2** applies it to whole arrays with `numpy`. We check both on every pair of changes from -2 to +2, then use the rule on our 12 SST-2 harnesses. The current harness is `A-zero`; the search set plays held-in and our 60 held-out reviews play held-out.

```python
def accept(d_in, d_ho):
    """Method 1: the rule of Self-Harness, Section 3.4, for one candidate (pass-count changes)."""
    return d_in >= 0 and d_ho >= 0 and max(d_in, d_ho) > 0


def accept_np(d_in, d_ho):
    """Method 2: the same rule on whole arrays with numpy."""
    d_in, d_ho = np.asarray(d_in), np.asarray(d_ho)
    return (d_in >= 0) & (d_ho >= 0) & (np.maximum(d_in, d_ho) > 0)
```

```text
rule checked on all 25 pairs of changes from -2 to +2: both methods agree: True
examples: (+3, 0) -> True, (+5, -1) -> False, (0, 0) -> False
current harness: A-zero (search 50/60, held-out 52/60)
  A-4shot-s1     d_in  +5  d_ho  +5  -> accept
  A-4shot-s2     d_in  +4  d_ho  +5  -> accept
  A-4shot-s3     d_in  +4  d_ho  -1  -> reject
  A-4shot-s4     d_in  +3  d_ho  +2  -> accept
  A-4shot-s5     d_in  +6  d_ho  +4  -> accept
  B-zero         d_in -10  d_ho -11  -> reject
  B-4shot-s1     d_in  -2  d_ho  +0  -> reject
  B-4shot-s2     d_in  -2  d_ho  -3  -> reject
  B-4shot-s3     d_in  +0  d_ho  -1  -> reject
  B-4shot-s4     d_in  +0  d_ho  -5  -> reject
  B-4shot-s5     d_in  -4  d_ho  -8  -> reject
accepted 4 of 11
largest accepted gain: A-4shot-s1: test 91.3% (current harness 87.3%)
```

The gate does its job on the clear cases. It rejects `A-4shot-s3`, which gained 4 search reviews but lost 1 held-out review.

Self-Harness merges all accepted edits into one harness. Prompts with different examples cannot be merged, so we keep the accepted one with the largest total gain. Two candidates tie at +10 reviews; we break the tie by the larger held-out gain, which picks `A-4shot-s1`. It scores 91.3 percent on the test set, up from 87.3 for the current harness. Here the stricter rule chose a better test harness than the best search score did (90.0 percent). With 12 candidates and 60 reviews per split, this is one small example, not a general result.

### Recursive Harness Self-Improvement (July 2026)

> [!PAPER] Lee et al. (2026), Recursive Harness Self-Improvement · Abstract
> [![The abstract of Recursive Harness Self-Improvement, with represents the harness as a prompt-level specification of the agent loop, pairwise feedback over its own revision history, and reducing inference cost by up to 60% highlighted](/img/papers/meta-harness/p5-rhi-abstract.png)](/img/papers/meta-harness/p5-rhi-abstract.png)
>
> **What it says:** RHI "represents the harness as a prompt-level specification of the agent loop" and refines it with "pairwise feedback over its own revision history". On 30 synthetic machine-learning research tasks, a few iterations beat the matching maximum reasoning-effort setting while "reducing inference cost by up to 60%".
>
> **What it means:**
>
> RHI targets the cost limit. Its tasks have no single right answer: the output is a whole code repository. So a judge model compares two outputs and says which is better. RHI compares the new harness only with the one before it, and keeps a history of these verdicts to steer the next edit.
>
> Its harness is text, not code: the roles of the agents, what they pass to each other, and the order of their steps. The authors also report a limit of their own (their §6.1). RHI on a weaker model does not reliably close the gap to a stronger model.

> [!DEFINITION] Pairwise judgment
> A judge looks at two outputs and says which is better, or that they are equal. Here the judge is another language model, and no output gets a score. It works for tasks where no test can mark an answer right or wrong, such as a research report.

> [!PAPER] Lee et al. (2026), Recursive Harness Self-Improvement · §3.1
> [![A passage of the RHI paper, with Meta-Harness is better viewed as optimizing the finite-candidate approximation in Equation (2) highlighted](/img/papers/meta-harness/p5-rhi-mh-a.png)](/img/papers/meta-harness/p5-rhi-mh-a.png)
> [![The continuation, with it therefore does not instantiate Equation (3), whose comparator is only the immediately previous harness highlighted](/img/papers/meta-harness/p5-rhi-mh.png)](/img/papers/meta-harness/p5-rhi-mh.png)
> [![Table 1 of the RHI paper: per-iteration cost. The ideal objective needs M traces and M choose 2 pairs, finite-population search needs m traces and m choose 2 pairs, and trajectory-local RHI needs 1 trace and 1 pair](/img/papers/meta-harness/p5-rhi-cost.png)](/img/papers/meta-harness/p5-rhi-cost.png)
>
> **What it says:** "Meta-Harness is better viewed as optimizing the finite-candidate approximation in Equation (2)": it keeps a population and lets the proposer read any earlier candidate. RHI instead compares only with "the immediately previous harness". Its Table 1 counts the cost per step: $$m$$ traces and $$\binom{m}{2}$$ pairs for a population of $$m$$, against 1 trace and 1 pair for RHI.
>
> **What it means:**
>
> The RHI authors place Meta-Harness in a family of population methods. All of them score many candidates and keep the best, which is accurate but expensive when each candidate is a long agent run. RHI gives up the population to pay for one new run and one comparison per step.
>
> The cost table counts pairwise comparisons, which RHI needs because it has no scores. Meta-Harness has scores, so it does not compare pairs; its bill is the $$m$$ evaluations. The table still shows the main trade: a population buys a broad search, and a chain of single steps buys a cheap one.

The number of pairs among $$m$$ outputs is $$\binom{m}{2} = m(m-1)/2$$: each of the $$m$$ outputs meets the $$m-1$$ others, and each pair is counted twice. **Method 1** counts the pairs one by one; **Method 2** uses `math.comb`.

```python
def pairs_by_hand(m):
    """Method 1: count the pairwise comparisons in a population of m outputs, one by one."""
    c = 0
    for i in range(m):
        for j in range(i + 1, m):
            c += 1
    return c
```

```text
  population   2:     1 pairs by hand,     1 with math.comb, trajectory-local: 1 pair
  population  10:    45 pairs by hand,    45 with math.comb, trajectory-local: 1 pair
  population  40:   780 pairs by hand,   780 with math.comb, trajectory-local: 1 pair
  population 109:  5886 pairs by hand,  5886 with math.comb, trajectory-local: 1 pair
```

With the 109 candidates of the math search, comparing every pair would take 5,886 judgments. RHI takes one per step.

### Rethinking the Evaluation of Harness Evolution (July 2026)

> [!PAPER] Wang et al. (2026), Rethinking the Evaluation of Harness Evolution for Agents · Abstract
> [![The abstract of the Rethinking paper, with violating the standard separation between training and test data, fails to outperform test-time scaling, and exhibits limited generalization highlighted](/img/papers/meta-harness/p5-rethink-abstract.png)](/img/papers/meta-harness/p5-rethink-abstract.png)
> [![Table 3 of the Rethinking paper: on Terminal-Bench 2.1 with disjoint search and evaluation tasks, the initial harness and the evolved harness both average 63.2 percent](/img/papers/meta-harness/p5-rethink-table3.png)](/img/papers/meta-harness/p5-rethink-table3.png)
>
> **What it says:** earlier harness-evolution papers search on the same benchmark they report on, "violating the standard separation between training and test data". Under matched budgets, harness evolution "fails to outperform" plain test-time scaling and "exhibits limited generalization". On disjoint tasks (Table 3), the average stays at 63.2 percent before and after evolution.
>
> **What it means:**
>
> This paper tests the two weak points of this page directly. First, it gives other methods the same budget. One runs the agent several times and keeps a good answer; one lets it revise its own answer. On Terminal-Bench 2.1 those plain uses of compute did as well or better. Second, it splits the 89 tasks into 45 for search, 10 for choosing, and 34 for the test. On the 34 unseen tasks, the evolved harness gains 1.2 points with Opus 4.6, nothing with GPT-5.4, and loses 1.4 with GPT-5.4 mini.
>
> Read the limits of this result too. The harness evolution they test is AHE with its explore agent turned off, not Meta-Harness itself, with a budget of five rounds. The same paper also adapts a harness to each single task, and that gave clear gains on long-horizon games such as ARC-AGI-3 and EdgeBench. So the fair summary is not "harness search does not work". It is: on Terminal-Bench, report a held-out split and a matched-budget baseline, or the gain may be search on the test.

> [!DEFINITION] Test-time scaling
> Spending more computation on each task when it is solved, instead of on a better model or harness. **Parallel sampling** runs the agent several times and keeps the best answer. **Sequential refinement** lets the agent read its last attempt and try again.

### Harness-Zero (September 2026)

> [!PAPER] Ye et al. (2026), Harness-Zero · Abstract
> [![The abstract of Harness-Zero, with agent harness distillation, transferring the behaviors it induces into model weights, macro-average task success from 23.3% to 44.3%, and even exceeding the 41.7% it reaches with that harness still attached highlighted](/img/papers/meta-harness/p5-hzero-abstract.png)](/img/papers/meta-harness/p5-hzero-abstract.png)
>
> **What it says:** Harness-Zero studies "agent harness distillation": it uses an optimized harness during training and transfers "the behaviors it induces into model weights". With the special harness removed, the model's "macro-average task success" rises "from 23.3% to 44.3%", "even exceeding the 41.7% it reaches with that harness still attached".
>
> **What it means:**
>
> This answers the co-evolution idea of §5. A harness found by search helps only while it runs around the model. Harness-Zero first evolves a harness, then lets a second agent read that harness and correct the model's answers during training runs. The model is fine-tuned on the corrected runs. At deployment it runs with a minimal harness, a single Bash tool, and keeps most of the gain.
>
> The base model is Qwen3.5-9B, tested on spreadsheets, multi-app tool use and USPTO chemistry. The evolved harness does not come from Meta-Harness itself but from the authors' own skill-guided loop. The paper names Meta-Harness as one method that can play this role. On USPTO the distilled model (30.0 percent) still trails the model with the harness attached (38.0). Procedures transfer into weights more easily than deep domain knowledge.

> [!DEFINITION] Distillation (fine-tuning on demonstrations)
> Training a model to copy good behaviour that something else produced. Here the "teacher" is the model inside a strong harness plus a reviewing agent. The student learns from their corrected runs, so it can behave the same way without the harness.

## The series in one page

Here is the whole story of the paper in six steps, with the page that covers each.

- [**Background**](background-what-a-harness-is.md): a harness is the code around a model call: memory, retrieval, prompt assembly and tool loops. Earlier optimizers improved prompts or single functions from scores or short feedback.
- [**Part 1**](part-1-the-big-idea.md): the same model can score very differently with different harnesses. Meta-Harness lets a coding agent search over harness code, reading full history from a filesystem.
- [**Part 2**](part-2-the-method.md): the objective is the expected reward of a harness. The loop of Algorithm 1 proposes, validates, evaluates and logs every candidate. The proposer reads the logs with ordinary tools such as `grep` and `cat`.
- [**Part 3**](part-3-text-classification.md): on text classification the best discovered harness reaches 48.6 percent with far less context than ACE. Raw traces matter: the scores-only and summary versions of the loop do much worse.
- [**Part 4**](part-4-math-and-terminalbench.md): a discovered retrieval router helps five models on IMO-level maths. On TerminalBench-2 a small, additive environment snapshot gives 76.4 percent with Opus 4.6 and 37.6 percent with Haiku 4.5.
- **Part 5** (this page): the TerminalBench-2 result uses the tasks it searched on, and every gain needs a check against luck. Later work moved the loop to the task model, cheaper steps, stricter gates and the weights.
- [**Part 6**](part-6-build-it-from-scratch.md): the whole loop in Python on a laptop, with a test set the proposer never sees.

> [!TAKEAWAYS] Key takeaways
> - Picking the best of many candidates on a small search set inflates its score. With 40 equal candidates and 100 examples, the winner looks 10.7 points better than it is.
> - On text classification and math, Meta-Harness reports test results on data the search never saw. On TerminalBench-2, the search and the final score used the same 89 tasks.
> - On 89 tasks, one task is 1.12 points and one run's standard error near 64 percent is about 5 points. Many steps in the Appendix A.2 log are within that noise.
> - Figure 7 shows search scores predict test scores well on LawBench, but on USPTO the best search scores do not give the best test scores.
> - A run evaluates about 60 harnesses, and a single evaluation can log up to 10 million tokens. The paper reports time and counts but no price.
> - All results use one frontier proposer. The authors say the method only became practical around early 2026.
> - AHE adds structure and a prediction for every edit; its agent foresees fixes well but most regressions not at all.
> - Self-Harness lets the task model improve its own harness and keeps an edit only if neither held-in nor held-out tasks get worse.
> - RHI needs one run and one pairwise judgment per step. A matched-budget study found no gain from harness evolution on held-out Terminal-Bench tasks, but per-task harness changes helped on long-horizon games.
> - Harness-Zero distils an evolved harness into the model weights, so the gain survives with a minimal harness.

## The paper and the code

[Paper (arXiv 2603.28052)](https://arxiv.org/abs/2603.28052) · [Project page](https://yoonholee.com/meta-harness/) · [Our scripts](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/meta-harness)
