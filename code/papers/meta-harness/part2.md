---
title: "The Method: A Loop That Reads Every Earlier Attempt"
description: "How Meta-Harness searches for better harness code: the goal written as one formula, the loop of Figure 2 and Algorithm 1, and the folder of past attempts that the proposer reads. We run the whole loop in Python with a small local model and a scripted proposer."
part: 2
covers: "§2, Table 1, §3, Figure 2, Algorithm 1, Appendix A.1, Table 8, Appendix D, Appendix E"
date: 2026-10-18
tags: [meta-harness, harness-engineering, agents, optimization, llm]
---

Meta-Harness is a loop. An agent reads a folder full of earlier attempts, writes a new harness, and the loop scores it and adds it to the folder. Each piece below shows the exact lines of the paper, then explains them, and then runs them as code on a laptop.

> [!DEFINITION] Harness
> The code around a language model that decides what the model sees. It builds the prompt, picks examples, keeps a memory, and reads the answer. The model itself does not change. [The Background page](background-what-a-harness-is.md) and [Part 1](part-1-the-big-idea.md) explain harnesses in more depth.

## Where Meta-Harness sits {§2}

> [!PAPER] · §2
> [![The first paragraph of Section 2, with credit assignment, assigns credit at the harness level, and rewrites the external code that governs future behavior highlighted](/img/papers/meta-harness/p2-related-intro.png)](/img/papers/meta-harness/p2-related-intro.png)
>
> **What it says:** Meta-Harness takes ideas from "credit assignment" and meta-learning. Rather than updating model weights, it "assigns credit at the harness level". It reads past rollouts, finds the steps that caused failures, and then "rewrites the external code that governs future behavior".
>
> **What it means:**
>
> Think of a football team that loses a match. Credit assignment is the question "which player, at which moment, caused the goal against us?". A coach who only sees the final score cannot answer it. A coach who watches the full video can.
>
> Machine learning usually answers this question with gradients, and then changes the weights of the model. Meta-Harness leaves the weights alone. It answers the question by reading records of past runs, and then it changes the code around the model. The "player" that gets the blame is a line of harness code: a retrieval rule, a prompt, a memory update.
>
> The paper names three older lines of work that it builds on. The next three boxes take them one by one.

> [!DEFINITION] Credit assignment
> Finding which earlier choice caused a later result. In a long task, a choice at step 2 can cause a failure at step 40. The hard part is to connect the two.

> [!DEFINITION] Meta-learning
> Learning how to learn. A meta-learning system does not solve one task. It improves the procedure that solves many tasks. Here, the procedure is the harness, and the system that improves it is Meta-Harness.

> [!PAPER] · §2
> [![The paragraph External memory and adaptive access, with adaptive access to external context and selectively inspects a large external history of code, scores, and execution traces highlighted](/img/papers/meta-harness/p2-external-memory.png)](/img/papers/meta-harness/p2-external-memory.png)
>
> **What it says:** retrieval-augmented generation, interleaved retrieval and reasoning, memory-based agents and recursive language models are all "mechanisms for adaptive access to external context". Meta-Harness uses the same pattern: the proposer "selectively inspects a large external history of code, scores, and execution traces".
>
> **What it means:**
>
> A student in an open-book exam does not read the whole library before each question. She looks up the one page she needs. That is adaptive access. The books stay on the shelf, outside her head, until a question calls for them.
>
> Retrieval-augmented generation does this for facts: it fetches a few passages and puts them in the prompt. Meta-Harness does it for experience. Its "library" is the record of every earlier harness. The proposer opens only the files that look useful for the problem in front of it.

> [!DEFINITION] Retrieval-augmented generation (RAG)
> A harness that searches a collection of documents for text that matches the question, and puts the best matches into the prompt. The model then answers with that text in front of it.

> [!PAPER] · §2
> [![The paragraph Executable code search, with evolve designated functions within fixed program scaffolds, deliberately minimal, and unrestricted filesystem access to prior experience highlighted](/img/papers/meta-harness/p2-code-search.png)](/img/papers/meta-harness/p2-code-search.png)
>
> **What it says:** earlier methods search over code: some use large models as mutation operators in evolutionary search, some "evolve designated functions within fixed program scaffolds", some search over agent workflows or memory designs. The outer loop of Meta-Harness is "deliberately minimal". It gives the proposer "unrestricted filesystem access to prior experience".
>
> **What it means:**
>
> Evolutionary search copies nature. Keep a population of programs, make small random changes, keep the ones that score best, repeat. Later methods let a language model make the changes. They still fix the frame: a "scaffold" program with one marked function, and only that function may change.
>
> Meta-Harness removes the frame. There is no fixed scaffold, no rule for which parent to change, and no special memory. There is only a folder of everything that happened and an agent that may read any of it. The bet is that a capable agent makes better choices than a fixed search rule.

> [!DEFINITION] Evolutionary search
> A search method that keeps a population of candidate solutions. In each round it changes some of them (mutation), sometimes mixes two (crossover), scores the results, and keeps the best. AlphaEvolve and OpenEvolve do this with a language model that writes the changes.

> [!PAPER] · §2
> [![The paragraph Text optimization methods, with complete executable procedure, hard to summarize up front, and reason over failed examples and their execution traces highlighted](/img/papers/meta-harness/p2-text-opt.png)](/img/papers/meta-harness/p2-text-opt.png)
>
> **What it says:** ProTeGi, TextGrad, OPRO, GEPA, AlphaEvolve/OpenEvolve and Feedback Descent improve prompts and other text with feedback from earlier attempts. They fit harness engineering less well. There, the target is a "complete executable procedure", and the useful feedback is spread over code, scores and traces "in a way that is hard to summarize up front".
>
> **What it means:**
>
> A prompt optimizer changes one piece of text, runs it, and reads a score or a short critique. That works when one prompt gives one answer. A harness is a whole program. It may call the model many times, store things, and fetch things. A bad choice at the start can show up as a wrong answer much later.
>
> To find such a cause, you need the details: which examples failed, what the prompt looked like at that moment, what the model said. A summary written before anyone knows the bug must guess which details matter, and it often guesses wrong. Meta-Harness keeps all the details and lets the proposer look for the cause itself.

> [!DEFINITION] Execution trace
> The full record of one run of a harness: every prompt it built, every model answer, every tool call and every change to its memory. In our code below, a trace is one line of JSON per task.

## Table 1: how much each optimizer sees {Table 1}

> [!PAPER] · Table 1
> [![Table 1 of the paper. Six methods with their history type, log content and MTok per iteration: OPRO 0.002, TextGrad 0.015, AlphaEvolve 0.022, GEPA 0.008, Feedback Descent 0.012, TTT-Discover 0.026. Meta-Harness has Full history, all logs and scores, 10.0](/img/papers/meta-harness/p2-table1.png)](/img/papers/meta-harness/p2-table1.png)
>
> **What it says:** for each method, the table gives what history the optimizer keeps (Window, Last, Summary or Full), what the log holds, and "MTok/iter", the authors' estimate of the context made by one evaluation. Meta-Harness has "all logs and scores", about 10.0 million tokens, which is "orders-of-magnitude more context per artifact evaluation".
>
> **What it means:**
>
> Read the "History" column first. **Last** means the optimizer sees only the current attempt. **Window** means it sees a few recent attempts. **Summary** means it sees a short text that someone, often another model, wrote about the attempts. **Full** means it can see every attempt in full.
>
> Then read the last column. The earlier methods get between 2,000 and 26,000 tokens per step. Meta-Harness gets up to 10 million. That is far more than any model can read at once. This is why the next sections are about a folder, and an agent that searches it.
>
> The numbers are the authors' own estimates for the largest setting in each paper. They show the scale of the gap, not an exact measurement.

> [!DEFINITION] Token and MTok
> A **token** is the unit of text a language model reads: a word or a part of a word. "MTok" means a million tokens. A page of English text is roughly 500 to 700 tokens.

{{FIG:p2_table1|Table 1 drawn on a log scale, where each grid line is ten times the one before. The six earlier methods sit between 2,000 and 26,000 tokens. Meta-Harness sits at 10,000,000.}}

A few lines of Python turn the table into ratios:

```python
TABLE1 = {'OPRO': 0.002, 'TextGrad': 0.015, 'AlphaEvolve': 0.022, 'GEPA': 0.008,
          'Feedback Descent': 0.012, 'TTT-Discover': 0.026, 'Meta-Harness': 10.0}
largest_prior = max(v for k, v in TABLE1.items() if k != 'Meta-Harness')
```

```text
== 1. Table 1: context per evaluation (MTok/iter, the paper's estimates)
  OPRO                0.002 MTok =        2,000 tokens   x     0.1 of the largest prior
  TextGrad            0.015 MTok =       15,000 tokens   x     0.6 of the largest prior
  AlphaEvolve         0.022 MTok =       22,000 tokens   x     0.8 of the largest prior
  GEPA                0.008 MTok =        8,000 tokens   x     0.3 of the largest prior
  Feedback Descent    0.012 MTok =       12,000 tokens   x     0.5 of the largest prior
  TTT-Discover        0.026 MTok =       26,000 tokens   x     1.0 of the largest prior
  Meta-Harness       10.000 MTok =   10,000,000 tokens   x   384.6 of the largest prior
  Meta-Harness / largest prior (TTT-Discover): 385x
  log10 of that ratio: 2.59 orders of magnitude
```

The largest earlier budget is TTT-Discover's 26,000 tokens. Meta-Harness has about 385 times that. On a log scale, 385 is 2.59 steps of ten. The paper rounds this to "three orders of magnitude" in §1.

> [!PAPER] · Appendix E
> [![Two paragraphs of Appendix E. AlphaEvolve / OpenEvolve, with program database with scalar scores and single stateless function with a clean scalar objective highlighted. GEPA, with operates on one candidate at a time and access to all prior candidates simultaneously highlighted](/img/papers/meta-harness/p2-appE.png)](/img/papers/meta-harness/p2-appE.png)
>
> **What it says:** AlphaEvolve and OpenEvolve give the proposer a "program database with scalar scores" (4K to 22K tokens per step). They suit problems where the target is a "single stateless function with a clean scalar objective". GEPA reads rollout traces, but it "operates on one candidate at a time" (2K to 8K tokens per step). Meta-Harness gives "access to all prior candidates simultaneously".
>
> **What it means:**
>
> This appendix explains the rows of Table 1 that are closest to Meta-Harness. AlphaEvolve is very good at problems like "find a faster sorting function". There, the function has no memory, and one number says how good it is. A small change gives a small, local effect.
>
> A harness is different: it is "stateful". It keeps a memory across many examples, so one choice about what to store can change every later answer. GEPA comes closer, because it reads traces. But it reads the traces of one candidate at a time. To see why a retrieval rule helps on one kind of problem and hurts on another, you must compare traces across many candidates. That is what the full folder allows.

> [!DEFINITION] Stateful program
> A program that remembers things between calls. A calculator is stateless: 2 + 2 always gives 4. A harness that stores every labelled example it has seen is stateful: its answer to the next question depends on what it stored before.

## A harness for optimizing harnesses {§3}

> [!PAPER] · §3
> [![The first two paragraphs of Section 3, with selectively inspect prior code and execution traces via filesystem access and Meta-Harness is itself a harness highlighted](/img/papers/meta-harness/p2-section3-intro.png)](/img/papers/meta-harness/p2-section3-intro.png)
>
> **What it says:** the method lets a proposer "selectively inspect prior code and execution traces via filesystem access", in place of lossy summaries or a hand-designed search structure. It repeatedly proposes, evaluates and logs new harnesses. "Meta-Harness is itself a harness", because it decides what the proposer model sees.
>
> **What it means:**
>
> The name is a small joke with a real point. A harness decides what a model sees. Meta-Harness decides what the proposer sees: it hands the proposer a folder and tools to search it. So Meta-Harness is a harness too, one level up.
>
> On this page there are two models, so keep them apart. The **task model** $$M$$ does the task, for example "label this review". It never changes. The **proposer** $$P$$ writes harness code for $$M$$. In the paper, $$P$$ is a strong coding agent. On the rest of this page, "harness" means the task harness that $$P$$ writes.

## The objective {§3}

> [!PAPER] · §3
> [![The Objective paragraph of Section 3 and its equation, with stateful program that wraps a language model, task distribution, fixed language model, rollout trajectory, reward function and maximizes the expected final reward highlighted, and the equation in a yellow box](/img/papers/meta-harness/p2-objective.png)](/img/papers/meta-harness/p2-objective.png)
>
> **What it says:** a harness is a "stateful program that wraps a language model". Let $$M$$ be a "fixed language model" and $$\mathcal{X}$$ a "task distribution". For a harness $$H$$ and a task $$x \sim \mathcal{X}$$, we run a "rollout trajectory" $$\tau \sim p_M(H, x)$$. A "reward function" $$r(\tau, x)$$ scores it. The goal is the harness that "maximizes the expected final reward".
>
> **What it means:**
>
> Picture a cooking contest. The cook is fixed: that is the model. You may only change the recipe card you hand to the cook: that is the harness. The judges taste many dishes, from many requests, on many days. You want the recipe card with the best average score over all of them.
>
> The formula says exactly this. It has an average over tasks, because one task proves little. It also has an average over runs, because a language model can answer differently each time. The section below reads the formula symbol by symbol and computes it by hand.

### The equation, symbol by symbol

$$
H^{*} = \arg\max_{H} \; \mathbb{E}_{x \sim \mathcal{X},\; \tau \sim p_M(H, x)} \; r(\tau, x)
$$

Read it from the inside out.

- $$r(\tau, x)$$ is the **reward**: one number that says how well the run $$\tau$$ did on the task $$x$$. For a classification task it is 1 for the right label and 0 for a wrong one.
- $$\tau$$ is the **rollout trajectory**: everything that happened in one run. The harness builds a prompt, the model answers, the harness updates its state, and so on.
- $$\tau \sim p_M(H, x)$$ says that the run is random. The same harness $$H$$ and task $$x$$ can give different runs, because the model $$M$$ can sample different words. $$p_M$$ is the chance of each run.
- $$x \sim \mathcal{X}$$ says that the task is drawn from a distribution $$\mathcal{X}$$: the kind of tasks we care about, such as "movie reviews to label".
- $$\mathbb{E}$$ is the **expected value**: the average reward over both kinds of chance, the task and the run.
- $$\arg\max_H$$ means "the $$H$$ that makes the average largest". $$H^{*}$$ is that best harness. Note what is missing: $$M$$ does not appear under the $$\arg\max$$. The model is fixed; only $$H$$ may change.

> [!DEFINITION] Expected value
> The long-run average of a random number. If a coin gives 1 for heads and 0 for tails, its expected value is $$0.5 \times 1 + 0.5 \times 0 = 0.5$$. Each outcome is weighted by its chance.

> [!DEFINITION] Distribution
> A rule that says how likely each outcome is. A task distribution says how often each kind of task occurs. In practice we never know it exactly; we only have a sample of tasks drawn from it.

**A worked example.** Take two tasks, $$x_1$$ and $$x_2$$, each with a chance of 0.5. Take two harnesses. With harness $$H_A$$ the model gets $$x_1$$ right 90% of the time and $$x_2$$ right 50% of the time. With $$H_B$$ the numbers are 70% and 80%. The reward is 1 for right and 0 for wrong.

$$
\mathbb{E}[r \mid H_A] = 0.5 \times 0.9 + 0.5 \times 0.5 = 0.70, \qquad \mathbb{E}[r \mid H_B] = 0.5 \times 0.7 + 0.5 \times 0.8 = 0.75
$$

So $$H^{*} = H_B$$. Note that $$H_A$$ is the better harness on $$x_1$$. It still loses, because the objective averages over all tasks. A harness that shines on one example and fails on others does not win.

### Method 1 and Method 2: the exact sum and the sampled average

Method 1 follows the definition: a sum over tasks and runs, each weighted by its chance. Method 2 does what an evaluation really does: draw tasks, run them, and average the rewards. The second way uses NumPy, the standard Python library for arrays of numbers.

```python
P_TASK = {'x1': 0.5, 'x2': 0.5}
P_RIGHT = {'H_A': {'x1': 0.9, 'x2': 0.5}, 'H_B': {'x1': 0.7, 'x2': 0.8}}


def expected_reward_exact(h):
    """Method 1: the definition. Sum over tasks of p(x) times the sum over rollouts of p(tau) r(tau, x)."""
    total = 0.0
    for x, px in P_TASK.items():
        p = P_RIGHT[h][x]
        total += px * (p * 1.0 + (1 - p) * 0.0)        # two rollouts: right (reward 1) or wrong (reward 0)
    return total


def expected_reward_sampled(h, n=100_000, seed=0):
    """Method 2: Monte Carlo. Draw a task, draw a rollout, average the rewards (what evaluation really does)."""
    rng = np.random.default_rng(seed)
    xs = rng.choice(list(P_TASK), size=n, p=list(P_TASK.values()))           # x ~ X
    p = np.array([P_RIGHT[h][x] for x in xs])
    r = (rng.random(n) < p).astype(float)                                      # tau ~ p_M(H, x), r(tau, x) in {0, 1}
    return r.mean(), r.std() / np.sqrt(n)
```

```text
== 2. The objective: E over tasks x and rollouts tau of r(tau, x)
  H_A: exact 0.7000   Monte Carlo 0.7011 (+/- 0.0014)   difference 0.0011
  H_B: exact 0.7500   Monte Carlo 0.7513 (+/- 0.0014)   difference 0.0013
  argmax over the two harnesses: H_B
```

The two methods agree to about 0.001. The sampled average is never exact: the "+/-" is its standard error, the typical size of its mistake. With 100,000 draws it is 0.0014. A real evaluation has far fewer draws, and the next run shows what that means.

> [!DEFINITION] Monte Carlo estimate
> Estimating an average by drawing random samples and averaging them. With $$n$$ samples, the typical error shrinks like $$1 / \sqrt{n}$$: four times the samples halves the error.

### The objective with a real model

Now the same objective with a real language model. The task model $$M$$ is Qwen2.5-0.5B-Instruct, a small open model, run locally. The task is SST-2, a public set of movie-review sentences labelled positive or negative. We estimate the expected reward of two harnesses on 24 reviews (the search set built further below). Each harness runs three times with random sampling (temperature 1.0), and once in greedy mode, which always picks the most likely word.

```text
== 2b. E[r] with Qwen2.5-0.5B-Instruct on the 24 search reviews
  zero_shot   sampled (T=1.0) runs [0.708, 0.458, 0.583] -> mean 0.583;  greedy 0.500
  few_shot_4  sampled (T=1.0) runs [0.125, 0.208, 0.208] -> mean 0.181;  greedy 0.167
  argmax of the sampled estimate: zero_shot
```

Look at the three sampled runs of zero-shot: 0.708, 0.458 and 0.583. Same harness, same 24 reviews, same model, and the score moves by 0.25. That is $$\tau \sim p_M(H, x)$$ at work. One run is a noisy guess of the expected reward; the mean of three runs, 0.583, is a better one.

On this search set, the zero-shot harness wins clearly over the few-shot harness (0.583 against 0.181). That looks odd, since examples usually help. The traces explain it, and we read them further below. Note also that the greedy score of zero-shot is exactly 0.500. That is by design: we built the search set from 12 reviews that greedy zero-shot got wrong and 12 that it got right.

> [!DEFINITION] Temperature and greedy decoding
> A language model gives a chance to every possible next token. **Greedy** decoding always takes the most likely one, so the same prompt gives the same answer. Sampling with **temperature** 1.0 draws tokens at random by those chances, so answers can differ from run to run.

### When there is more than one goal

> [!PAPER] · §3
> [![The paragraph after the equation, with Pareto dominance and report the resulting frontier highlighted](/img/papers/meta-harness/p2-pareto.png)](/img/papers/meta-harness/p2-pareto.png)
>
> **What it says:** "When multiple objectives are relevant (e.g., accuracy and context cost), we evaluate candidates under Pareto dominance and report the resulting frontier." Until now, people did this search by hand.
>
> **What it means:**
>
> The formula above has one number to maximize. Real systems care about two or more: accuracy, but also how many tokens each call costs. A harness that is 1 point more accurate but sends five times more text is not clearly better.
>
> So the paper does not force the two numbers into one. It keeps every harness that no other harness beats on both at once. That set is the **Pareto frontier**. A user can then pick a point on it: cheap and good, or expensive and best.

> [!DEFINITION] Pareto dominance and the Pareto frontier
> Harness A **dominates** harness B when A is at least as good on every goal and strictly better on at least one. Here: accuracy at least as high, context at most as large, and one of them strictly better. The **Pareto frontier** is the set of harnesses that no other harness dominates.

Here is Pareto dominance in code, run on the numbers of the paper's Table 2 (accuracy in %, extra context in thousands of tokens; [Part 3](part-3-text-classification.md) explains that table). Method 1 checks every pair. Method 2 is the classic faster way: sort by context, then walk along and keep each point that beats the best accuracy so far.

```python
def dominates(a, b):
    """a = (accuracy, context). a dominates b if it is no worse on both and strictly better on one."""
    return a[0] >= b[0] and a[1] <= b[1] and (a[0] > b[0] or a[1] < b[1])


def pareto_brute(points):
    """Method 1: keep every point that no other point dominates (compare all pairs)."""
    return sorted(k for k, p in points.items() if not any(dominates(q, p) for j, q in points.items() if j != k))


def pareto_sweep(points):
    """Method 2: sort by context (then by accuracy, high first); keep a point if it beats the best accuracy so far."""
    out, best = [], -float('inf')
    for k, (acc, ctx) in sorted(points.items(), key=lambda kv: (kv[1][1], -kv[1][0])):
        if acc > best:
            out.append(k)
            best = acc
    return sorted(out)
```

```text
== 3. Pareto dominance: higher accuracy and lower context are both better
  brute force: ['Few-Shot (32)', 'Few-Shot (8)', 'Meta-Harness', 'Zero-Shot']
  sweep      : ['Few-Shot (32)', 'Few-Shot (8)', 'Meta-Harness', 'Zero-Shot']
  same frontier: True
  Few-Shot (all) is dominated by ['Meta-Harness']
  MCE is dominated by ['Few-Shot (all)', 'Meta-Harness']
  ACE is dominated by ['Meta-Harness']
```

Both methods find the same four points. Few-Shot (all) has 40.8% at 12.3K tokens, but Meta-Harness has 48.6% at 11.4K: better on both, so Few-Shot (all) drops out. ACE has 40.9% at 50.8K, and Meta-Harness beats it on both too. Zero-Shot stays on the frontier even with the lowest accuracy, because nothing uses less context than zero.

{{FIG:p2_pareto|The numbers of Table 2 as points. Up is more accurate and left is cheaper. The line joins the four points of the Pareto frontier. Every point off the line has a point on the line that is both higher and further left.}}

## The search loop {§3, Figure 2}

> [!PAPER] · Figure 2
> [![Figure 2 of the paper: a filesystem with all experience on the left, an agent that proposes harness code, a harness wrapped around a frozen model that is evaluated on tasks, and an arrow that stores all logs back into the filesystem](/img/papers/meta-harness/p2-figure2.png)](/img/papers/meta-harness/p2-figure2.png)
>
> **What it says:** "(1) An agent reads a filesystem containing all prior candidates' source code, execution traces, and scores, and proposes a new harness. (2) We evaluate the proposed harness on evaluation tasks. (3) All logs (proposed code, reasoning traces, evaluation scores) are stored in the filesystem in a new directory, and the loop repeats."
>
> **What it means:**
>
> Follow the three numbers. At (1), the small robot is the proposer. It reads the folder on the left, which holds one sub-folder per earlier harness. It writes new harness code. At (2), the new harness wraps the frozen model (the snowflake means "frozen") and runs on the tasks. At (3), everything from that run goes into a new sub-folder.
>
> The word to notice is "all". Nothing is thrown away, and nothing is shortened into a summary. The folder only grows. The proposer never sees all of it at once; it chooses what to read.

> [!PAPER] · §3
> [![The paragraph Meta-Harness search loop, with single coding-agent proposer, growing filesystem D, decides which prior artifacts to inspect, far larger than the proposer's context window, and grep and cat highlighted](/img/papers/meta-harness/p2-search-loop.png)](/img/papers/meta-harness/p2-search-loop.png)
> [![Footnote 1 of page 4, with only became practical recently highlighted](/img/papers/meta-harness/p2-footnote.png)](/img/papers/meta-harness/p2-footnote.png)
>
> **What it says:** Meta-Harness uses a "single coding-agent proposer" with a "growing filesystem $$\mathcal{D}$$" as its feedback channel. The agent "decides which prior artifacts to inspect", which failures to fix, and how big an edit to make. The filesystem is "far larger than the proposer's context window", so the agent queries it with "grep and cat". The footnote adds that this "only became practical recently", after coding agents improved around early 2026.
>
> **What it means:**
>
> Older optimizers build one prompt for the model: "here are the last five attempts and their scores, write a better one". The outer loop decides what goes into that prompt. Meta-Harness turns this around. The outer loop decides almost nothing. The agent looks around like a programmer on a new codebase: it lists folders, greps for errors, opens a trace, compares two versions.
>
> This is why the proposer must be an agent and not a bare model. A bare model reads one fixed prompt and writes one answer. An agent can run tools many times, read the results, and decide what to look at next. The footnote is honest about the cost of this design: it needs a strong agent, and the authors think such agents only arrived recently.

> [!DEFINITION] Coding agent
> A language model inside a loop that lets it use developer tools: list files, read them, search them, run commands, and edit code. It decides which tool to call next from what it has seen so far. The paper's proposer is Anthropic's coding agent, running Opus 4.6.

> [!DEFINITION] Context window
> The largest amount of text a model can read in one call, counted in tokens. Anything beyond it must be left out, cut down, or fetched in pieces.

> [!DEFINITION] grep and cat
> Two standard command-line tools. `cat file` prints a file. `grep pattern files` prints the lines (or, with `-l`, the files) that contain a pattern. Together they let an agent find and read the few files it needs in a large folder.

> [!PAPER] · §3
> [![The paragraph that starts Meta-Harness maintains a population, with imposes no parent-selection rule, final test-set evaluation on the Pareto frontier, never sees test-set results and the search set highlighted](/img/papers/meta-harness/p2-population.png)](/img/papers/meta-harness/p2-population.png)
>
> **What it says:** Meta-Harness keeps a population and a Pareto frontier, but "imposes no parent-selection rule". It runs for a fixed number of iterations and does a "final test-set evaluation on the Pareto frontier". The proposer "never sees test-set results"; its only feedback comes from "the search set" and the traces logged on it.
>
> **What it means:**
>
> In evolutionary search, a rule picks which programs get changed next: often the best few. Meta-Harness has no such rule. The proposer may start from the best harness, from an old failure with one good idea, or from two harnesses at once. In practice it often starts from a strong one, but that is its own choice.
>
> The second half of the paragraph is about honesty. The tasks are split in two.
>
> The **search set** is what every candidate is scored on during the search; the proposer reads those scores and traces. The **test set** is locked away until the end. Only the final frontier runs on it, once. This is the same rule as "do not train on the exam". A harness tuned on the search set may be good only on those examples. The test set shows whether it is.

> [!DEFINITION] Search set and test set
> Two separate groups of tasks. The **search set** gives the scores and traces that guide the search. The **test set** is used once, after the search, to measure how well the chosen harnesses do on tasks the search never saw. A large gap between the two scores is a sign of overfitting.

## Algorithm 1: the outer loop {Algorithm 1}

> [!PAPER] · Algorithm 1
> [![Algorithm 1 of the paper, Meta-Harness outer loop over harnesses, thirteen lines, with Proposer P queries filesystem D, if H passes interface validation then, and return Pareto frontier of harnesses stored in D highlighted](/img/papers/meta-harness/p2-algorithm1.png)](/img/papers/meta-harness/p2-algorithm1.png)
>
> **What it says:** the inputs are tasks $$\mathcal{X}$$, a model $$M$$, a proposer $$P$$ and a number of iterations $$N$$. Evaluate each harness of the starting population and store it in $$\mathcal{D}$$. Then, $$N$$ times: the "Proposer P queries filesystem D", proposes $$k$$ new harnesses, and each one that passes "interface validation" is evaluated and stored. At the end, "return Pareto frontier of harnesses stored in D".
>
> **What it means:**
>
> The algorithm is short on purpose, and each line has one job.
>
> 1. Lines 1 to 3 set up. The **population** $$\mathcal{H}$$ holds the starting harnesses: the usual baselines of the task, such as zero-shot and few-shot prompting. $$\mathcal{D}$$ starts empty.
> 2. Lines 4 to 6 score every starting harness and store it. $$E_H$$ is the result of the evaluation: the scores and the traces.
> 3. Line 8 is the whole intelligence of the method, and it is one line: the proposer reads whatever it wants from $$\mathcal{D}$$.
> 4. Line 9 writes $$k$$ new harness files. In the text-classification run, $$k = 2$$.
> 5. Line 11 is a cheap gate. A harness that does not even load, or has the wrong function names, is dropped before it costs a full evaluation.
> 6. Line 12 is the same as line 6: evaluate and store. The folder only grows.
> 7. Line 13 returns the Pareto frontier. The test set is not in the algorithm at all; it comes after.

> [!DEFINITION] Population
> The set of candidate solutions a search keeps at one time. In Meta-Harness, the population is every harness evaluated so far; all of them stay in $$\mathcal{D}$$.

{{FIG:p2_loop|Algorithm 1 with the folders of our own run. The proposer reads the filesystem and writes new harness files. Each file passes an interface check, runs on the search set with the frozen model, and becomes a new folder. The test set is used only after the loop.}}

### Algorithm 1 as Python

We now run Algorithm 1 for real, small. The pieces:

- **Tasks.** SST-2 movie reviews. A memory of 64 labelled reviews from the training split, which harnesses may show to the model. A search set of 24 reviews, and a test set of 48 other reviews.
- **The model $$M$$.** Qwen2.5-0.5B-Instruct, frozen, greedy decoding, on a laptop.
- **A harness.** One Python file with one function, `run(x, model, memory)`, that returns a label and a trace.
- **The proposer $$P$$.** A stub: a short Python function that reads the folder the way an agent would. It writes a diagnosis and picks the next edit from a fixed list. It is not a language model. [Part 6](part-6-build-it-from-scratch.md) replaces it with a real local model.

The search set follows a tip from the paper's Appendix D (shown at the end of this page): make it hard for the baseline. We ran the zero-shot harness on 96 reviews and kept 12 that it got wrong and 12 that it got right.

```text
== search set: 96 candidate reviews, zero-shot greedy
  zero-shot on the pool: 0.875 (12 misses, 1 unclear answers), 273 s
  search set: 12 misses + 12 hits = 24 reviews;  test set: 48 other reviews
```

Here is the first harness of the population, exactly as the loop writes it to `000_zero_shot/harness.py`:

```python
"""Zero-shot: ask for the label, show no examples."""

def parse(out):
    """Read the model's answer: 'positive', 'negative', or None if it is not clear."""
    o = out.lower()
    pos, neg = 'positive' in o, 'negative' in o
    return 'positive' if pos and not neg else 'negative' if neg and not pos else None


def run(x, model, memory):
    prompt = f"Review: {x}\nIs the sentiment of this review positive or negative?"
    out = model(prompt)
    return parse(out), [dict(step='answer', prompt=prompt, output=out)]
```

`evaluate` runs one harness file on a list of tasks. It returns the scores and one trace record per task. `validate` is line 11 of Algorithm 1: it loads the file and calls `run` once on a fake model.

```python
def evaluate(path, tasks, model):
    """Run one harness on a list of (x, gold) tasks. Return the scores and one trace record per task."""
    h = load_harness(path)
    traces, right, chars = [], 0, 0
    for i, (x, gold) in enumerate(tasks):
        pred, steps = h.run(x, model, memory)
        right += pred == gold
        chars += sum(len(s.get('prompt', '')) for s in steps)
        traces.append(dict(id=i, x=x, gold=gold, pred=pred, correct=pred == gold, steps=steps))
    return dict(accuracy=right / len(tasks), avg_prompt_chars=chars / len(tasks), n=len(tasks)), traces


def validate(path):
    """Interface validation (line 11): the file imports, has run(), and run() answers one tiny task."""
    try:
        h = load_harness(path)
        label, trace = h.run('a fine film .', lambda prompt: 'positive', memory[:4])
        return label in ('positive', 'negative', None) and isinstance(trace, list)
    except Exception as e:                              # a syntax error, a missing run(), a crash...
        return False
```

`store` is lines 6 and 12: one new folder per harness, with the code, the scores and the traces.

```python
def store(name, source, proposal, model):
    """D <- D + {(H, Evaluate(H, M, X))}: one new folder with the code, the scores and the traces."""
    d = D_ROOT / f'{len([p for p in D_ROOT.iterdir() if p.is_dir()]):03d}_{name}'
    d.mkdir()
    (d / 'harness.py').write_text(source)
    (d / 'proposal.md').write_text(proposal)
    scores, traces = evaluate(d / 'harness.py', search, model)
    (d / 'scores.json').write_text(json.dumps(scores, indent=1))
    (d / 'traces.jsonl').write_text(''.join(json.dumps(t) + '\n' for t in traces))
    log(f'  stored {d.name:24s} search acc {scores["accuracy"]:.3f}   prompt chars {scores["avg_prompt_chars"]:7.1f}')
    return d
```

The stub proposer is line 8 and line 9. It reads every score file, the code of the two best harnesses and every earlier proposal note. It also reads the list of rejected candidates and the traces of the best harness. It counts unclear answers and wrong labels in those traces and writes them into its note. Then it takes the next untried edits from its list. The function `read` is its `cat`: it opens a file and records the access.

```python
def stub_proposer(it, k=2):
    """Stand-in for the paper's coding agent. It reads D like an agent would (scores, code, traces), writes a
    short diagnosis, and returns k new harness files. Unlike the real proposer, it picks the edits from a fixed list."""
    folders = sorted(p for p in D_ROOT.iterdir() if p.is_dir())
    scores = {f.name: json.loads(read(f / 'scores.json', it)) for f in folders}       # every score
    ranked = sorted(scores, key=lambda n: -scores[n]['accuracy'])
    for n in ranked[:2]:
        read(D_ROOT / n / 'harness.py', it)                                             # the code of the top two
    for f in folders:
        read(f / 'proposal.md', it)                                                     # what earlier steps tried
    best = ranked[0]
    traces = [json.loads(l) for l in read(D_ROOT / best / 'traces.jsonl', it).splitlines()]
    unclear = sum(t['pred'] is None for t in traces)
    wrong = sum(t['pred'] is not None and not t['correct'] for t in traces)
    tried = {f.name.split('_', 1)[1] for f in folders}
    rejected = D_ROOT / 'rejected.jsonl'
    if rejected.exists():                                                               # candidates that failed line 11
        tried |= {json.loads(l)['name'] for l in read(rejected, it).splitlines()}
    new = [(n, why, src) for n, why, src in EDITS if n not in tried][:k]
    out = []
    for n, why, src in new:
        note = (f'# iteration {it}: {n}\nRead {len(folders)} score files. Best so far: {best} '
                f'(search acc {scores[best]["accuracy"]:.3f}). In its traces: {unclear} unclear answers, '
                f'{wrong} wrong labels out of {len(traces)}.\nEdit: answers "{why}".\n')
        out.append((n, src, note))
    return out
```

And the loop itself, with the line numbers of Algorithm 1 in the comments:

```python
if D_ROOT.exists():
    shutil.rmtree(D_ROOT)
D_ROOT.mkdir(parents=True)                                                  # line 3: D <- empty
CALLS['n'] = 0
t = time.time()
log('-- lines 4-6: evaluate the initial population')
for name, src in SEEDS.items():
    store(name, src, f'# seed: {name}\nA baseline harness from the initial population.\n', greedy)
N = 2
for it in range(1, N + 1):                                                  # line 7
    log(f'-- iteration {it}: the proposer reads D and proposes')
    for name, src, note in stub_proposer(it):                               # lines 8-9
        cand = D_ROOT.parent / 'part2_candidate.py'
        cand.write_text(src)
        ok = validate(cand)                                                 # line 11
        cand.unlink()
        if ok:
            store(name, src, note, greedy)                                  # line 12
        else:
            log(f'  rejected {name}: failed interface validation (not evaluated; one line in rejected.jsonl)')
            with open(D_ROOT / 'rejected.jsonl', 'a') as fh:
                fh.write(json.dumps(dict(iteration=it, name=name, reason='interface validation failed')) + '\n')
log(f'  search done: {CALLS["n"]} model calls, {time.time() - t:.0f} s')

# line 13: the Pareto frontier of D (search accuracy up, prompt characters down)
pts = {f.name: (json.loads((f / 'scores.json').read_text())['accuracy'],
                json.loads((f / 'scores.json').read_text())['avg_prompt_chars']) for f in sorted(D_ROOT.iterdir()) if f.is_dir()}
front = pareto_sweep(pts)
assert front == pareto_brute(pts)
log(f'-- line 13: Pareto frontier of D: {front}')
(D_ROOT / 'frontier.json').write_text(json.dumps(front, indent=1))

# After the search: the test set, once, on the frontier only. Written outside D.
log('')
log('== after the search: the held-out test set (48 reviews), frontier only')
T = {}
for n in front:
    s, _ = evaluate(D_ROOT / n / 'harness.py', test, greedy)
    T[n] = dict(search=pts[n][0], test=s['accuracy'], prompt_chars=s['avg_prompt_chars'])
    log(f'  {n:24s} search {pts[n][0]:.3f}   test {s["accuracy"]:.3f}   gap {pts[n][0] - s["accuracy"]:+.3f}')
TEST_OUT.write_text(json.dumps(T, indent=1))
```

```text
== 4. Algorithm 1: the outer loop
-- lines 4-6: evaluate the initial population
  stored 000_zero_shot            search acc 0.500   prompt chars   178.9
  stored 001_few_shot_4           search acc 0.167   prompt chars   430.9
-- iteration 1: the proposer reads D and proposes
  stored 002_one_word_answer      search acc 0.583   prompt chars   230.9
  rejected broken_rename: failed interface validation (not evaluated; one line in rejected.jsonl)
-- iteration 2: the proposer reads D and proposes
  stored 003_retrieve_4           search acc 0.333   prompt chars   637.5
  stored 004_retrieve_balanced    search acc 0.292   prompt chars   660.8
  search done: 120 model calls, 315 s
-- line 13: Pareto frontier of D: ['000_zero_shot', '002_one_word_answer']
== after the search: the held-out test set (48 reviews), frontier only
  000_zero_shot            search 0.500   test 0.917   gap -0.417
  002_one_word_answer      search 0.583   test 0.938   gap -0.354
```

Read the log from the top. The two seeds score 0.500 and 0.167 on the search set.

In iteration 1 the stub reads the folder and sees one unclear answer in the best harness. It proposes two files. The one-word-answer harness scores 0.583, the best so far. The renamed file fails the interface check in a second, and it costs no model call at all. In iteration 2 the stub proposes two retrieval harnesses, and both score worse than zero-shot.

The frontier has two harnesses. Zero-shot uses the shortest prompts (178.9 characters on average). The one-word-answer harness has the best accuracy for 230.9 characters. Each of the other three is both less accurate and longer, so each is dominated.

Now the traces. Here is the reading a real proposer would do, in a few lines of Python over the saved folder. It calls no model.

```python
for folder in sorted(p for p in D.iterdir() if p.is_dir()):
    traces = [json.loads(l) for l in (folder / 'traces.jsonl').read_text().splitlines()]
    unclear = [t for t in traces if t['pred'] is None]
    wrong = [t for t in traces if t['pred'] is not None and not t['correct']]
    log(f'{folder.name:22s} unclear {len(unclear):2d}   wrong {len(wrong):2d}   right {len(traces) - len(unclear) - len(wrong):2d}   of {len(traces)}')
    if unclear:
        out = unclear[0]['steps'][0]['output'].replace('\n', ' ')
        log(f'   first unclear answer: {out[:90]!r}')
```

```text
== what the traces of each harness say
000_zero_shot          unclear  1   wrong 11   right 12   of 24
   first unclear answer: 'The sentiment of this review is clearly negative. The reviewer criticizes Jason X for bein'
001_few_shot_4         unclear 12   wrong  8   right  4   of 24
   first unclear answer: 'Based on the sentiment analysis provided for each review, I would categorize them as follo'
002_one_word_answer    unclear  0   wrong 10   right 14   of 24
003_retrieve_4         unclear  8   wrong  8   right  8   of 24
   first unclear answer: "I apologize, but I'm not able to review or provide feedback on any specific movie reviews "
004_retrieve_balanced  unclear  8   wrong  9   right  7   of 24
   first unclear answer: 'The sentiment of this review can be categorized as **negative**.  ### Analysis: 1. **Posit'
```

This output holds the real diagnosis. The few-shot harness gave 12 unclear answers out of 24. Its first one starts "Based on the sentiment analysis provided for each review, I would categorize them as follows". With four labelled reviews in one message, the small model thinks it must analyse all five reviews, and it never gives one label. The retrieval harnesses copy that prompt format, so they copy the bug: 8 unclear answers each.

Our stub cannot see this. It counts unclear answers only in the best harness, and it picks its edits from a list. A proposer that reads the few-shot traces would see the bug at once. It might then write a harness that asks for one word after the examples. This is the paper's argument in small form: the score said "few-shot is bad", but only the trace said why.

Last, the test set. Both frontier harnesses score much higher on the 48 test reviews (0.917 and 0.938) than on the search set.

That is not a miracle. The search set was built to be hard: half of it is reviews that zero-shot got wrong. The test set is a random sample, where zero-shot gets most reviews right. So the two numbers measure different things, and only the test number estimates accuracy on new reviews. On the test set, the one-word-answer harness is ahead by 0.021, which is one review out of 48.

{{FIG:p2_run|The harnesses on the Pareto frontier of our run, with their accuracy on the 24 search reviews and on the 48 test reviews. The search set was built to be hard for zero-shot, so the two scores are not the same measurement.}}

The whole search made 120 model calls and took 315 seconds on a laptop GPU that other jobs also used. The paper's runs evaluate about 60 harnesses with large models; Part 5 discusses that cost.

## Why search in code space {§3}

> [!PAPER] · §3
> [![The paragraph Advantages of code-space search on page 4, with small changes to retrieval, memory, or prompt-construction logic can affect behavior many steps later highlighted](/img/papers/meta-harness/p2-advantages-1.png)](/img/papers/meta-harness/p2-advantages-1.png)
> [![Its continuation on page 5, with infer why a harness failed, identify confounded edits, natural regularization bias, and coherent algorithms rather than brittle, hard-coded solutions highlighted](/img/papers/meta-harness/p2-advantages-2.png)](/img/papers/meta-harness/p2-advantages-2.png)
>
> **What it says:** in code space, "small changes to retrieval, memory, or prompt-construction logic can affect behavior many steps later", so local search rules fit badly. Traces let the proposer "infer why a harness failed" and "identify confounded edits". Writing harnesses as programs gives a "natural regularization bias": coding models propose "coherent algorithms rather than brittle, hard-coded solutions".
>
> **What it means:**
>
> Two ideas sit in this paragraph.
>
> The first is about cause and effect. When a candidate changes two things and gets worse, a score alone cannot say which change hurt. That is a **confounded edit**. With the traces, the proposer can look at the failed examples and see which change caused them. Appendix A shows a real case: two candidates each changed a bug fix and a prompt together, and both got worse. The proposer found that the prompt was the cause.
>
> The second idea is about what kind of solution the search finds. A search could "win" on the search set with a rule like "if the review contains the word 'Seagal', answer negative". That is brittle: it fits a few examples and nothing else. A coding model, trained on real software, tends to write general code instead, such as "retrieve similar examples and show both labels". That habit pushes the search toward harnesses that also work on new data. The authors call it a regularization bias, and [Part 5](part-5-limits-and-what-came-after.md) checks how far it holds.

> [!DEFINITION] Confounded edit
> A change that alters two or more things at once. If the score moves, you cannot tell which part caused it. The fix is to test each part alone.

> [!DEFINITION] Regularization
> Anything that pushes a learner toward general solutions, so it does not just memorize its training examples. Here, the push comes from the habits of a model that writes code.

> [!PAPER] · §3
> [![The paragraph Practical implementation, with single-file Python program, minimal domain-specific skill, always frozen, and roughly 60 harnesses over 20 iterations highlighted](/img/papers/meta-harness/p2-practical.png)](/img/papers/meta-harness/p2-practical.png)
>
> **What it says:** each harness is a "single-file Python program" that changes prompting, retrieval, memory and orchestration. The proposer is Anthropic's coding agent with Opus 4.6, guided by a "minimal domain-specific skill" that says where to write harnesses, how to read old ones, and which files it may not change. The base model is "always frozen". A typical run evaluates "roughly 60 harnesses over 20 iterations".
>
> **What it means:**
>
> Three practical choices make the loop work. One file per harness keeps each candidate easy to read, diff and load. A **skill** is a short instruction document for the agent: the rules of the game, not a recipe for how to think. And the task model never changes, so any gain comes from the harness alone.
>
> The last number sets the scale. Sixty harnesses in 20 iterations is about three per iteration. Each harness needs a full evaluation on the search set, and evaluation is the expensive part. That is why the search set must be small, a point Appendix D comes back to.

> [!DEFINITION] Skill (for an agent)
> A short text file with instructions that an agent reads before it starts. It gives the agent its role, the folder layout and the commands it may run. It also says what the agent must not touch and what it must produce.

## How the proposer uses the filesystem {Appendix A.1, Table 8}

> [!PAPER] · Appendix A.1
> [![Appendix A.1 and Table 8, with median of 82 files per iteration, non-Markovian, majority of available history, and Files read per iteration (median) highlighted](/img/papers/meta-harness/p2-table8.png)](/img/papers/meta-harness/p2-table8.png)
>
> **What it says:** in the TerminalBench-2 run (10 iterations, Opus 4.6), the proposer read a "median of 82 files per iteration" (range 69 to 99). About 41% were harness source code, 40% execution traces, 6% score summaries and 13% other files. Its access pattern is "non-Markovian": it reads the "majority of available history".
>
> **What it means:**
>
> This table answers a fair doubt. Maybe the agent ignores the big folder and only edits the last harness? The log of its file reads says no. It opens about 80 files each time, and it splits its attention almost evenly between code and traces. The scores themselves are only 6% of its reading. It looks up the numbers, then spends its time on the evidence behind them.
>
> "Markovian" means "depends only on the last step". An optimizer that only edits the latest candidate is Markovian. Reading most of the history is the opposite, and it is the behaviour the design was built to allow.

> [!DEFINITION] Markovian
> A process whose next step depends only on its current state, not on how it got there. A proposer that only reads the most recent harness is Markovian. One that reads the whole history is not.

Our stub proposer records its reads in the same way. The numbers below are not a result about agents: the stub follows a fixed reading plan, and our folder is tiny. They only show the bookkeeping that produced Table 8.

```text
== the proposer's file reads (Table 8 style; a scripted proposer, so only a picture of the bookkeeping)
  iteration 1: 7 files read
  iteration 2: 10 files read
  source   4   23.5%
  trace    2   11.8%
  score    5   29.4%
  other    6   35.3%
```

The stub read 7 files in iteration 1 and 10 in iteration 2, because the folder had grown. Its mix is very different from Table 8: it reads every score file but only one trace file per iteration. That is the stub's fixed plan, and it is the reason it missed the few-shot bug above. The paper's agent spent 40% of its reads on traces.

### The folder, listed and searched

The paper's proposer moves through the folder with standard tools. Here is our folder after the run. Method 1 walks it with our own loop over `os.walk`. Method 2 lets Python's `pathlib` library do the walking. Then the same two ways for `grep`: a search in Python, and the real `grep` program that an agent would call.

```python
def tree(root):
    """Method 1: walk the folders ourselves with os.walk; return every file path relative to root."""
    out = []
    for folder, dirs, files in os.walk(root):
        dirs.sort()
        for f in sorted(files):
            out.append(os.path.relpath(os.path.join(folder, f), root))
    return sorted(out)


def tree_lib(root):
    """Method 2: the standard library's pathlib does the walking."""
    return sorted(str(p.relative_to(root)) for p in Path(root).rglob('*') if p.is_file())


def grep(root, pattern):
    """Method 1: grep -rl in Python. Return the files that contain the pattern."""
    rx = re.compile(pattern)
    return sorted(p for p in tree(root) if rx.search((Path(root) / p).read_text()))


def grep_tool(root, pattern):
    """Method 2: the real grep program, the tool the paper's proposer calls."""
    out = subprocess.run(['grep', '-rlE', pattern, '.'], cwd=root, capture_output=True, text=True).stdout
    return sorted(l[2:] for l in out.split())
```

```text
== 5. The filesystem D
  000_zero_shot/harness.py                     520 bytes
  000_zero_shot/proposal.md                     66 bytes
  000_zero_shot/scores.json                     71 bytes
  000_zero_shot/traces.jsonl                13,110 bytes
  001_few_shot_4/harness.py                    589 bytes
  001_few_shot_4/proposal.md                    67 bytes
  001_few_shot_4/scores.json                    86 bytes
  001_few_shot_4/traces.jsonl               19,204 bytes
  002_one_word_answer/harness.py               605 bytes
  002_one_word_answer/proposal.md              209 bytes
  002_one_word_answer/scores.json               86 bytes
  002_one_word_answer/traces.jsonl          11,852 bytes
  003_retrieve_4/harness.py                    825 bytes
  003_retrieve_4/proposal.md                   229 bytes
  003_retrieve_4/scores.json                    73 bytes
  003_retrieve_4/traces.jsonl               28,608 bytes
  004_retrieve_balanced/harness.py           1,009 bytes
  004_retrieve_balanced/proposal.md            228 bytes
  004_retrieve_balanced/scores.json             85 bytes
  004_retrieve_balanced/traces.jsonl        24,594 bytes
  frontier.json                                 44 bytes
  rejected.jsonl                                83 bytes
  os.walk: 22 files, pathlib: 22 files, same list: True
  files with an unclear answer ("pred": null): ['000_zero_shot/traces.jsonl', '001_few_shot_4/traces.jsonl', '003_retrieve_4/traces.jsonl', '004_retrieve_balanced/traces.jsonl']
  Python grep and grep -rlE agree: True
  D holds 102,243 characters: about 25,561 tokens by the rule of thumb, 25,303 by the Qwen tokenizer
  Table 1 says up to 10,000,000 tokens per evaluation in the paper: 395x our whole D
```

Each evaluated harness has its own folder with four files: `harness.py` (the code), `proposal.md` (why it was written), `scores.json` and `traces.jsonl`. The rejected candidate left one line in `rejected.jsonl` and no folder. The traces are by far the largest files, as in the paper.

Both ways of listing give the same 22 files, and both ways of searching give the same four trace files with an unclear answer. One `grep` found every harness that has the parsing problem, without opening a single file by hand. That is the kind of query the paper's proposer runs.

The last two lines compare sizes. A token count by the rule of thumb (characters divided by 4) gives 25,561; the real Qwen tokenizer gives 25,303, close enough. Our whole folder, after five harnesses, is about the size of one TTT-Discover step in Table 1 (26,000 tokens). The paper's 10 million tokens per evaluation is 395 times our whole folder.

[![Terminal output of mh_part2.py with every result on this page](/img/papers/meta-harness/part2-run.png)](/img/papers/meta-harness/part2-run.png)

## Practical tips from the authors {Appendix D}

Appendix D lists what the authors learned while building the system. They say clearly that these are engineering lessons, not scientific claims. Four of them shape how anyone would build this loop.

> [!PAPER] · Appendix D
> [![The tip Write a good skill, with primary interface for steering the search, not the proposer's diagnosis procedure, and iterating on the skill text had a larger effect on search quality highlighted](/img/papers/meta-harness/p2-tip-skill.png)](/img/papers/meta-harness/p2-tip-skill.png)
>
> **What it says:** the skill text is the "primary interface for steering the search". It should constrain outputs and safety, "not the proposer's diagnosis procedure". After enough iterations, the traces shape the proposer more than the skill does. Still, "iterating on the skill text had a larger effect on search quality" than changing the number of iterations or the population size.
>
> **What it means:**
>
> Think of a manager and a skilled engineer. A good manager says what to deliver, what is off limits, and how success is measured. A bad manager also dictates how to debug. The skill should be the good manager: rules about outputs, not a checklist for thinking.
>
> The last sentence is a surprise worth keeping. The wording of one instruction file mattered more than running the search longer. The authors suggest a few short runs of 3 to 5 iterations just to fix the skill before the real run.

> [!PAPER] · Appendix D
> [![The tip Start with a baseline harness and a search set that is hard for it, with filtering for examples that the baseline gets wrong and roughly 50 full evaluations per run highlighted](/img/papers/meta-harness/p2-tip-searchset-1.png)](/img/papers/meta-harness/p2-tip-searchset-1.png)
> [![Its end on the next page, with a fast, discriminative eval is more valuable than a large one highlighted](/img/papers/meta-harness/p2-tip-searchset-2.png)](/img/papers/meta-harness/p2-tip-searchset-2.png)
>
> **What it says:** write a baseline first, then build the search set by "filtering for examples that the baseline gets wrong" or by picking hard, varied examples. Keep it small enough for "roughly 50 full evaluations per run": 50 to 100 examples for classification, 88 problems for math. "A fast, discriminative eval is more valuable than a large one."
>
> **What it means:**
>
> If the baseline already gets 95% of the search set right, every new harness scores between 95% and 100%. The differences are lost in noise, and the search has nothing to learn from. A search set full of hard cases makes good and bad ideas look different. That is what "discriminative" means here.
>
> Size is a trade. A larger set gives a more exact score, but each evaluation costs more, and the loop needs about 50 of them. Our run took this advice: its search set is half zero-shot misses. The price is a search score that is not a fair estimate of accuracy on new reviews. That is one more reason to keep a separate test set.

> [!PAPER] · Appendix D
> [![The tips Log everything in a format that is easy to navigate and Make logs queryable through a small CLI, with machine-readable formats such as JSON and lists the Pareto frontier highlighted](/img/papers/meta-harness/p2-tip-logging.png)](/img/papers/meta-harness/p2-tip-logging.png)
>
> **What it says:** write code, scores and traces in "machine-readable formats such as JSON", in a clear folder tree with steady file names, so that regex search works well. A small command-line tool that "lists the Pareto frontier", shows the top harnesses and diffs two runs helps the proposer save tokens.
>
> **What it means:**
>
> The proposer is a reader, and the folder is its book. A well-organized book is faster to read. If every folder has the same four files with the same names, one `grep` over `*/scores.json` finds every score. If a trace is one JSON object per line, one `grep` for `"pred": null` finds every unclear answer, as our run did above.
>
> The small command-line tool is a shortcut for questions the agent asks again and again. Without it, the agent spends tokens to work out the frontier by hand each time. With it, one command gives the answer.

> [!DEFINITION] JSON
> A plain-text format for structured data, such as `{"accuracy": 0.75, "n": 24}`. Programs and language models can both read it reliably.

> [!PAPER] · Appendix D
> [![The tips Lightweight validation before expensive benchmarks and Automate evaluation outside the proposer, with imports the module, instantiates the class, and calls both methods, catch most malformed or nonfunctional candidates in seconds, and Automate evaluation outside the proposer highlighted](/img/papers/meta-harness/p2-tip-validation.png)](/img/papers/meta-harness/p2-tip-validation.png)
>
> **What it says:** write a small test that "imports the module, instantiates the class, and calls both methods" on a few examples. Every proposed harness must pass it first; it can "catch most malformed or nonfunctional candidates in seconds". And "Automate evaluation outside the proposer": a separate program scores candidates and writes the results.
>
> **What it means:**
>
> This is line 11 of Algorithm 1 in practice. A full evaluation can take many minutes and cost real money. A broken file should fail in a second, not after it has used that budget. Our `validate` function does exactly this, and in our run it caught the renamed entry point without any call to the model.
>
> The second tip keeps the roles clean. The proposer writes harnesses; a fixed program runs them. If the proposer ran its own evaluations, it could make mistakes in them, or change how they work. Keeping evaluation outside the proposer also makes every score in the folder come from the same code.

## What a small local run cannot show

Everything above runs, but at a toy scale, and it is fair to say what that hides. Our task model has 0.5 billion parameters; the paper's text-classification model is GPT-OSS-120B. Our search set has 24 reviews, so one review moves the score by 1/24.

Above all, our proposer is a script. It reads the folder and writes a diagnosis, but it does not reason over the traces, and its edits come from a list we wrote. The paper's main claim is that a strong coding agent can do that reasoning. A stub cannot test that claim; it can only show the machinery around it. [Part 6](part-6-build-it-from-scratch.md) tries a local language model as the proposer and reports what happens.

> [!TAKEAWAYS] Key takeaways
> - Meta-Harness assigns credit at the level of harness code: it reads records of past runs, finds what caused failures, and rewrites the code. The model's weights never change.
> - Earlier optimizers see 2,000 to 26,000 tokens of feedback per step (Table 1). Meta-Harness can see up to 10 million, about 385 times more.
> - The objective is $$H^{*} = \arg\max_H \mathbb{E}_{x \sim \mathcal{X}, \tau \sim p_M(H,x)} r(\tau, x)$$: the harness with the best average reward over tasks and over random runs, for a fixed model.
> - With two goals, such as accuracy and context cost, the paper keeps the Pareto frontier: every harness that no other harness beats on both.
> - The loop: a coding agent reads a growing folder of code, scores and traces with tools like grep and cat. It writes new harnesses, and each one is checked, evaluated and stored.
> - There is no parent-selection rule and no summary step. The proposer decides what to read and what to change.
> - The proposer sees only the search set. The test set runs once, on the final frontier.
> - In the TerminalBench-2 run, the proposer read a median of 82 files per iteration, 41% code and 40% traces (Table 8).
> - Appendix D: the skill text matters most. Make the search set small and hard for the baseline, log in JSON, validate cheaply, and keep evaluation outside the proposer.
> - Our laptop run shows the machinery with a 0.5B model and a scripted proposer. It cannot show the reasoning of a strong proposer.

## The paper and the code

[Paper (arXiv 2603.28052)](https://arxiv.org/abs/2603.28052) · [Project page](https://yoonholee.com/meta-harness/) · [Our scripts for this series](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/meta-harness)
