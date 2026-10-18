---
title: "Meta-Harness: End-to-End Optimization of Model Harnesses"
short: Meta-Harness
description: "The 2026 paper that lets a coding agent write the code around a language model, by reading every earlier attempt's code, scores and traces. Six parts, with the paper's own text, worked examples, and a small Meta-Harness loop built from scratch in Python."
authors: [Yoonho Lee, Roshen Nair, Qizheng Zhang, Kangwook Lee, Omar Khattab, Chelsea Finn]
org: Stanford, KRAFTON, MIT
year: 2026
venue: COLM 2026
arxiv: "2603.28052"
code: https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact
accent: "#7fb069"
date: 2026-10-18
parts: 6
tags: [agents, harness-engineering, llm, optimization, coding-agents]
learn:
  - "What a harness is: the code that decides what a language model stores, retrieves and sees"
  - "Why the same model can score six times higher or lower with a different harness"
  - "Why earlier prompt optimizers fail here: they forget, see only a score, or squeeze feedback into a summary"
  - "The Meta-Harness loop: a proposer agent reads a filesystem of every earlier harness, its traces and its score, and writes the next one"
  - "What it found on text classification, math retrieval and TerminalBench-2, and what the ablations show"
  - "Where it can overfit, what it costs, and what later work changed"
  - "How to build a small harness-search loop yourself, with a scoring set and a held-out test set"
---

## Why this paper

A language model never works alone. Code around it decides what to remember, what to look up and what to show the model before each call. That code is the **harness**. Change the harness, and the same model can do much better or much worse on the same task.

Harnesses are mostly written by hand: a person reads failures, guesses a fix, and tries again. This paper asks whether an agent can do that job. Its system, **Meta-Harness**, gives a coding agent a folder with every earlier harness, the full record of how each one ran, and its score. The agent reads what it needs and writes a better harness. The paper tests this on three very different tasks and reports gains over hand-written harnesses on each.

## How the six parts are organised

**Before Part 1 (optional): [What a Harness Is](background-what-a-harness-is.md).** The model, the loop around it, and the earlier ways people tried to optimise prompts and programs.

1. **[The Big Idea](part-1-the-big-idea.md)**: the title, Figure 1, the abstract and §1.
2. **[The Method](part-2-the-method.md)**: §2, Table 1, §3, Figure 2 and Algorithm 1. The objective, the filesystem of experience, and the proposer.
3. **Text Classification** (coming soon): §4.1, Tables 2 to 5, Figures 3 and 4, and the discovered harnesses of the appendix.
4. **[Math Retrieval and TerminalBench-2](part-4-math-and-terminalbench.md)**: §4.2 and §4.3, Tables 6 and 7, and the discovered harnesses.
5. **[Limits, and What Came After](part-5-limits-and-what-came-after.md)**: §5, the proposer's behaviour in Appendix A, and the later work on self-improving harnesses.
6. **Build It from Scratch** (coming soon): a small Meta-Harness loop in Python, run on a laptop.

## How to read the boxes

- A **teal "From the paper" box** shows the exact lines of the paper as a highlighted screenshot. Click the screenshot to see it at full size. Under it, **What it says** gives the main words of the paper. **What it means** then explains those lines, and what the authors mean but do not write. When the box shows a different paper, the box gives the name of that paper.
- A **yellow definition box** explains a technical word the first time it occurs. If you know the word, skip the box.
- The small **Paper §3** tag above a heading tells you which section of the paper you read.
- A **green "Key takeaways" box** ends each part with the facts to remember.

Code blocks show real code, and the output under a code block is what that code printed. The scripts are in [`code/papers/meta-harness`](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/meta-harness). They run on a laptop with small open models; the paper used large commercial models, and each part says what that changes.

## What you need to know first

You need to know what a language model is and how a prompt works. Some Python helps for Part 6. The [Transformer series](/papers/transformer/) explains the model inside, but this paper treats the model as a black box.
