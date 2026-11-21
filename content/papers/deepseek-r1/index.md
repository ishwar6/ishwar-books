---
title: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
short: DeepSeek-R1
description: "The January 2025 paper that taught a language model to reason with reinforcement learning and two simple rules for reward. Eight parts, with the paper's own text, worked equations, GRPO written from scratch in Python, and real runs on the released distilled model."
authors: [Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, and others]
org: DeepSeek-AI
year: 2025
venue: arXiv 2025 (Nature 2025)
arxiv: "2501.12948"
code: https://github.com/deepseek-ai/DeepSeek-R1
accent: "#4d6bfe"
date: 2026-11-21
parts: 8
tags: [deepseek, reasoning, reinforcement-learning, grpo, llm]
learn:
  - "What a reasoning model is, and why longer thinking at answer time gives better answers"
  - "How GRPO works: sample a group of answers, score them, and push the model toward the better ones without a critic model"
  - "Why two rule-based rewards (is the answer right, is the format right) were enough, and why the authors avoided a neural reward model"
  - "How DeepSeek-R1-Zero learned to think longer and to check its own work, and what the aha moment really shows"
  - "The four stages that turn R1-Zero's raw skill into DeepSeek-R1, a model people can read and use"
  - "How 800,000 examples from R1 teach a small model to reason, and why distillation beat RL on the same small model"
  - "What the results tables show, what failed (process rewards and tree search), and what R1 still cannot do"
  - "How to write GRPO yourself in Python and train a small model to reason on your own laptop"
---

## Why this paper

Large language models were good at writing, but weak at hard multi-step problems such as competition maths. In 2024 one company showed that a model gets much better when it thinks for a long time before it answers. It did not say how it trained that model.

This paper shows one way, in the open. The authors took a base model and gave it only reinforcement learning, with two simple rules for reward: is the final answer right, and is the thinking inside the right tags. No human wrote an example of good reasoning. The model learned to think longer, to check its own steps and to try a new approach when the first one failed. The paper then turns this raw skill into DeepSeek-R1, a model that matched the best closed reasoning model of the time, and releases the weights.

The paper also shows that the reasoning of a big model can be copied into small models with plain fine-tuning. A 1.5-billion-parameter model trained this way runs on a laptop, and we use it in this series.

## How the eight parts are organised

**Before Part 1 (optional): [The Road to R1](background-the-road-to-r1.md).** How language models are trained, chain-of-thought, reinforcement learning from human feedback, PPO, and thinking at answer time.

Each part follows the paper's own sections:

1. **[The Big Idea](part-1-the-big-idea.md)**: the title, the abstract, Figure 1 and §1. Reasoning by reinforcement learning alone, and the three things the paper releases.
2. **[GRPO](part-2-grpo.md)**: §2.1 and §2.2.1, Equations 1 to 3. The learning algorithm, term by term, written from scratch and checked against a library.
3. **[Rewards and the Template](part-3-rewards-and-the-template.md)**: §2.2.2, §2.2.3 and Table 1. The accuracy reward, the format reward, and the one-paragraph prompt that starts it all.
4. **[R1-Zero and the Aha Moment](part-4-r1-zero-and-the-aha-moment.md)**: §2.2.4, Table 2, Figures 2 and 3, Table 3. What pure RL achieved, why answers got longer, and the moment the model said "Wait".
5. **[The R1 Pipeline](part-5-the-r1-pipeline.md)**: §2.3. Cold-start data, reasoning RL with a language reward, rejection sampling, and the last RL stage for all tasks.
6. **[Distillation](part-6-distillation.md)**: §2.4, §3.2, §4.1, Tables 5 and 6. Teaching small models with R1's answers, and why this beat RL on the small model.
7. **[Results, Failures and Limits](part-7-results-failures-and-limits.md)**: §3, §3.1, §4.2, §5 and Table 4. How the models are scored, what the results show, what did not work, and what is still open.
8. **[Train a Reasoner from Scratch](part-8-train-a-reasoner-from-scratch.md)**: the whole method in one short Python package. We write GRPO and the rewards ourselves and train a small model to reason on a laptop.

## How to read the boxes

- A **teal "From the paper" box** shows the exact lines of the paper as a highlighted screenshot. Click the screenshot to see it at full size. Under it, **What it says** gives the main words of the paper. **What it means** then explains those lines, and what the authors mean but do not write. When the box shows a different paper, the box gives the name of that paper.
- A **yellow definition box** explains a technical word the first time it occurs. If you know the word, skip the box.
- The small **Paper §2.2** tag above a heading tells you which section of the paper you read.
- A **green "Key takeaways" box** ends each part with the facts to remember.

Code blocks show real code, and the output under a code block is what that code printed. The scripts are in [`code/papers/deepseek-r1`](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/deepseek-r1).

We follow the first version of the paper (arXiv v1, 22 January 2025), the version most people read. A longer version came out later with the Nature article; where it adds a useful fact, the part says so.

## What you need to know first

You need to know what a language model does: it reads text and predicts the next token. Every other term is explained when it first occurs. The [Transformer series](/papers/transformer/) explains the model inside, and [DeepSeek, from the Ground Up](/writings/deepseek-1-every-model-and-what-is-open/) places R1 among DeepSeek's other models.
