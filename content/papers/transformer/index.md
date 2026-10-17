---
title: "Attention Is All You Need"
short: Transformer
description: "The 2017 paper that replaced recurrence with attention and gave us the Transformer, the design behind every large language model today. Six parts, with the paper's own text, worked equations and real code on a released translation model."
authors: [Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin]
org: Google Brain, Google Research, University of Toronto
year: 2017
venue: NeurIPS 2017
arxiv: "1706.03762"
code: https://github.com/tensorflow/tensor2tensor
accent: "#e0a458"
date: 2026-10-18
parts: 6
tags: [transformer, attention, machine-translation, deep-learning, nlp]
learn:
  - "Why recurrent networks were slow to train, and how attention removes the step-by-step loop"
  - "The encoder and the decoder, with every shape of the base model"
  - "Scaled dot-product attention and multi-head attention, worked by hand and on real weights"
  - "Why the paper divides by the square root of the key size, and what the mask in the decoder does"
  - "Sine and cosine position encodings, and why they let the model read word order"
  - "The training recipe: the learning-rate warm-up, dropout and label smoothing"
  - "What the results and the model variations show, and what the paper started"
---

## Why this paper

Before 2017, the best translation systems read a sentence one word at a time. Each step waited for the step before it, so training could not use the full power of a GPU. This paper removed that loop. Its model, the **Transformer**, lets every word look at every other word in one step, through a mechanism called **attention**.

The Transformer translated better than the systems before it and trained in a fraction of the time. The same design now runs every large language model, many image models and speech models.

## How the six parts are organised

**Before Part 1 (optional): [The Road to the Transformer](background-the-road-to-the-transformer.md).** Recurrent networks, encoder-decoder translation, and the first attention mechanism.

Each part follows the paper's own sections:

1. **[The Big Idea](part-1-the-big-idea.md)**: the title, the abstract, §1 and §2. What was wrong with recurrence, and the claim that attention alone is enough.
2. **[The Architecture](part-2-the-architecture.md)**: §3, Figure 1, §3.1, §3.3, §3.4 and §3.5. The encoder and decoder stacks, residual connections, layer normalisation, the feed-forward network, embeddings and position encodings.
3. **[Attention](part-3-attention.md)**: §3.2 and Figure 2. Scaled dot-product attention, multi-head attention, masking, and the three ways the model uses attention.
4. **[Why Self-Attention](part-4-why-self-attention.md)**: §4 and Table 1. Cost per layer, steps in sequence and path length, against recurrent and convolutional layers.
5. **[Training](part-5-training.md)**: §5. The data, batching, hardware, the learning-rate schedule, dropout and label smoothing.
6. **[Results, Variations and Impact](part-6-results-and-impact.md)**: §6, §7, Tables 2 to 4 and the attention pictures of the appendix. What the paper showed, and what came after.

## How to read the boxes

- A **teal "From the paper" box** shows the exact lines of the paper as a highlighted screenshot. Click the screenshot to see it at full size. Under it, **What it says** gives the main words of the paper. **What it means** then explains those lines, and what the authors mean but do not write. When the box shows a different paper, the box gives the name of that paper.
- A **yellow definition box** explains a technical word the first time it occurs. If you know the word, skip the box.
- The small **Paper §3** tag above a heading tells you which section of the paper you read.
- A **green "Key takeaways" box** ends each part with the facts to remember.

Code blocks show real code, and the output under a code block is what that code printed. The scripts are in [`code/papers/transformer`](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/transformer).

## What you need to know first

You need to know what a vector and a matrix are, and how to multiply them. Every other term is explained when it first occurs. The [attention series](/writings/attention-1-self-attention/) explains self-attention from zero, and is a good companion to Part 3.
