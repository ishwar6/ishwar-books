---
title: "A Mathematical Framework for Transformer Circuits"
short: Circuits
description: "The 2021 paper that started reverse-engineering transformers by hand. It writes a small attention-only model as a sum of simple pieces you can read straight from the weights: bigram tables, skip-trigrams and induction heads. Every equation is derived step by step, next to the paper's own figures and real code on small open models."
authors: [Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, Chris Olah]
org: Anthropic
year: 2021
venue: Transformer Circuits Thread
url: https://transformer-circuits.pub/2021/framework/index.html
accent: "#a77bf3"
date: 2026-10-12
parts: 7
tags: [interpretability, mechanistic-interpretability, transformers, circuits, attention]
learn:
  - "What mechanistic interpretability is, and why the paper compares it to reverse-engineering a compiled program"
  - "Why the paper studies tiny attention-only models, and what dropping MLPs, biases and layer norm costs"
  - "The whole model in four equations, with every symbol, every shape and a worked example by hand"
  - "The residual stream as a shared channel: every layer reads with a linear map and writes by adding"
  - "Virtual weights, subspaces and the bandwidth problem, with real numbers from small open models"
  - "Why a zero-layer transformer can only learn bigrams, derived step by step, and the bigram table read straight from real weights"
  - "How one attention head splits into a QK circuit and an OV circuit (Part 2), and how heads compose into induction heads (Parts 5 and 6)"
---

## Why this paper

Large language models work, but nobody wrote down the rules they follow. The rules are hidden inside millions or billions of learned numbers. In December 2021 a team at Anthropic published a long web article that asked: can we read those rules out of the numbers, the way a programmer reads a compiled program back into source code?

They started small. They took transformers with zero, one and two layers, kept only the attention part, and showed that each model can be written as a **sum of simple pieces**. Each piece can be read directly from the weights, without running the model. A zero-layer model is a table of word pairs (bigrams). A one-layer model adds "skip-trigrams" and a simple kind of copying. A two-layer model builds **induction heads**, a small circuit that lets a model continue a pattern it has seen earlier in the text. Induction heads became one of the best-known findings in interpretability.

The paper also gave the field its vocabulary: the **residual stream**, **virtual weights**, the **QK circuit** and the **OV circuit**, **path expansion**. Most later work on reading the inside of language models uses these words. The paper is long and dense, so this series reads it slowly, in its own order, and derives every equation by hand.

## How the parts are organised

1. **[The Big Idea and the Residual Stream](part-1-the-residual-stream.md)**: the introduction, the model simplifications, the model written as four equations, the residual stream as a communication channel, virtual weights, bandwidth, and the zero-layer transformer.
2. **[One Attention Head, Taken Apart: the QK and OV Circuits](part-2-one-attention-head.md)**: why heads are independent and additive, attention as information movement, the tensor-product form of a head, the two low-rank matrices $${W_{\mathrm{Q}}}^{\top} W_{\mathrm{K}}$$ and $$W_{\mathrm{O}} W_{\mathrm{V}}$$, and the QK and OV circuits read from real weights.
3. **[Paths Through the Model](part-3-paths-through-the-model.md)**: the path expansion trick, which writes a whole model as a sum of simple end-to-end terms.
4. **[One-Layer Models: Skip-Trigrams, Copying and Their Bugs](part-4-one-layer-models.md)**: reading the bigram and skip-trigram tables from the weights of a one-layer model.
5. **[Two-Layer Models: How Heads Compose](part-5-two-layer-models.md)**: Q-, K- and V-composition, and virtual attention heads.
6. **[Induction Heads](part-6-induction-heads.md)**: the two-head circuit behind in-context copying, and how the paper checks it.
7. **What the Paper Left Out, and What Came After** (coming soon): MLPs, superposition, dictionary learning and circuit tracing.

## How to read the boxes

- A **teal "From the paper" box** shows the exact words of the paper as a highlighted screenshot (click it to open it full size), followed by **what it says**, the main words of the paper, and **what it means**, an explanation of those lines and of what the authors mean but do not write.
- A **yellow definition box** explains a technical word the first time it appears. If you already know the word, skip the box.
- The small tag above a heading names the section of the paper you are reading, so you can follow along on the paper's web page.
- A **green "Key takeaways" box** ends every part with what to remember.

Code blocks show real code, and the output under them is what that code printed. The scripts are in [`code/papers/circuits`](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/circuits). They use the open-source TransformerLens library and small public models, and they run on the CPU of an Apple M5 Pro (64 GB) in under a minute.

## What you need to know first

You need to know what a vector and a matrix are, and how to multiply a matrix by a vector. Everything else is explained when it first appears. It helps to know roughly what a transformer is and what attention does; the [attention series](/writings/attention-1-self-attention/) explains both from zero. Part 1 needs almost none of it: attention itself is opened up in Part 2.
