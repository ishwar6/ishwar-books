---
title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
short: BERT
description: "The 2018 paper that taught one model to read text in both directions at once, then reuse that skill for almost any language task. Explained line by line in six parts, with the paper's own text, simple pictures and real code."
authors: [Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova]
org: Google AI Language
year: 2018
venue: NAACL 2019
arxiv: "1810.04805"
code: https://github.com/google-research/bert
date: 2026-10-04
tags: [bert, transformers, nlp, pre-training]
learn:
  - "Why reading in both directions matters, and why older models could not do it"
  - "How BERT turns text into numbers: WordPiece tokens, [CLS], [SEP], segment and position embeddings"
  - "The two pre-training tasks, masked language modelling and next sentence prediction, run on the real model"
  - "How one pre-trained model is fine-tuned for classification, question answering and more"
  - "What the ablation studies prove, and what later work (RoBERTa, ALBERT, DistilBERT, ELECTRA) changed"
---

## Why this paper

Before BERT, a model that understood language had to be built again for almost every task. BERT showed a simpler way: teach one model how language works by letting it fill in hidden words in billions of words of ordinary text, then adapt that same model to each task by adding one small layer. It set new best results on eleven tasks at once, won the Best Long Paper award at NAACL 2019, and within a year Google wrote that it was using BERT to understand searches. Most of the ideas you meet in modern language models (pre-training, fine-tuning, `[CLS]` vectors, sentence embeddings) are easiest to understand here, where they first came together.

The paper is short (nine pages plus an appendix), but dense. This breakdown reads it slowly, in its own order, and checks its claims with real code on the model the authors released.

## How the six parts are organised

Each part follows the paper's own sections:

1. **[The Big Idea: Reading in Both Directions](part-1-the-big-idea.md)**: the title, the Abstract and §1. Why one-directional language models are a limit, and BERT's fix.
2. **[Inside BERT: Architecture and Input](part-2-architecture-and-input.md)**: §2 and §3. The related work, the model's size and shape, and how text becomes the numbers BERT reads.
3. **[Pre-training: Masked LM and Next Sentence Prediction](part-3-pre-training.md)**: §3.1, Appendix A.1 and A.2. The two pre-training tasks, the data and the training recipe.
4. **[Fine-tuning and Results: GLUE, SQuAD and SWAG](part-4-fine-tuning-and-results.md)**: §3.2, §4, Appendix A.3, A.5 and B.1. How one model is adapted to each task, and every result table.
5. **[Ablations: What Really Matters](part-5-ablations.md)**: §5, Appendix A.4, C.1 and C.2. The experiments that take BERT apart to see which pieces matter.
6. **[After BERT: Impact, Limits and Summary](part-6-impact-and-summary.md)**: §6 and later work. What came next, what BERT cannot do, and the whole paper on one page.

## How to read the boxes

- A **teal "From the paper" box** shows the exact lines of the paper as a highlighted screenshot (click it to open it full size), followed by its **context**, **what it says** in plain English, and **why it matters**.
- A **yellow definition box** explains a technical word the first time it appears. If you already know the word, skip the box.
- The small **Paper §3.1** tag above a heading tells you which section of the paper you are reading, so you can follow along in the PDF.
- A **green "Key takeaways" box** ends every part with what to remember.

Code blocks show real code, and the output shown under them is what that code actually printed. The scripts are in [`code/papers/bert`](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/bert).

## What you need to know first

Nothing. Every term is explained when it first appears. BERT is built on **attention**, the mechanism that lets each word look at the other words; the paper only needs the idea, but if you want to see how attention works inside, the [attention series](/writings/attention-1-self-attention/) explains it from zero.
