---
title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
short: ViT
description: "The 2020 paper that cut an image into 16×16 patches, fed them to an unchanged Transformer, and beat the best convolutional networks once it was given enough data. Explained line by line in six parts, with the paper's own text, simple pictures, worked equations and real code on the released models."
authors: [Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby]
org: Google Research, Brain Team
year: 2020
venue: ICLR 2021
arxiv: "2010.11929"
code: https://github.com/google-research/vision_transformer
accent: "#e8a33d"
date: 2026-10-06
parts: 6
tags: [vit, vision-transformer, transformers, computer-vision, pre-training]
learn:
  - "Why convolutional networks ruled computer vision for a decade, and what 'inductive bias' means"
  - "How an image becomes a sequence of tokens: patches, the flattening maths, the linear projection, the [class] token and position embeddings"
  - "Every equation of the model (Eq. 1 to 8) with real numbers from the released ViT-B/16 checkpoint"
  - "How ViT is fine-tuned at a higher resolution, including the 2D interpolation of position embeddings, worked by hand"
  - "What the results prove: large-scale training beats inductive bias, and ViT costs 2 to 4 times less compute than ResNets for the same accuracy"
  - "What the model learns inside: embedding filters, position-embedding similarity, attention distance and attention maps, recomputed on the real model"
  - "What came after: DeiT, Swin, BEiT, MAE, DINO, CLIP and the scaling of ViT to 22 billion parameters"
---

## Why this paper

For ten years, computer vision belonged to convolutional neural networks. Every image model, from AlexNet in 2012 to the big ResNets of 2020, was built around the convolution, a small filter slid across the picture. Meanwhile, language models had switched to the Transformer, a design with no convolutions at all, and were growing faster than anyone expected.

This paper asked a blunt question: what happens if you take the Transformer from language, change almost nothing, and feed it an image cut into 16×16 pixel squares, as if each square were a word? The answer surprised the field. On ordinary amounts of data the Transformer lost to ResNets. On very large amounts of data (14 million to 300 million images) it won, and did so with less training compute. The paper's own summary is five words: "large scale training trumps inductive bias".

The Vision Transformer (ViT) became the base of most of today's image models and of the image side of multimodal models. The paper is also unusually well written and short (nine pages plus an appendix), which makes it a good paper to read in full.

## How the six parts are organised

Each part follows the paper's own sections:

1. **[The Big Idea: An Image Is Worth 16×16 Words](part-1-the-big-idea.md)**: the title, the Abstract and §1. Why convolutions dominated, what a patch is, and the claim that data beats built-in assumptions.
2. **Inside ViT: Patches, Embeddings and the Encoder** (coming soon): §2, §3.1, Figure 1, Equations 1 to 4 and Appendix A. The related work, and every step from pixels to a class prediction, with the real model's numbers.
3. **Fine-tuning, Higher Resolution and the Setup** (coming soon): §3.2, §4.1, Tables 1, 3 and 4, Appendix B.1. How a pre-trained ViT is adapted, the datasets, the baselines, and the training recipe.
4. **Results: Beating Big CNNs, and How Much Data It Takes** (coming soon): §4.2, §4.3, Table 2, Figures 2 to 4, Table 5, Appendix D.1, D.9 and D.10. Every result table and what it proves.
5. **Scaling, Looking Inside ViT, and Self-supervision** (coming soon): §4.4 to §4.6, Figures 5 to 7, Table 6, Appendix B.1.2 and D.2 to D.8. Compute versus accuracy, what the model learns, and masked patch prediction.
6. **After ViT: Impact, Limits and Summary** (coming soon): §5 and later work. What came next, what ViT cannot do, and the whole paper on one page.

## How to read the boxes

- A **teal "From the paper" box** shows the exact lines of the paper as a highlighted screenshot (click it to open it full size), followed by its **context**, **what it says** in plain English, and **why it matters**.
- A **yellow definition box** explains a technical word the first time it appears. If you already know the word, skip the box.
- The small **Paper §3.1** tag above a heading tells you which section of the paper you are reading, so you can follow along in the PDF.
- A **green "Key takeaways" box** ends every part with what to remember.

Code blocks show real code, and the output shown under them is what that code actually printed. The scripts are in [`code/papers/vit`](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/vit). They run on the models Google released, through the Hugging Face `transformers` library, on a laptop.

## What you need to know first

Nothing about computer vision. Every term is explained when it first appears. ViT is a Transformer **encoder**, the same design as [BERT](/papers/bert/), and the paper says so on almost every page; if you have read the BERT breakdown, Part 2 will feel familiar. If attention itself is new to you, the [attention series](/writings/attention-1-self-attention/) explains it from zero, but the paper only needs the idea: each patch looks at every other patch.
