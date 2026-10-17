---
title: "Segment Anything"
short: SAM
description: "The 2023 paper that made one model cut out any object in any picture from a click, a box or a rough mask. It shows a new task, the model SAM and a dataset of 1.1 billion masks. Six parts, with the paper's own text, simple pictures, worked equations and real code on the released models."
authors: [Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick]
org: Meta AI Research, FAIR
year: 2023
venue: ICCV 2023
arxiv: "2304.02643"
code: https://github.com/facebookresearch/segment-anything
accent: "#5bb8a6"
date: 2026-10-15
parts: 6
tags: [sam, segmentation, foundation-models, computer-vision, vision-transformer]
learn:
  - "What segmentation is, what a mask is, and how to measure if two masks agree (a number called IoU)"
  - "What a foundation model is, and how the paper moves the idea of a prompt from language to segmentation"
  - "Why one click is ambiguous, and how SAM gives three masks and a score for each one"
  - "How the image encoder, the prompt encoder and the mask decoder work, with the real shapes and numbers of the released models"
  - "How the data engine used SAM to label data that trained a better SAM, until it made 1.1 billion masks with no people"
  - "What the zero-shot experiments show on 23 datasets, edge detection, object proposals, instance segmentation and text prompts"
  - "What SAM cannot do, and what came after it"
---

## Why this paper

Segmentation means to find the exact pixels of an object in a picture. Before 2023, a segmentation model usually knew a fixed list of object types. It also worked well only on pictures like its training pictures. For a new kind of object or a new kind of picture, you collected new labels and trained a new model.

This paper asks a different question. Can one model cut out *any* object, in *any* picture, when you only show it *which* object you want? You show the object with a click, a box or a rough mask. The model returns a mask. The paper calls this a **prompt**, the same word that language models use.

The paper has three linked parts. A new task: promptable segmentation. A new model: the Segment Anything Model (SAM). A new dataset: SA-1B, with 1.1 billion masks on 11 million pictures. SAM helped people make the dataset, and the dataset trained SAM. SAM is now a standard tool in labelling software, in medical imaging research and in robotics.

## How the six parts are organised

**Before Part 1 (optional): [The Road to SAM](background-the-road-to-sam.md).** How FCN, YOLO and Mask R-CNN cut out objects from a fixed list, how GPT-3 and CLIP brought the prompt, and how YOLO and SAM differ on the same photo.

Each part follows the paper's own sections:

1. **[The Big Idea: Segment Anything](part-1-the-big-idea.md)**: the title, Figure 1, the abstract and §1. What a mask is, what a foundation model is, and the three parts of the project. With a real run of SAM on one picture.
2. **[The Task: Promptable Segmentation](part-2-the-task.md)**: §2 and Figure 3. Prompts, valid masks, ambiguity, pre-training and zero-shot transfer.
3. **[The Model: Encoder, Prompts and Decoder](part-3-the-model.md)**: §3, Figure 4, Figure 14 and Appendix A. Every block of SAM with the real numbers of the released models, the losses and the training loop.
4. **The Data Engine and SA-1B** (coming soon): §4, §5, §6 and Appendices B and C. Three stages of labelling, the automatic mask generator, and the analysis of the dataset.
5. **Zero-Shot Experiments** (coming soon): §7 and Appendix D. Points to masks on 23 datasets, edges, object proposals, instance segmentation, text to mask, and the ablations.
6. **After SAM: Impact, Limits and Summary** (coming soon): §8 and the work that came later.

## How to read the boxes

- A **teal "From the paper" box** shows the exact lines of the paper as a highlighted screenshot. Click the screenshot to see it at full size. Under it, **What it says** gives the main words of the paper. **What it means** then explains those lines, and what the authors mean but do not write. When the box shows a different paper, the box gives the name of that paper.
- A **yellow definition box** explains a technical word the first time it occurs. If you know the word, skip the box.
- The small **Paper §3** tag above a heading tells you which section of the paper you read. Use it to find the text in the PDF.
- A **green "Key takeaways" box** ends each part with the facts to remember.

Code blocks show real code. The output under a code block is the output that the code printed. The scripts are in [`code/papers/sam`](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/sam). They use the models that Meta released, through the Hugging Face `transformers` library, on an Apple M5 Pro (64 GB).


## What you need to know first

You do not need to know about computer vision. Each term has an explanation when it first occurs. SAM's image encoder is a Vision Transformer (ViT). The [ViT breakdown](/papers/vit/) explains it from zero, but Part 1 and Part 2 do not need it. If attention is new to you, the [attention series](/writings/attention-1-self-attention/) starts from zero. For Part 3 you need only this idea: each token looks at the other tokens.
