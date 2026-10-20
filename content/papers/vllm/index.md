---
title: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
short: vLLM
description: "The 2023 paper behind vLLM: it stores the memory of each chat request in small blocks, the way an operating system stores memory in pages, and serves 2 to 4 times more requests on the same GPU. Read with the paper's own text and real code."
authors: [Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica]
org: UC Berkeley, Stanford, UC San Diego
year: 2023
venue: SOSP 2023
arxiv: "2309.06180"
code: https://github.com/vllm-project/vllm
accent: "#5b9bd5"
date: 2026-10-19
parts: 6
tags: [llm-serving, inference, kv-cache, gpu-memory, systems]
learn:
  - "Why a language model on a GPU is limited by memory, not by arithmetic, when it writes text"
  - "What the KV cache is, how large it gets, and why it grows and shrinks with every request"
  - "Why storing each request's cache in one contiguous chunk wastes 60% to 80% of the memory"
  - "PagedAttention: attention over keys and values stored in small blocks, as an operating system stores pages"
  - "How vLLM shares blocks between samples, schedules and preempts requests, and runs on many GPUs"
  - "What the experiments show: 2 to 4 times the throughput of earlier systems at the same latency"
  - "How to build a small paged KV cache and serving loop yourself, in Python"
---

## Why this paper

A chatbot answers one word at a time. To write the next word, the model looks back at every earlier word of the conversation, so it keeps a record of them in GPU memory: the **KV cache**. Thousands of people send requests at the same time, and the GPU can serve many of them together only if their caches fit in memory.

This paper shows that the serving systems of 2023 wasted most of that memory. They reserved one large chunk per request, sized for the longest possible answer. The authors borrow an old idea from operating systems: cut memory into small pages and hand them out on demand. The result is **PagedAttention** and the serving engine **vLLM**, which serves 2 to 4 times more requests on the same hardware. vLLM became one of the most used open-source engines for serving language models.

## How the six parts are organised

**Before Part 1 (optional): [The KV Cache and Paging](background-the-kv-cache-and-paging.md).** How attention uses keys and values, and how an operating system gives each program its own pages of memory.

1. **[The Big Idea](part-1-the-big-idea.md)**: the title, the abstract, Figures 1 and 2, and §1. Where the memory of a GPU goes, and why most of the KV cache memory was wasted.
2. **[Why Memory Is the Bottleneck](part-2-why-memory-is-the-bottleneck.md)**: §2 and §3, Figure 3. The two phases of generation, batching, and the three kinds of waste.
3. **PagedAttention** (coming soon): §4.1 to §4.3, Figures 4 to 7. Attention over blocks, the block table, and the KV cache manager.
4. **Sharing, Scheduling and Many GPUs** (coming soon): §4.4 to §4.6 and §5, Figures 8 to 10. Copy-on-write, beam search, preemption and swapping, and distributed execution.
5. **The Experiments** (coming soon): §6 to §10. Throughput against latency, sharing, the ablations, and the limits.
6. **Build It from Scratch** (coming soon): a small paged KV cache and serving loop in Python, checked against a library model.

## How to read the boxes

- A **teal "From the paper" box** shows the exact lines of the paper as a highlighted screenshot. Click the screenshot to see it at full size. Under it, **What it says** gives the main words of the paper. **What it means** then explains those lines, and what the authors mean but do not write.
- A **yellow definition box** explains a technical word the first time it occurs. If you know the word, skip the box.
- The small **Paper §1** tag above a heading tells you which section of the paper you read.
- A **green "Key takeaways" box** ends each part with the facts to remember.

Code blocks show real code, and the output under a code block is what that code printed. The scripts are in [`code/papers/vllm`](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/vllm). They run on a laptop with a small OPT model. The paper used NVIDIA A100 GPUs, and each part says what that changes.

## What you need to know first

You need to know what a language model is and what a token is. Some Python helps. The [Transformer series](/papers/transformer/) explains attention, keys and values in depth, but each part defines what it uses.
