---
title: GPU Programming
subtitle: "From zero to kernel engineer"
description: "A complete course that teaches you to predict how fast a kernel can go, measure how fast it does go, and close the gap. From CPU parallelism to Hopper, Blackwell, FlashAttention and multi-GPU inference."
status: in-progress
accent: "#76b900"
order: 1
chapters: 32
topics: [CUDA, Triton, Tensor Cores, FlashAttention, Profiling, Multi-GPU]
---

A GPU kernel is fast when it keeps the machine's *bottleneck resource* busy. There are only a handful of candidates (memory bandwidth, memory latency, compute throughput, instruction issue, synchronisation, launch overhead and interconnect), and this book teaches you to identify which one you are fighting and what to do about it.

Every chapter ends with the kind of interview questions asked for kernel and performance roles, with answers.

> [!NOTE] Part I is published
> Chapters 1 to 5 (Foundations) are available now: CPU parallelism, why GPUs exist, the hardware, the programming model and the toolchain. Later parts are being edited and will appear here as they are ready.

## Running the code

Chapter 1 runs on any laptop CPU. From Chapter 2 on you need an NVIDIA GPU; a free Google Colab T4 is enough. The code lives in [`code/gpu`](https://github.com/ishwar6/ishwar-books/tree/main/code/gpu) in this site's repository.
