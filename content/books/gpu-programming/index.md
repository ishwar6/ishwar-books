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

> [!NOTE]
> Chapters are being edited for publication and will appear here one by one.
