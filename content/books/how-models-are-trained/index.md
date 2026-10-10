---
title: "How Models Are Trained"
subtitle: "From next-word predictor to assistant: pretraining, SFT, RLHF, DPO and RL for reasoning"
description: "How a language model goes from predicting the next word to following instructions, holding a conversation and reasoning step by step. The history from 2017 to today, the research papers as highlighted screenshots, every equation explained symbol by symbol, and real code run on real models, in plain English for beginners."
status: in-progress
accent: "#7c8cff"
order: 3
chapters: 5
topics: [Pretraining, SFT, RLHF, DPO, GRPO, Evaluation]
---

A model that has only read the internet can finish your sentence, but it cannot answer your question. Something happens between the raw model and the assistant you talk to, and that something has a history, a set of equations and a lot of practical tricks. This book explains all of it, one step at a time.

It is written for beginners, in simple English. Every idea comes with the research paper that introduced it (shown as highlighted screenshots), a simple picture, the maths explained one symbol at a time with real numbers, and small pieces of real code that run on a laptop, with every line explained.

**Part 1, the big picture**, covers what training actually changes inside a model, how the methods began and evolved, and how anyone can tell whether a model got better.

**Part 2, from base model to instruction follower**, covers pretraining (the data, the compute arithmetic, scaling laws, and a tiny GPT trained from scratch on a laptop) and supervised fine-tuning (instruction datasets, chat templates, the masked loss, LoRA, and a real fine-tune of a small open model).
