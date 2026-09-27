---
title: "RAG: From First Principles"
subtitle: "To terabyte-scale agentic systems"
description: "Retrieval-augmented generation built and measured step by step. Embeddings, chunking, hybrid search, reranking, evaluation, hallucination, agentic RAG with LangGraph, and ingesting a million PDFs."
status: complete
accent: "#a78bfa"
order: 1
topics: [Embeddings, Qdrant, Hybrid Search, Rerankers, Evaluation, Agentic RAG]
---

A complete, self-contained course on Retrieval-Augmented Generation, written to be read start to finish, in order, with code you run and numbers you measure at every step. One learning path, one corpus, one stack.

**Stack (current as of September 2026):** Python 3.12 · LangChain 1.4 · LangGraph 1.2 · OpenAI (chat and embeddings) · Qdrant 1.19 · FastAPI. Chapters 1 to 13 use LangChain only; LangGraph enters in Chapter 14, FastAPI and Docker in Chapter 17. Every API was checked against the official docs, so the old `RetrievalQA` / `create_react_agent` / `MemorySaver` world is not used anywhere (see [Appendix D](appendix/D-api-cheatsheet.md) for the old to new table).

**What this is:** the curriculum for someone who has to *build and defend* a RAG system. Every chapter ends with the questions engineers are actually asked about it, with answers.

**What this is not:** an API tour. You will learn why each piece exists, how to measure whether it works, and what to change when it does not.

## The one idea that organizes everything

A RAG system is only as good as the **worst of three stages**, and you cannot fix a stage you have not measured:

| Stage | Question it answers | Measured by | Chapters |
|---|---|---|---|
| **Ingestion** | Did the right facts end up as retrievable chunks? | chunk stats, coverage | 5, 6, 7, 12 |
| **Retrieval** | Given a question, did the right chunks come back, near the top? | recall@k, precision@k, MRR, nDCG | 4, 8, 9, 10 |
| **Generation** | Did the model answer *from* those chunks, correctly, and admit when it can't? | faithfulness, correctness, refusal accuracy | 3, 10, 11, 12 |

Everything in this book is either how to measure one of these, or what to do about it. Agentic RAG (Part 5) is what you build when a single pass through the three stages is not enough and the system has to decide to retrieve again, differently, or not at all.

Parts I to VII build the system. **Part VIII (Chapters 21 to 30) is the layer underneath:** HNSW built from scratch, rank fusion derived, rerankers measured, and hallucination causes injected on purpose, so you can explain *why* each component works and not only *how* to wire it up.

## Before you start

- **Prerequisites.** Python at the level of "I can write a function, a class, and read a stack trace." No ML background needed; Chapter 1 gives you the vocabulary.
- **Setup.** [Five minutes](00-start-here/01-setup.md): `uv sync`, put your OpenAI key in `.env`, run one script. Qdrant runs embedded (no Docker needed) until Chapter 17.
- **Cost.** The whole book, run once end to end, costs roughly **$1 to $3** in OpenAI usage with the default models. Evaluation loops are the expensive part; they have `--limit` flags.
- **Time.** About 40 to 60 hours including exercises. The [study plan](00-start-here/02-study-plan.md) has a 4-week plan and a 5-day sprint.
- **Corpus.** Every chapter uses the same data: the internal handbook of a fictional robotics company (14 documents) and a golden set of 47 questions with reference answers, including 5 the corpus *cannot* answer and one deliberate contradiction. See [the corpus](00-start-here/03-the-corpus.md).

> [!TIP] Get the code
> All 82 chapter scripts, the shared `ragbook` helpers and the corpus live in [`code/rag`](https://github.com/ishwar6/ishwar-books/tree/main/code/rag) in this site's repository. Clone it, `cd code/rag`, and every command in the book runs as written.

## How to read this book

- **In order.** Each chapter's code assumes the previous chapter's ideas, and the corpus, golden set and helpers are introduced progressively.
- **Run everything.** Each chapter has a "Run it" section with the exact command and the output you should see. The numbers you measure are the course.
- **Keep a `notes.md`.** For every evaluation you run, record the config, the numbers, and why you think they moved.
