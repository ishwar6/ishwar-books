# Study Plan

Three versions. Pick one and put it in your calendar.

- **[The 4-week plan](#the-4-week-plan)**: ~10–12 h/week. The main path; finishes the book with exercises.
- **[The 5-day interview sprint](#the-5-day-interview-sprint)**: you have an interview soon.

> Reading without running teaches vocabulary. Running without recording teaches nothing
> durable. Keep a `notes.md`: for every evaluation, write the config, the numbers, and your
> explanation of why they moved. That file is what you talk about in the interview.

## Before day 1

1. [SETUP.md](01-setup.md): `uv sync`, key in `.env`, `uv run python code/ch03/first_rag.py` works.
2. Read [`ragbook/common.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/ragbook/common.py) and [`ragbook/index.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/ragbook/index.py) (10 minutes). Every chapter imports them.
3. Skim [`data/handbook/`](https://github.com/ishwar6/ishwar-books/tree/main/code/rag/data/handbook) so you know what the corpus contains: you will judge answers against it.

## The 4-week plan

### Week 1: Foundations and a real pipeline (Part I, Part II)
- **Read:** Ch 1, 2, 3, 4, 5, 6, 7.
- **Do:** all "Run it" sections; exercises in Ch 2, 3, 5, 6.
- **Deliverable:** the chunking comparison table from Ch 5 for *your* run, plus one sentence per
  strategy on why it won or lost.
- **Checkpoint:** you can draw the ingestion and query loops from memory, explain cosine
  similarity, and say what a payload filter is and why it needs an index.

### Week 2: Retrieval and evaluation (Part III, Ch 10)
- **Read:** Ch 8, 9, 10.
- **Do:** run dense vs BM25 vs hybrid; run [`code/ch10/run_eval.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch10/run_eval.py) with at least three configs
  (change k, turn hybrid on, turn reranking on).
- **Deliverable:** an eval table with recall@5, MRR, nDCG@5, faithfulness, correctness and cost
  per question for each config, and a paragraph on the trade-offs.
- **Checkpoint:** you can define every metric in Appendix B without looking, and for each say
  what you would change if it were low.

### Week 3: Quality and agents (Ch 11–16)
- **Read:** Ch 11, 12, 13, 14, 15, 16.
- **Do:** make the Ch 11 faithfulness gate fail on purpose and watch it abstain; calibrate the
  Ch 12 threshold on the golden set; build the Ch 15 agentic graph and count LLM calls on a
  multi-hop question vs the Ch 6 pipeline; break the Ch 16 multi-agent loop, then fix it.
- **Deliverable:** a one-page "agentic vs single-pass" comparison on the golden set: accuracy,
  calls, cost, latency.
- **Checkpoint:** you can explain state/nodes/edges/reducers, why the loop needs a cap, and three
  ways to enforce retrieval-first behaviour.

### Week 4: Production, scale, frontier, interview (Ch 17–20, appendices)
- **Read:** Ch 17, 18, 19, 20; Appendix A tiers 1–3.
- **Do:** run the FastAPI service and hit it; run `scale_math.py` and sanity-check its numbers by
  hand; create a quantized collection and compare recall with rescoring on/off.
- **Deliverable:** your answer, written out, to "design a RAG system for 10M documents and
  1,000 QPS" (Ch 20 has a model answer: write yours first).
- **Checkpoint:** you can do the 1 GB → 10 GB → 1 TB arithmetic on a whiteboard.

## The 5-day interview sprint

Each day is ~4 hours. Read the chapter, run its main script once, then read its interview
questions out loud and answer before looking.

| Day | Read | Run | Say out loud |
|---|---|---|---|
| 1 | Ch 1, 3, 4, 5 | `ch03/first_rag.py`, `ch05/chunking_lab.py` | end-to-end RAG; why embeddings; chunking for a PDF with images |
| 2 | Ch 8, 9, 10, App. B | `ch08/hybrid_qdrant.py`, `ch10/run_eval.py --limit 15` | recall vs precision; faithfulness; how you evaluate; how you ensure quality |
| 3 | Ch 11, 12, 13 | `ch11/faithfulness_gate.py`, `ch12/coverage_report.py` | preventing hallucination; no relevant docs; debugging a bad answer |
| 4 | Ch 14, 15, 16 | `ch15/agentic_rag_graph.py`, `ch16/planner_retriever_analyst.py` | agentic vs generative RAG; agents looping; RAG-first; MCP vs REST; failing agent |
| 5 | Ch 17, 18, 19, 20 | `ch17/app.py` + `client_demo.py`, `ch18/scale_math.py` | async vs sync; Dockerfile vs Compose; 1 GB → 10 GB; is RAG dead; the 10M-docs design |

Then read Appendix A's "10 papers if you only have time for 10" the night before.
