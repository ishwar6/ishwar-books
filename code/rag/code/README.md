# Code

One directory per chapter, one runnable script per idea. Every script runs from the repo root:

```bash
uv run python code/ch03/first_rag.py
```

Scripts that need a model or a vector store import the shared helpers from
[`ragbook/`](../ragbook/) (`get_llm`, `get_embeddings`, `get_qdrant_client`, `load_handbook`,
`load_golden`, `build_handbook_index`). The pure-computation scripts deliberately import nothing
from `ragbook/` and make no API calls, so they run with no key and no cost: `ch10/metrics.py`,
`ch21/metric_illusions.py`, `ch21/significance.py`, `ch22/hnsw.py`, `ch22/ivf_and_pq.py`,
`ch22/recall_degradation.py`, `ch23/score_distributions.py`, `ch23/fusion_methods.py`,
`ch23/bm25_parameters.py` and `ch23/learned_sparse.py`.

Set `QDRANT_MODE=memory` to run scripts concurrently or throwaway; the default embedded on-disk
Qdrant allows one process at a time. Set `RAG_FAKE_EMBEDDINGS=1` for deterministic fake vectors
with no key and no cost: enough to prove a pipeline *runs*, never enough to judge its results.
`ch10/run_eval.py` accepts `--limit N`; `ch13/cost_and_latency.py` accepts `--n N`.

`scripts/validate_all.py` runs every script below in memory mode and reports pass/fail.

## What ships

| Script | Ch | What it demonstrates |
|---|---|---|
| `ch02/embeddings_playground.py` | 2 | embed sentences, cosine similarity by hand vs dot vs euclidean, nearest neighbours, Matryoshka dimensions, token cost |
| `ch03/first_rag.py` | 3 | the smallest complete RAG, LangChain only: load → chunk → InMemoryVectorStore → grounded prompt → cited answer |
| `ch04/qdrant_raw_client.py` | 4 | raw Qdrant client: create collection, upsert with payload, query_points, filters, count, scroll, delete |
| `ch04/qdrant_langchain.py` | 4 | the same through `QdrantVectorStore`: add_documents with ids, filtered search, retrievers (threshold, MMR) |
| `ch05/chunking_lab.py` | 5 | fixed / recursive / token / markdown-header / semantic chunking, chunk stats and hit-rate@5 on the golden set |
| `ch05/semantic_chunker.py` | 5 | a semantic chunker in ~40 lines: split where consecutive-sentence similarity drops |
| `ch06/ingest.py` | 6 | load → chunk → stable uuid5 ids → embed → upsert; idempotent re-ingest |
| `ch06/retrieve.py` | 6 | retrieval with k, metadata filter, score threshold; numbered context for citations |
| `ch06/generate.py` | 6 | prompt \| llm with citations; token usage and cost per answer |
| `ch06/rag_cli.py` | 6 | `rag_cli.py "question"` or `--interactive` |
| `ch07/download_gutenberg.py` | 7 | public-domain books as distractor data |
| `ch07/synthesize_docs.py` | 7 | LLM-generated handbook docs with Q/A pairs via structured output |
| `ch07/ingest_batch.py` | 7 | batched embedding, content-hash dedup, incremental manifest (add/update/delete), before/after hit rate |
| `ch08/bm25_from_scratch.py` | 8 | BM25 formula implemented by hand vs `rank_bm25` vs `BM25Retriever`; where lexical wins |
| `ch08/hybrid_qdrant.py` | 8 | hybrid three ways: RRF by hand, `RetrievalMode.HYBRID` with FastEmbed sparse vectors, raw `query_points` prefetch + RRF |
| `ch09/reranking.py` | 9 | retrieve 20 → rerank to 5 with an LLM judge and a cross-encoder |
| `ch09/query_transforms.py` | 9 | multi-query, HyDE, step-back, decomposition for multi-hop questions |
| `ch09/parent_child.py` | 9 | small-to-big: index small children, return the parent |
| `ch09/metadata_filtering.py` | 9 | self-query filter extraction, recency preference (FAQ vs policy conflict), MMR |
| `ch10/metrics.py` | 10 | hit rate, recall@k, precision@k, F1@k, MRR, MAP, nDCG@k, context precision/recall: each with its formula |
| `ch10/llm_judges.py` | 10 | faithfulness (claim-level), answer relevance, correctness, citation accuracy, refusal correctness |
| `ch10/run_eval.py` | 10 | one config → one table; saves JSON to `data/eval_results/` |
| `ch11/grounded_answering.py` | 11 | strict grounding prompt, sentence-level citations, citation verification |
| `ch11/faithfulness_gate.py` | 11 | score gate, faithfulness gate with retry/abstain, self-consistency |
| `ch12/coverage_report.py` | 12 | score distributions answerable vs unanswerable, threshold calibration, gaps report |
| `ch12/abstain_and_fallback.py` | 12 | abstain, clarify, or route to a fallback tool |
| `ch13/traced_rag.py` | 13 | JSON-lines logging, LangSmith tracing (optional), callback handler for usage |
| `ch13/cost_and_latency.py` | 13 | per-stage latency breakdown and cost table |
| `ch14/hello_graph.py` | 14 | state, nodes, conditional edges, streaming, checkpointer memory across turns |
| `ch14/rag_graph.py` | 14 | the Chapter 6 pipeline as a graph with a grade → rewrite loop |
| `ch15/agentic_rag_graph.py` | 15 | retrieval as a tool, ToolNode, document grading, query rewriting, bounded loop |
| `ch15/agentic_rag_create_agent.py` | 15 | the same with `create_agent` + limit middleware + structured response; an adaptive router |
| `ch16/planner_retriever_analyst.py` | 16 | a multi-agent loop that runs away, then the fixed version with termination, turn and token budgets |
| `ch16/guardrails.py` | 16 | RAG-first enforcement (forced tool choice, after-model check), PII redaction, role-based Qdrant filters, citation rate |
| `ch17/app.py` + `client_demo.py` | 17 | FastAPI service: async endpoints, streaming, cache, API-key auth + RBAC; `Dockerfile` and root `docker-compose.yml` |
| `ch18/scale_math.py` | 18 | chunks, embedding cost, RAM (fp32/int8/binary), ingest time and $/query for 1 GB → 1 TB |
| `ch18/quantized_collection.py` | 18 | scalar and binary quantization, rescoring/oversampling, HNSW parameters, on-disk vectors |
| `ch18/multitenant_collection.py` | 18 | tenant payload index, per-tenant HNSW, blue/green re-embedding with collection aliases |

## Part 8 · the deep dives (Chapters 21–30)

These scripts **measure** rather than demonstrate: they build the structure, sweep the parameter,
and print the curve. Several are useful as tools long after you have read the chapter: in
particular `ch27/diagnose.py` (point it at any failing query) and `ch26/triage.py` (point it at any
bad answer).

Shared helpers imported by their neighbours, not entry points: every module starting with `_`
(`ch24/_shared.py`, `ch25/_embedders.py`) plus `ch22/viz.py` (ASCII curve and bar helpers) and
`ch23/retrievers.py` (tunable BM25, cached dense scores, the rank-aware metrics).

| Script | Ch | What it measures |
|---|---|---|
| `ch21/metric_illusions.py` | 21 | ranking cases with identical recall but very different MRR/nDCG; the nDCG example worked by hand. No API calls |
| `ch21/recall_vs_correct.py` | 21 | the four-stage funnel (retrieved → in context → used → correct) and the gap recall@k cannot see |
| `ch21/chunk_level_labels.py` | 21 | document-level vs chunk-level relevance labels, and how much the coarse ones inflate recall |
| `ch21/significance.py` | 21 | bootstrap confidence intervals and a paired permutation test on saved eval runs |
| `ch22/hnsw.py` | 22 | HNSW built from scratch: levels, the neighbour-selection heuristic, beam search |
| `ch22/parameter_study.py` | 22 | recall, distance calls, latency and memory as `ef_search`, `M` and `ef_construction` sweep |
| `ch22/recall_degradation.py` | 22 | recall at a *fixed* `ef` as the index grows: why yesterday's tuning expires |
| `ch22/ivf_and_pq.py` | 22 | minimal IVF (`nprobe`) and product quantization, with the rescoring recovery |
| `ch22/filtered_search.py` | 22 | pre-filter vs post-filter vs native filtered search; post-filtering returning almost nothing |
| `ch22/embed_cache.py` | 22 | builds `data/cache/ch22_real_vectors.npz` so the sweeps run on real embeddings, not Gaussian noise (`--n 1500`, well under a cent) |
| `ch23/score_distributions.py` | 23 | why BM25 and cosine cannot be added: the distributions, and a "50/50" blend that is 98% one side |
| `ch23/fusion_methods.py` | 23 | RRF with a `k` sweep, weighted RRF, CombSUM/CombMNZ, z-score fusion, DBSF |
| `ch23/bm25_parameters.py` | 23 | `k1` saturation and `b` length-normalisation curves; negative IDF |
| `ch23/learned_sparse.py` | 23 | BM25 vs a learned sparse model: better on paraphrase, worse on identifiers |
| `ch24/bottleneck_demo.py` | 24 | the bi-encoder bottleneck: the answering chunk at rank 15 under exact cosine, fixed by a cross-encoder |
| `ch24/cascade_design.py` | 24 | recall@N against final nDCG@5 with latency and cost, so candidate depth is chosen with evidence |
| `ch24/reranker_shootout.py` | 24 | cross-encoders, ColBERT late interaction and LLM rerankers compared |
| `ch24/truncation_and_limits.py` | 24 | what a 512-token reranker does to a long chunk, plus latency percentiles |
| `ch25/embedding_bakeoff.py` | 25 | six models over one corpus: recall, nDCG, memory, cost, throughput |
| `ch25/prefix_and_length.py` | 25 | instruction prefixes (the difference between 0.534 and 0.954), and chunks over the model's limit |
| `ch25/dimension_curve.py` | 25 | quality against dimensions and memory, with the arithmetic at 1M and 100M vectors |
| `ch26/failure_injection.py` | 26 | eight hallucination causes induced on purpose, and which metric catches each |
| `ch26/judge_bias.py` | 26 | position and verbosity bias, judge self-consistency as context grows, kappa against human labels |
| `ch26/triage.py` | 26 | runs the diagnosis tree on a live answer and names the first stage that failed |
| `ch27/diagnose.py` | 27 | **the rank-17 playbook**: twelve diagnostics on one query, with a verdict and a ranked fix list |
| `ch27/chunk_experiment.py` | 27 | 45-cell factorial sweep of strategy × size × overlap against answer coverage and cost |
| `ch27/query_rewriting_lab.py` | 27 | expansion, HyDE, step-back and decomposition measured: including where rewriting *hurts* |
| `ch28/make_sample_pdf.py` | 28 | generates the fixture PDF: two-column prose, a table, a vector chart, a tree diagram, a raster figure |
| `ch28/parser_shootout.py` | 28 | pypdf vs pymupdf on the same file: text, blocks, coordinates, images, timings |
| `ch28/tables_and_representations.py` | 28 | the same table in five representations, measured by retrieval |
| `ch28/figures_to_text.py` | 28 | a diagram captioned by a vision model under a generic and a structure-eliciting prompt |
| `ch28/ocr_noise.py` | 28 | plausible OCR corruption and the recall it costs |
| `ch29/lost_in_the_middle.py` | 29 | answer accuracy with the gold chunk first, middle and last, across context lengths |
| `ch29/dedup_and_compress.py` | 29 | near-duplicate removal and three compression strategies, as a tokens-versus-quality trade |
| `ch29/assembler.py` | 29 | the shippable context builder: budget, dedup, ordering, headers, conflicts, injection defence |
| `ch30/ingest_queue.py` | 30 | a real durable queue: leases, retries with jitter, a dead-letter queue, resumability (`--fail-rate` to watch it recover) |
| `ch30/dedup_minhash.py` | 30 | MinHash near-duplicate detection, scored against deliberately perturbed copies |
| `ch30/batch_and_ratelimit.py` | 30 | embedding throughput against batch size, a token bucket, cache hit rate, the 1M-document cost table |
| `ch30/migrate_embeddings.py` | 30 | version-marked blue/green re-embedding with pre-switch validation and an alias flip |
