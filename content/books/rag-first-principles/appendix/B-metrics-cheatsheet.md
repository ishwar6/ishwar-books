# Appendix B · Every Metric on One Sheet

> For each metric: what it measures, the formula, how to compute it (function names refer
> to [`code/ch10/metrics.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch10/metrics.py)), what "good" looks like on a corpus like the handbook, and
> **what to change when it is low**. Chapter 10 has the code and worked examples; this is the
> reference you keep open.

Notation: for one question, `R` = set of relevant chunk ids (from the golden set's
`sources`, mapped to chunk ids), `L = [l₁, l₂, …]` = retrieved chunk ids in rank order,
`L@k` = first k of them, `rel(i) = 1` if `lᵢ ∈ R` else 0. All metrics are computed per
question and **averaged over the golden set**; report the mean and, ideally, the
per-tag breakdown (numeric / multi_hop / unanswerable …).

---

## B.1 Retrieval metrics (no LLM needed: run these first, they are free)

| Metric | Measures | Formula | Code | Good on handbook | If low → |
|---|---|---|---|---|---|
| **Hit rate@k** (a.k.a. success@k) | Did *any* relevant chunk make the top k? | `1 if |L@k ∩ R| > 0 else 0` | `hit_rate_at_k(retrieved, relevant, k)` | ≥ 0.9 at k=5 | The fact is not being found at all: check the doc was ingested; try hybrid (BM25 for numbers/names); increase k as a diagnostic; check chunking did not split the fact; verify query and index used the same embedding model. |
| **Recall@k** | Fraction of relevant chunks retrieved | `|L@k ∩ R| / |R|` | `recall_at_k` | ≥ 0.8 at k=5 | Same levers as hit rate; for multi-source questions add multi-query or decomposition so each sub-fact gets its own search; small-to-big so one hit brings its siblings. |
| **Precision@k** | Fraction of retrieved that are relevant | `|L@k ∩ R| / k` | `precision_at_k` | ≥ 0.4 at k=5 (many chunks are near-duplicates, so this is naturally modest) | Add a reranker; lower k; tighten metadata filters; dedupe near-identical chunks; improve chunk boundaries so chunks are about one thing. |
| **F1@k** | Harmonic mean of P@k and R@k | `2PR/(P+R)` | `f1_at_k` | ≥ 0.5 | Whichever of P/R is lower. |
| **MRR** (mean reciprocal rank) | How high is the *first* relevant chunk? | `1 / rank_of_first_relevant` (0 if none) | `mrr` | ≥ 0.7 | Reranking is the direct fix; query rewriting; hybrid. MRR up while recall flat = ranking fixed, retrieval unchanged. |
| **MAP** (mean average precision) | Precision averaged at each relevant hit | `(1/|R|) Σ_{i: rel(i)=1} P@i` | `mean_average_precision` | ≥ 0.6 | Reranking; better negatives if you fine-tune an embedding model. |
| **nDCG@k** | Ranking quality with position discount (supports graded relevance) | `DCG@k = Σᵢ₌₁ᵏ rel(i)/log₂(i+1)`; `nDCG = DCG / IDCG` (IDCG = DCG of the ideal ordering) | `ndcg_at_k` | ≥ 0.7 | Reranker; chunk ordering; if graded labels exist, tune to them. The standard metric in IR papers: know how to derive it. |
| **Recall vs brute force** (ANN recall) | Is the *index* losing results the exact search would find? | `|ANN top-k ∩ exact top-k| / k` on a sample | compute with `client.query_points(..., search_params=SearchParams(exact=True))` vs default | ≥ 0.95 | Raise HNSW `ef` (query) or `m`/`ef_construct` (rebuild); if quantised, enable `rescore` and `oversampling`. This isolates index error from embedding error. |

**Reading them together**
- Hit rate high, recall low → finding *some* evidence but not all; multi-source questions
  suffer. Multi-query / decomposition.
- Recall high, precision low → context is noisy; the generator pays in tokens and
  distraction. Rerank, lower k.
- Recall high, MRR low → right chunks, wrong order. Rerank.
- Everything low for `numeric`/`code` tags only → embeddings blur numbers; add BM25.
- Everything low for one document only → chunking or parsing problem in that document.

---

## B.2 Context quality (LLM-judged, per retrieved chunk)

| Metric | Measures | Formula | Code | Good | If low → |
|---|---|---|---|---|---|
| **Context precision** | Of the retrieved chunks, how many are actually useful for the answer: weighted by rank | judge each chunk `useful ∈ {0,1}` given question + reference answer; `Σᵢ (P@i · usefulᵢ) / Σ usefulᵢ` (RAGAS form) or simply `Σ usefulᵢ / k` | `context_precision_at_k` (rank-weighted, doc-level labels; Chapter 10) | ≥ 0.6 | Reranker; smaller k; better chunk boundaries. |
| **Context recall** | Does the retrieved context cover the reference answer? | split reference answer into statements; `supported statements / total statements` | `context_recall_docs` (doc-level; the claim-level LLM version is `faithfulness`-style judging in `llm_judges.py`) | ≥ 0.85 | Retrieval recall levers; larger chunks or parent-document retrieval so the whole fact is present. |
| **Context utilisation** (RAGChecker) | Of the relevant context that *was* retrieved, how much did the answer use? | `claims in answer supported by context / relevant claims available in context` | - | ≥ 0.7 | Generation problem: prompt ordering (lost in the middle), too much context, model too small. |

---

## B.3 Generation metrics (LLM-judged, on the answer)

| Metric | Measures | Formula | Code | Good | If low → |
|---|---|---|---|---|---|
| **Faithfulness** (groundedness) | Are the answer's claims supported by the retrieved context? | split answer into atomic claims; judge each `supported ∈ {0,1}` against the context only; `supported / total` | `faithfulness(answer, context)` | ≥ 0.9; 1.0 on `numeric` | Hallucination. Grounding prompt ("only from context"); require citations by chunk id and validate them; faithfulness gate that rewrites or refuses; lower verbosity; check context actually contained the fact (if not, this is a retrieval failure disguised). |
| **Answer relevance** | Does the answer address the question (regardless of truth)? | generate n questions from the answer, mean cosine(generated, original) (RAGAS); or direct 1–5 judge with rubric | `answer_relevance` | ≥ 0.8 | Prompt: answer the question first; stop the model from summarising the whole context; check query rewriting did not change intent. |
| **Answer correctness** | Does the answer agree with the reference? | judge `correct ∈ {0, 0.5, 1}` with the reference; plus cheap checks: `keywords` all present (`keyword_hit`), numeric exact match | `answer_correctness(question, answer, reference)` + `keyword_hit(answer, keywords)` | ≥ 0.8 | If faithfulness is high but correctness low → wrong/insufficient context (retrieval) or conflicting sources (add recency metadata). If faithfulness low too → generation. |
| **Completeness** | Did it cover all parts of a multi-part reference? | reference statements covered / total | part of `answer_correctness` | ≥ 0.8 | Decomposition; larger k for `multi_hop`; ask for all parts explicitly. |
| **Citation accuracy** | Do cited chunk ids actually support the sentence that cites them? | judge per citation; `valid / total citations` | `citation_accuracy` ([`code/ch10/llm_judges.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch10/llm_judges.py)) | ≥ 0.9 | Structured output with citations; validate ids exist; penalise uncited claims in the prompt. |
| **Refusal precision / recall** | Refuses when it should, answers when it can | over golden `answerable` flag: refusal-precision = correct refusals / all refusals; refusal-recall = correct refusals / unanswerable questions | `is_abstention` + the `answerable` flag in `run_eval.py` (Chapter 10/12) | both ≥ 0.8 | Low recall (answers made-up stuff): retrieval score threshold, "answerable" field, grounding prompt. Low precision (over-refuses): threshold too strict, prompt too cautious, retrieval missing real answers. |
| **Conciseness / length** | Tokens in the answer | `len(tokens)` | trivial | 30–150 tokens for factoid | Prompt; lower verbosity settings. Long answers also inflate judge scores (verbosity bias). |

---

## B.4 End-to-end and business metrics

| Metric | Measures | How | Good | If low → |
|---|---|---|---|---|
| **Exact / keyword match** | cheap correctness proxy | all `keywords` present (case-insensitive, normalised numbers) | ≥ 0.85 | Same as correctness; check normalisation (₹1,500 vs 1500). |
| **Task success rate** | did the user get what they needed | thumbs up/down, "was this helpful", follow-up-question rate | > 70% helpful | Look at the traces of the failures; cluster them; fix the biggest cluster. |
| **Escalation / deflection rate** | support use cases: tickets avoided | tickets before vs after | domain-specific | Coverage analysis: which questions have no answer in the corpus (Chapter 12). |
| **Retrieval-call rate** (agents) | fraction of factual questions where the agent actually retrieved | from traces | ~100% on factual | Force retrieval structurally (Chapter 16). |
| **Steps / tokens per task** (agents) | loop cost | from traces / `usage_metadata` | ≤ 3 steps median | Limits, termination condition, better tools. |

---

## B.5 Operational metrics

| Metric | Measures | How | Typical target | If bad → |
|---|---|---|---|---|
| **Latency p50 / p95 / p99** | end to end and per stage (embed, retrieve, rerank, generate) | timers in the trace; `time.perf_counter()` per stage | p95 < 2–3 s for chat; retrieval < 100 ms | Stage that dominates: generation → stream, smaller model, shorter context; rerank → fewer candidates, lighter model, GPU batching; retrieval → `ef`, quantisation, payload indexes, caching. |
| **Time to first token** | perceived latency | stream and time the first chunk | < 1 s | Stream; move checks after streaming; prompt caching. |
| **Cost per query** | $ | `(input_tokens × p_in + output_tokens × p_out)` from `usage_metadata` + embedding + infra/QPS | know your number | Shorter context (k, chunk size), smaller model for easy questions, caching, prompt-prefix ordering for cache hits. |
| **Tokens per query** (input/output) | the driver of cost and latency | `usage_metadata` | e.g. 1,500 in / 200 out | Same. |
| **Throughput (QPS)** and **error rate** | capacity, reliability | load test; 429/5xx counts | error < 0.5% | Async I/O, connection pools, provider fallback, retries with backoff. |
| **Index freshness lag** | time from source change to searchable | timestamp diff | < 5 min | Event-driven ingestion; smaller batches. |
| **Cache hit rate** | semantic/prefix cache effectiveness | hits / lookups | 20–40% semantic; > 70% prefix | Stable prompt prefix first; threshold tuning; normalise queries. |

---

## B.6 Evaluation-process metrics (is your judge trustworthy?)

| Metric | Measures | Formula | Target | If low → |
|---|---|---|---|---|
| **Judge–human agreement** | raw agreement | `matches / n` on a human-labelled sample (50–100 items) | > 85% | Rubric with explicit criteria; claim-level decomposition; stronger judge model; different family from the generator. |
| **Cohen's kappa** | agreement corrected for chance | `κ = (p_o − p_e)/(1 − p_e)` where `p_o` = observed agreement, `p_e` = expected by chance from marginals | κ > 0.6 (substantial), > 0.8 (near-perfect) | Same as above; binary labels are easier to agree on than 1–5 scales. |
| **Position-swap consistency** | pairwise judge stability | fraction of pairs where verdict is the same after swapping A/B | > 90% | Always evaluate both orders and average; use absolute rubric scoring instead of pairwise. |
| **Judge variance** | run-to-run noise | std of scores over repeated runs | small relative to the effect you care about | Fixed prompts, lower randomness, more items. |
| **Golden-set coverage** | does the eval set represent traffic | fraction of real-query clusters with ≥ n golden items | all major clusters | Add real questions from logs; regenerate synthetic ones per new document. |

---

## B.7 The formulas you should be able to write on a whiteboard

```
precision@k = |retrieved@k ∩ relevant| / k
recall@k    = |retrieved@k ∩ relevant| / |relevant|
MRR         = mean over questions of 1 / rank(first relevant)      (0 if none)
AP          = (1/|relevant|) · Σ_{i : rel(i)=1} precision@i ;  MAP = mean(AP)
DCG@k       = Σ_{i=1..k} rel(i) / log2(i + 1)
nDCG@k      = DCG@k / IDCG@k
faithfulness = supported claims / total claims           (claims from the answer, judged vs context)
context recall = reference statements supported by context / total reference statements
RRF(d)      = Σ_lists 1 / (60 + rank_list(d))
BM25(q,d)   = Σ_t IDF(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl)),  k1≈1.5, b≈0.75
cosine(a,b) = a·b / (|a||b|)
kappa       = (p_o − p_e) / (1 − p_e)
```

---

## B.8 A minimal eval report template

```
Config: collection=handbook_v3  chunk=800/120  embed=text-embedding-3-small  k=5  rerank=none  model=gpt-5.4-mini
Golden: 47 questions (42 answerable, 5 unanswerable)   Judge: gpt-5.4-mini, rubric v2

Retrieval        hit@5 0.93   recall@5 0.86   precision@5 0.41   MRR 0.78   nDCG@5 0.81
Generation       faithfulness 0.94   relevance 0.88   correctness 0.83   keyword-hit 0.86
Refusal          precision 0.83   recall 1.00
Cost/latency     1,420 in / 96 out tokens   $0.0015/q   p50 1.9 s   p95 3.4 s
Worst tags       multi_hop: recall 0.62   numeric: correctness 0.75
Judge check      agreement 0.90 (n=40)   kappa 0.78
```

Write one of these per change. The diff between two reports is the result of your
experiment; anything not in the report did not happen.

## B.9 Before you trust any number above

Every metric in this appendix is an estimate from a sample, computed against labels you invented,
and half of them are produced by a stochastic judge. Four checks separate a result from a story:
all are derived and measured in **[Chapter 21](../part8-deep-dives/21-retrieval-metrics-deep.md)**
and **[Chapter 26](../part8-deep-dives/26-hallucination-forensics.md)**.

| Check | Question it answers | Where |
|---|---|---|
| **Bootstrap CI** on every headline metric | is this difference bigger than the noise? On 42 questions the 95% interval on recall is about ±0.08: a 2-point "win" is nothing | Ch 21 §21.5, [`code/ch21/significance.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch21/significance.py) |
| **Paired** permutation test, not two independent means | did config B beat config A *on the same questions*? Pairing removes question difficulty as a confound | Ch 21 §21.5 |
| **Label granularity** stated out loud | document-level labels score a retrieval as perfect when the answer-bearing chunk was never returned; on this corpus they inflate recall by ~5 points | Ch 21 §21.3, [`code/ch21/chunk_level_labels.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch21/chunk_level_labels.py) |
| **Judge stability** re-measured as context grows | the same judge scoring the same objectively-correct citation was 100% self-consistent on a 1-chunk context and 60% on a 10-chunk one: judge error grows with exactly the variable you vary when tuning k | Ch 26 §26.5, [`code/ch26/judge_bias.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch26/judge_bias.py) |

The practical rule: **no single metric moving alone is a result.** Trust changes where several
related metrics move together, where the CI excludes zero, and where you can name the mechanism.

And the sentence to have ready in an interview: *recall@k is stage one of four*: retrieved, then
survived context assembly, then used by the model, then correct. Quoting retrieval recall as an
accuracy figure claims the other three stages are lossless.
