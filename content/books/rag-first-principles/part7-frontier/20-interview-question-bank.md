# Chapter 20 · The Interview Question Bank

> **Goal:** you can answer every RAG, agent, evaluation and deployment question you are
> likely to be asked (including all the ones from the Capgemini Invent interview that
> started this book) with a strong, concrete, structured answer. Part A is that transcript,
> answered the way a strong candidate answers. Part B is the deeper bank (Google/Meta-style).
> Part C is one system-design question worked end to end. **Part D is the second interview**:
> the one that went below the API layer into retrieval mechanics, where the honest answers are
> harder; each Part D answer links to the Part 8 chapter that derives and measures it.
>
> How to use it: read a question, answer out loud for 60–90 seconds, then compare. Where an
> answer says "Chapter N", that chapter has the code.

Answer shape that works in interviews: **definition in one sentence → how it works → the
trade-off → a number or name → what you would do in production.** Most answers below follow it.

---

## Part A · The transcript, answered well

### A1. Explain the end-to-end RAG pipeline.

Two phases. **Ingestion (offline):** load documents (PDF, HTML, Markdown, databases) →
parse into text with structure (headings, tables) → chunk (recursive, header-aware, 300–800
tokens with overlap) → embed each chunk with an embedding model → store vector + text +
metadata (source, section, date, permissions) in a vector database like Qdrant, with payload
indexes on the metadata you filter on. **Query (online):** embed the question with the *same*
model → retrieve candidates (dense ANN via HNSW, plus BM25 for hybrid, fused with RRF) →
optionally filter by metadata and rerank with a cross-encoder → take the top k → build a
prompt: system instructions, numbered context chunks, the question → LLM generates a
grounded answer with citations → post-checks (faithfulness, PII) → log the trace and score
it. Everything is measured against a golden set: recall@k for retrieval, faithfulness and
correctness for generation. (Chapters 3, 6.)

### A2. What is an embedding, and why do we convert to embeddings and store them?

An embedding is a fixed-length vector (e.g. 1,536 floats) produced by a neural model such
that texts with similar *meaning* land close together, measured by cosine similarity. It
turns "is this passage relevant to this question" into arithmetic, and handles synonyms and
paraphrase that keyword search misses ("time off" ≈ "leave" ≈ "PTO"). We *store* them
because embedding is the expensive step (a model call per chunk); doing it once at ingest
means each query costs one embedding call plus a sub-10 ms nearest-neighbour search. The
vector DB exists to make that search fast at scale (HNSW index instead of comparing against
every vector) and to attach filters and payloads. (Chapters 2, 4.)

### A3. A PDF contains text, images and hyperlinks. What chunking strategy do you use?

Parse by *modality* first, then chunk each appropriately, and keep everything linked by
metadata. **Text:** use a layout-aware parser (Docling, Unstructured, or pypdf plus a
heading heuristic) so headings, paragraphs and tables survive; then recursive,
header-aware chunking (split on headings first, fall back to paragraphs, then sentences),
300–800 tokens, 10–15% overlap; keep tables as whole chunks rendered to Markdown, never split
mid-table. **Images:** OCR for scanned text; for diagrams and charts generate a caption with a
vision model and embed the caption, storing the image reference in the payload: or use
ColPali to embed page images directly if the PDFs are visual. **Hyperlinks:** do not embed
URLs; store them as metadata on the chunk (`links: [...]`) so an agent can follow them, and
optionally fetch and index the target as a related document. Add contextual retrieval (a
one-sentence LLM summary prepended to each chunk) if retrieval is weak. Then *measure* the
choice on a golden set: chunk size is an empirical parameter. (Chapters 5, 9, 19.)

### A4. How do you secure sensitive information in a RAG system?

Defence in depth across the pipeline. **At ingest:** classify documents (Public/Internal/
Confidential/Restricted), redact or tokenise PII (emails, IDs, card numbers) before embedding:
the embedding itself can leak content. **At storage:** encrypt at rest, and store
permission attributes (`tenant`, `groups`, `classification`) in the payload with a payload
index. **At retrieval:** enforce access control *as a filter in the query*, never as a
post-filter the model applies: the user's groups become a Qdrant `Filter(must=[...])`, so
unauthorised chunks are never retrieved. **At generation:** send data only to LLM providers
that passed a vendor security review, with no-training terms; add PII middleware on input
and output. **Against injection:** treat retrieved text as data, delimit it, trust-filter
sources. **Around it:** audit logs of who retrieved what, secrets in a vault, rate limits,
and red-team evals with injected documents. (Chapters 16, 17.)

### A5. How do you prevent hallucinations?

Hallucination in RAG has two sources: the retriever did not bring the fact, or the model
ignored the context. Attack both. **Retrieval side:** improve recall (hybrid search,
reranking, better chunking) and *detect when retrieval failed* (a similarity or reranker
score threshold, or an LLM relevance grade) so the system can say "I don't know" instead of
inventing. **Generation side:** a strict grounding prompt ("answer only from the context;
if it is not there say so"), mandatory citations by chunk id, and low-verbosity output.
**Post-generation:** a faithfulness check: split the answer into claims and verify each
against the retrieved chunks with a judge model; drop or flag unsupported claims. **Process
side:** measure faithfulness and refusal-correctness on a golden set that includes
unanswerable questions, and track them per release. Choosing k is a tuning knob, not a
hallucination control. (Chapters 11, 12.)

### A6. How do you evaluate LLM outputs?

With a golden dataset and layered metrics. Build 50–500 questions with reference answers
and the source passages, including unanswerable and adversarial ones. **Retrieval metrics**
(no LLM needed): hit rate, recall@k, precision@k, MRR, nDCG. **Generation metrics** with an
LLM judge: faithfulness (claims supported by context), answer relevance, answer correctness
against the reference (with exact/keyword checks where possible). **Behavioural:** refusal
rate on unanswerable questions, citation accuracy. **Operational:** latency p50/p95, cost per
query. Validate the judge against a human-labelled sample (agreement/kappa) before trusting
it, and run the whole suite in CI so a change to chunk size or prompt shows up as a number
diff. (Chapter 10, Appendix B.)

### A7. How do you ensure the quality of your implementation?

Same way as any software, plus evals. Unit tests for deterministic parts (chunker produces
expected boundaries, filters build correctly, metrics functions match hand-computed values).
Integration tests with a fake embedding model so the pipeline runs in CI without cost. An
**eval suite** on the golden set with thresholds that fail the build if recall or
faithfulness drops. **Tracing** (LangSmith) on every run so a bad answer can be replayed:
what was retrieved, what prompt was sent, what the model returned. Versioned prompts and
collection names so changes are diffable and reversible. A canary rollout with online
signals (thumbs, refusal rate, latency). Code review that includes the prompt. (Chapters 10,
13, 17.)

### A8. Multi-agent system: planner, retriever and analyst agents loop forever, tokens explode. Why, and how do you fix it?

**Why:** each agent's output is the next agent's input with no terminal condition. Typical
causes: the planner re-plans whenever the analyst says "insufficient", the analyst says
insufficient whenever retrieval is weak, and nobody owns "stop"; the shared message history
grows every hop so every call gets more expensive; agents re-issue near-identical retrievals
because state is not deduplicated. **Fix, in layers:** (1) a hard **recursion / step limit**
on the graph (LangGraph `recursion_limit`) and per-run **model-call and tool-call limits**
(`ModelCallLimitMiddleware`, `ToolCallLimitMiddleware`) that end gracefully with a partial
answer; (2) an explicit **termination condition** in state: a `done` flag or a confidence
score with a threshold: checked by a conditional edge; (3) **token budget** per request
and per agent, tracked from `usage_metadata`, with summarisation of history
(`SummarizationMiddleware`) when it grows; (4) **idempotency**: cache retrieval results by
query so repeated calls are free and visible; (5) an **orchestrator** node that validates
each proposed hop against the budget before executing it, and observability so loops are
visible in traces. Test with an adversarial question that has no answer. (Chapter 16.)

### A9. An agent answers from its own knowledge instead of calling the retrieval agent. How do you enforce RAG-first behaviour?

Do not rely on the prompt alone; make it structural. Options from weakest to strongest:
(1) system prompt: "you must call `search` before answering any factual question and cite
results"; (2) **force the first tool call**: bind tools with `tool_choice="search"` for
the first model turn, or start the graph with a retrieval node so the model never sees the
question without context; (3) a **middleware/after_model check** that rejects any final
answer produced with zero retrieval calls in the run and re-prompts; (4) require
**citations** in structured output and validate that each cited id exists in the retrieved
set: an answer with no valid citations is rejected; (5) evaluate it: a "retrieval-call
rate" metric on factual questions, in CI. In the graph version this is just an edge:
START → retrieve → generate, with the LLM having no path to generate without retrieval.
(Chapters 15, 16.)

### A10. How does MCP differ from a REST API?

REST is a contract between programs: a developer reads docs and writes code to call
endpoints. MCP (Model Context Protocol) is a contract between a *model-driven application*
and services: the server publishes its **tools** (name, description, JSON schema), **resources**
(readable documents) and **prompts** in a machine-readable form; the client discovers them at
runtime and the *model* decides which to call, with a standard JSON-RPC transport,
capability negotiation and (since 2025) OAuth-based authorisation. So MCP is one level up: an
MCP server typically *wraps* REST APIs or databases (Notion, SQL, Slack) and turns them into
tool calls any MCP-capable agent can use without bespoke glue. Analogy: REST is USB for
software; MCP is USB-C for models: a single connector. The 2026-07-28 spec made the core
stateless for scaling. (Chapter 19.)

### A11. When should you use async vs sync endpoints in FastAPI?

Use `async def` when the handler spends its time **waiting on I/O** that has an async client:
LLM API calls, embedding calls, Qdrant (`AsyncQdrantClient`), HTTP, async database drivers.
The event loop serves other requests during the wait, so one worker handles hundreds of
concurrent RAG requests. Use plain `def` when the work is **CPU-bound or uses a blocking
library** (a sync SQLAlchemy session, PDF parsing, a local reranker): FastAPI runs sync
handlers in a threadpool so they do not block the loop. The bug to avoid: calling a blocking
function inside `async def`, which freezes every request. For long jobs (ingestion), return
202 and hand off to a queue/background task. Vector search is I/O; use the async client.
(Chapter 17.)

### A12. How do you manage DB sessions in FastAPI?

One session per request, created by a dependency with `yield` and always closed: a `get_db()`
dependency opens a session from a connection pool (SQLAlchemy engine with `pool_size`,
`max_overflow`), yields it to the endpoint, and in `finally` closes it so the connection
returns to the pool even on exceptions. Endpoints declare `db: Session = Depends(get_db)`.
Never share a session across requests or threads; commit explicitly in the service layer;
use the async engine (`AsyncSession`) with async endpoints. Pool sizing: roughly
`workers × pool_size` must stay under the database's connection limit: use PgBouncer if
not. Same pattern for the Qdrant client, except the client is long-lived (one per process,
created in the `lifespan` handler) because it manages its own connections.

### A13. How do you implement authentication?

For users: login → verify credentials → issue a short-lived **JWT access token** (15–60 min,
signed, with `sub`, `exp`, `scope`/`groups`) plus a refresh token in a secure httpOnly
cookie. Every request carries `Authorization: Bearer <jwt>`; a FastAPI dependency verifies
signature and expiry, loads the user, and checks scopes; the endpoint gets the user object.
For services: **API keys** (hashed at rest, shown once, rotatable, rate-limited per key) or
mTLS. For enterprise: OIDC via an identity provider (Okta, Entra) and validate their tokens
with the JWKS endpoint. Then authorisation: the user's groups flow *into the retrieval
filter* so RAG only sees permitted documents. Log auth failures; rotate signing keys.

### A14. Which cloud services have you used, and how would you deploy a RAG service?

Compute: containerised FastAPI on Cloud Run / ECS / GKE (or EC2 for simple cases), with a
Dockerfile and a Compose file for local dev. Vector DB: managed Qdrant Cloud or Qdrant on
Kubernetes with persistent volumes. Storage: S3/GCS for raw documents; a queue (SQS/Pub/Sub)
feeding ingestion workers so uploads are decoupled from indexing. Secrets in Secret
Manager/Vault; observability in CloudWatch/Cloud Logging plus LangSmith for traces; IAM
service accounts with least privilege. Vertex AI or Bedrock when data residency requires a
model inside the cloud boundary. (Chapter 17.)

### A15. Your RAG gives poor responses. How do you debug it?

Separate retrieval from generation with the trace. **Step 1:** look at what was retrieved
for the failing question. If the answer is not in the top-k → retrieval problem: check
chunking (is the fact split across chunks?), try hybrid search (numbers, codes and names
are BM25's job), add a reranker, check metadata filters are not excluding it, check the
embedding model was the same at ingest and query, check the document was actually ingested.
**Step 2:** if the answer *is* in the context but the response is wrong → generation
problem: prompt (grounding instructions, context order, too much context), model choice,
or conflicting chunks (old FAQ vs new policy → add recency metadata and prefer it).
**Step 3:** classify failures over the eval set, fix the biggest bucket, rerun. Never tune by
eyeballing one question. (Chapters 10, 13.)

### A16. Agentic RAG vs generative (naive) RAG?

Naive RAG: retrieve once, generate once, fixed k, no judgement. Agentic RAG: the model has
retrieval (and other tools) as callable actions and decides *whether* to retrieve, *what* to
ask, *whether the results are good enough*, and *whether to try again*: a loop with
grading and rewriting (Self-RAG, CRAG, Adaptive-RAG). Agentic wins on multi-hop, ambiguous
and mixed questions ("compare X and Y" needs two searches); naive wins on predictability,
latency and cost for FAQ-style traffic. In production: route simple questions to the cheap
pipeline and hard ones to the agent, and bound the agent with call limits. (Chapter 15.)

### A17. An agent in a multi-agent chain fails. How do you handle it?

Classify the failure. **Transient** (429, timeout, 5xx): retry with exponential backoff
and jitter, capped attempts, and a per-call timeout: `ModelRetryMiddleware` /
`ToolRetryMiddleware` do this; make calls **idempotent** so retries are safe. **Persistent**
(bad output, schema violation): validate with structured output, re-prompt once with the
error, then **fall back**: a simpler model, a cached answer, or a degraded response that
says which part failed. **Systemic** (downstream is down): a **circuit breaker** stops
hammering it and the orchestrator routes around or returns partial results. Always: the
orchestrator owns the timeout for the whole request, state is checkpointed so a resumed run
does not redo finished steps, and every failure is in the trace with the input that caused
it. Design so a failed analyst never blocks returning what the retriever already found.

### A18. Recall@K vs precision@K, faithfulness, correctness: define and say when each matters.

**Recall@K** = fraction of relevant chunks that appear in the top K; it says whether
retrieval *found* the evidence: the metric to raise first, since the model cannot use what
it does not see. **Precision@K** = fraction of the top K that are relevant; it controls
noise and tokens in the prompt: matters when K is large or context is expensive.
**Faithfulness** = fraction of the answer's claims supported by the retrieved context; it
measures hallucination independent of whether the answer is right. **Correctness** =
agreement of the answer with the reference; the end-to-end metric. An answer can be faithful
but incorrect (context was wrong or incomplete) or correct but unfaithful (model knew it
anyway). Report all four; they localise the failure. (Chapter 10, Appendix B.)

### A19. Your corpus grows from 1 GB to 10 GB. What changes?

Numbers first: 1 GB of text ≈ 250M tokens ≈ 600K chunks of 400 tokens ≈ 3.7 GB of float32
vectors at 1,536 dims; 10 GB is ten times that: 37 GB of vectors plus the HNSW graph, which
no longer fits comfortably in one node's RAM. So: (1) **quantise** (int8 → 4×, binary → 32×
with rescoring) and consider fewer dimensions (Matryoshka 512 → 3×); (2) put vectors
**on disk** with memory-mapped HNSW, or **shard** across nodes; (3) **tune HNSW** (`m`,
`ef_construct`, `ef`): recall drops as the index grows at the same `ef`, so re-measure;
(4) **filtering matters more**: payload indexes and tenant-aware indexing so a filtered
search does not scan; (5) **ingestion** becomes a pipeline (queue, batching, rate limits on
the embedding API, idempotent upserts, incremental updates), not a script; (6)
**latency** rises: add a reranker to keep k small, cache, and use hybrid search because
dense recall degrades with more distractors; (7) **re-embedding cost** at ~$0.02/M tokens
is ~$50 per re-index at 10 GB: fine, but plan for it. (Chapter 18.)

### A20. Explain the transformer briefly.

The transformer (Vaswani et al., 2017, "Attention Is All You Need") replaced recurrence
with **self-attention**: every token computes query, key and value vectors; attention
weights are softmax(QKᵀ/√d), so each token's new representation is a weighted mix of all
tokens' values: every position sees every other in one step, in parallel, which is why it
trains far faster than an RNN on GPUs and handles long-range dependencies. Multiple heads
learn different relations; positional encodings supply order; stacks of attention +
feed-forward layers with residuals and layer norm build depth. Encoder-only models (BERT)
produce embeddings; decoder-only models (GPT) generate. Cost is quadratic in sequence
length, which is why long context is expensive and retrieval still matters. (Appendix A.)

### A21. SQL or NoSQL for an e-commerce system?

Relational (Postgres/MySQL) for the core: orders, payments, inventory and customers have
fixed schemas, need transactions (an order must decrement stock and charge exactly once),
foreign keys and ad-hoc reporting. Add NoSQL where its strengths apply: a document store
for a flexible product catalogue with per-category attributes, Redis for carts and sessions,
a search engine for product search, and a vector DB for semantic search and
recommendations. Choose per workload, not per company. Say what you would actually do:
Postgres with JSONB covers most of the "flexible attributes" case without a second database.

### A22. What is an "AI harness"?

The harness is everything around the model that makes it a usable system: the **orchestration
loop** (LangGraph) that decides the next step; the **tools** it can call and their schemas;
**memory** (checkpointed state, long-term store); **retrieval**; **guardrails** (limits,
PII, injection defences, human-in-the-loop); **evals and tracing**; **prompt management**;
and the **runtime** (API, queues, budgets). The model is a function from context to text;
the harness is the program. In interviews: "the harness is the code that turns a model into
a product: and most of the engineering, and most of the quality, is in the harness."
Claude Code, deep-research agents and this book's Chapter 16 agent are harnesses.

### A23. Zero-shot vs one-shot vs few-shot prompting?

Zero-shot: instructions only, no examples ("Answer using the context; cite sources").
One-shot: one worked example of input → desired output. Few-shot: several examples,
ideally covering edge cases (an answerable question with a citation, an unanswerable one
with a refusal, a multi-source one). Examples teach *format and behaviour* better than
prose and cost tokens; with strong models zero-shot plus a clear rubric is often enough,
and few-shot is the fix when the model keeps getting the *shape* wrong (forgetting to cite,
over-answering). In RAG, retrieved chunks are not few-shot examples: they are evidence;
you can additionally retrieve *past good answers* as dynamic few-shot examples. Keep
examples in the stable prompt prefix so they are cached.

### A24. Dockerfile vs Docker Compose?

A **Dockerfile** describes how to build **one image**: base image, dependencies, copied
code, the command to run: it produces the artifact for one service (your FastAPI app).
**Docker Compose** describes how to run **several containers together** as one
application: your app *and* Qdrant *and* Postgres, with networks, volumes, environment
variables, ports and start order, in one `docker-compose.yml`, started with
`docker compose up`. Dockerfile = recipe for one dish; Compose = the menu and the table. In
production Compose is usually replaced by Kubernetes manifests, but the images from the
Dockerfiles are the same. (Chapter 17 ships both.)

---

## Part B · The deeper bank

### B1. Retrieval and indexing

**Q: How does HNSW work, and what do `m`, `ef_construct` and `ef` control?**
A: Hierarchical Navigable Small World graphs: every vector is a node with links to
neighbours; there are several layers, sparse at the top (long-range links) and dense at the
bottom. A search starts at the top, greedily walks to the nearest node, drops a layer, and
repeats; at the bottom layer it keeps a candidate list of size `ef` and returns the best
k. `m` = links per node (higher → better recall, more memory; 16–64 typical);
`ef_construct` = candidate list size *while building* (higher → better graph, slower
build); `ef` (search-time) = candidate list size while searching (higher → higher recall,
higher latency; must be ≥ k). Complexity is roughly logarithmic in N. Recall is not 100%;
you measure it against brute force on a sample. (Paper: Malkov & Yashunin 2016.)

**Q: IVF vs HNSW vs PQ?**
A: Three orthogonal ideas. **IVF** (inverted file): cluster vectors with k-means; at query
time search only the `nprobe` nearest clusters: simple, memory-light, good for huge
static sets. **HNSW**: a graph, best recall/latency at moderate scale, memory-hungry,
supports incremental inserts well. **PQ** (product quantisation): compress each vector by
splitting it into sub-vectors and replacing each with a centroid id: 8–64× smaller,
lossy; usually combined with IVF or HNSW (IVF-PQ, HNSW-PQ) and followed by rescoring on
exact vectors. Qdrant uses HNSW plus optional scalar/binary/product quantisation.

**Q: Cosine similarity vs dot product: and why normalise?**
A: Dot product = |a||b|cos θ; cosine is the dot product of unit vectors. If vectors are
L2-normalised the two are identical and cosine can be computed as a plain dot product
(faster, SIMD-friendly). Unnormalised dot product also rewards *magnitude*, which some
models use to encode confidence or length, so a long generic chunk can outscore a short
precise one. Most embedding APIs (OpenAI included) return normalised vectors; Qdrant's
`Distance.COSINE` normalises on insert. Euclidean on unit vectors is monotonic with cosine,
so the ranking is the same.

**Q: How do you choose embedding dimensionality?**
A: More dimensions → slightly higher quality, linearly more memory and bandwidth, and
slower search. Matryoshka-trained models (OpenAI text-embedding-3, Voyage, Qwen3) let you
truncate to 256–1,024 dims with small loss, because the leading dimensions carry the most
information. Procedure: run your golden set at 1,536, 1,024, 512, 256 and look at recall@k;
pick the smallest that stays within your tolerance (often 512). Combine with int8
quantisation for another 4×.

**Q: Explain BM25.**
A: A bag-of-words ranking function: score(q, d) = Σ over query terms of
IDF(t) × [tf(t,d)·(k₁+1)] / [tf(t,d) + k₁·(1 − b + b·|d|/avgdl)]. IDF weights rare
terms; term frequency saturates (k₁ ≈ 1.2–2 controls how fast) so ten mentions are not ten
times better than one; `b` (≈ 0.75) normalises for document length. It is exact-match,
fast, needs no model, and is unbeatable on identifiers, error codes, product names and rare
words: which is why hybrid search exists. (Chapter 8.)

**Q: How does hybrid search fuse results? Explain RRF.**
A: Dense and sparse scores live on different scales, so fuse *ranks*, not scores.
**Reciprocal Rank Fusion**: for each document, score = Σ over result lists of
1/(k + rank), with k ≈ 60; a document ranked high in both lists wins, one ranked high in
only one still gets credit. Robust, parameter-light, no normalisation needed; Qdrant does it
server-side with `prefetch` + `RrfQuery`. Alternative: DBSF (distribution-based score
fusion) normalises scores by their mean/std and sums: better when one retriever is much
stronger. Then rerank the fused top 20–50. (Cormack et al. 2009; Chapter 8.)

**Q: Bi-encoder vs cross-encoder vs late interaction?**
A: Bi-encoder: embed query and document *separately*, compare vectors: precomputable,
fast, less precise. Cross-encoder: feed query and document *together* through a
transformer and output a relevance score: precise, but O(candidates) model calls at query
time, so only for reranking the top 20–100. Late interaction (ColBERT): embed separately
but keep per-token vectors and score by MaxSim: precomputable *and* nearly cross-encoder
precise, at ~100× the storage. Standard stack: bi-encoder (+BM25) for candidates,
cross-encoder for the final order. (Chapters 9, 19.)

**Q: How do you choose chunk size?**
A: Empirically. Build the golden set, index the corpus at several sizes (e.g. 200, 400,
800, 1,200 tokens, with and without header-aware splitting), and compare recall@k and
end-to-end correctness. Small chunks: precise retrieval, lose context, more chunks per
answer. Large chunks: more context, diluted embeddings, more tokens per query. Typical
optimum is 300–800 tokens with 10–15% overlap for prose, whole units for tables and code.
Decouple with small-to-big: retrieve small child chunks, return their parent section.
(Chapters 5, 9.)

**Q: What is contextual retrieval / late chunking?** → see Chapter 19 §19.6.

**Q: How do you keep the index fresh and consistent with the source?**
A: Stable ids (hash of source + chunk index) so re-ingestion is an idempotent upsert;
a content hash per document so unchanged documents are skipped; deletes propagated
(delete by `doc_id` filter before re-adding); an ingestion queue triggered by source events
(webhooks, CDC) rather than nightly full crawls; a `version`/`updated_at` payload field so
you can prefer recent chunks and audit staleness; and blue/green collections with an alias
switch for full re-embeds. Eventual consistency of seconds to minutes is normal; say so.
(Chapter 7.)

**Q: How does metadata filtering interact with ANN search?**
A: Filtering *after* HNSW search can return fewer than k results (or none) when the filter
is selective. Qdrant filters *during* graph traversal and, for very selective filters,
falls back to a payload-index scan; that is why you create payload indexes and, for
multi-tenant data, mark the tenant field with `is_tenant=True` so vectors are co-located
and can have per-tenant HNSW graphs (`payload_m`, `m=0`). Always benchmark filtered recall,
not just unfiltered.

### B2. Generation and prompting

**Q: How do you order context in the prompt?**
A: Stable material first (system prompt, tool schemas, examples) so provider prompt caching
hits; then retrieved chunks; then the question last, so it is closest to the answer. Within
the chunks, either most-relevant first *and* last ("lost in the middle" says edges are
attended best), or original document order if chunks come from one document (In Defense of
RAG showed that helps). Number chunks and ask for citations by number.

**Q: How do you make the model say "I don't know"?**
A: Give it permission and a trigger: the prompt states that if the context does not contain
the answer it must say so, with an example; a retrieval-quality gate (score threshold or LLM
grade) can short-circuit *before* generation; structured output with an `answerable: bool`
field makes the decision explicit and measurable; and the golden set contains unanswerable
questions so refusal-precision and refusal-recall are tracked. (Chapter 12.)

**Q: Temperature and sampling for RAG?**
A: Low or default-deterministic settings for factual QA; you want the model to copy from
context, not be creative. Note GPT-5-family models do not accept `temperature`; use
reasoning/verbosity controls instead. Measure variance by running the eval set several
times.

**Q: How do you handle conflicting information in retrieved chunks?**
A: Detect and resolve, not average. Store `effective_date`/`version`/`authority` in the
payload; prefer or filter by recency and authority at retrieval; instruct the model to
surface conflicts ("the FAQ says 20 days, the 2026 policy says 24; the policy supersedes")
rather than pick silently; and fix the corpus: conflicting sources are a data-quality bug
you report to the owner. (Chapter 12.)

### B3. Evaluation

**Q: How do you build an evaluation dataset?**
A: Start from real user questions (logs, support tickets): 50 is enough to begin, 200–500
is comfortable. Add synthetic questions generated per chunk by an LLM, then *filter* them
(is it answerable from the chunk alone? would a real user ask it?). Label each with the
source chunk ids (for retrieval metrics) and a reference answer (for correctness). Include
20% unanswerable/out-of-scope, multi-hop, and adversarial (injected) items. Version it;
never tune on the whole set: hold out a slice. (Chapters 7, 10.)

**Q: Derive nDCG@k.**
A: DCG@k = Σᵢ₌₁ᵏ rel_i / log₂(i+1): relevance gains discounted by position (a relevant
result at rank 1 counts fully, at rank 4 counts 1/log₂5 ≈ 0.43). IDCG@k is the DCG of the
ideal ordering (all relevant items first). nDCG@k = DCG@k / IDCG@k ∈ [0, 1]. Unlike
recall it rewards *ranking* relevant items higher, and handles graded relevance. Use it
when order matters (it does: the model attends to early chunks more). (Appendix B.)

**Q: What are the known biases of LLM judges and how do you mitigate them?**
A: Position bias (prefers the first of two answers), verbosity bias (longer looks better),
self-preference (prefers its own family's style), and leniency on confident tone.
Mitigations: pairwise comparisons with both orders averaged; absolute scoring with a
concrete rubric and claim-level decomposition (faithfulness as per-claim yes/no);
reference-guided grading; a different model family as judge than as generator; and
calibration against 50–100 human labels (report Cohen's kappa; aim > 0.6). (Chapter 10.)

**Q: How would you A/B test a RAG change (e.g. new chunk size)?**
A: Offline first: run the golden set on both configurations (separate collection names),
compare recall@k, faithfulness, correctness, latency, cost; require a pre-declared
improvement. Online: route a percentage of traffic, keyed by user so a person sees one
variant; log variant id in the trace; compare thumbs-up rate, refusal rate, follow-up-
question rate (a proxy for unhelpful answers), latency; run long enough for significance;
guard with a kill switch. Report both offline and online results.

### B4. Systems and scale

**Q: Estimate memory for 10M chunks at 1,536 dims.**
A: 10M × 1,536 × 4 bytes = 61 GB of raw float32 vectors; HNSW adds ~m × 2 × 4 bytes per
vector of links (m=16 → ~1.3 GB) plus payload. int8 scalar quantisation → ~15 GB; binary →
~2 GB with the originals on disk for rescoring; Matryoshka 512 dims + int8 → ~5 GB. So a
single 64 GB node works with quantisation; without it you shard. (Chapter 18.)

**Q: How do you isolate tenants?**
A: Three levels. Payload-based: one collection, `tenant_id` in the payload with a tenant
index (`is_tenant=True`) and a mandatory filter added *server-side* from the auth token:
cheapest, fine for many small tenants. Shard-based: custom sharding with a shard key per
large tenant, so a tenant's data lives on its own shards. Collection/cluster-based: separate
collections or clusters for regulated tenants who need physical isolation. Never let the
filter be constructed from user input. (Chapter 18.)

**Q: Caching strategies for RAG?**
A: (1) Provider prompt caching by putting stable prefixes first; (2) embedding cache keyed
by text hash so re-ingestion and repeated queries do not re-embed; (3) retrieval cache
keyed by normalised query + filters + collection version; (4) semantic answer cache with a
high similarity threshold and corpus-version key; (5) HTTP/CDN caching for static
resources. Invalidate on corpus change. (Chapter 19 §19.14.)

**Q: How do you model the cost of a RAG system?**
A: Per query: embedding (~$0.02/M tokens × ~50 tokens ≈ negligible) + retrieval compute
(Qdrant node cost / QPS) + reranker + generation (input tokens ≈ system prompt + k × chunk
tokens + question; output tokens) × price. Example: 1,500 prompt + 300 output tokens on a
$0.75/$4.50 model ≈ $0.0025; at 100K queries/day ≈ $250/day. Ingestion: tokens × embedding
price, once, plus contextual-retrieval LLM calls if used. Then infra: vector DB RAM is the
dominant fixed cost: hence quantisation. Put it in a spreadsheet and show the interviewer
the levers. (Chapter 18.)

**Q: When should you NOT use RAG?**
A: When the knowledge is small and static (put it in the prompt); when the task is
reasoning or transformation over user-provided input, not lookup; when the "knowledge" is
really structured data best answered by SQL; when the model's parametric knowledge is
sufficient and freshness is not needed; when latency budgets are tighter than a retrieval
round-trip allows; and when you cannot build an eval set: you would be flying blind.
Fine-tuning is for *style and format*, not facts; it does not replace RAG for knowledge.

**Q: GraphRAG vs vector RAG?** → Chapter 19 §19.4.

**Q: Explain prompt injection through documents and defences.** → Chapter 19 §19.13.

### B5. Agents

**Q: Describe the Self-RAG / CRAG / Adaptive-RAG family in one sentence each.**
A: Self-RAG trains the model to emit reflection tokens deciding whether to retrieve and
whether the retrieved passage is relevant and the answer supported. CRAG adds a retrieval
evaluator that grades results as correct/ambiguous/incorrect and triggers web search or
knowledge refinement when they are bad. Adaptive-RAG classifies the question's complexity
and routes it to no-retrieval, single-step, or multi-step retrieval. All three are
"grade, then decide" loops; Chapter 15 implements the pattern in LangGraph.

**Q: How do you evaluate an agent, not just its answer?**
A: Trajectory metrics from traces: did it call retrieval when the question was factual
(tool-use precision/recall), number of steps and tokens per task, loop/termination rate,
tool-error rate, time to answer; plus the final-answer metrics. Build task suites with
expected tool sequences where they are deterministic; use an LLM judge with the trace for
the rest. (Chapters 13, 16.)

**Q: Checkpointing in LangGraph: why does it matter for RAG agents?**
A: A checkpointer persists graph state after every step under a `thread_id`, giving you
multi-turn memory, resumability after crashes or interrupts (human-in-the-loop), and
time-travel debugging (replay from a step). For RAG it means a follow-up question can reuse
retrieved context, and a failed generation does not redo retrieval. `InMemorySaver` for
dev, Postgres/SQLite savers in production. (Chapter 14.)

---

## Part C · System design: RAG over 10M documents at 1,000 QPS

Say the structure before the details: requirements → data → indexing → serving →
quality → operations → cost.

**Requirements.** 10M documents ≈ 200M chunks? No: clarify: 10M documents × ~20 chunks =
200M chunks is a different system from 10M chunks. Assume 10M documents, average 5 pages,
~50M chunks of 400 tokens. 1,000 QPS peak; p95 latency target 2 s end-to-end; multi-tenant
with permissions; freshness within 5 minutes of a document change; citations required.

**Ingestion.** Sources emit change events → queue (Pub/Sub/Kafka) → parser workers
(layout-aware) → chunker → contextualiser (optional LLM summary, cached by document) →
embedding workers batching 100s of chunks per API call with rate limiting → upsert to the
vector DB with stable ids and `tenant`, `acl`, `updated_at`, `doc_id`, `section` payload.
Dead-letter queue for failures; idempotent everywhere. Full re-embeds go to a new
collection behind an alias.

**Index.** Qdrant cluster: 50M × 1,536 dims float32 = 307 GB; with int8 scalar
quantisation and 512-dim Matryoshka embeddings ≈ 26 GB of quantised vectors plus originals
on disk for rescoring. Shard by tenant group (custom shard keys for large tenants,
payload partitioning for small), replication factor 2 for availability, HNSW `m=16`,
`ef_construct=200`, tenant-aware indexing. Sparse (BM25) vectors alongside for hybrid.
Payload indexes on every filtered field.

**Serving.** Stateless FastAPI/async service behind a load balancer: auth → build mandatory
ACL filter → embed query (cache) → hybrid `query_points` with prefetch dense + sparse,
RRF, limit 50 → cross-encoder rerank to top 8 (GPU reranker service, batched) → prompt
assembly with stable prefix first → LLM stream → faithfulness check on a sample or on
low-confidence answers → response with citations. Semantic answer cache in front for
repeated questions. Budget: retrieval ≤ 100 ms, rerank ≤ 150 ms, generation ~1–1.5 s.

**Capacity.** 1,000 QPS × 50 candidates rerank = 50K passage scores/s → several GPU
reranker replicas or a lighter reranker; embedding calls 1,000/s → batch and cache; LLM
1,000 concurrent streams → provider rate limits negotiated, multiple providers as fallback.
Vector DB: a few nodes with 64–128 GB RAM each handle this with quantisation; benchmark
filtered recall at target `ef`.

**Quality.** Golden set per tenant type; offline eval in CI (recall@8, nDCG, faithfulness,
correctness, refusal metrics); online signals; LLM-judge sampled 1–5% of traffic; monthly
recalibration of the judge; adversarial documents in the corpus test.

**Operations.** Tracing on every request (LangSmith/OpenTelemetry) with retrieved ids;
dashboards for p50/p95, recall proxy (reranker top score distribution), refusal rate, cost
per query; alerts on ingestion lag and index freshness; blue/green collection swaps;
runbooks for embedding-provider outage (fallback provider, degraded BM25-only mode).

**Cost sketch.** Generation dominates: 1,000 QPS × 86,400 s × 20% duty ≈ 17M queries/day ×
~$0.003 ≈ $50K/day at mini-model prices: so the real design levers are caching, a small
model for easy questions with routing to a larger one, and a short k. Infra (vector DB,
rerankers) is ~$1–3K/day. State that explicitly; interviewers want to see you know where
the money goes.

---

## Part D · The second transcript: below the API layer

A later interview for the same kind of role went one layer deeper. Where Part A asked *what you
built*, these asked *why it works and how you measured it*: and that is where most candidates,
including strong ones, come apart. Each answer below is the 30–60 second spoken version; the
chapter link is where the mechanism is derived and measured.

### D1. Your Recall@5 is 95%. Does that mean the system answers 95% of questions correctly?

No, and conflating the two is the classic mistake. Recall@5 says that for 95% of questions, a
chunk I had *labelled* relevant showed up in the top five: under my labelling, on my evaluation
set. It says nothing about what the generator did with it. The model can use only half the
evidence, be derailed by a second chunk that contradicts the first, lose the fact in the middle of
a long context, or ignore the context and answer from parametric memory. Retrieval recall is an
upper bound on answer quality. In our own harness, recall runs 0.94–1.00 while answer correctness
runs 0.79–0.83, and that gap is precisely what faithfulness, correctness and citation metrics
exist to measure. → [Chapter 21](../part8-deep-dives/21-retrieval-metrics-deep.md)

### D2. What do `M`, `ef_construction` and `ef_search` control in HNSW?

`M` is the number of neighbours kept per node in the graph. Higher `M` means a denser graph, better
recall, more memory: roughly `M × 2 × 4` bytes of links per vector: and slower construction.
`ef_construction` is the size of the candidate list while inserting a node: it buys index *quality*
at the cost of build time, and it is fixed once the index is built. `ef_search` is the same beam
width at query time, and it is the only one of the three you can change in production without a
rebuild: raising it explores more of the graph, which lifts recall and costs latency roughly
linearly until recall saturates. Sensible starting point: `M = 16`, `ef_construction = 100`,
`ef_search = 64–128`, then measure. → [Chapter 22](../part8-deep-dives/22-vector-search-internals.md)

### D3. How would you tune HNSW for 10 million vectors?

I would not start from the parameters, I would start from the requirement. First build ground truth:
take a sample of real queries, run exact brute-force search, and keep the true top-k. Then set a
recall target from the product: 0.95 or 0.99 against that ground truth. Choose `M` from the memory
budget, because `M` and the vector dimension decide RAM. Then sweep `ef_search` and pick the lowest
value that meets the recall target, reading p95 and p99 latency rather than the mean. If memory is
tight before recall is met, that is when quantization enters: scalar int8 with rescoring typically
keeps recall while cutting memory four-fold. And I would re-measure after growth: recall at a fixed
`ef` *degrades* as the index gets larger, so a setting validated at one million is not validated at
ten. → [Chapter 22 §22.5](../part8-deep-dives/22-vector-search-internals.md)

### D4. If Qdrant already returns similarity scores, why do you need a reranker?

Because those scores come from a bi-encoder, which has to compress each document into a fixed
vector *before* it has ever seen the query. That is an information bottleneck: the vector encodes
general topicality, not which entity, condition or qualifier this particular question is about. A
cross-encoder concatenates query and document and runs full attention across both, so it can tell
"payload of the A2" from "payload of the A2 Lite": but it must run one forward pass per pair, so
it cannot touch a million chunks. The resolution is a cascade: ANN gives recall over millions
cheaply, the cross-encoder gives precision over the 50–100 survivors. First stage optimises recall
and speed; second stage optimises ordering. → [Chapter 24](../part8-deep-dives/24-rerankers.md)

### D5. Bi-encoder versus cross-encoder.

| | Bi-encoder | Cross-encoder |
|---|---|---|
| Input | query and document encoded **separately** | query and document **together** |
| Output | vectors, compared by cosine or dot | a relevance score |
| Precompute documents | yes, at index time | no, at query time only |
| Cost at query | one embedding + ANN lookup | one forward pass **per candidate** |
| Scale | millions | tens to hundreds |
| Role | retrieval | reranking |

Late interaction (ColBERT) sits between them: it keeps a vector per token, precomputes documents
like a bi-encoder, and scores with MaxSim, recovering much of the cross-encoder's accuracy at a
fraction of the cost, for several times the storage.
→ [Chapter 24 §24.2](../part8-deep-dives/24-rerankers.md)

### D6. You use hybrid retrieval: how are the BM25 and dense results actually combined?

Not by adding the scores, because they are not on a common scale. BM25 is unbounded and depends on
corpus statistics, term frequencies and document length; cosine similarity lives in a bounded range
with a completely different distribution, and both shift from query to query. A weighted sum is
therefore dominated by whichever number happens to be larger, not by whichever retriever is right.
The standard answer is Reciprocal Rank Fusion: score each document as the sum over retrievers of
`1 / (k + rank)`, conventionally with `k = 60`. It uses only the ordering, so it needs no
per-corpus calibration and is robust to either retriever producing garbage scores. The cost is that
it discards magnitude (a retriever that is *confidently* right is treated the same as one that
barely preferred the document) which is what weighted RRF and distribution-based fusion try to
recover. → [Chapter 23](../part8-deep-dives/23-fusion-and-sparse.md)

### D7. The retrieved chunks are correct, but the model still hallucinates. What do you do?

Temperature would not be my first move; it changes sampling entropy, not grounding, and a model at
temperature zero still fabricates when the evidence is incomplete or contradictory. I would walk a
diagnosis tree. Is the evidence actually still in the final context after assembly and truncation,
or did it get cut? Does the prompt permit outside knowledge, or does it mandate grounding and a
fixed refusal? Are two retrieved chunks in conflict, so the model is averaging them? Is the gold
chunk buried in the middle of a long context? Is the answer a synthesis across chunks that the
evidence does not actually support? Each cause has a different fix, so I measure rather than guess:
claim-level faithfulness against the retrieved context tells me *what fraction* of the answer is
unsupported, and citation accuracy tells me whether the citations are real or decorative.
→ [Chapter 26](../part8-deep-dives/26-hallucination-forensics.md)

### D8. The correct document comes back at rank 17. What do you do?

I would resist the instinct to raise k from 5 to 20: that hides the fault and I pay the token cost
on every query forever. First I find out which of three things is wrong: the query, the index, or
the document's representation. Concretely: is the answer split across a chunk boundary; is there
vocabulary mismatch between the question and the chunk, which BM25 or a rewrite would fix; does
exact brute-force search rank it higher than the ANN index, which would make it a recall problem
rather than a ranking problem; is a metadata filter excluding it; is it crowded out by near
duplicates; and does a cross-encoder pull it into the top five, which tells me the evidence is
retrievable but misranked. The diagnosis chooses the fix, and I apply fixes in order of cost:
query rewriting and hybrid before re-chunking and re-embedding the corpus.
→ [Chapter 27](../part8-deep-dives/27-retrieval-debugging.md)

### D9. Why a chunk size of 500? Why not 1,000 or 2,000?

There is no universally right number: chunk size is an experiment variable, and the honest answer
describes the experiment. I sweep size against overlap and strategy and measure recall@k, nDCG,
answer faithfulness and correctness, plus tokens and cost per query, then pick from the Pareto
front. What moves the optimum is predictable: single-fact lookup questions favour smaller chunks
because precision rises and the fact is not diluted; synthesis questions favour larger ones;
tables and lists must not be split at all; the embedding model's maximum sequence length caps you;
and if you have a reranker you can afford to retrieve many small chunks and let it sort them. If I
need both, parent/child retrieval breaks the trade-off: search small children, give the model the
parent. → [Chapter 27 §27.5](../part8-deep-dives/27-retrieval-debugging.md)

### D10. How would you choose an embedding model?

By running a bake-off, not by reading a leaderboard. Fix the corpus, the chunking and the golden
set, vary only the model, and report recall@5, recall@10, MRR and nDCG@10 alongside embedding
latency, index memory and cost per million tokens. Then the constraints that are not on the
leaderboard: maximum sequence length versus my chunk sizes, whether it needs instruction prefixes
like `query:` and `passage:` (forgetting those quietly costs several points of recall) whether
the task is symmetric or asymmetric, licensing and self-hosting, languages, and whether dimensions
can be truncated with Matryoshka to trade a little quality for a lot of memory. MTEB is a
shortlisting tool: it tells me which five to test, never which one to ship.
→ [Chapter 25](../part8-deep-dives/25-choosing-an-embedding-model.md)

### D11. A PDF contains a tree diagram. How is it represented and indexed?

By default it is not indexed at all, and that is the failure people miss. A PDF has no semantic
structure (it is positioned glyphs plus drawing operators) so a diagram is either vector drawing
commands or a raster image, and a text extractor returns nothing for it beyond stray labels. There
are four options. Ignore it, which is the silent default. OCR it, which recovers text inside the
image but not its structure. Have a vision model caption it, index the caption as text with a link
back to the page image: and here the prompt matters enormously, because a generic caption gives
"a diagram showing a hierarchy", whereas asking explicitly for parent-to-child edges gives you the
structure that a query can actually match. Or use multimodal or late-interaction page embeddings
such as ColPali, which skip OCR and embed the page image directly: the strongest option for
figure-heavy documents, at higher storage and infrastructure cost.
→ [Chapter 28 §28.5](../part8-deep-dives/28-documents-in-the-real-world.md)

### D12. How do you index tables?

Never as flattened text, because flattening destroys the row-column association: "250 kg" stops
belonging to "Atlas A2". I keep the table intact in a structured representation (markdown or HTML
preserves alignment and models read both well) and I never split a table across chunks. For wide
or long tables, one row per chunk with the header repeated works well, because each chunk is then
a self-contained fact. A further step that helps retrieval is verbalising each row into a sentence,
which matches natural-language queries far better than a pipe-delimited row. Whichever I choose, I
measure it: the same table questions against each representation.
→ [Chapter 28 §28.4](../part8-deep-dives/28-documents-in-the-real-world.md)

### D13. How does your approach change between 10 PDFs and 1 million PDFs?

At ten, a script on a laptop is correct and anything else is over-engineering. At a million, every
stage becomes a distributed system with its own failure mode. Ingestion becomes a durable queue
with leased work items, retries with exponential backoff, and a dead-letter queue, because parsing
is the slow, flaky stage and some PDFs will crash any parser. Every step has to be idempotent and
keyed by content hash so a restart does not duplicate or double-bill; documents need versioning and
tombstones so updates and deletions propagate; near-duplicates need MinHash-style detection before
you pay to embed them. Embedding becomes a batched, rate-limited stage budgeted in tokens per
minute, with the asynchronous batch API where latency allows. The index needs sharding, quantization
and a memory plan, plus tenant isolation through payload partitioning. And you need a re-embedding
migration story (blue/green collections behind an alias) because the day you change embedding
model you cannot mix vector spaces. Finally observability per stage: lag, DLQ depth, index
freshness and cost per thousand documents. → [Chapter 30](../part8-deep-dives/30-ingesting-a-million-pdfs.md)

### D14. How do you order the chunks in the prompt, and does it matter?

It matters, and it is measurable. Models attend unevenly across a long context (strongest at the
beginning and end, weakest in the middle) so a gold chunk placed in the middle of twenty
distractors is materially more likely to be missed than the same chunk at position one. Practical
policies: most-relevant-first is the sane default; a relevance sandwich puts the two best chunks at
the start and the end; chronological ordering is right when facts supersede each other, paired with
dates in the chunk headers so the model can prefer the newer one. Beyond ordering, the assembler
should deduplicate near-identical chunks, because repetition biases the model toward the repeated
claim, and it should carry per-chunk metadata headers so citation and recency logic have something
to work with. → [Chapter 29](../part8-deep-dives/29-context-construction.md)

### D15. When does query rewriting help, and when does it hurt?

It helps when the question and the corpus use different vocabulary: user language versus internal
jargon, acronyms, or a question so terse that it embeds poorly. Expansion, HyDE and step-back
generalisation all attack that. Decomposition helps when one question needs two independent facts,
which single-shot retrieval cannot satisfy. It hurts when the query contains a precise identifier
(an error code, a ticket reference, a version number) because paraphrasing destroys the exact token
that lexical search would have matched perfectly. It also adds a model call to every request, which
is latency and cost on the critical path. My rule: route rather than rewrite everything: send
identifier-shaped queries straight to hybrid retrieval, and reserve rewriting for queries that fail
a first retrieval attempt. → [Chapter 27 §27.4](../part8-deep-dives/27-retrieval-debugging.md)

### D16. You made a change and recall went up two points. How do you know it is real?

On a golden set of fifty questions, two points is one and a half questions, which is inside the
noise band. I would report a confidence interval rather than a point estimate: bootstrap the
per-question scores to get a 95% interval, and use a paired test across the *same* questions rather
than comparing two independent means, since pairing removes question difficulty as a confound. For
LLM-judged metrics I also re-run the judge, because it is stochastic and contributes its own
variance. The practical rule: decide the minimum effect you care about, then size the eval set for
it: detecting a 2-point difference reliably needs several hundred questions, not fifty. Until
then, I trust changes that move several related metrics in the same direction, and I treat a single
metric moving alone as a hypothesis, not a result.
→ [Chapter 21 §21.5](../part8-deep-dives/21-retrieval-metrics-deep.md)

## Key takeaways

- Structure every answer: definition → mechanism → trade-off → number → production choice.
- Retrieval and generation fail differently; every debugging and evaluation answer starts by
  separating them.
- Enforce behaviour structurally (filters, forced tool calls, limits, structured output),
  not by prompt alone.
- Know the numbers: HNSW parameters, RRF's k=60, BM25's k₁ and b, bytes per vector,
  price per million tokens.
- The system-design answer is a pipeline with a budget at every stage and evals at the end.

## Next

→ [Appendix A: Research papers](../appendix/A-research-papers.md)

Appendix A gives the papers behind these answers, in the order to learn them; Appendix B is
the metrics sheet; Appendix D the API cheatsheet to keep beside you while coding.
