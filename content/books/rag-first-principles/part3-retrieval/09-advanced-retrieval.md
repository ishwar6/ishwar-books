# Chapter 9 · Advanced Retrieval: Re-ranking, Query Transforms, Parent/Child, Metadata

> **Goal:** by the end of this chapter you have a toolbox of seven retrieval upgrades, each implemented in plain functions, each with a *symptom* that tells you when to reach for it. You will know why a cross-encoder beats a bi-encoder, when to rewrite the query instead of the index, why small chunks should be searched but big chunks should be read, and how to make "the newer policy wins" a rule instead of a hope.
>
> Files: [`code/ch09/reranking.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch09/reranking.py), [`code/ch09/query_transforms.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch09/query_transforms.py), [`code/ch09/parent_child.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch09/parent_child.py), [`code/ch09/metadata_filtering.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch09/metadata_filtering.py)

---

## 9.1 The shape of a mature retriever

Chapter 6's retriever was one step: embed → nearest-k. Every production system ends up as a **funnel**:

```
query ──► [rewrite / expand] ──► [retrieve wide: dense + sparse, filtered] ──► [re-rank] ──► top-n ──► LLM
             cheap LLM call          k = 20–50, milliseconds                 slow, accurate     3–8
```

Each stage exists because the previous one has a specific weakness. This chapter is that funnel, stage by stage.

## 9.2 Re-ranking: bi-encoder vs cross-encoder

**Bi-encoder** (what an embedding model is): encodes the query and the chunk *separately* into vectors, compares them with a dot product. The chunk vector is computed once at index time, so search is a nearest-neighbour lookup: fast for millions of chunks. The price: the model never sees the query and chunk together, so it cannot do fine-grained matching ("does this table row answer *this* number?").

**Cross-encoder**: takes `(query, chunk)` as *one* input and outputs a relevance score. It attends across both texts, so it is far more accurate: and it must run once per candidate at query time, so it is far too slow to run on the whole collection.

Hence the two-stage pattern: **retrieve wide (k=20) with the bi-encoder, re-rank narrow (top 5) with a cross-encoder**.

[`code/ch09/reranking.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch09/reranking.py) implements both kinds of re-ranker:

```python
from fastembed.rerank.cross_encoder import TextCrossEncoder
_cross = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")   # 90 MB, runs on CPU

def rerank_cross_encoder(query, docs, top_n=5):
    scores = list(_cross.rerank(query, [d.page_content for d in docs]))
    return sorted(zip(docs, scores), key=lambda x: -x[1])[:top_n]
```

and the LLM itself as a **listwise** re-ranker: one structured call that scores all 20 candidates:

```python
class RerankResult(BaseModel):
    scores: list[ChunkScore]        # {index, score 0–10}

llm = get_llm().with_structured_output(RerankResult)
result = llm.invoke(RERANK_PROMPT.format(question=query, chunks=listing))
```

Output on the 120-robot pricing question (q32):

```
dense top-5 (bi-encoder):
    10-pricing-and-plans.md | ## Software (per robot, per month)
    10-pricing-and-plans.md | ## Implementation services
    01-company-overview.md  | ## Products
    10-pricing-and-plans.md | # Pricing and Plans (Internal Price Book)
    05-beacon-fleet-software.md

cross-encoder top-5 (167 ms for 20 pairs):
      6.78 10-pricing-and-plans.md | ## Software (per robot, per month)
      3.80 01-company-overview.md  | ## Products
      1.12 10-pricing-and-plans.md | ## Implementation services
     -2.35 01-company-overview.md  | ## Scale
     -4.15 10-pricing-and-plans.md | # Pricing and Plans

LLM re-ranker top-5 (1.7 s, one call):
      10.0 10-pricing-and-plans.md | ## Software (per robot, per month)
       2.0 01-company-overview.md  | ## Products
       0.0 ...
```

Notice the LLM's scores: one 10, one 2, everything else 0. An LLM re-ranker is **decisive**: great for precision, and it also tells you when *nothing* is relevant (all zeros → abstain, Chapter 12). The cross-encoder is 10× faster and free.

MRR@5 over the golden set: dense 1.000, cross-encoder 0.988, LLM 1.000. On this corpus dense already ranks the gold chunk first, so re-ranking cannot help and the small cross-encoder even demotes one chunk. That is the expected result on an easy benchmark: re-ranking earns its cost when first-stage precision is *low*: large corpora, near-duplicate chunks, hybrid candidate lists full of BM25 noise. Chapter 10's eval table shows the same thing with generation metrics.

| Re-ranker | Latency (20 cands) | Cost | Quality | Use when |
|---|---|---|---|---|
| cross-encoder (MiniLM, local) | ~150 ms CPU | free | good | default second stage |
| bigger cross-encoder (bge-reranker-v2, Cohere Rerank API) | 200–800 ms | $ | better | quality-critical |
| LLM listwise | 1–3 s | $$ | best + explains | low QPS, or as a *judge* offline |

> The architectures behind §9.2 (the bi-encoder information bottleneck, cross-encoders, late
> interaction, LLM rerankers, and how deep the candidate set should be) are
> [Chapter 24](../part8-deep-dives/24-rerankers.md). Diagnosing *which* of these techniques a given
> failure needs is [Chapter 27](../part8-deep-dives/27-retrieval-debugging.md).

## 9.3 Query transforms: fix the query, not the index

[`code/ch09/query_transforms.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch09/query_transforms.py): four plain functions, each `prompt | llm`.

### Multi-query

Vague or informally phrased questions embed poorly. Generate 3 paraphrases, retrieve for each, fuse with RRF:

```python
multi_chain = multi_prompt | llm.with_structured_output(Queries)   # Queries = list[str]
variants = [question] + multi_chain.invoke({"question": question}).queries
return rrf([store.similarity_search(v, k=k) for v in variants])[:k]
```

```
Q: can I get money back for using my own car
   variants: ['Can I be reimbursed for using my personal car for work?',
              'Am I eligible for mileage reimbursement when I drive my own vehicle?',
              'How do I claim compensation for driving my own car for business use?']
```

Costs one cheap LLM call and 3× the embedding/search calls; raises recall on sloppy queries. (`MultiQueryRetriever` in `langchain_classic` does this; you now know it is 6 lines.)

### HyDE: Hypothetical Document Embeddings

Questions and answers live in different regions of embedding space ("what does the robot do when Wi-Fi drops?" vs "a robot that loses connectivity parks itself"). HyDE asks the LLM to *write a plausible answer* and embeds **that** instead of the question. The fake answer may be wrong in its details: it only has to be *shaped* like the real chunk:

```
Q: robot wifi dropped mid-task, what does it do
  baseline: ['12-postmortem...', '08-runbook...', '04-atlas-a2-specification.md']
  hypothetical: "If a robot loses Wi-Fi during a task, it should automatically switch to its local fallback behavior..."
  HyDE:     ['04-atlas-a2-specification.md', '12-postmortem...', '08-runbook...']
```

The spec (the chunk with the real rule) moved from rank 3 to rank 1. HyDE shines on statement-shaped corpora (specs, policies) and hurts on corpora where the LLM's guess is confidently wrong-domain. Never show the hypothetical to the user.

### Step-back prompting

A specific question ("Can I fly business to Austin from Pune?") is answered by a general rule ("flights ≥ 6 h may be business"). Ask the LLM for the *general* question, retrieve for both, fuse:

```
   step-back question: What are the flight class rules?
```

### Decomposition for multi-hop

Some questions need two facts from two places. Split, retrieve per sub-question, answer over the union:

```
Q (q42): If I am a new Sales hire, can I take two weeks off in the last week of the quarter during my first month?
   sub-questions: ['Can a new Sales hire take two weeks off ... ?',
                   'What company policy applies to new Sales hires requesting extended time off during quarter-end?',
                   'Are there ... probation-period restrictions on taking PTO in the first month ...?']
  answer: No. ... probation for the first 6 months ... at most 5 days ... last week of each quarter is a PTO
          blackout period for Sales and Support ...
  gold  : No. During probation at most 5 days of PTO can be taken, and the last week of the quarter is a blackout period for Sales.
```

Correct. But look at q32:

```
   sub-questions: ['What is the monthly per-robot price for Beacon and Compass ...?',
                   'Does the pricing ... include any setup, minimums, or additional fees ...?',
                   'What is 120 times the monthly per-robot price ...?']
  answer: ... 120 × $220 = $26,400 per month
  gold  : ... $26,400; with the 15% discount for fleets over 100 robots it is $22,440 per month.
```

**Partially wrong**: the decomposer never asked about discounts, so the discount chunk was never retrieved. Decomposition is only as good as the sub-questions; the fix is either a broader union (also retrieve for the original question) or an agent that can notice "the worked example mentions a discount" and ask a follow-up: Chapter 15.

## 9.4 Parent/child retrieval (small-to-big)

There is a tension in chunk size:

- **Small chunks embed well.** One idea → one clean vector → precise matching.
- **Big chunks read well.** The LLM gets the surrounding table header, the section title, the caveat two lines below.

Parent/child resolves it: **index the small, return the big**. [`code/ch09/parent_child.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch09/parent_child.py) splits each 800-char parent into ~200-char children that carry `parent_id` in metadata, indexes only the children, and swaps back to parents at query time:

```python
def retrieve_parents(store, parent_by_id, query, k_children=8, k_parents=3):
    seen, out = set(), []
    for child in store.similarity_search(query, k=k_children):
        pid = child.metadata["parent_id"]
        if pid not in seen:
            seen.add(pid); out.append(parent_by_id[pid])
        if len(out) == k_parents:
            break
    return out
```

```
56 parent chunks (~800 chars) → 276 child chunks (~200 chars) indexed

Q: What is the P1 response time for Platinum support?
children hit (what we SEARCH):
   09-support-sla.md | '| **Platinum** | 24×7 | **15 minutes** | **4 hours** | 1 hour | **dedicated Technical Acco'
   09-support-sla.md | 'If a response target is missed, the customer can escalate ...'
parents returned (what the LLM READS):
   09-support-sla.md | 792 chars | '# Customer Support Tiers and SLAs ...'     ← includes the table HEADER row
```

The child that matched is a single table row without its header: useless to read alone ("15 minutes" of what?). The parent carries the header. That is the whole point. Doc-level hit-rate barely changes (1.00 → 0.98); the win is in **answer correctness**, which Chapter 10 measures. Storage cost: 5× more vectors; parents can live in a plain key-value store or as Qdrant payload.

Variants: **sentence-window retrieval** (index sentences, return ±N neighbours) and **auto-merging** (if enough siblings match, return the grandparent).

## 9.5 Metadata filtering and self-query

Similarity cannot express "only HR documents", "effective 2026 or later", "not the FAQ". Metadata can. Qdrant stores metadata as **payload** and applies filters *during* the HNSW search (pre-filtering), so filtering never silently drops your recall the way a post-filter on top-k does.

[`code/ch09/metadata_filtering.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch09/metadata_filtering.py) adds `category` and `effective_year` to every chunk and shows four patterns.

**1. Hand-written filters**

```python
only_hr = models.Filter(must=[models.FieldCondition(key="metadata.category", match=models.MatchValue(value="hr"))])
store.similarity_search(q, k=3, filter=only_hr)
not_faq = models.Filter(must_not=[models.FieldCondition(key="metadata.doc_id", match=models.MatchValue(value="14-faq"))])
```

Note the key: LangChain stores your metadata under the `metadata.` prefix in the payload. On a Qdrant *server* create a payload index for every field you filter on (`client.create_payload_index`): the embedded local mode ignores indexes, a server needs them for speed.

**2. Self-query**: let the LLM extract the filter from the question with structured output:

```python
class QueryPlan(BaseModel):
    search_text: str
    category: Literal["hr", "finance", "product", ...] | None = None
    min_year: int | None = None
```

```
Q: What does the 2026 leave policy say about carry-over?
plan: {'search_text': 'leave policy carry-over', 'category': 'hr', 'min_year': 2026}
   → ['02-pto-and-leave-policy.md(2026)'] ×3
Q: How much does the robot weigh?
plan: {'search_text': 'how much does the robot weigh', 'category': 'product'}
```

The `Literal` type is doing real work: the model cannot invent a category that does not exist. Use `models.Range(gte=...)` for numeric fields, `MatchAny` for lists.

**3. Recency preference**: the FAQ (2024, "20 days") vs the policy (2026, "24 days"):

```
by score  : ['02-pto(2026) 0.55', '02-pto(2026) 0.53', '14-faq(2024) 0.46', '02-pto(2026) 0.45']
recency   : ['02-pto(2026)', '02-pto(2026)', '02-pto(2026)', '14-faq(2024)']
```

`recency_rerank` sorts by score bucket, then by `effective_year` descending: near-ties resolve to the newer document. In Chapter 11 the LLM also *sees* the year and is told to prefer it; belt and braces.

**4. MMR (Maximal Marginal Relevance)**: for broad questions, plain similarity returns four near-identical chunks. MMR trades a little relevance for diversity: `as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5})`.

```
Q: tell me about Beacon
similarity: ['05-beacon', '11-release-notes', '14-faq', '05-beacon']
mmr       : ['05-beacon', '14-faq', '07-onboarding-guide', '11-release-notes']
```

`lambda_mult=1` is pure similarity, `0` is pure diversity.

## 9.6 Contextual retrieval

A chunk that says "The rate limit is 600 requests per minute" does not say *what* has that rate limit. Anthropic's **contextual retrieval** (2024) prepends a one-sentence, LLM-written situating context to each chunk *before* embedding: "This chunk is from the Beacon fleet software document, section API." Retrieval failures dropped by ~35% in their measurement (49% when contextual BM25 was added for hybrid, 67% with re-ranking on top). Cost: one LLM call per chunk at index time, cacheable.

Try it as Exercise 4 below: the 56-chunk handbook costs a few cents. Measure hit-rate@1 with `run_eval.py` (Chapter 10) before and after.

## 9.7 Decision table: symptom → technique

| Symptom (from your eval or your logs) | First thing to try |
|---|---|
| right doc in top-20, wrong chunk in top-3 | re-ranking (cross-encoder) |
| users type ids / commands / error strings | hybrid (Ch. 8) + exact-match payload filter |
| informal or vague questions miss | multi-query |
| question phrasing ≠ document phrasing (Q vs statement) | HyDE |
| very specific question, general rule in the docs | step-back |
| two facts from two places needed | decomposition; or an agent (Ch. 15) |
| retrieved chunk is a table row / fragment without context | parent/child, or contextual retrieval |
| stale doc outranks current one | `effective_date` metadata + recency preference |
| answers mix departments / products / tenants | metadata filter (self-query) |
| top-k are near-duplicates | MMR, or dedupe by parent |
| everything is slow | fewer stages; cache query embeddings; Qdrant-native hybrid |

## Run it

```bash
QDRANT_MODE=memory uv run python code/ch09/reranking.py            # downloads the MiniLM cross-encoder (~90 MB) once
QDRANT_MODE=memory uv run python code/ch09/query_transforms.py
QDRANT_MODE=memory uv run python code/ch09/parent_child.py
QDRANT_MODE=memory uv run python code/ch09/metadata_filtering.py
```

Each prints the before/after lists shown above. `reranking.py` ends with MRR@5 for dense / cross-encoder / LLM over the golden set (~42 LLM calls, a few cents).

## Exercises

1. In `reranking.py`, retrieve k=20 with the **hybrid** store from Chapter 8 instead of dense, then re-rank. Does the cross-encoder now change MRR? (It should: BM25 adds noise the re-ranker can remove.)
2. Fix the q32 decomposition failure: change the decomposition prompt so it always adds a sub-question about "discounts, conditions or exceptions". Re-run.
3. Turn `retrieve_parents` into sentence-window retrieval: index sentences, return the sentence plus its two neighbours.
4. Implement contextual retrieval: for each chunk, ask the LLM for a one-sentence context given the whole document (`load_handbook()`), prepend it, index into `ch09_contextual`, and compare hit-rate@1 on the golden set with `ragbook.build_handbook_index`.
5. Add `effective_year` to the self-query plan as a `max_year` too, and ask "what did the 2024 FAQ say about PTO?". Does the filter route you to the FAQ?

## Interview questions

**Q: What is a re-ranker and why not just retrieve better in the first place?**
A cross-encoder that scores (query, chunk) pairs jointly: more accurate than comparing two independently computed vectors, but O(candidates) model calls at query time. You cannot run it over a million chunks, so you retrieve a wide candidate set cheaply and re-rank only that. Two stages, each doing what it is good at.

**Q: Bi-encoder vs cross-encoder?**
Bi-encoder: encode query and document separately, compare vectors; documents pre-computed, search is ANN lookup. Cross-encoder: one forward pass over the concatenated pair, outputs relevance; sees token-level interactions, so much more precise, but nothing can be pre-computed. Embedding models are bi-encoders; re-rankers are cross-encoders.

**Q: What is HyDE and when does it fail?**
Generate a hypothetical answer with the LLM and embed that instead of the question, because answers are closer to answer-chunks than questions are. It fails when the LLM's guess is in the wrong domain or style (e.g. it writes marketing prose for a legal corpus), and it adds a generation call of latency. Good for policy/spec corpora; verify on your eval set.

**Q: How do you handle multi-hop questions in RAG?**
Decompose into sub-questions, retrieve for each, answer over the union: or let an agent iterate: retrieve, read, decide what is still missing, retrieve again (Chapter 15). Plain single-shot retrieval fails because the second fact is never mentioned in the question, so it is never in the query embedding.

**Q: Small chunks or big chunks?**
Both: search small, read big. Small chunks give precise embeddings; big chunks give the LLM the context (headers, caveats) it needs to answer correctly. Parent/child (small-to-big) retrieval indexes children with a parent id and returns parents. Contextual retrieval is the alternative: make the small chunk self-describing before embedding it.

**Q: A stale document keeps outranking the current one. How do you fix it?**
Put `effective_date`/`version` in metadata at ingestion; filter or boost by it at retrieval (prefer newer on near-ties); pass the date into the prompt and instruct the model to prefer newer sources and flag conflicts; and, upstream, retire superseded documents from the index. Detection comes from evals with conflict questions like our q01.

**Q: How does metadata filtering interact with vector search performance?**
Naive post-filtering (search top-k, then drop non-matching) loses recall when the filter is selective. Qdrant pre-filters inside the HNSW traversal using payload indexes, so you get exactly k matching results with little overhead: as long as the filtered field is indexed. For extremely selective filters (a single tenant) use tenant-aware indexing (Chapter 18).

**Q: What is MMR?**
Maximal Marginal Relevance: pick results that are relevant to the query *and* dissimilar to results already picked, weighted by λ. Use it when the top-k are redundant (near-duplicate chunks) and the LLM benefits from breadth.

## Key takeaways

- Production retrieval is a funnel: rewrite → retrieve wide (hybrid, filtered) → re-rank → few chunks to the LLM.
- Cross-encoders re-rank far better than bi-encoders can rank, at O(candidates) cost: so only re-rank 20–50.
- Query transforms (multi-query, HyDE, step-back, decomposition) are 6-line functions; use them for the specific symptom they fix.
- Search small, read big: parent/child or contextual retrieval gives the LLM chunks it can actually use.
- Metadata is how you express what similarity cannot: department, tenant, date. Filter in Qdrant with payload indexes; let the LLM extract filters with structured output.

## Next

→ [Chapter 10: Evaluating RAG: Every Metric, How to Compute It, How to Move It](../part4-quality/10-evaluation.md)

Chapter 10: how to *know* which of these helped: every retrieval and generation metric, implemented by hand, and an eval harness that turns "I think it's better" into a table.
