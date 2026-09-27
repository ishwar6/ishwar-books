# Chapter 24 · Rerankers: Bi-encoders, Cross-encoders and Late Interaction

> **Goal:** by the end of this chapter you can answer "if Qdrant already returns similarity
> scores, why rerank?" with a mechanism, a cost table and your own measurements; draw the
> bi-encoder / cross-encoder / late-interaction comparison from memory; derive your candidate
> depth from a recall curve instead of copying "retrieve 50, rerank to 5" from a blog post; and
> say precisely what reranking *cannot* fix: because on this repo's corpus, four different
> rerankers fail the same question, and the fix turns out to live somewhere else entirely.
>
> Files: [`code/ch24/bottleneck_demo.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch24/bottleneck_demo.py), `cascade_design.py`, `reranker_shootout.py`,
> `truncation_and_limits.py`. Chapter 9 §9.2 introduced reranking as an API call; this is the
> layer underneath.

---

## 24.1 The information bottleneck

Here is the whole argument in one sentence: **a bi-encoder must decide what a chunk means
before it has seen your question.**

```
   INDEX TIME                          QUERY TIME
   ──────────                          ──────────
   chunk ──► encoder ──► [0.02,        query ──► encoder ──► [0.01, ...]
                          -0.11,                                │
                          ...]  one vector                      │
                            │                                   ▼
                            └────────────── cosine ────────────►  score
                         computed WITHOUT                     one number
                         ever seeing the query
```

That single vector has to serve every question anyone will ever ask about the chunk. It
therefore encodes what the chunk is *mostly about*. A chunk whose heading matches your topic
looks better than a chunk that merely *contains your answer*.

Chapter 10 §10.8 hit this for real. The question `Which database and message broker does Beacon
use?` never got a correct answer in any configuration. [`code/ch24/bottleneck_demo.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch24/bottleneck_demo.py) reproduces
it with exact cosine search, so it is not an ANN recall artefact:

```
1. BI-ENCODER (exact cosine, so this is not an ANN recall problem)
          0.590  05-beacon-fleet-software.md      '# Beacon - Fleet Management Software (Architec'
          0.518  05-beacon-fleet-software.md      '## API'
          0.470  11-release-notes-beacon-4.2.md   '# Beacon 4.2 - Release Notes'
      ...
          0.324  <- the answering chunk, at rank 15
```

The chunk at rank 15 is the one that says:

> Persistent state in **PostgreSQL 16**; event stream on **Apache Kafka**.

It is a perfect answer. It ranks 15th out of 56 because it never uses the words "database" or
"message broker", while the title chunk of the same document says "Architecture and Operations":
close, in embedding space, to a question about architecture.

This is the bottleneck. It is not a bug in the embedding model and it is not fixed by a better
one. It is what happens when one vector must answer every possible question.

**A cross-encoder does not have this constraint.** It takes the query and the chunk as a single
input and runs attention across both, so it can ask "does this text answer *this* question?"
rather than "is this text about roughly this topic?".

```
   bi-encoder                        cross-encoder
   ──────────                        ─────────────
   [CLS] chunk tokens                [CLS] query tokens [SEP] chunk tokens
      │                                 │        ↕ attention ↕        │
      ▼                                 └────────────┬───────────────┘
   one vector (cached)                               ▼
                                              one relevance score
                                        (nothing cached; runs per pair)
```

So the cross-encoder should fix our question. Section 24.5 shows that it does not: and why
that is the most useful thing in this chapter. First, the architectures and their prices.

## 24.2 Three architectures, and what each costs

| | bi-encoder | cross-encoder | late interaction (ColBERT) |
|---|---|---|---|
| Input | query and doc **separately** | query + doc **together** | separately, but **per token** |
| Output | a vector each | one relevance score | a score from token interactions |
| Score | `E(q) · E(d)` | `f([q ; d])` | `Σᵢ maxⱼ E(q)ᵢ · E(d)ⱼ` |
| Precompute docs? | yes, one vector | **no** | yes, one vector **per token** |
| Indexable by ANN? | yes | no | yes (MaxSim-aware) |
| Work per query | one ANN lookup | one forward pass **per candidate** | cheap dot products per candidate |
| Storage per chunk | 1 × dim floats | 0 | ~100 × dim floats |
| Typical use | first stage, millions | rerank 20–100 | first stage or rerank, millions |

The late-interaction score deserves writing out, because "ColBERT" is usually said and never
explained. [`code/ch24/_shared.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch24/_shared.py):

```python
def maxsim(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    sim = query_tokens @ doc_tokens.T          # (n_q_tokens, n_d_tokens)
    return float(sim.max(axis=1).sum())
```

For every **query** token, find its best-matching **document** token, and sum those maxima. The
document keeps one vector per token instead of being crushed into one vector, so detail
survives; but the comparison is still a dot product, so it stays precomputable and indexable.
It is the middle of the design space: most of the cross-encoder's sensitivity, most of the
bi-encoder's speed, at roughly 100× the storage.

### Why you cannot simply cross-encode everything

The measured numbers, from a laptop CPU (`cascade_design.py` and the ColBERT timing in
`bottleneck_demo.py`):

| Scorer | Measured throughput | Work for ONE query over 1M chunks |
|---|---|---|
| bi-encoder + HNSW | - | **~1 ms** (Chapter 13 §13.5 measured `search_ms` p50 = 1 ms) |
| ColBERT MaxSim, brute force | ~200,000 docs/s | **~5 s** |
| cross-encoder MiniLM-L6 | 160 pairs/s | **1.7 hours** |

(The MaxSim figure is the noisiest of the three: it is a numpy matmul, so it moves by a factor
of two with machine load; `bottleneck_demo.py` prints what your machine does. The ratio is what
matters, and the ratio does not move.) A cross-encoder is roughly **1,000× slower per document
than MaxSim and a million times slower than an ANN lookup**. That is the entire reason the funnel
exists. You are not choosing between architectures; you are spending each one where its cost is
affordable:

```
  1,000,000 chunks
        │  bi-encoder + ANN        ~1 ms        (recall-oriented, cheap, blunt)
        ▼
     top 20–100
        │  cross-encoder           ~130 ms      (precision-oriented, costly, sharp)
        ▼
      top 3–8
        │  LLM                     ~1–3 s
        ▼
      answer
```

Each stage exists to buy the next stage a smaller, better input. Stage 1 is measured by
**recall** (did the answer survive?), stage 2 by **ranking quality** (is it at the top?).

## 24.3 Choosing the candidate depth, from a curve

"Retrieve 50, rerank to 5" is folklore until you derive the 50. Two facts decide it:

1. A reranker can only reorder what stage 1 returned, so **recall@N is a hard ceiling** on
   everything downstream.
2. Cross-encoder cost is **linear in N**: one forward pass per candidate.

So plot both. `cascade_design.py` runs the golden set against the ~1,000-chunk corpus
(handbook + generated documents + novel distractors) using the realistic "vague" query
phrasings from §24.4:

```
    N  recall@N  recall@5   nDCG@5     MRR  rerank ms   ceiling
    5     0.917     0.917    0.891   0.907         31   ██████████████
   10     0.964     0.952    0.925   0.935         62   ████████████████████
   20     0.988     0.964    0.937   0.945        125   ███████████████████████
   50     1.000     0.964    0.937   0.945        312   ████████████████████████
  100     1.000     0.964    0.937   0.945        623   ████████████████████████
  200     1.000     0.964    0.937   0.945       1246   ████████████████████████
```

Read it in two passes:

- **The ceiling keeps rising to N=50** (recall@N hits 1.000). Every gold document is in the
  candidate list by then.
- **Final quality stops improving at N=20** (nDCG@5 = 0.937). Going from N=20 to N=200 costs
  **+1,121 ms per query and returns +0.000 nDCG**.

That gap is the lesson. A rising ceiling is *necessary but not sufficient*: the extra documents
between N=20 and N=50 are real gold documents, but they are ones the reranker ranks below the
five it already had. Depth you cannot exploit is pure latency.

**Pick the knee, not the ceiling.** Here that is N=20 for 125 ms. If your curve looks different
(and it will, on a corpus with real near-duplicates) this script is how you find out, in about
a minute.

## 24.4 Six rerankers, measured twice

`reranker_shootout.py` runs the same candidates through every family. It runs them on **two
query sets**, and that choice matters more than the models do:

- **original**: the golden questions, which were written *from* the documents.
- **vague**: the same information needs, rewritten as someone would actually type them
  ("How much PTO do full-time folks usually get each year?"). One cached LLM call per 14
  questions; the cache makes every later run free and identical.

```
=== original queries (42 questions) ===
reranker                   recall@5  nDCG@5    MRR  ms/query   $/1k q
stage 1 only (no rerank)      0.988   0.943  0.960         0     0.00
cross-enc MiniLM-L6           0.976   0.956  0.970       129     0.00
cross-enc bge-base            0.976   0.956  0.969       787     0.00
cross-enc jina-tiny           0.952   0.931  0.949        98     0.00
ColBERT (late interact)       0.976   0.976  0.988       234     0.00

=== vague queries (42 questions) ===
reranker                   recall@5  nDCG@5    MRR  ms/query   $/1k q
stage 1 only (no rerank)      0.940   0.899  0.917         0     0.00
cross-enc MiniLM-L6           0.964   0.937  0.945       130     0.00
cross-enc bge-base            0.976   0.947  0.960       779     0.00
cross-enc jina-tiny           0.940   0.876  0.872        98     0.00
ColBERT (late interact)       0.964   0.938  0.940       229     0.00

--- LLM rerankers, vague queries, first 6 questions ---
reranker                   recall@5  nDCG@5    MRR  ms/query   $/1k q
stage 1 only (n=6)            1.000   0.917  0.889         0     0.00
LLM pointwise (N=10, n=6)     1.000   1.000  1.000      8582     2.18
LLM listwise (n=6)            1.000   1.000  1.000      1576     2.75
```

Five things an interviewer would want you to notice:

**1. On the original questions, reranking is a wash.** Stage 1 is already at 0.943 nDCG; MiniLM
adds 0.013 and *loses* recall@5. If you benchmark on questions copied from your documents, you
will conclude reranking does nothing and ship a worse system. That is a property of the
benchmark, not of reranking.

**2. On realistic phrasing, stage 1 drops 4.4 points of nDCG (0.943 → 0.899) and the rerankers
recover most of it** (bge-base back to 0.947). **A reranker's value is proportional to how noisy
its input is.** Vague queries, hybrid candidate lists full of BM25 noise, large corpora with
near-duplicates: that is when you pay for one.

**3. A reranker can make things worse.** `jina-reranker-v1-tiny` takes vague queries from 0.899
to **0.876**: below doing nothing at all. "Add a reranker" is not a safe default; it is a model
choice you must measure on your data. A tiny distilled reranker outside its training
distribution is a random number generator with a GPU bill.

**4. Quality costs latency, not linearly.** `bge-reranker-base` beats MiniLM by 0.010 nDCG and
takes **6× longer** (779 ms vs 130 ms for 20 candidates). On a 2-second answer budget that is
the difference between comfortable and tight, for one point of nDCG.

**5. The LLM rerankers are perfect and unaffordable.** Both hit 1.000 on the six-question sample:
and cost ~$2.20–$2.75 per 1,000 queries and 1.6–8.6 s per query. Note the shapes: pointwise
makes **10 sequential round trips** (8.6 s) while listwise makes **one** (1.6 s). The dollar
columns are not directly comparable here (pointwise saw N=10, listwise N=20), but the structural
point is: pointwise's cost and latency scale with N, listwise's do not. Use an LLM reranker
offline as a *judge* (Chapter 10), or online only at low QPS.

## 24.5 What reranking cannot fix

Now back to the question from §24.1, where reranking should have shone. All three cross-encoders
*and* ColBERT get the top-30 candidate list containing the answering chunk at rank 15:

```
2. RERANKERS over the top-30 candidates (the answering chunk is in there at 15)
   Xenova/ms-marco-MiniLM-L-6-v2      rank 26   top-1 '# Beacon - Fleet Management Softwa'
   BAAI/bge-reranker-base             rank 24   top-1 '# Beacon - Fleet Management Softwa'
   jinaai/jina-reranker-v1-tiny-en    rank 19   top-1 '## Breaking changes'
   ColBERT late interaction           rank 24   (MaxSim over token vectors)
```

**Every one of them makes it worse.** Rank 15 → 19, 24, 24, 26.

This is the most useful result in the chapter. The query says "database" and "message broker";
the chunk says "PostgreSQL 16" and "Apache Kafka" and contains neither query word. Joint
attention has nothing to attend *to*. A reranker is a **model with a training distribution**, not
an oracle for correctness: these were trained on MS MARCO-style web passages, where a bulleted
infrastructure list is not what an answer looks like.

What does fix it is changing the chunk's **representation**. One sentence prepended at ingest
(contextual retrieval, Chapter 9 §9.6):

> This section of the Beacon fleet-management document lists the infrastructure Beacon runs on:
> its database, its message broker/queue, its cluster and its robot transport.

```
3. CONTEXTUAL RETRIEVAL - one sentence prepended to the chunk at ingest
   bi-encoder rank: 15 -> 2   (no reranker involved)
```

**Rank 15 → 2, with no reranker in the pipeline at all.** Fixing the index beat reordering the
results. This is the order of operations from Chapter 10 §10.7 showing up as a concrete case:
retrieval representation first, then ranking, then generation.

Reranking cannot fix: content missing from the index; chunk boundaries that split the answer;
vocabulary mismatch between question and answer text; questions needing synthesis across
several chunks (Chapter 15's agentic loop); or a candidate list that never contained the answer.

### Three failure modes that bite in production

`truncation_and_limits.py` measures all three.

**Truncation.** Cross-encoders have a fixed window: 512 tokens for the ms-marco models, shared
between query and passage. Past it, text is not down-weighted; it is *unread*:

```
    prefix chars  ~tokens   needle first   needle last
               0        0         10.034        10.034
             474      118          9.196         6.981
            1896      474          7.153         2.664
            4740     1185          6.369       -11.217
            9480     2370          6.369       -11.217
           18960     4740          6.369       -11.217
```

The same sentence, in the same passage, scores **10.03 at the front and −11.22 at the back**.
And notice the last three rows are identical: adding 14,000 more characters changes nothing,
because nothing past the window is read. Parent-document retrieval (Chapter 9 §9.4) walks
straight into this: rerank the **child** you retrieved, then expand to the parent for the LLM.

**Calibration.** Cross-encoder outputs are pairwise-classifier logits, not probabilities, and
they are not comparable across queries. Eight questions that *all* retrieved a correct chunk:

```
     cosine  ce logit  question
      0.827     7.887  How many days of PTO do full-time employees get per
      0.731     1.220  How many sick days do I get and when do I need a med
      0.631    -7.953  When can I fly business class?
```

A correct top-1 scores **+7.89 for one question and −7.95 for another**. A global cut-off like
"drop anything below zero" would throw away correct answers for some questions and keep wrong
ones for others. **Order within a query: yes. Threshold across queries: no**: which is exactly
why Chapter 12 calibrates its abstention threshold on the first stage's cosine instead.

**Latency tails.** N=20 candidates on this CPU: p50 126 ms, p95 141 ms, max 146 ms. Budget with
p95, and remember this sits *in series* with the embedding call and generation (Chapter 13
§13.5). If the reranker has not answered in X ms, return the stage-1 order and log it: a
slightly worse ranking beats a timeout.

## 24.6 Training or fine-tuning a reranker

Usually you should not. Buy a better model first, fix your representation second (§24.5), and
only then consider training. When you do:

**Data.** Triples of `(query, positive passage, hard negatives)`. The positives come from your
logs: clicked results, thumbs-up answers, the golden set. The magic is in the **hard
negatives**: mine them from your own first stage by retrieving top-50 for each training query
and taking the high-ranked passages that are *not* the positive. Random negatives teach nothing
(the model learns "is this even the same topic?"); hard negatives teach the fine distinctions
that are exactly what a reranker is for. Roughly 4–8 hard negatives per positive is standard.

**Losses.** Pointwise cross-entropy (relevant/not) is simplest and weakest. Pairwise margin
(`max(0, margin − s⁺ + s⁻)`) directly optimises the ordering you care about. Listwise losses
(LambdaRank-style, optimising nDCG) are strongest when you have graded labels. Most open
rerankers are trained pairwise or with a listwise distillation objective.

**Distillation.** A cheap and underrated move: use a strong cross-encoder (or an LLM) to score
many `(query, passage)` pairs, then train your *bi-encoder* to match those scores. You push
reranker quality into stage 1, where it costs nothing at query time. This is how most modern
retrievers (E5, BGE) are actually trained.

**When it is worth it.** You have ≥10k domain queries with feedback; an off-the-shelf reranker
measurably underperforms on your data (like jina-tiny above); and the domain vocabulary is
genuinely unusual (clinical codes, legal citations, internal part numbers). Expect single-digit
nDCG points, not miracles, and budget for maintaining the model as your corpus drifts.

## 24.7 Production notes

- **Batch.** `TextCrossEncoder.rerank` takes a `batch_size`; one batch of 20 pairs is far cheaper
  than 20 calls. Never loop one pair at a time.
- **Quantise.** ONNX int8 cross-encoders run 2–4× faster on CPU for a fraction of a point of
  nDCG. fastembed ships ONNX models by default, which is why 160 pairs/s is achievable on a
  laptop CPU at all.
- **GPU only if you need it.** At 20 candidates and ~130 ms, CPU is fine to a few hundred QPS.
  GPU matters when N is large or the model is (bge-reranker-large and up).
- **Cache.** Rerank scores are deterministic for a `(query, chunk)` pair. Repeated queries are
  common; a small LRU on `(query_hash, chunk_id)` removes the stage entirely for them.
- **Degrade gracefully.** Timeout → stage-1 order. Model fails to load → stage-1 order plus an
  alert. Never let the reranker take the request down.
- **Log the reordering.** Store both orders for sampled requests (Chapter 13). "The reranker
  moved the gold chunk down" is a class of regression you cannot see from answer quality alone.

## The interview answer

**"If Qdrant already gives similarity scores, why do you need to rerank?"**

> Because those scores come from a bi-encoder, which had to compress each chunk into one vector
> *before it ever saw my query*. That vector encodes what the chunk is mostly about, so a chunk
> whose heading matches the topic can outrank a chunk that actually contains the answer. I have
> a case in my own corpus: the chunk naming PostgreSQL and Kafka ranks 15th for "which database
> and message broker does Beacon use", because it never uses the words "database" or "message
> broker", while the document's title chunk does. A cross-encoder reads the query and the chunk
> together with full attention, so it scores "does this answer *this* question" instead of
> "same topic". I can't run that over a million chunks: I measured 160 pairs a second, so a
> million chunks would be about 1.7 hours for one query: so the pattern is a cascade: ANN
> retrieves the top 20–50 in about a millisecond, the cross-encoder reranks those in about 130
> milliseconds, and the LLM sees five.

**"Bi-encoder versus cross-encoder?"**: draw the table from §24.2: separate versus joint
encoding, precomputable versus not, ANN-able versus not, O(1) versus O(N) per query, and
mention late interaction as the middle point that keeps one vector per token and scores with
MaxSim.

**"How deep should the candidate set be?"**

> I measure two curves: recall@N, which is the ceiling because a reranker can only reorder what
> it was given, and final nDCG@5 after reranking each depth. On my corpus recall@N keeps rising
> to 1.0 at N=50, but final nDCG stops improving at N=20: going to N=200 costs a second of extra
> latency for zero gain. So I take the knee, not the ceiling, and I re-measure it when the corpus
> or the first stage changes.

And the answer that separates seniors: **say what reranking cannot fix.** On that PostgreSQL
question all three cross-encoders and ColBERT made the rank *worse*; one sentence of contextual
prefix at ingest moved it from 15 to 2. Rerankers reorder; they do not repair representations.

## Run it

```bash
QDRANT_MODE=memory uv run python code/ch24/bottleneck_demo.py        # ~40 s, a few cents
QDRANT_MODE=memory uv run python code/ch24/cascade_design.py         # ~90 s, free after cache
QDRANT_MODE=memory uv run python code/ch24/reranker_shootout.py      # ~5 min, ~$0.05
QDRANT_MODE=memory uv run python code/ch24/truncation_and_limits.py  # ~30 s, free
```

The first run downloads three cross-encoders and a ColBERT model (~400 MB total) into
`~/.cache/fastembed`; after that everything except the OpenAI embeddings is local and free.
`reranker_shootout.py --limit 12 --llm-limit 4` cuts it to about a minute.

Expected: the answering chunk at dense rank 15 and every reranker making it worse; the cascade
knee at N=20; jina-tiny scoring *below* no-rerank on vague queries; the truncation table where
the last three rows are identical.

## Exercises

1. Add a fourth control probe to `bottleneck_demo.py` that the bi-encoder gets wrong. Hint: look
   for a chunk whose answer is a number or an identifier with no surrounding vocabulary. Then
   check whether contextual prefixing fixes it too.
2. Re-run `cascade_design.py` with `encode_passages(..., model_name="sentence-transformers/all-MiniLM-L6-v2")`:
a weaker stage 1. Does the knee move right? Explain why in one sentence.
3. In `reranker_shootout.py`, feed the rerankers a **hybrid** candidate list (Chapter 23's RRF of
   BM25 + dense) instead of dense-only. Does reranking help more or less? This is the setting
   most production systems are actually in.
4. Implement pairwise LLM reranking (ask the model "is A or B more relevant?" over a tournament)
   and add it to the shootout. Compare its cost and quality against pointwise and listwise.
5. Take the truncation experiment and find *your* model's real limit: binary-search the prefix
   length where the needle-last score first equals the fully-truncated score. Compare with the
   512 tokens the model card claims.
6. Build the per-query calibration the chapter says you need: instead of a global threshold, use
   the margin between top-1 and top-2 cross-encoder scores as an abstention signal, and evaluate
   it against Chapter 12's unanswerable questions.

## Interview questions

**Q: Why rerank if the vector store already returns scores?**
Because the vector store's scores come from a bi-encoder that compressed each document into one
vector without seeing the query, so it measures topical similarity rather than whether the text
answers the question. A cross-encoder processes query and document jointly, which is far more
accurate but costs one forward pass per candidate: measured at 160 pairs/s on CPU, or 1.7 hours
for a million chunks. So you retrieve cheaply and widely, then rerank a small candidate set.

**Q: Bi-encoder vs cross-encoder: the table.**
Bi-encoder: separate encoding, one vector per document precomputed at index time, dot-product
scoring, ANN-indexable, O(1) per document at query time, used for first-stage retrieval over
millions. Cross-encoder: query and document concatenated into one input with cross-attention, no
precomputation possible, O(N) forward passes, not indexable, used to rerank 20–100 candidates.
Late interaction (ColBERT) sits between: one vector per *token* precomputed, scored with
MaxSim = Σᵢ maxⱼ qᵢ·dⱼ, indexable, ~100× the storage.

**Q: How do you pick the candidate depth N?**
Plot recall@N (the ceiling: a reranker can only reorder what it received) and final nDCG@5
after reranking, against N, on realistic queries. Take the knee of the *final* curve, not the
point where recall saturates. On my corpus recall@N reaches 1.0 at N=50 but final nDCG plateaus
at N=20, so N=200 buys a second of latency and zero quality.

**Q: What is late interaction and when would you use it?**
ColBERT keeps a vector per token instead of per document and scores with MaxSim: for each query
token take its best-matching document token and sum. It preserves fine-grained term matching
while staying precomputable and ANN-indexable, so it can serve as a first stage or a very fast
reranker. The cost is storage (roughly 100 vectors per chunk) and more complex indexing. In my
measurements MaxSim scored on the order of 200,000 docs/s versus 160 pairs/s for a
cross-encoder: about a thousand to one.

**Q: Does adding a reranker always improve results?**
No. On vague queries in my corpus, `jina-reranker-v1-tiny` scored 0.876 nDCG@5 against 0.899 for
no reranking at all: it actively hurt. A reranker is a model with a training distribution; if
your domain is far from it, or your passages don't look like what it was trained on, it
misranks. Always measure the with/without comparison on your own data before shipping.

**Q: A reranker didn't fix your bad retrieval. What next?**
Check whether the problem is *ranking* or *representation*. If the answering chunk shares no
vocabulary with the question, no reranker can help: attention needs something to attend to. In
my corpus, a chunk saying "PostgreSQL 16" and "Apache Kafka" ranked 15th for "database and
message broker"; three cross-encoders and ColBERT all pushed it further down, but one contextual
sentence prepended at ingest moved it to rank 2. Fix the index before you re-order the results.

**Q: Why can't you threshold on cross-encoder scores?**
They are logits from a pairwise classifier, not calibrated probabilities, and their scale shifts
per query. I measured correct top-1 results scoring +7.89 for one question and −7.95 for
another. A global cut-off would discard correct answers for some queries and admit wrong ones
for others. Use them to order within a query; for abstention, calibrate on the first stage's
similarity or on the margin between top-1 and top-2.

**Q: What breaks when you rerank long documents?**
Cross-encoders truncate at a fixed window (512 tokens for ms-marco models). Content past it is
not read at all. I measured the same sentence scoring 10.03 when placed at the start of a
passage and −11.22 at the end, with the score unchanged as the passage grew further: proof
nothing beyond the window is seen. With parent-document retrieval you must rerank the child
chunk and only then expand to the parent.

**Q: Pointwise, pairwise or listwise LLM reranking?**
Pointwise scores each candidate independently: simple, parallelisable, but N sequential calls
(8.6 s for 10 candidates in my run) and no cross-candidate comparison. Pairwise compares two at
a time: accurate, but O(N log N) calls at best. Listwise puts all candidates in one prompt and
returns a ranking: one call, 1.6 s, and the model can compare candidates directly; limited by
context length and position bias. Listwise is the usual production choice; all of them are
expensive enough (~$2–3 per 1,000 queries) that they belong offline or at low QPS.

**Q: How would you train a domain reranker?**
Collect (query, positive) pairs from logs and the golden set, then mine **hard negatives** by
retrieving top-50 with the current first stage and taking high-ranked non-positives: random
negatives teach nothing useful. Train with a pairwise margin or listwise objective. Better
still, distil: score many pairs with a strong cross-encoder or LLM and train the *bi-encoder* to
match, moving quality into stage 1 where it is free at query time. Expect single-digit nDCG
gains, and only attempt it after representation and model-selection fixes are exhausted.

**Q: How do you keep reranking from hurting latency?**
Cap N at the measured knee, batch all candidates in one call, use an ONNX-quantised model, cache
scores per (query, chunk) pair, and set a timeout that falls back to stage-1 order. Budget on
p95, not the mean, and remember the stage is in series with embedding and generation: my p95
for N=20 was 141 ms against an end-to-end budget of ~2 s.

## Key takeaways

- A bi-encoder compresses a chunk into one vector before seeing the query, so it ranks by "what
  is this mostly about"; a cross-encoder reads the pair jointly and ranks by "does this answer
  the question". That difference is the entire reason reranking exists.
- The cascade is an economic argument: ~1 ms for ANN over a million chunks, ~130 ms for a
  cross-encoder over 20, ~1.7 hours for a cross-encoder over a million. Spend each model where
  it is affordable.
- Derive N from a recall@N ceiling curve *and* a final-quality curve. Recall saturating is not a
  reason to go deeper; my knee was N=20 while recall kept climbing to N=50.
- Reranking is not free and not always positive: one reranker scored below no-reranking, and the
  best one cost 6× the latency of the fastest for one nDCG point. Measure on queries that look
  like your traffic, not on questions copied from your documents.
- Rerankers reorder; they do not repair. Vocabulary mismatch, truncation past 512 tokens and
  cross-query score incomparability are structural limits: fix representation, chunking and
  calibration instead.

## Next

→ [Chapter 25: Choosing and Evaluating an Embedding Model](25-choosing-an-embedding-model.md)

Reranking improves the order of what stage 1 found. Chapter 25 is about stage 1 itself: how to
choose the model that decides what is findable at all, with a framework instead of a vibe.
