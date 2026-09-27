# Chapter 30 · Ingesting a Million PDFs

> **Goal:** you can answer "how does your approach change between 10 PDFs and a million?" as a
> system design rather than a list of adjectives: with the arithmetic that says what breaks
> first, a work-item state machine that survives crashes and poison documents, dedup that
> catches near-copies, batching and rate-limit maths that set your real ceiling, an index
> layout that holds at a million tenants' worth of data, and a re-embedding migration that
> nobody notices. All of it runs on your laptop at small scale, with every mechanism intact.
>
> Files: [`code/ch30/ingest_queue.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch30/ingest_queue.py), `dedup_minhash.py`, `batch_and_ratelimit.py`,
> `migrate_embeddings.py`.

[Chapter 18](../part6-scale/18-terabyte-scale.md) sized the *index*: vectors, RAM,
quantization. This chapter is about getting the documents into it, which is where the actual
engineering time goes.

---

## 30.1 What changes, by order of magnitude

The honest answer to the interview question is not "you add a queue". It is that different
things break at each order of magnitude, and knowing *which* is the signal you are speaking
from experience. Assume a 20-page PDF, ~60 chunks of ~250 tokens, and the Chapter 18 numbers
(text-embedding-3-small at $0.02/1M tokens, 1536 dims = 6 KB/vector fp32).

| Documents | Chunks | Embed tokens | Embed $ | Vectors, fp32 | What breaks first | What you add |
|---|---|---|---|---|---|---|
| **10** | 600 | 150k | $0.003 | 4 MB | nothing | a `for` loop |
| **1k** | 60k | 15M | $0.30 | 370 MB | one bad PDF kills the run; re-runs re-embed everything | content-hash skip, try/except, a manifest |
| **100k** | 6M | 1.5B | $30 | 37 GB | single process too slow (days); rate limits; memory | a durable queue, parallel workers, batching, quantization |
| **1M** | 60M | 15B | $300 | 369 GB | one machine cannot hold the index; parse failures are constant; duplicates crowd results | sharding/replication, DLQ, dedup, per-stage observability |
| **100M** | 6B | 1.5T | $30k | 37 TB | everything: storage, re-embedding is a quarter-long project, cost governance | a data platform: partitioning, tiering, lifecycle policy, incremental everything |

Three readings of that table that interviewers are listening for:

- **Embedding money is never the problem; embedding *time* is.** $300 to embed a million
  documents is a rounding error next to the engineers. But at a 1M tokens/minute limit, 15B
  tokens is 250 hours on one key: see §30.5.
- **Parsing dominates the compute**, not embedding. Parsing a 20-page PDF takes 0.5–5 seconds
  of CPU (layout models: much more); embedding its chunks is a couple of network calls you can
  batch and overlap. Ten to a hundred times more of your wall-clock and your failures live in
  the parse stage: and it is the stage that crashes.
- **The step change is between 1k and 100k**, where "re-run it" stops being a recovery
  strategy. Everything after that is scaling the same architecture.

## 30.2 The pipeline is stages, not a loop

```
 discover ──► fetch ──► parse ──► chunk ──► embed ──► upsert ──► verify
   (list)    (bytes)   (slow,     (cheap)  (rate-    (idempot-  (count,
                        flaky)              limited)  ent)       sample eval)
      │          │         │         │         │          │
      └──────────┴─────────┴─────────┴─────────┴──────────┘
              every stage: idempotent, retryable, independently scalable
                        back-pressure flows leftward
```

Each stage has its own failure mode and its own scaling knob:

| Stage | Typical failures | Retry? | Scaling knob |
|---|---|---|---|
| discover | source API paging, permissions | yes | one scheduler, watermark by `modified_at` |
| fetch | 404, 403, timeout, huge files | yes (transient) | concurrency + per-host limit |
| **parse** | corrupt/encrypted PDF, OOM, infinite loop, missing fonts | **sometimes** | CPU workers; hard timeout + memory cap per document |
| chunk | pathological documents (one 2 MB "paragraph") | no (deterministic) | free |
| embed | 429, 500, context length, provider outage | yes, with backoff | batch size, TPM budget, keys |
| upsert | index unavailable, payload too large | yes (idempotent) | batch, `wait=False` |
| verify | - | - | sampled golden-set eval |

Two rules make the whole thing tractable:

1. **Isolate the parse stage.** Run it with a timeout and a memory cap, in a separate worker
   pool, ideally a separate process so an OOM kills one worker and not the run. PDFs from the
   open internet *will* include files that hang a parser.
2. **Separate retryable from permanent.** A 429 is retryable. An encrypted PDF is not: it will
   fail identically forever. Retrying permanent failures is how a pipeline spends its budget on
   twelve corrupt files.

The work-item state machine that encodes this:

```
        submit()
           │
           ▼
      ┌─────────┐   lease()    ┌──────────┐  success   ┌──────┐
      │ queued  │─────────────►│  leased  │───────────►│ done │
      └─────────┘              └──────────┘            └──────┘
           ▲                        │
           │  retryable failure     │ permanent failure
           │  (backoff + jitter)    │ or attempts exhausted
           └────────────────────────┤
                                    ▼
                               ┌────────┐
                               │  dead  │  (dead-letter queue: a human looks)
                               └────────┘

      a crashed worker never blocks anything: its lease expires and the item
      returns to `queued` on its own - at-least-once delivery
```

`ingest_queue.py` implements exactly that in SQLite so you can watch it work:

```
$ QDRANT_MODE=memory uv run python code/ch30/ingest_queue.py --reset
submitted 19 documents: 19 new, 0 changed, 0 unchanged (skipped)

finished in 3.6s with 4 workers (fail-rate 0%)
  states        {'done': 19}
  documents     19 indexed, 0 retries, 0 dead-lettered
  chunks        97 embedded -> 97 points in Qdrant
  throughput    5.2 docs/s, 26.7 chunks/s
  worker time   parse 0.0s | embed 13.1s | upsert 0.1s  (summed across workers)
```

The `worker time` line is the one to read: 13.1 seconds of summed embed time inside 3.6
seconds of wall clock, because four workers overlapped their network waits. That ratio is the
entire argument for concurrency at the embed stage: and the reason the right worker count is
set by the API's limits, not by your CPU count. (Parse time is ~0 here because our "documents"
are markdown; on real PDFs this column dominates and the balance flips.)

## 30.3 Identity, idempotency, and duplicates

**Three different keys, three different jobs. Confusing them causes most re-ingest bugs.**

- **`doc_id`**: stable identity of *the thing* (a URL, a source-system primary key). Survives
  edits. This is what you delete by.
- **`content_hash`**: identity of *this version's bytes*. Changes on every edit. This is what
  you skip by.
- **point id**: `uuid5(namespace, f"{doc_id}::{chunk_index}")`. Deterministic, so re-upserting
  overwrites instead of duplicating. This is what makes at-least-once delivery safe.

That triple gives incremental indexing for free, and a second run of a finished crawl costs
nothing at all:

```
$ QDRANT_MODE=memory uv run python code/ch30/ingest_queue.py
submitted 19 documents: 0 new, 0 changed, 19 unchanged (skipped)
nothing to do - this is what a re-run of a finished crawl costs.
queue states: {'done': 19}
```

**Versioning.** When `content_hash` changes, re-chunk and diff at chunk level: upsert changed
chunks, delete chunk indices that no longer exist (the new version may be shorter), leave the
rest untouched. Deleting the document is a filter delete on `doc_id`. If you need
point-in-time answers ("what did the policy say in March?"), keep versions as separate points
with a `valid_from`/`valid_to` payload and filter at query time; otherwise use tombstones and
let retrieval never see the old version. Decide this early: retrofitting versioning into a
live index is painful.

**Near-duplicates need a different tool.** Exact hashing catches byte-identical copies and
nothing else. `dedup_minhash.py` measures that claim against the alternatives:

```
base document: 02-pto-and-leave-policy.md (387 words, 357 shingles), 128 permutations

variant                 sha256  true J  MinHash J    err  SimHash d
exact copy               MATCH   1.000      1.000  0.000          0
whitespace/case              -   1.000      1.000  0.000          0
1% words changed             -   0.919      0.898  0.021          7
5% words changed             -   0.610      0.609  0.001         12
20% words changed            -   0.162      0.125  0.037         28
paragraphs reordered         -   0.733      0.688  0.046          5
50% truncated                -   0.504      0.484  0.020         17
UNRELATED doc                -   0.001      0.000                31
```

Look at row two: the same document with whitespace collapsed and case changed has a true
Jaccard of **1.000** on word shingles and a completely different SHA-256. That one line is why
exact hashing is not a dedup strategy.

MinHash estimates the true overlap within 0.02–0.05 with 128 permutations, and the reason is a
small, elegant identity worth being able to state: for a random permutation `h`,
`P(min_h(A) == min_h(B)) == Jaccard(A, B)` exactly. Each permutation is one Bernoulli sample of
that probability, so the signature is an unbiased estimator whose error shrinks as
`1/sqrt(n_perms)`. SimHash instead has each feature vote on every bit of a 64-bit fingerprint;
similar documents differ in few bits (5–12 here) and unrelated ones in about half (31).

Choosing the threshold is an evaluation problem like any other:

```
  266 pairs, 84 true duplicates, 182 distinct

threshold  precision  recall     F1
      0.3       1.00    0.83   0.91
      0.5       1.00    0.83   0.91
      0.7       1.00    0.62   0.76
      0.9       1.00    0.44   0.61
  -> best F1 0.91 at threshold 0.5
```

Precision stays at 1.00 (unrelated documents simply do not collide) while recall falls as
you tighten. Recall caps at 0.83 because my label set calls the "20% of words changed" variant
a duplicate, and at a Jaccard of 0.16 it genuinely is not one. That is a labelling
disagreement, not an algorithm failure, and noticing which of the two you are looking at is the
skill.

Comparing every pair is O(n²): impossible at a million documents. LSH banding makes it
tractable: split the signature into `b` bands of `r` rows, and only compare pairs that match
on at least one whole band. The probability of that is `1 - (1 - s^r)^b`, an S-curve whose knee
you place where your threshold is:

```
LSH banding with 128 rows:
  bands x rows     0.3    0.5    0.7    0.8    0.9      <- Jaccard
        8 x 16    0.00   0.00   0.03   0.20   0.81
       16 x 8     0.00   0.06   0.61   0.95   1.00
       32 x 4     0.23   0.87   1.00   1.00   1.00
```

32×4 catches almost everything above 0.5 but also drags in 23% of the 0.3-similar pairs as
candidates to verify; 8×16 is cheap but catches only four pairs in five at Jaccard 0.9 and one
in five at 0.8, so it misses most genuine near-duplicates. Pick the row that matches the
threshold you just chose. Dedup **before** embedding (that is where the savings are) and
keep a `duplicate_of` pointer rather than silently dropping, so a user asking about the
duplicate still finds the canonical copy.

## 30.4 Queues, leases, and the dead-letter queue

The mechanisms, and what each one is defending against:

- **At-least-once delivery.** Assume every item can be delivered twice: a worker that died
  after upserting but before acknowledging will see its item again. Exactly-once is a
  distributed-systems fantasy; idempotent writes (§30.3) make at-least-once *equivalent* to it,
  which is the practical trick.
- **Leases (visibility timeouts).** A worker takes an item for N seconds. If it crashes, the
  lease expires and another worker picks the item up. No supervisor, no heartbeat protocol:
  `WHERE state='leased' AND lease_until < now` is the whole recovery mechanism. Set N above
  your p99 processing time, or slow items get processed twice concurrently.
- **Exponential backoff with jitter.** `delay = 2^attempts * base * (0.5 + random())`. The
  jitter is not decoration: without it, every worker that failed on the same provider outage
  retries at the same instant and rebuilds the stampede.
- **The dead-letter queue.** After `MAX_ATTEMPTS`, stop. The item goes to `dead` with its last
  error, visible for a human.

With injected failures and two permanently-poisoned documents:

```
$ QDRANT_MODE=memory uv run python code/ch30/ingest_queue.py --reset --fail-rate 0.2 --poison 2
submitted 19 documents: 19 new, 0 changed, 0 unchanged (skipped)
poison documents (always fail): 01-company-overview, 02-pto-and-leave-policy

finished in 3.1s with 4 workers (fail-rate 20%)
  states        {'dead': 2, 'done': 17}
  documents     17 indexed, 4 retries, 2 dead-lettered
  chunks        88 embedded -> 88 points in Qdrant

dead-letter queue (2 documents - a human looks at these):
  01-company-overview     ValueError: unsupported encrypted PDF (permanent)
  02-pto-and-leave-policy ValueError: unsupported encrypted PDF (permanent)
```

Four transient failures were retried into success; two permanent ones stopped costing money
after three attempts and are sitting in a queryable table with their error text. **A DLQ with
no one reading it is just a slower data loss**: put its depth on a dashboard and its
error-class breakdown in a weekly review. In practice the same three or four error classes
account for almost everything, and each becomes either a parser fix or a documented exclusion.

**One writer, many workers.** The design here fans out parse+embed across threads but funnels
upserts through a single writer. Writes get batched, the vector DB sees one connection, and
ordering is simple. It also keeps the code honest about Qdrant's embedded mode, which wants one
client per process.

Everything above maps onto whatever infrastructure you have:

| Here (SQLite) | SQS | Kafka | Celery/RQ | Temporal |
|---|---|---|---|---|
| `state`, `lease_until` | visibility timeout | consumer group offset + rebalance | task state in broker | workflow state |
| `attempts` + backoff | `ApproximateReceiveCount` + redrive | retry topic / manual offset | `autoretry_for`, `retry_backoff` | activity retry policy |
| `dead` rows | dead-letter queue | dead-letter topic | failed queue | terminated workflow |
| `submit()` skip-by-hash | idempotent producer | log compaction by key | task dedup key | workflow id dedup |
| the `writer` thread | batch consumer | sink connector | dedicated queue | activity |

Nothing here is exotic; the value is in knowing which property you are relying on.

## 30.5 Batching, rate limits, caching

Three measurements that set your wall clock. First, batch size against the real API:

```
embedding 128 chunks (~18,454 tokens) with text-embedding-3-small

 batch  calls  seconds  texts/s  ms/call
     1    128    74.04      1.7      578
     8     16    14.11      9.1      882
    32      4     3.92     32.6      981
    64      2     3.71     34.5     1854
   128      1     2.35     54.5     2349
```

**32× faster for the same tokens and the same price.** Per-call latency is dominated by the
round trip until the payload gets big, so the first few doublings are nearly free and the curve
then flattens. Batch until you hit the request size limit or your latency budget; for a
backfill, batch to the maximum.

Second (and this is the ceiling that actually matters) **providers meter tokens per minute,
not requests per minute**:

```
1M tokens/min, 2,500 tokens/document:
    350,000 TPM ->     140 docs/min -> 1M documents in  119.0 hours on one key
  1,000,000 TPM ->     400 docs/min -> 1M documents in   41.7 hours on one key
  5,000,000 TPM ->   2,000 docs/min -> 1M documents in    8.3 hours on one key
```

`docs/min = TPM / tokens_per_doc`. No amount of concurrency beats it; extra workers past that
point only generate 429s. Implement the limiter yourself (a token bucket: refill at the
sustained rate, bucket size is your allowed burst) rather than discovering the limit through
errors, because a self-imposed limit degrades gracefully while a provider-imposed one arrives
as a thundering herd of retries. To go faster you need more quota, more keys/accounts, a
second provider, or a local embedding model: and for a one-time backfill, the async **Batch
API at ~50% cost with a 24-hour turnaround** is usually the right answer. Use it for
backfills, never for live updates.

Third, the cache that makes retries and re-crawls free:

```
--- content-hash cache, the same corpus twice ---
  cold  2.50s   warm  0.00s   hit rate 50% (128/256)
```

Key it by **content hash, not document id**: the same boilerplate paragraph in forty documents
gets embedded once. Persist it (Redis, a table, object storage) so it survives restarts, and
remember that it is invalidated wholesale when you change embedding model: which is §30.7.

And the money, for a million documents:

```
pages/doc  chunks/doc       chunks         tokens   embed $  hours@1M TPM
        1           3    3,000,000    750,000,000        15          12.5
        5          15   15,000,000  3,750,000,000        75          62.5
       20          60   60,000,000 15,000,000,000       300         250.0
       50         150  150,000,000 37,500,000,000       750         625.0
```

Two-hundred-and-fifty hours for the 20-page case (ten days on one key) against $300 of
tokens. Every serious decision here is about time and failure, not about the invoice.

## 30.6 Index layout: partitions, tenants, shards

With 60M vectors you can no longer say "put it in a collection" and move on.

**One collection with payload partitioning** is the default. Tenant, source, language,
classification and date all live in the payload; every query carries a mandatory filter. For
tenancy specifically, index the tenant field with `is_tenant=True` and set `m=0` with
`payload_m=16` so each tenant gets its own HNSW sub-graph instead of one global graph
([Chapter 18 §18.3](../part6-scale/18-terabyte-scale.md)). One tenant's data then stays
co-located on disk and a filtered search does not wander through everyone else's vectors.

**A collection per tenant** is tempting and fails predictably past a few hundred. Every
collection carries fixed overhead: its own segments, its own HNSW graph, open file handles,
its own optimizer threads. At 10,000 tenants you have 10,000 of each, most of them holding a
dozen vectors, and operations that should be trivial (a schema change, a re-embed, a restart)
become 10,000 orchestrated steps. Use it only for a handful of tenants who need hard physical
isolation, usually for contractual reasons.

**Custom shard keys** are the middle ground: one collection, but the tenant (or region) decides
which shard the point lands on. You get isolation of blast radius and the ability to move a
large tenant to its own hardware, without the per-collection overhead. This is also how you
satisfy data-residency rules.

**Hot/cold splitting** is the highest-leverage move at this size, and it is usually driven by
time. Most corpora have a long tail nobody queries: keep the last N months in a RAM-resident,
quantized collection and push the rest to an on-disk collection, searching cold only when the
hot result is weak or the user asks for history. If 10% of your data serves 90% of queries, you
have just cut your memory bill by an order of magnitude.

The memory arithmetic that decides the tier (Chapter 18): 60M vectors × 1536 dims × 4 bytes =
**369 GB fp32**, or 92 GB with int8 scalar quantization, or 11 GB binary: plus roughly
`m × 2 × 4` bytes per vector for the graph. Quantization is not an optimisation at this scale;
it is what makes the index fit on machines you can afford. Replication is for availability and
read throughput, and it multiplies all of those numbers.

## 30.7 Re-embedding without downtime

You will change embedding model: a better one ships, or you shrink dimensions to cut the
memory bill. Vectors from two models are not comparable, so a half-migrated collection returns
nonsense for exactly the queries that touch the new half. The pattern:

```
   v1 (live)  ◄── alias "handbook_live" ◄── application
   v2 (dark)  ◄── backfill, every point marked embed_version=v2
      │
      ├─ validate v2 on the golden set BEFORE anyone sees it
      ├─ flip the alias (atomic: readers never see a half-built index)
      └─ keep v1 until you are sure; rollback is another flip
```

`migrate_embeddings.py` runs it end to end, migrating 1536-dim vectors to 512-dim ones:

```
v1 built: 56 points at 1536 dims
alias handbook_live -> ['ch30_v1']  (this is what production reads)

backfilling v2 at 512 dims, in the dark...
v2 built: 56 points at 512 dims (3.0x smaller vectors)
every point marked embed_version={'v2'} - a resumable backfill asks for the ones that are not

validating BEFORE the flip:
  collection   dims   hit@1     MRR
   v1 (live)   1536   1.000   1.000
   v2 (dark)    512   1.000   1.000
       delta         +0.000  +0.000

ACCEPTED: alias handbook_live -> ['ch30_v2'] (atomic switch)
storage: 6.0 KB -> 2.0 KB per vector, 6.1 GB -> 2.0 GB per 1M points
rollback drill: alias handbook_live -> ['ch30_v1'] again, no re-indexing, no downtime
```

Three-fold storage cut with no measured quality loss: but read the gate, not the conclusion.
Run the same migration to 48 dimensions and it refuses:

```
$ ... --new-dims 48
   v1 (live)   1536   1.000   1.000
   v2 (dark)     48   0.833   0.899
       delta         -0.167  -0.101

REFUSED: MRR dropped 0.101 > tolerance 0.020.
alias stays on ['ch30_v1']; v2 is kept for investigation.
```

That refusal is the entire reason to build v2 in the dark instead of migrating in place. Note
also what the accepted run *cannot* tell you: both collections scored 1.000 on a 14-document
corpus, so the gate had no headroom and would not have caught a small regression. Size the
golden set to the decision you are making ([Chapter 21](21-retrieval-metrics-deep.md)).

The details that make this work at 100M vectors rather than 56:

- **Version-mark every point** (`embed_version`) so the backfill is resumable and auditable:
  "how many points are still v1?" is a query, and a crashed backfill restarts where it stopped.
- **Dual-write during the backfill.** New and updated documents must go to *both* collections
  while it runs, or v2 is stale the moment it finishes. This is the step people forget.
- **Validate on a sample before the flip**, and prefer a paired comparison on the same
  questions: the gate above is a mini version of it.
- **Consider a canary**: point 5% of traffic at the alias'd v2 first if your router allows it.
- **Keep v1 for a full traffic cycle.** Rollback is an alias flip and costs nothing; rebuilding
  a deleted 100M-vector collection costs days.
- **Budget it honestly**: re-embedding 100M vectors is the §30.5 arithmetic again: the
  tokens are cheap, the calendar is not.

## 30.8 Observability and SLOs

Ingestion fails silently far more often than it fails loudly. The metrics that catch it:

| Signal | Why it is the one that matters | Alert when |
|---|---|---|
| **age of oldest unprocessed item** | the honest measure of "are we keeping up": a queue that is 4 hours behind is broken even if throughput looks fine | > your freshness SLO |
| throughput per stage (docs/s) | tells you *which* stage is the bottleneck | drops >50% week-on-week |
| failure rate by error class | three classes usually explain everything | any class doubles |
| **DLQ depth and arrival rate** | silent data loss lives here | any sustained arrival |
| retry rate | rising retries precede an outage | > 2× baseline |
| **index freshness** (change → searchable) | what users actually feel | > SLO (e.g. 15 min) |
| documents indexed vs discovered | the gap is documents you *think* you have | gap grows |
| cost per 1k documents | catches a parser change that doubled chunk counts | > 1.5× baseline |
| chunks per document distribution | a shifted distribution means a parser regression | distribution shift |
| **sampled retrieval eval on new data** | the only check that the index is *useful*, not just full | recall drops > 0.03 |

The last one closes the loop with [Chapter 10](../part4-quality/10-evaluation.md): run the
golden set nightly against production and alert on regression. An index can be complete, fast,
and useless.

Set explicit SLOs: *95% of documents searchable within 15 minutes of change; < 0.1% in the
DLQ; p95 parse under 10s*: because they turn "the pipeline is slow" into a decision about
workers, quota or scope.

## 30.9 The whiteboard answer

```
   sources          ┌──────────────┐
   (S3, SharePoint, │  discovery   │  watermark on modified_at, cursor per source
    web, DB CDC) ──►│  (scheduler) │
                    └──────┬───────┘
                           │ doc_id + content_hash   ◄── skip if hash unchanged
                           ▼
                    ┌──────────────┐
                    │ work queue   │  state: queued/leased/done/dead
                    │ (SQS/Kafka)  │  lease, attempts, backoff+jitter
                    └──────┬───────┘
                ┌──────────┼──────────┐
                ▼          ▼          ▼
            ┌───────┐ ┌───────┐ ┌───────┐     parse workers: CPU-bound,
            │ parse │ │ parse │ │ parse │     timeout + memory cap,
            └───┬───┘ └───┬───┘ └───┬───┘     crash-isolated  ──► DLQ
                └──────────┼──────────┘
                           ▼
                   chunk ──► dedup (MinHash/LSH, before embedding)
                           │
                           ▼
                    ┌──────────────┐
                    │ embed worker │  batch 128, token-bucket at TPM,
                    │  + cache     │  content-hash cache, Batch API for backfill
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │   upsert     │  uuid5(doc_id, chunk_idx) => idempotent
                    │  (writer)    │  wait=False, batched
                    └──────┬───────┘
                           ▼
                 Qdrant: one collection, payload partitions,
                 tenant index (is_tenant), quantized, sharded,
                 alias "live" ──► v1 | v2   (blue/green re-embed)
                           │
                           ▼
                 verify: counts, freshness, nightly golden-set eval
```

Say it in this order, and you will have covered every item on the list:

1. **Identity first**: `doc_id`, `content_hash`, deterministic point ids. Everything else
   depends on it.
2. **Queue + workers with leases**, at-least-once plus idempotent writes.
3. **Parse is the dangerous stage**: isolate, timeout, cap memory, DLQ the poison.
4. **Retryable vs permanent**, backoff with jitter, bounded attempts.
5. **Dedup before embedding**, MinHash + LSH, keep a `duplicate_of` pointer.
6. **Batch and respect TPM**; cache by content hash; Batch API for the backfill.
7. **Index layout**: payload partitions, tenant-aware HNSW, shards for isolation,
   quantization for memory, hot/cold split.
8. **Migrations**: build dark, version-mark, dual-write, validate, flip the alias, keep the
   rollback.
9. **Observability**: oldest-unprocessed age, DLQ depth, freshness, cost per 1k, nightly eval.
10. **Then the numbers**: 1M × 20 pages ≈ 60M chunks ≈ $300 and ~250 hours of embedding on one
    key at 1M TPM, ~369 GB fp32 or ~92 GB int8.

## The interview answer

**Asked: "How does your approach change between 10 PDFs and 1 million?"**

"At ten, a for-loop is correct and anything else is over-engineering. The architecture changes
at three points, and each one is forced by a specific thing breaking.

Around a thousand, re-running stops being free and one corrupt PDF kills the whole run. So I
add identity: a stable `doc_id`, a `content_hash` to skip unchanged documents, and deterministic
point ids: `uuid5(doc_id, chunk_index)`: so an upsert overwrites instead of duplicating. That
one change makes re-runs cost nothing and makes retries safe.

Around a hundred thousand, a single process is too slow and rate limits appear. That is where
the loop becomes a queue: work items with leases, so a crashed worker's item just comes back
when its lease expires; exponential backoff with jitter; bounded attempts; and a dead-letter
queue, because retrying a permanently encrypted PDF forever is how you spend a budget on twelve
files. The key insight is that parsing, not embedding, is the expensive and flaky stage
(ten to a hundred times more CPU, and it's the one that crashes) so it gets its own pool with a
timeout and a memory cap.

At a million (call it 60 million chunks) three new things matter. Rate limits are the real
ceiling: providers meter tokens per minute, so at 1M TPM and ~15,000 tokens for a 20-page
document that's under 70 documents a minute, roughly 250 hours on one key. Halve the document
to 2,500 tokens and the same quota gives you 400 a minute and 42 hours: which is the point
worth making out loud: the ceiling is set by tokens per document, not by worker count, so it
is the first number I'd ask for. The fix is batching, more quota, or the async Batch API at
half price for the backfill. Duplicates start crowding your top-k, so I
dedup with MinHash and LSH before embedding: exact hashing is useless, a whitespace change
gives a different SHA and identical content. And the index needs a layout: payload partitioning
with a tenant-aware HNSW rather than a collection per tenant, quantization because 60M vectors
is 369 GB in fp32 and 92 GB in int8, and a hot/cold split by date.

The last thing I'd mention is the one people forget: changing the embedding model later. You
build the new collection dark, mark every point with its version, dual-write while the backfill
runs, validate on a golden set before anyone sees it, and flip an alias. I've measured the
gate: migrating 1536 dims to 512 cost nothing and saved two-thirds of the memory, but to 48
dims MRR dropped 0.10 and the migration should be refused. Without the dark build you'd have
found that out in production."

## Run it

```bash
QDRANT_MODE=memory uv run python code/ch30/ingest_queue.py --reset          # clean run
QDRANT_MODE=memory uv run python code/ch30/ingest_queue.py                  # incremental: skips everything
QDRANT_MODE=memory uv run python code/ch30/ingest_queue.py --reset --fail-rate 0.2 --poison 2
uv run python code/ch30/dedup_minhash.py                                    # free, pure arithmetic
uv run python code/ch30/batch_and_ratelimit.py --texts 128                  # real API, a fraction of a cent
QDRANT_MODE=memory uv run python code/ch30/migrate_embeddings.py
QDRANT_MODE=memory uv run python code/ch30/migrate_embeddings.py --new-dims 48   # the gate refuses
```

Expected: 19 documents indexed to 97 points; a second run that does nothing; a run with
retries and two dead letters; the MinHash/SimHash table and LSH S-curve; the batch-size and
TPM tables; and both migration outcomes. Throughput and timing vary with your network; the
ratios do not. Note that `ingest_queue.py` keeps its queue in `data/ch30_queue.sqlite` while
`QDRANT_MODE=memory` throws the index away at exit: so an incremental re-run in memory mode
reports "unchanged" against an empty index. On a server both are durable; use `--reset` when
experimenting locally.

## Exercises

1. Run `ingest_queue.py --reset --repeat 3 --workers 1`, then `--workers 8`. Plot documents/s
   against workers. Where does it stop improving, and which stage is the ceiling?
2. Kill the process mid-run (Ctrl-C) and restart it without `--reset`. Confirm that leased
   items are reclaimed and the point count is identical to a clean run: that is the lease
   mechanism and idempotent ids working together.
3. Implement the versioning half of §30.3: change one paragraph of a handbook document,
   re-submit, and make the pipeline upsert only the changed chunks and delete the ones that no
   longer exist. Prove it by counting points before and after.
4. Add MinHash dedup to `ingest_queue.py` between chunk and embed, using the 0.5 threshold
   from §30.3. How many chunks does it remove from the handbook plus `data/generated/`? How
   much embedding spend would that save at a million documents?
5. Give `TokenBucket` a real clock and wire it into the embed stage, then run with a
   deliberately low TPM. Watch the pipeline throttle instead of collecting 429s.
6. Extend `migrate_embeddings.py` with dual-write: while v2 backfills, new documents must land
   in both collections. Then prove a document added mid-migration is searchable in v2 after
   the flip.

## Interview questions

**Q: What actually breaks first as you scale ingestion?**
At ~1k documents, re-runs and crash recovery: one corrupt file kills the run and a restart
re-embeds everything, which is fixed by content hashing and deterministic ids. At ~100k, wall
clock and rate limits, which forces a durable queue with parallel workers and batching. At ~1M,
memory for the index and the constant drip of parse failures and duplicates, which forces
quantization, sharding, a dead-letter queue and dedup. Naming the specific break at each scale
is what separates a design from a list of buzzwords.

**Q: How do you make ingestion idempotent?**
Deterministic point ids: `uuid5(namespace, f"{doc_id}::{chunk_index}")`: so any re-delivery
overwrites the same points instead of duplicating them, plus a `content_hash` check that skips
unchanged documents entirely. That combination means at-least-once delivery is safe, which is
the only delivery guarantee you can actually get from a queue.

**Q: Why do you need a dead-letter queue?**
To separate retryable from permanent failures. A 429 or a timeout should be retried with
backoff; an encrypted or corrupt PDF will fail identically forever, and retrying it burns
budget and hides real failures behind noise. After bounded attempts the item goes to a DLQ with
its error, where its depth is alerted on and its error classes get reviewed: a DLQ nobody
reads is just slower data loss.

**Q: Exact hashing catches duplicates. Why MinHash?**
Because exact hashing catches only byte-identical copies. I measured a document with collapsed
whitespace and changed case: true Jaccard 1.000 on word shingles, completely different SHA-256.
MinHash estimates the actual Jaccard overlap: `P(min_h(A)==min_h(B)) == Jaccard(A,B)`, so 128
permutations estimate it within about 0.02: and LSH banding makes finding candidates
sub-quadratic instead of comparing all pairs.

**Q: What is your real throughput ceiling for embedding?**
Tokens per minute, not requests per minute. At 1M TPM and ~2,500 tokens per document that is
400 documents a minute, so a million documents is about 42 hours on one key regardless of how
many workers you run: extra concurrency past that just produces 429s. Batching is free speed
below that ceiling: I measured 1.7 texts/s at batch size 1 versus 54.5 at 128, same tokens and
same price. For backfills the async Batch API halves the cost at a 24-hour turnaround.

**Q: How do you re-embed 100M vectors with a new model and no downtime?**
Build a second collection dark behind an alias, mark every point with its embedding version so
the backfill is resumable, dual-write new and changed documents to both while it runs, validate
the new collection on a golden set before any traffic sees it, then flip the alias atomically
and keep the old one for a full traffic cycle so rollback is another flip. I gate the flip on a
metric: migrating to 512 dims cost nothing, to 48 dims MRR fell 0.10 and the migration is
refused.

**Q: Collection per tenant, or one collection with a tenant filter?**
One collection with a tenant payload field indexed as a tenant key, `m=0` and `payload_m` set so
each tenant gets its own HNSW sub-graph, and a mandatory tenant filter on every query. A
collection per tenant carries fixed overhead (segments, graph, file handles, optimizer threads)
per tenant, so at thousands of tenants ordinary operations become thousands of orchestrated
steps. Custom shard keys are the middle ground when you need blast-radius isolation or data
residency for specific tenants.

**Q: Which stage of the pipeline is the expensive one, and why does that matter?**
Parsing. A 20-page PDF costs seconds of CPU to parse, far more with a layout model, while its
embeddings are a couple of batched network calls; parsing is also the stage that crashes, hangs
and OOMs. It matters because it sets where you spend money (CPU workers, not GPU), where you
need isolation (timeouts, memory caps, separate processes), and because people instinctively
optimise the embedding call instead.

**Q: What do you monitor on an ingestion pipeline?**
Age of the oldest unprocessed item (the honest "are we keeping up" signal) plus per-stage
throughput, failure rate by error class, DLQ depth and arrival rate, index freshness from
change to searchable, discovered-versus-indexed gap, cost per 1k documents, and chunks per
document to catch parser regressions. Above all, a nightly golden-set retrieval eval against
production: an index can be complete, fast and useless.

**Q: How do you handle a document that changes?**
Detect it by content hash, re-chunk, and diff at chunk level: upsert changed chunks, delete
chunk indices that no longer exist because the new version is shorter, leave the rest alone.
Deleting a document is a filter delete on `doc_id`. If answers must be reproducible as of a
past date, keep versions as separate points with validity metadata and filter at query time:
but decide that early, because retrofitting versioning into a live index is painful.

**Q: Where does the money actually go at a million documents?**
Not embeddings: 60M chunks is about $300 one time. It goes to parse compute, the engineering
time for failure handling, index memory (369 GB fp32, ~92 GB int8, which decides your machine
bill), and re-embedding whenever you change models. That is why the design decisions worth
arguing about are parser choice, quantization and migration strategy: not the per-token price.

## Key takeaways

- Different things break at each order of magnitude: re-runs and crash recovery at ~1k, wall
  clock and rate limits at ~100k, index memory and constant parse failures at ~1M. Name the
  break, then name the fix.
- Identity is the foundation: `doc_id` for the thing, `content_hash` for the version,
  `uuid5(doc_id, chunk_index)` for the point. That triple gives you incremental indexing and
  makes at-least-once delivery equivalent to exactly-once.
- Parsing, not embedding, is the expensive and dangerous stage: isolate it with timeouts and
  memory caps, and dead-letter the poison instead of retrying forever.
- Tokens per minute is the real throughput ceiling; batching is a free 32× below it, caching by
  content hash makes retries and re-crawls free, and the Batch API halves the cost of backfills.
- Re-embedding is a blue/green migration: build dark, version-mark, dual-write, validate on a
  golden set, flip an alias, keep the rollback. Gate the flip on a metric with enough headroom
  to detect a regression.

## Next

→ [Appendix A: Research papers](../appendix/A-research-papers.md)

That is the end of Part 8. The appendices are the reference layer: the papers behind these
techniques in reading order, the metrics cheat-sheet, the glossary, and the API cheat-sheet to
keep open while you code.
