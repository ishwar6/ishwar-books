# Chapter 18 · Terabyte Scale

> **Goal:** you can size a RAG system on paper (chunks, embedding cost, RAM, ingest time,
> cost per query) for 1 GB to 1 TB of text; you know which Qdrant knobs change the numbers
> (quantization, HNSW, on-disk storage, multitenancy, sharding) and can write the code for
> them; and you can describe an ingestion pipeline that survives a model change without
> downtime.

---

## 18.1 Do the arithmetic first: [`code/ch18/scale_math.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch18/scale_math.py)

Assumptions (state them, then change them): ~4 characters per token, 800-character chunks
with 120 overlap (what `ragbook.index` uses), `text-embedding-3-small` at 1536 dims and
$0.02 per 1M tokens, a sustained embedding rate of 20k tokens/s against one key.

```
  corpus    tokens    chunks   embed $  embed h  fp32 GB  int8 GB  bin GB  text GB
----------------------------------------------------------------------------------
    1 GB     0.27B      1.6M         6      4.4      9.0      2.3    0.28      1.6
   10 GB     2.68B     15.8M        63     43.9     90.4     22.6    2.82     16.2
  100 GB    26.84B    157.9M       632    438.6    903.5    225.9   28.24    161.8
1,024 GB   274.88B   1616.9M     6,468   4491.5   9252.1   2313.0  289.13   1656.5
```

Read the table the way an interviewer wants you to:

- **Embedding cost is not the problem.** $6.5k to embed a terabyte, once. The cheap model is
  fine; use the OpenAI Batch API (50% off, 24 h turnaround) for the initial load.
- **Embedding *time* is.** 4,500 hours at one key's rate. You parallelise across keys/
  workers, batch requests (up to 2,048 inputs per call), and accept days for the first load.
  Hence: ingestion is a pipeline with a queue, not a script.
- **RAM is the architecture decision.** 1.6 billion 1536-dim float32 vectors = **9.2 TB of
  RAM** for vectors alone (6 KB each) plus HNSW links (~128 B each at `m=16`). Nobody buys
  that. You reach for: int8 scalar quantization (÷4 → 2.3 TB), binary quantization (÷32 →
  290 GB), vectors on disk with only the quantized copy in RAM, fewer dimensions
  (Matryoshka: `dimensions=512` on `text-embedding-3-*` keeps most of the quality at a third
  of the bytes), bigger chunks (fewer vectors), and de-duplication (large corpora are 20–40%
  near-duplicates).
- **The text payload itself is 1.6 TB.** Store it in object storage or a document DB and keep
  only ids and short metadata in Qdrant if RAM/disk on the vector nodes is the constraint.

Per query, the LLM dominates cost:

```
Per query (k=4, ~200-token chunks, gpt-5.4-mini): $0.00125  ->  1M queries/month ≈ $1,253
  of which LLM = 100.0%, question embedding = 0.03%. Retrieval is cheap; generation is where the money goes.
Throughput: 1M queries/month = 0.39 qps average; plan for 10x peak = 4 qps
```

So at a million queries a month, retrieval hardware is trivial and the model bill is the
budget line. Caching (exact + semantic), smaller `k` with a reranker, and a cheaper model for
easy questions (Adaptive routing, Chapter 15) move that number.

### "You scale from 1 GB to 10 GB. What changes?"

Everything grows 10×, but not equally painfully:

| Dimension | 1 GB | 10 GB | What you revisit |
|---|---|---|---|
| Vectors in RAM (fp32) | 9 GB: one box | 90 GB: a big box or quantize | int8/binary quantization, `on_disk`, dims 512 |
| Index build | minutes | hours; HNSW build is O(n log n) and CPU-heavy | `ef_construct`, build on ingest nodes, `wait=False` |
| Ingest | one afternoon | days at one key's rate | queue + parallel workers + batch API |
| Query latency | ~5 ms search | still ~10 ms *if in RAM*; 50–200 ms if paging from disk | keep quantized vectors in RAM; `hnsw_ef` tuning |
| Recall | fine | more near-neighbours compete; irrelevant-but-similar chunks | better chunking, hybrid search, reranking, metadata filters |
| Filtering | any filter works | unindexed payload filters scan | payload indexes, `is_tenant`, pre-filtering |
| Freshness | re-index nightly | incremental upserts with stable ids, deletes | idempotent ids (`uuid5`), change feeds |

The candidate said "latency will increase, revisit chunking, embedding, vector DB". Give
this table instead.

## 18.2 Quantization and HNSW: [`code/ch18/quantized_collection.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch18/quantized_collection.py)

Three collections over the same 56 handbook chunks, differing only in configuration:

```python
models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(
    type=models.ScalarType.INT8, quantile=0.99, always_ram=True))        # 4x smaller

models.BinaryQuantization(binary=models.BinaryQuantizationConfig(always_ram=True))  # 32x smaller

hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100, on_disk=False)
vectors_config=models.VectorParams(size=DIM, distance=COSINE, on_disk=True)  # originals on disk
```

and at query time:

```python
search_params=models.SearchParams(
    hnsw_ef=128,
    quantization=models.QuantizationSearchParams(ignore=False, rescore=True, oversampling=2.0),
)
```

**How it works.** The HNSW graph is searched using the small quantized vectors kept in RAM
(`always_ram=True`). With `oversampling=2.0` it collects 2× `limit` candidates; `rescore=True`
then recomputes exact scores for those candidates from the original vectors (on disk) and
returns the top `limit`. You pay a handful of disk reads per query instead of holding 6 KB
per vector in RAM. Scalar int8 loses ~1% recall; binary loses more on short/low-dim
embeddings but works well on 1024+-dim OpenAI-style vectors: always rescore with binary.

**HNSW knobs.** `m` = links per node: higher → better recall, more RAM (`m*2*4` bytes per
vector). `ef_construct` = build-time beam width: higher → better graph, slower ingest.
`hnsw_ef` = query-time beam width: the dial you turn per request to trade latency for
recall. Start at `m=16, ef_construct=100, hnsw_ef=64–128`; measure recall@k against exact
search on a sample (Chapter 10) before changing.

Observed:

```
ch18_fp32      4.5 ms  [(0.545, '04-atlas-a2-specification.md'), (0.501, '04-atlas-a2-specification.md'), (0.43, '14-faq.md')]
ch18_int8      1.8 ms  [(0.545, ...same...)]
ch18_binary    1.8 ms  [(0.545, ...same...)]
```

Same hits and same scores after rescoring. You will also see:

```
UserWarning: Local mode performs exact (brute-force) search, so `search_params` has no effect
```

That is the honest caveat: the **embedded/local client does exact search and ignores HNSW
and quantization search params**. The configuration API is identical on a server, and that
is where the memory and latency effects appear. Run the same script against
`docker compose up -d` + `QDRANT_URL=http://localhost:6333` to see the warning disappear.

## 18.3 Multitenancy and zero-downtime re-embedding: [`code/ch18/multitenant_collection.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch18/multitenant_collection.py)

**One collection, many customers.** A collection per tenant does not scale past a few
hundred (each has its own index and overhead). Instead:

```python
client.create_collection(NAME, vectors_config=...,
    hnsw_config=models.HnswConfigDiff(m=0, payload_m=16))      # no global graph; one per tenant
client.create_payload_index(NAME, "tenant_id",
    models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True))
...
query_filter=models.Filter(must=[models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant))])
```

`m=0` disables the global HNSW graph; `payload_m=16` builds a small graph *per tenant
value*, so a filtered search walks only that tenant's graph. `is_tenant=True` co-locates a
tenant's vectors on disk. The filter is evaluated *inside* the search (Qdrant's filterable
HNSW), not as a post-filter that would throw away 99% of results. This is Chapter 16's RBAC
filter at customer scale. For very large tenants, or hard isolation requirements, use
**custom sharding**: `shard_number`, `sharding_method=CUSTOM`, `create_shard_key(tenant)`,
and route upserts/queries with `shard_key_selector`: each tenant's data lives on chosen
nodes. `replication_factor=2` gives you node-failure tolerance. Both are server-only.

**Changing the embedding model** (or chunking) means re-embedding everything, and old and new
vectors cannot share a collection (different spaces). Do it blue/green with an **alias**:

```python
client.update_collection_aliases(change_aliases_operations=[
    models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name="handbook_live")),
    models.CreateAliasOperation(create_alias=models.CreateAlias(collection_name="ch18_v2", alias_name="handbook_live")),
])
```

The application only ever reads `handbook_live`. Build `v2` next to `v1`, verify recall on
the golden set, flip the alias atomically, delete `v1` later. Observed:

```
alias handbook_live -> ['ch18_v1']
alias handbook_live -> ['ch18_v2']
```

> This section sketches the pipeline; [Chapter 30](../part8-deep-dives/30-ingesting-a-million-pdfs.md)
> builds it: a real queue with leases, retries, a dead-letter queue, MinHash deduplication,
> rate-limited batch embedding and a zero-downtime re-embedding migration.

## 18.4 The ingestion pipeline at scale

```
sources --> change feed / crawler --> queue (Kafka / SQS / Pub/Sub)
        --> parse+chunk workers --> dedup (content hash) --> embed workers (batched, rate-limited)
        --> upsert (wait=False, stable ids) --> Qdrant  --> nightly recall check on golden set
```

Rules that hold from 1 GB to 1 TB:

- **Stable ids** (`uuid5(doc_id, chunk_index)` as in `ragbook.index`) make every upsert
  idempotent; re-running a failed batch is safe; deleting a document is `delete` by
  `doc_id` filter.
- **Batch** embeddings (hundreds of chunks per call) and **respect rate limits** with backoff;
  the Batch API for bulk loads.
- **Dedup before embedding**: hash normalised chunk text; near-dup with MinHash if the
  corpus is web-like.
- **`wait=False`** on upsert lets Qdrant index in the background; monitor
  `collection.status` and the optimizer.
- **Metadata at ingest**: tenant, classification, source, timestamp: you cannot filter on
  what you did not store.
- **Freshness**: incremental upserts from a change feed; `updated_at` payload so retrieval
  can prefer recent versions (Chapter 9's conflict problem).
- **Two caches**: an embedding cache keyed by text hash (never embed the same chunk twice)
  and a semantic answer cache (Chapter 17).

## Run it

```bash
uv run python code/ch18/scale_math.py
QDRANT_MODE=memory uv run python code/ch18/quantized_collection.py
QDRANT_MODE=memory uv run python code/ch18/multitenant_collection.py
```

Outputs are quoted in the sections above. Two `UserWarning`s from the embedded client are
expected: *local mode performs exact search* and *payload indexes have no effect in local
Qdrant*. They are the reminder that HNSW, quantization and payload indexes are server
features; the code is what you would ship.

## Exercises

1. Change `CHUNK_CHARS` to 1600 and `DIMS` to 512 in `scale_math.py`. How much RAM does
   1 TB need with int8 now? What did you give up (hint: Chapter 5, Chapter 2)?
2. Start Qdrant with `docker compose up -d`, set `QDRANT_URL`, re-run
   `quantized_collection.py`. Then generate 200k random vectors, load them into the int8 and
   fp32 collections, and compare `query_points` latency and the dashboard's RAM figures.
3. Measure recall@10 of the binary collection *without* `rescore` vs *with*, against the
   fp32 collection as ground truth, on the golden questions.
4. Add `tenant_id` to `ragbook.index.chunk_documents` metadata and make Chapter 17's API
   filter on the tenant from the API key.
5. Write the blue/green procedure as a script: build `v2` with `chunk_size=400`, run the
   Chapter 10 eval on both, flip the alias only if recall@4 did not drop.

## Interview questions

**Q: You go from 1 GB to 10 GB of documents. What changes and what would you revisit?**
Vectors ×10 (≈90 GB fp32 at 1536 dims) so RAM becomes the constraint → quantization
(int8/binary), on-disk originals, fewer dims; index build and ingest go from hours to days →
queue-based parallel ingestion, batch embeddings; latency stays low only if the quantized
index stays in RAM; recall degrades as more similar chunks compete → hybrid search,
reranking, metadata filters, chunking review; unindexed filters start to hurt → payload
indexes; freshness needs incremental upserts with stable ids. Re-run the golden-set eval
after each change.

**Q: How much memory does a vector index need?**
vectors × dims × bytes-per-component, plus the HNSW graph (~`m*2*4` bytes per vector). At
1536 dims: 6 KB fp32, 1.5 KB int8, 192 B binary per vector. A billion vectors is 6 TB fp32
or ~190 GB binary: which is why quantization is not optional at scale.

**Q: What is quantization and what does it cost in quality?**
Compressing each vector component from 32-bit floats to 8-bit ints (scalar, 4×) or 1 bit
(binary, 32×). Search runs on the compressed vectors in RAM; `rescore` re-ranks the top
candidates with the originals. Scalar costs ~1% recall; binary more unless dims are high
and you oversample + rescore.

**Q: HNSW parameters: what do `m`, `ef_construct`, `ef` do?**
`m`: links per node: recall and RAM. `ef_construct`: build-time search width: graph quality
vs ingest time. `ef` (`hnsw_ef`): query-time search width: the latency/recall dial.
Tune with recall@k against brute-force on a sample.

**Q: How do you do multitenancy in a vector DB?**
One collection with a `tenant_id` payload, indexed with `is_tenant=True`, `m=0` +
`payload_m` so each tenant gets its own HNSW sub-graph, and a mandatory tenant filter
inside every search. For hard isolation or huge tenants: custom shard keys per tenant.
Never a collection per tenant beyond a few hundred.

**Q: How do you change the embedding model in production?**
Blue/green: build a new collection with the new model, re-embed from source (not from old
vectors), validate on the golden set, flip a collection alias the app reads from, delete
the old collection later. Old and new vectors never mix.

**Q: What does a query cost and where does the money go?**
Question embedding is ~$0.0000004; vector search is amortised hardware; the LLM call is
>99% of the marginal cost (context tokens in, answer tokens out). Reduce with smaller `k`
plus a reranker, caching, a cheaper model for easy queries, shorter chunks in the prompt.

## Key takeaways

- Size it on paper: chunks → vectors → bytes. RAM, not embedding cost, decides the design.
- Quantize (int8, binary with rescore), keep originals on disk, cut dims, dedup, chunk
  bigger: in that order of ease.
- HNSW is tuned with `m`, `ef_construct`, `hnsw_ef`; validate with recall@k, not vibes.
- Multitenancy = payload filter inside the search with per-tenant graphs; sharding for
  isolation.
- Ingestion is a queue-fed pipeline with stable ids; model changes are alias flips.
- The embedded Qdrant client is exact search for learning; the same code is HNSW on a
  server.

## Next

[Chapter 19: What is new in RAG](../part7-frontier/19-whats-new-in-rag.md): long context vs
retrieval, GraphRAG, late interaction, contextual retrieval, agentic search: and whether
RAG is "dead".
