# Chapter 4 · Qdrant: A Real Vector Database

> **Goal:** you can explain why a vector database beats a numpy array (approximate
> nearest-neighbour indexes, filtering, persistence), drive Qdrant with its raw client
> (collections, points, payloads, filters, thresholds, deletes), and then use LangChain's
> `QdrantVectorStore` knowing exactly what it stores and where the sharp edges are.

Code: [`code/ch04/qdrant_raw_client.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch04/qdrant_raw_client.py), [`code/ch04/qdrant_langchain.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch04/qdrant_langchain.py).

---

## 4.1 Why not numpy?

Chapter 2's `rank()` compares the query with *every* stored vector. That is exact and
fine for 55 chunks. It stops being fine at:

- **Scale.** 10 M vectors × 1536 floats is 60 GB of float32. A full scan per query is
  seconds, not milliseconds, and you cannot hold it in one process's RAM.
- **Filtering.** "Only chunks from the FAQ" or "only documents this user may see": a
  numpy array has no metadata, so you filter after the fact and hope enough survive.
- **Persistence and updates.** Re-embedding the corpus every process start is not a plan;
  neither is rebuilding the whole array when one document changes.
- **Concurrency.** Ingestion writing while queries read.

A vector database gives you all four. **Qdrant** is the one this book uses: open source,
written in Rust, runs embedded in Python for learning and as a server (Docker or Cloud) for
production, with the *same client API* for all modes.

## 4.2 HNSW in one page

Exact nearest-neighbour search is O(n). Every vector DB instead builds an **approximate**
index; the dominant one is **HNSW: Hierarchical Navigable Small World**.

Picture every vector as a node in a graph, each linked to its `m` nearest neighbours.
Searching means: start somewhere, hop to whichever neighbour is closest to the query,
repeat until no neighbour is closer. That greedy walk finds *a* local optimum quickly.
HNSW stacks several such graphs: a sparse top layer with long-range links for coarse
navigation, denser layers below for precision, like a motorway → main road → street
descent. Search is roughly O(log n).

Three knobs, all of which trade **recall** (did we find the true nearest?) against
**latency/memory**:

| Knob | Set when | Effect |
|---|---|---|
| `m` (links per node, default 16) | collection creation | higher = better recall, more RAM (each link is an integer per node) |
| `ef_construct` (default 100) | collection creation | higher = better graph, slower indexing |
| `ef` / `hnsw_ef` (if unset, Qdrant uses at least `limit`) | each query | higher = search visits more candidates, better recall, slower |

"Approximate" is real: with defaults you get ~95–99% of the true top-k. For RAG that is
fine (the reranker and the LLM smooth over the occasional miss) but you should *know* the
number for your collection (Chapter 18 measures it).

Qdrant also keeps the vectors themselves (for exact re-scoring of candidates), the
**payload** (your metadata as JSON) and, on a server, **payload indexes** so filters are
applied *inside* the graph walk rather than after it. That last point is why "filter by
tenant then search" is fast in Qdrant and slow in a naive system.

> This section is the working knowledge. The mechanism underneath: building HNSW from scratch,
> measuring what `m`, `ef_construct` and `ef` actually cost, IVF and product quantization, and a
> tuning recipe for 10M vectors: is [Chapter 22](../part8-deep-dives/22-vector-search-internals.md).
> If an interviewer asks you to tune HNSW, that chapter is the answer.

## 4.3 Three ways to run it

```python
from ragbook import get_qdrant_client
client = get_qdrant_client()
```

`get_qdrant_client()` picks a mode from `.env`:

| Mode | How | Use for |
|---|---|---|
| `QDRANT_MODE=memory` | `QdrantClient(":memory:")`: in-process, gone at exit | tests, experiments, this book's validation runs |
| default | `QdrantClient(path="./.qdrant_data")`: embedded, persisted to disk | learning; **one process at a time** (it takes a lock) |
| `QDRANT_URL=http://localhost:6333` | `QdrantClient(url=...)`: a server | anything real; `docker compose up -d qdrant` at the repo root starts one (the compose file also defines Chapter 17's API service), with a dashboard at http://localhost:6333/dashboard |

In embedded mode this also means **one client object per process**: a second `QdrantClient(path=...)` in the same script raises `already accessed by another instance`. Create the client once and pass it to every function that needs it (`build_handbook_index(..., client=client)`).

Embedded mode runs the same Rust engine in your process, with two caveats you will hit:
**payload indexes are ignored** (filters still work, just by scanning) and **two processes
cannot open the same path**. The server has neither limitation.

## 4.4 The raw client: what a vector DB actually stores

`qdrant_raw_client.py` does everything by hand once. Read it top to bottom.

**A collection** is a table whose rows carry a vector of fixed size and distance metric:

```python
client.create_collection(
    collection_name="ch04_raw",
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
)
```

**A point** is id + vector + payload. Ids must be integers or UUIDs. Deriving the id from
the content's *location* (`uuid5(doc_id#paragraph)`) makes re-ingest an overwrite instead
of a duplicate:

```python
models.PointStruct(
    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{p['doc_id']}#{p['para']}")),
    vector=v,
    payload={"text": para, "doc_id": doc.metadata["doc_id"], "para": i},
)
client.upsert(COLLECTION, points=points)
```

**Query** = embed the question, ask for the nearest `limit` points:

```python
res = client.query_points(COLLECTION, query=qv, limit=3, with_payload=True)
for pt in res.points:
    pt.score, pt.payload["doc_id"], pt.payload["text"]
```

```
Q: how long does a full charge take
   0.598  14-faq                     | **Q: How long does it take to charge an Atlas?** A: A full charge take...
   0.441  04-atlas-a2-specification  | - Lithium iron phosphate (LFP) pack, **48 V, 60 Ah**...
```

**Filter**: the same query, restricted by payload. `must` = AND, `should` = OR,
`must_not` = NOT; conditions match values, ranges, text, geo, nesting:

```python
query_filter=models.Filter(must=[
    models.FieldCondition(key="doc_id", match=models.MatchValue(value="14-faq"))
])
```

**Score threshold**: return nothing rather than junk. This is the DB-side version of the
"I don't know" problem from Chapter 3:

```
'recipe for pancakes' with score_threshold=0.45 -> 0 results
```

**Scroll** reads rows by filter without a vector (browse, export, count-by-doc).
**Delete** removes by ids or by filter: "document X was removed from the wiki" becomes one
call:

```python
client.delete(COLLECTION, points_selector=models.FilterSelector(filter=models.Filter(
    must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value="14-faq"))])))
```

```
after deleting 14-faq: 104 points remain
```

That is the entire data model: collections of points, each with a vector and a JSON
payload, queried by nearness plus filter. Everything else is configuration.

## 4.5 The LangChain wrapper, and what it decides for you

`QdrantVectorStore` makes Qdrant look like the `InMemoryVectorStore` of Chapter 3. Under
the hood it fixes one thing you must know: the **payload layout**:

```
payload = {
    "page_content": "<the chunk text>",
    "metadata":     {"source": ..., "doc_id": ..., "title": ...}     # your Document.metadata
}
```

Hence **filters use `metadata.<key>`**:

```python
store.similarity_search(q, k=2, filter=models.Filter(must=[
    models.FieldCondition(key="metadata.doc_id", match=models.MatchValue(value="14-faq"))]))
```

Forget the prefix and the filter silently matches nothing.

**Creating the store.** Two routes:

```python
# Route A - one-liner that builds its OWN client from location/url/path kwargs:
QdrantVectorStore.from_documents(chunks, embedding, location=":memory:", collection_name="x")

# Route B - you own the client and the collection; the store wraps them:
client.create_collection("x", vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE))
store = QdrantVectorStore(client=client, collection_name="x", embedding=embeddings)
store.add_documents(chunks, ids=[...])
```

Gotcha we hit writing this chapter: **`from_documents` does not accept `client=`**: it
raises `TypeError: Client.__init__() got an unexpected keyword argument 'client'`
because the kwargs are forwarded to a *new* `QdrantClient`. When you hold a client object
(you always will, via `get_qdrant_client()`), use Route B. It is also the route that lets
you set HNSW, quantization and sparse-vector configuration on the collection (Chapters 8,
18).

**Reading back**: `similarity_search`, `similarity_search_with_score` (raw cosine),
`similarity_search_with_relevance_scores` (rescaled: see below), and the retriever
interface:

```python
strict = store.as_retriever(search_type="similarity_score_threshold",
                            search_kwargs={"k": 4, "score_threshold": 0.66})
mmr = store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5})
```

`as_retriever` returns a Runnable: `retriever.invoke("question")` gives `list[Document]`:
which is the interface tools and graphs consume in Part 5.

**Second gotcha: two different scores.** The retriever's `score_threshold` is applied to
LangChain's *relevance* score, which for cosine is `(cosine + 1) / 2`, not the cosine you
see in `similarity_search_with_score`:

```
cosine vs relevance score for the same hits:
   cosine 0.438 -> relevance 0.719   03-travel-and-expense-policy.md
   cosine 0.278 -> relevance 0.639   03-travel-and-expense-policy.md
threshold retriever (relevance >= 0.66), 'pancake recipe' -> 0 docs
threshold retriever (relevance >= 0.66), 'what is the hotel cap in Europe' -> 1 docs
```

A threshold of 0.5 on the relevance scale is cosine 0.0: it filters nothing. Print both
before you choose a number.

**MMR (maximal marginal relevance**) fetches `fetch_k` candidates and picks `k` that are
relevant *and different from each other* (`lambda_mult` = 1 pure relevance, 0 pure
diversity). Useful when the top hits are five near-identical chunks from one document and
you would rather see three documents:

```
MMR retriever (diverse sources):
  [1] 12-postmortem-2026-03-rotterdam-halt.md
  [2] 04-atlas-a2-specification.md
  [3] 08-runbook-fleet-outage.md
```

**Re-attaching** later is the constructor again:
`QdrantVectorStore(client=client, collection_name=..., embedding=...)`: the collection
already has the vectors; you just need the same embedding model for queries (Chapter 2,
Rule 1).

## Run it

```bash
QDRANT_MODE=memory uv run python code/ch04/qdrant_raw_client.py
QDRANT_MODE=memory uv run python code/ch04/qdrant_langchain.py
```

Expected (trimmed):

```
created 'ch04_raw' with 1536-dim cosine vectors
upserted 116 points from 14 docs
payload index on doc_id: no effect in embedded mode (server only)
Q: how long does a full charge take
   0.598  14-faq  | **Q: How long does it take to charge an Atlas?** ...
'recipe for pancakes' with score_threshold=0.45 -> 0 results (good: nothing relevant exists)
scroll: 8 paragraphs belong to 10-pricing-and-plans
after deleting 14-faq: 104 points remain
```

```
55 chunks -> 55 points
payload keys: ['page_content', 'metadata'] | metadata keys: ['source', 'doc_id', 'title']
Q: what is the hotel cap in Europe
   0.438  03-travel-and-expense-policy.md
cosine 0.438 -> relevance 0.719
threshold retriever (relevance >= 0.66), 'pancake recipe' -> 0 docs
re-attached: 2 docs found
```

Then, without `QDRANT_MODE=memory`, run either script twice: the second run finds the
collection on disk in `.qdrant_data/`. To try the server, start Docker Desktop and:

```bash
docker compose up -d qdrant    # Qdrant on :6333, dashboard at /dashboard
QDRANT_URL=http://localhost:6333 uv run python code/ch04/qdrant_raw_client.py
```

## Exercises

1. In the raw script, change `Distance.COSINE` to `Distance.DOT`. Do the scores change? The
   ranking? Why (Chapter 2)?
2. Add a `should` filter that matches `doc_id` in {`14-faq`, `04-atlas-a2-specification`}.
   Then a `must_not`. Confirm the results with `scroll`.
3. Set `score_threshold` so that "how long does a full charge take" returns exactly the
   two relevant paragraphs. Now try that threshold on three other questions. Does one
   number work for all? (This is why Chapter 12 exists.)
4. Run `qdrant_langchain.py` against the Docker server and open the dashboard. Find your
   points and their payload. Create the payload index there and observe that it now works.
5. Write the LangChain filter for "chunks from the PTO policy OR the FAQ, but not chunk 0".

## Interview questions

**Q: Why use a vector database instead of numpy or Postgres?**
Approximate nearest-neighbour indexes (HNSW) make similarity search sub-linear; the DB
stores vectors with metadata and applies filters during the graph walk; it persists,
updates incrementally and handles concurrent reads/writes. Postgres with pgvector is a
legitimate vector database for moderate scale; numpy is a full scan with no metadata.

**Q: Explain HNSW.**
A layered proximity graph: each vector links to its m nearest neighbours; upper layers
are sparse for coarse navigation, lower layers dense for precision. A query greedily walks
from an entry point toward closer neighbours layer by layer. Roughly logarithmic search;
recall tuned by `ef` at query time and `m`/`ef_construct` at build time, trading RAM and
latency.

**Q: What is a payload, and why index it?**
Arbitrary JSON attached to each point: your metadata. A payload index lets Qdrant apply
`must`/`should`/`must_not` filters inside the HNSW traversal instead of post-filtering,
which keeps filtered queries fast and keeps k results even when the filter is very
selective (e.g. per-tenant).

**Q: How does LangChain's Qdrant integration store a Document?**
Payload `{"page_content": text, "metadata": {...}}`. Filters therefore address
`metadata.<key>`. The vector store returns Documents with the same metadata, so citations
and access filters flow through unchanged.

**Q: What is MMR and when do you use it?**
Maximal marginal relevance: pick results that are relevant to the query *and* dissimilar
to the results already picked. Use it when the top-k collapses onto one document or
near-duplicate chunks and you want coverage across sources.

**Q: A colleague sets `score_threshold=0.5` on a LangChain retriever and nothing gets
filtered. Why?**
`similarity_score_threshold` uses LangChain's relevance score, which rescales cosine to
0–1 via (cos + 1)/2; 0.5 corresponds to cosine 0. They should inspect
`similarity_search_with_relevance_scores` and pick a threshold on that scale, or threshold
raw cosine via `similarity_search_with_score` / Qdrant's own `score_threshold`.

**Q: In-memory vs local-path vs server Qdrant: when each?**
In-memory for tests. Embedded local path for single-process development (no payload
indexes, one process at a time). Server (Docker/Cloud) for anything shared, persistent
under concurrency, or that needs payload indexes, sharding, replication and snapshots.

## Key takeaways

- A vector DB = points (id, vector, payload) in collections, searched by nearness *plus*
  filter, with an ANN index (HNSW) that trades a little recall for a lot of speed.
- Stable ids make ingestion idempotent; filters make retrieval scoped; thresholds make
  "nothing relevant" expressible.
- `QdrantVectorStore` stores `page_content` and `metadata.*`: filter with the prefix.
- Hold your own client, create the collection yourself, then wrap it; `from_documents`
  wants to build its own client.
- Two score scales exist (cosine vs relevance). Know which one your threshold is on.

## Next

→ [Chapter 5: Chunking, Measured](05-chunking.md)

The 55 chunks we have been using were produced by a splitter we never questioned. Chapter 5
tries six chunking strategies on the same corpus and measures which one actually retrieves
the answers.
