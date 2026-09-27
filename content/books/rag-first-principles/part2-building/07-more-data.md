# Chapter 7 · More Data: Growing and Maintaining the Corpus

> **Goal:** you can grow a corpus three ways (download, synthesize, ingest incrementally),
> run ingestion in batches with dedup and a manifest so it is idempotent *and* handles
> deletions, estimate time and cost, and (crucially) measure what more data did to
> retrieval instead of assuming. You will also learn what "noise" does and does not do.

Code: [`code/ch07/download_gutenberg.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch07/download_gutenberg.py), [`code/ch07/synthesize_docs.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch07/synthesize_docs.py),
[`code/ch07/ingest_batch.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch07/ingest_batch.py).

---

## 7.1 Fifty-six chunks is a demo

Real corpora have three properties the handbook lacks: they are **big** (retrieval must
find 5 chunks among millions), **mostly irrelevant to any given question** (the right
chunk competes with thousands of distractors), and **they change** (documents are added,
edited, removed every day). This chapter adds all three to our small world.

## 7.2 Three ways to get more data

**(a) Download something public.** `download_gutenberg.py` fetches three public-domain
novels (≈2.4 MB of text) into `data/gutenberg/` and strips Project Gutenberg's licence
header and footer:

```python
r = httpx.get(url, timeout=30, follow_redirects=True)
r.raise_for_status()
path.write_text(strip_gutenberg_boilerplate(r.text))
```

Network calls fail; the script prints `FAILED … skipping` and continues, and the rest of
the chapter works with whatever arrived. They are **noise**: nobody will ask the handbook
about whales. That is exactly what most of a real corpus is relative to any one question.

**(b) Synthesize with the LLM.** `synthesize_docs.py` asks the model for new handbook-style
documents *together with* three question/answer pairs each, as structured output:

```python
class GeneratedDoc(BaseModel):
    title: str
    body: str = Field(description="400-600 words of markdown with headings, concrete numbers, ... at least one table")
    qa_pairs: list[QA] = Field(min_length=3, max_length=3)

llm = get_llm().with_structured_output(GeneratedDoc)
doc = llm.invoke(f"You write internal policy documents for Lumora Robotics ... Write the document: {topic}")
```

`with_structured_output(PydanticModel)` makes the model return a validated object, not
prose you have to parse. Five documents and 15 golden pairs land in `data/generated/`:

```
wrote g01-procurement-and-purchasing-approvals.md  (529 words, 3 QA pairs)
wrote g02-code-review-and-branching-standards-for-the-fleet-.md  (700 words, 3 QA pairs)
...
15 generated Q/A pairs in data/generated/qa.yaml
```

Synthetic data is how you get labelled evaluation data when you have none, and how you
stress-test a system with more documents *about the same domain*: the hard kind of
distractor. It comes with a warning you can see in the first generated file (yours will
differ: the model invents new details every run): ours says international travel "requires
approval at least **10 business days** before departure" and sets its own purchasing
thresholds (department head from **$501**). The real travel policy says book **14 days**
ahead and get written pre-approval above **₹1,00,000 (~$1,200)**. **Synthetic documents
introduce contradictions with the real ones.** In this book that is a feature (Chapters 11–12 need
conflicts to work on); in production, generated documents must never be indexed alongside
authoritative ones without a `kind: generated` flag you can filter on: which is exactly
what `ingest_batch.py` stamps on them.

**(c) Ingest incrementally**, which is the rest of the chapter.

## 7.3 Content-hash ids: dedup for free

Chapter 6 used `uuid5(doc_id::chunk_index)`: *location* ids. `ingest_batch.py` uses
*content* ids:

```python
def content_id(text, source):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, hashlib.sha256(f"{source}\n{text}".encode()).hexdigest()))
```

Same text in the same source → same id → the second ingest is a no-op. Change one word →
new id (and the old one is deleted by the manifest, below). Two identical paragraphs in one
document → one point. Location ids cannot do the last two.

## 7.4 The manifest: knowing what is already indexed

```python
manifest = json.loads(MANIFEST.read_text())          # {point_id: source} from the last run
wanted   = {content_id(c.page_content, c.metadata["source"]): c for c in chunks}
new_ids  = [i for i in wanted if i not in manifest]  # embed only these
gone_ids = [i for i in manifest if i not in wanted]  # delete these
```

That diff is the entire incremental-ingestion algorithm. In production the manifest is a
table (Postgres, DynamoDB) keyed by point id with source, content hash and timestamp; here
it is `data/.manifest_ch07.json`. Watch it work across the four phases of the run:

```
1) handbook only
  56 chunks wanted, 0 already indexed -> 56 to embed, 0 to delete
2) re-run with the same input (should embed nothing)
  56 chunks wanted, 56 already indexed -> 0 to embed, 0 to delete
3) add generated docs + Gutenberg noise
  997 chunks wanted, 56 already indexed -> 941 to embed, 0 to delete
4) remove the noise again (manifest deletes the vanished chunks)
  97 chunks wanted, 997 already indexed -> 0 to embed, 900 to delete
  97 points remain
```

Phase 2 is the idempotency test: zero API calls. Phase 4 is the deletion test that
location ids fail: 900 points whose source vanished are removed with one `delete` call.

## 7.5 Batching, rate limits, throughput, cost

```python
for start in range(0, len(new_ids), BATCH):            # BATCH = 100
    store.add_documents([wanted[i] for i in batch_ids], ids=batch_ids)
```

```
  embedded 941 chunks in 18.6s = 51 chunks/s, ~140,228 tokens, ~$0.0028
```

The numbers that matter, and how to think about them:

- **Batch size.** One HTTP call per chunk is dominated by latency; one call for 10,000
  chunks hits request-size limits and loses everything on a failure. 64–256 texts per
  embedding call is the sweet spot. `OpenAIEmbeddings` additionally splits by its own
  `chunk_size` (default 1000 texts) and token limits internally.
- **Rate limits.** OpenAI throttles by requests/minute and tokens/minute per model. The
  client retries `429` and `5xx` with exponential backoff (default `max_retries=2`); for
  bulk jobs raise it (`OpenAIEmbeddings(max_retries=6)`) and honour `Retry-After`. If you
  write your own loop, back off exponentially with jitter: *never* retry immediately in a
  tight loop, that is how you turn a 60-second throttle into a 10-minute one.
- **Throughput.** ~50 chunks/s single-threaded here. The bottleneck is network round-trips,
  so run 4–8 batches concurrently (threads or `asyncio` with `aembed_documents`) and you
  approach the tokens/minute ceiling. Qdrant upserts are not the bottleneck until you are
  well past 1,000 points/s.
- **Cost.** ~4 characters per token; `$0.02` per million tokens for `-small`. 941 chunks ≈
  140 k tokens ≈ **$0.003**. Embedding is the cheap part of RAG; it is the LLM at query
  time that costs money (Chapter 6).

**Scaling preview: 1 GB to 10 GB.** Straight multiplication, and it stays linear:

| | 1 GB text | 10 GB text |
|---|---|---|
| tokens | ~250 M | ~2.5 B |
| chunks @ 500 tokens | ~500 k | ~5 M |
| embedding cost (`-small`) | ~$5 | ~$50 |
| embedding time @ 50 chunks/s, 8 workers | ~20 min | ~3.5 h |
| vectors, 1536 × float32 | ~3 GB | ~30 GB |
| vectors, 1536 × int8 (scalar quantization) | ~0.8 GB | ~8 GB |

What does *not* stay linear: query latency (HNSW is ~log n, so 10× the data is a few more
hops), and **quality**: more chunks means more near-misses competing for the top-k, which
is where Chapter 8's hybrid search and Chapter 9's reranking earn their keep. RAM for the
HNSW index and vectors is the cost that hurts at 10 GB; Chapter 18 covers quantization,
on-disk vectors and sharding.

## 7.6 What more data did to retrieval: measured

The run answers the 42 golden questions before and after adding 941 chunks of noise:

```
1) handbook only            hit@5 1.00  hit@1 1.00  noise@5 0.00   (56 points)
3) + generated + gutenberg  hit@5 1.00  hit@1 1.00  noise@5 0.15   (997 points)
```

Two findings, and both matter more than the cliché "more data hurts".

**Irrelevant noise did not move hit rate at all.** Three novels are so far from "how many
PTO days" in embedding space that they never reach the top 5 *for the right document*.
Dense retrieval is robust to *off-topic* volume. Do not fear big corpora for that reason.

**But 15% of the context slots were taken by noise.** `noise@5` is the share of the top-5
that came from generated or Gutenberg chunks. The right document still ranks first, and
then positions 2–5 fill with a synthetic policy that *sounds* like the handbook (the
generated docs, not the novels: in our run 30 of the 32 polluted slots were generated
docs, spread over 19 of the 42 questions: check by printing sources). The model now reads
contradicting numbers alongside the right ones. Hit rate is blind to this; the answer is
not. This is the real cost of more data: **context pollution by on-topic near-misses**,
and it is what a reranker (Chapter 9), a score threshold (Chapter 12) and a `kind` filter
fix.

Rule: when you add data, measure **hit@k, hit@1 and what else is in the top-k**. The
third one is where quality actually degrades.

## 7.7 Ingestion as a pipeline

What we ran as one script is, in production, a pipeline with these stages: each of which
is where something goes wrong at scale:

```
source change event ──► queue ──► parse ──► chunk ──► hash+diff ──► embed (batched, rate-limited) ──► upsert ──► manifest ──► metrics
   (webhook, cron)      (SQS,      (per       (per      (skip         (N workers,                       (Qdrant)    (table)   (count, lag,
                        Pub/Sub)   worker)    doc)      unchanged)     backoff)                                                 failures)
```

- **Idempotency** at every stage (content ids, manifest) so a crashed worker can be
  replayed.
- **Backpressure**: the queue absorbs a 10,000-document dump without the embedding API
  throttling everything else.
- **Observability**: documents waiting, chunks/min, embedding errors, *index freshness*
  ("the newest indexed document is 4 minutes old"): the number your users actually feel.
- **Reindexing**: changing the embedding model or chunk size means a full rebuild into a
  *new* collection, then an alias swap (Qdrant supports collection aliases). Never rebuild
  in place under traffic.

Chapter 18 sizes this for a terabyte.

## Run it

```bash
uv run python code/ch07/download_gutenberg.py                 # ~2.4 MB, once
uv run python code/ch07/synthesize_docs.py --n 5              # 5 LLM calls, ~1 minute, ~1 cent
QDRANT_MODE=memory uv run python code/ch07/ingest_batch.py    # ~1 minute, ~$0.003
```

Expected: `saved pride-and-prejudice.txt (734 KB)` ×3; five `wrote gNN-….md` lines; then
the four-phase output quoted in §7.4–7.6. Re-running `synthesize_docs.py` says `have …`
and costs nothing. Pass `--max-chunks-per-book 50` to `ingest_batch.py` for a faster run.

## Exercises

1. Print the sources of the noise in the top-5 for phase 3. Are they novels or generated
   docs? Filter by `metadata.kind` and confirm that excluding `generated` restores
   `noise@5` to ~0.
2. Edit one sentence in [`data/handbook/02-pto-and-leave-policy.md`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/data/handbook/02-pto-and-leave-policy.md) and re-run
   `ingest_batch.py`. How many chunks were re-embedded? How many deleted? Revert the edit.
3. Set `BATCH = 10` and then `BATCH = 500`. Measure chunks/s. Explain the shape.
4. Make ingestion concurrent: embed batches with a `ThreadPoolExecutor(max_workers=4)`.
   What speedup do you get, and when do you first see a `429`?
5. The generated docs contradict the handbook (read your `data/generated/g01-*.md` next to
   [`data/handbook/03-travel-and-expense-policy.md`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/data/handbook/03-travel-and-expense-policy.md) and list the disagreements: ours had
   10 business days' travel approval vs 14 days' advance booking). Ask
   [`code/ch06/rag_cli.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch06/rag_cli.py) a question that hits one of them against a collection that
   contains both. What happens? (Save the answer for Chapter 11.)

## Interview questions

**Q: How would you design the ingestion pipeline for a corpus that changes daily?**
Event-driven: source changes go to a queue; workers parse, chunk, compute content hashes
and diff against a manifest so only new or changed chunks are embedded, in batches with
exponential backoff against rate limits; upsert by stable id; delete ids whose source
vanished; record freshness and error metrics. Full re-embeds (new model, new chunking) go
to a new collection behind an alias.

**Q: How do you avoid duplicate vectors?**
Deterministic ids derived from content (hash of source + text). Re-ingesting identical
content upserts the same id. A manifest of indexed ids lets you skip unchanged chunks
entirely and delete stale ones.

**Q: Going from 1 GB to 10 GB of documents: what changes?**
Linearly: chunk count, embedding cost (~$5 → $50 with small models), ingestion time,
vector storage (~3 → 30 GB float32, ~4× less with int8 quantization). Sub-linearly: query
latency (HNSW ~log n). Non-obviously: retrieval quality, because more on-topic near-misses
compete for the top-k: which drives you to hybrid search, reranking and metadata filters.
RAM becomes the cost driver, so quantization and on-disk storage enter.

**Q: How do you handle embedding API rate limits during bulk ingestion?**
Batch 64–256 texts per call, run a bounded number of concurrent workers, retry 429/5xx with
exponential backoff and jitter honouring `Retry-After`, and put a queue in front so a
burst of documents does not starve query-time embedding. Track tokens/minute against the
quota.

**Q: Does adding irrelevant data hurt retrieval?**
Off-topic volume barely affects dense retrieval: the right chunks stay at the top. What
hurts is *on-topic* near-duplicates and contradicting documents filling the remaining
top-k slots, polluting the context. Measure not just hit rate but what else is in the
top-k, and control it with filters, thresholds and reranking.

**Q: How would you use an LLM to create evaluation data?**
Generate documents (or take real ones) and ask the model, with structured output, for
question/answer pairs grounded in each passage, recording the source id. Review a sample
by hand, flag synthetic items with a tag so they can be excluded from production
retrieval, and use them for recall and faithfulness evals. It bootstraps a golden set
when none exists.

**Q: Why is synthetic data risky in a production index?**
It can contradict authoritative documents with plausible numbers. If indexed without a
`kind`/`source_type` flag and filtered out at query time, the model will read both and may
pick the invented one. Tag everything at ingest; filter at retrieval.

## Key takeaways

- Real corpora are big, mostly irrelevant per question, and changing; build ingestion for
  all three from day one.
- Content-hash ids plus a manifest give you dedup, skip-unchanged and delete-removed:
the whole incremental algorithm is a set difference.
- Batch, back off, run a few workers; embedding is cheap in dollars, bounded by rate
  limits in time.
- Off-topic noise leaves hit rate alone; on-topic near-misses pollute the context. Measure
  what else is in the top-k.
- Synthetic documents are excellent evaluation material and dangerous production data.
  Tag them.

## Next

→ [Chapter 8: BM25 and Hybrid Search](../part3-retrieval/08-bm25-and-hybrid.md)

Chapter 8 addresses the gap we saw in Chapter 2 (embeddings are fuzzy on exact terms)
by adding keyword search (BM25) and fusing it with dense retrieval into hybrid search,
natively in Qdrant.
