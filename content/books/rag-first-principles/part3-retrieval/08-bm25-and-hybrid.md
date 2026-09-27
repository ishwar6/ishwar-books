# Chapter 8 · BM25 and Hybrid Search

> **Goal:** by the end of this chapter you can explain the BM25 formula on a whiteboard, implement it in 40 lines, know exactly which queries embeddings get wrong, and build a hybrid (dense + sparse) retriever in Qdrant three different ways: including the one that fuses results server-side in a single request.
>
> Files: [`code/ch08/bm25_from_scratch.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch08/bm25_from_scratch.py), [`code/ch08/hybrid_qdrant.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch08/hybrid_qdrant.py)

---

## 8.1 Why embeddings alone are not enough

Chapters 2–7 built a *dense* retriever: every chunk becomes one vector, the query becomes one vector, and we return the nearest ones by cosine similarity. It is excellent at **meaning**: "how fast can the robot move" finds the chunk that says "maximum speed 2.0 m/s" even though they share no words.

It is bad at **exact tokens**. An embedding model compresses a chunk into 1,536 numbers; a ticket id like `BEACON-4187`, a command like `beaconctl certs renew`, a version like `3.8`, a standard like `ISO 3691-4` or a customer name like `Veldmark` is a tiny fraction of that chunk's meaning and mostly gets averaged away. Users, however, type exactly these things: a support engineer pastes the ticket id, an ops person pastes the command.

This is the oldest problem in information retrieval, and it has a 30-year-old answer: keyword scoring. The standard formula is **BM25** (Best Match 25, Robertson & Walker, 1994). Every search engine you have used ran some variant of it. In RAG it is the "sparse" half of **hybrid search**.

(The interview transcript calls it "PM25". It is BM25.)

## 8.2 From TF-IDF to BM25

Start with the intuition:

- A document mentioning the query term *more often* is more relevant → **term frequency (TF)**.
- A term that appears in *every* document ("the", "Beacon" in a Beacon manual) tells you nothing → down-weight by **inverse document frequency (IDF)**.
- A long document mentions everything a bit more often just by being long → **length normalisation**.

BM25 is TF-IDF with two fixes that matter in practice: TF **saturates** (the 10th occurrence adds far less than the 2nd), and length normalisation is tunable.

```
score(q, d) = Σ_{t ∈ q}  IDF(t) · tf(t,d) · (k1 + 1) / ( tf(t,d) + k1 · (1 − b + b · |d| / avgdl) )

IDF(t) = ln( (N − n_t + 0.5) / (n_t + 0.5) + 1 )
```

| Symbol | Meaning | Typical value |
|---|---|---|
| `tf(t,d)` | how many times term `t` occurs in chunk `d` | - |
| `N`, `n_t` | number of chunks; number of chunks containing `t` | - |
| `|d|`, `avgdl` | chunk length; average chunk length (in tokens) | - |
| `k1` | TF saturation: how fast repeats stop helping | 1.2–2.0 (we use 1.5) |
| `b` | length normalisation: 0 = ignore length, 1 = fully normalise | 0.75 |
| `+1` inside the log | keeps IDF positive for very common terms | - |

Two things to notice. With `k1 → 0` the TF factor becomes a constant: BM25 turns into "count matching terms weighted by IDF". With `b = 0` the denominator stops depending on length. Those two knobs are the whole story; everything else is bookkeeping.

## 8.3 BM25 in 40 lines

[`code/ch08/bm25_from_scratch.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch08/bm25_from_scratch.py):

```python
def tokenize(text: str) -> list[str]:
    """Keep digits, dots and hyphens inside tokens so "3.8", "BEACON-4187" and
    "beaconctl" survive as searchable units."""
    return re.findall(r"[a-z0-9][a-z0-9.\-]*", text.lower())

class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tokenized = [tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.tokenized]
        self.avgdl = sum(self.doc_len) / len(docs)
        self.tf = [Counter(t) for t in self.tokenized]
        df = Counter(term for toks in self.tokenized for term in set(toks))
        n = len(docs)
        self.idf = {t: math.log((n - c + 0.5) / (c + 0.5) + 1) for t, c in df.items()}

    def score(self, query: str, i: int) -> float:
        s = 0.0
        for t in tokenize(query):
            if t not in self.tf[i]:
                continue
            f = self.tf[i][t]
            norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
            s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + norm)
        return s
```

The **tokenizer is half the algorithm**. The default tokenizer in most libraries splits on whitespace or on `\w+`, which turns `(BEACON-4187)` into `beacon` and `4187`: and now "BEACON-4187" matches every chunk that mentions Beacon. We will see this bite the Qdrant sparse model in 8.6.

The script checks our implementation against `rank_bm25.BM25Okapi`, the reference library. Absolute scores differ slightly (rank_bm25 uses a different IDF smoothing) but the **ranking is identical**, which is what matters:

```
OURS      q='beaconctl rollout undo'
  10.559  08-runbook-fleet-outage.md    '## Diagnosis checklist ...'
   8.039  05-beacon-fleet-software.md   '## Release process ...'
   3.011  11-release-notes-beacon-4.2.md
rank_bm25 q='beaconctl rollout undo'
  10.320  08-runbook-fleet-outage.md
   7.861  05-beacon-fleet-software.md
   2.880  11-release-notes-beacon-4.2.md
✓ same top-3 ordering as rank_bm25
```

## 8.4 BM25 as a LangChain retriever

You rarely ship your own BM25. `langchain_community.retrievers.BM25Retriever` wraps `rank_bm25` behind the standard `Retriever` interface, so it drops into the Chapter 6 pipeline unchanged:

```python
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(chunks, k=3, preprocess_func=tokenize)
docs = bm25_retriever.invoke("BEACON-4187")
```

Always pass your own `preprocess_func`. And note a BM25 quirk: it always returns `k` results even if only one chunk matched: the rest have score 0 and are effectively random. If you use BM25 alone, drop zero-score results.

## 8.5 Where each retriever wins

The script runs four queries through both:

```
q='BEACON-4187'
  BM25 : ['11-release-notes-beacon-4.2.md', ...]      ← the ticket is in the release notes
  dense: ['05-beacon-fleet-software.md', '14-faq.md', ...]   ← embeddings just see "Beacon"

q='firmware 3.8'
  BM25 : ['11-release-notes-beacon-4.2.md', '04-atlas-a2-specification.md', ...]
  dense: ['04-atlas-a2-specification.md', '11-release-notes-beacon-4.2.md', ...]   ← both fine

q='how fast can the robot move'
  BM25 : ['14-faq.md', '01-company-overview.md', '07-onboarding-guide.md']   ← no word overlap with "maximum speed"
  dense: ['04-atlas-a2-specification.md'] ×3                                  ← meaning wins

q='what happens if I get sick for a week'
  BM25 : ['14-faq.md', '14-faq.md', '02-pto-and-leave-policy.md']
  dense: ['02-pto-and-leave-policy.md', '02-pto-and-leave-policy.md', ...]
```

| Query type | Winner | Why |
|---|---|---|
| ids, codes, commands, SKUs, error strings | BM25 | exact rare tokens carry huge IDF |
| version numbers, dates, names | BM25 (dense is OK) | embeddings blur digits |
| paraphrases, synonyms, questions vs statements | dense | no lexical overlap |
| long natural-language questions | dense | BM25 is diluted by stop-words |
| multilingual / typos | dense (with a multilingual model) | BM25 is literal |

Neither is "better". Real query logs contain both kinds, so production systems run both and **fuse**.

> Why rank fusion rather than adding the scores, where the `k = 60` in RRF comes from, weighted
> RRF, DBSF, BM25's `k1`/`b` derived, and learned sparse retrieval: 
> [Chapter 23](../part8-deep-dives/23-fusion-and-sparse.md).

## 8.6 Hybrid search, three ways

### A. Reciprocal Rank Fusion by hand

Dense scores are cosines in ~[0.2, 0.8]; BM25 scores are unbounded. You cannot add them. **RRF** (Cormack et al., 2009) ignores scores and fuses **ranks**:

```
RRF(d) = Σ_{lists L}  1 / (k + rank_L(d))        k = 60
```

A document ranked 1st in one list and absent from the other gets 1/61 ≈ 0.0164; one ranked 3rd in both gets 2/63 ≈ 0.0317: agreement beats a single strong opinion. `k=60` is the value from the paper and works everywhere; smaller `k` makes top ranks dominate.

```python
def rrf(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, doc_id in enumerate(lst, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
```

That is the whole algorithm. `EnsembleRetriever` in `langchain_classic` does exactly this with weights; now you do not need it.

### B. Qdrant-native hybrid through LangChain

Qdrant can store **several named vectors per point**, including *sparse* vectors: a `{token_id: weight}` dictionary, which is what BM25 is once you write it down. So instead of running BM25 in Python over all chunks, you store the sparse vector next to the dense one and let Qdrant search both.

Two ingredients:

1. A sparse *embedding model*. `FastEmbedSparse(model_name="Qdrant/bm25")` runs locally (downloads a few MB once) and turns text into a sparse vector using BM25 weights.
2. A collection with **both** vector configs: this has to be created before wrapping it in `QdrantVectorStore`:

```python
client.create_collection(
    collection_name=collection,
    vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
    sparse_vectors_config={"sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))},
)
store = QdrantVectorStore(
    client=client, collection_name=collection,
    embedding=dense,                       # OpenAI
    sparse_embedding=sparse,               # FastEmbedSparse("Qdrant/bm25")
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name="dense", sparse_vector_name="sparse",
)
store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
```

After that, `store.similarity_search(q, k)` is a hybrid query. `RetrievalMode.DENSE` and `RetrievalMode.SPARSE` give you either half alone from the same collection: handy for A/B tests.

### C. The raw client: `query_points` with `prefetch`

This is what B does under the hood, and what you write when you outgrow the wrapper (custom limits per branch, filters per branch, a re-scoring stage):

```python
res = client.query_points(
    collection_name=...,
    prefetch=[
        models.Prefetch(query=models.SparseVector(indices=sp.indices, values=sp.values), using="sparse", limit=20),
        models.Prefetch(query=dense_vec, using="dense", limit=20),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),   # or models.Fusion.DBSF
    limit=5,
)
```

Each `Prefetch` is one candidate list (top-20 sparse, top-20 dense); the outer `query` fuses them **inside Qdrant**, one network round-trip. `Fusion.DBSF` (distribution-based score fusion) normalises each list's scores by their mean/stddev and adds them: it uses score magnitudes where RRF only uses ranks, so it can be sharper when one retriever is confidently right, and noisier when it is confidently wrong. Start with RRF.

The same `prefetch` mechanism does **multi-stage retrieval**: prefetch 1,000 with a cheap quantised vector, re-score the survivors with the full vector (Chapter 18).

## 8.7 Measuring it

`hybrid_qdrant.py` runs all four retrievers over the 42 answerable golden questions and reports **hit-rate@k**: did any of the top-k chunks come from a gold source document?

```
hit-rate@k over 42 answerable golden questions
retriever            @1      @3      @5   misses@1
dense              1.00    1.00    1.00   []
bm25               0.74    0.95    0.98   ['q01', 'q02', 'q04', 'q06', 'q07', 'q15', 'q17', 'q22', 'q31', 'q40', 'q42']
rrf_by_hand        0.98    1.00    1.00   ['q07']
qdrant_hybrid      0.95    1.00    1.00   ['q14', 'q17']
```

Read this honestly:

- **Dense is already perfect on this benchmark.** Fourteen well-written documents and full-sentence questions is the easiest case for embeddings. On a real corpus (thousands of docs, near-duplicates, users typing fragments) dense hit-rate@1 is typically 0.6–0.8 and hybrid adds 5–15 points. Do not conclude from a toy corpus that hybrid is useless; conclude that **you must measure on your own queries** (Chapter 10).
- **BM25 alone loses 26% at @1**: on paraphrased questions, exactly as 8.5 predicted.
- **Hybrid can slightly hurt** on clean queries (q14, q17 dropped to rank 2) because a noisy BM25 list dilutes a correct dense list. RRF's `k` and per-branch `limit` control this.

The ID-style probe at the end shows the tokenizer lesson from 8.3 in the wild:

```
query                 dense                                 qdrant_hybrid
BEACON-4187           05-beacon-fleet-software.md           05-beacon-fleet-software.md   ← both wrong!
beaconctl certs renew 08-runbook-fleet-outage.md            08-runbook-fleet-outage.md
Veldmark              12-postmortem-...                     12-postmortem-...
```

Our hand-rolled RRF with our tokenizer put the release notes first for `BEACON-4187 fix`; the `Qdrant/bm25` sparse model tokenizes `BEACON-4187` into `beacon` + `4187` and the id's signal is diluted by every Beacon chunk. **When ids matter, control the tokenizer**: or add an exact-match payload filter (Chapter 9) for strings that look like ids.

## 8.8 When to use what

| Situation | Recommendation |
|---|---|
| Prototype, < 1k chunks | dense only; add BM25 only when you *see* keyword failures |
| Support / ops / legal / code corpora | hybrid from day one: ids and exact phrases are everywhere |
| Multilingual users | dense with a multilingual embedding model; BM25 per language if needed |
| Huge corpus, latency-critical | Qdrant-native hybrid (one round-trip, server-side fusion) |
| Two retrievers you cannot co-locate | RRF in Python |

## Run it

```bash
QDRANT_MODE=memory uv run python code/ch08/bm25_from_scratch.py
QDRANT_MODE=memory uv run python code/ch08/hybrid_qdrant.py     # first run downloads Qdrant/bm25 (~few MB)
```

Expected: the first script prints our scores next to rank_bm25's with identical ordering, then the four lexical-vs-dense probes. The second prints the three hybrid implementations on `BEACON-4187 fix`, then the hit-rate table above and the ID-style probe.

## Exercises

1. Change `b` to 0 and to 1 in `BM25` and re-run the `beaconctl rollout undo` query. Which chunk moves, and why? (Look at chunk lengths.)
2. Break the tokenizer on purpose: replace it with `text.lower().split()`. Which of the four probe queries change?
3. Set RRF `k=1` and `k=1000`. Explain the hit-rate@1 change in one sentence each.
4. Swap `Fusion.RRF` for `Fusion.DBSF` in `raw_hybrid` and compare the top-3 for three golden questions.
5. Add a fifth retriever to the table: `RetrievalMode.SPARSE` alone on the Qdrant collection. Does it match your hand BM25? If not, why not?

## Interview questions

**Q: Explain BM25 to me.**
A ranking function for keyword search. For each query term in a document it multiplies an IDF weight (rare terms count more) by a saturating term-frequency factor (repeats help less and less, controlled by `k1`) and normalises by document length relative to the average (controlled by `b`). Scores are summed over query terms. It is the baseline every neural retriever is compared against and it still wins on exact-match queries.

**Q: What does hybrid search mean in RAG and why do you need it?**
Running a lexical retriever (BM25 / sparse vectors) and a semantic retriever (dense embeddings) on the same query and fusing the results. Embeddings capture paraphrase and intent but blur rare exact tokens: ids, codes, versions, names. Keyword search is the opposite. Real query logs contain both kinds of queries, so hybrid raises recall on the tail without hurting the head.

**Q: How do you combine scores from BM25 and cosine similarity? They are on different scales.**
You don't combine scores; you combine ranks with Reciprocal Rank Fusion: each result gets Σ 1/(k + rank) over the lists it appears in, k≈60. No normalisation, works for any number of retrievers. Alternatives: DBSF (z-score normalise each list then add), or learned fusion weights if you have click data.

**Q: What is a sparse vector and how does Qdrant use it for hybrid search?**
A vector with mostly zeros, stored as `{index: value}` pairs: BM25 weights over a vocabulary are exactly that. Qdrant lets a point carry named dense and sparse vectors; a `query_points` call with two `Prefetch` branches (one per vector) and a `FusionQuery(RRF)` performs hybrid search server-side in one request.

**Q: Your hybrid retriever returned worse results than dense alone. What happened?**
Most likely the sparse branch is noisy: bad tokenizer (ids split into common words), stop-words not handled, or too large a `limit` on the sparse prefetch flooding the fusion with irrelevant hits. Check per-branch results separately, tighten the sparse limit, fix tokenization, and verify on a labelled set rather than a few queries.

**Q: What are k1 and b?**
`k1` controls term-frequency saturation (higher = more credit for repeated terms; typical 1.2–2.0). `b` controls length normalisation (0 = none, 1 = full; typical 0.75). For short chunks of similar length `b` matters little; for mixed-length documents it matters a lot.

**Q: Would you ever use BM25 without embeddings?**
Yes: when queries are mostly ids/codes (log search, ticket search), when you cannot run an embedding model (air-gapped, cost), or as a cheap first-stage filter before an expensive re-ranker. And always as a baseline in your eval table: if dense does not beat BM25 on your data, something is wrong with your embeddings or chunking.

## Key takeaways

- Embeddings model meaning, not tokens; BM25 models tokens, not meaning. Production query logs need both.
- BM25 = IDF × saturating TF × length normalisation. Two knobs: `k1`, `b`. The tokenizer decides whether ids are searchable.
- Never add heterogeneous scores; fuse ranks with RRF (k=60) or use Qdrant's server-side `FusionQuery`.
- Qdrant stores dense and sparse vectors on the same point; `RetrievalMode.HYBRID` in LangChain or `prefetch` in the raw client gives one-round-trip hybrid search.
- A toy benchmark saturates at 1.00; the value of hybrid shows up on real, messy queries: measure on yours.

## Next

→ [Chapter 9: Advanced Retrieval: Re-ranking, Query Transforms, Parent/Child, Metadata](09-advanced-retrieval.md)

Chapter 9: re-ranking, query rewriting, parent/child chunks, metadata filters: the techniques that fix the *specific* failures your eval table shows.
