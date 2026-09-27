# Chapter 22 · Vector Search Internals: HNSW, IVF and Quantization

> **Goal:** by the end of this chapter you can build an HNSW index from scratch, say exactly what
> `M`, `ef_construction` and `ef_search` trade against each other (with measured curves, not
> adjectives) and answer "how would you tune this for 10 million vectors?" as a procedure:
> ground truth, recall target, memory budget, `ef` sweep, p95 latency, re-measure after growth.
> You will also know why your recall can fall without anyone changing any code, and why
> "search then filter" is a bug rather than a design.
>
> Files: [`code/ch22/hnsw.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch22/hnsw.py), `parameter_study.py`, `recall_degradation.py`, `ivf_and_pq.py`,
> `filtered_search.py`.

[Chapter 4](../part2-building/04-qdrant.md) gave you HNSW in one page and a table of three knobs:
enough to configure a collection. That is the API layer. This chapter is underneath it: we build
the graph, break it deliberately, and measure what each parameter buys. Interviewers ask about
HNSW precisely because it separates people who have *used* a vector database from people who
understand what it is doing.

---

## 22.1 The thing we are approximating

Exact k-nearest-neighbour search is embarrassingly simple: compute the distance from the query to
every vector, sort, take k. For `n` vectors of `d` dimensions that is `n·d` multiply-adds per
query. No index, no training, perfect recall.

The arithmetic is why nobody ships it at scale:

| n | d | multiply-adds per query | rough single-core time |
|---|---|---|---|
| 10,000 | 1536 | 15 million | ~5 ms |
| 100,000 | 1536 | 154 million | ~50 ms |
| 1,000,000 | 1536 | 1.5 billion | ~500 ms |
| 10,000,000 | 1536 | 15 billion | ~5 s |

Those are single-core numbers; SIMD and threads buy you maybe 10–20×, which moves the problem
one order of magnitude and no further. Meanwhile your latency budget for retrieval is a few
milliseconds, because the LLM call after it already costs you a second (Chapter 13 measured
this: vector search was 1 ms of a 1,342 ms request).

So we trade exactness for speed. **Recall@k** is the price tag:

```
recall@k = |ANN top-k  ∩  exact top-k| / k
```

A recall of 0.95 at k=10 means that on average 9.5 of the 10 chunks you retrieve are the ones
exact search would have returned. For RAG this is usually fine (the reranker and the LLM smooth
over a swapped rank-9 chunk) but "usually fine" is not a number. The whole discipline of this
chapter is *measuring* it rather than assuming it.

One caveat worth holding onto from the start: approximate search is only a win when `n` is large.
[`code/ch22/parameter_study.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch22/parameter_study.py) measures our index doing **2,030 distance calls per query at
`ef_search=512` on a 2,000-vector index**: that is 101.5% of exact search. At small `n` the graph
walk is pure overhead, which is exactly why Qdrant keeps segments under `indexing_threshold`
in brute-force mode and does not build a graph at all. (That threshold is measured in kilobytes
of vector data, default 20,000: Qdrant's own "1 kB = 1 vector of size 256", so it is 20,000
points at 256 dimensions and only about 3,300 at 1536.)

## 22.2 From greedy walks to navigable small worlds

Put every vector in a graph, linking each to some of its nearest neighbours. To search: start at
some node, look at its neighbours, hop to whichever is closest to the query, repeat until no
neighbour improves. That is **greedy search**, and it costs you a handful of distance computations
per hop instead of `n`.

Two problems, and HNSW is exactly the two fixes.

**Problem 1: local minima.** If every link is short, the walk is a blind man feeling his way
across a country one step at a time. It gets stuck in whatever valley it started in.

```
   greedy walk on a short-link-only graph

   start                          query
     o---o---o                      x
          \                        /
           o---o  <-- stuck here: every neighbour of this
                     node is further from x than it is
```

The fix is a **beam**: instead of keeping the single best node, keep the best `ef` nodes found so
far and keep expanding the most promising unexplored one until none can improve. `ef` is the beam
width. This is the whole of `_search_layer` in [`code/ch22/hnsw.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch22/hnsw.py), and it is where `ef_search`
and `ef_construction` both land: they are the same parameter used at two different times.

**Problem 2: you need long links for coarse navigation and short links for precision.** A graph
with only long links overshoots; only short links crawls. HNSW's answer is a *hierarchy*: a stack
of graphs over the same points, where layer 0 holds everything and each layer above holds an
exponentially thinned sample.

```
  layer 2      o- - - - - - - - - - - -o          sparse, long hops
                \                     /
  layer 1      o-o- - - - -o- - - - -o-o          medium
               | |         |         | |
  layer 0   o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o       every point, short links

  search: enter at the top, greedily descend (ef=1, cheap), and only at
          layer 0 open the beam to ef_search.
```

A point is assigned to level `l` with probability that decays geometrically:
`level = floor(-ln(U) · mL)` with `mL = 1/ln(M)`. That constant is not arbitrary: it makes the
expected number of layers `~log_M(n)`, so the descent costs `O(log n)` hops. Our 2,000-vector
index builds **4 layers**; the 1,200-vector real-embedding index also builds 4.

This is the "navigable small world" idea: like a road network with motorways, A-roads and streets,
you can get anywhere in a few hops if the long-range links exist, and the hierarchy is a cheap way
to manufacture them.

## 22.3 Building it: [`code/ch22/hnsw.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch22/hnsw.py)

The implementation is ~200 lines of numpy and `heapq`, written to be read. The insertion path is
where the parameters live:

```python
def add(self, vec):
    idx = len(self.vectors); self.vectors.append(vec)
    level = self._random_level()                 # geometric: most points are layer-0 only

    # Phase 1: greedy descent with ef=1 - coarse navigation, cheap.
    ep = [self.entry_point]
    for l in range(self.max_layer, level, -1):
        ep = [self._search_layer(vec, ep, 1, l)[0][1]]

    # Phase 2: from `level` down to 0, search WIDE and link.
    for l in range(min(level, self.max_layer), -1, -1):
        found = self._search_layer(vec, ep, self.ef_construction, l)   # <-- ef_construction
        m = self.M0 if l == 0 else self.M                              # <-- M
        for n in self._select_neighbours(found, m):
            self._link(idx, n, l)
            self._prune(n, l)                    # links are bidirectional; neighbours overflow
        ep = [n for _, n in found]
```

Three details that only become obvious once you write it:

**Layer 0 gets `2M` links, not `M`.** Every point lives there, so it needs more connectivity to
stay navigable. Qdrant, hnswlib and Faiss all do this. When you set `m=16` you are buying 32 links
per node at the bottom plus 16 per node on each upper layer the point reaches.

**Links are bidirectional, so insertion damages your neighbours.** Adding `idx → n` also adds
`n → idx`, which can push `n` over its budget. `_prune` then re-selects `n`'s links. This is why
build time grows faster than linearly in `M` (measured below: M=4 builds in 1.0 s, M=64 in 21.7 s
for the same 2,000 points).

**The neighbour-selection heuristic is not an optimisation, it is the thing that makes the graph
work.** The obvious rule: keep the `M` nearest candidates: builds cliques: inside a dense cluster
every node links only to its own cluster, and no edge leaves. The paper's heuristic keeps a
candidate only if it is closer to the new point than to any already-selected neighbour, which
deliberately keeps edges pointing in *different directions*.

[`code/ch22/parameter_study.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch22/parameter_study.py) measures the difference at identical `M=8` and `ef_search=32`:

```
   selection  recall@10     links  ms/query
------------------------------------------------------------
   heuristic      0.820    32,670      0.25
       naive      0.740    32,204      0.24

heuristic - naive = +0.080 recall at identical M and ef_search.
```

Eight points of recall, same memory, same latency, purely from *which* neighbours you keep. If you
are ever asked "what is the clever part of HNSW?", this is a better answer than "the hierarchy".

## 22.4 The parameter study

Now the three knobs, measured on 2,000 uniform random 64-dimensional vectors (the pessimistic
case: see 22.5) with 25 queries and exact brute force as ground truth.

### `ef_search`: the query-time beam

```
 ef_search  recall@10  dist calls  ms/query   vs exact
------------------------------------------------------------
         8      0.692         367      0.14     18.4%
        16      0.792         486      0.17     24.3%
        32      0.928         761      0.29     38.0%
        64      0.992        1167      0.46     58.4%
       128      1.000        1607      0.74     80.4%
       256      1.000        1918      1.16     95.9%
       512      1.000        2030      1.56    101.5%
```

```
recall@10 vs ef_search (saturates)        distance calls (climbs ~linearly)
   1.000 |            *    *    *    *   2029.760 |                    *    *
   0.889 |       *                       1660.356 |               *
   0.778 |  *                            1290.951 |          *
   0.667 |                                921.547 |     *
         +---------------------------            552.142 |*    *
            8    16   32   64  128                       +--------------------
                                                            8  16  32  64 128
```

**Recall saturates; cost does not.** That asymmetry is the whole point. Going from `ef=64` to
`ef=512` costs you 74% more distance computations and 3.4× the latency to buy **zero** recall.
Every millisecond past the saturation point is waste, and the only way to find that point is to
measure it on your data.

`ef_search` must also be ≥ `k`: a beam narrower than the number of results you want is
incoherent, which is why Qdrant silently uses at least `limit` if you set `hnsw_ef` lower.

### `M`: links per node

```
   M  recall@10      links  link MB   vec MB  build s  ms/query
----------------------------------------------------------------
   4      0.692     16,608     0.07     0.51      1.0      0.29
   8      0.940     32,670     0.13     0.51      1.6      0.36
  16      0.992     64,238     0.26     0.51      3.7      0.49
  32      1.000    127,816     0.51     0.51     11.1      0.63
  64      1.000    247,490     0.99     0.51     21.7      0.79
```

Three readings:

- **Recall has sharply diminishing returns.** 4 → 8 buys 25 points; 16 → 32 buys 0.8; 32 → 64 buys
  nothing. The default of 16 is the default because it sits at the knee for typical data.
- **Memory is exactly linear in `M`** (0.07 → 0.99 MB as M goes 4 → 64). Links are a budget you
  spend, not a quality dial you turn up.
- **Build time grows super-linearly** (1.0 s → 21.7 s) because of the bidirectional pruning
  described above. `M` is the parameter you cannot change later without rebuilding, so it is the
  one to think hardest about.

At M=64 the graph (0.99 MB) costs more than the vectors themselves (0.51 MB): at 64 dimensions.
At 1536 dimensions the ratio inverts completely, which changes the advice; see the sizing table in
22.5.

### `ef_construction`: the build-time beam

```
 ef_constr   build s  recall@10      links
--------------------------------------------
        16       0.9      0.968     51,398
        50       3.3      0.988     64,172
       100       3.8      0.992     64,238
       200       4.7      0.992     64,212
       400       5.6      0.992     64,190
```

`ef_construction` controls how hard each insertion looks for good neighbours. It **saturates even
earlier than `ef_search`**: by 50–100 the graph is as good as it is going to get, and the extra
build time past that buys nothing measurable. Notice the link count stops changing too: the graph
is structurally converged.

This is the one parameter where the cost is paid once, offline, so the usual advice is "set it
higher than you think" (200 is a common production value). Our measurement says the *benefit*
plateaus; the reason to go higher anyway is that harder, clustered, higher-dimensional data
saturates later than this toy.

### The summary you should be able to recite

| Knob | When | Costs | Buys | Change later? |
|---|---|---|---|---|
| `M` | build | RAM (linear), build time (super-linear) | recall ceiling, graph connectivity | no: rebuild |
| `ef_construction` | build | build time only | graph quality (saturates ~100–200) | no: rebuild |
| `ef_search` | query | latency (linear), CPU | recall up to saturation | yes: per query |

"`M` and `ef_construction` are decisions; `ef_search` is a dial" is the sentence to have ready.

## 22.5 Tuning for 10 million vectors

This is the question that separates answers. It is not "M=16, ef=128"; it is a procedure.

**Step 0: know your data.** Uniform random vectors are the hardest possible case: in high
dimensions every point is roughly equidistant from every other, so there is no structure for the
graph to exploit. Real embeddings are clustered and anisotropic. [`code/ch22/embed_cache.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch22/embed_cache.py)
measures this on 1,500 OpenAI embeddings:

```
  mean pairwise cosine, real vectors   : +0.154  (std 0.109)
  mean pairwise cosine, random vectors : -0.000  (std 0.026)
```

The consequence is large. The same index code, same `M`, same `ef_construction`, on real
embeddings:

```
 ef_search  recall@10  dist calls  ms/query   vs exact      (n=1,200, dim=1,536)
------------------------------------------------------------
         8      0.995         191      0.21     15.9%
        16      1.000         249      0.24     20.7%
        32      1.000         353      0.32     29.5%
```

**Recall 0.995 at `ef_search=8`**, where the synthetic set needed `ef=128` for the same quality.
Never quote a recall number from someone else's benchmark; the distribution of *your* embeddings
decides it.

**Step 1: build ground truth.** Sample 1,000–10,000 real queries from production logs. Compute
exact top-k for each with brute force. This is a one-off batch job and it is not optional: without
it, "recall" is a word, not a number.

**Step 2: set a recall target from the product, not from taste.** If a reranker re-scores the top
50 and the LLM reads the top 5, your first stage needs high recall@50 and you do not care about
exact ordering. That argues for a lower `ef` than a system that feeds the top 5 straight to the
model. Chapter 24 makes this trade explicit.

**Step 3: pick `M` from the memory budget.** [`code/ch22/recall_degradation.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch22/recall_degradation.py) prints the
arithmetic:

```
RAM for n = 10,000,000 vectors, HNSW with M=16
per node links: layer0 keeps 2M=32 ids; only ~1/M of nodes reach layer 1 at
all, so the upper layers add ~M/(M-1)=1.1 ids on average. 4 bytes each
          -> ~132 bytes/node of graph (our 2,000-point index measures 32.1 ids/node at M=16)

   dim  vectors GB  graph GB  total GB   int8 GB  binary GB
------------------------------------------------------------
   384        15.4       1.3      16.7       5.2        1.8
   768        30.7       1.3      32.0       9.0        2.3
  1536        61.4       1.3      62.8      16.7        3.2
  3072       122.9       1.3     124.2      32.0        5.2
```

At 1536 dimensions the vectors are 61 GB and the graph is 1.3 GB. So: **raising `M` from 16 to 32
costs +1.3 GB; halving the dimension saves 30 GB.** Optimise dimension first (Matryoshka
truncation, Chapter 25), then quantization, then worry about `M`. Candidates get eliminated fast:
61 GB means a memory-optimised instance or on-disk vectors, not "we'll fit it on the API box".

**Step 4: sweep `ef_search` to the recall target, at minimum latency.** The script does this
end to end:

```
tuning ef_search for recall@10 >= 0.95 at n=4,000
 ef_search  recall@10   p50 ms   p95 ms  verdict
--------------------------------------------------
         8      0.540     0.16     0.18
        16      0.676     0.21     0.24
        32      0.856     0.33     0.36
        64      0.972     0.57     0.60  <- meets target
       128      0.996     0.95     0.99
       256      1.000     1.54     1.64
```

**Step 5: report p95 and p99, never the mean.** The beam explores a data-dependent number of
nodes, so latency has a tail even on identical hardware and an idle machine. A p50 of 0.57 ms with
a p95 of 0.60 ms is a healthy index; a p95 that is 5× the p50 usually means some queries land in
awkward regions of the graph, and it is a signal to raise `M` rather than `ef`.

**Step 6: re-measure after the index grows.** See 22.6.

A defensible starting point for 10M × 1536-d, stated as a starting point: `m=16`,
`ef_construct=200`, int8 scalar quantization with rescoring, `hnsw_ef` swept to hit the recall
target (typically 64–128), vectors on disk if the working set does not fit RAM, and a nightly job
that re-measures recall@10 against exact search on a 1,000-query sample.

## 22.6 Recall decays as you ingest: and nobody deployed anything

This is the most useful operational fact in the chapter, and it is a favourite interview scenario:
*"retrieval quality dropped last month, no code changed. Why?"*

[`code/ch22/recall_degradation.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch22/recall_degradation.py) holds `ef_search=32` and `M=16` fixed and grows the index:

```
recall@10 at FIXED ef_search=32, M=16, as the index grows

       n  recall@10  dist calls  % of exact  build s
------------------------------------------------------
     500      1.000         433      86.6%      0.6
   1,000      0.976         615      61.5%      1.6
   2,500      0.904         819      32.7%      5.4
   5,000      0.752         924      18.5%     13.1
  10,000      0.708        1036      10.4%     36.5
```

Recall falls from **1.000 to 0.708** with no change other than more data. The mechanism is in the
"% of exact" column: `ef_search` is a fixed *budget*, the graph grows, so that budget covers an
ever-smaller fraction of the space, and the number of near-ties you must disambiguate rises.

Three consequences for production:

- Recall is a property of **(index size, data distribution, `ef_search`)**. It is not a property of
  your code, and it will drift on its own.
- Put recall@k against exact search on a query sample into your monitoring, not just latency.
  Chapter 13's alert list should include it.
- When quality regresses with no deploy, the suspects are: index grew (this), index was rebuilt,
  embedding model changed under you, or the query mix changed. In that order.

## 22.7 Filtered search: why "search then filter" is a bug

Every real system filters: by tenant, by date, by document type, by permission. There are three
ways to combine a filter with a vector search and only two of them are correct.

```
  post-filter      ANN top-k  ->  drop non-matching  ->  fewer than k results (!)
  pre-filter       matching ids  ->  brute force over them  ->  correct, cost ∝ matches
  filtered ANN     walk the graph, collect only matching   ->  correct, cost ∝ 1/selectivity
```

[`code/ch22/filtered_search.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch22/filtered_search.py) measures all three on 4,000 vectors as the filter gets more
selective:

```
 selectivity       strategy  returned/k  recall@10  ms/query
------------------------------------------------------------------
        50%    post-filter     5.3/10       0.530      0.66
                  post x20    10.0/10       1.000      1.97
                pre-filter      10/10       1.000      0.10   (scans 1,995 vectors)
              filtered ANN    10.0/10       1.000      1.04   (2,259 dist calls)

         5%    post-filter     0.8/10       0.075      0.63
                  post x20     9.5/10       0.950      1.94
                pre-filter      10/10       1.000      0.01   (scans 208 vectors)
              filtered ANN    10.0/10       1.000      3.38   (4,035 dist calls)

         1%    post-filter     0.1/10       0.010      0.63
                  post x20     2.3/10       0.230      1.75
                pre-filter      10/10       1.000      0.00   (scans 42 vectors)
              filtered ANN    10.0/10       1.000      6.07   (4,048 dist calls)
```

Read that table slowly, because it contains four separate lessons.

**Post-filtering fails silently.** At 5% selectivity you asked for 10 results and got 0.8. Nothing
threw an exception; the user just saw a thin answer. This is the single most common home-made
RAG bug with multi-tenant data, and the reason it survives to production is that it looks fine at
50% selectivity in testing.

**Over-fetching is a patch, not a fix.** Fetching 20× more candidates rescues 5% selectivity
(9.5/10) and still collapses at 1% (2.3/10), while tripling latency. There is no constant
multiplier that is correct for an unknown filter.

**The two correct strategies have opposite cost curves.** As the filter gets *more* selective,
pre-filtering gets **cheaper** (0.10 → 0.00 ms; it scans 42 vectors instead of 1,995) while
filtered graph traversal gets **more expensive** (1.04 → 6.07 ms; the walk spends its time in
forbidden territory looking for the few allowed nodes). They cross over.

**Which is why the engine, not you, should choose.** Qdrant estimates the filter's cardinality
from the payload index and picks: graph traversal for loose filters, brute force over the matching
set for tight ones, switching at `full_scan_threshold` (default 10,000 kB of vectors, ~1,600
points at 1536 dimensions). This only works if
the field **has a payload index**: without one it cannot estimate cardinality or enumerate
matching ids, which is the real reason Chapter 4 told you to create them. For hard multi-tenancy
where one tenant is a tiny fraction of the collection, the fix is structural: `m=0` plus
`payload_m=16` builds a separate small graph per tenant instead of one global graph (Chapter 18).

The embedded Qdrant in this book brute-forces everything, so it cannot demonstrate the crossover:
which is exactly why `filtered_search.py` measures it on our own index and then shows the Qdrant
API separately. Be that explicit about your measurement setup in an interview; it reads as rigour,
not as an excuse.

## 22.8 IVF and PQ: the other two families

HNSW is a graph. The two other classical approaches partition or compress, and you should be able
to say when each wins because "why HNSW and not IVF?" is a natural follow-up.

**IVF (inverted file)** runs k-means over the vectors, assigns each to its nearest centroid, and at
query time searches only the `nprobe` nearest partitions exactly.

```
IVF - recall vs nprobe (n=8,000, dim=128, nlist=100)
 nprobe  recall@10  vectors scanned  % of exact
------------------------------------------------
      1      0.084               79       1.0%
      5      0.216              393       4.9%
     20      0.520            1,589      19.9%
     50      0.824            3,995      49.9%
    100      1.000            8,000     100.0%
```

`nprobe` is IVF's recall dial, exactly as `ef_search` is HNSW's. The characteristic IVF failure is
the boundary case: if the true neighbour sits just across a cell boundary and you probe one cell,
no amount of exact re-scoring inside that cell recovers it. Two operational differences matter more
than the recall curves: IVF needs a **training step** whose centroids are fitted to the data you
had then (distribution drift degrades it, so you retrain), and it handles **deletes and updates**
more gracefully than a graph does. HNSW needs no training, which is a genuine advantage for a
continuously-ingesting corpus.

**PQ (product quantization)** compresses instead of partitioning: split each vector into `m`
subvectors, run k-means with 256 centroids in each subspace, and store one byte per subvector. A
1536-dimensional float32 vector (6,144 bytes) becomes `m` bytes. Distances are computed by
**asymmetric distance computation**: the query stays full precision, you precompute its distance to
each subspace centroid, and then every database vector is a sum of `m` table lookups: no
multiplication at all.

On real 1536-d OpenAI embeddings:

```
 m (bytes)  compression  recall@10   recall@10
                            raw PQ +rescore100
-----------------------------------------------
        96          64x      0.730       1.000
       192          32x      0.815       1.000
```

**Raw PQ recall looks alarming; rescoring fixes it completely.** Retrieve a wide candidate set in
the compressed space, then re-rank those candidates with the original vectors. That is precisely
what `QuantizationSearchParams(rescore=True, oversampling=2.0)` does in Qdrant (Chapter 18), and
it is why "32× compression" is a statement about RAM, not about storage: you still keep the full
vectors somewhere to rescore against.

### Choosing

| Situation | Index |
|---|---|
| n < ~20,000, or a very selective filter | flat / brute force. Qdrant does this for you below `indexing_threshold` |
| general purpose, continuous ingest, updates | **HNSW**: no training, good recall, graceful |
| RAM-bound at 10M+ vectors | HNSW + scalar (int8) quantization with rescoring |
| extreme RAM pressure, 1024-d+ | binary quantization + rescoring, or IVF-PQ |
| billion-scale, batch workload, GPU available | IVF-PQ (Faiss), or DiskANN if SSD-resident |
| high-dimensional and latency-critical with a reranker after | HNSW at a *lower* `ef`, let the reranker fix ordering |

## 22.9 Operating a vector index

Tuning is not only the three parameters. The operational surface that a production question will
reach for:

- **Indexing threshold.** Qdrant leaves small segments unindexed on purpose (brute force is faster
  there: we measured it in 22.1). A freshly-ingested collection can therefore be *faster and more
  accurate* than a mature one, which confuses people during load tests.
- **Segments and optimizers.** Qdrant stores points in segments, merged and re-indexed by
  background optimizers. During a big ingest the optimizer is competing with your queries for CPU;
  p95 latency rises. Schedule bulk ingest away from peak, or throttle it.
- **Memmap vs in-RAM.** `on_disk=True` for vectors trades latency for RAM and relies on the page
  cache: the first query after a restart is slow (cold cache), steady state is fine. Warm the
  index after deploy if p99 on the first minute matters.
- **Rebuilds are not free.** Changing `m`, the distance metric, or the embedding model means a new
  collection. Do it blue/green behind an alias (Chapter 18): never in place.
- **What to monitor**: recall@k against exact on a query sample (the one everybody omits), p95/p99
  search latency, segment count and optimizer status, RAM vs resident set, ingest lag, and the
  index's point count against what your source of truth says it should be.

## The interview answer

**"What do M, ef_construction and ef_search do?"**

> M is how many links each node keeps in the graph: 2M on the bottom layer, which holds every
> point. It sets the recall ceiling and it is the memory you spend: memory is linear in M, about
> 130 bytes a node at M=16: 2M ids on layer 0 and roughly one more on the layers above, because
> only about one node in M ever reaches layer 1. ef_construction is the beam width while inserting, so it controls graph
> quality and build time; in my measurements it saturates around 50 to 100 and extra build time
> after that buys nothing. ef_search is the beam width at query time: the only one you can change
> per query. Recall saturates as you raise it but cost keeps growing linearly: on my test index,
> going from ef 64 to 512 cost 74% more distance computations for zero extra recall. So M and
> ef_construction are decisions you make at build time and can only change by rebuilding, while
> ef_search is a dial you tune against a latency budget.

**"How would you tune HNSW for 10 million vectors?"**

> As a procedure, not a preset. First I build ground truth: sample a few thousand real queries and
> compute exact top-k by brute force, because otherwise recall is not measurable. Then I set the
> recall target from the product: if a cross-encoder reranks the top 50, I need high recall at 50
> and I do not care about exact ordering, which lets me run a cheaper ef. Then M comes out of the
> memory budget: at 1536 dimensions, 10 million vectors is 61 GB of float32 vectors and only about
> 1.3 GB of graph at M=16, so the lever that matters is dimension and quantization, not M: int8
> with rescoring takes it to roughly 17 GB. Then I sweep ef_search to the lowest value that hits the
> recall target, and I report p95 and p99, not the mean, because beam search has a data-dependent
> tail. Finally I re-measure after the index grows, because recall at a fixed ef decays with n: I
> have measured it going from 1.00 to 0.71 as an index went from 500 to 10,000 vectors. So recall
> monitoring against an exact-search sample is part of the deployment, not a one-off.

**"Why not just use exact search?"**

> Because it is O(n·d) per query. At 10 million 1536-dimensional vectors that is about 15 billion
> multiply-adds, seconds of single-core time, and no amount of SIMD closes a three-order-of-
> magnitude gap. The flip side is that exact search *wins* when n is small: on a 2,000-vector
> index at ef 512 my HNSW did more distance computations than brute force. That is why Qdrant
> keeps small segments (under 20 MB of vector data) unindexed. Exact search is also the right answer when a
> filter is very selective: scanning the 42 matching vectors beats walking a graph looking for them.

## Run it

```bash
QDRANT_MODE=memory uv run python code/ch22/hnsw.py                 # ~4 s, the index itself
QDRANT_MODE=memory uv run python code/ch22/embed_cache.py          # ~10 s, one-off, <1 cent
QDRANT_MODE=memory uv run python code/ch22/parameter_study.py      # ~2 min, the three sweeps
QDRANT_MODE=memory uv run python code/ch22/parameter_study.py --quick   # ~20 s
QDRANT_MODE=memory uv run python code/ch22/recall_degradation.py   # ~70 s
QDRANT_MODE=memory uv run python code/ch22/ivf_and_pq.py           # ~45 s
QDRANT_MODE=memory uv run python code/ch22/filtered_search.py      # ~12 s
```

`hnsw.py` prints the shape of the whole chapter in six lines:

```
building HNSW: n=2000 dim=64 M=16 ef_construction=100
  layers=4  links=64,238  vector bytes=512,000  link bytes=256,952
  ef_search=  10  recall@10=0.690  distance calls/query=    365  (exact would be 2000)
  ef_search=  32  recall@10=0.930  distance calls/query=    765  (exact would be 2000)
  ef_search= 128  recall@10=1.000  distance calls/query=   1611  (exact would be 2000)
```

The **recall, distance-call, link-count and memory columns are deterministic**: they reproduce
exactly, because the index is seeded. The **millisecond columns are not**: they move 20–40% with
machine load, and they are pure Python, so treat them as *relative* costs within a single run, never
as absolute performance. Every *shape* (saturation, linear cost growth, decay with n, the filtering
crossover) reproduces regardless.

## Exercises

1. **Break the hierarchy.** Force `_random_level` to always return 0, so there is one flat layer.
   Re-run the `ef_search` sweep. How much recall do you lose at a fixed budget, and how many more
   distance calls do you need to get it back? This tells you what the layers are worth.
2. **Find your saturation point.** Run the `ef_search` sweep on the real embeddings with
   `n=1500`, then again after duplicating the data to `n=6000` (tile the array). Where does
   `ef_search` need to be for recall ≥ 0.98 in each case? Now you have measured 22.6 yourself.
3. **Make the heuristic matter more.** Generate clustered data (e.g. 20 Gaussian blobs) instead of
   uniform random, and re-run `heuristic_vs_naive`. The gap should widen: the naive rule's failure
   is a clustering failure. Explain the result in two sentences.
4. **Cost out your own corpus.** Take the sizing table and redo it for the corpus you actually
   work on: how many chunks, what dimension, what does it cost in RAM at float32, int8 and binary?
   Which of those fits on the instance you have?
5. **Fix a post-filter bug.** Write a retrieval function with the post-filter bug, run it at 5%
   selectivity, and watch it return 1 result for `k=10`. Then fix it two ways: pre-filter and
   Qdrant `query_filter`: and compare latency at 50% and 1% selectivity.

## Interview questions

**Q: Explain HNSW to someone who knows what a vector is but not what an index is.**
It is a graph where each vector links to some of its nearest neighbours, plus a hierarchy of
sparser graphs above it for coarse navigation. Search enters at the top, greedily hops toward the
query through the sparse upper layers, and then does a beam search at the bottom layer where every
point lives. Because the upper layers provide long-range links, you reach the right neighbourhood
in about log n hops instead of scanning everything, and the beam width controls how thoroughly you
explore once you are there. It is approximate: you trade a few percent of recall for two or three
orders of magnitude of speed.

**Q: What does M control, and what does raising it cost?**
M is the maximum links per node per layer, with 2M on layer 0. Raising it improves recall with
sharply diminishing returns (in my measurements 4 → 8 bought 25 points of recall, 16 → 32 bought
under one, and 32 → 64 bought nothing) while memory grows exactly linearly and build time grows
faster than linearly because every insertion can force neighbours to re-prune their link sets. M
cannot be changed without rebuilding the index, so it is a design decision, not a tuning dial.

**Q: Your p95 search latency doubled after a big ingest. What happened?**
Most likely the optimizer is re-indexing segments in the background and competing for CPU, or the
collection just crossed the indexing threshold and started building a graph. If it stays high, the
index is simply bigger: the beam walks more nodes. Check whether recall also moved: if recall fell
and latency rose, it is size; if latency rose and recall is unchanged, it is the optimizer or the
page cache after a restart. Either way, measure recall against exact search before touching
parameters.

**Q: Recall@10 was 0.97 in staging and 0.85 in production with the same config. Why?**
Different `n` and a different query distribution. `ef_search` is a fixed budget, so recall decays
as the index grows: I have measured 1.00 → 0.71 going from 500 to 10,000 vectors at a fixed
ef=32. Staging almost always has a smaller index. The second suspect is the data distribution:
clustered real embeddings are much easier than diverse ones, and production queries are messier
than the test set. The fix is to measure recall on production-sized data with production queries
and re-sweep `ef`.

**Q: What is the neighbour-selection heuristic and why does it exist?**
When inserting, you have a candidate list and need to keep M links. The naive rule keeps the M
nearest, which produces clusters that link only inward: the graph loses the edges that connect
regions and greedy search gets trapped. The heuristic keeps a candidate only if it is closer to the
new point than to any neighbour already kept, so the retained edges point in diverse directions. In
my measurement it was worth +0.08 recall at identical M, ef_search, memory and latency.

**Q: When is IVF a better choice than HNSW?**
When you are RAM-bound and can tolerate a training step: IVF-PQ compresses far more aggressively
and is the standard at billion scale, especially in batch or GPU settings. HNSW wins for
continuously-updated corpora because it needs no training: IVF's centroids are fitted to a
snapshot and degrade as the distribution drifts, so you retrain and re-assign. IVF also degrades
predictably at cell boundaries, where a true neighbour just outside the probed cells is simply
unreachable regardless of re-scoring.

**Q: How does product quantization work, and what is rescoring?**
Split each vector into m subvectors, cluster each subspace into 256 centroids, and store one byte
per subvector: a 1536-d float32 vector goes from 6,144 bytes to 96. At query time you keep the
query in full precision and precompute its distance to every centroid, so scoring a database vector
is m table lookups. Raw recall suffers: I measured 0.73 at 64× compression. Rescoring recovers it
completely: take a wide candidate set from the compressed index, re-rank with the original
vectors, and recall went back to 1.00. That is why compression is a RAM claim, not a storage claim.

**Q: How do you combine a metadata filter with a vector search?**
Never by searching and then filtering: that returns fewer than k results, silently. At 5%
selectivity I measured post-filtering returning 0.8 results out of 10, with recall 0.075. The two
correct strategies are pre-filtering (enumerate matching ids, brute force over them) and filtered
graph traversal (walk the graph, collect only matching nodes). Their costs move in opposite
directions as selectivity tightens, so a good engine picks between them using a cardinality
estimate from a payload index: which is why the filtered field must be indexed. For extreme
multi-tenancy you go further and build a per-tenant graph with `payload_m`.

**Q: How do you measure recall in production, where you have no ground truth?**
You manufacture it: sample real queries from logs, run exact brute-force search offline over the
same index contents, and treat that as ground truth. It is a batch job you can run nightly on a
sample of a few thousand queries; the cost is linear scans you are not doing in the request path.
Then you chart recall@k over time next to p95 latency, and you alert on the drop. Without this,
"our retrieval is fine" is an opinion.

**Q: Would you ever set `ef_search` lower than the recall you can afford?**
Yes: if there is a reranking stage after it. The first stage's job is candidate recall at, say,
top-50, not perfect ordering, and the cross-encoder fixes the order (Chapter 24). Spending latency
on `ef` to perfect an ordering you are about to discard is wasted budget. This is why "tune the
retrieval stack", not "tune HNSW", is the right frame.

**Q: Why does Qdrant sometimes ignore your HNSW settings entirely?**
Because below `indexing_threshold` (default 20,000 kB of vectors per segment, which is 20,000
points only if they are 256-dimensional) it does not build a graph:
brute force is genuinely faster on small segments, and I have measured a graph walk at high `ef`
doing *more* distance computations than exact search on a 2,000-vector index. The embedded/local
mode in this book always brute-forces. It is an important thing to know before you "benchmark"
HNSW parameters and conclude they do nothing.

## Key takeaways

- `M` and `ef_construction` are build-time decisions (RAM, graph quality, rebuild to change);
  `ef_search` is the per-query dial. Recall saturates in all three; cost does not.
- Real embeddings are clustered (mean pairwise cosine +0.15 vs 0.00 for random), so they need a far
  smaller `ef` than a synthetic benchmark suggests. Measure on your own vectors.
- Recall at fixed `ef` decays as the index grows: 1.00 → 0.71 from 500 to 10,000 vectors here.
  Monitor recall against an exact-search sample, not just latency.
- Post-filtering silently returns fewer than k results; pre-filtering and filtered traversal have
  opposite cost curves, which is why the engine should choose using a payload index.
- Quantization is the big memory lever (64× with rescoring restoring full recall), dimension is the
  bigger one; the graph itself is a rounding error at 1536-d.

## Next

→ [Chapter 23: Fusion: Score Distributions, RRF, and Learned Sparse Retrieval](23-fusion-and-sparse.md)

You can now find the nearest vectors quickly and know what that costs. But the nearest vectors are
not always the right ones: a BM25 index disagrees, and the two ranked lists have to be combined
by something better than adding numbers that are not on the same scale.
