# Chapter 23 · Fusion: Score Distributions, RRF, and Learned Sparse Retrieval

> **Goal:** by the end of this chapter you can explain (with the distributions on screen) why a
> BM25 score and a cosine score cannot be added, derive Reciprocal Rank Fusion and say what its
> `k` does, choose between rank fusion and score fusion for a reason, defend BM25's `k1` and `b`
> from first principles, and say where learned sparse retrieval (SPLADE) helps and where it
> actively hurts.
>
> Files: [`code/ch23/retrievers.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch23/retrievers.py), `score_distributions.py`, `fusion_methods.py`,
> `bm25_parameters.py`, `learned_sparse.py`.

[Chapter 8](../part3-retrieval/08-bm25-and-hybrid.md) built BM25, showed the RRF formula, and wired
up hybrid search three ways. It asserted that "dense scores are cosines in ~[0.2, 0.8]; BM25 scores
are unbounded: you cannot add them." This chapter proves it, measures what happens if you try
anyway, and derives the alternatives. The interview question that exposes the gap is the plain one:
*"you said you use hybrid retrieval: how are the two result sets actually combined?"*

---

## 23.1 Two numbers that are not the same kind of number

[`code/ch23/score_distributions.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch23/score_distributions.py) scores all 56 handbook chunks against all 42 answerable golden
questions with both retrievers, and describes the two populations:

```
  dense      min=  -0.060  max=   0.743  mean=  0.207  std= 0.114  range=  0.803
  bm25       min=   0.000  max=  35.721  mean=  2.440  std= 3.239  range= 35.721
```

```
dense cosine distribution:                BM25 score distribution:
    -0.060 | # 24                              0.000 | ######################## 1846
     0.020 | ################## 332            3.572 | ######## 356
     0.101 | ######################## 753      7.144 | ## 79
     0.181 | ####################### 661      10.716 | # 36
     0.261 | ################ 295             14.289 |  19
     0.342 | ######## 148                     17.861 |  7
     0.422 | ##### 91                         21.433 |  6
     0.502 | ## 29                            25.005 |  1
     0.583 | # 16                             28.577 |  1
     0.663 |  3                               32.149 |  1
```

Three structural differences, not just a scale difference:

- **Range.** Cosine is bounded by construction; BM25 is unbounded above, and its maximum depends on
  query length (more query terms = more summands), term rarity and corpus size.
- **Support.** Every chunk has a dense score. **1,846 of 2,352 BM25 scores are exactly zero**: a
  chunk that shares no query term scores nothing at all. Dense scores are a dense matrix; BM25 is a
  sparse one with a huge spike at zero.
- **Floor.** The dense scores cluster around 0.207, not around 0. Embedding space is anisotropic
  (Chapter 22 measured mean pairwise cosine at +0.154 on real embeddings), so "0.2" is the
  *baseline similarity of unrelated text*, not evidence of relevance. A cosine of 0.25 means
  "nothing", which is precisely what makes raw cosine thresholds so treacherous.

### What happens if you add them anyway

The script picks the query with the largest BM25 peak and computes the innocent-looking
`0.5 * bm25 + 0.5 * dense`:

```
query: "How long does a full charge of an Atlas A2 take, and how much runtime does a fast top-up give?"

chunk source                                bm25   dense  0.5b+0.5d
--------------------------------------------------------------------
   54 14-faq                               35.72   0.466      18.09
   13 04-atlas-a2-specification            21.86   0.582      11.22
   28 07-onboarding-guide                  12.17   0.354       6.26

the BM25 term contributes 98% of the winning score; dense 2%.
```

You wrote `0.5 / 0.5`. You got **98% BM25 / 2% dense**. The weights in your config are not the
weights in your ranking; the scale ate them. Worse, the effective weighting *changes per query*:
a query with rare terms produces big BM25 scores and becomes almost pure lexical, while a query of
common words produces small ones and becomes almost pure dense. Your system silently switches
retrieval strategy based on vocabulary.

Note also the substantive disagreement: the two retrievers share only **3 of their top 5** on this
query. Fusion is not a formality over near-identical lists.

## 23.2 Normalisation, and why it is not the fix

The obvious repair is to put both on [0,1] first. Two standard choices, both measured in the script.

**Min-max**: `(v - min) / (max - min)`. It guarantees the top result of every list is exactly 1.0:
including for a query where the retriever found nothing:

```
query                                                 raw max  minmax max
---------------------------------------------------------------------------
How long does a full charge of an Atlas A2 take, a      35.72        1.00
Which database and message broker does Beacon use?       4.98        1.00
```

A retriever that scored 35.7 (an unambiguous lexical hit) and one that scored 4.98 (essentially
noise) both report "1.00, maximum confidence" after normalisation. Fusing those equally is how a
weak retriever poisons a strong one. Min-max is also destroyed by a single outlier: one runaway
score compresses everything else toward zero.

**Z-score**: `(v - mean) / std`. Better behaved on outliers, but on BM25 the mean and standard
deviation are dominated by the mass of zeros, so they encode *how many chunks failed to match*
rather than how good the match is:

```
query                                                 nonzero  z of top1
---------------------------------------------------------------------------
What are the dimensions and weight of the Atlas A2         54       4.28
What are Beacon's RTO and RPO?                             54       3.71
How often do privileged account passwords rotate?           8       5.73
When can I fly business class?                             15       4.18
```

The query with only 8 matching chunks gets the *highest* z-score (5.73), not because its match is
better but because its distribution is emptier. Across queries these numbers are not comparable, so
any fixed fusion weight you tune on one query set mis-weights another.

**The requirement, stated properly:** to combine two retrievers you need a quantity comparable
*across retrievers* and *across queries*. Raw scores are neither. Normalised scores are still not
the second. Ranks are both: a rank of 1 means the same thing in every list and for every query.
That is the entire argument for rank fusion.

## 23.3 Reciprocal Rank Fusion, derived

```
                    Query
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
      BM25 / sparse          dense / ANN
           │                     │
      ranked list A         ranked list B
           │                     │
           └──────────┬──────────┘
                      ▼
          RRF(d) = Σ_lists 1 / (k + rank_L(d))
                      ▼
               fused ranking  →  reranker (Ch 24)  →  top-k → LLM
```

Two design choices are worth understanding separately.

**Why `1/rank`?** Because relevance by rank falls off roughly like a power law: the gap between
rank 1 and rank 2 is worth much more than the gap between rank 21 and rank 22. A reciprocal is the
simplest function with that shape, needs no tuning, and (crucially) is bounded, so no single list
can contribute an unbounded amount. Compare that with scores, where one list's outlier dominates
everything.

**What does `k` do?** It flattens the head of the curve. [`code/ch23/fusion_methods.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch23/fusion_methods.py) prints the
weights:

```
     k        r1       r2       r3       r5      r10      r20      r50   r1/r10
------------------------------------------------------------------------------
     0    1.0000   0.5000   0.3333   0.2000   0.1000   0.0500   0.0200     10.0x
     1    0.5000   0.3333   0.2500   0.1667   0.0909   0.0476   0.0196      5.5x
    10    0.0909   0.0833   0.0769   0.0667   0.0500   0.0333   0.0167      1.8x
    60    0.0164   0.0161   0.0159   0.0154   0.0143   0.0125   0.0091      1.1x
   200    0.0050   0.0050   0.0049   0.0049   0.0048   0.0045   0.0040      1.0x
```

Read the last column. At `k=0`, rank 1 is worth **10×** rank 10: one retriever's top hit decides the
fusion outright. At `k=60` it is worth **1.1×**: position barely matters within the top 10, so what
dominates is *how many lists contain the document at all*. That is the design intent: RRF assumes
both lists are informative and neither is authoritative, so **agreement beats confidence**.

The consequences follow directly:

- A document ranked 1st in one list and absent from the other scores `1/61 ≈ 0.0164`.
- A document ranked 3rd in *both* scores `2/63 ≈ 0.0317`: nearly double. Two moderate votes beat
  one strong one.
- Choose `k` by how much you trust individual rank positions: small `k` if one retriever is
  genuinely authoritative at the top, large `k` if you want consensus. `k=60` is the value from
  Cormack et al. (2009) and is a sane default precisely because it is so flat.

**What RRF throws away** is real and you should be able to say it: score magnitude. A query where
dense is highly confident and BM25 is returning noise is fused *exactly the same* as one where both
are confident, because RRF never learns that one list is garbage. The mitigations are weighted RRF
(a fixed prior per retriever) and the distribution-aware methods below.

**Weighted RRF** simply scales each list: `Σ w_L / (k + rank_L(d))`. Use it when you have measured
that one retriever is better on your corpus: as we are about to.

**Ties.** With `k=60` the weights are so flat that near-ties are common; make tie-breaking
deterministic (our `to_ranks` uses a stable sort) or your results will jitter between identical
runs, which is maddening to debug.

## 23.4 Measuring every fusion method

All methods, 42 answerable golden questions, document-level labels, recall/MRR/nDCG as in
[Chapter 10](../part4-quality/10-evaluation.md):

```
method            recall@5     MRR   nDCG@5
-------------------------------------------
dense only           0.988   1.000    0.980
bm25 only            0.988   0.859    0.881
CombSUM              1.000   0.940    0.956
CombMNZ              1.000   0.940    0.956
z-score sum          1.000   0.940    0.956
DBSF                 1.000   0.988    0.987
RRF k=60             1.000   0.976    0.976
RRF k=0              1.000   0.964    0.966
RRF k=200            1.000   0.976    0.976
wRRF 2:1 dense       1.000   0.988    0.983
wRRF 1:2 bm25        1.000   0.913    0.931
```

Be honest about what this shows, because an interviewer will push on it.

- **Every fusion method raises recall@5 to 1.000** and **every one lowers MRR below dense alone.**
  On this corpus dense retrieval is already perfect at rank 1, so mixing in a weaker list can only
  displace correct top hits. Fusion bought us the last 1.2% of recall and paid for it in precision
  at rank 1.
- **The score-based methods (CombSUM/CombMNZ/z-score, all 0.940) sit below the rank-based ones**,
  which is 23.1 and 23.2 showing up in a metric: they are still distorted by scale.
- **DBSF (0.988) and weighted RRF favouring dense (0.988) do best**, both for the same reason:
they use information that plain RRF discards (score distribution, or a prior that dense is
  stronger here).
- **`k=0` is worse than `k=60`** (0.964 vs 0.976): letting one list's rank-1 dominate hurts when
  that list is the weaker one.

The correct conclusion is not "hybrid is useless": it is that **on a small, clean, well-chunked
corpus with full-sentence questions, dense retrieval is already at the ceiling**, so there is no
headroom for fusion to recover. Chapter 8 measured the same thing. You would reach a different
conclusion on a corpus full of identifiers, which is the next table.

### Where BM25 actually wins

The golden set is paraphrase-style questions, dense retrieval's home ground. The lexical case needs
lexical queries:

```
query                    gold doc                            dense r  bm25 r  RRF r
-----------------------------------------------------------------------------------
BEACON-4187              11-release-notes-beacon-4.2               5       1      1
beaconctl rollout undo   08-runbook-fleet-outage                   2       1      2
ISO 3691-4               04-atlas-a2-specification                 1       1      1
Veldmark                 12-postmortem-2026-03-rotterdam-          1       1      1
firmware 3.8             11-release-notes-beacon-4.2               2       1      1

BM25 ranked the gold chunk higher on 3/5 identifier queries; dense on 0.
```

`BEACON-4187` is the clean case: dense puts the right document at **rank 5**, BM25 at **rank 1**,
and RRF recovers rank 1. An embedding maps that string into a neighbourhood of "Beacon-ish things"
because that is what the subword tokens mean; nothing makes the exact identifier special. BM25 has
no notion of meaning at all, but IDF makes a rare token overwhelming evidence: which is exactly
right here.

**This asymmetry is the whole argument for hybrid retrieval,** and it is the answer to "when does
hybrid help?": when your corpus contains identifiers, SKUs, error codes, command names, version
numbers, or product names (support, legal, ops and code corpora) and your users type them.

### The fusion mechanism, visible

Sorting the golden questions by how much the two retrievers disagree:

```
question                                        dense r  bm25 r  RRF r
----------------------------------------------------------------------
When can I fly business class?                        1       7      2
What are the four data classification levels          1       4      1
How many sick days do I get and when do I ne          1       3      1
```

RRF lands at or near the better list's answer rather than averaging to the middle. `1/61` from the
good list plus `1/67` from the bad one still beats a chunk that neither list ranked highly. Fusion
is insurance against one retriever's blind spot, not a way to beat the better retriever on its own
ground.

## 23.5 DBSF: using the distribution instead of ignoring it

Qdrant offers `Fusion.DBSF` alongside `Fusion.RRF`. Distribution-Based Score Fusion normalises each
list using **mean ± 3 standard deviations** as the bounds, then sums:

```python
def dbsf(score_lists):
    out = 0
    for s in score_lists:
        lo, hi = s.mean() - 3 * s.std(), s.mean() + 3 * s.std()
        out = out + np.clip((s - lo) / (hi - lo + 1e-9), 0, 1)
    return out
```

Using 3σ bounds instead of min/max is what makes it usable: a single outlier no longer rescales the
whole list, because anything beyond 3σ simply clips to 1.0. It keeps the magnitude information RRF
discards, which is why it scored best in our table (MRR 0.988 vs RRF's 0.976).

The trade-off is that it is only as good as the assumption that the score distribution is
meaningful and roughly stable. RRF is the safer default across unknown corpora and query mixes;
DBSF is worth measuring when one retriever is confidently right often enough that throwing away its
confidence costs you. Since Qdrant implements both as a one-word change, this is a decision you can
settle with your eval harness in an afternoon rather than by argument.

## 23.6 BM25's parameters, derived rather than copied

`k1 = 1.5, b = 0.75` appears in every tutorial including Chapter 8. Here is what each one is for.

### `k1`: term-frequency saturation

The TF factor is `f(k1+1)/(f + k1)`, which rises with occurrences `f` and asymptotes at `k1+1`:

```
 occurrences f     k1=0.0    k1=0.5    k1=1.2    k1=2.0   k1=10.0
------------------------------------------------------------------
             1       1.00      1.00      1.00      1.00      1.00
             2       1.00      1.20      1.38      1.50      1.83
             3       1.00      1.29      1.57      1.80      2.54
             5       1.00      1.36      1.77      2.14      3.67
            10       1.00      1.43      1.96      2.50      5.50
            20       1.00      1.46      2.08      2.73      7.33
```

- `k1 = 0` collapses to **boolean retrieval**: one occurrence counts the same as twenty.
- `k1 = 1.2` is nearly saturated by five occurrences: the 2nd occurrence adds ~0.38, the 20th
  adds ~0.01.
- `k1 = 10` is nearly linear over this range, i.e. close to raw TF-IDF.

The reason saturation exists: a chunk that says "battery" twenty times is not twenty times more
about batteries than one that says it twice: it is usually just longer or repetitive. Saturation
is what stops keyword stuffing from winning.

### `b`: length normalisation

The denominator is `k1(1 - b + b·|d|/avgdl)`, so `b` decides how much a document's length counts
against it:

```
   |d|/avgdl      b=0.0     b=0.3    b=0.75     b=1.0
------------------------------------------------------
        0.25       1.50      1.16      0.66      0.38
        1.00       1.50      1.50      1.50      1.50
        4.00       1.50      2.85      4.88      6.00
```

At `b=0`, a 10,000-word page beats a precise two-line answer because it contains every term at
least once. At `b=1` a term in a short chunk counts far more. `b=0.75` is the empirical compromise
from the TREC experiments.

**In RAG this parameter matters less than in web search, and it is worth knowing why:** you control
chunk size. If every chunk is ~800 characters then `|d|/avgdl ≈ 1` for nearly everything and the
`b` term is almost constant. **Your chunking strategy has already done the length normalisation.**

### IDF, and the `+1` that is not cosmetic

```
term               df  idf (lucene)   idf (raw)  note
------------------------------------------------------
the                48         0.161      -1.741  in most chunks
and                46         0.204      -1.488  in most chunks
project             1         3.638       3.611  unique

7 terms have NEGATIVE raw idf: the, in, and, a, to, is
```

The textbook IDF `ln((N - n_t + 0.5)/(n_t + 0.5))` goes **negative** once a term appears in more
than half the documents. A negative weight means a document is *penalised* for containing the word
"the": a chunk that never says it would outrank one that does. Lucene's `ln(1 + x)` form keeps IDF
positive and monotonic. Knowing this distinguishes "I used BM25" from "I understand BM25".

### Does tuning them matter?

```
    k1      b   recall@5      MRR    nDCG@5
--------------------------------------------
   0.0   0.00      0.952    0.856     0.865
   1.2   0.75      0.988    0.847     0.872
   1.2   1.00      0.988    0.895     0.908
   2.0   0.75      0.988    0.871     0.886
   4.0   0.75      1.000    0.883     0.904

best MRR 0.907 at k1=1.5, b=1.0
```

The spread across the whole grid is a few points of MRR, and the best cell (`b=1.0`) differs from
the default mostly because our chunks vary in length more than the `b` discussion above assumes.
The honest finding: **`k1` and `b` are second-order.** Spend the tuning budget on the tokenizer
(Chapter 8: `BEACON-4187` being split into `beacon` + `4187` cost far more than any `k1`), on
chunking, and on whether to run BM25 at all. Sweep them once, keep sane defaults, and be able to
explain what they do.

## 23.7 Learned sparse: SPLADE

BM25 assigns weights from corpus statistics. **Learned sparse** models assign them with a
transformer, while keeping the output a sparse `{term: weight}` vector: so it still lives in an
inverted index, still supports exact term matching, and still gives you interpretable term weights.
The headline capability is **term expansion**: weight on terms that are not in the text.

[`code/ch23/learned_sparse.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch23/learned_sparse.py) runs SPLADE++ and decodes what it produces:

```
query: "How long does a full charge of an Atlas A2 robot take?"
  SPLADE produces 24 weighted terms:
    from the query : long(2.28), atlas(2.04), charge(1.95), a2(1.57), robot(1.47), full(1.28)
    ADDED by model : time(1.46), m2(0.98), charging(0.96), cost(0.62), distance(0.51),
                     vehicle(0.46), hour(0.43), robots(0.36)
```

The added terms are the point: a chunk saying "90 minutes to charge" and a question asking about
"runtime" share no rare token, but both get weight on `time`, `hour`, `charging`, so an inverted
index matches them. This is lexical retrieval with the vocabulary-mismatch problem solved.

Now the honest part:

```
query: "What is the rate limit for the Beacon API?"
    from the query : beacon(3.06), rate(2.21), api(2.16), limit(1.91)
    ADDED by model : price(2.03), lighthouse(1.46), cost(0.95), money(0.61), fee(0.16)
```

It read *rate* as a tariff and *beacon* as a lighthouse. Expansion adds terms that are
**associated**, not synonymous, and those weights pull in genuinely irrelevant chunks.

Measured over the golden set and the identifier probes:

```
retriever               recall@5     MRR   nDCG@5   nnz/doc
------------------------------------------------------------
dense (OpenAI)             0.988   1.000    0.980      1536
BM25 (ours)                0.988   0.859    0.881         -
SPLADE++                   0.952   0.944    0.921       123
BM25 (Qdrant model)        0.964   0.954    0.932        48

query                     dense r  BM25 r  SPLADE r
----------------------------------------------------
BEACON-4187                     5       1         8
beaconctl rollout undo          2       1         4
ISO 3691-4                      1       1         1
Veldmark                        1       1         1
```

**SPLADE beats BM25 on paraphrase questions (MRR 0.944 vs 0.859) and loses badly on identifiers
(rank 8 vs rank 1).** Its vocabulary is BERT wordpiece, so `BEACON-4187` is shredded into subwords
and the rare-token signal BM25 depends on disappears; then the expansion terms dilute what is left.
Learned sparse is **not a strict upgrade** over BM25: it trades exact-match precision for
vocabulary reach.

It also costs more. 123 non-zero terms per document against BM25's 48 means longer posting lists
and queries that touch more of the index, plus a transformer forward pass per document at ingest
and per query at search time. BM25's weights are free to compute.

**When to use it:** vocabulary mismatch is your measured failure mode, and you would rather operate
one inverted index than an inverted index plus a vector database. **When not to:** your corpus is
full of identifiers and codes (keep BM25), or you already run dense retrieval with a reranker: a
cross-encoder fixes most of what SPLADE fixes, with less operational surface (Chapter 24).

To Qdrant all of this is the same shape: BM25, BM42, miniCOIL and SPLADE are all just
`{index: weight}` maps in a `SparseVectorParams` collection. Swapping the model changes one string.

## 23.8 Choosing

| Situation | Retrieval |
|---|---|
| Prototype, prose corpus, paraphrase questions | dense only; add lexical when you *measure* keyword failures |
| Identifiers, SKUs, error codes, commands, versions | **hybrid, non-negotiable**: dense put `BEACON-4187`'s home at rank 5 |
| Corpus jargon differs from user vocabulary | learned sparse (SPLADE), or dense + query rewriting (Chapter 9) |
| Two retrievers, no reliable prior on which is better | **RRF, k=60** |
| One retriever measurably stronger | weighted RRF, or DBSF if the scores are informative |
| Both retrievers in Qdrant, latency-sensitive | server-side `Fusion.RRF` via `prefetch`: one round trip (Chapter 8) |
| A reranker follows the fusion | fuse for **recall**, use generous per-branch limits, let the reranker sort it out |

**When hybrid is not worth it:** when your eval says dense already saturates (our golden set: MRR
1.000 → every fusion method *lowered* it), when your queries are natural-language questions with no
rare tokens, or when the second retriever's operational cost buys a fraction of a point. Measure
before you ship the complexity: that sentence is itself a strong interview answer.

## The interview answer

**"You said you use hybrid retrieval. How are the dense and BM25 results combined?"**

> Not by adding the scores, because they are not comparable. Cosine is bounded in about −1 to 1 and
> every chunk has one; BM25 is unbounded and mostly exactly zero: in my corpus about 78% of BM25
> scores are zero and the max is 35 while the max cosine is 0.74. If you write
> `0.5·bm25 + 0.5·cosine` you get something that is 98% BM25, and the effective weighting changes
> per query depending on how rare the query's terms are. So you fuse ranks instead. Reciprocal Rank
> Fusion scores each document as the sum over lists of `1/(k + rank)`, with `k` conventionally 60.
> A rank means the same thing in every list and for every query, which is exactly the property raw
> or normalised scores lack.

**"What is RRF and why not just normalise the scores?"**

> RRF is `Σ 1/(k + rank)` across the ranked lists. The `k` flattens the head of the curve: at `k=0`
> rank 1 is worth ten times rank 10, at `k=60` it is worth 1.1 times: so agreement between
> retrievers matters more than one retriever's confidence. A document at rank 3 in both lists beats
> a document at rank 1 in only one. You can normalise instead, but min-max makes every list's top
> hit exactly 1.0 even when that retriever found nothing relevant, and z-score on BM25 is dominated
> by how many documents scored zero rather than by relevance: neither is comparable across
> queries. The cost of RRF is that it discards score magnitude, so if one retriever is confidently
> right you can either weight it or use distribution-based fusion; Qdrant implements DBSF for
> exactly that, and it beat RRF on my eval set, 0.988 MRR against 0.976.

**"When does hybrid search actually help, and when doesn't it?"**

> It helps when the corpus has tokens that carry meaning as *strings*: identifiers, error codes,
> SKUs, command names, version numbers. I measured a ticket id where dense retrieval put the right
> document at rank 5 and BM25 put it at rank 1: embeddings map the id into a neighbourhood of
> similar-looking topics, whereas IDF treats a rare token as overwhelming evidence. It does not
> help when dense is already saturated: on my clean 14-document benchmark, dense MRR was 1.000 and
> *every* fusion method lowered it, because mixing in a weaker list can only displace correct top
> hits. So hybrid is a decision you make from your eval table and your corpus vocabulary, not a
> default you add because it sounds thorough.

## Run it

```bash
QDRANT_MODE=memory uv run python code/ch23/score_distributions.py   # ~15 s (first run embeds+caches)
QDRANT_MODE=memory uv run python code/ch23/fusion_methods.py        # ~10 s
QDRANT_MODE=memory uv run python code/ch23/bm25_parameters.py       # ~25 s (24-cell sweep)
QDRANT_MODE=memory uv run python code/ch23/learned_sparse.py        # ~60 s + 530 MB download once
```

Expected: the two distributions and the 98%-BM25 naive sum; the RRF weight table and the fusion
comparison; the `k1`/`b`/IDF tables; SPLADE's expanded terms and the identifier regression. Dense
scores are cached in `data/cache/`, so only the first run costs anything (well under a cent).

## Exercises

1. **Make fusion earn its keep.** The golden set is too easy for hybrid to help. Write ten queries
   the way a hurried user would (fragments, ids, misspellings, no verbs) label their gold
   documents, and re-run `fusion_methods.py` on those. Does RRF now beat dense alone?
2. **Tune `k` properly.** Sweep RRF's `k` over `0, 1, 5, 10, 30, 60, 120, 300` on both query sets
   from exercise 1. Where is the optimum on each, and does the direction match the theory that
   small `k` trusts individual rank positions?
3. **Break the tokenizer, then fix it.** Change `tokenize` in `retrievers.py` to the naive `\w+`
   and re-run the identifier probe. Watch `BEACON-4187` collapse. Then add a rule that keeps
   `[A-Z]+-\d+` patterns intact and measure the recovery.
4. **Weighted RRF from data.** Use half the golden set to estimate which retriever is stronger per
   question `tag`, derive per-tag weights, and evaluate on the other half. Did tag-conditional
   weighting beat a single global weight, or did you just overfit 21 questions?
5. **Post-hoc DBSF.** Implement DBSF with min-max bounds instead of 3σ and compare. How much of
   DBSF's advantage comes from the robust bounds rather than from using scores at all?

## Interview questions

**Q: Why can't you just add a BM25 score to a cosine similarity?**
Because they are different kinds of number. Cosine is bounded and every document has one; BM25 is
unbounded above, depends on query length and term rarity, and is exactly zero for any document not
sharing a query term: in my corpus 78% of the scores were zero while the maximum was 35 against a
maximum cosine of 0.74. Adding them with equal weights gives you a ranking that is almost entirely
BM25, and the effective weight shifts per query. You either fuse ranks, or normalise with a method
whose failure modes you have measured.

**Q: Write RRF on the whiteboard and explain `k`.**
`RRF(d) = Σ_L 1/(k + rank_L(d))`, summed over the ranked lists a document appears in, with `k = 60`
by convention. `k` controls how peaked the weighting is: at `k = 0` rank 1 is worth ten times rank
10, so one list's top hit dominates; at `k = 60` it is worth about 1.1 times, so what matters is
whether both lists ranked the document at all. Large `k` favours consensus, small `k` favours a
retriever you trust to be authoritative at the top.

**Q: What does RRF lose compared with score fusion?**
Magnitude. If dense retrieval is confidently right and BM25 is returning noise, RRF weights their
opinions identically because it only sees positions. Weighted RRF patches this with a fixed prior,
and distribution-based fusion (Qdrant's DBSF) normalises each list by mean ± 3σ and sums, keeping
the confidence information while staying robust to outliers. On my evaluation DBSF scored 0.988 MRR
against RRF's 0.976: worth measuring, not worth assuming.

**Q: What do `k1` and `b` do in BM25?**
`k1` controls term-frequency saturation: the TF factor asymptotes at `k1 + 1`, so at `k1 = 0` BM25
becomes boolean (one occurrence equals twenty) and at large `k1` it approaches raw TF-IDF. Typical
values of 1.2–2.0 mean the second occurrence of a term adds a lot and the twentieth adds almost
nothing, which is what stops keyword stuffing. `b` controls length normalisation from 0 (ignore
length) to 1 (fully normalise), with 0.75 as the empirical default. In RAG `b` matters less than in
web search because your chunker has already made documents roughly equal length.

**Q: Why does BM25's IDF have a `+1` inside the logarithm?**
Without it, `ln((N − n + 0.5)/(n + 0.5))` goes negative for any term appearing in more than half the
documents, so containing a common word would *reduce* a document's score. In my 56-chunk corpus
seven terms had negative raw IDF, including "the" at −1.74. The Lucene form `ln(1 + x)` keeps IDF
positive and monotonically decreasing in document frequency.

**Q: Dense retrieval missed a ticket id that BM25 found at rank 1. Why?**
Embeddings represent meaning, and an identifier's meaning is mostly its literal string. The
tokenizer splits `BEACON-4187` into subwords, and the resulting vector lands in the neighbourhood
of "Beacon-related things" rather than pointing at the one chunk that names it: in my measurement
dense put it at rank 5. BM25 has no semantics at all, but IDF gives a token appearing in one
document an enormous weight, which is exactly the right behaviour for identifiers. It is the
canonical argument for running both.

**Q: What is SPLADE, and is it a better BM25?**
It is a learned sparse retriever: a transformer predicts a weight for every vocabulary term,
including terms not present in the text, and the output stays a sparse vector that lives in an
inverted index. The expansion solves vocabulary mismatch: it added "time", "hour" and "charging"
to a query about a full charge. But it is not a strict upgrade: on identifier queries it did worse
than plain BM25 (rank 8 versus rank 1) because wordpiece tokenisation destroys rare literal tokens,
and its expansion sometimes adds mere associations: it expanded "Beacon API rate limit" with
"price", "money" and "lighthouse". It also produced 123 non-zero terms per document versus BM25's
48, and needs a model at both index and query time.

**Q: Your hybrid search performs worse than dense alone. What do you do?**
First accept the measurement rather than tuning blindly: it usually means dense is already near the
ceiling on that query mix, so the second list can only displace correct top hits: on my benchmark
dense MRR was 1.000 and every fusion method reduced it. Then check whether your query mix is
representative; hybrid earns its keep on fragments and identifiers, which polished benchmark
questions rarely contain. If it still loses, either weight the dense list up, restrict fusion to
queries that look lexical (containing digits, hyphenated tokens, ALL-CAPS), or drop the second
retriever and save the latency.

**Q: How would you fuse three or more retrievers?**
RRF generalises directly: sum `1/(k + rank)` over all lists: and that is its practical advantage
over pairwise score blending, which needs a weight matrix. With more lists, `k` becomes more
important because consensus effects strengthen; and you should check correlation between the
retrievers first, because two near-identical dense models will out-vote a genuinely complementary
lexical one purely by count. Weight by measured per-retriever quality, or drop redundant ones.

**Q: Where does fusion sit relative to reranking?**
Fusion is a cheap recall stage: it merges candidate lists without reading the documents. Reranking
is an expensive precision stage that reads query and document together. So fuse generously:
wide per-branch limits, `k = 60`, optimise for recall at 50 or 100: and let the cross-encoder
produce the final ordering. Tuning fusion for perfect rank-1 accuracy is wasted effort when a
reranker will re-sort the list anyway (Chapter 24).

## Key takeaways

- BM25 and cosine scores are structurally different (unbounded and mostly-zero versus bounded and
  dense); adding them gives a ranking dominated by scale: 98% BM25 from nominal 50/50 weights.
- Normalisation does not fix comparability across queries: min-max makes an empty result look
  maximally confident, z-score on BM25 measures how many documents failed to match.
- RRF fuses ranks because a rank is comparable across lists and queries. `k` flattens the head:
  `k = 60` makes agreement matter more than one retriever's confidence.
- Fusion is insurance against blind spots, not a free upgrade: on a corpus where dense already
  scores MRR 1.000, every fusion method made rank-1 precision worse while lifting recall.
- Hybrid is non-negotiable for identifiers and codes (dense rank 5 vs BM25 rank 1 on a ticket id);
  learned sparse trades exact-match precision for vocabulary reach and is not a strict upgrade.

## Next

→ [Chapter 24: Rerankers: Bi-encoders, Cross-encoders and Late Interaction](24-rerankers.md)

Fusion produced a merged candidate list without ever reading a document against the query. The next
stage does exactly that: and it is the answer to "if Qdrant already returns similarity scores, why
rerank at all?"
