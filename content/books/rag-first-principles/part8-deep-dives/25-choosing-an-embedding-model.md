# Chapter 25 · Choosing and Evaluating an Embedding Model

> **Goal:** by the end of this chapter you can answer "how would you choose an embedding model?"
> with a repeatable procedure rather than a preference; run a six-model bake-off on your own
> corpus in one command; name the three mistakes that silently invalidate such a comparison: and
> recognise them, because this chapter's own bake-off hit one and reported a good model at *half*
> its real quality; and decide dimensions and quantization from a measured curve instead of a
> default.
>
> Files: [`code/ch25/embedding_bakeoff.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch25/embedding_bakeoff.py), `prefix_and_length.py`, `dimension_curve.py`,
> `_embedders.py`. Chapter 2 explained what an embedding *is*; this chapter is how you pick one.

---

## 25.1 The framework

The interview question is "how do you choose an embedding model?" The weak answer lists
properties: cost, dimensions, multimodality, whether it is open source. Those are *inputs*. The
strong answer is a procedure that ends in a number:

```
  fix:    corpus + chunking + golden set + search method
  vary:   ONLY the embedding model
  report: recall@5, recall@10, MRR, nDCG@10          quality
          latency to encode, latency to query        speed
          bytes per vector x corpus size             memory
          $ per 1M tokens x corpus tokens            cost
          licence, hosting, max seq length, languages constraints
  decide: against the constraint that actually binds you
```

Three details that make it a real experiment rather than a demo:

**Search must be exact.** `embedding_bakeoff.py` uses brute-force cosine, not HNSW. If you
benchmark models through an ANN index you are measuring model quality *plus* that index's recall
error, and the two are confounded. Chapter 22 measures ANN error deliberately and separately.

**Queries must look like production.** The golden questions in this repo were written *from* the
documents, so they reuse document vocabulary and every model looks good. The bake-off therefore
runs on the "vague" rewrites introduced in Chapter 24: the same information needs phrased the
way somebody would actually type them. This is not a detail; §25.3 shows it changes the ranking.

**Everything is cached.** `_embedders.py` stores every matrix under `data/cache/ch25_embeddings/`
keyed by model and text hash. Without a cache you re-pay the API on each run and quietly shrink
the corpus until the experiment stops meaning anything.

## 25.2 The bake-off

Six candidates: three OpenAI configurations and three self-hosted models, over 997 chunks
(handbook + generated documents + novel distractors) and 42 realistic queries.

```
model                  dim  recall@5  recall@10  nDCG@10    MRR   MB/1M  corpus $  s/1k docs
openai-3-large        3072     0.988      0.988    0.973  0.988   12288    0.0186       40.8
openai-3-small        1536     0.976      0.988    0.972  0.988    6144    0.0029       21.8
openai-3-large@512     512     0.988      0.988    0.971  0.988    2048    0.0186       38.8
arctic-embed-s         384     0.952      0.976    0.954  0.976    1536    0.0000       13.8
bge-small-en-v1.5      384     0.964      0.976    0.918  0.925    1536    0.0000       19.2
all-MiniLM-L6-v2       384     0.940      0.940    0.875  0.901    1536    0.0000        4.0
```

What a hiring manager would want you to read out of that table:

- **The expensive model is not meaningfully better than the cheap one.** `3-large` beats
  `3-small` by 0.001 nDCG and costs **6.4× more** ($0.0186 vs $0.0029 to embed this corpus) and
  **twice the memory**. On this corpus, paying for `3-large` buys nothing.
- **Matryoshka truncation is nearly free.** `3-large@512` keeps 0.971 of 0.973 at **one sixth**
  the memory of full `3-large`. §25.5 pushes that further.
- **A 384-dimension self-hosted model is within 0.019 nDCG of the best API model**, at zero
  marginal cost and 8× less memory. For most systems that is the interesting row.
- **`s/1k docs` is not a model-speed column for the API rows.** Those are wall-clock seconds
  including network and batching (Chapter 13 §13.5 measured ~512 ms for a single query
  embedding). Compare local models to each other, and read API rows as "what bulk ingest will
  actually cost you in time".

And the entry that is *not* in the table: this bake-off originally ranked `arctic-embed-s` last,
at **0.534 nDCG**, and I nearly wrote it off as a weak model. It is 0.954. That mistake is the
next section.

## 25.3 Three things that silently invalidate a bake-off

### The instruction prefix: worth 0.42 nDCG here

Many open models are trained **asymmetrically**: the query is encoded with an instruction prefix,
the passage without. `snowflake-arctic-embed`'s card says prefixes are *necessary*; `bge-v1.5`
says *not so necessary*. Both statements turn out to be exactly true:

```
1. PREFIX ABLATION (realistic queries, 997-chunk corpus)
   model                  no prefix  with prefix    delta  card says
   arctic-embed-s             0.534        0.954   +0.419  REQUIRED
   bge-small-en-v1.5          0.910        0.918   +0.008  'not so necessary'
   all-MiniLM-L6-v2           0.875        0.875   +0.000  none (symmetric)
```

**One string of boilerplate: `"Represent this sentence for searching relevant passages: "`: is
worth 0.42 nDCG.** Without it, arctic is the worst model in the line-up. With it, arctic is
competitive with a 3072-dimension paid API model while being free and self-hosted.

The trap is that the library *appears* to handle this. fastembed exposes `query_embed()` and
`passage_embed()` next to `embed()`. In fastembed 0.8.0 they are no-ops for these models: not
"approximately equal", byte-identical:

```
0. DOES THE LIBRARY APPLY THE PREFIX FOR YOU?
   embed() == query_embed()   : True
   embed() == passage_embed() : True
```

So the harness applies prefixes by hand (`_embedders.py`, `QUERY_INSTRUCTION`). The general
lesson is worth more than the specific bug: **an API named after the thing you want is not
evidence that it does the thing you want.** Three lines of `np.allclose` would have saved the
wrong conclusion, and "we picked the wrong embedding model because a helper was a no-op" is a
genuinely expensive failure.

### Maximum sequence length: truncation you cannot see

Every encoder has a window. Text past it is not down-weighted, it is **absent from the vector**,
at index time, permanently. And the documented limit is not always the real one:

```
   arctic-embed-s       documented  512 | measured  512
   bge-small-en-v1.5    documented  512 | measured  512
   all-MiniLM-L6-v2     documented  256 | measured  128   <-- card and reality disagree

   800-char chunks (book default): 56 chunks
      arctic-embed-s       window  512 | median seen  170 | truncated   0/56 | nDCG@10 0.970
      bge-small-en-v1.5    window  512 | median seen  170 | truncated   0/56 | nDCG@10 0.972
      all-MiniLM-L6-v2     window  128 | median seen  128 | truncated  40/56 | nDCG@10 0.931

   2400-char chunks: 19 chunks
      arctic-embed-s       window  512 | median seen  512 | truncated  13/19 | nDCG@10 0.943
      bge-small-en-v1.5    window  512 | median seen  512 | truncated  13/19 | nDCG@10 0.896
      all-MiniLM-L6-v2     window  128 | median seen  128 | truncated  17/19 | nDCG@10 0.811
```

`all-MiniLM-L6-v2` truncates at **128 tokens**, half its documented 256, so **40 of the book's 56
default chunks are already being cut**. Its lower score is less "a weaker model" than "a model
that never saw most of your corpus". At 2,400-character chunks even the 512-window models lose
13 of 19 chunks' tails and `bge` drops from 0.972 to 0.896.

Two practical consequences. First, measure the clamp empirically: encode something enormous and
see where `token_count` stops rising. Second, **chunk size and embedding model are one decision,
not two**: a chunking experiment (Chapter 5) run under a model that truncates is measuring the
wrong thing.

### Questions written from your documents

The third invalidator is the one from Chapter 10 §10.2 and Chapter 24 §24.4: if your evaluation
questions were written by reading the chunks, they inherit the chunks' vocabulary and every model
scores near the ceiling. Run `embedding_bakeoff.py --easy` to see the differences compress. You
will pick a model that is good at a task nobody performs.

## 25.4 MTEB and public leaderboards

Use them as a **shortlist generator, never as an answer**. What a leaderboard tells you is the
average over dozens of tasks (classification, clustering, STS, retrieval) on public datasets,
which is not your task on your corpus. Specifically:

- **Task mix.** A model can top the overall table on the strength of classification and
  clustering while being mid-table on retrieval, which is the only column you care about.
- **Contamination.** Public benchmarks leak into training sets over time. Rankings drift for
  reasons unrelated to your data.
- **Size and licence.** A 7B embedding model at the top of the board may be untenable for your
  latency, memory or licence constraints. Filter the board by what you could actually deploy
  before reading the scores.
- **Recency.** The leaders change every few months; any specific ranking written in a book is
  stale on arrival.

The landscape as of writing: OpenAI `text-embedding-3-small`/`-large`, Voyage, Cohere `embed v4`,
Google Gemini embeddings on the API side; BGE, E5, Nomic, Arctic and Qwen3-Embedding among open
models. Vendor comparisons are marketing until you reproduce them: treat published deltas as
hypotheses for your own bake-off, which takes about a minute per model once the harness exists.
That harness is the deliverable, not the shortlist.

## 25.5 Dimensions, Matryoshka and quantization

Matryoshka Representation Learning trains a model so its first *n* dimensions are themselves a
valid embedding. For `text-embedding-3-*` this means the dimension count is a dial you turn after
measuring: and you can explore the entire curve from one full-precision encode, because
truncating locally and renormalising is what the API's `dimensions=` parameter does:

```
1. LOCAL TRUNCATION vs THE API's dimensions= PARAMETER (256 dims, 20 chunks)
   cosine(api_256, locally_truncated_256): min 0.999607  mean 0.999970
```

Then the curve:

```
     dims  recall@5  nDCG@10  vs full   GB/1M  GB/100M
     3072     0.988    0.973   +0.000    12.3     1229
     2048     0.988    0.969   -0.004     8.2      819
     1024     0.988    0.974   +0.002     4.1      410
      512     0.988    0.971   -0.002     2.0      205
      256     0.964    0.931   -0.042     1.0      102
      128     0.929    0.870   -0.103     0.5       51
       64     0.833    0.795   -0.177     0.3       26
```

**Flat to 512, then a cliff.** You keep full quality at one sixth of the memory (1,229 GB → 205 GB
at 100M vectors) and then lose 0.042 in the next halving. Note also that 1024 dims scores
*fractionally above* 3072: at this corpus size those differences are noise, which is itself the
lesson: do not chase 0.002.

Now the comparison that decides large deployments. Given a fixed byte budget, do you spend it on
**dimensions** or on **precision**?

```
   configuration                      bytes/vec  recall@5  nDCG@10
   3072-d float32 (reference)             12288     0.988    0.973
   3072-d int8                             3072     0.988    0.973
   768-d float32                           3072     0.988    0.970
   1024-d int8                             1024     0.988    0.974
   256-d float32                           1024     0.964    0.931
```

Read the last two rows: **the same 1,024 bytes buys 0.974 nDCG as 1024-d int8, or 0.931 as 256-d
float32.** Same memory, +0.043 quality. And 3072-d int8 matches full float32 exactly at a quarter
of the bytes.

**Spend your bytes on dimensions, not on precision.** Quantize first, truncate second. The caveat
is honest: this is exact search, and on a real HNSW index quantization also perturbs graph
traversal: which is precisely why Qdrant rescores candidates with full vectors (Chapter 18
§18.2). Measure the two together before committing.

## 25.6 Domain fit, and the ladder of adaptation

General-purpose models fail in predictable places: internal identifiers (`BEACON-4187`,
`beaconctl`), product codes, clinical or legal shorthand, non-English text, and very long
single-topic documents. Chapter 8 measured this directly: BM25 beat dense retrieval on exactly
the identifier-style questions, because a lexical match on a rare token is information a
semantic model smooths away.

When quality is short on your domain, work **up this ladder** and stop as soon as the numbers are
good enough. It is ordered by effort, and the cheap rungs are where most of the wins are:

1. **Fix the representation.** Chunk on structure; add contextual prefixes. Chapter 24 §24.5 moved
   a chunk from rank 15 to rank 2 this way, which no model swap achieved.
2. **Check the prefix and the window** (§25.3). Free, and worth 0.42 nDCG in this very chapter.
3. **Add hybrid retrieval.** BM25 alongside dense covers the exact-token cases by construction
   (Chapter 23).
4. **Try a different general model.** One command with the harness above.
5. **Use a domain model** if one exists for your field (biomedical, legal, code).
6. **Fine-tune** on `(query, positive)` pairs mined from your logs, with hard negatives from your
   own first stage: the same recipe as Chapter 24 §24.6, applied to the bi-encoder. Budget a few
   thousand pairs minimum and expect single-digit point gains, plus the ongoing cost of
   retraining as the corpus drifts.

A fine-tuning dataset is unglamorous: `{"query": "...", "positive": "chunk text", "negatives":
["...", "..."]}`, a few thousand rows, harvested from clicked results, thumbs-up answers and your
golden set. The honest expectation is that steps 1–3 deliver more than step 6 for most teams.

This repo cannot demonstrate step 6 credibly: 14 documents and 42 questions is far too small to
fine-tune on without fooling yourself, and a chapter that faked it would teach the wrong lesson.
The measurable claim here is the ladder's ordering, which §25.3 and Chapter 24 §24.5 both support.

## 25.7 Migration: you cannot mix embedding spaces

Two models produce vectors in unrelated spaces. A cosine between them is noise. So changing the
embedding model means **re-embedding the entire corpus**: there is no incremental path, and the
migration is an operational project rather than a config change:

1. Build a **new collection** with the new model (blue/green). Never write both models' vectors
   into one collection.
2. **Backfill** it from your source of truth, not from the old index: you need the original text,
   which is why Chapter 7's manifest exists.
3. **Dual-write** new documents to both collections during the backfill so the new one does not
   fall behind.
4. **Re-run the golden set against the new collection** and compare against the old one's stored
   results. This is the gate; a model that wins on a leaderboard can lose on your corpus.
5. **Switch the alias** atomically (Chapter 18 §18.3 has the code:
   `client.update_collection_aliases`), keeping the old collection until you are confident.
6. **Record the model name and dimension in the collection's metadata.** Chapter 2's Rule 1 exists
   because mixed spaces fail silently: retrieval returns plausible nonsense rather than an error.

Budget the re-embedding cost from §25.2's table: at 100M chunks of ~200 tokens, `3-small` is
about $400 and `3-large` about $2,600, plus the ingest time from Chapter 18's arithmetic.

## 25.8 The decision table

| If this binds you | Choose | Because |
|---|---|---|
| Nothing much; small corpus | `text-embedding-3-small` | 0.972 nDCG here, $0.02/1M, no ops burden |
| Memory / index size at scale | `3-large@512` or int8 | full quality at 1/6 the bytes (§25.5) |
| Cost at very large scale | self-hosted arctic/BGE | 0.954 here, zero marginal cost, 384-d |
| Data cannot leave your network | self-hosted arctic/BGE | no API dependency at all |
| Query latency | self-hosted small model | no network round trip (~512 ms saved, ch 13) |
| Identifiers, codes, rare tokens | any dense + **BM25 hybrid** | lexical matching is a different capability (ch 8, 23) |
| Long documents | a 512+ window model, and re-chunk | truncation is invisible and permanent (§25.3) |
| Multilingual corpus | a multilingual model (E5, Cohere, Gemini) | English-only models fail silently on other scripts |
| Nothing above is decided yet | run the bake-off | one command, one minute per model |

## The interview answer

**"How would you choose an embedding model?"**

> I run a bake-off rather than pick from a leaderboard. I fix the corpus, the chunking and a
> golden set of questions phrased the way users actually ask, use exact search so ANN error
> doesn't contaminate the comparison, and vary only the model. I report recall@5, recall@10,
> nDCG@10 and MRR for quality, plus encode latency, bytes per vector at my corpus size, and cost
> per million tokens. Then I decide against whichever constraint actually binds. On my corpus
> that table showed `text-embedding-3-large` beating `3-small` by 0.001 nDCG for 6× the cost and
> twice the memory: so the expensive model was the wrong answer, and I'd never have known from
> a leaderboard.

**"How do you know model A is better than model B for your data?"**

> Only by measuring on my data, and I'd flag three things that make such a comparison lie. First,
> instruction prefixes: asymmetric models need a query prefix, and in my bake-off forgetting it
> cost arctic-embed 0.42 nDCG: it looked like the worst model and was actually competitive with
> a 3072-dimension API model. Worse, the library's `query_embed()` silently didn't apply it; I
> caught it with `np.allclose`. Second, sequence length: `all-MiniLM-L6-v2` truncates at 128
> tokens, not the documented 256, so 40 of my 56 chunks were being cut: that's not a weaker
> model, it's a model that never saw the corpus. Third, questions written from the documents make
> every model look equal. Also, differences under ~0.01 on 40 questions are noise, so I'd bootstrap
> a confidence interval before believing a small win.

**"When would you fine-tune an embedding model?"**

> Last, not first. There's a ladder: fix chunking and add contextual prefixes, check prefixes and
> truncation, add BM25 hybrid for identifiers and rare tokens, try other general models, try a
> domain model: and only then fine-tune on query-positive pairs with hard negatives mined from
> my own first stage. I'd want a few thousand real query-document pairs from logs, and I'd expect
> single-digit point gains plus an ongoing retraining commitment. In this corpus a one-sentence
> contextual prefix moved a chunk from rank 15 to rank 2, which no model swap achieved: the
> cheap rungs usually win.

## Run it

```bash
QDRANT_MODE=memory uv run python code/ch25/embedding_bakeoff.py     # ~3 min first run, ~$0.03
QDRANT_MODE=memory uv run python code/ch25/prefix_and_length.py     # ~2 min, free after cache
QDRANT_MODE=memory uv run python code/ch25/dimension_curve.py       # ~1 min, ~$0.01
```

The first run downloads three small models (~300 MB) to `~/.cache/fastembed` and writes embedding
matrices to `data/cache/ch25_embeddings/`; every later run reads the cache, so iterating on the
analysis is free. Useful variants:

```bash
uv run python code/ch25/embedding_bakeoff.py --models openai-3-small,arctic-embed-s
uv run python code/ch25/embedding_bakeoff.py --easy      # original wording: watch the gaps close
```

Expected: `3-large` and `3-small` within 0.001 nDCG; arctic at 0.954 *with* its prefix and 0.534
without; MiniLM's measured window of 128 against a documented 256; the dimension curve flat to
512 then falling; and 1024-d int8 beating 256-d float32 at equal bytes.

## Exercises

1. Run the bake-off with `--easy` and put the two tables side by side. How much does the gap
   between the best and worst model shrink? Write one sentence explaining what that says about
   benchmarks built from documents.
2. Add `intfloat/e5-small-v2` to `default_models()` with the correct `query_prefix="query: "` and
   `doc_prefix="passage: "`. Confirm from the ablation that *both* prefixes matter for E5, unlike
   the query-only models already there.
3. The bake-off reports point estimates on 42 questions. Add a bootstrap confidence interval
   (resample questions with replacement 1,000 times) and re-read the table: which of the
   differences survive?
4. Re-run `dimension_curve.py` against `text-embedding-3-small` instead of `-large`. Does its
   cliff arrive at the same dimension, or earlier? Explain why that would be expected.
5. Combine the two dials: measure 512-d int8 and 256-d int8 and add them to the equal-byte table.
   At what point does quantization stop being free?
6. Write the migration script sketched in §25.7 for this repo: build a second collection with a
   different model, run the golden set against both, and print a per-question diff of which
   questions each one wins. That diff is what you would show a reviewer before switching.

## Interview questions

**Q: How do you choose an embedding model?**
Bake-off on my own data: fix corpus, chunking, golden set and use exact search; vary only the
model; report recall@k, nDCG@10, MRR, encode latency, bytes per vector at my scale, and cost per
1M tokens; then decide against the binding constraint. On my corpus the premium model beat the
cheap one by 0.001 nDCG at 6× the cost: a leaderboard would have sent me the other way.

**Q: What's the most common mistake in an embedding comparison?**
Forgetting instruction prefixes on asymmetric models. In my run it cost arctic-embed 0.42 nDCG:
the difference between "worst model tested" and "competitive with a 3072-d API model". Compounding
it, the library helper that is supposed to add the prefix was a no-op, which I only found by
comparing `embed()` and `query_embed()` output with `np.allclose`. Verify the library does what
its function name claims.

**Q: What is Matryoshka representation learning and why does it matter operationally?**
The model is trained so that the first n dimensions are themselves a valid embedding, so you can
truncate and renormalise instead of re-encoding. It turns dimensionality into a tunable dial: I
measured full quality down to 512 dims (1,229 GB → 205 GB at 100M vectors) with a cliff at 256.
Truncating locally matches the API's `dimensions=` parameter to within float noise: cosine
above 0.999 on every chunk I checked: so one encode buys the whole curve.

**Q: Dimensions or quantization: where do you spend a fixed memory budget?**
Quantization first. At an equal 1,024 bytes per vector I measured 1024-d int8 at 0.974 nDCG
against 256-d float32 at 0.931: same bytes, 0.043 better. And 3072-d int8 matched full float32
exactly at a quarter of the size. Caveat: that's exact search; on HNSW, quantization also affects
traversal, so vector stores rescore top candidates with full-precision vectors.

**Q: Why can't you mix two embedding models in one collection?**
Different models produce vectors in unrelated geometric spaces, so a cosine between them is
meaningless: and it fails *silently*, returning plausible-looking but wrong neighbours rather
than an error. Changing models means re-embedding everything: blue/green collections, backfill
from the source text, dual-write during migration, re-run the golden set as the gate, then switch
the alias. Record the model name and dimension in the collection metadata.

**Q: How does max sequence length affect retrieval quality?**
Text beyond the window is dropped at index time, so it is absent from the vector and no
query-side technique recovers it. I measured `all-MiniLM-L6-v2` clamping at 128 tokens against a
documented 256, which truncated 40 of my 56 default chunks; at 2,400-character chunks even
512-window models lost 13 of 19 chunks' tails and nDCG fell from 0.972 to 0.896. Chunk size and
model choice have to be decided together.

**Q: How should you use MTEB?**
As a shortlist, filtered to models you could actually deploy, and read the retrieval column rather
than the average: a model can top the overall board on classification and clustering. Then
reproduce it on your corpus, because public benchmarks suffer contamination over time and your
task mix is not theirs. The bake-off harness is the deliverable; the leaderboard just picks which
three models to run through it.

**Q: Your new model scores 0.01 higher on 40 questions. Ship it?**
No. That is within noise for 40 questions; one question is 2.5 points. I'd bootstrap a confidence
interval or run a paired test first, check whether the win is concentrated in one question type
using the tags, and confirm it doesn't regress latency, memory or cost. And if it's a different
model family, remember shipping it means re-embedding the whole corpus: the bar should be
proportional to that cost.

**Q: When is a self-hosted model the right call?**
When cost at scale, query latency, or data residency binds. My self-hosted 384-d model scored
0.954 against 0.973 for the best API model, at zero marginal cost, 8× less memory, and no network
round trip: Chapter 13 measured query embedding at ~512 ms p50 against the API. The trade is ops
burden: you now own model hosting, versioning and capacity.

**Q: The general model is bad on our jargon. What do you try, in order?**
Fix representation first (structural chunking, contextual prefixes), then verify prefixes and
truncation, then add BM25 hybrid: lexical matching handles exact identifiers by construction,
which Chapter 8 measured directly. Then try other general models, then a domain-specific model,
and only then fine-tune with hard negatives mined from my own retriever. Fine-tuning needs a few
thousand real pairs and buys single-digit points; the earlier rungs are usually bigger and
cheaper.

**Q: How do you estimate the cost of switching embedding models?**
Corpus tokens × price per 1M for the re-embed, plus ingest wall-clock from the measured
throughput, plus the new index's memory. For 100M chunks at ~200 tokens that's roughly $400 with
`3-small` or $2,600 with `3-large`, and the memory column changes by the dimension ratio. That
arithmetic is usually what kills a "let's just use the biggest model" proposal.

## Key takeaways

- Choosing an embedding model is a measurement procedure, not a preference: fix everything, vary
  the model, report quality *and* latency *and* memory *and* cost, decide on the binding
  constraint. The premium model lost on this corpus.
- Three things silently invalidate the comparison: missing instruction prefixes (worth 0.42 nDCG
  here, and the library's helper was a no-op), truncation at a window that may not match the
  documentation (128 vs a documented 256), and questions written from your own documents.
- Matryoshka truncation is close to free down to a cliff: full quality at 512 of 3072 dimensions
  here: and local truncation is equivalent to the API's `dimensions=` parameter.
- At equal bytes, more dimensions in int8 beats fewer dimensions in float32 (0.974 vs 0.931).
  Quantize first, truncate second, and rescore on a real index.
- You cannot mix embedding spaces: a model change is a full re-embed behind a blue/green
  collection and an alias switch, gated by re-running the golden set.

## Next

→ [Chapter 26: Generation Evaluation and Hallucination Forensics](26-hallucination-forensics.md)

Chapters 24 and 25 make retrieval find the right evidence. Chapter 26 is about the half that
happens afterwards: proving the model actually used it, and diagnosing the failure when it
did not.
