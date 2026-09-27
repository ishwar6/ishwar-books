# Chapter 27 · Retrieval Debugging: the Rank-17 Playbook

> **Goal:** somebody shows you a query where the right document comes back at rank 17.
> By the end of this chapter you diagnose before you treat: you can name twelve things
> that put a chunk at rank 17, you have a script that tests all twelve in thirty seconds
> and tells you which one fired, and you can defend the fix you chose over the six you
> rejected. You also stop guessing at chunk size and start running a factorial experiment
> whose output is a Pareto frontier, not an opinion.
>
> Files: [`code/ch27/diagnose.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch27/diagnose.py), [`code/ch27/chunk_experiment.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch27/chunk_experiment.py), [`code/ch27/query_rewriting_lab.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch27/query_rewriting_lab.py)

---

## 27.1 Why "raise k" is the wrong first move

The instinct is understandable. The document is at rank 17, k is 5, so make k 20 and the
bug is gone. Here is what you actually bought:

**You pay on every query, forever.** k is not a per-question setting. Going from k=5 to
k=20 at 140 tokens per chunk adds ~2,100 input tokens to *every* request. Chapter 13
measured the shape of that bill: cost is input-token dominated, so this is roughly a 3×
increase in the cost of your product to fix one ticket.

**You lower precision on every other query.** Chapter 10's eval table measured exactly
this trade: k=3 → k=6 raised document recall from 0.94 to 0.98 but dropped precision from
0.74 to 0.54 and (the part people miss) dropped the *correct answer rate* from 0.83 to
0.79. More context made answers worse, because the model had more plausible-but-wrong
material to blend.

**You may not even fix it.** Rank 17 in a candidate list is not the same as rank 17 in the
context. If the failure is that the answer is split across two chunks, or that a filter
excludes the document, or that the chunk is a heading with no content, k=20 changes
nothing.

**And you have destroyed the evidence.** The ticket is closed, the root cause is still in
the index, and it will produce different symptoms next month.

There is one case where raising k *is* the right answer: when you raise it as the
**candidate depth of a first stage that feeds a re-ranker**, so the LLM still reads five
chunks. That is not "k=20", that is "retrieve 20, re-rank to 5": a different system, and
Chapter 24 covers what it costs.

So: diagnose first. Every failure lands in one of three buckets.

```
                          "the right chunk is at rank 17"
                                        │
         ┌──────────────────────────────┼──────────────────────────────┐
         ▼                              ▼                              ▼
   QUERY problem                 INDEX problem            REPRESENTATION problem
   the question does not         the chunk is not         the chunk cannot be
   look like the answer          reachable or not         matched: split answer,
   (vocabulary, phrasing,        ranked (ANN recall,      heading-only, boilerplate,
   multi-hop, missing            filters, duplicates)     table without header
   context from earlier turns)
         │                              │                              │
   rewrite / HyDE /              hybrid, ef_search,       re-chunk, parent/child,
   decompose / route             dedup, filter audit      contextual prefix
   ── cheap, reversible ──►   ── medium ──►   ── expensive, re-embeds the corpus ──►
```

Notice the cost gradient along the bottom. **Fixes get more expensive left to right**, and
the diagnosis tells you how far right you are forced to go.

## 27.2 The twelve diagnostics

[`code/ch27/diagnose.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch27/diagnose.py) takes a question and the document you expected, and runs all
twelve. It is deliberately generic: the same checks run for any question, and every
verdict is a measurement it just made rather than a rule of thumb.

```bash
uv run python code/ch27/diagnose.py --golden q17
uv run python code/ch27/diagnose.py --question "how fast is the robot" \
      --expect 04-atlas-a2-specification --keywords 2.0 speed
```

| # | Diagnostic | What a FAIL means | Bucket |
|---|---|---|---|
| 1 | document is in the index | ingestion dropped it; nothing downstream can help | index |
| 2 | answer lives in one chunk | the answer straddles a chunk boundary | representation |
| 3 | rank and score gap | how far off it is, and whether the top hit is close or miles ahead | - |
| 4 | query/chunk vocabulary overlap | the chunk never uses the asker's words | query |
| 5 | lexical/hybrid ranks it higher | the terms *are* there; the embedding under-weights them | index |
| 6 | ANN recall vs exact | the graph never returned it (low `ef_search`, quantization) | index |
| 7 | metadata filters exclude it | a filter is silently removing it | index |
| 8 | near-duplicates crowd it out | k slots wasted on copies of one chunk | representation |
| 9 | chunk is a usable unit | heading-only or table-syntax chunks embed layout, not meaning | representation |
| 10 | single-hop question | the question needs two retrievals, not a bigger k | query |
| 11 | HyDE moves it up | confirms a question-vs-statement phrasing gap | query |
| 12 | re-ranker rescues it | it is *found* but mis-ordered: the textbook re-ranking case | - |

Three of these deserve a note.

**#3 is a score gap, not just a rank.** Rank tells you where it is; the gap tells you how
hard it will be to move. A gold chunk scoring 0.58 behind a 0.59 top hit is one good
re-ranker away from first place. A gold chunk at 0.32 behind a 0.59 is in a different
neighbourhood of the vector space, and no amount of re-ranking within the top-20 will find
it: you need to change the *query* or the *chunk*.

**#6 separates a ranking fault from an index fault**, and it is the one people skip. Run
the same query as an exact brute-force scan over every vector. If exact search puts the
chunk at rank 2 and your production search puts it at 40, the ANN index lost it: `ef_search`
is too low, or quantization is too aggressive, or the graph is badly built. If exact search
*also* says rank 40, the index is innocent and your embedding model simply disagrees with
you about relevance. In this book's embedded Qdrant the search is already exact, so the
script reports that and tells you what to re-run against a server. Chapter 22 is the
machinery behind this check.

**#12 is the honest test for "should I buy a re-ranker".** If a cross-encoder lifts the
chunk from 17 to 3, your first stage is fine and you have a ranking problem: deploy the
re-ranker. If it does not, re-ranking is not your fix, no matter what the vendor's blog
says.

### The output

Running it on golden question q17: the failure Chapter 10 found and could not fully
explain:

```
question : Which database and message broker does Beacon use?
expects  : 05-beacon-fleet-software    keywords: ['PostgreSQL', 'Kafka']

BASELINE dense top-5:
   1. 0.590  05-beacon-fleet-software.md        '# Beacon - Fleet Management Software (Architecture a'
   2. 0.518  05-beacon-fleet-software.md        '## API'
   3. 0.470  11-release-notes-beacon-4.2.md     '# Beacon 4.2 - Release Notes'
   4. 0.465  14-faq.md                          '**Q: What is the Beacon API rate limit?**'
   5. 0.463  05-beacon-fleet-software.md        '## Release process'

   rank of the first chunk from 05-beacon-fleet-software: 1
   rank of the chunk that CONTAINS the answer : 15      <- the number that matters
   document-level recall@5 scores this retrieval 1.0. It failed.
```

Read those last three lines again. **Document-level recall@5 is 1.00 and the system cannot
answer the question.** Three chunks of the right document are in the top five; the one
containing "PostgreSQL 16" and "Apache Kafka" is at rank 15 of 56. This is the concrete
version of the metric lesson in Chapter 21: your labels decide what your metrics can see,
and document-level labels cannot see this failure at all.

Then the verdicts:

```
#  check                             verdict  detail
------------------------------------------------------------------------------
1  document is in the index          OK       5 chunks indexed
2  answer lives in one chunk         OK       1 chunk(s) contain every keyword
3  rank / score gap (dense)          FAIL     answer chunk at rank 15 (score 0.324); top hit 0.590 (gap 0.266)
4  query/chunk vocabulary overlap    FAIL     0/4 query terms appear in the answer chunk
                                              (missing: ['beacon', 'broker', 'database', 'message'])
5  lexical / hybrid ranks it higher  FAIL     dense 15 → BM25 8, hybrid(RRF) 10
6  ANN recall vs exact search        INFO     this run is exact, so rank is a RANKING fault, not an index fault
7  metadata filters exclude it       INFO     no filter applied in this run
8  near-duplicates crowd it out      OK       14 chunks rank above it, none ≥0.90 similar
9  chunk is a usable unit            OK       77 words, 60% structural lines
10 single-hop question               OK       one chunk can answer it
11 HyDE moves it up                  FAIL     rank 15 → 7 with a hypothetical answer (1858 ms)
12 reranker rescues it               WARN     cross-encoder DEMOTES it 15 → 20: the reranker shares the
                                              bi-encoder's blind spot here, so re-ranking is not the fix

RECOMMENDATION (cheapest first; each line is backed by a check above)
   [cheap    ] (from #4) Vocabulary gap: the chunk never uses the asker's words. HyDE or a rewrite
                          bridges it; contextual retrieval fixes it at ingest.
   [cheap    ] (from #5) Turn on hybrid search: the terms are IN the chunk, the embedding just does
                          not weight them.
   [cheap    ] (from #11) Add HyDE for this class of question - confirms a question/statement gap.
```

Diagnostic 4 is the smoking gun: **zero of the four content words in the question appear in
the chunk that answers it.** The user says "database" and "message broker"; the document
says "PostgreSQL 16" and "Apache Kafka". A dense embedding is supposed to bridge exactly
that gap, and here it does not bridge it far enough: the words "Architecture and
Operations" in the document's *title* chunk are a better literal match for an
infrastructure question than the section that names the technologies.

And diagnostic 12 is the one that saves you money: the cross-encoder **demotes** the chunk
from 15 to 20. The small re-ranker shares the bi-encoder's blind spot. Had you bought a
re-ranker to fix this ticket, you would have made it worse and still been billed.

### Two contrasting runs

A healthy retrieval, `--golden q11` ("What is the maximum payload of the Atlas A2?"):

```
   rank of the chunk that CONTAINS the answer : 1
3  rank / score gap (dense)          OK       answer chunk at rank 1 (score 0.664); gap 0.000
4  query/chunk vocabulary overlap    OK       4/4 query terms present (100%)
9  chunk is a usable unit            WARN     100% of lines are headings/table syntax
RECOMMENDATION
   [medium] (from #9) Prepend the heading path, or add a one-line summary before embedding.
```

Even a passing query has something to say: every line of the winning chunk is a heading or
table syntax. It works
today because the question is lexically close. It is fragile.

And a genuine multi-hop failure, `--golden q42 --keywords blackout "5 days"`:

```
2  answer lives in one chunk         FAIL     no single chunk has all of ['blackout', '5 days'];
                                              fragments in chunk_index [1, 2]
10 single-hop question               FAIL     the keywords live in different chunks
RECOMMENDATION (cheapest first)
   [cheap    ] (from #10) Decompose the question or let an agent retrieve twice.
   [expensive] (from #2) Re-chunk: the answer straddles a boundary...
```

Same symptom class, completely different cause and fix. This is why you run the playbook
instead of reaching for k.

### A word about labels

Run q42 with its *golden* keywords and diagnostic 2 reports "OK: 34 chunks contain every
keyword". The golden keyword for q42 is `["No"]`, which matches most of the corpus. The
script now detects this and says so:

```
2  answer lives in one chunk         INFO     keywords ['No'] match a quarter of the corpus -
                                              too generic to locate the answer chunk
```

**A diagnostic is only as good as the ground truth you feed it.** A vague label does not
produce a wrong answer, it produces a confidently useless one: which is worse. This is the
same failure mode as document-level relevance labels, one level down.

## 27.3 Reading the verdicts

The recommendation block sorts fixes by cost, but you still choose. This table maps
combinations of fired diagnostics to root causes: combinations, because a single
diagnostic is rarely decisive.

| Fired | Root cause | Fix | Cost |
|---|---|---|---|
| 1 | not indexed | fix ingestion; check the manifest and the DLQ (Ch 30) | cheap |
| 4 + 11 | vocabulary gap between question and document | HyDE or query rewriting for this class; contextual retrieval at ingest | cheap → expensive |
| 4 + 5 | the terms are present but under-weighted | hybrid retrieval with RRF (Ch 23) | cheap |
| 5 only | rare exact tokens (ids, versions, error codes) | hybrid; never rewrite the identifier away (27.4) | cheap |
| 6 | the ANN graph lost it | raise `ef_search`, rescore quantized vectors, rebuild with higher `m` (Ch 22) | medium |
| 7 | filter excludes it | audit the filter; check for missing metadata on old documents | cheap |
| 8 | near-duplicates | dedup at ingest, MMR at query time (Ch 29) | cheap |
| 9 | the chunk is layout, not content | prepend heading path; contextual prefix; re-chunk | medium |
| 2 + 10 | the answer is split / multi-hop | decompose, or parent/child so the reader gets the section | cheap → expensive |
| 12 | found but mis-ranked | retrieve wide, re-rank narrow (Ch 24) | medium |
| 3 with a small gap and nothing else | genuinely close call | accept it; raise k *behind* a re-ranker | medium |

Two rules for using it.

**Change one thing.** Then re-run the diagnostic *and* the Chapter 10 eval. A fix that
moves one question and breaks three is common; only the eval sees the three.

**Prefer query-side fixes for one-off failures and index-side fixes for classes.** If three
tickets in a month all show diagnostic 4, that is not three tickets, it is a corpus whose
vocabulary does not match your users', and the real fix is contextual retrieval at ingest.

## 27.4 The fixes, in order of cost

### Query rewriting, properly

Rewriting is the cheapest lever: one LLM call, no re-indexing, reversible by deleting a
line. [`code/ch27/query_rewriting_lab.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch27/query_rewriting_lab.py) implements six variants and measures them all on
the full golden set.

| Transform | What it does | Reach for it when |
|---|---|---|
| **rewrite** | restate the question in the corpus's formal vocabulary, preserving identifiers | informal or slang queries |
| **expand** | append synonyms and expanded acronyms, keeping the original words | jargon/acronym mismatch |
| **HyDE** | write a hypothetical *answer*, embed that instead | question-shaped query vs statement-shaped corpus (#4, #11) |
| **step-back** | ask the more general question too, fuse both result lists | specific instance answered by a general rule |
| **decompose** | split into sub-questions, retrieve for each, fuse | multi-hop (#10) |
| **conversational** | resolve "it", "that policy", "and for contractors?" against the previous turns | any chat interface |

The last one is not optional in a chat product, and it is the one most often forgotten. "How
many days do I get?" followed by "and during probation?" is not a retrievable query: it
must first become "how many PTO days can an employee take during probation?". Rewrite
against the last 2–3 turns, not the whole history, or the rewrite drags in stale topics.

Now the measurement. Forty-two questions, exact search, so the only variable is the query text:

```
transform       MRR  recall@5  answer@3  ms/query  LLM calls
------------------------------------------------------------
baseline      1.000     0.988     0.952       573          0
rewrite       0.988     0.964     0.952      1530          1   +0.00 answer@3
expand        1.000     0.976     0.929      1594          1   -0.02 answer@3
hyde          0.984     0.988     0.929      1781          1   -0.02 answer@3
step_back     1.000     0.988     0.952      2093          1   +0.00 answer@3
decompose     1.000     0.988     0.929      2949          1   -0.02 answer@3
```

**Not one transform beats the baseline, and three make it worse.** Every one of them at least
doubles latency, and the multi-query ones triple to quintuple it. If you had adopted "HyDE
improves RAG" from a blog post, you would have shipped a 3× slower retriever that answers no
more questions correctly.

This is not an argument against rewriting: diagnostic 11 showed HyDE moving q17 from rank
15 to 7. It is an argument against applying it *globally*. The transforms help the
questions that have the matching defect and hurt the ones that do not.

The lab prints exactly which questions each transform broke:

```
WHERE A TRANSFORM BROKE A QUESTION THE BASELINE GOT RIGHT
   expand: broke ['q14']
      q14: What safety standard is the Atlas A2 compliant with, and what is its IP rating?
         became -> 'What safety standard is the Atlas A2 compliant with, and what is its IP rating?
                    compliance certifica...'
   hyde: broke ['q14']
         became -> 'The Atlas A2 is compliant with IEC 62368-1 for audio/video and ICT equipment safety.
                    It has an IP65 ...'
   decompose: broke ['q14']
         became -> 'What is the IP rating of the Atlas A2?'
```

Look at the HyDE line. The true answer is **ISO 3691-4 and IP54**. The hypothetical answer
confidently invented **IEC 62368-1 and IP65** (plausible standards, wrong ones) and the
search then went looking in the neighbourhood of that hallucination. *HyDE searches with a
hallucination; when the hallucinated specifics are wrong, you search the wrong
neighbourhood.* That is the mechanism, and it is why HyDE helps vague questions and hurts
precise technical ones. (HyDE's probe is regenerated on every run, so whether it breaks q14 in
particular comes and goes; `expand` and `decompose` break it every time.)

Decomposition broke the same question differently: it split a two-part question and the
retained sub-question dropped the safety-standard half.

**The conclusion is a router, not a transform.** Classify the query first (identifier-like,
vague, multi-part, conversational) and apply the transform that class needs. A cheap
classifier plus four branches beats any single global transform. And whatever your rewriter
does, it must preserve identifiers verbatim: the careful prompt in the lab keeps
`BEACON-4187` intact, and a thirty-second prompt may not.

### The rest of the ladder

**Metadata filters** (cheap): free precision when the question implies a scope. Audit them
when diagnostic 7 fires: the classic bug is documents ingested before a field existed,
which then silently fail every filter.

**Hybrid retrieval** (cheap): the fix for diagnostic 5. Chapter 23 derives why rank fusion
rather than score addition.

**Deeper candidates plus re-ranking** (medium): the fix for diagnostic 12 and the *only*
legitimate version of "raise k". Chapter 24.

**A better embedding model** (medium): a real option, with a real evaluation protocol in
Chapter 25. Re-embeds the corpus, so it is a migration, not a config change (Chapter 30).

**Chunking changes** (expensive): re-embeds everything and changes every metric at once.
Which is why you do not do it by feel.

## 27.5 Chunk size as a designed experiment

> *"Why 500 tokens? Why not 1,000 or 2,000?"*

The honest answer is that chunk size is a variable with an optimum that depends on your
corpus, your questions, your k and your model: and that you can find it in ten minutes.
[`code/ch27/chunk_experiment.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch27/chunk_experiment.py) runs the factorial sweep:

```
strategy ∈ {recursive, markdown-aware, sentence-packing}
  ×  chunk_size ∈ {200, 400, 800, 1200, 1600} characters
  ×  overlap    ∈ {0%, 10%, 25%}                        = 45 configurations
```

Each cell is measured with document `recall@5`, `nDCG@5`, and (the metric that actually
separates them) **`answer@3`**: do all the golden keywords survive in the concatenated
top-3 chunks? Plus the context tokens that configuration costs per query, and the dollar
cost at 1,000 queries.

Retrieval in the sweep is exact (a numpy dot product over every chunk) so that nothing from
the ANN layer contaminates a chunking measurement. The question embeddings are computed
once and reused across all 45 cells, which is what makes the whole sweep cost about a cent.

The top and bottom of the 45-row table:

```
strategy    size   ovl  chunks  mean_tok  recall@5   ndcg@5  answer@3  ctx_tok   $/1k q
---------------------------------------------------------------------------------------
sentence     800   25%      56       167      1.00     0.98      1.00      872     1.01
markdown     400   10%     126        71      1.00     0.99      0.98      379     0.64
markdown     400    0%     125        71      0.99     0.99      0.98      381     0.64
markdown     400   25%     126        73      1.00     0.99      0.98      387     0.65
sentence    1600    0%      28       280      0.96     0.93      0.98     1428     1.43
...
recursive    400   10%     139        57      1.00     0.98      0.90      338     0.61
sentence     200   10%     202        40      0.99     0.98      0.86      227     0.53
markdown     200    0%     215        46      0.99     0.99      0.83      243     0.54
recursive    200    0%     298        26      1.00     0.99      0.79      167     0.49
recursive    200   25%     302        26      1.00     0.99      0.79      168     0.49
```

Four things fall out of that table, and they are all interview answers.

**1. `recall@5` is useless here and `answer@3` is not.** Document recall sits between 0.93 and
1.00 across all 45 configurations: a 7-point spread, and only two of the 45 cells fall below
0.95. Answer coverage spreads from **0.79 to 1.00**, a 21-point range on the same corpus with
the same embedding model. If you tune chunk size
on document-level recall you will conclude that chunk size does not matter, and you will be
wrong. (Chapter 21 is the general version of this.)

**2. Smaller is not "more precise".** 200-character recursive chunks give you 298 vectors,
the cheapest context in the sweep (167 tokens), and the *worst* answer coverage (0.79). One
fact in five is cut in half. The people who say "small chunks are precise" are measuring
precision at the wrong granularity.

**3. Structure beats size.** The markdown-aware splitter at 400 characters reaches 0.98
answer coverage using 379 context tokens. The sentence splitter needs 800-character chunks
and **872 tokens**: 2.3× the context: to buy the last two points. Splitting where the
author split, and carrying the heading path into the chunk text, is worth more than any
amount of size tuning, because it is the only strategy that keeps a table row attached to
its header.

**4. Overlap does almost nothing here: because the corpus has headings.** At markdown/400,
0%, 10% and 25% all score 0.98; at markdown/800 and above they are identical to two decimals. Overlap earns its storage on
unstructured prose where boundaries are arbitrary; on a structured handbook the heading
boundary is already the right cut.

### The Pareto frontier

There is no winner, there is a frontier: the configurations that nothing beats on *both*
quality and cost:

```
PARETO FRONTIER (nothing beats these on both answer@3 and context cost)
   recursive  size=200   ovl=0%    answer@3=0.79  ctx=167 tok  $0.49/1k queries
   sentence   size=200   ovl=10%   answer@3=0.86  ctx=227 tok  $0.53/1k queries
   recursive  size=400   ovl=10%   answer@3=0.90  ctx=338 tok  $0.61/1k queries
   markdown   size=400   ovl=10%   answer@3=0.98  ctx=379 tok  $0.64/1k queries
   sentence   size=800   ovl=25%   answer@3=1.00  ctx=872 tok  $1.01/1k queries
```

Now the decision is a business one, stated in one sentence: *the last two points of answer
coverage cost 58% more per query.* At 100 queries a day nobody cares and you take the 1.00.
At 10 million queries a day that is a meaningful line item and you take markdown/400 and
spend the difference on a re-ranker. **That sentence is what a senior engineer is expected
to produce, and it is not available to anyone who picked 500 because a tutorial did.**

### What moves the optimum

The frontier above is for *this* corpus. The reasoning transfers:

| Factor | Pushes chunks **smaller** | Pushes chunks **larger** |
|---|---|---|
| Question type | single-fact lookup ("what is the payload?") | synthesis, comparison, summarisation |
| Fact density | dense spec sheets, FAQs, tables | narrative prose, reasoning-heavy docs |
| Structure | strong headings to split on | none, so boundaries are arbitrary |
| A re-ranker in the pipeline | yes: retrieve many small, re-rank, send few | no: each chunk must stand alone |
| Embedding model | short max-token limits | long-context embeddings |
| Generation model | weak at long context | strong at long context |
| Cost pressure | high QPS | low QPS |

And two techniques that stop you having to trade at all, both from Chapter 9:
**parent/child**: index 200-character children for precise matching, send the 800-character
parent to the model, which on this sweep would combine the 0.99 nDCG of small chunks with
the 1.00 answer coverage of large ones; and **contextual chunking**: prefix each chunk with
an LLM-written sentence saying what it is about, which is the ingest-time fix for
diagnostic 4 and 9.

With `--judge` the script also runs faithfulness and answer correctness on the top
configurations, because keyword coverage is a proxy and at some point you have to check the
thing you actually care about.

## 27.6 Debugging in production

A golden set finds the failures you thought of. Production finds the rest.

**Sample everything cheap, judge a slice.** Chapter 13's JSON line already stores the
retrieved ids and their scores. That is enough to re-run any of the twelve diagnostics
offline, months later, without reproducing the request.

**Watch the "nothing above threshold" bucket.** Queries where the top score never clears
your abstention threshold (Chapter 12) are your highest-value debugging queue: they are
failures the system already knows about. Cluster them by embedding and the clusters *are*
your content gaps and your vocabulary gaps.

**Cluster the misses, do not read them.** Ten thumbs-down on ten different questions is
noise. Ten thumbs-down that cluster around one topic is a missing document, and one
ingestion fixes all ten.

**Every confirmed miss becomes a golden question.** This is the flywheel: production finds
a failure, the playbook diagnoses it, the fix is measured against the golden set, and the
failure joins the golden set so it can never return silently. A golden set that does not
grow is a golden set that is drifting away from your users.

## 27.7 Case study: fixing q17

The full loop on the question Chapter 10 flagged and this chapter diagnosed.

**Symptom.** q17 abstains in every configuration of the Chapter 10 eval. Document recall@5
= 1.00.

**Diagnosis** (27.2): the answer chunk is at rank 15; zero of four query terms appear in it;
BM25 ranks it 8 and hybrid 10; HyDE moves it to 7; the cross-encoder demotes it to 20.
Verdict: a **vocabulary gap**, confirmed from three directions.

**Candidate fixes, in cost order.**

| Fix | Evidence for it | Evidence against |
|---|---|---|
| Raise k to 20 | it would put rank 15 in context | 3× tokens on every query; q17's chunk still arrives at rank 15 inside a 20-chunk context, where §29.2 shows position starts to matter |
| Hybrid | BM25 ranks it 8 vs dense 15 | 8 is still not 5; helps but does not fix |
| HyDE | 15 → 7 | still not top-5; costs 1.8 s; measured as a net *negative* across the golden set (27.4) |
| Re-ranking | - | measured to make it *worse* (15 → 20) |
| Contextual prefix at ingest | attacks the actual cause | re-embeds the corpus |

**The decision.** Every cheap fix is a partial fix, because they all treat a symptom of one
root cause: *the chunk does not say what it is about*. It lists PostgreSQL and Kafka under
a heading that never uses the words "database" or "message broker".

The fix that matches the diagnosis is the ingest-time one: give each chunk a one-line
LLM-written prefix describing what it contains ("This section lists the infrastructure
components Beacon runs on: database, event stream, and transport"). That single change
addresses diagnostic 4 (vocabulary), diagnostic 9 (structural chunks), and helps the
markdown-400 configuration from the sweep at the same time. It costs one cheap LLM call per
chunk, once: for this corpus, 56 calls.

Chapter 9 §9.6 describes contextual retrieval and Exercise 3 below has you implement and
measure it. The point of the case study is not the fix; it is that **five candidate fixes
were on the table and four were eliminated by measurement rather than by argument**.

## The interview answer

> **"The correct document is at rank 17. What do you do?"**

"I'd find out why before I changed anything, because rank 17 has at least a dozen causes
and they need different fixes. Raising k from 5 to 20 is the tempting move and it's usually
wrong: it triples my input tokens on every query forever, and in our own eval going from
k=3 to k=6 raised recall but dropped the correct-answer rate from 0.83 to 0.79, because the
model had more wrong material to blend.

So I run three checks first. Is it a *query* problem: does the question share any
vocabulary with the chunk? On one of our questions, zero of four content words appeared in
the chunk that answered it: the user said 'database and message broker', the document said
'PostgreSQL 16' and 'Apache Kafka'. Is it an *index* problem: does exact brute-force search
rank it higher than my ANN search? If yes, my `ef_search` or quantization lost it; if no,
the index is innocent and it's a ranking problem. Is it a *representation* problem: is the
answer even inside one chunk, or did chunking split it?

Then I pick the cheapest fix that matches the diagnosis: hybrid search if the terms are
literally there, HyDE or a rewrite for a phrasing gap, decomposition if it's multi-hop,
re-ranking if a cross-encoder can lift it: and I verify that. On our case the cross-encoder
actually *demoted* it from 15 to 20, so re-ranking would have been money spent to make it
worse. And I change one thing at a time and re-run the full eval, because fixes that move
one question and break three are the normal case."

> **"Why 500 tokens and not 1,000 or 2,000?"**

"Because I measured it: 500 by itself is just a number I inherited from a tutorial. I run a
factorial sweep of strategy × size × overlap and score every cell on answer coverage, not
document recall, because document recall stayed inside a 0.93–1.00 band across all 45 of our
configurations while answer coverage spread from 0.79 to 1.00. On our corpus the winner
wasn't a size at all, it was a strategy: markdown-aware splitting at 400 characters got
0.98 answer coverage on 379 context tokens, while sentence-packing needed 800-character
chunks and 872 tokens for the last two points. So the real output is a Pareto frontier and a
sentence for the business: the last two points of accuracy cost 58% more per query. Then I
say what moves that optimum (single-fact lookups and a re-ranker push smaller, synthesis
questions and unstructured prose push larger) and I mention that parent/child retrieval
lets you stop trading at all."

## Run it

```bash
# the playbook on the failure Chapter 10 could not explain
QDRANT_MODE=memory uv run python code/ch27/diagnose.py --golden q17

# a healthy query, and a multi-hop one, for contrast
QDRANT_MODE=memory uv run python code/ch27/diagnose.py --golden q11
QDRANT_MODE=memory uv run python code/ch27/diagnose.py --golden q42 --keywords blackout "5 days"

# your own question
QDRANT_MODE=memory uv run python code/ch27/diagnose.py \
      --question "how fast is the robot" --expect 04-atlas-a2-specification --keywords 2.0

# the chunking experiment: 12 cells (~1 min) or the full 45 (~4 min)
QDRANT_MODE=memory uv run python code/ch27/chunk_experiment.py --quick
QDRANT_MODE=memory uv run python code/ch27/chunk_experiment.py
QDRANT_MODE=memory uv run python code/ch27/chunk_experiment.py --quick --judge --limit 10

# six query transforms, measured (42 questions, ~5 min, ~$0.10)
QDRANT_MODE=memory uv run python code/ch27/query_rewriting_lab.py --limit 12
QDRANT_MODE=memory uv run python code/ch27/query_rewriting_lab.py
```

Expect: the verdict tables in §27.2 (HyDE's rank varies by a place or two between runs:
it is an LLM writing the probe), the 45-row grid and Pareto frontier in §27.5, and the
transform table in §27.4 including the q14 regression. The chunking sweep is deterministic;
the transform table moves by about one question (±0.02) between runs.

## Exercises

1. Run `diagnose.py` on every golden question whose answer chunk is outside the top-5
   (write the loop). Tally which diagnostics fire most often. That tally, not any single
   ticket, tells you what to fix in the pipeline.
2. Add diagnostic 13: "would a larger `chunk_size` keep the answer whole?" Re-chunk at
   1600 characters inside the script and report the answer chunk's new rank.
3. Implement contextual retrieval from §27.7: one cheap LLM call per chunk writing a
   one-line "this chunk is about…" prefix, embedded with the chunk: and measure q17's rank
   before and after, then the whole Chapter 10 eval to check nothing else regressed.
4. In `query_rewriting_lab.py`, add a router: classify each question as
   identifier / vague / multi-part / precise-technical with one cheap structured call, and
   apply only the matching transform. Does the router beat the baseline where every global
   transform failed?
5. Add `conversational` to the lab with three-turn histories you write yourself, and measure
   what happens when you rewrite against the last 2 turns versus the last 10.
6. Take the Pareto frontier from §27.5 and add a parent/child configuration (index at 200,
   return the 800-character parent). Where does it land on the frontier?

## Interview questions

**Q: The correct document comes back at rank 17. Walk me through what you do.**
See "The interview answer" above: diagnose into query / index / representation before
treating. Concretely: check vocabulary overlap between question and chunk, compare ANN
ranking against exact brute-force ranking to separate an index fault from a ranking fault,
check whether the answer even survives inside one chunk, then apply the cheapest matching
fix and re-run the full eval. Raising k is the last resort and only as candidate depth
behind a re-ranker.

**Q: Why is raising k a bad default fix?**
It charges every future query for one query's bug: k=5→20 roughly triples input tokens, and
input tokens dominate RAG cost. It lowers precision, which measurably lowered our correct
answer rate (0.83 → 0.79 going k=3 → k=6). And it does not fix whole classes of failure
(split answers, filters, unreachable chunks) it just hides them.

**Q: How do you tell a ranking problem from an index problem?**
Run the query as an exact brute-force scan over all vectors and compare ranks. If exact
search ranks the chunk far higher than production, the ANN layer lost it: raise
`ef_search`, turn on rescoring for quantized vectors, or rebuild with a higher `m`. If both
agree it is low, the index is innocent: the embedding model genuinely does not think that
chunk answers that query, so change the query or the chunk.

**Q: Your document-level recall@5 is 1.00 and the system still can't answer. How?**
Because document-level labels cannot see chunk-level failure. In our corpus three chunks of
the right document were in the top five and the chunk containing the actual answer was at
rank 15: recall@5 = 1.00, answer wrong. Fix the labels (chunk-level, or a keyword coverage
metric over the retrieved text) before you trust the metric.

**Q: When does re-ranking fix this and when does it not?**
It fixes it when the evidence is inside your candidate list but mis-ordered: test it by
re-ranking the top-20 and seeing whether the chunk moves into the top-5. It does not fix a
vocabulary gap: on our q17 a cross-encoder demoted the answer chunk from 15 to 20, because a
small cross-encoder shares the bi-encoder's blind spot. Always measure before buying a
stage.

**Q: How do you choose chunk size?**
A factorial sweep over strategy × size × overlap, scored on answer coverage and context
cost, producing a Pareto frontier rather than a winner. On our corpus document recall
stayed inside a 0.93–1.00 band everywhere while answer coverage spread 0.79–1.00, and the biggest lever
was structure-aware splitting, not size. Then I state the trade in business terms: the last
two points of accuracy cost 58% more per query.

**Q: Does chunk overlap matter?**
It depends entirely on whether your documents have natural boundaries. On our heading-rich
handbook, 0%, 10% and 25% overlap were within a point of each other, because the heading was
already the right cut. On unstructured prose where boundaries are arbitrary, overlap is what
stops a fact being cut in half: 10–20% is the usual range, and the cost is storage plus
near-duplicate chunks in the top-k.

**Q: Does HyDE improve retrieval?**
Sometimes, and it is not free. On our golden set HyDE was net negative (−0.02 answer
coverage) and tripled latency, yet on a specific failing question it moved the answer chunk
from rank 15 to 7. It also *broke* a question by hallucinating "IEC 62368-1, IP65" when the
truth was "ISO 3691-4, IP54": you search in the neighbourhood of the hallucination. So:
route it to vague, statement-mismatched questions; keep it away from precise technical ones.

**Q: How do you rewrite queries in a chat interface?**
Resolve references against the last two or three turns before retrieval: "and during
probation?" is not a retrievable query until it becomes "how many PTO days can an employee
take during probation?". Use a short window, because rewriting against the full history
drags in stale topics, and preserve identifiers verbatim; a careless rewriter deletes the
error code that was the only reason the query worked.

**Q: You have 200 production queries with thumbs-down. How do you prioritise?**
Cluster them by embedding rather than reading them. Ten complaints scattered across ten
topics are noise; ten that cluster on one topic are usually one missing document or one
vocabulary gap, and one fix closes all ten. Then run the playbook on a representative
member of each cluster and tally which diagnostics fire: the tally tells you whether to fix
the pipeline or the corpus.

**Q: How do you know a fix worked?**
Re-run the diagnostic on that question *and* the full evaluation on every question. A fix
that moves one question and regresses three is the normal outcome, and only the eval sees
it. Then add the original failure to the golden set so it can never regress silently.

## Key takeaways

- Rank 17 is a symptom with at least twelve causes in three buckets: query, index,
  representation. Diagnose before you treat; the bucket determines how expensive the fix is.
- Raising k charges every query forever for one query's bug, lowers precision, and can
  leave the real fault in place. The only good version of "raise k" is deeper candidates
  behind a re-ranker.
- Exact brute-force search is the check that separates a ranking fault from an ANN fault,
  and a cross-encoder trial is the check that tells you whether re-ranking is your fix: on
  our failing question it made things worse.
- Document-level recall@5 was 1.00 for a retrieval whose answer chunk sat at rank 15.
  Measure at the granularity where the failure lives.
- Chunk size is a designed experiment, not a preference. Ours produced a Pareto frontier
  where structure-aware 400-character chunks reached 0.98 answer coverage at 379 tokens,
  and the last two points cost 58% more.
- Query transforms are per-class tools, not global settings: on the full golden set none
  beat the baseline and three lost ground, while one of them fixed the specific question
  whose defect it matches.

## Next

→ [Chapter 28: Documents in the Real World](28-documents-in-the-real-world.md)

Every measurement in this chapter assumed clean markdown. Chapter 28 is what happens when
the corpus is PDFs: layout, reading order, tables, scans, and the diagram nobody can index.
