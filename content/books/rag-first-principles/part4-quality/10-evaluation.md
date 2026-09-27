# Chapter 10 · Evaluating RAG: Every Metric, How to Compute It, How to Move It

> **Goal:** by the end of this chapter you can define and compute every retrieval metric (hit rate, recall@k, precision@k, F1, MRR, MAP, nDCG, context precision/recall) by hand, implement the generation metrics (faithfulness, answer relevance, correctness, completeness, citation accuracy, abstention rates) as LLM judges, run one command that turns a RAG configuration into a table, and (the part interviewers care about) say what to change when a given number is low.
>
> Files: [`code/ch10/metrics.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch10/metrics.py), [`code/ch10/llm_judges.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch10/llm_judges.py), [`code/ch10/run_eval.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch10/run_eval.py). Results: `data/eval_results/*.json`.

---

## 10.1 Why evaluation is the whole job

Every chapter so far ended with "measure it". Here is why: a RAG system has at least eight knobs (chunk size, overlap, k, embedding model, hybrid on/off, re-ranker, prompt, LLM) and every knob interacts with every other. Turning one by feel is guessing. The only way to make progress is a **fixed set of questions with known answers** and a **script that prints the same table every time**. That is what this chapter builds. In the interview transcript the candidate said "recall@K, precision@K, faithfulness, correctness, golden dataset": the right words. This chapter is what those words mean and how you would defend them.

RAG evaluation splits cleanly in two, and you must measure both:

```
            question ──► RETRIEVAL ──► chunks ──► GENERATION ──► answer
                          │                          │
   retrieval metrics: was the right stuff        generation metrics: is the answer faithful
   in the top-k, and how high?                   to the chunks, relevant, correct, complete?
   (ids only, no LLM, free, deterministic)       (needs reading → LLM judge, costs cents)
```

A bad answer with perfect retrieval is a prompt/model problem. A bad answer with bad retrieval is a chunking/embedding/search problem. Without both halves you cannot tell which.

## 10.2 The golden dataset

Everything rests on [`data/golden/qa.yaml`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/data/golden/qa.yaml). Each item:

```yaml
- id: q32
  question: What would 120 robots with Beacon and Compass cost per month, paying monthly?
  answer: List is 120 × ($180 + $40) = $26,400; with the 15% discount ... $22,440 per month.
  sources: [10-pricing-and-plans]        # doc ids that contain the answer → retrieval labels
  answerable: true
  keywords: ["22,440"]                   # cheap exact-match proxy
  tags: [numeric, multi_hop]
```

Design rules that came from building it:

| Rule | Why |
|---|---|
| **40–100 questions** to start, grow to a few hundred | below 40 the noise swamps the signal (one question = 2.5 points); above a few hundred, judge cost bites |
| **Real user phrasing**, not restated document sentences | questions copied from the doc make retrieval look perfect (ours are close to that: see 10.8) |
| **Mix of types**: numeric, factual, table lookup, reasoning, multi-hop, conflict, code | each fails differently; `tags` let you slice |
| **10–15% unanswerable**, including *trap* questions (u03 "Atlas A3") and off-topic (u05) | otherwise you never measure refusal behaviour |
| **`sources` at document level**, chunk level if you can afford it | doc-level labels are cheap but over-estimate recall (10.8, q17) |
| **Reference answers short and factual** | judges compare facts, not prose |
| **Version it** in git next to the corpus | when the corpus changes, answers change; a golden set is a test suite |

**Synthetic generation** (asking the LLM to write questions per chunk) gets you to 200 items in ten minutes and is a fine *start*. Its risks: questions leak the chunk's vocabulary (inflates retrieval scores), they are uniformly "easy lookup" type, and they never include what users actually ask. Use synthetic to bootstrap, then replace with logged real questions, hand-labelled, as they arrive (Chapter 12's coverage report is the source).

## 10.3 Retrieval metrics, by hand

[`code/ch10/metrics.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch10/metrics.py). Inputs everywhere: `retrieved`: ranked list of ids; `relevant`: set of correct ids; `k`. Run the file for the worked example:

```
retrieved (ranked): ['B', 'A', 'D', 'C', 'E']   relevant: ['A', 'C']   k=5
  hit_rate@5             1.000   at least one relevant in top-5
  recall@5               1.000   2 found / 2 relevant
  precision@5            0.400   2 relevant / 5 shown
  f1@5                   0.571   2·1·0.4/(1+0.4)
  recall@3               0.500   only A in top-3 → 1/2
  MRR (1 q)              0.500   first relevant at rank 2 → 1/2
  AP                     0.500   (P@2 + P@4)/2 = (0.5 + 0.5)/2
  nDCG@5                 0.651   (1/log2 3 + 1/log2 5) / (1/log2 2 + 1/log2 3)
  context_precision@5    0.500   (P@2·1 + P@4·1)/2
```

| Metric | Formula | Question it answers | Range / good value |
|---|---|---|---|
| **hit rate@k** (success@k) | 1 if any of top-k ∈ relevant | did we find *anything* useful? | 0/1; average ≥ 0.9 |
| **recall@k** | \|relevant ∩ top-k\| / \|relevant\| | of what we needed, how much did we get? | ≥ 0.8; the ceiling on answer quality |
| **precision@k** | \|relevant ∩ top-k\| / k | of what we showed the LLM, how much was useful? | ≥ 0.5; noise → distraction, cost |
| **F1@k** | 2PR/(P+R) | one number when choosing k | - |
| **MRR** | mean of 1/rank of *first* relevant | how high is the first good chunk? | ≥ 0.8; "one good chunk is enough" tasks |
| **MAP** | mean of AP = (1/\|rel\|) Σ P@i over relevant ranks i | are *all* relevant items ranked early? | multi-source questions |
| **nDCG@k** | DCG@k / IDCG@k, DCG = Σ gain_i / log2(i+1) | order-aware, position-discounted quality | ≥ 0.85; the standard in search |
| **context precision@k** | Σ (P@i · rel_i) / (# relevant in top-k) | are the useful chunks at the *top* of the window? | RAGAS name; ≈ 1 when relevant come first |
| **context recall** | fraction of gold sources (or gold-answer claims) covered by the context | did the context contain what the reference answer needs? | RAGAS: claim-level, needs LLM; ours: doc-level |

> **The question that fails candidates.** *"Your Recall@5 is 95%. Does that mean the system
> answers 95% of questions correctly?"* **No.** It means that for 95% of questions, a chunk you
> *labelled* relevant appeared somewhere in the top 5: under your labelling, on your evaluation
> set. The generator can still misread the evidence, use only part of it, be derailed by a
> conflicting chunk, or ignore it and answer from memory. Retrieval recall is an **upper bound**
> on answer quality, not a measure of it. §10.6 shows the gap in our own numbers: recall 0.94–1.00,
> correct rate 0.79–0.83. Closing that gap is what the generation metrics in §10.4 are for.
> [Chapter 21](../part8-deep-dives/21-retrieval-metrics-deep.md) decomposes the gap question by
> question and shows how to tell a real improvement from noise.

Two subtleties that the code handles and interviews probe:

- **Chunk vs document granularity.** Our labels are document ids; the retriever returns chunks. Five chunks from the right document is one hit, not five. For rank metrics (MRR, nDCG, recall) `run_eval.py` first collapses the chunk list to unique documents in order (`unique_in_order`). For precision and context precision it keeps chunks: "what fraction of the k things the LLM reads are useful" is a chunk question. State which you are using.
- **Why nDCG discounts by log2(i+1).** Position 1 has weight 1, position 2 → 0.63, position 3 → 0.5, position 10 → 0.29. The LLM reads in order and attends more to the start and end of its context ("lost in the middle"), so a relevant chunk at rank 1 is genuinely worth more than one at rank 5. nDCG normalises by the best possible ordering so different questions (1 vs 3 relevant docs) are comparable.

## 10.4 Generation metrics: the LLM as judge

[`code/ch10/llm_judges.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch10/llm_judges.py). Every judge is a Pydantic model + `with_structured_output`: numbers out, never prose. The test at the bottom of the file judges one good answer, one hallucinated answer and one abstention:

```
[good] You get 24 days of PTO per year [1], and you can carry over up to 5 days [1].
  faithfulness: 1.0   claims: ['You get 24 days of PTO per year.', 'You can carry over up to 5 days.'] supported: [True, True]
  relevance   : 5
  correctness : correct  completeness=1.0
  citations   : 1.0

[hallucinated] You get 24 days of PTO per year [1], plus 10 extra days after 5 years of service [1].
  faithfulness: 0.5   supported: [True, False]        ← caught
  correctness : correct completeness=1.0               ← NOT caught: "extra detail does not conflict"
  citations   : 0.0                                    ← caught: [1] does not support sentence 2

[abstain] I don't know - the handbook does not mention this.
  faithfulness: 1.0 (nothing fabricated)   correctness: wrong   abstention: True
```

That middle case is the most important line in this chapter: **the correctness judge passed a hallucinated answer**, because the reference facts were present and the invented fact "did not conflict". (On other runs the same judge says `partial` for this answer: the verdict is not stable on exactly the case that matters, which is the point.) Correctness and faithfulness measure different things; you need both.

| Metric | How it is computed | Interpreting it |
|---|---|---|
| **faithfulness** (groundedness) | LLM extracts atomic claims from the answer → LLM marks each supported/unsupported by the context → ratio | **1 − faithfulness = hallucination rate.** Independent of truth: faithful to a wrong doc is still 1.0 |
| **answer relevance** | judge rates 1–5 whether the answer addresses the question (no context, no reference) | catches evasive or off-topic answers; a confident wrong answer scores 5 |
| **answer correctness** | judge compares to the reference answer: correct / partial / wrong; numbers must match | the metric stakeholders care about; needs a golden answer |
| **completeness** | judge: fraction of reference facts present | separates "right but thin" from "right" |
| **citation accuracy** | for each sentence with [n], does chunk n support it? | measures whether citations are real or decorative |
| **correct-abstention rate** | on unanswerable questions: did it refuse? (regex on the fixed refusal phrase) | should be ≈ 1.0 |
| **wrong-abstention rate** | on answerable questions: did it refuse? | the price of strict grounding; watch it *with* faithfulness |
| **keyword hit** | golden keywords present in the answer (no LLM) | free smoke alarm; run on every commit |

Judge design rules that keep these honest: binary or small-scale rubrics (yes/no per claim beats "rate 1–10"); one job per call; the judge never sees what it does not need (the faithfulness judge does not see the question or the reference); a *different* prompt from the generator. Known biases: **position** (first option preferred (randomise), **verbosity** (longer answers rated higher) rubric on facts, not length), **self-preference** (a model likes its own style (use a different model for judging when it matters), and **calibration drift**) label 30–50 answers by hand once and check the judge agrees ≥ 90%; re-check when you change the judge model.

## 10.5 The harness

[`code/ch10/run_eval.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch10/run_eval.py) takes a configuration and produces one table:

```bash
uv run python code/ch10/run_eval.py --k 3
uv run python code/ch10/run_eval.py --k 6
uv run python code/ch10/run_eval.py --k 4 --rerank
uv run python code/ch10/run_eval.py --k 8 --chunk-size 400 --no-judges     # fast, free
uv run python code/ch10/run_eval.py --k 3 --limit 10                       # while developing
```

For each golden question it retrieves, generates with a grounded prompt (fixed refusal phrase, [n] citations), computes the retrieval metrics on the ids, runs the judges on the answer, records latency and `usage_metadata` → cost, and writes `data/eval_results/<config>.json` with a summary plus every per-question row: the rows are where the learning is.

## 10.6 Results: four configurations, 47 questions each

Real numbers from `data/eval_results/` (gpt-5.4-mini, text-embedding-3-small, 14 documents,
42 answerable + 5 unanswerable). The `k=8, cs400` column was run with `--no-judges`, so it has
retrieval numbers only.

| metric | k=3, cs800 | k=6, cs800 | k=4 + rerank | k=8, cs400 (no judges) |
|---|---|---|---|---|
| hit@k (doc) | 1.000 | 1.000 | 1.000 | 1.000 |
| recall@k (doc) | 0.940 | **0.976** | **0.976** | **1.000** |
| precision@k (chunk) | **0.738** | 0.536 | 0.554 | 0.589 |
| MRR (doc) | 1.000 | 1.000 | 0.988 | 0.984 |
| nDCG@k (doc) | 0.954 | 0.975 | 0.973 | 0.975 |
| context precision@k | **0.992** | 0.917 | 0.938 | 0.847 |
| keyword hit | 0.952 | 0.940 | 0.917 | - |
| wrong-abstention rate | 0.071 | 0.095 | 0.095 | - |
| correct-abstention rate | 1.000 | 1.000 | 1.000 | - |
| faithfulness | 0.976 | 0.968 | **0.983** | - |
| hallucination rate | 0.024 | 0.032 | **0.017** | - |
| relevance (1–5) | 4.81 | 4.81 | 4.69 | - |
| correct rate | **0.857** | **0.857** | 0.833 | - |
| partial rate | 0.071 | 0.048 | 0.071 | - |
| completeness | **0.900** | 0.881 | 0.876 | - |
| citation accuracy | **1.000** | 0.974 | **1.000** | - |
| p50 / p95 latency (s) | 1.45 / 2.30 | 1.43 / 2.27 | 1.70 / 2.98 | 1.26 / 2.70 |
| avg input tokens | **594** | 1078 | 785 | 747 |

Reading the table like an engineer:

- **k buys recall and costs precision and money.** k=3→6 lifts document recall 0.94→0.98, and that
  part is real: recall is computed from ids, with no judge and no sampling noise. Everything else
  about that move is *not* distinguishable here: correct rate is identical (0.857), faithfulness
  and hallucination differ by less than one question, and input tokens nearly double. The
  defensible summary is "more recall, 1.8× the input tokens, no measurable answer gain on this
  corpus", not a story about hedging. [Chapter 21 §21.5](../part8-deep-dives/21-retrieval-metrics-deep.md)
  runs the paired significance test that makes that distinction.
- **Re-ranking at this corpus size buys nothing measurable.** MRR *falls* to 0.988 (the small
  cross-encoder demoted one gold chunk) and it adds ~250 ms. The first stage here is already
  precise; re-ranking pays when it is not, and
  [Chapter 24 §24.4](../part8-deep-dives/24-rerankers.md) builds a corpus where it does. Do not
  ship a stage your table does not justify.
- **Smaller chunks (400) with larger k (8)** reach recall 1.00 at fewer tokens than k=6/cs800
  (747 vs 1078): more, smaller pieces let the window hold more *distinct* facts. Context
  precision drops to 0.85: more of the window is filler.
  [Chapter 27 §27.5](../part8-deep-dives/27-retrieval-debugging.md) runs the full 45-cell sweep.
- **Wrong-abstention is 3–4 questions in every configuration.** It does not move with k, so it is
  not a k problem. See 10.8.
- **Citation accuracy is 0.97–1.00, and the first version of this table said 0.68–0.73.**
  That is worth dwelling on, because the difference was a **bug in the metric, not in the system**.
  `citation_accuracy` split the answer on sentence boundaries; when the model wrote
  `...24 days per year. [1]` the trailing `[1]` became its own "sentence" with no claim in it, and
  the judge was asked whether chunk 1 supports the string `"[1]"`: a coin flip. 37 of 39 cited
  answers contained one. The lesson is the one this whole chapter is about: **an evaluation harness
  is code, and code has bugs.** A metric that disagrees with your reading of the outputs is a
  hypothesis about your *metric* first. Spot-check the raw judged pairs before you believe a number:
and before you go optimise the thing it accuses.

Judges are stochastic: rerunning moves any judged number by ±0.02–0.05 on 42 questions. Never
celebrate a 0.01 improvement; a real change moves several related metrics together, and
Chapter 21 shows how to test whether it moved at all.

## 10.7 How to improve each metric

The table interviewers actually want. "Low X" → the most likely cause → what to try first (and what to check it against).

| Low metric | Most likely cause | Try first | Then |
|---|---|---|---|
| **hit rate / recall@k** | chunk boundaries split the fact; embedding misses the vocabulary; k too small | bigger `k` (check precision); chunk by headings (Ch. 5); hybrid (Ch. 8) | HyDE / multi-query (Ch. 9); better embedding model; contextual retrieval |
| **precision@k / context precision** | k too big; near-duplicate chunks; header/boilerplate chunks | smaller `k`; re-rank 20→4 (Ch. 9); MMR; drop chunks < 100 chars | metadata filters (Ch. 9); dedupe by parent |
| **MRR / nDCG** | right doc found but ranked low | re-ranker (cross-encoder); hybrid with RRF; contextual retrieval | fine-tune embeddings on your query→chunk pairs |
| **faithfulness** (hallucination high) | permissive prompt; answer synthesised from *nearby* facts; conflicting chunks | grounding rules + fixed refusal (Ch. 11); dates in context; more context (k↑) if recall is the gap | citation verifier; faithfulness gate with retry/abstain; smaller/cleaner chunks |
| **answer correctness** | retrieval recall (check first!); multi-hop question single-shot; arithmetic | fix recall; decomposition (Ch. 9) or agentic loop (Ch. 15); "show the arithmetic" instruction | stronger model for generation only; parent/child so the LLM sees the full table |
| **completeness** | k too small for multi-fact answers; chunks cut mid-list | k↑ for `multi_doc` tags; chunk on lists/tables intact | "answer all parts" instruction; decomposition |
| **answer relevance** | question rewritten/decomposed badly; over-cautious model | check the query transform output; loosen the prompt's brevity rule | different model |
| **citation accuracy** | citations are decorative; chunk numbering confusing | structured citations (Ch. 11); verify and drop; fewer, longer chunks | quote-then-cite prompting |
| **wrong-abstention** | strict prompt + the exact chunk missing (recall at *chunk* level); reasoning across chunks refused | inspect retrieved chunks for those ids; k↑ or HyDE for them; allow "answer from combined passages" | relax refusal rule; retrieval-then-sufficiency routing (Ch. 12) |
| **correct-abstention** (too low) | prompt allows prior knowledge; score gate absent | grounding rule 1; fixed refusal phrase; sufficiency judge | score threshold as pre-gate |
| **latency p95** | generation dominates; sequential retrieval steps; big k | stream tokens; cache query embeddings; Qdrant-native hybrid (one round trip) | smaller model; fewer stages; async |
| **cost / question** | input tokens ≈ k × chunk size | k↓ with re-ranking; shorter chunks; prompt caching | cheaper model for retrieval-side calls (judges, rewrites) |

Order of operations when a table looks bad: **retrieval first** (if recall is low nothing downstream can be fixed), then **prompt/grounding**, then **k and chunking trade-offs**, then **model**. Changing the LLM first is the most common and most expensive mistake.

## 10.8 What the per-question rows taught us

The summary hides the interesting part. Every configuration wrong-abstained on exactly three answerable questions: and mostly the same three:

```
q17  Which database and message broker does Beacon use?        → abstained (all four configs)
q23  Can I paste customer robot telemetry into a public AI assistant?  → abstained (all but k=8/cs400)
q42  New Sales hire, two weeks off in the last week of the quarter?    → abstained (all but k=4 + rerank)
```

(The odd ones out: k=4 + rerank abstained on q14 instead of q42; k=8/cs400 on q41 instead of q23. Same rate, different rows.)

Look at what was retrieved for **q17** at k=6: the chunks from the *right* document sit at ranks 1, 2 and 5; the other three are FAQ, release-notes and spec chunks:

```
0.59 05-beacon  '# Beacon - Fleet Management Software (Architecture and Operations) ...'   ← header/intro chunk
0.52 05-beacon  '## API ...'
0.46 05-beacon  '## Release process ...'
```

The `## Architecture` chunk that says "PostgreSQL 16 ... Apache Kafka" is **not there**. Doc-level labels report hit=1, recall=1: a perfect score for a retrieval that failed. The embedding of "database and message broker" landed on the document's title chunk (which contains the words "Architecture and Operations") rather than the section that names the technologies. Fixes, in the order you would try them: chunk-level labels so the metric *sees* it; hybrid (BM25 on "database" would match "PostgreSQL 16": no; but "broker"? no. Hybrid does not help here: the vocabulary genuinely differs); HyDE ("Beacon stores state in PostgreSQL and streams events on Kafka" embeds right next to the architecture chunk); contextual retrieval (prefix "This chunk lists Beacon's infrastructure components").

**q23** is the opposite: the right chunks *were* retrieved (`## AI tools` at 0.50, `## Data classification` at 0.40), and the model still refused. The answer requires combining two passages ("telemetry is Confidential" + "Confidential data only with reviewed tools") and rule 2 of the strict prompt made the model prefer refusal over inference. That is the wrong-abstention cost of grounding: and the reason faithfulness must be reported next to it.

**q42** needs the probation chunk (rank 4) *and* the blackout chunk (not in top-6). Multi-hop: decomposition or an agent (Chapters 9, 15).

Three failures, three different root causes, three different fixes: and the summary table showed one number, 7.1%, for all of them. **Always read the rows.**

## 10.9 Beyond offline eval

- **Offline eval** (this chapter): fixed golden set, run on every change, in CI. Gate merges on "no metric drops more than 0.03" rather than on absolute values.
- **Online eval**: sample 1–5% of production traffic, run the faithfulness and sufficiency judges asynchronously, chart the daily rate (Chapter 13). Add thumbs-up/down and *route* every thumbs-down to a human labeller: that is how the golden set grows with real questions.
- **A/B tests**: for prompt or k changes with a measurable business outcome (ticket deflection, time-to-answer). Offline metrics pick the candidates; A/B picks the winner.
- **Tools**: RAGAS (defines context precision/recall, faithfulness: currently incompatible with LangChain 1.x, which is why we implemented them), DeepEval (pytest-style assertions), LangSmith evaluators and datasets (hosted, integrates with the tracing in Chapter 13), Arize Phoenix, TruLens. Knowing what they compute (this chapter) matters more than which one you pick.

## Run it

```bash
uv run python code/ch10/metrics.py                         # worked example, no API calls
uv run python code/ch10/llm_judges.py                      # 3 answers × 4 judges ≈ 12 calls
QDRANT_MODE=memory uv run python code/ch10/run_eval.py --k 3 --limit 10    # ~1 min
QDRANT_MODE=memory uv run python code/ch10/run_eval.py --k 6               # 47 questions, ~5 min, ~$0.15
```

Expected: the metric table above (judged numbers within ±0.05), one ✓/✗ line per question, and a JSON file per configuration in `data/eval_results/`.

## Exercises

1. Run `--k 8 --chunk-size 400` *with* judges and add the column to the table. Did recall 1.00 turn into a higher correct rate, or did precision 0.59 hurt it?
2. Convert the golden labels to **chunk level** for five questions (find the `chunk_id`s that contain the answer) and recompute recall@k for q17. Compare with the doc-level number.
3. Fix q17 with HyDE (Chapter 9's `hyde()` as the retriever in `run_eval.py`). Does it break anything else?
4. Judge calibration: hand-label the 42 answers in `k3_cs800.json` as correct/partial/wrong. Compute agreement with the judge. Which kind of disagreement dominates?
5. Add an `nDCG` variant with graded relevance: 2 for a chunk that contains the exact answer, 1 for a chunk from the right document. You will need chunk-level labels from exercise 2.

## Interview questions

**Q: How do you evaluate a RAG system?**
Separately for retrieval and generation, on a versioned golden set of real-style questions with reference answers, source labels and 10–15% unanswerables. Retrieval: recall@k, precision@k, MRR, nDCG computed from ids: deterministic and free. Generation: LLM judges with structured output for faithfulness (claims supported by context), answer correctness vs the reference, relevance, completeness, citation accuracy, plus abstention rates on both answerable and unanswerable questions. Plus latency and cost. One script, one table per configuration, in CI; read the per-question rows, not just the summary.

**Q: How do you evaluate LLM outputs specifically?**
Structured LLM-as-judge with narrow rubrics: split the answer into atomic claims and check each against the context (faithfulness); compare facts against a reference (correctness); check each citation supports its sentence. Keep judges binary where possible, hide what they do not need, use a different prompt (ideally model) from the generator, and calibrate against 30–50 human labels. Complement with free proxies (keyword hits, refusal regex) for every commit.

**Q: How do you ensure the quality of your implementation?**
A golden set as a test suite, an eval harness that runs on every change, thresholds that fail the build on regression (e.g. faithfulness −0.03), per-question diffs between runs, tracing in production to sample real traffic into the judges, and a feedback loop that turns thumbs-down and abstentions into new golden questions. Quality is a process, not a number.

**Q: Recall@k vs precision@k in RAG: which matters more?**
Recall sets the ceiling: if the answer is not in the top-k, no prompt can recover it. Precision controls noise, cost and distraction: the model reads everything you give it. Start by getting recall ≥ 0.9 with a generous k, then recover precision with a re-ranker so you can send fewer chunks. Report both; the trade-off is the design decision.

**Q: What is faithfulness and how is it different from correctness?**
Faithfulness: fraction of the answer's claims supported by the retrieved context: measures hallucination, needs no reference answer. Correctness: does the answer match the ground truth: needs a reference. An answer can be faithful and wrong (the context was wrong or stale) or correct and unfaithful (right from memory). Our judge test showed the second case: a correctness judge passed an answer with an invented fact.

**Q: What is nDCG and why use it instead of precision?**
Precision ignores order; nDCG rewards relevant items near the top with a log2(rank+1) discount and normalises by the ideal ordering, so it is comparable across questions with different numbers of relevant docs. LLMs attend unevenly across the context, so order matters: nDCG measures what precision cannot.

**Q: What are the pitfalls of LLM-as-judge?**
Position bias, verbosity bias, self-preference, inconsistent scales on 1–10 rubrics, and drift when the judge model changes. Mitigate with binary per-claim rubrics, randomised order, a different judge model, and periodic calibration against human labels. And remember judges are stochastic: treat ±0.03 on 40 questions as noise.

**Q: How would you build the golden dataset?**
Start with 40–100 questions: bootstrap with synthetic generation per chunk, then replace with real logged questions labelled by domain owners. Include numeric, table, multi-hop, conflict and unanswerable types with tags. Store as YAML in git next to the corpus; every new production failure becomes a new item.

**Q: How do you go from an offline metric to knowing the product is better?**
Offline metrics select candidates; online sampling (judges on 1–5% of traffic) confirms no regression; A/B tests on a business metric (deflection, CSAT) decide. Offline numbers on a small golden set cannot distinguish a 0.02 change from noise: the rows and the business metric can.

## Key takeaways

- Evaluate retrieval and generation separately; the first is free and deterministic, the second needs structured LLM judges.
- Know the formulas: recall/precision@k, MRR, MAP, nDCG (log2 discount), context precision; know chunk- vs doc-level granularity and say which you use.
- Faithfulness ≠ correctness; wrong-abstention ≠ hallucination. Report the pairs together or you will optimise one by wrecking the other.
- One command → one table → one JSON per configuration; diff the rows, not the summary. Three identical 7.1% failures had three different causes.
- Improve in order: retrieval recall → grounding prompt → k/chunking trade-off → model. Changing the model first is the expensive mistake.

## Next

→ [Chapter 11: Stopping Hallucination](11-hallucination.md)

Chapter 11: the defences that move faithfulness and citation accuracy: grounding prompts that constrain, verified citations, and gates that abstain instead of guessing.
