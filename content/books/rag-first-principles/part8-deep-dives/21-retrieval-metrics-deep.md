# Chapter 21 · Retrieval Metrics Under the Microscope

> **Goal:** by the end of this chapter you can answer, without hesitating, what "Recall@5 = 95%" does and does not claim; you can construct two retrievers with identical recall and opposite quality; you can derive nDCG on a whiteboard; you know how much of your retrieval score is an artefact of labelling documents instead of chunks; and you can say whether a +0.03 improvement is real, with a confidence interval and a p-value.
>
> Files: [`code/ch21/metric_illusions.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch21/metric_illusions.py), [`code/ch21/recall_vs_correct.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch21/recall_vs_correct.py), [`code/ch21/chunk_level_labels.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch21/chunk_level_labels.py), [`code/ch21/significance.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch21/significance.py)
>
> Prerequisite: [Chapter 10](../part4-quality/10-evaluation.md) defined every metric and built the harness. This chapter is the layer underneath: where the metrics lie to you.

---

## 21.1 The question that ends interviews

> *"Your Recall@5 is 95%. Does that mean the RAG system answers 95% of questions correctly?"*

**No.** It means that for 95% of the questions in **that** evaluation set, under **that**
definition of relevance, at least the labelled evidence appeared somewhere in the top 5 results.
It is a claim about one stage of a four-stage pipeline, and it is the stage with the most
generous labels.

Everything that happens after retrieval can still destroy the answer:

- the retrieved *document* is right but the retrieved *chunk* does not contain the fact,
- the evidence is there and the model refuses anyway,
- the evidence is there and the model reads it wrong, does the arithmetic wrong, or blends two
  passages,
- two retrieved passages disagree and the model silently picks one.

Say it as a product of conditional probabilities and it becomes obvious:

```
P(correct answer)  =  P(right document retrieved)          ← this is what recall@k reports
                   ×  P(evidence actually in the context | document retrieved)
                   ×  P(model answers at all | evidence present)
                   ×  P(answer is right | model answered)
```

Recall@k is the **first factor only**. Quoting it as an accuracy number asserts that the other
three factors equal 1.0.

They do not. [`code/ch21/recall_vs_correct.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch21/recall_vs_correct.py) measures all four on the golden set, at k=5:

```
THE FUNNEL   (k=5, n=42 answerable questions)
  stage                         kept    of n  conditional
  0 asked                         42   1.000
  1 gold doc retrieved            42   1.000        1.000   ← recall@k reports THIS
  2 evidence in the context       40   0.952        0.952
  3 model answered                38   0.905        0.950
  4 answer correct                35   0.833        0.921   ← what the user experiences

  P(correct) = 1.000 x 0.952 x 0.950 x 0.921 = 0.833
  stage-1 success (a.k.a. 'hit rate@k'): 1.000
  end-to-end correct:                    0.833   → the gap recall cannot see: +0.167
```

**Document-level retrieval was perfect (42 out of 42) and one answer in six was still wrong.**
That 16.7-point gap is the entire content of the interview question. A candidate who says "95%
recall means 95% correct" has claimed a gap of zero.

The script also reports *where* each question died, which is the part that makes it useful
rather than merely humbling:

```
  WHERE QUESTIONS DIE
    2-evidence     2  q17, q37     right document, wrong chunk
    3-refusal      2  q23, q42     evidence present, model refused
    4-generation   3  q08, q29, q41  evidence present, answer wrong
    ok            35
```

Three different stages, three different fixes, and **recall@k reported 1.000 for all seven
failures**. This is why "which metric do you optimise?" has only one defensible answer: none of
them alone. You optimise the funnel, and you look at the stage that is losing the most.

A note on honesty in both directions: a question can also fail an early stage and still come out
right, because the fact appears in a document you did not label. The script counts those
separately and calls them what they are: correct by luck.

## 21.2 Metrics are not interchangeable

The second thing an interviewer probes is whether you understand that these metrics measure
genuinely different things. The cheapest way to prove you do is to construct cases where two
systems tie on one metric and separate on another. [`code/ch21/metric_illusions.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch21/metric_illusions.py) is nothing
but such cases; it calls no model and no database, so the numbers are identical on every
machine.

**Identical recall, identical precision, three different systems:**

```
  system    top-5 (relevant starred)      hit  recall   prec    MRR    MAP   nDCG   ctxP
  top       *A* *C* X Y Z                1.00   1.000  0.400  1.000  1.000  1.000  1.000
  middle    X *A* *C* Y Z                1.00   1.000  0.400  0.500  0.583  0.693  0.583
  bottom    X Y Z *A* *C*                1.00   1.000  0.400  0.250  0.325  0.501  0.325
```

Recall spread: 0.000. Precision spread: 0.000. MRR spread: 0.750. The "bottom" system puts three
distractors in front of the evidence: for a generator that reads in order and attends unevenly,
that is a materially worse system, and set-based metrics are structurally incapable of noticing.

**Identical MRR = 1.000, one third of the evidence:**

```
  complete  *A* *B* *C* X Y              1.00   1.000  0.600  1.000  1.000  1.000
  one-hit   *A* X Y Z W                  1.00   0.333  0.200  1.000  0.333  0.469
```

For a question whose answer needs three documents, MRR gives a perfect score to a system that
found one. MRR asks "how soon was the *first* good result": a question that only makes sense
when one good chunk is genuinely enough. Say that condition out loud when you quote MRR.

**Identical hit rate, a fluke:**

```
  good      *A* *B* *C* X Y              1.00   1.000  0.600  1.000
  lucky     X Y Z W *C*                  1.00   0.333  0.200  0.200
```

Hit rate@k is the most flattering metric in RAG and therefore the one most often quoted in a
standup. Anything scraping into last place scores the same as a perfect ranking.

**And k is not a detail:**

```
    k  recall@k   prec@k    f1@k   nDCG@k
    1     0.333    1.000   0.500    1.000
    3     0.667    0.667   0.667    0.704
    5     0.667    0.400   0.500    0.704
    8     1.000    0.375   0.545    0.871
```

Recall rises monotonically with k; precision falls. "Our recall is 95%" without a k is not a
claim. Recall@100 is nearly free and nearly useless, because the generator has to read all
hundred.

The summary to have ready:

| Metric | Sees | Blind to |
|---|---|---|
| hit rate@k | did anything relevant appear | how much, how high, how noisy |
| recall@k | coverage of the gold set | rank order, precision, the value of k |
| precision@k | noise fraction in the window | rank order, whether gold is missing entirely |
| MRR | rank of the **first** hit | every other relevant document |
| MAP | all relevant docs, rank-weighted | graded relevance; needs complete labels |
| nDCG@k | order **and** graded gains, normalised | what the generator does with the context |
| context precision | are the good chunks at the top | absolute coverage |
| **all of them** | the retrieval stage | **whether the answer was right** |

## 21.3 Your labels are the experiment

Every number above is computed against a set of labels you chose. The golden set labels
relevance at **document** level (`sources: [05-beacon-fleet-software]`) because that is cheap.
The retriever returns **chunks**. So "the right document appeared" scores a hit even when the
chunk holding the fact never made it into the context.

This is not a hypothetical. [`code/ch21/chunk_level_labels.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch21/chunk_level_labels.py) builds chunk-level ground truth
(one LLM call per question: show it every chunk of the gold document, ask which actually contain
the answer), saves it to [`data/golden/chunk_labels.yaml`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/data/golden/chunk_labels.yaml), and re-scores:

```
id     doc hit  doc rec  chunk hit  chunk rec  chunk MRR   verdict
q15       1.00     1.00       1.00       0.50       1.00   doc-level optimistic
q17       1.00     1.00       0.00       0.00       0.00   ← doc-level says HIT, the answer chunk is absent
q19       1.00     0.50       1.00       0.50       1.00
...
OPTIMISM BIAS OF DOCUMENT-LEVEL LABELS   (k=5, n=20 questions)
  metric                   doc-level   chunk-level   inflation
  hit rate@k                   1.000         0.950      +0.050
  recall@k                     0.950         0.900      +0.050
```

Five points of pure labelling artefact on this corpus: and the corpus is 14 short documents.
The inflation scales with document length: a document that produces 6 chunks gives you a 1-in-6
shot at looking perfect. On a corpus of 40-page PDFs, document-level labels are close to
meaningless.

Note *which* question the chunk labels exposed: **q17**, the same question [Chapter 10
§10.8](../part4-quality/10-evaluation.md) found by reading the per-question rows by hand. Two
independent methods, one bug. The chunk labels find it mechanically, which is the difference
between an anecdote and a metric.

Three labelling decisions to be able to defend:

- **Binary vs graded.** A chunk containing the exact figure and a chunk from the same section
  that merely discusses the topic are not equally useful. With binary labels they are identical.
  Case 6 in `metric_illusions.py` shows two systems tying at binary nDCG 1.000 and separating at
  graded nDCG (1.000 vs 0.860). Grade 2 for "contains the answer", 1 for "related", 0 otherwise
  is enough resolution for most systems.
- **Chunk vs document.** Chunk labels are the only way to see chunking failures at all. They
  cost about a minute of human time (or one LLM call) per question. Label at chunk level for the
  questions you care about; leave the long tail at document level; and **always say which you
  used** when you quote a number.
- **Completeness.** Recall assumes you labelled *every* relevant document. If a fact appears in
  three documents and you labelled one, your recall is pessimistic and your precision is
  nonsense. This is why MAP, which assumes complete labels, is fragile on hand-built sets.

## 21.4 nDCG, derived

You will be asked to explain nDCG without slides. There are four steps.

**1. Gain.** Each retrieved item has a relevance gain. Binary: 1 if relevant. Graded: 2 for
"contains the answer", 1 for "related". (The other common convention is exponential gain, `2^rel
− 1`, which is what the original paper and most search stacks use; it rewards highly-relevant
items much more aggressively. Say which you use.)

**2. Discount.** An item at rank *i* is worth `1 / log2(i + 1)`:

```
  rank    1     2     3     4     5    10
  weight  1.00  0.63  0.50  0.43  0.39  0.29
```

Rank 1 is worth 1.00; rank 5 is worth 0.43. That ratio *is* the modelling assumption: you are
asserting that a user (or a generator) gets less value from later results. For RAG the
assumption is well founded: the model reads the context in order and attends unevenly to the
middle.

**3. DCG** is the sum of discounted gains:

```
  DCG@k = Σ  gain_i / log2(i + 1)

   rank i  doc  gain  log2(i+1)  discount  contribution
        1    B     0      1.000     1.000         0.000
        2    A     1      1.585     0.631         0.631
        3    D     0      2.000     0.500         0.000
        4    C     1      2.322     0.431         0.431
        5    E     0      2.585     0.387         0.000
  DCG@5 = 1.0616
```

**4. Normalise by the ideal.** IDCG is the DCG of the best possible ordering: all relevant
items first:

```
  IDCG@5 = 1/log2(2) + 1/log2(3) = 1.0000 + 0.6309 = 1.6309
  nDCG@5 = 1.0616 / 1.6309 = 0.6509
```

That last step is what makes nDCG comparable **across questions**. A question with one relevant
document and a question with three have different maximum DCGs; dividing by each question's own
ideal puts both on a 0–1 scale so you can average them. MRR and MAP cannot express graded
relevance at all, and precision cannot express order. nDCG is the only one of the four that does
both: which is why it is the default in search, and why it is the right headline retrieval
metric for RAG.

### MAP, and the averaging trap underneath every metric

MAP is the other rank-aware metric you will be asked to derive, and it is simpler than it looks.
Average Precision for one question is the mean of `precision@i` evaluated **only at the ranks
where a relevant item appears**:

```
  ranking: B *A* D *C* E        relevant = {A, C}
  relevant at rank 2 → precision@2 = 1/2 = 0.500
  relevant at rank 4 → precision@4 = 2/4 = 0.500
  AP = (0.500 + 0.500) / |relevant| = 0.500
```

MAP is the mean of AP over questions. Dividing by `|relevant|` rather than by the number found
is what penalises *missing* documents: retrieve one of three gold documents perfectly at rank 1
and AP is 0.333, not 1.0. That also makes MAP fragile on hand-built sets: it assumes your
labels are **complete**, and an unlabelled relevant document is scored as a mistake.

The subtlety worth knowing, because it silently changes every number on your dashboard, is **how
you average across questions**:

- **Macro-averaging** computes the metric per question, then averages the per-question values.
  Every question carries equal weight. This is what `run_eval.py` and every script in this book
  does, and it is almost always what you want: each question represents a user need.
- **Micro-averaging** pools the raw counts across all questions first (total relevant retrieved
  ÷ total relevant) so questions with many gold documents dominate. A single question with 10
  gold documents outweighs five questions with one each.

On a golden set where most questions have one source and a few have three, the two can differ by
several points on identical data. Neither is wrong; quoting one while your colleague computes
the other is. State which you use, and prefer macro unless you have a specific reason: with
micro, adding one richly-labelled question silently reweights your entire benchmark.

The same choice appears in the funnel of 21.1: `P(evidence in context | document retrieved)`
averaged per question is not the same as pooling all chunks. And it appears again in generation
metrics: macro-averaging faithfulness over questions versus pooling all claims gives different
answers whenever answer lengths vary.

## 21.5 Is the improvement real?

The deadliest habit in RAG work is shipping a change because a number moved.
[`code/ch21/significance.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch21/significance.py) pairs two saved runs question by question and applies three tests.
Real output comparing k=3 against k=6:

```
metric             n   mean A   mean B    B - A              95% CI       p   verdict
recall@k          42    0.940    0.976   +0.036    [+0.000, +0.083]   0.247   no evidence
ndcg@k            42    0.954    0.975   +0.021    [+0.000, +0.048]   0.247   no evidence
precision@k       42    0.738    0.536   -0.202    [-0.258, -0.147]   0.000   SIGNIFICANT
keyword_hit       42    0.952    0.940   -0.012    [-0.071, +0.036]   1.000   no evidence
faithfulness      42    0.976    0.968   -0.008    [-0.056, +0.040]   1.000   no evidence
correct           42    0.857    0.857   +0.000    [-0.071, +0.071]   1.000   no evidence
citation_acc      38    1.000    0.974   -0.026    [-0.079, +0.000]   1.000   no evidence
```

**Exactly one row is significant, and it is the one nobody would have written a story about.**
Precision falls by 0.202 with a confidence interval nowhere near zero, because precision@k is
arithmetic on the ids: at k=6 you are showing the model twice as many chunks and the same number
are relevant. Every judged row (faithfulness, correctness, citations) has an interval straddling
zero, and `correct` is *identical* at 0.857.

This table is also a receipt. An earlier draft of Chapter 10 narrated this same comparison as
"lifts recall, cuts hallucination five-fold, costs correct rate": a causal story about the model
hedging when given more chunks. Running this script killed it: the hallucination gap rested on two
questions out of 42, and when a bug in the citation metric was fixed (Chapter 10 §10.6) the
correctness gap vanished entirely. **The narrative survived three drafts and did not survive one
significance test.** That is the whole argument for this section.

Three techniques, and why each:

**Paired bootstrap.** Resample *questions* with replacement 10,000 times, keeping each
question's two measurements together, and take the 2.5th and 97.5th percentiles of the resampled
differences. It answers "if I had drawn a different golden set of the same size, how much would
this difference move?" No normality assumption.

**Paired permutation test.** Under the null hypothesis that the two systems are the same, it is
a coin flip which one produced which measurement for a given question. Flip each pair
independently 10,000 times and count how often the shuffled difference is at least as extreme as
the observed one. With 42 questions there are 2^42 sign patterns, so 10,000 draws is a Monte Carlo
estimate rather than a full enumeration: but reporting it as `(count + 1)/(trials + 1)` keeps the
test valid at its nominal level, and nothing anywhere assumes normality.

**Pairing matters.** Some questions are hard for every configuration (q17 fails everywhere).
Comparing two independent means throws that information away. The paired view of `correct`:

```
                       B correct   B wrong
  A correct                   35         1
  A wrong                      1         5
  concordant: 40/42 questions - they carry NO information about which
  system is better. McNemar's test uses only the 2 discordant pairs:
  A-only-right = 1, B-only-right = 1, exact two-sided p = 1.000
  the questions that actually differ: ['q04', 'q37']
```

Forty of the 42 questions answer identically under both configurations. **The entire comparison
rests on two questions.** That is the honest description of most RAG A/B results on a small
golden set, and it is why the script prints the ids: go read those two rows instead of arguing
about the average.

### How many questions do you need?

```
  Planning table - minimum detectable effect on a binary metric (80% power,
  alpha 0.05), as a function of how often the two systems disagree at all:
   n questions   disagree 10%   disagree 20%   disagree 30%
            20              -              -              -
            42              -           0.20           0.25
           100           0.09           0.14           0.16
           300           0.06           0.08           0.09
          1000           0.03           0.04           0.06
```

With 42 questions you cannot reliably detect anything smaller than about 20 points. To detect
2–3 points you need hundreds to thousands of questions. And confidence intervals shrink with the
square root of n, so halving the interval costs four times the labelling:

```
      n   mean diff                95% CI    width
     10      +0.100      [+0.000, +0.250]    0.250
     20      +0.050      [+0.000, +0.125]    0.125
     42      +0.036      [+0.000, +0.083]    0.083
```

The practical escape is not always "label more questions". A metric with more resolution per
question (graded relevance, chunk-level labels, partial credit) extracts more than one bit
from each question you already have, which buys power at no labelling cost beyond the labels
themselves.

## 21.6 What to actually put on the dashboard

Six numbers, each protecting against a different failure, reported together because each is
gameable alone:

| Number | Protects against | Typical target | Gamed by |
|---|---|---|---|
| **recall@k (chunk-level)** | the answer never reaching the model | ≥ 0.90 | raising k |
| **nDCG@k** | the answer arriving buried | ≥ 0.85 | nothing much: this is the honest ranking metric |
| **end-to-end correct rate** | everything downstream of retrieval | product decision | a lenient judge |
| **faithfulness** | fabrication | ≥ 0.95 | refusing everything |
| **wrong-abstention rate** | the above being gamed | ≤ 0.05 | answering everything |
| **p95 latency and $/question** | a system nobody can afford to run | product decision | - |

Plus two process rules. Never report a difference without an interval. Never report a summary
without reading the rows that changed.

## The interview answer

> *"Your Recall@5 is 95%. Does that mean the RAG system answers 95% of questions correctly?"*

> "No. Recall@5 says that for 95% of the questions in that eval set, the labelled evidence appeared somewhere in the top 5: and usually the labels are at document level, so it may not even mean the right *chunk* was retrieved. It is the first of four stages. After retrieval the evidence still has to survive into the context window, the model still has to answer rather than refuse, and the answer still has to be right. On my own handbook benchmark, document-level hit rate was 100% and end-to-end correctness was 83%: a 17-point gap, split across chunking failures, over-refusals and synthesis errors. So I'd report recall next to end-to-end correctness, and I'd decompose the gap before touching anything."

> *"Which metric would you optimise?"*

> "Recall first, because it is the ceiling: nothing downstream can recover a fact that was never retrieved. Once recall is above about 0.9 I stop pushing it and optimise nDCG, because with the answer present the remaining retrieval problem is ranking, and the generator reads in order. I report precision alongside, since it is the cost and distraction term, and I fix it with a reranker rather than by cutting k. But the number I actually defend to a product owner is end-to-end correctness with faithfulness and wrong-abstention next to it, because the retrieval metrics are means, not ends."

> *"How do you know an improvement is real?"*

> "Paired runs on the same golden set, then a paired bootstrap confidence interval and a permutation test. When I compared k=3 with k=6 on 42 questions, the only change that survived was precision; the recall, faithfulness and correctness movements all had intervals straddling zero. The paired view showed the whole comparison rested on two questions out of 42. With around 40 questions the minimum detectable effect is roughly 20 points, so I treat anything smaller as a hypothesis, not a result: and I read the per-question diff rather than the average."

## Run it

```bash
uv run python code/ch21/metric_illusions.py                               # deterministic, no API calls
uv run python code/ch21/significance.py                                   # deterministic, reads saved runs
uv run python code/ch21/significance.py --a k3_cs800 --b k4_cs800_rerank
QDRANT_MODE=memory uv run python code/ch21/recall_vs_correct.py --limit 12   # ~1 min, ~$0.04
QDRANT_MODE=memory uv run python code/ch21/recall_vs_correct.py              # all 42, ~4 min, ~$0.12
QDRANT_MODE=memory uv run python code/ch21/chunk_level_labels.py             # cached labels, or builds 12
QDRANT_MODE=memory uv run python code/ch21/chunk_level_labels.py --build 20  # 20 LLM calls
```

`metric_illusions.py` and `significance.py` print byte-identical output on every run (seeded, no
model calls). The two scripts that call a model will vary a little: the funnel's stage-2 and
stage-4 counts move by a question or two between runs, and which questions land in `2-evidence`
versus `4-generation` can change. The shape (perfect document-level retrieval, a double-digit
gap to end-to-end correctness) does not.

## Exercises

1. Re-run `recall_vs_correct.py` at `--k 3` and `--k 8`. Which conditional probability in the
   funnel moves most? Does the stage that loses the most questions change?
2. `significance.py --a k3_cs800 --b k8_cs400_nojudge` compares configurations that differ in
   chunk size *and* k. What can you conclude, and what can you not conclude, from a comparison
   that moves two variables?
3. Extend `chunk_level_labels.py` to all 42 answerable questions and recompute the optimism
   bias. Does it grow with the number of chunks in the gold document? (Plot inflation against
   `n_candidate_chunks`.)
4. Implement exponential gain (`2^rel − 1`) in the graded nDCG of `metric_illusions.py` using
   the grades in `chunk_labels.yaml`. Which questions change rank order most?
5. Add a `--metric` flag to `significance.py` and run it on `ndcg@k` for every pair of saved
   runs in `data/eval_results/`. Build the matrix of which configurations are distinguishable at
   all with 42 questions.
6. The funnel treats "correct by luck" as a footnote. Modify `recall_vs_correct.py` to report it
   as its own row, and find a question where the answer is right despite the gold document never
   being retrieved.

## Interview questions

**Q: Does Recall@5 = 95% mean 95% of answers are correct?**
No: see the full answer above. It is stage one of four, measured against labels you chose,
usually at document level. Measured end to end on the handbook benchmark, perfect document-level
retrieval still produced 83% correctness.

**Q: Recall@k vs precision@k: which matters more?**
Recall sets the ceiling; precision controls noise, cost and distraction. Push recall until
roughly 0.9 with a generous k, then recover precision with a reranker so you can send fewer
chunks. Reporting either alone is gameable: recall@100 is nearly free, and precision@1 is
trivially maximised by returning nothing risky.

**Q: Why is nDCG better than precision for RAG?**
Precision ignores order and grades; nDCG applies a `1/log2(i+1)` discount so a relevant chunk at
rank 1 counts more than one at rank 5, supports graded relevance, and normalises by each
question's ideal ranking so questions with different numbers of relevant documents can be
averaged. The generator reads in order and attends unevenly, so order is a real quality
difference that precision cannot express.

**Q: Derive nDCG@5 for a ranking with relevant items at positions 2 and 4.**
DCG = 1/log2(3) + 1/log2(5) = 0.6309 + 0.4307 = 1.0616. IDCG puts both first: 1/log2(2) +
1/log2(3) = 1.6309. nDCG = 0.651.

**Q: Two retrievers have the same recall@5. How can one be much better?**
Because recall is a set metric. Put the relevant documents at ranks 1–2 versus ranks 4–5 and
recall is identical while MRR goes from 1.00 to 0.25 and nDCG from 1.00 to 0.50. The second
system fills the front of the context window with distractors.

**Q: When is MRR the wrong metric?**
Whenever the answer needs more than one source. MRR only looks at the first relevant hit, so a
system that retrieves 1 of 3 required documents scores a perfect 1.0. Use it for "one good chunk
is enough" lookups; use recall/MAP/nDCG for multi-source questions.

**Q: What is wrong with document-level relevance labels?**
They score a hit whenever any chunk of the right document is retrieved, so they cannot see
chunking failures: the single most common real failure. Measured on this corpus they inflate
hit rate and recall by 5 points, and the inflation grows with document length. Chunk-level
labels cost about one minute or one LLM call per question.

**Q: How do you know a change is a real improvement?**
Paired evaluation on a fixed golden set, a paired bootstrap confidence interval, and a
permutation test; report the interval, not just the delta. Check the discordant questions: most
comparisons rest on a handful of rows. With ~40 questions the minimum detectable effect is about
20 points, so small deltas are hypotheses to be tested on a larger set, not results.

**Q: How big should the golden set be?**
Big enough for the effect you need to detect. For a binary metric with 20% disagreement between
systems, 42 questions detect ~20 points, 100 detect ~14, 300 detect ~8, 1000 detect ~4.
Alternatively, increase resolution per question with graded and chunk-level labels so each
question carries more than one bit.

**Q: Your eval says +2% and the CI is [-3%, +7%]. What do you ship?**
Nothing, on that evidence. Either gather more questions, use a metric with more resolution, or
run an online A/B on a business outcome. Shipping on a 2-point offline delta is how a pipeline
accumulates changes that individually looked positive and collectively did nothing.

**Q: What is Cohen's kappa doing in a retrieval-metrics discussion?**
It belongs to the label-quality half of the problem: when humans (or a judge) produce your
relevance labels, you need their agreement corrected for chance before you trust any metric
computed on top. Raw agreement flatters a labeller who always says "relevant". [Chapter
26](26-hallucination-forensics.md) measures it for generation judges.

## Key takeaways

- Recall@k is one factor in a four-factor product. On this corpus, perfect document-level
  retrieval coexisted with 83% end-to-end correctness: a 16.7-point gap, spread across
  chunking, refusal and synthesis failures.
- Metrics are not interchangeable: identical recall with MRR spread 0.75, identical MRR with a
  third of the evidence, identical hit rate with a fluke ranking. Know which blindness each
  metric has.
- Your labels are the experiment. Document-level labels inflated hit rate and recall by 5 points
  here and hid the one genuine chunking failure; graded labels break ties that binary labels
  cannot express.
- nDCG is the honest headline ranking metric: order-aware, graded, and normalised so questions
  are comparable. Be able to derive it.
- Report differences with intervals. Comparing k=3 and k=6 on 42 questions, only the precision
  change was significant, and the whole comparison rested on two questions.

## Next

→ [Chapter 22: Vector Search Internals](22-vector-search-internals.md)

Chapter 22 goes one layer further down: what the index is actually doing when it returns those
top-k results, and what `m`, `ef_construction` and `ef_search` cost you in recall, latency and
memory.
