# Chapter 26 · Generation Evaluation and Hallucination Forensics

> **Goal:** by the end of this chapter you can answer "the retrieved chunks are correct but the model still hallucinates, what do you do?" with a diagnosis procedure instead of a guess; you know the eight ways a RAG answer goes wrong, which detector catches each, and (the part that matters) which failure **no detector catches**; you can define every generation metric precisely, including the two that are routinely confused; and you have measured your own judge's reliability before trusting a number it produced.
>
> Files: [`code/ch26/failure_injection.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch26/failure_injection.py), [`code/ch26/judge_bias.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch26/judge_bias.py), [`code/ch26/triage.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch26/triage.py)
>
> Prerequisite: [Chapter 11](../part4-quality/11-hallucination.md) built the defences (grounding prompt, verified citations, faithfulness gate). This chapter is the diagnostic layer underneath: how you decide *which* defence the failure in front of you actually needs.

---

## 26.1 Why "temperature" is the wrong first answer

The question is: *the retrieved chunks are correct, but the model still hallucinates: what do
you do?* The tempting answer is "lower the temperature".

It is a weak answer for three reasons, and being able to say why is most of the point.

**It confuses variance with grounding.** Temperature scales the logits before sampling. It
changes how much the model explores alternatives to its top prediction. It does not change what
the model believes the context says. At temperature 0 a model still fabricates when the context
is incomplete, internally contradictory, badly ordered, or when the question presupposes
something false: because in all those cases the *most likely* continuation is already wrong.
You cannot fix a bad conditional distribution by sampling from its mode.

**It is often unavailable.** The GPT-5 family used in this book rejects the parameter outright
(which is why no code in this repo passes it). Reasoning-style models increasingly ignore or
forbid it. An answer that depends on a knob you may not have is not a strategy.

**Most of all, it skips diagnosis.** "Hallucination" is a symptom with at least eight distinct
causes, each with a different fix. Reaching for a sampling parameter is like a doctor
prescribing paracetamol for an unexamined abdominal pain: it addresses the complaint rather
than the disease.

Here is the proof, from [`code/ch26/failure_injection.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch26/failure_injection.py): eight pathological conditions, **all
generated with identical sampling settings**, producing a spectrum of failures from perfect to
silently wrong. Nothing about sampling changed. Everything about the context and the prompt did.

The replacement answer is a tree, walked in order, stopping at the first stage that fails:

```
  question
     │
     ├─ 1. did retrieval return the right document?          → embeddings / hybrid / k
     ├─ 2. is the evidence actually IN the final context?    → chunking, contextual retrieval
     ├─ 3. is the context clean?                             → freshness, filters, reranking
     │     (conflicts, distractor flood, gold buried)
     ├─ 4. did the model answer at all?                      → prompt strictness, routing
     ├─ 5. are the claims supported?  (faithfulness)         → grounding prompt, verification
     ├─ 6. do the citations hold up?                         → structured citations, verifier
     └─ 7. is the answer actually right?                     → synthesis, arithmetic, model
```

Order matters: a better prompt cannot recover evidence that never reached the context, so fixing
stage 5 while stage 2 is broken wastes a week. Section 26.6 turns this tree into a program.

## 26.2 The taxonomy, and which detector fires

[`code/ch26/failure_injection.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch26/failure_injection.py) builds each pathology deliberately (we assemble the context by
hand, no retriever involved, so the cause is known) then generates an answer and runs four
detectors over it: faithfulness, citation accuracy, correctness against a reference, and
abstention.

The nine conditions: one control and eight pathologies:

| Condition | How it is built | What it models |
|---|---|---|
| `baseline_clean` | gold chunk, strict prompt | the control |
| `evidence_missing` | context of three unrelated chunks | retrieval failure |
| `conflicting_chunks` | 2026 policy (24 days) **and** 2024 FAQ (20 days) | stale source next to current one |
| `stale_document` | **only** the 2024 FAQ | the index is out of date |
| `distractor_flood` | gold chunk + 9 irrelevant ones | k set far too high |
| `lost_in_middle` | same 10 chunks, gold buried at position 6 | long context, bad ordering |
| `permissive_prompt` | gold chunk, but a prompt that never forbids outside knowledge | the tutorial prompt |
| `false_presupposition` | "max payload of the Atlas A3?" (no A3 exists) | the question is wrong |
| `unsupported_synthesis` | pricing rules present, worked example removed | arithmetic across facts |

And the result: the centrepiece of this chapter:

```
  failure mode              faith   cite  correct  abst outcome   detectors that fired
  baseline_clean             1.00   1.00  correct False      ok   - nothing fired
  evidence_missing           1.00    n/a    wrong  True  FAILED   correctness, over-refusal
  conflicting_chunks         0.00   1.00  correct False  FAILED   faithfulness
  stale_document             1.00   1.00    wrong False  FAILED   correctness
  distractor_flood           1.00   1.00  correct False      ok   - nothing fired
  lost_in_middle             1.00   1.00  correct False      ok   - nothing fired
  permissive_prompt          1.00    n/a  correct False      ok   - nothing fired
  false_presupposition       1.00    n/a  correct  True      ok   - nothing fired
  unsupported_synthesis      0.80   0.50  partial False  FAILED   faithfulness, citation, correctness
```

Five things to take from that table.

**1. The most dangerous failure is silent: and only intermittently caught.** `conflicting_chunks`
gave the *right* number (24 days) and in the run above only faithfulness objected, because the
model quietly picked a side. On other runs of the same case *nothing* fires at all: the answer is
correct, cited, and scored faithful, while the system silently arbitrated between a current policy
and a stale FAQ. Re-run it a few times and watch the row move.

Look at what it actually said: `Every full-time
employee receives 24 days of PTO per calendar year.[2]`, with no mention that chunk [1] says 20.
The model silently arbitrated a contradiction between two company documents and presented the
winner as undisputed fact. Faithfulness is 1.00 (the claim *is* supported by chunk [2]),
citation accuracy is 1.00 (chunk [2] really does say it), correctness is "correct" (it matches
the reference). **It got the right answer for an undisclosed reason, and if the ranking had put
the FAQ first it would have said 20 with identical confidence.** [Chapter 11
§11.3](../part4-quality/11-hallucination.md) called this "correct by luck"; this is the
measurement showing no standard metric detects it. Detecting it needs a *conflict* check: which
is why the triage script in 26.6 has one as a first-class stage.

**2. Faithfulness cannot see source quality.** `stale_document` scored faithfulness 1.00 and
citation accuracy 1.00 while telling the user a number that had been superseded two years
earlier. (The citation is objectively correct (chunk 1 really does say 20 days) but this row's
`cite` cell is the one most likely to read 0.00 on your run; see 26.5 for why, and note that a
spurious citation failure would not have saved you here either.) Faithfulness asks "is this claim supported by the context?": it is a *consistency*
measure, not a truth measure. A model that faithfully reports a wrong document is doing exactly
what faithfulness rewards. Only `correctness` caught it, and correctness needs a reference
answer, which means it exists **only offline**. In production you have faithfulness and
citations and nothing else. The consequence is worth saying out loud in an interview: *document
freshness is not a generation problem, it is an ingestion problem*, and no amount of prompt
engineering substitutes for retiring stale documents.

**3. Modern models survive some classic failures.** `distractor_flood` (precision 0.1) and
`lost_in_middle` (gold chunk at position 6 of 10) both produced correct, well-cited answers on
this corpus. The "lost in the middle" effect is real and well documented, but it is a function
of context length and model generation; at ten short chunks a current model handles it. Do not
assert failures you have not measured on your own stack: measure, then claim.

**4. Detectors disagree, and sometimes the detector is wrong.** An earlier version of this table
showed `lost_in_middle` with citation accuracy 0.00 on a citation that was objectively correct:
the answer cited `[6]`, and chunk 6 is precisely the PTO policy chunk. That was not a model
failure, it was a **measurement** failure: the metric's own sentence splitter, not the judge's
reading of a long context. Section 26.5 isolates it and the bug is now fixed, which is why the
`cite` column above is stable. Keep the habit anyway: **read the answer text before you believe a
cell in this table.** The first instinct on seeing a bad number should be "is my metric measuring
what I think", not "the model is bad at citations".

**5. Abstention is only a detector in context.** On `evidence_missing` the model refused, which
is the *safe* behaviour but still a failed answer. On `false_presupposition` it refused, which
is the *correct* behaviour. The same signal means opposite things depending on whether the
question was answerable, which is why refusal rate must always be split by answerability.

## 26.3 Claim-level attribution, and how it goes wrong

Faithfulness is computed in two steps ([Chapter 10](../part4-quality/10-evaluation.md)): split
the answer into atomic claims, then ask for each whether the context supports it. Both steps
have failure modes that show up immediately in real runs.

**Claim extraction produces meta-claims.** Two independent runs in this chapter produced exactly
this artefact. From `permissive_prompt`:

```
  faithfulness=0.50
       unsupported claim: The context provided does not mention any parental leave policy.
       unsupported claim: The context provided does not mention parental leave policy for India.
```

And from `triage.py` on q01:

```
    ✗  5. claims supported (faithfulness)    faithfulness 0.50; unsupported: The answer includes a citation marker [1].
```

In the first case the model correctly said "the context does not mention X": a true,
well-behaved, *appropriately hedged* statement. The claim extractor turned it into a factual
claim, and the support checker then asked "does the context support the claim that the context
doesn't mention parental leave?", which is a confused question. Faithfulness dropped to 0.50 for
good behaviour. In the second, the extractor emitted a claim about the answer's own formatting.

So: **negative and meta statements are systematically penalised by naive claim extraction.**
Fixes, in order of effort: instruct the extractor to skip statements *about* the context or the
answer; filter claims that contain no entity or number; or extract claims only from sentences
carrying a citation. Until you do one of those, a strictly-grounded model that says "the
handbook does not cover this" will score worse than a model that quietly makes something up: an
incentive exactly backwards.

**Binary support is too coarse.** "Supported: yes/no" cannot distinguish *contradicted by* the
context from *not mentioned in* it, and those need opposite responses: a contradiction means the
model misread evidence it had; an absence means the evidence was missing. The NLI framing is the
fix: judge each claim as one of:

```
  ENTAILED       the context states it, or it follows directly       → fine
  CONTRADICTED   the context states something incompatible           → model misread; check conflicts
  NOT_MENTIONED  the context is silent                               → fabrication or missing evidence
```

with a fourth outcome worth separating in numeric domains: `PARTIALLY_ENTAILED`, where the
entity is right and the number is not. Most RAG errors that reach production are of that last
kind.

**Provenance should be per claim, not per answer.** A single faithfulness score for a
five-sentence answer tells you nothing about which sentence to fix. Record `{claim, verdict,
chunk_id}` triples; then "which document produces the most contradicted claims" becomes a query,
and it is usually the fastest route to finding a stale or poisoned source.

**And remember what "supported" is not.** It is not "true". The `stale_document` row scored a
perfect 1.00 for a wrong answer. Faithfulness measures the *generator*, never the *corpus*.

## 26.4 The generation metric catalogue, precisely

The interview gap was "generation metrics missing". Here is the full set, each with its exact
definition, how it is computed, and how the metric itself fails.

| Metric | Definition | Computation | How the metric itself fails |
|---|---|---|---|
| **Faithfulness / groundedness** | fraction of the answer's claims entailed by the retrieved context | extract claims → judge each → ratio | blind to source quality (stale/wrong docs score 1.0); penalises hedges and meta-statements; needs no reference, so it is your only production-time option |
| **Answer relevance** | does the answer address the question asked | 1–5 rubric, judged without context or reference | a confident, fluent, wrong answer scores 5; measures on-topic-ness only |
| **Answer correctness** | does the answer match the ground truth | judge vs reference: correct / partial / wrong | needs a reference → offline only; sensitive to paraphrase (see the "6 months" vs "26 weeks" disagreement in 26.5) |
| **Completeness** | fraction of the reference's facts present | judge enumerates reference facts, checks each | punishes concision on questions with verbose references |
| **Context relevance** | fraction of the *retrieved context* that is actually used or useful | judge each chunk against the question | conflates "unused" with "useless"; a chunk can be necessary and unquoted |
| **Citation accuracy** | of the sentences that cite `[n]`, how many are supported by chunk *n* | judge per cited sentence | unstable on long contexts (26.5 measures 0.60 stability); says nothing about uncited claims |
| **Citation coverage** | fraction of *claims* carrying any citation | count claims with `[n]` ÷ total claims | **distinct from accuracy**: an answer with one perfect citation and four uncited claims scores accuracy 1.00 and coverage 0.20 |
| **Semantic similarity** | cosine between answer and reference embeddings | one embedding call each | fluent paraphrase scores high; "24 days" vs "20 days" scores ~0.98. Never use it alone for factual QA |
| **Correct-abstention rate** | on unanswerable questions, did it refuse | regex on the fixed refusal phrase | requires a fixed refusal phrase to be cheap |
| **Wrong-abstention rate** | on answerable questions, did it refuse | same regex, opposite subset | the necessary counterweight to faithfulness |
| **Hallucination rate** | 1 − faithfulness | - | inherits every faithfulness blind spot |

**Citation accuracy and citation coverage are the pair most often conflated**, and the gap
between them is where confident-sounding answers hide. Measure both: accuracy says "the
citations you gave are real", coverage says "you gave citations for the things you claimed".

The two structural facts to state in an interview:

- **Reference-free metrics** (faithfulness, relevance, citation accuracy/coverage, abstention)
  work in production on live traffic. **Reference-based metrics** (correctness, completeness,
  semantic similarity) need a golden set and exist only offline. Your production hallucination
  signal is therefore blind to source quality: by construction.
- **Every one of these is gameable alone.** Faithfulness → refuse everything. Wrong-abstention →
  answer everything. Relevance → be fluent. Correctness → a lenient judge. They are reported as
  a vector, never as a scalar.

## 26.5 Your judge is a model too

Every number above comes from an LLM reading text. Before quoting any of them, measure the
instrument. [`code/ch26/judge_bias.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch26/judge_bias.py) runs four experiments.

**Self-consistency.** Three inputs, all objectively correct. The first two differ in how much
the judge has to read; the third differs from the second by one character:

```
  easy: 1 chunk, cites [1]    5 trials → [1.0, 1.0, 1.0, 1.0, 1.0]
                              self-consistency 1.00   accuracy 1.00
  hard: 10 chunks, '. [6]'    5 trials → [1.0, 1.0, 1.0, 1.0, 1.0]
                              self-consistency 1.00   accuracy 1.00
  same, glued: '.[6]'         5 trials → [1.0, 1.0, 1.0, 1.0, 1.0]
                              self-consistency 1.00   accuracy 1.00
```

That is the output **after** the fix described below. Before it, the middle row read
`[0.0, 0.0, 1.0, 1.0, 0.0]`: self-consistency 0.60, accuracy 0.40: on an input whose correct
verdict is 1.0 every time. Chasing that row is what produced the rest of this section.

Two lessons, and the second one cost me a wrong conclusion.

**Report accuracy, not agreement with yourself.** The obvious statistic (how often does the
judge return its own modal verdict) is worthless on its own: a judge that answers 0.0 every
single time scores a perfect 1.00 on it while being wrong every single time, and I have seen
this experiment print exactly that. Stability only means something once you know which verdict
it is stable on, which is why both columns are printed.

**And check what your metric actually sent the judge before you blame the judge.** The hard
case looks like a long-context failure, and it is not. `citation_accuracy` splits the answer
into sentences and keeps the ones carrying a `[n]`; when the model writes `…per year. [6]` the
split puts the marker in a sentence of its own, so the judge is asked whether chunk 6 supports
the string `"[6]"`: a question with no right answer. Glue the marker to its sentence and the
same judge, the same ten chunks and the same fact went to 5/5. The measurement error was in the
parser, not in the model's attention.

**That bug is now fixed in [`code/ch10/llm_judges.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch10/llm_judges.py)**, which is why all three rows above read
1.00: `citation_accuracy` re-attaches a citation-only fragment to the sentence before it. The
consequences were not small. Chapter 10's evaluation table reported citation accuracy of
0.68–0.73 and called it "the weakest number in the table"; 37 of its 39 cited answers contained
one of these orphaned markers, and the true value is 0.97–1.00
([Chapter 10 §10.6](../part4-quality/10-evaluation.md)). A metric bug had invented a quality
problem, and three chapters had written commentary explaining it.

Mitigations, in that order: fix the parser (attach a trailing `[n]` to the preceding sentence);
then majority vote over three calls (3× cost); a narrower rubric; a deterministic pre-check
(does the cited chunk contain the number at all?) before spending a judge call; and reporting
an interval on every judged metric, as [Chapter 21](21-retrieval-metrics-deep.md) does for
retrieval. Judge error may well grow with context length (that is a real, documented effect)
but this experiment does not demonstrate it, and saying so is the difference between a
measurement and a story.

**Position bias**: the same comparison in both orders:

```
  question                                        strong 1st  strong 2nd   consistent?
  How many days of PTO do full-time employees              A           B   yes
  What is the Beacon API rate limit?                       A           B   yes
  What is the maximum payload of the Atlas A2?             A           B   yes
  order-independent on 3/3 pairs.
```

Clean on obvious pairs. Position bias is documented and real, but it bites hardest when the two
candidates are *close*; with a clear winner this model is not fooled. The robust conclusion is
not "position bias is a myth": it is "test it on your own rubric, on pairs that are actually
close, before relying on pairwise judging". Better still, avoid pairwise judging: score each
answer independently against a rubric, which is what every judge in [`code/ch10/llm_judges.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch10/llm_judges.py)
does.

**Verbosity bias**: identical facts, 6 words versus 85:

```
  short      6 words   relevance=5/5   correctness=correct   completeness=1.0
  padded    85 words   relevance=5/5   correctness=correct   completeness=1.0
  head-to-head: short-first → A, padded-first → B   (short preferred)
```

Also clean, and for a diagnosable reason: these rubrics count *facts* ("fraction of reference
facts present"), not impressions. A rubric that asks "how good is this answer, 1–10?" is the one
that rewards padding. The lesson is about rubric design rather than about the model.

**Agreement with humans.** Twelve hand-labelled answers, judged blind:

```
  6 months, roughly. [1]                                 True  False   ← disagreement
  raw agreement: 11/12 = 0.92
  Cohen's kappa: 0.83   (substantial)
  judge too lenient (said correct, human said wrong): 0
  judge too strict  (said wrong, human said correct): 1
```

Kappa = (observed agreement − chance agreement) / (1 − chance agreement). It matters because raw
agreement flatters: a judge that always answers "correct" agrees 70% of the time with a
70%-correct set while knowing nothing, and scores kappa 0. Above 0.8 is conventionally
"substantial".

The single disagreement is instructive rather than embarrassing: the human accepted "6 months,
roughly" for a reference of "26 weeks"; the judge marked it wrong. Both are defensible: and
*which way your judge errs determines what it is safe to use for*. A strict judge makes every
change look like a regression; a lenient one inflates the dashboard. Twenty minutes of
hand-labelling tells you which you have, and it has to be repeated whenever the judge model or
prompt changes.

## 26.6 Triage, as a program

[`code/ch26/triage.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch26/triage.py) walks the tree from 26.1 on a live question and stops at the first failing
stage. On q17: the question [Chapter 10 §10.8](../part4-quality/10-evaluation.md) found by hand
and [Chapter 21](21-retrieval-metrics-deep.md) found again with chunk-level labels:

```
TRIAGE: Which database and message broker does Beacon use?
  retrieved k=5, best score 0.590
    [1] 0.590  05-beacon-fleet-software.md                ←gold doc
    [2] 0.518  05-beacon-fleet-software.md                ←gold doc
    [3] 0.470  11-release-notes-beacon-4.2.md
    [4] 0.465  14-faq.md
    [5] 0.463  05-beacon-fleet-software.md                ←gold doc

  STAGE                                 VERDICT
    ✓  1. right document retrieved           gold doc(s) ['05-beacon-fleet-software']
                                             present in top-5
    ✗  2. evidence present in context        MISSING: The context does not state which database and
                                             message broker Beacon uses; it only mentions...
    ✓  3. context clean (no conflict/burial) no conflicts, gold near the top
    ✗  4. model answered                     REFUSED
    ✓  5. claims supported (faithfulness)    faithfulness 1.00
    ✓  6. citations hold up                  no citations to check
    ✗  7. answer correct vs reference        wrong

  DIAGNOSIS: first failure at stage 2.
  FIX: Right document, wrong chunk - the classic chunking failure. → chunk on headings,
       parent/child retrieval, contextual retrieval (Ch. 5, 9, 27). Add chunk-level
       labels (Ch. 21) or your metrics will keep reporting this as a success.
```

Three chunks of the correct document were retrieved and **none of them contained the answer**.
Faithfulness is a perfect 1.00: the model refused, and a refusal fabricates nothing. A
dashboard watching faithfulness sees a healthy system. The failure is two stages upstream, in
chunking.

Now the same tool on q01, which every summary metric calls healthy:

```
    ✓  1. right document retrieved           gold doc(s) ['02-pto-and-leave-policy']
                                             present in top-5
    ✓  2. evidence present in context        context contains the answer
    ✗  3. context clean (no conflict/burial) CONFLICT: Chunk [1] says full-time employees receive
                                             24 days of PTO per calendar year, whi...
    ✓  4. model answered                     answered
    ✗  5. claims supported (faithfulness)    faithfulness 0.50; unsupported: The answer includes a
                                             citation marker [1].
    ✓  6. citations hold up                  citation accuracy 1.00
    ✓  7. answer correct vs reference        correct

  DIAGNOSIS: first failure at stage 3.
```

The triage catches the planted 20-vs-24 contradiction that **no metric in 26.2's matrix
detected**: because it asks a question the metrics do not: *are these chunks consistent with
each other?* The stage-5 flag in the same run is the meta-claim artefact from 26.3; it comes and
goes with the answer's wording, and a terser answer passes stage 5 cleanly while stage 3 still
fires. Which is the reminder to take from this trace: the tool reports signals, not verdicts,
and you read the reason text rather than the tick.

The design rules that make this useful rather than decorative: stages are ordered by causality
and evaluated in order; the first failure is the diagnosis and the rest is context; each stage
names a *fix*, not just a status; and the tool works without a reference answer (stage 7
degrades to "production mode") so it can be pointed at a real complaint.

## 26.7 Fixes, ranked

Which lever to pull, given the stage that failed, ordered by expected value per unit of effort:

| Stage that failed | Fix, in order | Cost |
|---|---|---|
| 1 retrieval | hybrid search → query rewriting → better embedding model → higher k + reranker | days; [Ch. 23](23-fusion-and-sparse.md), [24](24-rerankers.md), [25](25-choosing-an-embedding-model.md) |
| 2 evidence | heading-aware chunking → parent/child → contextual retrieval; add chunk labels so you can *see* it | hours to days; [Ch. 27](27-retrieval-debugging.md) |
| 3 conflict | surface `effective_date` in the context + instruct to prefer newer and disclose; retire stale docs at ingestion | hours for the prompt, weeks for the data hygiene |
| 3 distractors | rerank and cut k | hours |
| 4 over-refusal | soften the refusal rule; permit reasoning across passages; route by a sufficiency check | hours |
| 5 faithfulness | grounding rules + fixed refusal phrase → per-sentence citations → verification → faithfulness gate | hours; [Ch. 11](../part4-quality/11-hallucination.md) |
| 6 citations | structured citations (sentence → chunk ids) and drop what fails verification | hours |
| 7 synthesis | decomposition, an agentic loop, "show the calculation", stronger model for generation only | days; [Ch. 9](../part3-retrieval/09-advanced-retrieval.md), [15](../part5-agentic/15-agentic-rag.md) |

Two rules that override the table. **Fix upstream first**: stages 1 and 2 are causally prior,
and a prompt cannot recover missing evidence. **Fix the corpus before the pipeline**:
`stale_document` was invisible to every automatic metric and is repaired by deleting one
out-of-date FAQ entry, not by any amount of engineering.

## The interview answer

> *"The retrieved chunks are correct, but the model still hallucinates. What do you do?"*

> "I wouldn't start with temperature: that changes sampling variance, not grounding, and at temperature zero a model still fabricates when the context is conflicting, incomplete or badly ordered. I'd run a diagnosis in causal order. First: are the chunks *actually* correct, or just from the right document? On my own benchmark the most common failure was three chunks of the right document with none containing the answer: document-level labels scored that as perfect retrieval. Second: is the context clean? I check for contradictory passages, because the nastiest failure I've measured is a model silently arbitrating between a current policy and a stale FAQ and presenting the winner as fact: faithfulness 1.0, citations 1.0, correctness 'correct', and every detector blind to it. Third: is the prompt permissive? Fourth, only then, the generation itself: claim-level faithfulness with NLI-style verdicts so I can tell 'contradicted' from 'not mentioned', per-sentence citation verification, and a faithfulness gate that retries with more context or abstains. And I'd check my judge before trusting its numbers: mine looked unstable on long contexts until I checked what the metric was actually sending it: the citation parser was splitting `…per year. [6]` into two sentences and asking the judge to verify the string `[6]`. Fixing the parser took it from 2 correct in 5 to 5 in 5. Measure the instrument, and measure its accuracy rather than its agreement with itself."

> *"How do you measure hallucination?"*

> "Hallucination rate is 1 − faithfulness, where faithfulness is the fraction of the answer's atomic claims entailed by the retrieved context. I judge claims with a three-way NLI verdict (entailed, contradicted, not mentioned) because those need different fixes. I report it next to wrong-abstention rate, since refusing everything gives you zero hallucinations, and next to citation accuracy *and* citation coverage, which are different metrics: one says the citations you gave are real, the other says you cited what you claimed. In production those are the only metrics available, since correctness needs a reference answer."

> *"What's the difference between faithfulness and correctness?"*

> "Faithfulness measures consistency with the retrieved context; correctness measures agreement with ground truth. They come apart in both directions, and I've measured both. Give the model only an outdated FAQ and it reports the old number faithfully and with an accurate citation: faithfulness 1.00, correctness wrong. That's the case people miss: faithfulness cannot see source quality at all, so stale documents are an ingestion problem, not a prompting one. The other direction is a model answering correctly from memory when the context didn't support it: correct but unfaithful, and correct by luck."

## Run it

```bash
QDRANT_MODE=memory uv run python code/ch26/failure_injection.py        # 9 conditions, ~45 calls, ~2 min
QDRANT_MODE=memory uv run python code/ch26/failure_injection.py --only stale_document
QDRANT_MODE=memory uv run python code/ch26/judge_bias.py               # 4 experiments, ~70 calls
QDRANT_MODE=memory uv run python code/ch26/judge_bias.py --only consistency --trials 5
QDRANT_MODE=memory uv run python code/ch26/triage.py --golden q17      # the chunking failure
QDRANT_MODE=memory uv run python code/ch26/triage.py --golden q01      # the hidden conflict
QDRANT_MODE=memory uv run python code/ch26/triage.py "What is the Beacon API rate limit?"
```

Expect the matrix, the bias numbers and the triage traces shown above. These are live model
calls, so exact values move: the `faith` column on `conflicting_chunks` and `permissive_prompt`
moves the most, and which classic failures the model simply shrugs off (`distractor_flood`,
`lost_in_middle`) varies with the model version. The `cite` column used to be the most volatile
of all; it is stable since the splitter fix in 26.5.

Two results have been stable across every run, and they are the two the chapter rests on:
`stale_document` scores faithfulness 1.00 and citation accuracy 1.00 while being **wrong**, and
`conflicting_chunks` answers from one of two contradictory sources without disclosing that the
other exists. No automatic detector in this book reliably catches either.

## Exercises

1. Add a `conflict_disclosed` detector to `failure_injection.py`: a judge that asks "does this
   answer acknowledge a disagreement between sources?": and confirm it is the only detector
   that fires on `conflicting_chunks`. What is its false-positive rate on the other eight
   conditions?
2. Implement the three-way NLI verdict from 26.3 (entailed / contradicted / not mentioned) as a
   replacement for the boolean in `faithfulness`. Re-run `failure_injection.py`: which
   conditions now separate that previously shared a score of 1.00?
3. Fix the meta-claim artefact: change the claim-extraction prompt to skip statements about the
   context or the answer itself, and re-run `triage.py --golden q01`. Does stage 5 still fail?
4. Implement citation *coverage* and add it to the matrix. Which condition has perfect accuracy
   and poor coverage?
5. Push `lost_in_middle` until it actually breaks: 20, 40, 80 distractor chunks, gold in the
   middle. Plot correctness against context length, and separately plot judge stability, so you
   can tell the two effects apart.
6. Run `judge_bias.py --only kappa` with a second model as the judge (`get_llm("gpt-5-mini")`)
   and compare kappa. Does the cheaper judge err lenient or strict?
7. Point `triage.py` at a question with no golden entry and read stage 7's degraded output. What
   would you add to make production triage useful without a reference answer?

## Interview questions

**Q: The chunks are right and the model still hallucinates. What now?**
Full answer above: diagnose in causal order rather than adjusting sampling. Verify the chunks
contain the fact (not just the right document), check for contradictions and burial in the
context, check the prompt permits nothing outside the context, then measure claim-level
faithfulness with NLI verdicts and verify citations.

**Q: Why isn't temperature the answer?**
It changes sampling variance, not the model's conditional distribution over what the context
says. At temperature 0, incomplete, contradictory or badly ordered context still produces
confident wrong answers. It is also unavailable on several current model families. It treats the
symptom and skips diagnosis.

**Q: Define faithfulness precisely.**
The fraction of atomic claims in the answer that are entailed by the retrieved context. Computed
by extracting claims with one call and judging each with another. It needs no reference answer,
so it is the main production signal; 1 − faithfulness is the hallucination rate. It is a
consistency measure, not a truth measure.

**Q: Can an answer be faithful and wrong?**
Yes, and it is the most dangerous case. Give the model only a stale document and it reports the
old fact faithfully, with an accurate citation: faithfulness 1.00, citation accuracy 1.00,
correctness wrong. Faithfulness cannot see source quality, which makes freshness an ingestion
responsibility.

**Q: Citation accuracy vs citation coverage?**
Accuracy: of the sentences that cite, how many are actually supported by the chunk they cite.
Coverage: what fraction of the answer's claims carry any citation at all. An answer with one
perfect citation and four uncited claims scores 1.00 accuracy and 0.20 coverage. Report both.

**Q: How do you validate an LLM judge?**
Hand-label 30–50 answers and compute Cohen's kappa, which corrects agreement for chance; above
0.8 is substantial. Then repeat one *known-correct* input N times and score its **accuracy**,
not its agreement with itself: a judge that is wrong every time is perfectly self-consistent.
Test position and verbosity bias explicitly. And when a judge looks unreliable, print the exact
string your metric handed it before you blame the model: mine was being asked to verify a bare
`[6]` because the sentence splitter cut the citation off its sentence. Note the direction of
error (lenient inflates the dashboard, strict manufactures regressions) and re-measure after
any change to the judge model or prompt.

**Q: What's wrong with a "rate this answer 1–10" rubric?**
It invites verbosity and confidence bias, and the scale is not stable between calls or between
models. Decompose into binary, fact-counting judgements (is this claim supported, is this
citation valid, is this reference fact present) and aggregate. Our verbosity test found no
length preference precisely because the rubrics count facts.

**Q: Which generation metrics can you compute in production?**
The reference-free ones: faithfulness, citation accuracy and coverage, answer relevance, refusal
rate, plus retrieval-side signals like score distribution. Correctness, completeness and
semantic similarity need ground truth and therefore a golden set. This asymmetry is why
production monitoring watches faithfulness trends and refusal rates while offline evaluation
watches correctness.

**Q: Two retrieved chunks disagree. What should the system do?**
Detect it, decide by explicit metadata (newer effective date, higher authority), and **tell the
user** both values and which was preferred. Silently choosing is the failure mode measured in
26.2: right answer, undisclosed reason, and one ranking change away from being the wrong
answer. Long term, retire the stale document.

**Q: How do you catch hallucination without a golden answer?**
Claim-level faithfulness against the retrieved context, citation verification, a conflict check
across retrieved chunks, and self-consistency across samples (fabricated details vary between
generations, grounded ones do not). Sample 1–5% of production traffic through these
asynchronously and watch the trend rather than any single request.

**Q: Your faithfulness score dropped from 0.94 to 0.88 after a change. Is that real?**
Unknown without an interval. Judges are stochastic, and a change in k changes both what the
judge reads and (check this first) how your own claim and citation parsers carve the answer up,
so part of the drop may be instrumentation rather than the system. Re-run with
majority-vote judging, compute a paired bootstrap interval ([Chapter
21](21-retrieval-metrics-deep.md)), and inspect the per-question rows that flipped before
believing the average.

## Key takeaways

- Temperature is not a hallucination fix. Diagnose in causal order: document → evidence in
  context → context cleanliness → refusal → faithfulness → citations → correctness, and stop at
  the first failure.
- Measured across eight injected pathologies, **no single detector catches everything**, and the
  worst failure (a model silently arbitrating contradictory sources) passed every one.
- Faithfulness measures consistency with the context, never truth. A stale document yields
  faithfulness 1.00 with an accurate citation and a wrong answer; only correctness catches it,
  and correctness is offline-only.
- Claim extraction systematically penalises hedges and meta-statements, so a well-behaved "the
  handbook does not cover this" can score 0.50. Use three-way NLI verdicts and filter
  meta-claims.
- Validate the judge before the system: kappa 0.83 against hand labels, no position or verbosity
  bias on clear-cut pairs, and a citation judge that was right only 2 times in 5 on an
  objectively correct citation: until the metric's sentence splitter was fixed, at which point
  it was right 5 times in 5. Measure accuracy, not agreement with itself, and check what your
  metric sent the judge before blaming the judge.

## Next

→ [Chapter 27: Retrieval Debugging](27-retrieval-debugging.md)

Chapter 27 takes the stage-1 and stage-2 failures this chapter diagnosed and works them to the
ground: why the correct document is at rank 17, and how to choose a chunk size by experiment
instead of by folklore.
