# Chapter 12 · When the Data Isn't There

> **Goal:** by the end of this chapter you can tell, with numbers, how well retrieval scores separate answerable from unanswerable questions (badly), know the two extra signals that work, and have a router that abstains fast, asks a clarifying question, or hands off: plus a coverage report that turns "the bot doesn't know" into a to-do list for the people who own the documents.
>
> Files: [`code/ch12/coverage_report.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch12/coverage_report.py), [`code/ch12/abstain_and_fallback.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch12/abstain_and_fallback.py)

---

## 12.1 The most common RAG failure nobody measures

Every RAG demo is built on questions the corpus can answer. Every RAG *product* gets questions it cannot: the policy that was never written down, the product that does not exist, the thing that lives in a different system, the general-knowledge question the user asked because the chat box was there.

A system that cannot say "I don't know" turns each of these into a hallucination. A system that says it too often is useless. Getting this right is a three-part problem:

1. **Detect** that the context does not contain the answer.
2. **Respond** appropriately: abstain, clarify, or fall back.
3. **Close the loop**: tell the content owners what is missing, so next month it *is* there.

## 12.2 Detecting missing data

### Signal 1: retrieval score · weaker than you think

Intuition says an unanswerable question should have a low best-match score. `coverage_report.py` tests that on the golden set:

```
best-chunk cosine score - answerable (n=42):  min 0.38  median 0.57  max 0.74
best-chunk cosine score - unanswerable (n=5): [0.15, 0.48, 0.58, 0.62, 0.64]
   u01 0.58   What was Lumora's revenue in 2025?
   u02 0.62   Who is the head of marketing at Lumora?
   u03 0.64   What is the maximum payload of the Atlas A3?
   u04 0.48   How do I configure the Kafka retention period in Beacon?
   u05 0.15   What is the capital of France?
   lowest answerable:
   q07 0.38   When can I fly business class?
   q25 0.40   What is the annual learning budget?
```

Four of the five unanswerable questions score *higher* than the hardest answerable ones. Of course they do: "Atlas A3 payload" is very similar to the Atlas A2 spec sheet; "head of marketing at Lumora" is very similar to the company overview. **Cosine similarity measures topical closeness, not the presence of an answer.**

The threshold sweep makes it concrete:

```
threshold sweep (predict unanswerable if best score < t):
     t   abstain-when-should   false-abstain  accuracy
  0.25                   1/5            0/42      0.91
  0.35                   1/5            0/42      0.91
  0.40                   1/5            2/42      0.87
  0.50                   2/5           12/42      0.68
→ best threshold ≈ 0.25 (accuracy 0.91)        ← 0.25, 0.30 and 0.35 tie; anything in that band works
```

A score threshold catches exactly one category: **off-topic** questions (u05). It is still worth having (it is free and saves an LLM call) but set it low (≈0.30–0.35 for `text-embedding-3-small`; recalibrate for every embedding model, the scales differ) and expect it to do one job only.

### Signal 2: the sufficiency judge

The signal that works is reading. One structured call: *does this context contain enough to answer this question?*

```python
class Sufficiency(BaseModel):
    sufficient: bool
    missing: str        # what would be needed
```

```
sufficiency judge on unanswerable + invented questions:
   [GAP] What was Lumora's revenue in 2025?                        → 2025 revenue
   [GAP] What is the maximum payload of the Atlas A3?              → Atlas A3 payload specification
   [GAP] How do I configure the Kafka retention period in Beacon?  → Kafka retention configuration details
   [GAP] How do I reset my Okta password?                          → Okta password reset procedure
   [GAP] How many vacation days do interns get?                    → intern vacation policy
```

11 of 11 probes correctly flagged, each with a *named* gap. This is the same mechanism as the grounded prompt's rule 2 (Chapter 11), pulled out into a separate step so you can act on it *before* generating and log it *as data*.

### Signal 3: disagreement

If three sampled answers disagree (Chapter 11's self-consistency), or the LLM re-ranker (Chapter 9) scores every candidate 0, the context is probably not answering the question. Cheap to add when you already run those components.

| Signal | Cost | Catches | Misses |
|---|---|---|---|
| best score < τ | free | off-topic questions | on-topic questions with no answer |
| sufficiency judge | 1 small LLM call | almost everything | very subtle partial coverage |
| re-ranker all-zero | free if you re-rank | same as judge, less precise | - |
| self-consistency disagreement | 3× generation | fabricated specifics | consistent wrong refusals |

## 12.3 Responding well

`abstain_and_fallback.py` routes each question through, in order:

```python
def answer_or_fallback(store, question, k=4):
    scored = store.similarity_search_with_score(question, k=k)
    if scored[0][1] < HOPELESS:                   # 1. off-topic → abstain fast, no LLM cost
        return {"route": "abstain_fast", ...}
    t = triage_chain.invoke(...)                   # 2. sufficient? ambiguous? what's missing?
    if t.sufficient:
        return {"route": "answer", "reply": grounded.invoke(...)}
    if t.ambiguous and t.clarifying_question:      # 3a. ask ONE clarifying question
        return {"route": "clarify", ...}
    extra = secondary_source(question)             # 3b. another index / web tool / ticket system
    if extra: return {"route": "fallback_source", ...}
    return {"route": "handoff", ...}               # 3c. honest hand-off + logged gap
```

```
Q: What is the hotel cap per night for travel in Europe?
   route=answer           best_score=0.55   → €180 per night before taxes [1]

Q: What was Lumora's revenue in 2025?
   route=handoff          best_score=0.58   → The handbook doesn't cover this (A source stating Lumora's 2025
                                              revenue). I've noted it for the content team; for now please ask ...

Q: What is the capital of France?
   route=abstain_fast     best_score=0.15   → This doesn't look like something the handbook covers ...

Q: What does the policy say about carry-over?
   route=answer           best_score=0.36   → Up to 5 unused PTO days may be carried over ...
Q: How much is the stipend?
   route=answer           best_score=0.40   → The one-time home office stipend is ₹15,000 ...
```

The last two were meant to trigger **clarification** ("which policy?", "home-office stipend or learning budget?") and did not: the triage judge decided the top chunks made the intent clear enough and answered. That is a real finding, not a bug to hide: **LLM judges under-detect ambiguity**, because the retrieved context resolves it for them. If your product needs clarifying questions, add a structural signal: e.g. "top-k chunks come from ≥2 documents with different `category`": rather than trusting the judge alone. And be careful what you wish for: a bot that asks "which stipend?" when 95% of people mean the home-office one is more annoying than one that answers and adds "(if you meant the learning budget, that is ₹60,000)".

Design rules for the abstention message itself:

- **Say what you do know.** "The handbook covers PTO, sick leave and parental leave, but not intern leave" beats "I don't know".
- **Route, don't dead-end.** Name the channel or person.
- **Never fake certainty about the negative.** "The handbook does not cover this" is honest; "Lumora has no dress code" is a hallucination in a refusal costume.
- **Log it** with the `missing` field: that is the input to 12.4.

## 12.4 The coverage report

The unanswerable questions are the cheapest product research you will ever get: real users telling you exactly which document to write next. `coverage_report.py` produces `data/reports/gaps.md`:

```
# Coverage gaps report
11 of 11 probed questions have no sufficient support in the handbook.

## Missing topics
- Lumora company questions
- Atlas A2/A3 specs
- Beacon configuration
- HR and payroll policies
- General knowledge

| Question | What is missing | Nearest existing docs |
| How do I reset my Okta password? | Okta password reset procedure | 07-onboarding-guide.md, 06-security-policy.md |
| How many vacation days do interns get? | intern vacation policy | 02-pto-and-leave-policy.md, ... |
```

In production the input is the log of `handoff`/`abstain` routes (Chapter 13), clustered weekly. "Nearest existing docs" tells the writer *where* the new paragraph belongs. The "General knowledge" cluster tells you to add a scope note to the UI. This loop (measure gaps → write docs → re-index → gaps shrink) is the single highest-leverage activity in a RAG product after the first month, and almost nobody does it.

## 12.5 Small-corpus and cold-start tactics

When the corpus is *thin* rather than *missing*:

- **Synthetic augmentation** (Chapter 7): generate Q/A pairs or restatements per chunk and index them alongside: more "surface area" for questions to land on. Label them so you can exclude them from citations.
- **Query expansion** (Chapter 9): multi-query and HyDE make a small corpus behave as if the questions were phrased the way the documents are.
- **Lower k, higher precision**: with few documents, k=8 is mostly noise; k=3 with a re-ranker.
- **Cold start (zero documents)**: do not launch RAG. Launch a scoped assistant that collects questions, run the coverage report on them, write the top-20 documents, *then* turn on retrieval. Two weeks of question logs beat two months of guessing what to write.

## Run it

```bash
QDRANT_MODE=memory uv run python code/ch12/coverage_report.py        # ~13 LLM calls, writes data/reports/gaps.md
QDRANT_MODE=memory uv run python code/ch12/abstain_and_fallback.py   # ~8 LLM calls
```

Expected: the two score distributions and the threshold sweep (numbers will match closely: embeddings are deterministic), the sufficiency verdicts, the gaps report path, then the five routing decisions.

## Exercises

1. Add 5 more invented questions to `INVENTED_USER_QUESTIONS` that you think *are* answerable. Does the sufficiency judge agree? Note any disagreement and decide who is right.
2. Implement the structural ambiguity signal: in `answer_or_fallback`, if the top-4 chunks span ≥2 `doc_id`s from different topics, force `ambiguous=True`. Re-run the stipend question.
3. Make `secondary_source` real: index [`data/handbook/14-faq.md`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/data/handbook/14-faq.md) alone in a second collection and fall back to it. (Then argue why this particular fallback is a bad idea given q01.)
4. Re-calibrate `HOPELESS` for `text-embedding-3-large`: set `RAG_EMBED_MODEL` in `.env`, rebuild, re-run the sweep. Did the scale move?
5. Write the abstention message for u03 by hand, following the four design rules. Then make the prompt produce it.

## Interview questions

**Q: What do you do when RAG has no relevant documents for a question?**
Detect it: a low retrieval score catches off-topic questions, a sufficiency judge (one structured LLM call: "does this context answer this?") catches on-topic questions with no answer. Then route: abstain with an honest, specific message and a pointer to the right channel; ask one clarifying question if the question is genuinely ambiguous; or fall back to a secondary source or a human. Log every abstention with what was missing, and feed that to the content owners as a coverage report.

**Q: Can you use the similarity score to decide whether to answer?**
Only for off-topic detection. Similarity measures topical closeness, not answer presence: "Atlas A3 payload" scores 0.64 against the Atlas A2 spec even though the A3 does not exist. Use a low threshold to skip obviously unrelated questions cheaply, and a sufficiency judge for the rest.

**Q: How do you calibrate that threshold?**
With labelled data: score the best chunk for answerable and unanswerable questions from your golden set, sweep the threshold, pick the value that maximises accuracy (or that keeps false abstentions under a budget). Re-calibrate whenever the embedding model changes: scales are not comparable across models.

**Q: How do you decide between abstaining and asking a clarifying question?**
Clarify only when the ambiguity would change the answer *and* the context does not resolve it; otherwise answer the most likely reading and mention the alternative. LLM judges under-detect ambiguity because the retrieved context disambiguates for them; add a structural signal (top hits from different topics) if clarification matters to your product.

**Q: How do you improve a RAG system whose corpus is too small?**
Find out what is missing (coverage report from abstention logs), write or ingest those documents, and meanwhile squeeze more from what you have: synthetic Q/A augmentation, multi-query/HyDE, tighter k with re-ranking. Do not launch retrieval on an empty corpus: collect questions first.

**Q: How would you measure "I don't know" behaviour?**
Two rates on a golden set that includes unanswerable questions: correct-abstention rate (on unanswerables) and wrong-abstention rate (on answerables). Report them together with faithfulness; moving one usually moves another.

## Key takeaways

- Unanswerable questions are normal traffic; a system that cannot refuse hallucinates on all of them.
- Retrieval scores detect off-topic questions only; use a low threshold as a free first gate, and a sufficiency judge for the real decision.
- Respond by routing: abstain fast, clarify when genuinely ambiguous, fall back to another source, hand off: and always say what is *not* covered without asserting the negative.
- Every abstention is a data point; a weekly coverage report is the highest-leverage RAG improvement after launch.
- Small corpus: augment synthetically, expand queries, lower k; zero corpus: collect questions before you retrieve.

## Next

→ [Chapter 13: Observability: Logging, Tracing, Cost and Latency](13-observability.md)

Chapter 13: logging, tracing, cost and latency: seeing what your RAG system actually did on every request.
