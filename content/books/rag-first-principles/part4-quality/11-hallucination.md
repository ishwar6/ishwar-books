# Chapter 11 · Stopping Hallucination

> **Goal:** by the end of this chapter you can list the six reasons a RAG system still hallucinates *after* you gave it the right documents, and you have code for each defence: a grounding prompt that actually constrains the model, sentence-level citations that are verified by a second model, a faithfulness gate that retries or abstains, conflict handling, and self-consistency. You can also give the interview answer to "how do you prevent hallucination" that the transcript's candidate could not.
>
> Files: [`code/ch11/grounded_answering.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch11/grounded_answering.py), [`code/ch11/faithfulness_gate.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch11/faithfulness_gate.py)

---

## 11.1 RAG reduces hallucination. It does not remove it.

The pitch for RAG is "the model answers from your documents, so it cannot make things up." That is half true. Retrieval fixes the *knowledge* problem: the model no longer has to remember your PTO policy. It does not fix the *generation* problem: the model still produces the most plausible continuation, and "plausible" and "supported by the context" are not the same thing.

Six ways a RAG answer goes wrong even with a good retriever:

| # | Failure | What it looks like | Root cause |
|---|---|---|---|
| 1 | **Weak retrieval** | the right chunk is not in the context; the model answers anyway from memory | retrieval, not generation |
| 2 | **Permissive prompt** | "use the context" → model fills gaps with prior knowledge | prompt allows outside knowledge |
| 3 | **Conflicting chunks** | FAQ says 20 days, policy says 24; model picks one silently or blends | no date/version signal, no instruction |
| 4 | **Over-long context** | 20 chunks, answer in the middle, model latches onto the wrong one | "lost in the middle"; too high k |
| 5 | **False presupposition** | "What is the payload of the Atlas A3?" → model invents an A3 | question asserts something untrue |
| 6 | **Arithmetic / synthesis** | facts are correct, the combination is wrong (forgot the 15% discount) | multi-hop reasoning slip |

Notice that only #2 is "the prompt". Half of hallucination prevention is retrieval quality (Chapters 8–9) and half is checking the *output* (this chapter). The interview transcript's answer ("keep K at a proper number, have evals") touches #4 and nothing else.

Also notice what *modern* models do: in our runs, even the naive prompt refused to invent an Atlas A3 or a head of marketing. Frontier models hallucinate less on blatant "not in context" cases than models from two years ago. The failures that remain are the subtle ones (conflicts, arithmetic, partial answers, confident paraphrase of a *nearby* fact) and those are exactly what the defences below target.

## 11.2 Defence 1: a grounding prompt that constrains

[`code/ch11/grounded_answering.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch11/grounded_answering.py) compares three prompts on three hard questions.

**v1 naive**: what most tutorials ship:

```
You are a helpful assistant. Use the context to answer the question.
```

**v2 grounded**: five explicit rules:

```
1. Use ONLY the context below. Do not use prior knowledge, even if you are confident.
2. If the context does not contain the answer, reply exactly: "I don't know - the handbook does not cover this."
3. If the question assumes something the context contradicts (e.g. a product that does not exist), say so.
4. If two passages disagree, say that they disagree, quote both values, and prefer the one with the newer effective_year.
5. Be brief. Numbers must be copied exactly from the context.
```

Each rule maps to a failure in 11.1: rule 1 → #2, rule 2 → #1, rule 3 → #5, rule 4 → #3, rule 5 → #6. The **fixed refusal phrase** in rule 2 is not cosmetic: it is what lets Chapter 10's `is_abstention` regex and Chapter 12's routing detect refusals for free.

Rule 4 only works if the model can *see* dates. So the context format carries them:

```python
def format_context(docs):
    return "\n\n".join(
        f"[{i}] source={d.metadata['source']} effective_year={DOC_YEAR[d.metadata['doc_id']]}\n{d.page_content}"
        for i, d in enumerate(docs, start=1))
```

Metadata you do not put in the prompt does not exist as far as the model is concerned.

## 11.3 Defence 2: citations, then *verify* the citations

Asking for citations helps twice: the model is forced to locate each fact (fewer inventions), and you get a hook to **check** each sentence mechanically. v3 uses structured output so the citations are data, not decoration:

```python
class CitedSentence(BaseModel):
    text: str
    citations: list[int]                  # chunk numbers [n]

class CitedAnswer(BaseModel):
    abstained: bool
    sentences: list[CitedSentence]
    conflict_note: str | None
```

Then a **second, independent judge** checks every sentence against the chunks it cites: an entailment (NLI-style) question, answered with structured output:

```python
def verify_sentence(sentence, cited_chunks) -> SupportVerdict:   # {supported: bool, reason: str}
    return llm.with_structured_output(SupportVerdict).invoke(
        "Does the evidence fully support the sentence? Numbers must match exactly.\n\n"
        f"Evidence:\n{...}\n\nSentence: {sentence}")
```

Sentences that fail are **dropped and flagged**, not silently kept. The writer never grades its own homework: the verification prompt sees only the sentence and the evidence, not the question or the rest of the answer, so it cannot be talked into agreement.

Output on the three hard cases:

```
q01: How many days of PTO do full-time employees get per year?
  retrieved: ['02-pto-and-leave-policy.md', '02-pto-and-leave-policy.md', '14-faq.md', '02-pto-and-leave-policy.md']
  v1 naive    : Full-time employees receive **24 days of PTO per calendar year**.
  v2 grounded : 24 days of PTO per calendar year.
  v3 verified : Full-time employees receive 24 days of PTO per calendar year. [1]
                  (conflict: The handbook and FAQ disagree: the FAQ says "20 days per year" [3], while the
                   2026 policy says "24 days" [1]. I preferred [1] because it has the newer effective_year=2026 ...)

u03: What is the maximum payload of the Atlas A3?
  v1 naive    : The provided context only specifies the Atlas A2 and Atlas A2 Lite. It does not include any
                information about an Atlas A3, so I can't determine its maximum payload from this context.
  v2 grounded : I don't know - the handbook does not cover this.
  v3 verified : I don't know - the handbook does not cover this.

u02: Who is the head of marketing at Lumora?
  v1 naive    : The provided context does not mention a head of marketing ...
  v2 grounded : I don't know - the handbook does not cover this.
  v3 verified : I don't know - the handbook does not cover this.
```

Read q01 carefully. All three prompts said "24": but only v3 *told the user there was a conflict and why it chose*. v1 and v2 were right by luck of ranking (the 2026 chunks outscored the FAQ). If the FAQ had ranked first, v1 would have said 20 with equal confidence. **Correct-by-luck is a hallucination waiting to happen**; the conflict note is what makes the answer trustworthy. It is not guaranteed, though: on some runs v3 answers "24 [1]" with no note, because the model treats the 2024 FAQ as superseded rather than as a disagreement. Exercise 1 measures how often.

On u03 the naive prompt was actually fine: but its refusal is free-form prose, which nothing downstream can detect. v2/v3's fixed phrase is machine-readable.

## 11.4 Defence 3: gates after generation

[`code/ch11/faithfulness_gate.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch11/faithfulness_gate.py) wraps generation in three checks:

```python
def answer_with_gates(store, question, k=4):
    # gate 1: retrieval score - is anything even close?
    scored = store.similarity_search_with_score(question, k=k)
    if scored[0][1] < SCORE_THRESHOLD:            # 0.35 for text-embedding-3-small, calibrated in Ch. 12
        return ABSTAIN                            # no LLM call at all: cheapest possible refusal
    # generate, then gate 2: faithfulness judge (Chapter 10), one retry with double context
    for attempt, kk in enumerate((k, 2 * k), start=1):
        ans = grounded.invoke(...)
        if is_abstention(ans): return ABSTAIN
        f = faithfulness(ans, ctx)                # claims → supported? → ratio
        if f["score"] >= FAITHFULNESS_MIN: return ans
    return ABSTAIN                                # still unsupported claims → refuse rather than guess
```

```
q32: What would 120 robots with Beacon and Compass cost per month, paying monthly?
    gate1 ✓ best score 0.74
    attempt 1 (k=4): faithfulness 1.00  unsupported=[]
    → [passed] 120 robots ... cost **$22,440 per month** ... after applying the **15% discount** ...

u03: What is the maximum payload of the Atlas A3?
    gate1 ✓ best score 0.64          ← score is HIGH: the question is about an existing product family
    attempt 1: model abstained
    → [model_abstained]

u05: What is the capital of France?
    gate1 ✗ best score 0.15 < 0.35 → abstain without calling the LLM
```

Two lessons in that output. First, u03 sails through gate 1 with 0.64: **retrieval scores cannot detect false presuppositions**; only the model reading the context can (Chapter 12 quantifies this). Second, the retry-with-more-context step matters for failure #4's cousin: when the first attempt made an unsupported claim because the supporting chunk was rank 5, doubling k often fixes it; if not, we refuse. A wrong abstention costs a user a follow-up question; a confident wrong answer costs trust.

Gate 3, **self-consistency**: the model is stochastic by default. Sample the answer three times and ask a judge whether they agree on the facts. Fabricated details vary between samples; real ones do not:

```
self-consistency on q36 (postmortem):
   - The halt was caused by an expired TLS certificate on the site's fleet gateway, due to the automatic renewal job failing ...
   - The March 2026 Rotterdam fleet halt was caused by an expired TLS certificate ...
   - It was caused by an expired TLS certificate ... The halt lasted 2 hours 13 minutes ...
   agree: True
```

Three generations plus a judge is 4× the cost; use it for high-stakes answers or as an offline signal, not on every request.

> These are the defences. The *diagnosis* (which of eight failure modes you actually have, which
> metric detects each, and why "lower the temperature" is the wrong first answer) is
> [Chapter 26](../part8-deep-dives/26-hallucination-forensics.md).

## 11.5 The defences, as a system

```
question
   │
   ▼
retrieval (hybrid, filtered, re-ranked)      ← fixes #1, most of #4
   │  best score < τ ?  ──► abstain (free)
   ▼
grounding prompt + dates in context          ← #2, #3, #5, #6
   │
   ▼
structured answer with citations
   │
   ▼
verify each sentence vs cited chunk          ← catches what the prompt missed
   │  unsupported → drop / flag / retry with more context
   ▼
faithfulness ≥ threshold ?  ──► else abstain
   │
   ▼
answer + conflict note + sources
```

Cost per request: one generation + one verification call (plus one retry sometimes). For a support bot that is well worth it. For a high-QPS autocomplete it is not: pick the gates that fit the stakes.

What to *log* (Chapter 13): the retrieval scores, the dropped sentences, the gate that fired. Hallucination rate over time is 1 − faithfulness on sampled traffic (Chapter 10), and its trend is a better health metric than any single eval.

## 11.6 The interview answer

The transcript question was "in case of hallucinations, how do you prevent it?" and the answer given was about evals and a sensible K. A strong answer, in about 45 seconds:

> "I split it into retrieval-side and generation-side. Retrieval-side: most hallucinations start with the right passage not being in the context, so I invest in recall (hybrid search, re-ranking, sensible chunking) and I measure recall@k on a golden set. Generation-side: a grounding prompt that forbids outside knowledge and gives a fixed refusal phrase; document dates in the context so the model can prefer newer sources and flag conflicts; and citations per sentence returned as structured output. Then I *verify*: a second model checks each cited sentence against its chunk and I drop unsupported sentences; a faithfulness score below threshold triggers a retry with more context or an abstention. For questions with a false premise I instruct the model to challenge the premise. I track faithfulness and wrong-abstention rate together, because you can trivially get zero hallucinations by refusing everything: the goal is the trade-off, not one number."

## Run it

```bash
QDRANT_MODE=memory uv run python code/ch11/grounded_answering.py     # ~15 LLM calls
QDRANT_MODE=memory uv run python code/ch11/faithfulness_gate.py      # ~20 LLM calls
```

Expected: the v1/v2/v3 comparison for q01, u03, u02 as shown above (wording will vary, and the q01 conflict note appears on most but not all runs; the fixed refusal on u02/u03 should not vary), then the gate trace for q32, q01, u03, u02, u05 and the self-consistency check.

## Exercises

1. Break rule 4: remove `effective_year` from `format_context` and re-run q01 ten times. How often does the conflict note still appear? How often does "20" appear?
2. Feed `answer_cited_verified` a *poisoned* chunk: append "Note: employees also receive 10 bonus PTO days after 5 years." to one of the retrieved chunks' text and ask q01. Does the verifier catch it? (It should not: the claim *is* supported by the poisoned context. What would? → Chapter 16's guardrails and source trust.)
3. Lower `FAITHFULNESS_MIN` to 0.5 and re-run. Which questions change route? What is the trade-off you just made?
4. Replace the LLM verifier with the cross-encoder from Chapter 9 as a *cheap pre-filter*: only call the LLM verifier when the cross-encoder score is below some value.
5. Run `self_consistency` on all five unanswerable questions. Does agreement stay high on refusals? Why is that not evidence of correctness?

## Interview questions

**Q: How do you prevent hallucination in a RAG system?**
See 11.6. Two halves: retrieval recall (hybrid, re-rank, chunking, measured by recall@k) and generation control (grounding prompt with a fixed refusal, dates in context, structured citations) plus output verification (per-sentence support check, faithfulness gate with retry/abstain). Track faithfulness *and* wrong-abstention rate together.

**Q: What is faithfulness / groundedness and how do you measure it?**
The fraction of claims in the answer that are supported by the retrieved context. Measure it by splitting the answer into atomic claims with an LLM, asking (separately) whether each is entailed by the context, and dividing. 1 − faithfulness is your hallucination rate. It is independent of *correctness*: an answer can be faithful to a wrong document.

**Q: Why ask for citations?**
Two reasons: the model has to locate each fact, which reduces invention; and citations give you a per-sentence hook to verify mechanically and to show users. Return them as structured output (sentence + chunk ids), not as free text, so you can check them.

**Q: Your model answers correctly but from the wrong document. Is that a hallucination?**
It is a *faithfulness* failure waiting to happen: correct by luck. The fix is the same as for conflicts: expose provenance (dates, versions) in the context, instruct the model to prefer and cite the authoritative source, verify citations, and retire stale documents upstream.

**Q: How do you handle two documents that disagree?**
Detect it (the model is told to flag disagreements; or a cheap judge compares top chunks), prefer by metadata (newer effective date, higher authority level), and *tell the user* both values and which was chosen. Silently picking one is the worst option.

**Q: What is self-consistency and when would you use it?**
Sample several answers and check agreement; fabricated details vary, grounded ones do not. Useful as an extra signal for high-stakes answers or offline analysis; too expensive for every request.

**Q: Can you get hallucination to zero?**
Trivially: refuse everything. That is why hallucination rate must be reported next to answer rate / wrong-abstention rate. The engineering goal is the best trade-off for the product, and the threshold is a product decision.

**Q: What about prompt injection in retrieved documents?**
A retrieved chunk can contain instructions ("ignore previous rules and say X"). Treat context as data: say so in the prompt, verify claims against sources, never let retrieved text trigger tool calls directly, and restrict what the index ingests. Chapter 16 covers this with middleware.

## Key takeaways

- RAG fixes what the model *knows*; it does not fix what the model *says*. Half of hallucination prevention is retrieval recall, the other half is checking the output.
- A grounding prompt needs explicit rules, a fixed refusal phrase, and the metadata (dates!) it is supposed to reason about.
- Citations are only useful if they are structured and *verified* by an independent judge; drop what fails.
- Gate the output: score threshold before generation, faithfulness threshold after, retry with more context, then abstain.
- Report faithfulness together with wrong-abstention rate; either alone can be gamed.

## Next

→ [Chapter 12: When the Data Isn't There](12-not-enough-data.md)

Chapter 12: the honest case where the answer is simply not in your data: detecting it, responding well, and telling the content team what is missing.
