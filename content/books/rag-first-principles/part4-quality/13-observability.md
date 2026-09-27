# Chapter 13 · Observability: Logging, Tracing, Cost and Latency

> *(The outline said "dogging"; we read it as logging/debugging/observability: the practice of being able to see what your RAG system did on any request.)*
>
> **Goal:** by the end of this chapter every request your RAG system serves leaves a structured record you can grep (query, retrieved chunk ids with scores, tokens, per-step latency, answer) and you can turn on LangSmith tracing with two environment variables. You know where the milliseconds and dollars go, what to alert on, and the three debugging recipes that resolve most "the bot said something weird" tickets.
>
> Files: [`code/ch13/traced_rag.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch13/traced_rag.py), [`code/ch13/cost_and_latency.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch13/cost_and_latency.py)

---

## 13.1 The question you will be asked every week

"Why did it answer *that*?" A RAG answer depends on a query embedding, a nearest-neighbour search, a prompt template, a stochastic model and whatever was in the index at that second. If you did not record those, you cannot answer the question: you can only reproduce and hope. Observability for RAG is one discipline: **record the inputs to every decision** so the eval judges of Chapter 10 can be run on production traffic *after the fact*.

Three layers, cheapest first:

| Layer | What | Cost | You get |
|---|---|---|---|
| structured logs | one JSON line per request with ids, scores, tokens, timings | ~0 | grep-able history, offline eval on real traffic |
| tracing | a tree of spans per request (retrieve → prompt → model → parse) | small | click-through debugging, latency waterfalls |
| metrics + alerts | aggregates over time (p95, cost/day, refusal rate) | small | knowing *before* users tell you |

## 13.2 Structured logs: the JSON line

[`code/ch13/traced_rag.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch13/traced_rag.py) writes one line per request to `data/logs/rag.jsonl`:

```python
record = {
    "ts": ..., "request_id": "feb2563f", "model": CHAT_MODEL, "k": 4,
    "question": question,
    "retrieved": [{"chunk_id": ..., "source": "02-pto-and-leave-policy.md", "score": 0.588}, ...],
    "usage": {"input_tokens": 765, "output_tokens": 19},
    "timing_ms": {"retrieve_ms": 926, "generate_ms": 843, "total_ms": 1769},
    "answer": "Full-time employees get **24 days of PTO per calendar year** [1].",
}
logger.info(json.dumps(record, ensure_ascii=False))
```

What goes in, and why each field:

- **`request_id`**: the handle users, logs and traces share. Show it in the UI ("ref: feb2563f") so a complaint arrives with it.
- **`retrieved` with `chunk_id` and `score`**: the single most useful field. 80% of bad answers are explained by looking at what was retrieved. Storing the *ids*, not the text, keeps the log small; you can re-fetch text from Qdrant by id.
- **`usage`**: from `AIMessage.usage_metadata`; this is your cost ledger.
- **`timing_ms` per step**: retrieval vs generation is the first split when something is slow.
- **`answer`**: and, if you run Chapter 11's gates, which gate fired and what was dropped.
- Not logged: the API key (never), and: depending on your data classification: the raw question if it may contain personal data (hash it, or log it to a restricted sink). The security policy in our own corpus would classify robot telemetry as Confidential; your logs inherit that.

The logging is standard `logging` with a `FileHandler` whose formatter is just `%(message)s`: the message *is* the JSON. In production the handler is stdout and your platform (Cloud Logging, Datadog, Loki) ingests it. Debugging is then:

```
grep feb2563f data/logs/rag.jsonl | python -m json.tool
```

## 13.3 Tracing with LangSmith

Logs are flat; traces are trees. LangChain and LangGraph emit spans automatically once tracing is on: no code changes:

```bash
# .env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=rag-learning
```

Every `prompt | llm` invocation then appears in the LangSmith UI as a run with its inputs, outputs, latency and token counts, nested under whatever called it. To group the retrieval step with the model call under one parent span, wrap the request function:

```python
from langsmith import traceable

@traceable(name="rag_answer", tags=["ch13"])
def rag_answer(store, llm, question, k=4): ...
```

`@traceable` is a **no-op when tracing is off**: the script prints `LangSmith tracing: off` and runs normally: so it is safe to leave in production code and flip by environment. What you get when it is on: a waterfall per request (where did the 1.7 s go?), the exact prompt the model saw (with the chunks inlined), filterable by tag/metadata, plus the ability to add a trace to a dataset with one click: the bridge from "this answer was wrong" to a new golden question.

Alternatives with the same shape: **Langfuse** (open source, self-hostable), **Arize Phoenix** (OpenTelemetry-based), **Datadog LLM Observability** (if your infra is already there), **OpenTelemetry** directly. The LangSmith callbacks are the easiest with LangChain; the concepts transfer.

## 13.4 Callbacks: capturing usage from inside the chain

Sometimes you need the numbers in code, not in a UI: to enforce a per-request token budget, or to attribute cost to a tenant. A LangChain callback handler sees every model call:

```python
class UsageHandler(BaseCallbackHandler):
    def __init__(self): self.calls = []
    def on_llm_end(self, response, **kwargs):
        for gen_list in response.generations:
            for gen in gen_list:
                usage = getattr(gen.message, "usage_metadata", None) or {}
                self.calls.append({"input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0)})

resp = (prompt | llm).invoke(inputs, config={"callbacks": [handler]})
handler.totals   # {'input_tokens': 765, 'output_tokens': 19}
```

Attach it per request via `config`, and it collects usage across *all* model calls in that chain (generation, judges, rewrites) which a single `resp.usage_metadata` cannot. Chapter 16 uses the same idea to enforce budgets in agents.

## 13.5 Where the time and money go

[`code/ch13/cost_and_latency.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch13/cost_and_latency.py) times each step separately on 12 golden questions:

```
12 questions, k=4, rerank=False, model=gpt-5.4-mini

step              p50 ms    p95 ms   share
embed_ms             512      1339     38%
search_ms              1         4      0%
generate_ms          727      1043     54%
total_ms            1342      2106

tokens/question: 736 in, 24 out   cost/question: $0.00066
     1,000 questions/day → $    0.66/day   $   19.78/month
    10,000 questions/day → $    6.59/day   $  197.82/month
   100,000 questions/day → $   65.94/day   $ 1978.23/month

same traffic (10k/day), other chat models:
   gpt-5.4-mini     $    6.59/day
   gpt-5-mini       $    2.32/day
   gpt-5.6-luna     $    1.76/day
   gpt-5.6-terra    $   17.58/day
   gpt-5.5          $   43.94/day
```

The lessons generalise well beyond this toy:

- **The vector search is free.** 1 ms at p50. People optimise HNSW parameters while 99% of latency is two HTTPS calls to a model provider. (At a billion vectors this changes (Chapter 18) but not before.)
- **Embedding the query costs as much as you'd expect a network call to cost**: ~500 ms p50, 1.3 s p95: and the p95 is pure provider variance. Cache query embeddings (identical questions repeat), or run a small local embedding model for queries if latency matters more than a point of recall.
- **Generation dominates and its p95 is the user's experience.** Stream tokens so the first word appears in ~500 ms; the total does not change but the *felt* latency does.
- **Cost is input-token dominated**: 736 in vs 24 out. So cost ∝ k × chunk size, and the cheap "mini/luna" tier is 3–25× cheaper than flagships for the same RAG answer. Use the cheap tier for retrieval-side calls (rewrites, judges, re-ranking) always; use a bigger model for the final answer only if the eval table says it helps.
- **Every re-ranker, rewrite, or judge you add in the request path adds its own p95.** A 5-stage funnel with five 1-second p95s has a multi-second p95. Run judges asynchronously *after* responding; keep the synchronous path to embed → search → generate wherever the stakes allow.

## 13.6 What to alert on

Averages hide everything; alert on tails and rates, computed from the JSON lines:

| Signal | Why it matters | Typical threshold |
|---|---|---|
| **p95 latency** per step | provider degradation shows here first | > 2× the weekly baseline for 10 min |
| **refusal rate** (fixed phrase in answers) | a jump = index broke, ingestion stopped, or a prompt regression | > 1.5× baseline |
| **mean best-retrieval score** | drift = embedding model changed, index empty, or new question mix | drops by > 0.1 |
| **tokens / request** | k or chunking changed by accident; prompt injection padding | > 2× baseline |
| **cost / day** | the bill | budget |
| **error rate** (429s, timeouts) | rate limits, outages | > 1% |
| **offline eval regression** (CI) | a code change moved faithfulness/recall | −0.03 |
| **sampled online faithfulness** | hallucination trend on real traffic | weekly downward trend |

The retrieval-score and refusal-rate alerts are RAG-specific and catch the failure that ordinary APM misses entirely: **the service is up, fast, and answering nonsense** because the index was rebuilt empty at 3 a.m.

## 13.7 Debugging recipes

**"The answer is wrong."** Find the `request_id`. Look at `retrieved` first: is the right chunk there? No → retrieval problem: re-run the query against Qdrant, check the score, check whether the chunk exists (`client.retrieve(ids=[...])`), check ingestion logs for that document. Yes → generation problem: pull the exact prompt from the trace, re-run it in isolation, check for conflicting chunks, check the grounding rules. 80/20 rule: 80% of wrong answers are retrieval.

**"It's slow."** `timing_ms` split. Retrieval slow → embedding call (network) vs search (index size, filters without payload index, `ef` too high). Generation slow → output tokens (is the model writing essays? shorten with the prompt), input tokens (k too high), provider p95 (compare with the provider status page; add a fallback model: Chapter 16's `ModelFallbackMiddleware`).

**"It's expensive."** `usage` per step per request. Input tokens ∝ k × chunk size → reduce k with a re-ranker, or chunk smaller. Many calls per request → which stage? Judges and rewrites on a cheap model; cache query embeddings and identical questions; prompt caching for the long fixed system prompt.

**"It changed and nobody changed anything."** Diff the retrieval scores and refusal rate around the time. Suspects: the embedding model provider updated (scores shift), the index was re-ingested (chunk ids changed → compare counts), the LLM was silently updated (re-run the golden set and diff the JSON).

## Run it

```bash
QDRANT_MODE=memory uv run python code/ch13/traced_rag.py              # 3 requests → data/logs/rag.jsonl
QDRANT_MODE=memory uv run python code/ch13/cost_and_latency.py --n 12 # timing + cost table
QDRANT_MODE=memory uv run python code/ch13/cost_and_latency.py --n 12 --rerank   # adds a rerank_ms row
```

Expected: the tracing status line, three request records with ids, and the log file path; then the p50/p95 breakdown and the cost projections. Absolute latencies vary with your network; the *shape* (search ≈ 0, generate > embed) should not.

## Exercises

1. Add the Chapter 11 gates to `rag_answer` and log which gate fired. Then compute the refusal rate from the log file with a 5-line script.
2. Create a free LangSmith account, set the three env vars, re-run `traced_rag.py`, and find the retrieved chunks inside the trace UI. Add one trace to a dataset.
3. Cache query embeddings in a dict keyed by the question. Re-run `cost_and_latency.py` with repeated questions. How much does p50 drop?
4. Write `alerts.py`: read `rag.jsonl`, compute p95 latency, refusal rate and mean best score over the last N lines, and print WARN when any exceeds a threshold you choose.
5. Make `UsageHandler` enforce a budget: raise if cumulative input tokens for one request exceed 4,000. Trigger it with k=30.

## Interview questions

**Q: How do you debug a RAG system that gave a bad answer in production?**
Start from the request id and the structured log: check what was retrieved (ids, scores) before anything else: most bad answers are retrieval misses. If retrieval was right, pull the exact prompt from the trace and reproduce the generation in isolation, looking for conflicts or prompt weaknesses. Then decide whether it is a data gap (Chapter 12), a chunking/retrieval issue, or a grounding issue: and add the case to the golden set.

**Q: What do you log for each RAG request?**
Request id, timestamp, model and config (k, prompt version), the question (or a hash if sensitive), retrieved chunk ids with scores and sources, token usage per model call, per-step latency, the answer, and which guard rails fired. Ids not text, so logs stay small and can be re-joined to the index. Never the API key.

**Q: How do you trace LangChain / LangGraph applications?**
Set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`: every runnable emits spans automatically. Wrap your own functions with `@traceable` to group steps under one parent. Alternatives: Langfuse, Arize Phoenix, Datadog LLM Observability, raw OpenTelemetry; the callback mechanism is the same.

**Q: Where does latency go in a RAG request?**
Two network calls to model providers: query embedding (hundreds of ms, high variance) and generation (dominant, proportional to output tokens). The vector search itself is ~1 ms at small-to-medium scale. Mitigations: cache query embeddings, stream tokens, keep the synchronous path short, run judges asynchronously.

**Q: How do you control cost?**
Cost is input-token dominated (k × chunk size), so reduce k with a re-ranker, use a cheap model tier for all retrieval-side calls (rewrites, judges), cache repeated queries and embeddings, use prompt caching for long fixed system prompts, and capture `usage_metadata` per request so you can attribute cost per tenant/feature and alert on the daily total.

**Q: What would you alert on for a RAG system that ordinary APM doesn't cover?**
Refusal rate ("I don't know" frequency), mean retrieval score drift, tokens per request, and sampled online faithfulness. These catch the "service is healthy but answering nonsense" failure (an empty or stale index, a changed embedding model, a prompt regression) which latency and error-rate alerts never see.

**Q: How do you evaluate production traffic without ground truth?**
Sample 1–5% of requests and run reference-free judges asynchronously: faithfulness (claims vs retrieved context), sufficiency, relevance. Chart them daily. Route thumbs-down and abstentions to human labelling to grow the golden set. Chapter 10's judges, Chapter 13's logs.

## Key takeaways

- Record the inputs to every decision: retrieved ids + scores, tokens, per-step timings, answer, request id. One JSON line per request answers most "why" questions with `grep`.
- Tracing (LangSmith et al.) is two env vars with LangChain; `@traceable` groups your own steps and is a no-op when off.
- Vector search is ~1 ms; the two model calls are the latency and the cost. Cache query embeddings, stream generation, keep the synchronous path short.
- Cost ∝ input tokens ∝ k × chunk size; cheap model tiers for everything except (maybe) the final answer.
- Alert on refusal rate, retrieval-score drift and sampled faithfulness: the RAG-specific failures APM cannot see.

## Next

→ [Chapter 14: LangGraph Basics: RAG as a State Machine](../part5-agentic/14-langgraph-basics.md)

Part 5: LangGraph: turning the retrieve-then-generate pipeline into a graph that can decide, loop, grade its own retrieval and try again.
