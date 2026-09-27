# Chapter 16 · Multi-Agent Systems and Guard Rails

> **Goal:** you can explain *why* a planner / retriever / analyst loop runs away and burns
> tokens, fix it with termination criteria, turn caps and token budgets; force an agent to
> retrieve before it answers; secure a RAG system (PII, role-based access enforced in the
> vector search); and make a chain of agents survive a failing member. Every one of these
> was an interview question.

---

## 16.1 The scenario: three agents in a loop

> *"We have a planner agent, a retrieval agent and an analysis agent. In production they
> enter a loop, repeatedly calling each other, and token usage explodes. Why does this
> happen and how do you fix it?"*

**Why it happens.** Nobody owns termination. Each agent's job is phrased as "find what is
missing": the planner always proposes one more search, the analyst always finds one more
gap, and the graph has a cycle with no exit except LangGraph's `recursion_limit`: a
safety net, not a design. Three things make it worse: (1) every lap *appends* to the shared
context, so each call is bigger than the last (tokens grow quadratically with laps), (2)
agents re-run queries they already ran because nothing records what was done, (3) there is
no budget anyone checks.

[`code/ch16/planner_retriever_analyst.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch16/planner_retriever_analyst.py) builds it both ways. The runaway version:

```python
b.add_edge(START, "planner")
b.add_conditional_edges("planner", route_after_planner)   # END only if planner says done
b.add_edge("retriever", "analyst")
b.add_edge("analyst", "planner")                           # unconditional cycle
```

with a planner prompted to be "never satisfied" (an exaggeration of the real failure: in
practice the prompt says "be thorough" and the model obliges). Run with
`{"recursion_limit": 12}`; it stops only when LangGraph throws `GraphRecursionError`,
having spent tokens on every lap.

**The fix is layered: you need all of them:**

| Layer | Mechanism | In the code |
|---|---|---|
| Explicit termination | a `done` flag set by the *analyst* (the consumer of the work), not the planner | `Analysis.sufficient` → `done` |
| Turn cap | `max_turns` counter in state, checked by the routing edge | `MAX_TURNS = 4` |
| Token budget | accumulate `usage_metadata.total_tokens` into state with an `operator.add` reducer; stop when over | `TOKEN_BUDGET`, `tokens: Annotated[int, operator.add]` |
| Orchestrator validation | a node that checks the proposed action *before* executing it (reject repeated queries) | `validate_query` |
| Context control | keep only the last N findings in the planner prompt; summarise older ones | `findings[-4:]`; `SummarizationMiddleware` for agents |
| Hard cap | `recursion_limit` in the config; with `create_agent`, `ModelCallLimitMiddleware` / `ToolCallLimitMiddleware` | both |

```python
def route_after_analyst_guarded(state) -> Literal["planner", "__end__"]:
    if state["done"]:                   return END
    if state["turns"] >= MAX_TURNS:     return END   # log it
    if state["tokens"] >= TOKEN_BUDGET: return END   # log it
    return "planner"
```

The pattern to say out loud: **the agent proposes, the orchestrator decides.** Each agent
returns a *proposal* (next query, "I think we are done"); a deterministic orchestrator node
or edge validates it against state (budget, turns, repeats) and only then executes. The
LLM never holds the loop condition.

## 16.2 Enforcing RAG-first behaviour

> *"If an agent answers from its own knowledge instead of invoking the retrieval agent, how
> do you enforce RAG-first behaviour?"*

Prompting ("always search first") is the weakest lever: it works most of the time and
fails silently the rest. [`code/ch16/guardrails.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch16/guardrails.py) shows five stronger ones, in order of
strength:

**(a) Make retrieval a fixed node, not a tool.** In a `StateGraph`, `START → retrieve →
generate`. The model never gets the option. This is Chapter 14's graph; use it when every
question needs retrieval.

**(b) Force the first tool call.** With `create_agent`, a `wrap_model_call` middleware
overrides `tool_choice` until a `ToolMessage` exists:

```python
@wrap_model_call
def force_first_search(request: ModelRequest, handler):
    if not any(isinstance(m, ToolMessage) for m in request.messages):
        request = request.override(tool_choice="retrieve_handbook")
    return handler(request)
```

`tool_choice` is enforced by the provider's API, not by persuasion: the first response
*must* be a call to `retrieve_handbook`.

**(c) Reject ungrounded answers after the fact.** An `after_model` hook that inspects the
answer; if the model answered without any retrieval in the history, it appends a
correction and jumps back to the model:

```python
@after_model
@hook_config(can_jump_to=["model"])
def reject_ungrounded(state: AgentState, runtime):
    last = state["messages"][-1]
    used_tool = any(isinstance(m, ToolMessage) for m in state["messages"])
    if isinstance(last, AIMessage) and not last.tool_calls and not used_tool:
        return {"messages": [HumanMessage("You must call retrieve_handbook before answering.")],
                "jump_to": "model"}
```

**(d) A faithfulness gate as the last step.** Even after retrieving, the model may add
facts from memory. `faithfulness_gate(answer, context)` asks a structured-output judge
whether every claim is supported by the retrieved context and refuses otherwise. This is
Chapter 11's check, placed as a node.

**(e) Measure it.** The script ends with a citation-rate eval: % of golden answers that
contain at least one `[n]` citation. Track that number in CI; a regression means one of the
layers above broke.

## 16.3 Securing a RAG system

> *"How would you secure sensitive information in a RAG system?"*

Think in three places: **before the index**, **inside retrieval**, **around the model**.

**Before the index.** Classify documents at ingest (the handbook's `Confidential` price
book), store the classification and allowed roles as *payload* on every chunk, redact PII
you do not need (emails, phone numbers) before embedding: once it is in a vector store it is
retrievable. Never index secrets.

**Inside retrieval: enforce access in the vector search, not in the prompt.**

```python
@tool
def retrieve_handbook(query: str, runtime: ToolRuntime[UserContext]) -> str:
    denied = ROLE_DENY[runtime.context.role]
    flt = models.Filter(must_not=[models.FieldCondition(
        key="metadata.doc_id", match=models.MatchAny(any=sorted(denied)))]) if denied else None
    return format_docs(store.similarity_search(query, k=4, filter=flt))

agent = create_agent(..., context_schema=UserContext)
agent.invoke({"messages": [...]}, context=UserContext(role="employee"))
```

The caller's identity arrives through `context` (from your JWT / session), reaches the
tool via `ToolRuntime`, and becomes a Qdrant filter. Chunks the user may not see are never
retrieved, so they cannot leak through the model, the logs or a prompt injection. In
Chapter 18 this generalises to multitenancy.

**Around the model.** `PIIMiddleware("email", strategy="redact", apply_to_input=True)`
strips emails from user input before it reaches the provider (strategies: `block`, `redact`,
`mask`, `hash`; types include `credit_card`, `ip`, `url`, or a custom regex). Add
`apply_to_output=True` to scrub model output. Treat retrieved text as untrusted (prompt
injection), log prompts with secrets masked, keep API keys in a secret manager, and put
audit logging on *who asked what and which chunks were returned*.

## 16.4 When an agent in the chain fails

> *"An agent in a multi-agent chain fails. What do you do?"*

Distinguish **transient** failures (rate limit, timeout, 5xx) from **logical** ones (bad
output, tool error). For transient: retries with exponential backoff and jitter:
`ModelRetryMiddleware(max_retries=2, backoff_factor=2.0, initial_delay=0.5)` and
`ToolRetryMiddleware`, plus transport timeouts on the model itself
(`get_llm(timeout=30, max_retries=2)`), plus a fallback model
(`ModelFallbackMiddleware("openai:gpt-5-mini")`). For logical: validate outputs with
structured schemas, make tools **idempotent** (a retried "create ticket" must not create
two), and let the orchestrator route around a failed agent (skip the analyst, return partial
findings with a flag) rather than crash the run. Trace every step (Chapter 13) so you can
see *which* agent failed with *what* input.

## 16.5 A note on MCP vs REST

> *"You have used MCP tools. How do they differ from a REST API?"*

MCP (Model Context Protocol) is a **standard protocol for exposing tools, resources and
prompts to LLM applications**. An MCP *server* wraps something (a database, Notion, Slack,
a REST API) and advertises tools with JSON schemas. An MCP *client* (your agent) connects,
*discovers* the tools at runtime, and the *model* chooses which to call and with what
arguments; the client executes them over a stateful session (stdio or HTTP). A REST API is
one service's specific contract that a *developer* wires in by hand. So: REST is what the
server calls internally; MCP is the uniform, self-describing, LLM-facing layer on top:
"USB-C for tools". In LangChain, `langchain-mcp-adapters` turns MCP tools into `@tool`
objects you pass to `create_agent`.

## Run it

```bash
QDRANT_MODE=memory uv run python code/ch16/planner_retriever_analyst.py
QDRANT_MODE=memory uv run python code/ch16/guardrails.py
```

Observed (trimmed):

```
=== RUNAWAY: no termination criteria (recursion_limit=12) ===
  [planner] turn=1 done=False next='Search the handbook for Gold-tier support entitlements ...'
  [analyst] sufficient=True
  [planner] turn=2 done=False next='Search the handbook for any Gold-tier inclusions beyond ...'
  [analyst] sufficient=True
  [planner] turn=3 done=False next='... exclusions/limitations ... taxes, shipping ...'
  [analyst] sufficient=True
  [planner] turn=4 done=False next='... whether Gold support is 24×7 for all severities ...'
  [analyst] sufficient=True
STOPPED by GraphRecursionError: Recursion limit of 12 reached without hitting a stop condition ...

=== FIXED: done flag + max_turns + token budget (recursion_limit=25) ===
  [planner] turn=1 done=False next='Atlas A2 pricing and Gold-tier P1 support coverage'
  [analyst] sufficient=True
finished: turns=1 tokens=930
REPORT: Gold-tier customers get 24×7 support, with a 30-minute P1 response target and a 6-hour
P1 resolution target ... An Atlas A2 robot costs $48,000 list price (USD). Sources: 09-support-sla.md and 10-pricing-and-plans.md.
```

Look at the runaway run: the *analyst said "sufficient" on lap one* and the planner kept
going anyway, each proposed query longer and more speculative than the last. That is the
production failure in miniature.

```
--- PII redaction: the email never reaches the model ---
A: The hotel cap in Europe is **€180**. [1]

--- RBAC: same question, two roles ---
employee -> I don't know based on the handbook.
sales    -> The Atlas A2 robot costs **$48,000** list price in USD.[2]

--- citation-rate eval on a golden subset ---
  q01 cited=True | Full-time employees receive **24 days of PTO per calendar year**.[1]
  ...
citation rate: 8/8 = 100%
```

## Exercises

1. Set `TOKEN_BUDGET = 500` and watch the budget guard fire before `MAX_TURNS`. Log which
   guard stopped the run into state.
2. Replace the planner/analyst functions with two `create_agent` sub-agents, each with
   `ModelCallLimitMiddleware(run_limit=2)`, and wire them into the same supervisor graph.
3. Remove `force_first_search` and change the system prompt to "answer from your own
   knowledge". Show `reject_ungrounded` catching it (look for the `[after_model]` line).
4. Add `PIIMiddleware("credit_card", strategy="mask")` and pass a fake card number. Then set
   `apply_to_output=True` on the email rule and make the model quote an email from a chunk.
5. Make the RBAC deny list come from chunk payload instead of a Python dict: add
   `classification` to metadata in [`ragbook/index.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/ragbook/index.py) and filter on
   `metadata.classification`.

## Interview questions

**Q: Why do multi-agent systems loop, and how do you fix it?**
Because termination is left to the agents, whose prompts reward finding "more", and the
graph has a cycle with no state-based exit. Fix with an explicit `done` criterion owned by
the consumer of the work, a max-turns counter, a per-run token budget accumulated from
`usage_metadata`, an orchestrator that validates each proposed action (no repeats), context
trimming/summarisation, and a hard `recursion_limit` / call-limit middleware as the last
line.

**Q: How do you enforce RAG-first?**
Strongest to weakest: make retrieval a fixed graph node before generation; force the first
model call with `tool_choice`; reject answers produced without a `ToolMessage` via an
`after_model` hook and jump back; add a faithfulness gate; and measure citation rate in
eval. Prompting alone is not enforcement.

**Q: How do you secure sensitive data in RAG?**
Classify and redact at ingest; store access metadata as payload; enforce access as a filter
*inside* the vector search from the caller's verified identity (never in the prompt);
redact PII before the provider call; treat retrieved text as untrusted; encrypt at rest;
audit who retrieved what; keep keys in a secret manager.

**Q: An agent in the chain keeps failing: what do you do?**
Classify the failure. Transient → retries with exponential backoff and jitter, timeouts,
model fallback. Logical → schema-validated outputs, idempotent tools, orchestrator routes
around the failed step and returns partial results with a flag. Trace every step so you can
see the failing input.

**Q: How do you control token usage per request?**
Budget in state (sum `usage_metadata.total_tokens`), stop routing when exceeded; cap
context (keep last N findings, summarise the rest, `SummarizationMiddleware`); cap calls
(`ModelCallLimitMiddleware`, `ToolCallLimitMiddleware`); smaller `k`; a cheaper model for
grading/routing.

**Q: MCP vs REST API?**
MCP is a standard, self-describing protocol between an LLM app (client) and tool servers
that wrap systems (DBs, SaaS, REST APIs). The model discovers tools with JSON schemas and
chooses calls at run time over a session; REST is one hand-wired contract. MCP servers
usually call REST or a DB internally.

**Q: What is an orchestrator in a multi-agent system?**
Deterministic code (a node or edge) that receives proposals from agents, validates them
against state and policy (budget, turns, repeats, permissions) and decides what executes
next. It separates *judgement* (LLM) from *control* (code).

## Key takeaways

- Loops happen when the LLM holds the exit condition. Put termination, turn caps and token
  budgets in state; let the orchestrator decide.
- Context grows every lap; trim or summarise, or cost grows quadratically.
- RAG-first is enforced by graph structure, `tool_choice`, post-checks and evals: not by
  asking nicely.
- Access control belongs in the vector search filter, keyed by verified identity.
- Retries with backoff, timeouts, fallbacks and idempotent tools make an agent chain survive
  a failing member.

## Next

[Chapter 17: Production](../part6-scale/17-production.md): serve the pipeline with
FastAPI (async done right), stream tokens, cache, authenticate, and package it with Docker
and Compose next to a Qdrant server.
