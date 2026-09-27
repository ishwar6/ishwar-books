# Chapter 14 · LangGraph Basics: RAG as a State Machine

> **Goal:** by the end of this chapter you can build a LangGraph graph from scratch (state,
> nodes, edges, a conditional branch, a bounded loop, and a checkpointer that gives it
> memory) and you can rebuild the Chapter 6 RAG pipeline as a graph whose *retrieval
> quality is a decision*, not an assumption. That decision is the doorway to agentic RAG.

Everything before this chapter used LangChain alone: a retriever, a prompt, a model, glued
together with Python. That is enough for "2-step RAG": retrieve, then generate. It stops
being enough the moment you want the pipeline to *branch* (was retrieval good?), *loop*
(try again with a better query), *pause* (ask a human), *remember* (multi-turn), or
*stream* progress to a UI. You can hand-write all of that with `if`/`while` and a dict,
and for a while you should: but you end up re-inventing a workflow engine badly.
LangGraph is that engine, written once.

---

## 14.1 The mental model: a state machine for LLM apps

A LangGraph graph has exactly four ideas:

| Idea | What it is | In code |
|---|---|---|
| **State** | one dict that flows through the graph; the *only* thing nodes share | a `TypedDict` |
| **Node** | a plain function `state -> partial update` | `def retrieve(state): return {"docs": ...}` |
| **Edge** | which node runs after which; conditional edges are functions that *return the name* of the next node | `add_edge`, `add_conditional_edges` |
| **Reducer** | how a field's update is merged into state; default is overwrite | `Annotated[list, add_messages]` |

Two consequences you should internalise:

1. **Nodes return *partial* updates.** `{"answer": "..."}` updates one key and leaves the
   rest alone. You never return the whole state.
2. **Reducers decide append vs overwrite.** A `messages` field with `add_messages` *appends*;
   a `turns: int` with no reducer is *replaced*. Forget the reducer on a list and every node
   wipes the history: the single most common LangGraph bug.

The three things you get for free by describing your pipeline this way (and cannot easily
get from plain functions) are **persistence** (a checkpointer snapshots state after every
node, keyed by a `thread_id`), **streaming** (the graph yields after every node, so a UI can
show "searching… grading… writing…"), and **interrupts** (pause before a node for human
approval, resume later from the checkpoint). Cycles come free too: an edge back to an
earlier node is just an edge.

## 14.2 Your first graph: [`code/ch14/hello_graph.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch14/hello_graph.py)

The whole file is ~90 lines. The core:

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]   # reducer: APPEND
    turns: int                                # no reducer: overwrite

def call_model(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response], "turns": state.get("turns", 0) + 1}

def add_signature(state: State) -> dict:
    last = state["messages"][-1]
    return {"messages": [AIMessage(content=f"(turn {state['turns']}) {last.text}")]}

def should_sign(state: State) -> Literal["add_signature", "__end__"]:
    return "add_signature" if state["turns"] % 2 == 1 else END

builder = StateGraph(State)
builder.add_node("call_model", call_model)
builder.add_node("add_signature", add_signature)
builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", should_sign)
builder.add_edge("add_signature", END)
graph = builder.compile(checkpointer=InMemorySaver())
```

Read `should_sign` carefully: a conditional edge is a function that looks at state and
returns a **node name** (or `END`). It does no work itself. "Nodes do the work, edges say
what happens next" is the whole discipline.

### Memory is the checkpointer, not the model

```python
config = {"configurable": {"thread_id": "demo-1"}}
graph.invoke({"messages": [HumanMessage("My name is Maya ...")], "turns": 0}, config)
graph.invoke({"messages": [HumanMessage("What is my name?")]}, config)   # remembers
graph.invoke({"messages": [HumanMessage("What is my name?")]},
             {"configurable": {"thread_id": "demo-2"}})                    # does not
```

`InMemorySaver` stores a snapshot of the state after every node, keyed by `thread_id`. On
the second call with the same thread, LangGraph loads the snapshot, applies the reducer
(`add_messages` appends your new question to the stored history), and runs. On a new
thread there is nothing to load. For production swap in `langgraph-checkpoint-postgres`
or `-sqlite`; the code does not change.

### Streaming

`graph.stream(..., stream_mode="updates")` yields `{node_name: partial_update}` after each
node; `stream_mode="values"` yields the full state instead. This is how you show progress
in a UI without threading anything yourself.

## 14.3 RAG as a graph: [`code/ch14/rag_graph.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch14/rag_graph.py)

Version A is the Chapter 6 pipeline with a new coat of paint:

```python
class State(TypedDict):
    question: str; query: str; docs: list[Document]; best_score: float
    retries: int; answer: str; sources: list[str]

def retrieve(state):  hits = store.similarity_search_with_score(state["query"], k=4); ...
def generate(state):  ... llm.invoke(prompt with format_docs(state["docs"])) ...

b.add_edge(START, "retrieve"); b.add_edge("retrieve", "generate"); b.add_edge("generate", END)
```

Nothing is gained yet. Version B adds one conditional edge and one node:

```python
def grade(state) -> Literal["generate", "rewrite"]:
    if state["best_score"] >= SCORE_THRESHOLD or state["retries"] >= MAX_REWRITES:
        return "generate"
    return "rewrite"

def rewrite(state):
    new_query = llm.invoke("Rewrite this search query so it matches how an employee "
                           "handbook would phrase it ... " + state["query"]).text
    return {"query": new_query, "retries": state["retries"] + 1}

b.add_conditional_edges("retrieve", grade)
b.add_edge("rewrite", "retrieve")          # the cycle
```

Three design points hide in those lines:

- **`question` vs `query`.** The user's question never changes; the search string does.
  Keep them separate or your final answer will address the rewritten question.
- **The loop is bounded by state**, not by hope. `retries` is checked in the edge. Chapter 16
  is about what happens when people skip this.
- **Grading here is a cheap heuristic** (cosine score). Chapter 15 replaces it with an LLM
  judgement using structured output. Same edge, better grader: that is the value of the
  shape.

## 14.4 Why this is the bridge to agentic RAG

Look at what changed between A and B. In A, the program flow was fixed by you at design
time. In B, the *graph* looks at the state at run time and chooses a path. That is what
"agentic" means at the smallest scale: a decision made from evidence during execution. In
Chapter 15 the decisions get bigger (*whether* to retrieve at all, *what* to search for,
*whether* the results are relevant) and some of them are made by the model via tool calls.
The machinery stays exactly this: state, nodes, edges, reducers.

## Run it

```bash
uv run python code/ch14/hello_graph.py
QDRANT_MODE=memory uv run python code/ch14/rag_graph.py
```

Observed output (trimmed):

```
--- invoke #1 ---
(turn 1) Hi Maya - glad to hear you like Qdrant!

--- invoke #2, same thread (follow-up needs memory) ---
Your name is Maya, and you like Qdrant.
turns so far: 2 | messages in state: 5

--- invoke #3, NEW thread (no memory) ---
(turn 1) I don't know your name from the information I have here. ...

--- stream(stream_mode='updates') ---
  node=call_model     keys=['messages', 'turns']
  node=add_signature  keys=['messages']
```

```
=== Version B: with grading + bounded rewrite loop ===
Q: How many PTO days do I get per year?
  [retrieve] query='How many PTO days do I get per year?' best_score=0.553
A: Every full-time employee receives **24 days of PTO per calendar year** [1].
sources: ['02-pto-and-leave-policy.md', '14-faq.md'] | rewrites: 0

Q: whats the deal w/ the robot thing stopping when wifi drops
  [retrieve] query='whats the deal w/ the robot thing stopping when wifi drops' best_score=0.433
  [rewrite]  -> 'What is the policy regarding the robot stopping operation when Wi-Fi is disconnected?'
  [retrieve] query='What is the policy regarding ...' best_score=0.471
A: When a robot loses connectivity for more than 60 seconds, it parks itself safely and waits [1]. ...
sources: [...] | rewrites: 1
```

The sloppy question scored below the threshold, got rewritten once, and then cleared it.
The clean question never entered the loop. Both graphs print their ASCII diagram first
(`graph.get_graph().draw_ascii()`: `grandalf` is in the project dependencies).

## Exercises

1. Change `turns` to `Annotated[int, operator.add]` and return `{"turns": 1}` from
   `call_model`. Same behaviour, different mechanism: explain why.
2. Add a third node to `rag_graph.py` that runs *after* `generate` and appends a
   "confidence" field based on `best_score`. Route to it unconditionally.
3. Replace the cosine threshold in `grade` with the golden set: for each answerable question,
   record whether the loop fired and whether the final sources contained the expected
   `sources`. Did rewriting help recall@4?
4. Swap `InMemorySaver` for `langgraph.checkpoint.sqlite.SqliteSaver` (`uv add
   langgraph-checkpoint-sqlite`). Kill the process between invoke #1 and #2 and show memory
   survives.
5. Use `stream_mode="values"` and print only the `query` field each step. Watch it change.

## Interview questions

**Q: What is LangGraph and why would you use it over plain LangChain?**
LangGraph is a library for building stateful, multi-step LLM applications as graphs: a
typed state, nodes that are functions, edges (including conditional ones) that route.
Plain LangChain composes a *linear* pipeline well (prompt | model | parser). The moment
you need branches, cycles, persistence across turns, human-in-the-loop pauses, or
per-step streaming, LangGraph gives you those as first-class features instead of ad hoc
`while` loops and globals.

**Q: What is a reducer and why does it matter?**
A reducer is the function that merges a node's update for a field into the existing state.
The default replaces the value; `add_messages` appends (and de-duplicates by message id);
`operator.add` concatenates lists or sums ints. Without a reducer on the messages list,
every node would overwrite the conversation history.

**Q: How does LangGraph give an agent memory?**
Through a checkpointer. After every node it saves a state snapshot keyed by
`thread_id` from the invoke config. The next invoke with the same thread loads the snapshot
and merges the new input through the reducers. Short-term memory is therefore *the
conversation state*; long-term memory across threads is a separate store (e.g. a vector
store you write to).

**Q: Where do you put the decision "retrieval was not good enough, try again"?**
In a conditional edge that reads a grade from state (a score or an LLM judgement) and
returns either `generate` or `rewrite`, with an edge from `rewrite` back to `retrieve`.
The loop must be bounded by a counter in state that the edge checks.

**Q: Difference between `stream_mode="values"` and `"updates"`?**
`values` yields the complete state after each step; `updates` yields only the delta each
node returned, keyed by node name. `updates` is what you use for progress UIs and logs;
`values` when the consumer needs the whole picture (e.g. the latest message).

**Q: What is `recursion_limit`?**
The maximum number of steps (super-steps) a single invoke may execute before LangGraph
raises `GraphRecursionError`; default 25. It is a safety net against infinite cycles, not a
substitute for explicit termination logic in your state (Chapter 16).

## Key takeaways

- A graph is state + nodes + edges + reducers. Nodes do the work; edges decide what is next.
- Nodes return partial updates; reducers decide append vs overwrite. Put `add_messages` on
  the messages list.
- Memory lives in the checkpointer, keyed by `thread_id`, not in the model.
- A conditional edge that grades retrieval turns a pipeline into a system that decides;
  bound every loop with a counter in state.
- Keep `question` (what the user asked) and `query` (what you search for) as separate fields.

## Next

[Chapter 15: Agentic RAG](15-agentic-rag.md): let the model decide *whether* and *what*
to retrieve, grade the results with structured output, and compare a hand-built graph with
`create_agent`.
