"""Chapter 16 - a planner / retriever / analyst loop that (1) runs away, then (2) is fixed.

The interview scenario: three agents call each other in a cycle, token usage explodes.
WHY it happens: nobody owns termination. The planner always finds "one more thing",
the analyst always finds "a gap", and the graph has a cycle with no exit condition
except LangGraph's recursion_limit (a safety net, not a design).

Run:  QDRANT_MODE=memory uv run python code/ch16/planner_retriever_analyst.py
"""
import operator
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from ragbook import build_handbook_index, format_docs, get_llm

llm = get_llm()
store = build_handbook_index("ch16_handbook")

TASK = "Write a short brief for a Gold-tier customer: what P1 support they get and what an Atlas A2 costs."


class State(TypedDict):
    task: str
    findings: Annotated[list[str], operator.add]   # reducer: each retrieval APPENDS
    queries_done: Annotated[list[str], operator.add]
    next_query: str
    report: str
    turns: int
    tokens: Annotated[int, operator.add]           # reducer: every LLM call ADDS its usage
    done: bool


class Plan(BaseModel):
    done: bool = Field(description="True when the findings are enough to write the brief")
    next_query: str = Field(default="", description="Next handbook search if not done")


class Analysis(BaseModel):
    sufficient: bool
    report: str = Field(description="The brief, if sufficient; otherwise what is missing")


def _tokens(msg) -> int:
    return (msg.usage_metadata or {}).get("total_tokens", 0) if hasattr(msg, "usage_metadata") else 0


# ------------------------------------------------------------ agents/nodes ---
def make_planner(runaway: bool):
    def planner(state: State) -> dict:
        instruction = (
            "You are never satisfied: ALWAYS return done=false and propose one more search."
            if runaway else
            "Return done=true as soon as the findings cover every part of the task. "
            "Never repeat a query from queries_done."
        )
        structured = llm.with_structured_output(Plan, include_raw=True)
        out = structured.invoke(
            f"{instruction}\nTask: {state['task']}\nQueries done: {state['queries_done']}\n"
            f"Findings so far:\n" + "\n---\n".join(state["findings"][-4:])
        )
        plan, raw = out["parsed"], out["raw"]
        print(f"  [planner] turn={state['turns'] + 1} done={plan.done} next={plan.next_query!r}")
        return {"done": plan.done, "next_query": plan.next_query, "turns": state["turns"] + 1, "tokens": _tokens(raw)}
    return planner


def retriever(state: State) -> dict:
    docs = store.similarity_search(state["next_query"], k=3)
    return {"findings": [format_docs(docs)], "queries_done": [state["next_query"]]}


def analyst(state: State) -> dict:
    out = llm.with_structured_output(Analysis, include_raw=True).invoke(
        f"Task: {state['task']}\nFindings:\n" + "\n---\n".join(state["findings"])
        + "\nIf the findings cover the task, write the brief (3 sentences, cite sources)."
    )
    a = out["parsed"]
    print(f"  [analyst] sufficient={a.sufficient}")
    return {"report": a.report, "done": a.sufficient, "tokens": _tokens(out["raw"])}


# ------------------------------------------------------------- guard rails ---
MAX_TURNS = 4
TOKEN_BUDGET = 12_000


def route_after_planner(state: State) -> Literal["retriever", "__end__"]:
    return END if state["done"] else "retriever"


def route_to_validate(state: State) -> Literal["validate", "__end__"]:
    """FIXED graph only: the planner's proposal goes to the orchestrator first."""
    return END if state["done"] else "validate"


def route_after_analyst_guarded(state: State) -> Literal["planner", "__end__"]:
    """The ORCHESTRATOR's check: several independent reasons to stop."""
    if state["done"]:
        return END
    if state["turns"] >= MAX_TURNS:
        print(f"  [guard] max_turns={MAX_TURNS} reached -> stop")
        return END
    if state["tokens"] >= TOKEN_BUDGET:
        print(f"  [guard] token budget {TOKEN_BUDGET} exceeded ({state['tokens']}) -> stop")
        return END
    return "planner"


def validate_query(state: State) -> dict:
    """Orchestrator validates the proposed action BEFORE executing it (dedupe)."""
    if state["next_query"] in state["queries_done"]:
        print(f"  [orchestrator] rejected repeat query {state['next_query']!r}")
        return {"done": True}
    return {}


def build(runaway: bool):
    b = StateGraph(State)
    b.add_node("planner", make_planner(runaway))
    b.add_node("retriever", retriever)
    b.add_node("analyst", analyst)
    b.add_edge(START, "planner")
    if runaway:
        b.add_conditional_edges("planner", route_after_planner)
        b.add_edge("retriever", "analyst")
        b.add_edge("analyst", "planner")                    # <- unconditional cycle
    else:
        b.add_node("validate", validate_query)
        b.add_conditional_edges("planner", route_to_validate)     # planner PROPOSES ...
        b.add_conditional_edges("validate", route_after_planner)  # ... orchestrator DECIDES, then executes
        b.add_edge("retriever", "analyst")
        b.add_conditional_edges("analyst", route_after_analyst_guarded)
    return b.compile()


def run(graph, label: str, recursion_limit: int = 25) -> None:
    print(f"\n=== {label} (recursion_limit={recursion_limit}) ===")
    init = {"task": TASK, "findings": [], "queries_done": [], "next_query": "", "report": "", "turns": 0, "tokens": 0, "done": False}
    try:
        out = graph.invoke(init, {"recursion_limit": recursion_limit})
        print(f"finished: turns={out['turns']} tokens={out['tokens']}")
        print("REPORT:", out["report"][:400])
    except GraphRecursionError as e:
        print("STOPPED by GraphRecursionError:", str(e)[:90], "...")
        print("  -> this is the safety net firing. Tokens were burned on every lap before it did.")


if __name__ == "__main__":
    run(build(runaway=True), "RUNAWAY: no termination criteria", recursion_limit=12)
    run(build(runaway=False), "FIXED: done flag + max_turns + token budget")
