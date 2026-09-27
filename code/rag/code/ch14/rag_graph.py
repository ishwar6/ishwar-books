"""Chapter 14 - the Chapter 6 RAG pipeline, rebuilt as a LangGraph graph.

Version A:  retrieve -> generate
Version B:  retrieve -> grade -> (good) generate | (weak) rewrite -> retrieve ...
            with a `retries` counter so the loop is bounded (max 2 rewrites).

The point: once retrieval quality is a *decision* in the graph, you have taken the
first step from "naive RAG" to "agentic RAG" (Chapter 15).

Run:  QDRANT_MODE=memory uv run python code/ch14/rag_graph.py
"""
from typing import Literal

from typing_extensions import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from ragbook import build_handbook_index, format_docs, get_llm

llm = get_llm()
store = build_handbook_index("ch14_handbook")

MAX_REWRITES = 2
SCORE_THRESHOLD = 0.45   # cosine similarity below this = "retrieval looks weak"


class State(TypedDict):
    question: str          # what the user asked (never changes)
    query: str             # what we actually search with (rewrites change this)
    docs: list[Document]
    best_score: float
    retries: int
    answer: str
    sources: list[str]


# ---------------------------------------------------------------- nodes -----
def retrieve(state: State) -> dict:
    hits = store.similarity_search_with_score(state["query"], k=4)
    docs = [d for d, _ in hits]
    best = max((s for _, s in hits), default=0.0)
    print(f"  [retrieve] query={state['query']!r} best_score={best:.3f}")
    return {"docs": docs, "best_score": best}


def rewrite(state: State) -> dict:
    prompt = (
        "Rewrite this search query so it matches how an employee handbook would phrase it. "
        "Return only the rewritten query.\n\nQuery: " + state["query"]
    )
    new_query = llm.invoke(prompt).text.strip()
    print(f"  [rewrite]  -> {new_query!r}")
    return {"query": new_query, "retries": state["retries"] + 1}


def generate(state: State) -> dict:
    prompt = (
        "Answer the question using ONLY the context. Cite sources like [1]. "
        "If the context does not contain the answer, say you don't know.\n\n"
        f"Context:\n{format_docs(state['docs'])}\n\nQuestion: {state['question']}"
    )
    answer = llm.invoke(prompt).text
    return {"answer": answer, "sources": sorted({d.metadata["source"] for d in state["docs"]})}


# --------------------------------------------------------------- routing ----
def grade(state: State) -> Literal["generate", "rewrite"]:
    """The decision: is retrieval good enough, or should we try a better query?"""
    if state["best_score"] >= SCORE_THRESHOLD or state["retries"] >= MAX_REWRITES:
        return "generate"
    return "rewrite"


def build_simple():
    b = StateGraph(State)
    b.add_node("retrieve", retrieve)
    b.add_node("generate", generate)
    b.add_edge(START, "retrieve")
    b.add_edge("retrieve", "generate")
    b.add_edge("generate", END)
    return b.compile()


def build_with_grading():
    b = StateGraph(State)
    b.add_node("retrieve", retrieve)
    b.add_node("rewrite", rewrite)
    b.add_node("generate", generate)
    b.add_edge(START, "retrieve")
    b.add_conditional_edges("retrieve", grade)   # grade() returns the next node's name
    b.add_edge("rewrite", "retrieve")            # <- the cycle; bounded by MAX_REWRITES
    b.add_edge("generate", END)
    return b.compile()


def ask(graph, question: str) -> None:
    print(f"\nQ: {question}")
    out = graph.invoke({"question": question, "query": question, "retries": 0, "docs": [], "best_score": 0.0, "answer": "", "sources": []})
    print("A:", out["answer"])
    print("sources:", out["sources"], "| rewrites:", out["retries"])


if __name__ == "__main__":
    print("=== Version A: retrieve -> generate ===")
    print(build_simple().get_graph().draw_ascii())
    ask(build_simple(), "How many PTO days do I get per year?")

    print("\n=== Version B: with grading + bounded rewrite loop ===")
    graph_b = build_with_grading()
    print(graph_b.get_graph().draw_ascii())
    ask(graph_b, "How many PTO days do I get per year?")
    # vague wording -> likely low score -> the graph rewrites before answering
    ask(graph_b, "whats the deal w/ the robot thing stopping when wifi drops")
