"""Chapter 15 - the same agentic RAG in ~30 lines with `create_agent`,
plus a tiny Adaptive-RAG router.

create_agent gives you the model->tools->model loop for free. Middleware adds the
guard rails we hand-wrote in the graph version (call limits). response_format gives
a typed final answer.

Run:  QDRANT_MODE=memory uv run python code/ch15/agentic_rag_create_agent.py
"""
from typing import Literal

from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain.tools import tool

from ragbook import build_handbook_index, format_docs, get_llm

llm = get_llm()
store = build_handbook_index("ch15_handbook")


@tool
def retrieve_handbook(query: str) -> str:
    """Search the Lumora employee handbook (policies, robot specs, pricing, runbooks)."""
    return format_docs(store.similarity_search(query, k=4))


class Answer(BaseModel):
    answer: str = Field(description="Concise answer with [n] citations, or 'I don't know'")
    sources: list[str] = Field(description="Source file names actually used")
    confidence: Literal["high", "medium", "low"]


SYSTEM = (
    "You answer questions about Lumora Robotics using the retrieve_handbook tool. "
    "Always search before answering factual questions; search more than once if the first "
    "results do not cover every part of the question (e.g. price AND discount). "
    "Only state facts that appear in retrieved text. If nothing relevant is found, say so."
)

agent = create_agent(
    model=llm,
    tools=[retrieve_handbook],
    system_prompt=SYSTEM,
    middleware=[
        ToolCallLimitMiddleware(tool_name="retrieve_handbook", run_limit=3),  # max 3 searches / question
        ModelCallLimitMiddleware(run_limit=6, exit_behavior="end"),          # hard stop on runaway loops
    ],
    response_format=Answer,
)


# ------------------------------------------------- Adaptive RAG: a router -----
class Route(BaseModel):
    """How much retrieval does this question need?"""
    route: Literal["no_retrieval", "single_step", "multi_step"]
    reason: str


def route_question(question: str) -> Route:
    return llm.with_structured_output(Route).invoke(
        "Classify the question for a handbook assistant.\n"
        "no_retrieval: greetings/chit-chat/pure reasoning. single_step: one fact from one place. "
        "multi_step: needs several facts combined (e.g. price * count with discount, or a policy "
        "applied to a situation).\n\nQuestion: " + question
    )


def ask(question: str) -> None:
    print(f"\nQ: {question}")
    r = route_question(question)
    print(f"  [router] {r.route}: {r.reason}")
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    tool_calls = sum(len(m.tool_calls) for m in result["messages"] if getattr(m, "tool_calls", None))
    ai_msgs = sum(1 for m in result["messages"] if type(m).__name__ == "AIMessage")
    structured = result.get("structured_response")
    print(f"  searches={tool_calls} model_calls={ai_msgs}")
    if structured:
        print(f"A ({structured.confidence}): {structured.answer}")
        print("  sources:", structured.sources)
    else:
        print("A:", result["messages"][-1].text)


if __name__ == "__main__":
    ask("What is the maximum payload of the Atlas A2?")                                    # single
    ask("If I am a new Sales hire, can I take two weeks off in the last week of the quarter during my first month?")  # q42, multi-hop
    ask("What is the maximum payload of the Atlas A3?")                                    # trap: no A3
