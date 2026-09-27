"""Chapter 16 - guard rails for an agentic RAG system, with `create_agent` middleware.

  1. RAG-first: force the first model call to search (wrap_model_call + tool_choice)
  2. RAG-first: reject an answer produced without retrieval (after_model + jump_to)
  3. Security: PII redaction on input, role-based access as a Qdrant payload filter
  4. Reliability: retries with backoff, call limits
  5. Faithfulness gate + a citation-rate eval on the golden set

Run:  QDRANT_MODE=memory uv run python code/ch16/guardrails.py
"""
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel
from qdrant_client import models

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    ModelCallLimitMiddleware,
    ModelRequest,
    ModelRetryMiddleware,
    PIIMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
    after_model,
    hook_config,
    wrap_model_call,
)
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain.tools import ToolRuntime, tool

from ragbook import build_handbook_index, format_docs, get_llm, load_golden

llm = get_llm(timeout=30, max_retries=2)     # transport-level timeout + retries live on the model
store = build_handbook_index("ch16_handbook")

# --------------------------------------------------- 3a. role-based access ----
# Which documents each role may see. In production this comes from your IAM system and
# is stored as payload (metadata.allowed_roles / metadata.classification) at ingest time.
CONFIDENTIAL_DOCS = {"10-pricing-and-plans"}                 # the price book is Confidential
ROLE_DENY = {"employee": CONFIDENTIAL_DOCS, "sales": set(), "admin": set()}


@dataclass
class UserContext:
    role: Literal["employee", "sales", "admin"]


@tool
def retrieve_handbook(query: str, runtime: ToolRuntime[UserContext]) -> str:
    """Search the Lumora handbook. Results are filtered by the caller's role."""
    denied = ROLE_DENY[runtime.context.role]
    flt = None
    if denied:
        # must_not on doc_id: the filter is applied INSIDE the vector search, so
        # confidential chunks never even reach the model. Enforce access in retrieval, not in the prompt.
        flt = models.Filter(must_not=[models.FieldCondition(key="metadata.doc_id", match=models.MatchAny(any=sorted(denied)))])
    docs = store.similarity_search(query, k=4, filter=flt)
    return format_docs(docs) if docs else "No documents found."


# -------------------------------------------------------- 1. force search ----
@wrap_model_call
def force_first_search(request: ModelRequest, handler):
    """Until a ToolMessage exists in the conversation, the model MUST call the tool."""
    if not any(isinstance(m, ToolMessage) for m in request.messages):
        request = request.override(tool_choice="retrieve_handbook")
    return handler(request)


# ------------------------------------------- 2. reject ungrounded answers ----
@after_model
@hook_config(can_jump_to=["model"])
def reject_ungrounded(state: AgentState, runtime):
    last = state["messages"][-1]
    used_tool = any(isinstance(m, ToolMessage) for m in state["messages"])
    if isinstance(last, AIMessage) and not last.tool_calls and not used_tool:
        print("  [after_model] answer without retrieval -> rejected, jumping back to model")
        return {"messages": [HumanMessage("You must call retrieve_handbook before answering.")], "jump_to": "model"}
    return None


SYSTEM = (
    "You answer questions about Lumora Robotics ONLY from retrieve_handbook results. "
    "Cite every fact as [n]. If the results do not contain the answer, reply exactly: "
    "I don't know based on the handbook."
)

agent = create_agent(
    model=llm,
    tools=[retrieve_handbook],
    system_prompt=SYSTEM,
    context_schema=UserContext,
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_input=True),   # 3b. never send emails to the LLM
        force_first_search,
        reject_ungrounded,
        ToolCallLimitMiddleware(tool_name="retrieve_handbook", run_limit=3),
        ModelCallLimitMiddleware(run_limit=6, exit_behavior="end"),
        ModelRetryMiddleware(max_retries=2, backoff_factor=2.0, initial_delay=0.5),   # 4. exponential backoff
        ToolRetryMiddleware(max_retries=2),
    ],
)


# ------------------------------------------------------ 5. faithfulness gate ---
class Faithful(BaseModel):
    supported: bool
    unsupported_claims: list[str]


def faithfulness_gate(answer: str, context: str) -> Faithful:
    return llm.with_structured_output(Faithful).invoke(
        "Is every factual claim in the ANSWER supported by the CONTEXT? List unsupported claims.\n"
        f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
    )


def ask(question: str, role: str = "employee") -> str:
    result = agent.invoke({"messages": [HumanMessage(question)]}, context=UserContext(role=role))
    msgs = result["messages"]
    answer = msgs[-1].text
    context = "\n".join(m.text for m in msgs if isinstance(m, ToolMessage))
    if context and "I don't know" not in answer:
        verdict = faithfulness_gate(answer, context)
        if not verdict.supported:
            print(f"  [gate] unsupported claims {verdict.unsupported_claims} -> refusing")
            answer = "I don't know based on the handbook."
    return answer


if __name__ == "__main__":
    print("\n--- PII redaction: the email never reaches the model ---")
    print("A:", ask("My email is maya@lumora.example. What is the hotel cap in Europe?"))

    print("\n--- RBAC: same question, two roles ---")
    q = "How much does an Atlas A2 robot cost?"
    print("employee ->", ask(q, role="employee"))
    print("sales    ->", ask(q, role="sales"))

    print("\n--- citation-rate eval on a golden subset ---")
    golden = [g for g in load_golden() if g["answerable"]][:8]
    cited = 0
    for g in golden:
        a = ask(g["question"], role="admin")
        has_cite = "[" in a and "]" in a
        cited += has_cite
        print(f"  {g['id']} cited={has_cite} | {a[:90]}")
    print(f"citation rate: {cited}/{len(golden)} = {cited / len(golden):.0%}")
