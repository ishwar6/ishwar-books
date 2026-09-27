"""Chapter 14 - your first LangGraph graph.

A LangGraph graph is a state machine:
  * STATE   one dict that flows through the graph (we describe its shape with a TypedDict)
  * NODES   plain Python functions: state in -> partial state update out
  * EDGES   which node runs next; a conditional edge is a function that returns the name
  * REDUCER how a field's update is merged (default = overwrite; `add_messages` appends)

Run:  uv run python code/ch14/hello_graph.py
"""
from typing import Annotated, Literal

from typing_extensions import TypedDict

from langchain.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from ragbook import get_llm


# ---------------------------------------------------------------- 1. state --
class State(TypedDict):
    # `add_messages` is a REDUCER: a node returning {"messages": [m]} APPENDS m
    # instead of replacing the list. Without it, every node would wipe history.
    messages: Annotated[list, add_messages]
    # no reducer -> last write wins (a plain counter)
    turns: int


# ---------------------------------------------------------------- 2. nodes --
llm = get_llm()


def call_model(state: State) -> dict:
    """Ask the model to continue the conversation. Returns a PARTIAL update."""
    response = llm.invoke(state["messages"])
    return {"messages": [response], "turns": state.get("turns", 0) + 1}


def add_signature(state: State) -> dict:
    """A node that does no LLM work at all: nodes are just functions."""
    last = state["messages"][-1]
    return {"messages": [AIMessage(content=f"(turn {state['turns']}) {last.text}")]}


# --------------------------------------------------------------- 3. routing --
def should_sign(state: State) -> Literal["add_signature", "__end__"]:
    """A conditional edge: look at state, return the NAME of the next node."""
    if state["turns"] % 2 == 1:      # sign odd turns only, just to show routing
        return "add_signature"
    return END


# ----------------------------------------------------------------- 4. graph --
builder = StateGraph(State)
builder.add_node("call_model", call_model)
builder.add_node("add_signature", add_signature)
builder.add_edge(START, "call_model")                 # entry point
builder.add_conditional_edges("call_model", should_sign)  # branch
builder.add_edge("add_signature", END)

# A checkpointer saves state after every step, keyed by thread_id.
# That is what gives the graph MEMORY across separate .invoke() calls.
graph = builder.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    print(graph.get_graph().draw_ascii())          # needs `grandalf` (already in deps)

    config = {"configurable": {"thread_id": "demo-1"}}

    print("\n--- invoke #1 ---")
    out = graph.invoke(
        {"messages": [HumanMessage("My name is Maya and I like Qdrant. Say hi in one line.")], "turns": 0},
        config,
    )
    print(out["messages"][-1].text)

    # Same thread_id -> the checkpointer restores the previous messages, so the model
    # can answer a follow-up that needs the earlier turn. NEW thread_id -> no memory.
    print("\n--- invoke #2, same thread (follow-up needs memory) ---")
    out = graph.invoke({"messages": [HumanMessage("What is my name and what do I like?")]}, config)
    print(out["messages"][-1].text)
    print("turns so far:", out["turns"], "| messages in state:", len(out["messages"]))

    print("\n--- invoke #3, NEW thread (no memory) ---")
    out = graph.invoke(
        {"messages": [HumanMessage("What is my name?")], "turns": 0},
        {"configurable": {"thread_id": "demo-2"}},
    )
    print(out["messages"][-1].text)

    # stream_mode="updates" yields {node_name: partial_update} after each node -
    # this is how UIs show "thinking..." steps. "values" yields the full state instead.
    print("\n--- stream(stream_mode='updates') ---")
    for update in graph.stream({"messages": [HumanMessage("Count to three.")]}, config, stream_mode="updates"):
        for node, delta in update.items():
            print(f"  node={node:<14} keys={list(delta)}")
