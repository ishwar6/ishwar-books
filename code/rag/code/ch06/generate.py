"""Chapter 6 - generate.py: the read path, part 2.

context + question -> prompt -> LLM -> answer with citations, plus token usage and cost.

Run:  uv run python code/ch06/generate.py "What is the Beacon API rate limit?"
"""
import sys
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from ragbook import CHAT_MODEL, get_llm

sys.path.insert(0, str(Path(__file__).parent))
from retrieve import build_context, get_store, retrieve  # noqa: E402

# $ per 1M tokens (in, out). Update when you change RAG_CHAT_MODEL.
PRICES = {"gpt-5.4-mini": (0.75, 4.50), "gpt-5-mini": (0.25, 2.00), "gpt-5.6-luna": (0.20, 1.20)}

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the Lumora Robotics handbook assistant.\n"
            "Answer using ONLY the numbered context. If the answer is not in the context, say: "
            "I don't know based on the handbook.\n"
            "If two context items disagree, say so and prefer the one marked as a policy or more recent.\n"
            "Cite every fact with its context number like [2]. Keep answers under 4 sentences.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def cost_usd(usage: dict, model: str = CHAT_MODEL) -> float:
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    return usage["input_tokens"] * price_in / 1e6 + usage["output_tokens"] * price_out / 1e6


def answer(question: str, k: int = 4, store=None) -> dict:
    """The whole read path in one call. Returns a dict so callers (CLI, API, evals) can log everything."""
    store = store or get_store()
    hits = retrieve(store, question, k=k)
    chain = PROMPT | get_llm()
    response = chain.invoke({"context": build_context(hits), "question": question})
    usage = response.usage_metadata or {"input_tokens": 0, "output_tokens": 0}
    return {
        "question": question,
        "answer": response.text,
        "sources": [{"n": i, "source": d.metadata["source"], "chunk": d.metadata["chunk_index"], "score": round(s, 3)} for i, (d, s) in enumerate(hits, 1)],
        "usage": {"input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"]},
        "cost_usd": round(cost_usd(usage), 6),
    }


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "What is the Beacon API rate limit?"
    result = answer(question)
    print(f"Q: {result['question']}\nA: {result['answer']}\n")
    for s in result["sources"]:
        print(f"   [{s['n']}] {s['score']}  {s['source']} (chunk {s['chunk']})")
    print(f"\ntokens {result['usage']}  cost ${result['cost_usd']}")
