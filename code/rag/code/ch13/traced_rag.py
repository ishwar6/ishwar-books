"""Chapter 13 - See inside your RAG: structured logs, LangSmith traces, callbacks.

Every request writes ONE JSON line to data/logs/rag.jsonl with: the query, the
retrieved chunk ids + scores, token usage, latency per step, and the answer.
That file answers 90% of "why did it say that?" questions.

Also shown:
  - @traceable from LangSmith: a no-op unless LANGSMITH_TRACING=true + LANGSMITH_API_KEY are set,
    so this file runs with or without an account.
  - a LangChain callback handler capturing token usage from every model call.

Run:  uv run python code/ch13/traced_rag.py
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from ragbook import CHAT_MODEL, ROOT, build_handbook_index, format_docs, get_llm, load_golden

# ----------------------------------------------------------- JSON-lines log ---
LOG_PATH = ROOT / "data" / "logs" / "rag.jsonl"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("rag")
logger.setLevel(logging.INFO)
_h = logging.FileHandler(LOG_PATH)
_h.setFormatter(logging.Formatter("%(message)s"))   # the message IS the JSON
logger.addHandler(_h)


# ---------------------------------------------------------- usage callback ---
class UsageHandler(BaseCallbackHandler):
    """Collects token usage from every chat-model call made while it is attached."""

    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def on_llm_end(self, response, **kwargs):
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                usage = getattr(msg, "usage_metadata", None) or {}
                self.calls.append({"input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0)})

    @property
    def totals(self) -> dict[str, int]:
        return {k: sum(c[k] for c in self.calls) for k in ("input_tokens", "output_tokens")}


prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the context, briefly, citing [n]. If not covered, say: I don't know."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


@traceable(name="rag_answer", tags=["ch13"])          # shows up as a tree in LangSmith when enabled
def rag_answer(store, llm, question: str, k: int = 4) -> dict:
    request_id = str(uuid.uuid4())[:8]
    t: dict[str, float] = {}

    t0 = time.perf_counter()
    scored = store.similarity_search_with_score(question, k=k)      # embed query + ANN search
    t["retrieve_ms"] = (time.perf_counter() - t0) * 1000

    handler = UsageHandler()
    t1 = time.perf_counter()
    resp = (prompt | llm).invoke({"context": format_docs([d for d, _ in scored]), "question": question},
                                 config={"callbacks": [handler]})
    t["generate_ms"] = (time.perf_counter() - t1) * 1000
    t["total_ms"] = (time.perf_counter() - t0) * 1000

    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "request_id": request_id, "model": CHAT_MODEL, "k": k,
        "question": question,
        "retrieved": [{"chunk_id": d.metadata["chunk_id"], "source": d.metadata["source"], "score": round(s, 3)} for d, s in scored],
        "usage": handler.totals, "timing_ms": {kk: round(v) for kk, v in t.items()},
        "answer": resp.text,
    }
    logger.info(json.dumps(record, ensure_ascii=False))
    return record


if __name__ == "__main__":
    tracing = os.getenv("LANGSMITH_TRACING", "").lower() == "true" and bool(os.getenv("LANGSMITH_API_KEY"))
    print("LangSmith tracing:", "ON → https://smith.langchain.com" if tracing else "off (set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env to enable)")

    store, llm = build_handbook_index("ch13_handbook"), get_llm()
    for g in load_golden()[:3]:
        r = rag_answer(store, llm, g["question"])
        print(f"\n[{r['request_id']}] {g['question']}")
        print(f"   retrieved: {[(x['source'], x['score']) for x in r['retrieved'][:2]]} ...")
        print(f"   usage: {r['usage']}   timing: {r['timing_ms']}")
        print(f"   answer: {r['answer'][:120].replace(chr(10), ' ')}")

    print(f"\nlog file: {LOG_PATH.relative_to(ROOT)}  ({sum(1 for _ in open(LOG_PATH))} lines)")
    print("debug a bad answer:  grep <request_id> data/logs/rag.jsonl | python -m json.tool")
