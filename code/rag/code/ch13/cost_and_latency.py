"""Chapter 13 - Where do the milliseconds and the dollars go?

Runs N golden questions and breaks each request into steps:
  embed query → vector search → (optional) rerank → generate
then prints p50/p95 per step, tokens and cost per question, and a projection
for a production load. Also a what-if: how k changes cost.

Run:  uv run python code/ch13/cost_and_latency.py [--n 12] [--rerank]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ch10"))
from metrics import PRICES, cost_usd, percentile  # noqa: E402

from ragbook import CHAT_MODEL, EMBED_MODEL, build_handbook_index, format_docs, get_embeddings, get_llm, load_golden  # noqa: E402

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the context, briefly, citing [n]."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t) * 1000


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--rerank", action="store_true")
    args = ap.parse_args()

    store, emb, llm = build_handbook_index("ch13_handbook"), get_embeddings(), get_llm()
    reranker = None
    if args.rerank:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")

    steps = {"embed_ms": [], "search_ms": [], "rerank_ms": [], "generate_ms": [], "total_ms": []}
    tokens_in, tokens_out, costs = [], [], []
    for g in load_golden()[: args.n]:
        t0 = time.perf_counter()
        vec, ms = timed(lambda: emb.embed_query(g["question"]))
        steps["embed_ms"].append(ms)
        fetch = args.k * 4 if reranker else args.k
        docs, ms = timed(lambda: store.similarity_search_by_vector(vec, k=fetch))
        steps["search_ms"].append(ms)
        if reranker:
            (docs, ms) = timed(lambda: [d for d, _ in sorted(zip(docs, reranker.rerank(g["question"], [d.page_content for d in docs])), key=lambda x: -x[1])[: args.k]])
            steps["rerank_ms"].append(ms)
        resp, ms = timed(lambda: (prompt | llm).invoke({"context": format_docs(docs), "question": g["question"]}))
        steps["generate_ms"].append(ms)
        steps["total_ms"].append((time.perf_counter() - t0) * 1000)
        u = resp.usage_metadata or {}
        tokens_in.append(u.get("input_tokens", 0))
        tokens_out.append(u.get("output_tokens", 0))
        q_tokens = len(g["question"]) // 4
        costs.append(cost_usd(CHAT_MODEL, u.get("input_tokens", 0), u.get("output_tokens", 0)) + cost_usd(EMBED_MODEL, q_tokens, 0))

    print(f"{args.n} questions, k={args.k}, rerank={args.rerank}, model={CHAT_MODEL}\n")
    print(f"{'step':<14}{'p50 ms':>10}{'p95 ms':>10}{'share':>8}")
    total_p50 = percentile(steps["total_ms"], 50)
    for name, vals in steps.items():
        if not vals:
            continue
        p50 = percentile(vals, 50)
        share = f"{p50 / total_p50:.0%}" if name != "total_ms" else ""
        print(f"{name:<14}{p50:>10.0f}{percentile(vals, 95):>10.0f}{share:>8}")

    avg_in, avg_out, avg_cost = statistics.mean(tokens_in), statistics.mean(tokens_out), statistics.mean(costs)
    print(f"\ntokens/question: {avg_in:.0f} in, {avg_out:.0f} out   cost/question: ${avg_cost:.5f}")
    for qpd in (1_000, 10_000, 100_000):
        print(f"   {qpd:>7,} questions/day → ${avg_cost * qpd:8.2f}/day   ${avg_cost * qpd * 30:9.2f}/month")

    print("\nsame traffic (10k/day), other chat models (input-token dominated, so cheap models win):")
    for m, (pin, pout) in PRICES.items():
        if m.startswith("gpt"):
            c = avg_in / 1e6 * pin + avg_out / 1e6 * pout
            print(f"   {m:<16} ${c * 10_000:8.2f}/day")
    print(f"\nwhat-if k: input tokens scale ~linearly with k; k={args.k * 2} ≈ {avg_in * 2:.0f} in-tokens ≈ ${avg_cost * 1.8:.5f}/question")
