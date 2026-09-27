"""Chapter 10 - The evaluation harness: one config → one table.

  uv run python code/ch10/run_eval.py --k 3 --limit 10
  uv run python code/ch10/run_eval.py --k 6
  uv run python code/ch10/run_eval.py --k 6 --rerank
  uv run python code/ch10/run_eval.py --chunk-size 400 --k 8 --no-judges

For every golden question: retrieve → generate (grounded prompt with [n] citations)
→ retrieval metrics (metrics.py) → LLM judges (llm_judges.py) → latency + cost.
Prints a summary table and writes data/eval_results/<config>.json so you can diff runs.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

sys.path.insert(0, str(Path(__file__).parent))  # so `import metrics, llm_judges` works from repo root
import llm_judges as J  # noqa: E402
import metrics as M  # noqa: E402

from ragbook import CHAT_MODEL, ROOT, build_handbook_index, format_docs, get_llm, load_golden  # noqa: E402

SYSTEM = (
    "You answer questions about the Lumora employee handbook using ONLY the context. "
    "Cite the chunk number like [2] after each sentence that uses it. "
    "If the context does not contain the answer, reply exactly: "
    "\"I don't know - the handbook does not cover this.\" Never use outside knowledge."
)
prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", "Context:\n{context}\n\nQuestion: {question}")])


def rerank_cross_encoder(query, docs, top_n):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    global _ce
    if "_ce" not in globals():
        _ce = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
    scores = list(_ce.rerank(query, [d.page_content for d in docs]))
    return [d for d, _ in sorted(zip(docs, scores), key=lambda x: -x[1])[:top_n]]


def run(args) -> dict:
    store = build_handbook_index(f"ch10_handbook_{args.chunk_size}", chunk_size=args.chunk_size)
    llm = get_llm()
    chain = prompt | llm
    golden = load_golden()[: args.limit] if args.limit else load_golden()

    per_q = []
    for g in golden:
        t0 = time.perf_counter()
        fetch_k = args.k * 4 if args.rerank else args.k
        docs = store.similarity_search(g["question"], k=fetch_k)
        if args.rerank:
            docs = rerank_cross_encoder(g["question"], docs, args.k)
        t_retrieve = time.perf_counter() - t0

        context = format_docs(docs)
        resp = chain.invoke({"context": context, "question": g["question"]})
        answer = resp.text
        t_total = time.perf_counter() - t0
        usage = resp.usage_metadata or {}
        cost = M.cost_usd(CHAT_MODEL, usage.get("input_tokens", 0), usage.get("output_tokens", 0))

        retrieved = [d.metadata["doc_id"] for d in docs]     # one entry per CHUNK, labelled by its doc
        docs_ranked = M.unique_in_order(retrieved)            # one entry per DOC, for rank metrics
        relevant = set(g["sources"])
        row = {
            "id": g["id"], "answerable": g["answerable"], "answer": answer,
            "retrieved": retrieved, "latency_s": t_total, "retrieve_s": t_retrieve, "cost_usd": cost,
            "input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0),
        }
        if g["answerable"]:
            row.update({
                # chunk-level: what fraction of the k chunks the LLM reads are useful?
                "hit@k": M.hit_rate_at_k(retrieved, relevant, args.k),
                "precision@k": M.precision_at_k(retrieved, relevant, args.k),
                "context_precision@k": M.context_precision_at_k(retrieved, relevant, args.k),
                # doc-level: did we cover every gold document, and how high was the first one?
                "recall@k": M.recall_at_k(docs_ranked, relevant, args.k),
                "mrr": M.reciprocal_rank(docs_ranked, relevant),
                "ndcg@k": M.ndcg_at_k(docs_ranked, relevant, args.k),
                "keyword_hit": J.keyword_hit(answer, g["keywords"]),
                "wrong_abstain": J.is_abstention(answer),          # abstained although answerable
            })
            if args.judges:
                f = J.faithfulness(answer, context)
                c = J.answer_correctness(g["question"], answer, g["answer"])
                row.update({
                    "faithfulness": f["score"], "relevance": J.answer_relevance(g["question"], answer),
                    "correct": c.verdict == "correct", "partial": c.verdict == "partial", "completeness": c.completeness,
                    "citation_acc": J.citation_accuracy(answer, [d.page_content for d in docs]),
                })
        else:
            row["correct_abstain"] = J.is_abstention(answer)      # abstained AND should have
        per_q.append(row)
        flag = "✓" if row.get("hit@k", row.get("correct_abstain")) else "✗"
        print(f"  {flag} {g['id']}  {t_total:4.1f}s  {answer[:80].replace(chr(10), ' ')}")

    return summarise(args, per_q)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else float("nan")


def summarise(args, per_q) -> dict:
    ans = [r for r in per_q if r["answerable"]]
    un = [r for r in per_q if not r["answerable"]]
    s = {
        "config": {"k": args.k, "chunk_size": args.chunk_size, "rerank": args.rerank, "model": CHAT_MODEL, "n": len(per_q)},
        "retrieval": {m: mean([r[m] for r in ans]) for m in ["hit@k", "recall@k", "precision@k", "mrr", "ndcg@k", "context_precision@k"]},
        "generation": {
            "keyword_hit": mean([r["keyword_hit"] for r in ans]),
            "wrong_abstain_rate": mean([r["wrong_abstain"] for r in ans]),
            "correct_abstain_rate": mean([r["correct_abstain"] for r in un]) if un else None,
        },
        "ops": {
            "p50_latency_s": M.percentile([r["latency_s"] for r in per_q], 50),
            "p95_latency_s": M.percentile([r["latency_s"] for r in per_q], 95),
            "avg_cost_usd": mean([r["cost_usd"] for r in per_q]),
            "avg_input_tokens": mean([r["input_tokens"] for r in per_q]),
        },
    }
    if args.judges and ans:
        s["generation"].update({
            "faithfulness": mean([r["faithfulness"] for r in ans]),
            "hallucination_rate": 1 - mean([r["faithfulness"] for r in ans]),
            "relevance_1to5": mean([r["relevance"] for r in ans]),
            "correct_rate": mean([r["correct"] for r in ans]),
            "partial_rate": mean([r["partial"] for r in ans]),
            "completeness": mean([r["completeness"] for r in ans]),
            "citation_accuracy": mean([r["citation_acc"] for r in ans]),
        })
    out = ROOT / "data" / "eval_results"
    out.mkdir(parents=True, exist_ok=True)
    # a --limit run gets its own file (…_nN.json) so it never overwrites the full-golden-set results
    name = f"k{args.k}_cs{args.chunk_size}{'_rerank' if args.rerank else ''}{'_nojudge' if not args.judges else ''}{f'_n{args.limit}' if args.limit else ''}"
    (out / f"{name}.json").write_text(json.dumps({"summary": s, "per_question": per_q}, indent=2))
    print(f"\n=== {name} ===")
    for section, vals in s.items():
        if section == "config":
            continue
        for k, v in vals.items():
            if v is not None:
                print(f"  {k:<22}{v:8.3f}" if isinstance(v, float) else f"  {k:<22}{v}")
    print(f"  saved → data/eval_results/{name}.json")
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N golden questions")
    ap.add_argument("--no-judges", dest="judges", action="store_false", help="skip LLM judges (fast, cheap)")
    run(ap.parse_args())
