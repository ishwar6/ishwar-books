"""Chapter 24 - how deep should the candidate set be? Measure, don't guess.

"Retrieve 50, rerank to 5" is folklore until you can derive the 50. Two facts
decide it:

  1. A reranker can only re-order what stage 1 returned. recall@N is therefore a
     CEILING on everything downstream. Where recall@N flattens, extra depth buys
     nothing.
  2. Reranking cost is linear in N. A cross-encoder runs one forward pass per
     candidate, so doubling N doubles the reranking latency.

So: plot the ceiling and the achieved quality against N, and read off the knee.
Queries are the realistic (vague) phrasings - on questions copied from the
documents every curve is flat at 1.0 and you learn nothing.

Run:  QDRANT_MODE=memory uv run python code/ch24/cascade_design.py
      QDRANT_MODE=memory uv run python code/ch24/cascade_design.py --limit 10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ch10"))
from _shared import (  # noqa: E402
    bar,
    build_corpus,
    cross_encoder,
    encode_passages,
    encode_queries,
    exact_search,
    vague_queries,
)
from metrics import ndcg_at_k, recall_at_k, reciprocal_rank, unique_in_order  # noqa: E402

from ragbook import load_golden  # noqa: E402

DEPTHS = [5, 10, 20, 50, 100, 200]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="use only the first N questions")
    args = ap.parse_args()

    golden = [g for g in load_golden() if g["answerable"]]
    vague = vague_queries(golden)
    if args.limit:
        golden = golden[: args.limit]

    chunks = build_corpus(hard=True)
    texts = [c.page_content for c in chunks]
    print(f"corpus {len(chunks)} chunks (handbook + generated + novel distractors)")
    t0 = time.perf_counter()
    matrix = encode_passages(texts)
    print(f"stage-1 encoder: BAAI/bge-small-en-v1.5, {matrix.shape[1]}-d, "
          f"{len(chunks)} chunks in {time.perf_counter() - t0:.1f}s")

    queries = [vague[g["id"]] for g in golden]
    qvecs = encode_queries(queries)
    ce = cross_encoder()
    if ce is None:
        print("cross-encoder unavailable - cannot measure the cascade")
        return

    # Retrieve once at the deepest N, then slice: identical to retrieving at each N,
    # but without repeating the search.
    deepest = max(DEPTHS)
    per_q = []
    for g, qv in zip(golden, qvecs):
        hits = exact_search(qv, matrix, deepest)
        per_q.append([chunks[i] for i, _ in hits])

    # Cross-encoder scores for the deepest list; a prefix of these is exactly what
    # you would compute at a smaller N.
    print(f"\nscoring {len(golden)} queries x {deepest} candidates with the cross-encoder...")
    t0 = time.perf_counter()
    ce_scores = [np.array(list(ce.rerank(q, [d.page_content for d in docs])))
                 for q, docs in zip(queries, per_q)]
    elapsed = time.perf_counter() - t0
    pairs = len(golden) * deepest
    throughput = pairs / elapsed
    print(f"{pairs} pairs in {elapsed:.1f}s -> {throughput:.0f} pairs/s on this CPU")

    rows = []
    for n in DEPTHS:
        ceiling, r5, nd5, rr = [], [], [], []
        for g, docs, scores in zip(golden, per_q, ce_scores):
            rel = set(g["sources"])
            cand = docs[:n]
            ids_stage1 = unique_in_order([d.metadata["doc_id"] for d in cand])
            ceiling.append(recall_at_k(ids_stage1, rel, n))          # can the reranker win at all?

            order = np.argsort(-scores[:n])
            final = [cand[i] for i in order]
            ids = unique_in_order([d.metadata["doc_id"] for d in final])
            r5.append(recall_at_k(ids, rel, 5))
            nd5.append(ndcg_at_k(ids, rel, 5))
            rr.append(reciprocal_rank(ids, rel))
        rows.append((n, np.mean(ceiling), np.mean(r5), np.mean(nd5), np.mean(rr), n / throughput * 1000))

    print(f"\n{'N':>5} {'recall@N':>9} {'recall@5':>9} {'nDCG@5':>8} {'MRR':>7} {'rerank ms':>10}   ceiling")
    for n, ceil, r5, nd5, rr, ms in rows:
        print(f"{n:>5} {ceil:>9.3f} {r5:>9.3f} {nd5:>8.3f} {rr:>7.3f} {ms:>10.0f}   {bar(ceil, 24, 0.8, 1.0)}")

    print("\nfinal nDCG@5 vs candidate depth")
    lo = min(r[3] for r in rows) - 0.01
    hi = max(r[3] for r in rows) + 0.005
    for n, _, _, nd5, _, ms in rows:
        print(f"  N={n:<4} {nd5:.3f} |{bar(nd5, 40, lo, hi)}  (+{ms:.0f} ms)")

    best = max(rows, key=lambda r: r[3])
    knee = next((r for r in rows if r[3] >= best[3] - 0.003), best)
    deepest_row = rows[-1]
    print(
        f"\nread it: the stage-1 ceiling keeps rising to recall@N = {deepest_row[1]:.3f} at "
        f"N={deepest_row[0]},\nbut final quality stops improving at N={knee[0]} "
        f"(nDCG@5 {knee[3]:.3f}, {knee[5]:.0f} ms of reranking).\n"
        f"Going from N={knee[0]} to N={deepest_row[0]} costs "
        f"{deepest_row[5] - knee[5]:.0f} ms more per query and returns "
        f"{deepest_row[3] - knee[3]:+.3f} nDCG.\n"
        "A rising ceiling is necessary, not sufficient: those extra candidates are ones the\n"
        "reranker ranks below the five it already had. Pick the knee, not the ceiling."
    )


if __name__ == "__main__":
    main()
