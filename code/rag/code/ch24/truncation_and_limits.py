"""Chapter 24 - the three ways a reranker bites you in production.

1. TRUNCATION. A cross-encoder has a fixed input window (512 tokens for the
   ms-marco models, shared between query and passage). Anything past it is not
   "weighted less" - it is not read at all. Parent-document retrieval (ch 9 §9.4)
   hands the reranker long parents and quietly walks into this.

2. CALIBRATION. Cross-encoder outputs are logits from a pairwise classifier, not
   probabilities and not comparable ACROSS queries. A correct top-1 can score
   -1.4 for one question and +8 for another. You may order with them; you may not
   threshold with them (Chapter 12 calibrates thresholds on the first stage's
   cosine for exactly this reason).

3. LATENCY TAILS. Reranking adds a fixed multiple of candidates to your critical
   path. The mean is fine; p95 is what your users feel.

Run:  QDRANT_MODE=memory uv run python code/ch24/truncation_and_limits.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ch10"))
from _shared import build_corpus, cross_encoder, encode_passages, encode_queries, exact_search  # noqa: E402
from metrics import percentile  # noqa: E402

from ragbook import load_golden  # noqa: E402

NEEDLE = "The Pune test hall emergency shutdown code is BRAVO-47-SIERRA."
QUESTION = "What is the Pune test hall emergency shutdown code?"
FILLER = (
    "Lumora Robotics operates autonomous mobile robots in warehouses. Beacon assigns "
    "tasks to robots and plans their paths. Compass reports throughput and battery "
    "health. Field service visits each site on a schedule agreed with the customer. "
)


def main() -> None:
    ce = cross_encoder()
    if ce is None:
        print("cross-encoder unavailable")
        return

    # --- 1. truncation ------------------------------------------------------
    print("1. TRUNCATION - the same sentence, pushed further from the start")
    print(f"   question: {QUESTION}")
    print(f"   needle  : {NEEDLE}\n")
    print(f"   {'prefix chars':>13} {'~tokens':>8} {'needle first':>14} {'needle last':>13}")
    for n_filler in (0, 2, 8, 20, 40, 80):
        prefix = FILLER * n_filler
        approx_tokens = len(prefix) // 4
        first = list(ce.rerank(QUESTION, [NEEDLE + " " + prefix]))[0]
        last = list(ce.rerank(QUESTION, [prefix + " " + NEEDLE]))[0]
        print(f"   {len(prefix):>13} {approx_tokens:>8} {first:>14.3f} {last:>13.3f}")
    print(
        "\n   With the needle FIRST the score stays high however long the passage gets.\n"
        "   With the needle LAST it collapses once the passage passes the window: the\n"
        "   model never saw the sentence. Same text, same question - position decides.\n"
        "   Fix: rerank the CHILD chunk you retrieved, then expand to the parent for the\n"
        "   LLM. Never hand a 4,000-token parent to a 512-token cross-encoder.\n"
    )

    # --- 2. calibration -----------------------------------------------------
    print("2. CALIBRATION - top-1 scores are not comparable across queries")
    golden = [g for g in load_golden() if g["answerable"]][:8]
    chunks = build_corpus(hard=False)
    matrix = encode_passages([c.page_content for c in chunks])
    qvecs = encode_queries([g["question"] for g in golden])
    tops = []
    print(f"   {'cosine':>8} {'ce logit':>9}  question")
    for g, qv in zip(golden, qvecs):
        hits = exact_search(qv, matrix, 10)
        docs = [chunks[i] for i, _ in hits]
        scores = list(ce.rerank(g["question"], [d.page_content for d in docs]))
        best = int(np.argmax(scores))
        tops.append(scores[best])
        print(f"   {hits[best][1]:>8.3f} {scores[best]:>9.3f}  {g['question'][:52]}")
    print(
        f"\n   top-1 cross-encoder logit ranges {min(tops):.2f} to {max(tops):.2f} across eight\n"
        "   questions that ALL retrieved a correct chunk. A global cut-off like\n"
        "   'drop anything below 0' would discard correct answers for some questions and\n"
        "   keep wrong ones for others. Order within a query: yes. Threshold across\n"
        "   queries: no - unless you calibrate per query (e.g. margin to the runner-up).\n"
    )

    # --- 3. latency tails ---------------------------------------------------
    print("3. LATENCY - the cost you actually pay per request")
    golden_all = [g for g in load_golden() if g["answerable"]]
    qall = encode_queries([g["question"] for g in golden_all])
    per_q = [[chunks[i] for i, _ in exact_search(qv, matrix, 20)] for qv in qall]
    lat = []
    for g, docs in zip(golden_all, per_q):
        t0 = time.perf_counter()
        list(ce.rerank(g["question"], [d.page_content for d in docs]))
        lat.append((time.perf_counter() - t0) * 1000)
    print(f"   N=20 candidates, {len(lat)} queries, local CPU")
    print(f"   p50 {percentile(lat, 50):7.0f} ms")
    print(f"   p95 {percentile(lat, 95):7.0f} ms")
    print(f"   max {max(lat):7.0f} ms")
    print(
        f"\n   Budget with p95, not the mean. At {percentile(lat, 95):.0f} ms p95 this stage alone\n"
        "   eats a noticeable slice of a 2-second answer, and it sits in series with the\n"
        "   embedding call and generation (Chapter 13 §13.5). Mitigations: cap N, use the\n"
        "   smaller model, batch, run on GPU, or degrade gracefully - if the reranker has\n"
        "   not answered in X ms, return the stage-1 order and log it."
    )


if __name__ == "__main__":
    main()
