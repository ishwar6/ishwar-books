"""Chapter 25 - six embedding models, one corpus, one golden set, one table.

The framework: fix the corpus, the chunking and the questions; vary ONLY the
embedding model; report retrieval quality AND latency AND memory AND cost. Then
decide. "It felt better" is not a result; neither is "it's top of MTEB".

Search is exact cosine on purpose - we are comparing models, so ANN recall error
must not leak into the comparison (Chapter 22 measures that separately).

Queries are the realistic "vague" phrasings from Chapter 24: questions copied out
of the documents make every model look equally good and hide the differences.

Run:  QDRANT_MODE=memory uv run python code/ch25/embedding_bakeoff.py
      QDRANT_MODE=memory uv run python code/ch25/embedding_bakeoff.py --models openai-3-small,bge-small-en-v1.5
      QDRANT_MODE=memory uv run python code/ch25/embedding_bakeoff.py --easy   # handbook only
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "ch24"))
sys.path.insert(0, str(HERE.parents[1] / "ch10"))
from _embedders import default_models, embed_cost, encode, index_mb  # noqa: E402
from _shared import build_corpus, vague_queries  # noqa: E402
from metrics import ndcg_at_k, recall_at_k, reciprocal_rank, unique_in_order  # noqa: E402

from ragbook import load_golden  # noqa: E402


def approx_tokens(texts: list[str]) -> int:
    """~4 characters per token. Good enough for a cost estimate; use tiktoken when
    the bill matters (Chapter 2 §2.6)."""
    return sum(len(t) for t in texts) // 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default="", help="comma-separated subset")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--easy", action="store_true", help="handbook only, original question wording")
    args = ap.parse_args()

    golden_all = [g for g in load_golden() if g["answerable"]]
    vague = vague_queries(golden_all)
    golden = golden_all[: args.limit] if args.limit else golden_all
    queries = [g["question"] if args.easy else vague[g["id"]] for g in golden]

    chunks = build_corpus(hard=not args.easy)
    texts = [c.page_content for c in chunks]
    doc_ids = [c.metadata["doc_id"] for c in chunks]
    n_tokens = approx_tokens(texts)

    models = default_models()
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        models = [m for m in models if m.name in wanted]

    print(f"corpus   : {len(chunks)} chunks (~{n_tokens:,} tokens), "
          f"{'handbook only' if args.easy else 'handbook + generated + novels'}")
    print(f"questions: {len(golden)} ({'original wording' if args.easy else 'realistic/vague wording'})")
    print(f"search   : exact cosine (no ANN error in the comparison)\n")

    rows = []
    for emb in models:
        try:
            dm = encode(emb, texts, "doc")
            qm = encode(emb, queries, "query")
        except Exception as e:
            print(f"  {emb.name}: unavailable ({type(e).__name__}: {str(e)[:70]})")
            continue

        r5, r10, nd10, rr = [], [], [], []
        dn = dm / np.clip(np.linalg.norm(dm, axis=1, keepdims=True), 1e-12, None)
        for g, qv in zip(golden, qm):
            q = qv / max(float(np.linalg.norm(qv)), 1e-12)
            order = np.argsort(-(dn @ q))[:10]
            ids = unique_in_order([doc_ids[i] for i in order])
            rel = set(g["sources"])
            r5.append(recall_at_k(ids, rel, 5))
            r10.append(recall_at_k(ids, rel, 10))
            nd10.append(ndcg_at_k(ids, rel, 10))
            rr.append(reciprocal_rank(ids, rel))

        # Latency on a fixed 50-chunk sample, always fresh (never cached), so the
        # column is comparable on every run. For API models this is wall clock and
        # therefore mostly network + batching, which is exactly what you pay in
        # production (Chapter 13 §13.5 measured ~512 ms p50 for one query embedding).
        t0 = time.perf_counter()
        encode(emb, texts[:50], "doc", cache=False)
        per_1k = (time.perf_counter() - t0) / 50 * 1000
        rows.append({
            "name": emb.name, "dim": emb.out_dim,
            "r5": np.mean(r5), "r10": np.mean(r10), "ndcg": np.mean(nd10), "mrr": np.mean(rr),
            "mb_1m": index_mb(1_000_000, emb.out_dim),
            "cost": embed_cost(n_tokens, emb.price_per_1m),
            "s_per_1k": per_1k, "notes": emb.notes,
        })

    print(f"{'model':<20}{'dim':>6}{'recall@5':>10}{'recall@10':>11}{'nDCG@10':>9}{'MRR':>7}"
          f"{'MB/1M':>8}{'corpus $':>10}{'s/1k docs':>11}")
    for r in sorted(rows, key=lambda x: -x["ndcg"]):
        s = f"{r['s_per_1k']:.1f}" if r["s_per_1k"] else "-"
        print(f"{r['name']:<20}{r['dim']:>6}{r['r5']:>10.3f}{r['r10']:>11.3f}{r['ndcg']:>9.3f}"
              f"{r['mrr']:>7.3f}{r['mb_1m']:>8.0f}{r['cost']:>10.4f}{s:>11}")

    if rows:
        best = max(rows, key=lambda r: r["ndcg"])
        cheap = min(rows, key=lambda r: (r["mb_1m"], -r["ndcg"]))
        gap = best["ndcg"] - cheap["ndcg"]
        print(
            f"\nbest quality : {best['name']} (nDCG@10 {best['ndcg']:.3f}, "
            f"{best['mb_1m']:.0f} MB per 1M vectors)"
            f"\nleanest      : {cheap['name']} (nDCG@10 {cheap['ndcg']:.3f}, "
            f"{cheap['mb_1m']:.0f} MB per 1M vectors)"
            f"\nthe trade    : {gap:+.3f} nDCG for {best['mb_1m'] / max(cheap['mb_1m'], 1):.1f}x the memory"
            f" and {'an API dependency' if best['cost'] else 'no API dependency'}."
            "\n\nDecide with the column that binds you. If 8 GB of RAM is the constraint, the"
            "\nquality column is advice; the memory column is the answer."
        )
        for r in rows:
            if r["notes"]:
                print(f"   {r['name']:<20} {r['notes']}")


if __name__ == "__main__":
    main()
