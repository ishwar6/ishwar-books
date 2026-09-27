"""Chapter 22 - "we changed nothing and recall got worse": ef_search is not a
recall setting, it is a *budget*. Hold it fixed and grow the index, and recall falls.

    QDRANT_MODE=memory uv run python code/ch22/recall_degradation.py
    QDRANT_MODE=memory uv run python code/ch22/recall_degradation.py --max-n 20000

Then the arithmetic you must be able to do on a whiteboard: what a 10M-vector
HNSW index costs in RAM, and how to pick M and ef_search for a latency target.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hnsw import Hnsw, exact_knn, normalize, recall_at_k   # noqa: E402
from viz import ascii_curve                                # noqa: E402

K = 10


def degradation(sizes: list[int], ef_fixed: int = 32, dim: int = 64, n_q: int = 25):
    """Same ef_search, same M, same data distribution - only n changes."""
    rng = np.random.default_rng(0)
    Q = normalize(rng.normal(size=(n_q, dim)).astype(np.float32))
    print(f"recall@{K} at FIXED ef_search={ef_fixed}, M=16, as the index grows")
    print(f"\n{'n':>8} {'recall@10':>10} {'dist calls':>11} {'% of exact':>11} {'build s':>8}")
    print("-" * 60)
    rows = []
    for n in sizes:
        X = normalize(rng.normal(size=(n, dim)).astype(np.float32))
        t0 = time.perf_counter()
        idx = Hnsw(dim=dim, M=16, ef_construction=100, seed=0).build(X)
        build = time.perf_counter() - t0
        truth = [exact_knn(X, q, K) for q in Q]
        idx.distance_calls = 0
        rec = float(np.mean([recall_at_k(idx.search(q, K, ef_fixed), truth[i]) for i, q in enumerate(Q)]))
        calls = idx.distance_calls / n_q
        rows.append((n, rec, calls, build))
        print(f"{n:>8,} {rec:>10.3f} {calls:>11.0f} {calls / n:>10.1%} {build:>8.1f}")
    print(ascii_curve([f"{r[0] // 1000}k" if r[0] >= 1000 else str(r[0]) for r in rows],
                      [r[1] for r in rows],
                      ylabel=f"\nrecall@{K} at fixed ef_search={ef_fixed} vs index size",
                      ymin=0.0, ymax=1.0))
    print("""
Why: ef_search fixes how many nodes the beam may hold. The number of nodes that
must be explored to reach the true top-k grows with n (roughly log n hops, and
more near-ties to disambiguate), so a constant budget covers a shrinking fraction
of the graph - the '% of exact' column collapses. Nothing about your code changed.

Operationally: recall is a property of (index size, data distribution, ef_search).
Re-measure it after every significant ingest, and alert on it. A retrieval quality
regression with no deploy is almost always this, a re-index, or an embedding-model
change.""")
    return rows


def find_ef_for_target(target: float = 0.95, n: int = 4000, dim: int = 64, n_q: int = 25):
    """The procedure itself: sweep ef until recall clears the target, report the cost."""
    rng = np.random.default_rng(1)
    X = normalize(rng.normal(size=(n, dim)).astype(np.float32))
    Q = normalize(rng.normal(size=(n_q, dim)).astype(np.float32))
    truth = [exact_knn(X, q, K) for q in Q]
    idx = Hnsw(dim=dim, M=16, ef_construction=100, seed=0).build(X)
    print(f"\n{'=' * 68}\ntuning ef_search for recall@{K} >= {target:.2f} at n={n:,}\n{'=' * 68}")
    print(f"{'ef_search':>10} {'recall@10':>10} {'p50 ms':>8} {'p95 ms':>8}  verdict")
    print("-" * 68)
    chosen = None
    for ef in (8, 16, 32, 64, 128, 256):
        lat, recs = [], []
        for i, q in enumerate(Q):
            t0 = time.perf_counter()
            got = idx.search(q, K, ef)
            lat.append((time.perf_counter() - t0) * 1000)
            recs.append(recall_at_k(got, truth[i]))
        rec = float(np.mean(recs))
        p50, p95 = float(np.percentile(lat, 50)), float(np.percentile(lat, 95))
        ok = rec >= target
        if ok and chosen is None:
            chosen = (ef, rec, p50, p95)
        print(f"{ef:>10} {rec:>10.3f} {p50:>8.2f} {p95:>8.2f}  {'<- meets target' if ok and chosen[0] == ef else ''}")
    if chosen:
        print(f"\npick ef_search={chosen[0]}: recall {chosen[1]:.3f}, p50 {chosen[2]:.2f} ms, p95 {chosen[3]:.2f} ms")
        print("Report p95, not the mean: the beam explores a data-dependent number of")
        print("nodes, so query latency has a tail even on identical hardware.")


def sizing_table(dims=(384, 768, 1536, 3072), n: int = 10_000_000, M: int = 16):
    """RAM arithmetic for 10M vectors. Do this before choosing anything."""
    print(f"\n{'=' * 68}\nRAM for n = {n:,} vectors, HNSW with M={M}\n{'=' * 68}")
    print("per vector: dim*4 bytes (float32) + links")
    print(f"per node links: layer0 keeps 2M={2 * M} ids; only ~1/M of nodes reach layer 1 at")
    print(f"all, so the upper layers add ~M/(M-1)={M / (M - 1):.1f} ids on average. 4 bytes each")
    link_bytes = round((2 * M + M / (M - 1)) * 4)
    print(f"          -> ~{link_bytes} bytes/node of graph "
          f"(our 2,000-point index measures 32.1 ids/node at M=16)\n")
    print(f"{'dim':>6} {'vectors GB':>11} {'graph GB':>9} {'total GB':>9} {'int8 GB':>9} {'binary GB':>10}")
    print("-" * 68)
    for d in dims:
        vec = n * d * 4 / 1e9
        graph = n * link_bytes / 1e9
        print(f"{d:>6} {vec:>11.1f} {graph:>9.1f} {vec + graph:>9.1f} "
              f"{vec / 4 + graph:>9.1f} {vec / 32 + graph:>10.1f}")
    print("""
Reading this table in an interview:
  - float32 1536-d at 10M is ~61 GB of vectors alone: that is a memory-optimised
    instance, not a laptop, and it is why quantization exists (Chapter 18).
  - the graph is ~1.3 GB at M=16 - small next to the vectors, so raising M to 32
    buys recall for +1.3 GB, while halving the dimension saves 30 GB. Dimension
    first, then quantization, then M.
  - int8 scalar quantization: /4 on vectors, keep originals on disk for rescoring.
  - binary: /32, only viable with rescoring and high-dimensional (>=1024) vectors.""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=10000, help="largest index to build (slow above 20k)")
    args = ap.parse_args()

    sizes = [n for n in (500, 1000, 2500, 5000, 10000, 20000) if n <= args.max_n]
    degradation(sizes)
    find_ef_for_target()
    sizing_table()
