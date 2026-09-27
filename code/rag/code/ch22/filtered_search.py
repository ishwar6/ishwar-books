"""Chapter 22 - "search, then filter" is a bug. Three ways to combine a metadata
filter with a vector search, and what each one costs.

    QDRANT_MODE=memory uv run python code/ch22/filtered_search.py

  post-filter : ANN top-k, then drop rows that fail the filter.
                Returns FEWER THAN K results. Silently. This is the bug.
  pre-filter  : find matching ids first, brute-force only those.
                Always correct; cost grows with how many rows match.
  filtered ANN: traverse the graph but only collect matching nodes (Qdrant).
                Correct and fast - until the filter is so selective that the walk
                wanders through mostly-forbidden territory.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hnsw import Hnsw, normalize      # noqa: E402

K = 10


def true_filtered_topk(X: np.ndarray, q: np.ndarray, allowed: np.ndarray, k: int) -> list[int]:
    """Ground truth: exact search restricted to the allowed ids."""
    ids = np.where(allowed)[0]
    return list(ids[np.argsort(-(X[ids] @ q))[:k]])


def experiment(n: int = 4000, dim: int = 64, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = normalize(rng.normal(size=(n, dim)).astype(np.float32))
    Q = normalize(rng.normal(size=(20, dim)).astype(np.float32))
    print(f"building index: n={n:,} dim={dim} M=16 ef_construction=100")
    index = Hnsw(dim=dim, M=16, ef_construction=100, seed=seed).build(X)

    print(f"\n{'selectivity':>12} {'strategy':>14} {'returned/k':>11} {'recall@10':>10} {'ms/query':>9}")
    print("-" * 78)
    for sel in (0.5, 0.2, 0.05, 0.01):
        mask = rng.random(n) < sel                     # the "tenant" / "category" filter
        allowed_set = set(np.where(mask)[0].tolist())
        truth = [true_filtered_topk(X, q, mask, K) for q in Q]

        # --- 1. post-filter: the tempting one-liner ------------------------------
        t0 = time.perf_counter()
        returned, recs = [], []
        for q, t in zip(Q, truth):
            hits = [h for h in index.search(q, K, ef_search=64) if h in allowed_set]
            returned.append(len(hits))
            recs.append(len(set(hits) & set(t)) / max(len(t), 1))
        ms = (time.perf_counter() - t0) / len(Q) * 1000
        print(f"{sel:>11.0%} {'post-filter':>14} {np.mean(returned):>7.1f}/{K:<3} "
              f"{np.mean(recs):>10.3f} {ms:>9.2f}")

        # --- 2. post-filter with oversampling: the usual patch -------------------
        t0 = time.perf_counter()
        returned, recs = [], []
        for q, t in zip(Q, truth):
            hits = [h for h in index.search(q, K * 20, ef_search=256) if h in allowed_set][:K]
            returned.append(len(hits))
            recs.append(len(set(hits) & set(t)) / max(len(t), 1))
        ms = (time.perf_counter() - t0) / len(Q) * 1000
        print(f"{'':>11} {'post x20':>14} {np.mean(returned):>7.1f}/{K:<3} "
              f"{np.mean(recs):>10.3f} {ms:>9.2f}")

        # --- 3. pre-filter: exact search over the matching subset ----------------
        ids = np.where(mask)[0]
        t0 = time.perf_counter()
        recs = []
        for q, t in zip(Q, truth):
            hits = ids[np.argsort(-(X[ids] @ q))[:K]]
            recs.append(len(set(hits.tolist()) & set(t)) / max(len(t), 1))
        ms = (time.perf_counter() - t0) / len(Q) * 1000
        print(f"{'':>11} {'pre-filter':>14} {min(len(ids), K):>7}/{K:<3} "
              f"{np.mean(recs):>10.3f} {ms:>9.2f}   (scans {len(ids):,} vectors)")

        # --- 4. filtered graph traversal: what Qdrant does ----------------------
        t0 = time.perf_counter()
        returned, recs, calls = [], [], []
        for q, t in zip(Q, truth):
            index.distance_calls = 0
            hits = index.search(q, K, ef_search=64, allowed=allowed_set)
            calls.append(index.distance_calls)
            returned.append(len(hits))
            recs.append(len(set(hits) & set(t)) / max(len(t), 1))
        ms = (time.perf_counter() - t0) / len(Q) * 1000
        print(f"{'':>11} {'filtered ANN':>14} {np.mean(returned):>7.1f}/{K:<3} "
              f"{np.mean(recs):>10.3f} {ms:>9.2f}   ({np.mean(calls):,.0f} dist calls)")
        print()


def qdrant_api():
    """The same three strategies in Qdrant terms."""
    from qdrant_client import models

    from ragbook import get_qdrant_client

    print("=" * 78)
    print("in Qdrant you do NOT choose strategy 1 or 2 - you pass a filter and the")
    print("engine picks, using the payload index's cardinality estimate:")
    print("=" * 78)
    client = get_qdrant_client()
    name = "ch22_filter_demo"
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(name, vectors_config=models.VectorParams(size=32, distance=models.Distance.COSINE))
    rng = np.random.default_rng(0)
    V = normalize(rng.normal(size=(300, 32)).astype(np.float32))
    client.upsert(name, points=[
        models.PointStruct(id=i, vector=v.tolist(), payload={"tenant": f"t{i % 20}"})
        for i, v in enumerate(V)])

    res = client.query_points(
        name, query=V[0].tolist(), limit=5,
        query_filter=models.Filter(must=[models.FieldCondition(
            key="tenant", match=models.MatchValue(value="t0"))]),
    )
    print(f"  filtered query returned {len(res.points)} points, all tenant=t0: "
          f"{all(p.payload['tenant'] == 't0' for p in res.points)}")
    print("""
  Qdrant's rule of thumb:
    - filter matches MANY points  -> walk the HNSW graph, skip non-matching (our #4)
    - filter matches FEW points   -> ignore the graph, brute-force the matching set
                                     (our #3). The switch point is governed by
                                     `full_scan_threshold` (default 10,000 kB of vectors).
    - this only works if the field has a PAYLOAD INDEX; without one Qdrant cannot
      estimate cardinality or enumerate matching ids cheaply.
    - for a hard multi-tenant split, `payload_m` + `m=0` builds a separate small
      graph PER TENANT instead of one global graph (Chapter 18) - that is the fix
      when one tenant is 0.01% of the collection.""")
    client.delete_collection(name)


if __name__ == "__main__":
    experiment()
    qdrant_api()
