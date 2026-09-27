"""Chapter 22 - what M, ef_construction and ef_search actually do, measured.

    QDRANT_MODE=memory uv run python code/ch22/parameter_study.py
    QDRANT_MODE=memory uv run python code/ch22/parameter_study.py --quick

Four experiments, all against exact brute-force ground truth:

  1. ef_search  sweep -> recall@10, distance calls, latency        (query-time knob)
  2. M          sweep -> recall, links, memory, build time         (build-time, memory)
  3. ef_constr. sweep -> build time, recall                        (build-time, quality)
  4. neighbour-selection heuristic vs "keep the M nearest"         (why the heuristic exists)

The synthetic set (uniform random, 64-d) is the pessimistic case. The real set
(OpenAI embeddings of handbook sentences, 1536-d) is what production looks like.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_cache import load_or_build                      # noqa: E402
from hnsw import Hnsw, exact_knn, normalize, recall_at_k   # noqa: E402
from viz import ascii_curve, ascii_xy, bar                 # noqa: E402

K = 10


def ground_truth(X: np.ndarray, Q: np.ndarray, k: int = K) -> list[list[int]]:
    return [exact_knn(X, q, k) for q in Q]


def evaluate(index: Hnsw, X: np.ndarray, Q: np.ndarray, truth: list[list[int]], ef: int):
    """Recall, mean distance calls and mean latency for one ef_search setting."""
    index.distance_calls = 0
    t0 = time.perf_counter()
    recalls = [recall_at_k(index.search(q, K, ef), truth[i]) for i, q in enumerate(Q)]
    dt = (time.perf_counter() - t0) / len(Q) * 1000
    return float(np.mean(recalls)), index.distance_calls / len(Q), dt


def sweep_ef_search(X, Q, truth, label: str, efs: list[int], M=16, ef_c=100):
    print(f"\n{'=' * 78}\n1. ef_search sweep - {label}  (n={len(X)}, dim={X.shape[1]}, M={M}, ef_construction={ef_c})\n{'=' * 78}")
    t0 = time.perf_counter()
    index = Hnsw(dim=X.shape[1], M=M, ef_construction=ef_c, seed=0).build(X)
    print(f"built in {time.perf_counter() - t0:.1f}s   layers={len(index.graph)}   links={index.link_count():,}")
    print(f"\n{'ef_search':>10} {'recall@10':>10} {'dist calls':>11} {'ms/query':>9}  {'vs exact':>9}")
    print("-" * 78)
    rows = []
    for ef in efs:
        rec, calls, ms = evaluate(index, X, Q, truth, ef)
        rows.append((ef, rec, calls, ms))
        print(f"{ef:>10} {rec:>10.3f} {calls:>11.0f} {ms:>9.2f}  {calls / len(X):>8.1%}")
    print(f"\n(exact search touches all {len(X)} vectors per query)")
    print(ascii_curve([r[0] for r in rows], [r[1] for r in rows],
                      ylabel=f"recall@{K} vs ef_search (saturates)", ymin=0.0, ymax=1.0))
    print(ascii_curve([r[0] for r in rows], [r[2] for r in rows],
                      ylabel="distance calls vs ef_search (keeps climbing, ~linear)"))
    print("\nthe trade-off curve you actually care about:")
    print(ascii_xy([r[3] for r in rows], [r[1] for r in rows],
                   xlabel="ms/query", ylabel=f"recall@{K}"))
    return index, rows


def sweep_m(X, Q, truth, Ms: list[int], ef_c=100, ef_s=64):
    print(f"\n{'=' * 78}\n2. M sweep - links per node (ef_construction={ef_c}, ef_search={ef_s})\n{'=' * 78}")
    print(f"{'M':>4} {'recall@10':>10} {'links':>10} {'link MB':>8} {'vec MB':>8} {'build s':>8} {'ms/query':>9}")
    print("-" * 78)
    rows = []
    for M in Ms:
        t0 = time.perf_counter()
        idx = Hnsw(dim=X.shape[1], M=M, ef_construction=ef_c, seed=0).build(X)
        build = time.perf_counter() - t0
        rec, _, ms = evaluate(idx, X, Q, truth, ef_s)
        mem = idx.memory_bytes()
        rows.append((M, rec, idx.link_count(), mem["links"] / 1e6, mem["vectors"] / 1e6, build, ms))
        print(f"{M:>4} {rec:>10.3f} {idx.link_count():>10,} {mem['links'] / 1e6:>8.2f} "
              f"{mem['vectors'] / 1e6:>8.2f} {build:>8.1f} {ms:>9.2f}")
    print(ascii_curve([r[0] for r in rows], [r[1] for r in rows],
                      ylabel="recall@10 vs M (diminishing returns)", ymin=0.0, ymax=1.0))
    print("\nmemory is LINEAR in M - links per node is a budget, not a quality dial:")
    mx = max(r[3] for r in rows)
    for M, _, links, mb, _, _, _ in rows:
        print(f"  M={M:<3} {mb:6.2f} MB links  {bar(mb, mx)}")
    return rows


def sweep_ef_construction(X, Q, truth, efcs: list[int], M=16, ef_s=64):
    print(f"\n{'=' * 78}\n3. ef_construction sweep - graph quality (M={M}, ef_search={ef_s})\n{'=' * 78}")
    print(f"{'ef_constr':>10} {'build s':>9} {'recall@10':>10} {'links':>10}")
    print("-" * 78)
    rows = []
    for ef_c in efcs:
        t0 = time.perf_counter()
        idx = Hnsw(dim=X.shape[1], M=M, ef_construction=ef_c, seed=0).build(X)
        build = time.perf_counter() - t0
        rec, _, _ = evaluate(idx, X, Q, truth, ef_s)
        rows.append((ef_c, build, rec, idx.link_count()))
        print(f"{ef_c:>10} {build:>9.1f} {rec:>10.3f} {idx.link_count():>10,}")
    print(ascii_curve([r[0] for r in rows], [r[1] for r in rows],
                      ylabel="build seconds vs ef_construction (linear-ish: you pay once)"))
    print(ascii_curve([r[0] for r in rows], [r[2] for r in rows],
                      ylabel="recall@10 vs ef_construction (saturates early)", ymin=0.0, ymax=1.0))
    return rows


def heuristic_vs_naive(X, Q, truth, M=8, ef_c=100, ef_s=32):
    """Why the paper's neighbour-selection heuristic exists at all."""
    print(f"\n{'=' * 78}\n4. neighbour selection: heuristic vs 'keep the M nearest' (M={M}, ef_search={ef_s})\n{'=' * 78}")
    print(f"{'selection':>12} {'recall@10':>10} {'links':>9} {'ms/query':>9}")
    print("-" * 78)
    out = {}
    for name, use_h in (("heuristic", True), ("naive", False)):
        idx = Hnsw(dim=X.shape[1], M=M, ef_construction=ef_c, seed=0, use_heuristic=use_h).build(X)
        rec, _, ms = evaluate(idx, X, Q, truth, ef_s)
        out[name] = rec
        print(f"{name:>12} {rec:>10.3f} {idx.link_count():>9,} {ms:>9.2f}")
    d = out["heuristic"] - out["naive"]
    print(f"\nheuristic - naive = {d:+.3f} recall at identical M and ef_search.")
    print("The naive rule links each node to its nearest neighbours only, so dense")
    print("regions become cliques with no edges leaving them; the greedy walk gets")
    print("trapped. The heuristic keeps an edge only if it points somewhere no")
    print("already-kept edge covers - diversity of direction beats pure proximity.")
    return out


def qdrant_note():
    """Show the real Qdrant knobs, and be honest about what embedded mode does."""
    from qdrant_client import models

    from ragbook import get_qdrant_client

    print(f"\n{'=' * 78}\n5. the same three knobs in Qdrant\n{'=' * 78}")
    client = get_qdrant_client()
    name = "ch22_hnsw_demo"
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=64, distance=models.Distance.COSINE),
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),   # build-time knobs
    )
    rng = np.random.default_rng(0)
    V = normalize(rng.normal(size=(500, 64)).astype(np.float32))
    client.upsert(name, points=[models.PointStruct(id=i, vector=v.tolist(), payload={"i": i})
                                for i, v in enumerate(V)])
    res = client.query_points(
        name, query=V[0].tolist(), limit=10,
        search_params=models.SearchParams(hnsw_ef=128),              # query-time knob
    )
    cfg = client.get_collection(name).config.hnsw_config
    print(f"  created with hnsw_config: m={cfg.m}, ef_construct={cfg.ef_construct}")
    print(f"  queried with SearchParams(hnsw_ef=128) -> {len(res.points)} hits, "
          f"top score {res.points[0].score:.3f}")
    print("""
  NOTE, and say this in an interview if asked how you measured:
  the embedded/local Qdrant used by this book does BRUTE-FORCE search. It accepts
  m / ef_construct / hnsw_ef and ignores them, so recall is always 1.0 and no sweep
  on it means anything. The sweeps above therefore run on our own implementation.
  To measure Qdrant's real HNSW you need a server (`docker compose up -d qdrant`)
  and a collection above `indexing_threshold` (default 20,000 kB of vectors) - below that
  Qdrant deliberately stays exact, because brute force wins on small segments.""")
    client.delete_collection(name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller n and fewer points per sweep")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    n_syn = 1200 if args.quick else 2000
    n_q = 15 if args.quick else 25
    X_syn = normalize(rng.normal(size=(n_syn, 64)).astype(np.float32))
    Q_syn = normalize(rng.normal(size=(n_q, 64)).astype(np.float32))
    T_syn = ground_truth(X_syn, Q_syn)

    efs = [8, 16, 32, 64, 128] if args.quick else [8, 16, 32, 64, 128, 256, 512]
    sweep_ef_search(X_syn, Q_syn, T_syn, "synthetic uniform vectors", efs)
    sweep_m(X_syn, Q_syn, T_syn, [4, 8, 16, 32] if args.quick else [4, 8, 16, 32, 64])
    sweep_ef_construction(X_syn, Q_syn, T_syn, [16, 50, 100, 200] if args.quick else [16, 50, 100, 200, 400])
    heuristic_vs_naive(X_syn, Q_syn, T_syn)

    # Real embeddings: one ef_search sweep. Same knobs, much easier data.
    n_real = 600 if args.quick else 1200
    X_real = load_or_build(n_real)[:n_real]
    Q_real = X_real[:20] + rng.normal(scale=0.05, size=(20, X_real.shape[1])).astype(np.float32)
    Q_real = normalize(Q_real)          # "near-duplicate" queries, like a real paraphrase
    T_real = ground_truth(X_real, Q_real)
    sweep_ef_search(X_real, Q_real, T_real, "real OpenAI embeddings", efs)

    qdrant_note()
