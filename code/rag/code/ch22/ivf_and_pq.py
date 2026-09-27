"""Chapter 22 - the other two index families, implemented small enough to read.

    QDRANT_MODE=memory uv run python code/ch22/ivf_and_pq.py

IVF  (inverted file): partition the space with k-means, search only the `nprobe`
     nearest partitions. Cuts *how many* vectors you compare against.
PQ   (product quantization): split each vector into subvectors, replace each with
     a 1-byte codebook id. Cuts *how big* each comparison is.

They are orthogonal and usually combined (Faiss's IVF-PQ). HNSW is a third
approach - a graph rather than a partition - and is what Qdrant uses.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_cache import load_or_build      # noqa: E402
from hnsw import normalize                 # noqa: E402
from viz import ascii_curve                # noqa: E402

K = 10
SEED = 0


def recall(approx: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean([len(set(a) & set(t)) / len(t) for a, t in zip(approx, truth)]))


# ------------------------------------------------------------------- IVF -----
class IVF:
    """Coarse quantizer + inverted lists. ~25 lines, the real thing is the same idea."""

    def __init__(self, nlist: int = 100, seed: int = SEED):
        self.nlist = nlist
        self.km = KMeans(n_clusters=nlist, n_init=3, random_state=seed)

    def fit(self, X: np.ndarray) -> "IVF":
        self.assign = self.km.fit_predict(X)
        self.centroids = self.km.cluster_centers_
        self.lists = [np.where(self.assign == c)[0] for c in range(self.nlist)]
        return self

    def search(self, X: np.ndarray, q: np.ndarray, k: int, nprobe: int):
        # 1. which partitions could hold the answer? (cheap: nlist dot products)
        order = np.argsort(-(self.centroids @ q))[:nprobe]
        cand = np.concatenate([self.lists[c] for c in order if len(self.lists[c])])
        if len(cand) == 0:
            return np.array([], dtype=int), 0
        # 2. exact search inside those partitions only
        sims = X[cand] @ q
        top = cand[np.argsort(-sims)[:k]]
        return top, len(cand)          # also report how many vectors we touched


def ivf_study(X: np.ndarray, Q: np.ndarray, truth: np.ndarray, nlist: int = 100):
    print(f"\n{'=' * 72}\nIVF - recall vs nprobe (n={len(X):,}, dim={X.shape[1]}, nlist={nlist})\n{'=' * 72}")
    t0 = time.perf_counter()
    ivf = IVF(nlist).fit(X)
    print(f"k-means trained in {time.perf_counter() - t0:.1f}s; "
          f"mean list length {np.mean([len(l) for l in ivf.lists]):.0f}, "
          f"max {max(len(l) for l in ivf.lists)}")
    print(f"\n{'nprobe':>7} {'recall@10':>10} {'vectors scanned':>16} {'% of exact':>11}")
    print("-" * 72)
    rows = []
    for nprobe in (1, 2, 5, 10, 20, 50, nlist):
        got, scanned = zip(*[ivf.search(X, q, K, nprobe) for q in Q])
        r = recall(got, truth)
        avg = float(np.mean(scanned))
        rows.append((nprobe, r, avg))
        print(f"{nprobe:>7} {r:>10.3f} {avg:>16,.0f} {avg / len(X):>10.1%}")
    print(ascii_curve([r[0] for r in rows], [r[1] for r in rows],
                      ylabel="recall@10 vs nprobe", ymin=0.0, ymax=1.0))
    print("""
The IVF failure mode is the edge case: if the true neighbour sits just across a
partition boundary and you only probe 1 cell, you never see it - no amount of
exact re-scoring inside the cell recovers it. That is why nprobe is the recall
dial, and why IVF needs retraining when the data distribution drifts (the
centroids are fitted to the data you had at training time; HNSW has no such
training step, which is a real operational advantage).""")
    return rows


# -------------------------------------------------------------------- PQ -----
class PQ:
    """Product quantization: m subspaces, 256 centroids each -> m bytes per vector."""

    def __init__(self, m: int = 8, nbits: int = 8, seed: int = SEED):
        self.m, self.ncent = m, 2 ** nbits
        self.seed = seed

    def fit(self, X: np.ndarray) -> "PQ":
        n, d = X.shape
        assert d % self.m == 0, "dim must divide by m"
        self.sub = d // self.m
        self.codebooks = []                              # m x ncent x sub
        self.codes = np.zeros((n, self.m), dtype=np.uint8)
        for i in range(self.m):
            part = X[:, i * self.sub:(i + 1) * self.sub]
            km = KMeans(n_clusters=min(self.ncent, len(X)), n_init=2, random_state=self.seed).fit(part)
            self.codebooks.append(km.cluster_centers_)
            self.codes[:, i] = km.labels_
        return self

    def search(self, q: np.ndarray, k: int):
        """Asymmetric distance: the QUERY stays full precision, only the database
        is quantized. Precompute query-to-centroid dot products per subspace, then
        every database vector is a sum of m table lookups - no distance maths at all."""
        tables = [self.codebooks[i] @ q[i * self.sub:(i + 1) * self.sub] for i in range(self.m)]
        scores = np.zeros(len(self.codes), dtype=np.float32)
        for i in range(self.m):
            scores += tables[i][self.codes[:, i]]        # lookup, not multiply
        return np.argsort(-scores)[:k], scores

    def bytes_per_vector(self) -> int:
        return self.m


def pq_study(X: np.ndarray, Q: np.ndarray, truth: np.ndarray, ms: list[int]):
    d = X.shape[1]
    print(f"\n{'=' * 72}\nPQ - compression vs recall (n={len(X):,}, dim={d})\n{'=' * 72}")
    print(f"{'m (bytes)':>10} {'compression':>12} {'recall@10':>10} {'recall@10':>11} {'train s':>8}")
    print(f"{'':>10} {'':>12} {'raw PQ':>10} {'+rescore100':>11}")
    print("-" * 72)
    rows = []
    for m in ms:
        t0 = time.perf_counter()
        pq = PQ(m=m).fit(X)
        train = time.perf_counter() - t0
        raw, resc = [], []
        for q, t in zip(Q, truth):
            top, _ = pq.search(q, K)
            raw.append(len(set(top) & set(t)) / len(t))
            # rescoring: take PQ's top-100 candidates, re-rank with the FULL vectors
            cand, _ = pq.search(q, 100)
            exact_order = cand[np.argsort(-(X[cand] @ q))][:K]
            resc.append(len(set(exact_order) & set(t)) / len(t))
        comp = (d * 4) / m
        rows.append((m, comp, float(np.mean(raw)), float(np.mean(resc))))
        print(f"{m:>10} {comp:>11.0f}x {np.mean(raw):>10.3f} {np.mean(resc):>11.3f} {train:>8.1f}")
    print(ascii_curve([r[0] for r in rows], [r[2] for r in rows],
                      ylabel="raw PQ recall@10 vs bytes per vector", ymin=0.0, ymax=1.0))
    print("""
Two lessons that transfer straight to Qdrant's quantization settings:
  - raw quantized recall looks bad; RESCORING fixes most of it. Search wide in the
    compressed space, then re-rank the survivors with the original vectors. That is
    exactly `QuantizationSearchParams(rescore=True, oversampling=2.0)` (Chapter 18).
  - you keep the original vectors somewhere (disk/memmap) to rescore at all. "32x
    compression" is a RAM statement, not a storage statement.""")
    return rows


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    n, d = 8000, 128
    X = normalize(rng.normal(size=(n, d)).astype(np.float32))
    Q = normalize(rng.normal(size=(25, d)).astype(np.float32))
    truth = np.array([np.argsort(-(X @ q))[:K] for q in Q])

    ivf_study(X, Q, truth)
    pq_study(X, Q, truth, ms=[8, 16, 32, 64])

    # Real embeddings: the case that matters, 1536-d OpenAI vectors.
    Xr = load_or_build(1500)
    Qr = normalize(Xr[:20] + rng.normal(scale=0.05, size=(20, Xr.shape[1])).astype(np.float32))
    tr = np.array([np.argsort(-(Xr @ q))[:K] for q in Qr])
    print(f"\n{'#' * 72}\nreal OpenAI embeddings, 1536-d\n{'#' * 72}")
    pq_study(Xr, Qr, tr, ms=[96, 192])
    print("\n1536-d float32 = 6,144 bytes/vector. PQ at m=96 -> 96 bytes = 64x less RAM.")
    print("(Codebooks here are trained on only 1,500 vectors; production wants")
    print(" >= a few hundred thousand training vectors per subspace codebook.)")
