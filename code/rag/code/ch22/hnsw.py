"""Chapter 22 - HNSW, implemented from scratch so the parameters stop being folklore.

This is the module the other ch22 scripts import. Run it directly for a demo:

    QDRANT_MODE=memory uv run python code/ch22/hnsw.py

Readability over speed: distances are plain numpy dot products, the priority queues
are `heapq`, and nothing is vectorised across queries. A production HNSW (hnswlib,
Qdrant's Rust core) is the same algorithm with SIMD, memory pooling and no Python.

The three parameters everyone is asked about, and where they live in this file:

    M                -> `Hnsw.__init__`  : links kept per node per layer. Memory + recall.
    ef_construction  -> `Hnsw.__init__`  : beam width WHILE INSERTING. Graph quality + build time.
    ef_search        -> `Hnsw.search`    : beam width AT QUERY TIME. Recall + latency.

Everything is cosine distance on L2-normalised vectors: dist = 1 - dot(a, b).
"""
from __future__ import annotations

import heapq
import math

import numpy as np


def normalize(X: np.ndarray) -> np.ndarray:
    """Cosine similarity == dot product once every vector has unit length."""
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)


class Hnsw:
    """A Hierarchical Navigable Small World index.

    The structure is a stack of graphs ("layers"). Every point lives in layer 0;
    a point also appears in layer l with probability that decays exponentially, so
    the upper layers are a sparse, long-range road network over the same data.
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 100,
        seed: int = 0,
        use_heuristic: bool = True,
    ) -> None:
        self.dim = dim
        self.M = M                       # links per node on layers >= 1
        self.M0 = 2 * M                  # layer 0 gets 2M: it holds every point, so it
                                         # needs more connectivity to stay navigable
        self.ef_construction = ef_construction
        self.use_heuristic = use_heuristic
        # Level assignment uses mL = 1/ln(M): the paper's choice, it makes the expected
        # number of layers ~log_M(n) so descent costs O(log n).
        self.mL = 1.0 / math.log(M) if M > 1 else 1.0
        self.rng = np.random.default_rng(seed)

        self.vectors: list[np.ndarray] = []
        # graph[layer] = {node_id: set(neighbour_ids)}
        self.graph: list[dict[int, set[int]]] = []
        self.entry_point: int | None = None
        self.max_layer = -1
        self.distance_calls = 0          # the real cost unit of ANN search

    # ---------------------------------------------------------------- basics --
    def _dist(self, q: np.ndarray, idx: int) -> float:
        self.distance_calls += 1
        return 1.0 - float(np.dot(q, self.vectors[idx]))

    def _dists(self, q: np.ndarray, idxs: list[int]) -> np.ndarray:
        """Batched distance: same maths, one numpy call. Counts as len(idxs) calls."""
        self.distance_calls += len(idxs)
        return 1.0 - (self.vectors_arr(idxs) @ q)

    def vectors_arr(self, idxs: list[int]) -> np.ndarray:
        return np.stack([self.vectors[i] for i in idxs])

    def _random_level(self) -> int:
        """P(level >= l) decays geometrically. floor(-ln(U) * mL) is the paper's formula."""
        return int(-math.log(self.rng.random() + 1e-12) * self.mL)

    # ----------------------------------------------------------- graph search --
    def _search_layer(self, q: np.ndarray, entry: list[int], ef: int, layer: int,
                      allowed: set[int] | None = None) -> list[tuple[float, int]]:
        """Beam search inside ONE layer. Returns up to `ef` (distance, node) pairs.

        Two structures, as in the paper:
          - `candidates`: min-heap of things still worth expanding (closest first)
          - `best`: max-heap (negated) of the ef closest found so far

        The loop stops when the nearest unexpanded candidate is further than the
        worst of the current best set - nothing closer can be reached from here.
        `ef` is exactly the size of `best`, which is why it controls both recall
        and cost: a wider beam escapes local minima, and visits more nodes doing it.

        `allowed` is filtered search done the way Qdrant does it: the walk still
        traverses non-matching nodes (they are the roads between matching ones), but
        only matching nodes are collected. The cost of that is visible in 22.7.
        """
        g = self.graph[layer]
        visited: set[int] = set(entry)
        d_entry = self._dists(q, entry)
        candidates = [(float(d), n) for d, n in zip(d_entry, entry)]
        heapq.heapify(candidates)
        best = [(-d, n) for d, n in candidates if allowed is None or n in allowed]
        heapq.heapify(best)

        while candidates:
            d_c, c = heapq.heappop(candidates)
            if best and d_c > -best[0][0] and len(best) >= ef:
                break                                   # no improvement possible
            unseen = [n for n in g.get(c, ()) if n not in visited]
            if not unseen:
                continue
            visited.update(unseen)
            for d_n, n in zip(self._dists(q, unseen), unseen):
                d_n = float(d_n)
                heapq.heappush(candidates, (d_n, n))    # traverse through everything
                if allowed is not None and n not in allowed:
                    continue                            # ...but do not collect it
                if len(best) < ef:
                    heapq.heappush(best, (-d_n, n))
                elif d_n < -best[0][0]:
                    heapq.heapreplace(best, (-d_n, n))
        return sorted((-d, n) for d, n in best)

    # ------------------------------------------------------ neighbour choice --
    def _select_neighbours(self, cands: list[tuple[float, int]], m: int) -> list[int]:
        """Pick which `m` of the candidates actually become links.

        Naive ("keep the m nearest") is what most people assume. It builds *clusters*:
        inside a dense blob every node links only to its blob, and the graph loses the
        long-range edges that make the walk navigable - search then gets trapped.

        The paper's heuristic keeps a candidate only if it is closer to the query point
        than to any already-selected neighbour. That deliberately keeps edges pointing
        in *different directions*, preserving connectivity between clusters.
        `use_heuristic=False` reproduces the naive version so the difference is measurable.
        """
        if not self.use_heuristic:
            return [n for _, n in sorted(cands)[:m]]

        selected: list[int] = []
        for d_new, n in sorted(cands):
            if len(selected) >= m:
                break
            keep = True
            for s in selected:
                # distance from the candidate to an already-chosen neighbour
                self.distance_calls += 1
                if 1.0 - float(np.dot(self.vectors[n], self.vectors[s])) < d_new:
                    keep = False        # `s` already covers this direction, and better
                    break
            if keep:
                selected.append(n)
        # If the heuristic was too strict, top up with the nearest remaining.
        if len(selected) < m:
            for _, n in sorted(cands):
                if n not in selected:
                    selected.append(n)
                if len(selected) >= m:
                    break
        return selected

    def _link(self, a: int, b: int, layer: int) -> None:
        self.graph[layer].setdefault(a, set()).add(b)
        self.graph[layer].setdefault(b, set()).add(a)

    def _prune(self, node: int, layer: int) -> None:
        """Links are bidirectional, so adding one can push a neighbour over its budget.
        When that happens we re-run the selection heuristic on its full link set."""
        m = self.M0 if layer == 0 else self.M
        nbrs = self.graph[layer][node]
        if len(nbrs) <= m:
            return
        nbr_list = list(nbrs)
        d = self._dists(self.vectors[node], nbr_list)
        keep = set(self._select_neighbours([(float(x), n) for x, n in zip(d, nbr_list)], m))
        for dropped in nbrs - keep:
            self.graph[layer][dropped].discard(node)
        self.graph[layer][node] = keep

    # ---------------------------------------------------------------- insert --
    def add(self, vec: np.ndarray) -> int:
        idx = len(self.vectors)
        self.vectors.append(vec)
        level = self._random_level()
        while self.max_layer < level:                 # grow the stack if needed
            self.graph.append({})
            self.max_layer += 1
        for l in range(level + 1):
            self.graph[l].setdefault(idx, set())

        if self.entry_point is None:
            self.entry_point = idx
            return idx

        # Phase 1: greedy descent from the top down to level+1 with ef=1.
        # Cheap coarse navigation - this is what the upper layers are for.
        ep = [self.entry_point]
        for l in range(self.max_layer, level, -1):
            ep = [self._search_layer(vec, ep, 1, l)[0][1]]

        # Phase 2: from level down to 0, do a WIDE (ef_construction) search and link.
        for l in range(min(level, self.max_layer), -1, -1):
            found = self._search_layer(vec, ep, self.ef_construction, l)
            m = self.M0 if l == 0 else self.M
            for n in self._select_neighbours(found, m):
                self._link(idx, n, l)
                self._prune(n, l)
            ep = [n for _, n in found]                # reuse as entry for the layer below

        if level > self.max_layer_of(self.entry_point):
            self.entry_point = idx
        return idx

    def max_layer_of(self, node: int) -> int:
        return max((l for l in range(len(self.graph)) if node in self.graph[l]), default=0)

    def build(self, X: np.ndarray) -> "Hnsw":
        for v in X:
            self.add(v)
        return self

    # ---------------------------------------------------------------- search --
    def search(self, q: np.ndarray, k: int = 10, ef_search: int = 64,
               allowed: set[int] | None = None) -> list[int]:
        """Descend the layers greedily, then one wide beam search on layer 0.

        `allowed` (optional) restricts which nodes may be RETURNED, not which may be
        traversed - filtered vector search, as a real engine implements it."""
        if self.entry_point is None:
            return []
        ef = max(ef_search, k)                        # a beam narrower than k is nonsense
        ep = [self.entry_point]
        for l in range(self.max_layer, 0, -1):
            ep = [self._search_layer(q, ep, 1, l)[0][1]]   # descent ignores the filter
        found = self._search_layer(q, ep, ef, 0, allowed=allowed)
        return [n for _, n in found[:k]]

    # ----------------------------------------------------------------- stats --
    def link_count(self) -> int:
        return sum(len(v) for layer in self.graph for v in layer.values())

    def memory_bytes(self) -> dict[str, int]:
        """Vectors are float32; links are 4-byte ids (both directions are stored)."""
        n = len(self.vectors)
        return {
            "vectors": n * self.dim * 4,
            "links": self.link_count() * 4,
            "layers": len(self.graph),
        }


def exact_knn(X: np.ndarray, q: np.ndarray, k: int) -> list[int]:
    """Ground truth. O(n·d) - the thing HNSW is approximating."""
    return list(np.argsort(-(X @ q))[:k])


def recall_at_k(approx: list[int], truth: list[int]) -> float:
    return len(set(approx) & set(truth)) / len(truth)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, dim, k = 2000, 64, 10
    X = normalize(rng.normal(size=(n, dim)).astype(np.float32))
    queries = normalize(rng.normal(size=(20, dim)).astype(np.float32))

    print(f"building HNSW: n={n} dim={dim} M=16 ef_construction=100")
    index = Hnsw(dim=dim, M=16, ef_construction=100, seed=0).build(X)
    mem = index.memory_bytes()
    print(f"  layers={mem['layers']}  links={index.link_count():,}  "
          f"vector bytes={mem['vectors']:,}  link bytes={mem['links']:,}")

    for ef in (10, 32, 128):
        index.distance_calls = 0
        rec = np.mean([recall_at_k(index.search(q, k, ef), exact_knn(X, q, k)) for q in queries])
        print(f"  ef_search={ef:4d}  recall@{k}={rec:.3f}  "
              f"distance calls/query={index.distance_calls / len(queries):7.0f}  "
              f"(exact would be {n})")
