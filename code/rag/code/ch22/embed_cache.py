"""Chapter 22 - a cache of REAL embeddings, so the parameter sweeps can run on
realistic vectors without paying OpenAI on every run.

Random Gaussian vectors are the *hardest* case for a graph index: no cluster
structure, and in high dimensions every point is roughly equidistant from every
other ("concentration of distances"). Real embeddings are clustered and live on a
lower-dimensional manifold, so HNSW does better on them. Both belong in the study -
the synthetic set is the pessimistic bound, the real set is what you ship.

    QDRANT_MODE=memory uv run python code/ch22/embed_cache.py --n 1500

Writes data/cache/ch22_real_vectors.npz (a few MB). Cost: ~1500 short texts at
$0.02/1M tokens is well under one cent.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from ragbook import DATA_DIR, get_embeddings, load_handbook

CACHE = DATA_DIR / "cache" / "ch22_real_vectors.npz"


def collect_texts(limit: int) -> list[str]:
    """Sentence-ish units from every corpus we have on disk, deduplicated."""
    texts: list[str] = []
    for doc in load_handbook():
        texts += [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", doc.page_content)]
    for extra in ("generated", "gutenberg"):          # created by Chapter 7, optional
        d = DATA_DIR / extra
        if d.exists():
            for path in sorted(d.glob("*.*"))[:4]:
                body = path.read_text(encoding="utf-8", errors="ignore")[:400_000]
                texts += [s.strip() for s in re.split(r"(?<=[.!?])\s+", body)]

    seen, out = set(), []
    for t in texts:
        t = " ".join(t.split())
        if 40 <= len(t) <= 400 and t not in seen:     # skip headers and boilerplate
            seen.add(t)
            out.append(t)
        if len(out) >= limit:
            break
    return out


def load_or_build(n: int = 1500) -> np.ndarray:
    """Return an (n, dim) float32 array of L2-normalised real embeddings."""
    if CACHE.exists():
        X = np.load(CACHE)["X"]
        if len(X) >= n:
            return X[:n]
    texts = collect_texts(n)
    print(f"embedding {len(texts)} real text snippets (cached afterwards)...")
    vecs = get_embeddings().embed_documents(texts)
    X = np.asarray(vecs, dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, X=X)
    print(f"cached {X.shape} -> {CACHE.relative_to(DATA_DIR.parent)}")
    return X


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    args = ap.parse_args()
    X = load_or_build(args.n)
    print(f"real embedding matrix: {X.shape}  dtype={X.dtype}")

    # How clustered is it? Mean pairwise cosine on a sample says a lot: random
    # high-dimensional vectors average ~0, real text embeddings are strongly positive.
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(400, len(X)), replace=False)
    S = X[idx] @ X[idx].T
    off = S[~np.eye(len(idx), dtype=bool)]
    R = rng.normal(size=(len(idx), X.shape[1])).astype(np.float32)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    SR = R @ R.T
    offr = SR[~np.eye(len(idx), dtype=bool)]
    print(f"  mean pairwise cosine, real vectors   : {off.mean():+.3f}  (std {off.std():.3f})")
    print(f"  mean pairwise cosine, random vectors : {offr.mean():+.3f}  (std {offr.std():.3f})")
    print("  -> real embeddings are anisotropic and clustered; ANN finds them easier.")
