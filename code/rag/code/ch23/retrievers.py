"""Chapter 23 - the shared pieces: a tunable BM25, cached dense scores, and the
three retrieval metrics. Imported by the other ch23 scripts.

Chapter 8 built a BM25 with fixed k1/b and measured hit-rate. Here k1 and b are
parameters we sweep, scores are kept (not just ranks) so we can study their
distributions, and the metrics are rank-aware.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from ragbook import DATA_DIR, chunk_documents, get_embeddings, load_golden, load_handbook

CACHE = DATA_DIR / "cache" / "ch23_dense.npz"


def tokenize(text: str) -> list[str]:
    """Keep digits, dots and hyphens inside tokens so "3.8", "BEACON-4187" and
    "beaconctl" survive as single searchable units (Chapter 8, §8.3)."""
    return re.findall(r"[a-z0-9][a-z0-9.\-]*", text.lower())


class BM25:
    """BM25 with k1 and b exposed, plus the pieces needed to plot its internals."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.toks = [tokenize(d) for d in docs]
        self.len = np.array([len(t) for t in self.toks], dtype=np.float32)
        self.avgdl = float(self.len.mean())
        self.tf = [Counter(t) for t in self.toks]
        df = Counter(term for t in self.toks for term in set(t))
        n = len(docs)
        # Lucene-style IDF: the +1 inside the log keeps it positive even for a term
        # in every document (without it, df > N/2 gives a NEGATIVE idf).
        self.idf = {t: math.log((n - c + 0.5) / (c + 0.5) + 1) for t, c in df.items()}
        self.raw_idf = {t: math.log((n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.df = df
        self.n = n

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        for t in tokenize(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in range(self.n):
                f = self.tf[i].get(t, 0)
                if not f:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avgdl)
                out[i] += idf * f * (self.k1 + 1) / (f + norm)
        return out


def load_corpus():
    """56 handbook chunks, their doc ids, and the golden questions."""
    chunks = chunk_documents(load_handbook())
    texts = [c.page_content for c in chunks]
    doc_ids = [c.metadata["doc_id"] for c in chunks]
    golden = [g for g in load_golden() if g["answerable"]]
    return chunks, texts, doc_ids, golden


def dense_scores(texts: list[str], questions: list[str], tag: str = "golden") -> np.ndarray:
    """(n_questions, n_chunks) cosine matrix, cached so sweeps are free.

    `tag` names the cache file: different query sets must not share one, or they
    overwrite each other and every run pays the embedding cost again."""
    path = CACHE.with_name(f"ch23_dense_{tag}.npz")
    if path.exists():
        d = np.load(path, allow_pickle=True)
        if len(d["texts"]) == len(texts) and len(d["questions"]) == len(questions):
            return d["S"]
    emb = get_embeddings()
    print(f"embedding {len(texts)} chunks + {len(questions)} queries [{tag}] (cached afterwards)...")
    C = np.asarray(emb.embed_documents(texts), dtype=np.float32)
    Q = np.asarray([emb.embed_query(q) for q in questions], dtype=np.float32)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    S = Q @ C.T
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, S=S, texts=np.array(texts, dtype=object),
                        questions=np.array(questions, dtype=object))
    return S


# ------------------------------------------------------------------ metrics --
def docs_in_order(ranked_chunks: list[int], doc_ids: list[str]) -> list[str]:
    """Collapse a ranked chunk list to unique doc ids, preserving order.
    Labels are document-level (Chapter 10, §10.3), so metrics must be too."""
    seen, out = set(), []
    for i in ranked_chunks:
        d = doc_ids[i]
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def recall_at_k(ranked_docs: list[str], gold: set[str], k: int) -> float:
    return len(set(ranked_docs[:k]) & gold) / len(gold)


def mrr(ranked_docs: list[str], gold: set[str]) -> float:
    for r, d in enumerate(ranked_docs, start=1):
        if d in gold:
            return 1.0 / r
    return 0.0


def ndcg_at_k(ranked_docs: list[str], gold: set[str], k: int) -> float:
    dcg = sum((1.0 if d in gold else 0.0) / math.log2(r + 1)
              for r, d in enumerate(ranked_docs[:k], start=1))
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate(rank_fn, doc_ids: list[str], golden: list[dict], k: int = 5) -> dict:
    """rank_fn(qi) -> ranked chunk indices. Returns averaged metrics."""
    rec, rr, nd = [], [], []
    for qi, g in enumerate(golden):
        gold = set(g["sources"])
        docs = docs_in_order(rank_fn(qi), doc_ids)
        rec.append(recall_at_k(docs, gold, k))
        rr.append(mrr(docs, gold))
        nd.append(ndcg_at_k(docs, gold, k))
    return {f"recall@{k}": float(np.mean(rec)), "mrr": float(np.mean(rr)),
            f"ndcg@{k}": float(np.mean(nd))}
