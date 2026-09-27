"""Chapter 23 - every way to combine two ranked lists, measured on the golden set.

    QDRANT_MODE=memory uv run python code/ch23/fusion_methods.py

  CombSUM/CombMNZ  : normalise scores, then add (CombMNZ multiplies by #lists hit)
  z-score fusion   : standardise, then add
  DBSF             : Qdrant's distribution-based fusion (mean +/- 3 sigma bounds)
  RRF              : ignore scores entirely, add 1/(k + rank)
  weighted RRF     : same, with a per-retriever weight

Then the breakdown that actually tells you something: which question TYPES each
retriever wins, using the golden set's `tags`.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrievers import BM25, dense_scores, evaluate, load_corpus   # noqa: E402

K = 5


# ------------------------------------------------------------- fusion rules --
def to_ranks(scores: np.ndarray) -> np.ndarray:
    """rank 1 = best. Ties broken by index, deterministically."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.int32)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def rrf(score_lists: list[np.ndarray], k: int = 60, weights: list[float] | None = None) -> np.ndarray:
    w = weights or [1.0] * len(score_lists)
    return sum(wi / (k + to_ranks(s)) for wi, s in zip(w, score_lists))


def minmax(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def combsum(score_lists: list[np.ndarray]) -> np.ndarray:
    return sum(minmax(s) for s in score_lists)


def combmnz(score_lists: list[np.ndarray], topn: int = 20) -> np.ndarray:
    """CombSUM x (number of lists in which the doc appears in the top-n)."""
    hits = sum((to_ranks(s) <= topn).astype(np.float32) for s in score_lists)
    return combsum(score_lists) * hits


def zsum(score_lists: list[np.ndarray]) -> np.ndarray:
    return sum((s - s.mean()) / (s.std() + 1e-9) for s in score_lists)


def dbsf(score_lists: list[np.ndarray]) -> np.ndarray:
    """Qdrant's DBSF: normalise each list to [0,1] using mean +/- 3 sigma as the
    bounds (robust to a single outlier, unlike min-max), then sum."""
    out = np.zeros_like(score_lists[0])
    for s in score_lists:
        lo, hi = s.mean() - 3 * s.std(), s.mean() + 3 * s.std()
        out = out + np.clip((s - lo) / (hi - lo + 1e-9), 0, 1)
    return out


# ------------------------------------------------------------------- study --
def weight_curve() -> None:
    """What k does to the shape of the RRF weight."""
    print("=" * 74)
    print("RRF weight 1/(k+rank) by rank, for several k")
    print("=" * 74)
    ranks = [1, 2, 3, 5, 10, 20, 50]
    print(f"{'k':>6} " + "".join(f"{('r' + str(r)):>9}" for r in ranks) + f"{'r1/r10':>9}")
    print("-" * 74)
    for k in (0, 1, 10, 60, 200):
        w = [1 / (k + r) for r in ranks]
        ratio = (1 / (k + 1)) / (1 / (k + 10))
        print(f"{k:>6} " + "".join(f"{x:>9.4f}" for x in w) + f"{ratio:>9.1f}x")
    print("""
k is a flattener. At k=0 rank 1 is worth 10x rank 10, so a single retriever's top
hit decides the fusion. At k=60 it is worth 1.1x - agreement between retrievers
matters more than one retriever's confidence. That is the design intent: RRF
assumes both lists are informative but neither is authoritative.""")


def main() -> None:
    chunks, texts, doc_ids, golden = load_corpus()
    questions = [g["question"] for g in golden]
    D = dense_scores(texts, questions)
    bm = BM25(texts)
    B = np.stack([bm.scores(q) for q in questions])

    weight_curve()

    methods = {
        "dense only":      lambda qi: D[qi],
        "bm25 only":       lambda qi: B[qi],
        "CombSUM":         lambda qi: combsum([D[qi], B[qi]]),
        "CombMNZ":         lambda qi: combmnz([D[qi], B[qi]]),
        "z-score sum":     lambda qi: zsum([D[qi], B[qi]]),
        "DBSF":            lambda qi: dbsf([D[qi], B[qi]]),
        "RRF k=60":        lambda qi: rrf([D[qi], B[qi]], k=60),
        "RRF k=0":         lambda qi: rrf([D[qi], B[qi]], k=0),
        "RRF k=200":       lambda qi: rrf([D[qi], B[qi]], k=200),
        "wRRF 2:1 dense":  lambda qi: rrf([D[qi], B[qi]], k=60, weights=[2.0, 1.0]),
        "wRRF 1:2 bm25":   lambda qi: rrf([D[qi], B[qi]], k=60, weights=[1.0, 2.0]),
    }

    print("\n" + "=" * 74)
    print(f"fusion methods over {len(golden)} answerable golden questions (doc-level labels)")
    print("=" * 74)
    print(f"{'method':<16} {'recall@5':>9} {'MRR':>7} {'nDCG@5':>8}")
    print("-" * 74)
    results = {}
    for name, fn in methods.items():
        m = evaluate(lambda qi, fn=fn: list(np.argsort(-fn(qi))), doc_ids, golden, k=K)
        results[name] = m
        print(f"{name:<16} {m['recall@5']:>9.3f} {m['mrr']:>7.3f} {m['ndcg@5']:>8.3f}")

    print("""
Read this honestly: on 14 clean documents with well-formed questions, dense
retrieval is already near-perfect, so fusion has almost no headroom (Chapter 8
made the same point). What the table DOES show is the failure ordering - the
score-based methods sit below the rank-based ones, because they are still being
distorted by the scale problem the previous script measured.""")

    # ------------------------------------------------------ tag breakdown --
    print("\n" + "=" * 74)
    print("where each retriever wins: MRR by question tag")
    print("=" * 74)
    by_tag: dict[str, list[int]] = defaultdict(list)
    for qi, g in enumerate(golden):
        for t in g.get("tags", []):
            by_tag[t].append(qi)

    shown = {"dense only": D, "bm25 only": B}
    print(f"{'tag':<12} {'n':>3} " + "".join(f"{n:>13}" for n in shown) + f"{'RRF k=60':>11}")
    print("-" * 74)
    for tag, idxs in sorted(by_tag.items(), key=lambda kv: -len(kv[1])):
        if len(idxs) < 3:
            continue
        sub = [golden[i] for i in idxs]
        row = f"{tag:<12} {len(idxs):>3} "
        for name, S in shown.items():
            m = evaluate(lambda j, S=S, idxs=idxs: list(np.argsort(-S[idxs[j]])), doc_ids, sub, k=K)
            row += f"{m['mrr']:>13.3f}"
        m = evaluate(lambda j, idxs=idxs: list(np.argsort(-rrf([D[idxs[j]], B[idxs[j]]]))),
                     doc_ids, sub, k=K)
        row += f"{m['mrr']:>11.3f}"
        print(row)

    # ------------------------------------------- the case fusion is built for --
    print("\n" + "=" * 74)
    print("the queries where the two retrievers most disagree")
    print("=" * 74)
    print(f"{'question':<46} {'dense r':>8} {'bm25 r':>7} {'RRF r':>6}")
    print("-" * 74)
    rows = []
    for qi, g in enumerate(golden):
        gold = set(g["sources"])

        def first_gold_rank(scores):
            for r, i in enumerate(np.argsort(-scores), start=1):
                if doc_ids[i] in gold:
                    return r
            return 999

        rd, rb = first_gold_rank(D[qi]), first_gold_rank(B[qi])
        rf = first_gold_rank(rrf([D[qi], B[qi]]))
        rows.append((abs(rd - rb), questions[qi], rd, rb, rf))
    for _, q, rd, rb, rf in sorted(rows, reverse=True)[:6]:
        print(f"{q[:44]:<46} {rd:>8} {rb:>7} {rf:>6}")
    print("""
This is the mechanism in one table: when one retriever puts the right chunk at
rank 1 and the other buries it, RRF lands between them but near the winner -
1/61 from the good list plus a small contribution from the bad one still beats a
chunk that neither list liked. Fusion is insurance against one retriever's blind
spot, not a way to beat the better retriever on its own ground.""")

    # ------------------------------------------------ where BM25 actually wins --
    # The golden set is paraphrase-style questions, which is dense retrieval's home
    # ground. The lexical case needs lexical queries: identifiers, commands, codes.
    print("\n" + "=" * 74)
    print("the other direction: exact-identifier queries (BM25's home ground)")
    print("=" * 74)
    probes = [
        ("BEACON-4187", "11-release-notes-beacon-4.2"),
        ("beaconctl rollout undo", "08-runbook-fleet-outage"),
        ("ISO 3691-4", "04-atlas-a2-specification"),
        ("Veldmark", "12-postmortem-2026-03-rotterdam-halt"),
        ("firmware 3.8", "11-release-notes-beacon-4.2"),
    ]
    emb_q = dense_scores(texts, [p[0] for p in probes], tag="idprobe")
    print(f"{'query':<24} {'gold doc':<34} {'dense r':>8} {'bm25 r':>7} {'RRF r':>6}")
    print("-" * 74)
    wins_b = wins_d = 0
    for j, (q, gold_doc) in enumerate(probes):
        bs = bm.scores(q)

        def rank_of(scores, gold_doc=gold_doc):
            for r, i in enumerate(np.argsort(-scores), start=1):
                if doc_ids[i] == gold_doc:
                    return r
            return 999

        rd, rb = rank_of(emb_q[j]), rank_of(bs)
        rf = rank_of(rrf([emb_q[j], bs]))
        wins_b += rb < rd
        wins_d += rd < rb
        print(f"{q:<24} {gold_doc[:32]:<34} {rd:>8} {rb:>7} {rf:>6}")
    print(f"\nBM25 ranked the gold chunk higher on {wins_b}/{len(probes)} identifier queries; "
          f"dense on {wins_d}.")
    print("""
That is the asymmetry hybrid search exists for. Embeddings map 'BEACON-4187' to a
region about Beacon-ish things; the exact token is not special to them. BM25 has
no notion of meaning but treats a rare token as overwhelming evidence, because
that is precisely what IDF encodes. Neither is right in general - which is why
you run both and fuse, and why a corpus of ids, SKUs, error codes or commands
makes hybrid non-optional.""")


if __name__ == "__main__":
    main()
