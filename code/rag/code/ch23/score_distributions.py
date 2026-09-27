"""Chapter 23 - why you cannot add a BM25 score to a cosine score.

    QDRANT_MODE=memory uv run python code/ch23/score_distributions.py

Prints the two score distributions, shows a query where the naive weighted sum is
decided entirely by scale rather than by relevance, and then measures the
normalisation family that tries to fix it (min-max, z-score) and its failure modes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrievers import BM25, dense_scores, load_corpus, tokenize   # noqa: E402


def histogram(vals: np.ndarray, bins: int = 10, width: int = 40, label: str = "") -> str:
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    counts, edges = np.histogram(vals, bins=bins, range=(lo, hi))
    top = counts.max() or 1
    out = [label]
    for c, e in zip(counts, edges[:-1]):
        out.append(f"  {e:8.3f} | {'#' * int(round(c / top * width))} {c}")
    return "\n".join(out)


def describe(name: str, v: np.ndarray) -> None:
    print(f"  {name:<10} min={v.min():8.3f}  max={v.max():8.3f}  mean={v.mean():7.3f}  "
          f"std={v.std():6.3f}  range={v.max() - v.min():7.3f}")


def main() -> None:
    chunks, texts, doc_ids, golden = load_corpus()
    questions = [g["question"] for g in golden]
    D = dense_scores(texts, questions)            # (n_q, n_chunks) cosine
    bm = BM25(texts)

    print("=" * 74)
    print("1. the two scales, over all questions x all chunks")
    print("=" * 74)
    B = np.stack([bm.scores(q) for q in questions])
    describe("dense", D.ravel())
    describe("bm25", B.ravel())
    print()
    print(histogram(D.ravel(), label="dense cosine distribution:"))
    print(histogram(B.ravel(), label="\nBM25 score distribution:"))
    print("""
Dense scores are bounded, dense (every pair has one), and never near zero because
embedding space is anisotropic - the floor here is the corpus's own similarity
baseline, not 0. BM25 is unbounded above, and MOSTLY EXACTLY ZERO: a chunk sharing
no query term scores nothing at all. Two different kinds of number.""")

    print("\n" + "=" * 74)
    print("2. the naive sum: 0.5*bm25 + 0.5*dense")
    print("=" * 74)
    # pick the query with the largest BM25 max - the one where scale dominates hardest
    qi = int(np.argmax(B.max(axis=1)))
    q = questions[qi]
    print(f'query: "{q}"\n')
    naive = 0.5 * B[qi] + 0.5 * D[qi]
    print(f"{'chunk':>5} {'source':<34} {'bm25':>7} {'dense':>7} {'0.5b+0.5d':>10}")
    print("-" * 74)
    for i in np.argsort(-naive)[:5]:
        print(f"{i:>5} {doc_ids[i][:34]:<34} {B[qi][i]:>7.2f} {D[qi][i]:>7.3f} {naive[i]:>10.2f}")
    contrib_b = B[qi].max() / (B[qi].max() + D[qi].max())
    print(f"\nthe BM25 term contributes {contrib_b:.0%} of the winning score; dense "
          f"{1 - contrib_b:.0%}.")
    print("The 'weights' are 0.5/0.5 but the FUSION is ~{:.0%} BM25 - the weight you".format(contrib_b))
    print("wrote is not the weight you got. Scale ate it.")

    # how far apart are the two rankings?
    top_d = set(np.argsort(-D[qi])[:5].tolist())
    top_b = set(np.argsort(-B[qi])[:5].tolist())
    print(f"\ntop-5 overlap between the two retrievers on this query: {len(top_d & top_b)}/5")

    print("\n" + "=" * 74)
    print("3. normalisation, and why it is not a fix")
    print("=" * 74)

    def minmax(v):
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)

    def zscore(v):
        return (v - v.mean()) / (v.std() + 1e-9)

    print("a) min-max makes every query's best result exactly 1.0 - even a query")
    print("   where nothing matched. Compare a query with a real lexical hit against")
    print("   one with none:\n")
    nnz = [(i, int((B[i] > 0).sum()), float(B[i].max())) for i in range(len(questions))]
    strong = max(nnz, key=lambda t: t[2])
    weak = min([t for t in nnz if t[2] > 0], key=lambda t: t[2])
    print(f"{'query':<52} {'raw max':>8} {'minmax max':>11}")
    print("-" * 74)
    for i, _, mx in (strong, weak):
        print(f"{questions[i][:50]:<52} {mx:>8.2f} {minmax(B[i]).max():>11.2f}")
    print("\n   Same 1.00 after normalisation. A retriever that found nothing now looks")
    print("   as confident as one that found an exact id match. Fusing those equally")
    print("   is how a weak retriever poisons a strong one.")

    print("\nb) z-score is sensitive to the shape of the tail. BM25's mass sits at")
    print("   exactly zero, so the mean and std are set by how many chunks failed to")
    print("   match - a property of the corpus, not of this query's relevance:\n")
    print(f"{'query':<52} {'nonzero':>8} {'z of top1':>10}")
    print("-" * 74)
    for i, nz, _ in sorted(nnz, key=lambda t: -t[1])[:3] + sorted(nnz, key=lambda t: t[1])[:2]:
        z = zscore(B[i])
        print(f"{questions[i][:50]:<52} {nz:>8} {z.max():>10.2f}")
    print("""
   The same relevance produces a different z depending on how many chunks happen
   to contain any query term. Across queries these numbers are not comparable, so
   a fixed fusion weight tuned on one query set silently mis-weights another.

Conclusion: to combine two retrievers you need a quantity that is comparable
across retrievers AND across queries. Ranks are exactly that. -> RRF, next script.""")


if __name__ == "__main__":
    main()
