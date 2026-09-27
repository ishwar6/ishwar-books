"""Chapter 21 - What each retrieval metric can and cannot see.

Chapter 10 defined the metrics. This file shows you where they DISAGREE, because
that is what an interviewer probes. Two retrievers can have identical recall@5 and
be wildly different systems; two can have identical MRR and differ in recall.

Nothing here calls an LLM or a database: every number is computed from a ranked
list of document ids, so the output is identical on every machine, every run.

Run:  uv run python code/ch21/metric_illusions.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ch10"))  # reuse Chapter 10's metrics
import metrics as M  # noqa: E402

K = 5


def row(name: str, retrieved: list[str], relevant: set[str], k: int = K) -> dict:
    """Every metric for one ranked list. `retrieved` is doc ids, best first."""
    return {
        "name": name,
        "ranking": " ".join(f"*{r}*" if r in relevant else r for r in retrieved[:k]),
        "hit": M.hit_rate_at_k(retrieved, relevant, k),
        "recall": M.recall_at_k(retrieved, relevant, k),
        "prec": M.precision_at_k(retrieved, relevant, k),
        "mrr": M.reciprocal_rank(retrieved, relevant),
        "map": M.average_precision(retrieved, relevant),
        "ndcg": M.ndcg_at_k(retrieved, relevant, k),
        "ctxp": M.context_precision_at_k(retrieved, relevant, k),
    }


def table(title: str, rows: list[dict], lesson: str) -> None:
    print(f"\n{title}")
    print(f"  {'system':<10}{'top-5 (relevant starred)':<28}{'hit':>5}{'recall':>8}{'prec':>7}"
          f"{'MRR':>7}{'MAP':>7}{'nDCG':>7}{'ctxP':>7}")
    for r in rows:
        print(f"  {r['name']:<10}{r['ranking']:<28}{r['hit']:>5.2f}{r['recall']:>8.3f}{r['prec']:>7.3f}"
              f"{r['mrr']:>7.3f}{r['map']:>7.3f}{r['ndcg']:>7.3f}{r['ctxp']:>7.3f}")
    print(f"  → {lesson}")


def spread(rows: list[dict], metric: str) -> float:
    vals = [r[metric] for r in rows]
    return max(vals) - min(vals)


# ---------------------------------------------------------------------------
# Case 1. Same recall, same precision, very different user experience.
#   Both systems return the same TWO relevant docs inside the top 5 - recall and
#   precision literally cannot tell them apart. Only rank-aware metrics can.
# ---------------------------------------------------------------------------
def case_ranking_order() -> None:
    relevant = {"A", "C"}
    rows = [
        row("top", ["A", "C", "X", "Y", "Z"], relevant),      # gold first
        row("middle", ["X", "A", "C", "Y", "Z"], relevant),
        row("bottom", ["X", "Y", "Z", "A", "C"], relevant),   # gold last
    ]
    table("CASE 1 - identical recall@5 AND precision@5, three different systems",
          rows,
          f"recall spread {spread(rows, 'recall'):.3f}, precision spread {spread(rows, 'prec'):.3f}, "
          f"but MRR spread {spread(rows, 'mrr'):.3f} and nDCG spread {spread(rows, 'ndcg'):.3f}.")
    print("     'bottom' feeds the LLM three distractors before the evidence. An LLM reads in")
    print("     order and attends unevenly ('lost in the middle'), so this is a real difference")
    print("     that recall@5 reports as zero difference.")


# ---------------------------------------------------------------------------
# Case 2. Same MRR = 1.0, completely different recall.
#   MRR only looks at the FIRST relevant hit. For a multi-source question that is
#   exactly the wrong thing to look at.
# ---------------------------------------------------------------------------
def case_mrr_blindness() -> None:
    relevant = {"A", "B", "C"}     # a question whose answer needs all three docs
    rows = [
        row("complete", ["A", "B", "C", "X", "Y"], relevant),
        row("one-hit", ["A", "X", "Y", "Z", "W"], relevant),
    ]
    table("CASE 2 - identical MRR (1.000) and identical hit rate (1.00)", rows,
          f"recall spread {spread(rows, 'recall'):.3f} - 'one-hit' has 1/3 of the evidence.")
    print("     A question needing 3 sources scores a perfect MRR with 1 source retrieved.")
    print("     Report MRR only when one good chunk is genuinely enough.")


# ---------------------------------------------------------------------------
# Case 3. Same hit rate = 1.0, and hit rate is what most teams actually report.
# ---------------------------------------------------------------------------
def case_hit_rate_hides() -> None:
    relevant = {"A", "B", "C"}
    rows = [
        row("good", ["A", "B", "C", "X", "Y"], relevant),
        row("lucky", ["X", "Y", "Z", "W", "C"], relevant),
    ]
    table("CASE 3 - identical hit rate@5 (1.00)", rows,
          f"recall spread {spread(rows, 'recall'):.3f}, MRR spread {spread(rows, 'mrr'):.3f}, "
          f"nDCG spread {spread(rows, 'ndcg'):.3f}.")
    print("     'lucky' scraped one relevant doc into last place. Hit rate is the most")
    print("     flattering metric in RAG and the one most often quoted in a standup.")


# ---------------------------------------------------------------------------
# Case 4. Precision and recall move in opposite directions with k.
#   Same retriever, same ranking - only the window changes.
# ---------------------------------------------------------------------------
def case_k_tradeoff() -> None:
    retrieved = ["A", "X", "B", "Y", "Z", "C", "W", "V"]
    relevant = {"A", "B", "C"}
    print("\nCASE 4 - one ranking, one retriever, k is the only variable")
    print(f"  ranking: {' '.join('*' + r + '*' if r in relevant else r for r in retrieved)}")
    print(f"  {'k':>3}{'recall@k':>10}{'prec@k':>9}{'f1@k':>8}{'nDCG@k':>9}")
    for k in (1, 3, 5, 8):
        print(f"  {k:>3}{M.recall_at_k(retrieved, relevant, k):>10.3f}"
              f"{M.precision_at_k(retrieved, relevant, k):>9.3f}"
              f"{M.f1_at_k(retrieved, relevant, k):>8.3f}"
              f"{M.ndcg_at_k(retrieved, relevant, k):>9.3f}")
    print("  → recall rises monotonically with k and precision falls. Quoting 'recall@k'")
    print("    without k is meaningless: recall@100 is nearly free and nearly useless.")


# ---------------------------------------------------------------------------
# Case 5. nDCG worked by hand, so you can derive it on a whiteboard.
# ---------------------------------------------------------------------------
def case_ndcg_by_hand() -> None:
    retrieved = ["B", "A", "D", "C", "E"]
    relevant = {"A", "C"}
    print("\nCASE 5 - nDCG@5 derived step by step (the whiteboard version)")
    print(f"  retrieved: {retrieved}   relevant: {sorted(relevant)}")
    print("\n  DCG@k = Σ  gain_i / log2(i + 1)        gain_i = 1 if item i is relevant else 0")
    print(f"  {'rank i':>7}{'doc':>5}{'gain':>6}{'log2(i+1)':>11}{'discount':>10}{'contribution':>14}")
    dcg = 0.0
    for i, d in enumerate(retrieved, start=1):
        gain = 1.0 if d in relevant else 0.0
        disc = math.log2(i + 1)
        contrib = gain / disc
        dcg += contrib
        print(f"  {i:>7}{d:>5}{gain:>6.0f}{disc:>11.3f}{1 / disc:>10.3f}{contrib:>14.3f}")
    print(f"  DCG@5 = {dcg:.4f}")
    print("\n  IDCG = the DCG of the BEST POSSIBLE ordering (both relevant docs first):")
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    print(f"  IDCG@5 = 1/log2(2) + 1/log2(3) = {1 / math.log2(2):.4f} + {1 / math.log2(3):.4f} = {idcg:.4f}")
    print(f"  nDCG@5 = DCG/IDCG = {dcg:.4f} / {idcg:.4f} = {dcg / idcg:.4f}")
    print(f"  library agrees: {M.ndcg_at_k(retrieved, relevant, 5):.4f}")
    print("\n  Why log2(i+1)? It is a smooth, slowly decaying discount with discount(rank 1) = 1.")
    print("  Position weights: " + "  ".join(f"r{i}={1 / math.log2(i + 1):.2f}" for i in range(1, 6)))
    print("  Rank 1 is worth 1.00, rank 5 only 0.43 - that ratio is the modelling assumption.")
    print("  Normalising by IDCG is what makes a 1-gold question comparable with a 3-gold one.")


# ---------------------------------------------------------------------------
# Case 6. Graded relevance changes the ORDER of the leaderboard.
#   Binary labels say two systems tie; graded labels say one is clearly better.
# ---------------------------------------------------------------------------
def dcg_graded(gains: list[float], k: int) -> float:
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains[:k], start=1))


def ndcg_graded(retrieved: list[str], grades: dict[str, float], k: int) -> float:
    gains = [grades.get(d, 0.0) for d in retrieved[:k]]
    ideal = sorted(grades.values(), reverse=True)[:k] + [0.0] * k
    idcg = dcg_graded(ideal, k)
    return 0.0 if idcg == 0 else dcg_graded(gains, k) / idcg


def case_graded_relevance() -> None:
    # A: the chunk with the exact number. B: same document, adjacent section (useful
    # context but does not contain the answer). Binary labelling calls both "relevant".
    grades = {"A": 2.0, "B": 1.0}
    binary = {"A", "B"}
    sys1 = ["A", "B", "X", "Y", "Z"]   # exact-answer chunk first
    sys2 = ["B", "A", "X", "Y", "Z"]   # merely-related chunk first
    print("\nCASE 6 - binary labels tie; graded labels break the tie")
    print("  grades: A=2 (contains the exact answer), B=1 (same doc, related section)")
    print(f"  {'system':<13}{'ranking':<22}{'binary nDCG@5':>15}{'graded nDCG@5':>16}")
    for name, r in [("answer-1st", sys1), ("related-1st", sys2)]:
        print(f"  {name:<13}{' '.join(r):<22}{M.ndcg_at_k(r, binary, K):>15.3f}{ndcg_graded(r, grades, K):>16.3f}")
    print("  → binary nDCG: identical (both put 'relevant' docs at ranks 1-2).")
    print("    graded nDCG: answer-1st wins. If your users need the number, not the topic,")
    print("    binary labels cannot express your product requirement.")


# ---------------------------------------------------------------------------
# What each metric is blind to - the summary an interviewer wants.
# ---------------------------------------------------------------------------
def blindness_table() -> None:
    print("\nWHAT EACH METRIC IS BLIND TO")
    print(f"  {'metric':<20}{'sees':<34}{'blind to'}")
    rows = [
        ("hit rate@k", "did anything relevant appear", "how much, how high, how noisy"),
        ("recall@k", "coverage of the gold set", "rank order, precision, k itself"),
        ("precision@k", "noise fraction in the window", "rank order, whether gold is missing"),
        ("MRR", "rank of the FIRST hit", "every other relevant document"),
        ("MAP", "all relevant docs, rank-weighted", "graded relevance; needs complete labels"),
        ("nDCG@k", "order + graded gains, normalised", "what the generator does with it"),
        ("context precision", "are the good chunks at the top", "absolute coverage (recall)"),
    ]
    for name, sees, blind in rows:
        print(f"  {name:<20}{sees:<34}{blind}")
    print("\n  And the one every metric above is blind to: whether the ANSWER was right.")
    print("  That is Chapter 21.1 and the recall_vs_correct.py script.")


if __name__ == "__main__":
    print("=" * 96)
    print("METRIC ILLUSIONS - where two retrievers look identical and are not")
    print("=" * 96)
    case_ranking_order()
    case_mrr_blindness()
    case_hit_rate_hides()
    case_k_tradeoff()
    case_ndcg_by_hand()
    case_graded_relevance()
    blindness_table()
