"""Chapter 10 - Every retrieval metric, by hand, with its formula.

All functions take:
  retrieved : list of ids in RANK ORDER (best first), e.g. doc_ids of the top-k chunks
  relevant  : set of ids that count as correct (from the golden set's `sources`)

Nothing here calls an LLM. Run the file to see a worked example with the numbers.

Run:  uv run python code/ch10/metrics.py
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def unique_in_order(ids: Sequence[str]) -> list[str]:
    """Collapse a ranked list of CHUNK ids mapped to DOC ids into a ranked list of
    unique docs (first occurrence wins). Rank metrics (MRR, AP, nDCG) assume each
    item appears once; five chunks of the same doc are one hit, not five."""
    seen, out = set(), []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ------------------------------------------------------ set-based @k metrics ---
def hit_rate_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """1 if ANY of the top-k is relevant, else 0.  (a.k.a. success@k, "did we find it at all")
    Averaged over questions it answers: how often does the right doc appear in the window?"""
    return 1.0 if any(r in relevant for r in retrieved[:k]) else 0.0


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """|relevant ∩ top-k| / |relevant|.   "Of everything we SHOULD have found, how much did we?"
    Low recall → the answer is not in the context → the LLM cannot answer correctly."""
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """|relevant ∩ top-k| / k.   "Of what we showed the LLM, how much was useful?"
    Low precision → context full of noise → distraction, cost, and hallucination risk."""
    if k == 0:
        return 0.0
    return len([r for r in retrieved[:k] if r in relevant]) / k


def f1_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Harmonic mean of precision@k and recall@k: 2PR / (P + R). One number when you must pick k."""
    p, r = precision_at_k(retrieved, relevant, k), recall_at_k(retrieved, relevant, k)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


# ---------------------------------------------------- rank-aware metrics ---
def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """1 / rank of the FIRST relevant result (rank starts at 1); 0 if none.
    MRR = mean over questions. Cares only about the first hit - right for
    'one good chunk is enough' use cases."""
    for i, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def average_precision(retrieved: Sequence[str], relevant: set[str]) -> float:
    """AP = (1/|relevant|) * Σ_{ranks i where item i is relevant} precision@i.
    MAP = mean over questions. Rewards putting ALL relevant items early."""
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for i, r in enumerate(retrieved, start=1):
        if r in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant)


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """Discounted Cumulative Gain: Σ_{i=1..k} gain_i / log2(i + 1).
    Position 1 has discount 1, position 2 → 0.63, position 3 → 0.5 ..."""
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains[:k], start=1))


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """nDCG@k = DCG@k / IDCG@k, where IDCG is the DCG of the ideal ordering
    (all relevant items first). Binary gains here (1 if relevant). Range 0..1.
    The standard metric when ORDER within the top-k matters (it does: LLMs
    attend more to the beginning of the context - 'lost in the middle')."""
    retrieved = unique_in_order(retrieved)
    gains = [1.0 if r in relevant else 0.0 for r in retrieved[:k]]
    # ideal ordering: all relevant items first (capped at k)
    n_ideal_ones = min(len(relevant), k)
    ideal = [1.0] * n_ideal_ones + [0.0] * (k - n_ideal_ones)
    idcg = dcg_at_k(ideal, k)
    return 0.0 if idcg == 0 else dcg_at_k(gains, k) / idcg


# ------------------------------------------------- RAGAS-style context metrics ---
def context_precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """RAGAS 'context precision':  Σ_{i=1..k} (precision@i · rel_i) / (number of relevant items in top-k).
    Like AP but normalised by relevant-found instead of relevant-total → 'were the
    useful chunks ranked at the top of the window?'. Equals 1.0 if all relevant chunks
    come before all irrelevant ones."""
    rel = [1 if r in relevant else 0 for r in retrieved[:k]]
    found = sum(rel)
    if found == 0:
        return 0.0
    return sum(precision_at_k(retrieved, relevant, i) * rel[i - 1] for i in range(1, len(rel) + 1)) / found


def context_recall_docs(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Document-level context recall: fraction of gold SOURCE DOCS that have at least
    one chunk in the top-k. (RAGAS's version is claim-level and needs an LLM: fraction
    of golden-answer sentences attributable to the context - see llm_judges.py.)"""
    return recall_at_k(retrieved, relevant, k)


# ----------------------------------------------------- operational metrics ---
def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile. p50 = median, p95 = 'the slow tail users notice'."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, math.ceil(p / 100 * len(s)) - 1)
    return s[idx]


# $ per 1M tokens (input, output) - official OpenAI pricing page, Sep 2026.
PRICES = {
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.5": (5.00, 30.00),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """usage_metadata → dollars. Unknown model → 0 (and you should add it to PRICES)."""
    pin, pout = PRICES.get(model, (0.0, 0.0))
    return input_tokens / 1e6 * pin + output_tokens / 1e6 * pout


# ---------------------------------------------------------- worked example ---
if __name__ == "__main__":
    # A question whose answer lives in docs A and C. Our retriever returned 5 chunks
    # from docs: B, A, D, C, E  (so relevant items sit at ranks 2 and 4).
    retrieved = ["B", "A", "D", "C", "E"]
    relevant = {"A", "C"}
    k = 5
    print("retrieved (ranked):", retrieved, "  relevant:", sorted(relevant), f"  k={k}\n")
    rows = [
        ("hit_rate@5", hit_rate_at_k(retrieved, relevant, k), "1.0 - at least one relevant in top-5"),
        ("recall@5", recall_at_k(retrieved, relevant, k), "2 found / 2 relevant = 1.0"),
        ("precision@5", precision_at_k(retrieved, relevant, k), "2 relevant / 5 shown = 0.4"),
        ("f1@5", f1_at_k(retrieved, relevant, k), "2·1·0.4/(1+0.4) = 0.571"),
        ("recall@3", recall_at_k(retrieved, relevant, 3), "only A in top-3 → 1/2 = 0.5"),
        ("MRR (1 q)", reciprocal_rank(retrieved, relevant), "first relevant at rank 2 → 0.5"),
        ("AP", average_precision(retrieved, relevant), "(P@2 + P@4)/2 = (0.5 + 0.5)/2 = 0.5"),
        ("nDCG@5", ndcg_at_k(retrieved, relevant, k), "(1/log2 3 + 1/log2 5)/(1/log2 2 + 1/log2 3) ≈ 0.651"),
        ("context_precision@5", context_precision_at_k(retrieved, relevant, k), "(P@2·1 + P@4·1)/2 = 0.5"),
    ]
    for name, val, why in rows:
        print(f"  {name:<22}{val:6.3f}   {why}")
    print("\nlatencies p50/p95 of [0.8,1.1,0.9,3.2,1.0]s:", percentile([0.8, 1.1, 0.9, 3.2, 1.0], 50), percentile([0.8, 1.1, 0.9, 3.2, 1.0], 95))
    print("cost of 2,000 in + 300 out tokens on gpt-5.4-mini: $%.5f" % cost_usd("gpt-5.4-mini", 2000, 300))


# ------------------------------------------------- aliases used in Appendix B ---
def mrr(retrieved_lists: Sequence[Sequence[str]], relevant_sets: Sequence[set[str]]) -> float:
    """Mean Reciprocal Rank over many questions."""
    pairs = list(zip(retrieved_lists, relevant_sets))
    return sum(reciprocal_rank(r, s) for r, s in pairs) / len(pairs) if pairs else 0.0


def mean_average_precision(retrieved_lists: Sequence[Sequence[str]], relevant_sets: Sequence[set[str]]) -> float:
    """MAP over many questions."""
    pairs = list(zip(retrieved_lists, relevant_sets))
    return sum(average_precision(r, s) for r, s in pairs) / len(pairs) if pairs else 0.0


context_precision = context_precision_at_k
context_recall = context_recall_docs

try:  # generation judges live in llm_judges.py (same folder); re-exported here for convenience
    from llm_judges import answer_correctness, answer_relevance, faithfulness  # noqa: F401
except ImportError:  # pragma: no cover
    pass
