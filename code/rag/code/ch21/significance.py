"""Chapter 21 - Is that improvement real, or did you re-run the dice?

Takes two saved evaluation runs from data/eval_results/ (written by
code/ch10/run_eval.py), pairs them question by question, and answers three
questions an interviewer will ask:

  1. What is the difference, with a 95% confidence interval?   (paired bootstrap)
  2. Could that difference have happened by chance?            (paired permutation test)
  3. How many questions would I need to detect it?             (McNemar power simulation)

Everything is seeded, so the numbers below reproduce exactly. No LLM calls: this
reads the per-question rows that the eval harness already wrote.

Run:  uv run python code/ch21/significance.py
      uv run python code/ch21/significance.py --a k3_cs800 --b k4_cs800_rerank
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "data" / "eval_results"
SEED = 20260916

# Metrics worth testing. Binary ones can also use the McNemar machinery below.
METRICS = ["recall@k", "ndcg@k", "precision@k", "keyword_hit", "faithfulness", "correct", "citation_acc"]


def load_pairs(name_a: str, name_b: str, metric: str) -> tuple[list[float], list[float], list[str]]:
    """Per-question values for the SAME questions in both runs, in the same order.

    Pairing is the whole point: question q17 is hard in every configuration, so
    comparing means of two independent samples throws away the information that
    the same questions appear in both."""
    def rows(name: str) -> dict[str, dict]:
        data = json.loads((RESULTS / f"{name}.json").read_text())
        return {r["id"]: r for r in data["per_question"] if r["answerable"]}

    ra, rb = rows(name_a), rows(name_b)
    ids = [i for i in sorted(ra.keys() & rb.keys())
           if ra[i].get(metric) is not None and rb[i].get(metric) is not None]
    return ([float(ra[i][metric]) for i in ids], [float(rb[i][metric]) for i in ids], ids)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


# ------------------------------------------------------------- bootstrap ---
def paired_bootstrap_ci(a: list[float], b: list[float], trials: int = 10_000,
                        alpha: float = 0.05, seed: int = SEED) -> tuple[float, float]:
    """95% CI for mean(b) - mean(a).

    Resample QUESTIONS (not values) with replacement, keeping each question's two
    measurements together. The spread of the resampled differences estimates how
    much the difference would move if you had drawn a different golden set of the
    same size. No normality assumption, works for any metric in [0, 1]."""
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(trials):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(mean([b[i] for i in idx]) - mean([a[i] for i in idx]))
    diffs.sort()
    lo = diffs[int((alpha / 2) * trials)]
    hi = diffs[int((1 - alpha / 2) * trials) - 1]
    return lo, hi


# ----------------------------------------------------------- permutation ---
def paired_permutation_p(a: list[float], b: list[float], trials: int = 10_000,
                         seed: int = SEED) -> float:
    """Two-sided p-value for 'the two configurations are the same'.

    Under that null hypothesis the two measurements of one question are
    exchangeable: it is a coin flip which system produced which. So flip each
    pair independently, many times, and see how often the shuffled difference is
    at least as extreme as the observed one. 10,000 draws out of 2^n sign patterns
    is a Monte Carlo estimate, not a full enumeration, but the (count+1)/(trials+1)
    form keeps the test valid at its nominal level. No t-test assumptions."""
    rng = random.Random(seed + 1)
    observed = abs(mean(b) - mean(a))
    per_q = [bi - ai for ai, bi in zip(a, b)]
    count = 0
    for _ in range(trials):
        flipped = mean([d if rng.random() < 0.5 else -d for d in per_q])
        if abs(flipped) >= observed - 1e-12:
            count += 1
    return (count + 1) / (trials + 1)      # add-one: never report p = 0


# ------------------------------------------------------ McNemar + power ---
def binom_two_sided_p(successes: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value, computed with math.comb (no scipy)."""
    if n == 0:
        return 1.0
    def pmf(x: int) -> float:
        return math.comb(n, x) * p ** x * (1 - p) ** (n - x)
    obs = pmf(successes)
    return min(1.0, sum(pmf(x) for x in range(n + 1) if pmf(x) <= obs + 1e-12))


def mcnemar(a: list[float], b: list[float]) -> tuple[int, int, float]:
    """For a BINARY metric: count the discordant pairs and test them.

    Questions where both systems succeed (or both fail) carry no information
    about which is better - McNemar's insight. Only b_wins and a_wins matter,
    and under the null they split 50/50."""
    a_wins = sum(1 for x, y in zip(a, b) if x > y)
    b_wins = sum(1 for x, y in zip(a, b) if y > x)
    return a_wins, b_wins, binom_two_sided_p(b_wins, a_wins + b_wins)


def power_for(n: int, delta: float, discordance: float, trials: int = 2000,
              alpha: float = 0.05, seed: int = SEED) -> float:
    """Probability of detecting a true effect `delta` with `n` questions.

    Model: a fraction `discordance` of questions are answered differently by the
    two systems; of those, the split is tilted to produce the true effect delta.
    Simulate, run the exact McNemar test, count rejections. `discordance` is
    measured from the real runs, so this is calibrated to this eval set, not to
    a textbook."""
    rng = random.Random(seed + 2)
    p_b = (discordance + delta) / 2          # P(only B correct)
    p_a = (discordance - delta) / 2          # P(only A correct)
    if p_a < 0 or p_b < 0:
        return float("nan")
    hits = 0
    for _ in range(trials):
        a_wins = b_wins = 0
        for _ in range(n):
            u = rng.random()
            if u < p_b:
                b_wins += 1
            elif u < p_b + p_a:
                a_wins += 1
        if binom_two_sided_p(b_wins, a_wins + b_wins) < alpha:
            hits += 1
    return hits / trials


def mde(n: int, discordance: float) -> float:
    """Smallest true effect this many questions can detect at 80% power.

    An effect can never exceed the discordance rate: if two systems answer the
    same way on 95% of questions, the most either can win by is 5 points."""
    for delta in [d / 100 for d in range(1, 61)]:
        if delta > discordance:
            return float("nan")
        if power_for(n, delta, discordance, trials=600) >= 0.80:
            return delta
    return float("nan")


# ------------------------------------------------------------------ main ---
def compare(name_a: str, name_b: str) -> None:
    print("=" * 92)
    print(f"A = {name_a}      B = {name_b}      (paired on the same golden questions)")
    print("=" * 92)
    print(f"\n{'metric':<16}{'n':>4}{'mean A':>9}{'mean B':>9}{'B - A':>9}"
          f"{'95% CI':>20}{'p':>8}   verdict")
    for metric in METRICS:
        a, b, ids = load_pairs(name_a, name_b, metric)
        if not a:
            continue
        diff = mean(b) - mean(a)
        lo, hi = paired_bootstrap_ci(a, b)
        p = paired_permutation_p(a, b)
        # A CI that straddles zero means the sign of the difference is not established.
        verdict = "SIGNIFICANT" if p < 0.05 else ("no evidence" if lo <= 0 <= hi else "borderline")
        print(f"{metric:<16}{len(a):>4}{mean(a):>9.3f}{mean(b):>9.3f}{diff:>+9.3f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>20}{p:>8.3f}   {verdict}")

    # ---- the paired structure, made visible, on the binary metric -----------
    a, b, ids = load_pairs(name_a, name_b, "correct")
    both = sum(1 for x, y in zip(a, b) if x == y == 1)
    neither = sum(1 for x, y in zip(a, b) if x == y == 0)
    a_wins, b_wins, p_mcnemar = mcnemar(a, b)
    n = len(a)
    print(f"\nPAIRED VIEW of 'correct'  (n = {n} answerable questions)")
    print(f"  {'':<18}{'B correct':>12}{'B wrong':>10}")
    print(f"  {'A correct':<18}{both:>12}{a_wins:>10}")
    print(f"  {'A wrong':<18}{b_wins:>12}{neither:>10}")
    print(f"  concordant: {both + neither}/{n} questions - they carry NO information about which")
    print(f"  system is better. McNemar's test uses only the {a_wins + b_wins} discordant pairs:")
    print(f"  A-only-right = {a_wins}, B-only-right = {b_wins}, exact two-sided p = {p_mcnemar:.3f}")
    ids_flip = [i for i, (x, y) in zip(ids, zip(a, b)) if x != y]
    print(f"  the questions that actually differ: {ids_flip}")

    # ---- how big a golden set do you need? ---------------------------------
    discordance = (a_wins + b_wins) / n
    print(f"\nHOW MANY QUESTIONS DO YOU NEED?")
    print(f"  These two runs disagree on only {a_wins + b_wins}/{n} questions "
          f"(discordance {discordance:.3f}), so the largest")
    print(f"  effect that could possibly exist between them is {discordance:.3f} - and at that")
    print(f"  discordance, {n} questions have {power_for(n, discordance, discordance, trials=600):.2f} power. "
          f"This comparison was underpowered before it started.")
    print("\n  Planning table - minimum detectable effect on a binary metric (80% power,")
    print("  alpha 0.05), as a function of how often the two systems disagree at all:")
    print(f"  {'n questions':>12}{'disagree 10%':>15}{'disagree 20%':>15}{'disagree 30%':>15}")
    for nq in (20, 42, 100, 300, 1000):
        cells = []
        for d in (0.10, 0.20, 0.30):
            m = mde(nq, d)
            cells.append(f"{m:.2f}" if m == m else "  -")
        print(f"  {nq:>12}{cells[0]:>15}{cells[1]:>15}{cells[2]:>15}")
    print("\n  Read the '20%' column: 42 questions cannot reliably detect anything smaller")
    print("  than ~20 points. Every '+2% better' claim on a 40-question golden set is noise.")
    print("  To detect 2-3 points you need hundreds to thousands of questions - or a metric")
    print("  with more resolution per question (graded relevance, per-chunk labels, partial")
    print("  credit) so that each question carries more than one bit.")

    # ---- CI width shrinks like 1/sqrt(n) -----------------------------------
    print("\nCI WIDTH vs GOLDEN-SET SIZE  (recall@k, subsampled from the real runs, seeded)")
    a, b, _ = load_pairs(name_a, name_b, "recall@k")
    rng = random.Random(SEED + 3)
    print(f"  {'n':>5}{'mean diff':>12}{'95% CI':>22}{'width':>9}")
    for nq in (10, 20, len(a)):
        idx = rng.sample(range(len(a)), nq)
        sa, sb = [a[i] for i in idx], [b[i] for i in idx]
        lo, hi = paired_bootstrap_ci(sa, sb, trials=4000)
        print(f"  {nq:>5}{mean(sb) - mean(sa):>+12.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>22}{hi - lo:>9.3f}")
    print("  → halving the interval takes 4x the questions. Budget your labelling accordingly.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="k3_cs800", help="baseline run name in data/eval_results")
    ap.add_argument("--b", default="k6_cs800", help="candidate run name")
    args = ap.parse_args()
    compare(args.a, args.b)
