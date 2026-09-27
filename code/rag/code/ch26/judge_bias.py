"""Chapter 26 - Your judge is a model too. Measure it before you trust it.

Every generation metric in this book is produced by an LLM reading an answer. That
judge has a false-positive rate, a preference for whichever option it sees first,
a weakness for long answers, and a tendency to agree with itself less often than
you would like. If you do not measure those, your eval dashboard is decorated
noise.

Four experiments, each printing a number you can quote:

  1. SELF-CONSISTENCY  same input, N times - how often does the verdict flip?
  2. POSITION BIAS     A-vs-B and B-vs-A - does the winner depend on the order?
  3. VERBOSITY BIAS    identical facts, short vs padded - does length buy a score?
  4. HUMAN AGREEMENT   Cohen's kappa against hand labels below.

  uv run python code/ch26/judge_bias.py                 # all four, ~70 calls
  uv run python code/ch26/judge_bias.py --trials 3 --only consistency
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ch10"))
import llm_judges as J  # noqa: E402

from ragbook import get_llm  # noqa: E402

# ---------------------------------------------------------------------------
# 1. SELF-CONSISTENCY
#    A citation that is objectively correct: the cited chunk literally contains
#    the sentence's fact. A perfect judge returns 1.0 every time.
# ---------------------------------------------------------------------------
CTX_CHUNK = (
    "# Paid Time Off (PTO) and Leave Policy\n\n*Effective 1 January 2026.*\n\n"
    "- Every full-time employee receives **24 days of PTO per calendar year**, accrued monthly\n"
    "  at 2 days per month.\n- Up to **5 unused days may be carried over** into the next year."
)
CITED_ANSWER = "Every full-time employee receives 24 days of PTO per calendar year. [1]"


def self_consistency(trials: int) -> None:
    """Three inputs, all objectively correct. The first two differ in how much the
    judge reads; the third differs from the second by one space. Report ACCURACY
    against the known-correct verdict, not just agreement with itself."""
    from failure_injection import CASES, fmt          # same folder: reuse the built contexts

    hard = [c for c in CASES if c["name"] == "lost_in_middle"][0]
    hard_answer = "Every full-time employee receives 24 days of PTO per calendar year. [6]"

    print("\n" + "=" * 78)
    print("1. SELF-CONSISTENCY - the same objectively-correct input, judged repeatedly")
    print("=" * 78)
    print("  Every case below cites a chunk that literally states the fact, so the correct")
    print("  verdict is 1.0 every single time. They differ in how much the judge reads -")
    print("  and, for the last two, in one character of punctuation.\n")

    # The third case is the control that tells you WHY the second one misbehaves.
    # citation_accuracy() splits the answer on sentence boundaries and keeps the
    # sentences carrying a [n]. "...year. [6]" splits into two, so the judge is
    # handed the string "[6]" with no claim in it. Glue the marker to the sentence
    # and the same judge, the same context and the same fact behave differently.
    cases = [
        ("easy: 1 chunk, cites [1]", CITED_ANSWER, [CTX_CHUNK]),
        ("hard: 10 chunks, '. [6]'", hard_answer, hard["chunks"]),
        ("same, glued: '.[6]'", hard_answer.replace(". [6]", ".[6]"), hard["chunks"]),
    ]
    results = {}
    for label, answer, chunks in cases:
        vals = [J.citation_accuracy(answer, chunks) for _ in range(trials)]
        modal = max(set(vals), key=vals.count)
        stability = vals.count(modal) / len(vals)          # does it agree with ITSELF?
        accuracy = vals.count(1.0) / len(vals)             # does it agree with the TRUTH?
        results[label] = accuracy
        flag = "" if accuracy == 1.0 else "   ← WRONG on an objectively correct citation"
        print(f"  {label:<28}{trials} trials → {vals}")
        print(f"  {'':<28}self-consistency {stability:.2f}   accuracy {accuracy:.2f}{flag}")

    faiths = [J.faithfulness(CITED_ANSWER, CTX_CHUNK)["score"] for _ in range(trials)]
    print(f"  {'faithfulness (easy case)':<28}{trials} trials → {faiths}")

    print("\n  Read the ACCURACY column, not the self-consistency one: a judge that returns")
    print("  0.0 every time is perfectly self-consistent and perfectly wrong. Stability is")
    print("  only interesting once you know the verdict it is stable on.")
    spaced, glued = results["hard: 10 chunks, '. [6]'"], results["same, glued: '.[6]'"]
    if glued > spaced:
        print("  And the cause here is not the model's reading of a long context: gluing the")
        print("  citation marker to its sentence fixes it. The metric's own sentence splitter")
        print("  was handing the judge a bare '[6]' to verify. Before you blame the judge for")
        print("  a bad number, read what your metric actually sent it.")
    print("  Mitigations: fix the parser first (attach a trailing [n] to the preceding")
    print("  sentence), then majority vote over 3 calls (3x cost), a narrower rubric, a")
    print("  deterministic pre-check (does the cited chunk contain the number at all?), or")
    print("  reporting a confidence interval on every judged metric.")


# ---------------------------------------------------------------------------
# 2. POSITION BIAS
# ---------------------------------------------------------------------------
class Preference(BaseModel):
    winner: str = Field(description="'A' or 'B' - the better answer")
    reason: str = Field(description="one short sentence")


PAIRS = [
    {
        "q": "How many days of PTO do full-time employees get per year?",
        "strong": "Full-time employees receive 24 days of PTO per calendar year, accrued at 2 days per month. [1]",
        "weak": "Employees get a generous amount of paid leave each year, typically a few weeks.",
    },
    {
        "q": "What is the Beacon API rate limit?",
        "strong": "600 requests per minute per API key; above that the API returns HTTP 429 with a Retry-After header. [1]",
        "weak": "There is a rate limit on the Beacon API to prevent abuse.",
    },
    {
        "q": "What is the maximum payload of the Atlas A2?",
        "strong": "250 kg for the Atlas A2; the A2 Lite carries 120 kg. [1]",
        "weak": "The Atlas A2 can carry a substantial load, more than the Lite version.",
    },
]


def judge_pair(question: str, first: str, second: str) -> str:
    return get_llm().with_structured_output(Preference).invoke(
        "Which answer is better for the question? Judge factual specificity and usefulness.\n\n"
        f"Question: {question}\n\nAnswer A:\n{first}\n\nAnswer B:\n{second}"
    ).winner.strip().upper()


def position_bias() -> None:
    print("\n" + "=" * 78)
    print("2. POSITION BIAS - the same comparison, presented in both orders")
    print("=" * 78)
    print("  Each pair has an obviously better answer. A judge free of position bias picks")
    print("  the strong answer in BOTH orders, so the two verdicts must disagree on the")
    print("  letter (A then B) and agree on the content.\n")
    print(f"  {'question':<46}{'strong 1st':>12}{'strong 2nd':>12}   consistent?")
    consistent = 0
    for p in PAIRS:
        v1 = judge_pair(p["q"], p["strong"], p["weak"])      # strong is A
        v2 = judge_pair(p["q"], p["weak"], p["strong"])      # strong is B
        ok = (v1 == "A" and v2 == "B")
        consistent += ok
        print(f"  {p['q'][:44]:<46}{v1:>12}{v2:>12}   {'yes' if ok else 'NO - order decided it'}")
    print(f"\n  order-independent on {consistent}/{len(PAIRS)} pairs.")
    print("  Mitigation: randomise or run both orders and keep only agreeing verdicts.")
    print("  Better: avoid pairwise judging entirely - score each answer against a rubric")
    print("  independently, which is what every judge in code/ch10/llm_judges.py does.")


# ---------------------------------------------------------------------------
# 3. VERBOSITY BIAS
# ---------------------------------------------------------------------------
SHORT = "24 days per calendar year. [1]"
PADDED = (
    "That is a great question, and it is important to understand your leave entitlement. "
    "According to the Lumora employee handbook, which was updated effective 1 January 2026 "
    "and supersedes the earlier 2024 policy, every full-time employee at the company is "
    "entitled to receive a total of 24 days of paid time off per calendar year. [1] This "
    "entitlement accrues gradually over the course of the year rather than being granted all "
    "at once. Please do consult your People partner if you have any further questions."
)


def verbosity_bias() -> None:
    print("\n" + "=" * 78)
    print("3. VERBOSITY BIAS - identical facts, 6 words vs 85 words")
    print("=" * 78)
    q = "How many days of PTO do full-time employees get per year?"
    ref = "24 days per calendar year."
    for name, ans in [("short", SHORT), ("padded", PADDED)]:
        rel = J.answer_relevance(q, ans)
        c = J.answer_correctness(q, ans, ref)
        print(f"  {name:<8}{len(ans.split()):>4} words   relevance={rel}/5   "
              f"correctness={c.verdict}   completeness={c.completeness}")
    winner = judge_pair(q, SHORT, PADDED)
    winner2 = judge_pair(q, PADDED, SHORT)
    print(f"  head-to-head: short-first → {winner}, padded-first → {winner2}"
          f"   ({'padded preferred' if (winner == 'B' and winner2 == 'A') else 'short preferred' if (winner == 'A' and winner2 == 'B') else 'order-dependent'})")
    print("\n  Both answers contain exactly one fact and it is the same fact. Any score")
    print("  difference is length, not quality. This is why rubrics must count FACTS")
    print("  ('fraction of reference facts present') rather than ask 'how good is this'.")


# ---------------------------------------------------------------------------
# 4. HUMAN AGREEMENT (Cohen's kappa)
#    Hand-labelled by the author against the handbook. Ground truth for the judge.
# ---------------------------------------------------------------------------
HAND_LABELLED = [
    # (question, answer, reference, human verdict: True = correct)
    ("How many PTO days do I get?", "24 days per calendar year. [1]", "24 days per calendar year.", True),
    ("How many PTO days do I get?", "20 days per year. [1]", "24 days per calendar year.", False),
    ("How many PTO days do I get?", "About three to four weeks. [1]", "24 days per calendar year.", False),
    ("What is the Beacon API rate limit?", "600 requests per minute per API key. [1]",
     "600 requests per minute per API key.", True),
    ("What is the Beacon API rate limit?", "600 requests per hour per key. [1]",
     "600 requests per minute per API key.", False),
    ("What is the max payload of the Atlas A2?", "250 kg. [1]", "250 kg (A2 Lite: 120 kg).", True),
    ("What is the max payload of the Atlas A2?", "120 kg. [1]", "250 kg (A2 Lite: 120 kg).", False),
    ("How long is parental leave for a primary caregiver?", "26 weeks at full pay. [1]",
     "26 weeks at full pay (8 weeks secondary).", True),
    ("How long is parental leave for a primary caregiver?", "6 months, roughly. [1]",
     "26 weeks at full pay (8 weeks secondary).", True),     # 26 weeks ≈ 6 months: a genuine edge case
    ("What is the P1 response time for Platinum support?", "15 minutes, 24x7, with a dedicated TAM. [1]",
     "15 minutes; includes a dedicated Technical Account Manager.", True),
    ("What is the P1 response time for Platinum support?", "30 minutes. [1]",
     "15 minutes; includes a dedicated Technical Account Manager.", False),
    ("When will Beacon API v1 be removed?", "In Beacon 4.4. [1]", "In Beacon 4.4.", True),
]


def cohens_kappa(a: list[bool], b: list[bool]) -> float:
    """Agreement corrected for agreement by chance.

      kappa = (p_observed - p_chance) / (1 - p_chance)

    Raw agreement is misleading when one label dominates: a judge that always says
    'correct' agrees 70% of the time with a 70%-correct set while knowing nothing.
    kappa scores that judge 0. Above 0.8 is usually called substantial agreement."""
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa_true, pb_true = sum(a) / n, sum(b) / n
    pe = pa_true * pb_true + (1 - pa_true) * (1 - pb_true)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def human_agreement() -> None:
    print("\n" + "=" * 78)
    print("4. AGREEMENT WITH HUMAN LABELS - Cohen's kappa")
    print("=" * 78)
    human, judge = [], []
    print(f"  {'answer':<52}{'human':>7}{'judge':>7}")
    for q, ans, ref, h in HAND_LABELLED:
        v = J.answer_correctness(q, ans, ref).verdict == "correct"
        human.append(h)
        judge.append(v)
        mark = "" if h == v else "   ← disagreement"
        print(f"  {ans[:50]:<52}{str(h):>7}{str(v):>7}{mark}")
    n = len(human)
    agree = sum(x == y for x, y in zip(human, judge))
    k = cohens_kappa(human, judge)
    fp = sum(1 for h, v in zip(human, judge) if v and not h)
    fn = sum(1 for h, v in zip(human, judge) if h and not v)
    print(f"\n  raw agreement: {agree}/{n} = {agree / n:.2f}")
    print(f"  Cohen's kappa: {k:.2f}   "
          f"({'substantial' if k >= 0.8 else 'moderate' if k >= 0.6 else 'weak - do not ship this judge'})")
    print(f"  judge too lenient (said correct, human said wrong): {fp}")
    print(f"  judge too strict  (said wrong, human said correct): {fn}")
    print("\n  Which direction the judge errs decides what it is safe to use for. A lenient")
    print("  correctness judge inflates your dashboard; a strict one makes every change look")
    print("  like a regression. Re-measure kappa whenever you change the judge model or prompt.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5, help="repeats for the self-consistency test")
    ap.add_argument("--only", default="", choices=["", "consistency", "position", "verbosity", "kappa"])
    args = ap.parse_args()

    if args.only in ("", "consistency"):
        self_consistency(args.trials)
    if args.only in ("", "position"):
        position_bias()
    if args.only in ("", "verbosity"):
        verbosity_bias()
    if args.only in ("", "kappa"):
        human_agreement()
