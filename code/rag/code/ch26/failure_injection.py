"""Chapter 26 - Manufacture every kind of hallucination on purpose, then see
which metric notices.

The interview question is: "the retrieved chunks are correct, but the model still
hallucinates - what do you do?" You cannot answer that by staring at one bad
answer. You answer it by knowing the failure TAXONOMY and knowing which detector
fires for each entry, because the detector that fires tells you which fix to apply.

So: build each pathological context deliberately (we control the context here, no
retriever involved), generate an answer, and run four detectors over it:

  faithfulness      claims supported by the context        (Ch. 10 judge)
  citation accuracy does [n] support the sentence it tags  (Ch. 10 judge)
  correctness       does the answer match the reference    (Ch. 10 judge)
  abstention        did the model refuse                   (regex)

The output matrix is the point of the chapter: NO SINGLE DETECTOR CATCHES
EVERYTHING, and the one you would reach for first (faithfulness) is blind to the
most dangerous case of all.

  uv run python code/ch26/failure_injection.py              # all cases, ~45 calls
  uv run python code/ch26/failure_injection.py --limit 3
  uv run python code/ch26/failure_injection.py --only stale_document
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ch10"))
import llm_judges as J  # noqa: E402

from ragbook import chunk_documents, get_llm, load_handbook  # noqa: E402

# --------------------------------------------------------------- prompts ---
GROUNDED = (
    "You answer questions about the Lumora employee handbook using ONLY the context. "
    "Cite the chunk number like [2] after each sentence that uses it. "
    "If the context does not contain the answer, reply exactly: "
    "\"I don't know - the handbook does not cover this.\" Never use outside knowledge. "
    "If two passages disagree, say so and quote both."
)
PERMISSIVE = "You are a helpful assistant. Use the context below to answer the question."


def chunks_by_doc() -> dict[str, list]:
    out: dict[str, list] = {}
    for c in chunk_documents(load_handbook()):
        out.setdefault(c.metadata["doc_id"], []).append(c)
    return out


C = chunks_by_doc()


def text(doc: str, idx: int) -> str:
    return C[doc][idx].page_content


def distractors(n: int) -> list[str]:
    """Chunks that are real handbook content but irrelevant to PTO questions."""
    pool = [text("03-travel-and-expense-policy", 0), text("04-atlas-a2-specification", 1),
            text("05-beacon-fleet-software", 1), text("08-runbook-fleet-outage", 0),
            text("09-support-sla", 0), text("12-postmortem-2026-03-rotterdam-halt", 0),
            text("13-remote-work-policy", 0), text("06-security-policy", 0),
            text("07-onboarding-guide", 0), text("11-release-notes-beacon-4.2", 0)]
    return pool[:n]


def fmt(chunks: list[str]) -> str:
    return "\n\n".join(f"[{i}]\n{c}" for i, c in enumerate(chunks, start=1))


# ------------------------------------------------------------- the cases ---
PTO_Q = "How many days of PTO do full-time employees get per year?"
PTO_REF = "24 days per calendar year, accrued at 2 days per month."

PRICING_NO_EXAMPLE = text("10-pricing-and-plans", 1).split("## Worked example")[0]

CASES: list[dict] = [
    {
        "name": "baseline_clean",
        "why": "gold chunk, strict prompt - the control. Everything should pass.",
        "question": PTO_Q, "reference": PTO_REF, "prompt": GROUNDED,
        "chunks": [text("02-pto-and-leave-policy", 0)],
        "answerable": True,
    },
    {
        "name": "evidence_missing",
        "why": "retrieval failed: nothing about PTO is in the context at all.",
        "question": PTO_Q, "reference": PTO_REF, "prompt": GROUNDED,
        "chunks": distractors(3),
        "answerable": True,
    },
    {
        "name": "conflicting_chunks",
        "why": "2026 policy (24 days) and the 2024 FAQ (20 days) both present.",
        "question": PTO_Q, "reference": PTO_REF, "prompt": GROUNDED,
        "chunks": [text("14-faq", 0), text("02-pto-and-leave-policy", 0)],
        "answerable": True,
    },
    {
        "name": "stale_document",
        "why": "ONLY the outdated FAQ (20 days). The model is perfectly faithful to a wrong source.",
        "question": PTO_Q, "reference": PTO_REF, "prompt": GROUNDED,
        "chunks": [text("14-faq", 0)],
        "answerable": True,
    },
    {
        "name": "distractor_flood",
        "why": "gold chunk plus 9 irrelevant ones - precision 0.1, the k=30 mistake.",
        "question": PTO_Q, "reference": PTO_REF, "prompt": GROUNDED,
        "chunks": [text("02-pto-and-leave-policy", 0)] + distractors(9),
        "answerable": True,
    },
    {
        "name": "lost_in_middle",
        "why": "same chunks as the flood, but the gold chunk is buried at position 6 of 10.",
        "question": PTO_Q, "reference": PTO_REF, "prompt": GROUNDED,
        "chunks": distractors(5) + [text("02-pto-and-leave-policy", 0)] + distractors(4),
        "answerable": True,
    },
    {
        "name": "permissive_prompt",
        "why": "gold chunk present, but the prompt never forbids outside knowledge.",
        "question": "How many days of PTO do full-time employees get, and what is the parental leave in India specifically?",
        "reference": "24 days of PTO. The handbook does not state an India-specific parental leave policy.",
        "prompt": PERMISSIVE,
        "chunks": [text("02-pto-and-leave-policy", 0)],
        "answerable": True,
    },
    {
        "name": "false_presupposition",
        "why": "the question invents a product. Correct behaviour is to refuse, not to answer.",
        "question": "What is the maximum payload of the Atlas A3?",
        "reference": "Not in the handbook; there is no Atlas A3, only the A2 and A2 Lite.",
        "prompt": GROUNDED,
        "chunks": [text("04-atlas-a2-specification", 1), text("04-atlas-a2-specification", 0)],
        "answerable": False,
    },
    {
        "name": "unsupported_synthesis",
        "why": "all facts present, but the answer needs arithmetic + applying a discount rule.",
        "question": "What would 120 robots with Beacon and Compass cost per month, paying monthly?",
        "reference": "120 x ($180 + $40) = $26,400, less the 15% fleet discount = $22,440 per month.",
        "prompt": GROUNDED,
        "chunks": [PRICING_NO_EXAMPLE],
        "answerable": True,
    },
]


def run_case(case: dict) -> dict:
    context = fmt(case["chunks"])
    chain = ChatPromptTemplate.from_messages(
        [("system", case["prompt"]), ("human", "Context:\n{context}\n\nQuestion: {question}")]
    ) | get_llm()
    answer = chain.invoke({"context": context, "question": case["question"]}).text

    abstained = J.is_abstention(answer)
    # did the answer DISCLOSE a disagreement rather than silently picking a side?
    low = answer.lower()
    disclosed = any(w in low for w in ("disagree", "conflict", "contradic", "outdated", "supersed", "differs"))
    f = J.faithfulness(answer, context)
    cite = J.citation_accuracy(answer, case["chunks"])
    corr = J.answer_correctness(case["question"], answer, case["reference"]).verdict
    return {
        "name": case["name"], "answerable": case["answerable"], "answer": answer.replace("\n", " "),
        "faith": f["score"], "cite": cite, "corr": corr, "abstained": abstained,
        "disclosed": disclosed,
        "unsupported": [c for c, ok in zip(f["claims"], f["supported"]) if not ok],
    }


def main(args) -> None:
    cases = [c for c in CASES if not args.only or c["name"] == args.only]
    if args.limit:
        cases = cases[: args.limit]

    rows = []
    for case in cases:
        print(f"\n─── {case['name']} " + "─" * (60 - len(case["name"])))
        print(f"  why: {case['why']}")
        print(f"  Q  : {case['question'][:88]}")
        r = run_case(case)
        rows.append(r)
        print(f"  A  : {r['answer'][:260]}")
        cite_txt = "n/a" if r["cite"] is None else f"{r['cite']:.2f}"
        print(f"  faithfulness={r['faith']:.2f}  citation={cite_txt}  "
              f"correctness={r['corr']}  abstained={r['abstained']}")
        for c in r["unsupported"]:
            print(f"       unsupported claim: {c[:100]}")

    # ---------------------------------------------------------- the matrix ---
    print("\n" + "=" * 94)
    print("DETECTION MATRIX - which detector notices which failure")
    print("=" * 94)
    print(f"  {'failure mode':<24}{'faith':>7}{'cite':>7}{'correct':>9}{'abst':>6}{'outcome':>8}   "
          f"{'detectors that fired':<34}")
    missed = []
    for r in rows:
        # a detector "fires" when it signals that something is wrong
        f_fire = r["faith"] < 0.9
        c_fire = r["cite"] is not None and r["cite"] < 0.9
        # for an unanswerable question the desired verdict IS a refusal
        k_fire = (r["corr"] != "correct") if r["answerable"] else (not r["abstained"])
        a_fire = r["abstained"] if r["answerable"] else False

        caught = [n for n, fired in
                  [("faithfulness", f_fire), ("citation", c_fire),
                   ("correctness", k_fire), ("over-refusal", a_fire)] if fired]
        # Did the model actually FAIL? (separate question from whether a detector fired)
        if r["answerable"]:
            failed = r["corr"] != "correct" or r["abstained"]
            # answering a conflict without disclosing it is a silent failure
            if r["name"] == "conflicting_chunks" and not r["disclosed"]:
                failed = True
        else:
            failed = not r["abstained"]
        cite_txt = " n/a" if r["cite"] is None else f"{r['cite']:.2f}"
        status = "FAILED" if failed else "ok"
        print(f"  {r['name']:<24}{r['faith']:>7.2f}{cite_txt:>7}{r['corr']:>9}"
              f"{str(r['abstained']):>6}{status:>8}   {', '.join(caught) if caught else '- nothing fired':<34}")
        if failed and not caught:
            missed.append(r["name"])

    print("\n  Reading the matrix")
    print("  ------------------")
    print("  * faithfulness is measured AGAINST THE CONTEXT, so any failure whose context is")
    print("    internally consistent scores 1.00 - most importantly 'stale_document', where the")
    print("    model faithfully reports a number that is two years out of date. Faithfulness is")
    print("    the metric everyone reaches for first and it cannot see source quality at all.")
    print("  * correctness needs a reference answer, so it only exists offline, on your golden")
    print("    set. In production you have faithfulness and citations and nothing else - which")
    print("    is exactly why document freshness and source trust are ingestion problems.")
    print("  * abstention is a detector only in context: on an answerable question it means")
    print("    over-refusal; on 'false_presupposition' it is the CORRECT behaviour.")
    if missed:
        print(f"  * SILENT FAILURES - the model got it wrong and nothing fired: {', '.join(missed)}")
    else:
        print("  * no silent failures in this run: every case the model got wrong lit up at")
        print("    least one detector. Cases marked 'ok' are ones the model simply handled.")
    print("\n  Note what is NOT in this table: temperature. Every run above used the same")
    print("  sampling settings. The failures were produced entirely by what was put in the")
    print("  context and what the prompt permitted.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="run a single case by name")
    main(ap.parse_args())
