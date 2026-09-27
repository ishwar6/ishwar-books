"""Chapter 11 - Guard rails AFTER generation: score gate, faithfulness gate, self-consistency.

A grounded prompt lowers hallucination; it does not eliminate it. So we also
check the OUTPUT before showing it:

  gate 1  retrieval score threshold  : best chunk too far from the question → abstain, don't generate
  gate 2  faithfulness judge         : claims not supported → retry with more context, then abstain
  gate 3  self-consistency (optional): sample 3 answers; if they disagree on facts, lower confidence

Run:  uv run python code/ch11/faithfulness_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ch10"))
from llm_judges import faithfulness, is_abstention  # noqa: E402  (Chapter 10's judge, reused)

from ragbook import build_handbook_index, format_docs, get_llm, load_golden  # noqa: E402

llm = get_llm()
ABSTAIN = "I don't know - the handbook does not cover this."
SCORE_THRESHOLD = 0.35     # calibrated in Chapter 12 (cosine similarity, text-embedding-3-small)
FAITHFULNESS_MIN = 0.99    # every claim must be supported

grounded = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the context. If it does not contain the answer, reply exactly: " + repr(ABSTAIN)),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
]) | llm


def answer_with_gates(store, question: str, k: int = 4, verbose: bool = True) -> dict:
    log = lambda *a: print("   ", *a) if verbose else None

    # gate 1: is anything even close?
    scored = store.similarity_search_with_score(question, k=k)
    best = scored[0][1] if scored else 0.0
    if best < SCORE_THRESHOLD:
        log(f"gate1 ✗ best score {best:.2f} < {SCORE_THRESHOLD} → abstain without calling the LLM")
        return {"answer": ABSTAIN, "gate": "score", "faithfulness": None}
    log(f"gate1 ✓ best score {best:.2f}")

    # generate, then gate 2 with one retry on more context
    for attempt, kk in enumerate((k, k * 2), start=1):
        docs = [d for d, _ in store.similarity_search_with_score(question, k=kk)]
        ctx = format_docs(docs)
        ans = grounded.invoke({"context": ctx, "question": question}).text
        if is_abstention(ans):
            log(f"attempt {attempt}: model abstained")
            return {"answer": ABSTAIN, "gate": "model_abstained", "faithfulness": 1.0}
        f = faithfulness(ans, ctx)
        log(f"attempt {attempt} (k={kk}): faithfulness {f['score']:.2f}  unsupported={[c for c, ok in zip(f['claims'], f['supported']) if not ok]}")
        if f["score"] >= FAITHFULNESS_MIN:
            return {"answer": ans, "gate": "passed", "faithfulness": f["score"]}
    log("gate2 ✗ still unsupported claims after retry → abstain")
    return {"answer": ABSTAIN, "gate": "faithfulness", "faithfulness": f["score"]}


# ------------------------------------------------------- self-consistency ---
class Agreement(BaseModel):
    agree: bool = Field(description="true if all answers state the same facts and numbers")
    disagreement: str | None = None


def self_consistency(store, question: str, n: int = 3) -> dict:
    """Sample n answers (the model is stochastic by default) and ask a judge whether
    they agree. Disagreement is a cheap hallucination detector: fabricated details
    vary between samples, real ones don't."""
    docs = store.similarity_search(question, k=4)
    ctx = format_docs(docs)
    answers = [grounded.invoke({"context": ctx, "question": question}).text for _ in range(n)]
    verdict = llm.with_structured_output(Agreement).invoke(
        "Do these answers agree on the facts?\n\n" + "\n\n".join(f"Answer {i + 1}: {a}" for i, a in enumerate(answers))
    )
    return {"answers": answers, "agree": verdict.agree, "disagreement": verdict.disagreement}


if __name__ == "__main__":
    store = build_handbook_index("ch11_handbook")
    golden = {g["id"]: g for g in load_golden()}
    for qid in ["q32", "q01", "u03", "u02", "u05"]:
        q = golden[qid]["question"]
        print(f"\n{qid}: {q}")
        r = answer_with_gates(store, q)
        print(f"    → [{r['gate']}] {r['answer'][:220].replace(chr(10), ' ')}")
        print(f"    gold: {golden[qid]['answer']}")

    print("\nself-consistency on q36 (postmortem):")
    sc = self_consistency(store, golden["q36"]["question"])
    for a in sc["answers"]:
        print("   -", a[:140].replace("\n", " "))
    print("   agree:", sc["agree"], "|", sc["disagreement"])
