"""Chapter 29 - lost_in_the_middle.py: does WHERE you put the evidence matter?

Retrieval gave you k chunks. You now choose an order. This script measures what
that choice is worth: it puts the chunk that actually contains the answer at the
FIRST, MIDDLE and LAST position among N-1 distractors, for N = 4, 10 and 20, and
scores the answer with a free, deterministic check (do the golden keywords
appear?). No LLM judge, so the only noise is the generator itself.

  uv run python code/ch29/lost_in_the_middle.py --limit 12
  uv run python code/ch29/lost_in_the_middle.py --limit 20 --sizes 4 10 20 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from langchain_core.prompts import ChatPromptTemplate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ch10"))
from llm_judges import keyword_hit  # noqa: E402

from ragbook import chunk_documents, get_embeddings, get_llm, load_golden, load_handbook  # noqa: E402

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using ONLY the numbered context. Cite sources as [n]. "
               "If the context does not contain the answer, say exactly: I don't know based on the handbook."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


def build_context(chunks) -> str:
    return "\n\n".join(f"[{i}] (source: {c.metadata['source']})\n{c.page_content}"
                       for i, c in enumerate(chunks, start=1))


def place(gold, distractors, position: str, n: int):
    """Put the gold chunk first, in the middle, or last among n-1 distractors."""
    others = distractors[: n - 1]
    if position == "first":
        return [gold] + others
    if position == "last":
        return others + [gold]
    mid = len(others) // 2
    return others[:mid] + [gold] + others[mid:]


def bar(x: float, width: int = 24) -> str:
    return "█" * round(x * width) + "·" * (width - round(x * width))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--sizes", type=int, nargs="*", default=[4, 10, 20])
    args = ap.parse_args()

    chunks = chunk_documents(load_handbook())
    emb = get_embeddings()
    vecs = np.array(emb.embed_documents([c.page_content for c in chunks]), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    llm = get_llm()
    golden = [g for g in load_golden() if g["answerable"] and g["keywords"]]

    # Keep only questions where ONE chunk holds the whole answer: otherwise
    # "where is the gold chunk" is not a well-defined question.
    cases = []
    for g in golden:
        kws = [k.lower() for k in g["keywords"]]
        gold = [c for c in chunks if all(k in c.page_content.lower() for k in kws)
                and c.metadata["doc_id"] in g["sources"]]
        if len(gold) != 1:
            continue
        q = np.array(emb.embed_query(g["question"]), dtype=np.float32)
        q /= np.linalg.norm(q)
        order = np.argsort(-(vecs @ q))
        distractors = [chunks[i] for i in order
                       if chunks[i].metadata["chunk_id"] != gold[0].metadata["chunk_id"]][:40]
        cases.append((g, gold[0], distractors))
        if len(cases) >= args.limit:
            break

    print(f"{len(cases)} questions whose answer lives in exactly one chunk\n")
    results: dict[tuple[int, str], float] = {}
    for n in args.sizes:
        for position in ("first", "middle", "last"):
            hits = []
            for g, gold, distractors in cases:
                ctx = build_context(place(gold, distractors, position, n))
                answer = (PROMPT | llm).invoke({"context": ctx, "question": g["question"]}).text
                hits.append(keyword_hit(answer, g["keywords"]) or 0.0)
            score = float(np.mean(hits))
            results[(n, position)] = score
            print(f"  n={n:<3} gold at {position:<7} keyword-hit {score:.3f}  {bar(score)}")

    print(f"\n{'context size':>13}{'first':>9}{'middle':>9}{'last':>9}{'spread':>9}")
    print("-" * 49)
    for n in args.sizes:
        row = [results[(n, p)] for p in ("first", "middle", "last")]
        print(f"{n:>13}{row[0]:>9.3f}{row[1]:>9.3f}{row[2]:>9.3f}{max(row) - min(row):>9.3f}")

    worst = min(results.items(), key=lambda kv: kv[1])
    best = max(results.items(), key=lambda kv: kv[1])
    print(f"\nbest  : n={best[0][0]}, gold {best[0][1]} -> {best[1]:.3f}")
    print(f"worst : n={worst[0][0]}, gold {worst[0][1]} -> {worst[1]:.3f}")
    print("Read the SPREAD column: it is the price of ordering at that context size.")


if __name__ == "__main__":
    main()
