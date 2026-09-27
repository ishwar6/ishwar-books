"""Chapter 27 - query_rewriting_lab.py: six ways to rewrite a query, measured.

Rewriting is the cheapest fix in the playbook: it changes one LLM call, not the
index. But every transform trades something, and two of them actively destroy
certain queries. This lab runs all six over the golden set, reports MRR /
recall@5 / answer@3 / latency / LLM calls for each, and then shows the failure
case that stops you from turning rewriting on globally.

  uv run python code/ch27/query_rewriting_lab.py --limit 12
  uv run python code/ch27/query_rewriting_lab.py                # all 42 questions

Retrieval is exact (numpy) so the only variable is the query text.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose import World, rank_of, rrf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ch10"))
from metrics import recall_at_k, unique_in_order  # noqa: E402

from ragbook import get_llm, load_golden  # noqa: E402

llm = get_llm()


class Queries(BaseModel):
    queries: list[str]


def _chain(system: str, structured: bool = False):
    p = ChatPromptTemplate.from_messages([("system", system), ("human", "{q}")])
    return p | (llm.with_structured_output(Queries) if structured else llm)


# Each transform returns the list of query strings to search with. One string →
# one search; several → RRF over the result lists.
REWRITE = _chain("Rewrite the question as a single search query using the formal vocabulary an "
                 "internal handbook would use. Keep every identifier, number and product name "
                 "exactly as written. Return only the query.")
EXPAND = _chain("Rewrite the question, appending 3-6 extra terms: synonyms, expanded acronyms and "
                "likely handbook wording. Keep the original words too. Return only the query.")
HYDE = _chain("Write two sentences from an internal company handbook that would answer the question. "
              "Invent plausible specifics; this text is a search probe, never shown to a user.")
STEP_BACK = _chain("Write the more general question whose answer would let you answer this one. "
                   "Example: 'Can I fly business to Rotterdam?' -> 'What are the flight class rules?'. "
                   "Return only the question.")
DECOMPOSE = _chain("Split the question into 2-3 self-contained sub-questions that must each be "
                   "answered. If it is already atomic, return it unchanged.", structured=True)


def t_none(q: str) -> list[str]:
    return [q]


def t_rewrite(q: str) -> list[str]:
    return [REWRITE.invoke({"q": q}).text.strip()]


def t_expand(q: str) -> list[str]:
    return [EXPAND.invoke({"q": q}).text.strip()]


def t_hyde(q: str) -> list[str]:
    return [HYDE.invoke({"q": q}).text.strip()]


def t_step_back(q: str) -> list[str]:
    return [q, STEP_BACK.invoke({"q": q}).text.strip()]


def t_decompose(q: str) -> list[str]:
    return [q] + DECOMPOSE.invoke({"q": q}).queries


TRANSFORMS = {"baseline": t_none, "rewrite": t_rewrite, "expand": t_expand,
              "hyde": t_hyde, "step_back": t_step_back, "decompose": t_decompose}
LLM_CALLS = {"baseline": 0, "rewrite": 1, "expand": 1, "hyde": 1, "step_back": 1, "decompose": 1}


def search(world: World, queries: list[str]) -> list[str]:
    """Rank DOC ids for one or more query strings (RRF when there is more than one)."""
    rankings = [world.dense_rank(q) for q in queries]
    merged = rrf(rankings) if len(rankings) > 1 else rankings[0]
    return unique_in_order([world.by_id[cid].metadata["doc_id"] for cid, _ in merged[:30]]), merged


def evaluate(world: World, golden: list[dict], name: str) -> tuple[dict, dict]:
    fn = TRANSFORMS[name]
    rr, rec, ans3, lat = [], [], [], []
    per_q: dict[str, dict] = {}
    for g in golden:
        t0 = time.perf_counter()
        queries = fn(g["question"])
        docs_ranked, merged = search(world, queries)
        lat.append((time.perf_counter() - t0) * 1000)
        relevant = set(g["sources"])
        r = rank_of([(d, 0.0) for d in docs_ranked], relevant)
        rr.append(1 / r if r else 0.0)
        rec.append(recall_at_k(docs_ranked, relevant, 5))
        top3 = " ".join(world.by_id[cid].page_content for cid, _ in merged[:3]).lower()
        ans3.append(1.0 if all(k.lower() in top3 for k in g["keywords"]) else 0.0)
        per_q[g["id"]] = {"answer@3": ans3[-1], "query": queries[-1]}
    return ({"transform": name, "MRR": float(np.mean(rr)), "recall@5": float(np.mean(rec)),
             "answer@3": float(np.mean(ans3)), "ms/query": float(np.mean(lat)),
             "llm_calls": LLM_CALLS[name]}, per_q)


def regressions(rows_per_q: dict[str, dict], golden: list[dict]) -> None:
    """The honest version of "when rewriting hurts": every question the baseline
    answered and a transform broke. No contrived example - these are the rows
    from the run above."""
    by_id = {g["id"]: g for g in golden}
    base = rows_per_q["baseline"]
    print("\nWHERE A TRANSFORM BROKE A QUESTION THE BASELINE GOT RIGHT")
    any_found = False
    for name, per_q in rows_per_q.items():
        if name == "baseline":
            continue
        broke = [qid for qid in per_q if base[qid]["answer@3"] == 1.0 and per_q[qid]["answer@3"] == 0.0]
        fixed = [qid for qid in per_q if base[qid]["answer@3"] == 0.0 and per_q[qid]["answer@3"] == 1.0]
        if broke or fixed:
            any_found = True
            print(f"   {name}: broke {broke or '-'}  fixed {fixed or '-'}")
            for qid in broke[:2]:
                print(f"      {qid}: {by_id[qid]['question']}")
                print(f"         became -> {per_q[qid]['query'][:100]!r}")
    if not any_found:
        print("   none - every transform was a wash on this corpus.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    golden = [g for g in load_golden() if g["answerable"] and g["keywords"]]
    if args.limit:
        golden = golden[: args.limit]
    world = World()
    print(f"{len(golden)} questions, {len(world.chunks)} chunks, exact search\n")

    rows, rows_per_q = [], {}
    for name in TRANSFORMS:
        summary, per_q = evaluate(world, golden, name)
        rows.append(summary)
        rows_per_q[name] = per_q

    print(f"{'transform':<12}{'MRR':>7}{'recall@5':>10}{'answer@3':>10}{'ms/query':>10}{'LLM calls':>11}")
    print("-" * 60)
    base = rows[0]
    for r in rows:
        flag = ""
        if r["transform"] != "baseline":
            d = r["answer@3"] - base["answer@3"]
            flag = f"   {d:+.2f} answer@3"
        print(f"{r['transform']:<12}{r['MRR']:>7.3f}{r['recall@5']:>10.3f}{r['answer@3']:>10.3f}"
              f"{r['ms/query']:>10.0f}{r['llm_calls']:>11}{flag}")

    regressions(rows_per_q, golden)
    print("\nConclusion: apply transforms by QUESTION CLASS (a router), never globally.")


if __name__ == "__main__":
    main()
