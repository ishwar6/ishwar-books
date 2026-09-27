"""Chapter 27 - chunk_experiment.py: "why 500 and not 2000?" as a designed experiment.

Chunk size is not a preference, it is a variable with an optimum that depends on
your corpus, your questions and your k. This script runs a factorial sweep

    strategy  x  chunk_size  x  overlap

and reports, for every cell: document recall@5, nDCG@5, whether the ANSWER
survives in the top-3 chunks, how many tokens that costs per query, and the
resulting $/query. Then it prints the Pareto frontier - the configurations that
are not beaten on both quality and cost - because that, not a single winner, is
what you take to a design review.

  uv run python code/ch27/chunk_experiment.py                # full 45-cell grid (~4 min)
  uv run python code/ch27/chunk_experiment.py --quick        # 12 cells
  uv run python code/ch27/chunk_experiment.py --quick --judge --limit 10   # + LLM metrics

Retrieval here is EXACT (numpy dot product over every chunk), so the numbers
measure chunking alone - no ANN recall, no HNSW parameters mixed in.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ch10"))
from metrics import cost_usd, ndcg_at_k, recall_at_k, unique_in_order  # noqa: E402

from ragbook import CHAT_MODEL, get_embeddings, load_golden, load_handbook  # noqa: E402

ENC = tiktoken.get_encoding("cl100k_base")
PROMPT_OVERHEAD = 120          # system prompt + question, in tokens


def ntok(text: str) -> int:
    return len(ENC.encode(text))


# ------------------------------------------------------------- strategies ---
def split_recursive(docs, size: int, overlap: int) -> list[Document]:
    sp = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap,
                                        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""])
    return [Document(page_content=p, metadata=dict(d.metadata))
            for d in docs for p in sp.split_text(d.page_content)]


def split_markdown(docs, size: int, overlap: int) -> list[Document]:
    """Split at headings first (the author's own boundaries), then cap oversized
    sections. The heading path stays in metadata and is prepended to the text so
    a table row still knows which table it belongs to."""
    md = MarkdownHeaderTextSplitter([("#", "h1"), ("##", "h2"), ("###", "h3")])
    cap = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    out: list[Document] = []
    for d in docs:
        for sec in md.split_text(d.page_content):
            path = " > ".join(str(sec.metadata[h]) for h in ("h1", "h2", "h3") if h in sec.metadata)
            for piece in cap.split_text(sec.page_content):
                text = f"{path}\n{piece}" if path else piece
                out.append(Document(page_content=text, metadata=dict(d.metadata)))
    return out


SENT = re.compile(r"(?<=[.!?:])\s+|\n{2,}")


def split_sentences(docs, size: int, overlap: int) -> list[Document]:
    """Pack whole sentences up to the size limit - never cuts mid-sentence."""
    out: list[Document] = []
    for d in docs:
        sents = [s.strip() for s in SENT.split(d.page_content) if s.strip()]
        buf: list[str] = []
        for s in sents:
            if buf and sum(len(x) + 1 for x in buf) + len(s) > size:
                out.append(Document(page_content=" ".join(buf), metadata=dict(d.metadata)))
                keep, total = [], 0
                for prev in reversed(buf):                     # sentence-level overlap
                    if total + len(prev) > overlap:
                        break
                    keep.insert(0, prev); total += len(prev)
                buf = keep
            buf.append(s)
        if buf:
            out.append(Document(page_content=" ".join(buf), metadata=dict(d.metadata)))
    return out


STRATEGIES = {"recursive": split_recursive, "markdown": split_markdown, "sentence": split_sentences}


# ------------------------------------------------------------------ sweep ---
def evaluate(chunks: list[Document], qvecs: np.ndarray, golden: list[dict], emb) -> dict:
    vecs = np.array(emb.embed_documents([c.page_content for c in chunks]), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    sims = qvecs @ vecs.T                                       # (questions, chunks)

    rec, nd, ans3, ctx = [], [], [], []
    for i, g in enumerate(golden):
        order = np.argsort(-sims[i])[:10]
        docs_ranked = unique_in_order([chunks[j].metadata["doc_id"] for j in order])
        relevant = set(g["sources"])
        rec.append(recall_at_k(docs_ranked, relevant, 5))
        nd.append(ndcg_at_k(docs_ranked, relevant, 5))
        top3 = " ".join(chunks[j].page_content for j in order[:3]).lower()
        ans3.append(1.0 if all(k.lower() in top3 for k in g["keywords"]) else 0.0)
        ctx.append(sum(ntok(chunks[j].page_content) for j in order[:5]))

    tok = float(np.mean(ctx))
    return {
        "n_chunks": len(chunks),
        "mean_tok": float(np.mean([ntok(c.page_content) for c in chunks])),
        "recall@5": float(np.mean(rec)),
        "ndcg@5": float(np.mean(nd)),
        "answer@3": float(np.mean(ans3)),
        "ctx_tok@5": tok,
        "usd_1k_q": cost_usd(CHAT_MODEL, int(tok + PROMPT_OVERHEAD), 60) * 1000,
    }


def pareto(rows: list[dict]) -> list[dict]:
    """Keep configs that nothing beats on BOTH answer@3 (higher) and ctx_tok@5 (lower)."""
    keep = []
    for r in rows:
        if not any(o["answer@3"] >= r["answer@3"] and o["ctx_tok@5"] <= r["ctx_tok@5"]
                   and (o["answer@3"] > r["answer@3"] or o["ctx_tok@5"] < r["ctx_tok@5"]) for o in rows):
            keep.append(r)
    return sorted(keep, key=lambda r: r["ctx_tok@5"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="12 cells instead of 45")
    ap.add_argument("--limit", type=int, help="use only the first N golden questions")
    ap.add_argument("--judge", action="store_true", help="also run faithfulness/correctness on the top 3 configs")
    args = ap.parse_args()

    sizes = [400, 800, 1600] if args.quick else [200, 400, 800, 1200, 1600]
    fracs = [0.0, 0.25] if args.quick else [0.0, 0.10, 0.25]
    strategies = ["recursive", "markdown"] if args.quick else list(STRATEGIES)

    docs = load_handbook()
    golden = [g for g in load_golden() if g["answerable"] and g["keywords"]]
    if args.limit:
        golden = golden[: args.limit]

    emb = get_embeddings()
    qv = np.array(emb.embed_documents([g["question"] for g in golden]), dtype=np.float32)
    qv /= np.linalg.norm(qv, axis=1, keepdims=True)             # embed the questions ONCE
    print(f"{len(docs)} documents, {len(golden)} questions, "
          f"{len(strategies) * len(sizes) * len(fracs)} configurations\n")

    rows: list[dict] = []
    for strat in strategies:
        for size in sizes:
            for frac in fracs:
                overlap = int(size * frac)
                chunks = STRATEGIES[strat](docs, size, overlap)
                r = {"strategy": strat, "size": size, "ovl": f"{frac:.0%}", **evaluate(chunks, qv, golden, emb)}
                rows.append(r)
                print(f"  {strat:<10}{size:>5}{r['ovl']:>6}  "
                      f"chunks={r['n_chunks']:<4} mean_tok={r['mean_tok']:5.0f}  "
                      f"recall@5={r['recall@5']:.2f} ndcg@5={r['ndcg@5']:.2f} "
                      f"answer@3={r['answer@3']:.2f} ctx={r['ctx_tok@5']:5.0f}")

    print("\n" + "=" * 92)
    print(f"{'strategy':<11}{'size':>5}{'ovl':>6}{'chunks':>8}{'mean_tok':>10}"
          f"{'recall@5':>10}{'ndcg@5':>9}{'answer@3':>10}{'ctx_tok':>9}{'$/1k q':>9}")
    print("-" * 92)
    for r in sorted(rows, key=lambda r: (-r["answer@3"], r["ctx_tok@5"])):
        print(f"{r['strategy']:<11}{r['size']:>5}{r['ovl']:>6}{r['n_chunks']:>8}{r['mean_tok']:>10.0f}"
              f"{r['recall@5']:>10.2f}{r['ndcg@5']:>9.2f}{r['answer@3']:>10.2f}"
              f"{r['ctx_tok@5']:>9.0f}{r['usd_1k_q']:>9.2f}")

    front = pareto(rows)
    print("\nPARETO FRONTIER (nothing beats these on both answer@3 and context cost)")
    for r in front:
        print(f"   {r['strategy']:<10} size={r['size']:<5} ovl={r['ovl']:<5} "
              f"answer@3={r['answer@3']:.2f}  ctx={r['ctx_tok@5']:.0f} tok  ${r['usd_1k_q']:.2f}/1k queries")

    if args.judge:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ch10"))
        import llm_judges as J
        from langchain_core.prompts import ChatPromptTemplate

        from ragbook import format_docs, get_llm
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer using ONLY the context. Cite [n]. If the context does not contain the answer, "
                       "say exactly: I don't know based on the handbook."),
            ("human", "Context:\n{context}\n\nQuestion: {question}")])
        chain = prompt | get_llm()
        print("\nGENERATION METRICS on the top-3 configs by answer@3 "
              f"({len(golden)} questions each - this is the expensive part)")
        for r in sorted(rows, key=lambda r: -r["answer@3"])[:3]:
            chunks = STRATEGIES[r["strategy"]](docs, r["size"], int(r["size"] * float(r["ovl"].rstrip("%")) / 100))
            vecs = np.array(emb.embed_documents([c.page_content for c in chunks]), dtype=np.float32)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            faith, corr = [], []
            for i, g in enumerate(golden):
                top = [chunks[j] for j in np.argsort(-(qv[i] @ vecs.T))[:5]]
                ctx = format_docs(top)
                ans = chain.invoke({"context": ctx, "question": g["question"]}).text
                faith.append(J.faithfulness(ans, ctx)["score"])
                corr.append(1.0 if J.answer_correctness(g["question"], ans, g["answer"]).verdict == "correct" else 0.0)
            print(f"   {r['strategy']:<10} size={r['size']:<5} ovl={r['ovl']:<5} "
                  f"faithfulness={np.mean(faith):.2f}  correct={np.mean(corr):.2f}")


if __name__ == "__main__":
    main()
