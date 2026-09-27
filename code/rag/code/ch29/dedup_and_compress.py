"""Chapter 29 - dedup_and_compress.py: buy back context budget without losing answers.

Two ways to spend fewer tokens on the same evidence:

  DEDUPLICATION - drop chunks that say what another chunk already said.
                  (near-duplicates also bias the model: three copies of a claim
                  read like three witnesses.)
  COMPRESSION   - shrink each chunk: keep only the sentences that matter
                  (extractive) or have a cheap model rewrite it (abstractive).

Every method is scored the same way: tokens in the final context, latency added,
and whether the answer still contains the golden keywords. A method that saves
40% of tokens and loses 10 points of accuracy is not a saving.

  uv run python code/ch29/dedup_and_compress.py --limit 12 --k 10
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import tiktoken
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ch10"))
from llm_judges import keyword_hit  # noqa: E402

from ragbook import chunk_documents, get_embeddings, get_llm, load_golden, load_handbook  # noqa: E402

ENC = tiktoken.get_encoding("cl100k_base")
SENT = re.compile(r"(?<=[.!?:])\s+|\n(?=[-|#*])")

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY the numbered context. Cite [n]. If it is not there, say exactly: "
               "I don't know based on the handbook."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])
SUMMARISE = ChatPromptTemplate.from_messages([
    ("system", "Compress the passage to the sentences that could answer the question. Keep every number, "
               "name and unit verbatim. If nothing in it is relevant, reply exactly: IRRELEVANT."),
    ("human", "Question: {question}\n\nPassage:\n{passage}"),
])


def ntok(text: str) -> int:
    return len(ENC.encode(text))


def render(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i}] (source: {d.metadata['source']})\n{d.page_content}"
                       for i, d in enumerate(docs, start=1))


# ---------------------------------------------------------------- dedup ----
def shingles(text: str, n: int = 5) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def dedup_embeddings(docs: list[Document], vecs: np.ndarray, threshold: float = 0.90):
    """Keep a chunk only if it is not ~the same vector as one already kept.
    Cheap, and it catches paraphrases that share no words."""
    keep, kept_vecs = [], []
    for d, v in zip(docs, vecs):
        if all(float(v @ kv) < threshold for kv in kept_vecs):
            keep.append(d); kept_vecs.append(v)
    return keep


def dedup_shingles(docs: list[Document], threshold: float = 0.50):
    """Lexical near-duplicate detection: no embeddings, microseconds, catches the
    overlap that `chunk_overlap` itself creates."""
    keep, sets = [], []
    for d in docs:
        s = shingles(d.page_content)
        if all(jaccard(s, t) < threshold for t in sets):
            keep.append(d); sets.append(s)
    return keep


# ----------------------------------------------------------- compression ---
def compress_extractive(docs: list[Document], qvec: np.ndarray, emb, keep_frac: float = 0.5):
    """Keep the sentences closest to the query. No LLM, so it cannot invent
    anything. Note the cost: one embedding call PER CHUNK, which is why the
    prep-latency column in 29.4 is seconds - batching every chunk's sentences
    into a single call would cut it several-fold."""
    out = []
    for d in docs:
        sents = [s.strip() for s in SENT.split(d.page_content) if s.strip()]
        if len(sents) <= 2:
            out.append(d); continue
        sv = np.array(emb.embed_documents(sents), dtype=np.float32)
        sv /= np.linalg.norm(sv, axis=1, keepdims=True)
        n = max(1, int(len(sents) * keep_frac))
        best = sorted(sorted(range(len(sents)), key=lambda i: -float(sv[i] @ qvec))[:n])  # keep reading order
        out.append(Document(page_content=" ".join(sents[i] for i in best), metadata=d.metadata))
    return out


def compress_abstractive(docs: list[Document], question: str, llm):
    """A cheap model rewrites each chunk against the question. Strongest
    compression, and the only one that can DROP a chunk entirely - also the only
    one that can silently delete the number you needed."""
    chain = SUMMARISE | llm
    out = []
    for d in docs:
        text = chain.invoke({"question": question, "passage": d.page_content}).text.strip()
        if text.upper().startswith("IRRELEVANT"):
            continue
        out.append(Document(page_content=text, metadata=d.metadata))
    return out or docs[:1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--k", type=int, default=10, help="chunks retrieved before any trimming")
    args = ap.parse_args()

    chunks = chunk_documents(load_handbook())
    emb, llm = get_embeddings(), get_llm()
    vecs = np.array(emb.embed_documents([c.page_content for c in chunks]), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    golden = [g for g in load_golden() if g["answerable"] and g["keywords"]][: args.limit]
    print(f"{len(golden)} questions, k={args.k} chunks retrieved before trimming\n")

    methods = ["none", "dedup-embed", "dedup-shingle", "extractive-50%", "abstractive"]
    stats = {m: {"tok": [], "ms": [], "hit": [], "kept": []} for m in methods}

    for g in golden:
        q = np.array(emb.embed_query(g["question"]), dtype=np.float32)
        q /= np.linalg.norm(q)
        order = np.argsort(-(vecs @ q))[: args.k]
        base = [chunks[i] for i in order]
        base_vecs = vecs[order]

        for m in methods:
            t0 = time.perf_counter()
            if m == "none":
                docs = base
            elif m == "dedup-embed":
                docs = dedup_embeddings(base, base_vecs)
            elif m == "dedup-shingle":
                docs = dedup_shingles(base)
            elif m == "extractive-50%":
                docs = compress_extractive(base, q, emb)
            else:
                docs = compress_abstractive(base, g["question"], llm)
            prep_ms = (time.perf_counter() - t0) * 1000

            ctx = render(docs)
            answer = (PROMPT | llm).invoke({"context": ctx, "question": g["question"]}).text
            stats[m]["tok"].append(ntok(ctx))
            stats[m]["ms"].append(prep_ms)
            stats[m]["hit"].append(keyword_hit(answer, g["keywords"]) or 0.0)
            stats[m]["kept"].append(len(docs))

    base_tok = float(np.mean(stats["none"]["tok"]))
    print(f"{'method':<16}{'chunks':>8}{'ctx tokens':>12}{'saved':>8}{'prep ms':>10}{'keyword-hit':>13}")
    print("-" * 67)
    for m in methods:
        s = stats[m]
        tok = float(np.mean(s["tok"]))
        print(f"{m:<16}{np.mean(s['kept']):>8.1f}{tok:>12.0f}"
              f"{(1 - tok / base_tok) * 100:>7.0f}%{np.mean(s['ms']):>10.0f}{np.mean(s['hit']):>13.3f}")

    print("\nRule: a saving that costs accuracy is not a saving. Compare each row's "
          "keyword-hit with the 'none' row before you ship it.")


if __name__ == "__main__":
    main()
