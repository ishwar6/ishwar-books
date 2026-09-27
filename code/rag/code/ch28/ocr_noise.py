"""Chapter 28 - what OCR errors do to retrieval, measured.

    QDRANT_MODE=memory uv run python code/ch28/ocr_noise.py

A scanned PDF reaches your index through OCR, and OCR is never perfect: 'rn'
becomes 'm', 'l' becomes '1', 'O' becomes '0', spaces vanish. The queries stay
clean - users type real words - so every error widens the gap between the
question and the text that answers it.

We corrupt the corpus at increasing rates and measure hit-rate@5 for the two
retrievers, using exact (brute-force) cosine so the only moving part is the
noise. The two curves separate for a reason worth being able to explain.
"""
from __future__ import annotations

import argparse
import random
import re

import numpy as np
from rank_bm25 import BM25Okapi

from ragbook import chunk_documents, get_embeddings, load_golden, load_handbook

# The classic confusion pairs of a character-based OCR engine.
CONFUSIONS = [
    ("rn", "m"), ("m", "rn"), ("cl", "d"), ("l", "1"), ("I", "l"), ("i", "j"),
    ("O", "0"), ("0", "O"), ("S", "5"), ("5", "S"), ("B", "8"), ("g", "q"),
    ("e", "c"), ("c", "e"), ("t", "f"), ("u", "ii"),
]


def corrupt(text: str, rate: float, rng: random.Random) -> str:
    """Apply OCR-like damage at roughly `rate` per character."""
    if rate <= 0:
        return text
    out, i = [], 0
    while i < len(text):
        if rng.random() < rate:
            choice = rng.random()
            if choice < 0.7:                                   # glyph confusion
                for src, dst in CONFUSIONS:
                    if text.startswith(src, i):
                        out.append(dst)
                        i += len(src)
                        break
                else:
                    out.append(text[i])
                    i += 1
            elif choice < 0.85 and text[i] == " ":             # lost word space
                i += 1
            else:                                              # spurious break
                out.append(text[i] + "-\n" if text[i].isalpha() else text[i])
                i += 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def score(ranked_doc_ids: list[list[str]], golden: list[dict], k: int) -> tuple[float, float, float]:
    """(hit@1, hit@k, MRR) at document level.

    hit@5 over a 14-document corpus saturates at 1.00 and measures nothing -
    pick a metric that can still move before you run the experiment. hit@1 and
    MRR keep resolution here.
    """
    hit1 = hitk = rr = 0.0
    for ids, item in zip(ranked_doc_ids, golden):
        gold = set(item["sources"])
        hit1 += ids[0] in gold if ids else 0
        hitk += any(d in gold for d in ids[:k])
        for rank, d in enumerate(ids[:k], start=1):
            if d in gold:
                rr += 1 / rank
                break
    n = len(golden)
    return hit1 / n, hitk / n, rr / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    chunks = chunk_documents(load_handbook())
    golden = [g for g in load_golden() if g["answerable"]]
    doc_ids = [c.metadata["doc_id"] for c in chunks]
    emb = get_embeddings()

    print(f"{len(chunks)} chunks, {len(golden)} answerable questions, k={args.k}")
    print("queries stay CLEAN; only the corpus is damaged\n")

    q_vecs = np.asarray(emb.embed_documents([g["question"] for g in golden]), dtype=np.float32)
    q_vecs /= np.linalg.norm(q_vecs, axis=1, keepdims=True)
    q_tokens = [tokenize(g["question"]) for g in golden]

    rates = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30]
    rows = []
    for rate in rates:
        rng = random.Random(args.seed)
        texts = [corrupt(c.page_content, rate, rng) for c in chunks]

        vecs = np.asarray(emb.embed_documents(texts), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        order = np.argsort(-(q_vecs @ vecs.T), axis=1)[:, : args.k]
        dense = score([[doc_ids[j] for j in row] for row in order], golden, args.k)

        bm25 = BM25Okapi([tokenize(t) for t in texts])
        ranked = [[doc_ids[j] for j in np.argsort(-bm25.get_scores(q))[: args.k]]
                  for q in q_tokens]
        lexical = score(ranked, golden, args.k)

        sample = corrupt("Full charge time 90 minutes, ingress protection IP54.", rate, rng)
        rows.append((rate, dense, lexical, sample))

    print(f"{'noise':>6} | {'dense h@1':>9} {'h@5':>5} {'MRR':>5} "
          f"| {'BM25 h@1':>9} {'h@5':>5} {'MRR':>5}")
    print("-" * 58)
    for rate, dense, lexical, _ in rows:
        print(f"{rate:6.0%} | {dense[0]:9.2f} {dense[1]:5.2f} {dense[2]:5.2f} "
              f"| {lexical[0]:9.2f} {lexical[1]:5.2f} {lexical[2]:5.2f}")

    print("\nwhat the corpus looks like at each rate:")
    for rate, _, _, sample in rows:
        print(f"{rate:6.0%}  {sample.replace(chr(10), '')[:64]}")

    print("\nMRR, plotted (# = 0.025):")
    for rate, dense, lexical, _ in rows:
        print(f"{rate:6.0%}  dense {'#' * round(dense[2] * 40):<40}{dense[2]:.2f}")
        print(f"        BM25  {'#' * round(lexical[2] * 40):<40}{lexical[2]:.2f}")

    base, worst = rows[0], rows[-1]
    print(f"\nfrom clean to {worst[0]:.0%} noise: "
          f"dense MRR {base[1][2]:.2f} -> {worst[1][2]:.2f} "
          f"({(worst[1][2]-base[1][2])/base[1][2]:+.0%}), "
          f"BM25 MRR {base[2][2]:.2f} -> {worst[2][2]:.2f} "
          f"({(worst[2][2]-base[2][2])/base[2][2]:+.0%})")


if __name__ == "__main__":
    main()
