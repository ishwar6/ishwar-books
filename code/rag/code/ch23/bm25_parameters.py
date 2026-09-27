"""Chapter 23 - k1, b and IDF: what the three parts of BM25 actually do.

    QDRANT_MODE=memory uv run python code/ch23/bm25_parameters.py

Chapter 8 gave the formula and used k1=1.5, b=0.75 without justifying them. Here
each term is plotted on its own, then swept against retrieval quality.

    score(q,d) = SUM_t  IDF(t) . tf(t,d)(k1+1) / (tf(t,d) + k1(1 - b + b|d|/avgdl))
                         ^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         rarity              saturating, length-normalised TF
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrievers import BM25, evaluate, load_corpus, tokenize   # noqa: E402


def tf_saturation() -> None:
    print("=" * 74)
    print("1. k1 - how fast repetition stops helping")
    print("=" * 74)
    print("   tf_factor = f(k1+1)/(f+k1)   (b=0 for clarity; the asymptote is k1+1)")
    print(f"\n{'occurrences f':>14} " + "".join(f"{('k1=' + str(k)):>10}" for k in (0.0, 0.5, 1.2, 2.0, 10.0)))
    print("-" * 74)
    for f in (1, 2, 3, 5, 10, 20):
        row = f"{f:>14} "
        for k1 in (0.0, 0.5, 1.2, 2.0, 10.0):
            row += f"{f * (k1 + 1) / (f + k1):>10.2f}" if k1 > 0 else f"{1.0:>10.2f}"
        print(row)
    print("""
  k1=0   : term presence only. 1 occurrence == 20 occurrences (boolean retrieval).
  k1=1.2 : the 2nd occurrence adds ~0.38, the 20th adds ~0.01. Nearly saturated by 5.
  k1=10  : nearly linear in f over this range - closer to raw TF-IDF.

  Why saturate at all? Because a chunk saying "battery" 20 times is not 20x more
  about batteries than one saying it twice; it is usually just longer or repetitive.
  Saturation is what stops keyword-stuffed documents from dominating.""")


def length_norm() -> None:
    print("\n" + "=" * 74)
    print("2. b - how hard long documents are penalised")
    print("=" * 74)
    print("   denominator factor = k1(1 - b + b.|d|/avgdl), shown as the divisor at k1=1.5")
    print(f"\n{'|d|/avgdl':>12} " + "".join(f"{('b=' + str(b)):>10}" for b in (0.0, 0.3, 0.75, 1.0)))
    print("-" * 74)
    for ratio in (0.25, 0.5, 1.0, 2.0, 4.0):
        row = f"{ratio:>12.2f} "
        for b in (0.0, 0.3, 0.75, 1.0):
            row += f"{1.5 * (1 - b + b * ratio):>10.2f}"
        print(row)
    print("""
  b=0    : length ignored. A 10,000-word page beats a precise 2-line answer,
           because it contains every term at least once.
  b=1    : fully normalised. A term in a short chunk counts far more.
  b=0.75 : the empirical compromise from the TREC experiments.

  In RAG this parameter matters LESS than in web search, because you control chunk
  size: if every chunk is ~800 characters, |d|/avgdl is near 1 for everything and
  the b term is nearly constant. Chunking is doing the length normalisation for you.""")


def idf_behaviour(bm: BM25) -> None:
    print("\n" + "=" * 74)
    print("3. IDF - and why the +1 inside the log is not cosmetic")
    print("=" * 74)
    n = bm.n
    print(f"corpus: {n} chunks\n")
    print(f"{'term':<16} {'df':>4} {'idf (lucene)':>13} {'idf (raw)':>11}  note")
    print("-" * 74)
    interesting = sorted(bm.df.items(), key=lambda kv: -kv[1])[:4]
    rare = [(t, c) for t, c in bm.df.items() if c == 1][:3]
    for t, c in interesting + rare:
        note = "in most chunks" if c > n / 2 else ("unique" if c == 1 else "")
        print(f"{t[:16]:<16} {c:>4} {bm.idf[t]:>13.3f} {bm.raw_idf[t]:>11.3f}  {note}")
    neg = [t for t, c in bm.df.items() if bm.raw_idf[t] < 0]
    print(f"\n{len(neg)} terms have NEGATIVE raw idf (they appear in more than half the")
    print(f"chunks): {', '.join(neg[:6])}")
    print("""
  Without the +1, a term in >N/2 documents gets a negative weight, so a document
  is PENALISED for containing it - a chunk mentioning "the" would score below one
  that never says it. Lucene's log(1 + x) form keeps IDF positive and monotonic.
  This is the kind of detail that separates "I used BM25" from "I understand BM25".""")


def sweep(texts, doc_ids, golden) -> None:
    print("\n" + "=" * 74)
    print("4. does any of this move retrieval quality? (42 golden questions)")
    print("=" * 74)
    print(f"{'k1':>6} {'b':>6} {'recall@5':>10} {'MRR':>8} {'nDCG@5':>9}")
    print("-" * 74)
    best = None
    for k1 in (0.0, 0.6, 1.2, 1.5, 2.0, 4.0):
        for b in (0.0, 0.5, 0.75, 1.0):
            bm = BM25(texts, k1=k1, b=b)
            B = np.stack([bm.scores(g["question"]) for g in golden])
            m = evaluate(lambda qi: list(np.argsort(-B[qi])), doc_ids, golden, k=5)
            if best is None or m["mrr"] > best[0]:
                best = (m["mrr"], k1, b)
            if b in (0.0, 0.75) or k1 in (1.2,):
                print(f"{k1:>6.1f} {b:>6.2f} {m['recall@5']:>10.3f} {m['mrr']:>8.3f} {m['ndcg@5']:>9.3f}")
    print(f"\nbest MRR {best[0]:.3f} at k1={best[1]}, b={best[2]}")
    print("""
  The spread across this whole grid is small, and that is the honest finding: on a
  corpus of uniform, well-chunked documents, k1 and b are second-order. Spend your
  tuning budget on the tokenizer (Chapter 8 showed BEACON-4187 being split into
  'beacon'+'4187'), on chunking, and on whether you run BM25 at all - not on k1.
  Sweep them once, keep the defaults unless the data says otherwise, and be able to
  explain what they do.""")


if __name__ == "__main__":
    chunks, texts, doc_ids, golden = load_corpus()
    tf_saturation()
    length_norm()
    idf_behaviour(BM25(texts))
    sweep(texts, doc_ids, golden)
