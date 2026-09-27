"""Chapter 8 - BM25 from scratch, then compared to the libraries.

BM25 is the 1990s keyword-ranking formula that still powers most search engines.
It is the "sparse" / "lexical" half of hybrid search. Here we:

  1. implement it in ~40 lines on the handbook chunks,
  2. check our numbers against rank_bm25.BM25Okapi (the reference implementation),
  3. wrap the same thing as a LangChain retriever (BM25Retriever),
  4. show queries where keywords win over embeddings, and where they lose.

Run:  uv run python code/ch08/bm25_from_scratch.py
"""
from __future__ import annotations

import math
import re
import warnings
from collections import Counter

warnings.filterwarnings("ignore", category=DeprecationWarning)  # langchain-community sunset notice

from langchain_community.retrievers import BM25Retriever
from rank_bm25 import BM25Okapi

from ragbook import build_handbook_index, chunk_documents, load_handbook

# --------------------------------------------------------------- tokenizer ---
def tokenize(text: str) -> list[str]:
    """Keep digits, dots and hyphens inside tokens so "3.8", "BEACON-4187" and
    "beaconctl" survive as searchable units."""
    return re.findall(r"[a-z0-9][a-z0-9.\-]*", text.lower())


# ------------------------------------------------------------- BM25 by hand ---
class BM25:
    """Okapi BM25.

    score(q, d) = sum over query terms t of
        IDF(t) * tf(t,d) * (k1 + 1) / ( tf(t,d) + k1 * (1 - b + b * |d| / avgdl) )

    IDF(t) = ln( (N - n_t + 0.5) / (n_t + 0.5) + 1 )     # +1 keeps IDF positive
      N     = number of documents, n_t = documents containing t
    k1 (≈1.5) : how quickly repeated terms stop adding score (term-frequency saturation)
    b  (≈0.75): how much to penalise long documents (0 = not at all, 1 = fully)
    """

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tokenized = [tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.tokenized]
        self.avgdl = sum(self.doc_len) / len(docs)
        self.tf = [Counter(t) for t in self.tokenized]
        df = Counter(term for toks in self.tokenized for term in set(toks))
        n = len(docs)
        self.idf = {t: math.log((n - c + 0.5) / (c + 0.5) + 1) for t, c in df.items()}

    def score(self, query: str, i: int) -> float:
        s = 0.0
        for t in tokenize(query):
            if t not in self.tf[i]:
                continue
            f = self.tf[i][t]
            norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
            s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + norm)
        return s

    def top_k(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(len(self.tokenized))]
        return sorted(scores, key=lambda x: -x[1])[:k]


def show(title: str, chunks, ranked: list[tuple[int, float]]):
    print(f"\n{title}")
    for i, s in ranked:
        c = chunks[i]
        print(f"  {s:6.3f}  {c.metadata['source']:<40} {c.page_content[:70].replace(chr(10), ' ')!r}")


if __name__ == "__main__":
    chunks = chunk_documents(load_handbook())
    texts = [c.page_content for c in chunks]
    print(f"{len(chunks)} chunks, avg length {sum(len(tokenize(t)) for t in texts) / len(texts):.0f} tokens")

    # 1. ours
    ours = BM25(texts)
    # 2. reference implementation, same tokenizer, same k1/b
    ref = BM25Okapi([tokenize(t) for t in texts], k1=1.5, b=0.75)

    q = "beaconctl rollout undo"
    show(f"OURS      q={q!r}", chunks, ours.top_k(q, 3))
    ref_scores = ref.get_scores(tokenize(q))
    ref_top = sorted(enumerate(ref_scores), key=lambda x: -x[1])[:3]
    show(f"rank_bm25 q={q!r}", chunks, ref_top)
    # rank_bm25 uses a slightly different IDF (no +1, floor for negatives), so
    # absolute numbers differ a little; the ORDER is what matters and should match.
    assert [i for i, _ in ours.top_k(q, 3)] == [i for i, _ in ref_top], "ranking mismatch"
    print("\n✓ same top-3 ordering as rank_bm25")

    # 3. the LangChain wrapper - a Retriever like any other (Ch. 6 pipeline can swap it in)
    bm25_retriever = BM25Retriever.from_documents(chunks, k=3, preprocess_func=tokenize)
    print("\nBM25Retriever.invoke('BEACON-4187'):")
    for d in bm25_retriever.invoke("BEACON-4187"):
        print("   ", d.metadata["source"], "|", d.page_content[:70].replace("\n", " "))

    # 4. lexical vs dense - where each wins
    store = build_handbook_index("ch08_handbook")
    for q in [
        "BEACON-4187",                       # exact ticket id: keyword should nail it
        "firmware 3.8",                      # version number
        "how fast can the robot move",       # paraphrase of "maximum speed": dense should win
        "what happens if I get sick for a week",  # paraphrase: dense should win
    ]:
        b = [chunks[i].metadata["source"] for i, _ in ours.top_k(q, 3)]
        d = [x.metadata["source"] for x in store.similarity_search(q, k=3)]
        print(f"\nq={q!r}\n  BM25 : {b}\n  dense: {d}")
