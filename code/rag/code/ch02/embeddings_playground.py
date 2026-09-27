"""Chapter 2 - Embeddings playground.

What this shows, in order:
  1. An embedding is just a list of floats. Same model -> same length, every time.
  2. Cosine similarity by hand (numpy), and why it equals a dot product on
     normalized vectors while euclidean distance ranks the same way.
  3. Nearest-neighbour search over a handful of sentences = the core of RAG.
  4. Matryoshka: asking for 256 dims instead of 1536 barely changes rankings.
  5. Tokens and cost: how much embedding a corpus actually costs.

Run:  uv run python code/ch02/embeddings_playground.py
"""
import numpy as np
import tiktoken
from langchain_openai import OpenAIEmbeddings

from ragbook import EMBED_MODEL, get_embeddings

# Ten "document" sentences, written like the handbook. Note that some pairs
# share words but not meaning (robots/kg vs PTO/days), and some share meaning
# but not words ("leave" vs "vacation").
SENTENCES = [
    "Full-time employees receive 24 days of PTO per calendar year.",
    "Up to 5 unused vacation days can be carried over until 31 March.",
    "The Atlas A2 robot carries a maximum payload of 250 kg.",
    "A full charge of the Atlas battery takes 90 minutes.",
    "The Beacon API is limited to 600 requests per minute per API key.",
    "Report security incidents to security@lumora.example within one hour.",
    "Hotel rooms in Europe are capped at 180 euros per night.",
    "Beacon ships a release every two weeks on a Tuesday.",
    "Privileged account passwords rotate every 90 days.",
    "Parental leave is 26 weeks for the primary caregiver.",
]

QUERIES = [
    "how much holiday do I get",                 # no shared words with sentence 0
    "how heavy a load can the robot lift",       # paraphrase of sentence 2
    "how often do passwords expire",             # 'expire' vs 'rotate'
    "90",                                        # a bare number: matches two sentences lexically
]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """cos(theta) = (a . b) / (|a| |b|). Ranges -1..1; 1 = same direction."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def rank(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
    """Top-k indices by cosine similarity. This IS a vector search, brute force."""
    sims = [cosine(query_vec, d) for d in doc_vecs]
    order = np.argsort(sims)[::-1][:k]
    return [(int(i), sims[i]) for i in order]


def main() -> None:
    embeddings = get_embeddings()

    # ---- 1. what an embedding is -------------------------------------------
    doc_vecs = np.array(embeddings.embed_documents(SENTENCES))
    print(f"model={EMBED_MODEL}  shape={doc_vecs.shape}  (10 sentences x {doc_vecs.shape[1]} floats)")
    print("first 5 numbers of sentence 0:", np.round(doc_vecs[0][:5], 4))
    print("vector norm (OpenAI embeddings come out unit-length):", round(float(np.linalg.norm(doc_vecs[0])), 4))

    # ---- 2. three ways to compare vectors ----------------------------------
    a, b = doc_vecs[0], doc_vecs[1]      # PTO vs carry-over (related)
    c = doc_vecs[2]                      # robot payload (unrelated)
    print("\ncosine(PTO, carry-over) =", round(cosine(a, b), 3), " cosine(PTO, payload) =", round(cosine(a, c), 3))
    print("dot   (PTO, carry-over) =", round(float(a @ b), 3), " dot   (PTO, payload) =", round(float(a @ c), 3))
    print("eucl  (PTO, carry-over) =", round(euclidean(a, b), 3), " eucl  (PTO, payload) =", round(euclidean(a, c), 3))
    print("-> on unit vectors: cosine == dot, and euclidean^2 == 2 - 2*cosine. Same ranking, three names.")

    # ---- 3. nearest neighbours: the heart of retrieval ---------------------
    print("\nNearest neighbours (cosine):")
    for q in QUERIES:
        qv = np.array(embeddings.embed_query(q))
        print(f"\n  Q: {q!r}")
        for i, s in rank(qv, doc_vecs):
            print(f"     {s:.3f}  {SENTENCES[i]}")

    # ---- 4. Matryoshka: shorter vectors, almost the same ranking -----------
    small = OpenAIEmbeddings(model=EMBED_MODEL, dimensions=256)
    small_docs = np.array(small.embed_documents(SENTENCES))
    print(f"\nSame model with dimensions=256 -> shape {small_docs.shape}")
    # Compare the top-3 SETS, not just the top-1: when two candidates are nearly
    # tied (0.420 vs 0.406 above) any small change can swap them, and that is fine.
    overlap = 0
    for q in QUERIES:
        full_top = {i for i, _ in rank(np.array(embeddings.embed_query(q)), doc_vecs, k=3)}
        small_top = {i for i, _ in rank(np.array(small.embed_query(q)), small_docs, k=3)}
        overlap += len(full_top & small_top)
    print(f"top-3 overlap 1536 vs 256 dims: {overlap}/{3 * len(QUERIES)} slots agree  (6x less storage)")

    # ---- 5. tokens and money ------------------------------------------------
    enc = tiktoken.get_encoding("cl100k_base")   # tokenizer used by text-embedding-3-*
    total_tokens = sum(len(enc.encode(s)) for s in SENTENCES)
    print(f"\n{len(SENTENCES)} sentences = {total_tokens} tokens")
    for model, price in [("text-embedding-3-small", 0.02), ("text-embedding-3-large", 0.13)]:
        gb_tokens = 250_000_000          # ~1 GB of English text ~ 250M tokens
        print(f"  {model}: ${price}/1M tokens -> this corpus ${total_tokens * price / 1e6:.6f}; 1 GB of text ~ ${gb_tokens * price / 1e6:,.0f}")


if __name__ == "__main__":
    main()
