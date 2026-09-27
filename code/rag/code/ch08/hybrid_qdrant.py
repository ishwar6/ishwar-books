"""Chapter 8 - Hybrid search: dense + BM25, three ways.

  A. Reciprocal Rank Fusion (RRF) by hand over two ranked lists.
  B. Qdrant-native hybrid through LangChain: QdrantVectorStore(retrieval_mode=HYBRID)
     with a dense vector (OpenAI) and a sparse vector (FastEmbed "Qdrant/bm25").
  C. The same thing with the raw client: query_points(prefetch=[...], query=FusionQuery(RRF)).

Then we measure hit-rate@1/3/5 for dense / BM25 / RRF-by-hand / Qdrant-hybrid on the golden set.

Run:  uv run python code/ch08/hybrid_qdrant.py
First run downloads the small "Qdrant/bm25" FastEmbed model (~few MB).
"""
from __future__ import annotations

import re
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import models

from ragbook import build_handbook_index, chunk_documents, get_embeddings, get_qdrant_client, load_golden, load_handbook


def tokenize(text: str) -> list[str]:
    """Same tokenizer as bm25_from_scratch.py. The default BM25Retriever tokenizer
    only splits on whitespace, so "(BEACON-4187)" would never match "BEACON-4187"."""
    return re.findall(r"[a-z0-9][a-z0-9.\-]*", text.lower())


# ------------------------------------------------------------ A. RRF by hand ---
def rrf(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion.

    score(d) = sum over lists L of 1 / (k + rank_L(d))      (rank starts at 1)

    k=60 is the value from the original 2009 paper; it damps the advantage of
    being #1 in one list versus #3 in two lists. Scores from different systems are
    never compared directly - only ranks are - which is why RRF needs no
    normalisation and works for any pair of retrievers."""
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, doc_id in enumerate(lst, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


# ----------------------------------------------- B. Qdrant-native hybrid store ---
def build_hybrid_store(chunks: list[Document], collection: str = "ch08_hybrid", client=None) -> QdrantVectorStore:
    client = client or get_qdrant_client()
    dense = get_embeddings()
    sparse = FastEmbedSparse(model_name="Qdrant/bm25")  # BM25 as a *sparse vector*: {token_id: weight}

    if not client.collection_exists(collection):
        dim = len(dense.embed_query("probe"))
        # A collection can hold several NAMED vectors per point. Here: one dense, one sparse.
        client.create_collection(
            collection_name=collection,
            vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))},
        )
        store = QdrantVectorStore(
            client=client,
            collection_name=collection,
            embedding=dense,
            sparse_embedding=sparse,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
        return store
    return QdrantVectorStore(
        client=client, collection_name=collection, embedding=dense, sparse_embedding=sparse,
        retrieval_mode=RetrievalMode.HYBRID, vector_name="dense", sparse_vector_name="sparse",
    )


# ------------------------------------------------ C. raw client hybrid query ---
def raw_hybrid(store: QdrantVectorStore, query: str, k: int = 5):
    """What QdrantVectorStore(HYBRID) does under the hood: two prefetches fused server-side."""
    dense_vec = store.embeddings.embed_query(query)
    sp = store.sparse_embeddings.embed_query(query)
    res = store.client.query_points(
        collection_name=store.collection_name,
        prefetch=[
            models.Prefetch(query=models.SparseVector(indices=sp.indices, values=sp.values), using="sparse", limit=20),
            models.Prefetch(query=dense_vec, using="dense", limit=20),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),  # or models.Fusion.DBSF
        limit=k,
        with_payload=True,
    )
    return [(p.payload["metadata"]["source"], round(p.score, 4)) for p in res.points]


def hit(docs_sources: list[str], gold_sources: list[str]) -> bool:
    return any(s in gold_sources for s in docs_sources)


if __name__ == "__main__":
    chunks = chunk_documents(load_handbook())
    by_id = {c.metadata["chunk_id"]: c for c in chunks}

    client = get_qdrant_client()          # ONE client per process (embedded mode locks the folder)
    dense_store = build_handbook_index("ch08_handbook", client=client)
    bm25 = BM25Retriever.from_documents(chunks, k=10, preprocess_func=tokenize)
    hybrid_store = build_hybrid_store(chunks, client=client)

    q = "BEACON-4187 fix"
    print(f"q={q!r}")
    dense_ids = [d.metadata["chunk_id"] for d in dense_store.similarity_search(q, k=10)]
    bm25_ids = [d.metadata["chunk_id"] for d in bm25.invoke(q)]
    print("  A. RRF by hand:")
    for cid, s in rrf([dense_ids, bm25_ids])[:3]:
        print(f"     {s:.4f} {by_id[cid].metadata['source']}")
    print("  B. QdrantVectorStore HYBRID:")
    for d, s in hybrid_store.similarity_search_with_score(q, k=3):
        print(f"     {s:.4f} {d.metadata['source']}")
    print("  C. raw query_points + FusionQuery(RRF):", raw_hybrid(hybrid_store, q, 3))

    # ------------------------------------------ hit-rate@k on the golden set
    # hit-rate@k = fraction of questions where at least one of the top-k chunks
    # comes from a gold source document. We report k=1, 3, 5: on a 14-document
    # corpus everything looks perfect at k=5, so k=1 is where retrievers differ.
    golden = [g for g in load_golden() if g["answerable"]]
    names = ["dense", "bm25", "rrf_by_hand", "qdrant_hybrid"]
    ks = (1, 3, 5)
    hits = {n: {k: 0 for k in ks} for n in names}
    misses_at1 = {n: [] for n in names}
    for g in golden:
        gold = [s + ".md" for s in g["sources"]]
        d_docs = dense_store.similarity_search(g["question"], k=10)
        b_docs = bm25.invoke(g["question"])
        fused = rrf([[d.metadata["chunk_id"] for d in d_docs], [d.metadata["chunk_id"] for d in b_docs]])
        results = {
            "dense": [d.metadata["source"] for d in d_docs],
            "bm25": [d.metadata["source"] for d in b_docs],
            "rrf_by_hand": [by_id[c].metadata["source"] for c, _ in fused],
            "qdrant_hybrid": [d.metadata["source"] for d in hybrid_store.similarity_search(g["question"], k=5)],
        }
        for name, srcs in results.items():
            for k in ks:
                if hit(srcs[:k], gold):
                    hits[name][k] += 1
            if not hit(srcs[:1], gold):
                misses_at1[name].append(g["id"])

    n = len(golden)
    print(f"\nhit-rate@k over {n} answerable golden questions")
    print(f"{'retriever':<15}" + "".join(f"{'@'+str(k):>8}" for k in ks) + "   misses@1")
    for name in names:
        row = "".join(f"{hits[name][k] / n:>8.2f}" for k in ks)
        print(f"{name:<15}{row}   {misses_at1[name]}")

    # ------------------------------------------ where hybrid earns its keep: ID-style queries
    # Golden questions are full sentences, which embeddings handle well. Real users
    # type fragments: ticket ids, commands, standards, names. Compare top-1 source.
    print("\nID-style queries - top-1 source")
    print(f"{'query':<22}{'dense':<42}{'qdrant_hybrid'}")
    for q in ["BEACON-4187", "beaconctl certs renew", "ISO 3691-4", "Veldmark", "cert-renewer", "IP54"]:
        d = dense_store.similarity_search(q, k=1)[0].metadata["source"]
        h = hybrid_store.similarity_search(q, k=1)[0].metadata["source"]
        print(f"{q:<22}{d:<42}{h}")

