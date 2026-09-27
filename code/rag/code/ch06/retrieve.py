"""Chapter 6 - retrieve.py: the read path, part 1.

question -> embedding -> top-k chunks (optionally filtered / thresholded) -> context string.

Run:  uv run python code/ch06/retrieve.py "How many PTO days do I get?"
"""
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models

from ragbook import format_docs, get_embeddings, get_qdrant_client

sys.path.insert(0, str(Path(__file__).parent))
from ingest import COLLECTION, ingest  # noqa: E402


def get_store(collection: str = COLLECTION) -> QdrantVectorStore:
    """Attach to the collection; build it first if it is missing (memory mode, fresh machine)."""
    client = get_qdrant_client()
    if not client.collection_exists(collection):
        return ingest(collection, client=client)   # reuse THIS client: embedded mode allows one per process
    return QdrantVectorStore(client=client, collection_name=collection, embedding=get_embeddings())


def retrieve(
    store: QdrantVectorStore,
    question: str,
    k: int = 4,
    doc_id: str | None = None,
    score_threshold: float | None = None,
) -> list[tuple[Document, float]]:
    """Top-k chunks with cosine scores.

    doc_id           restrict to one document (payload filter - cheap and exact)
    score_threshold  drop weak matches instead of padding the context with noise
    """
    flt = None
    if doc_id:
        flt = models.Filter(must=[models.FieldCondition(key="metadata.doc_id", match=models.MatchValue(value=doc_id))])
    return store.similarity_search_with_score(question, k=k, filter=flt, score_threshold=score_threshold)


def build_context(hits: list[tuple[Document, float]]) -> str:
    """Numbered context so the model can cite [n] and we can check it."""
    return format_docs([doc for doc, _ in hits])


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "How many PTO days do I get?"
    store = get_store()
    hits = retrieve(store, question, k=4)
    print(f"Q: {question}\n")
    for i, (doc, score) in enumerate(hits, start=1):
        print(f"[{i}] {score:.3f}  {doc.metadata['source']}  chunk {doc.metadata['chunk_index']}")
    print("\n--- context the model will see ---")
    print(build_context(hits)[:1200], "...")
