"""Chapter 6 - ingest.py: load -> chunk -> stable ids -> embed -> upsert.

This is the "write path" of a RAG system. It runs when documents change, not
when users ask questions. Idempotent: run it twice, get the same collection.

Run:  uv run python code/ch06/ingest.py [--recreate]
"""
import argparse
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models

from ragbook import DATA_DIR, chunk_documents, get_embeddings, get_qdrant_client, load_handbook

COLLECTION = "ch06_handbook"


def load_sources() -> list[Document]:
    """Load everything we want to index.

    The handbook comes through load_handbook(). Two other loaders you will meet
    in real projects (not used here, so the run stays offline-safe):

        from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
        DirectoryLoader("data/handbook", glob="**/*.md", loader_cls=TextLoader).load()
        PyPDFLoader("report.pdf").load()          # one Document per page, metadata["page"]

    Whatever the loader, the output is the same shape: list[Document(page_content, metadata)].
    """
    docs = load_handbook()
    generated = DATA_DIR / "generated"                       # Chapter 7 adds synthetic docs here
    for path in sorted(generated.glob("*.md")) if generated.exists() else []:
        docs.append(Document(page_content=path.read_text(), metadata={"source": path.name, "doc_id": path.stem, "title": path.stem}))
    return docs


def ingest(collection: str = COLLECTION, recreate: bool = False, chunk_size: int = 800, chunk_overlap: int = 120, client=None) -> QdrantVectorStore:
    # Pass a client in when you call ingest() more than once in a process: one client per
    # process is how Qdrant is meant to be used (and in memory mode a fresh client would be
    # a fresh, empty database - no idempotency to demonstrate).
    client = client or get_qdrant_client()
    embeddings = get_embeddings()

    if recreate and client.collection_exists(collection):
        client.delete_collection(collection)
    if not client.collection_exists(collection):
        dim = len(embeddings.embed_query("probe"))
        client.create_collection(collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))

    store = QdrantVectorStore(client=client, collection_name=collection, embedding=embeddings)

    docs = load_sources()
    chunks = chunk_documents(docs, chunk_size, chunk_overlap)   # adds chunk_index + chunk_id (uuid5 of doc_id::index)

    # Stable ids are the whole trick to idempotency: the same chunk always maps
    # to the same point id, so a re-run overwrites instead of duplicating.
    t0 = time.perf_counter()
    store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    dt = time.perf_counter() - t0

    print(f"{len(docs)} docs -> {len(chunks)} chunks -> {client.count(collection).count} points in {collection!r} ({dt:.1f}s)")
    return store


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true", help="drop and rebuild the collection")
    args = ap.parse_args()
    client = get_qdrant_client()
    ingest(recreate=args.recreate, client=client)
    ingest(client=client)   # second run, same client: same point count -> idempotent
