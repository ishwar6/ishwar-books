"""Build (or reuse) a Qdrant index over the handbook corpus.

Chapter 6 builds this pipeline step by step in `code/ch06/`. From Chapter 7 on,
every chapter needs "an index of the handbook" as a *starting point*, so the
finished pipeline lives here. It is deliberately small: load -> chunk -> embed
-> upsert, with stable ids so re-running is idempotent.
"""
from __future__ import annotations

import uuid
import warnings

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragbook.common import get_embeddings, get_qdrant_client, load_handbook

# Qdrant imports are deliberately *inside* the function below. `langchain_qdrant` pulls in
# fastembed/onnxruntime and grpc at import time; keeping them lazy means the LangChain-only
# chapters (2 and 3) never load them, and start faster.

NAMESPACE = uuid.UUID("6f1c2a3e-0000-4000-8000-00000000ba9e")


def chunk_documents(
    docs: list[Document], chunk_size: int = 800, chunk_overlap: int = 120
) -> list[Document]:
    """Recursive character chunking. Each chunk keeps the parent's metadata and
    gets `chunk_index` + a stable `chunk_id` (so re-ingesting overwrites, never duplicates)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],  # prefer heading boundaries
    )
    chunks: list[Document] = []
    for doc in docs:
        for i, piece in enumerate(splitter.split_text(doc.page_content)):
            meta = dict(doc.metadata, chunk_index=i)
            meta["chunk_id"] = str(uuid.uuid5(NAMESPACE, f"{meta['doc_id']}::{i}"))
            chunks.append(Document(page_content=piece, metadata=meta))
    return chunks


def build_handbook_index(
    collection_name: str = "handbook",
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    recreate: bool = False,
    client=None,
    embeddings=None,
):
    """Return a QdrantVectorStore over the handbook. Builds the collection if it
    does not exist (or if recreate=True); otherwise reuses what is on disk."""
    from langchain_qdrant import QdrantVectorStore, RetrievalMode
    from qdrant_client import models

    client = client or get_qdrant_client()
    embeddings = embeddings or get_embeddings()

    if recreate and client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    if not client.collection_exists(collection_name):
        dim = len(embeddings.embed_query("dimension probe"))
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        # payload indexes make metadata filters fast on a Qdrant *server*
        # (Chapters 4 and 18); the embedded local mode ignores them, so hush its warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            client.create_payload_index(collection_name, "metadata.doc_id", models.PayloadSchemaType.KEYWORD)
            client.create_payload_index(collection_name, "metadata.source", models.PayloadSchemaType.KEYWORD)

        store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
            retrieval_mode=RetrievalMode.DENSE,
        )
        chunks = chunk_documents(load_handbook(), chunk_size, chunk_overlap)
        store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
        return store

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        retrieval_mode=RetrievalMode.DENSE,
    )
