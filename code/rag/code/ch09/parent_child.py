"""Chapter 9 - Parent/child retrieval ("small-to-big").

Tension: small chunks EMBED well (one idea → one clean vector) but READ badly
(the LLM gets a fragment with no surrounding context). Big chunks read well but
embed badly (many ideas → one blurry vector).

Fix: index SMALL child chunks, each carrying the id of its BIG parent; search on
children; hand the LLM the parents. Here parents are the 800-char chunks from
ragbook.chunk_documents and children are ~200-char pieces of them.

Run:  uv run python code/ch09/parent_child.py
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import models

from ragbook import chunk_documents, get_embeddings, get_qdrant_client, load_golden, load_handbook


def build_parent_child(collection: str = "ch09_children"):
    parents = chunk_documents(load_handbook(), chunk_size=800, chunk_overlap=120)
    parent_by_id = {p.metadata["chunk_id"]: p for p in parents}

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    children: list[Document] = []
    for p in parents:
        for j, piece in enumerate(child_splitter.split_text(p.page_content)):
            meta = dict(p.metadata, parent_id=p.metadata["chunk_id"], child_index=j)
            children.append(Document(page_content=piece, metadata=meta))

    client = get_qdrant_client()
    emb = get_embeddings()
    if not client.collection_exists(collection):
        dim = len(emb.embed_query("probe"))
        client.create_collection(collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
        store = QdrantVectorStore(client=client, collection_name=collection, embedding=emb, retrieval_mode=RetrievalMode.DENSE)
        store.add_documents(children)
    else:
        store = QdrantVectorStore(client=client, collection_name=collection, embedding=emb, retrieval_mode=RetrievalMode.DENSE)
    return store, parent_by_id, len(parents), len(children)


def retrieve_parents(store, parent_by_id, query: str, k_children: int = 8, k_parents: int = 3) -> list[Document]:
    """Search children, de-duplicate to parents preserving best-child order."""
    seen, out = set(), []
    for child in store.similarity_search(query, k=k_children):
        pid = child.metadata["parent_id"]
        if pid not in seen:
            seen.add(pid)
            out.append(parent_by_id[pid])
        if len(out) == k_parents:
            break
    return out


if __name__ == "__main__":
    store, parents, n_par, n_child = build_parent_child()
    print(f"{n_par} parent chunks (~800 chars) → {n_child} child chunks (~200 chars) indexed")

    q = "What is the P1 response time for Platinum support?"
    print(f"\nQ: {q}")
    print("children hit (what we SEARCH):")
    for c in store.similarity_search(q, k=3):
        print(f"   {c.metadata['source']} | {c.page_content[:90].replace(chr(10), ' ')!r}")
    print("parents returned (what the LLM READS):")
    for p in retrieve_parents(store, parents, q):
        print(f"   {p.metadata['source']} | {len(p.page_content)} chars | {p.page_content[:70].replace(chr(10), ' ')!r}")

    # Compare doc-level hit-rate@1 of child-search vs parent-search over the golden set.
    from ragbook import build_handbook_index
    plain = build_handbook_index("ch09_handbook", client=store.client)   # reuse the one client (embedded mode)
    golden = [g for g in load_golden() if g["answerable"]]
    hit_plain = hit_pc = 0
    for g in golden:
        if plain.similarity_search(g["question"], k=1)[0].metadata["doc_id"] in g["sources"]:
            hit_plain += 1
        if retrieve_parents(store, parents, g["question"], k_parents=1)[0].metadata["doc_id"] in g["sources"]:
            hit_pc += 1
    print(f"\nhit-rate@1 (doc level): plain 800-char chunks = {hit_plain / len(golden):.2f}, "
          f"child-search→parent = {hit_pc / len(golden):.2f}   (n={len(golden)})")
