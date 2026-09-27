"""Chapter 7 - ingest_batch.py: incremental, batched ingestion with a manifest.

Adds three things the Chapter 6 ingest did not have:
  1. content-hash ids   -> identical text is never embedded twice (dedup)
  2. a manifest         -> only NEW chunks are embedded; chunks whose source vanished are deleted
  3. batching + timing  -> embed 100 chunks per API call, report throughput and cost
Then it measures what the extra (mostly irrelevant) data does to retrieval quality.

Run:  uv run python code/ch07/ingest_batch.py            (handbook + generated + gutenberg)
      uv run python code/ch07/ingest_batch.py --max-chunks-per-book 150
"""
import argparse
import hashlib
import json
import time
import uuid

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import models

from ragbook import DATA_DIR, chunk_documents, get_embeddings, get_qdrant_client, load_golden, load_handbook

COLLECTION = "ch07_corpus"
MANIFEST = DATA_DIR / ".manifest_ch07.json"      # {point_id: source}; in production this is a table
EMBED_PRICE_PER_M = 0.02                          # text-embedding-3-small
BATCH = 100


def content_id(text: str, source: str) -> str:
    """Same text in the same source -> same id. Re-ingest = no-op. Changed text -> new id (old one deleted via manifest)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, hashlib.sha256(f"{source}\n{text}".encode()).hexdigest()))


def load_noise(max_chunks_per_book: int) -> list[Document]:
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    for path in sorted((DATA_DIR / "gutenberg").glob("*.txt")):
        book = Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name, "doc_id": path.stem, "kind": "noise"})
        docs.extend(splitter.split_documents([book])[:max_chunks_per_book])
    return docs


def load_generated() -> list[Document]:
    return [Document(page_content=p.read_text(), metadata={"source": p.name, "doc_id": p.stem, "kind": "generated"})
            for p in sorted((DATA_DIR / "generated").glob("*.md"))]


def sync(store: QdrantVectorStore, chunks: list[Document]) -> None:
    """Bring the collection in line with `chunks`: add what is new, delete what is gone."""
    manifest: dict[str, str] = json.loads(MANIFEST.read_text()) if MANIFEST.exists() and store.client.collection_exists(COLLECTION) else {}
    wanted = {content_id(c.page_content, c.metadata["source"]): c for c in chunks}

    new_ids = [i for i in wanted if i not in manifest]
    gone_ids = [i for i in manifest if i not in wanted]
    print(f"  {len(wanted)} chunks wanted, {len(manifest)} already indexed -> {len(new_ids)} to embed, {len(gone_ids)} to delete")

    if gone_ids:
        store.client.delete(COLLECTION, points_selector=models.PointIdsList(points=gone_ids))

    t0, chars = time.perf_counter(), 0
    for start in range(0, len(new_ids), BATCH):
        batch_ids = new_ids[start : start + BATCH]
        batch_docs = [wanted[i] for i in batch_ids]
        # OpenAI's client retries 429/5xx with exponential backoff by default (max_retries=2);
        # raise it in OpenAIEmbeddings(max_retries=6) for bulk jobs. Respect Retry-After.
        store.add_documents(batch_docs, ids=batch_ids)
        chars += sum(len(d.page_content) for d in batch_docs)
        print(f"    batch {start // BATCH + 1}: {len(batch_ids)} chunks  ({time.perf_counter() - t0:.1f}s so far)")
    dt = time.perf_counter() - t0
    if new_ids:
        approx_tokens = chars / 4
        print(f"  embedded {len(new_ids)} chunks in {dt:.1f}s = {len(new_ids) / dt:.0f} chunks/s, ~{approx_tokens:,.0f} tokens, ~${approx_tokens * EMBED_PRICE_PER_M / 1e6:.4f}")

    manifest = {i: c.metadata["source"] for i, c in wanted.items()}
    MANIFEST.write_text(json.dumps(manifest, indent=0))


def hit_rates(store: QdrantVectorStore, golden: list[dict]) -> tuple[float, float, float]:
    """hit@5, hit@1, and noise@5 = share of top-5 slots taken by chunks that are NOT handbook
    (the context window the model sees gets polluted long before hit-rate drops)."""
    h5 = h1 = noise = 0
    for q in golden:
        docs = store.similarity_search(q["question"], k=5)
        ids = [d.metadata["doc_id"] for d in docs]
        h5 += bool(set(ids) & set(q["sources"]))
        h1 += ids[0] in q["sources"]
        noise += sum(d.metadata.get("kind") in ("noise", "generated") for d in docs) / 5
    n = len(golden)
    return h5 / n, h1 / n, noise / n


def main(max_chunks_per_book: int) -> None:
    client, embeddings = get_qdrant_client(), get_embeddings()
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)          # start clean so the before/after is honest
    MANIFEST.unlink(missing_ok=True)
    client.create_collection(COLLECTION, vectors_config=models.VectorParams(size=len(embeddings.embed_query("p")), distance=models.Distance.COSINE))
    store = QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=embeddings)
    golden = [q for q in load_golden() if q["answerable"]]

    handbook = chunk_documents(load_handbook())
    print("1) handbook only")
    sync(store, handbook)
    h5, h1, nz = hit_rates(store, golden)
    print(f"  hit@5 {h5:.2f}  hit@1 {h1:.2f}  noise@5 {nz:.2f}   ({client.count(COLLECTION).count} points)")

    print("2) re-run with the same input (should embed nothing)")
    sync(store, handbook)

    print("3) add generated docs + Gutenberg noise")
    everything = handbook + chunk_documents(load_generated()) + load_noise(max_chunks_per_book)
    sync(store, everything)
    h5, h1, nz = hit_rates(store, golden)
    print(f"  hit@5 {h5:.2f}  hit@1 {h1:.2f}  noise@5 {nz:.2f}   ({client.count(COLLECTION).count} points)")

    print("4) remove the noise again (manifest deletes the vanished chunks)")
    sync(store, handbook + chunk_documents(load_generated()))
    print(f"  {client.count(COLLECTION).count} points remain")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chunks-per-book", type=int, default=300)
    main(ap.parse_args().max_chunks_per_book)
