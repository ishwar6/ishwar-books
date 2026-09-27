"""Chapter 4 - Qdrant with the raw client (no LangChain).

Doing it once by hand shows what the LangChain wrapper does for you later:
  create a collection -> upsert points (id, vector, payload) -> query -> filter -> delete.

Run:  uv run python code/ch04/qdrant_raw_client.py
"""
import uuid
import warnings

from qdrant_client import models

from ragbook import get_embeddings, get_qdrant_client, load_handbook

COLLECTION = "ch04_raw"


def main() -> None:
    client = get_qdrant_client()          # memory / local disk / server - same API
    embeddings = get_embeddings()

    # ---- 1. a collection = a table whose rows have a vector -----------------
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    dim = len(embeddings.embed_query("probe"))            # 1536 for text-embedding-3-small
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        # HNSW knobs (server mode): m = links per node, ef_construct = build effort.
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
    )
    print(f"created {COLLECTION!r} with {dim}-dim cosine vectors")

    # ---- 2. points = id + vector + payload ---------------------------------
    # We store one point per *paragraph* here (chunking proper is Chapter 5).
    docs = load_handbook()
    texts, payloads = [], []
    for doc in docs:
        for i, para in enumerate(p for p in doc.page_content.split("\n\n") if len(p) > 80):
            texts.append(para)
            payloads.append({"text": para, "doc_id": doc.metadata["doc_id"], "para": i})
    vectors = embeddings.embed_documents(texts)            # one API call, batched internally

    points = [
        models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{p['doc_id']}#{p['para']}")),  # stable id -> re-run = overwrite
            vector=v,
            payload=p,
        )
        for v, p in zip(vectors, payloads)
    ]
    client.upsert(COLLECTION, points=points)
    print(f"upserted {client.count(COLLECTION).count} points from {len(docs)} docs")

    # ---- 3. payload index: makes filters fast on a server (ignored in local mode)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client.create_payload_index(COLLECTION, "doc_id", models.PayloadSchemaType.KEYWORD)
    print("payload index on doc_id:", "no effect in embedded mode (server only)" if caught else "created")

    # ---- 4. query: nearest vectors -----------------------------------------
    question = "how long does a full charge take"
    qv = embeddings.embed_query(question)
    res = client.query_points(COLLECTION, query=qv, limit=3, with_payload=True)
    print(f"\nQ: {question}")
    for pt in res.points:
        print(f"   {pt.score:.3f}  {pt.payload['doc_id']}  | {pt.payload['text'][:70].replace(chr(10), ' ')}...")

    # ---- 5. same query, restricted to one document with a payload filter ----
    res = client.query_points(
        COLLECTION,
        query=qv,
        limit=3,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value="14-faq"))]
        ),
    )
    print("\nsame query, filter doc_id == 14-faq:")
    for pt in res.points:
        print(f"   {pt.score:.3f}  {pt.payload['doc_id']}  | {pt.payload['text'][:70].replace(chr(10), ' ')}...")

    # ---- 6. score threshold: return nothing rather than junk ---------------
    res = client.query_points(COLLECTION, query=embeddings.embed_query("recipe for pancakes"), limit=3, score_threshold=0.45)
    print(f"\n'recipe for pancakes' with score_threshold=0.45 -> {len(res.points)} results (good: nothing relevant exists)")

    # ---- 7. scroll = read rows without a vector (filter-only browsing) -----
    rows, _ = client.scroll(COLLECTION, scroll_filter=models.Filter(
        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value="10-pricing-and-plans"))]), limit=100)
    print(f"scroll: {len(rows)} paragraphs belong to 10-pricing-and-plans")

    # ---- 8. delete by filter (e.g. a document was removed from the source) -
    client.delete(COLLECTION, points_selector=models.FilterSelector(filter=models.Filter(
        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value="14-faq"))])))
    print(f"after deleting 14-faq: {client.count(COLLECTION).count} points remain")


if __name__ == "__main__":
    main()
