"""Chapter 4 - the same thing through LangChain's QdrantVectorStore.

The wrapper decides the payload layout for you:
    payload = {"page_content": <text>, "metadata": {<your Document.metadata>}}
so filters use keys like "metadata.doc_id".

Run:  uv run python code/ch04/qdrant_langchain.py
"""
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import models

from ragbook import get_embeddings, get_qdrant_client, load_handbook, print_docs

COLLECTION = "ch04_langchain"


def main() -> None:
    client = get_qdrant_client()
    embeddings = get_embeddings()
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_documents(load_handbook())

    # 1. Create the collection yourself, then wrap it. (The one-liner
    #    QdrantVectorStore.from_documents(..., location=":memory:" | url=...) also exists, but it
    #    builds its OWN client from those kwargs and does not accept client= - with a shared
    #    client object, "create then wrap" is the pattern that works everywhere.)
    dim = len(embeddings.embed_query("probe"))
    client.create_collection(COLLECTION, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
    store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embeddings,
        retrieval_mode=RetrievalMode.DENSE,  # SPARSE and HYBRID come in Chapter 8
    )
    store.add_documents(chunks)              # embeds + upserts; ids=None -> random uuids (Chapter 6 uses stable ids)
    print(f"{len(chunks)} chunks -> {client.count(COLLECTION).count} points")

    # 2. Peek at one stored point to see the payload layout the wrapper chose.
    pt = client.scroll(COLLECTION, limit=1, with_payload=True)[0][0]
    print("payload keys:", list(pt.payload.keys()), "| metadata keys:", list(pt.payload["metadata"].keys()))

    # 3. Plain similarity search with scores.
    q = "what is the hotel cap in Europe"
    print(f"\nQ: {q}")
    for doc, score in store.similarity_search_with_score(q, k=3):
        print(f"   {score:.3f}  {doc.metadata['source']}")

    # 4. Metadata filter - note the "metadata." prefix.
    print("\nsame query, only the FAQ:")
    docs = store.similarity_search(q, k=2, filter=models.Filter(
        must=[models.FieldCondition(key="metadata.doc_id", match=models.MatchValue(value="14-faq"))]))
    print_docs(docs, 80)

    # 5. Retriever objects: the interface the rest of LangChain speaks.
    #    GOTCHA: the retriever's score_threshold is NOT the cosine score you saw above.
    #    LangChain rescales cosine to a 0..1 "relevance" = (cosine + 1) / 2, so
    #    cosine 0.44 -> relevance 0.72. Look at both before choosing a threshold.
    print("\ncosine vs relevance score for the same hits:")
    for (d, cos), (_, rel) in zip(store.similarity_search_with_score(q, k=2),
                                  store.similarity_search_with_relevance_scores(q, k=2)):
        print(f"   cosine {cos:.3f} -> relevance {rel:.3f}   {d.metadata['source']}")
    strict = store.as_retriever(search_type="similarity_score_threshold",
                                search_kwargs={"k": 4, "score_threshold": 0.66})   # ~ cosine 0.32
    print(f"threshold retriever (relevance >= 0.66), 'pancake recipe' -> {len(strict.invoke('pancake recipe'))} docs")
    print(f"threshold retriever (relevance >= 0.66), {q!r} -> {len(strict.invoke(q))} docs")

    mmr = store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5})
    print("\nMMR retriever (diverse sources):")
    print_docs(mmr.invoke("what happens when a robot loses connection"), 80)

    # 6. Re-attach to an existing collection later (another process, next day...).
    again = QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=embeddings)
    print(f"\nre-attached: {len(again.similarity_search('PTO', k=2))} docs found")


if __name__ == "__main__":
    main()
