"""Chapter 9 - Metadata filtering, self-query, recency, and MMR.

Similarity search finds what is *similar*; it cannot express "only HR docs",
"only things effective in 2026", or "not the FAQ". Metadata can. Qdrant stores
metadata as *payload* and filters it during the vector search (pre-filtering),
so you never lose recall to a post-filter.

  1. hand-written filter        models.Filter(must=[FieldCondition(...)])
  2. self-query                 LLM extracts {category, min_year} from the question → filter
  3. recency preference         FAQ (2024) vs policy (2026): sort ties by effective year
  4. MMR                        as_retriever(search_type="mmr") for diverse results

Run:  uv run python code/ch09/metadata_filtering.py
"""
from __future__ import annotations

import re
from typing import Literal

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from pydantic import BaseModel, Field
from qdrant_client import models

from ragbook import chunk_documents, get_embeddings, get_llm, get_qdrant_client, load_handbook

CATEGORY = {  # in real life this comes from your CMS / folder structure / a classifier
    "01-company-overview": "company", "02-pto-and-leave-policy": "hr", "03-travel-and-expense-policy": "finance",
    "04-atlas-a2-specification": "product", "05-beacon-fleet-software": "engineering", "06-security-policy": "security",
    "07-onboarding-guide": "hr", "08-runbook-fleet-outage": "engineering", "09-support-sla": "support",
    "10-pricing-and-plans": "sales", "11-release-notes-beacon-4.2": "engineering",
    "12-postmortem-2026-03-rotterdam-halt": "engineering", "13-remote-work-policy": "hr", "14-faq": "faq",
}


def build(collection: str = "ch09_meta") -> QdrantVectorStore:
    docs = load_handbook()
    years = {}
    for d in docs:  # year is in the doc header; copy to every chunk of that doc
        m = re.search(r"(20\d\d)", d.page_content[:400])
        years[d.metadata["doc_id"]] = int(m.group(1)) if m else 2024
    chunks = chunk_documents(docs)
    for c in chunks:
        c.metadata["category"] = CATEGORY[c.metadata["doc_id"]]
        c.metadata["effective_year"] = years[c.metadata["doc_id"]]
    client, emb = get_qdrant_client(), get_embeddings()
    if not client.collection_exists(collection):
        client.create_collection(collection, vectors_config=models.VectorParams(size=len(emb.embed_query("x")), distance=models.Distance.COSINE))
        store = QdrantVectorStore(client=client, collection_name=collection, embedding=emb, retrieval_mode=RetrievalMode.DENSE)
        store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
        return store
    return QdrantVectorStore(client=client, collection_name=collection, embedding=emb, retrieval_mode=RetrievalMode.DENSE)


# ------------------------------------------------------------- self-query ---
class QueryPlan(BaseModel):
    """What the LLM extracts from the user's question."""
    search_text: str = Field(description="the semantic part of the query, without filter words")
    category: Literal["hr", "finance", "product", "engineering", "security", "support", "sales", "company", "faq"] | None = None
    min_year: int | None = Field(default=None, description="only documents effective in this year or later")


def self_query(store: QdrantVectorStore, question: str, k: int = 3) -> list[Document]:
    planner = get_llm().with_structured_output(QueryPlan)
    plan = planner.invoke(
        "Extract a search plan from the question. Categories: hr (leave, onboarding, remote work), finance (travel, expenses), "
        "product (robot specs), engineering (Beacon, runbooks, releases, incidents), security, support (SLAs), sales (pricing), "
        "company, faq. Only set a filter if the question clearly implies it.\n\nQuestion: " + question
    )
    print("   plan:", plan.model_dump(exclude_none=True))
    must = []
    if plan.category:
        must.append(models.FieldCondition(key="metadata.category", match=models.MatchValue(value=plan.category)))
    if plan.min_year:
        must.append(models.FieldCondition(key="metadata.effective_year", range=models.Range(gte=plan.min_year)))
    return store.similarity_search(plan.search_text, k=k, filter=models.Filter(must=must) if must else None)


def recency_rerank(docs_scores: list[tuple[Document, float]], tolerance: float = 0.08) -> list[Document]:
    """If two chunks score within `tolerance`, prefer the newer effective_year.
    This is how you make 'the policy wins over the stale FAQ' systematic."""
    return [d for d, _ in sorted(docs_scores, key=lambda x: (-round(x[1] / tolerance), -x[0].metadata["effective_year"]))]


def src(docs):
    return [f"{d.metadata['source']}({d.metadata['effective_year']})" for d in docs]


if __name__ == "__main__":
    store = build()

    q = "How many days of PTO do I get?"
    print(f"Q: {q}")
    print("  1. no filter          :", src(store.similarity_search(q, k=3)))
    only_hr = models.Filter(must=[models.FieldCondition(key="metadata.category", match=models.MatchValue(value="hr"))])
    print("  1. filter category=hr :", src(store.similarity_search(q, k=3, filter=only_hr)))
    not_faq = models.Filter(must_not=[models.FieldCondition(key="metadata.doc_id", match=models.MatchValue(value="14-faq"))])
    print("  1. filter NOT faq     :", src(store.similarity_search(q, k=3, filter=not_faq)))

    print("\n  2. self-query:")
    for q2 in ["What does the 2026 leave policy say about carry-over?", "How much does the robot weigh?"]:
        print(f"   Q: {q2}")
        print("      →", src(self_query(store, q2)))

    print("\n  3. recency preference on the PTO conflict:")
    pairs = store.similarity_search_with_score(q, k=6)
    print("     by score  :", [f"{d.metadata['doc_id']}({d.metadata['effective_year']}) {s:.2f}" for d, s in pairs[:4]])
    print("     recency   :", src(recency_rerank(pairs)[:4]))

    print("\n  4. MMR (diversity) vs plain similarity for a broad question:")
    q3 = "tell me about Beacon"
    plain = store.as_retriever(search_kwargs={"k": 4}).invoke(q3)
    mmr = store.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5}).invoke(q3)
    print("     similarity:", [d.metadata["source"] for d in plain])
    print("     mmr       :", [d.metadata["source"] for d in mmr])
