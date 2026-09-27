"""Chapter 9 - Query transforms: fix the QUERY, not the index.

Users ask badly-phrased, under-specified or multi-part questions. Four cheap
transforms, each a plain function built from `prompt | llm`:

  multi_query   : 3 paraphrases → retrieve each → fuse with RRF        (recall for vague queries)
  hyde          : write a hypothetical answer → embed THAT               (bridges question↔statement gap)
  step_back     : ask the more general question first                    (specific → principle)
  decompose     : split a multi-hop question into sub-questions          (q32, q42 need two facts)

Run:  uv run python code/ch09/query_transforms.py
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from ragbook import build_handbook_index, get_llm, load_golden, format_docs

llm = get_llm()


class Queries(BaseModel):
    queries: list[str]


def rrf(lists: list[list[Document]], k: int = 60) -> list[Document]:
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for lst in lists:
        for rank, d in enumerate(lst, start=1):
            cid = d.metadata["chunk_id"]
            docs[cid] = d
            scores[cid] = scores.get(cid, 0.0) + 1 / (k + rank)
    return [docs[c] for c, _ in sorted(scores.items(), key=lambda x: -x[1])]


# ------------------------------------------------------------ multi-query ---
multi_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user question as 3 different search queries that a support agent might type. "
               "Vary the vocabulary (synonyms, formal/informal). Return only the queries."),
    ("human", "{question}"),
])
multi_chain = multi_prompt | llm.with_structured_output(Queries)


def multi_query(store, question: str, k: int = 5) -> list[Document]:
    variants = [question] + multi_chain.invoke({"question": question}).queries
    print("   variants:", variants[1:])
    return rrf([store.similarity_search(v, k=k) for v in variants])[:k]


# ------------------------------------------------------------------ HyDE ---
hyde_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a short, plausible paragraph from an internal company handbook that answers the question. "
               "Invent specifics if needed - this text is only used as a search probe, never shown to the user."),
    ("human", "{question}"),
])
hyde_chain = hyde_prompt | llm


def hyde(store, question: str, k: int = 5) -> list[Document]:
    """Hypothetical Document Embeddings: a fake answer lives in the same 'region' of
    embedding space as the real answer chunk, closer than the question does."""
    fake = hyde_chain.invoke({"question": question}).text
    print("   hypothetical:", fake[:120].replace("\n", " "), "...")
    return store.similarity_search(fake, k=k)


# ------------------------------------------------------------- step-back ---
step_back_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a specific question, write the more general question whose answer would help. "
               "Example: 'Can I fly business to Rotterdam?' -> 'What are the flight class rules?'. Return only the question."),
    ("human", "{question}"),
])
step_back_chain = step_back_prompt | llm


def step_back(store, question: str, k: int = 5) -> list[Document]:
    general = step_back_chain.invoke({"question": question}).text.strip()
    print("   step-back question:", general)
    return rrf([store.similarity_search(question, k=k), store.similarity_search(general, k=k)])[:k]


# ------------------------------------------------------------- decompose ---
decompose_prompt = ChatPromptTemplate.from_messages([
    ("system", "Split the question into 2-3 self-contained sub-questions that must each be answered "
               "to answer the whole. If it is already atomic, return it unchanged."),
    ("human", "{question}"),
])
decompose_chain = decompose_prompt | llm.with_structured_output(Queries)


def decompose_and_answer(store, question: str, k: int = 3) -> str:
    subs = decompose_chain.invoke({"question": question}).queries
    print("   sub-questions:", subs)
    context_parts = []
    for sq in subs:
        docs = store.similarity_search(sq, k=k)
        context_parts.append(f"Sub-question: {sq}\n{format_docs(docs)}")
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using only the context. Show the arithmetic if any. If something is missing, say so."),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    return (answer_prompt | llm).invoke({"context": "\n\n".join(context_parts), "question": question}).text


def sources(docs):
    return [d.metadata["source"] for d in docs]


if __name__ == "__main__":
    store = build_handbook_index("ch09_handbook")
    golden = {g["id"]: g for g in load_golden()}

    q = "can I get money back for using my own car"          # vague, informal → multi-query
    print(f"\nQ: {q}\n  baseline:", sources(store.similarity_search(q, k=3)))
    print("  multi-query:", sources(multi_query(store, q, k=3)))

    q = "robot wifi dropped mid-task, what does it do"       # question ≠ statement phrasing → HyDE
    print(f"\nQ: {q}\n  baseline:", sources(store.similarity_search(q, k=3)))
    print("  HyDE:", sources(hyde(store, q, k=3)))

    q = "Can I fly business class to Austin from Pune?"     # specific → step back to the rule
    print(f"\nQ: {q}\n  baseline:", sources(store.similarity_search(q, k=3)))
    print("  step-back:", sources(step_back(store, q, k=3)))

    for qid in ("q32", "q42"):                               # multi-hop → decompose
        q = golden[qid]["question"]
        print(f"\nQ ({qid}): {q}")
        print("  answer:", decompose_and_answer(store, q).replace("\n", "\n          "))
        print("  gold  :", golden[qid]["answer"])
