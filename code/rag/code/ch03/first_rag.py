"""Chapter 3 - The smallest complete RAG, LangChain only.

Ingest:  load -> chunk -> embed -> store (in memory)
Query:   embed question -> top-k chunks -> prompt -> LLM -> answer with [n] citations

Run:  uv run python code/ch03/first_rag.py
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragbook import format_docs, get_embeddings, get_llm, load_golden, load_handbook

# The prompt is the contract with the model. Three rules, all of them matter:
#  - only the context (stops it using memorised "facts")
#  - say "I don't know" (gives it a legal way out instead of guessing)
#  - cite [n] (lets us check the answer against the chunk it came from)
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about the Lumora Robotics employee handbook.\n"
            "Use ONLY the numbered context below. If the context does not contain the answer, "
            "reply exactly: I don't know based on the handbook.\n"
            "Cite the context items you used like [1] or [2][3]. Be brief.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def build_store() -> InMemoryVectorStore:
    """Ingest. 14 markdown files -> 55 chunks -> 55 vectors in a Python dict."""
    docs = load_handbook()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)          # metadata (source, doc_id) is copied to every chunk
    print(f"{len(docs)} documents -> {len(chunks)} chunks")
    return InMemoryVectorStore.from_documents(chunks, embedding=get_embeddings())


def answer(store: InMemoryVectorStore, question: str, k: int = 4) -> str:
    """Query. Retrieval and generation are two separate steps - keep them separate
    in your head too; almost every RAG bug is in one or the other, rarely both."""
    hits = store.similarity_search_with_score(question, k=k)   # [(Document, cosine score)]
    print(f"\nQ: {question}")
    for i, (doc, score) in enumerate(hits, start=1):
        print(f"   [{i}] {score:.3f}  {doc.metadata['source']}  | {doc.page_content[:70].replace(chr(10), ' ')}...")

    chain = PROMPT | get_llm()                                  # Runnable pipe: prompt output feeds the model
    response = chain.invoke({"context": format_docs([d for d, _ in hits]), "question": question})
    print(f"A: {response.text}")
    print(f"   tokens: {response.usage_metadata['input_tokens']} in / {response.usage_metadata['output_tokens']} out")
    return response.text


if __name__ == "__main__":
    store = build_store()
    golden = {q["id"]: q for q in load_golden()}
    for qid in ["q01", "q16", "u02"]:
        answer(store, golden[qid]["question"])
        print(f"   reference: {golden[qid]['answer']}")
