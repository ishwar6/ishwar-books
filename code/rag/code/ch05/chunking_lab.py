"""Chapter 5 - Chunking lab.

For each strategy: chunk the handbook, report chunk statistics, index into a
throwaway Qdrant collection, and measure hit-rate@5 on the golden questions
(= did at least one of the top-5 chunks come from a document that holds the answer?).

Run:  uv run python code/ch05/chunking_lab.py
"""
import sys
from pathlib import Path

import tiktoken
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models
from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

sys.path.insert(0, str(Path(__file__).parent))
from semantic_chunker import semantic_chunks  # noqa: E402

from ragbook import get_embeddings, get_qdrant_client, load_golden, load_handbook  # noqa: E402

ENC = tiktoken.get_encoding("cl100k_base")


def by_markdown_headers(docs: list[Document]) -> list[Document]:
    """Split on headings, keep the heading text as metadata, then cap size."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False
    )
    capped = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0)
    out = []
    for doc in docs:
        for sec in header_splitter.split_text(doc.page_content):   # returns Documents with h1/h2 metadata
            sec.metadata.update(doc.metadata)
            out.extend(capped.split_documents([sec]))
    return out


def strategies(embeddings):
    return {
        "fixed_500":        lambda docs: CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=0).split_documents(docs),
        "recursive_800":    lambda docs: RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_documents(docs),
        "recursive_300":    lambda docs: RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(docs),
        "tokens_200":       lambda docs: TokenTextSplitter(encoding_name="cl100k_base", chunk_size=200, chunk_overlap=30).split_documents(docs),
        "markdown_headers": by_markdown_headers,
        "semantic":         lambda docs: [c for d in docs for c in semantic_chunks(d, embeddings)],
    }


def stats(chunks: list[Document]) -> tuple[int, float, int, int]:
    toks = [len(ENC.encode(c.page_content)) for c in chunks]
    return len(chunks), sum(toks) / len(toks), min(toks), max(toks)


def new_store(client, name: str, embeddings, chunks: list[Document]) -> QdrantVectorStore:
    """Fresh collection for this strategy, then embed + upsert the chunks."""
    if client.collection_exists(name):
        client.delete_collection(name)
    dim = len(embeddings.embed_query("probe"))
    client.create_collection(name, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
    store = QdrantVectorStore(client=client, collection_name=name, embedding=embeddings)
    store.add_documents(chunks)
    return store


def evaluate(store: QdrantVectorStore, golden: list[dict]) -> tuple[float, float, float]:
    """Three cheap retrieval metrics (Chapter 10 has the full set):
      hit@5      a top-5 chunk comes from a document holding the answer (lenient)
      hit@1      the very first chunk does (strict)
      answer@3   ALL of the question's answer keywords appear in the top-3 chunk text
                 -> did the chunking keep the answer in one piece?"""
    hit5 = hit1 = ans3 = 0
    for q in golden:
        docs = store.similarity_search(q["question"], k=5)
        ids = [d.metadata["doc_id"] for d in docs]
        hit5 += bool(set(ids) & set(q["sources"]))
        hit1 += ids[0] in q["sources"]
        text = " ".join(d.page_content for d in docs[:3])
        ans3 += all(kw in text for kw in q["keywords"])
    n = len(golden)
    return hit5 / n, hit1 / n, ans3 / n


def main() -> None:
    docs = load_handbook()
    golden = [q for q in load_golden() if q["answerable"]]
    embeddings = get_embeddings()
    client = get_qdrant_client()

    print(f"{'strategy':<18}{'chunks':>7}{'mean_tok':>10}{'min':>6}{'max':>6}{'hit@5':>8}{'hit@1':>8}{'answer@3':>10}")
    for name, fn in strategies(embeddings).items():
        chunks = fn(docs)
        n, mean, lo, hi = stats(chunks)
        store = new_store(client, f"ch05_{name}", embeddings, chunks)
        h5, h1, a3 = evaluate(store, golden)
        print(f"{name:<18}{n:>7}{mean:>10.0f}{lo:>6}{hi:>6}{h5:>8.2f}{h1:>8.2f}{a3:>10.2f}")

    print("\nExample: the pricing table under 'fixed_500' (does it survive?)")
    fixed = strategies(embeddings)["fixed_500"](docs)
    for c in fixed:
        if "Atlas A2** robot" in c.page_content:
            print(c.page_content[:400])
            break


if __name__ == "__main__":
    main()
