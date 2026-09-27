"""Chapter 28 - making a diagram retrievable: the tree-diagram question, measured.

    QDRANT_MODE=memory uv run python code/ch28/figures_to_text.py

Three indexes over the same two figure pages:

  extracted      what the PDF parser gives you: the node LABELS and the axis
                 labels. No edges, no bar values. This is the silent default.
  generic        + a vision-model caption written from "describe this figure".
  structured     + a caption written from a prompt that DEMANDS the structure
                 (parent -> child edges; one row per bar with its value).

Then we ask questions that can only be answered from the figure's structure.
A caption only contains what you asked the model to look at - which is the
whole answer to "what happens to a tree diagram in a PDF?".

Cost: 4 vision calls (2 figures x 2 prompts), a few cents.
"""
from __future__ import annotations

import argparse
import base64

import pymupdf
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.messages import HumanMessage
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models

from ragbook import (DATA_DIR, chunk_documents, format_docs, get_embeddings,
                     get_llm, get_qdrant_client, load_handbook)

PDF = DATA_DIR / "pdfs" / "lumora_report.pdf"
FIGURE_PAGES = {4: "bar chart of robots per site", 5: "tree diagram of the org chart"}

GENERIC_PROMPT = "Describe this figure."

STRUCTURED_PROMPT = (
    "Extract this figure's STRUCTURE as text that someone could search later.\n"
    "If it is a hierarchy or flow chart: list every edge as 'parent -> child', "
    "one per line, covering every box.\n"
    "If it is a chart: give one line per series/bar as 'label: value' using the "
    "axis scale to read the values.\n"
    "Start with one sentence naming the figure, then the lines. No commentary."
)

# (question, must contain, must NOT contain). The negative list matters: a
# caption that dumps every box in the diagram will "contain" the right words
# while answering the question wrongly, and a naive keyword scorer would call
# that a pass. Scorers need negatives as much as positives.
QUESTIONS = [
    ("Which teams report into Fleet Platform?", ["beacon", "compass"],
     ["navigation", "firmware"]),
    ("Who does the Navigation team report to?", ["robot platform"], []),
    ("How many robots are deployed at the Rotterdam site?", ["410"], []),
    ("Which site has the most robots deployed?", ["rotterdam"], ["pune", "austin"]),
]

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the context. If it does not answer the question, "
               "reply exactly: I don't know."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


def render(page_no: int, dpi: int = 110) -> tuple[str, str]:
    """Render one PDF page to PNG. Returns (base64, extracted text)."""
    doc = pymupdf.open(PDF)
    page = doc[page_no - 1]
    png = page.get_pixmap(dpi=dpi).tobytes("png")
    text = page.get_text()
    doc.close()
    return base64.b64encode(png).decode(), text


def caption(llm, b64: str, instruction: str) -> str:
    msg = HumanMessage(content=[
        {"type": "text", "text": instruction},
        {"type": "image", "source_type": "base64", "mime_type": "image/png", "data": b64},
    ])
    return llm.invoke([msg]).text.strip()


def index_of(client, emb, dim, name: str, docs: list[Document]) -> QdrantVectorStore:
    coll = f"ch28_fig_{name}"
    if client.collection_exists(coll):
        client.delete_collection(coll)
    client.create_collection(coll, vectors_config=models.VectorParams(
        size=dim, distance=models.Distance.COSINE))
    store = QdrantVectorStore(client=client, collection_name=coll, embedding=emb)
    store.add_documents(docs)
    return store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--dpi", type=int, default=110)
    args = ap.parse_args()
    if not PDF.exists():
        raise SystemExit("run: uv run python code/ch28/make_sample_pdf.py")

    llm = get_llm()
    pages = {n: render(n, args.dpi) for n in FIGURE_PAGES}

    captions: dict[str, dict[int, str]] = {"generic": {}, "structured": {}}
    for page_no, (b64, _) in pages.items():
        print(f"\n=== page {page_no} - {FIGURE_PAGES[page_no]} ===")
        for style, instruction in (("generic", GENERIC_PROMPT),
                                   ("structured", STRUCTURED_PROMPT)):
            text = caption(llm, b64, instruction)
            captions[style][page_no] = text
            print(f"\n--- {style} caption ---\n{text}")

    # ------------------------------------------------ build the three indexes
    # The company overview states the org structure and the Rotterdam figure in
    # prose, so leaving it in would let every index answer from text and measure
    # nothing about the figures. Exclude it: we want the FIGURE to be the only
    # source of these facts.
    contaminating = {"01-company-overview"}
    noise = chunk_documents([d for d in load_handbook()
                             if d.metadata["doc_id"] not in contaminating])
    extracted = [
        Document(page_content=text, metadata={"source": f"report.pdf p{n}"})
        for n, (_, text) in pages.items()
    ]
    variants = {
        "extracted": extracted,
        "generic": extracted + [
            Document(page_content=f"Figure on page {n}: {c}",
                     metadata={"source": f"report.pdf p{n} (caption)"})
            for n, c in captions["generic"].items()],
        "structured": extracted + [
            Document(page_content=f"Figure on page {n}: {c}",
                     metadata={"source": f"report.pdf p{n} (caption)"})
            for n, c in captions["structured"].items()],
    }

    client = get_qdrant_client()
    emb = get_embeddings()
    dim = len(emb.embed_query("probe"))
    chain = PROMPT | llm

    print(f"\n\n{len(QUESTIONS)} questions that only the FIGURE can answer "
          f"({len(noise)} handbook chunks as distractors, k={args.k};\n"
          f"the company-overview doc is excluded - it states these same facts in prose)\n")
    print(f"{'index':12} " + " ".join(f"q{i+1:<2}" for i in range(len(QUESTIONS))) + "  score")
    print("-" * 44)
    answers: dict[str, list[str]] = {}
    for name, docs in variants.items():
        store = index_of(client, emb, dim, name, docs + noise)
        marks, got = [], []
        for question, expected, forbidden in QUESTIONS:
            context = format_docs(store.similarity_search(question, k=args.k))
            answer = chain.invoke({"context": context, "question": question}).text.strip()
            low = answer.lower()
            ok = all(e in low for e in expected) and not any(f in low for f in forbidden)
            marks.append("OK " if ok else "-  ")
            got.append(answer.replace("\n", " ")[:70])
        answers[name] = got
        print(f"{name:12} " + " ".join(marks) + f"  {sum(m == 'OK ' for m in marks)}/{len(QUESTIONS)}")

    print("\nwhat each index actually answered:")
    for i, (question, _, _) in enumerate(QUESTIONS):
        print(f"\n  Q{i+1}: {question}")
        for name in variants:
            print(f"     {name:11} {answers[name][i]}")


if __name__ == "__main__":
    main()
