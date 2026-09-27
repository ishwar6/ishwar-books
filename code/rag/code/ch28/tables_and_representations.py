"""Chapter 28 - six ways to put a table in a vector index, measured.

    QDRANT_MODE=memory uv run python code/ch28/tables_and_representations.py

A table's meaning lives in the GRID: "250 kg" means nothing until you know it
sits in the row "Maximum payload" and the column "Atlas A2". Flatten it to a
string and chunk it, and that association is the first thing you lose - the
number survives, the question "which robot?" does not.

Each representation is indexed into its own collection, queried with real
table-lookup questions, and scored twice:
  retrieval  - did the top-k text contain the correct value at all?
  answer     - did the model then answer with the correct value?
The gap between those two columns is the whole lesson.
"""
from __future__ import annotations

import argparse

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models

from ragbook import (chunk_documents, format_docs, get_embeddings, get_llm,
                     get_qdrant_client, load_handbook)

# ------------------------------------------------------------- the tables ---
ATLAS = {
    "title": "Atlas A2 - technical specification",
    "header": ["Property", "Atlas A2", "Atlas A2 Lite"],
    "rows": [
        ["Maximum payload", "250 kg", "120 kg"],
        ["Maximum speed", "2.0 m/s", "1.6 m/s"],
        ["Battery runtime", "8 hours", "8 hours"],
        ["Full charge time", "90 minutes", "75 minutes"],
        ["Fast top-up", "20 minutes", "18 minutes"],
        ["Length", "1120 mm", "920 mm"],
        ["Width", "720 mm", "620 mm"],
        ["Robot weight", "145 kg", "110 kg"],
        ["Ingress protection", "IP54", "IP54"],
        ["Operating temperature", "0 to 40 C", "0 to 40 C"],
        ["Lift height", "60 mm", "45 mm"],
    ],
}
SUPPORT = {
    "title": "Customer support tiers and SLAs",
    "header": ["Tier", "Hours", "P1 response", "P1 resolution", "Spare parts"],
    "rows": [
        ["Bronze", "9x5", "4 hours", "next business day", "standard shipping"],
        ["Silver", "12x5", "1 hour", "8 hours", "2 business days"],
        ["Gold", "24x7", "30 minutes", "6 hours", "next business day"],
        ["Platinum", "24x7", "15 minutes", "4 hours", "on-site spares kit"],
    ],
}
TABLES = [ATLAS, SUPPORT]

QUESTIONS = [
    ("What is the maximum payload of the Atlas A2 Lite?", "120"),
    ("What is the maximum payload of the Atlas A2?", "250"),
    ("How long does a full charge of the Atlas A2 Lite take?", "75"),
    ("What is the Atlas A2 Lite's top speed?", "1.6"),
    ("What is the P1 response time for Platinum support?", "15"),
    ("What is the P1 response time for Bronze support?", "4 hour"),
    ("Which support tier includes an on-site spares kit?", "Platinum"),
    ("How wide is the Atlas A2 Lite?", "620"),
    ("What is the lift height of the Atlas A2 Lite?", "45"),
]


def distractors() -> list[Document]:
    """The rest of the handbook, so the table chunks have to be FOUND, not just
    be the only thing in the index. The two documents that state these same
    facts in prose are excluded - otherwise we would be measuring those."""
    excluded = {"04-atlas-a2-specification", "09-support-sla"}
    docs = [d for d in load_handbook() if d.metadata["doc_id"] not in excluded]
    return chunk_documents(docs)


# -------------------------------------------------- the representations -----
def rep_flattened(t: dict) -> list[Document]:
    """What naive extraction + fixed-size chunking does: one text stream, cut
    every ~160 characters with no idea where the grid was."""
    flat = f"{t['title']} " + " ".join(" ".join(r) for r in [t["header"], *t["rows"]])
    size = 200
    return [Document(page_content=flat[i:i + size]) for i in range(0, len(flat), size)]


def rep_markdown(t: dict) -> list[Document]:
    lines = ["| " + " | ".join(t["header"]) + " |",
             "|" + "---|" * len(t["header"])]
    lines += ["| " + " | ".join(r) + " |" for r in t["rows"]]
    return [Document(page_content=f"{t['title']}\n" + "\n".join(lines))]


def rep_html(t: dict) -> list[Document]:
    head = "<tr>" + "".join(f"<th>{c}</th>" for c in t["header"]) + "</tr>"
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in t["rows"])
    return [Document(page_content=f"{t['title']}\n<table>{head}{body}</table>")]


def rep_csv(t: dict) -> list[Document]:
    lines = [",".join(t["header"])] + [",".join(r) for r in t["rows"]]
    return [Document(page_content=f"{t['title']}\n" + "\n".join(lines))]


def rep_row_chunks(t: dict) -> list[Document]:
    """One chunk per row, with the title and header REPEATED into every chunk.
    Each chunk is now self-contained: it can survive on its own in a top-k list."""
    return [
        Document(page_content=f"{t['title']}\n" + " | ".join(t["header"]) + "\n" + " | ".join(r))
        for r in t["rows"]
    ]


def rep_verbalised(t: dict) -> list[Document]:
    """One sentence per cell-group, in the vocabulary a user would actually type.
    Costs tokens at ingest; embeds far closer to natural questions."""
    docs = []
    for row in t["rows"]:
        key, values = row[0], row[1:]
        sentence = f"In the {t['title']}, for {key}: " + "; ".join(
            f"{col} is {val}" for col, val in zip(t["header"][1:], values)
        ) + "."
        docs.append(Document(page_content=sentence))
    return docs


REPS = {
    "flattened (naive)": rep_flattened,
    "markdown": rep_markdown,
    "html": rep_html,
    "csv": rep_csv,
    "row-per-chunk": rep_row_chunks,
    "verbalised": rep_verbalised,
}

def enforce_chunk_limit(docs: list[Document], limit: int | None) -> list[Document]:
    """A generic chunker, applied AFTER the representation was chosen.

    This is the realistic regime for a big table: your splitter does not know a
    table is a table, so it cuts every N characters. Representations that put
    the header in one chunk and the rows in another lose the column names; a
    representation whose chunks are individually self-contained does not care.
    """
    if limit is None:
        return docs
    out: list[Document] = []
    for d in docs:
        text = d.page_content
        for i in range(0, len(text), limit):
            out.append(Document(page_content=text[i:i + limit], metadata=dict(d.metadata)))
    return out


PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the context. If the context does not clearly answer, "
               "reply exactly: I don't know. Be terse - the value and its unit."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


def run_experiment(client, emb, dim, chain, questions, noise, k, chunk_limit) -> None:
    label = ("no chunk limit - the whole table fits in one chunk"
             if chunk_limit is None else
             f"chunk limit {chunk_limit} chars - a big table gets cut up")
    print(f"\n=== {label} ===")
    print(f"{'representation':20} {'tbl ch':>6} {'retrieval':>10} {'answer':>8}")
    print("-" * 48)
    results = {}
    for name, fn in REPS.items():
        table_docs = enforce_chunk_limit([d for t in TABLES for d in fn(t)], chunk_limit)
        docs = table_docs + noise
        coll = f"ch28_{name.split()[0].replace('-', '_')}_{chunk_limit or 0}"
        if client.collection_exists(coll):
            client.delete_collection(coll)
        client.create_collection(coll, vectors_config=models.VectorParams(
            size=dim, distance=models.Distance.COSINE))
        store = QdrantVectorStore(client=client, collection_name=coll, embedding=emb)
        store.add_documents(docs)

        found = correct = 0
        misses = []
        for question, expected in questions:
            hits = store.similarity_search(question, k=k)
            context = format_docs(hits)
            in_context = expected.lower() in context.lower()
            answer = chain.invoke({"context": context, "question": question}).text
            ok = expected.lower() in answer.lower()
            found += in_context
            correct += ok
            if in_context and not ok:
                misses.append((question, answer.strip().replace("\n", " ")[:60]))
        results[name] = (len(table_docs), found, correct, misses)
        print(f"{name:20} {len(table_docs):6} {found:>7}/{len(questions)} "
              f"{correct:>5}/{len(questions)}")

    print("\nthe interesting cases - value was retrieved, answer still wrong")
    print("(the grid was destroyed: the number is there, its column is not):")
    any_miss = False
    for name, (_, _, _, misses) in results.items():
        for q, a in misses:
            any_miss = True
            print(f"  [{name}] {q}\n      -> {a}")
    if not any_miss:
        print("  none in this regime")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--limit", type=int, default=len(QUESTIONS))
    ap.add_argument("--chunk-limit", type=int, default=300,
                    help="character budget a naive splitter would enforce")
    args = ap.parse_args()
    questions = QUESTIONS[: args.limit]

    client = get_qdrant_client()          # ONE client per process (embedded mode locks)
    emb = get_embeddings()
    dim = len(emb.embed_query("probe"))
    chain = PROMPT | get_llm()

    noise = distractors()
    print(f"{len(questions)} table questions, k={args.k}, "
          f"{len(noise)} distractor chunks from the rest of the handbook")
    print("retrieval = the value appeared in the top-k text; "
          "answer = the model stated it")
    for limit in (None, args.chunk_limit):
        run_experiment(client, emb, dim, chain, questions, noise, args.k, limit)


if __name__ == "__main__":
    main()
