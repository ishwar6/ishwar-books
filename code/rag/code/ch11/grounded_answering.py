"""Chapter 11 - Grounded answering: the prompt, the citations, the verification.

Three versions of "answer from context", each stricter than the last:

  v1 naive     : "Use the context to answer."            → fills gaps from memory
  v2 grounded  : explicit rules + a fixed refusal phrase → says "I don't know" when it should
  v3 cited     : structured output: sentences + citation ids, then every citation is VERIFIED
                 by a judge; unsupported sentences are dropped and flagged

Tested on three nasty questions: q01 (two docs disagree), u03 (false presupposition:
there is no "Atlas A3"), u02 (simply not in the corpus).

Run:  uv run python code/ch11/grounded_answering.py
"""
from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ragbook import build_handbook_index, get_llm, load_golden, load_handbook

llm = get_llm()

# ------------------------------------------------- context with document dates ---
# Conflicts are usually "old doc vs new doc". Give the model the date so it can prefer the newer one.
DOC_YEAR = {}
for _d in load_handbook():
    m = re.search(r"(20\d\d)", _d.page_content[:400])
    DOC_YEAR[_d.metadata["doc_id"]] = int(m.group(1)) if m else 2024


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] source={d.metadata['source']} effective_year={DOC_YEAR[d.metadata['doc_id']]}\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


# ------------------------------------------------------------------ v1 naive ---
naive = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the context to answer the question."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
]) | llm

# --------------------------------------------------------------- v2 grounded ---
GROUNDED_RULES = """You answer questions about the Lumora employee handbook.
Rules:
1. Use ONLY the context below. Do not use prior knowledge, even if you are confident.
2. If the context does not contain the answer, reply exactly: "I don't know - the handbook does not cover this."
3. If the question assumes something the context contradicts (e.g. a product that does not exist), say so.
4. If two passages disagree, say that they disagree, quote both values, and prefer the one with the newer effective_year.
5. Be brief. Numbers must be copied exactly from the context."""
grounded = ChatPromptTemplate.from_messages([
    ("system", GROUNDED_RULES),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
]) | llm


# ------------------------------------------------------ v3 cited + verified ---
class CitedSentence(BaseModel):
    text: str
    citations: list[int] = Field(description="chunk numbers [n] that support this sentence")


class CitedAnswer(BaseModel):
    abstained: bool = Field(description="true if the context does not answer the question")
    sentences: list[CitedSentence] = Field(default_factory=list)
    conflict_note: str | None = Field(default=None, description="if passages disagree, explain which you preferred and why")


cited = ChatPromptTemplate.from_messages([
    ("system", GROUNDED_RULES + "\nReturn every sentence with the chunk numbers that support it. A sentence with no support must not be written."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
]) | llm.with_structured_output(CitedAnswer)


class SupportVerdict(BaseModel):
    supported: bool
    reason: str


def verify_sentence(sentence: str, cited_chunks: list[str]) -> SupportVerdict:
    """NLI-style check: does the cited text ENTAIL the sentence? Independent second
    opinion - the writer is not allowed to grade its own homework."""
    return llm.with_structured_output(SupportVerdict).invoke(
        "Does the evidence fully support the sentence? Numbers must match exactly.\n\n"
        f"Evidence:\n{chr(10).join(cited_chunks)}\n\nSentence: {sentence}"
    )


def answer_cited_verified(question: str, docs: list[Document]) -> str:
    ctx = format_context(docs)
    out: CitedAnswer = cited.invoke({"context": ctx, "question": question})
    if out.abstained or not out.sentences:
        return "I don't know - the handbook does not cover this."
    kept, dropped = [], []
    for s in out.sentences:
        chunks = [docs[i - 1].page_content for i in s.citations if 0 < i <= len(docs)]
        if not chunks:
            dropped.append((s.text, "no citation"))
            continue
        v = verify_sentence(s.text, chunks)
        (kept if v.supported else dropped).append((s.text + " " + "".join(f"[{c}]" for c in s.citations), v.reason))
    text = " ".join(t for t, _ in kept) or "I don't know - the handbook does not cover this."
    if out.conflict_note:
        text += f"\n  (conflict: {out.conflict_note})"
    for t, why in dropped:
        text += f"\n  [dropped unsupported sentence: {t!r} - {why}]"
    return text


if __name__ == "__main__":
    store = build_handbook_index("ch11_handbook")
    golden = {g["id"]: g for g in load_golden()}
    for qid in ["q01", "u03", "u02"]:
        q = golden[qid]["question"]
        docs = store.similarity_search(q, k=4)
        ctx = format_context(docs)
        print(f"\n{'=' * 80}\n{qid}: {q}\n  retrieved: {[d.metadata['source'] for d in docs]}")
        print("  v1 naive    :", naive.invoke({"context": ctx, "question": q}).text.replace("\n", " ")[:300])
        print("  v2 grounded :", grounded.invoke({"context": ctx, "question": q}).text.replace("\n", " ")[:300])
        print("  v3 verified :", answer_cited_verified(q, docs).replace("\n", "\n                "))
        print("  gold        :", golden[qid]["answer"])
