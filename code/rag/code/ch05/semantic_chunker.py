"""Chapter 5 - a semantic chunker in ~40 lines.

Idea: split text into sentences, embed each, and cut a new chunk wherever the
similarity between consecutive sentences DROPS (a topic change). This is what
"semantic chunker" classes in libraries do; here you can see every step.
"""
import re

import numpy as np
from langchain_core.documents import Document


def split_sentences(text: str) -> list[str]:
    # Good enough for prose: split after . ! ? or a newline. Markdown headers become their own "sentence".
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def semantic_chunks(doc: Document, embeddings, breakpoint_percentile: float = 25, min_sentences: int = 2) -> list[Document]:
    sentences = split_sentences(doc.page_content)
    if len(sentences) <= min_sentences:
        return [doc]
    vecs = np.array(embeddings.embed_documents(sentences))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    # similarity between each sentence and the next one
    sims = np.sum(vecs[:-1] * vecs[1:], axis=1)
    # cut where similarity is unusually LOW (bottom X percentile of this document)
    threshold = np.percentile(sims, breakpoint_percentile)

    chunks, current = [], [sentences[0]]
    for sent, sim in zip(sentences[1:], sims):
        if sim < threshold and len(current) >= min_sentences:
            chunks.append(" ".join(current))
            current = [sent]
        else:
            current.append(sent)
    chunks.append(" ".join(current))

    return [Document(page_content=c, metadata=dict(doc.metadata, chunk_index=i)) for i, c in enumerate(chunks)]


if __name__ == "__main__":
    from ragbook import get_embeddings, load_handbook

    doc = load_handbook()[1]   # the PTO policy
    for c in semantic_chunks(doc, get_embeddings())[:6]:
        print(f"--- chunk {c.metadata['chunk_index']} ({len(c.page_content)} chars)")
        print(c.page_content[:160].replace("\n", " "), "...")
