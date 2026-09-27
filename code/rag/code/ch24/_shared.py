"""Chapter 24 - shared pieces for the reranking experiments.

Why this file exists: all four ch24 scripts need the same three things -
a corpus (easy or hard), an encoder, and an *exact* search. Exact search
matters here: we are measuring what RERANKING does, so the first stage must
not contribute ANN recall error of its own. At 1k chunks a numpy dot product
is instant, so we pay nothing for that rigour. (Chapter 4 explains why you
would never do this at a million chunks.)

Not a runnable chapter script - the other files in code/ch24/ import it.
"""
from __future__ import annotations

import random
import re
import time
from functools import lru_cache

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragbook import DATA_DIR, chunk_documents, load_handbook

SEED = 20260916


# --------------------------------------------------------------- corpora ---
def handbook_chunks(chunk_size: int = 800, chunk_overlap: int = 120) -> list[Document]:
    """The easy corpus: 14 curated documents, ~56 chunks, no distractors."""
    return chunk_documents(load_handbook(), chunk_size, chunk_overlap)


def distractor_chunks(n_novel: int = 900, chunk_size: int = 800) -> list[Document]:
    """Chunks of public-domain novels (Chapter 7 downloaded them) used as noise.

    They are *topically* unrelated, which is the easy kind of distractor. The
    hard kind - near-duplicate policy text - is what `generated_chunks()` adds.
    """
    out: list[Document] = []
    novels = sorted((DATA_DIR / "gutenberg").glob("*.txt"))
    if not novels:
        return out
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    for path in novels:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for i, piece in enumerate(splitter.split_text(text)):
            out.append(
                Document(
                    page_content=piece,
                    metadata={"source": path.name, "doc_id": path.stem, "chunk_index": i},
                )
            )
    random.Random(SEED).shuffle(out)
    return out[:n_novel]


def generated_chunks(chunk_size: int = 800, chunk_overlap: int = 120) -> list[Document]:
    """LLM-written handbook-style documents (Chapter 7). These are the *dangerous*
    distractors: same voice, same company, same vocabulary, different facts."""
    out: list[Document] = []
    gen = DATA_DIR / "generated"
    if not gen.exists():
        return out
    docs = [
        Document(
            page_content=p.read_text(encoding="utf-8"),
            metadata={"source": p.name, "doc_id": p.stem, "title": p.stem},
        )
        for p in sorted(gen.glob("*.md"))
    ]
    return chunk_documents(docs, chunk_size, chunk_overlap) if docs else out


def build_corpus(hard: bool) -> list[Document]:
    """easy = handbook only (56 chunks). hard = handbook + generated + novels (~1k)."""
    chunks = handbook_chunks()
    if hard:
        chunks = chunks + generated_chunks() + distractor_chunks()
    return chunks


# -------------------------------------------------------------- encoders ---
@lru_cache(maxsize=4)
def _text_embedder(model_name: str):
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name)


def encode_passages(texts: list[str], model_name: str = "BAAI/bge-small-en-v1.5") -> np.ndarray:
    """Document-side vectors from a LOCAL bi-encoder. Free, so we can afford to
    re-embed a 1,000-chunk corpus in every experiment."""
    m = _text_embedder(model_name)
    return np.array(list(m.passage_embed(texts)), dtype=np.float32)


def encode_queries(texts: list[str], model_name: str = "BAAI/bge-small-en-v1.5") -> np.ndarray:
    """Query-side vectors. `query_embed` applies the model's own query prefix if it
    has one - Chapter 25 measures what forgetting that costs."""
    m = _text_embedder(model_name)
    return np.array(list(m.query_embed(texts)), dtype=np.float32)


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def exact_search(qvec: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Brute-force cosine top-k. Returns [(row index, score)] best first."""
    sims = normalize(matrix) @ normalize(qvec)
    idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in idx]


# -------------------------------------------------------------- rerankers ---
@lru_cache(maxsize=4)
def cross_encoder(model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
    """A cross-encoder scores (query, passage) jointly. Returns None (with a note)
    if the model cannot be fetched, so the scripts still run offline."""
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        return TextCrossEncoder(model_name=model_name)
    except Exception as e:  # pragma: no cover
        print(f"(cross-encoder {model_name} unavailable: {type(e).__name__}: {str(e)[:80]})")
        return None


@lru_cache(maxsize=2)
def colbert(model_name: str = "answerdotai/answerai-colbert-small-v1"):
    try:
        from fastembed import LateInteractionTextEmbedding

        return LateInteractionTextEmbedding(model_name=model_name)
    except Exception as e:  # pragma: no cover
        print(f"(ColBERT {model_name} unavailable: {type(e).__name__}: {str(e)[:80]})")
        return None


def maxsim(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    """ColBERT's late-interaction score, written out so you can see it:

        score(q, d) = Σ_i  max_j  q_i · d_j

    For every QUERY token, find its best-matching DOCUMENT token, and add those
    up. One vector per token instead of one per document, so the document keeps
    its detail - but the comparison is still a dot product, so it stays indexable.
    """
    sim = query_tokens @ doc_tokens.T          # (n_q_tokens, n_d_tokens)
    return float(sim.max(axis=1).sum())


# ------------------------------------------------------------- utilities ---
def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, (time.perf_counter() - t0) * 1000.0


def first_rank(docs: list[Document], predicate) -> int | None:
    """1-based rank of the first document satisfying `predicate`, else None."""
    for r, d in enumerate(docs, start=1):
        if predicate(d):
            return r
    return None


def contains(pattern: str):
    """Predicate: chunk text matches this regex (case-insensitive)."""
    rx = re.compile(pattern, re.I)
    return lambda d: bool(rx.search(d.page_content))


def bar(value: float, width: int = 28, lo: float = 0.0, hi: float = 1.0) -> str:
    n = int(round((value - lo) / (hi - lo) * width)) if hi > lo else 0
    return "█" * max(0, min(width, n))


# ------------------------------------------------------- realistic queries ---
VAGUE_CACHE = DATA_DIR / "cache" / "ch24_vague_queries.json"


def vague_queries(golden: list[dict]) -> dict[str, str]:
    """The golden questions rewritten the way someone would actually type them.

    Questions written *from* a document reuse its vocabulary, which flatters
    retrieval (Chapter 10 §10.2 warns about exactly this). Rerankers are judged
    unfairly on such a set. One cached LLM call per 14 questions fixes it; the
    cache means every later run is free and identical.
    """
    import json

    if VAGUE_CACHE.exists():
        return json.loads(VAGUE_CACHE.read_text())

    from pydantic import BaseModel, Field

    from ragbook import get_llm

    class Item(BaseModel):
        id: str
        vague: str = Field(description="how a busy employee would actually ask this in Slack")

    class Out(BaseModel):
        items: list[Item]

    llm = get_llm().with_structured_output(Out)
    out: dict[str, str] = {}
    for i in range(0, len(golden), 14):
        batch = golden[i : i + 14]
        listing = "\n".join(f"{g['id']}: {g['question']}" for g in batch)
        r = llm.invoke(
            "Rewrite each question the way a busy employee would type it into Slack: "
            "vague, conversational, no jargon from the document, possibly indirect. "
            "Keep the same information need.\n\n" + listing
        )
        out.update({it.id: it.vague for it in r.items})
    VAGUE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    VAGUE_CACHE.write_text(json.dumps(out, indent=1))
    return out
