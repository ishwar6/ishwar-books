"""Chapter 25 - one interface over every embedding model, plus a disk cache.

A bake-off is only fair if every model sees the same corpus, the same queries and
the same chunking, and only the model changes. That is what this file enforces.

The cache matters more than it looks: without it you re-pay the API on every run
and you are tempted to shrink the corpus until the experiment is meaningless.

Not a runnable chapter script - the ch25 scripts import it.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ragbook import DATA_DIR

CACHE_DIR = DATA_DIR / "cache" / "ch25_embeddings"

# $ per 1M tokens, from the official pricing page (Sep 2026). Local models are free
# to call but cost you CPU/RAM and an ops burden - priced at 0 here, not at "free".
PRICE_PER_1M = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}


# Instruction prefixes the model cards require. We apply these OURSELVES, by hand,
# because fastembed's query_embed()/passage_embed() return vectors identical to
# embed() for these models - verified with np.allclose in ch25's prefix experiment.
# Trusting the helper silently cost arctic-embed 0.42 nDCG. See §25.3.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@dataclass
class Embedder:
    """A model under test. `slice_dim` implements Matryoshka truncation."""

    name: str                      # label in the tables
    backend: str                   # "openai" | "fastembed"
    model_id: str
    dim: int
    price_per_1m: float = 0.0
    slice_dim: int | None = None   # keep only the first n dims, then renormalise
    query_prefix: str = ""         # prepended to QUERIES before encoding
    doc_prefix: str = ""           # prepended to DOCUMENTS before encoding
    notes: str = ""
    timings: dict = field(default_factory=dict)

    @property
    def out_dim(self) -> int:
        return self.slice_dim or self.dim


def _norm(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12, None)


def _truncate(m: np.ndarray, d: int | None) -> np.ndarray:
    """Matryoshka: the first d dimensions are themselves a usable embedding -
    but you must renormalise, because the tail you dropped carried some norm.
    This is what OpenAI's `dimensions=` parameter does server-side."""
    return _norm(m[:, :d]) if d else m


@lru_cache(maxsize=8)
def _fastembed(model_id: str):
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_id)


def _key(emb: Embedder, kind: str, texts: list[str]) -> str:
    h = hashlib.sha256(("\x00".join(texts)).encode()).hexdigest()[:16]
    slug = emb.name.replace("/", "_").replace(" ", "_")
    return f"{slug}.{kind}.{h}.npy"


def _cached(emb: Embedder, kind: str, texts: list[str], compute) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / _key(emb, kind, texts)
    if path.exists():
        emb.timings.setdefault(kind, 0.0)   # cached: no honest timing to report
        return np.load(path)
    t0 = time.perf_counter()
    m = compute()
    emb.timings[kind] = time.perf_counter() - t0
    np.save(path, m)
    return m


def encode(
    emb: Embedder, texts: list[str], kind: str, with_prefix: bool = True, cache: bool = True
) -> np.ndarray:
    """kind is "doc" or "query" - the distinction matters for asymmetric models.

    with_prefix=False measures what forgetting the instruction costs (§25.3).
    cache=False forces a fresh encode, for honest latency timing.
    """
    prefix = (emb.query_prefix if kind == "query" else emb.doc_prefix) if with_prefix else ""
    prepared = [prefix + t for t in texts] if prefix else texts

    def compute() -> np.ndarray:
        if emb.backend == "openai":
            from langchain_openai import OpenAIEmbeddings

            model = OpenAIEmbeddings(model=emb.model_id)
            raw = (
                model.embed_documents(prepared)
                if kind == "doc"
                else [model.embed_query(t) for t in prepared]
            )
            return np.array(raw, dtype=np.float32)

        # Note we call plain embed() on already-prefixed text rather than
        # query_embed()/passage_embed(): those helpers are no-ops here.
        return np.array(list(_fastembed(emb.model_id).embed(prepared)), dtype=np.float32)

    if not cache:
        return _truncate(compute(), emb.slice_dim)
    return _truncate(_cached(emb, kind, prepared, compute), emb.slice_dim)


def index_mb(n_vectors: int, dim: int, bytes_per_value: int = 4) -> float:
    """Raw float32 vector storage. Real indexes add HNSW graph links and payload
    on top (Chapter 18); this is the floor, not the bill."""
    return n_vectors * dim * bytes_per_value / 1e6


def embed_cost(n_tokens: int, price_per_1m: float) -> float:
    return n_tokens / 1e6 * price_per_1m


# The line-up. Keep every entry runnable on a laptop.
def default_models() -> list[Embedder]:
    return [
        Embedder("openai-3-small", "openai", "text-embedding-3-small", 1536,
                 PRICE_PER_1M["text-embedding-3-small"], notes="the book's default"),
        Embedder("openai-3-large", "openai", "text-embedding-3-large", 3072,
                 PRICE_PER_1M["text-embedding-3-large"], notes="best OpenAI quality"),
        Embedder("openai-3-large@512", "openai", "text-embedding-3-large", 3072,
                 PRICE_PER_1M["text-embedding-3-large"], slice_dim=512,
                 notes="Matryoshka truncation of the row above"),
        Embedder("bge-small-en-v1.5", "fastembed", "BAAI/bge-small-en-v1.5", 384,
                 query_prefix=QUERY_INSTRUCTION, notes="67 MB self-hosted; card: prefix 'not so necessary'"),
        Embedder("all-MiniLM-L6-v2", "fastembed", "sentence-transformers/all-MiniLM-L6-v2", 384,
                 notes="2021 baseline, symmetric, 256-token limit, no prefix"),
        Embedder("arctic-embed-s", "fastembed", "snowflake/snowflake-arctic-embed-s", 384,
                 query_prefix=QUERY_INSTRUCTION, notes="2024; card: prefix REQUIRED - and it is"),
    ]
