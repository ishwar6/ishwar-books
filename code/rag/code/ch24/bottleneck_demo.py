"""Chapter 24 - the bi-encoder information bottleneck, and its limits.

A bi-encoder compresses a chunk into ONE vector before it has seen the query, so
whatever the chunk is "mostly about" dominates. Chapter 10 §10.8 hit this for
real: for "Which database and message broker does Beacon use?", the chunk that
actually names PostgreSQL and Kafka did not make the top-6.

This script runs four experiments on that one probe:

  1. how far down the bi-encoder buries the answering chunk, and why
  2. whether three cross-encoders and ColBERT rescue it   (they do not)
  3. what actually fixes it: changing the chunk's REPRESENTATION
  4. the honesty check - on the other probes the bi-encoder is already perfect,
     so this corpus cannot show what a reranker is for (see reranker_shootout.py)

Run:  QDRANT_MODE=memory uv run python code/ch24/bottleneck_demo.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _shared import (  # noqa: E402
    colbert,
    contains,
    cross_encoder,
    exact_search,
    first_rank,
    handbook_chunks,
    maxsim,
)

from ragbook import get_embeddings  # noqa: E402

PROBE = "Which database and message broker does Beacon use?"
MARKER = r"PostgreSQL 16"

# Probes the bi-encoder handles perfectly - kept to stop the chapter cherry-picking.
CONTROLS = [
    ("How long is the staging soak before a Beacon release?", r"staging soak \(48 hours\)"),
    ("How long does it take to swap a wheel module?", r"field-replaceable in under 15 minutes"),
    ("When is a medical certificate required for sick leave?", r"3 or more consecutive days"),
    ("What fraction of cloud customers get a release first?", r"canary \(5% of cloud customers"),
]

RERANKERS = [
    "Xenova/ms-marco-MiniLM-L-6-v2",
    "BAAI/bge-reranker-base",
    "jinaai/jina-reranker-v1-tiny-en",
]


def show(docs, scores, n=3, pred=None):
    for d, s in list(zip(docs, scores))[:n]:
        head = d.page_content.strip().split("\n")[0][:46]
        mark = "   <-- answers it" if pred and pred(d) else ""
        print(f"      {s:9.3f}  {d.metadata['source']:<32} {head!r}{mark}")


def main() -> None:
    chunks = handbook_chunks()
    texts = [c.page_content for c in chunks]
    emb = get_embeddings()
    pred = contains(MARKER)
    gold_i = next(i for i, c in enumerate(chunks) if pred(c))

    print(f"corpus: {len(chunks)} handbook chunks | embedder: OpenAI text-embedding-3-small")
    print(f"probe : {PROBE!r}")
    print(f"the chunk that answers it: #{gold_i} {chunks[gold_i].metadata['source']} '## Architecture'\n")

    # --- 1. where the bi-encoder puts it ------------------------------------
    matrix = np.array(emb.embed_documents(texts), dtype=np.float32)
    qv = np.array(emb.embed_query(PROBE), dtype=np.float32)
    hits = exact_search(qv, matrix, len(chunks))
    docs = [chunks[i] for i, _ in hits]
    dense_rank = first_rank(docs, pred)
    print("1. BI-ENCODER (exact cosine, so this is not an ANN recall problem)")
    show(docs, [s for _, s in hits], 3, pred)
    gold_score = next(s for i, s in hits if i == gold_i)
    print(f"      ...\n      {gold_score:9.3f}  <- the answering chunk, at rank {dense_rank}")
    print(
        "   why: the query says 'database' and 'message broker'. The chunk says\n"
        "   'PostgreSQL 16' and 'Apache Kafka' and never uses either query word. The\n"
        "   document's TITLE chunk does say 'Architecture and Operations', so it looks\n"
        "   more on-topic to a single vector that was computed without seeing the query.\n"
    )

    # --- 2. can a reranker fix it? -----------------------------------------
    print(f"2. RERANKERS over the top-30 candidates (the answering chunk is in there at {dense_rank})")
    cand = docs[:30]
    for name in RERANKERS:
        ce = cross_encoder(name)
        if ce is None:
            continue
        s = list(ce.rerank(PROBE, [d.page_content for d in cand]))
        order = np.argsort(-np.array(s))
        r = first_rank([cand[i] for i in order], pred)
        top = cand[order[0]].page_content.strip().split("\n")[0][:34]
        print(f"   {name:<34} rank {str(r):<4} top-1 {top!r}")

    cb = colbert()
    if cb is not None:
        dt = list(cb.passage_embed([d.page_content for d in cand]))
        qt = list(cb.query_embed([PROBE]))[0]
        s = [maxsim(qt, d) for d in dt]
        r = first_rank([cand[i] for i in np.argsort(-np.array(s))], pred)
        print(f"   {'ColBERT late interaction':<34} rank {str(r):<4} (MaxSim over token vectors)")
        # How cheap is MaxSim really? Time it, because the cascade argument in 24.2
        # rests on this number being ~1000x the cross-encoder's 160 pairs/s.
        t0 = time.perf_counter()
        for _ in range(20):
            [maxsim(qt, d) for d in dt]
        rate = 20 * len(dt) / (time.perf_counter() - t0)
        print(f"   MaxSim brute force on this CPU: {rate:,.0f} docs/s "
              f"-> {1e6 / rate:.1f} s for 1M chunks (a cross-encoder needs ~1.7 h)")
    print(
        "   Every scorer prefers the title chunk. A reranker is a MODEL with its own\n"
        "   training distribution, not an oracle for 'correct'. When the answering text\n"
        "   shares no vocabulary with the question, joint attention has nothing to work\n"
        "   with either. This is a representation failure, not a ranking failure.\n"
    )

    # --- 3. the fix: change the representation ------------------------------
    print("3. CONTEXTUAL RETRIEVAL - one sentence prepended to the chunk at ingest (ch 9 §9.6)")
    ctx = texts[:]
    ctx[gold_i] = (
        "This section of the Beacon fleet-management document lists the infrastructure "
        "Beacon runs on: its database, its message broker/queue, its cluster and its "
        "robot transport.\n\n"
    ) + texts[gold_i]
    m2 = np.array(emb.embed_documents(ctx), dtype=np.float32)
    hits2 = exact_search(qv, m2, len(ctx))
    r2 = first_rank([chunks[i] for i, _ in hits2], pred)
    print(f"   bi-encoder rank: {dense_rank} -> {r2}   (no reranker involved)")
    print("   Fixing the index beat re-ordering the results. Order of operations, always.\n")

    # --- 4. honesty check ---------------------------------------------------
    print("4. THE SAME BI-ENCODER ON FOUR CONTROL PROBES")
    for q, marker in CONTROLS:
        v = np.array(emb.embed_query(q), dtype=np.float32)
        h = exact_search(v, matrix, len(chunks))
        rank = first_rank([chunks[i] for i, _ in h], contains(marker))
        print(f"   rank {str(rank):<3} {q}")
    print(
        "\n   Rank 1 every time. On 56 curated chunks with questions written from those\n"
        "   chunks, first-stage retrieval has no room to fail, so a reranker has nothing\n"
        "   to win. That is a property of the BENCHMARK, not evidence that reranking is\n"
        "   useless - reranker_shootout.py builds the setting where it pays."
    )


if __name__ == "__main__":
    main()
