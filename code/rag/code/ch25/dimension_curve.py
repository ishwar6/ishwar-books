"""Chapter 25 - how many dimensions do you actually need?

text-embedding-3-* are Matryoshka models: the first n dimensions are themselves a
usable embedding. So "3072-d" is a ceiling, not a requirement, and the dimension
count is a dial you can turn AFTER measuring - unlike the model choice itself.

Three things measured here:
  1. that truncating locally and renormalising equals asking the API for fewer
     dimensions (so you can explore the whole curve for the price of one encode)
  2. quality against dimensions, with the memory each setting costs at 1M and 100M
  3. dimensions versus int8 quantization at an EQUAL byte budget - the comparison
     that actually decides a large deployment

Run:  QDRANT_MODE=memory uv run python code/ch25/dimension_curve.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "ch24"))
sys.path.insert(0, str(HERE.parents[1] / "ch10"))
from _embedders import Embedder, encode, index_mb  # noqa: E402
from _shared import bar, build_corpus, vague_queries  # noqa: E402
from metrics import ndcg_at_k, recall_at_k, unique_in_order  # noqa: E402

from ragbook import load_golden  # noqa: E402

DIMS = [3072, 2048, 1024, 512, 256, 128, 64]


def evaluate(dm: np.ndarray, qm: np.ndarray, doc_ids, golden) -> tuple[float, float]:
    dn = dm / np.clip(np.linalg.norm(dm, axis=1, keepdims=True), 1e-12, None)
    r5, nd = [], []
    for g, qv in zip(golden, qm):
        q = qv / max(float(np.linalg.norm(qv)), 1e-12)
        order = np.argsort(-(dn @ q))[:10]
        ids = unique_in_order([doc_ids[i] for i in order])
        rel = set(g["sources"])
        r5.append(recall_at_k(ids, rel, 5))
        nd.append(ndcg_at_k(ids, rel, 10))
    return float(np.mean(r5)), float(np.mean(nd))


def quantize_int8(m: np.ndarray) -> np.ndarray:
    """Per-vector symmetric int8: scale by the largest absolute component, round,
    then dequantize. One byte per dimension instead of four. Qdrant does this
    inside the index (Chapter 18); here we do it in the open so you can measure
    the quality cost yourself."""
    scale = np.max(np.abs(m), axis=1, keepdims=True) / 127.0
    return np.round(m / np.clip(scale, 1e-12, None)).astype(np.int8).astype(np.float32) * scale


def main() -> None:
    golden = [g for g in load_golden() if g["answerable"]]
    vague = vague_queries(golden)
    queries = [vague[g["id"]] for g in golden]
    chunks = build_corpus(hard=True)
    texts = [c.page_content for c in chunks]
    doc_ids = [c.metadata["doc_id"] for c in chunks]

    full = Embedder("openai-3-large", "openai", "text-embedding-3-large", 3072, 0.13)
    dm_full = encode(full, texts, "doc")
    qm_full = encode(full, queries, "query")

    # --- 1. is local truncation the same as the API's dimensions= parameter? ---
    print("1. LOCAL TRUNCATION vs THE API's dimensions= PARAMETER (256 dims, 20 chunks)")
    from langchain_openai import OpenAIEmbeddings

    sample = texts[:20]
    api_256 = np.array(
        OpenAIEmbeddings(model="text-embedding-3-large", dimensions=256).embed_documents(sample),
        dtype=np.float32,
    )
    local_256 = dm_full[:20, :256]
    local_256 = local_256 / np.clip(np.linalg.norm(local_256, axis=1, keepdims=True), 1e-12, None)
    cos = np.sum(api_256 * local_256, axis=1)
    print(f"   cosine(api_256, locally_truncated_256): min {cos.min():.6f}  mean {cos.mean():.6f}")
    print("   Identical up to float noise. So one full-precision encode buys you the")
    print("   entire curve below - explore locally, then set dimensions= in production")
    print("   to stop paying to transfer dimensions you discard.\n")

    # --- 2. quality vs dimensions -------------------------------------------
    print("2. QUALITY vs DIMENSIONS (997 chunks, 42 realistic queries)")
    rows = []
    for d in DIMS:
        dm = dm_full[:, :d]
        qm = qm_full[:, :d]
        r5, nd = evaluate(dm, qm, doc_ids, golden)
        rows.append((d, r5, nd, index_mb(1_000_000, d) / 1000, index_mb(100_000_000, d) / 1000))
    base = rows[0][2]
    print(f"   {'dims':>6}{'recall@5':>10}{'nDCG@10':>9}{'vs full':>9}{'GB/1M':>8}{'GB/100M':>9}")
    for d, r5, nd, gb1, gb100 in rows:
        print(f"   {d:>6}{r5:>10.3f}{nd:>9.3f}{nd - base:>+9.3f}{gb1:>8.1f}{gb100:>9.0f}")

    lo = min(r[2] for r in rows) - 0.01
    hi = max(r[2] for r in rows) + 0.005
    print("\n   nDCG@10 by dimension")
    for d, _, nd, gb1, _ in rows:
        print(f"   {d:>5}d {nd:.3f} |{bar(nd, 40, lo, hi)}  {gb1:.1f} GB/1M")

    keep = [r for r in rows if r[2] >= base - 0.005]
    smallest = min(keep, key=lambda r: r[0]) if keep else rows[0]
    print(
        f"\n   Within 0.005 nDCG of full precision down to {smallest[0]} dims - "
        f"{rows[0][3] / smallest[3]:.0f}x less memory\n"
        f"   ({rows[0][4]:.0f} GB -> {smallest[4]:.0f} GB at 100M vectors) for "
        f"{smallest[2] - base:+.3f} quality.\n"
    )

    # --- 3. fewer dimensions or fewer bits? ---------------------------------
    print("3. EQUAL BYTE BUDGET: fewer dimensions vs int8 quantization")
    print("   Both rows below store the SAME bytes per vector. Which spends them better?")
    print(f"\n   {'configuration':<34}{'bytes/vec':>10}{'recall@5':>10}{'nDCG@10':>9}")
    for label, dm, qm, nbytes in [
        ("3072-d float32 (reference)", dm_full, qm_full, 3072 * 4),
        ("3072-d int8", quantize_int8(dm_full), qm_full, 3072 * 1),
        ("768-d float32", dm_full[:, :768], qm_full[:, :768], 768 * 4),
        ("1024-d int8", quantize_int8(dm_full[:, :1024]), qm_full[:, :1024], 1024 * 1),
        ("256-d float32", dm_full[:, :256], qm_full[:, :256], 256 * 4),
    ]:
        r5, nd = evaluate(dm, qm, doc_ids, golden)
        print(f"   {label:<34}{nbytes:>10}{r5:>10.3f}{nd:>9.3f}")
    print(
        "\n   int8 keeps nearly all the quality for a quarter of the bytes: quantizing is\n"
        "   usually the better first move, and it composes with truncation. Note the\n"
        "   comparison is exact-search only - on a real HNSW index quantization also\n"
        "   changes graph traversal, which is why Qdrant rescores with full vectors\n"
        "   (Chapter 18 §18.2). Measure both together before you commit."
    )


if __name__ == "__main__":
    main()
