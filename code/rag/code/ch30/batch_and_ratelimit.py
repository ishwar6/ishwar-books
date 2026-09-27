"""Chapter 30 - batching, rate limits, caching, and what a million documents cost.

    uv run python code/ch30/batch_and_ratelimit.py
    uv run python code/ch30/batch_and_ratelimit.py --texts 128     # cheaper/faster

Three things decide how long a bulk embed takes, and none of them is the model:

  batch size   one HTTPS round trip per call. Batching amortises it; the curve
               flattens once the payload, not the round trip, dominates.
  rate limit   providers meter TOKENS per minute, not requests. Your ceiling is
               TPM / tokens-per-document documents per minute, full stop.
  cache        content-hash your chunks. Re-crawls and retries then cost zero,
               which matters more than any of the above.

The batch-size numbers below are measured against the real API. The rate limiter
runs on a virtual clock so you can watch an hour of throttling in a millisecond.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import time

from ragbook import EMBED_MODEL, chunk_documents, get_embeddings, load_handbook

EMBED_PRICE_PER_1M = {"text-embedding-3-small": 0.02, "text-embedding-3-large": 0.13}


# ------------------------------------------------------------ rate limiter --
class TokenBucket:
    """The standard limiter: tokens refill at a constant rate, work waits for
    capacity. Burst = bucket size, sustained rate = refill rate.

    `clock` is injected so tests (and this demo) can run on a virtual clock.
    """

    def __init__(self, tokens_per_minute: int, burst: int | None = None, clock=time.monotonic):
        self.rate = tokens_per_minute / 60.0
        self.capacity = burst or tokens_per_minute
        self.tokens = float(self.capacity)
        self.clock = clock
        self.last = clock()

    def consume(self, tokens: int) -> float:
        """Returns how long the caller had to wait."""
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens >= tokens:
            self.tokens -= tokens
            return 0.0
        wait = (tokens - self.tokens) / self.rate
        self.tokens = 0.0
        self.last = now + wait
        return wait


def approx_tokens(text: str) -> int:
    """~4 characters per token is close enough for budgeting; use tiktoken when
    you need the real number for a billing forecast."""
    return max(1, len(text) // 4)


# ------------------------------------------------------------------ cache ---
class EmbeddingCache:
    """Keyed by content hash, not by document id: the same paragraph in forty
    documents is embedded once. Persist this in production (Redis/S3/a table)."""

    def __init__(self):
        self.store: dict[str, list[float]] = {}
        self.hits = self.misses = 0

    def embed(self, texts: list[str], embedder) -> list[list[float]]:
        keys = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
        todo = [t for t, k in zip(texts, keys) if k not in self.store]
        if todo:
            fresh = embedder.embed_documents(todo)
            for t, vec in zip(todo, fresh):
                self.store[hashlib.sha256(t.encode()).hexdigest()] = vec
        self.hits += len(texts) - len(todo)
        self.misses += len(todo)
        return [self.store[k] for k in keys]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", type=int, default=256)
    ap.add_argument("--batches", type=int, nargs="*", default=[1, 8, 32, 64, 128, 256])
    args = ap.parse_args()

    chunks = chunk_documents(load_handbook())
    texts = [c.page_content for c in chunks]
    while len(texts) < args.texts:                    # pad by reusing chunks
        texts += texts
    texts = texts[: args.texts]
    tokens = sum(approx_tokens(t) for t in texts)
    emb = get_embeddings()

    # ---- 1. throughput vs batch size (real API calls) ----------------------
    print(f"embedding {len(texts)} chunks (~{tokens:,} tokens) with {EMBED_MODEL}\n")
    print(f"{'batch':>6} {'calls':>6} {'seconds':>8} {'texts/s':>8} {'ms/call':>8}")
    print("-" * 40)
    baseline = None
    for size in args.batches:
        if size > len(texts):
            continue
        t0 = time.perf_counter()
        calls = 0
        for i in range(0, len(texts), size):
            emb.embed_documents(texts[i:i + size])
            calls += 1
        dt = time.perf_counter() - t0
        baseline = baseline or dt
        print(f"{size:6} {calls:6} {dt:8.2f} {len(texts)/dt:8.1f} {dt/calls*1000:8.0f}")
    print(f"\nbatching {args.batches[0]} -> {args.batches[-1]} is the cheapest speedup you")
    print("will ever get: same tokens, same price, fewer round trips.")

    # ---- 2. the rate limit is the real ceiling -----------------------------
    print("\n--- what a token-per-minute limit means for a bulk load ---")
    now = [0.0]
    bucket = TokenBucket(tokens_per_minute=1_000_000, burst=200_000, clock=lambda: now[0])
    per_doc_tokens = 2_500                            # ~10 chunks of 250 tokens
    waited = 0.0
    for _ in range(2_000):                            # 2,000 documents
        wait = bucket.consume(per_doc_tokens)
        waited += wait
        now[0] += wait + 0.001
    print(f"1M tokens/min, {per_doc_tokens:,} tokens/document:")
    print(f"  2,000 documents took {now[0]/60:.1f} simulated minutes "
          f"({waited/60:.1f} min of it waiting on the limiter)")
    for tpm in (350_000, 1_000_000, 5_000_000):
        docs_per_min = tpm / per_doc_tokens
        print(f"  {tpm:>9,} TPM -> {docs_per_min:7,.0f} docs/min -> "
              f"1M documents in {1_000_000/docs_per_min/60:6.1f} hours on one key")

    # ---- 3. the cache ------------------------------------------------------
    print("\n--- content-hash cache, the same corpus twice ---")
    cache = EmbeddingCache()
    t0 = time.perf_counter()
    cache.embed(texts, emb)
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    cache.embed(texts, emb)
    warm = time.perf_counter() - t0
    total = cache.hits + cache.misses
    print(f"  cold {cold:5.2f}s   warm {warm:5.2f}s   "
          f"hit rate {cache.hits/total:.0%} ({cache.hits}/{total})")
    dupes = len(texts) - len({hashlib.sha256(t.encode()).hexdigest() for t in texts})
    print(f"  {dupes} of {len(texts)} chunks were duplicates of another chunk - "
          f"never embedded twice")
    print("  (this corpus is padded by repetition to reach --texts, so that ratio is an")
    print("   artifact; on a real crawl expect 1-20% exact-duplicate chunks)")

    # ---- 4. the cost table -------------------------------------------------
    print("\n--- one million documents ---")
    price = EMBED_PRICE_PER_1M.get(EMBED_MODEL, 0.02)
    print(f"{'pages/doc':>9} {'chunks/doc':>11} {'chunks':>12} {'tokens':>14} "
          f"{'embed $':>9} {'hours@1M TPM':>13}")
    for pages, per_doc in ((1, 3), (5, 15), (20, 60), (50, 150)):
        total_chunks = per_doc * 1_000_000
        total_tokens = total_chunks * 250
        hours = total_tokens / 1_000_000 / 60
        print(f"{pages:9} {per_doc:11} {total_chunks:12,} {total_tokens:14,} "
              f"{total_tokens/1e6*price:9,.0f} {hours:13,.1f}")
    print(f"\nprices: {EMBED_MODEL} at ${price}/1M tokens. The async Batch API is ~50%")
    print("cheaper with a 24h turnaround - right for a backfill, wrong for live updates.")
    print("Parsing, not embedding, is what actually costs you: see 30.2.")


if __name__ == "__main__":
    main()
