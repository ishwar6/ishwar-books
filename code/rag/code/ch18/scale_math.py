"""Chapter 18 - do the arithmetic BEFORE you pick an architecture.

Run:  uv run python code/ch18/scale_math.py
"""
CHARS_PER_TOKEN = 4          # English prose, rough
CHUNK_CHARS = 800            # what ragbook.index uses
OVERLAP_CHARS = 120
EMBED_PRICE_PER_M = 0.02     # text-embedding-3-small, $ per 1M tokens
DIMS = 1536
EMBED_TOKENS_PER_SEC = 20_000   # a realistic sustained rate against one API key (tier-dependent)
QPS_HNSW_PER_NODE = 300          # ballpark searches/sec one Qdrant node serves for ~10M vectors in RAM

GB = 1024 ** 3


def row(size_bytes: int) -> dict:
    tokens = size_bytes / CHARS_PER_TOKEN
    effective_chunk = CHUNK_CHARS - OVERLAP_CHARS          # overlap means you embed ~15% extra
    chunks = size_bytes / effective_chunk
    embed_tokens = chunks * CHUNK_CHARS / CHARS_PER_TOKEN  # overlap included
    return {
        "corpus": f"{size_bytes / GB:,.0f} GB",
        "tokens": tokens,
        "chunks": chunks,
        "embed_cost_$": embed_tokens / 1e6 * EMBED_PRICE_PER_M,
        "embed_hours": embed_tokens / EMBED_TOKENS_PER_SEC / 3600,
        "ram_fp32_GB": chunks * DIMS * 4 / GB,
        "ram_int8_GB": chunks * DIMS * 1 / GB,
        "ram_binary_GB": chunks * DIMS / 8 / GB,
        "payload_GB": chunks * (CHUNK_CHARS + 300) / GB,   # text + metadata stored alongside
    }


if __name__ == "__main__":
    sizes = [1 * GB, 10 * GB, 100 * GB, 1024 * GB]
    rows = [row(s) for s in sizes]
    hdr = f"{'corpus':>8} {'tokens':>9} {'chunks':>9} {'embed $':>9} {'embed h':>8} {'fp32 GB':>8} {'int8 GB':>8} {'bin GB':>7} {'text GB':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['corpus']:>8} {r['tokens']/1e9:>8.2f}B {r['chunks']/1e6:>8.1f}M {r['embed_cost_$']:>9,.0f} "
              f"{r['embed_hours']:>8.1f} {r['ram_fp32_GB']:>8.1f} {r['ram_int8_GB']:>8.1f} {r['ram_binary_GB']:>7.2f} {r['payload_GB']:>8.1f}")

    print("\nPer-vector memory at 1536 dims: fp32 = 6,144 B | int8 = 1,536 B | binary = 192 B")
    print("HNSW graph adds roughly m*2*4 bytes per vector (m=16 -> ~128 B) on top.\n")

    # cost per query: embed the question + LLM tokens; the vector search itself is ~free at the margin
    q_embed_tokens, ctx_tokens, out_tokens = 20, 4 * 200 + 150, 120
    embed = q_embed_tokens / 1e6 * EMBED_PRICE_PER_M
    llm_in, llm_out = ctx_tokens / 1e6 * 0.75, out_tokens / 1e6 * 4.50       # gpt-5.4-mini list prices
    per_q = embed + llm_in + llm_out
    print(f"Per query (k=4, ~200-token chunks, gpt-5.4-mini): ${per_q:.5f}  ->  1M queries/month ≈ ${per_q*1e6:,.0f}")
    print(f"  of which LLM = {(llm_in+llm_out)/per_q:.1%}, question embedding = {embed/per_q:.2%}. Retrieval is cheap; generation is where the money goes.")
    print(f"  Add a reranker over 20 candidates (+~4k tokens with an LLM reranker): ≈ +${4000/1e6*0.75:.4f}/query")
    print(f"\nThroughput: 1M queries/month = {1e6/30/86400:.2f} qps average; plan for 10x peak = "
          f"{1e6/30/86400*10:.0f} qps -> {max(1, round(1e6/30/86400*10/QPS_HNSW_PER_NODE))} Qdrant node(s) at ~{QPS_HNSW_PER_NODE} qps each.")
