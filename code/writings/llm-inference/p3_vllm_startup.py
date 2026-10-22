"""Part 3: work out, by hand, the two KV lines that vLLM prints at startup, for Llama 3.1 8B on one H100 80 GB.
COMPUTED, NOT MEASURED: there is no NVIDIA GPU here. The formulas follow vLLM v0.32.0:
  vllm/v1/worker/utils.py          requested_memory = total_memory x gpu_memory_utilization
  vllm/v1/worker/gpu_worker.py     available KV memory = requested_memory - non-KV memory (weights, activations...)
                                   logs "Available KV cache memory: %s GiB"
  vllm/v1/core/kv_cache_utils.py   num_blocks = available // (page bytes per layer x layers)
                                   max concurrency = num_blocks / blocks per request at max_model_len
                                   logs "%s KV cache size: %s tokens, Maximum concurrency for %s tokens per request: %.2fx"
Assumptions (stated in the text): total memory = the spec's 80 GB; 4 GB of non-weight working memory, as in the
Part 3 simulation (the real number comes from vLLM's profiling run). Output: results/p3_vllm_startup_stdout.txt"""
TOTAL = 80e9
UTIL = 0.92                         # vLLM v0.32.0 default (vllm/config/cache.py)
WEIGHTS = 8_030_261_248 * 2         # Llama 3.1 8B in BF16 (p2_gpu_capacity.py)
WORKING = 4e9                       # assumption
LAYERS, KV_HEADS, HEAD_DIM, BLOCK = 32, 8, 128, 16
GiB = 1024 ** 3

requested = TOTAL * UTIL
print(f"1. vLLM may use {UTIL} x {TOTAL / 1e9:.0f} GB = {requested / 1e9:.2f} GB")
avail = requested - WEIGHTS - WORKING
print(f"2. minus weights {WEIGHTS / 1e9:.2f} GB and working memory {WORKING / 1e9:.0f} GB = {avail / 1e9:.2f} GB for the KV cache")
print(f"   log line: Available KV cache memory: {avail / GiB:.2f} GiB")
for dtype, bpn in (('auto (BF16)', 2), ('fp8', 1)):
    page = 2 * BLOCK * KV_HEADS * HEAD_DIM * bpn          # one block of one layer: K and V for 16 tokens
    per_block = page * LAYERS
    blocks = int(avail // per_block)
    tokens = blocks * BLOCK
    print(f"\n--kv-cache-dtype {dtype}: one block = {page:,} bytes per layer x {LAYERS} layers = {per_block / 2**20:.0f} MiB")
    print(f"3. blocks = {avail / 1e9:.2f} GB / {per_block / 2**20:.0f} MiB = {blocks:,} blocks = {tokens:,} tokens")
    for max_len in (131072, 8192):
        conc = blocks / -(-max_len // BLOCK)
        print(f"4. --max-model-len {max_len:,}: {max_len // BLOCK:,} blocks per request -> "
              f"log line: GPU KV cache size: {tokens:,} tokens, Maximum concurrency for {max_len:,} tokens per request: {conc:.2f}x")
    for L in (1024, 4096):
        print(f"   requests of {L:,} tokens (prompt + answer): {blocks // (L // BLOCK):,} fit at once")
