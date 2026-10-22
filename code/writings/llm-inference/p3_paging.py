"""Part 3 of "LLM Inference from the Ground Up": paging the KV cache, with small checks you can follow by hand.

What it shows (every number in Part 3 that is not from the paper or from Parts 1 and 2 comes from here):
  1. The allocation simulation already published in Part 3 (four policies, 200,000 requests, Llama 3.1 8B, 80 GB).
  2. The static vs continuous batching simulation already published (4,096 requests, 64 slots).
  3. The worked examples of the three wastes: a 37-token request with a 2,048-slot reservation vs 16-token blocks.
  4. One request growing token by token: when it gets a new block.
  5. External fragmentation on a tiny memory of 16 slots.
  6. PagedAttention by hand: attention read block by block through a block table equals ordinary attention.
  7. Copy-on-write for two samples of one prompt, with reference counts.
  8. Prefix caching: chained block hashes for three prompts.
Pure Python + numpy + torch (CPU). Run: ~/.venvs/attn/bin/python p3_paging.py
Output: results/p3_paging_stdout.txt
"""
import hashlib, heapq, math, os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'p3_paging_stdout.txt')
os.makedirs(os.path.dirname(OUT), exist_ok=True)
_lines = []


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    _lines.append(s)
    open(OUT, 'w').write('\n'.join(_lines) + '\n')


# ---------------------------------------------------------------- 1. allocation policies (same code as the article)
log('== 1. How much reserved KV memory holds real tokens (simulation, Llama 3.1 8B, one 80 GB GPU)')
rng = np.random.default_rng(42)
PER_TOKEN = 2 * 32 * 8 * 128 * 2                    # 128 KiB, from the config
BUDGET = 0.9 * 80e9 - 8.03e9 * 2 - 4e9              # 90% of memory, minus weights and ~4 GB working memory
N, MAX_LEN, MAX_TOKENS, BLOCK = 200_000, 8192, 2048, 16
prompt = np.clip(rng.lognormal(math.log(512), 0.9, N), 16, 6000).astype(int)
output = np.clip(rng.lognormal(math.log(256), 0.9, N), 8, MAX_TOKENS).astype(int)
now = prompt + (rng.uniform(0, 1, N) * output).astype(int)   # a random moment in each request's life
policies = {
    "reserve max length": np.full(N, MAX_LEN),
    "reserve prompt + max_tokens": prompt + MAX_TOKENS,
    "reserve exact final length": prompt + output,
    "paged, 16-token blocks": np.ceil(now / BLOCK).astype(int) * BLOCK,
}
for name, reserved in policies.items():
    log(f"{name:28s} useful {now.sum() / reserved.sum():6.1%}   fits {BUDGET / (reserved.mean() * PER_TOKEN):5.0f} requests")
paged = policies["paged, 16-token blocks"]
log(f"paged: mean empty slots in the last block {np.mean(paged - now):.1f} (always 0 to 15)")

# ---------------------------------------------------------------- 2. static vs continuous batching
log('\n== 2. Static vs continuous batching (simulation, 4,096 requests, 64 slots)')
SLOTS = 64
lens = np.clip(rng.lognormal(math.log(256), 0.9, 4096), 8, 2048).astype(int)
static = sum(lens[i:i + SLOTS].max() for i in range(0, len(lens), SLOTS))
free = [0] * SLOTS
for L in lens:
    heapq.heappush(free, heapq.heappop(free) + L)
continuous = max(free)
log(f"static {static:,} steps, useful {lens.sum() / (static * SLOTS):.0%}")
log(f"continuous {continuous:,} steps, useful {lens.sum() / (continuous * SLOTS):.0%}")
log(f"ratio {static / continuous:.1f}x")

# ---------------------------------------------------------------- 3. the three wastes, one request
log('\n== 3. One request of 37 tokens: a 2,048-slot reservation vs 16-token blocks')
TOK, RES, B = 37, 2048, 16
blocks = math.ceil(TOK / B)
log(f"contiguous: reserved {RES:,} slots, used {TOK}, empty {RES - TOK:,} ({TOK / RES:.1%} used)")
log(f"paged: {blocks} blocks x {B} = {blocks * B} slots, used {TOK}, empty {blocks * B - TOK} ({TOK / (blocks * B):.1%} used)")
kib = PER_TOKEN / 1024
log(f"Llama 3.1 8B, {kib:.0f} KiB per token: contiguous holds {RES * kib / 1024:.0f} MiB, "
    f"paged holds {blocks * B * kib / 1024:.0f} MiB, one 16-token block is {B * kib / 1024:.0f} MiB")
log(f"KV memory in the simulation's budget: {BUDGET / 1e9:.2f} GB = {int(BUDGET // (B * PER_TOKEN)):,} blocks of 16 tokens")
log(f"worst case for paged: {B - 1} empty slots in the last block; best case: 0")
FINAL = 300                                   # suppose the answer ends when the request holds 300 tokens
log(f"if the request will end at {FINAL} tokens: of the {RES - TOK:,} empty slots now, {FINAL - TOK} are reserved slots "
    f"(filled later) and {RES - FINAL:,} are internal fragmentation (never filled)")
fb = math.ceil(FINAL / B)
log(f"at the end, contiguous: {RES - FINAL:,} slots never used ({FINAL / RES:.1%} used); "
    f"paged: {fb} blocks = {fb * B} slots, {fb * B - FINAL} never used ({FINAL / (fb * B):.1%} used)")

# ---------------------------------------------------------------- 4. one request grows token by token
log('\n== 4. A request with a 30-token prompt grows to 50 tokens (block size 16)')
table, free_pool = [], [7, 1, 3, 12, 5]       # physical block ids, in the order the allocator hands them out
events = []
for n in range(1, 51):
    if (n - 1) % B == 0:                      # the last block is full (or there is none): get a new one
        table.append(free_pool.pop(0))
        events.append((n, list(table)))
    if n in (30, 31, 32, 33, 48, 49, 50):
        filled = n - (len(table) - 1) * B
        log(f"tokens {n:2d}: block table {table}, last block filled {filled:2d}/16, empty slots {len(table) * B - n:2d}")
log('new block at token:', ', '.join(f"{n} -> {t[-1]}" for n, t in events))

# ---------------------------------------------------------------- 5. external fragmentation on 16 slots
log('\n== 5. External fragmentation: 16 slots of memory, three requests finished, one new request')
mem = list('AAAA..BBB...CC..')        # '.' = free slot; A, B, C are requests still running
gaps, run = [], 0
for c in mem + ['#']:
    if c == '.':
        run += 1
    elif run:
        gaps.append(run); run = 0
log(f"memory {''.join(mem)}  free slots {mem.count('.')}, free gaps {gaps}, largest gap {max(gaps)}")
need = 6
log(f"a new request needs {need} slots in one piece: fits contiguously? {max(gaps) >= need}")
log(f"with blocks of 2 slots: free blocks {mem.count('.') // 2}, the request needs {need // 2}: fits? {mem.count('.') // 2 >= need // 2}")

# ---------------------------------------------------------------- 6. PagedAttention by hand
log('\n== 6. Attention read block by block through a block table equals ordinary attention (CPU, FP32)')
torch.manual_seed(0)
d, n_tok = 64, 37
q = torch.randn(d)
K = torch.randn(n_tok, d)
V = torch.randn(n_tok, d)
# store the 37 keys and values in a pool of 8 physical blocks, scattered by a block table
pool_k = torch.zeros(8, B, d)
pool_v = torch.zeros(8, B, d)
block_table = [7, 1, 3]
for i in range(n_tok):
    pool_k[block_table[i // B], i % B] = K[i]
    pool_v[block_table[i // B], i % B] = V[i]


def paged_attention(q, pool_k, pool_v, block_table, n_tok):
    """Method 1: the paper's blockwise form. Visit blocks in logical order, look up each physical block."""
    scores, values = [], []
    for j, phys in enumerate(block_table):
        filled = min(B, n_tok - j * B)                  # the last block may be partly filled
        Kj = pool_k[phys, :filled]                      # (filled, d): one lookup, one block
        scores.append(Kj @ q / math.sqrt(d))            # q . k for the tokens of this block
        values.append(pool_v[phys, :filled])
    s = torch.cat(scores)
    a = torch.exp(s - s.max()); a = a / a.sum()          # softmax over all 37 tokens
    return a @ torch.cat(values)


o_paged = paged_attention(q, pool_k, pool_v, block_table, n_tok)
o_lib = torch.nn.functional.scaled_dot_product_attention(q[None, None, None], K[None, None], V[None, None])[0, 0, 0]
log(f"block table {block_table}, {n_tok} tokens, block size {B}")
log(f"Method 1 (blocks through the table) vs Method 2 (PyTorch on one contiguous array): max difference {(o_paged - o_lib).abs().max():.1e}")
log(f"block lookups for this token: {len(block_table)}; contiguous reads: 1")

# ---------------------------------------------------------------- 7. copy-on-write
log('\n== 7. Copy-on-write: two samples of one 20-token prompt (block size 16)')
ref = {7: 2, 1: 2}                     # physical block -> reference count
tables = {'sample 1': [7, 1], 'sample 2': [7, 1]}
log(f"after prefill: tables {tables}, ref counts {ref}, blocks stored {len(ref)} (without sharing: 4)")
next_free = 3
for s in ('sample 1', 'sample 2'):     # each sample writes token 21 into its last block (block 1, slots 5 to 16 free)
    last = tables[s][-1]
    if ref[last] > 1:
        ref[last] -= 1
        tables[s][-1] = next_free; ref[next_free] = 1
        log(f"{s} writes token 21: block {last} is shared (ref {ref[last] + 1}), copy it to block {next_free}, ref of {last} -> {ref[last]}")
        next_free += 1
    else:
        log(f"{s} writes token 21: block {last} has ref {ref[last]}, write in place")
log(f"now: tables {tables}, ref counts {ref}, blocks stored {len(ref)}")

# ---------------------------------------------------------------- 8. prefix caching: chained hashes
log('\n== 8. Prefix caching: each full block is named by a hash of its tokens and of the block before it')


def block_hashes(tokens):
    hs, parent = [], b''
    for i in range(0, len(tokens) - len(tokens) % B, B):          # only full blocks get a hash
        h = hashlib.sha256(parent + repr(tokens[i:i + B]).encode()).hexdigest()[:8]
        hs.append(h); parent = h.encode()
    return hs


system = list(range(1000, 1032))                                    # a 32-token system prompt = 2 full blocks
p1 = system + [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
p2 = system + [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]
p3 = [999] + system[1:] + p1[32:]                                   # first token differs
cache = set()
for name, p in (('request 1', p1), ('request 2', p2), ('request 3', p3)):
    hs = block_hashes(p)
    hit = 0
    for h in hs:
        if h in cache: hit += 1
        else: break
    cache.update(hs)
    log(f"{name}: {len(p)} tokens, full-block hashes {hs}, reused from cache {hit} blocks = {hit * B} tokens, prefill {len(p) - hit * B} tokens")
