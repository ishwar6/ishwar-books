"""Part 6 experiment 3: a toy Engram, the hashed n-gram memory of DeepSeek-V4.1-Flash (CPU, seconds).

Engram (arXiv 2601.07372) looks up vectors in big tables. The address of a row is a hash of the last 2, 3 (and 4)
token IDs, so finding the row needs no search and no matrix product. This script shows, on a tiny scale:

  1. tokenizer compression: "The" and "the" get one canonical ID;
  2. the multiplicative-XOR hash of the official demo (github.com/deepseek-ai/Engram, engram_demo_v1.py):
     mix = (t0 * a0) XOR (t1 * a1) XOR ...,  row = mix mod p  (a_k random odd numbers, p a prime per head);
  3. why several hash heads: two n-grams can share a row in one table, but rarely in all heads at once;
  4. the context gate: alpha = sigmoid(RMSNorm(h) . RMSNorm(k) / sqrt(d));
  5. a lookup costs about the same time whether the table has a thousand rows or millions.
Run:  ~/.venvs/attn/bin/python part6_engram_toy.py
"""
import time
import numpy as np
import torch
from common import Log, save

log = Log('part6_engram_toy')
rng = np.random.default_rng(0)

# ---- 1. tokenizer compression ---------------------------------------------------------------------------------
text = 'Only Alexander the Great could tame the horse Bucephalus . The horse obeyed Alexander the Great'.split()
raw_vocab = {w: i for i, w in enumerate(dict.fromkeys(text))}          # case-sensitive IDs, like a real tokenizer
canon = {w: i for i, w in enumerate(dict.fromkeys(w.lower() for w in text))}
raw_ids = [raw_vocab[w] for w in text]
ids = [canon[w.lower()] for w in text]                                  # the surjective map P: V -> V'
log('1) tokenizer compression')
log(f'   raw IDs      : {raw_ids}   ({len(raw_vocab)} distinct)')
log(f'   canonical IDs: {ids}   ({len(canon)} distinct; "The" and "the" now share one ID)')

# ---- 2. multiplicative-XOR hash, 2 heads per n-gram order ------------------------------------------------------
ORDERS, HEADS, DIM = (2, 3), 2, 4
mult = rng.integers(0, 2**20, size=max(ORDERS)) * 2 + 1               # random odd multipliers, one per offset
PRIMES = {2: [1009, 1013], 3: [1019, 1021]}                             # a distinct prime table size per head
tables = {(n, k): rng.standard_normal((PRIMES[n][k], DIM)).astype(np.float32) for n in ORDERS for k in range(HEADS)}


def rows_for(seq, t, n):
    """The suffix n-gram ending at position t -> one row number per hash head."""
    gram = seq[max(0, t - n + 1): t + 1]
    gram = [0] * (n - len(gram)) + gram                                  # pad at the start of the text
    mix = 0
    for k, tok in enumerate(reversed(gram)):                             # offset 0 = current token
        mix ^= int(tok) * int(mult[k])
    return [mix % p for p in PRIMES[n]], gram


log('\n2) lookups for the last tokens of "... obeyed Alexander the Great"')
t = len(ids) - 1
for n in ORDERS:
    r, gram = rows_for(ids, t, n)
    words = ' '.join(text[t - n + 1: t + 1])
    log(f'   {n}-gram "{words}" -> canonical IDs {gram} -> rows {r} in tables of size {PRIMES[n]}')
e = np.concatenate([tables[(n, k)][rows_for(ids, t, n)[0][k]] for n in ORDERS for k in range(HEADS)])
log(f'   memory vector e_t = concatenation of {len(ORDERS) * HEADS} rows x {DIM} = {e.size} numbers')
r1 = rows_for(ids, 3, 3)[0]
r2 = rows_for(ids, len(ids) - 1, 3)[0]
log(f'   "Alexander the Great" at position 3 and at the end give the same rows: {r1} and {r2}: same memory, both times')

# ---- 3. collisions: one head vs two heads ----------------------------------------------------------------------
N, P1, P2 = 50_000, 65_537, 65_539
grams = rng.integers(0, 100_000, size=(N, 2))
grams = np.unique(grams, axis=0)
mix = (grams[:, 0] * int(mult[0])) ^ (grams[:, 1] * int(mult[1]))
h1, h2 = mix % P1, mix % P2


def collided(h):
    _, inv, cnt = np.unique(h, return_inverse=True, return_counts=True)
    return cnt[inv] > 1


c1, c2 = collided(h1), collided(h2)
both = collided(h1.astype(np.int64) * P2 + h2)
log(f'\n3) {len(grams):,} distinct 2-grams hashed into tables of {P1:,} and {P2:,} rows')
log(f'   share a row with another 2-gram in head 1: {c1.mean() * 100:.1f}%   in head 2: {c2.mean() * 100:.1f}%')
log(f'   share BOTH rows with the same other 2-gram:  {both.mean() * 100:.3f}%')

# ---- 4. the context gate --------------------------------------------------------------------------------------
d = 16


def rms(v):
    return v / np.sqrt((v ** 2).mean())


k = rng.standard_normal(d)
log(f'\n4) gate alpha = sigmoid(RMSNorm(h) . RMSNorm(k) / sqrt({d}))')
for label, h in [('hidden state agrees with the memory (h = k)', k), ('unrelated hidden state (random h)', rng.standard_normal(d)),
                 ('hidden state contradicts the memory (h = -k)', -k)]:
    a = 1 / (1 + np.exp(-(rms(h) @ rms(k)) / np.sqrt(d)))
    log(f'   {label:46s}: alpha = {a:.3f}')

# ---- 5. lookup time does not grow with the table ---------------------------------------------------------------
torch.manual_seed(0)
B, D = 8192, 64
log(f'\n5) gather {B:,} rows of {D} FP16 numbers (CPU, best of 20 runs)')
times = {}
for rows in (1_000, 100_000, 4_000_000):
    table = torch.randn(rows, D, dtype=torch.float16)
    idx = torch.randint(0, rows, (B,))
    best = 1e9
    for _ in range(20):
        t0 = time.perf_counter(); out = table[idx]; best = min(best, time.perf_counter() - t0)
    times[rows] = best
    log(f'   table of {rows:>9,} rows ({rows * D * 2 / 1e6:7.1f} MB): {best * 1e3:.3f} ms')
    del table

log('\n   For contrast, a dense layer touches every weight: 2 FLOPs per weight per token.')
log(f'   196.6B dense weights would cost {2 * 196.6e9 / 1e9:,.0f} GFLOPs per token; Engram reads 48 rows x 256 = 12,288 numbers.')
save('part6_engram_toy', dict(collide_1head=float(c1.mean()), collide_2heads=float(both.mean()),
                              lookup_ms={str(k): v * 1e3 for k, v in times.items()}))
