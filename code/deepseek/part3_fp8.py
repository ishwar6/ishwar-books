"""Part 3 experiment: why DeepSeek-V3 scales FP8 numbers in small groups, and why it adds up in FP32 every 128 terms.

What it shows (CPU only, a few seconds):
  A. What FP8 E4M3 can hold: its largest number, its step size, and what happens to tiny and huge numbers.
  B. One activation matrix (128 tokens x 1,024 channels, random normal) with ONE large outlier.
     We quantize it to FP8 three ways: one scale for the whole tensor, one scale per 1x128 tile (V3's choice
     for activations), one scale per 128x128 block (V3's choice for weights). We report the error of the
     ordinary numbers and how many of them become exactly zero.
  C. A matrix whose rows (tokens) differ in size by up to 10**S times (S = 0 to 8), like activation gradients, quantized
     the same three ways, alone and inside a matrix product with a weight matrix.
  D. A SIMULATION of limited accumulation precision: 4,096-term dot products added up in an accumulator
     that keeps only 14 significant bits (our stand-in for the paper's "around 14 bits"), with and without
     moving the partial sum to an FP32 total every 128 terms. This models the idea; it is not the H800 circuit.
Run: ~/.venvs/attn/bin/python part3_fp8.py   ->  results/part3_fp8_stdout.txt, results/part3_fp8.json
"""
import torch
from common import Log, save

torch.manual_seed(0)
log = Log('part3_fp8')
F8 = torch.float8_e4m3fn
FMAX = torch.finfo(F8).max                       # 448.0, the largest E4M3 number


def to_fp8_and_back(x, scale):
    """Quantize: multiply by the scale, clamp to the FP8 range, round to FP8. Dequantize: divide by the scale."""
    q = (x * scale).clamp(-FMAX, FMAX).to(F8)    # stored as 1 byte per number
    return q.float() / scale


def per_tensor(x):
    scale = FMAX / x.abs().max()                 # one scale: the largest |x| maps to 448
    return to_fp8_and_back(x, scale)


def per_tile(x, n=128):
    """One scale per row-piece of n channels (1 x n tiles): per token, per 128 channels."""
    r, c = x.shape
    t = x.reshape(r, c // n, n)
    scale = FMAX / t.abs().amax(dim=2, keepdim=True).clamp(min=1e-12)
    return to_fp8_and_back(t, scale).reshape(r, c)


def per_block(x, n=128):
    """One scale per n x n block."""
    r, c = x.shape
    t = x.reshape(r // n, n, c // n, n)
    scale = FMAX / t.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    return to_fp8_and_back(t, scale).reshape(r, c)


def rel_err(a, ref):
    return ((a - ref).norm() / ref.norm()).item()


# ---------------- A. the format ----------------
log('A. FP8 E4M3 (4 exponent bits, 3 mantissa bits)')
log(f'   largest number: {FMAX:g}   smallest normal: {torch.finfo(F8).tiny:g}   smallest above zero: {2**-9:g}')
for v in [1.0, 1.1, 0.3, 300.0, 0.004, 0.0001]:
    log(f'   {v:>8g} -> stored as {torch.tensor(v).to(F8).float().item():g}')
log('   between 1 and 2 there are only 8 steps: ' + ', '.join(f'{1 + k / 8:g}' for k in range(8)) + ', 2')

# worked example: one small group of four numbers, with and without an outlier in it
log('')
log('A2. Worked example: scale, round, scale back')
for group in ([0.5, -1.2, 2.0, 0.01], [0.5, -1.2, 1000.0, 0.01]):
    g = torch.tensor(group)
    sc = FMAX / g.abs().max()
    back = to_fp8_and_back(g, sc)
    log(f'   group {group}: scale = 448 / {g.abs().max().item():g} = {sc.item():.4g}')
    log(f'      0.01 x scale = {0.01 * sc.item():.5g} -> FP8 {(0.01 * sc).to(F8).float().item():.5g} -> back {back[3].item():.5g}'
        f'  (error {abs(back[3].item() - 0.01) / 0.01:.1%})')

# ---------------- B. one outlier, three ways to scale ----------------
X = torch.randn(128, 1024)
log('')
log('B. Activation matrix 128 x 1024, random normal (typical |x| about 0.8), one outlier at row 5, column 300')
log(f'   {"outlier":>8} | {"error of the ordinary numbers":^40} | {"ordinary numbers turned into 0":^34}')
log(f'   {"":>8} | {"tensor":>12} {"1x128 tile":>13} {"128x128 blk":>13} | {"tensor":>10} {"tile":>10} {"block":>10}')
rows = []
normal = torch.ones_like(X, dtype=torch.bool)
normal[5, 300] = False
for big in [0, 100, 1000, 10000, 100000]:
    Xo = X.clone()
    if big:
        Xo[5, 300] = big
    res = {}
    for name, f in [('tensor', per_tensor), ('tile', per_tile), ('block', per_block)]:
        d = f(Xo)
        res[name] = (rel_err(d[normal], Xo[normal]), (d[normal] == 0).float().mean().item() * 100)
    rows.append({'outlier': big, **{k: {'err': v[0], 'zero_pct': v[1]} for k, v in res.items()}})
    log(f'   {big:>8,} | {res["tensor"][0]:>11.2%} {res["tile"][0]:>13.2%} {res["block"][0]:>13.2%} | '
        f'{res["tensor"][1]:>9.2f}% {res["tile"][1]:>9.2f}% {res["block"][1]:>9.2f}%')
bf = rel_err(X.to(torch.bfloat16).float(), X)
log(f'   for comparison: BF16 (7 stored mantissa bits) gives {bf:.2%} on the same numbers')

# ---------------- C. rows of very different sizes ----------------
log('')
log('C. Rows (tokens) of very different sizes: row i is multiplied by 10**u, u uniform in [-S, 0]')
log('   (activation gradients look like this: some tokens carry far larger values than others)')
log('   error = mean over rows of each row\'s relative error, so small rows count as much as large ones')


def row_err(d, ref):
    return ((d - ref).norm(dim=1) / ref.norm(dim=1)).mean().item()


W = torch.randn(1024, 1024) * 0.02
Wq = per_block(W).double()                                    # W always in 128x128 blocks, as in V3
log(f'   {"S":>3} | {"error of G":^36} | {"error of G W^T":^36}')
log(f'   {"":>3} | {"tensor":>10} {"tile":>12} {"block":>12} | {"tensor":>10} {"tile":>12} {"block":>12}')
imb = []
for S in [0, 2, 4, 6, 8]:
    G = torch.randn(128, 1024) * 10 ** (-S * torch.rand(128, 1))
    Y = G.double() @ W.double().T
    r = {'S': S}
    for name, f in [('tensor', per_tensor), ('tile', per_tile), ('block', per_block)]:
        d = f(G)
        r[name] = {'err': row_err(d, G), 'mm': row_err(d.double() @ Wq.T, Y)}
    imb.append(r)
    log(f'   {S:>3} | {r["tensor"]["err"]:>10.2%} {r["tile"]["err"]:>12.2%} {r["block"]["err"]:>12.2%} | '
        f'{r["tensor"]["mm"]:>10.2%} {r["tile"]["mm"]:>12.2%} {r["block"]["mm"]:>12.2%}')

# ---------------- D. accumulation (simulation) ----------------
log('')
log('D. SIMULATION: 4,096 dot products of length K = 4,096 (FP8-rounded random inputs)')


def keep_bits(x, bits=14):
    """Keep `bits` significant bits of each number by cutting off the rest (round toward zero)."""
    m, e = torch.frexp(x)                         # x = m * 2**e, 0.5 <= |m| < 1
    return torch.ldexp(torch.trunc(m * 2 ** bits) / 2 ** bits, e)


N, K = 4096, 4096
a = torch.randn(N, K).to(F8).double()
b = torch.randn(N, K).to(F8).double()
prod = a * b                                      # each product is exact in float64
exact = prod.sum(1)
low = torch.zeros(N, dtype=torch.float64)
part = torch.zeros(N, dtype=torch.float64)
total = torch.zeros(N, dtype=torch.float64)
for k in range(K):
    low = keep_bits(low + prod[:, k])                 # (1) everything in the narrow accumulator
    part = keep_bits(part + prod[:, k])               # (2) narrow accumulator for 128 terms...
    if (k + 1) % 128 == 0:
        total = (total + part).float().double()       # ...then moved into an FP32 total
        part.zero_()
scale_ref = prod.abs().sum(1)                         # error relative to the size of the terms
acc = {}
for name, v in [('narrow accumulator for all 4,096 terms', low), ('narrow for 128 terms, then FP32', total)]:
    e = (v - exact).abs() / scale_ref
    acc[name] = {'max': e.max().item(), 'mean': e.mean().item()}
    log(f'   {name:<40} max error {e.max().item():.4%}   mean error {e.mean().item():.4%}')
log('   (error = |result - exact sum| / sum of |terms|, so it does not blow up when the sum is near zero)')

save('part3_fp8', {'outlier_sweep': rows, 'bf16_err': bf, 'imbalanced': imb, 'accumulation': acc, 'K': K})
