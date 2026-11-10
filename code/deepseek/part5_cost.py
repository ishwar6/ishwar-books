"""Part 5: an attention-cost calculator for DeepSeek-V3's real dimensions, dense MLA versus DSA.

Everything here is arithmetic (no GPU, no weights). Inputs are the published config.json values of
DeepSeek-V3 / V3.2 (61 layers, 128 heads, kv_lora_rank 512, rope dim 64, nope dim 128, v dim 128;
V3.2 adds index_n_heads 64, index_head_dim 128, index_topk 2048) and the FlashMLA FP8 cache row
(656 bytes per token per layer) plus the indexer key (128 FP8 bytes + one FP32 scale = 132 bytes).
It also reproduces NSA's Table 4 (keys read per decoding step).

Counts are multiply-adds (MACs) of the attention core only: the score dot products and the weighted
sum of values. Projections and experts are counted separately as about 2 x 37B FLOPs per token.
"""
from common import Log, save

log = Log('part5_cost')

LAYERS, HEADS = 61, 128
D_C, D_R, D_NOPE, D_V = 512, 64, 128, 128
IDX_HEADS, IDX_DIM, TOPK = 64, 128, 2048
ACTIVE = 37e9                                   # active parameters per token (V3 paper)
ROW_MLA_FP8 = 656                               # bytes per token per layer (FlashMLA "FP8 with scale")
ROW_IDX_FP8 = IDX_DIM + 4                       # 128 FP8 bytes + 1 float32 scale
ROW_MLA_BF16 = (D_C + D_R) * 2                  # 1,152 bytes: the plain BF16 latent cache of V3

# Multiply-adds per (query, key) pair, per layer
MAC_MQA = HEADS * ((D_C + D_R) + D_C)           # absorbed form: score over 576 dims, sum over 512 dims
MAC_MHA = HEADS * ((D_NOPE + D_R) + D_V)        # expanded form: 192-dim keys, 128-dim values per head
MAC_IDX = IDX_HEADS * IDX_DIM                   # lightning indexer: 64 heads x 128 dims

CTX = [8192, 16384, 32768, 65536, 131072]


def pairs(L):                                   # causal (query, key) pairs in a prompt of L tokens
    return L * (L + 1) // 2


def pairs_topk(L, k=TOPK):                      # sum over t of min(t, k)
    return pairs(min(L, k)) + max(0, L - k) * k


def g(x):
    return f'{x:,.0f}'


log('Per (query, key) pair, per layer, multiply-adds:')
log(f'  dense MLA, MQA mode (decode):   {HEADS} x ({D_C}+{D_R} + {D_C}) = {g(MAC_MQA)}')
log(f'  dense MLA, MHA mode (prefill):  {HEADS} x ({D_NOPE}+{D_R} + {D_V}) = {g(MAC_MHA)}')
log(f'  lightning indexer (FP8):        {IDX_HEADS} x {IDX_DIM} = {g(MAC_IDX)}')
log(f'  ratio MQA-mode / indexer = {MAC_MQA / MAC_IDX:.1f}x')
log()

out = {'ctx': CTX, 'decode': [], 'prefill': [], 'bytes': [], 'nsa': []}

log('DECODE: one new token at position L, whole model (61 layers), attention core only')
log(f'{"L":>8} {"dense GFLOP":>12} {"DSA idx":>9} {"DSA main":>9} {"DSA tot":>9} {"dense/DSA":>9} {"weights":>8} {"attn/weights":>12}')
for L in CTX:
    dense = 2 * LAYERS * MAC_MQA * L / 1e9
    idx = 2 * LAYERS * MAC_IDX * L / 1e9
    main = 2 * LAYERS * MAC_MQA * min(L, TOPK) / 1e9
    w = 2 * ACTIVE / 1e9
    log(f'{L:>8,} {dense:>12,.1f} {idx:>9,.1f} {main:>9,.1f} {idx + main:>9,.1f} {dense / (idx + main):>8.1f}x {w:>8.0f} {dense / w:>11.1f}x')
    out['decode'].append(dict(L=L, dense=dense, idx=idx, main=main, dsa=idx + main, weights=w))
L_eq = 2 * ACTIVE / (2 * LAYERS * MAC_MQA)
log(f'Dense attention FLOPs equal the 2 x 37B weight FLOPs at L = {L_eq:,.0f} tokens')
log()

log('DECODE: cache bytes read for one new token of one sequence (61 layers)')
log(f'{"L":>8} {"dense FP8 MB":>13} {"dense BF16 MB":>14} {"DSA idx MB":>11} {"DSA main MB":>12} {"DSA tot MB":>11} {"dense/DSA":>9}')
for L in CTX:
    d8 = LAYERS * ROW_MLA_FP8 * L / 1e6
    d16 = LAYERS * ROW_MLA_BF16 * L / 1e6
    i = LAYERS * ROW_IDX_FP8 * L / 1e6
    m = LAYERS * ROW_MLA_FP8 * min(L, TOPK) / 1e6
    log(f'{L:>8,} {d8:>13,.1f} {d16:>14,.1f} {i:>11,.1f} {m:>12,.1f} {i + m:>11,.1f} {d8 / (i + m):>8.1f}x')
    out['bytes'].append(dict(L=L, dense_fp8=d8, dense_bf16=d16, idx=i, main=m, dsa=i + m))
HBM = 4.8e12                                    # H200 memory bandwidth, bytes per second (NVIDIA spec)
log(f'At 128K on one H200 (4.8 TB/s): reading the dense FP8 cache takes {5245.0e6 / HBM * 1e3:.2f} ms per new token, '
    f'the DSA reads {out["bytes"][-1]["dsa"] * 1e6 / HBM * 1e3:.2f} ms (one sequence, before any batching)')
log()

log('PREFILL: a whole prompt of L tokens, whole model, attention core only (TFLOP)')
log(f'{"L":>8} {"dense MHA":>10} {"dense MQA":>10} {"DSA idx":>9} {"DSA main":>9} {"DSA tot":>9} {"MHA/DSA":>8} {"weights":>8}')
for L in CTX:
    mha = 2 * LAYERS * MAC_MHA * pairs(L) / 1e12
    mqa = 2 * LAYERS * MAC_MQA * pairs(L) / 1e12
    idx = 2 * LAYERS * MAC_IDX * pairs(L) / 1e12
    main = 2 * LAYERS * MAC_MQA * pairs_topk(L) / 1e12
    w = 2 * ACTIVE * L / 1e12
    log(f'{L:>8,} {mha:>10,.1f} {mqa:>10,.1f} {idx:>9,.1f} {main:>9,.1f} {idx + main:>9,.1f} {mha / (idx + main):>7.1f}x {w:>8,.1f}')
    out['prefill'].append(dict(L=L, mha=mha, mqa=mqa, idx=idx, main=main, dsa=idx + main, weights=w))
log()

# Where does DSA start to beat dense MHA-mode prefill, per query at position t?
#   MAC_MHA * t  =  MAC_IDX * t + MAC_MQA * TOPK      (for t > TOPK)
t_eq = MAC_MQA * TOPK / (MAC_MHA - MAC_IDX)
t_eq_fp8 = MAC_MQA * TOPK / (MAC_MHA - MAC_IDX / 2)    # if FP8 indexer math runs twice as fast as BF16
log(f'Per-query break-even, dense MHA-mode vs DSA: t = {t_eq:,.0f} tokens '
    f'({t_eq_fp8:,.0f} if the FP8 indexer counts half)')
log(f'Below that, a dense pass is cheaper: why V3.2 runs short prefills in a masked MHA mode.')
log()

log('KV cache per token (61 layers):')
log(f'  V3, BF16 latent:            61 x 576 x 2 B = {g(LAYERS * (D_C + D_R) * 2)} B')
log(f'  V3, FlashMLA FP8 row:       61 x 656 B     = {g(LAYERS * ROW_MLA_FP8)} B')
log(f'  V3.2, FP8 row + indexer:    61 x (656 + 132) B = {g(LAYERS * (ROW_MLA_FP8 + ROW_IDX_FP8))} B '
    f'(+{100 * ROW_IDX_FP8 / ROW_MLA_FP8:.1f}%)')
log(f'  at 128K tokens: V3 FP8 {LAYERS * ROW_MLA_FP8 * 131072 / 1e9:.2f} GB, '
    f'V3.2 {LAYERS * (ROW_MLA_FP8 + ROW_IDX_FP8) * 131072 / 1e9:.2f} GB per sequence')
out['kv'] = dict(v3_bf16=LAYERS * (D_C + D_R) * 2, v3_fp8=LAYERS * ROW_MLA_FP8, v32=LAYERS * (ROW_MLA_FP8 + ROW_IDX_FP8))
log()

log('NSA (arXiv 2502.11089) keys read per decoding step: t/d + n*l\' + w with l=32, d=16, n=16, l\'=64, w=512')
for t in [8192, 16384, 32768, 65536]:
    n_tok = t // 16 + 16 * 64 + 512
    exact = (t - 32) // 16 + 1 + 16 * 64 + 512
    log(f'  t = {t:>6,}: {n_tok:>5,} tokens (exact count {exact:,}), full attention reads {t:,}: {t / n_tok:.1f}x fewer')
    out['nsa'].append(dict(t=t, nsa=n_tok, ratio=t / n_tok))

save('part5_cost', out)
