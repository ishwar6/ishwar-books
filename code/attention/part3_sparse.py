"""Part 3: sliding-window attention, rolling KV cache, receptive field, local:global KV memory, decode timing.
Every number in the article's sliding-window sections comes from this script (writes results/part3.json)."""
import json, math, time
import torch
import torch.nn.functional as F

torch.manual_seed(0)
OUT = {}


def window_mask(T, W, device=None):
    """True where query i may look at key j: j <= i (causal) and i - j < W (inside the window)."""
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    return (j <= i) & (i - j < W)


def attend(q, k, v, allowed):
    """Plain attention with a boolean 'allowed' mask. q, k, v: (..., T, d)."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~allowed, float('-inf'))
    return torch.softmax(scores, -1) @ v


def rope(x, pos, base=10000.0):
    """Rotary position embedding on the last dimension (rotate-half form)."""
    d = x.shape[-1]
    inv = base ** (-torch.arange(0, d, 2, dtype=x.dtype) / d)
    ang = pos[:, None].to(x.dtype) * inv[None, :]
    cos, sin = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


# 1. Sliding-window attention: basic checks ------------------------------------------------------
T, d, W = 24, 16, 6
x = torch.randn(T, d, dtype=torch.float64)
Wq, Wk, Wv = (torch.randn(d, d, dtype=torch.float64) / math.sqrt(d) for _ in range(3))
q, k, v = x @ Wq, x @ Wk, x @ Wv

swa = attend(q, k, v, window_mask(T, W))
ref = F.scaled_dot_product_attention(q[None], k[None], v[None], attn_mask=window_mask(T, W)[None])[0]
full_causal = attend(q, k, v, window_mask(T, T))
wide = attend(q, k, v, window_mask(T, T + 5))           # window larger than the text = ordinary causal attention

x2 = x.clone()
x2[: T - W] = torch.randn(T - W, d, dtype=torch.float64)  # change every token outside the last token's window
q2, k2, v2 = x2 @ Wq, x2 @ Wk, x2 @ Wv
swa2 = attend(q2, k2, v2, window_mask(T, W))
full2 = attend(q2, k2, v2, window_mask(T, T))

OUT['swa_checks'] = {
    'T': T, 'W': W,
    'vs_sdpa': (swa - ref).abs().max().item(),
    'wide_window_vs_causal': (wide - full_causal).abs().max().item(),
    'last_token_change_swa': (swa2[-1] - swa[-1]).abs().max().item(),
    'last_token_change_full': (full2[-1] - full_causal[-1]).abs().max().item(),
    'mask_example': window_mask(10, 4).int().tolist(),
}
print('1. SWA vs PyTorch SDPA with the same mask:  max |diff| = %.1e' % OUT['swa_checks']['vs_sdpa'])
print('   window wider than the text vs causal:   max |diff| = %.1e' % OUT['swa_checks']['wide_window_vs_causal'])
print('   change all %d tokens outside the window of the last token:' % (T - W))
print('      last output with sliding window moves by %.1e' % OUT['swa_checks']['last_token_change_swa'])
print('      last output with full attention moves by %.2f' % OUT['swa_checks']['last_token_change_full'])

# 2. Receptive field: stacked layers see further ---------------------------------------------------
T, d, W = 40, 16, 4
rf = {}
for L in range(1, 7):
    layers = [[torch.randn(d, d, dtype=torch.float64) / math.sqrt(d) for _ in range(3)] for _ in range(L)]
    x = torch.randn(T, d, dtype=torch.float64, requires_grad=True)
    h = x
    for Wq_, Wk_, Wv_ in layers:
        h = h + attend(h @ Wq_, h @ Wk_, h @ Wv_, window_mask(T, W))   # residual connection, like a real model
    h[-1].sum().backward()
    reach = (x.grad.abs().sum(-1) > 0).nonzero().flatten().tolist()
    rf[L] = {'tokens_reached': len(reach), 'farthest_back': T - 1 - min(reach), 'formula': L * (W - 1) + 1}
OUT['receptive_field'] = {'W': W, 'T': T, 'by_layers': rf}
print('2. Receptive field of the last token, window W = %d' % W)
for L, r in rf.items():
    print('   %d layer(s): depends on %2d tokens (formula L*(W-1)+1 = %2d), reaches %2d tokens back'
          % (L, r['tokens_reached'], r['formula'], r['farthest_back']))

# 3. Rolling buffer KV cache: decode token by token with a cache of only W slots --------------------
T, d, W, H = 50, 16, 8, 2
x = torch.randn(T, H * d, dtype=torch.float64)
Wq, Wk, Wv = (torch.randn(H * d, H * d, dtype=torch.float64) / math.sqrt(H * d) for _ in range(3))
pos = torch.arange(T)
heads = lambda t: t.view(T, H, d).transpose(0, 1)                      # (H, T, d)
Q = rope(heads(x @ Wq), pos); K = rope(heads(x @ Wk), pos); V = heads(x @ Wv)
batch_out = attend(Q, K, V, window_mask(T, W))                          # all tokens at once

k_cache = torch.zeros(H, W, d, dtype=torch.float64)                     # the whole cache: W slots, never more
v_cache = torch.zeros(H, W, d, dtype=torch.float64)
step_out = []
for i in range(T):
    xi = x[i:i + 1]
    qi = rope((xi @ Wq).view(1, H, d).transpose(0, 1), pos[i:i + 1])
    ki = rope((xi @ Wk).view(1, H, d).transpose(0, 1), pos[i:i + 1])
    vi = (xi @ Wv).view(1, H, d).transpose(0, 1)
    slot = i % W                                                        # position i goes to slot i mod W
    k_cache[:, slot], v_cache[:, slot] = ki[:, 0], vi[:, 0]
    n = min(i + 1, W)                                                   # how many slots are filled
    step_out.append(attend(qi, k_cache[:, :n], v_cache[:, :n], torch.ones(1, n, dtype=torch.bool)))
step_out = torch.cat(step_out, 1)
OUT['rolling_cache'] = {'T': T, 'W': W, 'max_diff': (step_out - batch_out).abs().max().item(),
                        'slots_used': W, 'slots_full_cache': T}
print('3. Rolling buffer cache (W = %d slots) vs computing all %d tokens at once: max |diff| = %.1e'
      % (W, T, OUT['rolling_cache']['max_diff']))

# 4. KV cache memory: real configurations ------------------------------------------------------------
def kv_bytes(n, layers, kv_heads, head_dim, window=None, bytes_per=2):
    keep = n if window is None else min(n, window)
    return 2 * layers * kv_heads * head_dim * keep * bytes_per

ctx = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
g = dict(layers=62, kv_heads=16, head_dim=128, window=1024, pattern=6)
n_global = sum(1 for i in range(g['layers']) if (i + 1) % g['pattern'] == 0)
n_local = g['layers'] - n_global
gemma = {'n_global': n_global, 'n_local': n_local, 'context': ctx,
         'all_global_gib': [kv_bytes(n, 62, 16, 128) / 2**30 for n in ctx],
         'five_to_one_gib': [(kv_bytes(n, n_global, 16, 128) + kv_bytes(n, n_local, 16, 128, 1024)) / 2**30 for n in ctx]}
mistral_full = kv_bytes(32768, 32, 8, 128) / 2**30
mistral_swa = kv_bytes(32768, 32, 8, 128, 4096) / 2**30
oss_full = kv_bytes(131072, 24, 8, 64) / 2**30
oss_mix = (kv_bytes(131072, 12, 8, 64) + kv_bytes(131072, 12, 8, 64, 128)) / 2**30
OUT['kv_memory'] = {'gemma3_27b': gemma,
                    'mistral_7b_32k': {'full_gib': mistral_full, 'swa_gib': mistral_swa},
                    'gpt_oss_20b_128k': {'full_gib': oss_full, 'alternating_gib': oss_mix}}
print('4. KV cache for one sequence, 16-bit numbers')
print('   Gemma 3 27B (%d local + %d global layers):' % (n_local, n_global))
for n, a, b in zip(ctx, gemma['all_global_gib'], gemma['five_to_one_gib']):
    print('      %7d tokens: all global %6.2f GiB   5 local : 1 global %5.2f GiB   (%.1fx smaller)' % (n, a, b, a / b))
print('   Mistral 7B at 32K: full %.2f GiB, window 4096 %.2f GiB (%.0fx smaller)' % (mistral_full, mistral_swa, mistral_full / mistral_swa))
print('   gpt-oss-20b at 128K: full %.2f GiB, alternating window 128 %.2f GiB' % (oss_full, oss_mix))

# 5. Decode step time: full cache vs sliding window --------------------------------------------------
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
Hh, D, Wd = 16, 128, 1024
timing = {}
def bench(n, iters=40):
    q = torch.randn(1, Hh, 1, D, device=dev, dtype=torch.bfloat16)
    kk = torch.randn(1, Hh, n, D, device=dev, dtype=torch.bfloat16)
    vv = torch.randn(1, Hh, n, D, device=dev, dtype=torch.bfloat16)
    for _ in range(5):
        F.scaled_dot_product_attention(q, kk, vv)
    torch.mps.synchronize() if dev == 'mps' else None
    t0 = time.perf_counter()
    for _ in range(iters):
        F.scaled_dot_product_attention(q, kk, vv)
    torch.mps.synchronize() if dev == 'mps' else None
    return (time.perf_counter() - t0) / iters * 1000
for n in [1024, 4096, 16384, 65536, 131072]:
    timing[n] = {'full_ms': min(bench(n) for _ in range(3)), 'window_ms': min(bench(min(n, Wd)) for _ in range(3))}
OUT['decode_timing'] = {'device': dev, 'heads': Hh, 'head_dim': D, 'window': Wd, 'results': timing}
print('5. One decode step of attention (%d heads, d = %d, %s, bf16)' % (Hh, D, dev))
for n, r in timing.items():
    print('   %6d tokens so far: full %.3f ms, window %d: %.3f ms' % (n, r['full_ms'], Wd, r['window_ms']))

json.dump(OUT, open('results/part3.json', 'w'), indent=1)
