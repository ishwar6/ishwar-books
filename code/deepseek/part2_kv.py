"""Part 2: how much KV cache DeepSeek-V3's MLA stores, and the "absorb" trick, checked in code.

1. Bytes per token for V3's MLA (BF16 and the FP8 serving layout) against MHA, GQA and MQA built with the same
   layer count, and the cache for one 128K-token conversation.
2. MLA from scratch with V3's real shapes (random weights, 128 heads, d_c = 512, RoPE part 64, 64 cached tokens):
   Method 1 rebuilds every head's keys and values from the cached latent ("naive");
   Method 2 folds W^UK into the query and W^UV into the output ("absorb", as in V3's inference/model.py).
   Both must give the same attention output.
"""
import json, os
import torch
from common import Log, save

log = Log('part2_kv')
HERE = os.path.dirname(os.path.abspath(__file__))
c = json.load(open(os.path.join(HERE, 'results/part2_v3_config.json')))
L, nh, dc, dr = c['num_hidden_layers'], c['num_attention_heads'], c['kv_lora_rank'], c['qk_rope_head_dim']
dn, dv = c['qk_nope_head_dim'], c['v_head_dim']
CTX = 128 * 1024

rows = [
    ('MHA, V3 head shapes (K 192, V 128 per head)', nh * ((dn + dr) + dv) * L * 2),
    ('MHA, textbook (2 x 128 heads x 128)', 2 * nh * 128 * L * 2),
    ('GQA, 8 KV groups x 128', 2 * 8 * 128 * L * 2),
    ('MQA, 1 KV head x 128', 2 * 1 * 128 * L * 2),
    ('MLA, BF16 (512 latent + 64 RoPE key)', (dc + dr) * L * 2),
    ('MLA, FP8 layout (512 B + 16 B scales + 128 B RoPE)', (512 + 16 + 128) * L),
]
log(f'== KV cache per token, {L} layers ==')
log(f'  MLA stores d_c + d_R = {dc} + {dr} = {dc + dr} numbers per layer; x {L} layers x 2 bytes = {(dc + dr) * L * 2:,} bytes')
mla = (dc + dr) * L * 2
out = []
for name, b in rows:
    gib = b * CTX / 2 ** 30
    log(f'  {name:52s} {b:>10,} B/token  {b / 1024:8.1f} KiB  {b / mla:6.2f}x MLA   128K tokens: {gib:7.2f} GiB')
    out.append(dict(name=name, bytes=b, gib_128k=gib))

# ---------------------------------------------------------------- MLA naive vs absorbed
log('\n== MLA, naive vs absorbed, V3 shapes, random weights ==')
torch.manual_seed(0)
T, d = 64, c['hidden_size']
h = torch.randn(T, d, dtype=torch.float64)                       # hidden states of 64 earlier tokens
W_dkv = torch.randn(dc, d, dtype=torch.float64) / d ** 0.5       # W^DKV: d -> 512 (the down-projection)
W_kr = torch.randn(dr, d, dtype=torch.float64) / d ** 0.5        # W^KR: d -> 64 (the shared RoPE key)
W_uk = torch.randn(nh, dn, dc, dtype=torch.float64) / dc ** 0.5  # W^UK: 512 -> 128 heads x 128
W_uv = torch.randn(nh, dv, dc, dtype=torch.float64) / dc ** 0.5  # W^UV: 512 -> 128 heads x 128
W_o = torch.randn(d, nh * dv, dtype=torch.float64) / (nh * dv) ** 0.5
q_c = torch.randn(nh, dn, dtype=torch.float64)                   # the new token's query, content part, per head
q_r = torch.randn(nh, dr, dtype=torch.float64)                   # the new token's query, RoPE part, per head
scale = (dn + dr) ** -0.5

c_kv = h @ W_dkv.T                                               # (T, 512)  <- this is what is cached
k_r = h @ W_kr.T                                                 # (T, 64)   <- and this (RoPE already applied in V3)
log(f'  cache: c_kv {tuple(c_kv.shape)} + k_r {tuple(k_r.shape)} = {c_kv.shape[1] + k_r.shape[1]} numbers per token')

# Method 1: naive. Rebuild keys and values of every head for every cached token.
k_c = torch.einsum('hdc,tc->htd', W_uk, c_kv)                    # (128, T, 128)
v_c = torch.einsum('hdc,tc->htd', W_uv, c_kv)                    # (128, T, 128)
scores1 = (torch.einsum('hd,htd->ht', q_c, k_c) + q_r @ k_r.T) * scale
o1 = torch.einsum('ht,htd->hd', scores1.softmax(-1), v_c)        # (128, 128)
u1 = W_o @ o1.reshape(-1)
log(f'  naive  : rebuilt keys + values hold {k_c.numel() + v_c.numel():,} numbers for {T} tokens')

# Method 2: absorbed. Move W^UK onto the query side and W^UV after the weighted sum.
q_lat = torch.einsum('hd,hdc->hc', q_c, W_uk)                    # (128, 512): query seen in latent space
scores2 = (q_lat @ c_kv.T + q_r @ k_r.T) * scale                 # score straight against the cache
x = scores2.softmax(-1) @ c_kv                                   # (128, 512): weighted sum of latents
o2 = torch.einsum('hc,hdc->hd', x, W_uv)                         # apply W^UV once, at the end
u2 = W_o @ o2.reshape(-1)
log(f'  absorbed: reads only the cache, {c_kv.numel() + k_r.numel():,} numbers for {T} tokens')
err_s = (scores1 - scores2).abs().max().item()
err_u = ((u1 - u2).abs().max() / u1.abs().max()).item()
log(f'  largest score difference {err_s:.1e}; largest output difference (relative) {err_u:.1e}')

# The two orders cost different arithmetic per cached token (multiply-adds, one decode step, one layer)
naive_ops = nh * (dn + dv) * dc + nh * (dn + dr + dv)            # rebuild K and V, then score and mix
absorb_ops = nh * (dc + dr + dc)                                 # score in latent space, mix latents
log(f'  multiply-adds per cached token per layer: naive {naive_ops:,}, absorbed {absorb_ops:,}')

save('part2_kv', dict(rows=out, ctx=CTX, err_scores=err_s, err_out=err_u, naive_ops=naive_ops, absorb_ops=absorb_ops))
