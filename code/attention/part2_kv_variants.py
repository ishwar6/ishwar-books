"""Attention from the ground up, Part 2: MQA, GQA and MLA.

  1. GQA from scratch: MHA and MQA are its two extremes. Checked against PyTorch.
  2. A real model: recompute layer 0 attention of Qwen2.5-0.5B (14 query heads, 2 KV heads, RoPE) by hand.
  3. Decode speed: one attention step over a long KV cache with 32, 8 and 1 KV heads.
  4. MLA from scratch: the compressed-cache path equals the full path; applying RoPE naively breaks it.
  5. KV cache per token for real models.
Run: python part2_kv_variants.py   (writes results/part2.json)
"""
import json, math, statistics, time
import torch
import torch.nn.functional as F

torch.manual_seed(0)
OUT = {}


# ---------------------------------------------------------------- 1. GQA (with MHA and MQA as special cases)
def gqa(q, k, v, causal=True):
    """q: (B, Hq, T, d); k, v: (B, Hkv, T, d) with Hq a multiple of Hkv.
    Each group of Hq/Hkv query heads shares one key/value head. No copy of K or V is made."""
    B, Hq, T, d = q.shape
    Hkv = k.shape[1]
    g = Hq // Hkv
    qg = q.view(B, Hkv, g, T, d)                                       # group the query heads
    scores = torch.einsum('bhgqd,bhkd->bhgqk', qg, k) / math.sqrt(d)
    if causal:
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), 1)
        scores = scores.masked_fill(mask, float('-inf'))
    out = torch.einsum('bhgqk,bhkd->bhgqd', torch.softmax(scores, -1), v)
    return out.reshape(B, Hq, T, d)


B, T, Hq, d = 2, 12, 8, 16
q = torch.randn(B, Hq, T, d, dtype=torch.float64)
checks = {}
for Hkv, name in ((8, 'MHA (8 KV heads)'), (2, 'GQA (2 KV heads)'), (1, 'MQA (1 KV head)')):
    k, v = torch.randn(B, Hkv, T, d, dtype=torch.float64), torch.randn(B, Hkv, T, d, dtype=torch.float64)
    ours = gqa(q, k, v)
    # reference: give every query head its own copy of its group's K and V, then plain attention
    rep = Hq // Hkv
    ref = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), is_causal=True)
    checks[name] = (ours - ref).abs().max().item()
    print(f'1. {name:18s} vs PyTorch with repeated K/V: max |diff| = {checks[name]:.1e}')
OUT['gqa_checks'] = checks


# ---------------------------------------------------------------- 2. a real GQA model, recomputed by hand
from transformers import AutoModelForCausalLM, AutoTokenizer
name = 'Qwen/Qwen2.5-0.5B'
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, attn_implementation='eager').eval()
cfg = model.config
attn = model.model.layers[0].self_attn
captured = {}
def hook(mod, args, kwargs, output):
    captured['x'] = kwargs['hidden_states']
    captured['cos_sin'] = kwargs['position_embeddings']
    captured['out'] = output[0]
handle = attn.register_forward_hook(hook, with_kwargs=True)
ids = tok('Grouped-query attention lets several query heads share one key and value head.', return_tensors='pt').input_ids
with torch.no_grad():
    model(ids)
handle.remove()

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

with torch.no_grad():
    x = captured['x']                                                  # (1, T, 896): input to layer 0's attention
    cos, sin = (t.unsqueeze(1) for t in captured['cos_sin'])           # rotary angles for each position
    Tn, hd = x.shape[1], cfg.hidden_size // cfg.num_attention_heads
    split = lambda t, h: t.view(1, Tn, h, hd).transpose(1, 2)
    qh = split(attn.q_proj(x), cfg.num_attention_heads)               # 14 query heads
    kh = split(attn.k_proj(x), cfg.num_key_value_heads)               # only 2 key heads
    vh = split(attn.v_proj(x), cfg.num_key_value_heads)               # only 2 value heads
    qh, kh = qh * cos + rotate_half(qh) * sin, kh * cos + rotate_half(kh) * sin   # RoPE
    heads = gqa(qh, kh, vh)
    mine = attn.o_proj(heads.transpose(1, 2).reshape(1, Tn, -1))
OUT['qwen_layer0'] = {
    'query_heads': cfg.num_attention_heads, 'kv_heads': cfg.num_key_value_heads, 'head_dim': hd, 'tokens': Tn,
    'max_abs_diff': (mine - captured['out']).abs().max().item(), 'output_scale': captured['out'].abs().max().item(),
}
print(f"2. Qwen2.5-0.5B layer 0 ({cfg.num_attention_heads} query heads share {cfg.num_key_value_heads} KV heads): "
      f"our GQA vs the model's own output, max |diff| = {OUT['qwen_layer0']['max_abs_diff']:.1e} "
      f"(outputs are up to {OUT['qwen_layer0']['output_scale']:.1f})")
del model


# ---------------------------------------------------------------- 3. decode speed: reading the KV cache
dev = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
sync = (torch.mps.synchronize if dev == 'mps' else torch.cuda.synchronize if dev == 'cuda' else (lambda: None))
Bd, ctx, Hq, d = 8, 4096, 32, 128
speed = {}
for Hkv in (32, 8, 1):
    qd = torch.randn(Bd, Hq, 1, d, device=dev, dtype=torch.bfloat16)          # one new token per sequence
    kc = torch.randn(Bd, Hkv, ctx, d, device=dev, dtype=torch.bfloat16)        # the cache it must read
    vc = torch.randn(Bd, Hkv, ctx, d, device=dev, dtype=torch.bfloat16)
    step = lambda: gqa(qd, kc, vc, causal=False)
    for _ in range(3):
        step(); sync()
    ts = []
    for _ in range(20):
        sync(); t0 = time.perf_counter(); step(); sync(); ts.append(time.perf_counter() - t0)
    ms = statistics.median(ts) * 1000
    mb = 2 * kc.numel() * kc.element_size() / 1e6
    speed[Hkv] = {'ms': ms, 'kv_mb': mb}
    print(f'3. {Hkv:2d} KV heads: cache read per step {mb:7.1f} MB, attention step {ms:6.2f} ms')
OUT['decode_speed'] = {'device': dev, 'batch': Bd, 'context': ctx, 'query_heads': Hq, 'head_dim': d, 'results': speed}


# ---------------------------------------------------------------- 4. MLA from scratch
def rope(x, pos, base=10000.0):
    """Rotary position embedding on the last dimension (rotate-half form)."""
    dim = x.shape[-1]
    inv = 1.0 / base ** (torch.arange(0, dim, 2, dtype=x.dtype) / dim)
    ang = pos[:, None].to(x.dtype) * inv[None]
    cos, sin = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)
    return x * cos + rotate_half(x) * sin

class MLA(torch.nn.Module):
    """Multi-head latent attention (DeepSeek-V2 style, without query compression).
    Keys and values are rebuilt from a small shared latent c (d_c numbers per token);
    position information travels in a separate, shared RoPE key of d_r numbers."""
    def __init__(self, d_model=256, h=8, d_nope=32, d_rope=16, d_v=32, d_c=64):
        super().__init__()
        self.h, self.dn, self.dr, self.dv, self.dc = h, d_nope, d_rope, d_v, d_c
        L = lambda i, o: torch.nn.Linear(i, o, bias=False)
        self.W_dkv = L(d_model, d_c)                 # down-projection: token -> latent (this is what gets cached)
        self.W_kr = L(d_model, d_rope)               # shared RoPE key (also cached)
        self.W_uk = L(d_c, h * d_nope)               # latent -> per-head keys
        self.W_uv = L(d_c, h * d_v)                  # latent -> per-head values
        self.W_q = L(d_model, h * d_nope)
        self.W_qr = L(d_model, h * d_rope)
        self.W_o = L(h * d_v, d_model)

    def naive(self, x, rope_on_latent_keys=False):
        """Rebuild full per-head K and V (what training does)."""
        T = x.shape[0]; pos = torch.arange(T)
        c, kr = self.W_dkv(x), rope(self.W_kr(x), pos)
        k_nope = self.W_uk(c).view(T, self.h, self.dn)
        if rope_on_latent_keys:                      # the tempting mistake: rotate the reconstructed keys too
            k_nope = torch.stack([rope(k_nope[:, i], pos) for i in range(self.h)], 1)
        v = self.W_uv(c).view(T, self.h, self.dv)
        q_nope = self.W_q(x).view(T, self.h, self.dn)
        if rope_on_latent_keys:
            q_nope = torch.stack([rope(q_nope[:, i], pos) for i in range(self.h)], 1)
        q_rope = torch.stack([rope(t, pos) for t in self.W_qr(x).view(T, self.h, self.dr).unbind(1)], 1)
        scores = (torch.einsum('qhd,khd->hqk', q_nope, k_nope) + torch.einsum('qhd,kd->hqk', q_rope, kr)) / math.sqrt(self.dn + self.dr)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        A = torch.softmax(scores.masked_fill(mask, float('-inf')), -1)
        return self.W_o(torch.einsum('hqk,khd->qhd', A, v).reshape(T, -1))

    def absorbed(self, x):
        """Inference path: cache only c (d_c) and the RoPE key (d_r) per token; never rebuild K or V."""
        T = x.shape[0]; pos = torch.arange(T)
        c, kr = self.W_dkv(x), rope(self.W_kr(x), pos)                     # <- the entire KV cache
        W_uk = self.W_uk.weight.view(self.h, self.dn, self.dc)
        W_uv = self.W_uv.weight.view(self.h, self.dv, self.dc)
        q_nope = self.W_q(x).view(T, self.h, self.dn)
        q_lat = torch.einsum('qhd,hdc->qhc', q_nope, W_uk)                  # absorb W_uk into the query
        q_rope = torch.stack([rope(t, pos) for t in self.W_qr(x).view(T, self.h, self.dr).unbind(1)], 1)
        scores = (torch.einsum('qhc,kc->hqk', q_lat, c) + torch.einsum('qhd,kd->hqk', q_rope, kr)) / math.sqrt(self.dn + self.dr)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        A = torch.softmax(scores.masked_fill(mask, float('-inf')), -1)
        o_lat = torch.einsum('hqk,kc->qhc', A, c)                            # attend in latent space
        o = torch.einsum('qhc,hvc->qhv', o_lat, W_uv)                        # then up-project once
        return self.W_o(o.reshape(T, -1))

mla = MLA().double()
xm = torch.randn(20, 256, dtype=torch.float64)
same = (mla.naive(xm) - mla.absorbed(xm)).abs().max().item()
broken = (mla.naive(xm, rope_on_latent_keys=True) - mla.absorbed(xm)).abs().max().item()
scale_out = mla.naive(xm).abs().max().item()
full_cache = mla.h * (mla.dn + mla.dr + mla.dv)
latent_cache = mla.dc + mla.dr
OUT['mla'] = {'naive_vs_absorbed': same, 'rope_on_latent_vs_absorbed': broken, 'output_scale': scale_out,
              'full_cache_per_token': full_cache, 'latent_cache_per_token': latent_cache}
print(f'4. MLA: full path vs compressed-cache path, max |diff| = {same:.1e}')
print(f'   with RoPE applied to the reconstructed keys, the compressed path is wrong by up to {broken:.2f} (outputs up to {scale_out:.2f})')
print(f'   numbers cached per token per layer: full K and V {full_cache}, latent + RoPE key {latent_cache} ({full_cache / latent_cache:.1f}x smaller)')


# ---------------------------------------------------------------- 5. KV cache per token, real models (BF16)
def kib(x):
    return x / 1024
models = {
    'Llama 2 7B (MHA)': dict(layers=32, kv=32, k_dim=128, v_dim=128),
    'Llama 3.1 8B (GQA, 8 KV heads)': dict(layers=32, kv=8, k_dim=128, v_dim=128),
    'Falcon 7B (MQA)': dict(layers=32, kv=1, k_dim=64, v_dim=64),
    'DeepSeek-V3 if it used MHA': dict(layers=61, kv=128, k_dim=128 + 64, v_dim=128),
    'DeepSeek-V3 (MLA)': dict(layers=61, latent=512 + 64),
}
sizes = {}
for m, c in models.items():
    per_layer = c['latent'] if 'latent' in c else c['kv'] * (c['k_dim'] + c['v_dim'])
    sizes[m] = per_layer * c['layers'] * 2                      # 2 bytes per BF16 number
    print(f'5. {m:32s} {kib(sizes[m]):8.1f} KiB per token')
OUT['kv_per_token_bytes'] = sizes
json.dump(OUT, open('results/part2.json', 'w'), indent=1)
print('saved results/part2.json')
