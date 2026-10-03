"""Part 4: linear attention, the delta rule, Gated DeltaNet. Proofs, memory and timing.
Writes results/part4.json."""
import json, math, time
import torch
import torch.nn.functional as F
from transformers.models.qwen3_next.modeling_qwen3_next import torch_recurrent_gated_delta_rule, torch_chunk_gated_delta_rule

torch.manual_seed(0)
OUT = {}
phi = lambda x: F.elu(x) + 1                     # the feature map of Katharopoulos et al. (2020): always positive


# 1. Linear attention: the parallel form and the recurrent form give the same answer ------------------
def linear_attention_parallel(q, k, v):
    """Like causal attention, but exp(q.k) is replaced by phi(q).phi(k), and there is no softmax."""
    Q, K = phi(q), phi(k)
    scores = (Q @ K.T).tril()                                  # (T, T), future set to 0
    return (scores @ v) / scores.sum(-1, keepdim=True)


def linear_attention_recurrent(q, k, v):
    """The same thing as an RNN: a fixed-size state S (d_k x d_v) and a normaliser z (d_k)."""
    S = torch.zeros(k.shape[-1], v.shape[-1], dtype=q.dtype)
    z = torch.zeros(k.shape[-1], dtype=q.dtype)
    out = []
    for t in range(q.shape[0]):
        S = S + torch.outer(phi(k[t]), v[t])                   # write: add this token's key-value pair
        z = z + phi(k[t])
        out.append((phi(q[t]) @ S) / (phi(q[t]) @ z))          # read: one small matrix-vector product
    return torch.stack(out)


T, dk, dv = 200, 16, 16
q, k, v = (torch.randn(T, d, dtype=torch.float64) for d in (dk, dk, dv))
par, rec = linear_attention_parallel(q, k, v), linear_attention_recurrent(q, k, v)
OUT['linear_equivalence'] = {'T': T, 'max_diff': (par - rec).abs().max().item(), 'state_numbers': dk * dv + dk}
print('1. Linear attention, parallel form vs recurrent form (T = %d): max |diff| = %.1e' % (T, OUT['linear_equivalence']['max_diff']))
print('   the recurrent form keeps only %d numbers, however long the text' % OUT['linear_equivalence']['state_numbers'])


# 2. Overwriting a memory: plain linear attention vs the delta rule -----------------------------------
def write_linear(S, k, v):
    return S + torch.outer(v, k)                               # S is d_v x d_k, as in the Gated DeltaNet paper


def write_delta(S, k, v, beta=1.0):
    return S - beta * torch.outer(S @ k - v, k)                # S (I - beta k k^T) + beta v k^T


d = 64
keys = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))[0].T   # orthonormal keys: no accidental overlap
x_key, y_key = keys[0], keys[1]
one, five, seven = (torch.full((d,), float(n), dtype=torch.float64) for n in (1, 5, 7))
story = [('x = 1', x_key, one), ('y = 7', y_key, seven), ('x = 5', x_key, five)]
S_lin = torch.zeros(d, d, dtype=torch.float64); S_del = torch.zeros(d, d, dtype=torch.float64)
for _, kk, vv in story:
    S_lin, S_del = write_linear(S_lin, kk, vv), write_delta(S_del, kk, vv)
OUT['overwrite'] = {'linear_x': (S_lin @ x_key).mean().item(), 'delta_x': (S_del @ x_key).mean().item(),
                    'linear_y': (S_lin @ y_key).mean().item(), 'delta_y': (S_del @ y_key).mean().item()}
print('2. Write "x = 1", "y = 7", then "x = 5". Read x and y back:')
print('   linear attention: x = %.2f, y = %.2f' % (OUT['overwrite']['linear_x'], OUT['overwrite']['linear_y']))
print('   delta rule:       x = %.2f, y = %.2f' % (OUT['overwrite']['delta_x'], OUT['overwrite']['delta_y']))


# 3. How many facts fit? Random keys, retrieve every value after all writes -------------------------
def recall_error(n, rule, d=64, trials=20):
    errs = []
    for _ in range(trials):
        K = F.normalize(torch.randn(n, d, dtype=torch.float64), dim=-1)
        V = torch.randn(n, d, dtype=torch.float64)
        S = torch.zeros(d, d, dtype=torch.float64)
        for i in range(n):
            S = write_linear(S, K[i], V[i]) if rule == 'linear' else write_delta(S, K[i], V[i])
        R = (S @ K.T).T                                        # read every key
        errs.append(((R - V).norm(dim=-1) / V.norm(dim=-1)).mean().item())
    return sum(errs) / len(errs)


ns = [4, 8, 16, 32, 48, 64, 96, 128, 192, 256]
cap = {'d': 64, 'n': ns, 'linear': [recall_error(n, 'linear') for n in ns], 'delta': [recall_error(n, 'delta') for n in ns]}
OUT['capacity'] = cap
print('3. Store n random facts in a 64 x 64 memory, then read all of them back (relative error, 0 = perfect):')
for n, a, b in zip(ns, cap['linear'], cap['delta']):
    print('   n = %3d: linear attention %.2f   delta rule %.2f' % (n, a, b))


# 4. The forget gate: how fast an old fact fades -----------------------------------------------------
fade = {a: [a ** t for t in (0, 10, 50, 100, 500)] for a in (0.9, 0.99, 0.999)}
OUT['fade'] = {'steps': [0, 10, 50, 100, 500], 'alphas': {str(a): f for a, f in fade.items()}}
print('4. Strength of a fact after t more tokens, with forget gate alpha (alpha^t):')
for a, f in fade.items():
    print('   alpha = %-5s: ' % a + '  '.join('t=%d: %.3f' % (t, x) for t, x in zip(OUT['fade']['steps'], f)))


# 5. Gated DeltaNet from scratch vs the reference code shipped with Qwen3-Next in transformers ---------
def gated_delta_rule(q, k, v, alpha, beta):
    """S_t = alpha_t * S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T ;  o_t = S_t q_t.
    q, k: (H, T, d_k) L2-normalised; v: (H, T, d_v); alpha, beta: (H, T) in (0, 1)."""
    H, T, dk = k.shape
    S = torch.zeros(H, v.shape[-1], dk, dtype=q.dtype)
    out = []
    for t in range(T):
        kt, vt = k[:, t], v[:, t]
        S = alpha[:, t, None, None] * S                                         # 1. forget a little (gate)
        S = S - beta[:, t, None, None] * torch.einsum('hv,hk->hvk', torch.einsum('hvk,hk->hv', S, kt) - vt, kt)   # 2. correct (delta)
        out.append(torch.einsum('hvk,hk->hv', S, q[:, t]))                       # 3. read
    return torch.stack(out, 1)


H, T, dk, dv = 4, 256, 32, 48
q = F.normalize(torch.randn(H, T, dk, dtype=torch.float64), dim=-1)
k = F.normalize(torch.randn(H, T, dk, dtype=torch.float64), dim=-1)
v = torch.randn(H, T, dv, dtype=torch.float64)
g = -F.softplus(torch.randn(H, T, dtype=torch.float64))          # log of the gate, as Qwen3-Next stores it
beta = torch.sigmoid(torch.randn(H, T, dtype=torch.float64))
mine = gated_delta_rule(q / math.sqrt(dk), k, v, g.exp(), beta)     # Qwen3-Next scales queries by 1/sqrt(d_k)
to_hf = lambda x: x.transpose(0, 1)[None]                          # (H, T, ...) -> (1, T, H, ...)
ref_rec, _ = torch_recurrent_gated_delta_rule(to_hf(q), to_hf(k), to_hf(v), g=g.T[None], beta=beta.T[None])
ref_chunk, _ = torch_chunk_gated_delta_rule(to_hf(q), to_hf(k), to_hf(v), g=g.T[None], beta=beta.T[None])
mine_hf = mine.transpose(0, 1)[None].float()
OUT['gdn_reference'] = {'vs_recurrent': (mine_hf - ref_rec.float()).abs().max().item(),
                        'vs_chunked': (mine_hf - ref_chunk.float()).abs().max().item(),
                        'output_scale': mine.abs().max().item()}
print('5. Our gated delta rule vs the Qwen3-Next reference code in transformers (float32 inside):')
print('   vs step-by-step version: max |diff| = %.1e' % OUT['gdn_reference']['vs_recurrent'])
print('   vs chunked version:      max |diff| = %.1e   (outputs up to %.2f)' % (OUT['gdn_reference']['vs_chunked'], OUT['gdn_reference']['output_scale']))


# 6. Memory: real hybrid models ------------------------------------------------------------------------
GiB, MiB = 2 ** 30, 2 ** 20
n = 262144
qn = dict(layers=48, full=12, kv_heads=2, head_dim=256, lin=36, v_heads=32, dk=128, dv=128)
qn_attn_per_tok = 2 * qn['kv_heads'] * qn['head_dim'] * 2                       # bytes per token per attention layer
qn_state = qn['lin'] * qn['v_heads'] * qn['dk'] * qn['dv'] * 2
mem = {'qwen3_next': {'context': n, 'hybrid_attention_gib': qn['full'] * qn_attn_per_tok * n / GiB,
                      'all_attention_gib': qn['layers'] * qn_attn_per_tok * n / GiB, 'gdn_state_mib': qn_state / MiB}}
km = dict(layers=27, mla=7, kda=20, heads=32, d=128, latent=512 + 64)
mem['kimi_linear'] = {'mla_layers': km['mla'], 'kda_layers': km['kda'],
                      'hybrid_bytes_per_token': km['mla'] * km['latent'] * 2, 'all_mla_bytes_per_token': km['layers'] * km['latent'] * 2,
                      'kda_state_mib': km['kda'] * km['heads'] * km['d'] * km['d'] * 2 / MiB}
mem['kimi_linear']['reduction'] = 1 - mem['kimi_linear']['hybrid_bytes_per_token'] / mem['kimi_linear']['all_mla_bytes_per_token']
nh = dict(attn=4, layers_attn_if_all=28, kv_heads=8, head_dim=128)
mem['nemotron_h_8b'] = {'attention_layers': 4, 'mamba_layers': 24, 'mlp_layers': 24}
ctxs = [4096, 16384, 65536, 262144]
mem['qwen3_next']['curve'] = {'context': ctxs,
                              'all_attention_gib': [qn['layers'] * qn_attn_per_tok * c / GiB for c in ctxs],
                              'hybrid_gib': [(qn['full'] * qn_attn_per_tok * c + qn_state) / GiB for c in ctxs]}
OUT['memory'] = mem
print('6. Memory, 2 bytes per number')
print('   Qwen3-Next-80B at %d tokens: 12 attention layers %.2f GiB (all 48 as attention: %.2f GiB), plus a fixed %.0f MiB of Gated DeltaNet state'
      % (n, mem['qwen3_next']['hybrid_attention_gib'], mem['qwen3_next']['all_attention_gib'], mem['qwen3_next']['gdn_state_mib']))
print('   Kimi Linear 48B: %d B per token with 7 MLA layers vs %d B if all 27 were MLA (%.0f%% less), plus a fixed %.0f MiB of KDA state'
      % (mem['kimi_linear']['hybrid_bytes_per_token'], mem['kimi_linear']['all_mla_bytes_per_token'], 100 * mem['kimi_linear']['reduction'], mem['kimi_linear']['kda_state_mib']))


# 7. One decode step: softmax attention over a growing cache vs one Gated DeltaNet state update --------
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
sync = (lambda: torch.mps.synchronize()) if dev == 'mps' else (lambda: None)
def timeit(fn, iters=50):
    for _ in range(5): fn()
    sync(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    sync(); return (time.perf_counter() - t0) / iters * 1000

Hq, Hkv, D = 16, 2, 256                                            # Qwen3-Next's gated-attention layer shape
Hv, Dk, Dv = 32, 128, 128                                          # Qwen3-Next's Gated DeltaNet layer shape
S = torch.zeros(1, Hv, Dk, Dv, device=dev, dtype=torch.float32)
kt = F.normalize(torch.randn(1, Hv, Dk, device=dev), dim=-1); qt = F.normalize(torch.randn(1, Hv, Dk, device=dev), dim=-1)
vt = torch.randn(1, Hv, Dv, device=dev); a_ = torch.rand(1, Hv, 1, 1, device=dev); b_ = torch.rand(1, Hv, 1, device=dev)
def gdn_step():
    St = S * a_
    delta = (vt - (St * kt[..., None]).sum(-2)) * b_
    St = St + kt[..., None] * delta[..., None, :]
    return (St * qt[..., None]).sum(-2)
timing = {}
for c in [4096, 16384, 65536, 262144]:
    qq = torch.randn(1, Hkv, Hq // Hkv, D, device=dev, dtype=torch.bfloat16)   # GQA: the 8 query heads of a group read one K/V head
    kk = torch.randn(1, Hkv, c, D, device=dev, dtype=torch.bfloat16)
    vv = torch.randn(1, Hkv, c, D, device=dev, dtype=torch.bfloat16)
    timing[c] = {'attention_ms': min(timeit(lambda: F.scaled_dot_product_attention(qq, kk, vv)) for _ in range(3)),
                 'gdn_ms': min(timeit(gdn_step) for _ in range(3))}
    del kk, vv
OUT['decode_timing'] = {'device': dev, 'results': timing}
print('7. One decode step on %s (Qwen3-Next layer shapes):' % dev)
for c, r in timing.items():
    print('   %6d tokens so far: gated attention %.3f ms   Gated DeltaNet %.3f ms' % (c, r['attention_ms'], r['gdn_ms']))

json.dump(OUT, open('results/part4.json', 'w'), indent=1)
