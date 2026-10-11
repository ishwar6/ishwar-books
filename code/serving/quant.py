"""Quantization on a real model, measured: Qwen2.5-0.5B on WikiText-2 (test split), PyTorch on the Apple GPU.

Every scheme is simulated ("fake quantization"): weights or activations are rounded to the low-precision grid
and immediately turned back into float32 numbers, so the model computes with exactly the values a real
low-bit kernel would see, but at float32 speed. This measures QUALITY. It cannot measure speed: no INT4 or
FP8 kernels run here. Memory sizes are counted from the formats.

Schemes (the embedding table, which Qwen2.5-0.5B shares with its output head, stays in BF16 in all of them):
  BF16             the released weights (baseline)
  INT8 / INT4 / INT3 weight-only, round-to-nearest (RTN), one scale per output channel or per group of 128
  INT4 g128 + AWQ  the same INT4 grid after an activation-aware per-input-channel scale search (AWQ, 2306.00978)
  FP8 W            weights rounded to FP8 E4M3, one scale per output channel
  FP8 W8A8         FP8 weights and FP8 activations (one dynamic scale per token) at every linear layer input
  INT8 W8A8        INT8 weights and INT8 activations, per token or one scale for the whole tensor
  FP8 KV           keys and values rounded to FP8 E4M3 as they would be stored in the KV cache
                   (rounded at the k_proj / v_proj outputs, before rotary position encoding)
Run: python code/serving/quant.py  -> results/quant.json, results/quant_stdout.txt
"""
import json, math, time
from pathlib import Path
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import Log

log = Log('quant')
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
MODEL = 'Qwen/Qwen2.5-0.5B'
WIN = 1024                      # tokens per evaluation window (non-overlapping)
KL_WINDOWS = 24                 # windows used for KL divergence and top-1 agreement against BF16
SNAP = Path.home() / '.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots'
wt = next(SNAP.glob('*/wikitext-2-raw-v1'))
tok = AutoTokenizer.from_pretrained(MODEL)
test_ids = tok('\n\n'.join(pq.read_table(wt / 'test-00000-of-00001.parquet')['text'].to_pylist()), return_tensors='pt').input_ids[0]
train_ids = tok('\n\n'.join(pq.read_table(wt / 'train-00000-of-00001.parquet')['text'].to_pylist()[:6000]), return_tensors='pt').input_ids[0]
n_win = len(test_ids) // WIN
log(f'{MODEL}, WikiText-2 test: {len(test_ids):,} tokens -> {n_win} windows of {WIN}; device {DEV}, float32 compute')


# ---------------------------------------------------------------- number formats
E4M3_MAX = 448.0
def fp8_e4m3(x):
    """Round to the nearest FP8 E4M3 value (4 exponent bits, 3 mantissa bits, max 448), as float32.
    The Apple GPU has no float8 type, so this is emulated; checked against torch's CPU float8 cast below."""
    a = x.abs().clamp(max=E4M3_MAX)
    e = torch.floor(torch.log2(a.clamp(min=2.0 ** -9))).clamp(min=-6)      # subnormals share 2^-6's spacing
    step = torch.exp2(e - 3)
    return torch.sign(x) * torch.round(a / step) * step

def q_fp8(w, dim=-1):
    s = w.abs().amax(dim=dim, keepdim=True).clamp(min=1e-12) / E4M3_MAX
    return fp8_e4m3(w / s) * s

def q_int_sym(w, bits, dim=-1):
    qmax = 2 ** (bits - 1) - 1
    s = w.abs().amax(dim=dim, keepdim=True).clamp(min=1e-12) / qmax
    return torch.round(w / s).clamp(-qmax - 1, qmax) * s

def q_int_group(w, bits, group=128):
    """Asymmetric (min-max with zero point) rounding, one scale and zero per group of `group` input weights."""
    o, i = w.shape
    g = w.reshape(o, i // group, group)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    s = (hi - lo).clamp(min=1e-12) / (2 ** bits - 1)
    z = torch.round(-lo / s)
    q = (torch.round(g / s) + z).clamp(0, 2 ** bits - 1)
    return ((q - z) * s).reshape(o, i)


# ---------------------------------------------------------------- the model and its linear layers
def load():
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()

def linears(m):
    return [(n, mod) for n, mod in m.named_modules() if isinstance(mod, torch.nn.Linear) and 'lm_head' not in n]


@torch.inference_mode()
def evaluate(m, ref=None):
    """Perplexity over all windows; with ref, also KL(ref || m) and top-1 agreement on the first KL_WINDOWS."""
    nll, n, kl, agree, cnt = 0.0, 0, 0.0, 0, 0
    for k in range(n_win):
        x = test_ids[k * WIN:(k + 1) * WIN].unsqueeze(0).to(DEV)
        logits = m(x).logits[0, :-1].float()
        lp = torch.log_softmax(logits, -1)
        nll += -lp.gather(1, x[0, 1:, None]).sum().item(); n += WIN - 1
        if ref is not None and k < KL_WINDOWS:
            rlp = torch.log_softmax(ref(x).logits[0, :-1].float(), -1)
            kl += (rlp.exp() * (rlp - lp)).sum().item(); cnt += WIN - 1
            agree += (rlp.argmax(-1) == lp.argmax(-1)).sum().item()
    return {'ppl': math.exp(nll / n), 'kl': kl / cnt if cnt else 0.0, 'top1': agree / cnt if cnt else 1.0}


def apply_weights(m, fn):
    for _, mod in linears(m):
        mod.weight.data = fn(mod.weight.data)

def hook_inputs(m, fn):
    hs = [mod.register_forward_pre_hook(lambda mod, a: (fn(a[0]),)) for _, mod in linears(m)]
    return hs

def hook_kv(m, fn):
    return [mod.register_forward_hook(lambda mod, a, out: fn(out)) for n, mod in linears(m) if n.endswith(('k_proj', 'v_proj'))]

def act_fp8_token(x):
    s = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / E4M3_MAX
    return fp8_e4m3(x / s) * s

def act_int8(x, per_token=True):
    s = (x.abs().amax(-1, keepdim=True) if per_token else x.abs().amax()).clamp(min=1e-12) / 127
    return torch.round(x / s).clamp(-128, 127) * s

def kv_fp8(y):
    s = y.abs().amax().clamp(min=1e-12) / E4M3_MAX        # one scale per layer and forward pass
    return fp8_e4m3(y / s) * s

def kv_fp8_token_head(y, head_dim=64):
    """One scale per token and per KV head (the idea behind vLLM's fp8_per_token_head cache type)."""
    z = y.reshape(*y.shape[:-1], -1, head_dim)
    s = z.abs().amax(-1, keepdim=True).clamp(min=1e-12) / E4M3_MAX
    return (fp8_e4m3(z / s) * s).reshape(y.shape)


# ---------------------------------------------------------------- AWQ-style activation-aware scaling
@torch.inference_mode()
def awq(m, bits=4, group=128, n_tokens=4096, grid=20):
    """For each set of linear layers that read the same input, try scales s = mean|x|^alpha per input channel,
    quantize W*s, and keep the alpha whose outputs Q(W*s)(x/s) best match W x on calibration text.
    Simplified from AWQ: no weight clipping search, scales searched per layer with full-precision inputs."""
    groups = {}
    for n, mod in linears(m):
        layer, kind = n.rsplit('.', 1)
        key = layer.replace('self_attn', 'attn_in') if kind in ('q_proj', 'k_proj', 'v_proj') else \
              layer.replace('mlp', 'mlp_in') if kind in ('gate_proj', 'up_proj') else n
        groups.setdefault(key, []).append(mod)
    xs = {}
    def grab(mod, a):
        if id(mod) not in xs:
            xs[id(mod)] = a[0].reshape(-1, a[0].shape[-1])[:n_tokens].clone()
    hs = [mod.register_forward_pre_hook(grab) for mods in groups.values() for mod in mods[:1]]
    m(train_ids[:n_tokens].unsqueeze(0).to(DEV))
    for h in hs: h.remove()
    chosen = []
    for key, mods in groups.items():
        x = xs[id(mods[0])]
        sx = x.abs().mean(0)
        ref = [x @ mod.weight.T for mod in mods]
        best = (float('inf'), 0.0, None)
        for k in range(grid + 1):
            alpha = k / grid
            s = sx.pow(alpha).clamp(min=1e-4); s = s / (s.max() * s.min()).sqrt()
            qs = [q_int_group(mod.weight * s, bits, group) / s for mod in mods]
            err = sum(((x @ q.T) - r).pow(2).mean().item() for q, r in zip(qs, ref))
            if err < best[0]: best = (err, alpha, qs)
        for mod, q in zip(mods, best[2]): mod.weight.data = q
        chosen.append(best[1])
    return sum(chosen) / len(chosen)


# ---------------------------------------------------------------- run
x = torch.randn(200_000) * 30
x[:1000] = torch.linspace(-0.02, 0.02, 1000)                     # include subnormals
emul_ok = torch.equal(fp8_e4m3(x), x.to(torch.float8_e4m3fn).float())
log(f'FP8 E4M3 emulation equals torch CPU float8 cast on 200,000 values: {emul_ok}')

ref = load()
n_lin = sum(mod.weight.numel() for _, mod in linears(ref))
n_emb = ref.model.embed_tokens.weight.numel()
log(f'linear-layer weights: {n_lin / 1e6:.1f} M, embedding/output table (tied): {n_emb / 1e6:.1f} M')
log()

def bytes_of(bits_per_weight, scale_bytes_per_weight=0.0, kv=False):
    return n_lin * (bits_per_weight / 8 + scale_bytes_per_weight) + n_emb * 2

SCHEMES = [
    # name, weight fn, act hook, kv hook, bits/weight, scale bytes per weight, AWQ?
    ('BF16 (baseline)', None, None, None, 16, 0, False),
    ('INT8 per-channel', lambda w: q_int_sym(w, 8), None, None, 8, 0, False),
    ('FP8 E4M3 per-channel', q_fp8, None, None, 8, 0, False),
    ('INT4 per-channel', lambda w: q_int_sym(w, 4), None, None, 4, 0, False),
    ('INT4 g128', lambda w: q_int_group(w, 4), None, None, 4, 4 / 128, False),
    ('INT4 g128 + AWQ scales', None, None, None, 4, 4 / 128, True),
    ('INT3 g128', lambda w: q_int_group(w, 3), None, None, 3, 4 / 128, False),
    ('FP8 W8A8 (per-token act.)', q_fp8, act_fp8_token, None, 8, 0, False),
    ('INT8 W8A8 (per-token act.)', lambda w: q_int_sym(w, 8), act_int8, None, 8, 0, False),
    ('INT8 W8A8 (per-tensor act.)', lambda w: q_int_sym(w, 8), lambda a: act_int8(a, False), None, 8, 0, False),
    ('FP8 KV, one scale per layer', None, None, kv_fp8, 16, 0, False),
    ('FP8 KV, scale per token+head', None, None, kv_fp8_token_head, 16, 0, False),
]
rows = []
log(f'{"scheme":30s} {"weights MiB":>11} {"PPL":>8} {"dPPL":>7} {"KL vs BF16":>11} {"top-1 same":>11} {"s":>5}')
base_ppl = None
for name, wfn, afn, kfn, bits, sb, use_awq in SCHEMES:
    t0 = time.time()
    m = load() if (wfn or afn or kfn or use_awq) else ref
    extra = {}
    if wfn: apply_weights(m, wfn)
    if use_awq: extra['mean_alpha'] = awq(m)
    hs = (hook_inputs(m, afn) if afn else []) + (hook_kv(m, kfn) if kfn else [])
    r = evaluate(m, ref if m is not ref else None)
    for h in hs: h.remove()
    base_ppl = base_ppl or r['ppl']
    mib = bytes_of(bits, sb) / 2**20
    rows.append({'scheme': name, 'weight_MiB': mib, 'bits': bits, **r, **extra})
    log(f'{name:30s} {mib:11.0f} {r["ppl"]:8.3f} {r["ppl"] - base_ppl:+7.3f} {r["kl"]:11.4f} {r["top1"] * 100:10.1f}% {time.time() - t0:5.0f}')
    if m is not ref: del m
    if DEV == 'mps': torch.mps.empty_cache()
log()
log('Weights MiB counts the linear layers at the stated bits (+ one 16-bit scale and one 16-bit zero per group of 128')
log('for g128) plus the BF16 embedding table. KL and top-1 use the first 24 windows (24,552 next-token predictions).')
log.save({'model': MODEL, 'windows': n_win, 'win': WIN, 'tokens': len(test_ids), 'fp8_emulation_matches_torch': emul_ok,
          'linear_params': n_lin, 'embedding_params': n_emb, 'rows': rows})
