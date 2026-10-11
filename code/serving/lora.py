"""Many LoRA fine-tunes on one base model: memory arithmetic and a real measurement.

1. Memory (arithmetic): a LoRA adapter for Llama 3.1 8B at ranks 8, 16 and 64 on all seven projections,
   against a full fine-tuned copy of the model.
2. Measured: Qwen2.5-0.5B (BF16) decode steps on this laptop's GPU, batch of 32 requests, each request
   with its own adapter on every linear layer. The adapter maths runs as two batched matrix products over
   gathered adapter weights (the idea behind Punica's SGMV and S-LoRA's kernels, written in plain PyTorch),
   or as a loop over adapters. Checked against merging each adapter into its own copy of the weights.
Run: python code/serving/lora.py  -> results/lora.json, results/lora_stdout.txt
"""
import statistics, time
import torch
from transformers import AutoModelForCausalLM, DynamicCache
from common import Log, LLAMA8B

log = Log('lora')
OUT = {}

# ---------------------------------------------------------------- 1. memory
h, inter, kvd = LLAMA8B['hidden'], 14336, LLAMA8B['kv_heads'] * LLAMA8B['head_dim']
shapes = {'q': (h, h), 'k': (h, kvd), 'v': (h, kvd), 'o': (h, h), 'gate': (h, inter), 'up': (h, inter), 'down': (inter, h)}
per_rank = LLAMA8B['layers'] * sum(i + o for i, o in shapes.values())
full = LLAMA8B['params'] * 2
log('1. MEMORY, Llama 3.1 8B: a LoRA adapter on q, k, v, o, gate, up, down in all 32 layers')
log(f'   parameters per unit of rank: 32 x sum(d_in + d_out) = {per_rank:,}')
rows = []
for r in (8, 16, 64):
    b = per_rank * r * 2
    rows.append({'rank': r, 'params': per_rank * r, 'bytes': b, 'per_80GB': None})
    log(f'   rank {r:2d}: {per_rank * r / 1e6:6.1f} M parameters = {b / 2**20:6.0f} MiB in BF16 ({b / full * 100:.2f}% of the {full / 1e9:.1f} GB model);'
        f' 100 adapters = {100 * b / 1e9:.1f} GB')
log(f'   100 full fine-tuned copies instead: {100 * full / 1e12:.2f} TB')
OUT['memory'] = {'per_rank': per_rank, 'full_bytes': full, 'rows': rows}

# ---------------------------------------------------------------- 2. measured
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
sync = {'mps': torch.mps.synchronize, 'cuda': torch.cuda.synchronize}.get(DEV, lambda: None)
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B', dtype=torch.bfloat16).to(DEV).eval()
lin = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and 'lm_head' not in n]
B, C, RANK, N_AD = 32, 512, 16, 32
g = torch.Generator().manual_seed(0)
ADAPTERS = {}
for n, m in lin:                      # random adapters: A (n_adapters, d_in, r), B (n_adapters, r, d_out)
    A = (torch.randn(N_AD, m.in_features, RANK, generator=g) / m.in_features ** 0.5).to(DEV, torch.bfloat16)
    Bm = (torch.randn(N_AD, RANK, m.out_features, generator=g) * 0.01).to(DEV, torch.bfloat16)
    ADAPTERS[n] = (A, Bm)
idx = torch.arange(B, device=DEV) % N_AD           # request i uses adapter idx[i]
mode = {'m': 'base'}


def lora_hook(name):
    A, Bm = ADAPTERS[name]
    def hook(mod, inp, out):
        if mode['m'] == 'base':
            return out
        x = inp[0]                                   # (batch, tokens, d_in)
        if mode['m'] == 'gather':                    # one batched product for all requests: x A[idx] B[idx]
            return out + torch.bmm(torch.bmm(x, A[idx]), Bm[idx])
        if mode['m'] == 'same':                      # every request uses adapter 0
            return out + (x @ A[0]) @ Bm[0]
        y = out.clone()                              # 'loop': one small product per distinct adapter
        for a in idx.unique().tolist():
            rows_ = (idx == a).nonzero().squeeze(1)
            y[rows_] += (x[rows_] @ A[a]) @ Bm[a]
        return y
    return hook

hooks = [m.register_forward_hook(lora_hook(n)) for n, m in lin]


@torch.inference_mode()
def step(cache, x):
    return model(input_ids=x, past_key_values=cache, logits_to_keep=1).logits


def make_cache():
    c = DynamicCache()
    step(c, torch.randint(0, 1000, (1, C), generator=g).to(DEV))
    c.batch_repeat_interleave(B)
    return c


x = torch.randint(0, 1000, (B, 1), generator=g).to(DEV)
cache0 = make_cache()
import copy
log()
log(f'2. MEASURED, Qwen2.5-0.5B BF16 on {DEV}: one decode step, batch {B}, {C} cached tokens each, rank-{RANK} adapters on all {len(lin)} linear layers')
res = {}
for m_ in ('base', 'same', 'gather', 'loop'):
    mode['m'] = m_
    times = []
    for i in range(14):
        c = copy.deepcopy(cache0)
        sync(); t = time.perf_counter(); step(c, x); sync()
        if i >= 4: times.append((time.perf_counter() - t) * 1000)
    res[m_] = {'median_ms': statistics.median(times), 'min_ms': min(times)}
desc = {'base': 'base model only', 'same': 'all 32 requests share one adapter', 'gather': f'{N_AD} different adapters, gathered + batched',
        'loop': f'{N_AD} different adapters, loop over adapters'}
for k, v in res.items():
    log(f'   {desc[k]:42s} median {v["median_ms"]:6.2f} ms  min {v["min_ms"]:6.2f} ms  ({v["median_ms"] / res["base"]["median_ms"]:.2f}x base)')

# correctness: gathered LoRA output == merging each request's adapter into its own weights (one layer, float32)
n0, m0 = lin[0]
A, Bm = ADAPTERS[n0]
xx = torch.randn(B, 1, m0.in_features, generator=g).to(DEV, torch.float32)
W = m0.weight.float()
gathered = xx @ W.T + torch.bmm(torch.bmm(xx, A[idx].float()), Bm[idx].float())
merged = torch.stack([xx[i] @ (W + (A[idx[i]].float() @ Bm[idx[i]].float()).T).T for i in range(B)])
diff = float((gathered - merged).abs().max())
log(f'   check on {n0}: gathered LoRA vs per-request merged weights, max abs difference {diff:.2e}')
OUT['measured'] = {'batch': B, 'context': C, 'rank': RANK, 'adapters': N_AD, 'device': DEV, **res, 'max_abs_diff': diff}
log.save(OUT)
