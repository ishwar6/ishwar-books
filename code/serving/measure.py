"""Step costs of a real model, measured: Qwen2.5-0.5B (BF16) through PyTorch on one GPU (Apple 'mps' here).

This is NOT a serving engine. It times the two kinds of forward pass an engine runs, so the queueing
simulator (simulate.py) can use real step costs instead of made-up ones:
  decode : b sequences, each with c tokens already in its KV cache, produce one token each
  prefill: n new prompt tokens on top of c cached tokens (c = 0 for a fresh prompt)
Then it fits one step-cost model to all of them:  t = a + b_tok * new_tokens + g * kv_tokens_read.
Random token IDs: the content of the text does not change how long the arithmetic takes.
Run: python code/serving/measure.py   -> results/measure.json and results/measure_stdout.txt
"""
import json, platform, statistics, sys, time
from pathlib import Path
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, DynamicCache

HERE = Path(__file__).resolve().parent
DEV = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
sync = {'mps': torch.mps.synchronize, 'cuda': torch.cuda.synchronize}.get(DEV, lambda: None)
MODEL = 'Qwen/Qwen2.5-0.5B'
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
V = model.config.vocab_size
g = torch.Generator().manual_seed(6)
ids = lambda b, n: torch.randint(0, V, (b, n), generator=g).to(DEV)
REPEATS, WARMUP = 30, 2
out = {'device': DEV, 'model': MODEL, 'torch': torch.__version__, 'transformers': transformers.__version__,
       'python': platform.python_version(), 'machine': platform.machine(), 'repeats': REPEATS, 'warmup': WARMUP}
lines = []
def log(s=''):
    print(s); lines.append(s)


@torch.inference_mode()
def forward(x, cache):
    return model(input_ids=x, past_key_values=cache, logits_to_keep=1)


def timed_once(kind, b_or_n, c):
    """One timed forward pass: decode (b sequences, one token each) or prefill (n tokens), on a c-token cache."""
    if kind == 'decode':
        cache, x = cache_of(b_or_n, c), ids(b_or_n, 1)
    else:
        cache, x = cache_of(1, c), ids(1, b_or_n)
    sync(); t = time.perf_counter(); forward(x, cache); sync()
    return (time.perf_counter() - t) * 1000


@torch.inference_mode()
def cache_of(b, c):
    """A KV cache holding c tokens for each of b sequences (one prefill, copied b times)."""
    cache = DynamicCache()
    if c:
        forward(ids(1, c), cache)
        if b > 1:
            cache.batch_repeat_interleave(b)
    return cache


log(f'$ python code/serving/measure.py')
log(f'{MODEL}, BF16, device={DEV}, fastest of {REPEATS} interleaved rounds after {WARMUP} warm-up rounds (median in brackets)')
weights = sum(p.numel() * p.element_size() for p in model.parameters())
out['weight_bytes'] = weights
log(f'weights in memory: {weights / 2**20:.0f} MiB')
log()
# Interleaved rounds: every configuration is timed once per round, so each one gets the same chances to hit
# a quiet moment on the shared GPU. Keep the fastest run (the least disturbed) and the median.
CONFIGS = [('decode', b, c) for c in (256, 1024, 4096) for b in (1, 4, 16, 32, 64, 128)] + \
          [('prefill', n, c) for n, c in [(16, 0), (64, 0), (128, 0), (256, 0), (512, 0), (1024, 0), (2048, 0), (4096, 0),
                                          (512, 2048), (512, 6144), (2048, 2048)]]
times = {k: [] for k in CONFIGS}
for rnd in range(WARMUP + REPEATS):
    for k in CONFIGS:
        t = timed_once(*k)
        if rnd >= WARMUP:
            times[k].append(t)
log('DECODE: b sequences, c cached tokens each, one new token each')
log(f'{"b":>5} {"c":>6} {"ms/step":>9} {"(median)":>9} {"tokens/s":>9}')
dec = []
for (kind, b, c), ts in times.items():
    if kind != 'decode': continue
    ms, med = min(ts), statistics.median(ts)
    dec.append({'b': b, 'c': c, 'ms': round(ms, 2), 'median_ms': round(med, 2), 'tok_s': round(b / ms * 1000)})
    log(f'{b:5d} {c:6d} {ms:9.2f} {f"({med:.1f})":>9} {b / ms * 1000:9.0f}')
out['decode'] = dec
log()
log('PREFILL: n new prompt tokens on top of c cached tokens (one sequence)')
log(f'{"n":>5} {"c":>6} {"ms":>9} {"(median)":>9} {"tokens/s":>9}')
pre = []
for (kind, n, c), ts in times.items():
    if kind != 'prefill': continue
    ms, med = min(ts), statistics.median(ts)
    pre.append({'n': n, 'c': c, 'ms': round(ms, 2), 'median_ms': round(med, 2), 'tok_s': round(n / ms * 1000)})
    log(f'{n:5d} {c:6d} {ms:9.2f} {f"({med:.1f})":>9} {n / ms * 1000:9.0f}')
out['prefill'] = pre

def fit(dec, pre):
    """One cost model for every step: t = a + b*new + g*pairs + k*kv. new = tokens computed this step; pairs =
    prefill attention pairs (each new prompt token attends to every earlier token); kv = cached tokens read
    (decode reads every sequence's whole cache, which is memory-bound and costs more per token than prefill)."""
    rows, y = [], []
    for r in dec:
        rows.append([1, r['b'], 0, r['b'] * (r['c'] + 1)]); y.append(r['ms'])
    for r in pre:
        rows.append([1, r['n'], r['n'] * r['c'] + r['n'] * (r['n'] + 1) / 2, r['c'] + r['n']]); y.append(r['ms'])
    X, y = np.array(rows, float), np.array(y)
    w = 1 / y                               # relative least squares: small steps matter as much as big ones
    coef, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    err = np.abs(X @ coef - y) / y
    return coef, err


coef, err = fit(dec, pre)
out['cost_model'] = {'a_ms': coef[0], 'per_new_token_ms': coef[1], 'per_kv_pair_ms': coef[2], 'per_kv_read_ms': coef[3],
                     'median_rel_err': float(np.median(err)), 'max_rel_err': float(err.max()),
                     'worst_point': (dec + pre)[int(err.argmax())]}
log()
log('COST MODEL fitted to all points above (relative least squares):')
log(f'  t = {coef[0]:.2f} ms + {coef[1] * 1000:.1f} us x new tokens + {coef[2] * 1e6:.2f} ns x prefill attention pairs'
    f' + {coef[3] * 1e6:.0f} ns x cached tokens read')
log(f'  median error {np.median(err) * 100:.0f}%, worst {err.max() * 100:.0f}% at {(dec + pre)[int(err.argmax())]}')
(HERE / 'results').mkdir(exist_ok=True)
(HERE / 'results/measure.json').write_text(json.dumps(out, indent=2) + '\n')
(HERE / 'results/measure_stdout.txt').write_text('\n'.join(lines) + '\n')
