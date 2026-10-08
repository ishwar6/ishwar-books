"""Real prefix reuse on a real model: Qwen2.5-0.5B (BF16) through PyTorch on one GPU.

This is NOT SGLang or vLLM. It measures the operation both engines rely on: run prefill
over a shared prefix once, keep its KV cache, and later prefill only the new suffix on top of it.
Cold  = prefill prefix + suffix from nothing.
Warm  = prefix KV already cached; prefill only the suffix tokens.
Random token IDs: the text content does not change how long the arithmetic takes.
Run: python code/sglang/measure_prefix.py   (results/measure_prefix.json and _stdout.txt)
"""
import copy, json, platform, statistics, sys, time
from pathlib import Path
import torch
import transformers
from transformers import AutoModelForCausalLM, DynamicCache

HERE = Path(__file__).resolve().parent
DEV = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
sync = {'mps': torch.mps.synchronize, 'cuda': torch.cuda.synchronize}.get(DEV, lambda: None)
MODEL = 'Qwen/Qwen2.5-0.5B'
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
cfg = model.config
g = torch.Generator().manual_seed(5)
ids = lambda n: torch.randint(0, cfg.vocab_size, (1, n), generator=g).to(DEV)
REPEATS, WARMUP = 9, 3


@torch.inference_mode()
def prefill(x, cache=None):
    out = model(input_ids=x, past_key_values=cache if cache is not None else DynamicCache(), logits_to_keep=1)
    return out.logits[0, -1].float(), out.past_key_values


def timed(fn):
    times = []
    for i in range(WARMUP + REPEATS):
        sync(); t = time.perf_counter(); r = fn(); sync()
        if i >= WARMUP:
            times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times), r


def cold_warm(P, S):
    prefix, suffix = ids(P), ids(S)
    full = torch.cat([prefix, suffix], 1)
    cold_ms, (cold_logits, _) = timed(lambda: prefill(full))
    _, prefix_cache = prefill(prefix)
    warm_runs = []
    for i in range(WARMUP + REPEATS):
        c = copy.deepcopy(prefix_cache)            # copying is not timed: an engine reuses blocks in place
        sync(); t = time.perf_counter(); warm_logits, _ = prefill(suffix, c); sync()
        if i >= WARMUP:
            warm_runs.append((time.perf_counter() - t) * 1000)
    warm_ms = statistics.median(warm_runs)
    diff = float((cold_logits - warm_logits).abs().max())
    same = int(cold_logits.argmax()) == int(warm_logits.argmax())
    return {'P': P, 'S': S, 'cold_ms': round(cold_ms, 2), 'warm_ms': round(warm_ms, 2),
            'speedup': round(cold_ms / warm_ms, 2), 'same_next_token': same, 'max_abs_logit_diff': round(diff, 4)}


@torch.inference_mode()
def decode_ms(context, steps=32):
    _, cache = prefill(ids(context))
    tok = ids(1)
    times = []
    for i in range(steps):
        sync(); t = time.perf_counter()
        out = model(input_ids=tok, past_key_values=cache, logits_to_keep=1)
        cache = out.past_key_values; tok = out.logits[:, -1:].argmax(-1)
        sync(); times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times[4:])


def main():
    log = [f'$ python code/sglang/measure_prefix.py',
           f'model {MODEL}  bf16  device {DEV}  torch {torch.__version__}',
           f'layers {cfg.num_hidden_layers}  kv_heads {cfg.num_key_value_heads}  head_dim {cfg.hidden_size // cfg.num_attention_heads}',
           '', 'SHARED PREFIX SWEEP (suffix S = 64 new tokens)',
           '     P      S   cold ms   warm ms   speedup   same token   max |logit diff|']
    rows = []
    for P in [512, 1024, 2048, 4096, 8192]:
        r = cold_warm(P, 64); rows.append(r)
        log.append(f"{P:6d} {64:6d} {r['cold_ms']:9.1f} {r['warm_ms']:9.1f} {r['speedup']:8.1f}x   {str(r['same_next_token']):>10}   {r['max_abs_logit_diff']:.4f}")
    log += ['', 'SUFFIX SWEEP (shared prefix P = 2048)',
            '     P      S   cold ms   warm ms   speedup']
    srows = []
    for S in [16, 64, 256, 1024]:
        r = cold_warm(2048, S); srows.append(r)
        log.append(f"{2048:6d} {S:6d} {r['cold_ms']:9.1f} {r['warm_ms']:9.1f} {r['speedup']:8.1f}x")
    dec = decode_ms(2112)
    log += ['', f'DECODE one token at context 2,112: {dec:.2f} ms per step']
    print('\n'.join(log))
    res = {'kind': 'Real PyTorch measurement of prefix-KV reuse; not an SGLang or vLLM benchmark',
           'model': MODEL, 'dtype': 'bfloat16', 'device': DEV, 'machine': platform.machine(), 'platform': platform.platform(),
           'chip': 'Apple M5 Pro' if DEV == 'mps' else 'unknown', 'torch': torch.__version__,
           'transformers': transformers.__version__, 'python': sys.version.split()[0],
           'repeats': REPEATS, 'warmup': WARMUP, 'statistic': 'median',
           'prefix_sweep': rows, 'suffix_sweep': srows, 'decode_ms_at_2112': round(dec, 2)}
    (HERE / 'results/measure_prefix.json').write_text(json.dumps(res, indent=2) + '\n')
    (HERE / 'results/measure_prefix_stdout.txt').write_text('\n'.join(log) + '\n')


if __name__ == '__main__':
    main()
