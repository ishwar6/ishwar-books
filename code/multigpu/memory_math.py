"""Section 1 arithmetic: do the weights and the KV cache fit on one GPU, and how fast can decode be?
Pure arithmetic from model shapes and spec-sheet numbers (common.py). Nothing is measured here.
Run: python code/multigpu/memory_math.py  -> results/memory_math.json"""
import math
from common import LLAMA, DSV3, GPUS, GB, GiB, llama_params, kv_bytes_per_token, dsv3_kv_bytes_per_token, save

H100 = GPUS['H100 SXM']
USABLE = 0.90          # assume 10% of GPU memory goes to the CUDA context, activations and fragmentation
out = {'assumptions': {'usable_fraction': USABLE, 'gpu': 'H100 SXM 80 GB, 3.35 TB/s'}}

print('MODEL MEMORY: weights + KV cache against one 80 GB H100')
print(f"{'model':16}{'params':>10}{'weights BF16':>14}{'KV/token':>11}{'KV 32x8K':>11}{'total':>10}{'H100s':>7}")
rows = []
for name, c in LLAMA.items():
    p = llama_params(c)
    w = p * 2
    kv = kv_bytes_per_token(c)
    kv_load = kv * 32 * 8192                        # 32 conversations of 8,192 tokens each
    total = w + kv_load
    gpus = math.ceil(total / (H100['mem'] * USABLE))
    rows.append(dict(model=name, params=p, weights_bf16=w, kv_per_token=kv, kv_32x8k=kv_load, total=total, h100s=gpus))
    print(f"{name:16}{p/1e9:9.1f}B{w/GB:12.0f} GB{kv/1024:8.0f} KiB{kv_load/GiB:8.0f} GiB{total/GB:7.0f} GB{gpus:7d}")

# DeepSeek-V3: weights released in FP8 (1 byte per parameter); MLA cache in BF16.
w8 = DSV3['total_params'] * 1
kv = dsv3_kv_bytes_per_token()
kv_load = kv * 32 * 8192
total = w8 + kv_load
gpus = math.ceil(total / (H100['mem'] * USABLE))
rows.append(dict(model='DeepSeek-V3 (FP8)', params=DSV3['total_params'], weights_fp8=w8, kv_per_token=kv,
                 kv_32x8k=kv_load, total=total, h100s=gpus))
print(f"{'DeepSeek-V3':16}{DSV3['total_params']/1e9:9.0f}B{w8/GB:9.0f} GB(FP8){kv/1024:6.1f} KiB{kv_load/GiB:8.0f} GiB{total/GB:7.0f} GB{gpus:7d}")
out['models'] = rows

# If Llama 3.1 70B used ordinary multi-head attention (64 KV heads), the cache would be 8x larger.
mha = dict(LLAMA['Llama 3.1 70B'], Hkv=64)
out['llama70b_if_mha_kv_per_token'] = kv_bytes_per_token(mha)
print(f"\nDeepSeek-V3 MLA cache per token: 61 x (512 + 64) x 2 B = {kv:,} B = {kv/1024:.1f} KiB "
      f"(Llama 3.1 405B: {kv_bytes_per_token(LLAMA['Llama 3.1 405B'])/1024:.0f} KiB)")

print('\nDECODE FLOOR: time to read every weight once per step (batch 1, perfect split, no communication)')
print(f"{'model':16}" + ''.join(f'{n:>10}' for n in ['1 GPU', '2', '4', '8', '16']))
floors = {}
for r in rows[:3]:
    w = r['weights_bf16']
    t = {n: w / (n * H100['hbm']) * 1e3 for n in [1, 2, 4, 8, 16]}
    floors[r['model']] = t
    fits = lambda n: (w + 0) <= n * H100['mem'] * USABLE
    print(f"{r['model']:16}" + ''.join(f"{t[n]:8.1f}ms" if fits(n) else f"{'(no fit)':>10}" for n in t))
active = DSV3['active_params'] * 1
floors['DeepSeek-V3 active 37B FP8'] = {1: active / H100['hbm'] * 1e3}
print(f"DeepSeek-V3: only the active 37B FP8 weights are read per token: {active/H100['hbm']*1e3:.1f} ms on one GPU's bandwidth"
      f"\n  (but all 671 GB must still be stored somewhere, and a large batch touches most experts)")
out['decode_floor_ms'] = {k: {str(n): v for n, v in d.items()} for k, d in floors.items()}
save('memory_math', out)
