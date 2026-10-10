"""Real timings on this laptop's Apple GPU (PyTorch "mps"): what a tensor-parallel shard costs.

1. Memory bandwidth: copy a 1 GiB BF16 tensor on the GPU (bytes read + bytes written per second).
2. One Llama 3.1 8B MLP matrix (14,336 x 4,096, BF16, 117 MB) multiplied by 1 to 4,096 tokens, whole and
   split into 2 and 4 row shards (the piece one GPU would hold under TP=2 and TP=4). Time per shard, the
   GB/s and TFLOP/s it reaches, and the shard's speedup over the whole matrix.
3. The bytes that shard must then exchange (its partial output: tokens x 4,096 x 2 bytes) and how long that
   takes on spec-sheet links, set against the measured compute time. The compute is measured here; the link
   times are arithmetic. Only the ratio's trend is the point.
Fastest of 40 runs after 5 warm-ups (the laptop was shared with other jobs while measuring, so the minimum is used).
Run: python code/multigpu/measure_mps.py  -> results/measure_mps.json"""
import time, statistics, platform
import torch
from common import LINKS, save

dev = 'mps'
sync = torch.mps.synchronize


def timeit(fn, reps=40, warm=5):
    """Fastest of `reps` runs. The laptop was shared with other jobs, so the minimum (the run least disturbed by
    them) is a steadier estimate of the work itself than the median."""
    for _ in range(warm):
        fn()
    sync()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t)
    return min(ts)


out = {'machine': platform.machine(), 'torch': torch.__version__}
print('1. GPU MEMORY BANDWIDTH (copy 1 GiB, BF16, on mps)')
x = torch.empty(2 ** 29, dtype=torch.bfloat16, device=dev)
y = torch.empty_like(x)
t = timeit(lambda: y.copy_(x))
bw = 2 * x.numel() * 2 / t
out['copy_GBps'] = bw / 1e9
print(f'   {t*1e3:.1f} ms per copy -> {bw/1e9:.0f} GB/s (read + write)')
del x, y

print('\n2. ONE 8B MLP MATRIX, WHOLE AND SPLIT (BF16, mps)')
W = torch.randn(14336, 4096, dtype=torch.bfloat16, device=dev) * 0.02
rows = []
for T in [1, 16, 128, 1024, 4096]:
    X = torch.randn(T, 4096, dtype=torch.bfloat16, device=dev)
    base = None
    for p in [1, 2, 4]:
        Wp = W[: 14336 // p].contiguous()                   # one GPU's shard of the output columns
        t = timeit(lambda: X @ Wp.T)
        flops = 2 * T * Wp.numel()
        bytes_ = Wp.numel() * 2 + X.numel() * 2 + T * Wp.shape[0] * 2
        base = base or t
        rows.append(dict(tokens=T, tp=p, ms=t * 1e3, TFLOPs=flops / t / 1e12, GBps=bytes_ / t / 1e9, speedup=base / t))
        print(f'   tokens {T:5}  TP={p}: {t*1e3:7.3f} ms  {flops/t/1e12:5.2f} TFLOP/s  {bytes_/t/1e9:5.0f} GB/s  speedup {base/t:4.2f}x')
out['shard'] = rows

print('\n3. COMPUTE PER SHARD (measured above, TP=4) AGAINST THE BYTES IT MUST EXCHANGE (arithmetic)')
ex = []
for r in rows:
    if r['tp'] != 4:
        continue
    N = r['tokens'] * 4096 * 2                               # partial output of the down projection, BF16
    line = dict(tokens=r['tokens'], compute_ms=r['ms'], bytes=N)
    for link, B in LINKS.items():
        line[link] = 2 * 3 / 4 * N / B * 1e3                 # all-reduce bytes per GPU over the link, no latency term
    ex.append(line)
    print(f"   tokens {r['tokens']:5}: shard compute {r['ms']:7.3f} ms; all-reduce of {N/1024:7.0f} KiB takes "
          + ', '.join(f"{line[l]:.3f} ms on {l.split(' (')[0]}" for l in list(LINKS)[:3]))
out['exchange'] = ex
save('measure_mps', out)
