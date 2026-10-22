"""Part 1, real data-centre GPUs. Vendor specs (dense BF16, no sparsity) -> ridge point, and Llama 3.1 8B decode
at batch 1: the least time per step (weights / bandwidth) and the most tokens per second for one user.
Pure arithmetic on official numbers; no GPU is needed. Output: results/p1_gpus_stdout.txt and results/p1_gpus.json."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
# name: (HBM GB, bandwidth TB/s, dense BF16 TFLOP/s, source)
GPUS = {
    'A100 80GB SXM': (80, 2.039, 312, 'nvidia.com/en-us/data-center/a100: 80GB HBM2e, 2,039 GB/s, BF16 312 TFLOPS (624 with sparsity)'),
    'H100 SXM': (80, 3.35, 989, 'nvidia.com/en-us/data-center/h100: 80GB, 3.35TB/s, BF16 1,979 TFLOPS with sparsity -> 989 dense'),
    'H200 SXM': (141, 4.8, 989, 'nvidia.com/en-us/data-center/h200: 141GB, 4.8TB/s, BF16 1,979 TFLOPS with sparsity -> 989 dense'),
    'B200 (HGX)': (180, 7.7, 2250, 'NVIDIA Blackwell datasheet, HGX B200 per GPU: 180 GB HBM3E, 7.7 TB/s, FP16/BF16 4.5 PFLOPS sparse -> 2,250 dense'),
    'MI300X': (192, 5.3, 1307.4, 'amd.com MI300X page and data sheet: 192 GB HBM3, 5.3 TB/s, BF16 1307.4 TFLOPS dense'),
}
WEIGHTS = 8.03e9 * 2          # Llama 3.1 8B in BF16: 8.03 billion weights x 2 bytes (Part 2)
print(f"Llama 3.1 8B weights in BF16: {WEIGHTS / 1e9:.1f} GB")
print(f"{'GPU':15s} {'HBM GB':>7s} {'TB/s':>6s} {'TFLOP/s':>8s} {'ridge':>6s} {'floor ms':>9s} {'max tok/s':>10s} {'weights % of HBM':>17s}")
out = {}
for name, (gb, bw, tf, src) in GPUS.items():
    ridge = tf / bw                                   # TFLOP/s divided by TB/s = FLOPs per byte
    floor_ms = WEIGHTS / (bw * 1e12) * 1e3            # one full read of the weights
    out[name] = dict(gb=gb, bw=bw, tflops=tf, ridge=ridge, floor_ms=floor_ms, max_tps=1000 / floor_ms, src=src)
    print(f"{name:15s} {gb:7.0f} {bw:6.2f} {tf:8.0f} {ridge:6.0f} {floor_ms:9.2f} {1000 / floor_ms:10.0f} {WEIGHTS / (gb * 1e9):16.0%}")
a, b = GPUS['A100 80GB SXM'], GPUS['B200 (HGX)']
print(f"A100 -> B200: compute x{b[2] / a[2]:.1f}, bandwidth x{b[1] / a[1]:.1f}, ridge {a[2] / a[1]:.0f} -> {b[2] / b[1]:.0f}")
a, h = GPUS['A100 80GB SXM'], GPUS['H100 SXM']
print(f"A100 -> H100: compute x{h[2] / a[2]:.1f}, bandwidth x{h[1] / a[1]:.1f}")
print(f"H100 -> H200: same compute, bandwidth x{4.8 / 3.35:.2f}, ridge {989 / 3.35:.0f} -> {989 / 4.8:.0f}")
print("batch needed to reach the ridge (weights only, about 1 FLOP per byte per sequence):",
      ', '.join(f"{k} {v['ridge']:.0f}" for k, v in out.items()))
for k, (gb, bw, tf, src) in GPUS.items():
    print(f"source {k}: {src}")
json.dump(out, open(os.path.join(HERE, 'results', 'p1_gpus.json'), 'w'), indent=1)
