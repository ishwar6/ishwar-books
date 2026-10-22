"""Part 2: how many conversations fit on a real GPU, from the KV size formula.
Models: Llama 3.1 8B and 70B in BF16 (weight counts worked out from each model's config.json).
KV cache in BF16 (2 bytes per number) or FP8 (1 byte). GPUs: H100 80 GB, H200 141 GB, B200 180 GB (vendor specs).
Assumption: 10% of each GPU's memory is kept for working memory (activations, CUDA graphs, the CUDA context),
so the KV cache gets 90% of memory minus the weights. Weights and KV heads of 70B are split evenly over N GPUs
(tensor parallelism, Part 3 and Part 7). Pure arithmetic. Output: results/p2_gpu_capacity_stdout.txt, .json"""
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
KEEP = 0.10                                       # working memory, an assumption stated in the text
GPUS = {'H100 80GB': 80e9, 'H200 141GB': 141e9, 'B200 180GB': 180e9}
CFG = {  # from huggingface.co/meta-llama/Llama-3.1-8B-Instruct and -70B-Instruct config.json
    'Llama 3.1 8B': dict(layers=32, hidden=4096, inter=14336, vocab=128256, heads=32, kv=8, hd=128),
    'Llama 3.1 70B': dict(layers=80, hidden=8192, inter=28672, vocab=128256, heads=64, kv=8, hd=128),
}


def n_weights(c):
    h, kvd = c['hidden'], c['kv'] * c['hd']
    layer = h * h * 2 + h * kvd * 2 + 3 * h * c['inter'] + 2 * h     # q, o, k, v, the MLP, two norms
    return c['layers'] * layer + 2 * c['vocab'] * h + h              # untied embedding and output head, final norm


out = {}
for m, c in CFG.items():
    w = n_weights(c)
    wb = w * 2
    print(f"{m}: {w:,} weights -> {wb / 1e9:.1f} GB in BF16; KV per token BF16 "
          f"{2 * c['layers'] * c['kv'] * c['hd'] * 2 // 1024} KiB, FP8 {2 * c['layers'] * c['kv'] * c['hd'] // 1024} KiB")
    out[m] = {'weights_gb': wb / 1e9}
print(f"assumption: {KEEP:.0%} of GPU memory kept for working memory\n")

for m, c in CFG.items():
    wb = out[m]['weights_gb'] * 1e9
    for g, cap in GPUS.items():
        # smallest GPU count (1, 2, 4, 8: it must divide the 8 KV heads) whose usable memory holds the weights
        for n in (1, 2, 4, 8):
            if n * cap * (1 - KEEP) > wb:
                break
        rows = []
        for need in sorted({n, 2 * n} if m.endswith('70B') else {1}):
            free = need * cap * (1 - KEEP) - wb
            r = {'gpus': need, 'kv_gb': free / 1e9}
            for dt, bpn in (('bf16', 2), ('fp8', 1)):
                per_tok = 2 * c['layers'] * c['kv'] * c['hd'] * bpn
                r[f'tokens_{dt}'] = int(free // per_tok)
                for L in (4096, 32768):
                    r[f'{dt}_{L}'] = int(free // (per_tok * L))
            rows.append(r)
            print(f"{m:14s} on {need} x {g:10s}: weights {wb / need / 1e9:5.1f} GB per GPU, KV memory {free / 1e9:6.1f} GB in total "
                  f"= {r['tokens_bf16']:>9,} tokens BF16 / {r['tokens_fp8']:>9,} FP8 | conversations that fit: "
                  f"4,096 tok BF16 {r['bf16_4096']:4d}, FP8 {r['fp8_4096']:4d}; 32,768 tok BF16 {r['bf16_32768']:3d}, FP8 {r['fp8_32768']:3d}")
        out[m][g] = rows
    print()

# why 70B does not fit on one GPU
wb = out['Llama 3.1 70B']['weights_gb']
for g, cap in GPUS.items():
    print(f"70B on one {g}: usable {cap * (1 - KEEP) / 1e9:.0f} GB vs weights {wb:.1f} GB -> "
          f"{'fits, ' + format(cap * (1 - KEEP) / 1e9 - wb, '.1f') + ' GB left for KV' if cap * (1 - KEEP) / 1e9 > wb else 'does not fit'}")
json.dump(out, open(os.path.join(HERE, 'results', 'p2_gpu_capacity.json'), 'w'), indent=1)
