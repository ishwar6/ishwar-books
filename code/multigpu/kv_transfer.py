"""Section 7 arithmetic: how many bytes of KV cache must move from a prefill GPU to a decode GPU, how long
that takes on each kind of link, and how much of it layer-by-layer sending can hide behind the prefill.
Pure arithmetic from model shapes and spec-sheet bandwidths; the prefill time uses an assumed 50% of peak FLOP/s.
Run: python code/multigpu/kv_transfer.py  -> results/kv_transfer.json"""
from common import LLAMA, DSV3, GPUS, LINKS, GB, GiB, llama_params, kv_bytes_per_token, dsv3_kv_bytes_per_token, save

G = GPUS['H100 SXM']
MFU = 0.5
out = {'mfu_assumed': MFU}

print('1. CHECK AGAINST DISTSERVE (Section 2.3): OPT-66B, 512-token prompt, 10 requests per second')
opt = dict(L=64, Hkv=72, hd=128)                                   # OPT-66B: 64 layers, 72 heads x 128 = 9216, full MHA
b = 2 * opt['L'] * opt['Hkv'] * opt['hd'] * 2 * 512
print(f'   2 x 64 layers x 9216 x 2 bytes x 512 tokens = {b:,} B = {b/GiB:.3f} GiB (paper: "approximately 1.13GB")')
print(f'   x 10 requests/s = {10*b/GiB:.2f} GiB/s = {10*b*8/GiB:.0f} Gibit/s (paper: 11.3GB per second, "90Gbps"; the paper counts in powers of 2)')
out['distserve_check'] = dict(bytes=b, GiB=b / GiB, gbps_at_10rps=10 * b * 8 / 1e9)


def prefill_s(P, L, d, T, gpus):
    """Dense prefill FLOPs: 2 P T for the weights plus 2 L T^2 d for causal attention (QK^T and AV, half masked)."""
    return (2 * P * T + 2 * L * T * T * d) / (gpus * G['bf16'] * MFU)


print('\n2. KV BYTES AND TRANSFER TIME FOR ONE PROMPT (BF16 cache)')
links = {'NVLink (same node)': LINKS['NVLink 4 (H100, via NVSwitch)'],
         '8 x 400G NICs (TP=8 to TP=8)': 8 * LINKS['InfiniBand NDR 400 Gb/s (one NIC)'],
         '1 x 400G NIC': LINKS['InfiniBand NDR 400 Gb/s (one NIC)'],
         '100G Ethernet': LINKS['Ethernet 100 Gb/s']}
models = {
    'Llama 3.1 8B (1 GPU)': (kv_bytes_per_token(LLAMA['Llama 3.1 8B']), llama_params(LLAMA['Llama 3.1 8B']), 32, 4096, 1),
    'Llama 3.1 70B (TP=8)': (kv_bytes_per_token(LLAMA['Llama 3.1 70B']), llama_params(LLAMA['Llama 3.1 70B']), 80, 8192, 8),
    'DeepSeek-V3 MLA (32 GPUs)': (dsv3_kv_bytes_per_token(), DSV3['active_params'], 61, 7168, 32),
}
rows = []
for name, (per_tok, P, L, d, gpus) in models.items():
    for T in [1024, 8192, 32768]:
        kv = per_tok * T
        pf = prefill_s(P, L, d, T, gpus)
        r = dict(model=name, tokens=T, kv_bytes=kv, prefill_ms=pf * 1e3)
        for ln, B in links.items():
            x = kv / B
            # Layer-by-layer: layer i's KV is sent while layers i+1.. compute. What is left after the prefill ends
            # is the larger of one layer's share and the part of the transfer that did not fit under the compute.
            exposed = max(x / L, x - pf * (L - 1) / L)
            r[ln] = dict(transfer_ms=x * 1e3, exposed_layerwise_ms=exposed * 1e3)
        rows.append(r)
        print(f"   {name:26} {T:6,} tokens: KV {kv/GiB:6.2f} GiB, prefill ~{pf*1e3:7.0f} ms; transfer "
              + ', '.join(f"{ln.split(' (')[0]} {r[ln]['transfer_ms']:7.1f} ms" for ln in links))
out['rows'] = rows

print('\n3. WHAT THE USER FEELS: extra delay before the second token (send all at the end vs layer by layer)')
for r in rows:
    if r['tokens'] != 8192:
        continue
    for ln in ['1 x 400G NIC', '100G Ethernet', 'NVLink (same node)']:
        print(f"   {r['model']:26} 8,192 tokens over {ln:18}: all at the end {r[ln]['transfer_ms']:7.1f} ms, "
              f"layer by layer {r[ln]['exposed_layerwise_ms']:6.2f} ms")
save('kv_transfer', out)
