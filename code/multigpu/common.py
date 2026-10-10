"""Shared model shapes and hardware numbers for Part 7. Every value names its source.
Model shapes: Llama 3 paper (arXiv 2407.21783) Table 3; DeepSeek-V3 report (arXiv 2412.19437) Section 4.2;
Qwen2.5-0.5B and OLMoE-1B-7B from their Hugging Face config.json files.
Hardware: NVIDIA product pages (links in README.md). These are peak spec-sheet numbers, not measurements."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
RES.mkdir(exist_ok=True)

GB = 1e9
GiB = 2 ** 30

# Dense Llama 3 models: layers, model dim, FFN dim, query heads, KV heads, head dim, vocab (paper Table 3).
LLAMA = {
    'Llama 3.1 8B':   dict(L=32,  d=4096,  ffn=14336, H=32,  Hkv=8, hd=128, vocab=128256),
    'Llama 3.1 70B':  dict(L=80,  d=8192,  ffn=28672, H=64,  Hkv=8, hd=128, vocab=128256),
    'Llama 3.1 405B': dict(L=126, d=16384, ffn=53248, H=128, Hkv=8, hd=128, vocab=128256),
}
# DeepSeek-V3 (report Section 4.2): 61 layers, hidden 7168, MLA caches a 512-wide latent plus a 64-wide RoPE key
# per token per layer; 1 shared + 256 routed experts (top-8), expert FFN width 2048; first 3 layers dense.
DSV3 = dict(L=61, d=7168, kv_latent=512, rope=64, experts=256, topk=8, expert_ffn=2048, dense_layers=3,
            total_params=671e9, active_params=37e9)

# Peak numbers from NVIDIA spec pages (see README for links). Bandwidths in bytes/s.
GPUS = {
    'H100 SXM': dict(mem=80 * GB, hbm=3.35e12, bf16=989e12, nvlink=900e9, pcie=128e9),
    'H200 SXM': dict(mem=141 * GB, hbm=4.8e12, bf16=989e12, nvlink=900e9, pcie=128e9),
}
# Links, per direction (the spec-sheet NVLink and PCIe figures are both directions added together).
LINKS = {
    'NVLink 4 (H100, via NVSwitch)': 450e9,
    'PCIe Gen5 x16': 64e9,
    'InfiniBand NDR 400 Gb/s (one NIC)': 50e9,
    'Ethernet 100 Gb/s': 12.5e9,
}


def save(name, obj):
    (RES / f'{name}.json').write_text(json.dumps(obj, indent=2) + '\n')


def llama_params(c):
    d, L = c['d'], c['L']
    attn = d * c['H'] * c['hd'] + 2 * d * c['Hkv'] * c['hd'] + c['H'] * c['hd'] * d
    mlp = 3 * d * c['ffn']
    norms = 2 * d
    return L * (attn + mlp + norms) + 2 * c['vocab'] * d + d   # untied input and output embeddings


def kv_bytes_per_token(c, bytes_per=2):
    """Dense GQA model: 2 (K and V) x layers x KV heads x head dim x bytes."""
    return 2 * c['L'] * c['Hkv'] * c['hd'] * bytes_per


def dsv3_kv_bytes_per_token(bytes_per=2):
    """MLA: one latent (512) plus one RoPE key (64) per layer, shared by all heads."""
    return DSV3['L'] * (DSV3['kv_latent'] + DSV3['rope']) * bytes_per
