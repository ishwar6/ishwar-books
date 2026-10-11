"""Shared helpers: a logger that saves what it prints, and the two step-cost models."""
import json
from pathlib import Path
from engine import LinearCost, RooflineCost

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
RES.mkdir(exist_ok=True)


class Log:
    def __init__(self, name, cmd=None):
        self.name, self.lines = name, [f'$ python code/serving/{cmd or name}.py']
        print(self.lines[0])
    def __call__(self, s=''):
        print(s); self.lines.append(str(s))
    def save(self, data):
        (RES / f'{self.name}.json').write_text(json.dumps(data, indent=2, default=float) + '\n')
        (RES / f'{self.name}_stdout.txt').write_text('\n'.join(self.lines) + '\n')


def laptop_cost():
    """Step cost of Qwen2.5-0.5B on this laptop's GPU, fitted in measure.py."""
    m = json.loads((RES / 'measure.json').read_text())['cost_model']
    return LinearCost(m['a_ms'], m['per_new_token_ms'], m['per_kv_pair_ms'], m['per_kv_read_ms'])


# Llama 3.1 8B (from its config.json): 32 layers, hidden 4096, 8 KV heads of 128, 8.03 B parameters,
# of which 2 x 128,256 x 4096 are the input embedding and output head.
LLAMA8B = dict(params=8.03e9, layers=32, hidden=4096, kv_heads=8, head_dim=128, vocab=128256)
LLAMA8B['non_embedding'] = LLAMA8B['params'] - 2 * LLAMA8B['vocab'] * LLAMA8B['hidden']
LLAMA8B['kv_bytes_per_token_bf16'] = 2 * LLAMA8B['layers'] * LLAMA8B['kv_heads'] * LLAMA8B['head_dim'] * 2
# NVIDIA H100 SXM datasheet: 80 GB HBM3, 3.35 TB/s, 989 TFLOPS dense BF16 (1,979 with sparsity).
H100 = dict(mem_bytes=80e9, bw=3.35e12, bf16_flops=989e12, fp8_flops=1979e12)


def h100_cost(weight_bytes_per_param=2, kv_bytes_scale=1.0, overhead_ms=0.0):
    """Roofline bound for Llama 3.1 8B on one H100: max(compute, memory) per step, no overheads.
    The output head (vocab x hidden) also runs per new token; it is counted in the weights read."""
    m = LLAMA8B
    return RooflineCost(n_params=m['params'] - m['vocab'] * m['hidden'],
                        weight_bytes=m['params'] * weight_bytes_per_param,
                        kv_bytes_per_token=m['kv_bytes_per_token_bf16'] * kv_bytes_scale,
                        attn_flops_per_pair=4 * m['layers'] * m['hidden'],
                        peak_flops=H100['bf16_flops'], peak_bw=H100['bw'], overhead_ms=overhead_ms)
