"""Part 2: how much memory the models of PPO and of GRPO need, for a 7B policy and for a DeepSeek-V3-size policy.

A rough count of the weights and the optimizer state only (no activations, no KV cache for sampling):
  - a trained model with mixed-precision Adam keeps 16 bytes per parameter: bf16 weights (2) + bf16 gradients (2)
    + fp32 master weights (4) + Adam's two fp32 moments (4 + 4). This is the usual rule from the ZeRO paper.
  - a frozen model only runs forward passes: bf16 weights, 2 bytes per parameter.
PPO: policy (trained) + reference (frozen) + reward model (frozen) + value model (trained, same size as the policy).
GRPO: policy (trained) + reference (frozen) + reward model (frozen); with rule-based rewards (R1-Zero) no reward model.
DeepSeek-V3 has 671B parameters in total (DeepSeek-V3 report). We assume the reward and value models have the
same size as the policy, as the papers say is typical."""
import math
from common import Log, save

log = Log('r1_part2_memory')
TRAINED, FROZEN = 16, 2            # bytes per parameter
R = {}


def table(name, n):
    rows = [
        ('PPO', [('policy', TRAINED), ('reference', FROZEN), ('reward model', FROZEN), ('value model', TRAINED)]),
        ('GRPO, reward model', [('policy', TRAINED), ('reference', FROZEN), ('reward model', FROZEN)]),
        ('GRPO, rule rewards', [('policy', TRAINED), ('reference', FROZEN)]),
    ]
    log(f'== {name}: {n / 1e9:.0f}B parameters per model')
    out = {}
    for algo, parts in rows:
        gb = [(p, n * b / 1e9) for p, b in parts]
        tot = sum(g for _, g in gb)
        out[algo] = tot
        log(f'{algo:20s} ' + ' + '.join(f'{p} {g:,.0f}' for p, g in gb) + f' = {tot:,.0f} GB')
    log(f'GRPO with rule rewards needs {out["GRPO, rule rewards"] / out["PPO"] * 100:.0f}% of the memory of PPO; '
        f'it saves {out["PPO"] - out["GRPO, rule rewards"]:,.0f} GB')
    log(f'80 GB GPUs just to hold these (rounded up): PPO {math.ceil(out["PPO"] / 80):,}, '
        f'GRPO with rule rewards {math.ceil(out["GRPO, rule rewards"] / 80):,}')
    critic = n * TRAINED / 1e9
    log(f'the value model alone: {critic:,.0f} GB, {math.ceil(critic / 80):,} GPUs of 80 GB (rounded up)')
    R[name] = out
    log('')


log('bytes per parameter: trained (mixed-precision Adam) 16, frozen (bf16) 2')
log('')
table('7B policy', 7e9)
table('DeepSeek-V3 size', 671e9)
save('r1_part2_memory', R)
