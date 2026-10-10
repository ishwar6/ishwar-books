"""Chapter 5: LoRA arithmetic on the real Qwen2.5-0.5B shapes. Counts the trainable parameters for several ranks (and checks the
count against peft), shows on a toy 4x4 matrix what a rank-1 update is, checks that LoRA starts as an exact no-op, and works out
the training memory for full fine-tuning, LoRA and QLoRA."""
import torch
from transformers import AutoModelForCausalLM
from common import Log, save

log = Log('ch5_lora_math')
R = {}
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B', dtype=torch.float32)
cfg = m.config
log(f'Qwen2.5-0.5B: {cfg.num_hidden_layers} layers, hidden {cfg.hidden_size}, MLP {cfg.intermediate_size}, '
    f'{cfg.num_attention_heads} query heads, {cfg.num_key_value_heads} key/value heads of {cfg.hidden_size // cfg.num_attention_heads}')
layer = m.model.layers[0]
mods = {'q_proj': layer.self_attn.q_proj, 'k_proj': layer.self_attn.k_proj, 'v_proj': layer.self_attn.v_proj,
        'o_proj': layer.self_attn.o_proj, 'gate_proj': layer.mlp.gate_proj, 'up_proj': layer.mlp.up_proj, 'down_proj': layer.mlp.down_proj}
total_all = sum(p.numel() for p in m.parameters())
R['shapes'] = {}
per_layer_full = 0
for k, lin in mods.items():
    d_out, d_in = lin.weight.shape
    R['shapes'][k] = [d_in, d_out]
    per_layer_full += d_in * d_out
    log(f'  {k:9s} W: {d_out:5d} x {d_in:5d} = {d_in * d_out:>10,} weights;  LoRA r=16 adds r(d_in + d_out) = 16 x {d_in + d_out:,} = {16 * (d_in + d_out):,}')
log(f'all parameters: {total_all:,}; the 7 linear layers x {cfg.num_hidden_layers} layers hold {per_layer_full * cfg.num_hidden_layers:,}')
R['ranks'] = []
for r in [1, 2, 4, 8, 16, 64, 256]:
    n = cfg.num_hidden_layers * sum(r * (a + b) for a, b in R['shapes'].values())
    log(f'  rank {r:3d}: {n:>11,} trainable = {n / total_all:.3%} of the model')
    R['ranks'].append([r, n, n / total_all])
R['total'] = total_all

from peft import LoraConfig, get_peft_model
pm = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, target_modules=list(mods), task_type='CAUSAL_LM'))
n_peft = sum(p.numel() for p in pm.parameters() if p.requires_grad)
log(f'peft count for r=16: {n_peft:,} (formula: {R["ranks"][4][1]:,})')
R['peft_r16'] = n_peft
x = torch.randint(0, 1000, (1, 12))
with torch.no_grad():
    base_out = m.get_base_model()(x).logits if hasattr(m, 'get_base_model') else None
    with pm.disable_adapter():
        a = pm(x).logits
    b = pm(x).logits
log(f'at initialisation B = 0, so LoRA changes nothing: max |logits with LoRA - without| = {(a - b).abs().max().item():.1e}')
R['init_diff'] = (a - b).abs().max().item()

log('')
log('== a rank-1 update on a 4 x 4 matrix ==')
torch.manual_seed(3)
B = torch.tensor([[1.0], [0.5], [-1.0], [2.0]])        # d_out x r
A = torch.tensor([[0.2, -0.4, 0.0, 0.6]])              # r x d_in
dW = B @ A
log('B (4x1) =', [round(v, 2) for v in B.flatten().tolist()], '  A (1x4) =', [round(v, 2) for v in A.flatten().tolist()])
for row in dW.tolist():
    log('   ' + '  '.join(f'{v:5.2f}' for v in row))
log(f'16 numbers in Delta W, built from 4 + 4 = 8 numbers; rank {torch.linalg.matrix_rank(dW).item()}')
R['toy'] = {'B': B.flatten().tolist(), 'A': A.flatten().tolist(), 'dW': dW.tolist()}

log('')
log('== training memory (weights + gradients + AdamW state, before activations) ==')
N = total_all
lora = R['ranks'][4][1]
rows = [('full fine-tuning, bf16 mixed precision', N * 16),
        ('LoRA r=16, frozen weights in bf16', N * 2 + lora * 16),
        ('QLoRA r=16, frozen weights in 4-bit NF4', N * 0.5 + lora * 16)]
for name, b in rows:
    log(f'  {name:42s} {b / 2**30:6.2f} GiB')
R['mem'] = rows
for name, n in [('7B', 7e9), ('65B', 65e9)]:
    log(f'  frozen weights alone of a {name} model: bf16 {n * 2 / 2**30:5.0f} GiB, 4-bit {n * 0.5 / 2**30:5.0f} GiB')
    R[f'w_{name}'] = [n * 2 / 2**30, n * 0.5 / 2**30]
save('ch5_lora_math', R)
