"""Chapter 5: why plain LoRA had trouble ending its turn. Qwen2.5-0.5B ties its input embedding and output layer, and LoRA
freezes both. Look at the row of <|im_end|>: is it a trained row, and which tokens sit closest to it?"""
import json, torch
from transformers import AutoModelForCausalLM
from common import Log, save
from ch5_common import tok

log = Log('ch5_special_tokens')
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B', dtype=torch.float32)
E = m.get_input_embeddings().weight.detach()
log(f'tied input/output embeddings: {m.config.tie_word_embeddings}; matrix {tuple(E.shape)}')
norms = E.norm(dim=1)
log(f'mean row norm over the 151,643 ordinary tokens: {norms[:151643].mean():.3f}')
for i in [151643, 151644, 151645]:
    log(f'  {tok.convert_ids_to_tokens(i):15s} id {i}: norm {norms[i]:.3f}')
same = ((norms - norms[151645]).abs() < 1e-3).nonzero().flatten().tolist()
log(f'rows with exactly the same norm as <|im_end|> ({norms[151645]:.4f}): {len(same)}, ids {same[:3]} ... {same[-3:]}')
cos0 = torch.nn.functional.cosine_similarity(E.double(), E[151645][None].double(), dim=1)
d = (E - E[151645][None]).abs().max(dim=1).values
near = ((cos0 > 0.9999) & (d < 1e-3)).nonzero().flatten().tolist()
log(f'rows almost identical to <|im_end|> (cosine > 0.9999 and every number within 0.001): {len(near)} (including itself)')
nb = [i for i in near if i != 151645][:3]
for i in nb:
    log(f'   e.g. id {i} {tok.decode([i])!r}: cosine {cos0[i]:.6f}, largest difference in any of the 896 numbers {d[i]:.2e}')
log(f'so in the frozen output layer these {len(near)} tokens get practically the same logit as <|im_end|>, whatever the hidden state:')
log(f'if they tied exactly, p(<|im_end|>) could not exceed 1/{len(near)}, a loss of at least ln({len(near)}) = {torch.tensor(float(len(near))).log():.2f} nats')
identical = near
cos = torch.nn.functional.cosine_similarity(E, E[151645][None], dim=1)
cos[151645] = -1
top = cos.topk(12)
R = {'norm_mean': norms[:151643].mean().item(), 'norm_imend': norms[151645].item(), 'same_norm': len(same), 'identical': len(identical), 'neighbours': []}
log('the 12 rows closest to <|im_end|> (cosine similarity):')
for c, i in zip(top.values.tolist(), top.indices.tolist()):
    s = tok.decode([i])
    log(f'  id {i:6d}  {s!r:20s} cos {c:.3f}')
    R['neighbours'].append([i, s, c])
plain = json.load(open('results/ch5_sft_lora_plain.json'))
log('what plain LoRA wrote where it should have ended its turn (first checks):')
for r in plain['checks'][:6]:
    log(f'  {r["reply"][:60]!r}')
save('ch5_special_tokens', R)
