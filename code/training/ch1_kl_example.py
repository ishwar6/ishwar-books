"""Chapter 1, experiment 5: KL divergence worked out by hand at two real positions.
Position 1: the first answer token for "Give me three tips for sleeping better." (style: how to open the answer).
Position 2: the token after "The capital of Australia is" inside the instruct model's answer (a fact).
For each, the top tokens of both models, and every term p * log(p / q) of KL(P_instruct || P_base)."""
import torch
from common import Log, save
from ch1_models import load, BASE, INSTRUCT, DEV

log = Log('ch1_kl_example')
tok, inst = load(INSTRUCT)
_, base = load(BASE)


def chat(q):
    ids = tok.apply_chat_template([{'role': 'user', 'content': q}], add_generation_prompt=True, tokenize=True, return_dict=False)
    return ids['input_ids'] if isinstance(ids, dict) else ids


@torch.no_grad()
def last_dist(model, ids):
    return torch.log_softmax(model(torch.tensor([ids], device=DEV)).logits[0, -1].float(), -1).cpu()


cases = [('style', chat('Give me three tips for sleeping better.')),
         ('fact', chat('What is the capital of Australia?') + tok('The capital of Australia is')['input_ids'])]
out = []
for name, ids in cases:
    lp, lq = last_dist(inst, ids), last_dist(base, ids)
    p, q = lp.exp(), lq.exp()
    kl = (p * (lp - lq)).sum().item()
    top = torch.topk(p, 4).indices.tolist()
    log(f'\n=== {name}: context ends with {tok.decode(ids[-6:])!r} ===')
    log('token            P_instruct   P_base    p*log(p/q)')
    rows = []
    for i in top:
        term = (p[i] * (lp[i] - lq[i])).item()
        rows.append(dict(tok=tok.decode([i]), p=p[i].item(), q=q[i].item(), term=term))
        log(f'{tok.decode([i])!r:<16} {p[i].item():9.4f}  {q[i].item():9.4f}   {term:9.4f}')
    rest = kl - sum(r['term'] for r in rows)
    log(f'all other {len(p) - 4:,} tokens together: {rest:.4f}')
    log(f'KL(P_instruct || P_base) = {kl:.3f} nats   (reverse direction KL(P_base || P_instruct) = {(q * (lq - lp)).sum().item():.3f})')
    btop = torch.topk(q, 4).indices.tolist()
    log('base model top-4: ' + ', '.join(f'{tok.decode([i])!r} {q[i].item():.3f}' for i in btop))
    out.append(dict(name=name, context=tok.decode(ids[-6:]), rows=rows, rest=rest, kl=kl,
                    base_top=[dict(tok=tok.decode([i]), q=q[i].item(), p=p[i].item()) for i in btop]))
save('ch1_kl_example', out)
