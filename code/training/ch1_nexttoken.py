"""Chapter 1, experiment 1: what a language model computes.
Qwen2.5-0.5B (base) reads "The capital of France is Paris." and we print, for every position, the probability it gave to the
token that really came next, the chain-rule product, the cross-entropy loss and the perplexity. Also the top-5 candidates
after "The capital of France is" with their raw scores (logits), to work through softmax by hand."""
import math, torch
from common import Log, save
from ch1_models import load, BASE, DEV

log = Log('ch1_nexttoken')
tok, model = load(BASE)
n_params = sum(p.numel() for p in model.parameters())
cfg = model.config
log(f'model: {BASE}   parameters: {n_params:,}   layers: {cfg.num_hidden_layers}   hidden size: {cfg.hidden_size}')
log(f'vocabulary: {len(tok):,} tokens (embedding rows: {cfg.vocab_size:,})')

sentence = 'The capital of France is Paris.'
ids = tok(sentence)['input_ids']
pieces = [tok.decode([i]) for i in ids]
log(f'\nsentence: {sentence!r}')
log('tokens:  ' + ' | '.join(repr(p) for p in pieces))
log('ids:     ' + ' '.join(str(i) for i in ids))

with torch.no_grad():
    logits = model(torch.tensor([ids], device=DEV)).logits[0].float().cpu()   # [T, V]: one score per vocabulary token, per position
probs = torch.softmax(logits, dim=-1)

log('\nposition  context -> actual next token   P(actual)   -log P   model top-1 (P)')
rows, logp_sum = [], 0.0
for t in range(len(ids) - 1):
    p = probs[t, ids[t + 1]].item()
    top = probs[t].argmax().item()
    logp_sum += math.log(p)
    ctx = tok.decode(ids[:t + 1])
    rows.append(dict(context=ctx, next=pieces[t + 1], p=p, nll=-math.log(p), top1=tok.decode([top]), top1_p=probs[t, top].item(),
                     rank=int((probs[t] > p).sum().item()) + 1))
    log(f'{t + 1:>8}  {ctx[-24:]!r:>28} -> {pieces[t + 1]!r:<9} {p:9.4f}  {-math.log(p):7.3f}   {tok.decode([top])!r} ({probs[t, top].item():.3f})')

n = len(ids) - 1
loss = -logp_sum / n
log(f'\nP(sentence after the first token) = product of the {n} probabilities = {math.exp(logp_sum):.3e}')
log(f'sum of log-probabilities = {logp_sum:.3f}')
log(f'average cross-entropy loss = {loss:.3f} nats per token')
log(f'perplexity = exp(loss) = {math.exp(loss):.2f}')

# the same loss, computed the way training code does it
with torch.no_grad():
    out = model(torch.tensor([ids], device=DEV), labels=torch.tensor([ids], device=DEV))
log(f'model(input_ids, labels=input_ids).loss = {out.loss.item():.3f}   (the library shifts the labels by one for us)')

# softmax by hand on the top-5 candidates after "The capital of France is"
t = len(tok('The capital of France is')['input_ids']) - 1
top5 = torch.topk(logits[t], 5)
log('\nafter "The capital of France is": top-5 next tokens')
log('token        logit    exp(logit - max)   softmax over all V   softmax over just these 5')
z = top5.values
e = torch.exp(z - z.max())
cands = []
for k in range(5):
    i = top5.indices[k].item()
    cands.append(dict(tok=tok.decode([i]), logit=z[k].item(), p=probs[t, i].item(), p5=(e[k] / e.sum()).item()))
    log(f'{tok.decode([i])!r:<12} {z[k].item():7.3f}   {e[k].item():10.4f}        {probs[t, i].item():10.4f}           {(e[k] / e.sum()).item():.4f}')
log(f'probability mass of these 5 tokens: {sum(c["p"] for c in cands):.4f}  (the other {cfg.vocab_size - 5:,} share the rest)')

# a less predictable sentence, for contrast
for s in ['The cat sat on the mat.', 'Purple ideas sleep furiously under quiet spoons.']:
    i2 = tok(s)['input_ids']
    with torch.no_grad():
        l2 = model(torch.tensor([i2], device=DEV), labels=torch.tensor([i2], device=DEV)).loss.item()
    log(f'{s!r}: loss {l2:.3f}, perplexity {math.exp(l2):.1f}')

save('ch1_nexttoken', dict(params=n_params, layers=cfg.num_hidden_layers, hidden=cfg.hidden_size, vocab=len(tok),
                           vocab_rows=cfg.vocab_size, sentence=sentence, pieces=pieces, ids=ids, rows=rows,
                           logp_sum=logp_sum, loss=loss, ppl=math.exp(loss), hf_loss=out.loss.item(), top5=cands))
