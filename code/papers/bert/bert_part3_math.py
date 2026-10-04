"""Part 3, the equations with real numbers.

1. The masked-LM head of bert-base-uncased: shapes, the tied embedding matrix, and one prediction worked by hand
   (scores -> softmax -> cross-entropy), checked against the library.
2. The 80/10/10 rule: the expected share of every kind of token, and the same shares measured (from part3_mlm.json).
3. The NSP head: C -> pooler -> W (2 x 768) -> softmax, worked by hand on the paper's two A.1 examples.
4. The total pre-training loss on the paper's two A.1 examples as one batch: every masked position's loss,
   the two means, their sum, and the loss BertForPreTraining computes.
5. GELU at a few points, exact and with the tanh formula of the released code.
6. Attention cost at length 128 and 512 for BERT-base: score entries, memory for the attention weights, and
   multiply-add work of the attention scores versus the rest of a layer.
Runs on the CPU in a few seconds."""
import json, math, os
import torch
import torch.nn.functional as Fn
from transformers import BertForPreTraining, BertTokenizer
from common import Log, save, RESULTS

log = Log('part3_math')
torch.manual_seed(0)
tok = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForPreTraining.from_pretrained('bert-base-uncased').eval()
V, H = model.config.vocab_size, model.config.hidden_size
out = {}

# ---------------------------------------------------------------- 1. the MLM head, one prediction by hand
head = model.cls.predictions
E = model.bert.embeddings.word_embeddings.weight
tied = head.decoder.weight.data_ptr() == E.data_ptr()
log('1. the masked-LM head of bert-base-uncased')
log(f'   transform: dense {tuple(head.transform.dense.weight.shape)} + GELU + LayerNorm({H})')
log(f'   decoder weight {tuple(head.decoder.weight.shape)}, bias {tuple(head.bias.shape)}')
log(f'   decoder weight is the input token-embedding matrix (same memory): {tied}')

sent = 'the man went to the [MASK] to buy a gallon of milk.'
target = 'store'
enc = tok(sent, return_tensors='pt')
toks = tok.convert_ids_to_tokens(enc.input_ids[0])
pos = toks.index('[MASK]')
with torch.no_grad():
    hid = model.bert(**enc).last_hidden_state[0]          # (n, 768): the T vectors
    T = hid[pos]                                            # T_i, 768 numbers
    h = head.transform(T)                                   # dense + GELU + LayerNorm
    z = h @ E.T + head.bias                                 # 30,522 scores (logits)
    z_lib = model(**enc).prediction_logits[0, pos]
t_id = tok.convert_tokens_to_ids(target)
m = float(z.max())
lse = m + math.log(float(torch.exp(z - m).sum()))           # log of the softmax denominator
p = torch.softmax(z, -1)
top = torch.topk(z, 5)
loss_hand = lse - float(z[t_id])                            # -log softmax(z)[target]
loss_lib = float(Fn.cross_entropy(z_lib[None], torch.tensor([t_id])))
log('')
log(f'   sentence: {" ".join(toks)}   (masked position i = {pos}, original word "{target}")')
log(f'   T_i: {tuple(T.shape)}, first 4 numbers: ' + ', '.join(f'{v:+.3f}' for v in T[:4].tolist()))
log(f'   scores z = h E^T + b: {tuple(z.shape)}; largest score {m:.3f}; smallest {float(z.min()):.3f}')
log('   top 5 scores and their probabilities:')
top5 = []
for v, i in zip(top.values.tolist(), top.indices.tolist()):
    w = tok.convert_ids_to_tokens(i)
    top5.append(dict(word=w, z=v, p=float(p[i]), exp=math.exp(v - m)))
    log(f'     {w:10s} z = {v:7.3f}   exp(z - max) = {math.exp(v - m):.4f}   p = {float(p[i]):.4f}')
log(f'   log of the softmax denominator, log sum_v exp(z_v) = {lse:.4f}')
log(f'   p("{target}") = exp({float(z[t_id]):.3f} - {lse:.3f}) = {float(p[t_id]):.4f}')
log(f'   cross-entropy = -log p("{target}") = {loss_hand:.4f}   (library: {loss_lib:.4f})')
log(f'   rank of "{target}": {int((z > z[t_id]).sum()) + 1} of {V:,}; probability mass in the top 5: {float(p[top.indices].sum()):.4f}')
log(f'   for scale: a uniform guess over {V:,} tokens gives -log(1/{V}) = {math.log(V):.4f}')
out['mlm'] = dict(sentence=toks, pos=pos, target=target, tied=tied, T4=T[:12].tolist(), zmax=m, zmin=float(z.min()), lse=lse,
                  z_target=float(z[t_id]), p_target=float(p[t_id]), loss=loss_hand, loss_lib=loss_lib, top5=top5,
                  uniform=math.log(V), shapes=dict(T=H, dense=[H, H], E=[V, H], b=V))

# ---------------------------------------------------------------- 2. the 80/10/10 budget
log('')
log('2. the 80/10/10 rule: expected share of all tokens')
budget = dict(mask=0.15 * 0.8, random=0.15 * 0.1, same=0.15 * 0.1, untouched=0.85)
for k, v in budget.items():
    log(f'   {k:10s} {v:.3%}')
log(f'   tokens that look like the original text: untouched + same = {budget["untouched"] + budget["same"]:.3%}')
log(f'   tokens that carry a loss (chosen): mask + random + same = {budget["mask"] + budget["random"] + budget["same"]:.3%}')
M = json.load(open(os.path.join(RESULTS, 'part3_mlm.json')))
c = M['counts']
meas = dict(mask=c['mask'] / c['tokens'], random=c['random'] / c['tokens'], same=c['same'] / c['tokens'],
            untouched=(c['tokens'] - c['chosen']) / c['tokens'])
log(f'   measured on {c["tokens"]:,} WikiText-103 tokens (part3_mlm.json, seed 0): ' +
    ', '.join(f'{k} {v:.2%}' for k, v in meas.items()) + f'; look original {meas["untouched"] + meas["same"]:.2%}')
for L_ in (128, 512):
    log(f'   per sequence of {L_} tokens: about {0.15 * L_:.1f} chosen = {0.12 * L_:.1f} [MASK] + {0.015 * L_:.2f} random + {0.015 * L_:.2f} unchanged')
out['budget'] = dict(expected=budget, measured=meas, tokens=c['tokens'])

# ---------------------------------------------------------------- 3. the NSP head by hand
log('')
log('3. the NSP head, worked by hand on the two examples of Appendix A.1')
W, bW = model.cls.seq_relationship.weight, model.cls.seq_relationship.bias
log(f'   pooler: dense {tuple(model.bert.pooler.dense.weight.shape)} + tanh; classifier W {tuple(W.shape)}, b {tuple(bW.shape)}; class 0 = IsNext, 1 = NotNext')
A1 = [('the man went to [MASK] store', 'he bought a gallon [MASK] milk', 0, ['the', 'of']),
      ('the man [MASK] to the store', 'penguin [MASK] are flightless birds', 1, ['went', '##s'])]
nsp = []
for a, b_, y, _ in A1:
    e = tok(a, b_, return_tensors='pt')
    with torch.no_grad():
        o = model.bert(**e)
        Craw = o.last_hidden_state[0, 0]
        C = torch.tanh(model.bert.pooler.dense(Craw))        # the pooled C the NSP head reads
        s = W @ C + bW
        s_lib = model(**e).seq_relationship_logits[0]
    pr = torch.softmax(s, -1)
    l = -math.log(float(pr[y]))
    nsp.append(dict(a=a, b=b_, label=['IsNext', 'NotNext'][y], scores=s.tolist(), p=pr.tolist(), loss=l, scores_lib=s_lib.tolist(), C=C[:12].tolist()))
    log(f'   A: {a} | B: {b_} | label {["IsNext", "NotNext"][y]}')
    log(f'     C: ({H},), first 4: ' + ', '.join(f'{v:+.3f}' for v in C[:4].tolist()))
    log(f'     scores W C + b = [{s[0]:.3f}, {s[1]:.3f}]   (library: [{s_lib[0]:.3f}, {s_lib[1]:.3f}])')
    log(f'     softmax: P(IsNext) = {pr[0]:.6f}, P(NotNext) = {pr[1]:.6f}; loss -log P({["IsNext", "NotNext"][y]}) = {l:.6f}')
out['nsp'] = nsp

# ---------------------------------------------------------------- 4. the total loss on one small batch
log('')
log('4. the pre-training loss on the two A.1 examples as one batch')
log('   (the paper does not print the hidden words; we restore the obvious ones: "the", "of", "went", and "##s" of "penguins")')
enc = tok([x[0] for x in A1], [x[1] for x in A1], padding=True, return_tensors='pt')
labels = torch.full_like(enc.input_ids, -100)
for r, (_, _, _, words) in enumerate(A1):
    where = (enc.input_ids[r] == tok.mask_token_id).nonzero().flatten().tolist()
    for i, w in zip(where, words):
        labels[r, i] = tok.convert_tokens_to_ids(w)
y = torch.tensor([x[2] for x in A1])
with torch.no_grad():
    o = model(**enc, labels=labels, next_sentence_label=y)
per = []
for r in range(2):
    for i in (labels[r] != -100).nonzero().flatten().tolist():
        l = float(Fn.cross_entropy(o.prediction_logits[r, i][None], labels[r, i][None]))
        w = tok.convert_ids_to_tokens(int(labels[r, i]))
        per.append(dict(pair=r + 1, pos=i, word=w, loss=l, p=math.exp(-l)))
        log(f'   pair {r + 1}, position {i:2d}, target "{w}": p = {math.exp(-l):.4f}, loss {l:.4f}')
mlm_mean = sum(x['loss'] for x in per) / len(per)
nsp_per = [float(Fn.cross_entropy(o.seq_relationship_logits[r][None], y[r][None])) for r in range(2)]
nsp_mean = sum(nsp_per) / 2
log(f'   mean MLM loss over {len(per)} masked positions: {mlm_mean:.4f}')
log('   NSP loss per pair: ' + ', '.join(f'{v:.6f}' for v in nsp_per) + f'; mean {nsp_mean:.6f}')
log(f'   total = {mlm_mean:.4f} + {nsp_mean:.6f} = {mlm_mean + nsp_mean:.4f}   (BertForPreTraining: {o.loss.item():.4f})')
out['batch'] = dict(per=per, mlm_mean=mlm_mean, nsp=nsp_per, nsp_mean=nsp_mean, total=mlm_mean + nsp_mean, lib=o.loss.item())

# ---------------------------------------------------------------- 5. GELU
log('')
log('5. GELU(x) = x * Phi(x), and the tanh formula of the released code')
g = []
for x in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0):
    Phi = 0.5 * (1 + math.erf(x / math.sqrt(2)))
    th = 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
    g.append(dict(x=x, Phi=Phi, gelu=x * Phi, tanh=th))
    log(f'   x = {x:+.1f}: Phi(x) = {Phi:.4f}, GELU = {x:+.1f} x {Phi:.4f} = {x * Phi:+.4f}; tanh formula {th:+.4f}; ReLU {max(0, x):+.1f}')
out['gelu'] = g

# ---------------------------------------------------------------- 6. attention cost
log('')
log('6. attention cost in BERT-base (L = 12 layers, A = 12 heads, H = 768), multiply-adds per layer')
cost = {}
for n in (128, 512):
    scores = n * n
    attn_mats = 12 * scores                     # one n x n weight matrix per head, per layer
    lin = 12 * n * H * H                         # Q, K, V, output (4 n H^2) + feed-forward (8 n H^2)
    att = 2 * n * n * H                          # Q K^T (n^2 H) + weights x V (n^2 H)
    cost[n] = dict(scores_per_head=scores, weights_per_layer=attn_mats, all_layers=12 * attn_mats,
                   linear=lin, attention=att, share=att / (lin + att), per_token=(lin + att) / n)
    log(f'   n = {n}: scores per head {scores:,}; per layer (12 heads) {attn_mats:,}; all 12 layers {12 * attn_mats:,} numbers')
    log(f'      multiply-adds per layer: projections + feed-forward 12 n H^2 = {lin:,}; attention scores and mixing 2 n^2 H = {att:,}')
    log(f'      attention share of a layer\'s work: {att / (lin + att):.1%}; work per token {(lin + att) / n:,.0f}')
r = cost[512]
q = cost[128]
log(f'   512 vs 128: score matrix {r["scores_per_head"] / q["scores_per_head"]:.0f}x, whole-sequence work {(r["linear"] + r["attention"]) / (q["linear"] + q["attention"]):.2f}x '
    f'(4x the tokens), work per token {r["per_token"] / q["per_token"]:.2f}x')
out['cost'] = {str(k): v for k, v in cost.items()}

save('part3_math', out)
