"""Part 1, the maths with real numbers.

1. The language-model chain rule on GPT-2: P(sentence) = product of P(word | words on its left).
2. The masked-word contrast on BERT: P(word | words on BOTH sides).
3. The attention mask on "the kid smiles": one GPT-2 head recomputed by hand (scores, mask, softmax),
   checked against the library, and the head-averaged first-layer attention of GPT-2 (causal) and BERT (bidirectional).
Runs on the CPU, so the numbers are the same on every machine (up to tiny float differences)."""
import math
import torch
from transformers import AutoTokenizer, BertForMaskedLM, GPT2LMHeadModel
from common import Log, save

torch.manual_seed(0)
log = Log('part1_math')
R = {}

gtok = AutoTokenizer.from_pretrained('openai-community/gpt2')
gpt2 = GPT2LMHeadModel.from_pretrained('openai-community/gpt2', attn_implementation='eager').eval()
btok = AutoTokenizer.from_pretrained('bert-base-uncased')
bert = BertForMaskedLM.from_pretrained('bert-base-uncased', attn_implementation='eager').eval()

# ---------------------------------------------------------------- 1. chain rule on GPT-2
SENT = 'The kid smiles at the dog.'
ids = [gtok.bos_token_id] + gtok(SENT).input_ids          # <|endoftext|> marks "start of text", so word 1 also gets a probability
with torch.no_grad():
    logp = torch.log_softmax(gpt2(torch.tensor([ids])).logits[0], -1)
log(f'1. The chain rule on GPT-2: "{SENT}"')
log(f'   vocabulary size V = {gpt2.config.vocab_size:,}; one output vector of V scores per position')
log(f'   {"i":>2}  {"word":<8} {"given (left side only)":<26} {"P(word | left)":>14} {"log P":>8}')
rows, total = [], 0.0
for i in range(1, len(ids)):
    lp = float(logp[i - 1, ids[i]])
    total += lp
    left = gtok.decode(ids[1:i]) or '(start)'
    w = gtok.decode([ids[i]])
    rows.append(dict(i=i, word=w.strip(), left=left, p=round(math.exp(lp), 6), logp=round(lp, 3)))
    log(f'   {i:>2}  {w.strip():<8} {left:<26} {math.exp(lp):>14.6f} {lp:>8.3f}')
n = len(ids) - 1
log(f'   n = {n} tokens')
log(f'   sum of log P = {total:.3f}')
log(f'   P(sentence) = exp({total:.3f}) = {math.exp(total):.3e}')
log(f'   average negative log-likelihood = {-total / n:.3f}  (perplexity exp of that = {math.exp(-total / n):.1f})')
R['chain'] = dict(sentence=SENT, rows=rows, n=n, sum_logp=round(total, 3), p_sentence=f'{math.exp(total):.3e}',
                  mean_nll=round(-total / n, 3), ppl=round(math.exp(-total / n), 1), vocab=gpt2.config.vocab_size)

# ---------------------------------------------------------------- 2. the masked contrast on BERT
log('')
log('2. The same word, two ways of predicting it: "smiles"')
g_left = gtok('The kid').input_ids
with torch.no_grad():
    pg = torch.softmax(gpt2(torch.tensor([[gtok.bos_token_id] + g_left])).logits[0, -1], -1)
sm = gtok(' smiles').input_ids
assert len(sm) == 1
p_gpt = float(pg[sm[0]])
top_g = [(gtok.decode([int(t)]).strip(), round(float(v), 4)) for v, t in zip(*pg.topk(5))]
enc = btok('the kid [MASK] at the dog.', return_tensors='pt')
pos = (enc.input_ids[0] == btok.mask_token_id).nonzero().item()
with torch.no_grad():
    out = bert(**enc, output_hidden_states=True)
pb = torch.softmax(out.logits[0, pos], -1)
p_bert = float(pb[btok.convert_tokens_to_ids('smiles')])
top_b = [(btok.decode([int(t)]).strip(), round(float(v), 4)) for v, t in zip(*pb.topk(5))]
log(f'   GPT-2  P(smiles | "The kid")                    = {p_gpt:.4f}   top 5: ' + ', '.join(f'{w} {p:.4f}' for w, p in top_g))
log(f'   BERT   P(smiles | "the kid [MASK] at the dog.") = {p_bert:.4f}   top 5: ' + ', '.join(f'{w} {p:.4f}' for w, p in top_b))
log(f'   BERT shapes: T_i has {out.hidden_states[-1].shape[-1]} numbers; output scores over V = {bert.config.vocab_size:,} WordPiece tokens')
R['masked'] = dict(p_gpt=round(p_gpt, 4), top_gpt=top_g, p_bert=round(p_bert, 4), top_bert=top_b,
                   H=out.hidden_states[-1].shape[-1], V=bert.config.vocab_size)

# ---------------------------------------------------------------- 3. the attention mask on "the kid smiles"
log('')
log('3. Attention on "the kid smiles": rows = queries (the word that looks), columns = keys (the word looked at)')
g = gtok('the kid smiles', return_tensors='pt').input_ids             # 3 tokens, no special tokens in GPT-2
toks = [gtok.decode([int(t)]).strip() for t in g[0]]
assert toks == ['the', 'kid', 'smiles'], toks
with torch.no_grad():
    o = gpt2(g, output_attentions=True)
    blk = gpt2.transformer.h[0]
    x = gpt2.transformer.wte(g) + gpt2.transformer.wpe(torch.arange(3)[None])
    q, k, v = blk.attn.c_attn(blk.ln_1(x)).split(768, dim=2)
HEAD = 0
d = 64
qh, kh = q[0, :, HEAD * d:(HEAD + 1) * d], k[0, :, HEAD * d:(HEAD + 1) * d]
S = (qh @ kh.T) / math.sqrt(d)
M = torch.triu(torch.full((3, 3), float('-inf')), diagonal=1)
A = torch.softmax(S + M, -1)
lib = o.attentions[0][0, HEAD]
log(f'   GPT-2 layer 1, head {HEAD + 1}: query and key vectors have d_k = {d} numbers each')
fmt = lambda r: '  '.join(f'{float(z):7.3f}' if math.isfinite(float(z)) else '   -inf' for z in r)
log('   scores s = q.k / sqrt(64):')
for t, r in zip(toks, S):
    log(f'     {t:<7}{fmt(r)}')
log('   mask M (0 = allowed, -inf = blocked: a key to the right of the query):')
for t, r in zip(toks, M):
    log(f'     {t:<7}{fmt(r)}')
log('   weights a = softmax(s + M), row by row:')
for t, r in zip(toks, A):
    log(f'     {t:<7}{fmt(r)}   (row sum {float(r.sum()):.3f})')
log(f'   max difference from the library\'s own attention weights: {float((A - lib).abs().max()):.1e}')
R['head'] = dict(toks=toks, head=HEAD + 1, scores=[[round(float(z), 3) for z in r] for r in S],
                 weights=[[round(float(z), 3) for z in r] for r in A], check=float((A - lib).abs().max()))

ga = o.attentions[0][0].mean(0)
e = btok('the kid smiles', return_tensors='pt')
with torch.no_grad():
    ba = bert(**e, output_attentions=True).attentions[0][0].mean(0)
btoks = btok.convert_ids_to_tokens(e.input_ids[0])
log('')
log('   layer 1, averaged over the 12 heads')
log('   GPT-2 (causal):')
for t, r in zip(toks, ga):
    log(f'     {t:<7}' + '  '.join(f'{float(z):5.3f}' for z in r))
log('   BERT (bidirectional):            ' + '  '.join(f'{t:>7}' for t in btoks))
for t, r in zip(btoks, ba):
    log(f'     {t:<7}' + '  '.join(f'{float(z):7.3f}' for z in r))
R['avg'] = dict(gpt_toks=toks, gpt=[[round(float(z), 3) for z in r] for r in ga],
                bert_toks=btoks, bert=[[round(float(z), 3) for z in r] for r in ba])
save('part1_math', R)
