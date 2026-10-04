"""Part 2, the equations with real numbers: attention on a tiny example and in the real model, why we divide by
sqrt(d_k), multi-head and feed-forward shapes, LayerNorm, the input embedding sum for one real token,
cosine similarity, a greedy WordPiece trace, and parameter counts (BERT-base, BERT-large, OpenAI GPT).
Runs on the CPU in under a minute. Writes results/part2_math.json and results/part2_math_stdout.txt."""
import math, torch
import torch.nn.functional as Fn
from transformers import BertConfig, BertModel, BertTokenizer, OpenAIGPTConfig, OpenAIGPTModel
from common import Log, save

torch.manual_seed(0)
torch.set_printoptions(precision=3, sci_mode=False)
log = Log('part2_math')
R = {}
fmt = lambda v: '[' + ', '.join(f'{x:6.3f}' for x in v) + ']'

tok = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').eval()

# ---------------------------------------------------------------- 1. attention on a tiny example (3 tokens, d_k = 4)
log('== 1. scaled dot-product attention, tiny example: 3 tokens, d_k = 4 ==')
words = ['the', 'kid', 'smiles']
Q = torch.tensor([[1., 0., 1., 0.], [0., 2., 0., 1.], [1., 1., 1., 1.]])
K = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 1.], [1., 1., 1., 0.]])
V = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
dk = Q.shape[1]
scores = Q @ K.T                                  # (3, 3): one score per (query, key) pair
scaled = scores / math.sqrt(dk)                   # divide by sqrt(4) = 2
weights = torch.softmax(scaled, dim=-1)           # each row sums to 1
out = weights @ V                                 # (3, 2): a weighted average of the value rows
for name, M in [('Q K^T', scores), ('/ sqrt(d_k)', scaled), ('softmax', weights), ('output = weights V', out)]:
    log(f'  {name}:')
    for w, r in zip(words, M):
        log(f'    {w:<7} ' + ' '.join(f'{x:6.3f}' for x in r))
log(f'  row sums of the weights: {[round(x, 3) for x in weights.sum(-1).tolist()]}')
R['tiny'] = dict(words=words, Q=Q.tolist(), K=K.tolist(), V=V.tolist(), scores=scores.tolist(), scaled=scaled.tolist(),
                 weights=[[round(x, 4) for x in r] for r in weights.tolist()], out=[[round(x, 4) for x in r] for r in out.tolist()])
log('')

# ---------------------------------------------------------------- 2. why divide by sqrt(d_k)
log('== 2. why divide by sqrt(d_k): dot products of random vectors with 64 numbers ==')
g = torch.Generator().manual_seed(0)
q = torch.randn(10000, 64, generator=g)
k = torch.randn(10000, 64, generator=g)
dots = (q * k).sum(-1)
sm_raw = torch.softmax(torch.randn(8, generator=g) * dots.std(), -1)
row = torch.randn(8, generator=torch.Generator().manual_seed(1))
p_raw = torch.softmax(row * 8.0, -1)               # scores with spread 8 (unscaled, d_k = 64)
p_scaled = torch.softmax(row, -1)                  # the same scores divided by sqrt(64) = 8
R['scale'] = dict(std_raw=round(dots.std().item(), 3), std_scaled=round((dots / 8).std().item(), 3),
                  row=[round(x, 3) for x in row.tolist()], p_raw=[round(x, 3) for x in p_raw.tolist()],
                  p_scaled=[round(x, 3) for x in p_scaled.tolist()])
log(f'  10,000 random pairs, each number drawn with mean 0 and spread 1')
log(f'  spread (standard deviation) of q.k:          {dots.std():.3f}   (about sqrt(64) = 8)')
log(f'  spread of q.k / sqrt(64):                     {(dots / 8).std():.3f}')
log(f'  one row of 8 scores (already divided):        {fmt(row.tolist())}')
log(f'  softmax WITHOUT the division (scores x 8):    {fmt(p_raw.tolist())}   max {p_raw.max():.3f}')
log(f'  softmax WITH the division:                    {fmt(p_scaled.tolist())}   max {p_scaled.max():.3f}')
log('')

# ---------------------------------------------------------------- 3. attention in the real model, layer 1
log('== 3. the real model: shapes of attention in layer 1, and one head\'s weights ==')
enc = tok('the kid smiles', return_tensors='pt')
toks = tok.convert_ids_to_tokens(enc.input_ids[0])
E = model.embeddings
lay = model.encoder.layer[0]
sa = lay.attention.self
with torch.no_grad():
    x = E(input_ids=enc.input_ids, token_type_ids=enc.token_type_ids)[0]      # (n, 768)
    Wq = sa.query.weight.T                                                     # (768, 768) = 12 heads x 64
    q = sa.query(x).view(-1, 12, 64).transpose(0, 1)                           # (12, n, 64)
    k = sa.key(x).view(-1, 12, 64).transpose(0, 1)
    v = sa.value(x).view(-1, 12, 64).transpose(0, 1)
    att = torch.softmax(q @ k.transpose(1, 2) / 8.0, -1)                       # (12, n, n)
    heads = (att @ v)                                                          # (12, n, 64)
    joined = heads.transpose(0, 1).reshape(-1, 768)                            # (n, 768)
    mha = lay.attention.output.dense(joined)                                   # times W_O (768 x 768)
n = x.shape[0]
log(f'  tokens: {toks}  (n = {n})')
log(f'  X (input embeddings) {tuple(x.shape)};  W_Q {tuple(Wq.shape)} holds 12 heads of 64 columns')
log(f'  per head: Q {tuple(q.shape[1:])}, K {tuple(k.shape[1:])}, V {tuple(v.shape[1:])}; Q K^T {tuple(att.shape[1:])}; output {tuple(heads.shape[1:])}')
log(f'  12 heads joined: {tuple(joined.shape)}; after W_O {tuple(lay.attention.output.dense.weight.shape)}: {tuple(mha.shape)}')
hsel = 0
log(f'  head {hsel + 1} of layer 1, attention weights (rows: the token looking, columns: the token looked at):')
log('    ' + ' ' * 8 + ''.join(f'{t:>8}' for t in toks))
for t, r in zip(toks, att[hsel]):
    log(f'    {t:<8}' + ''.join(f'{x:8.3f}' for x in r))
R['real_attention'] = dict(tokens=toks, head=hsel + 1, layer=1, weights=[[round(x, 4) for x in r] for r in att[hsel].tolist()],
                           shapes=dict(X=list(x.shape), WQ=list(Wq.shape), Q=list(q.shape[1:]), scores=list(att.shape[1:]),
                                       head_out=list(heads.shape[1:]), joined=list(joined.shape), WO=list(lay.attention.output.dense.weight.shape)))
log('')

# ---------------------------------------------------------------- 4. the feed-forward network on one token
log('== 4. the feed-forward network: 768 -> 3072 -> 768, for the token "kid" in layer 1 ==')
pos = toks.index('kid')
with torch.no_grad():
    h1 = lay.attention.output.LayerNorm(x + mha)                 # after add and normalise
    wide = lay.intermediate.dense(h1[pos])                        # (3072,)
    act = Fn.gelu(wide)
    back = lay.output.dense(act)                                  # (768,)
pos_share = (wide > 0).float().mean().item()
R['ffn'] = dict(token='kid', in_shape=list(h1[pos].shape), wide_shape=list(wide.shape), out_shape=list(back.shape),
                share_positive=round(pos_share, 4), W1=list(lay.intermediate.dense.weight.T.shape), W2=list(lay.output.dense.weight.T.shape),
                first6_wide=[round(v, 3) for v in wide[:6].tolist()], first6_gelu=[round(v, 3) for v in act[:6].tolist()])
log(f'  in {tuple(h1[pos].shape)} -> x W1 + b1 {tuple(wide.shape)} -> GELU -> x W2 + b2 {tuple(back.shape)}')
log(f'  W1 {tuple(lay.intermediate.dense.weight.T.shape)}, W2 {tuple(lay.output.dense.weight.T.shape)}')
log(f'  first 6 of the 3072 numbers before GELU: {fmt(wide[:6].tolist())}')
log(f'  the same 6 after GELU:                   {fmt(act[:6].tolist())}')
log(f'  share of the 3072 that are positive before GELU: {pos_share:.3f}')
log('')

# ---------------------------------------------------------------- 5. the input embedding of one token, number by number
log('== 5. input embedding of "dog" in the Figure 2 pair, first 6 of 768 numbers ==')
e2 = tok('my dog is cute', 'he likes playing', return_tensors='pt')
t2 = tok.convert_ids_to_tokens(e2.input_ids[0])
i = t2.index('dog')
with torch.no_grad():
    tv = E.word_embeddings.weight[e2.input_ids[0, i]]
    sv = E.token_type_embeddings.weight[e2.token_type_ids[0, i]]
    pv = E.position_embeddings.weight[i]
    s = tv + sv + pv
    mu, var = s.mean(), s.var(unbiased=False)
    normed = (s - mu) / torch.sqrt(var + E.LayerNorm.eps)
    ln = E.LayerNorm(s)
    model_e = E(input_ids=e2.input_ids, token_type_ids=e2.token_type_ids)[0, i]
log(f'  token "dog" id {e2.input_ids[0, i].item()}, segment A (0), position {i}')
log(f'  Tok[3899]           {fmt(tv[:6].tolist())}  ...')
log(f'  Seg[A]              {fmt(sv[:6].tolist())}  ...')
log(f'  Pos[{i}]              {fmt(pv[:6].tolist())}  ...')
log(f'  sum                 {fmt(s[:6].tolist())}  ...')
log(f'  mean of all 768 numbers of the sum: {mu:.4f}; spread (standard deviation): {var.sqrt():.4f}')
log(f'  (sum - mean)/spread {fmt(normed[:6].tolist())}  ...')
log(f'  x gamma + beta      {fmt(ln[:6].tolist())}  ...   (LayerNorm output)')
log(f'  model.embeddings    {fmt(model_e[:6].tolist())}  ...   max |difference| {(ln - model_e).abs().max():.1e}')
log(f'  lengths: |Tok| {tv.norm():.3f}, |Seg| {sv.norm():.3f}, |Pos| {pv.norm():.3f}, |sum| {s.norm():.3f}, |LayerNorm(sum)| {ln.norm():.3f}')
R['embed'] = dict(token='dog', id=int(e2.input_ids[0, i]), position=i,
                  tok=[round(v, 4) for v in tv[:6].tolist()], seg=[round(v, 4) for v in sv[:6].tolist()], pos=[round(v, 4) for v in pv[:6].tolist()],
                  sum=[round(v, 4) for v in s[:6].tolist()], mean=round(mu.item(), 4), std=round(var.sqrt().item(), 4),
                  normed=[round(v, 4) for v in normed[:6].tolist()], ln=[round(v, 4) for v in ln[:6].tolist()],
                  norms=dict(tok=round(tv.norm().item(), 3), seg=round(sv.norm().item(), 3), pos=round(pv.norm().item(), 3),
                             sum=round(s.norm().item(), 3), ln=round(ln.norm().item(), 3)))
log('')

# ---------------------------------------------------------------- 6. cosine similarity, by hand
log('== 6. cosine similarity, by hand ==')
a, b = torch.tensor([1., 2., 2.]), torch.tensor([2., 1., 2.])
c = (a @ b) / (a.norm() * b.norm())
log(f'  a = {a.tolist()}, b = {b.tolist()}: a.b = {a @ b:.0f}, |a| = {a.norm():.0f}, |b| = {b.norm():.0f}, cos = {a @ b:.0f}/({a.norm():.0f} x {b.norm():.0f}) = {c:.3f}')
R['cos_tiny'] = dict(a=a.tolist(), b=b.tolist(), dot=(a @ b).item(), na=a.norm().item(), nb=b.norm().item(), cos=round(c.item(), 4))
log('')

# ---------------------------------------------------------------- 7. greedy WordPiece, step by step
log('== 7. greedy longest-match-first WordPiece, every try, for "embeddings" ==')
trace = []
word, start = 'embeddings', 0
while start < len(word):
    end = len(word)
    while end > start:
        piece = ('##' if start > 0 else '') + word[start:end]
        hit = piece in tok.vocab
        trace.append((piece, hit))
        if hit:
            break
        end -= 1
    start = end
pieces = [p for p, h in trace if h]
log(f'  {len(trace)} lookups, {len(pieces)} pieces: {" ".join(pieces)}')
for p, h in trace:
    log(f'    {p:<14} {"in the vocabulary -> keep" if h else "not in the vocabulary"}')
R['wp_trace'] = dict(word=word, trace=trace, pieces=pieces, real=tok.tokenize(word))
log('')

# ---------------------------------------------------------------- 8. parameters, term by term, and GPT for comparison
log('== 8. parameters term by term (H = hidden size, I = 4H feed-forward size, V = vocabulary, P = positions) ==')
terms = {}
for name in ['bert-base-uncased', 'bert-large-uncased']:
    c = BertConfig.from_pretrained(name)
    H, I, V, P, L = c.hidden_size, c.intermediate_size, c.vocab_size, c.max_position_embeddings, c.num_hidden_layers
    t = [('token table V x H', V * H), ('position table P x H', P * H), ('segment table 2 x H', 2 * H), ('embedding LayerNorm 2H', 2 * H),
         ('Q, K, V, O matrices 4 H x H (per layer)', 4 * H * H), ('their biases 4H (per layer)', 4 * H),
         ('FFN W1 H x 4H + W2 4H x H (per layer)', 2 * H * I), ('FFN biases 4H + H (per layer)', I + H),
         ('two LayerNorms 4H (per layer)', 4 * H), ('pooler H x H + H', H * H + H)]
    per_layer = sum(v for k, v in t if 'per layer' in k)
    total = sum(v for k, v in t if 'per layer' not in k) + L * per_layer
    terms[name] = dict(L=L, H=H, terms=t, per_layer=per_layer, total=total)
    log(f'  {name}: L={L} H={H} I={I}')
    for k, v in t:
        log(f'    {k:<42} {v:>13,}')
    log(f'    one layer {per_layer:,} x {L} = {L * per_layer:,};  total {total:,}')
gc = OpenAIGPTConfig.from_pretrained('openai-community/openai-gpt')
with torch.device('meta'):
    gpt = OpenAIGPTModel(gc)
gpt_n = sum(p.numel() for p in gpt.parameters())
log(f'  OpenAI GPT (openai-gpt config): L={gc.n_layer} H={gc.n_embd} heads={gc.n_head} vocabulary={gc.vocab_size} positions={gc.n_positions}')
log(f'    parameters counted on the meta device: {gpt_n:,}')
R['params'] = terms
R['gpt'] = dict(L=gc.n_layer, H=gc.n_embd, A=gc.n_head, vocab=gc.vocab_size, positions=gc.n_positions, params=gpt_n)

save('part2_math', R)
