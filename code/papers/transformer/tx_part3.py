"""Part 3 (attention, §3.2): real numbers for every equation of the section.

1. Eq. 1 by hand on three tiny tokens (numbers a reader can check).
2. Footnote 4: the dot product of random vectors has variance d_k; what that does to the softmax and its gradient.
3. The released MarianMT en-de model (paper base shapes): heads, head size, the four projection matrices.
4. Rebuild multi-head self-attention of encoder layer 1 by hand from the weights and match the library output.
5. The three uses of attention (§3.2.3): encoder self-attention, causal decoder self-attention, cross-attention.
Marian is not the paper's model: same shapes, different data, training and tokenizer."""
import math, torch
from transformers import MarianMTModel, MarianTokenizer
from common import Log, save

torch.manual_seed(0)
log = Log('part3')
R = {}

# ---------------------------------------------------------------- 1. Eq. 1 by hand
log('$ python tx_part3.py')
log('== 1. Eq. 1 on three tokens, d_k = 2')
Q = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
K = torch.tensor([[2., 0.], [0., 2.], [1., 1.]])
V = torch.tensor([[10., 0.], [0., 10.], [5., 5.]])
S = Q @ K.T / math.sqrt(2)
W = S.softmax(-1)
O = W @ V
log('scores QK^T/sqrt(2):', [[round(x, 3) for x in r] for r in S.tolist()])
log('weights (softmax, row 1):', [round(x, 3) for x in W[0].tolist()])
log('output row 1:', [round(x, 2) for x in O[0].tolist()])
R['tiny'] = dict(S=S.tolist(), W=W.tolist(), O=O.tolist())

# ---------------------------------------------------------------- 2. footnote 4: variance of q.k
log('== 2. dot products of random vectors (mean 0, variance 1), 100,000 pairs each')
R['var'] = []
for dk in [4, 16, 64, 512]:
    q, k = torch.randn(100_000, dk), torch.randn(100_000, dk)
    dot = (q * k).sum(-1)
    log(f'd_k={dk:4d}  mean {dot.mean():+.3f}  variance {dot.var():8.2f}  std {dot.std():6.2f}  '
        f'scaled std {(dot / math.sqrt(dk)).std():.3f}')
    R['var'].append(dict(dk=dk, var=dot.var().item(), std=dot.std().item()))

for dk in [64, 512]:
  log(f'== softmax over 10 keys, one query, d_k = {dk} (same random draw, scaled and not scaled)')
  q, keys = torch.randn(dk), torch.randn(10, dk)
  raw = keys @ q
  for name, s in [(f'unscaled{dk}', raw), (f'scaled{dk}', raw / math.sqrt(dk))]:
    s = s.clone()
    p = s.softmax(-1)
    J = torch.autograd.functional.jacobian(lambda x: x.softmax(-1), s)
    log(f'{name:11s} scores range [{s.min():+.1f}, {s.max():+.1f}]  top weight {p.max():.4f}  '
        f'largest gradient entry {J.abs().max():.4f}')
    R[name] = dict(top=p.max().item(), grad=J.abs().max().item(), lo=s.min().item(), hi=s.max().item())

# ---------------------------------------------------------------- 3. the released model
log('== 3. Helsinki-NLP/opus-mt-en-de (MarianMT)')
name = 'Helsinki-NLP/opus-mt-en-de'
tok = MarianTokenizer.from_pretrained(name)
model = MarianMTModel.from_pretrained(name, attn_implementation='eager').eval()
c = model.config
att = model.model.encoder.layers[0].self_attn
log(f'd_model {c.d_model}, heads {c.encoder_attention_heads}, d_k = d_v = {c.d_model // c.encoder_attention_heads}, '
    f'encoder layers {c.encoder_layers}, decoder layers {c.decoder_layers}')
for p in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
    log(f'  {p:8s} weight {tuple(getattr(att, p).weight.shape)}')
n_att = sum(p.numel() for p in att.parameters())
log(f'parameters in one multi-head attention block: {n_att:,} (4 x 512 x 512 + 4 x 512 = {4*512*512+4*512:,})')
R['shapes'] = dict(d_model=c.d_model, heads=c.encoder_attention_heads, params=n_att)

src = 'The cat sat on the mat because it was tired.'
enc = tok(src, return_tensors='pt')
src_toks = tok.convert_ids_to_tokens(enc.input_ids[0])
log('source tokens:', src_toks)

grab = {}
def hook(mod, args, kwargs, out):
    grab['x'] = kwargs.get('hidden_states', args[0] if args else None)
    grab['y'] = out[0]
h = att.register_forward_hook(hook, with_kwargs=True)
with torch.no_grad():
    gen = model.generate(**enc, num_beams=1, max_new_tokens=40)
h.remove()
tgt = tok.decode(gen[0], skip_special_tokens=True)
log('translation (greedy):', tgt)

# ---------------------------------------------------------------- 4. multi-head attention by hand
log('== 4. encoder layer 1 self-attention rebuilt by hand')
x = grab['x'][0]                       # n x 512
n, H, dh = x.shape[0], c.encoder_attention_heads, c.d_model // c.encoder_attention_heads
with torch.no_grad():
    q = att.q_proj(x).view(n, H, dh).transpose(0, 1)    # 8 x n x 64
    k = att.k_proj(x).view(n, H, dh).transpose(0, 1)
    v = att.v_proj(x).view(n, H, dh).transpose(0, 1)
    w = (q @ k.transpose(1, 2) / math.sqrt(dh)).softmax(-1)  # 8 x n x n
    heads = w @ v                                        # 8 x n x 64
    concat = heads.transpose(0, 1).reshape(n, H * dh)    # n x 512
    y = att.out_proj(concat)
diff = (y - grab['y'][0]).abs().max().item()
log(f'x {tuple(x.shape)} -> Q,K,V per head {tuple(q.shape)} -> weights {tuple(w.shape)} -> heads {tuple(heads.shape)}'
    f' -> concat {tuple(concat.shape)} -> W^O -> {tuple(y.shape)}')
log(f'largest difference from the library output: {diff:.1e}')
R['rebuild_diff'] = diff
j = src_toks.index('▁it')
log(f'layer 1, query "it" (position {j}): top key per head')
R['it_heads'] = []
for hh in range(H):
    top = w[hh, j].topk(2)
    pair = [(src_toks[i], round(p, 2)) for p, i in zip(top.values.tolist(), top.indices.tolist())]
    log(f'  head {hh+1}: {pair}')
    R['it_heads'].append(pair)

# ---------------------------------------------------------------- 5. the three uses of attention
log('== 5. the three uses (full forward pass on the source and the translation)')
with torch.no_grad():
    out = model(**enc, decoder_input_ids=gen[:, :-1], output_attentions=True)
tgt_in = tok.convert_ids_to_tokens(gen[0, :-1])
log('decoder input tokens:', tgt_in)
log(f'encoder self-attention weights per layer: {tuple(out.encoder_attentions[0].shape)}  (batch, heads, n_src, n_src)')
log(f'decoder self-attention weights per layer: {tuple(out.decoder_attentions[0].shape)}  (batch, heads, n_tgt, n_tgt)')
log(f'cross-attention weights per layer:        {tuple(out.cross_attentions[0].shape)}  (batch, heads, n_tgt, n_src)')
ds = out.decoder_attentions[5][0].mean(0)              # last layer, mean over heads
above = torch.triu(torch.ones_like(ds), 1).bool()
log(f'decoder self-attention, weight above the diagonal (future tokens): max {ds[above].max().item():.1e}')
log('decoder self-attention, last layer, mean of heads, first 5 rows x 5 columns:')
for r in range(5):
    log('  ' + ' '.join(f'{v:.2f}' for v in ds[r, :5].tolist()) + f'   <- query {tgt_in[r]}')
R['dec_self'] = dict(tokens=tgt_in, W=ds.tolist())
# cross-attention: average of heads in decoder layer 3 (a middle layer)
L = 3
ca = out.cross_attentions[L - 1][0].mean(0)            # n_tgt x n_src
log(f'cross-attention, decoder layer {L}, mean of heads: the source token each German token attends to most')
for r in range(1, len(tgt_in)):
    i = ca[r].argmax().item()
    log(f'  query {tgt_in[r]:12s} -> {src_toks[i]:10s} ({ca[r, i]:.2f})')
R['cross'] = dict(layer=L, tgt=tgt_in, src=src_toks, W=ca.tolist())
es = out.encoder_attentions[0][0]
R['enc_self'] = dict(src=src_toks, W=es.mean(0).tolist())
save('part3', R)
