"""Part 3, code walkthroughs: attention written from scratch (Method 1), then with the library (Method 2),
and a check that both give the same numbers.
1. Equation 1 in NumPy on the three tokens of the worked example, with every matrix printed.
2. The same function in PyTorch against torch.nn.functional.scaled_dot_product_attention.
3. The causal mask built by hand and applied before the softmax, against the library's is_causal=True.
4. Multi-head attention from scratch (project, split heads, attend, merge, W^O) with the shape after every line,
   against torch.nn.MultiheadAttention with the same weights.
5. Our function on the real weights of Helsinki-NLP/opus-mt-en-de (MarianMT): encoder self-attention,
   masked decoder self-attention and cross-attention, against the library's own layers.
6. Three common mistakes, each with its wrong output: no scale, mask after the softmax, wrong transpose.
Saves results/part3b.json and results/part3b_stdout.txt. Runs on a CPU in a few seconds."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from common import Log, save

torch.manual_seed(0)
torch.set_printoptions(precision=3, sci_mode=False)
log = Log('part3b')
R = {}


def show(name, M, fmt='{:7.3f}'):
    """Print a small matrix, one row per line."""
    M = np.asarray(M, dtype=float)
    log(f'{name}  shape {M.shape}')
    for row in np.atleast_2d(M):
        log('   ' + ' '.join(('   -inf' if np.isneginf(v) else fmt.format(v)) for v in row))


log('$ python tx_part3b.py')

# ---------------------------------------------------------------- 1. Equation 1 in NumPy, every matrix printed
log('== 1. Method 1a: Equation 1 in NumPy, three tokens, d_k = 2')
Q = np.array([[1., 0.], [0., 1.], [1., 1.]])     # 3 queries, one row per token
K = np.array([[2., 0.], [0., 2.], [1., 1.]])     # 3 keys
V = np.array([[10., 0.], [0., 10.], [5., 5.]])   # 3 values
d_k = Q.shape[1]                                  # 2 numbers per query and key

scores = Q @ K.T                                  # (3, 3): row i, column j = query i . key j
show('Q K^T', scores)
scaled = scores / np.sqrt(d_k)                    # divide every score by sqrt(2) = 1.414
show('Q K^T / sqrt(d_k)', scaled)
e = np.exp(scaled - scaled.max(axis=1, keepdims=True))  # subtract the row maximum first: same softmax, no overflow
weights = e / e.sum(axis=1, keepdims=True)        # each row now adds up to 1
show('weights = softmax(rows)', weights)
log('row sums:', np.round(weights.sum(axis=1), 6).tolist())
out = weights @ V                                 # (3, 2): each output row is a weighted average of the value rows
show('output = weights V', out, '{:7.2f}')
R['np'] = dict(scores=scores.tolist(), scaled=scaled.tolist(), weights=weights.tolist(), out=out.tolist())


# ---------------------------------------------------------------- 2. the same in PyTorch, and the library function
def attention(Q, K, V, mask=None):
    """Equation 1: softmax(Q K^T / sqrt(d_k)) V.
    Q: (..., n_q, d_k)   K: (..., n_k, d_k)   V: (..., n_k, d_v)
    mask: True where a query may look at a key, False where it may not (None = no mask)."""
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)       # (..., n_q, n_k): every query against every key
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))    # forbidden pairs get minus infinity, BEFORE the softmax
    weights = torch.softmax(scores, dim=-1)                  # (..., n_q, n_k): each row adds up to 1
    return weights @ V, weights                              # (..., n_q, d_v), (..., n_q, n_k)


log('== 2. Method 1b (our PyTorch function) against Method 2 (F.scaled_dot_product_attention)')
Qt, Kt, Vt = (torch.tensor(a, dtype=torch.float32) for a in (Q, K, V))
ours, _ = attention(Qt, Kt, Vt)                       # Method 1: our function
lib = F.scaled_dot_product_attention(Qt, Kt, Vt)       # Method 2: the library
d1 = (ours - lib).abs().max().item()                   # the largest difference over all outputs
log('three tokens, ours   :', [[round(v, 4) for v in r] for r in ours.tolist()])
log('three tokens, library:', [[round(v, 4) for v in r] for r in lib.tolist()])
log(f'max abs difference: {d1:.1e}')
q, k, v = torch.randn(2, 8, 10, 64), torch.randn(2, 8, 10, 64), torch.randn(2, 8, 10, 64)   # (batch, heads, n, d_k)
ours, _ = attention(q, k, v)
lib = F.scaled_dot_product_attention(q, k, v)
d2 = (ours - lib).abs().max().item()
log(f'random (batch 2, heads 8, n 10, d_k 64): output {tuple(lib.shape)}, max abs difference {d2:.1e}')
R['sdpa_diff'] = d2

# ---------------------------------------------------------------- 3. the causal mask, by hand and with the library
log('== 3. The causal mask, built by hand (n = 5, d_k = 4)')
n = 5
allowed = torch.zeros(n, n, dtype=torch.bool)
for i in range(n):                 # query position i ...
    for j in range(n):             # ... may look at key position j only if j is not in the future
        allowed[i, j] = j <= i
log('same as torch.tril(ones):', torch.equal(allowed, torch.tril(torch.ones(n, n, dtype=torch.bool))))
show('allowed (1 = may look, 0 = future)', allowed.int(), '{:2.0f}')
g = torch.Generator().manual_seed(3)
q, k, v = torch.randn(n, 4, generator=g), torch.randn(n, 4, generator=g), torch.randn(n, 4, generator=g)
scores = q @ k.T / math.sqrt(4)
show('scores before the mask', scores)
masked = scores.masked_fill(~allowed, float('-inf'))
show('scores after the mask', masked)
out_mask, w_mask = attention(q, k, v, allowed)
show('weights', w_mask)
lib_causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)
lib_bool = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
d_causal = (out_mask - lib_causal).abs().max().item()
d_bool = (out_mask - lib_bool).abs().max().item()
log(f'ours vs library is_causal=True: max abs difference {d_causal:.1e}')
log(f'ours vs library attn_mask=allowed: max abs difference {d_bool:.1e}')
R['mask'] = dict(scores=scores.tolist(), weights=w_mask.tolist())

# ---------------------------------------------------------------- 4. multi-head attention from scratch, shapes after every line
log('== 4. Multi-head attention from scratch, shape after every line (batch 2, n 5, d_model 16, h 4)')
batch, n, d_model, h = 2, 5, 16, 4
d_k = d_model // h                                  # 4 numbers per head
x = torch.randn(batch, n, d_model)                  # the input: 2 sentences of 5 tokens, 16 numbers each
p = {f'W_{m}': torch.randn(d_model, d_model) / math.sqrt(d_model) for m in 'QKVO'}   # paper layout: x @ W
p.update({f'b_{m}': torch.randn(d_model) * 0.1 for m in 'QKVO'})

log(f'x                                    {tuple(x.shape)}   (batch, n, d_model)')
Qm = x @ p['W_Q'] + p['b_Q']                         # all 4 heads' queries in one product
log(f'Q = x W_Q + b_Q                      {tuple(Qm.shape)}   (batch, n, d_model)')
Qs = Qm.view(batch, n, h, d_k)                       # cut the last axis into h pieces of d_k
log(f'Q.view(batch, n, h, d_k)             {tuple(Qs.shape)}   (batch, n, h, d_k)')
Qh = Qs.transpose(1, 2)                              # move heads before tokens: each head is its own problem
log(f'.transpose(1, 2)                     {tuple(Qh.shape)}   (batch, h, n, d_k)')
Kh = (x @ p['W_K'] + p['b_K']).view(batch, n, h, d_k).transpose(1, 2)   # keys: the same three steps
Vh = (x @ p['W_V'] + p['b_V']).view(batch, n, h, d_k).transpose(1, 2)   # values: the same three steps
log(f'K, V the same way                    {tuple(Kh.shape)}   (batch, h, n, d_k)')
S = Qh @ Kh.transpose(-2, -1) / math.sqrt(d_k)       # one n x n score table per head
log(f'Q K^T / sqrt(d_k)                    {tuple(S.shape)}   (batch, h, n, n)')
Wt = torch.softmax(S, dim=-1)                        # each row of each table adds up to 1
log(f'softmax over the last axis           {tuple(Wt.shape)}   (batch, h, n, n)')
heads = Wt @ Vh                                      # each head mixes its own values
log(f'weights V                            {tuple(heads.shape)}   (batch, h, n, d_k)')
merged = heads.transpose(1, 2)                       # tokens first again
log(f'.transpose(1, 2)                     {tuple(merged.shape)}   (batch, n, h, d_k)')
concat = merged.reshape(batch, n, h * d_k)           # the h pieces side by side: the paper's Concat
log(f'.reshape(batch, n, h * d_k)          {tuple(concat.shape)}   (batch, n, d_model)')
y_steps = concat @ p['W_O'] + p['b_O']               # W^O mixes what the heads found
log(f'concat W_O + b_O                     {tuple(y_steps.shape)}   (batch, n, d_model)')


def split_heads(t, h):
    """(batch, n, d_model) -> (batch, h, n, d_k): cut each token vector into h pieces, one per head."""
    batch, n, d_model = t.shape
    return t.view(batch, n, h, d_model // h).transpose(1, 2)


def merge_heads(t):
    """(batch, h, n, d_k) -> (batch, n, h * d_k): put the h pieces of each token side by side (Concat)."""
    batch, h, n, d_k = t.shape
    return t.transpose(1, 2).reshape(batch, n, h * d_k)


def multi_head_attention(x_q, x_kv, p, h, mask=None):
    """MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O, with head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V).
    x_q (batch, n_q, d_model) makes the queries. x_kv (batch, n_k, d_model) makes the keys and values:
    the same tensor for self-attention, the encoder output for cross-attention.
    p: W_Q, W_K, W_V, W_O of shape (d_model, d_model) and b_Q, b_K, b_V, b_O of shape (d_model,).
    The h matrices W_i^Q of the paper are the h column blocks of W_Q, each d_model x d_k."""
    Q = split_heads(x_q @ p['W_Q'] + p['b_Q'], h)     # (batch, h, n_q, d_k)
    K = split_heads(x_kv @ p['W_K'] + p['b_K'], h)    # (batch, h, n_k, d_k)
    V = split_heads(x_kv @ p['W_V'] + p['b_V'], h)    # (batch, h, n_k, d_v)
    heads, weights = attention(Q, K, V, mask)         # (batch, h, n_q, d_v), (batch, h, n_q, n_k)
    return merge_heads(heads) @ p['W_O'] + p['b_O'], weights   # (batch, n_q, d_model)


y_ours, w_ours = multi_head_attention(x, x, p, h)
log(f'step by step vs multi_head_attention(): max abs difference {(y_steps - y_ours).abs().max().item():.1e}')

log('== 4b. Method 2: torch.nn.MultiheadAttention with the same weights')
mha = torch.nn.MultiheadAttention(d_model, h, batch_first=True)
with torch.no_grad():
    # The library stores each matrix as (out, in) and computes x @ W.T, so we copy the transpose.
    mha.in_proj_weight.copy_(torch.cat([p['W_Q'].T, p['W_K'].T, p['W_V'].T]))   # (3 d_model, d_model)
    mha.in_proj_bias.copy_(torch.cat([p['b_Q'], p['b_K'], p['b_V']]))
    mha.out_proj.weight.copy_(p['W_O'].T)
    mha.out_proj.bias.copy_(p['b_O'])
    y_lib, w_lib = mha(x, x, x, need_weights=True, average_attn_weights=False)
d_out = (y_ours - y_lib).abs().max().item()                  # Method 1 against Method 2: outputs
d_w = (w_ours - w_lib).abs().max().item()                      # and the attention weights of every head
causal = torch.tril(torch.ones(n, n, dtype=torch.bool))       # True = may look
with torch.no_grad():
    y_ours_c, _ = multi_head_attention(x, x, p, h, mask=causal)
    y_lib_c, _ = mha(x, x, x, attn_mask=~causal)                # nn.MultiheadAttention: True = may NOT look
d_causal = (y_ours_c - y_lib_c).abs().max().item()
n_par = sum(t.numel() for t in mha.parameters())
log(f'library output {tuple(y_lib.shape)}, weights {tuple(w_lib.shape)}')
log(f'output : max abs difference {d_out:.1e}')
log(f'weights: max abs difference {d_w:.1e}')
log(f'with the causal mask (library gets ~mask): max abs difference {d_causal:.1e}')
log(f'parameters: {n_par} = 4 x {d_model} x {d_model} + 4 x {d_model}')
R['mha_diff'] = d_out

# ---------------------------------------------------------------- 5. the real model: our function on Marian's weights
log('== 5. Our multi_head_attention() on the weights of Helsinki-NLP/opus-mt-en-de')
from transformers import MarianMTModel, MarianTokenizer
name = 'Helsinki-NLP/opus-mt-en-de'
tok = MarianTokenizer.from_pretrained(name)
model = MarianMTModel.from_pretrained(name, attn_implementation='eager').eval()
H = model.config.encoder_attention_heads


def marian_params(att):
    """Read one Marian attention block as the paper's matrices. nn.Linear stores (out, in): transpose to x @ W."""
    return dict(W_Q=att.q_proj.weight.T, b_Q=att.q_proj.bias, W_K=att.k_proj.weight.T, b_K=att.k_proj.bias,
                W_V=att.v_proj.weight.T, b_V=att.v_proj.bias, W_O=att.out_proj.weight.T, b_O=att.out_proj.bias)


blocks = {'encoder self-attention': model.model.encoder.layers[0].self_attn,
          'decoder self-attention': model.model.decoder.layers[0].self_attn,
          'cross-attention': model.model.decoder.layers[0].encoder_attn}
seen = {}


def catch(key):
    def hook(mod, args, kwargs, out):
        x_in = kwargs.get('hidden_states', args[0] if args else None)
        seen[key] = dict(x_q=x_in, x_kv=kwargs.get('key_value_states'), y=out[0])
    return hook


src = 'The cat sat on the mat because it was tired.'
enc = tok(src, return_tensors='pt')
with torch.no_grad():
    gen = model.generate(**enc, num_beams=1, max_new_tokens=40)
handles = [b.register_forward_hook(catch(k), with_kwargs=True) for k, b in blocks.items()]
with torch.no_grad():
    model(**enc, decoder_input_ids=gen[:, :-1])          # one forward pass: source and the translation so far
for hd in handles:
    hd.remove()
src_toks = tok.convert_ids_to_tokens(enc.input_ids[0])
tgt_toks = tok.convert_ids_to_tokens(gen[0, :-1])
log(f'source {len(src_toks)} tokens, decoder input {len(tgt_toks)} tokens')

R['marian'] = {}
with torch.no_grad():
    # encoder self-attention: queries, keys and values all come from the English tokens
    s = seen['encoder self-attention']
    y, w = multi_head_attention(s['x_q'], s['x_q'], marian_params(blocks['encoder self-attention']), H)
    diff = (y - s['y']).abs().max().item()                # against the output of the library's own block
    log(f'encoder self-attention : x {tuple(s["x_q"].shape)} -> weights {tuple(w.shape)}, max abs difference {diff:.1e}')
    R['marian']['enc'] = diff
    # decoder self-attention: the German tokens, with the causal mask
    s = seen['decoder self-attention']
    m = s['x_q'].shape[1]
    y, w = multi_head_attention(s['x_q'], s['x_q'], marian_params(blocks['decoder self-attention']), H,
                                mask=torch.tril(torch.ones(m, m, dtype=torch.bool)))
    diff = (y - s['y']).abs().max().item()
    log(f'decoder self-attention : x {tuple(s["x_q"].shape)} -> weights {tuple(w.shape)}, max abs difference {diff:.1e}')
    R['marian']['dec'] = diff
    # cross-attention: queries from the decoder, keys and values from the encoder output
    s = seen['cross-attention']
    y, w = multi_head_attention(s['x_q'], s['x_kv'], marian_params(blocks['cross-attention']), H)
    diff = (y - s['y']).abs().max().item()
    log(f'cross-attention        : queries {tuple(s["x_q"].shape)} from the decoder, keys and values '
        f'{tuple(s["x_kv"].shape)} from the encoder')
    log(f'                         -> weights {tuple(w.shape)} -> output {tuple(y.shape)}, max abs difference {diff:.1e}')
    R['marian']['cross'] = diff
    log('largest output value of each block: ' + ', '.join(f'{key} {seen[key]["y"].abs().max().item():.1f}' for key in blocks))
    j = tgt_toks.index('▁Katze')
    top = w[0, :, j].mean(0).topk(3)
    log(f'cross-attention layer 1, query "Katze", mean of {H} heads, top keys:',
        [(src_toks[i], round(v, 2)) for v, i in zip(top.values.tolist(), top.indices.tolist())])

# ---------------------------------------------------------------- 6. three common mistakes, each with its wrong output
log('== 6. Common mistakes (encoder layer 1 of Marian, and the 5-token causal example)')
s = seen['encoder self-attention']
pm = marian_params(blocks['encoder self-attention'])
it = src_toks.index('▁it')
with torch.no_grad():
    y_ok, w_ok = multi_head_attention(s['x_q'], s['x_q'], pm, H)
    # Mistake 1: forget the scale. Same code, but the scores are not divided by sqrt(d_k) = 8.
    Qh, Kh, Vh = (split_heads(s['x_q'] @ pm[f'W_{c}'] + pm[f'b_{c}'], H) for c in 'QKV')
    w_bad = torch.softmax(Qh @ Kh.transpose(-2, -1), dim=-1)          # no / math.sqrt(d_k)
    y_bad = merge_heads(w_bad @ Vh) @ pm['W_O'] + pm['b_O']
log('mistake 1, no scale:')
log(f'  query "it", head 1: top weight {w_ok[0, 0, it].max():.2f} with the scale, {w_bad[0, 0, it].max():.2f} without')
log(f'  largest weight in any row, all heads: {w_ok.max():.3f} with the scale, {w_bad.max():.3f} without')
log(f'  output: max abs difference from the library {(y_bad - s["y"]).abs().max().item():.2f}')
R['no_scale'] = dict(it_ok=w_ok[0, 0, it].max().item(), it_bad=w_bad[0, 0, it].max().item())

# Mistake 2: mask AFTER the softmax. The future weights become 0, but the rows no longer add up to 1.
w_after = torch.softmax(scores, dim=-1) * allowed                   # softmax first, then zero the future
w_zero = torch.softmax(scores.masked_fill(~allowed, 0.0), dim=-1)   # "mask" with 0 in place of minus infinity
log('mistake 2, mask after the softmax (5-token example of section 3):')
show('  weights, mask after softmax', w_after)
log('  row sums:', [round(v, 3) for v in w_after.sum(-1).tolist()], ' (correct: all 1.0)')
log('  variant: masked scores set to 0, not -inf; weight on the future per row:',
    [round(v, 3) for v in (w_zero * ~allowed).sum(-1).tolist()])
R['mask_after'] = dict(rowsums=w_after.sum(-1).tolist())

# Mistake 3: split the heads with view(batch, h, n, d_k) and no transpose. The shapes are right; the numbers are not.
with torch.no_grad():
    xq = s['x_q']
    b_, n_, dm = xq.shape
    Qw, Kw, Vw = ((xq @ pm[f'W_{c}'] + pm[f'b_{c}']).view(b_, H, n_, dm // H) for c in 'QKV')   # wrong split
    heads_w, _ = attention(Qw, Kw, Vw)
    y_w1 = heads_w.reshape(b_, n_, dm) @ pm['W_O'] + pm['b_O']
    Qr, Kr, Vr = (split_heads(xq @ pm[f'W_{c}'] + pm[f'b_{c}'], H) for c in 'QKV')            # right split
    heads_r, _ = attention(Qr, Kr, Vr)
    y_w2 = heads_r.reshape(b_, n_, dm) @ pm['W_O'] + pm['b_O']                                # wrong merge
log('mistake 3, wrong axes:')
log(f'  split with view(batch, h, n, d_k), no transpose: shapes {tuple(Qw.shape)} (look right), '
    f'max abs difference {(y_w1 - s["y"]).abs().max().item():.2f}')
log(f'  right split, but merge with reshape and no transpose: output {tuple(y_w2.shape)}, '
    f'max abs difference {(y_w2 - s["y"]).abs().max().item():.2f}')
log(f'  right split and right merge: max abs difference {(y_ok - s["y"]).abs().max().item():.1e}')
save('part3b', R)
