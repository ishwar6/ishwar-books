"""Part 3: paths through the model. The path expansion trick on real attention-only transformers.

Uses TransformerLens 3.x (HookedTransformer) with attn-only-1l and attn-only-2l. Prints and saves:
  1. the one-layer model as a product of three terms, (Id (x) W_U) . (Id + sum_h A^h (x) W_OV^h) . (Id (x) W_E),
     computed by hand at every position and compared with the model's logits;
  2. the product expanded into a sum of end-to-end paths: the direct path, one term per head, plus the parts the
     paper leaves out (position embedding, biases); each term's share of the logit of the predicted token, and the
     tokens each term pushes up most;
  3. one head term read in tokens: A^h (x) W_U W_OV^h W_E, i.e. attention weight times one column of the OV circuit,
     for every source token;
  4. the freezing trick: with the attention patterns (and layer-norm scales) frozen, the logits are a linear function
     of the token embeddings (checked on sums and multiples); without freezing they are not. Swapping one source token
     with frozen patterns changes the logits by exactly (weight) x (difference of two OV columns);
  5. the footnote: the pattern of one row depends on every source token (softmax), shown by changing one token;
  6. the two-layer model in outline: 1 + 16 + 64 = 81 path terms (direct, single heads, virtual heads) plus position
     and bias parts, summed and compared with the model's logits, and the size of each group.
Runs on the CPU in well under a minute after the models are downloaded."""
import torch
from transformer_lens import HookedTransformer
from common import Log, save, timer

torch.manual_seed(0)
torch.set_grad_enabled(False)
log = Log('part3')
R = {}
elapsed = timer()


def load(name):
    return HookedTransformer.from_pretrained(name, device='cpu').eval()


m1 = load('attn-only-1l')
c = m1.cfg
prompt = 'I like green tea, so I drink green'
R['prompt'] = prompt
toks = m1.to_tokens(prompt)
str_toks = m1.to_str_tokens(prompt)
n = len(str_toks)
last = n - 1
logits, cache = m1.run_with_cache(toks)
pred = int(logits[0, last].argmax())
TEA = m1.to_single_token(' tea')
R['tokens'] = str_toks
R['pred'] = m1.to_string(pred)
st = lambda ids: [m1.to_string(int(i)) for i in ids]

# Everything the model writes is centred (TransformerLens folds layer norm), so each layer norm only divides by a
# scale. We read the scales from the run; fixed scales make every step below a matrix product.
s1 = cache['blocks.0.ln1.hook_scale'][0, :, 0]                   # [n]: scale before the attention layer
sf = cache['ln_final.hook_scale'][0, :, 0]                       # [n]: scale before the unembedding
A = cache['blocks.0.attn.hook_pattern'][0]                       # [heads, n, n]
E = m1.W_E[toks[0]]                                              # [n, 512]: W_E t, one row per position
P = m1.W_pos[:n]                                                 # [n, 512]: the position embedding
W_U, b_U = m1.W_U, m1.b_U
OV = [m1.W_V[0, h] @ m1.W_O[0, h] for h in range(c.n_heads)]     # row form: x @ OV = (W_OV x)^T
bias_O = m1.b_O[0] + sum(m1.b_V[0, h] @ m1.W_O[0, h] for h in range(c.n_heads))   # b_V is 0 here (folded)

# ---------------------------------------------------------------- 1. the model as a product of three terms
log('== 1. THE ONE-LAYER MODEL AS A PRODUCT OF THREE TERMS (attn-only-1l)')
log(f'  prompt: {str_toks}  ({n} tokens); the model predicts {R["pred"]!r} next')
x0 = E + P                                                       # (Id (x) W_E) t, plus positions
mid = x0 + sum(A[h] @ (x0 / s1[:, None]) @ OV[h] for h in range(c.n_heads)) + bias_O   # (Id + sum_h A^h (x) W_OV^h) x0
prod = (mid / sf[:, None]) @ W_U + b_U                           # (Id (x) W_U) x1
R['product_diff'] = float((prod - logits[0]).abs().max())
log(f'  embed -> (Id + sum of 8 heads) -> unembed, by hand at all {n} positions: |by hand - model logits| max = {R["product_diff"]:.1e}')
log(f'  layer-norm scales at the last position: before attention {float(s1[last]):.3f}, before the unembedding {float(sf[last]):.3f}')

# ---------------------------------------------------------------- 2. expand the product into a sum of paths
log('\n== 2. EXPAND THE PRODUCT: A SUM OF END-TO-END PATHS (logits at the last position, " green")')
i = last
terms = {}
terms['direct path (token)'] = (E[i] / sf[i]) @ W_U                                   # Id (x) W_U W_E
for h in range(c.n_heads):
    terms[f'head 0.{h} (tokens)'] = (A[h, i] @ (E / s1[:, None]) @ OV[h] / sf[i]) @ W_U   # A^h (x) W_U W_OV^h W_E
terms['direct path (position)'] = (P[i] / sf[i]) @ W_U
terms['heads (positions)'] = sum((A[h, i] @ (P / s1[:, None]) @ OV[h] / sf[i]) @ W_U for h in range(c.n_heads))
terms['biases b_O and b_U'] = (bias_O / sf[i]) @ W_U + b_U
total = sum(terms.values())
R['sum_diff'] = float((total - logits[0, i]).abs().max())
log(f'  {len(terms)} terms: the direct path, 8 head terms (the paper\'s 9 terms), and 3 parts the paper leaves out')
log(f'  |sum of the terms - model logits| max over all {c.d_vocab:,} logits = {R["sum_diff"]:.1e}')
log(f'  {"term":26}{"adds to logit of tea":>22}   tokens it pushes up most')
R['terms'] = {}
for k, v in terms.items():
    top = v.topk(4).indices
    R['terms'][k] = dict(tea=round(float(v[TEA]), 3), top=st(top), size=round(float(v.std()), 3))
    log(f'  {k:26}{float(v[TEA]):>+22.3f}   {st(top)}')
R['tea_logit'] = float(logits[0, i, TEA])
log(f'  {"sum":26}{float(total[TEA]):>+22.3f}   (model logit of " tea": {R["tea_logit"]:.3f})')
p_tea = float(logits[0, i].softmax(-1)[TEA])
R['p_tea'] = p_tea
log(f'  probability of " tea": {p_tea:.3f}')

# ---------------------------------------------------------------- 3. one head term, read in tokens
log('\n== 3. ONE HEAD TERM IN TOKENS: A^h (x) W_U W_OV^h W_E (head 0.3, destination = last " green")')
h = 3
log('  each source token j adds  A[last, j] x (column t_j of the OV circuit)  to the logits; its effect on " tea":')
R['head3_sources'] = []
for j in range(n):
    col = (E[j] / s1[j]) @ OV[h] @ W_U / sf[i]                   # column t_j of W_U W_OV W_E (with the two scales)
    contrib = float(A[h, i, j] * col[TEA])
    R['head3_sources'].append(dict(tok=str_toks[j], a=round(float(A[h, i, j]), 3), col_tea=round(float(col[TEA]), 3), contrib=round(contrib, 3)))
    log(f'    j={j} {str_toks[j]!r:10} weight {float(A[h, i, j]):.3f} x column value {float(col[TEA]):+7.3f} = {contrib:+.3f}')
log(f'  sum = {sum(d["contrib"] for d in R["head3_sources"]):+.3f}  (term "head 0.3 (tokens)" above: {R["terms"]["head 0.3 (tokens)"]["tea"]:+.3f})')

# ---------------------------------------------------------------- 4. the freezing trick
log('\n== 4. FREEZE THE ATTENTION PATTERNS: THE LOGITS BECOME LINEAR IN THE TOKENS')
prompt_b = 'The cat sat on the mat and the dog'
toks_b = m1.to_tokens(prompt_b)
assert toks_b.shape == toks.shape
R['prompt_b'] = prompt_b
frozen = {'blocks.0.attn.hook_pattern': cache['blocks.0.attn.hook_pattern'],
          'blocks.0.ln1.hook_scale': cache['blocks.0.ln1.hook_scale'],
          'ln_final.hook_scale': cache['ln_final.hook_scale']}


def run(emb, freeze):
    """Run the model with the token embeddings replaced by emb [1, n, 512]; optionally freeze patterns and scales."""
    hooks = [('hook_embed', lambda x, hook: emb)]
    if freeze:
        hooks += [(k, (lambda v: (lambda x, hook: v))(v)) for k, v in frozen.items()]
    return m1.run_with_hooks(toks, fwd_hooks=hooks)[0]          # [n, vocab]


ea, eb = m1.W_E[toks], m1.W_E[toks_b]
zero = torch.zeros_like(ea)
R['freeze'] = {}
for freeze in [True, False]:
    f0 = run(zero, freeze)                                       # the constant part (positions and biases)
    g = lambda e: run(e, freeze) - f0                            # the part that depends on the tokens
    ga, gb = g(ea), g(eb)
    d_sum = float((g(ea + eb) - (ga + gb)).abs().max())
    d_two = float((g(2 * ea) - 2 * ga).abs().max())
    size = float(ga.abs().mean())
    key = 'frozen' if freeze else 'not frozen'
    R['freeze'][key] = dict(sum=d_sum, double=d_two, size=size)
    log(f'  {key:10}: g(a + b) vs g(a) + g(b): max diff {d_sum:.1e};  g(2a) vs 2 g(a): max diff {d_two:.1e}  (typical |g(a)| = {size:.2f})')
log('  g(t) = logits(t) - logits(no tokens): the part of the logits that depends on the tokens; a = our prompt, b = "The cat sat on the mat and the dog"')

# swap one source token, patterns frozen: the change is weight x (difference of two OV columns), summed over heads
log('\n  swap source token 4 " tea" -> " coffee", patterns and scales frozen, watch the last position:')
COF = m1.to_single_token(' coffee')
toks_c = toks.clone()
toks_c[0, 4] = COF
ec = m1.W_E[toks_c]
d_frozen = run(ec, True)[last] - run(ea, True)[last]
pred_change = sum(A[hh, last, 4] * ((m1.W_E[COF] - m1.W_E[TEA]) / s1[4]) @ OV[hh] for hh in range(c.n_heads)) @ W_U / sf[last]
R['swap_formula_diff'] = float((d_frozen - pred_change).abs().max())
log(f'    change of all logits = sum over heads of A[last, 4] x (OV column coffee - OV column tea): max diff {R["swap_formula_diff"]:.1e}')
R['swap'] = dict(tea=round(float(d_frozen[TEA]), 3), coffee=round(float(d_frozen[COF]), 3))
log(f'    frozen run: logit of " tea" changes by {float(d_frozen[TEA]):+.3f}, logit of " coffee" by {float(d_frozen[COF]):+.3f}')
real_c = m1(toks_c)[0, last]
top_c = real_c.softmax(-1).topk(3)
R['swap_real_top'] = [(m1.to_string(int(k)), round(float(v), 3)) for v, k in zip(top_c.values, top_c.indices)]
log(f'    the real model on "I like green coffee, so I drink green" predicts: {R["swap_real_top"]}')

# ---------------------------------------------------------------- 5. the footnote: a row of A depends on every source
log('\n== 5. ONE ROW OF THE PATTERN DEPENDS ON EVERY SOURCE TOKEN (softmax), head 0.3, last row')
toks_d = toks.clone()
toks_d[0, 8] = m1.to_single_token(' want')
_, cache_d = m1.run_with_cache(toks_d)
Ad = cache_d['blocks.0.attn.hook_pattern'][0, h, last]
R['row_before'] = [round(float(v), 3) for v in A[h, last]]
R['row_after'] = [round(float(v), 3) for v in Ad]
str_d = m1.to_str_tokens(toks_d)
R['tokens_d'] = str_d
log(f'  before: "{prompt}"   weight on " tea" {float(A[h, last, 4]):.3f}, on BOS {float(A[h, last, 0]):.3f}, on token 8 {str_toks[8]!r} {float(A[h, last, 8]):.3f}')
log(f'  after : "{"".join(str_d[1:])}"   weight on " tea" {float(Ad[4]):.3f}, on BOS {float(Ad[0]):.3f}, on token 8 {str_d[8]!r} {float(Ad[8]):.3f}')
log(f'  the raw score of " tea" is the same in both runs: {float(cache["blocks.0.attn.hook_attn_scores"][0, h, last, 4]):.3f} and {float(cache_d["blocks.0.attn.hook_attn_scores"][0, h, last, 4]):.3f}')

# ---------------------------------------------------------------- 6. two layers, in outline
log('\n== 6. TWO LAYERS IN OUTLINE (attn-only-2l): 1 + 16 + 64 PATH TERMS')
m2 = load('attn-only-2l')
logits2, c2 = m2.run_with_cache(toks)
L0s = c2['blocks.0.ln1.hook_scale'][0, :, 0]
L1s = c2['blocks.1.ln1.hook_scale'][0, :, 0]
Fs = c2['ln_final.hook_scale'][0, :, 0]
A0 = c2['blocks.0.attn.hook_pattern'][0]
A1 = c2['blocks.1.attn.hook_pattern'][0]
OV0 = [m2.W_V[0, k] @ m2.W_O[0, k] for k in range(8)]
OV1 = [m2.W_V[1, k] @ m2.W_O[1, k] for k in range(8)]
bO0 = m2.b_O[0] + sum(m2.b_V[0, k] @ m2.W_O[0, k] for k in range(8))
bO1 = m2.b_O[1] + sum(m2.b_V[1, k] @ m2.W_O[1, k] for k in range(8))
E2, P2 = m2.W_E[toks[0]], m2.W_pos[:n]


def unemb(v):
    return (v[last] / Fs[last]) @ m2.W_U


x0t, x0p = E2, P2                                               # the two parts of the embedding
h0 = {k: {'t': A0[k] @ (x0t / L0s[:, None]) @ OV0[k], 'p': A0[k] @ (x0p / L0s[:, None]) @ OV0[k]} for k in range(8)}
groups = {'direct path': [], 'layer-0 heads': [], 'layer-1 heads': [], 'virtual heads': [], 'positions and biases': []}
groups['direct path'].append(unemb(x0t))
groups['positions and biases'] += [unemb(x0p), unemb(bO0.expand(n, -1)), unemb(bO1.expand(n, -1)), m2.b_U]
for k in range(8):
    groups['layer-0 heads'].append(unemb(h0[k]['t']))
    groups['positions and biases'].append(unemb(h0[k]['p']))
for k in range(8):
    rd = lambda x: A1[k] @ (x / L1s[:, None]) @ OV1[k]          # layer-1 head k reading one part of the stream
    groups['layer-1 heads'].append(unemb(rd(x0t)))
    groups['positions and biases'] += [unemb(rd(x0p)), unemb(rd(bO0.expand(n, -1)))]
    for k0 in range(8):
        groups['virtual heads'].append(unemb(rd(h0[k0]['t'])))   # (A^{1.k} A^{0.k0}) (x) (W_U W_OV^{1.k} W_OV^{0.k0} W_E)
        groups['positions and biases'].append(unemb(rd(h0[k0]['p'])))
total2 = sum(sum(v) for v in groups.values())
R['two'] = dict(sum_diff=float((total2 - logits2[0, last]).abs().max()),
                counts={k: len(v) for k, v in groups.items()})
pred2 = int(logits2[0, last].argmax())
R['two']['pred'] = m2.to_string(pred2)
log(f'  same prompt; attn-only-2l predicts {R["two"]["pred"]!r} after the last " green"')
log(f'  terms: ' + ', '.join(f'{k} {len(v)}' for k, v in groups.items()) + f'  (paper\'s terms: 1 + 16 + 64 = 81)')
log(f'  |sum of all terms - model logits| max = {R["two"]["sum_diff"]:.1e}')
log(f'  {"group":22}{"adds to logit of " + repr(R["two"]["pred"]):>26}{"spread of the 48,262 logits (std)":>36}')
R['two']['groups'] = {}
for k, v in groups.items():
    s = sum(v)
    R['two']['groups'][k] = dict(pred=round(float(s[pred2]), 3), std=round(float(s.std()), 3))
    log(f'  {k:22}{float(s[pred2]):>+26.3f}{float(s.std()):>36.3f}')
R['two']['logit'] = float(logits2[0, last, pred2])
R['two']['p'] = float(logits2[0, last].softmax(-1)[pred2])
log(f'  {"sum":22}{float(total2[pred2]):>+26.3f}   (model logit {R["two"]["logit"]:.3f}, probability {R["two"]["p"]:.3f})')
vh = torch.stack(groups['virtual heads'])
big = vh.std(-1).topk(3)
R['two']['top_virtual'] = [round(float(v), 3) for v in big.values]
log(f'  the 3 largest single virtual heads (std of their logits): {R["two"]["top_virtual"]}; the 64 together: {float(vh.sum(0).std()):.3f}')

R['seconds'] = elapsed()
save('part3', R)
