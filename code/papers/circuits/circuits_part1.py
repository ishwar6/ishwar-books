"""Part 1: the residual stream of real attention-only transformers, taken apart.

Uses TransformerLens (version 3.x, the HookedTransformer API) and three small models:
  attn-only-1l and attn-only-2l (attention-only toy models trained in the style of the paper: 512-wide residual stream,
  8 heads of 64 dimensions per layer, a 48,262-token vocabulary) and gpt2 (small, for comparison).
Prints and saves:
  1. the shapes of every matrix the paper names (W_E, W_U, W_Q, W_K, W_V, W_O) and the parameter counts;
  2. that folding layer norm's scale into the weights changes nothing (largest logit difference);
  3. the residual stream at one position split into its parts (embedding, position, every head, biases), and the check
     that the parts add up to the real vector; then each part's share of the final logit (direct logit attribution);
  4. the direct path W_U W_E (the zero-layer model hidden inside every transformer): top next tokens for a few tokens;
  5. the read and write matrices of one head, and virtual weights between layers 0 and 1 of attn-only-2l;
  6. the bandwidth numbers (residual dimensions vs head dimensions and neurons) and how many dimensions the embedding uses;
  7. that rotating the residual stream (and every matrix touching it) leaves the output unchanged.
Runs on the CPU in under a minute after the first download (about 100 MB per toy model, 500 MB for gpt2)."""
import torch
from transformer_lens import HookedTransformer
from common import Log, save, timer

torch.manual_seed(0)
torch.set_grad_enabled(False)
log = Log('part1')
R = {}
elapsed = timer()


def load(name, **kw):
    return HookedTransformer.from_pretrained(name, device='cpu', **kw).eval()


# ---------------------------------------------------------------- 1. the matrices and their shapes
m1 = load('attn-only-1l')
m2 = load('attn-only-2l')
g2 = load('gpt2')
log('== 1. THE MATRICES (TransformerLens stores them transposed: row vectors, x @ W)')
for name, m in [('attn-only-1l', m1), ('attn-only-2l', m2), ('gpt2', g2)]:
    c = m.cfg
    n = sum(p.numel() for p in m.parameters())
    log(f'{name}: layers={c.n_layers} heads/layer={c.n_heads} d_model={c.d_model} d_head={c.d_head} '
        f'n_vocab={c.d_vocab} n_ctx={c.n_ctx} attn_only={c.attn_only} params={n:,}')
    R[name] = dict(n_layers=c.n_layers, n_heads=c.n_heads, d_model=c.d_model, d_head=c.d_head, d_vocab=c.d_vocab,
                   n_ctx=c.n_ctx, params=n, d_mlp=(c.d_mlp if not c.attn_only else 0))
c = m2.cfg
shapes = dict(W_E=tuple(m2.W_E.shape), W_pos=tuple(m2.W_pos.shape), W_U=tuple(m2.W_U.shape),
              W_Q=tuple(m2.W_Q.shape), W_K=tuple(m2.W_K.shape), W_V=tuple(m2.W_V.shape), W_O=tuple(m2.W_O.shape))
for k, v in shapes.items():
    log(f'  attn-only-2l {k}: {v}')
R['shapes_2l'] = shapes
R['embed_params_2l'] = m2.W_E.numel() + m2.W_U.numel()
R['we_sample'] = [round(float(v), 3) for v in m2.W_E[0, :3]]
log(f'  the first three numbers of W_E (token 0): {R["we_sample"]}')
log(f'  W_E + W_U hold {R["embed_params_2l"]:,} of {R["attn-only-2l"]["params"]:,} parameters')

# ---------------------------------------------------------------- 2. layer norm folding is exact
raw = load('attn-only-2l', fold_ln=False, center_writing_weights=False, center_unembed=False)
prompt = 'Mr and Mrs Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal. Mr and Mrs Durs'
toks = m2.to_tokens(prompt)
lp_a = m2(toks).log_softmax(-1)
lp_b = raw(toks).log_softmax(-1)
R['fold_ln_max_logprob_diff'] = float((lp_a - lp_b).abs().max())
log('')
log('== 2. FOLDING LAYER NORM INTO THE WEIGHTS')
log(f'  prompt has {toks.shape[1]} tokens; largest change in any log-probability after folding: {R["fold_ln_max_logprob_diff"]:.2e}')
del raw

# ---------------------------------------------------------------- 3. the residual stream is a sum of parts
logits, cache = m2.run_with_cache(toks)
pos = toks.shape[1] - 1
str_toks = m2.to_str_tokens(toks)
pred = int(logits[0, pos].argmax())
top5 = logits[0, pos].softmax(-1).topk(5)
log('')
log('== 3. THE RESIDUAL STREAM AT THE LAST POSITION, TAKEN APART (attn-only-2l)')
log(f'  last tokens: {str_toks[-6:]}  -> model predicts {m2.to_string([pred])!r}')
log('  top 5: ' + ', '.join(f'{m2.to_string([int(i)])!r} {float(p):.3f}' for p, i in zip(top5.values, top5.indices)))
parts = [('embedding W_E t', cache['hook_embed'][0, pos]), ('position W_pos', cache['hook_pos_embed'][0, pos])]
for L in range(c.n_layers):
    z = cache[f'blocks.{L}.attn.hook_z'][0, pos]                    # [heads, d_head]: the result vector r of each head
    for h in range(c.n_heads):
        parts.append((f'head {L}.{h}', z[h] @ m2.W_O[L, h]))         # W_O r: what head L.h writes
    parts.append((f'bias b_O layer {L}', m2.b_O[L]))
total = sum(v for _, v in parts)
real = cache[f'blocks.{c.n_layers - 1}.hook_resid_post'][0, pos]
R['resid_sum_max_abs_diff'] = float((total - real).abs().max())
R['resid_norm'] = float(real.norm())
log(f'  {len(parts)} parts; |sum of parts - real residual vector| max = {R["resid_sum_max_abs_diff"]:.2e}; ||x_final|| = {R["resid_norm"]:.2f}')
log(f'  first 4 coordinates of the real vector: {[round(float(v), 3) for v in real[:4]]}')
log(f'  first 4 coordinates of the sum:         {[round(float(v), 3) for v in total[:4]]}')
# direct logit attribution: logit = sum over parts of (part, centred and divided by the final scale) . W_U[:, pred]  + b_U
scale = cache['ln_final.hook_scale'][0, pos, 0]
contrib = []
for name, v in parts:
    v = v - v.mean()
    contrib.append((name, float(v @ m2.W_U[:, pred] / scale), float(v.norm())))
R['pred_token'] = m2.to_string([pred])
R['final_scale'] = float(scale)
R['logit_real'] = float(logits[0, pos, pred])
R['logit_from_parts'] = sum(x for _, x, _ in contrib) + float(m2.b_U[pred])
R['b_U_pred'] = float(m2.b_U[pred])
R['parts'] = [dict(name=n, logit=l, norm=nm) for n, l, nm in contrib]
log(f'  final layer-norm scale at this position: {R["final_scale"]:.3f}')
log(f'  {"part":<20}{"size ||v||":>12}{"adds to logit of " + repr(R["pred_token"]):>34}')
for n, l, nm in contrib:
    log(f'  {n:<20}{nm:>12.2f}{l:>+34.3f}')
log(f'  {"unembedding bias":<20}{"":>12}{R["b_U_pred"]:>+34.3f}')
log(f'  sum of all lines = {R["logit_from_parts"]:.3f}; the model\'s real logit = {R["logit_real"]:.3f}')
# where does the biggest contributor look? (attention pattern of the top head at the last position)
best = max(contrib[2:], key=lambda c: c[1])[0]
if best.startswith('head 1.'):
    hh = int(best.split('.')[1])
    patt = cache['blocks.1.attn.hook_pattern'][0, hh, pos]
    top3 = patt.topk(3)
    R['top_head'] = dict(name=best, attends=[(int(i), str_toks[int(i)], float(v)) for v, i in zip(top3.values, top3.indices)])
    log(f'  {best} at the last position attends to: ' + ', '.join(f'position {int(i)} {str_toks[int(i)]!r} {float(v):.2f}' for v, i in zip(top3.values, top3.indices)))
R['prompt'] = prompt
R['last_tokens'] = str_toks[-6:]
R['top5'] = [(m2.to_string([int(i)]), float(p)) for p, i in zip(top5.values, top5.indices)]

# ---------------------------------------------------------------- 4. the direct path W_U W_E: a bigram table
log('')
log('== 4. THE DIRECT PATH W_U W_E (top 5 "next tokens" from the token alone)')
words = [' Barack', ' United', ' New', ' Hong', ' Mr', ' according', ' Los', ' (']
R['direct'] = {}
for name, m in [('attn-only-1l', m1), ('attn-only-2l', m2), ('gpt2', g2)]:
    log(f'  {name}')
    R['direct'][name] = {}
    for w in words:
        ids = m.to_tokens(w, prepend_bos=False)[0]
        if len(ids) != 1:
            log(f'    {w!r}: not a single token, skipped')
            continue
        t = int(ids[0])
        row = m.W_E[t] @ m.W_U                                       # one row of the n_vocab x n_vocab table
        row = row - row.mean()
        top = row.topk(5)
        rank_self = int((row > row[t]).sum())
        R['direct'][name][w] = dict(top=[(m.to_string([int(i)]), float(v)) for v, i in zip(top.values, top.indices)], self_rank=rank_self)
        log(f'    {w!r:>12} -> ' + ', '.join(f'{m.to_string([int(i)])!r}' for i in top.indices) + f'   (rank of the token itself: {rank_self})')
# how often is the direct path's top choice the same token (the "inverse" test), over the first 2000 ordinary tokens
for name, m in [('attn-only-1l', m1), ('gpt2', g2)]:
    ids = torch.arange(1000, 3000)
    T = m.W_E[ids] @ m.W_U
    same = float((T.argmax(-1) == ids).float().mean())
    R['direct'][name]['_self_top1_rate'] = same
    log(f'  {name}: tokens 1000..2999, share whose top direct-path prediction is the token itself: {same:.3f}')

# ---------------------------------------------------------------- 5. read and write matrices, virtual weights
log('')
log('== 5. READING, WRITING AND VIRTUAL WEIGHTS (attn-only-2l, paper notation: W_V is d_head x d_model)')
WV0 = m2.W_V[0, 0].T          # paper's W_V: [64, 512]  reads from the residual stream
WO0 = m2.W_O[0, 0].T          # paper's W_O: [512, 64]  writes to the residual stream
log(f'  head 0.0 reads with W_V {tuple(WV0.shape)} and writes with W_O {tuple(WO0.shape)}')
WOV = WO0 @ WV0
R['wov_shape'] = tuple(WOV.shape)
R['wov_rank'] = int(torch.linalg.matrix_rank(WOV))
log(f'  W_O W_V is {tuple(WOV.shape)} = {WOV.numel():,} numbers, but its rank is only {R["wov_rank"]}')
# virtual weight: layer-1 head h2 reading (with W_V) what layer-0 head h1 wrote (with W_O)
V = torch.zeros(c.n_heads, c.n_heads)
for h1 in range(c.n_heads):
    for h2 in range(c.n_heads):
        V[h1, h2] = (m2.W_V[1, h2].T @ m2.W_O[0, h1].T).norm()
R['virtual_norms'] = V.tolist()
i, j = divmod(int(V.argmax()), c.n_heads)
k, l = divmod(int(V.argmin()), c.n_heads)
R['virtual_max'] = dict(h0=i, h1=j, norm=float(V.max()))
R['virtual_min'] = dict(h0=k, h1=l, norm=float(V.min()))
log(f'  virtual weight W_V(1.h2) W_O(0.h1): shape {tuple((m2.W_V[1, 0].T @ m2.W_O[0, 0].T).shape)}, one for each of the 64 head pairs')
log(f'  largest size (Frobenius norm): head 0.{i} -> head 1.{j}: {float(V.max()):.2f}; smallest: head 0.{k} -> head 1.{l}: {float(V.min()):.2f}')
log('  (raw sizes are not yet a fair score of how much two heads talk; Part 5 normalises them)')
WE_T = m2.W_E.T               # paper's W_E: [512, 48262]
R['we_paper_shape'] = tuple(WE_T.shape)
log(f'  the embedding as a "layer 0 writer": W_E is {tuple(WE_T.shape)}; head 0.0 reading it: W_V W_E is {tuple((WV0 @ WE_T).shape)}')

# ---------------------------------------------------------------- 6. bandwidth and how much room the embedding takes
log('')
log('== 6. BANDWIDTH')
R['bandwidth'] = {}
for name in ['attn-only-2l', 'gpt2']:
    d = R[name]
    head_dims = d['n_layers'] * d['n_heads'] * d['d_head']
    neurons = d['n_layers'] * d['d_mlp']
    R['bandwidth'][name] = dict(d_model=d['d_model'], head_dims=head_dims, neurons=neurons)
    log(f'  {name}: residual stream {d["d_model"]} dims; heads write {head_dims} dims in total; MLP neurons {neurons}; '
        f'ratio {(head_dims + neurons) / d["d_model"]:.1f}x')
R['pca'] = {}
for name, m in [('attn-only-2l', m2), ('gpt2', g2)]:
    out = {}
    for which, W in [('W_E', m.W_E), ('W_U', m.W_U.T)]:
        X = W - W.mean(0)
        s = torch.linalg.svdvals(X) ** 2
        cum = (s.cumsum(0) / s.sum())
        out[which] = {f'{int(q * 100)}%': int((cum < q).sum()) + 1 for q in (0.5, 0.9)}
    R['pca'][name] = out
    log(f'  {name}: principal directions needed for 50% / 90% of the variance: '
        f'W_E {out["W_E"]["50%"]} / {out["W_E"]["90%"]}, W_U {out["W_U"]["50%"]} / {out["W_U"]["90%"]} (of {m.cfg.d_model})')

# ---------------------------------------------------------------- 7. no privileged basis: rotate everything
log('')
log('== 7. ROTATE THE RESIDUAL STREAM (attn-only-1l)')
# A random rotation that keeps the all-ones direction fixed, so layer norm's "subtract the mean" step is not disturbed.
d = m1.cfg.d_model
g = torch.Generator().manual_seed(0)
ones = torch.ones(d, 1) / d ** 0.5
B, _ = torch.linalg.qr(torch.cat([ones, torch.randn(d, d - 1, generator=g)], 1))
B = B[:, 1:]                                     # an orthonormal basis of the directions with mean zero
Qs, _ = torch.linalg.qr(torch.randn(d - 1, d - 1, generator=g))
Rm = ones @ ones.T + B @ Qs @ B.T                # rotate inside that space, leave the all-ones direction alone
log(f'  rotation is orthogonal: max |R R^T - I| = {float((Rm @ Rm.T - torch.eye(d)).abs().max()):.1e}; keeps ones: {float((Rm @ ones - ones).abs().max()):.1e}')
toks1 = m1.to_tokens(prompt)
before = m1(toks1)
sd = {k: v.clone() for k, v in m1.state_dict().items()}
rot = load('attn-only-1l')
new = rot.state_dict()
new['embed.W_E'] = sd['embed.W_E'] @ Rm.T                    # rows are residual vectors: x -> R x
new['pos_embed.W_pos'] = sd['pos_embed.W_pos'] @ Rm.T
for key in ['W_Q', 'W_K', 'W_V']:                              # readers: x R^T then (R W) = x W
    new[f'blocks.0.attn.{key}'] = torch.einsum('ij,hjk->hik', Rm, sd[f'blocks.0.attn.{key}'])
new['blocks.0.attn.W_O'] = sd['blocks.0.attn.W_O'] @ Rm.T       # writer
new['blocks.0.attn.b_O'] = sd['blocks.0.attn.b_O'] @ Rm.T
new['unembed.W_U'] = Rm @ sd['unembed.W_U']                     # final reader
rot.load_state_dict(new)
after = rot(toks1)
R['rotation_max_logit_diff'] = float((before - after).abs().max())
R['rotation_W_E_change'] = float((new['embed.W_E'] - sd['embed.W_E']).abs().mean())
log(f'  every weight that touches the stream changed (mean |change| in W_E: {R["rotation_W_E_change"]:.3f})')
log(f'  largest change in any output logit: {R["rotation_max_logit_diff"]:.1e}')

R['seconds'] = round(elapsed(), 1)
log('')
log(f'done in {R["seconds"]} s on the CPU')
save('part1', R)
