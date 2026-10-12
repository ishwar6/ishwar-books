"""Part 2: one attention head taken apart, on real attention-only transformers.

Uses TransformerLens 3.x (HookedTransformer) with attn-only-1l and attn-only-2l. Prints and saves:
  1. the shapes of one head's four matrices, in the paper's notation (column vectors);
  2. "independent and additive": concatenating the eight result vectors and multiplying by the big W_O equals the sum of
     each head's result through its own slice of W_O (both layers of attn-only-2l, and the 1-layer model);
  3. a real attention pattern (attn-only-1l, a 10-token prompt), the three steps of a head (values, mix, output) redone
     by hand, and the tensor-product form h(x) = (A (x) W_O W_V) x checked in both orders;
  4. the attention pattern computed in one step from W_QK = W_Q^T W_K (biases folded in as an always-1 dimension);
  5. W_OV and W_QK of one head: shape, rank, singular values; how much the read and write subspaces overlap;
  6. keys, queries and values are "superficial": change them with any invertible 64 x 64 matrix, the head does not change;
  7. the OV circuit W_U W_OV W_E and the QK circuit W_E^T W_QK W_E of one head, read straight from the weights.
Runs on the CPU in well under a minute after the models are downloaded."""
import torch
from transformer_lens import HookedTransformer
from common import Log, save, timer

torch.manual_seed(0)
torch.set_grad_enabled(False)
log = Log('part2')
R = {}
elapsed = timer()


def load(name):
    return HookedTransformer.from_pretrained(name, device='cpu').eval()


m1 = load('attn-only-1l')
m2 = load('attn-only-2l')
c = m1.cfg
H = 3                       # the head we take apart: layer 0, head 3 of attn-only-1l ("head 0.3")
R['head'] = f'0.{H}'

# ---------------------------------------------------------------- 1. one head's matrices
log('== 1. ONE HEAD, FOUR MATRICES (attn-only-1l, head 0.3, paper shapes = TransformerLens shapes transposed)')
for name in ['W_Q', 'W_K', 'W_V', 'W_O']:
    W = getattr(m1, name)[0, H]
    log(f'  {name}: paper shape {tuple(W.T.shape)}  ({W.numel():,} numbers)')
log(f'  d_model={c.d_model} d_head={c.d_head} heads per layer={c.n_heads}  (8 x 64 = {c.n_heads * c.d_head})')
log(f'  scores are divided by sqrt(d_head) = {c.attn_scale:.0f} before the softmax (the paper folds this into W_Q)')

# ---------------------------------------------------------------- 2. independent and additive
log('\n== 2. CONCATENATE-THEN-MULTIPLY EQUALS THE SUM OF EACH HEAD THROUGH ITS OWN SLICE OF W_O')
prompt = 'I like green tea, so I drink green'
R['prompt'] = prompt
R['add'] = {}
for mname, m in [('attn-only-1l', m1), ('attn-only-2l', m2)]:
    toks = m.to_tokens(prompt)
    _, cache = m.run_with_cache(toks)
    for L in range(m.cfg.n_layers):
        z = cache[f'blocks.{L}.attn.hook_z'][0]                       # [pos, heads, d_head]: result vectors r^h
        W_O = m.W_O[L]                                                # [heads, d_head, d_model]
        concat = z.reshape(z.shape[0], -1)                            # [pos, 512]: r^h1, r^h2, ... stacked
        W_OH = W_O.reshape(-1, m.cfg.d_model)                         # [512, 512]: the big output matrix W_O^H
        big = concat @ W_OH                                           # the usual way: one big multiply
        per_head = torch.stack([z[:, h] @ W_O[h] for h in range(m.cfg.n_heads)])   # [heads, pos, 512]
        summed = per_head.sum(0)                                      # the paper's way: each head alone, then add
        real = cache[f'blocks.{L}.hook_attn_out'][0] - m.b_O[L]       # what the model really added (minus the bias)
        d1, d2 = float((big - summed).abs().max()), float((summed - real).abs().max())
        norms = per_head[:, -1].norm(dim=-1)
        R['add'][f'{mname} L{L}'] = dict(concat_vs_sum=d1, sum_vs_model=d2, W_OH=tuple(W_OH.T.shape),
                                         head_norms_last=[round(float(v), 3) for v in norms], out_norm_last=float(summed[-1].norm()))
        log(f'  {mname} layer {L}: W_O^H is {tuple(W_OH.T.shape)} (paper shape); |concat - sum| max = {d1:.1e}; '
            f'|sum - model| max = {d2:.1e}')
d = R['add']['attn-only-1l L0']
log(f'  attn-only-1l, last position: size of each head\'s 512-number output: {d["head_norms_last"]}; size of their sum {d["out_norm_last"]:.3f}')

# ---------------------------------------------------------------- 3. a real attention pattern, three steps, tensor form
log('\n== 3. ONE HEAD ON A REAL PROMPT (attn-only-1l, head 0.3)')
toks = m1.to_tokens(prompt)
str_toks = m1.to_str_tokens(prompt)
logits, cache = m1.run_with_cache(toks)
A = cache['blocks.0.attn.hook_pattern'][0, H]                         # [pos, pos]: row = destination, column = source
X = cache['blocks.0.ln1.hook_normalized'][0]                          # [pos, 512]: the stream as the head reads it
R['tokens'] = str_toks
R['pattern'] = [[round(float(v), 3) for v in row] for row in A]
R['model_pred'] = m1.to_string(logits[0, -1].argmax())
log(f'  prompt: {str_toks}  ({len(str_toks)} tokens); the model predicts {R["model_pred"]!r} next')
for i in [len(str_toks) - 1]:
    top = A[i].topk(3)
    log(f'  row of A for the last token {str_toks[i]!r}: ' + ', '.join(f'{str_toks[int(j)]!r} {float(v):.2f}' for v, j in zip(top.values, top.indices)))
log(f'  every row adds up to 1: max |row sum - 1| = {float((A.sum(-1) - 1).abs().max()):.1e}; '
    f'entries above the diagonal (the future) are {float(A.triu(1).abs().max()):.0f}')
# the three steps, by hand (row form: TransformerLens multiplies on the right)
W_V, W_O = m1.W_V[0, H], m1.W_O[0, H]
v = X @ W_V + m1.b_V[0, H]                                           # step 1: a value vector for every token  [pos, 64]
r = A @ v                                                             # step 2: mix value vectors with the pattern [pos, 64]
out = r @ W_O                                                         # step 3: write back into the stream     [pos, 512]
z_model = cache['blocks.0.attn.hook_z'][0, :, H]
R['three_steps_r_diff'] = float((r - z_model).abs().max())
log(f'  three steps by hand: |r - model r| max = {R["three_steps_r_diff"]:.1e}')
# the tensor-product form: (A (x) W_OV) x = A x W_OV^T in the paper's column form, = A @ X @ (W_V W_O) in row form
W_OV_row = W_V @ W_O                                                  # [512, 512] (the paper's W_OV, transposed)
first_A = (A @ X) @ W_OV_row                                          # mix across tokens first, then each vector
first_W = A @ (X @ W_OV_row)                                          # each vector first, then mix across tokens
R['tensor_order_diff'] = float((first_A - first_W).abs().max())
R['tensor_vs_steps_diff'] = float((first_A - out).abs().max())
log(f'  (A (x) W_OV) x, A first vs W_OV first: max diff {R["tensor_order_diff"]:.1e}; vs the three steps: {R["tensor_vs_steps_diff"]:.1e}')
# what the head moved: the head's output at the last position, read through the unembedding
pos = len(str_toks) - 1
dl = out[pos] @ m1.W_U
top = dl.topk(5)
R['moved_top'] = [(m1.to_string(int(i)), round(float(v), 2)) for v, i in zip(top.values, top.indices)]
log(f'  head 0.3 output at {str_toks[pos]!r}, read with W_U: top tokens {R["moved_top"]}')
ranks = (dl > dl[m1.to_single_token(' tea')]).sum().item() + 1
log(f'  rank of " tea" among {c.d_vocab:,} tokens: {ranks}')
R['tea_rank_in_head_out'] = ranks

# ---------------------------------------------------------------- 4. the attention pattern in one step, from W_QK
log('\n== 4. THE PATTERN IN ONE STEP: A = softmax(x^T W_QK x), WITHOUT KEYS OR QUERIES')
ones = torch.ones(X.shape[0], 1)
Xt = torch.cat([X, ones], 1)                                          # always-1 dimension carries the biases (Part 1)
WQt = torch.cat([m1.W_Q[0, H], m1.b_Q[0, H][None]], 0)                # [513, 64]
WKt = torch.cat([m1.W_K[0, H], m1.b_K[0, H][None]], 0)
W_QK_row = WQt @ WKt.T                                                # [513, 513]: the paper's W_Q^T W_K
scores = Xt @ W_QK_row @ Xt.T / c.attn_scale                          # every query-key pair in one product
mask = torch.ones_like(scores).triu(1).bool()
A1 = scores.masked_fill(mask, float('-inf')).softmax(-1)
R['qk_one_step_diff'] = float((A1 - A).abs().max())
log(f'  W_QK with the biases folded in: {tuple(W_QK_row.shape)}; |A from W_QK - model A| max = {R["qk_one_step_diff"]:.1e}')
s_last = scores[pos, :pos + 1]
R['scores_last'] = [round(float(v), 2) for v in s_last]
log(f'  raw scores of the last row (before softmax): {R["scores_last"]}')

# ---------------------------------------------------------------- 5. W_OV and W_QK: low rank
log('\n== 5. W_OV = W_O W_V AND W_QK = W_Q^T W_K (paper shapes)')
W_OV = (W_V @ W_O).T                                                  # paper: [512, 512] = W_O (512x64) W_V (64x512)
W_QK = (m1.W_Q[0, H] @ m1.W_K[0, H].T)                                # paper: W_Q^T W_K, [512, 512]
for name, M in [('W_OV', W_OV), ('W_QK', W_QK)]:
    S = torch.linalg.svdvals(M)
    rank = int(torch.linalg.matrix_rank(M))
    R[name] = dict(shape=tuple(M.shape), rank=rank, numel=M.numel(), sv=[round(float(s), 4) for s in S],
                   factors=2 * c.d_model * c.d_head)
    log(f'  {name}: shape {tuple(M.shape)} = {M.numel():,} numbers, rank {rank}; stored as two factors: {2 * c.d_model * c.d_head:,} numbers')
    log(f'    singular values: 1st {S[0]:.3f}, 2nd {S[1]:.3f}, 64th {S[63]:.4f}, 65th {S[64]:.1e}, 512th {S[-1]:.1e}')
# read and write subspaces (SVD: W_OV = U S V^T; V's first 64 columns = read, U's first 64 = write)
log('  overlap of the 64-dim read and write subspaces of W_OV (1 = the same subspace, 0 = at right angles, '
    f'random 64-dim subspaces of 512 dims: {64 / 512:.3f})')
R['overlap'] = {}
for h in range(c.n_heads):
    M = (m1.W_V[0, h] @ m1.W_O[0, h]).T
    U, S, Vh = torch.linalg.svd(M)
    read, write = Vh[:64].T, U[:, :64]
    ov = float((read.T @ write).pow(2).sum() / 64)
    R['overlap'][h] = ov
log('    ' + ', '.join(f'head 0.{h}: {v:.3f}' for h, v in R['overlap'].items()))

# ---------------------------------------------------------------- 6. keys, queries and values are superficial
log('\n== 6. CHANGE THE VALUES WITH ANY INVERTIBLE 64 x 64 MATRIX M: THE HEAD DOES NOT CHANGE')
M = torch.randn(64, 64, dtype=torch.float64)                          # any invertible matrix (float64 to keep rounding small)
Xd, Ad, W_Vd, W_Od = X.double(), A.double(), W_V.double(), W_O.double()
v, out = Xd @ W_Vd, (Ad @ (Xd @ W_Vd)) @ W_Od
v2 = v @ M                                                            # new value vectors (W_V -> M^T W_V in paper form)
out2 = (Ad @ v2) @ (torch.linalg.inv(M) @ W_Od)                       # and W_O -> W_O M^-T
q, k = Xd @ m1.W_Q[0, H].double(), Xd @ m1.W_K[0, H].double()
q2, k2 = q @ M, k @ torch.linalg.inv(M).T                             # new queries and keys
R['reparam'] = dict(value_change=float((v2 - v).abs().mean()), value_size=float(v.abs().mean()),
                    out_diff=float((out2 - out).abs().max()), out_size=float(out.abs().mean()),
                    qk_diff=float(((q2 @ k2.T) - (q @ k.T)).abs().max()), q_change=float((q2 - q).abs().mean()),
                    q_size=float(q.abs().mean()), score_size=float((q @ k.T).abs().mean()))
log(f'  value vectors: mean |change| {R["reparam"]["value_change"]:.2f} (their mean size is {R["reparam"]["value_size"]:.2f}); '
    f'head output: max |change| {R["reparam"]["out_diff"]:.1e}')
log(f'  queries: mean |change| {R["reparam"]["q_change"]:.2f} (mean size {R["reparam"]["q_size"]:.2f}); '
    f'every query-key score: max |change| {R["reparam"]["qk_diff"]:.1e} (mean size {R["reparam"]["score_size"]:.2f})')

# ---------------------------------------------------------------- 7. the OV and QK circuits, read from the weights
log('\n== 7. THE OV CIRCUIT W_U W_OV W_E AND THE QK CIRCUIT W_E^T W_QK W_E (head 0.3, read from the weights only)')
WE = m1.W_E - m1.W_E.mean(1, keepdim=True)
WE = WE / WE.std(1, keepdim=True)                                     # what is left of layer norm: each token's vector scaled to size
WU = m1.W_U
log(f'  each full circuit is {c.d_vocab:,} x {c.d_vocab:,} = {c.d_vocab ** 2 / 1e9:.2f} billion numbers, so we compute one column or row at a time')
st = lambda ids: [m1.to_string(int(i)) for i in ids]
R['ov'] = {}
for w in [' tea', ' green', ' Paris', ' Obama', ' coffee', ' cat', ' London', ' Potter']:
    t = m1.to_single_token(w)
    col = (WE[t] @ W_V @ W_O) @ WU                                    # column t of W_U W_OV W_E: "if the head attends to w"
    top = col.topk(6)
    R['ov'][w] = dict(top=[(m1.to_string(int(i)), round(float(v), 2)) for v, i in zip(top.values, top.indices)],
                      self_rank=int((col > col[t]).sum()) + 1)
    log(f'  OV  attend to {w!r:10} -> raises {st(top.indices)}')
ids = torch.arange(1000, 3000)
OVcols = (WE[ids] @ W_V @ W_O) @ WU
R['ov_self_top1'] = {}
for h in range(c.n_heads):
    cols = (WE[ids] @ m1.W_V[0, h] @ m1.W_O[0, h]) @ WU
    R['ov_self_top1'][h] = float((cols.argmax(-1) == ids).float().mean())
log('  share of tokens 1000..2999 whose top OV output is the token itself: ' +
    ', '.join(f'0.{h}: {v:.2f}' for h, v in R['ov_self_top1'].items()))
K = WE @ m1.W_K[0, H]                                                 # [vocab, 64]: the key of every token
R['qk'] = {}
for w in [' in', ' see', ' of', ' the', ' drink']:
    t = m1.to_single_token(w)
    row = (K @ (WE[t] @ m1.W_Q[0, H])) / c.attn_scale                 # row t of W_E^T W_QK W_E: "query w, which keys?"
    top = row.topk(8)
    R['qk'][w] = dict(top=[(m1.to_string(int(i)), round(float(v), 2)) for v, i in zip(top.values, top.indices)],
                      median=round(float(row.median()), 2))
    log(f'  QK  query {w!r:8} -> attends most to {st(top.indices)}  (scores {float(top.values[0]):.1f} .. {float(top.values[-1]):.1f}; median {R["qk"][w]["median"]:.1f})')

R['seconds'] = elapsed()
log(f'\ntotal time {R["seconds"]:.1f} s')
save('part2', R)
