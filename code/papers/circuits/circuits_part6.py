"""Part 6: induction heads, on real attention-only transformers (attn-only-1l and attn-only-2l, TransformerLens 3.x).

Prints and saves:
  1. the model we use, and how it differs from the paper's two-layer model;
  2. repeated random tokens: the loss on the first and on the second copy, for the one-layer and the two-layer model,
     and the loss at every position (for the figure);
  3. the previous-token score of every layer-0 head and the induction (prefix-matching) score of every layer-1 head;
  4. the attention patterns of the previous-token head and the induction head on a short repeated sequence;
  5. the copying score of every head: eigenvalue positivity of its OV circuit W_U W_OV W_E, and the share of tokens
     whose top output is the token itself;
  6. the "same matching" score: eigenvalue positivity of the K-composition term W_E^T W_QK^h W_OV^prev W_E;
  7. K-composition scores between every layer-0 head and the induction head (the measure Part 5 explains);
  8. how the key of the induction head is built: the attention score split into the part from the previous-token head
     and the part from everything else;
  9. ablations: replace one head's output with its average output and measure the induction score and the loss again;
 10. a sentence with a made-up name: what the induction head reads and adds, next to the one-layer model.
Runs on the CPU in about a minute after the models are downloaded."""
import torch
from transformer_lens import HookedTransformer
from common import Log, save

torch.manual_seed(0)
torch.set_grad_enabled(False)
log = Log('part6')
R = {}


def load(name):
    return HookedTransformer.from_pretrained(name, device='cpu').eval()


m1 = load('attn-only-1l')
m2 = load('attn-only-2l')
c = m2.cfg
NH = c.n_heads
BOS = m2.tokenizer.bos_token_id

# ---------------------------------------------------------------- 1. the model
log('== 1. THE MODEL: attn-only-2l (TransformerLens)')
log(f'  layers {c.n_layers}, heads per layer {NH}, d_model {c.d_model}, d_head {c.d_head}, vocabulary {c.d_vocab:,}, context {c.n_ctx}')
log(f'  position: {c.positional_embedding_type} (learned position vectors added to the stream; the paper\'s model has none in the stream)')
log('  the paper\'s two-layer model has 12 heads per layer; ours has 8, so head numbers differ')
R['model'] = dict(layers=c.n_layers, heads=NH, d_model=c.d_model, d_head=c.d_head, vocab=c.d_vocab)

# ---------------------------------------------------------------- 2. repeated random tokens
log('\n== 2. REPEATED RANDOM TOKENS: [BOS] t1 ... t50 t1 ... t50, 100 sequences, tokens drawn uniformly from the vocabulary')
N, S = 100, 50
rand = torch.randint(1, c.d_vocab, (N, S))
rand[rand == BOS] = 1
toks = torch.cat([torch.full((N, 1), BOS), rand, rand], 1)           # [N, 101]
R['rep'] = dict(n=N, half=S)
uniform = float(torch.log(torch.tensor(float(c.d_vocab))))
log(f'  a model that guesses uniformly over {c.d_vocab:,} tokens has loss ln({c.d_vocab}) = {uniform:.2f} nats on every token')
R['rep']['uniform'] = uniform
curves = {}
for name, m in [('attn-only-1l', m1), ('attn-only-2l', m2)]:
    logits = m(toks)
    lp = logits.log_softmax(-1)
    loss = -lp[:, :-1].gather(-1, toks[:, 1:, None])[..., 0]         # [N, 100]: loss of predicting token p+1 at position p
    acc = (logits[:, :-1].argmax(-1) == toks[:, 1:]).float()
    # position p predicts token p+1. Tokens 2..50 of the first copy: p = 1..49; tokens 2..50 of the second copy: p = 51..99
    first, second = loss[:, 1:S].mean(), loss[:, S + 1:].mean()
    a1, a2 = acc[:, 1:S].mean(), acc[:, S + 1:].mean()
    curves[name] = [round(float(v), 3) for v in loss.mean(0)]
    R['rep'][name] = dict(first=float(first), second=float(second), acc_first=float(a1), acc_second=float(a2))
    log(f'  {name}: loss on the first copy {first:.2f}, on the second copy {second:.2f} nats; '
        f'top-1 right: first {a1:.1%}, second {a2:.1%}')
R['rep']['curves'] = curves
p1, p2 = R['rep']['attn-only-1l'], R['rep']['attn-only-2l']
log(f'  probability the 2-layer model gives the right token on the second copy, on average: exp(-{p2["second"]:.2f}) = {float(torch.exp(torch.tensor(-p2["second"]))):.2f}')

# ---------------------------------------------------------------- 3. previous-token and induction scores
log('\n== 3. WHICH HEADS DO WHAT, MEASURED ON THE REPEATED SEQUENCES')
_, cache = m2.run_with_cache(toks, names_filter=lambda n: n.endswith('hook_pattern'))
P0, P1 = cache['blocks.0.attn.hook_pattern'], cache['blocks.1.attn.hook_pattern']   # [N, heads, 101, 101]
prev = P0.diagonal(-1, -2, -1).mean((0, -1))                          # weight on position i-1, averaged over i and sequences
prev1 = P1.diagonal(-1, -2, -1).mean((0, -1))
# induction: destination i in the second copy (i = 51..100) reads source i-49, the token after the earlier copy of token i
ind = P1.diagonal(-(S - 1), -2, -1)[..., 1:].mean((0, -1))
ind0 = P0.diagonal(-(S - 1), -2, -1)[..., 1:].mean((0, -1))
R['prev0'] = [round(float(v), 3) for v in prev]
R['prev1'] = [round(float(v), 3) for v in prev1]
R['ind1'] = [round(float(v), 3) for v in ind]
R['ind0'] = [round(float(v), 3) for v in ind0]
log('  previous-token score (average attention to the token just before):')
log('    layer 0: ' + ', '.join(f'0.{h} {float(v):.2f}' for h, v in enumerate(prev)))
log('    layer 1: ' + ', '.join(f'1.{h} {float(v):.2f}' for h, v in enumerate(prev1)))
log('  induction score (on the second copy, average attention to the token after the earlier copy of the present token):')
log('    layer 0: ' + ', '.join(f'0.{h} {float(v):.2f}' for h, v in enumerate(ind0)))
log('    layer 1: ' + ', '.join(f'1.{h} {float(v):.2f}' for h, v in enumerate(ind)))
PT = int(prev.argmax())
IH = int(ind.argmax())
IH2 = int(ind.argsort(descending=True)[1])
R['prev_head'], R['ind_head'], R['ind_head2'] = PT, IH, IH2
log(f'  previous-token head: 0.{PT}; induction head: 1.{IH} (the next best layer-1 head is 1.{IH2} with {float(ind[IH2]):.2f})')
log(f'  a head that spread its weight evenly over the 51 to 100 earlier tokens would score about {1 / 75:.3f}')

# ---------------------------------------------------------------- 4. the patterns on a short repeated sequence
log('\n== 4. THE TWO PATTERNS ON A SHORT REPEATED SEQUENCE ([BOS] + 8 random tokens, twice)')
g = torch.Generator().manual_seed(6)
short = torch.randint(1000, 20000, (8,), generator=g)
st = torch.cat([torch.tensor([BOS]), short, short])[None]
_, cs = m2.run_with_cache(st)
labels = ['BOS'] + [f'<{int(t)}>' for t in short] * 2
A_prev = cs['blocks.0.attn.hook_pattern'][0, PT]
A_ind = cs['blocks.1.attn.hook_pattern'][0, IH]
R['short'] = dict(labels=labels, words=m2.to_str_tokens(st[0]), prev=[[round(float(v), 3) for v in r] for r in A_prev],
                  ind=[[round(float(v), 3) for v in r] for r in A_ind])
log(f'  tokens: {labels}')
i = 9 + 3          # the 4th token of the second copy
log(f'  standing on position {i} ({labels[i]}, second copy): head 0.{PT} gives {float(A_prev[i, i - 1]):.2f} to position {i - 1} (the token before); '
    f'head 1.{IH} gives {float(A_ind[i, i - 7]):.2f} to position {i - 7} ({labels[i - 7]}, the token after the first {labels[i]})')
log(f'  head 1.{IH} on the first copy (nothing to find): weight on BOS, averaged over positions 2..8: {float(A_ind[2:9, 0].mean()):.2f}')
log(f'  head 1.{IH} on the second copy: weight on the token after the earlier copy, averaged: '
    f'{float(torch.stack([A_ind[j, j - 7] for j in range(10, 17)]).mean()):.2f}')

# ---------------------------------------------------------------- 5. OV circuits: copying
log('\n== 5. COPYING: EIGENVALUE POSITIVITY OF THE OV CIRCUIT W_U W_OV W_E (1 = all eigenvalues positive, -1 = all negative)')
WE = m2.W_E - m2.W_E.mean(1, keepdim=True)
WE = WE / WE.std(1, keepdim=True)                                     # each token as layer norm would scale it (as in Part 2)
WU = m2.W_U                                                           # [512, vocab]


def positivity(M):
    """sum(lambda) / sum(|lambda|) over the eigenvalues of a square matrix (the paper's summary statistic)."""
    ev = torch.linalg.eigvals(M.double())
    return float(ev.sum().real / ev.abs().sum())


UE = WU.T                                                             # paper's W_U: [vocab, 512]
# eigenvalues of W_U W_OV W_E (vocab x vocab) = eigenvalues of W_OV (W_E W_U) (512 x 512), apart from zeros
EW = WE.T                                                             # paper's W_E: [512, vocab]
E_U = EW @ UE                                                         # [512, 512]: W_E W_U in the paper's shapes
ov_pos, self_top = {}, {}
sample = torch.arange(1000, 3000)
for L in range(2):
    for h in range(NH):
        W_OV = (m2.W_V[L, h] @ m2.W_O[L, h]).T                         # paper's W_OV: [512, 512]
        ov_pos[f'{L}.{h}'] = positivity(W_OV @ E_U)
        cols = (WE[sample] @ m2.W_V[L, h] @ m2.W_O[L, h]) @ WU         # [2000, vocab]: column t of W_U W_OV W_E, as rows
        self_top[f'{L}.{h}'] = float((cols.argmax(-1) == sample).float().mean())
R['ov_pos'] = {k: round(v, 3) for k, v in ov_pos.items()}
R['self_top'] = {k: round(v, 3) for k, v in self_top.items()}
for L in range(2):
    log(f'  layer {L} positivity: ' + ', '.join(f'{L}.{h} {ov_pos[f"{L}.{h}"]:+.2f}' for h in range(NH)))
    log(f'  layer {L} share of tokens 1000..2999 whose top output is the token itself: ' + ', '.join(f'{L}.{h} {self_top[f"{L}.{h}"]:.2f}' for h in range(NH)))

# a look at one column of the induction head's OV circuit
for w in [' Dursley', ' Paris', ' tea']:
    t = m2.to_single_token(w) if len(m2.to_tokens(w, prepend_bos=False)[0]) == 1 else None
    if t is None:
        continue
    col = (WE[t] @ m2.W_V[1, IH] @ m2.W_O[1, IH]) @ WU
    top = m2.to_str_tokens(col.topk(5).indices)
    R.setdefault('ind_ov_cols', {})[w] = top
    log(f'  OV of 1.{IH}: attend to {w!r} -> raises {top}')

# ---------------------------------------------------------------- 6. QK: the K-composition term, "same matching"
log('\n== 6. SAME MATCHING: EIGENVALUE POSITIVITY OF THE K-COMPOSITION TERM W_E^T W_QK^h W_OV^p W_E (p = a layer-0 head)')
EE = EW @ EW.T                                                        # W_E W_E^T: [512, 512]
qk_pos = {}
for h in range(NH):
    W_QK = m2.W_Q[1, h] @ m2.W_K[1, h].T                              # paper's W_Q^T W_K: [512, 512]
    for p in range(NH):
        W_OVp = (m2.W_V[0, p] @ m2.W_O[0, p]).T
        qk_pos[f'1.{h}<-0.{p}'] = positivity(W_QK @ W_OVp @ EE)       # same eigenvalues as W_E^T W_QK W_OV W_E
direct = {f'1.{h}': positivity((m2.W_Q[1, h] @ m2.W_K[1, h].T) @ EE) for h in range(NH)}
R['qk_pos_prev'] = {f'1.{h}': round(qk_pos[f'1.{h}<-0.{PT}'], 3) for h in range(NH)}
R['qk_pos_all'] = {k: round(v, 3) for k, v in qk_pos.items()}
R['qk_pos_direct'] = {k: round(v, 3) for k, v in direct.items()}
log(f'  with the previous-token head 0.{PT}: ' + ', '.join(f'1.{h} {qk_pos[f"1.{h}<-0.{PT}"]:+.2f}' for h in range(NH)))
others = [qk_pos[f'1.{IH}<-0.{p}'] for p in range(NH) if p != PT]
log(f'  head 1.{IH} with each other layer-0 head: ' + ', '.join(f'0.{p} {qk_pos[f"1.{IH}<-0.{p}"]:+.2f}' for p in range(NH) if p != PT))
log(f'  direct term W_E^T W_QK^h W_E (no layer-0 head): ' + ', '.join(f'1.{h} {direct[f"1.{h}"]:+.2f}' for h in range(NH)))
log('  the scatter of the paper (layer-1 heads): x = OV positivity, y = K-composition positivity with the previous-token head')
for h in range(NH):
    log(f'    1.{h}: ({ov_pos[f"1.{h}"]:+.2f}, {qk_pos[f"1.{h}<-0.{PT}"]:+.2f})  induction score {float(ind[h]):.2f}')

# ---------------------------------------------------------------- 7. K-composition scores (the measure of Part 5)
log('\n== 7. K-COMPOSITION SCORE ||W_QK^h W_OV^p||_F / (||W_QK^h||_F ||W_OV^p||_F), INTO THE INDUCTION HEAD')
def fro(M):
    return float(M.norm())
kc = []
for p in range(NH):
    W_OVp = (m2.W_V[0, p] @ m2.W_O[0, p]).T
    W_QK = m2.W_Q[1, IH] @ m2.W_K[1, IH].T
    kc.append(fro(W_QK @ W_OVp) / (fro(W_QK) * fro(W_OVp)))
# baseline: random matrices with the same factor shapes
base = []
for _ in range(20):
    a = torch.randn(512, 64) @ torch.randn(64, 512)
    b = torch.randn(512, 64) @ torch.randn(64, 512)
    base.append(fro(a @ b) / (fro(a) * fro(b)))
R['kcomp'] = [round(v, 4) for v in kc]
R['kcomp_base'] = round(sum(base) / len(base), 4)
log('  into 1.' + str(IH) + ': ' + ', '.join(f'0.{p} {v:.3f}' for p, v in enumerate(kc)))
log(f'  the same measure for random rank-64 matrices of these shapes: {R["kcomp_base"]:.3f} (average of 20)')

# ---------------------------------------------------------------- 8. how the key is built
log(f'\n== 8. WHERE THE INDUCTION HEAD\'S KEY COMES FROM (head 1.{IH}, the repeated sequences)')
_, cf = m2.run_with_cache(toks[:20])
q = cf['blocks.1.attn.hook_q'][:, :, IH]                              # [20, 101, 64]
# the stream that layer 1 reads = embedding + position + the outputs of the 8 layer-0 heads (+ the bias); split the key
scale = cf['blocks.1.ln1.hook_scale']                                 # [20, 101, 1]: layer norm divides by this
parts = {'embed+pos': cf['hook_embed'] + cf['hook_pos_embed']}
z0 = cf['blocks.0.attn.hook_z']                                       # [20, 101, heads, 64]
for p in range(NH):
    parts[f'0.{p}'] = z0[:, :, p] @ m2.W_O[0, p]
parts['bias'] = m2.b_O[0].expand_as(parts['embed+pos'])
total = sum(parts.values())
def centre(x):
    return x - x.mean(-1, keepdim=True)
keys = {k: (centre(v) / scale) @ m2.W_K[1, IH] for k, v in parts.items()}
kb = m2.b_K[1, IH]
dst = torch.arange(S + 1, 2 * S + 1)                                  # destinations in the second copy
src = dst - (S - 1)                                                   # the token after the earlier copy
contrib = {}
for k, kv in keys.items():
    sc = (q[:, dst] * kv[:, src]).sum(-1) / c.attn_scale              # this part's share of the score at the induction target
    other = (q[:, dst, None] * kv[:, None, :].expand(-1, len(dst), -1, -1)).sum(-1) / c.attn_scale  # [20, 50, 101]
    mask = torch.arange(101)[None, :] <= dst[:, None]
    mean_other = (other * mask).sum(-1) / mask.sum(-1)
    contrib[k] = (float(sc.mean()), float(mean_other.mean()))
bias_sc = float((q[:, dst] @ kb).mean() / c.attn_scale)
full = (q[:, dst] * cf['blocks.1.attn.hook_k'][:, src, IH]).sum(-1).mean() / c.attn_scale
R['key_parts'] = {k: [round(a, 2), round(b, 2)] for k, (a, b) in contrib.items()}
log('  share of the attention score, at the induction target vs averaged over all allowed sources:')
for k, (a, b) in sorted(contrib.items(), key=lambda kv: -abs(kv[1][0] - kv[1][1])):
    log(f'    from {k:>9}: target {a:+.2f}, average source {b:+.2f}, lift {a - b:+.2f}')
log(f'    key bias part: {bias_sc:+.2f} (the same for every source, so the softmax ignores it)')
log(f'  sum of the parts at the target {sum(a for a, _ in contrib.values()) + bias_sc:.2f}; the model\'s own score {float(full):.2f}')
R['key_full'] = round(float(full), 2)

# ---------------------------------------------------------------- 9. ablations
log('\n== 9. ABLATIONS: REPLACE ONE HEAD\'S RESULT VECTORS WITH THEIR AVERAGE OVER ALL POSITIONS AND SEQUENCES')
_, cz = m2.run_with_cache(toks, names_filter=lambda n: n.endswith('hook_z'))
abl = {}
def run_ablated(L, h):
    mean = cz[f'blocks.{L}.attn.hook_z'][:, :, h].mean((0, 1))
    def hook(z, hook):
        z[:, :, h] = mean
        return z
    out = {}
    def grab(pat, hook):
        out['p'] = pat
    logits = m2.run_with_hooks(toks, fwd_hooks=[(f'blocks.{L}.attn.hook_z', hook), ('blocks.1.attn.hook_pattern', grab)])
    loss = -logits.log_softmax(-1)[:, :-1].gather(-1, toks[:, 1:, None])[..., 0]
    i_score = float(out['p'][:, IH].diagonal(-(S - 1), -2, -1)[..., 1:].mean())
    return float(loss[:, 1:S].mean()), float(loss[:, S + 1:].mean()), i_score
base_scores = (p2['first'], p2['second'], float(ind[IH]))
log(f'  no ablation:            loss first {base_scores[0]:.2f}, second {base_scores[1]:.2f}; induction score of 1.{IH} {base_scores[2]:.2f}')
for L, h, why in [(0, PT, 'the previous-token head'), (0, int(prev.argsort(descending=True)[1]), 'the second most previous-token head'),
                  (0, int(prev.argsort()[0]), 'the least previous-token layer-0 head'), (1, IH, 'the induction head'),
                  (1, IH2, 'the second induction-like head'), (1, int(ind.argsort()[0]), 'the least induction-like layer-1 head')]:
    f, s, i_s = run_ablated(L, h)
    abl[f'{L}.{h}'] = dict(first=round(f, 3), second=round(s, 3), ind=round(i_s, 3), why=why)
    log(f'  ablate {L}.{h} ({why}): loss first {f:.2f}, second {s:.2f}; induction score of 1.{IH} {i_s:.2f}')
R['ablate'] = abl

# ---------------------------------------------------------------- 10. a sentence
log('\n== 10. A MADE-UP NAME IN A SENTENCE')
text = 'The new teacher was Mrs Brindlewick. On Monday the class met Mrs Brindlewick'
st = m2.to_tokens(text)
words = m2.to_str_tokens(text)
log(f'  tokens: {words}')
second = [j for j in range(len(words)) if words[j] == ' Mrs'][1]
first = [j for j in range(len(words)) if words[j] == ' Mrs'][0]
name_len = second - first
_, ct = m2.run_with_cache(st)
A = ct['blocks.1.attn.hook_pattern'][0, IH]
rows = []
for name, m in [('attn-only-1l', m1), ('attn-only-2l', m2)]:
    lp = m(st)[0].log_softmax(-1)
    for j in range(second, len(words) - 1):
        rows.append((name, j, words[j], words[j + 1], float(lp[j, st[0, j + 1]].exp())))
R['sentence'] = dict(text=text, words=words, rows=[list(r) for r in rows])
for name, j, w, nxt, p in rows:
    extra = ''
    if name == 'attn-only-2l':
        top = int(A[j].argmax())
        r_h = ct['blocks.1.attn.hook_z'][0, j, IH] @ m2.W_O[1, IH]
        eff = float((r_h @ m2.W_U)[st[0, j + 1]])
        extra = f'; head 1.{IH} reads {words[top]!r} (pos {top}) with {float(A[j, top]):.2f}, adds {eff:+.2f} to the logit of {nxt!r}'
    log(f'  {name}: at {w!r} (pos {j}) the next token {nxt!r} gets probability {p:.2f}{extra}')
R['sentence']['first_mrs'] = first

save('part6', R)
log('\nsaved results/part6.json')
