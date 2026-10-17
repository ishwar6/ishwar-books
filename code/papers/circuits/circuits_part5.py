"""Part 5: two-layer attention-only models, how heads compose (attn-only-2l, TransformerLens 3.x).

Prints and saves:
  1. the model and how many terms each path expansion has (direct path, 16 head terms, 64 virtual heads);
  2. the path expansion of the logits done by hand on real text, with the attention patterns and the layer-norm
     scales frozen: direct path + one term per head + one term per pair of heads adds up to the model's logits;
  3. term importance: the loss when only some orders of terms are kept (uniform, order 0, order <= 1, order <= 2),
     the split of the head terms by layer, and the loss change when one virtual head or one head term is removed;
  4. the composition scores of the paper (Q-, K- and V-composition, Frobenius norms), with a baseline from random
     matrices of the same shapes, plus a previous-token score of every layer-0 head;
  5. the path expansion of the attention scores of each layer-1 head: no composition, Q-, K- and Q-and-K terms
     add up to the real scores; how much the pattern changes when one kind of term (or one head's K term) is removed;
  6. a virtual attention head: its pattern A^{h2} A^{h1} is a real attention pattern; which one looks two tokens back.
Text: the first 24 x 128 tokens of the WikiText-2 test split (raw). CPU only."""
import math
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from common import Log, save

torch.manual_seed(0)
torch.set_grad_enabled(False)
log = Log('part5')
R = {}

m = HookedTransformer.from_pretrained('attn-only-2l', device='cpu').eval()
c = m.cfg
NH, D = c.n_heads, c.d_model

# ---------------------------------------------------------------- 1. the model and the number of terms
log('== 1. THE MODEL AND THE NUMBER OF TERMS (attn-only-2l)')
log(f'  layers {c.n_layers}, heads per layer {NH}, d_model {D}, d_head {c.d_head}, vocabulary {c.d_vocab:,}; '
    f'd_head / d_model = {c.d_head / D:.3f}')
n_ind, n_virt = 2 * NH, NH * NH
log(f'  logit path expansion: 1 direct path + {n_ind} single-head terms + {NH} x {NH} = {n_virt} virtual-head terms '
    f'= {1 + n_ind + n_virt} terms')
log(f'  the paper\'s model has 12 heads per layer: 1 + 24 + 144 = {1 + 24 + 144} terms')
R['terms'] = dict(ind=n_ind, virt=n_virt)

# ---------------------------------------------------------------- the text
ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')
text = ' '.join(t.strip() for t in ds['text'] if len(t.strip()) > 0 and not t.strip().startswith('='))
ids = torch.tensor(m.tokenizer(text[:100000])['input_ids'])          # to_tokens would cut at 1,024 tokens
NSEQ, LEN = 24, 128
toks = torch.stack([torch.cat([torch.tensor([m.tokenizer.bos_token_id]), ids[i * LEN:(i + 1) * LEN]]) for i in range(NSEQ)])
B, P = toks.shape
log(f'  text: WikiText-2 test, {NSEQ} sequences of {LEN} tokens (+ BOS), {NSEQ * LEN:,} predictions')
R['data'] = dict(nseq=NSEQ, len=LEN)

logits, cache = m.run_with_cache(toks)


def loss_of(lg):
    lp = lg[:, :-1].log_softmax(-1)
    return float(-lp.gather(-1, toks[:, 1:, None]).mean())


loss_model = loss_of(logits)

# ---------------------------------------------------------------- 2. path expansion of the logits, by hand
log('\n== 2. PATH EXPANSION OF THE LOGITS ON REAL TEXT (patterns and layer-norm scales frozen)')
cen = lambda x: x - x.mean(-1, keepdim=True)
x0 = cache['hook_embed'] + cache['hook_pos_embed']                       # [B, P, 512]: token + position
s0, s1, sf = cache['blocks.0.ln1.hook_scale'], cache['blocks.1.ln1.hook_scale'], cache['ln_final.hook_scale']
A0, A1 = cache['blocks.0.attn.hook_pattern'], cache['blocks.1.attn.hook_pattern']   # [B, H, P, P]


def head_out(A, xin, L, h, bias=True):
    v = xin @ m.W_V[L, h] + (m.b_V[L, h] if bias else 0)
    return (A[:, h] @ v) @ m.W_O[L, h]


x0n = cen(x0) / s0
o0 = torch.stack([head_out(A0, x0n, 0, h) for h in range(NH)])          # [H, B, P, 512] layer-0 head outputs
x1 = x0 + o0.sum(0) + m.b_O[0]
d_x1 = float((x1 - cache['blocks.1.hook_resid_pre']).abs().max())
base1 = cen(x0 + m.b_O[0]) / s1                                          # what a layer-1 head reads from the direct path
o1_ind = torch.stack([head_out(A1, base1, 1, h) for h in range(NH)])    # [H2, B, P, 512]
o1_virt = torch.stack([torch.stack([head_out(A1, cen(o0[h1]) / s1, 1, h2, bias=False) for h1 in range(NH)])
                       for h2 in range(NH)])                             # [H2, H1, B, P, 512]
# stream-space terms (everything is linear from here: centre, divide by the frozen final scale, unembed)
T_direct = x0 + m.b_O[0] + m.b_O[1]
unembed = lambda s: (cen(s) / sf) @ m.W_U + m.b_U
full = T_direct + o0.sum(0) + o1_ind.sum(0) + o1_virt.sum((0, 1))
d_logits = float((unembed(full) - logits).abs().max())
log(f'  stream before layer 1 rebuilt from the terms: max |diff| = {d_x1:.1e}')
log(f'  logits rebuilt as direct + {n_ind} head terms + {n_virt} virtual heads: max |diff| from the model = {d_logits:.1e} '
    f'(logits range {float(logits.min()):.1f} .. {float(logits.max()):.1f})')
R['rebuild'] = dict(x1=d_x1, logits=d_logits)
# how big each kind of term is in the final stream (mean size of the vector it adds, last position excluded)
sz = lambda t: float(cen(t).norm(dim=-1).mean())
R['size'] = dict(direct=sz(T_direct), heads_L0=sz(o0.sum(0)), heads_L1=sz(o1_ind.sum(0)), virtual=sz(o1_virt.sum((0, 1))))
log('  mean size of the vector each group adds to the final stream: ' +
    ', '.join(f'{k} {v:.1f}' for k, v in R['size'].items()))

# ---------------------------------------------------------------- 3. term importance
log('\n== 3. TERM IMPORTANCE: LOSS WHEN ONLY SOME TERMS ARE KEPT (nats per token, lower is better)')
uni = math.log(c.d_vocab)
L0 = loss_of(unembed(T_direct))
L_heads0 = loss_of(unembed(T_direct + o0.sum(0)))
L_heads1 = loss_of(unembed(T_direct + o1_ind.sum(0)))
L1 = loss_of(unembed(T_direct + o0.sum(0) + o1_ind.sum(0)))
L2 = loss_of(unembed(full))
R['loss'] = dict(uniform=uni, order0=L0, order1=L1, order2=L2, model=loss_model, dir_L0=L_heads0, dir_L1=L_heads1)
log(f'  uniform guess over {c.d_vocab:,} tokens: ln({c.d_vocab}) = {uni:.2f}')
log(f'  order 0 (direct path only):                 {L0:.2f}   marginal {L0 - uni:+.2f}  ({(L0 - uni):+.2f} per term, 1 term)')
log(f'  order <= 1 (+ {n_ind} single-head terms):         {L1:.2f}   marginal {L1 - L0:+.2f}  ({(L1 - L0) / n_ind:+.3f} per term)')
log(f'  order <= 2 (+ {n_virt} virtual heads):            {L2:.2f}   marginal {L2 - L1:+.2f}  ({(L2 - L1) / n_virt:+.4f} per term)')
log(f'  the model itself:                           {loss_model:.2f}')
log('  head terms split by layer:')
log(f'    layer-0 heads: {L_heads0 - L0:+.2f} vs direct path only, {L1 - L_heads1:+.2f} vs direct path + layer-1 heads')
log(f'    layer-1 heads: {L_heads1 - L0:+.2f} vs direct path only, {L1 - L_heads0:+.2f} vs direct path + layer-0 heads')
# one term at a time, removed from the full sum
drop_v = torch.zeros(NH, NH)
for h2 in range(NH):
    for h1 in range(NH):
        drop_v[h2, h1] = loss_of(unembed(full - o1_virt[h2, h1])) - L2
drop_h = {}
for h in range(NH):
    drop_h[f'0.{h}'] = loss_of(unembed(full - o0[h])) - L2
    drop_h[f'1.{h}'] = loss_of(unembed(full - o1_ind[h])) - L2
R['drop_virtual'] = [[round(float(v), 4) for v in r] for r in drop_v]
R['drop_head'] = {k: round(v, 4) for k, v in drop_h.items()}
order = drop_v.flatten().argsort(descending=True)
top_v = [(f'1.{int(i) // NH} o 0.{int(i) % NH}', float(drop_v.flatten()[i])) for i in order[:5]]
log('  remove ONE term from the full sum, loss goes up by:')
log('    largest virtual heads (h2 o h1): ' + ', '.join(f'{n} {v:+.3f}' for n, v in top_v))
log(f'    virtual heads: {int((drop_v > 0.01).sum())} of {n_virt} above +0.01; median {float(drop_v.median()):+.4f}')
hs = sorted(drop_h.items(), key=lambda kv: -kv[1])
log('    single heads, largest: ' + ', '.join(f'{k} {v:+.3f}' for k, v in hs[:6]))
log('    single heads, smallest: ' + ', '.join(f'{k} {v:+.3f}' for k, v in hs[-3:]))
R['top_virtual'] = top_v
# where do the virtual heads get their information? split each layer-0 head's output into the part it copies from
# the start token (position 0, where many heads park their attention) and the part it copies from real tokens
o0_bos = torch.stack([A0[:, h, :, :1] @ ((x0n[:, :1] @ m.W_V[0, h]) @ m.W_O[0, h]) for h in range(NH)])
o0_tok = o0 - o0_bos
virt_from = lambda o: sum(head_out(A1, cen(o.sum(0)) / s1, 1, h2, bias=False) for h2 in range(NH))
v_bos, v_tok = virt_from(o0_bos), virt_from(o0_tok)
d_split = float((v_bos + v_tok - o1_virt.sum((0, 1))).abs().max())
L_vb = loss_of(unembed(T_direct + o0.sum(0) + o1_ind.sum(0) + v_bos))
L_vt = loss_of(unembed(T_direct + o0.sum(0) + o1_ind.sum(0) + v_tok))
R['virt_split'] = dict(bos=L_vb - L1, tok=L_vt - L1, check=d_split)
log(f'  virtual heads split by what the layer-0 head read (check {d_split:.1e}): from the start token {L_vb - L1:+.2f}, '
    f'from real tokens {L_vt - L1:+.2f} (each added alone to order <= 1)')

# ---------------------------------------------------------------- 4. composition scores
log('\n== 4. COMPOSITION SCORES  ||W_A W_B||_F / (||W_A||_F ||W_B||_F)  (paper shapes, layer-0 head h1 -> layer-1 head h2)')
W_OV = lambda L, h: (m.W_V[L, h] @ m.W_O[L, h]).T                       # paper: W_O W_V, 512 x 512
W_QK = lambda L, h: m.W_Q[L, h] @ m.W_K[L, h].T                          # paper: W_Q^T W_K, 512 x 512
fro = lambda M: float(M.norm())
score = lambda A, Bm: fro(A @ Bm) / (fro(A) * fro(Bm))
comp = {k: torch.zeros(NH, NH) for k in 'QKV'}
for h2 in range(NH):
    for h1 in range(NH):
        ov1 = W_OV(0, h1)
        comp['Q'][h2, h1] = score(W_QK(1, h2).T, ov1)
        comp['K'][h2, h1] = score(W_QK(1, h2), ov1)
        comp['V'][h2, h1] = score(W_OV(1, h2), ov1)
# baseline: random low-rank matrices of the same shapes (512 x 64 times 64 x 512)
g = torch.Generator().manual_seed(0)
rl = lambda: torch.randn(D, c.d_head, generator=g) @ torch.randn(c.d_head, D, generator=g)
base = sum(score(rl(), rl()) for _ in range(200)) / 200
log(f'  random baseline (two random rank-64 512 x 512 matrices, mean of 200): {base:.4f}')
R['comp_baseline'] = base
R['comp'] = {k: [[round(float(v), 4) for v in r] for r in M] for k, M in comp.items()}
for k in 'QKV':
    M = comp[k]
    i = int(M.argmax())
    log(f'  {k}-composition: max {float(M.max()):.3f} (head 1.{i // NH} reads head 0.{i % NH}), mean {float(M.mean()):.3f}, '
        f'pairs above baseline + 0.05: {int((M > base + 0.05).sum())} of {NH * NH}')
for k in 'QK':
    log(f'  {k}-composition, mean over the 8 layer-1 heads, per layer-0 head: ' +
        ', '.join(f'0.{h} {float(comp[k][:, h].mean()):.3f}' for h in range(NH)))
# previous-token score of each layer-0 head: mean attention from position i to i-1
prev = [float(A0[:, h].diagonal(-1, -2, -1).mean()) for h in range(NH)]
R['prev_score'] = prev
log('  layer-0 heads, mean attention to the previous token: ' + ', '.join(f'0.{h} {v:.2f}' for h, v in enumerate(prev)))
PT = int(torch.tensor(prev).argmax())
R['prev_head'] = PT
log(f'  K-composition with the previous-token head 0.{PT}, per layer-1 head: ' +
    ', '.join(f'1.{h} {float(comp["K"][h, PT]):.3f}' for h in range(NH)))
# induction score of each layer-1 head (only to label the heads; Part 6 explains it)
rnd = torch.randint(1000, 10000, (8, 50), generator=g)
rep = torch.cat([torch.full((8, 1), m.tokenizer.bos_token_id), rnd, rnd], 1)
_, rc = m.run_with_cache(rep)
pr = rc['blocks.1.attn.hook_pattern']                                    # [8, H, 101, 101]
ind = [float(pr[:, h].diagonal(-49, -2, -1)[:, 1:].mean()) for h in range(NH)]
R['induction_score'] = ind
log('  layer-1 heads, induction score on repeated random tokens (Part 6): ' + ', '.join(f'1.{h} {v:.2f}' for h, v in enumerate(ind)))

# ---------------------------------------------------------------- 5. path expansion of the attention scores
log('\n== 5. PATH EXPANSION OF THE ATTENTION SCORES OF EACH LAYER-1 HEAD (same text, layer-0 patterns frozen)')
scale = c.attn_scale
q_parts_in = [base1] + [cen(o0[h]) / s1 for h in range(NH)]              # query side = key side = sum of these
mask = torch.ones(P, P).triu(1).bool()
real_scores = cache['blocks.1.attn.hook_attn_scores']
R['qk'] = {}


def pattern(sc):
    return sc.masked_fill(mask, float('-inf')).softmax(-1)


def tv(Aa, Ab):                       # total variation distance between two patterns, mean over rows: 0 = same, 1 = disjoint
    return float(0.5 * (Aa - Ab).abs().sum(-1)[:, 1:].mean())


maxdiff = 0.0
for h2 in range(NH):
    qd = q_parts_in[0] @ m.W_Q[1, h2] + m.b_Q[1, h2]
    kd = q_parts_in[0] @ m.W_K[1, h2] + m.b_K[1, h2]
    qh = torch.stack([x @ m.W_Q[1, h2] for x in q_parts_in[1:]])          # [H1, B, P, 64]
    kh = torch.stack([x @ m.W_K[1, h2] for x in q_parts_in[1:]])
    S_none = qd @ kd.transpose(-1, -2) / scale
    S_q = torch.stack([q @ kd.transpose(-1, -2) for q in qh]) / scale     # per layer-0 head on the query side
    S_k = torch.stack([qd @ k.transpose(-1, -2) for k in kh]) / scale
    S_qk = (qh.sum(0) @ kh.sum(0).transpose(-1, -2)) / scale
    total = S_none + S_q.sum(0) + S_k.sum(0) + S_qk
    real = real_scores[:, h2]
    maxdiff = max(maxdiff, float((total - real).masked_fill(mask, 0).abs().max()))
    A_real = pattern(total)
    d = dict(no_K=tv(A_real, pattern(total - S_k.sum(0))), no_Q=tv(A_real, pattern(total - S_q.sum(0))),
             no_QK=tv(A_real, pattern(total - S_qk)), only_none=tv(A_real, pattern(S_none)),
             no_K_per_head=[tv(A_real, pattern(total - S_k[h1])) for h1 in range(NH)])
    low = lambda S: float(S.masked_fill(mask, 0).abs().sum() / (~mask).sum() / B)
    d['size'] = dict(none=low(S_none), Q=low(S_q.sum(0)), K=low(S_k.sum(0)), QK=low(S_qk))
    R['qk'][h2] = d
log(f'  no-composition + Q + K + Q-and-K terms vs the model\'s scores: max |diff| = {maxdiff:.1e}')
log('  how much the pattern changes (total variation, 0 = same, 1 = completely different) when a kind of term is removed:')
for h2 in range(NH):
    d = R['qk'][h2]
    log(f'    head 1.{h2}: without K terms {d["no_K"]:.2f}, without Q terms {d["no_Q"]:.2f}, without Q-and-K {d["no_QK"]:.2f}; '
        f'no-composition term alone {d["only_none"]:.2f}')
R['qk_maxdiff'] = maxdiff
log('  without the K term of ONE layer-0 head (rows: layer-1 heads; columns: layer-0 heads 0..7):')
for h2 in range(NH):
    log(f'    1.{h2}: ' + ' '.join(f'{v:.2f}' for v in R['qk'][h2]['no_K_per_head']))

# ---------------------------------------------------------------- 6. a virtual attention head
log('\n== 6. A VIRTUAL ATTENTION HEAD: PATTERN A^{h2} A^{h1}, OV MATRIX W_OV^{h2} W_OV^{h1}')


def where(Am):                         # mean attention to the start token, to itself, one back, two back (positions >= 5)
    return [round(float(Am[:, 5:, 0].mean()), 2)] + [round(float(Am.diagonal(-k, -2, -1)[:, 5:].mean()), 2) for k in range(3)]


log(f'  {NH} x {NH} = {NH * NH} virtual heads. Where heads look (mean weight on: start token, itself, 1 back, 2 back):')
R['where'] = {}
for L, A in [(0, A0), (1, A1)]:
    for h in range(NH):
        R['where'][f'{L}.{h}'] = where(A[:, h])
    log(f'    layer {L}: ' + '; '.join(f'{L}.{h} {R["where"][f"{L}.{h}"]}' for h in range(NH)))
two_back = torch.zeros(NH, NH)
for h2 in range(NH):
    for h1 in range(NH):
        two_back[h2, h1] = float((A1[:, h2] @ A0[:, h1]).diagonal(-2, -2, -1)[:, 5:].mean())
R['virtual'] = {}
for tag, (h2, h1) in [('most important', divmod(int(drop_v.argmax()), NH)), ('most two-back', divmod(int(two_back.argmax()), NH))]:
    Av = A1[:, h2] @ A0[:, h1]
    d = dict(h2=h2, h1=h1, where=where(Av), rows=float((Av.sum(-1) - 1).abs().max()), upper=float(Av.triu(1).abs().max()),
             drop=float(drop_v[h2, h1]), vcomp=float(comp['V'][h2, h1]))
    R['virtual'][tag] = d
    log(f'  {tag}: 1.{h2} o 0.{h1}: pattern {d["where"]}; rows add up to 1 (max |sum - 1| {d["rows"]:.1e}); '
        f'future entries {d["upper"]:.0f}; V-composition score {d["vcomp"]:.3f}; removing it: loss {d["drop"]:+.3f}')

save('part5', R)
