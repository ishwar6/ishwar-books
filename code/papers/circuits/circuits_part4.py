"""Part 4: one-layer models, read as skip-trigrams, on the real attn-only-1l (TransformerLens 3.x, CPU).

Prints and saves:
  1. how big the expanded QK and OV circuits of the model are, next to the number of weights it really stores;
  2. skip-trigram tables: fix a source token, list the destinations that attend to it most (QK) and the out tokens
     it raises most (OV), for several heads;
  3. skip-trigrams at work in the whole model: the "Ralph ... R -> alph" copy and "perfect ... looks -> perfect";
  4. primarily positional heads: the attention pattern from the position embeddings alone, the real pattern on text,
     and what those heads write;
  5. a skip-trigram "bug": head 0.2 links " keep" with " bay" and " mind" in a factored form;
  6. the OV circuit is low-rank, and its non-zero eigenvalues come from a 64 x 64 matrix;
  7. eigenvalues as a copying statistic for every head, against random matrices, and other copying measures
     (the diagonal, the top-1 and top-5 self share); the trace equals the sum of the eigenvalues.
Every number of Part 4 that does not come from the paper comes from this script (results/part4_stdout.txt)."""
import os
import torch
from transformer_lens import HookedTransformer
from common import Log, save

torch.manual_seed(0)
torch.set_grad_enabled(False)
log = Log('part4')
R = {}
HERE = os.path.dirname(os.path.abspath(__file__))

m = HookedTransformer.from_pretrained('attn-only-1l', device='cpu').eval()
m.set_use_attn_result(True)
c = m.cfg
V, NH = c.d_vocab, c.n_heads
WE = m.W_E - m.W_E.mean(1, keepdim=True)
WE = WE / WE.std(1, keepdim=True)          # what is left of layer norm (Part 2): every token vector scaled to the same size
WU = m.W_U
tok = m.to_single_token
st = lambda ids: [m.to_string(int(i)) for i in ids]
Q = [WE @ m.W_Q[0, h] for h in range(NH)]  # the query of every token, per head: [vocab, 64]
K = [WE @ m.W_K[0, h] for h in range(NH)]  # the key of every token


def ov_col(h, t):
    """Column t of the OV circuit W_U W_OV W_E: what reading token t does to the 48,262 logits."""
    return (WE[t] @ m.W_V[0, h] @ m.W_O[0, h]) @ WU


def qk_col(h, t):
    """Column t of the QK circuit W_E^T W_QK W_E: the score every destination token gives to source t."""
    return Q[h] @ K[h][t] / c.attn_scale


# ---------------------------------------------------------------- 1. sizes
log('== 1. HOW BIG THE EXPANDED CIRCUITS ARE (attn-only-1l)')
n_params = sum(p.numel() for p in m.parameters())
per = V * V
R['sizes'] = dict(vocab=V, heads=NH, per_circuit=per, all_circuits=2 * NH * per, params=n_params)
log(f'  vocabulary {V:,} tokens, {NH} heads; one expanded circuit is {V:,} x {V:,} = {per / 1e9:.2f} billion numbers')
log(f'  QK + OV for all {NH} heads: {2 * NH * per / 1e9:.1f} billion numbers; the model stores {n_params / 1e6:.1f} million weights')

# ---------------------------------------------------------------- 2. skip-trigram tables
log('\n== 2. SKIP-TRIGRAMS READ FROM THE WEIGHTS: fix a source, top destinations (QK) and top out tokens (OV)')
R['tables'] = []
for src, h in [(' perfect', 0), (' large', 0), (' Ralph', 0), (' Lloyd', 0), (' two', 6), (' couldn', 6), (' Paris', 7), (' two', 7)]:
    t = tok(src)
    qk, ov = qk_col(h, t), ov_col(h, t)
    d, o = qk.topk(5), ov.topk(5)
    row = dict(src=src, head=f'0.{h}', dest=st(d.indices), out=st(o.indices),
               qk_top=round(float(d.values[0]), 2), qk_median=round(float(qk.median()), 2),
               ov_top=round(float(o.values[0]), 2), ov_median=round(float(ov.median()), 2))
    R['tables'].append(row)
    log(f'  head 0.{h}  source {src!r:11} dest {row["dest"]}  out {row["out"]}')
log('  scale: ' + '; '.join(f'0.{r["head"][-1]} {r["src"]!r}: top QK {r["qk_top"]} (median {r["qk_median"]}), top OV {r["ov_top"]} (median {r["ov_median"]})'
                             for r in R['tables'][:3]))

# ---------------------------------------------------------------- 3. skip-trigrams at work
log('\n== 3. SKIP-TRIGRAMS AT WORK IN THE WHOLE MODEL (probability of the next token)')
R['work'] = []
for a, b, w in [('My friend Ralph called me today.\n"R', 'My friend Bob called me today.\n"R', 'alph'),
                ('We visited Ralph and his dog.\n"R', 'We visited Tom and his dog.\n"R', 'alph'),
                ('The perfect gift for her. It looks', 'The nice gift for her. It looks', ' perfect')]:
    pa = float(m(a)[0, -1].softmax(-1)[tok(w)])
    pb = float(m(b)[0, -1].softmax(-1)[tok(w)])
    R['work'].append(dict(a=a, b=b, w=w, pa=pa, pb=pb))
    log(f'  {a!r:42} p({w!r}) = {pa:.4f}   |   {b!r:40} p({w!r}) = {pb:.4f}')
prompt = 'My friend Ralph called me today.\n"R'
lg, cache = m.run_with_cache(prompt)
toks = m.to_str_tokens(prompt)
scale = cache['ln_final.hook_scale'][0, -1]
res = cache['blocks.0.attn.hook_result'][0, -1]                       # [heads, 512]: what each head adds at "R"
push = [float(res[h] @ WU[:, tok('alph')] / scale) for h in range(NH)]
j = toks.index(' Ralph')
A0 = float(cache['blocks.0.attn.hook_pattern'][0, 0, -1, j])
R['ralph'] = dict(tokens=toks, push=push, attn_00=A0)
log(f'  tokens: {toks}')
log(f'  each head\'s push on the logit of "alph" at the last token: ' + ', '.join(f'0.{h}: {v:+.2f}' for h, v in enumerate(push)))
log(f'  head 0.0 at "R" puts {A0:.2f} of its attention on " Ralph"')

# ---------------------------------------------------------------- 4. primarily positional heads
log('\n== 4. PRIMARILY POSITIONAL HEADS')
P = m.W_pos[:201]
Pn = (P - P.mean(1, keepdim=True)); Pn = Pn / Pn.std(1, keepdim=True)
R['pos_only'] = {}
log('  attention from the position embeddings alone (query at position 200; weight on the same token, 1 back, 2 back):')
for h in range(NH):
    s = (Pn[200] @ m.W_Q[0, h]) @ (Pn @ m.W_K[0, h]).T / c.attn_scale
    a = s.softmax(-1)
    R['pos_only'][h] = [round(float(a[200 - d]), 3) for d in range(3)]
    log(f'    head 0.{h}: {R["pos_only"][h]}')
text = open(os.path.join(HERE, 'part1.md')).read()
text = text[text.index('---', 4) + 3:]                                # skip the frontmatter
toks_t = m.to_tokens(text)[:, :1000]
_, ct = m.run_with_cache(toks_t)
Pat = ct['blocks.0.attn.hook_pattern'][0]
R['pos_text'] = {}
log(f'  real pattern on {toks_t.shape[1]} tokens of text (Part 1 of this series), mean weight on 0, 1, 2, 3 tokens back (positions 100+):')
for h in range(NH):
    R['pos_text'][h] = [round(float(torch.diagonal(Pat[h], -d)[100:].mean()), 3) for d in range(4)]
    log(f'    head 0.{h}: {R["pos_text"][h]}')
R['pos_ov'] = {}
for src, h in [(' corresponding', 5), (' couldn', 5), (' shouldn', 5), (' couldn', 1), (' shouldn', 1)]:
    o = ov_col(h, tok(src)).topk(5)
    R['pos_ov'][f'{src}|{h}'] = st(o.indices)
    log(f'  OV of head 0.{h}, source {src!r:15} -> {st(o.indices)}')

# ---------------------------------------------------------------- 5. a skip-trigram bug
log('\n== 5. A SKIP-TRIGRAM BUG: head 0.2 and " keep"')
H = 2
k = tok(' keep')
ov, qk = ov_col(H, k), qk_col(H, k)
outs = [' mind', ' bay', ' toes', ' wraps', ' forefront']
dests = [' in', ' at', ' on', ' under', ' going']
R['bug'] = dict(ov={w: round(float(ov[tok(w)]), 2) for w in outs}, ov_median=round(float(ov.median()), 2),
                qk={w: round(float(qk[tok(w)]), 2) for w in dests}, qk_median=round(float(qk.median()), 2),
                ov_top=st(ov.topk(6).indices), qk_top=st(qk.topk(6).indices))
log(f'  OV column " keep": {R["bug"]["ov"]}  (median over all tokens {R["bug"]["ov_median"]}); top {R["bug"]["ov_top"]}')
log(f'  QK column " keep": {R["bug"]["qk"]}  (median {R["bug"]["qk_median"]}); top destinations {R["bug"]["qk_top"]}')
R['bug']['prompts'] = []
for p in ['You need to keep this in', 'You need to keep the dogs at']:
    lg, cc = m.run_with_cache(p)
    tk = m.to_str_tokens(p)
    sc = cc['ln_final.hook_scale'][0, -1]
    pr = lg[0, -1].softmax(-1)
    jk = tk.index(' keep')
    a = float(cc['blocks.0.attn.hook_pattern'][0, H, -1, jk])
    v = cc['blocks.0.attn.hook_v'][0, jk, H]                          # the value vector of " keep" in head 0.2
    via = {w: float(a * (v @ m.W_O[0, H] @ WU[:, tok(w)]) / sc) for w in [' mind', ' bay']}
    head = {w: float(cc['blocks.0.attn.hook_result'][0, -1, H] @ WU[:, tok(w)] / sc) for w in [' mind', ' bay']}
    best = {w: max(range(NH), key=lambda h: float(cc['blocks.0.attn.hook_result'][0, -1, h] @ WU[:, tok(w)])) for w in [' mind']}
    row = dict(prompt=p, attn=a, via_keep=via, head=head, p={w: float(pr[tok(w)]) for w in [' mind', ' bay']},
               top=st(lg[0, -1].topk(3).indices),
               h7_mind=float(cc['blocks.0.attn.hook_result'][0, -1, 7] @ WU[:, tok(' mind')] / sc))
    R['bug']['prompts'].append(row)
    log(f'  {p!r}: head 0.2 puts {a:.2f} on " keep"; through " keep" it pushes mind {via[" mind"]:+.2f}, bay {via[" bay"]:+.2f}  '
        f'(ratio bay/mind {via[" bay"] / via[" mind"]:.2f}); whole head: mind {head[" mind"]:+.2f}, bay {head[" bay"]:+.2f}')
    log(f'      whole model: p(mind) = {row["p"][" mind"]:.4f}, p(bay) = {row["p"][" bay"]:.6f}; top 3 {row["top"]}; head 0.7 pushes mind {row["h7_mind"]:+.2f}')

# ---------------------------------------------------------------- 6. low rank, and the 64 x 64 trick
log('\n== 6. THE OV CIRCUIT IS LOW-RANK: ITS NON-ZERO EIGENVALUES LIVE IN A 64 x 64 MATRIX')
log(f'  OV circuit (row form) = (W_E W_V)(W_O W_U): [{V:,} x 64] times [64 x {V:,}], so its rank is at most 64 of {V:,}')
log('  eigenvalues of AB and BA are the same (apart from zeros), so we take those of (W_O W_U)(W_E W_V): 64 x 64')
# check the trick on a small piece: 300 tokens only, so the full matrix fits
ids = torch.arange(1000, 1300)
Asm = (WE[ids] @ m.W_V[0, 0]).double()                                # [300, 64]
Bsm = (m.W_O[0, 0] @ WU[:, ids]).double()                              # [64, 300]
big = torch.linalg.eigvals(Asm @ Bsm)
small = torch.linalg.eigvals(Bsm @ Asm)
bs = big[big.abs().argsort(descending=True)][:64]
ss = small[small.abs().argsort(descending=True)]
R['trick'] = dict(n=300, max_diff=float((bs.abs().sort().values - ss.abs().sort().values).abs().max()),
                  big_rest=float(big[big.abs().argsort(descending=True)][64:].abs().max()), sum_big=float(big.sum().real), sum_small=float(small.sum().real))
log(f'  check on a 300-token piece of head 0.0: 300 x 300 has 300 eigenvalues; the 64 largest match the 64 x 64 ones '
    f'to {R["trick"]["max_diff"]:.1e}; the other 236 are at most {R["trick"]["big_rest"]:.1e}')

# ---------------------------------------------------------------- 7. eigenvalues as a copying statistic
log('\n== 7. EIGENVALUES OF EACH HEAD\'S OV CIRCUIT (whole vocabulary), AND OTHER COPYING MEASURES')
R['eig'] = {}
ids = torch.arange(1000, 3000)
for h in range(NH):
    small = (m.W_O[0, h] @ WU).double() @ (WE @ m.W_V[0, h]).double()  # [64, 64]
    ev = torch.linalg.eigvals(small)
    score = float(ev.sum().real / ev.abs().sum())
    n_pos_real = int(((ev.imag.abs() < 1e-9) & (ev.real > 0)).sum())
    n_complex = int((ev.imag.abs() >= 1e-9).sum())
    n_right = int((ev.real > 0).sum())
    trace = float(small.trace())
    diag = ((WE @ m.W_V[0, h] @ m.W_O[0, h]) * WU.T).sum(-1).double()  # every diagonal entry of the 48,262 x 48,262 circuit
    cols = (WE[ids] @ m.W_V[0, h] @ m.W_O[0, h]) @ WU
    top1 = float((cols.argmax(-1) == ids).float().mean())
    top5 = float((cols.topk(5, -1).indices == ids[:, None]).any(-1).float().mean())
    R['eig'][h] = dict(score=score, n_pos_real=n_pos_real, n_complex=n_complex, n_right=n_right, trace=trace,
                       diag_sum=float(diag.sum()), diag_pos=float((diag > 0).float().mean()), top1=top1, top5=top5,
                       ev=[[round(float(e.real), 2), round(float(e.imag), 2)] for e in ev])
    log(f'  head 0.{h}: copy score sum(l)/sum(|l|) = {score:+.3f}; {n_right}/64 with positive real part '
        f'({n_pos_real} real and positive, {n_complex} complex); trace {trace:,.0f} = sum of diagonal {float(diag.sum()):,.0f}')
log('  other measures (diagonal over all 48,262 tokens; top-1 and top-5 self share over tokens 1000..2999):')
for h in range(NH):
    e = R['eig'][h]
    log(f'    head 0.{h}: diagonal > 0 for {e["diag_pos"]:.2f} of tokens; top-1 self {e["top1"]:.2f}; top-5 self {e["top5"]:.2f}')
n_copy = sum(R['eig'][h]['score'] > 0.5 for h in range(NH))
log(f'  heads with copy score above 0.5: {n_copy} of {NH}')
R['n_copy'] = n_copy

# random matrices with the same shapes
rs = []
g = torch.Generator().manual_seed(0)
for s in range(20):
    E = torch.randn(V, 64, generator=g, dtype=torch.float64)             # stands for W_E W_V
    B = torch.randn(64, V, generator=g, dtype=torch.float64)             # stands for W_O W_U
    ev = torch.linalg.eigvals(B @ E)
    if s == 0:
        R['random_ev'] = [[round(float(e.real), 2), round(float(e.imag), 2)] for e in ev]
    rs.append(dict(score=float(ev.sum().real / ev.abs().sum()), n_right=int((ev.real > 0).sum()),
                   n_complex=int((ev.imag.abs() >= 1e-9).sum())))
R['random'] = rs
sc = [r['score'] for r in rs]
log(f'  20 random Gaussian heads of the same shapes: copy score from {min(sc):+.3f} to {max(sc):+.3f} (mean {sum(sc) / len(sc):+.3f}); '
    f'positive real part {min(r["n_right"] for r in rs)} to {max(r["n_right"] for r in rs)} of 64; '
    f'complex {min(r["n_complex"] for r in rs)} to {max(r["n_complex"] for r in rs)} of 64')

save('part4', R)
