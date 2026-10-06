"""Part 2, Appendix A by hand: Eq. 5 to 8 (self-attention and multi-head self-attention) on a tiny example that fits
on paper: N = 4 tokens, D = 6, k = 2 heads, D_h = 3, with small fixed integer weights. Prints every matrix (z, U_qkv,
q, k, v, the scores, the softmax rows that sum to 1, SA(z) per head, the concatenation and U_msa) and then shows
that reordering the four input tokens reorders the output rows in exactly the same way (permutation equivariance),
which is why ViT needs position embeddings. Runs on the CPU in a second.
Writes results/part2_attn.json and results/part2_attn_stdout.txt."""
import math
import torch
from common import Log, save

torch.manual_seed(0)
log = Log('part2_attn')
R = {}
N, D, k = 4, 6, 2
Dh = D // k
names = ['t0', 't1', 't2', 't3']


def show(name, M, rows=None, dec=0):
    log(f'  {name}  shape {tuple(M.shape)}')
    for i, r in enumerate(M):
        lab = (rows[i] if rows else '') + ' '
        log('    ' + f'{lab:>6}' + ' '.join((f'{x:6.0f}' if dec == 0 else f'{x:6.{dec}f}') for x in r.tolist()))


g = torch.Generator().manual_seed(2037)   # a seed chosen so that the scores stay small enough to check by hand
z = torch.randint(0, 3, (N, D), generator=g).float()                # 4 tokens, 6 numbers each, small integers
U_qkv = [torch.randint(-1, 2, (D, 3 * Dh), generator=g).float() for _ in range(k)]   # one U_qkv per head: (6, 9)
U_msa = torch.randint(-1, 2, (k * Dh, D), generator=g).float()        # (6, 6)

log(f'== Eq. 5 to 8 on a tiny example: N = {N} tokens, D = {D}, k = {k} heads, D_h = D/k = {Dh} ==')
show('z (the input, one row per token)', z, names)
R['z'] = z.tolist()
heads_out = []
R['heads'] = []
for h in range(k):
    log(f'-- head {h + 1} --')
    show(f'U_qkv (head {h + 1}): D x 3 D_h', U_qkv[h])
    qkv = z @ U_qkv[h]                                                # Eq. 5: [q, k, v] = z U_qkv  -> (4, 9)
    q, kk, v = qkv[:, :Dh], qkv[:, Dh:2 * Dh], qkv[:, 2 * Dh:]
    show('[q, k, v] = z U_qkv  (the first 3 columns are q, the next 3 k, the last 3 v)', qkv, names)
    scores = q @ kk.T                                                 # (4, 4): one number per pair
    show('q k^T  (row i = query of token i, column j = key of token j)', scores, names)
    scaled = scores / math.sqrt(Dh)
    show(f'q k^T / sqrt(D_h) = q k^T / {math.sqrt(Dh):.3f}', scaled, names, 3)
    A = torch.softmax(scaled, -1)                                     # Eq. 6
    show('A = softmax(row by row)', A, names, 3)
    log(f'    row sums: {[round(s, 3) for s in A.sum(-1).tolist()]}')
    SA = A @ v                                                        # Eq. 7
    show('SA(z) = A v  (row i = weighted average of the value rows, weights = row i of A)', SA, names, 3)
    heads_out.append(SA)
    R['heads'].append(dict(U_qkv=U_qkv[h].tolist(), qkv=qkv.tolist(), scores=scores.tolist(), scaled=[[round(t, 4) for t in r] for r in scaled.tolist()],
                           A=[[round(t, 4) for t in r] for r in A.tolist()], SA=[[round(t, 4) for t in r] for r in SA.tolist()]))
log('-- Eq. 8 --')
concat = torch.cat(heads_out, -1)                                     # (4, 6) = [SA_1(z); SA_2(z)]
show('[SA_1(z); SA_2(z)]  (the two heads side by side: 4 x 6)', concat, names, 3)
show('U_msa: k D_h x D = 6 x 6', U_msa)
msa = concat @ U_msa                                                  # (4, 6)
show('MSA(z) = [SA_1(z); SA_2(z)] U_msa', msa, names, 3)
R['concat'] = [[round(t, 4) for t in r] for r in concat.tolist()]
R['U_msa'] = U_msa.tolist()
R['msa'] = [[round(t, 4) for t in r] for r in msa.tolist()]

# check one softmax entry by hand
i, j = 0, 0
e = torch.exp(torch.tensor([[round(t, 4) for t in r] for r in R['heads'][0]['scaled']][0]))
log('')
log(f'  check one entry by hand (head 1, row t0): exp of the scaled scores = {[round(t, 3) for t in e.tolist()]}, sum = {e.sum():.3f}')
log(f'  A[t0, t0] = {e[0]:.3f} / {e.sum():.3f} = {e[0] / e.sum():.3f}   (printed above: {R["heads"][0]["A"][0][0]:.3f})')
R['hand_check'] = dict(exp=[round(t, 4) for t in e.tolist()], sum=e.sum().item(), a00=(e[0] / e.sum()).item())

# permutation equivariance
log('')
log('== permutation equivariance: shuffle the input rows, the output rows shuffle the same way ==')


def msa_fn(z):
    outs = []
    for h in range(k):
        qkv = z @ U_qkv[h]
        q, kk, v = qkv[:, :Dh], qkv[:, Dh:2 * Dh], qkv[:, 2 * Dh:]
        outs.append(torch.softmax(q @ kk.T / math.sqrt(Dh), -1) @ v)
    return torch.cat(outs, -1) @ U_msa


perm = [2, 0, 3, 1]
z_perm = z[perm]
out_a = msa_fn(z)
out_b = msa_fn(z_perm)
log(f'  new order of the tokens: {[names[p] for p in perm]}')
show('MSA(z) in the original order', out_a, names, 3)
show('MSA(z shuffled)', out_b, [names[p] for p in perm], 3)
same = torch.allclose(out_b, out_a[perm], atol=1e-5)
log(f'  MSA(z shuffled) == MSA(z) with its rows shuffled the same way: {same}   (max |difference| {(out_b - out_a[perm]).abs().max():.1e})')
log('  so without position embeddings the model cannot tell where a patch came from: only the set of patches matters.')
# and with a position embedding added, the two runs differ
E_pos = torch.randint(0, 2, (N, D), generator=g).float()
out_c = msa_fn(z + E_pos)
out_d = msa_fn(z_perm + E_pos)
log(f'  with a position row added before attention (z + E_pos), the shuffled run no longer matches: max |difference| {(out_d - out_c[perm]).abs().max():.3f}')
R['perm'] = dict(perm=perm, out=[[round(t, 4) for t in r] for r in out_a.tolist()], out_perm=[[round(t, 4) for t in r] for r in out_b.tolist()],
                 same=same, maxdiff_with_pos=(out_d - out_c[perm]).abs().max().item(), E_pos=E_pos.tolist())
save('part2_attn', R)
