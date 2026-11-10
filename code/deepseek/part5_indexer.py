"""Part 5: a lightning indexer and top-k selection from scratch, on random tensors, against dense attention.

The setting copies DeepSeek-V3.2 in miniature:
  * main attention in "MQA mode" of MLA: H query heads share ONE cached latent per token (c_s), which is
    both the key and the value (the absorbed form of MLA);
  * a lightning indexer (eq. 1 of arXiv 2512.02556): I[t,s] = sum_j w[t,j] * ReLU(q_I[t,j] . k_I[s]),
    with few small heads and one shared indexer key per token;
  * top-k selection (eq. 2): attend only to the k entries with the highest index score;
  * the dense warm-up (eq. 3): train only the indexer, with a KL loss to the head-summed, L1-normalised
    main attention.
The ratio k / L is the real one: 64 of 4,096 tokens = 1.56%, like 2,048 of 131,072.
Method 1 is our own gather-based sparse attention; Method 2 checks it against PyTorch's
scaled_dot_product_attention with a boolean mask. CPU only, about a minute.
"""
import math, time
import torch
import torch.nn.functional as F
from common import Log, save

torch.manual_seed(0)
log = Log('part5_indexer')

D = 64            # hidden size of a token
H = 16            # main query heads (V3: 128)
DC = 32           # latent size, shared by all heads (V3: 512 + 64)
HI, DI = 16, 16   # indexer heads and indexer head size (V3.2: 64 and 128)
L = 4096          # tokens in the cache
T = 32            # query tokens per sequence
K = 64            # tokens each query keeps (V3.2: 2,048)
TEMP = 3.0        # sharpens the main attention so a few tokens matter more (real attention is peaky)
MIX = 0.5         # heads share one pattern plus their own part (real heads are correlated, not independent)

# ---- the frozen "main model": random projections, never trained ----
Wq = (torch.randn(D, DC) + MIX * torch.randn(H, D, DC)) / math.sqrt(D * (1 + MIX ** 2))   # per-head query projection
Wc = torch.randn(D, DC) / math.sqrt(D)             # the shared latent c_s (key and value, MQA mode)


def main_attention(h_q, h_kv):
    """Dense attention in MQA mode. h_q: (B,T,D), h_kv: (B,L,D) -> per-head probs (B,H,T,L), output (B,T,H,DC)."""
    q = torch.einsum('btd,hdc->bhtc', h_q, Wq)                     # (B,H,T,DC)
    c = h_kv @ Wc                                                  # (B,L,DC): one latent per token
    a = torch.softmax(TEMP * q @ c.transpose(1, 2).unsqueeze(1) / math.sqrt(DC), dim=-1)  # (B,H,T,L)
    return a, q, c


def target(a):
    """Eq. 3 target: sum the main attention over heads, then L1-normalise over the tokens."""
    p = a.sum(1)
    return p / p.sum(-1, keepdim=True)                             # (B,T,L)


class Indexer(torch.nn.Module):
    """I[t,s] = sum_j w[t,j] * ReLU(q_I[t,j] . k_I[s])."""
    def __init__(self):
        super().__init__()
        self.wq = torch.nn.Linear(D, HI * DI, bias=False)          # query token -> HI small queries
        self.wk = torch.nn.Linear(D, DI, bias=False)               # earlier token -> ONE small key
        self.ww = torch.nn.Linear(D, HI, bias=False)               # query token -> one weight per head

    def forward(self, h_q, h_kv, fp8=False):
        B = h_q.shape[0]
        q = self.wq(h_q).view(B, T, HI, DI)                        # (B,T,HI,DI)
        k = self.wk(h_kv)                                          # (B,L,DI)
        if fp8:                                                    # store and multiply in 8-bit floats
            q = q.to(torch.float8_e4m3fn).float()
            k = k.to(torch.float8_e4m3fn).float()
        w = self.ww(h_q)                                           # (B,T,HI)
        dots = torch.einsum('btjd,bsd->btjs', q, k)                # (B,T,HI,L)
        return torch.einsum('btj,btjs->bts', w, F.relu(dots))      # (B,T,L): the index scores


def sparse_attention(q, c, idx):
    """Method 1: gather the k selected latents and attend over them only. idx: (B,T,K)."""
    B = q.shape[0]
    sel = torch.gather(c.unsqueeze(1).expand(B, T, L, DC), 2, idx.unsqueeze(-1).expand(B, T, K, DC))  # (B,T,K,DC)
    s = TEMP * torch.einsum('bhtc,btkc->bhtk', q, sel) / math.sqrt(DC)
    a = torch.softmax(s, -1)
    return torch.einsum('bhtk,btkc->bthc', a, sel)                 # (B,T,H,DC)


def dense_out(a, c):
    return torch.einsum('bhtl,blc->bthc', a, c)


MU = torch.randn(D)   # hidden states share a mean direction, as real ones do (zero-mean noise alone is not realistic)


def batch(B=8):
    """Fresh random sequences: query tokens (B,T,D) and cached tokens (B,L,D)."""
    return torch.randn(B, T, D) + MU, torch.randn(B, L, D) + MU


def evaluate(name, idx, a, p, q, c, o_dense, k=K):
    o = sparse_attention(q, c, idx) if k == K else None
    mass = torch.gather(p, 2, idx).sum(-1).mean().item()           # share of the attention the k tokens hold
    best = p.topk(k, -1).indices
    rec = sum(len(set(x.tolist()) & set(y.tolist())) for x, y in zip(idx.flatten(0, 1), best.flatten(0, 1))) / (best.numel())
    err = ((o - o_dense).norm() / o_dense.norm()).item() if o is not None else float('nan')
    return dict(name=name, mass=mass, recall=rec, err=err)


ind = Indexer()
h_q, h_kv = batch(16)                                              # a fixed test batch, never trained on
with torch.no_grad():
    a, q, c = main_attention(h_q, h_kv)
    p = target(a)
    o_dense = dense_out(a, c)

log(f'Setup: {L:,} cached tokens, {T} queries per sequence, {H} heads sharing one {DC}-dim latent, '
    f'indexer {HI} heads x {DI} dims, keep k = {K} ({100 * K / L:.2f}%)')
top = p.topk(K, -1).values.sum(-1).mean().item()
log(f'Dense attention: the best {K} tokens hold {100 * top:.1f}% of the head-summed attention on average')
log()

res = []
with torch.no_grad():
    res.append(evaluate('random tokens', torch.stack([torch.randperm(L)[:K] for _ in range(16 * T)]).view(16, T, K), a, p, q, c, o_dense))
    res.append(evaluate('untrained indexer', ind(h_q, h_kv).topk(K, -1).indices, a, p, q, c, o_dense))

# ---- dense warm-up: freeze the main model, train only the indexer with KL(p || softmax(I)) ----
opt = torch.optim.Adam(ind.parameters(), lr=1e-3)
t0 = time.time()
for step in range(1, 401):
    hq, hkv = batch(8)
    with torch.no_grad():
        pt = target(main_attention(hq, hkv)[0])
    logq = torch.log_softmax(ind(hq, hkv), -1)
    loss = F.kl_div(logq, pt, reduction='batchmean')              # = sum_t KL(p_t || softmax(I_t)) / B
    opt.zero_grad(); loss.backward(); opt.step()
    if step in (1, 25, 100, 200, 400):
        log(f'warm-up step {step:>4}: KL loss {loss.item() / T:.3f} per query')
log(f'warm-up took {time.time() - t0:.0f} s on CPU')
log()

with torch.no_grad():
    I = ind(h_q, h_kv)
    res.append(evaluate('trained indexer', I.topk(K, -1).indices, a, p, q, c, o_dense))
    res.append(evaluate('oracle (true top-k)', p.topk(K, -1).indices, a, p, q, c, o_dense))
    log(f'{"selector":22s} {"attention kept":>15} {"recall of true top-k":>21} {"output error":>13}')
    for r in res:
        log(f'{r["name"]:22s} {100 * r["mass"]:>14.1f}% {100 * r["recall"]:>20.1f}% {100 * r["err"]:>12.1f}%')
    log()

    # k sweep for the figure
    sweep = []
    for k in [16, 32, 64, 128, 256, 512, 1024]:
        row = {'k': k}
        for name, sc in [('trained', I), ('oracle', p)]:
            row[name] = torch.gather(p, 2, sc.topk(k, -1).indices).sum(-1).mean().item()
        row['random'] = k / L
        sweep.append(row)
        log(f'k = {k:>5} ({100 * k / L:5.2f}% of tokens): attention kept, trained indexer {100 * row["trained"]:5.1f}%, '
            f'oracle {100 * row["oracle"]:5.1f}%, random {100 * row["random"]:5.1f}%')
    log()

    # FP8: does storing the indexer's q and k in 8 bits change which tokens win?
    I8 = ind(h_q, h_kv, fp8=True)
    a32, a8 = I.topk(K, -1).indices, I8.topk(K, -1).indices
    same = sum(len(set(x.tolist()) & set(y.tolist())) for x, y in zip(a32.flatten(0, 1), a8.flatten(0, 1))) / a32.numel()
    log(f'FP8 indexer (float8_e4m3fn q and k): {100 * same:.1f}% of the selected tokens are the same as in FP32')

    # Method 2: the library. A boolean mask that allows only the selected tokens, in PyTorch's SDPA.
    idx = I.topk(K, -1).indices
    allow = torch.zeros(16, T, L, dtype=torch.bool).scatter_(2, idx, True)          # (B,T,L)
    o_lib = F.scaled_dot_product_attention(q, c.unsqueeze(1).expand(-1, H, -1, -1), c.unsqueeze(1).expand(-1, H, -1, -1),
                                           attn_mask=allow.unsqueeze(1), scale=TEMP / math.sqrt(DC))  # (B,H,T,DC)
    o_ours = sparse_attention(q, c, idx).permute(0, 2, 1, 3)
    log(f'Method 1 (gather {K} latents) vs Method 2 (SDPA with a mask over all {L:,}): '
        f'max difference {(o_ours - o_lib).abs().max().item():.1e}')
    log()

# ---- multiply-adds per query, counted ----
dense_mac = H * (DC + DC) * L                 # scores over DC dims + weighted sum over DC dims, every token
idx_mac = HI * DI * L                         # the indexer touches every token, but small
main_mac = H * (DC + DC) * K                  # the main attention touches only k tokens
log('Multiply-adds per query token:')
log(f'  dense attention:          {H} x ({DC}+{DC}) x {L:,} = {dense_mac:,}')
log(f'  DSA indexer:              {HI} x {DI} x {L:,}      = {idx_mac:,}')
log(f'  DSA main (top-{K}):        {H} x ({DC}+{DC}) x {K}     = {main_mac:,}')
log(f'  DSA total:                {idx_mac + main_mac:,}  ({dense_mac / (idx_mac + main_mac):.1f}x fewer than dense)')

save('part5_indexer', dict(results=res, sweep=sweep, fp8_same=same, top_mass=top,
                           macs=dict(dense=dense_mac, idx=idx_mac, main=main_mac)))
