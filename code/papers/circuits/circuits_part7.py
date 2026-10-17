"""Part 7: what the paper left out (MLP layers), and the later work, on real models and real toy models.

Uses TransformerLens 3.x (HookedTransformer) and torch on the CPU. Prints and saves:
  1. how many parameters sit in the MLP layers of attn-only-1l, attn-only-2l, gelu-1l, gelu-2l and gpt2 (the paper: "2/3rds");
  2. the in-context learning score of the 2022 follow-up paper (loss at token 500 minus loss at token 50) on Wikipedia text
     (wikitext-103 test), for one-layer and two-layer models with and without MLPs, and gpt2;
  3. one MLP layer read with the paper's equations: W_U W_O for single neurons of gelu-1l (what a neuron writes),
     W_I W_E (which tokens excite it directly), and the text where it fires most;
  4. the toy model of superposition (5 features, 2 dimensions, x' = ReLU(W^T W x + b)) trained at four sparsities,
     and a linear model for comparison;
  5. a bigger toy model (40 features in 10 dimensions) and a sparse autoencoder trained on its hidden vectors:
     do single neurons match features, and does the autoencoder find the 40 feature directions?
Runs on the CPU in a few minutes after the models are downloaded."""
import math, re
import pandas as pd
import torch
from transformer_lens import HookedTransformer
from common import Log, save

torch.manual_seed(0)
log = Log('part7')
R = {}
WIKI = ('/Users/admin/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/'
        'b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1/')   # Salesforce/wikitext on Hugging Face


def load(name):
    return HookedTransformer.from_pretrained(name, device='cpu').eval()


def articles(split):
    """wikitext-103 lines, joined back into whole Wikipedia articles (a new article starts at ' = Title = ')."""
    lines = pd.read_parquet(WIKI + f'{split}-00000-of-00001.parquet')['text'].tolist()
    docs, cur = [], []
    for l in lines:
        if re.match(r'^ = [^=].* = \n$', l) and cur:
            docs.append(''.join(cur))
            cur = []
        cur.append(l)
    docs.append(''.join(cur))
    return docs


# ---------------------------------------------------------------- 1. where the parameters are
log('== 1. HOW MANY PARAMETERS ARE IN THE MLP LAYERS? (embedding and unembedding not counted)')
names = ['attn-only-1l', 'attn-only-2l', 'gelu-1l', 'gelu-2l', 'gpt2']
models = {}
R['params'] = {}
with torch.no_grad():
    for n in names:
        m = models[n] = load(n)
        attn = sum(p.numel() for k, p in m.named_parameters() if '.attn.' in k)
        mlp = sum(p.numel() for k, p in m.named_parameters() if '.mlp.' in k)
        emb = sum(p.numel() for k, p in m.named_parameters() if k.startswith(('embed', 'pos_embed', 'unembed')))
        frac = mlp / (attn + mlp)
        R['params'][n] = dict(layers=m.cfg.n_layers, d_model=m.cfg.d_model, d_mlp=m.cfg.d_mlp if mlp else 0,
                              attn=attn, mlp=mlp, emb=emb, mlp_frac=frac)
        log(f'  {n:13s} layers {m.cfg.n_layers:2d}  d_model {m.cfg.d_model}  attention {attn:>11,}  MLP {mlp:>11,}'
            f'  MLP share {frac:.3f}' + (f'  (d_mlp = {m.cfg.d_mlp} = {m.cfg.d_mlp // m.cfg.d_model} x d_model)' if mlp else ''))
g = models['gpt2'].cfg
log(f'  gpt2, one layer by hand: attention 4 x {g.d_model} x {g.d_model} + biases = {4 * g.d_model ** 2 + 4 * g.d_model:,}; '
    f'MLP 2 x {g.d_model} x {g.d_mlp} + biases = {2 * g.d_model * g.d_mlp + g.d_mlp + g.d_model:,}')

# ---------------------------------------------------------------- 2. in-context learning score
log('\n== 2. IN-CONTEXT LEARNING SCORE: LOSS AT TOKEN 500 MINUS LOSS AT TOKEN 50 (wikitext-103 test, 512-token windows)')
docs = articles('test')
W, NWIN = 512, 300
IDX = [1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500]
R['icl'] = {}
with torch.no_grad():
    for n in names:
        m = models[n]
        wins = []
        for d in docs:
            t = m.tokenizer(d)['input_ids']
            for s in range(0, len(t) - W + 1, W):
                wins.append([m.tokenizer.bos_token_id] + t[s:s + W - 1])
        wins = torch.tensor(wins[:NWIN])
        losses = []
        for i in range(0, len(wins), 10):
            losses.append(m(wins[i:i + 10], return_type='loss', loss_per_token=True))
        L = torch.cat(losses)                      # [windows, 511]: L[:, k-1] is the loss of predicting token k
        curve = {k: float(L[:, k - 1].mean()) for k in IDX}
        # a smoother version: average over tokens 41-60 and 491-510 (the paper uses the single tokens 50 and 500)
        early, late = float(L[:, 40:60].mean()), float(L[:, 490:510].mean())
        score = curve[500] - curve[50]
        se = float((L[:, 499] - L[:, 49]).std() / math.sqrt(len(L)))
        R['icl'][n] = dict(windows=len(L), curve=curve, score=score, se=se, early=early, late=late, smooth=late - early)
        log(f'  {n:13s} {len(L)} windows  loss at token 50: {curve[50]:.3f}  at token 500: {curve[500]:.3f}  '
            f'score {score:+.3f} (+/- {se:.3f})  [tokens 41-60 vs 491-510: {late - early:+.3f}]')
    log('  loss by token index (mean over windows):')
    log('  ' + 'index'.ljust(13) + ''.join(f'{k:>7d}' for k in IDX))
    for n in names:
        log(f'  {n:13s}' + ''.join(f'{R["icl"][n]["curve"][k]:7.2f}' for k in IDX))

# ---------------------------------------------------------------- 3. one MLP layer, read with the paper's equations
log('\n== 3. MLP NEURONS OF gelu-1l READ FROM THE WEIGHTS (TransformerLens folds the layer norm weights into W_in)')
m = models['gelu-1l']
c = m.cfg
log(f'  gelu-1l: d_model {c.d_model}, d_mlp {c.d_mlp} neurons, vocabulary {c.d_vocab:,}; '
    f'W_I is ({c.d_mlp}, {c.d_model}) and W_O is ({c.d_model}, {c.d_mlp}) in the paper\'s shapes')
toks = []
for d in docs:
    t = m.tokenizer(d)['input_ids']
    for s in range(0, len(t) - 255, 256):
        toks.append([m.tokenizer.bos_token_id] + t[s:s + 255])
X = torch.tensor(toks[:400])
with torch.no_grad():
    acts = []
    for i in range(0, len(X), 20):
        _, cache = m.run_with_cache(X[i:i + 20], names_filter='blocks.0.mlp.hook_post')
        acts.append(cache['blocks.0.mlp.hook_post'][:, 1:])
    A = torch.cat(acts)                                       # [windows, 255, 2048], BOS dropped
    Xs = X[:, 1:]
    flat = A.reshape(-1, c.d_mlp)
    log(f'  activations on {flat.shape[0]:,} tokens of Wikipedia text')
    logit_dir = m.W_out[0] @ m.W_U                            # [2048, vocab]: row n is W_U W_O[:, n], the neuron's direct effect
    direct_in = m.W_E @ m.W_in[0]                             # [vocab, 2048]: column n is W_I[n] W_E (direct path, no LN scale)
    R['neurons'] = {}
    for n in [847, 1996]:
        up = logit_dir[n].topk(8)
        down = (-logit_dir[n]).topk(5)
        din = direct_in[:, n].topk(8)
        v, idx = flat[:, n].topk(12)
        ctx = []
        for j in idx.tolist():
            dd, p = divmod(j, Xs.shape[1])
            ctx.append((m.to_string(Xs[dd, max(0, p - 5):p]), m.to_string(Xs[dd, p:p + 1])))
        on = float((flat[:, n] > 0.1).float().mean())
        R['neurons'][n] = dict(up=[(m.to_string(int(i)), round(float(x), 2)) for x, i in zip(up.values, up.indices)],
                               down=[(m.to_string(int(i)), round(-float(x), 2)) for x, i in zip(down.values, down.indices)],
                               direct_in=[(m.to_string(int(i)), round(float(x), 2)) for x, i in zip(din.values, din.indices)],
                               top=[(a, b, round(float(x), 2)) for (a, b), x in zip(ctx, v)], frac_on=on,
                               max=float(v[0]))
        log(f'  neuron {n}: fires (> 0.1) on {on * 100:.1f}% of tokens; largest activation {float(v[0]):.2f}')
        log(f'    W_U W_O (what it writes): raises ' + ', '.join(f'{t!r} {x:+.2f}' for t, x in R['neurons'][n]['up']))
        log(f'                              lowers ' + ', '.join(f'{t!r} {x:+.2f}' for t, x in R['neurons'][n]['down']))
        log(f'    W_I W_E (tokens that excite it directly): ' + ', '.join(f'{t!r} {x:.2f}' for t, x in R['neurons'][n]['direct_in']))
        log(f'    where it fires most (text before | the token):')
        for a, b, x in R['neurons'][n]['top'][:10]:
            log(f'      {x:5.2f}  {a!r} | {b!r}')

# ---------------------------------------------------------------- 4. toy model of superposition
log('\n== 4. TOY MODEL OF SUPERPOSITION: 5 FEATURES, 2 DIMENSIONS, x\' = ReLU(W^T W x + b), importance I_i = 0.9^i')
N, M_ = 5, 2
I = torch.tensor([0.9 ** i for i in range(N)])


def batch(S, size):
    x = torch.rand(size, N)
    return x * (torch.rand(size, N) >= S)                     # each feature is 0 with probability S, else uniform in [0, 1]


def train(S, relu=True, seed=0, steps=6000):
    g = torch.Generator().manual_seed(seed)
    Wt = torch.nn.Parameter(torch.randn(M_, N, generator=g) * 0.5)
    b = torch.nn.Parameter(torch.zeros(N))
    opt = torch.optim.Adam([Wt, b], lr=1e-2)
    torch.manual_seed(seed)
    for step in range(steps):
        x = batch(S, 1024)
        out = x @ Wt.T @ Wt + b
        if relu:
            out = torch.relu(out)
        loss = ((I * (x - out) ** 2).sum(-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        x = batch(S, 100000)
        out = x @ Wt.T @ Wt + b
        out = torch.relu(out) if relu else out
        loss = float(((I * (x - out) ** 2).sum(-1)).mean())
    return Wt.detach(), b.detach(), loss


def describe(Wt, b):
    norms = Wt.norm(dim=0)
    ang = [round(math.degrees(math.atan2(float(Wt[1, i]), float(Wt[0, i]))) % 360, 1) for i in range(N)]
    G = Wt.T @ Wt
    return dict(norms=[round(float(v), 3) for v in norms], angles=ang, b=[round(float(v), 3) for v in b],
                W=[[round(float(v), 3) for v in row] for row in Wt], WtW=[[round(float(v), 2) for v in row] for row in G],
                represented=int((norms > 0.5).sum()))


R['toy'] = {}
for S in [0.0, 0.7, 0.9, 0.97]:
    best = min((train(S, seed=s) for s in range(3)), key=lambda t: t[2])   # best of 3 random starts
    d = describe(best[0], best[1]); d['loss'] = best[2]
    R['toy'][f'relu S={S}'] = d
    log(f'  ReLU model, S = {S:<4} (1-S = {1 - S:.2f}): features represented {d["represented"]}/5; '
        f'|W_i| = {d["norms"]}; angles (deg) = {d["angles"]}; b = {d["b"]}; loss {d["loss"]:.4f}')
Wl, bl, ll = min((train(0.9, relu=False, seed=s) for s in range(3)), key=lambda t: t[2])
d = describe(Wl, bl); d['loss'] = ll
R['toy']['linear S=0.9'] = d
log(f'  linear model, S = 0.9: features represented {d["represented"]}/5; |W_i| = {d["norms"]}; angles = {d["angles"]}; loss {ll:.4f}')
d = R['toy']['relu S=0.9']
log(f'  ReLU, S = 0.9: W^T W (5 x 5) = {d["WtW"]}')
log(f'  ReLU, S = 0.9: the two hidden numbers, as rows of W (how much each feature moves each one):')
for k in range(M_):
    log(f'    hidden {k + 1}: {d["W"][k]}')

# ---------------------------------------------------------------- 5. a sparse autoencoder on a bigger toy model
log('\n== 5. A SPARSE AUTOENCODER ON A BIGGER TOY MODEL: 40 FEATURES IN 10 DIMENSIONS (S = 0.95, all importances 1)')
N2, M2, S2 = 40, 10, 0.95


def batch2(size):
    x = torch.rand(size, N2)
    return x * (torch.rand(size, N2) >= S2)


torch.manual_seed(0)
Wt = torch.nn.Parameter(torch.randn(M2, N2) * 0.3)
b = torch.nn.Parameter(torch.zeros(N2))
opt = torch.optim.Adam([Wt, b], lr=3e-3)
for step in range(10000):
    x = batch2(1024)
    loss = ((x - torch.relu(x @ Wt.T @ Wt + b)) ** 2).sum(-1).mean()
    opt.zero_grad(); loss.backward(); opt.step()
Wt = Wt.detach()
norms = Wt.norm(dim=0)
true_dirs = Wt / norms
neuron_cos = (true_dirs.T @ torch.eye(M2)).abs().max(1).values      # best match of each feature among the 10 neurons
per_neuron = (Wt.abs() > 0.3).sum(1)                                 # how many features push each neuron by more than 0.3
R['toy2'] = dict(n=N2, m=M2, S=S2, loss=float(loss), represented=int((norms > 0.5).sum()),
                 neuron_cos_median=float(neuron_cos.median()), neuron_cos_max=float(neuron_cos.max()),
                 features_per_neuron=per_neuron.tolist())
log(f'  toy model: {R["toy2"]["represented"]} of {N2} features represented (|W_i| > 0.5) in {M2} dimensions; loss {float(loss):.4f}')
log(f'  features that move each of the 10 neurons by more than 0.3: {per_neuron.tolist()}')
log(f'  best |cosine| between a feature direction and any single neuron: median {float(neuron_cos.median()):.3f}, '
    f'largest {float(neuron_cos.max()):.3f}  (1 would mean "this neuron is this feature")')
F, LAM = 40, 0.3
torch.manual_seed(1)
We = torch.nn.Parameter(torch.randn(F, M2) * 0.3)
be = torch.nn.Parameter(torch.zeros(F))
Wd = torch.nn.Parameter(torch.randn(M2, F) * 0.3)
bd = torch.nn.Parameter(torch.zeros(M2))
opt = torch.optim.Adam([We, be, Wd, bd], lr=1e-3)
for step in range(15000):
    h = batch2(4096) @ Wt.T                                          # the toy model's hidden vectors h = W x
    f = torch.relu((h - bd) @ We.T + be)                             # f = ReLU(W_enc (h - b_dec) + b_enc)
    hh = f @ Wd.T + bd                                               # h_hat = W_dec f + b_dec
    l = ((h - hh) ** 2).sum(-1).mean() + LAM * (f * Wd.norm(dim=0)).sum(-1).mean()
    opt.zero_grad(); l.backward(); opt.step()
with torch.no_grad():
    dirs = Wd / Wd.norm(dim=0, keepdim=True)
    cos = true_dirs.T @ dirs                                         # [40 true, 40 learned]
    best = cos.max(1).values
    xs = batch2(100000)
    h = xs @ Wt.T
    f = torch.relu((h - bd) @ We.T + be)
    hh = f @ Wd.T + bd
    ev = 1 - float(((h - hh) ** 2).sum(-1).mean() / ((h - h.mean(0)) ** 2).sum(-1).mean())
    active, true_active = float((f > 1e-6).float().sum(-1).mean()), float((xs > 0).float().sum(-1).mean())
R['sae'] = dict(F=F, lam=LAM, found99=int((best > 0.99).sum()), found9=int((best > 0.9).sum()),
                median=float(best.median()), minimum=float(best.min()), explained=ev, active=active, true_active=true_active)
log(f'  sparse autoencoder: {F} features, L1 weight {LAM}, 15,000 steps of 4,096 hidden vectors')
log(f'  true features with a learned direction at cosine > 0.99: {R["sae"]["found99"]} of {N2}; > 0.9: {R["sae"]["found9"]} of {N2}; '
    f'median best cosine {R["sae"]["median"]:.4f}; worst {R["sae"]["minimum"]:.3f}')
log(f'  the reconstruction keeps {ev * 100:.1f}% of the variance of h; on average {true_active:.2f} true features are on per input, '
    f'and the autoencoder turns on {active:.2f} of its features')
save('part7', R)

# ---------------------------------------------------------------- 6. the "Aunt Sally" prompt of the 2025 circuit-tracing paper
log('\n== 6. THE "AUNT SALLY" PROMPT ON ONE-LAYER AND TWO-LAYER ATTENTION-ONLY MODELS')
prompt = 'I always loved visiting Aunt Sally. Whenever I was feeling sad, Aunt'
R['sally'] = {}
with torch.no_grad():
    for n in ['attn-only-1l', 'attn-only-2l']:
        m = models[n]
        toks = m.to_tokens(prompt)
        st = m.to_str_tokens(prompt)
        logits, cache = m.run_with_cache(toks)
        p = logits[0, -1].softmax(-1)
        top = p.topk(5)
        sid = m.to_single_token(' Sally')
        rank = int((p > p[sid]).sum()) + 1
        spos = st.index(' Sally')
        L = m.cfg.n_layers - 1
        att = cache[f'blocks.{L}.attn.hook_pattern'][0, :, -1, spos]          # every head of the last layer: last token -> " Sally"
        h = int(att.argmax())
        R['sally'][n] = dict(top=[(m.to_string(int(i)), round(float(v), 3)) for v, i in zip(top.values, top.indices)],
                             p_sally=float(p[sid]), rank=rank, head=f'{L}.{h}', att=float(att[h]))
        log(f'  {n}: tokens {len(st)}; next-token top 5: ' + ', '.join(f'{t!r} {v:.3f}' for t, v in R['sally'][n]['top']))
        log(f'    P(" Sally") = {float(p[sid]):.3f} (rank {rank}); strongest head of layer {L} on " Sally": {L}.{h} with weight {float(att[h]):.2f}')
save('part7', R)
