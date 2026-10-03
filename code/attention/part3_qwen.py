"""Part 3, real model: Qwen2.5-0.5B (trained with full attention) run with
  (a) sliding-window masks, with and without 4 'sink' tokens, and
  (b) a DeepSeek-style lightning indexer per layer, trained with the dense warm-up KL loss, then used for top-k sparse attention.
Writes results/part3_qwen.json."""
import json, math, os, re, time, urllib.request
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
T = 2048                    # tokens per chunk
EVAL_FROM = 1024            # score only the second half, where windows actually cut something off
OUT = {'T': T, 'scored_positions': [EVAL_FROM, T - 1]}

tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B', attn_implementation='eager', dtype=torch.float32).to(dev).eval()
cfg = model.config
NL = cfg.num_hidden_layers


def book(num):
    """A public-domain book from Project Gutenberg, without its licence header and footer."""
    path = os.path.expanduser(f'~/.cache/gutenberg_{num}.txt')
    if not os.path.exists(path):
        urllib.request.urlretrieve(f'https://www.gutenberg.org/cache/epub/{num}/pg{num}.txt', path)
    text = open(path, encoding='utf-8').read()
    text = text.split('*** START OF', 1)[1].split('\n', 1)[1].split('*** END OF', 1)[0]
    return re.sub(r'\n{3,}', '\n\n', text)


def chunks(num, n, skip=20000):
    ids = tok(book(num), return_tensors='pt').input_ids[0][skip:]
    return [ids[i * T:(i + 1) * T][None].to(dev) for i in range(n)]


eval_chunks = chunks(1342, 4)      # Pride and Prejudice: measuring
train_chunks = chunks(1661, 6)     # The Adventures of Sherlock Holmes: training the indexers (different text)

i_ = torch.arange(T, device=dev)[:, None]
j_ = torch.arange(T, device=dev)[None, :]
CAUSAL = j_ <= i_


def additive(allowed):
    """Boolean (T, T) -> the additive float mask eager attention expects, shape (1, 1, T, T)."""
    return torch.zeros(T, T, device=dev).masked_fill(~allowed, torch.finfo(torch.float32).min)[None, None]


def window(W, sinks=0):
    allowed = CAUSAL & (i_ - j_ < W)
    if sinks:
        allowed |= CAUSAL & (j_ < sinks)
    return allowed


@torch.no_grad()
def nll(ids, mask=None):
    logits = model(ids, attention_mask=mask).logits[0, :-1].float()
    losses = F.cross_entropy(logits, ids[0, 1:], reduction='none')
    return losses[EVAL_FROM - 1:].mean().item()


def ppl(mask_fn=None):
    return math.exp(sum(nll(c, mask_fn() if mask_fn else None) for c in eval_chunks) / len(eval_chunks))


# sanity: our own full causal 4D mask must give exactly the model's normal output
with torch.no_grad():
    a = model(eval_chunks[0]).logits
    b = model(eval_chunks[0], attention_mask=additive(CAUSAL)).logits
OUT['mask_sanity_max_diff'] = (a - b).abs().max().item()
print('custom full mask vs default: max |diff| = %.1e' % OUT['mask_sanity_max_diff'])

t0 = time.time()
full_ppl = ppl()
OUT['full_ppl'] = full_ppl
print('full attention: perplexity %.2f' % full_ppl)
win = {}
for W in (64, 256, 1024):
    win[W] = {'window_only': ppl(lambda: additive(window(W))),
              'window_plus_4_sinks': ppl(lambda: additive(window(W - 4, sinks=4)))}   # same budget: W tokens
    print('budget %4d tokens: window only %8.2f   4 sinks + window %d: %6.2f' % (W, win[W]['window_only'], W - 4, win[W]['window_plus_4_sinks']))
OUT['window'] = win

# ---------------------------------------------------------------- DeepSeek-style lightning indexer
HI, DI = 4, 32          # indexer heads and their size: tiny next to the model's 14 heads of 64


def rope(x, base=10000.0):
    d = x.shape[-1]
    inv = base ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    ang = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)[:, None] * inv
    cos, sin = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)
    return x * cos + torch.cat([-x[..., d // 2:], x[..., : d // 2]], -1) * sin


class LightningIndexer(torch.nn.Module):
    """I[t, s] = sum_j w[t, j] * ReLU(q[t, j] . k[s])   (DeepSeek-V3.2, eq. 1). One shared key per token."""
    def __init__(self, d_model):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d_model)                        # Qwen's hidden states have a few huge values; tame them
        self.q = torch.nn.Linear(d_model, HI * DI, bias=False)
        self.k = torch.nn.Linear(d_model, DI, bias=False)
        self.w = torch.nn.Linear(d_model, HI, bias=False)

    def forward(self, h):                                              # h: (T, d_model)
        n, h = h.shape[0], self.norm(h)
        q = rope(self.q(h).view(n, HI, DI).transpose(0, 1))            # (HI, T, DI)
        k = rope(self.k(h))                                            # (T, DI)
        return torch.einsum('jt,jts->ts', self.w(h).T, F.relu(q @ k.T))   # (T, T)


captured = {}
def grab_input(layer):
    def hook(mod, args, kwargs):
        captured.setdefault(layer, {})['h'] = kwargs['hidden_states'][0].detach()
    return hook
def grab_weights(layer):
    def hook(mod, args, output):
        A = output[1][0]                                               # (heads, T, T)
        captured[layer]['p'] = (A.sum(0) / A.sum(0).sum(-1, keepdim=True)).detach()   # sum over heads, L1-normalise
    return hook


@torch.no_grad()
def dense_targets(ids):
    """One dense forward pass: every layer's attention input h and its head-summed, normalised attention p."""
    captured.clear()
    hs = [m.self_attn.register_forward_pre_hook(grab_input(l), with_kwargs=True) for l, m in enumerate(model.model.layers)]
    hs += [m.self_attn.register_forward_hook(grab_weights(l)) for l, m in enumerate(model.model.layers)]
    model(ids)
    for hk in hs:
        hk.remove()
    return {l: (v['h'].cpu(), v['p'].half().cpu()) for l, v in captured.items()}


def kl_loss(ix, h, p):
    I = ix(h).masked_fill(~CAUSAL, float('-inf'))
    logq = torch.log_softmax(I, -1)
    return -(p * logq.masked_fill(~CAUSAL, 0)).sum(-1).mean()          # KL(p || softmax(I)) up to a constant


def coverage(sel, p):
    """Share of the model's own attention that lands on the selected tokens (rows from EVAL_FROM on)."""
    return (p * sel).sum(-1)[EVAL_FROM:].mean().item()


def topk_select(I, k):
    I = I.masked_fill(~CAUSAL, float('-inf'))
    return torch.zeros(T, T, dtype=torch.bool, device=dev).scatter(1, I.topk(k, -1).indices, True) & CAUSAL


print('collecting dense attention targets ...')
train = [dense_targets(c) for c in train_chunks]
evals = [dense_targets(c) for c in eval_chunks]

indexers, curves = [], {}
STEPS, LR = 400, 1e-3                                                  # warm-up learning rate 1e-3, as in the paper
for l in range(NL):
    ix = LightningIndexer(cfg.hidden_size).to(dev)
    if l == 0:
        before_state = {k: v.clone() for k, v in ix.state_dict().items()}
    opt = torch.optim.Adam(ix.parameters(), lr=LR)
    hist = []
    for s in range(STEPS):
        h, p = train[s % len(train)][l]
        loss = kl_loss(ix, h.to(dev), p.to(dev).float())
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(ix.parameters(), 1.0); opt.step()
        if s % 50 == 0 or s == STEPS - 1:
            hist.append([s, loss.item()])
    curves[l] = hist
    indexers.append(ix.eval())
    print('layer %2d: warm-up loss %.3f -> %.3f' % (l, hist[0][1], hist[-1][1]))
OUT['indexer'] = {'heads': HI, 'dim': DI, 'steps': STEPS, 'lr': LR, 'loss_curves': curves,
                  'params_per_layer': sum(p.numel() for p in indexers[0].parameters()),
                  'attn_params_per_layer': sum(p.numel() for p in model.model.layers[0].self_attn.parameters())}

# how much of the real attention does each selection rule capture? (held-out text, dense hidden states)
cov = {}
untrained = LightningIndexer(cfg.hidden_size).to(dev); untrained.load_state_dict(before_state)
for k in (64, 256):
    r = {'window': [], 'sinks_window': [], 'indexer_untrained_layer0': [], 'indexer': []}
    with torch.no_grad():
        for ev in evals:
            for l in range(NL):
                h, p = ev[l][0].to(dev), ev[l][1].to(dev).float()
                r['window'].append(coverage(window(k), p))
                r['sinks_window'].append(coverage(window(k - 4, 4), p))
                r['indexer'].append(coverage(topk_select(indexers[l](h), k), p))
                if l == 0:
                    r['indexer_untrained_layer0'].append(coverage(topk_select(untrained(h), k), p))
    cov[k] = {name: sum(v) / len(v) for name, v in r.items()}
    print('budget %d: attention captured  window %.3f  sinks+window %.3f  indexer %.3f  (untrained, layer 0: %.3f)'
          % (k, cov[k]['window'], cov[k]['sinks_window'], cov[k]['indexer'], cov[k]['indexer_untrained_layer0']))
OUT['coverage'] = cov

# per-layer coverage at k = 64, for a figure
OUT['coverage_by_layer_64'] = {name: [] for name in ('sinks_window', 'indexer')}
with torch.no_grad():
    for l in range(NL):
        sw, ixc = [], []
        for ev in evals:
            h, p = ev[l][0].to(dev), ev[l][1].to(dev).float()
            sw.append(coverage(window(60, 4), p)); ixc.append(coverage(topk_select(indexers[l](h), 64), p))
        OUT['coverage_by_layer_64']['sinks_window'].append(sum(sw) / len(sw))
        OUT['coverage_by_layer_64']['indexer'].append(sum(ixc) / len(ixc))


# the real test: run the model with top-k sparse attention chosen by the indexers, in every layer
def sparse_hook(layer, k):
    def hook(mod, args, kwargs):
        h = kwargs['hidden_states'][0]
        kwargs['attention_mask'] = additive(topk_select(indexers[layer](h), k))
        return args, kwargs
    return hook


dsa = {}
for k in (64, 256):
    hs = [m.self_attn.register_forward_pre_hook(sparse_hook(l, k), with_kwargs=True) for l, m in enumerate(model.model.layers)]
    dsa[k] = ppl()
    for hk in hs:
        hk.remove()
    print('budget %4d tokens: indexer top-k perplexity %.2f' % (k, dsa[k]))
OUT['dsa_ppl'] = dsa

# one picture: which tokens does each rule pick for the last query in a middle layer?
l, ev = NL // 2, evals[0]
with torch.no_grad():
    h, p = ev[l][0].to(dev), ev[l][1].to(dev).float()
    sel = topk_select(indexers[l](h), 64)[-1].nonzero().flatten().tolist()
OUT['example'] = {'layer': l, 'query_pos': T - 1, 'indexer_selected': sel,
                  'attention_last_row_top': torch.topk(p[-1], 64).indices.tolist()}
OUT['seconds'] = time.time() - t0
json.dump(OUT, open('results/part3_qwen.json', 'w'), indent=1)
print('done in %.0f s' % OUT['seconds'])
