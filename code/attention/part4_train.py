"""Part 4: train five small models that differ ONLY in their token-mixing layer, on the same data, with the same budget.
  softmax   : ordinary causal softmax attention (with RoPE)
  gated     : softmax attention + a head-specific, elementwise sigmoid gate on its output (Qiu et al., 2025)
  linear    : linear attention, phi(x) = elu(x) + 1 (Katharopoulos et al., 2020)
  gdn       : Gated DeltaNet (Yang et al., 2025), computed with the reference chunked kernel shipped with Qwen3-Next
  hybrid    : 3 Gated DeltaNet layers + 1 gated softmax layer (the Qwen3-Next pattern)
Task 1: next-byte prediction on public-domain books.  Task 2: associative recall (store n key-value facts, then answer n lookups), a one-hop version of MQAR (Arora et al., 2023).
Writes results/part4_train.json."""
import json, math, os, re, sys, time, urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_next.modeling_qwen3_next import torch_chunk_gated_delta_rule

dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
D, H, LAYERS = 256, 4, 4
HD = D // H
PATTERNS = {'softmax': ['softmax'] * 4, 'gated': ['gated'] * 4, 'linear': ['linear'] * 4,
            'gdn': ['gdn'] * 4, 'hybrid': ['gdn', 'gdn', 'gdn', 'gated']}


def rope(x):                                                   # x: (B, H, T, d)
    d, T = x.shape[-1], x.shape[-2]
    inv = 10000 ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    ang = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * inv
    cos, sin = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)
    return x * cos + torch.cat([-x[..., d // 2:], x[..., : d // 2]], -1) * sin


class SoftmaxAttention(nn.Module):
    def __init__(self, gated):
        super().__init__()
        self.qkv, self.o = nn.Linear(D, 3 * D, bias=False), nn.Linear(D, D, bias=False)
        self.gate = nn.Linear(D, D, bias=False) if gated else None     # one gate value per head per channel

    def forward(self, x, return_attn=False):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, H, HD).permute(2, 0, 3, 1, 4)
        q, k = rope(q), rope(k)
        if return_attn:
            A = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(HD)).masked_fill(
                torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1), float('-inf')), -1)
            y = A @ v
        else:
            A, y = None, F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, D)
        if self.gate is not None:
            y = y * torch.sigmoid(self.gate(x))                  # Y' = Y * sigmoid(X W_theta), right after SDPA
        return self.o(y), A


class LinearAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv, self.o = nn.Linear(D, 3 * D, bias=False), nn.Linear(D, D, bias=False)

    def forward(self, x, return_attn=False):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, H, HD).permute(2, 0, 3, 1, 4)
        q, k = F.elu(q) + 1, F.elu(k) + 1
        s = (q @ k.transpose(-2, -1)).tril()                      # parallel form: same as the recurrent form
        y = (s @ v) / (s.sum(-1, keepdim=True) + 1e-6)
        return self.o(y.transpose(1, 2).reshape(B, T, D)), None


class GatedDeltaNet(nn.Module):
    """q, k, v -> short causal conv -> SiLU; q, k L2-normalised; alpha = exp(g), beta = sigmoid; output RMSNorm * SiLU gate."""
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.conv = nn.Conv1d(3 * D, 3 * D, 4, groups=3 * D, padding=3)
        self.ab = nn.Linear(D, 2 * H, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(H).uniform_(1, 16)))
        self.dt_bias = nn.Parameter(torch.zeros(H))
        self.out_gate, self.norm = nn.Linear(D, D, bias=False), nn.RMSNorm(HD)
        self.o = nn.Linear(D, D, bias=False)

    def forward(self, x, return_attn=False):
        B, T, _ = x.shape
        qkv = F.silu(self.conv(self.qkv(x).transpose(1, 2))[..., :T].transpose(1, 2))
        q, k, v = qkv.view(B, T, 3, H, HD).unbind(2)                # (B, T, H, HD) each
        a, b = self.ab(x).chunk(2, -1)
        g = -self.A_log.exp() * F.softplus(a + self.dt_bias)      # log of the forget gate alpha_t (always < 0)
        y, _ = torch_chunk_gated_delta_rule(q, k, v, g=g, beta=torch.sigmoid(b), use_qk_l2norm_in_kernel=True, chunk_size=16)
        y = self.norm(y) * F.silu(self.out_gate(x).view(B, T, H, HD))
        return self.o(y.reshape(B, T, D)), None


MIXERS = {'softmax': lambda: SoftmaxAttention(False), 'gated': lambda: SoftmaxAttention(True),
          'linear': LinearAttention, 'gdn': GatedDeltaNet}


class Block(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.n1, self.n2, self.mix = nn.RMSNorm(D), nn.RMSNorm(D), MIXERS[kind]()
        self.up, self.down = nn.Linear(D, 2 * 683, bias=False), nn.Linear(683, D, bias=False)

    def forward(self, x, return_attn=False):
        y, A = self.mix(self.n1(x), return_attn)
        x = x + y
        u, g = self.up(self.n2(x)).chunk(2, -1)
        return x + self.down(F.silu(g) * u), A


class LM(nn.Module):
    def __init__(self, kinds, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, D)
        self.blocks = nn.ModuleList(Block(k) for k in kinds)
        self.norm, self.head = nn.RMSNorm(D), nn.Linear(D, vocab, bias=False)

    def forward(self, ids, return_attn=False, ids2=None):
        x, attns = self.emb(ids), []
        if ids2 is not None:
            x = x + self.emb(ids2)                                     # a memory position carries a key AND its value
        for b in self.blocks:
            x, A = b(x, return_attn)
            attns.append(A)
        return self.head(self.norm(x)), attns


# ------------------------------------------------------------------ data
def book(num):
    path = os.path.expanduser(f'~/.cache/gutenberg_{num}.txt')
    if not os.path.exists(path):
        urllib.request.urlretrieve(f'https://www.gutenberg.org/cache/epub/{num}/pg{num}.txt', path)
    text = open(path, encoding='utf-8').read()
    text = text.split('*** START OF', 1)[1].split('\n', 1)[1].split('*** END OF', 1)[0]
    return re.sub(r'\s+', ' ', text)


TRAIN_BOOKS, VAL_BOOK = [1342, 1661, 11, 84, 2701, 98], 74   # Tom Sawyer is held out for validation
train_bytes = torch.tensor(list(' '.join(book(n) for n in TRAIN_BOOKS).encode('utf-8')), dtype=torch.long)
val_bytes = torch.tensor(list(book(VAL_BOOK).encode('utf-8')), dtype=torch.long)
T_LM, B_LM, STEPS_LM = 256, 32, int(os.environ.get('STEPS_LM', 1500))


def lm_batch(src, gen):
    i = torch.randint(0, len(src) - T_LM - 1, (B_LM,), generator=gen)
    x = torch.stack([src[j:j + T_LM + 1] for j in i.tolist()])
    return x[:, :-1].to(dev), x[:, 1:].to(dev)


N_PAIRS, KEY_VOCAB, VAL_VOCAB, STEPS_AR, B_AR = int(os.environ.get('N_PAIRS', 256)), 1024, 256, int(os.environ.get('STEPS_AR', 3000)), 32
NONE = KEY_VOCAB + VAL_VOCAB                                       # "no value here" token, used at query positions
AR_VOCAB = NONE + 1


def recall_batch(gen, n_pairs=None):
    if n_pairs is None:                                            # training: a different number of facts in each batch
        n_pairs = [32, 64, 128, 256][torch.randint(0, 4, (1,), generator=gen).item()]
    """n memory positions, each holding (key, value), then n query positions holding only a key, in a new order.
    The model must output the value stored under each queried key. Keys are distinct inside a sequence."""
    keys = torch.stack([torch.randperm(KEY_VOCAB, generator=gen)[:n_pairs] for _ in range(B_AR)])
    vals = torch.randint(0, VAL_VOCAB, (B_AR, n_pairs), generator=gen) + KEY_VOCAB
    order = torch.stack([torch.randperm(n_pairs, generator=gen) for _ in range(B_AR)])
    ids = torch.cat([keys, keys.gather(1, order)], 1)
    ids2 = torch.cat([vals, torch.full_like(vals, NONE)], 1)
    y = torch.cat([torch.full_like(vals, -100), vals.gather(1, order)], 1)
    return (ids.to(dev), ids2.to(dev)), y.to(dev)


def train(model, steps, lr, get_batch, gen):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    sched = lambda s: min(1, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / steps))
    t0, hist = time.time(), []
    for s in range(steps):
        for gparam in opt.param_groups:
            gparam['lr'] = lr * sched(s)
        x, y = get_batch(gen)
        logits, _ = (model(x[0], ids2=x[1]) if isinstance(x, tuple) else model(x))
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if s % 100 == 0 or s == steps - 1:
            hist.append([s, loss.item()])
    if dev == 'mps':
        torch.mps.synchronize()
    return hist, time.time() - t0


@torch.no_grad()
def eval_lm(model):
    gen, tot = torch.Generator().manual_seed(123), 0.0
    for _ in range(40):
        x, y = lm_batch(val_bytes, gen)
        tot += F.cross_entropy(model(x)[0].flatten(0, 1), y.flatten()).item()
    return tot / 40 / math.log(2)                                  # bits per byte


@torch.no_grad()
def eval_recall(model, n_pairs=N_PAIRS):
    gen, right, total = torch.Generator().manual_seed(321), 0, 0
    for _ in range(20):
        (ids, ids2), y = recall_batch(gen, n_pairs)
        pred = model(ids, ids2=ids2)[0].argmax(-1)
        m = y != -100
        right += (pred[m] == y[m]).sum().item(); total += m.sum().item()
    return right / total


@torch.no_grad()
def first_token_share(model):
    """Average attention weight on position 0, for queries at positions 64 and later, per softmax layer."""
    gen = torch.Generator().manual_seed(7)
    x, _ = lm_batch(val_bytes, gen)
    _, attns = model(x, return_attn=True)
    return [A[..., 64:, 0].mean().item() for A in attns if A is not None]


out_path = os.environ.get('OUT', 'results/part4_train.json')
R = json.load(open(out_path)) if os.path.exists(out_path) else {}
names = sys.argv[1:] or list(PATTERNS)
SEED = int(os.environ.get('SEED', 0))                             # extra runs: a different random start
RECALL_ONLY = os.environ.get('RECALL_ONLY') == '1'                 # extra runs: skip the book task
for name in names:
    kinds = PATTERNS[name]
    torch.manual_seed(SEED)
    lm = LM(kinds, 256).to(dev)
    params = sum(p.numel() for p in lm.parameters())
    if RECALL_ONLY:
        hist, secs, bpb, sink = [], 0.0, float('nan'), None
    else:
        hist, secs = train(lm, STEPS_LM, 2e-3, lambda g: lm_batch(train_bytes, g), torch.Generator().manual_seed(1 + SEED))
        bpb = eval_lm(lm)
        sink = first_token_share(lm) if any(k in ('softmax', 'gated') for k in kinds) else None
    torch.manual_seed(SEED)
    ar = LM(kinds, AR_VOCAB).to(dev)
    ar_hist, ar_secs = train(ar, STEPS_AR, float(os.environ.get('LR_AR', 1e-3)), recall_batch, torch.Generator().manual_seed(2 + SEED))
    acc = {n: eval_recall(ar, n) for n in (32, 64, 128, 256)}
    R[name] = {'layers': kinds, 'seed': SEED, 'recall_steps': STEPS_AR, 'params_lm': params, 'lm_loss_curve': hist, 'lm_seconds': secs, 'val_bits_per_byte': bpb,
               'first_token_share': sink, 'recall_loss_curve': ar_hist, 'recall_seconds': ar_secs, 'recall_accuracy': acc}
    print('%-8s params %.2fM | books: %.3f bits/byte (%.0f s) | recall accuracy  32 facts %.3f  64 facts %.3f  128 facts %.3f  256 facts %.3f (%.0f s)%s'
          % (name, params / 1e6, bpb, secs, acc[32], acc[64], acc[128], acc[256], ar_secs,
             '' if sink is None else ' | first-token attention per softmax layer: ' + ' '.join('%.3f' % s for s in sink)), flush=True)
    json.dump(R, open(out_path, 'w'), indent=1)
