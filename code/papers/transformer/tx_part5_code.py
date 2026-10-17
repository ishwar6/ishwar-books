"""Part 5 (training, §5), the code walk-through. Every rule of §5 twice: once from scratch, once with PyTorch,
and a check that both give the same numbers. Every number of the code sections of part5.md comes from this output.

1. The learning rate of Eq. 3 by hand, and with torch.optim.lr_scheduler.LambdaLR.
2. Adam by hand for five steps, and torch.optim.Adam with the paper's betas and eps.
3. Dropout by hand (a Bernoulli mask), and nn.Dropout with the same seed.
4. Label smoothing by hand, and nn.CrossEntropyLoss(label_smoothing=0.1); the variant that spreads eps over
   the V - 1 wrong tokens only; the real Marian logits of Part 5.
5. Batching by length with padding masks: our own collate function against the tokenizer's padding, and what
   the mask does inside the released Marian encoder.
6. One full training step written out (forward, loss, backward, clip, Adam, schedule), then 1,000 such steps on a
   tiny Transformer that learns to reverse a list of digits. CPU, a few seconds.
"""
import math
import torch
import torch.nn as nn
from common import Log, save

log = Log('part5_code')
R = {}


# ================================================================== 1. the learning rate of Eq. 3
log('== 1. The learning rate of Eq. 3: by hand and with LambdaLR')


def lrate(step_num, d_model=512, warmup_steps=4000):
    """Eq. 3 of the paper. step_num starts at 1."""
    rise = step_num * warmup_steps ** -1.5      # straight line up during warm-up
    fall = step_num ** -0.5                     # 1 / sqrt(step) after warm-up
    return d_model ** -0.5 * min(rise, fall)    # the smaller of the two, scaled by the width


# Method 2: PyTorch. LambdaLR multiplies the optimizer's lr by f(k), where k counts calls of
# scheduler.step() and starts at 0. So we set lr = 1.0 and shift k by one: f(k) = lrate(k + 1).
w = nn.Parameter(torch.zeros(1))
opt = torch.optim.Adam([w], lr=1.0, betas=(0.9, 0.98), eps=1e-9)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda k: lrate(k + 1))

worst = 0.0
for step_num in range(1, 20001):
    lib = opt.param_groups[0]['lr']             # the lr that this step uses
    ours = lrate(step_num)
    worst = max(worst, abs(lib - ours))
    if step_num in (1, 1000, 4000, 16000):
        log(f'step {step_num:>6,}: by hand {ours:.7f}   LambdaLR {lib:.7f}')
    opt.step()                                  # (no gradient here; only the schedule matters)
    sched.step()
log(f'max abs difference over 20,000 steps: {worst:.1e}')
R['lr_diff'] = worst

# ================================================================== 2. Adam
log('\n== 2. Adam: by hand and torch.optim.Adam (lr 0.001, betas (0.9, 0.98), eps 1e-9)')


class AdamByHand:
    """Adam, Section 5.3 of the paper, written out. One m and one v for every weight."""
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.98, eps=1e-9):
        self.params = list(params)
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = [torch.zeros_like(p) for p in self.params]   # running mean of the gradient
        self.v = [torch.zeros_like(p) for p in self.params]   # running mean of the squared gradient
        self.t = 0                                             # step counter

    @torch.no_grad()
    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            m.mul_(self.beta1).add_((1 - self.beta1) * g)          # m = b1 m + (1 - b1) g
            v.mul_(self.beta2).add_((1 - self.beta2) * g * g)      # v = b2 v + (1 - b2) g^2
            m_hat = m / (1 - self.beta1 ** self.t)                 # bias correction: m and v start at 0
            v_hat = v / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (v_hat.sqrt() + self.eps)       # the step


torch.manual_seed(0)
W0 = torch.randn(4, 3)                   # one small weight matrix
X = torch.randn(8, 4)                    # 8 inputs
Y = torch.randn(8, 3)                    # 8 targets
A = nn.Parameter(W0.clone())             # copy 1, trained by our Adam
B = nn.Parameter(W0.clone())             # copy 2, trained by PyTorch's Adam
ours = AdamByHand([A], lr=1e-3)
lib = torch.optim.Adam([B], lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
for t in range(1, 6):
    for P, o in ((A, ours), (B, lib)):
        P.grad = None
        loss = ((X @ P - Y) ** 2).mean()  # the same loss for both copies
        loss.backward()
        o.step()
    log(f'step {t}: loss {loss.item():.6f}   max |A - B| = {(A - B).abs().max().item():.1e}   '
        f'mean |change of a weight| = {(A - W0).abs().mean().item():.6f}')
R['adam_diff'] = (A - B).abs().max().item()

# ================================================================== 3. dropout
log('\n== 3. Dropout: by hand and nn.Dropout(p = 0.1)')


def dropout_by_hand(x, p=0.1, training=True):
    if not training:
        return x                                   # at inference dropout does nothing
    keep = torch.empty_like(x).bernoulli_(1 - p)   # 1 with probability 1 - p, else 0
    return x * keep * (1 / (1 - p))                # zero the dropped values, scale up the kept ones


x = torch.randn(64, 512)                           # 64 tokens of d_model = 512
torch.manual_seed(1)
ours = dropout_by_hand(x)
torch.manual_seed(1)
lib = nn.Dropout(0.1).train()(x)
log(f'same seed: max abs difference {(ours - lib).abs().max().item():.1e}; '
    f'zeros {int((ours == 0).sum())} of {x.numel()} ({100 * (ours == 0).float().mean():.2f}%)')
log(f'mean of x {x.mean():.4f}, mean after dropout {ours.mean():.4f}; inference: '
    f'{torch.equal(dropout_by_hand(x, training=False), nn.Dropout(0.1).eval()(x))}')
R['drop_diff'] = (ours - lib).abs().max().item()

# ================================================================== 4. label smoothing
log('\n== 4. Label smoothing: by hand and nn.CrossEntropyLoss(label_smoothing=0.1)')


def smoothed_targets(y, V, eps=0.1, spread_over='all'):
    """The target distribution q for each position. y: (n,) correct token ids."""
    n = y.shape[0]
    if spread_over == 'all':                 # eps / V to every token, the correct one too (PyTorch, Szegedy et al.)
        q = torch.full((n, V), eps / V)
        q[torch.arange(n), y] += 1 - eps
    else:                                    # eps / (V - 1) to the wrong tokens only
        q = torch.full((n, V), eps / (V - 1))
        q[torch.arange(n), y] = 1 - eps
    return q


def smoothed_loss_by_hand(logits, y, eps=0.1, spread_over='all', ignore_index=None):
    log_p = torch.log_softmax(logits, dim=-1)              # (n, V) log probabilities
    q = smoothed_targets(y.clamp(min=0), logits.shape[-1], eps, spread_over)
    per_position = -(q * log_p).sum(-1)                   # cross-entropy between q and p
    if ignore_index is not None:                          # padding positions do not count
        keep = y != ignore_index
        return per_position[keep].mean()
    return per_position.mean()


torch.manual_seed(0)
V = 5
logits = torch.tensor([[4.0, 0.5, 0.0, -0.5, 0.2]])      # one position, word 0 is clearly first
y = torch.tensor([0])
log(f'5 words, logits {[round(a, 2) for a in logits[0].tolist()]}, correct word 0, p = {[round(a, 4) for a in torch.softmax(logits, -1)[0].tolist()]}')
log(f'target, eps over all 5 words     : {[round(a, 3) for a in smoothed_targets(y, V)[0].tolist()]}')
log(f'target, eps over the 4 wrong ones: {[round(a, 4) for a in smoothed_targets(y, V, spread_over="wrong")[0].tolist()]}')
a = smoothed_loss_by_hand(logits, y)
b = nn.CrossEntropyLoss(label_smoothing=0.1)(logits, y)
c = smoothed_loss_by_hand(logits, y, spread_over='wrong')
log(f'loss by hand {a.item():.6f}   CrossEntropyLoss {b.item():.6f}   difference {abs(a - b).item():.1e}')
log(f'loss with eps over the wrong words only: {c.item():.6f}')

# a batch with padding: 4 rows of 6 positions, vocabulary 1,000, pad id 0
logits = torch.randn(24, 1000) * 3
y = torch.randint(1, 1000, (24,))
y[[5, 10, 11, 17, 22, 23]] = 0                            # six padding positions
a = smoothed_loss_by_hand(logits, y, ignore_index=0)
b = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=0)(logits, y)
log(f'24 positions (6 of them padding), V = 1,000: by hand {a.item():.6f}   library {b.item():.6f}   '
    f'difference {abs(a - b).item():.1e}')
R['ls_diff'] = abs(a - b).item()

# the real Marian logits of Part 5
from transformers import MarianMTModel, MarianTokenizer
NAME = 'Helsinki-NLP/opus-mt-en-de'
tok = MarianTokenizer.from_pretrained(NAME)
model = MarianMTModel.from_pretrained(NAME).eval()
src = 'The cat sat on the mat because it was tired.'
tgt = 'Die Katze saß auf der Matte, weil sie müde war.'
enc = tok(src, text_target=tgt, return_tensors='pt')
with torch.no_grad():
    logits = model(**enc).logits[0]                       # (14, 58101)
labels = enc['labels'][0]
a = smoothed_loss_by_hand(logits, labels)
b = nn.CrossEntropyLoss(label_smoothing=0.1)(logits, labels)
c = smoothed_loss_by_hand(logits, labels, spread_over='wrong')
log(f'Marian, {len(labels)} target tokens, V = {logits.shape[-1]:,}: by hand {a.item():.6f}   library {b.item():.6f}   '
    f'eps over the wrong tokens only {c.item():.6f}')
R['marian_ls'] = [a.item(), b.item(), c.item()]

# ================================================================== 5. batching by length, with padding masks
log('\n== 5. Batching by length with padding masks (Marian tokenizer)')
SENTS = [
    'Attention is all you need.',
    'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.',
    'We propose a new simple network architecture.',
    'Experiments on two machine translation tasks show these models to be superior in quality.',
    'It is more parallelizable.',
    'The Transformer generalizes well to other tasks.',
    'Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task.',
    'Recurrent models compute along the symbol positions of the input and output sequences.',
]
PAD = tok.pad_token_id


def make_batches(token_lists, max_tokens):
    """Sort by length, then fill each batch until rows x longest row would pass max_tokens."""
    order = sorted(range(len(token_lists)), key=lambda i: len(token_lists[i]))
    batches, cur = [], []
    for i in order:
        longest = max(len(token_lists[j]) for j in cur + [i])
        if cur and longest * (len(cur) + 1) > max_tokens:
            batches.append(cur)
            cur = []
        cur.append(i)
    batches.append(cur)
    return batches


def pad_and_mask(rows, pad_id):
    """Rows of different lengths -> one rectangle of ids and a mask (1 = real token, 0 = padding)."""
    n = max(len(r) for r in rows)
    ids = torch.full((len(rows), n), pad_id)                   # (batch, n), all padding
    mask = torch.zeros((len(rows), n), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = torch.tensor(r)                         # real tokens on the left
        mask[i, :len(r)] = 1
    return ids, mask


ids_list = [tok(s).input_ids for s in SENTS]
log('lengths in tokens:', [len(r) for r in ids_list])
batches = make_batches(ids_list, max_tokens=48)
for b in batches:
    ids, mask = pad_and_mask([ids_list[i] for i in b], PAD)
    log(f'batch of sentences {b}: shape {tuple(ids.shape)}, real tokens {int(mask.sum())}, '
        f'padding {ids.numel() - int(mask.sum())}')
pad_all = sum(max(len(r) for r in ids_list) - len(r) for r in ids_list)
log(f'one batch of all 8 in the given order would hold {pad_all} padding tokens; '
    f'the sorted batches hold {sum(pad_and_mask([ids_list[i] for i in b], PAD)[0].numel() for b in batches) - sum(len(r) for r in ids_list)}')

# Method 2: the tokenizer pads and builds the mask for us. Same tensors?
b0 = batches[-1]
ids, mask = pad_and_mask([ids_list[i] for i in b0], PAD)
lib = tok([SENTS[i] for i in b0], padding=True, return_tensors='pt')
log(f'last batch: our ids == tokenizer ids: {torch.equal(ids, lib.input_ids)}, '
    f'our mask == tokenizer mask: {torch.equal(mask, lib.attention_mask)}')

# What the mask does: the encoder output of the shortest sentence in that batch, alone and inside the batch.
short = min(b0, key=lambda i: len(ids_list[i]))
k = b0.index(short)
n_real = len(ids_list[short])
with torch.no_grad():
    enc = model.get_encoder()
    alone = enc(input_ids=torch.tensor([ids_list[short]])).last_hidden_state[0]
    with_mask = enc(input_ids=ids, attention_mask=mask).last_hidden_state[k, :n_real]
    no_mask = enc(input_ids=ids).last_hidden_state[k, :n_real]
log(f'sentence {short} ({n_real} tokens, padded to {ids.shape[1]}): encoder output alone vs in the batch')
log(f'  with the padding mask   : max abs difference {(alone - with_mask).abs().max().item():.1e}')
log(f'  without the padding mask: max abs difference {(alone - no_mask).abs().max().item():.2f}')
R['mask_diff'] = (alone - with_mask).abs().max().item()
R['nomask_diff'] = (alone - no_mask).abs().max().item()

# ================================================================== 6. one training step, then a tiny run
log('\n== 6. One training step written out, on a tiny Transformer that learns to reverse digits')
torch.manual_seed(0)
PAD_ID, BOS, EOS, V = 0, 1, 2, 13           # tokens 3..12 are the digits 0..9
D_MODEL, HEADS, D_FF, LAYERS = 64, 4, 128, 2


class TinyTransformer(nn.Module):
    """Embeddings + sinusoidal positions + nn.Transformer (post-norm, as in the paper) + an output layer."""
    def __init__(self, small_init=True):
        super().__init__()
        self.emb = nn.Embedding(V, D_MODEL, padding_idx=PAD_ID)
        if small_init:                        # small start, so x sqrt(d_model) gives about 1
            nn.init.normal_(self.emb.weight, std=D_MODEL ** -0.5)
        pos = torch.arange(32)[:, None]
        i = torch.arange(0, D_MODEL, 2)
        pe = torch.zeros(32, D_MODEL)
        pe[:, 0::2] = torch.sin(pos / 10000 ** (i / D_MODEL))
        pe[:, 1::2] = torch.cos(pos / 10000 ** (i / D_MODEL))
        self.register_buffer('pe', pe)
        self.drop = nn.Dropout(0.1)
        self.core = nn.Transformer(D_MODEL, HEADS, LAYERS, LAYERS, D_FF, dropout=0.1, batch_first=True)
        self.out = nn.Linear(D_MODEL, V)

    def embed(self, ids):                     # token embedding * sqrt(d_model) + position, then dropout
        return self.drop(self.emb(ids) * D_MODEL ** 0.5 + self.pe[:ids.shape[1]])

    def forward(self, src, tgt_in):
        m = tgt_in.shape[1]
        causal = torch.triu(torch.ones(m, m, dtype=torch.bool), diagonal=1)   # True = may not look (future)
        h = self.core(self.embed(src), self.embed(tgt_in), tgt_mask=causal, tgt_is_causal=True,
                      src_key_padding_mask=src == PAD_ID, tgt_key_padding_mask=tgt_in == PAD_ID,
                      memory_key_padding_mask=src == PAD_ID)
        return self.out(h)                    # (batch, m, V) scores for the next token


def make_batch(n_rows, gen):
    """Random digit lists of length 3 to 8; the target is the list reversed. Padded, with BOS and EOS."""
    lens = torch.randint(3, 9, (n_rows,), generator=gen)
    rows = [torch.randint(3, 13, (int(L),), generator=gen) for L in lens]
    src, _ = pad_and_mask([r.tolist() + [EOS] for r in rows], PAD_ID)
    tgt, _ = pad_and_mask([[BOS] + r.flip(0).tolist() + [EOS] for r in rows], PAD_ID)
    return src, tgt[:, :-1], tgt[:, 1:]      # decoder input is the target shifted right by one


def clip_by_hand(params, max_norm):
    """What clip_grad_norm_ does: one norm over all gradients, then one scale factor for all."""
    grads = [p.grad for p in params if p.grad is not None]
    total = torch.sqrt(sum((g ** 2).sum() for g in grads))
    scale = min(1.0, max_norm / (total.item() + 1e-6))
    return total, [g * scale for g in grads]


def train_step(model, batch, opt, sched, loss_fn, max_norm=1.0):
    src, tgt_in, tgt_out = batch
    model.train()                                              # dropout on
    logits = model(src, tgt_in)                                # 1. forward pass      (batch, m, V)
    loss = loss_fn(logits.reshape(-1, V), tgt_out.reshape(-1)) # 2. loss, smoothed, padding ignored
    opt.zero_grad()                                            # 3. forget the old gradients
    loss.backward()                                            # 4. backward pass: new gradients
    norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm)  # 5. clip the gradient norm
    opt.step()                                                 # 6. Adam changes every weight
    sched.step()                                               # 7. next learning rate of Eq. 3
    return loss.item(), norm.item()


model_t = TinyTransformer()
n_par = sum(p.numel() for p in model_t.parameters())
log(f'tiny model: d_model {D_MODEL}, {HEADS} heads, d_ff {D_FF}, {LAYERS}+{LAYERS} layers, {n_par:,} weights')
WARMUP, STEPS = 200, 1000
opt = torch.optim.Adam(model_t.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: lrate(k + 1, D_MODEL, WARMUP))
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=PAD_ID)
gen = torch.Generator().manual_seed(1)

# the clip check, on the gradients of the first batch
batch = make_batch(64, gen)
model_t.train()
loss = loss_fn(model_t(*batch[:2]).reshape(-1, V), batch[2].reshape(-1))
loss.backward()
params = [p for p in model_t.parameters() if p.grad is not None]
total, clipped = clip_by_hand(params, 1.0)
lib_total = nn.utils.clip_grad_norm_(params, 1.0)
diff = max((c - p.grad).abs().max().item() for c, p in zip(clipped, params))
log(f'gradient norm by hand {total.item():.4f}, clip_grad_norm_ {lib_total.item():.4f}; '
    f'clipped gradients max abs difference {diff:.1e}')
opt.zero_grad()
torch.manual_seed(0)

import time
t0 = time.time()
for step in range(1, STEPS + 1):
    lr_now = opt.param_groups[0]['lr']
    loss, norm = train_step(model_t, make_batch(64, gen), opt, sched, loss_fn)
    if step in (1, 50, 100, 200, 400, 600, 1000):
        log(f'step {step:4d}: lr {lr_now:.5f}   loss {loss:.3f}   gradient norm before clipping {norm:.2f}')
qy, qo = 0.9 + 0.1 / V, 0.1 / V
log(f'lowest possible smoothed loss with V = {V}: {-(qy * math.log(qy) + (V - 1) * qo * math.log(qo)):.3f}')
log(f'{STEPS} steps in {time.time() - t0:.0f} s on the CPU')


@torch.no_grad()
def greedy(model, src, max_len=12):
    model.eval()                                               # dropout off
    out = torch.full((src.shape[0], 1), BOS)
    for _ in range(max_len):
        nxt = model(src, out)[:, -1].argmax(-1, keepdim=True)  # best next token for every row
        out = torch.cat([out, nxt], 1)
    return out


test = make_batch(500, torch.Generator().manual_seed(99))


def exact_reversals(model):
    pred = greedy(model, test[0])
    right = 0
    for i in range(500):
        want = test[2][i][test[2][i] != PAD_ID].tolist()           # reversed digits + EOS
        got = pred[i, 1:1 + len(want)].tolist()
        right += got == want
    return right, pred


right, pred = exact_reversals(model_t)
show = test[0][0][test[0][0] > 2] - 3
log(f'example: {show.tolist()} -> {(pred[0, 1:][pred[0, 1:] > 2][:len(show)] - 3).tolist()}')
log(f'exact reversals on 500 new lists: {right} / 500')

# the same run with PyTorch's default start for the embeddings (spread 1): positions get drowned
torch.manual_seed(0)
model_b = TinyTransformer(small_init=False)
opt_b = torch.optim.Adam(model_b.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
sched_b = torch.optim.lr_scheduler.LambdaLR(opt_b, lambda k: lrate(k + 1, D_MODEL, WARMUP))
gen_b = torch.Generator().manual_seed(1)
make_batch(64, gen_b)                         # skip the batch of the clip check, as above
for step in range(1, STEPS + 1):
    loss_b, _ = train_step(model_b, make_batch(64, gen_b), opt_b, sched_b, loss_fn)
right_b, _ = exact_reversals(model_b)
log(f'same run, embeddings started with spread 1 (PyTorch default): loss {loss_b:.3f}, exact reversals {right_b} / 500')
R['tiny'] = dict(params=n_par, exact=right, exact_default_init=right_b)
save('part5_code', R)
