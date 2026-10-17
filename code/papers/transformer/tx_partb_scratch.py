"""Background page, built by hand: the models before the Transformer, written from scratch and checked against
PyTorch. CPU only, about two minutes.

  1. an RNN cell by hand (Method 1) against nn.RNNCell with the same weights (Method 2), on the page's example
  2. an LSTM cell by hand against nn.LSTMCell with copied weights; what the forget gate does at the start
  3. the vanishing gradient as code: the chain of step derivatives multiplied by hand, against autograd
  4. Bahdanau's additive attention by hand on the page's example, against the same maths in batched torch
  5. a tiny sequence-to-sequence model (reverse a list of digits): our own encoder and decoder loops, against
     nn.LSTM / nn.LSTMCell with the same weights
  6. the bottleneck, measured: the same tiny model with and without attention, accuracy by length
Writes results/partb_scratch_stdout.txt and results/partb_scratch.json."""
import math, random, time
import torch
import torch.nn as nn
from common import Log, save

log = Log('partb_scratch')
torch.manual_seed(0)
random.seed(0)
torch.set_num_threads(4)
R = {}


def diff(a, b):
    return (a - b).abs().max().item()


def r3(t):
    return [round(v, 3) for v in t.tolist()]


# ------------------------------------------------------------------ 1. RNN cell
def rnn_cell(x, h, W, U, b):
    """Method 1: one step of a basic RNN, h_t = tanh(W x_t + U h_(t-1) + b)."""
    return torch.tanh(W @ x + U @ h + b)


W = torch.tensor([[0.5, -0.3], [0.8, 0.2]])        # input weights   (hidden 2, input 2)
U = torch.tensor([[0.9, 0.0], [0.1, 0.4]])         # state weights   (hidden 2, hidden 2)
b = torch.tensor([0.0, 0.1])                       # bias            (hidden 2,)
words = {'the': [1.0, 0.0], 'cat': [0.0, 1.0], 'sat': [1.0, 1.0]}

lib = nn.RNNCell(2, 2, nonlinearity='tanh')        # Method 2: PyTorch's cell, given the same numbers
with torch.no_grad():
    lib.weight_ih.copy_(W)
    lib.weight_hh.copy_(U)
    lib.bias_ih.copy_(b)
    lib.bias_hh.zero_()                            # PyTorch has two biases; their sum plays the role of b

log('== 1. RNN cell: our function against nn.RNNCell (same W, U, b)')
h_ours = torch.zeros(2)                            # h_0 = [0, 0]
h_lib = torch.zeros(1, 2)                          # the library wants a batch dimension: (1, 2)
with torch.no_grad():
    for w, x in words.items():
        x = torch.tensor(x)
        h_ours = rnn_cell(x, h_ours, W, U, b)                 # step t needs h from step t - 1
        h_lib = lib(x.unsqueeze(0), h_lib)
        log(f'  {w!r:6} ours h = {r3(h_ours)}   nn.RNNCell h = {r3(h_lib[0])}')
log(f'  max |ours - library| over the 3 steps: {diff(h_ours, h_lib[0]):.1e}')


# ------------------------------------------------------------------ 2. LSTM cell
def lstm_cell(x, h, c, W, U, b):
    """Method 1: one LSTM step. W, U, b hold the four gates stacked in PyTorch's order: i, f, g, o."""
    z = x @ W.T + h @ U.T + b                      # (batch, 4 * hidden): all four gates in one product
    i, f, g, o = z.chunk(4, dim=-1)                # split into four (batch, hidden) pieces
    i = torch.sigmoid(i)                           # input gate: how much new content to write (0..1)
    f = torch.sigmoid(f)                           # forget gate: how much old memory to keep (0..1)
    g = torch.tanh(g)                              # the new content c~ (-1..1)
    o = torch.sigmoid(o)                           # output gate: how much of the cell to show (0..1)
    c = f * c + i * g                              # the cell line: c_t = f * c_(t-1) + i * c~
    h = o * torch.tanh(c)                          # the hidden state: h_t = o * tanh(c_t)
    return h, c, f


d_in, d_h = 4, 3
cell = nn.LSTMCell(d_in, d_h)                      # Method 2, with its own random weights
Wl, Ul = cell.weight_ih.detach(), cell.weight_hh.detach()          # (12, 4) and (12, 3): 4 gates x 3 numbers
bl = (cell.bias_ih + cell.bias_hh).detach()                         # (12,)
xs = torch.randn(5, 1, d_in)                                        # 5 steps, batch 1
h1, c1 = torch.zeros(1, d_h), torch.zeros(1, d_h)
h2, c2 = torch.zeros(1, d_h), torch.zeros(1, d_h)
with torch.no_grad():
    for t in range(5):
        h1, c1, _ = lstm_cell(xs[t], h1, c1, Wl, Ul, bl)
        h2, c2 = cell(xs[t], (h2, c2))
log('')
log(f'== 2. LSTM cell: our function against nn.LSTMCell (input {d_in}, hidden {d_h}, 5 steps)')
log(f'  weight_ih {tuple(Wl.shape)}  weight_hh {tuple(Ul.shape)}  bias {tuple(bl.shape)}  (4 gates x {d_h})')
log(f'  ours    h_5 = {r3(h1[0])}  c_5 = {r3(c1[0])}')
log(f'  library h_5 = {r3(h2[0])}  c_5 = {r3(c2[0])}')
log(f'  max |ours - library|: h {diff(h1, h2):.1e}, c {diff(c1, c2):.1e}')

big = nn.LSTMCell(64, 64)
x64 = torch.randn(1, 64)
with torch.no_grad():
    _, _, f_default = lstm_cell(x64, torch.zeros(1, 64), torch.zeros(1, 64), big.weight_ih, big.weight_hh,
                                big.bias_ih + big.bias_hh)
    b_open = big.bias_ih + big.bias_hh
    b_open[64:128] += 3.0                          # the f block is the 2nd quarter of the bias
    _, _, f_open = lstm_cell(x64, torch.zeros(1, 64), torch.zeros(1, 64), big.weight_ih, big.weight_hh, b_open)
log(f'  forget gate of an untrained LSTM (hidden 64): mean {f_default.mean():.3f}; with +3 on its bias: mean {f_open.mean():.3f}')
R['forget'] = [f_default.mean().item(), f_open.mean().item()]

# ------------------------------------------------------------------ 3. the vanishing gradient by hand
log('')
log('== 3. Vanishing gradient: multiply the step derivatives by hand, compare with autograd')
log('   tanh RNN, hidden 64, PyTorch default start, one random network, 64-bit numbers')
torch.manual_seed(1)
d = 64
net = nn.RNNCell(d, d).double()
Wd, Ud, bd = net.weight_ih.detach(), net.weight_hh.detach(), (net.bias_ih + net.bias_hh).detach()
T = 50
x_seq = torch.randn(T, d, dtype=torch.float64)

# Method 1: run the loop and keep each state; then build d h_T / d h_0 as a product of T small matrices
h = torch.zeros(d, dtype=torch.float64)
J = torch.eye(d, dtype=torch.float64)              # d h_0 / d h_0 = identity
norms_hand = {}
for t in range(T):
    h = torch.tanh(Wd @ x_seq[t] + Ud @ h + bd)
    step = torch.diag(1 - h ** 2) @ Ud             # d h_t / d h_(t-1) = diag(1 - h_t^2) U   (tanh' = 1 - tanh^2)
    J = step @ J                                   # chain rule: one more factor for each step
    norms_hand[t + 1] = (torch.ones(d, dtype=torch.float64) @ J).norm().item()   # gradient of sum(h_T)

# Method 2: autograd does the same chain rule for us
norms_auto = {}
for Tm in [1, 5, 10, 20, 50]:
    h0 = torch.zeros(d, dtype=torch.float64, requires_grad=True)
    h = h0
    for t in range(Tm):
        h = torch.tanh(Wd @ x_seq[t] + Ud @ h + bd)
    g, = torch.autograd.grad(h.sum(), h0)
    norms_auto[Tm] = g.norm().item()
for Tm in [1, 5, 10, 20, 50]:
    log(f'  T = {Tm:2d}: |d sum(h_T) / d h_0| by hand {norms_hand[Tm]:.3e}   autograd {norms_auto[Tm]:.3e}')
shrink = (norms_hand[50] / norms_hand[1]) ** (1 / 49)
log(f'  average shrink per step: x {shrink:.3f}; |U| largest stretch (top singular value) {torch.linalg.matrix_norm(Ud, 2):.3f}')
R['grad_hand'] = norms_hand

# ------------------------------------------------------------------ 4. Bahdanau's additive attention
log('')
log('== 4. Additive attention (Bahdanau): e_j = v . tanh(W s + U h_j), on the page example')
H = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])   # annotations of "the", "black", "cat": (3, 2)
s = torch.tensor([-1.0, 2.0])                             # decoder state before "noir": (2,)
Wa = torch.tensor([[1.0, 0.0], [0.0, 1.0]])               # small weights we chose by hand, so you can check
Ua = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
va = torch.tensor([-1.0, 1.0])


def additive_attention_loop(s, H, Wa, Ua, va):
    """Method 1: one source word at a time, exactly as the equations read."""
    e = []
    for h_j in H:                                          # j = 1, 2, 3
        e.append(va @ torch.tanh(Wa @ s + Ua @ h_j))      # a small network gives one score per word
    e = torch.stack(e)                                     # (3,)
    alpha = torch.exp(e) / torch.exp(e).sum()              # softmax: positive, adds up to 1
    c = sum(alpha[j] * H[j] for j in range(len(H)))        # weighted sum of the annotations
    return e, alpha, c


def additive_attention_batched(s, H, Wa, Ua, va):
    """Method 2: the same maths for all words at once, with torch's softmax and a matrix product."""
    e = torch.tanh(s @ Wa.T + H @ Ua.T) @ va               # (3, 2) -> (3,): all scores in one line
    alpha = torch.softmax(e, dim=0)
    return e, alpha, alpha @ H                             # (3,) @ (3, 2) -> (2,)


e1, a1, c1 = additive_attention_loop(s, H, Wa, Ua, va)
e2, a2, c2 = additive_attention_batched(s, H, Wa, Ua, va)
log(f'  W s = {r3(Wa @ s)}; U h_j = {[r3(Ua @ hj) for hj in H]}')
log(f'  tanh(W s + U h_j) = {[r3(torch.tanh(Wa @ s + Ua @ hj)) for hj in H]}')
log(f'  scores e = {r3(e1)}   weights alpha = {r3(a1)}   context c = {r3(c1)}')
log(f'  batched torch: e = {r3(e2)}   alpha = {r3(a2)}   c = {r3(c2)}   max difference {max(diff(a1, a2), diff(c1, c2)):.1e}')
e_dot = H @ s
a_dot = torch.softmax(e_dot, 0)
sdpa = torch.nn.functional.scaled_dot_product_attention(s.view(1, 1, 2), H.view(1, 3, 2), H.view(1, 3, 2), scale=1.0)
log(f'  dot score for comparison: e = {r3(e_dot)}  alpha = {r3(a_dot)}  c = {r3(a_dot @ H)};'
    f' F.scaled_dot_product_attention (scale 1) c = {r3(sdpa[0, 0])}')
R['additive'] = dict(e=e1.tolist(), alpha=a1.tolist(), c=c1.tolist())

# ------------------------------------------------------------------ 5. a tiny sequence-to-sequence model
# The task: read a list of digits, write it in reverse. "3 1 4" -> "4 1 3". Token 10 = start, 11 = end.
BOS, EOS, V = 10, 11, 12


class Seq2Seq(nn.Module):
    """Encoder LSTM reads the digits; decoder LSTM cell writes them. With attn=True, the decoder also gets
    a Bahdanau context vector at each step (the 2015 fix of the bottleneck)."""
    def __init__(self, hidden=64, attn=False):
        super().__init__()
        self.attn, self.H = attn, hidden
        self.emb = nn.Embedding(V, 16)
        self.enc = nn.LSTM(16, hidden, batch_first=True)
        self.dec = nn.LSTMCell(16 + (hidden if attn else 0), hidden)
        self.out = nn.Linear(hidden * (2 if attn else 1), V)
        if attn:
            self.Wa = nn.Linear(hidden, hidden, bias=False)
            self.Ua = nn.Linear(hidden, hidden, bias=False)
            self.va = nn.Linear(hidden, 1, bias=False)

    def context(self, s, ann):                          # s (B, H), ann (B, n, H)
        e = self.va(torch.tanh(self.Wa(s).unsqueeze(1) + self.Ua(ann))).squeeze(-1)   # (B, n)
        alpha = torch.softmax(e, -1)
        return (alpha.unsqueeze(-1) * ann).sum(1), alpha                               # (B, H)

    def forward(self, src, tgt_in):                     # teacher forcing: tgt_in = [BOS] + reversed digits
        ann, (h, c) = self.enc(self.emb(src))           # ann (B, n, H): one annotation per source digit
        h, c = h[0], c[0]                               # the fixed-length summary: (B, H) each
        logits = []
        for t in range(tgt_in.shape[1]):
            inp = self.emb(tgt_in[:, t])
            if self.attn:
                ctx, _ = self.context(h, ann)
                inp = torch.cat([inp, ctx], -1)
            h, c = self.dec(inp, (h, c))
            o = torch.cat([h, self.context(h, ann)[0]], -1) if self.attn else h
            logits.append(self.out(o))
        return torch.stack(logits, 1)                   # (B, m, V)


def batch(B, n):
    src = torch.randint(0, 10, (B, n))
    tgt = src.flip(1)                                   # the answer: the same digits in reverse order
    tgt_in = torch.cat([torch.full((B, 1), BOS), tgt], 1)
    tgt_out = torch.cat([tgt, torch.full((B, 1), EOS)], 1)
    return src, tgt_in, tgt_out


def train(model, steps, lengths, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(steps):
        src, tgt_in, tgt_out = batch(64, random.choice(lengths))
        loss = nn.functional.cross_entropy(model(src, tgt_in).reshape(-1, V), tgt_out.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


def greedy_ours(model, src, max_len=30):
    """Method 1: encoder and decoder loops written out, with our lstm_cell and the model's weights."""
    Wi, Wh, bb = model.enc.weight_ih_l0, model.enc.weight_hh_l0, model.enc.bias_ih_l0 + model.enc.bias_hh_l0
    h = c = torch.zeros(1, model.H)
    for t in range(src.shape[1]):                       # the encoder: one digit per step, in order
        h, c, _ = lstm_cell(model.emb(src[:, t]), h, c, Wi, Wh, bb)
    Di, Dh, db = model.dec.weight_ih, model.dec.weight_hh, model.dec.bias_ih + model.dec.bias_hh
    y, out = torch.tensor([BOS]), []
    for _ in range(max_len):                            # the decoder: write, then feed the word back in
        h, c, _ = lstm_cell(model.emb(y), h, c, Di, Dh, db)
        y = (h @ model.out.weight.T + model.out.bias).argmax(-1)     # the most likely next token
        if y.item() == EOS:
            break
        out.append(y.item())
    return out, h


def greedy_lib(model, src, max_len=30):
    """Method 2: the same loop with nn.LSTM for the encoder and nn.LSTMCell for the decoder."""
    _, (h, c) = model.enc(model.emb(src))
    h, c = h[0], c[0]
    y, out = torch.tensor([BOS]), []
    for _ in range(max_len):
        h, c = model.dec(model.emb(y), (h, c))
        y = model.out(h).argmax(-1)
        if y.item() == EOS:
            break
        out.append(y.item())
    return out, h


log('')
log('== 5. A tiny sequence-to-sequence model: reverse a list of digits')
LENS = list(range(3, 13))
t0 = time.time()
plain = Seq2Seq(64, attn=False)
loss_plain = train(plain, 2500, LENS, seed=0)
log(f'  encoder LSTM 16 -> 64, decoder LSTM cell 16 -> 64, output 64 -> 12; {sum(p.numel() for p in plain.parameters()):,} weights')
log(f'  trained 2,500 steps of 64 lists, lengths 3 to 12, teacher forcing; last loss {loss_plain:.3f}; {time.time() - t0:.0f} s on the CPU')
src = torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6]])
with torch.no_grad():
    o1, hh1 = greedy_ours(plain, src)
    o2, hh2 = greedy_lib(plain, src)
log(f'  input  {src[0].tolist()}')
log(f'  ours   {o1}')
log(f'  nn.LSTM / nn.LSTMCell {o2}; same tokens: {o1 == o2}; max |h ours - h library| {diff(hh1, hh2):.1e}')

# ------------------------------------------------------------------ 6. the bottleneck, and attention
log('')
log('== 6. The bottleneck: exact-match accuracy on 500 new lists of each length')
t0 = time.time()
att = Seq2Seq(64, attn=True)
loss_att = train(att, 2500, LENS, seed=0)
log(f'  same model plus additive attention: {sum(p.numel() for p in att.parameters()):,} weights; last loss {loss_att:.3f}; {time.time() - t0:.0f} s')


def accuracy(model, n, k=500):
    torch.manual_seed(100 + n)
    src, tgt_in, tgt_out = batch(k, n)
    with torch.no_grad():
        logits = model(src, tgt_in)                       # teacher-forced; exact match needs every token right
    pred = logits.argmax(-1)
    return (pred == tgt_out).all(1).float().mean().item(), (pred == tgt_out).float().mean().item()


acc = []
log(f'  {"length":>6}  {"no attention":>13}  {"with attention":>15}   (whole list right; tokens right)')
for n in [4, 8, 12, 16]:
    a_p, t_p = accuracy(plain, n)
    a_a, t_a = accuracy(att, n)
    acc.append(dict(n=n, plain=a_p, plain_tok=t_p, attn=a_a, attn_tok=t_a))
    log(f'  {n:>6}  {a_p:>6.1%} ({t_p:5.1%})  {a_a:>7.1%} ({t_a:5.1%})' + ('   longer than any training list' if n > 12 else ''))
R['seq2seq'] = acc
save('partb_scratch', R)
