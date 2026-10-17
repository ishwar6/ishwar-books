"""Part 4 (Why Self-Attention, §4 and Table 1): the numbers behind the three columns of Table 1.
1. Arithmetic: per-layer work n^2*d (self-attention) against n*d^2 (recurrent) and k*n*d^2 (convolution), d = 512.
2. Real sentence lengths n with the Marian en-de tokenizer (Helsinki-NLP/opus-mt-en-de), against d = 512.
3. Wall time on this CPU: one self-attention step against a recurrent loop over n positions (d = 512).
4. Path length: how much the last output depends on each input (gradient norm), in a tanh RNN and in one
   self-attention layer, both with random weights (d = 64, n = 50).
5. Convolution: layers needed to connect two ends, contiguous (n/k) and dilated (log_k n).
Saves results/part4.json and results/part4_stdout.txt."""
import math, statistics, time
import torch
from common import Log, save

log = Log('part4')
torch.manual_seed(0)
torch.set_num_threads(4)
R = {}

# ---------------------------------------------------------------- 1. arithmetic of Table 1
d, k, r = 512, 3, 16
log('1. Work per layer (multiply-adds), d = 512, k = 3, r = 16')
log(f'{"n":>6} {"self-attn n*n*d":>16} {"recurrent n*d*d":>16} {"conv k*n*d*d":>14} {"restricted r*n*d":>17} {"attn/rnn":>9}')
rows = []
for n in [10, 27, 100, 512, 2048, 8192]:
    a, rn, cv, rs = n * n * d, n * d * d, k * n * d * d, r * n * d
    rows.append(dict(n=n, attn=a, rnn=rn, conv=cv, restricted=rs, ratio=a / rn))
    log(f'{n:>6} {a:>16,} {rn:>16,} {cv:>14,} {rs:>17,} {a / rn:>9.3f}')
R['arith'] = rows

# ---------------------------------------------------------------- 2. real sentence lengths with the Marian tokenizer
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
S = ['The cat sat on the mat.',
     'Attention is all you need.',
     'The agreement on the European Economic Area was signed in August 1992.',
     'Self-attention layers are faster than recurrent layers when the sequence length is smaller than the representation dimensionality.',
     'Learning long-range dependencies is a key challenge in many sequence transduction tasks, and the length of the paths that signals must travel is one key factor.',
     'The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.']
log('')
log('2. Real sentence lengths (Marian en-de tokenizer, d_model = 512)')
lens = []
for s in S:
    ids = tok(s)['input_ids']
    lens.append(len(ids))
    log(f'  n = {len(ids):>3}  words = {len(s.split()):>2}  n/d = {len(ids) / 512:.3f}  {s[:60]}{"..." if len(s) > 60 else ""}')
pieces = tok.convert_ids_to_tokens(tok(S[3])['input_ids'])
log('  tokens of sentence 4:', ' '.join(pieces))
R['lengths'] = dict(sentences=S, n=lens, pieces4=pieces)

# ---------------------------------------------------------------- 3. wall time: parallel attention against a sequential loop
log('')
log('3. Wall time on CPU (4 threads), d = 512, median of 7 runs, milliseconds')
log(f'{"n":>6} {"self-attn":>10} {"recurrent loop":>15} {"steps attn":>11} {"steps rnn":>10}')
W = torch.randn(d, d) / math.sqrt(d); U = torch.randn(d, d) / math.sqrt(d)
Wq, Wk, Wv = (torch.randn(d, d) / math.sqrt(d) for _ in range(3))


def attn(x):
    q, kk, v = x @ Wq, x @ Wk, x @ Wv
    return torch.softmax(q @ kk.T / math.sqrt(d), -1) @ v


def rnn(x):
    h = torch.zeros(d)
    xw = x @ W                       # the input part can be done for all positions at once
    for t in range(x.shape[0]):      # the recurrent part cannot: h_t needs h_{t-1}
        h = torch.tanh(xw[t] + h @ U)
    return h


def ms(f, x, reps=7):
    f(x)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); f(x); ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


timing = []
with torch.no_grad():
    for n in [16, 64, 256, 1024, 4096]:
        x = torch.randn(n, d)
        ta, tr = ms(attn, x), ms(rnn, x)
        timing.append(dict(n=n, attn_ms=ta, rnn_ms=tr))
        log(f'{n:>6} {ta:>10.3f} {tr:>15.3f} {1:>11} {n:>10}')
R['timing'] = timing

# ---------------------------------------------------------------- 4. path length: dependence of the last output on each input
log('')
log('4. Gradient norm |d out_last / d x_i| (random weights, d = 64, n = 50), mean of 20 seeds')
dd, n = 64, 50
gr_r, gr_a = torch.zeros(n), torch.zeros(n)
for seed in range(20):
    g = torch.Generator().manual_seed(seed)
    Ur = torch.randn(dd, dd, generator=g) / math.sqrt(dd); Wr = torch.randn(dd, dd, generator=g) / math.sqrt(dd)
    Aq, Ak, Av = (torch.randn(dd, dd, generator=g) / math.sqrt(dd) for _ in range(3))
    x = torch.randn(n, dd, generator=g, requires_grad=True)
    h = torch.zeros(dd)
    for t in range(n):
        h = torch.tanh(x[t] @ Wr + h @ Ur)
    h.sum().backward()
    gr_r += x.grad.norm(dim=1) / 20
    x2 = x.detach().clone().requires_grad_(True)
    z = torch.softmax((x2 @ Aq) @ (x2 @ Ak).T / math.sqrt(dd), -1) @ (x2 @ Av)
    z[-1].sum().backward()
    gr_a += x2.grad.norm(dim=1) / 20
show = [0, 9, 19, 29, 39, 44, 48, 49]
log(f'{"input i":>8} {"distance":>9} {"RNN":>12} {"self-attn":>12}')
for i in show:
    log(f'{i + 1:>8} {n - 1 - i:>9} {gr_r[i].item():>12.2e} {gr_a[i].item():>12.2e}')
R['grad'] = dict(n=n, d=dd, rnn=gr_r.tolist(), attn=gr_a.tolist())

# ---------------------------------------------------------------- 5. convolution: layers to connect the two ends
log('')
log('5. Layers needed to connect position 1 and position n (kernel k = 3)')
conv = []
for n in [10, 27, 100, 512]:
    contig = math.ceil((n - 1) / (k - 1))      # each layer widens the receptive field by k - 1
    dil = math.ceil(math.log(n, k))            # dilations 1, k, k^2, ...: field k^L
    conv.append(dict(n=n, contiguous=contig, dilated=dil))
    log(f'  n = {n:>4}: contiguous {contig:>4} layers, dilated {dil:>2} layers, self-attention 1 layer, RNN {n - 1} steps')
R['conv'] = conv
save('part4', R)
