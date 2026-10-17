"""Background page experiments (the road to the Transformer). Small, CPU only, about one minute.
1. One RNN step by hand (2 numbers in the state), to check the equation h_t = tanh(W x_t + U h_{t-1} + b).
2. How far a signal travels back: the size of d h_T / d h_0 in an untrained tanh RNN and an untrained LSTM.
3. Bahdanau-style attention by hand: three source vectors, one decoder state, scores, softmax, context vector.
4. Time: an LSTM must take n steps one after another; one self-attention layer takes all n tokens at once.
Writes results/partb.json and results/partb_stdout.txt."""
import json, os, sys, time, io
import torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(0)
buf = io.StringIO()


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s); buf.write(s + '\n')


R = {}
torch.set_num_threads(4)

# ---------------------------------------------------------------- 1. one RNN step by hand
say('$ python tx_partb.py')
say('1. One RNN step, h_t = tanh(W x_t + U h_prev + b)')
W = torch.tensor([[0.5, -0.3], [0.8, 0.2]])
U = torch.tensor([[0.9, 0.0], [0.1, 0.4]])
b = torch.tensor([0.0, 0.1])
h = torch.zeros(2)
xs = {'the': torch.tensor([1.0, 0.0]), 'cat': torch.tensor([0.0, 1.0]), 'sat': torch.tensor([1.0, 1.0])}
steps = []
for w, x in xs.items():
    pre = W @ x + U @ h + b
    h = torch.tanh(pre)
    steps.append([w, [round(v, 3) for v in pre.tolist()], [round(v, 3) for v in h.tolist()]])
    say(f'   word {w!r:6} x={x.tolist()}  W x + U h + b = {[round(v, 3) for v in pre.tolist()]}  ->  h = {[round(v, 3) for v in h.tolist()]}')
R['rnn_steps'] = steps

# ---------------------------------------------------------------- 2. gradient through many steps
say('')
say('2. Size of d h_T / d h_0 (LSTM: d h_T / d c_0) (untrained, hidden size 64, 20 random starts)')
d, dists = 64, [1, 2, 5, 10, 20, 30, 50]
res = {'rnn': [], 'lstm': [], 'lstm_f3': []}
for kind in ['rnn', 'lstm', 'lstm_f3']:
    for T in dists:
        vals = []
        for seed in range(20):
            torch.manual_seed(seed)
            net = nn.RNN(d, d, nonlinearity='tanh') if kind == 'rnn' else nn.LSTM(d, d)
            if kind == 'lstm_f3':                  # forget gate starts open: bias +3, so f is about 0.95
                with torch.no_grad():
                    net.bias_ih_l0[d:2 * d] += 3.0
            x = torch.randn(T, 1, d)
            h0 = torch.zeros(1, 1, d, requires_grad=True)
            c0 = torch.zeros(1, 1, d, requires_grad=True)
            if kind == 'rnn':
                out, _ = net(x, h0)
                g, = torch.autograd.grad(out[-1].sum(), h0)     # d h_T / d h_0
            else:
                out, _ = net(x, (h0, c0))
                g, = torch.autograd.grad(out[-1].sum(), c0)     # d h_T / d c_0: the path through the cell
            vals.append(g.norm().item())
        res[kind].append(sum(vals) / len(vals))
    say(f'   {kind.upper():7}: ' + '  '.join(f'T={T}: {v:.2e}' for T, v in zip(dists, res[kind])))
R['grad'] = {'dists': dists, **res}

# ---------------------------------------------------------------- 3. attention by hand
say('')
say('3. Attention by hand (dot-product score, as in Luong et al.)')
H = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])   # annotations h_1..h_3 of "the", "black", "cat"
s = torch.tensor([-1.0, 2.0])                             # decoder state that is about to write "noir"
e = H @ s
a = torch.softmax(e, 0)
c = a @ H
say(f'   scores e = {[round(v, 3) for v in e.tolist()]}')
say(f'   weights alpha = softmax(e) = {[round(v, 3) for v in a.tolist()]}  (sum {a.sum().item():.3f})')
say(f'   context c = sum alpha_j h_j = {[round(v, 3) for v in c.tolist()]}')
R['attn'] = {'e': e.tolist(), 'alpha': a.tolist(), 'c': c.tolist()}

# ---------------------------------------------------------------- 4. sequential steps vs one parallel layer
say('')
say('4. Time for one layer over n tokens, d = 512, batch 32, CPU (median of 5 runs)')
d, B = 512, 32
lstm = nn.LSTM(d, d)
attn = nn.MultiheadAttention(d, 8)
lin = nn.Linear(d, 4 * d)


def med(f):
    ts = []
    for _ in range(5):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return sorted(ts)[2] * 1000


def lstm_cell_loop(x):
    cell = nn.LSTMCell(d, d)
    hh, cc = torch.zeros(B, d), torch.zeros(B, d)
    for t in range(x.shape[0]):            # each step must wait for the one before it
        hh, cc = cell(x[t], (hh, cc))
    return hh


tim = []
with torch.no_grad():
    for n in [16, 64, 256]:
        x = torch.randn(n, B, d)
        t_loop = med(lambda: lstm_cell_loop(x))
        t_lstm = med(lambda: lstm(x))
        t_att = med(lambda: attn(x, x, x, need_weights=False))
        tim.append([n, t_loop, t_lstm, t_att])
        say(f'   n={n:4}: LSTM step loop {t_loop:7.1f} ms ({n} steps in a row)   nn.LSTM {t_lstm:7.1f} ms   self-attention {t_att:6.1f} ms (1 step)')
R['time'] = tim

os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
json.dump(R, open(os.path.join(HERE, 'results', 'partb.json'), 'w'), indent=1)
open(os.path.join(HERE, 'results', 'partb_stdout.txt'), 'w').write(buf.getvalue())
