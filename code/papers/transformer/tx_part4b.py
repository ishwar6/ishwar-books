"""Part 4, code walkthroughs for Table 1: each column computed by code from the real shapes, then checked a second way.
1. Work per layer: we count the multiply-adds of a self-attention, a recurrent, a convolutional and a separable
   convolutional layer from their shapes (Method 1), then run the layers under torch.utils.flop_counter.FlopCounterMode
   (Method 2) and compare.
2. Sequential operations: the longest chain of steps in the real autograd graph of each layer, for growing n.
3. Maximum path length: a dependency graph for each layer type, stacked until the last output depends on the first
   input (Method 1), checked with autograd on real stacked layers: is d out_n / d x_1 zero or not (Method 2).
4. Restricted (local) attention: the window mask built by hand, against F.scaled_dot_product_attention with the same
   mask, then a banded version that computes only n * r scores, with its FLOP count.
Saves results/part4b.json and results/part4b_stdout.txt. CPU only, random weights."""
import math, sys
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from common import Log, save

torch.manual_seed(0)
sys.setrecursionlimit(10_000)          # the recurrent graph below is a deep chain
log = Log('part4b')
R = {}
log('$ python tx_part4b.py')


# ---------------------------------------------------------------- 1. work per layer: count by hand, then with a FLOP counter
def count_self_attention(n, d):
    """Multiply-adds of one self-attention layer, from the shapes (one head, as in Table 1)."""
    scores = n * n * d          # Q K^T: n x n dot products, d multiply-adds each
    mix = n * n * d             # weights V: n output rows, each a sum of n value vectors of d numbers
    proj = 4 * n * d * d        # x W_Q, x W_K, x W_V and the output W_O: n vectors times a d x d matrix, four times
    return scores + mix, proj


def count_recurrent(n, d):
    """Multiply-adds of one recurrent layer h_t = tanh(x_t W + h_{t-1} U)."""
    chain = n * d * d           # h_{t-1} U: one d x d product per position, one after another
    inp = n * d * d             # x_t W: the input part, also d x d per position
    return chain, inp


def count_conv(n, d, k):
    """Multiply-adds of one convolution with kernel width k, d channels in and out."""
    return k * n * d * d        # each of n outputs mixes k input vectors, each with a d x d matrix


def count_separable(n, d, k):
    """Multiply-adds of a depthwise separable convolution: a per-channel window, then a d x d mix."""
    depthwise = k * n * d       # each channel has its own k numbers: no mixing across channels
    pointwise = n * d * d       # a 1 x 1 convolution mixes the d channels at each position
    return depthwise + pointwise


def flops(fn, *args):
    """FLOPs that PyTorch's own counter sees (it counts 2 FLOPs per multiply-add)."""
    with FlopCounterMode(display=False) as fc:
        fn(*args)
    return fc.get_total_flops()


def attention_core(q, k, v):
    """The part of self-attention that Table 1 counts: the scores and the mix. q, k, v: (n, d)."""
    return torch.softmax(q @ k.T / math.sqrt(q.shape[1]), dim=-1) @ v       # (n, d)


def recurrent_chain(xw, U):
    """The part of a recurrent layer that must wait: h_t = tanh(xw_t + h_(t-1) U), one position after another."""
    h = torch.zeros(xw.shape[1])
    for t in range(xw.shape[0]):        # h_t needs h_(t-1): no way to compute the steps at the same time
        h = torch.tanh(xw[t] + h @ U)
    return h                             # the last memory (d,)


n, d, k = 27, 512, 3                                                       # 27 tokens, the base model's d
x = torch.randn(n, d)                                                      # the input: n vectors of d numbers
Wq, Wk, Wv, Wo, W, U = (torch.randn(d, d) / math.sqrt(d) for _ in range(6))
conv = torch.nn.Conv1d(d, d, k, padding=1, bias=False)                     # contiguous kernel of width 3
depthwise = torch.nn.Conv1d(d, d, k, padding=1, groups=d, bias=False)      # one small window per channel
pointwise = torch.nn.Conv1d(d, d, 1, bias=False)                           # mixes the channels
xc = x.T.unsqueeze(0)                                                      # Conv1d wants (batch, channels d, length n)

log(f'== 1. Work per layer, n = {n}, d = {d}, k = {k}: our count from the shapes vs FlopCounterMode / 2')
core, proj = count_self_attention(n, d)
chain, inp = count_recurrent(n, d)
with torch.no_grad():
    q, kk, v = x @ Wq, x @ Wk, x @ Wv
    rows = [  # (part of the layer, Method 1: our count, Method 2: what PyTorch's counter sees)
        ('self-attention core (QK^T and weights V)', core, flops(attention_core, q, kk, v)),
        ('self-attention projections W_Q W_K W_V W_O', proj, flops(lambda: (x @ Wq, x @ Wk, x @ Wv, x @ Wo))),
        ('recurrent chain h U (n steps)', chain, flops(recurrent_chain, x @ W, U)),
        ('recurrent input x W', inp, flops(lambda: x @ W)),
        ('convolution, k = 3', count_conv(n, d, k), flops(conv, xc)),
        ('separable convolution, k = 3', count_separable(n, d, k), flops(lambda: pointwise(depthwise(xc)))),
    ]
log(f'{"part of the layer":44s} {"our count":>12s} {"counter / 2":>12s}  equal')
R['work'] = []
for name, ours, fl in rows:
    log(f'{name:44s} {ours:>12,} {fl // 2:>12,}  {ours == fl // 2}')
    R['work'].append(dict(name=name, ours=ours, counter=fl // 2))
log(f'Table 1 terms: n^2 d = {n * n * d:,}, n d^2 = {n * d * d:,}, k n d^2 = {k * n * d * d:,}, '
    f'k n d + n d^2 = {k * n * d + n * d * d:,}')

log('')
log('separable convolution with k = n against self-attention + feed-forward (Table 1 terms, d = 512)')
for n2 in [27, 512]:
    sep = count_separable(n2, d, n2)
    sa_ff = n2 * n2 * d + n2 * d * d
    log(f'  n = k = {n2:>3}: separable k n d + n d^2 = {sep:>12,}   self-attention n^2 d + feed-forward n d^2 = {sa_ff:>12,}')
R['separable_eq'] = True


# ---------------------------------------------------------------- 2. sequential operations: depth of the real autograd graph
def graph_depth(t):
    """The longest chain of operations from the output back to the inputs, in PyTorch's autograd graph.
    Operations on one chain must run one after another; operations on different chains can run in parallel."""
    memo = {}

    def depth(node):
        if node is None:
            return 0
        if node not in memo:
            parents = [p for p, _ in node.next_functions]
            memo[node] = 1 + max([depth(p) for p in parents] or [0])
        return memo[node]
    return depth(t.grad_fn)


def self_attention_layer(x, Wq, Wk, Wv):
    """A whole self-attention layer (one head): the projections, then the core."""
    return attention_core(x @ Wq, x @ Wk, x @ Wv)                            # (n, d)


def recurrent_layer(x, W, U):
    """A whole recurrent layer: the input part x W for all positions at once, then the chain."""
    return recurrent_chain(x @ W, U)


log('')
log('== 2. Sequential operations: longest chain in the autograd graph (d = 64)')
log(f'{"n":>6} {"self-attention":>15} {"recurrent":>10}')
R['depth'] = []
dd = 64
Aq, Ak, Av, Wr, Ur = (torch.randn(dd, dd, requires_grad=True) for _ in range(5))
for n2 in [4, 16, 64, 256]:
    xs = torch.randn(n2, dd)
    da = graph_depth(self_attention_layer(xs, Aq, Ak, Av))
    dr = graph_depth(recurrent_layer(xs, Wr, Ur))
    log(f'{n2:>6} {da:>15} {dr:>10}')
    R['depth'].append(dict(n=n2, attn=da, rnn=dr))


# ---------------------------------------------------------------- 3. maximum path length
def window_links(n, offsets):
    """links[i, j] = True if output i of one layer reads input j directly, for j = i + offset.
    A one-sided kernel of width 3 has offsets (0, -1, -2); dilated by 3 it has (0, -3, -6)."""
    links = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n):
        for o in offsets:
            if 0 <= i + o < n:
                links[i, i + o] = True
    return links


def layers_to_connect(n, layer_links):
    """Stack layers until the last output depends on the first input; return how many layers that takes.
    layer_links(L) gives the links of layer L. The dependency of a stack is the boolean product of its layers."""
    reach = torch.eye(n, dtype=torch.bool)           # before any layer, each position depends only on itself
    for L in range(1, 10 * n):
        reach = (layer_links(L).int() @ reach.int()) > 0
        if reach[n - 1, 0]:                          # output n now depends on input 1
            return L
    return None


def recurrent_path(n):
    """x_1 enters the memory at h_1. To reach the output z_n = h_n, it must pass h_1 -> h_2 -> ... -> h_n."""
    t, hops = 1, 0                        # x_1 is now inside h_1
    while t < n:
        t += 1                            # one recurrent step: h_t is computed from h_(t-1)
        hops += 1
    return hops


log('')
log('== 3. Maximum path length (Method 1: dependency graph, layers until the last output depends on the first input)')
k = 3
log(f'{"n":>5} {"self-attn":>10} {"recurrent":>10} {"conv k=3":>9} {"dilated k=3":>12} {"restricted r=5":>15} {"r=17":>5}')
R['paths'] = []
for n2 in [8, 27, 64, 100]:
    full = torch.ones(n2, n2, dtype=torch.bool)
    p_attn = layers_to_connect(n2, lambda L: full)                                       # every query sees every key
    p_conv = layers_to_connect(n2, lambda L: window_links(n2, [0, -1, -2]))               # kernel 3, contiguous
    p_dil = layers_to_connect(n2, lambda L: window_links(n2, [0, -k ** (L - 1), -2 * k ** (L - 1)]))   # dilation 1, 3, 9, ...
    p_loc = layers_to_connect(n2, lambda L: window_links(n2, range(-2, 3)))               # window of r = 5, centred
    p_loc17 = layers_to_connect(n2, lambda L: window_links(n2, range(-8, 9)))             # window of r = 17, centred
    p_rnn = recurrent_path(n2)
    log(f'{n2:>5} {p_attn:>10} {p_rnn:>10} {p_conv:>9} {p_dil:>12} {p_loc:>15} {p_loc17:>5}')
    R['paths'].append(dict(n=n2, attn=p_attn, rnn=p_rnn, conv=p_conv, dilated=p_dil, local=p_loc, local17=p_loc17))


def conv_stack_grad(n, L, dilated=False):
    """Method 2: a real stack of L random one-sided Conv1d layers (kernel 3). Is d out_n / d x_1 zero?"""
    g = torch.Generator().manual_seed(L)
    x = torch.randn(1, 8, n, generator=g, requires_grad=True)         # 8 channels, n positions
    h = x
    for l in range(L):
        dil = 3 ** l if dilated else 1
        w = torch.randn(8, 8, 3, generator=g)
        h = F.pad(h, (2 * dil, 0))                                     # pad on the left only: output t reads t, t - dil, t - 2 dil
        h = torch.tanh(F.conv1d(h, w, dilation=dil))
    h[0, :, n - 1].sum().backward()                                    # the last output ...
    return x.grad[0, :, 0].abs().sum().item()                          # ... and its gradient at the first input


log('Method 2: real stacks of random Conv1d layers, n = 27, |d out_27 / d x_1|')
for L, dil in [(12, False), (13, False), (2, True), (3, True)]:
    gsum = conv_stack_grad(27, L, dil)
    log(f'  {"dilated" if dil else "contiguous":10s} {L:>2} layers: {gsum:.3e}  ({"connected" if gsum > 0 else "not connected"})')

# ---------------------------------------------------------------- 4. restricted (local) attention
log('')
log('== 4. Restricted self-attention: a window of r = 5 centred on each position (n = 10)')
n2, r = 10, 5
half = r // 2
local = torch.zeros(n2, n2, dtype=torch.bool)
for i in range(n2):
    for j in range(n2):
        local[i, j] = abs(i - j) <= half        # key j is inside the window of query i
log('window mask (1 = may look):')
for i in range(n2):
    log('   ' + ' '.join(str(int(v)) for v in local[i].tolist()))
log(f'allowed pairs: {int(local.sum())} of {n2 * n2} (at most r = {r} per row)')
R['local_mask'] = local.int().tolist()


def masked_attention(q, k, v, mask):
    """Equation 1 with a mask: computes all n x n scores, then hides the ones outside the window."""
    scores = q @ k.T / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask, float('-inf'))
    return torch.softmax(scores, dim=-1) @ v


def banded_attention(q, k, v, r):
    """The same result, but each query computes only its r scores: n * r work in place of n * n."""
    n, d = q.shape
    half = r // 2
    kp = F.pad(k, (0, 0, half, half))                # (n + 2 half, d): zero rows before the start and after the end
    vp = F.pad(v, (0, 0, half, half))
    k_win = kp.unfold(0, r, 1)                       # (n, d, r): for query i, the keys i - half ... i + half
    v_win = vp.unfold(0, r, 1)                       # (n, d, r)
    scores = (q.unsqueeze(1) @ k_win).squeeze(1) / math.sqrt(d)    # (n, r): r dot products per query
    pos = torch.arange(n).unsqueeze(1) + torch.arange(r) - half      # (n, r): the real position of each window slot
    scores = scores.masked_fill((pos < 0) | (pos >= n), float('-inf'))   # slots outside the sentence are padding
    w = torch.softmax(scores, dim=-1)                # (n, r)
    return (v_win @ w.unsqueeze(2)).squeeze(2)       # (n, d): mix the r values of each window


q, kk, v = torch.randn(n2, 16), torch.randn(n2, 16), torch.randn(n2, 16)
o_mask = masked_attention(q, kk, v, local)
o_lib = F.scaled_dot_product_attention(q, kk, v, attn_mask=local)
o_band = banded_attention(q, kk, v, r)
d_lib = (o_mask - o_lib).abs().max().item()       # Method 1 (our mask) against Method 2 (the library)
d_band = (o_band - o_mask).abs().max().item()     # the banded version against the masked one
log(f'masked (ours) vs F.scaled_dot_product_attention(attn_mask): max abs difference {d_lib:.1e}')
log(f'banded (n * r scores) vs masked (n * n scores):              max abs difference {d_band:.1e}')

log('FLOPs of the two versions, d = 64, r = 17 (FlopCounterMode, matrix products only)')
log(f'{"n":>6} {"masked n x n":>14} {"banded n x r":>13} {"ratio":>6}')
R['local_flops'] = []
d3, r3 = 64, 17
for n3 in [256, 1024, 4096]:
    q, kk, v = torch.randn(n3, d3), torch.randn(n3, d3), torch.randn(n3, d3)
    mask = (torch.arange(n3).unsqueeze(1) - torch.arange(n3)).abs() <= r3 // 2
    fm = flops(masked_attention, q, kk, v, mask)
    fb = flops(banded_attention, q, kk, v, r3)               # window of 17: 8 on each side and the centre
    log(f'{n3:>6} {fm:>14,} {fb:>13,} {fm / fb:>6.0f}')
    R['local_flops'].append(dict(n=n3, masked=fm, banded=fb))
save('part4b', R)
