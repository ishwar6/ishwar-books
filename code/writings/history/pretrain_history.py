"""Worked numbers for Part 5 of "How AI Got Here" (inside a modern language model: data, scale and architecture).
Every number in the text of Part 5 comes from this script or from the papers it cites.
Constants from papers: Chinchilla fit (Hoffmann et al. 2022, eq. 10), Llama 3 hyperparameters (Table 3 and section 3.4.1),
DeepSeek-V3 configuration (section 4.2) and cost (Table 1), the toy BPE vocabulary of Sennrich et al. (2016, Algorithm 1).
Run: ~/.venvs/attn/bin/python pretrain_history.py > results/pretrain_history_stdout.txt
It also writes results/pretrain_history.json (the iso-FLOP curves and the LR schedule for the figures)."""
import collections, json, math, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}


def head(s):
    print('\n' + '=' * 78 + '\n' + s + '\n' + '=' * 78)


# ---------------------------------------------------------------------------------------------
head('1. Byte-pair encoding on the toy vocabulary of Sennrich et al. (2016)')
vocab = {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3}


def pair_counts(v):
    pairs = collections.Counter()
    for word, f in v.items():
        s = word.split()
        for a, b in zip(s, s[1:]):
            pairs[a, b] += f
    return pairs


def merge(pair, v):
    pat = re.compile(r'(?<!\S)' + re.escape(' '.join(pair)) + r'(?!\S)')
    return {pat.sub(''.join(pair), w): f for w, f in v.items()}


print('start:', vocab)
print('first pair counts:', dict(sorted(pair_counts(vocab).items(), key=lambda x: -x[1])[:6]))
merges = []
for i in range(10):
    pc = pair_counts(vocab)
    best = max(pc, key=pc.get)       # ties: the first pair counted wins, as in the paper's code
    vocab = merge(best, vocab)
    merges.append(best)
    print(f'merge {i + 1:2d}: {best[0]!r:8} + {best[1]!r:8} (count {pc[best]:2d}) -> {list(vocab)}')


def encode(word, merges):
    s = list(word) + ['</w>']
    for a, b in merges:
        i, t = 0, []
        while i < len(s):
            if i + 1 < len(s) and s[i] == a and s[i + 1] == b:
                t.append(a + b); i += 2
            else:
                t.append(s[i]); i += 1
        s = t
    return s


for w in ['lowest', 'newer', 'wider', 'low']:
    print(f'encode {w!r}: {encode(w, merges)}')
OUT['bpe_merges'] = [list(m) for m in merges]

# ---------------------------------------------------------------------------------------------
head('2. Cross-entropy and perplexity')
for L in [11.76, 4.0, 2.0, 1.69]:
    print(f'loss {L:5.2f} nats per token -> perplexity exp(L) = {math.exp(L):10.2f}')
print(f'random guess over 128,000 tokens: loss ln(128000) = {math.log(128000):.2f}, perplexity 128,000')
for p in [0.9, 0.5, 0.01]:
    print(f'p(true token) = {p}: loss -ln p = {-math.log(p):.3f}')

# ---------------------------------------------------------------------------------------------
head('3. Training compute C = 6 N D (Kaplan et al. 2020)')
runs = [('GPT-3 175B', 175e9, 300e9), ('Gopher 280B', 280e9, 300e9), ('Chinchilla 70B', 70e9, 1.4e12),
        ('Llama 3 8B', 8e9, 15e12), ('Llama 3 405B', 405e9, 15.6e12), ('DeepSeek-V3 (37B active)', 37e9, 14.8e12)]
for name, N, D in runs:
    print(f'{name:26s} N = {N:.3g}, D = {D:.3g}: 6ND = {6 * N * D:.3g} FLOPs, tokens per parameter = {D / N:,.1f}')
print('tiny check: N = 1e9, D = 20e9 -> 6ND =', f'{6 * 1e9 * 20e9:.2g}')
print('per token, per parameter: 2 FLOPs forward (one multiply, one add) + 4 FLOPs backward = 6')

# ---------------------------------------------------------------------------------------------
head('4. The Chinchilla loss fit L(N, D) = E + A / N^alpha + B / D^beta (Hoffmann et al. 2022, eq. 10)')
E, A, B, al, be = 1.69, 406.4, 410.7, 0.34, 0.28


def L(N, D, E=E, A=A, B=B, al=al, be=be):
    return E + A / N ** al + B / D ** be


for name, N, D in [('Gopher 280B, 300B tokens', 280e9, 300e9), ('Chinchilla 70B, 1.4T tokens', 70e9, 1.4e12)]:
    print(f'{name:30s}: E = {E}, A/N^a = {A / N ** al:.4f}, B/D^b = {B / D ** be:.4f}, L = {L(N, D):.4f}')
a = be / (al + be); b = al / (al + be)
G = (al * A / (be * B)) ** (1 / (al + be))
print(f'compute-optimal exponents from the fit: N_opt ~ C^{a:.3f}, D_opt ~ C^{b:.3f}')


def opt(C):
    N = G * (C / 6) ** a
    return N, C / (6 * N)


print(f'{"budget C":>10} | {"fit: N_opt":>12} {"D_opt":>10} {"D/N":>6} | {"20 tokens/param: N":>19} {"D":>9}')
for C in [1e21, 1e22, 1e23, 5.76e23, 1e24, 3.8e25]:
    N, D = opt(C)
    N20 = math.sqrt(C / 120)
    print(f'{C:10.3g} | {N:12.3g} {D:10.3g} {D / N:6.1f} | {N20:19.3g} {20 * N20:9.3g}')
print('Note: the fit (Approach 3) gives fewer parameters and more tokens than the paper\'s Approaches 1 and 2;')
print('the paper\'s Table 3 (Approach 1) gives 67B parameters and 1.5T tokens for 5.76e23 FLOPs.')

head('4a. The replication fit of Besiroglu et al. (2024): E = 1.8172, A = 482.01, B = 2085.43, alpha = 0.3478, beta = 0.3658')
E2, A2, B2, al2, be2 = 1.8172, 482.01, 2085.43, 0.3478, 0.3658
a2 = be2 / (al2 + be2)
G2 = (al2 * A2 / (be2 * B2)) ** (1 / (al2 + be2))
for C in [5.76e23, 3.8e25]:
    N = G2 * (C / 6) ** a2
    print(f'C = {C:.3g}: N_opt = {N:.3g}, D_opt = {C / (6 * N):.3g}, tokens per parameter = {C / (6 * N) / N:.1f}')

iso = {}
for C in [1e21, 1e22, 1e23, 5.76e23]:
    Ns = np.logspace(7.5, 12, 200)
    Ls = [L(N, C / (6 * N)) for N in Ns]
    i = int(np.argmin(Ls))
    iso[f'{C:.3g}'] = dict(N=[float(x) for x in Ns], L=[float(x) for x in Ls], best_N=float(Ns[i]), best_L=float(Ls[i]))
    print(f'iso-FLOP C = {C:.3g}: lowest loss {Ls[i]:.4f} at N = {Ns[i]:.3g} (curve over N from 3.2e7 to 1e12)')
OUT['iso'] = iso

head('4b. Over-training: a small model on many tokens, under the same fit (an illustration only)')
N8, D8 = 8e9, 15e12
C8 = 6 * N8 * D8
No, Do = opt(C8)
print(f'Llama-3-8B-like run: C = {C8:.3g}; fit loss {L(N8, D8):.4f}')
print(f'compute-optimal under the fit for the same C: N = {No:.3g}, D = {Do:.3g}, loss {L(No, Do):.4f}')
print(f'extra loss of the small model: {L(N8, D8) - L(No, Do):.4f}; serving cost per token is {No / N8:.1f} times lower')
print(f'the same 8B model with 20 tokens per parameter (160B tokens): loss {L(8e9, 160e9):.4f}')

# ---------------------------------------------------------------------------------------------
head('5. Attention cost: the score matrix grows with the square of the length')
for n in [2048, 8192, 32768, 131072, 1048576]:
    bytes_ = n * n * 2
    print(f'n = {n:9,d}: n^2 = {n * n:.3g} scores; one head, bf16: {bytes_ / 2 ** 20:12,.0f} MiB')

# ---------------------------------------------------------------------------------------------
head('6. LayerNorm vs RMSNorm on x = [1, 2, 3, 4]')
x = np.array([1., 2., 3., 4.])
mu, sd = x.mean(), x.std()
rms = math.sqrt((x ** 2).mean())
print(f'LayerNorm: mean {mu}, std {sd:.4f} -> {np.round((x - mu) / sd, 4)}')
print(f'RMSNorm:   rms sqrt(mean(x^2)) = sqrt({(x ** 2).mean()}) = {rms:.4f} -> {np.round(x / rms, 4)}')

# ---------------------------------------------------------------------------------------------
head('7. SwiGLU')
silu = lambda z: z / (1 + math.exp(-z))
for z in [-2, -1, 0, 1, 2]:
    print(f'SiLU({z:+d}) = {silu(z):+.4f}')
print('tiny SwiGLU: x W = 2, x V = 3 -> SiLU(2) * 3 =', f'{silu(2) * 3:.4f}', '; x V = 0 closes the gate -> 0')
for name, d, f in [('Llama 3 8B', 4096, 14336), ('Llama 3 70B', 8192, 28672), ('Llama 3 405B', 16384, 53248)]:
    print(f'{name}: d = {d}, d_ff = {f} = {f / d:.2f} d; FFN weights per layer 3 d d_ff = {3 * d * f:,}')
print('original Transformer FFN: 2 matrices of d x 4d = 8 d^2; SwiGLU with d_ff = 8/3 d: 3 x d x 8/3 d = 8 d^2')

# ---------------------------------------------------------------------------------------------
head('8. RoPE: rotate by position, the dot product sees only the distance')


def rot(v, ang):
    c, s = math.cos(ang), math.sin(ang)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


q, k, th = np.array([1.0, 0.5]), np.array([0.8, -0.2]), 0.3
print(f'q = {q}, k = {k}, theta = {th}; plain q.k = {q @ k:.4f}')
for m, n in [(3, 1), (7, 5), (102, 100), (5, 1)]:
    qm, kn = rot(q, m * th), rot(k, n * th)
    print(f'q at position {m:3d}, k at position {n:3d} (distance {m - n}): q.k = {qm @ kn:.4f}')
for base in [10000, 500000]:
    d = 128
    ths = base ** (-np.arange(0, d, 2) / d)
    print(f'base {base:>7,}: theta_0 = {ths[0]:.4f} rad/token (wavelength {2 * math.pi / ths[0]:.2f} tokens), '
          f'slowest theta = {ths[-1]:.3g} (wavelength {2 * math.pi / ths[-1]:,.0f} tokens)')
print(f'position interpolation from 8K to 128K: scale s = {131072 // 8192}')

# ---------------------------------------------------------------------------------------------
head('9. KV cache per token and for a 128K context (bf16, 2 bytes)')


def kv(layers, kv_heads, dh, bytes_=2):
    return 2 * layers * kv_heads * dh * bytes_


ctx = 131072
rows = [('Llama-3-70B shape, MHA (64 KV heads)', kv(80, 64, 128)),
        ('Llama-3-70B shape, GQA (8 KV heads, as built)', kv(80, 8, 128)),
        ('Llama-3-70B shape, MQA (1 KV head)', kv(80, 1, 128)),
        ('DeepSeek-V3 shape, MHA (128 heads x 128)', kv(61, 128, 128)),
        ('DeepSeek-V3 shape, MLA (512 + 64 per layer, as built)', 61 * (512 + 64) * 2)]
for name, b_ in rows:
    print(f'{name:55s}: {b_:>10,d} bytes/token = {b_ / 1024:8.1f} KiB; 128K tokens: {b_ * ctx / 2 ** 30:7.2f} GiB')
print(f'GQA saving vs MHA (Llama shape): {rows[0][1] / rows[1][1]:.0f}x; MLA vs MHA (DeepSeek shape): {rows[3][1] / rows[4][1]:.1f}x')
OUT['kv'] = [[n, b_] for n, b_ in rows]

# ---------------------------------------------------------------------------------------------
head('10. A tiny mixture-of-experts router: softmax, top-2, Switch load-balancing loss')
logits = np.array([2.0, 1.0, 0.5, 0.1, -0.3, -1.0, 0.0, 0.3])
p = np.exp(logits) / np.exp(logits).sum()
top = np.argsort(-p)[:2]
w = p[top] / p[top].sum()
print('router logits for one token:', logits)
print('softmax probabilities:', np.round(p, 4))
print(f'top-2 experts: {top.tolist()}, probabilities {np.round(p[top], 4).tolist()}, renormalised gate weights {np.round(w, 4).tolist()}')
print('output = %.4f * FFN_%d(x) + %.4f * FFN_%d(x)' % (w[0], top[0], w[1], top[1]))
Nexp = 8


def switch_aux(f, P):
    return Nexp * float(np.sum(f * P))


f = np.full(Nexp, 1 / Nexp); P = np.full(Nexp, 1 / Nexp)
print(f'balanced: f_i = P_i = 1/8 -> aux = N sum f_i P_i = {switch_aux(f, P):.4f}')
f = np.zeros(Nexp); f[0] = 1; P = np.zeros(Nexp); P[0] = 1
print(f'collapsed: every token to expert 0 with probability 1 -> aux = {switch_aux(f, P):.4f}')
f = np.array([0.5, 0.5, 0, 0, 0, 0, 0, 0]); P = np.array([0.4, 0.4, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02])
print(f'two favourite experts: f = {f.tolist()}, P = {P.tolist()} -> aux = {switch_aux(f, P):.4f}')
rng = np.random.default_rng(0)
T = 4096
lg = rng.normal(size=(T, Nexp)) + np.array([1.5, 1.0, 0, 0, 0, 0, 0, 0])   # a router that likes experts 0 and 1
pr = np.exp(lg) / np.exp(lg).sum(1, keepdims=True)
top1 = pr.argmax(1)
f = np.bincount(top1, minlength=Nexp) / T; P = pr.mean(0)
print(f'random router biased to experts 0 and 1, {T} tokens, top-1: load f = {np.round(f, 3).tolist()}')
print(f'   aux = {switch_aux(f, P):.4f} (1.0 is perfectly balanced); with alpha = 0.01 the added loss is {0.01 * switch_aux(f, P):.5f}')
for name, tot, act in [('Mixtral 8x7B', 47, 13), ('DeepSeek-V2', 236, 21), ('DeepSeek-V3', 671, 37)]:
    print(f'{name}: {act}B of {tot}B parameters active per token = {100 * act / tot:.1f}%')
print('DeepSeek-V3 MoE layer: 1 shared + 8 of 256 routed experts active = 9 expert FFNs of 257')

# ---------------------------------------------------------------------------------------------
head('11. Learning-rate schedule: linear warm-up, then cosine decay (Llama 3 405B settings)')
peak, low, warm, total = 8e-5, 8e-7, 8000, 1_200_000


def lr(t):
    if t < warm:
        return peak * t / warm
    pr_ = min(1.0, (t - warm) / (total - warm))
    return low + 0.5 * (peak - low) * (1 + math.cos(math.pi * pr_))


for t in [0, 4000, 8000, 304000, 604000, 904000, 1_200_000]:
    print(f'step {t:>9,}: lr = {lr(t):.3e}')
print('assumption: the 1,200,000 steps count from the start, warm-up included')
OUT['lr'] = [[t, lr(t)] for t in range(0, total + 1, 10000)] + [[t, lr(t)] for t in range(0, 8001, 500)]

# ---------------------------------------------------------------------------------------------
head('12. One AdamW step')
w_, g, eta, b1, b2, eps, wd = 1.0, 0.5, 1e-3, 0.9, 0.95, 1e-8, 0.1
m = (1 - b1) * g; v = (1 - b2) * g * g
mh, vh = m / (1 - b1), v / (1 - b2)
step = eta * mh / (math.sqrt(vh) + eps)
decay = eta * wd * w_
print(f'w = {w_}, g = {g}, lr = {eta}, beta1 = {b1}, beta2 = {b2}, weight decay = {wd}')
print(f'm = {m:.4f}, v = {v:.6f}, m_hat = {mh:.4f}, v_hat = {vh:.4f}')
print(f'Adam step = lr * m_hat / sqrt(v_hat) = {step:.6f}; decay = lr * wd * w = {decay:.6f}; new w = {w_ - step - decay:.6f}')
g2 = 50.0
m = (1 - b1) * g2; v = (1 - b2) * g2 * g2
print(f'same step with g = {g2}: Adam step = {eta * (m / (1 - b1)) / (math.sqrt(v / (1 - b2)) + eps):.6f} (the first step is lr times the sign)')

# ---------------------------------------------------------------------------------------------
head('13. Memory: training states and inference weights')
for name, N in [('7.5B (ZeRO example)', 7.5e9), ('70B', 70e9), ('405B', 405e9)]:
    print(f'{name}: mixed-precision Adam, 16 bytes/param = {16 * N / 1e9:,.0f} GB; '
          f'80 GB GPUs needed just for these states: {math.ceil(16 * N / 80e9)}')
for bits in [16, 8, 4]:
    print(f'70B weights at {bits:2d} bits: {70e9 * bits / 8 / 1e9:.0f} GB')
print(f'DeepSeek-V3 671B weights at 8 bits (FP8): {671e9 / 1e9:.0f} GB; at 16 bits: {671e9 * 2 / 1e9:.0f} GB')
print('ZeRO-3 over 64 GPUs for 7.5B: 16 * 7.5e9 / 64 =', f'{16 * 7.5e9 / 64 / 1e9:.3f} GB per GPU')

# ---------------------------------------------------------------------------------------------
head('14. Parallelism: tensor split and the pipeline bubble')
for g_ in [4, 8]:
    print(f'tensor parallel, {g_} GPUs: a 4096 x 14336 FFN matrix becomes {g_} slices of 4096 x {14336 // g_}')
for p_, mb in [(4, 4), (4, 8), (4, 32), (16, 64)]:
    print(f'pipeline with p = {p_:2d} stages and m = {mb:2d} micro-batches: idle (bubble) fraction (p-1)/(m+p-1) = {(p_ - 1) / (mb + p_ - 1):.3f}')

# ---------------------------------------------------------------------------------------------
head('15. DeepSeek-V3 reported cost (Table 1), and an estimate of the achieved speed')
print('pre-training 2,664K + context extension 119K + post-training 5K =', 2664 + 119 + 5, 'K H800 GPU hours')
print('at $2 per GPU hour:', f'${(2664 + 119 + 5) * 1000 * 2 / 1e6:.3f}M', '; check: 180K GPU hours per trillion tokens x 14.8 =', 180 * 14.8, 'K')
Cds = 6 * 37e9 * 14.8e12
print(f'our estimate: 6 x 37e9 x 14.8e12 = {Cds:.3g} FLOPs over 2,664K GPU hours = '
      f'{Cds / (2664e3 * 3600) / 1e12:.0f} TFLOPS per GPU (attention FLOPs not counted)')
print(f'days on 2,048 GPUs: {2664e3 / 2048 / 24:.1f}')
print(f'that is {Cds / (2664e3 * 3600) / 989.4e12:.0%} of the H100/H800 dense BF16 peak of 989.4 TFLOPS')
Cl = 3.8e25
for mfu in [0.38, 0.43]:
    print(f'Llama 3 405B estimate: 3.8e25 FLOPs on 16,384 GPUs at 989.4 TFLOPS x {mfu:.0%} MFU = '
          f'{Cl / (16384 * 989.4e12 * mfu) / 86400:.0f} days')

# ---------------------------------------------------------------------------------------------
head('16. Speculative decoding: expected tokens per target-model pass, (1 - a^(g+1)) / (1 - a) (Leviathan et al.)')
for al_, gm in [(0.6, 4), (0.8, 4), (0.85, 1), (0.9, 1)]:
    print(f'acceptance a = {al_}, draft length g = {gm}: {(1 - al_ ** (gm + 1)) / (1 - al_):.3f} tokens per pass')

# ---------------------------------------------------------------------------------------------
head('17. Near-duplicate detection: Jaccard of word 3-grams, and MinHash buckets (FineWeb: 14 buckets of 8)')
s1 = 'the cat sat on the mat and looked out of the window'
s2 = 'the cat sat on the mat and looked out of the door'


def shingles(s, n=3):
    t = s.split()
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


A_, B_ = shingles(s1), shingles(s2)
print(f'{len(A_)} and {len(B_)} shingles, {len(A_ & B_)} shared, {len(A_ | B_)} in the union: Jaccard = {len(A_ & B_) / len(A_ | B_):.4f}')
for s in [0.5, 0.7, 0.75, 0.8, 0.9]:
    print(f'similarity {s:.2f}: chance to be flagged 1 - (1 - s^8)^14 = {1 - (1 - s ** 8) ** 14:.4f}')

# ---------------------------------------------------------------------------------------------
head('18. Images as tokens (ViT patches)')
for side, patch in [(224, 16), (896, 14), (1024, 16)]:
    print(f'{side} x {side} image, {patch} x {patch} patches: ({side}/{patch})^2 = {(side // patch) ** 2} tokens, '
          f'each patch has {patch * patch * 3} numbers (RGB)')

os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
json.dump(OUT, open(os.path.join(HERE, 'results', 'pretrain_history.json'), 'w'))
print('\nwrote results/pretrain_history.json')
