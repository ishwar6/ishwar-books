"""Chapter 4: small worked examples used in the text (learning-rate values, parameter counts, perplexity, MinHash on two sentences)."""
import math, json, hashlib, re
import numpy as np
from common import Log, save
from ch4_gpt import lr_at, GPT

log = Log('ch4_worked')
R = {}
log('== learning rate: warmup 100, cosine to 10% of 1e-3 over 1,500 steps ==')
for s in [0, 50, 99, 100, 800, 1499]:
    log(f'  step {s:5d}: lr = {lr_at(s, 1e-3, 100, 1500):.3e}')
    R[f'lr_{s}'] = lr_at(s, 1e-3, 100, 1500)

log('== parameter count of the tiny GPT ==')
d, L, V, T = 384, 6, 4096, 256
attn = 3 * d * d + 3 * d + d * d + d          # qkv weights + biases, output projection + bias
mlp = d * 4 * d + 4 * d + 4 * d * d + d       # up + bias, down + bias
ln = 2 * 2 * d                                # two LayerNorms, gain and bias
blk = attn + mlp + ln
log(f'  one block: attention {attn:,} + MLP {mlp:,} + LayerNorms {ln:,} = {blk:,}')
log(f'  12 d^2 per block = {12 * d * d:,} (the usual approximation, without biases)')
log(f'  6 blocks = {L * blk:,}; + final LayerNorm {2 * d} = {L * blk + 2 * d:,} non-embedding')
log(f'  embeddings: token {V * d:,} + position {T * d:,}; output layer shares the token matrix')
log(f'  total = {L * blk + 2 * d + V * d + T * d:,}; PyTorch count = {GPT(V).n_params(False):,}')
R['params'] = {'attn': attn, 'mlp': mlp, 'ln': ln, 'blk': blk, 'nonemb': L * blk + 2 * d, 'total': L * blk + 2 * d + V * d + T * d}

pre = json.load(open('results/ch4_pretrain.json'))
v = pre['val'][-1]
log(f'== perplexity: final validation loss {v:.3f} -> e^loss = {math.exp(v):.2f}; start {pre["val"][0]:.3f} -> {math.exp(pre["val"][0]):,.0f} (vocabulary 4,096; ln 4096 = {math.log(4096):.3f})')
R['ppl_end'], R['ppl_start'] = math.exp(v), math.exp(pre['val'][0])

log('== MinHash on two short sentences (word 3-grams, 64 hashes) ==')
a = 'the cat sat on the mat and looked out of the window at the rain'
b = 'the cat sat on the mat and looked out of the door at the rain'
sh = lambda t: {' '.join(t.split()[i:i + 3]) for i in range(len(t.split()) - 2)}
A, B = sh(a), sh(b)
jac = len(A & B) / len(A | B)
log(f'  A has {len(A)} shingles, B has {len(B)}, shared {len(A & B)}, union {len(A | B)}: Jaccard = {jac:.3f}')
P = (1 << 61) - 1
rng = np.random.default_rng(0)
coef = [(int(rng.integers(1, P)), int(rng.integers(0, P))) for _ in range(64)]
hv = lambda s: int.from_bytes(hashlib.sha1(s.encode()).digest()[:4], 'little')
mh = lambda S: [min((x * hv(s) + y) % P for s in S) for x, y in coef]
ma, mb = mh(A), mh(B)
est = sum(p == q for p, q in zip(ma, mb)) / 64
log(f'  MinHash estimate with 64 hashes: {sum(p == q for p, q in zip(ma, mb))}/64 = {est:.3f}')
R['mh'] = {'a': a, 'b': b, 'nA': len(A), 'nB': len(B), 'shared': len(A & B), 'union': len(A | B), 'jac': jac, 'est': est}
save('ch4_worked', R)
