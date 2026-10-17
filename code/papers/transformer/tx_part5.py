"""Part 5 (training, §5) experiments. Every number in part5.md comes from this output.

1. Batching by length (§5.1): the Marian tokenizer on the sentences of the paper itself; padding waste of random
   batches against length-sorted batches; how many sentences fill 25,000 tokens.
2. Hardware and schedule (§5.2): the arithmetic of steps, seconds, hours and tokens seen.
3. Adam (§5.3): one step worked by hand, and the memory of beta2 = 0.98 against 0.999 when the gradient grows.
4. The learning rate of Eq. 3 at chosen steps, for d_model = 512 and 1024.
5. Residual dropout (§5.4): a dropout layer on ten ones; the count of dropout places in the base model.
6. Label smoothing (§5.4): a 5-word example by hand, then the real Marian model (opus-mt-en-de) on one sentence.
"""
import math, re
import torch
from common import Log, save

log = Log('part5')
R = {}
torch.manual_seed(0)

from transformers import MarianMTModel, MarianTokenizer
NAME = 'Helsinki-NLP/opus-mt-en-de'
tok = MarianTokenizer.from_pretrained(NAME)
model = MarianMTModel.from_pretrained(NAME).eval()

# ------------------------------------------------------------------ 1. batching
log('== 1. Batching by length (Marian tokenizer, sentences of the paper)')
import pymupdf
doc = pymupdf.open(__import__('os').path.expanduser('~/.cache/papers/1706.03762.pdf'))
txt = ' '.join(doc[i].get_text() for i in range(0, 10)).replace('\n', ' ')
sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', txt) if 5 <= len(s.split()) <= 60]
lens = [len(tok(s).input_ids) for s in sents]
log(f'sentences: {len(sents)}  tokens: {sum(lens)}  mean length: {sum(lens)/len(lens):.1f}  min {min(lens)}  max {max(lens)}')

def pad_waste(order, budget=600):
    batches, cur = [], []
    for i in order:
        if cur and max(lens[j] for j in cur + [i]) * (len(cur) + 1) > budget:
            batches.append(cur); cur = []
        cur.append(i)
    batches.append(cur)
    real = sum(lens[i] for b in batches for i in b)
    slots = sum(max(lens[i] for i in b) * len(b) for b in batches)
    return len(batches), real, slots, 1 - real / slots

g = torch.Generator().manual_seed(0)
rand = torch.randperm(len(lens), generator=g).tolist()
srt = sorted(range(len(lens)), key=lambda i: lens[i])
for name, order in [('random order', rand), ('sorted by length', srt)]:
    nb, real, slots, w = pad_waste(order)
    log(f'{name:17s}: {nb:3d} batches of <= 600 slots, real tokens {real}, slots {slots}, padding {100*w:.1f}%')
    R['pad_' + name.split()[0]] = round(100 * w, 1)
R['n_sents'] = len(sents); R['mean_len'] = round(sum(lens) / len(lens), 1)
log(f'sentences in a batch of 25,000 tokens at the mean length: {25000/(sum(lens)/len(lens)):.0f}')

# ------------------------------------------------------------------ 2. schedule arithmetic
log('\n== 2. Hardware and schedule')
base_s = 0.4 * 100_000; big_s = 1.0 * 300_000
log(f'base: 100,000 steps x 0.4 s = {base_s:,.0f} s = {base_s/3600:.1f} hours (paper: 12 hours)')
log(f'big : 300,000 steps x 1.0 s = {big_s:,.0f} s = {big_s/86400:.2f} days (paper: 3.5 days)')
log(f'source tokens seen, base: 100,000 x 25,000 = {100_000*25_000:,}')
log(f'source tokens seen, big : 300,000 x 25,000 = {300_000*25_000:,}')
log(f'per GPU per step: 25,000 / 8 = {25000/8:,.0f} source tokens')

# ------------------------------------------------------------------ 3. Adam
log('\n== 3. Adam by hand (one weight, lr = 0.001, beta1 = 0.9, beta2 = 0.98, eps = 1e-9)')
b1, b2, eps, lr = 0.9, 0.98, 1e-9, 1e-3
m = v = 0.0; w = 1.0; g1 = 0.5
m = b1 * m + (1 - b1) * g1; v = b2 * v + (1 - b2) * g1 ** 2
mh = m / (1 - b1); vh = v / (1 - b2)
step = lr * mh / (math.sqrt(vh) + eps)
log(f'gradient g = {g1}: m = {m:.4f}  v = {v:.6f}  m_hat = {mh:.4f}  v_hat = {vh:.4f}  step = {step:.6f}  new w = {w-step:.6f}')
log('same first step with g = 50.0: step = %.6f (Adam steps do not grow with the gradient)' % (lr * (0.1*50/0.1) / (math.sqrt(0.02*2500/0.02) + eps)))

log('\nThe gradient is 1.0 for 200 steps, then jumps to 10.0. Size of the step (in units of lr):')
for B2 in (0.98, 0.999):
    m = v = 0.0; out = {}
    for t in range(1, 401):
        gg = 1.0 if t <= 200 else 10.0
        m = b1 * m + (1 - b1) * gg; v = B2 * v + (1 - B2) * gg * gg
        s = (m / (1 - b1 ** t)) / (math.sqrt(v / (1 - B2 ** t)) + eps)
        if t in (200, 201, 202, 205, 210, 250, 300, 400):
            out[t] = round(s, 3)
    log(f'beta2 = {B2}: memory about {1/(1-B2):.0f} steps; ' + '  '.join(f't={k}: {x}' for k, x in out.items()))
    R[f'adam_{B2}'] = out

# ------------------------------------------------------------------ 4. learning rate, Eq. 3
log('\n== 4. Learning rate of Eq. 3, warmup_steps = 4000')
def lrate(step, d=512, w=4000):
    return d ** -0.5 * min(step ** -0.5, step * w ** -1.5)
R['lr'] = {}
for s in (1, 100, 1000, 2000, 4000, 8000, 16000, 50000, 100000, 300000):
    log(f'step {s:>7,}: d=512 -> {lrate(s):.7f}   d=1024 -> {lrate(s, 1024):.7f}')
    R['lr'][s] = lrate(s)
log(f'peak (step 4000): 512^-0.5 = {512**-0.5:.5f}, 4000^-0.5 = {4000**-0.5:.5f}, product = {lrate(4000):.7f}')
R['curve'] = [[s, lrate(s), lrate(s, 1024)] for s in list(range(50, 20001, 50))]

# ------------------------------------------------------------------ 5. dropout
log('\n== 5. Residual dropout, P_drop = 0.1')
drop = torch.nn.Dropout(0.1).train()
x = torch.ones(10)
torch.manual_seed(7)
log('training :', [round(a, 3) for a in drop(x).tolist()])
log('inference:', [round(a, 3) for a in drop.eval()(x).tolist()])
log(f'kept values are scaled by 1 / (1 - 0.1) = {1/0.9:.3f}')
big = torch.nn.Dropout(0.1).train()(torch.ones(1_000_000))
log(f'on 1,000,000 ones: zeros {int((big == 0).sum()):,}, mean {big.mean():.4f}')
log('dropout places in the base model: encoder 6 x 2 = 12, decoder 6 x 3 = 18, embedding sums 2, total 32')
c = model.config
log(f'Marian config: dropout {c.dropout}, attention_dropout {c.attention_dropout}, activation_dropout {c.activation_dropout}')

# ------------------------------------------------------------------ 6. label smoothing
log('\n== 6. Label smoothing, eps = 0.1')
V = 5; e = 0.1
q = [(1 - e) + e / V if k == 0 else e / V for k in range(V)]
log(f'5-word vocabulary, correct word = word 0. Smoothed target: {[round(a, 2) for a in q]}')
def ce(qv, p):
    return -sum(a * math.log(b) for a, b in zip(qv, p))
for name, p in [('confident 0.999', [0.999] + [0.00025] * 4), ('model = target', q), ('unsure 0.60', [0.6] + [0.1] * 4)]:
    log(f'{name:16s}: p = {[round(a, 5) for a in p]}  plain loss {-math.log(p[0]):.4f}  smoothed loss {ce(q, p):.4f}')

src = 'The cat sat on the mat because it was tired.'
with torch.no_grad():
    out = model.generate(**tok(src, return_tensors='pt'), num_beams=4, max_new_tokens=40)
    tgt = tok.decode(out[0], skip_special_tokens=True)
    enc = tok(src, text_target=tgt, return_tensors='pt')
    logits = model(**enc).logits[0]
labels = enc['labels'][0]
lp = torch.log_softmax(logits, -1)
Vm = lp.shape[-1]
nll = -lp[torch.arange(len(labels)), labels]
uni = -lp.mean(-1)
ls = (1 - e) * nll + e * uni
log(f'\nMarian: "{src}" -> "{tgt}"   vocabulary {Vm:,} tokens')
log(f'{"token":>12s} {"p(token)":>9s} {"plain":>7s} {"smoothed":>9s}')
for k, t in enumerate(labels.tolist()):
    log(f'{tok.convert_ids_to_tokens(t):>12s} {nll[k].neg().exp():9.3f} {nll[k]:7.3f} {ls[k]:9.3f}')
log(f'mean plain loss {nll.mean():.3f} (perplexity {nll.mean().exp():.2f}),  mean smoothed loss {ls.mean():.3f}')
log(f'mean of the largest probability over the {len(labels)} positions: {lp.exp().max(-1).values.mean():.3f}')
R['marian'] = dict(src=src, tgt=tgt, V=Vm, plain=round(nll.mean().item(), 3), smooth=round(ls.mean().item(), 3),
                   ppl=round(nll.mean().exp().item(), 2))
save('part5', R)
qy, qo = (1 - e) + e / Vm, e / Vm
floor = -(qy * math.log(qy) + (Vm - 1) * qo * math.log(qo))
log(f'lowest possible smoothed loss with {Vm:,} tokens (prediction = target): {floor:.3f}; each other token gets {qo:.2e}')
R['marian']['floor'] = round(floor, 3)
save('part5', R)
