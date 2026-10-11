"""Part 1: the hand-sized worked examples (no model, only numpy-sized arithmetic in torch).

  A. a bigram table from a nine-word corpus, and why the best zero-layer logits are log-probabilities;
  B. a zero-layer transformer with 4 tokens and a 3-dimensional residual stream: x0 = W_E t, logits = W_U x0;
  C. a residual stream as a sum: x = x0 + h1 + h2, and the logits split into one piece per part;
  D. reading and writing: virtual weights W_I W_O between a writer and two readers (one connected, one not);
  E. folding a bias into the weights with an always-one dimension;
  F. rotating the residual stream: every reader and writer changes, the output does not.
Every number printed here is quoted in the article. Runs in a second."""
import math
import torch
from common import Log, save

torch.set_printoptions(precision=3, sci_mode=False)
log = Log('part1_toy')
R = {}
V = ['the', 'cat', 'dog', 'sat']


def r(x, k=3):
    return [round(float(v), k) for v in x]


# ---------------------------------------------------------------- A. bigram statistics
corpus = 'the cat sat . the cat ran . the dog sat .'.split()
pairs = list(zip(corpus, corpus[1:]))
after_the = [b for a, b in pairs if a == 'the']
counts = {w: after_the.count(w) for w in sorted(set(after_the))}
probs = {w: c / len(after_the) for w, c in counts.items()}
logs = {w: math.log(p) for w, p in probs.items()}
log('== A. BIGRAMS IN A TINY CORPUS')
log(f'  corpus: {" ".join(corpus)}')
log(f'  words that follow "the": {after_the}  counts {counts}')
log(f'  P(next | the): {({w: round(p, 3) for w, p in probs.items()})}')
log(f'  log P(next | the): {({w: round(v, 3) for w, v in logs.items()})}')
for c in [0.0, 5.0]:
    z = torch.tensor([logs['cat'] + c, logs['dog'] + c])
    log(f'  softmax(log P + {c}) = {r(z.softmax(0))}  (adding the same number to every logit changes nothing)')
R['A'] = dict(corpus=' '.join(corpus), counts=counts, probs=probs, logs=logs)

# ---------------------------------------------------------------- B. a zero-layer transformer, by hand
W_E = torch.tensor([[1., 0, 0, 0],      # d_model = 3 rows, n_vocab = 4 columns: column j is the vector of token j
                    [0, 1, 1, 0],
                    [0, 0, 1, 1]])
W_U = torch.tensor([[0., 0, 1],         # n_vocab = 4 rows: row i scores token i as the next token
                    [2, 0, 0],
                    [1, 0, 0],
                    [0, 2, 0]])
t = torch.tensor([1., 0, 0, 0])         # one-hot vector for "the"
x0 = W_E @ t
logits = W_U @ x0
p = logits.softmax(0)
T = W_U @ W_E
log('')
log('== B. ZERO-LAYER MODEL: 4 tokens, d_model = 3')
log(f'  vocabulary: {V}')
log(f'  t (one-hot "the") = {r(t, 0)}')
log(f'  x0 = W_E t = {r(x0, 0)}  (the first column of W_E)')
log(f'  logits = W_U x0 = {r(logits, 0)}')
log(f'  softmax(logits) = {r(p)}  -> after "the": ' + ', '.join(f'{w} {float(q):.3f}' for w, q in zip(V, p)))
log(f'  the whole table W_U W_E (row = next token, column = current token):')
for i, w in enumerate(V):
    log(f'    {w:>4}: {r(T[i], 0)}')
log(f'  sizes: W_E {tuple(W_E.shape)}, W_U {tuple(W_U.shape)}, W_U W_E {tuple(T.shape)}')
R['B'] = dict(W_E=W_E.tolist(), W_U=W_U.tolist(), x0=r(x0), logits=r(logits), probs=r(p), table=T.tolist(),
              e_sum=float(torch.exp(logits).sum()))

# ---------------------------------------------------------------- C. the residual stream is a sum
h1 = torch.tensor([0., 0.5, 0])
h2 = torch.tensor([0.2, 0, -0.3])
x1 = x0 + h1 + h2
parts = {'embedding x0': x0, 'head 1': h1, 'head 2': h2}
log('')
log('== C. RESIDUAL STREAM = SUM OF PARTS')
log(f'  x0 = {r(x0, 1)}; head 1 writes {r(h1, 1)}; head 2 writes {r(h2, 1)}')
log(f'  x_final = x0 + h1 + h2 = {r(x1, 1)}')
log(f'  logits = W_U x_final = {r(W_U @ x1, 2)}')
for k, v in parts.items():
    log(f'    W_U ({k}) = {r(W_U @ v, 2)}')
log(f'    sum of the three rows = {r(sum(W_U @ v for v in parts.values()), 2)}  (the same: the unembedding is linear)')
R['C'] = dict(h1=r(h1), h2=r(h2), x1=r(x1), logits=r(W_U @ x1), split={k: r(W_U @ v) for k, v in parts.items()})

# ---------------------------------------------------------------- D. reading, writing, virtual weights
W_O1 = torch.tensor([[0.], [0], [1]])   # layer 1 writes one number into dimension 3: d_model x 1
W_Ia = torch.tensor([[0., 0, 2]])       # reader a looks at dimension 3: 1 x d_model
W_Ib = torch.tensor([[1., 0, 0]])       # reader b looks at dimension 1 only
a = torch.tensor([0.7])                 # what layer 1 wants to send
log('')
log('== D. VIRTUAL WEIGHTS')
log(f'  writer W_O {tuple(W_O1.shape)} = {W_O1.flatten().tolist()}^T; sends a = {a.item():.1f} -> adds {r(W_O1 @ a, 1)} to the stream')
log(f'  reader a: W_I = {W_Ia.flatten().tolist()}; virtual weight W_I W_O = {(W_Ia @ W_O1).item():.1f}; reads {(W_Ia @ W_O1 @ a).item():.2f}')
log(f'  reader b: W_I = {W_Ib.flatten().tolist()}; virtual weight W_I W_O = {(W_Ib @ W_O1).item():.1f}; reads {(W_Ib @ W_O1 @ a).item():.2f}')
R['D'] = dict(a=a.item(), va=(W_Ia @ W_O1).item(), vb=(W_Ib @ W_O1).item(), read_a=(W_Ia @ W_O1 @ a).item())

# ---------------------------------------------------------------- E. folding a bias
W = torch.tensor([[2., 1], [0, 3]])
b = torch.tensor([1., -1])
x = torch.tensor([4., 5])
Wb = torch.cat([W, b[:, None]], 1)
xb = torch.cat([x, torch.ones(1)])
log('')
log('== E. A BIAS IS A WEIGHT ON AN ALWAYS-ONE INPUT')
log(f'  W x + b = {r(W @ x, 0)} + {r(b, 0)} = {r(W @ x + b, 0)}')
log(f'  [W | b] [x; 1] = {Wb.tolist()} . {r(xb, 0)} = {r(Wb @ xb, 0)}')
R['E'] = dict(Wx=r(W @ x), out=r(W @ x + b))

# ---------------------------------------------------------------- F. rotation: no privileged basis
th = math.radians(90)
Rot = torch.tensor([[1., 0, 0], [0, math.cos(th), -math.sin(th)], [0, math.sin(th), math.cos(th)]])
c = W_E[:, 1]                          # "cat"
cr = Rot @ c
W_Ur = W_U @ Rot.T
log('')
log('== F. ROTATE THE STREAM BY 90 DEGREES (dimensions 2 and 3)')
log(f'  embedding of "cat": was {r(c, 0)}, now R x = {r(cr, 2)}')
log(f'  new unembedding W_U R^T: rows {[r(row, 2) for row in W_Ur]}')
log(f'  logits for "cat" before: {r(W_U @ c, 2)}; after rotating both: {r(W_Ur @ cr, 2)}')
R['F'] = dict(cat_rot=r(cr, 2), W_Ur=[r(row, 2) for row in W_Ur], logits_before=r(W_U @ c, 2), logits_rot=r(W_Ur @ cr, 2))
save('part1_toy', R)
