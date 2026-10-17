"""Part 6 worked examples, small enough to check by hand (paper notation: vectors are columns).
  A. an induction head built by hand: 3 tokens (a, b, c), a 6-number stream (3 "this token" + 3 "previous token"
     dimensions), a previous-token head in layer 0 and an induction head in layer 1, on the sequence a b c a;
  B. the K-composition term W_E^T W_QK W_OV^prev W_E of the hand-built heads, and its eigenvalues;
  C. eigenvalue positivity, sum(lambda) / sum(|lambda|), on three 2 x 2 matrices."""
import numpy as np
from common import Log, save

log = Log('part6_toy')
R = {}
L = lambda a: (np.round(np.asarray(a, dtype=float), 3) + 0.0).tolist()
V = ['a', 'b', 'c']
seq = ['a', 'b', 'c', 'a']

# ---------------------------------------------------------------- A. the circuit by hand
log('== A. AN INDUCTION HEAD BUILT BY HAND (tokens a, b, c; sequence a b c a)')
I3, Z3 = np.eye(3), np.zeros((3, 3))
W_E = np.vstack([I3, Z3])                       # 6 x 3: a token writes a 1 into its own "this token" dimension
X = np.stack([W_E[:, V.index(t)] for t in seq], 1)    # 6 x 4: one column per position
log(f'  stream before layer 0, one row per position: {L(X.T)}')
# layer 0: a previous-token head. Pattern: each position reads the position before (position 1 reads itself).
A0 = np.zeros((4, 4))
A0[0, 0] = 1
for i in range(1, 4):
    A0[i, i - 1] = 1
W_OV0 = np.block([[Z3, Z3], [I3, Z3]])          # read "this token" dims, write them into the "previous token" dims
X1 = X + W_OV0 @ X @ A0.T                       # h(x) = A x W_OV^T in table form; columns here, so W_OV X A^T
X1[3:, 0] = 0                                   # position 1 has no previous token: keep that part empty
log(f'  layer-0 head pattern (row = destination): {L(A0)}')
log(f'  stream after layer 0 (first 3 = this token, last 3 = previous token): {L(X1.T)}')
# layer 1: the induction head. Score = lam * <this token at the destination, previous token at the source>.
lam = 4.0
W_QK = lam * np.block([[Z3, I3], [Z3, Z3]])     # x_dst^T W_QK x_src: rows = query side, columns = key side
dst = 3
s = np.array([X1[:, dst] @ W_QK @ X1[:, j] for j in range(4)])
w = np.exp(s) / np.exp(s).sum()
log(f'  scores from the last position (token a) to positions 1..4: {L(s)}')
log(f'  softmax: {L(w)}   (e^4 / (e^4 + 3) = {np.exp(4) / (np.exp(4) + 3):.3f})')
W_OV1 = np.block([[I3, Z3], [Z3, Z3]])          # read the source's "this token" dims and write them out
W_U = np.hstack([I3, Z3])                       # 3 x 6: read the "this token" dims as logits
out = W_U @ W_OV1 @ (X1 @ w)
log(f'  induction head output read as logits for a, b, c: {L(out)}  -> predicts {V[int(out.argmax())]!r}')
R['A'] = dict(X=L(X.T), X1=L(X1.T), A0=L(A0), scores=L(s), weights=L(w), logits=L(out), lam=lam)

# ---------------------------------------------------------------- B. the K-composition term
log('\n== B. THE K-COMPOSITION TERM W_E^T W_QK W_OV^prev W_E (3 x 3, one row and column per token)')
T = W_E.T @ W_QK @ W_OV0 @ W_E
ev = np.linalg.eigvals(T)
log(f'  term = {L(T)}; eigenvalues {L(np.sort(ev.real))}; positivity {ev.sum().real / np.abs(ev).sum():.2f}')
log('  entry (row a, column a) = lam: "the destination is a" meets "the token before the source is a"')
R['B'] = dict(T=L(T), ev=L(np.sort(ev.real)))

# ---------------------------------------------------------------- C. eigenvalue positivity
log('\n== C. EIGENVALUE POSITIVITY sum(lambda) / sum(|lambda|)')
R['C'] = {}
for name, M in [('copy', [[2, 0], [0, 1]]), ('swap', [[0, 1], [1, 0]]), ('mixed', [[2, 0], [0, -1]])]:
    M = np.array(M, dtype=float)
    ev = np.linalg.eigvals(M)
    pos = ev.sum().real / np.abs(ev).sum()
    R['C'][name] = dict(M=L(M), ev=L(np.sort(ev.real)[::-1]), pos=round(float(pos), 3))
    log(f'  {name:5}: M = {L(M)}, eigenvalues {L(np.sort(ev.real)[::-1])}, positivity {pos:.2f}')

save('part6_toy', R)
