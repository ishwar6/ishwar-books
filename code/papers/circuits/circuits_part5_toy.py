"""Part 5 worked examples, small enough to check by hand (paper notation: vectors are columns).
  A. the path expansion of a two-layer model with one head per layer and a 1-number stream: four terms add up;
  B. a virtual attention head: two previous-token patterns multiply into a "two tokens back" pattern;
  C. the path expansion of one attention score: query side and key side each have two parts, so four terms;
  D. the Frobenius norm and the composition score on a 3-dimensional stream (K-composition 1, Q-composition 0)."""
import numpy as np
from common import Log, save

log = Log('part5_toy')
R = {}
L = lambda a: (np.round(np.asarray(a, dtype=float), 3) + 0.0).tolist()

# ---------------------------------------------------------------- A. path expansion of the logits
log('== A. PATH EXPANSION: TWO LAYERS, ONE HEAD EACH, A 1-NUMBER STREAM, 3 TOKENS')
x0 = np.array([1.0, 2.0, 3.0])                         # the embedding of each token (one number per token)
A1 = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]], float)   # layer-1 head: every token reads the previous one (token 1 reads itself)
A2 = np.array([[1, 0, 0], [0.5, 0.5, 0], [0, 0, 1]], float)  # layer-2 head: some other pattern
w1, w2 = 0.5, 2.0                                      # the OV "matrix" of each head is one number
x1 = x0 + w1 * (A1 @ x0)                               # after layer 1: x + head 1
x2 = x1 + w2 * (A2 @ x1)                               # after layer 2: x + head 2 (reading x1)
direct = x0
h1 = w1 * (A1 @ x0)
h2 = w2 * (A2 @ x0)
virt = (w2 * w1) * (A2 @ A1 @ x0)
R['A'] = dict(x0=L(x0), A1=L(A1), A2=L(A2), w1=w1, w2=w2, x1=L(x1), x2=L(x2), direct=L(direct), h1=L(h1), h2=L(h2),
              virt=L(virt), sum=L(direct + h1 + h2 + virt), A2A1=L(A2 @ A1))
log(f'  x0 = {L(x0)}; x1 = x0 + w1 A1 x0 = {L(x1)}; x2 = x1 + w2 A2 x1 = {L(x2)}')
log(f'  direct x0 = {L(direct)}; head 1: w1 A1 x0 = {L(h1)}; head 2: w2 A2 x0 = {L(h2)}; '
    f'virtual: w2 w1 (A2 A1) x0 = {L(virt)}')
log(f'  sum of the four terms = {L(direct + h1 + h2 + virt)}  (same as x2)')

# ---------------------------------------------------------------- B. a virtual head attends two tokens back
log('\n== B. TWO PREVIOUS-TOKEN HEADS MAKE A VIRTUAL HEAD THAT LOOKS TWO TOKENS BACK (4 tokens)')
Pv = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], float)
PP = Pv @ Pv
R['B'] = dict(P=L(Pv), PP=L(PP))
log(f'  previous-token pattern P = {L(Pv)}')
log(f'  P P = {L(PP)}  (token 4 now reads token 2; every row still adds up to 1)')

# ---------------------------------------------------------------- C. path expansion of one attention score
log('\n== C. ONE ATTENTION SCORE AS FOUR TERMS (2-number query and key vectors)')
qd, qh = np.array([1.0, 0.0]), np.array([0.0, 1.0])    # query = direct part + part written by a layer-1 head
kd, kh = np.array([2.0, 1.0]), np.array([1.0, 3.0])    # key = direct part + part written by a layer-1 head
terms = dict(none=qd @ kd, Q=qh @ kd, K=qd @ kh, QK=qh @ kh)
R['C'] = dict(qd=L(qd), qh=L(qh), kd=L(kd), kh=L(kh), terms={k: float(v) for k, v in terms.items()},
              total=float((qd + qh) @ (kd + kh)))
log(f'  q = {L(qd)} + {L(qh)}, k = {L(kd)} + {L(kh)}')
log(f'  no composition qd.kd = {terms["none"]:.0f}; Q-composition qh.kd = {terms["Q"]:.0f}; '
    f'K-composition qd.kh = {terms["K"]:.0f}; both qh.kh = {terms["QK"]:.0f}')
log(f'  sum = {sum(terms.values()):.0f} = (qd + qh).(kd + kh) = {R["C"]["total"]:.0f}')

# ---------------------------------------------------------------- D. Frobenius norm and composition scores
log('\n== D. COMPOSITION SCORES ON A 3-DIMENSIONAL STREAM')
M = np.array([[1.0, 2.0], [2.0, 0.0]])
R['D'] = dict(M=L(M), fro=float(np.linalg.norm(M)))
log(f'  Frobenius norm of {L(M)}: sqrt(1 + 4 + 4 + 0) = {np.linalg.norm(M):.0f}')
e = np.eye(3)
OV1 = 2 * np.outer(e[2], e[0])            # head 1 reads dimension 1 and writes it, doubled, into dimension 3
QK2 = np.outer(e[0], e[2])                # head 2: query reads dimension 1, key reads dimension 3
QK3 = np.outer(e[0], e[1])                # head 3: query reads dimension 1, key reads dimension 2
s = lambda A, B: np.linalg.norm(A @ B) / (np.linalg.norm(A) * np.linalg.norm(B))
R['D'].update(OV1=L(OV1), QK2=L(QK2), QK3=L(QK3), K2=float(s(QK2, OV1)), Q2=float(s(QK2.T, OV1)),
              K3=float(s(QK3, OV1)), prodK2=L(QK2 @ OV1))
log(f'  W_OV of head 1 = {L(OV1)} (reads dim 1, writes dim 3)')
log(f'  head 2 W_QK = {L(QK2)}: W_QK W_OV = {L(QK2 @ OV1)}, K-composition = {R["D"]["K2"]:.2f}, '
    f'Q-composition = {R["D"]["Q2"]:.2f}')
log(f'  head 3 W_QK = {L(QK3)} (key reads dim 2): K-composition = {R["D"]["K3"]:.2f}')
save('part5_toy', R)
