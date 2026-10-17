"""Part 4 worked examples, small enough to check by hand (paper notation: M = W_U W_OV W_E, column = source token,
row = out token, so M[out, source]).
  A. a skip-trigram in factored form: (attention from the destination) x (OV effect of the source); why one head cannot
     make "keep ... in" prefer "mind" while "keep ... at" prefers "bay";
  B. eigenvalues of small OV circuits: a copying matrix, a swap matrix, a rotation, and a "pathological" matrix whose
     eigenvalues are all positive although one token lowers its own logit;
  C. the copy score sum(l) / sum(|l|), and trace = sum of eigenvalues = sum of the diagonal;
  D. AB and BA have the same non-zero eigenvalues (the trick that makes 48,262 x 48,262 circuits small)."""
import numpy as np
from common import Log, save

log = Log('part4_toy')
R = {}
L = lambda a: (np.round(np.asarray(a, dtype=float), 3) + 0.0).tolist()

# ---------------------------------------------------------------- A. factored form
log('== A. A SKIP-TRIGRAM IN FACTORED FORM: push on out = (attention of destination on " keep") x (OV of " keep")')
ov = {'mind': 2.0, 'bay': 6.0}                 # what reading " keep" does to two logits (OV column)
att = {'in': 0.5, 'at': 0.8}                   # how much each destination reads " keep" (from the QK circuit)
table = {d: {o: att[d] * ov[o] for o in ov} for d in att}
R['A'] = dict(ov=ov, att=att, table=table)
log(f'  OV column of " keep": {ov}; attention on " keep" from " in" {att["in"]}, from " at" {att["at"]}')
for d in att:
    log(f'  destination {d!r}: mind {att[d]} x {ov["mind"]} = {table[d]["mind"]:.1f}, bay {att[d]} x {ov["bay"]} = {table[d]["bay"]:.1f}'
        f'  (bay / mind = {table[d]["bay"] / table[d]["mind"]:.1f})')
log('  wanted: " in" -> mind, " at" -> bay. The ratio bay / mind is 3.0 for every destination, so one head cannot do it.')

# ---------------------------------------------------------------- B. eigenvalues
log('\n== B. EIGENVALUES OF SMALL OV CIRCUITS (columns = source token, rows = out token)')
cases = {
    'copy': (np.array([[2.0, 1.0], [1.0, 2.0]]), '" cat" and " Cat" each raise themselves (2) and each other (1)'),
    'swap': (np.array([[0.0, 1.0], [1.0, 0.0]]), '" cat" raises " cats" and " cats" raises " cat"'),
    'rotate': (np.array([[0.0, -1.0], [1.0, 0.0]]), 'token 1 raises token 2, token 2 lowers token 1'),
    'pathological': (np.array([[-1.0, 6.0], [-1.0, 4.0]]), 'all eigenvalues positive, yet token 1 lowers its own logit'),
}
R['B'] = {}
for name, (M, what) in cases.items():
    lam, vec = np.linalg.eig(M)
    order = np.argsort(-lam.real)
    lam, vec = lam[order], vec[:, order]
    score = float(lam.sum().real / np.abs(lam).sum())
    entry = dict(M=L(M), what=what, lam=[[round(float(z.real), 3), round(float(z.imag), 3)] for z in lam],
                 score=round(score, 3), trace=float(np.trace(M)), lam_sum=round(float(lam.sum().real), 3), diag=L(np.diag(M)))
    if np.isreal(lam).all():
        v = np.real(vec)
        v = v / np.sign(v[np.argmax(np.abs(v), 0), range(v.shape[1])])
        entry['vec'] = L(v.T)
        entry['angle'] = round(float(np.degrees(np.arccos(abs(v[:, 0] @ v[:, 1]) / np.linalg.norm(v[:, 0]) / np.linalg.norm(v[:, 1])))), 1)
    R['B'][name] = entry
    lam_s = ', '.join(f'{z.real:+.3f}' + (f'{z.imag:+.3f}i' if abs(z.imag) > 1e-12 else '') for z in lam)
    log(f'  {name:12} M = {L(M)}: {what}')
    log(f'               eigenvalues {lam_s}; copy score {score:+.3f}; trace {np.trace(M):+.1f} = sum of eigenvalues {lam.sum().real:+.1f}; diagonal {L(np.diag(M))}')
    if 'vec' in entry:
        log(f'               eigenvectors {entry["vec"]} (angle between them {entry["angle"]} degrees)')
M = cases['copy'][0]
v1 = np.array([1.0, 1.0])
R['B']['copy_check'] = L(M @ v1)
log(f'  check by hand: copy matrix times (1, 1) = {L(M @ v1)} = 3 x (1, 1)')
M = cases['pathological'][0]
R['B']['patho_col1'] = L(M[:, 0])
log(f'  pathological: reading token 1 gives logits {L(M[:, 0])}: token 1 goes down by 1')

# ---------------------------------------------------------------- D. AB and BA
log('\n== D. AB AND BA HAVE THE SAME NON-ZERO EIGENVALUES')
A = np.array([[1.0], [2.0], [0.0]])            # 3 x 1: "W_E W_V" of a 3-token vocabulary and d_head = 1
B = np.array([[1.0, 1.0, 1.0]])                # 1 x 3: "W_O W_U"
AB, BA = A @ B, B @ A
lam = np.sort(np.linalg.eigvals(AB).real)[::-1]
R['D'] = dict(A=L(A), B=L(B), AB=L(AB), BA=L(BA), lam_AB=L(lam))
log(f'  A = {L(A.ravel())} (column), B = {L(B.ravel())} (row)')
log(f'  AB = {L(AB)} (3 x 3) has eigenvalues {L(lam)}; BA = {L(BA)} (1 x 1)')
save('part4_toy', R)
