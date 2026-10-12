"""Part 2 worked examples, small enough to check by hand (paper notation: vectors are columns, x has one row per token).
  A. two heads: concatenate-then-multiply equals the sum of each head through its own slice of W_O;
  B. one head moving information: values, mixing with the attention pattern, output; W_OV = W_O W_V;
  C. the tensor product A (x) W: (A (x) W) x = A x W^T, in either order, and as one 4 x 4 (Kronecker) matrix;
     the mixed-product rule (A2 (x) W2)(A1 (x) W1) = (A2 A1) (x) (W2 W1);
  D. an attention score in one step: q . k = x_dst^T W_QK x_src;
  E. the head does not change when values are rescaled by M and W_O by 1/M."""
import numpy as np
from common import Log, save

log = Log('part2_toy')
R = {}
L = lambda a: np.round(np.asarray(a, dtype=float), 3).tolist()

# ---------------------------------------------------------------- A. independent and additive
log('== A. TWO HEADS: CONCATENATE THEN MULTIPLY, OR EACH HEAD ALONE THEN ADD')
WO1 = np.array([[1, 0], [0, 1], [1, 1]])        # head 1 writes its 2 numbers into a 3-dim stream
WO2 = np.array([[2, 0], [0, 0], [0, 1]])        # head 2 likewise
r1, r2 = np.array([1, 2]), np.array([3, -1])    # each head's result vector
WOH = np.hstack([WO1, WO2])                     # the big output matrix [W_O^1, W_O^2]: 3 x 4
concat = np.concatenate([r1, r2])               # [r^1; r^2]: 4 numbers
big = WOH @ concat
a, b = WO1 @ r1, WO2 @ r2
R['A'] = dict(WO1=L(WO1), WO2=L(WO2), r1=L(r1), r2=L(r2), WOH=L(WOH), concat=L(concat), big=L(big), h1=L(a), h2=L(b), sum=L(a + b))
log(f'  W_O^H = {L(WOH)} (3 x 4), [r1; r2] = {L(concat)}  ->  W_O^H [r1; r2] = {L(big)}')
log(f'  W_O^1 r1 = {L(a)}, W_O^2 r2 = {L(b)}, sum = {L(a + b)}')

# ---------------------------------------------------------------- B. one head moving information
log('\n== B. ONE HEAD MOVES INFORMATION (3 tokens, 3-dim stream, d_head = 1)')
X = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)     # one row per token: x_1, x_2, x_3
WV = np.array([[1, 1, 0]], dtype=float)                           # 1 x 3: reads dims 1 and 2
WO = np.array([[0], [0], [1]], dtype=float)                       # 3 x 1: writes into dim 3
A = np.array([[1, 0, 0], [0.5, 0.5, 0], [0.8, 0.2, 0]])           # rows = destination, columns = source
v = X @ WV.T                                                      # v_i = W_V x_i
r = A @ v                                                         # r_i = sum_j A_ij v_j
h = r @ WO.T                                                      # h(x)_i = W_O r_i
WOV = WO @ WV
h2 = A @ X @ WOV.T                                                # (A (x) W_OV) x = A x W_OV^T
R['B'] = dict(X=L(X), WV=L(WV), WO=L(WO), A=L(A), v=L(v.ravel()), r=L(r.ravel()), h=L(h), WOV=L(WOV), h2=L(h2))
log(f'  values v = W_V x_i: {L(v.ravel())}')
log(f'  results r_i = sum_j A_ij v_j: {L(r.ravel())}   (token 3: 0.8 x 2 + 0.2 x 1 = 1.8)')
log(f'  outputs h(x)_i = W_O r_i: {L(h)}')
log(f'  W_OV = W_O W_V = {L(WOV)}; A x W_OV^T = {L(h2)}  (same)')

# ---------------------------------------------------------------- C. the tensor product
log('\n== C. THE TENSOR PRODUCT: (A (x) W) x = A x W^T')
A2 = np.array([[1, 0], [0.5, 0.5]])          # 2 tokens: token 2 takes half of each
W2 = np.array([[0, 1], [1, 0]])              # per token: swap the two numbers
x2 = np.array([[1, 2], [3, 4]], dtype=float) # 2 tokens x 2 dims
w_first = x2 @ W2.T
a_first = A2 @ x2
res1, res2 = A2 @ w_first, a_first @ W2.T
K = np.kron(A2, W2)                          # the same map as one 4 x 4 matrix on the flattened x
res3 = (K @ x2.ravel()).reshape(2, 2)
R['C'] = dict(A=L(A2), W=L(W2), x=L(x2), w_first=L(w_first), a_first=L(a_first), res=L(res1), res_other=L(res2), kron=L(K), res_kron=L(res3))
log(f'  x = {L(x2)}; W first: x W^T = {L(w_first)}, then A: {L(res1)}')
log(f'  A first: A x = {L(a_first)}, then W: {L(res2)}')
log(f'  as one 4 x 4 matrix A (x) W = {L(K)}, times x flattened {L(x2.ravel())} = {L(res3.ravel())}')
B2 = np.array([[1, 0], [1, 0]]); V2 = np.array([[2, 0], [0, 3]])
lhs = np.kron(B2, V2) @ np.kron(A2, W2)
rhs = np.kron(B2 @ A2, V2 @ W2)
R['C']['mixed_product_diff'] = float(np.abs(lhs - rhs).max())
log(f'  mixed-product rule (B (x) V)(A (x) W) = (BA) (x) (VW): max difference {R["C"]["mixed_product_diff"]}')

# ---------------------------------------------------------------- D. a score in one step
log('\n== D. ONE ATTENTION SCORE: q . k = x_dst^T W_Q^T W_K x_src')
WQ = np.array([[1, 0]], dtype=float)         # 1 x 2: the query reads dim 1
WK = np.array([[0, 1]], dtype=float)         # 1 x 2: the key reads dim 2
xd, xs = np.array([2, 0.]), np.array([0, 3.])
q, k = WQ @ xd, WK @ xs
WQK = WQ.T @ WK
R['D'] = dict(WQ=L(WQ), WK=L(WK), xd=L(xd), xs=L(xs), q=float(q[0]), k=float(k[0]), score=float(q @ k), WQK=L(WQK), score1=float(xd @ WQK @ xs))
log(f'  q = W_Q x_dst = {float(q[0])}, k = W_K x_src = {float(k[0])}, q . k = {float(q @ k)}')
log(f'  W_QK = W_Q^T W_K = {L(WQK)}; x_dst^T W_QK x_src = {float(xd @ WQK @ xs)}')

# ---------------------------------------------------------------- E. values are superficial
log('\n== E. SCALE THE VALUES BY M = 2 AND W_O BY 1/2')
v_new = 2 * v
h_new = (A @ v_new) @ (WO / 2).T
R['E'] = dict(v_new=L(v_new.ravel()), same=bool(np.allclose(h_new, h)))
log(f'  new values {L(v_new.ravel())}; new outputs equal the old ones: {R["E"]["same"]}')
log(f'  W_OV is unchanged: {L((WO / 2) @ (2 * WV))}')

save('part2_toy', R)
