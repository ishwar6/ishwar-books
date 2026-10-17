"""Part 3 worked examples, small enough to check by hand.

A. Expanding a product of sums with numbers: (1 + 2)(1 + 3) = 1 + 2 + 3 + 6, the same move as the path expansion.
B. A toy one-layer model (vocabulary a, b, c; stream of 2 numbers; one head) on the two-token text "a b":
   the product (Id (x) W_U)(Id + A (x) W_OV)(Id (x) W_E) and the expanded sum Id (x) W_U W_E + A (x) W_U W_OV W_E
   give the same logits. Also as one 6 x 6 matrix (Kronecker product) acting on the flattened one-hot tokens.
C. Freezing: with A fixed, swapping the first token a -> c changes the logits at position 2 by exactly
   A[2,1] x (OV column c - OV column a); and the logits of a sum of inputs are the sum of the logits.
D. Counting paths: one layer with H heads gives 1 + H terms; two layers give 1 + 2H + H^2."""
import torch
from common import Log, save

log = Log('part3_toy')
R = {}
f = lambda t: [round(float(v), 3) for v in t.flatten()]

log('== A. EXPANDING A PRODUCT OF SUMS')
log(f'  (1 + 2) x (1 + 3) = {(1 + 2) * (1 + 3)};  1 + 2 + 3 + 2 x 3 = {1 + 2 + 3 + 2 * 3}')
R['A'] = [(1 + 2) * (1 + 3), 1 + 2 + 3 + 2 * 3]

log('\n== B. A TOY ONE-LAYER MODEL: PRODUCT = SUM OF PATHS')
W_E = torch.tensor([[1., 0., 1.], [0., 1., 1.]])        # columns: a = (1,0), b = (0,1), c = (1,1)
W_U = torch.tensor([[1., 0.], [0., 1.], [0., 0.]])       # logit of a = dim 1, of b = dim 2, of c = 0
W_OV = torch.tensor([[0., 1.], [1., 0.]])                # the head swaps the two numbers
A = torch.tensor([[1., 0.], [0.5, 0.5]])                 # token 1 reads itself; token 2 reads both, half each
t = torch.tensor([[1., 0., 0.], [0., 1., 0.]])           # "a b": one one-hot row per position
log(f'  W_E (columns a, b, c) = {W_E.tolist()};  W_U = {W_U.tolist()};  W_OV = {W_OV.tolist()};  A = {A.tolist()}')
# row form: a table x with one row per position; (A (x) W) x = A x W^T
x0 = t @ W_E.T                                           # (Id (x) W_E) t
x1 = x0 + A @ x0 @ W_OV.T                                # (Id + A (x) W_OV) x0
prod = x1 @ W_U.T                                        # (Id (x) W_U) x1
log(f'  product, step by step: x0 = {x0.tolist()}, x1 = {x1.tolist()}, logits = {prod.tolist()}')
direct = t @ (W_U @ W_E).T                               # Id (x) W_U W_E
OVc = W_U @ W_OV @ W_E                                   # the OV circuit, 3 x 3: column = source token
head = A @ t @ OVc.T                                     # A (x) W_U W_OV W_E
log(f'  W_U W_E (direct path table, column = current token) = {(W_U @ W_E).tolist()}')
log(f'  W_U W_OV W_E (OV circuit, column = source token)     = {OVc.tolist()}')
log(f'  direct path term = {direct.tolist()};  head term = {head.tolist()};  sum = {(direct + head).tolist()}')
R['B'] = dict(x0=x0.tolist(), x1=x1.tolist(), prod=prod.tolist(), WUWE=(W_U @ W_E).tolist(), OVc=OVc.tolist(),
              direct=direct.tolist(), head=head.tolist())
K = torch.kron(torch.eye(2), W_U @ W_E) + torch.kron(A, OVc)          # the whole model as one 6 x 6 matrix
flat = K @ t.flatten()
log(f'  as one 6 x 6 matrix (Id (x) W_U W_E + A (x) W_U W_OV W_E) times the flattened tokens {f(t)}: {f(flat)}')
R['B']['flat'] = f(flat)

log('\n== C. FREEZING THE PATTERN')
t2 = torch.tensor([[0., 0., 1.], [0., 1., 0.]])          # "c b": the first token swapped a -> c, A kept frozen
T = lambda tt: tt @ (W_U @ W_E).T + A @ tt @ OVc.T
d = T(t2) - T(t)
log(f'  "c b" with A frozen: logits {T(t2).tolist()}; change at position 2 = {f(d[1])}')
log(f'  A[2,1] x (OV column c - OV column a) = 0.5 x ({f(OVc[:, 2])} - {f(OVc[:, 0])}) = {f(0.5 * (OVc[:, 2] - OVc[:, 0]))}')
log(f'  linear: T("a b" + "c b") = {f(T(t + t2))};  T("a b") + T("c b") = {f(T(t) + T(t2))}')
R['C'] = dict(new=T(t2).tolist(), change=f(d[1]))

log('\n== D. COUNTING PATHS')
for H in [1, 2, 8, 12]:
    log(f'  {H:2} heads per layer: one layer {1 + H} terms; two layers {1 + 2 * H + H * H} terms (1 + {2 * H} + {H * H})')
R['D'] = {H: [1 + H, 1 + 2 * H + H * H] for H in [1, 2, 8, 12]}
save('part3_toy', R)
