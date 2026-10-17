"""Part 7 worked examples, small enough to check by hand.
  A. GeLU is not linear: GeLU(a + b) is not GeLU(a) + GeLU(b), so a sum cannot be pushed through an MLP layer;
  B. two features in one dimension (an antipodal pair), x' = ReLU(W^T W x + b), with and without the ReLU;
  C. the composition-score bug: trace and eigenvalues do not change when a product is turned around, the Frobenius norm does;
  D. one sparse-autoencoder feature by hand: encode, decode, and the two parts of the loss;
  E. the in-context learning score of three made-up windows."""
import math
import numpy as np
from common import Log, save

log = Log('part7_toy')
R = {}
L = lambda a: (np.round(np.asarray(a, dtype=float), 3) + 0.0).tolist()
gelu = lambda z: 0.5 * z * (1 + math.erf(z / math.sqrt(2)))

# ---------------------------------------------------------------- A
log('== A. GeLU IS NOT LINEAR')
a, b = 1.0, -1.0
R['A'] = dict(ga=gelu(a), gb=gelu(b), gsum=gelu(a + b), sum_g=gelu(a) + gelu(b))
log(f'  GeLU(1) = {gelu(a):.3f}, GeLU(-1) = {gelu(b):.3f}, their sum = {gelu(a) + gelu(b):.3f}')
log(f'  GeLU(1 + (-1)) = GeLU(0) = {gelu(a + b):.3f}  -> not the same, so the path terms of an MLP cannot be split apart')

# ---------------------------------------------------------------- B
log('\n== B. TWO FEATURES IN ONE DIMENSION: W = [1, -1], b = [0, 0]')
W = np.array([[1.0, -1.0]])                  # 1 x 2: feature 1 points up, feature 2 points down
R['B'] = {}
for x in [(0.7, 0.0), (0.0, 0.4), (0.7, 0.4)]:
    x = np.array(x)
    h = W @ x                                # the one hidden number
    lin = W.T @ h                            # linear read-out W^T h
    rel = np.maximum(lin, 0)                 # ReLU read-out
    R['B'][str(L(x))] = dict(h=L(h), linear=L(lin), relu=L(rel))
    log(f'  x = {L(x)}: h = W x = {L(h)[0]}; W^T h = {L(lin)}; ReLU(W^T h) = {L(rel)}')

# ---------------------------------------------------------------- C
log('\n== C. TURNING A PRODUCT AROUND: X Y versus Y X')
X = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]])     # 2 x 3
Y = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])   # 3 x 2
XY, YX = X @ Y, Y @ X
eXY = sorted(np.linalg.eigvals(XY).real.round(6), reverse=True)
eYX = sorted(np.linalg.eigvals(YX).real.round(6), reverse=True)
R['C'] = dict(X=L(X), Y=L(Y), XY=L(XY), YX=L(YX), trXY=float(np.trace(XY)), trYX=float(np.trace(YX)),
              eigXY=L(eXY), eigYX=L(eYX), fXY=float(np.linalg.norm(XY)), fYX=float(np.linalg.norm(YX)))
log(f'  X Y = {L(XY)} (2 x 2); Y X = {L(YX)} (3 x 3)')
log(f'  trace: {np.trace(XY):.0f} and {np.trace(YX):.0f}; eigenvalues: {L(eXY)} and {L(eYX)}')
log(f'  Frobenius norm: |X Y| = sqrt({np.sum(XY ** 2):.0f}) = {np.linalg.norm(XY):.3f}; '
    f'|Y X| = sqrt({np.sum(YX ** 2):.0f}) = {np.linalg.norm(YX):.3f}  -> different')

# ---------------------------------------------------------------- D
log('\n== D. ONE SPARSE AUTOENCODER FEATURE BY HAND (2 dimensions, b_dec = 0)')
d = np.array([0.6, 0.8])                      # decoder direction, length 1
e, be, lam = np.array([0.6, 0.8]), -0.1, 0.5  # encoder row, encoder bias, L1 weight
R['D'] = {}
for x in [0.5 * d, np.array([0.8, -0.6]) * 0.05]:
    pre = float(e @ x) + be
    f = max(pre, 0.0)
    xh = f * d
    mse = float(np.sum((x - xh) ** 2))
    R['D'][str(L(x))] = dict(pre=round(pre, 3), f=round(f, 3), xhat=L(xh), mse=round(mse, 4), l1=round(lam * f, 3))
    log(f'  x = {L(x)}: e . x + b_enc = {pre:.3f}; f = ReLU(...) = {f:.3f}; x_hat = f d = {L(xh)}; '
        f'|x - x_hat|^2 = {mse:.4f}; L1 part = {lam} x {f:.3f} = {lam * f:.3f}')

# ---------------------------------------------------------------- E
log('\n== E. IN-CONTEXT LEARNING SCORE OF THREE MADE-UP WINDOWS')
l50, l500 = np.array([5.0, 4.0, 6.0]), np.array([4.5, 3.2, 5.8])
R['E'] = dict(l50=L(l50), l500=L(l500), m50=float(l50.mean()), m500=float(l500.mean()), score=float(l500.mean() - l50.mean()))
log(f'  loss at token 50: {L(l50)} (mean {l50.mean():.2f}); at token 500: {L(l500)} (mean {l500.mean():.2f})')
log(f'  score = {l500.mean():.2f} - {l50.mean():.2f} = {l500.mean() - l50.mean():+.2f} nats')
save('part7_toy', R)
