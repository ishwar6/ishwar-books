"""Chapter 3: length-controlled win rate (Dubois et al. 2024), on simulated judge votes.
Two models of EQUAL quality answer the same 800 instructions; the "verbose" model writes longer answers.
A simulated judge prefers the verbose model's answer with probability
    logistic( 0 (no quality gap) + phi * tanh( (len_m - len_b) / std ) + noise per instruction )
so its raw win rate is above 50% for length alone. We then fit the length-controlled regression
    q(y = 1) = logistic( theta + phi * tanh(dlen / std) )
(the paper's third term, a per-instruction difficulty gamma_x, is left out here: the simulation draws it at random and
the fit treats it as noise) and report the length-controlled win rate logistic(theta): the same regression with the
length term set to 0.
A third model is genuinely better (theta = 0.5) but not longer, to show that LC keeps real quality differences."""
import numpy as np
from common import Log, save

log = Log('ch3_lc')
rng = np.random.default_rng(0)
N = 800
PHI_TRUE = 1.2


def simulate(theta_true, extra_len_mean):
    len_b = rng.normal(250, 60, N).clip(40)                     # baseline answer lengths (words)
    len_m = (len_b + rng.normal(extra_len_mean, 100, N)).clip(40)
    d = len_m - len_b
    gx = rng.normal(0, 0.5, N)                                   # per-instruction effect
    p = 1 / (1 + np.exp(-(theta_true + PHI_TRUE * np.tanh(d / d.std()) + gx)))
    y = (rng.random(N) < p).astype(float)
    return d, y


def fit(d, y, iters=4000, lr=0.5):
    """Logistic regression on [1, tanh(d / std(d))] by gradient descent (the instruction term is absorbed in the noise)."""
    X = np.stack([np.ones_like(d), np.tanh(d / d.std())], 1)
    w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        w -= lr * X.T @ (p - y) / len(y)
    return w


res = {}
for name, theta, extra in [('verbose, same quality', 0.0, 100.0), ('better, same length', 0.5, 0.0), ('baseline vs itself', 0.0, 0.0)]:
    d, y = simulate(theta, extra)
    th, ph = fit(d, y)
    raw = y.mean()
    lc = 1 / (1 + np.exp(-th))
    res[name] = dict(raw=raw, lc=lc, theta=th, phi=ph, mean_extra_words=float(d.mean()))
    if name.startswith('verbose'):
        keep = (d, th, ph)
    log(f'{name:24s} mean length difference {d.mean():+6.0f} words | raw win rate {100 * raw:5.1f}% | '
        f'fit: theta {th:+.3f}, phi {ph:+.3f} | length-controlled win rate logistic(theta) = {100 * lc:5.1f}%')
d, th, ph = keep
i = int(np.argmin(abs(d / d.std() - 1.0)))                     # an instruction about one std longer
t = np.tanh(d[i] / d.std())
log('')
log(f'worked example, one instruction of the verbose model: length difference {d[i]:.0f} words, std {d.std():.0f}')
log(f'  tanh({d[i]:.0f} / {d.std():.0f}) = tanh({d[i] / d.std():.2f}) = {t:.3f}')
log(f'  with length:    logistic({th:+.3f} + {ph:.3f} x {t:.3f}) = {1 / (1 + np.exp(-(th + ph * t))):.3f}')
log(f'  length removed: logistic({th:+.3f}) = {1 / (1 + np.exp(-th)):.3f}')
res['example'] = dict(d=float(d[i]), std=float(d.std()), tanh=float(t), theta=float(th), phi=float(ph))
save('ch3_lc', res)
