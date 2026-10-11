"""Chapter 7, overoptimisation experiment, step 2 (after ch7_bon_gen.py): best-of-n against our reward models,
judged by a stronger "gold" reward model, in the style of Gao et al. (2022).

For each prompt we have N = 32 samples. Best-of-n keeps the sample with the highest PROXY reward among n. We report the
expected proxy and gold reward of the kept sample for n = 1, 2, 4, 8, 16, 32 with the unbiased estimator of
Nakano et al. (2021): sort the N samples by proxy reward; the i-th smallest (1-based) is the best of a random n-subset with
probability C(i-1, n-1) / C(N, n). The KL between the best-of-n policy and the original policy is log n - (n-1)/n.
Proxies: main (3,200 pairs), small (400 pairs), the ensemble mean, and the ensemble minimum (worst case)."""
import json, math, os
import numpy as np
from common import Log, save
from ch7_common import CACHE, RM_DIR, load_many, score_texts, load_gold, score_gold

log = Log('ch7_bon')
data = json.load(open(os.path.join(CACHE, 'bon_samples.json')))
P = [d['prompt'] for d in data for _ in d['responses']]
Y = [y for d in data for y in d['responses']]
FIN = [f for d in data for f in d['finished']]
NP, N = len(data), len(data[0]['responses'])
log(f'{NP} prompts x {N} samples = {len(Y)} answers; finished within 256 tokens: {np.mean(FIN):.3f}')

scores_path = os.path.join(CACHE, 'bon_scores.json')
sc = json.load(open(scores_path)) if os.path.exists(scores_path) else {}       # cached scores, one list per model
if 'gold' not in sc:
    gt, gm = load_gold()
    sc['gold'] = score_gold(gt, gm, P, Y)
    json.dump(sc, open(scores_path, 'w'))
    del gm
    log('  scored with gold')
have = lambda n: os.path.exists(os.path.join(RM_DIR if n == 'main' else os.path.join(CACHE, n), 'adapter_model.safetensors'))
todo = [n for n in ['main', 'lr1', 'small', 'seed1', 'seed2', 'margin'] if n not in sc and have(n)]
if todo:
    m = load_many(todo)
    for n in todo:
        m.set_adapter(n)
        sc[n] = score_texts(m, P, Y)
        json.dump(sc, open(scores_path, 'w'))
        log(f'  scored with {n}')
    del m
if os.environ.get('SCORE_ONLY'):
    raise SystemExit
A = {k: np.array(v).reshape(NP, N) for k, v in sc.items()}
ntok = np.array([len(y.split()) for y in Y]).reshape(NP, N)
FINA = np.array(FIN, dtype=float).reshape(NP, N)


def z(x):                       # standardise each reward model over all samples so the scales are comparable
    return (x - x.mean()) / x.std()


Z = {k: z(v) for k, v in A.items()}
Z['ens_mean'] = (Z['main'] + Z['seed1'] + Z['seed2']) / 3
Z['ens_min'] = np.minimum(np.minimum(Z['main'], Z['seed1']), Z['seed2'])
ns = [1, 2, 4, 8, 16, 32]


def bon(proxy, target, n):
    """Expected target value of the sample with the highest proxy among n drawn without replacement from the N."""
    tot = 0.0
    for p, t in zip(proxy, target):
        order = np.argsort(p, kind='stable')
        w = np.array([math.comb(i, n - 1) for i in range(N)], dtype=float) / math.comb(N, n)   # i = number of smaller ones
        tot += float(np.dot(w, t[order]))
    return tot / len(proxy)


R = dict(ns=ns, kl=[math.log(n) - (n - 1) / n for n in ns], curves={})
log(f'{"proxy":9s} ' + ' '.join(f'n={n:<5d}' for n in ns) + '   (rewards in sd units, change from n=1)')
for k in ['main', 'small', 'ens_mean', 'ens_min', 'gold']:
    prox = [bon(Z[k], Z[k], n) for n in ns]
    gold = [bon(Z[k], Z['gold'], n) for n in ns]
    length = [bon(Z[k], ntok, n) for n in ns]
    fin = [bon(Z[k], FINA, n) for n in ns]
    R['curves'][k] = dict(proxy=[v - prox[0] for v in prox], gold=[v - gold[0] for v in gold], words=length, finished=fin)
    log(f'{k:9s} proxy ' + ' '.join(f'{v - prox[0]:+.3f}' for v in prox))
    log(f'{"":9s} gold  ' + ' '.join(f'{v - gold[0]:+.3f}' for v in gold))
    log(f'{"":9s} words ' + ' '.join(f'{v:6.1f}' for v in length))
    log(f'{"":9s} done  ' + ' '.join(f'{v:6.2f}' for v in fin))
log('KL(best-of-n || policy) in nats: ' + ' '.join(f'{v:.3f}' for v in R['kl']))

# robustness: only answers that finished within 256 tokens (a cut-off answer is an easy target for the gold RM)
NF = FINA.sum(1).astype(int)
keep = NF >= 8
log(f'\nfinished answers only: {keep.sum()} prompts with at least 8 finished answers (of 32)')


def bon_masked(proxy, target, n):
    tot = 0.0
    for i in np.where(keep)[0]:
        sel = FINA[i] > 0
        p, t = proxy[i][sel], target[i][sel]
        order = np.argsort(p, kind='stable')
        w = np.array([math.comb(j, n - 1) for j in range(len(p))], dtype=float) / math.comb(len(p), n)
        tot += float(np.dot(w, t[order]))
    return tot / keep.sum()


R['finished_only'] = dict(ns=[1, 2, 4, 8], n_prompts=int(keep.sum()), curves={})
for k in ['main', 'small', 'ens_mean', 'ens_min', 'gold']:
    prox = [bon_masked(Z[k], Z[k], n) for n in [1, 2, 4, 8]]
    gold = [bon_masked(Z[k], Z['gold'], n) for n in [1, 2, 4, 8]]
    words = [bon_masked(Z[k], ntok, n) for n in [1, 2, 4, 8]]
    R['finished_only']['curves'][k] = dict(proxy=[v - prox[0] for v in prox], gold=[v - gold[0] for v in gold], words=words)
    log(f'{k:9s} proxy ' + ' '.join(f'{v - prox[0]:+.3f}' for v in prox) + ' | gold ' + ' '.join(f'{v - gold[0]:+.3f}' for v in gold)
        + ' | words ' + ' '.join(f'{v:.0f}' for v in words))

# Gao et al.'s functional form for best-of-n: R(d) = d (alpha - beta d), d = sqrt(KL), fitted by least squares
for k in ['main', 'small']:
    d = np.sqrt(np.array(R['kl']))
    g = np.array(R['curves'][k]['gold'])
    X = np.stack([d, -d ** 2], 1)
    alpha, beta = np.linalg.lstsq(X, g, rcond=None)[0]
    peak = alpha / (2 * beta) if beta > 0 else float('inf')
    R['curves'][k]['fit'] = dict(alpha=float(alpha), beta=float(beta), d_peak=float(peak))
    if beta > 0 and peak ** 2 < 50:
        log(f'fit gold({k}) = d ({alpha:.3f} - {beta:.3f} d): peak at d = {peak:.2f} (KL {peak ** 2:.2f} nats, n about {math.exp(peak ** 2 + 1):.0f})')
    else:
        log(f'fit gold({k}) = d ({alpha:.3f} - ({beta:.3f}) d): no peak within reach (beta {"too small" if beta > 0 else "negative"})')

# how often does the proxy's favourite differ from the gold's favourite?
for k in ['main', 'small']:
    same = np.mean(np.argmax(A[k], 1) == np.argmax(A['gold'], 1))
    rho = np.mean([np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1] for a, b in zip(A[k], A['gold'])])
    log(f'{k}: picks the gold favourite of 32 in {same:.2f} of prompts; mean within-prompt rank correlation with gold {rho:.3f}')
    R['curves'][k]['same_top'] = float(same); R['curves'][k]['rank_corr'] = float(rho)

# example: a prompt where the proxy's top pick is a gold low
k = 'main'
gap = [(Z['gold'][i].max() - Z['gold'][i][np.argmax(Z[k][i])], i) for i in range(NP)]
_, i = max(gap)
j, jg = int(np.argmax(Z[k][i])), int(np.argmax(Z['gold'][i]))
R['example'] = dict(prompt=data[i]['prompt'], proxy_pick=data[i]['responses'][j], gold_pick=data[i]['responses'][jg],
                    proxy_scores=[float(Z[k][i][j]), float(Z['gold'][i][j])], gold_scores=[float(Z[k][i][jg]), float(Z['gold'][i][jg])])
log(f'\nlargest disagreement, prompt #{i}: {data[i]["prompt"][:120]!r}')
log(f'  proxy pick: proxy {Z[k][i][j]:+.2f}, gold {Z["gold"][i][j]:+.2f}: {data[i]["responses"][j][:200]!r}')
log(f'  gold pick:  proxy {Z[k][i][jg]:+.2f}, gold {Z["gold"][i][jg]:+.2f}: {data[i]["responses"][jg][:200]!r}')
save('ch7_bon', R)
