"""Chapter 3: confidence intervals for the benchmark results, by bootstrap.
The bootstrap: from the n per-question results (1 = right, 0 = wrong), draw n with replacement, take the mean, and repeat
B times. The middle 95% of those means is the 95% confidence interval. For a difference between two models on the
SAME questions (a paired comparison), resample question indices and recompute both models on the same draw."""
import json, math, os
import numpy as np
from common import Log, save

log = Log('ch3_stats')
B = 10000
rng = np.random.default_rng(0)


def boot(x, b=B):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (b, len(x)))
    means = x[idx].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi), means


def paired(x, y, b=B):
    x, y = np.asarray(x, float), np.asarray(y, float)
    idx = rng.integers(0, len(x), (b, len(x)))
    d = x[idx].mean(1) - y[idx].mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(x.mean() - y.mean()), float(lo), float(hi)


res = {}
# ---- the worked example: normal approximation versus bootstrap
M = json.load(open('results/ch3_mmlu.json'))
x = M['models']['base']['letter-5shot']['correct']
n, k = len(x), sum(x)
p = k / n
se = math.sqrt(p * (1 - p) / n)
acc, lo, hi, means = boot(x)
log('== worked example: base model, MMLU 5-shot letter ==')
log(f'{k} right out of {n}:  accuracy p = {k}/{n} = {p:.4f}')
log(f'standard error sqrt(p(1-p)/n) = sqrt({p:.4f} x {1 - p:.4f} / {n}) = {se:.4f}')
log(f'normal-approximation 95% interval: p +- 1.96 x SE = {p:.4f} +- {1.96 * se:.4f} = [{p - 1.96 * se:.4f}, {p + 1.96 * se:.4f}]')
log(f'bootstrap ({B:,} resamples) 95% interval: [{lo:.4f}, {hi:.4f}]')
log(f'first five resampled accuracies: ' + ', '.join(f'{v:.4f}' for v in means[:5]))
res['bootstrap_example'] = dict(B=B, n=n, k=k, acc=acc, lo=lo, hi=hi, se=se, samples=means[:4000].round(5).tolist(),
                                label='base, MMLU 5-shot letter')

# ---- MMLU, every method
log('')
log('== MMLU: accuracy and 95% bootstrap interval ==')
res['mmlu'] = {'n': M['n'], 'base': {}, 'instruct': {}}
for mname, methods in M['models'].items():
    for key, r in methods.items():
        a, l, h, _ = boot(r['correct'])
        res['mmlu'][mname][key] = dict(acc=a, lo=l, hi=h)
        log(f'{mname:8s} {key:18s} {100 * a:5.1f}%  [{100 * l:5.1f}, {100 * h:5.1f}]')
log('')
log('== paired differences (Instruct minus base, same questions) ==')
res['mmlu_diff'] = {}
for key in ['letter-5shot', 'letter-0shot', 'cloze', 'cloze-norm', 'generate']:
    d, l, h = paired(M['models']['instruct'][key]['correct'], M['models']['base'][key]['correct'])
    res['mmlu_diff'][key] = dict(diff=d, lo=l, hi=h)
    log(f'{key:14s} {100 * d:+5.1f} points  [{100 * l:+5.1f}, {100 * h:+5.1f}]' + ('   (interval contains 0)' if l < 0 < h else ''))

log('')
log('== moving the right answer: accuracy at A minus accuracy at D (paired, same questions) ==')
res['position_diff'] = {}
for mname in ['base', 'instruct']:
    d, l, h = paired(M['models'][mname]['answer-at-A']['correct'], M['models'][mname]['answer-at-D']['correct'])
    res['position_diff'][mname] = dict(diff=d, lo=l, hi=h)
    log(f'{mname:8s} {100 * d:+5.1f} points  [{100 * l:+5.1f}, {100 * h:+5.1f}]')

# ---- MMLU, Instruct generating with a 256-token budget
if os.path.exists('results/ch3_mmlu_long.json'):
    a, l, h, _ = boot(json.load(open('results/ch3_mmlu_long.json'))['correct'])
    res['mmlu']['instruct']['generate-256'] = dict(acc=a, lo=l, hi=h)
    log(f'instruct generate-256        {100 * a:5.1f}%  [{100 * l:5.1f}, {100 * h:5.1f}]')

# ---- GSM8K
G = json.load(open('results/ch3_gsm8k.json')) if os.path.exists('results/ch3_gsm8k.json') else {'n': 0, 'models': {}}
log('')
log('== GSM8K: exact match and 95% bootstrap interval ==')
res['gsm8k'] = {'n': G['n']}
for mname, r in G['models'].items():
    res['gsm8k'][mname] = {}
    for kind in ['strict', 'flexible']:
        a, l, h, _ = boot(r[f'{kind}_correct'])
        res['gsm8k'][mname][kind] = dict(acc=a, lo=l, hi=h)
        log(f'{mname:8s} {kind:8s} {100 * a:5.1f}%  [{100 * l:5.1f}, {100 * h:5.1f}]')
if os.path.exists('results/ch3_gsm8k_long.json'):
    a, l, h, _ = boot(json.load(open('results/ch3_gsm8k_long.json'))['strict_correct'])
    res['gsm8k']['instruct']['strict-1024'] = dict(acc=a, lo=l, hi=h)
    log(f'instruct strict, up to 1,024 tokens {100 * a:5.1f}%  [{100 * l:5.1f}, {100 * h:5.1f}]')
if G['models']:
    d, l, h = paired(G['models']['instruct']['strict_correct'], G['models']['base']['flexible_correct'])
    res['gsm8k_diff'] = dict(diff=d, lo=l, hi=h)
    log(f'Instruct (strict, 320 tokens) minus base (flexible): {100 * d:+.1f} points [{100 * l:+.1f}, {100 * h:+.1f}]')
    if os.path.exists('results/ch3_gsm8k_long.json'):
        d, l, h = paired(json.load(open('results/ch3_gsm8k_long.json'))['strict_correct'], G['models']['base']['flexible_correct'])
        res['gsm8k_diff_long'] = dict(diff=d, lo=l, hi=h)
        log(f'Instruct (strict, 1,024 tokens) minus base (flexible): {100 * d:+.1f} points [{100 * l:+.1f}, {100 * h:+.1f}]')

# ---- HumanEval greedy
if os.path.exists('results/ch3_humaneval.json'):
    H = json.load(open('results/ch3_humaneval.json'))
    a, l, h, _ = boot([int(v) for v in H['greedy_pass']])
    res['humaneval'] = dict(acc=a, lo=l, hi=h)
    log('')
    log(f'HumanEval greedy pass@1: {100 * a:.1f}%  [{100 * l:.1f}, {100 * h:.1f}]  (164 problems)')

# ---- how wide is a 95% interval for n questions, at 50% accuracy?
log('')
log('== width of a 95% interval at accuracy 50% (normal approximation: 2 x 1.96 x sqrt(0.25 / n)) ==')
res['width'] = []
for nn in [100, 400, 1000, 1319, 14042]:
    w = 2 * 1.96 * math.sqrt(0.25 / nn)
    res['width'].append(dict(n=nn, width=w))
    log(f'n = {nn:6d}: +- {50 * w:.1f} points')
save('ch3_stats', res)

# ---- best-of-n: how far from the original policy is "keep the best of n"? (Gao et al. 2022, from Stiennon et al. 2020)
log('')
log('== KL distance of best-of-n from the sampling policy: KL = ln n - (n - 1) / n ==')
res['bon_kl'] = {}
for nn in [1, 2, 4, 8, 16, 64, 1000]:
    kl = math.log(nn) - (nn - 1) / nn
    res['bon_kl'][nn] = kl
    log(f'n = {nn:5d}: {math.log(nn):.3f} - {(nn - 1) / nn:.3f} = {kl:.3f} nats')
save('ch3_stats', res)
