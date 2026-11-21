"""Part 7: the metrics of §3, written twice and checked. No model is loaded, so no heavy lock is needed.

1. pass@1 as the paper writes it: the mean correctness of k sampled answers.
2. The unbiased pass@k estimator of Chen et al. (2021):
   Method 1, our own product form (term by term, no big numbers);
   Method 2, the combinatorial form with math.comb, and the numpy one-liner of Chen et al., Figure 3;
   a brute-force check that lists every subset; a table for n = 16 samples with c correct.
3. Why the simple guess 1 - (1 - c/n)^k is biased: a simulation with a known true success rate.
4. cons@k (majority vote), with ties: Method 1 our own counting, Method 2 collections.Counter / numpy.unique.
"""
import itertools, math, random
from collections import Counter
import numpy as np
from common import Log, save

log = Log('r1_part7_passk')
R = {}

# ================================================================== 1. pass@1
log('== 1. pass@1 = mean correctness of k sampled answers')


def pass_at_1(p):
    """p: list of 0/1 correctness values p_i, one per sampled answer (length k)."""
    return sum(p) / len(p)                      # (1/k) * sum_i p_i


p = [1, 0, 1, 1, 0, 1, 1, 0]                    # 8 sampled answers to one question, 5 correct
log(f'p_i = {p}  ->  pass@1 = {sum(p)}/{len(p)} = {pass_at_1(p):.3f}')
R['pass1_example'] = pass_at_1(p)

# ================================================================== 2. pass@k, two ways
log()
log('== 2. the unbiased pass@k estimator (Chen et al. 2021)')


def pass_at_k_product(n, c, k):
    """Method 1, our own. n samples, c correct, budget k.
    P(all k picks wrong) = C(n-c, k) / C(n, k) = prod_{i=n-c+1}^{n} (1 - k/i)  (no big numbers)."""
    if n - c < k:                               # fewer than k wrong answers: every k-subset holds a correct one
        return 1.0
    all_wrong = 1.0
    for i in range(n - c + 1, n + 1):           # c factors, one per correct sample
        all_wrong *= 1 - k / i
    return 1 - all_wrong


def pass_at_k_comb(n, c, k):
    """Method 2a: the formula as written, with exact integers from math.comb."""
    return 1 - math.comb(n - c, k) / math.comb(n, k)   # math.comb(a, b) = 0 when b > a


def pass_at_k_numpy(n, c, k):
    """Method 2b: the numpy script of Chen et al., Figure 3."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def pass_at_k_brute(n, c, k):
    """Check: list every way to pick k of the n samples and count the picks that hold a correct one."""
    samples = [1] * c + [0] * (n - c)
    picks = list(itertools.combinations(range(n), k))
    good = sum(any(samples[j] for j in pick) for pick in picks)
    return good / len(picks), good, len(picks)


n, c, k = 5, 2, 2
b, good, total = pass_at_k_brute(n, c, k)
log(f'worked example: n={n} samples, c={c} correct, k={k}')
log(f'  all {total} pairs listed, {good} hold a correct answer  ->  {good}/{total} = {b:.3f}')
log(f'  formula 1 - C({n - c},{k})/C({n},{k}) = 1 - {math.comb(n - c, k)}/{math.comb(n, k)} = {pass_at_k_comb(n, c, k):.3f}')
log(f'  product 1 - (1-{k}/4)(1-{k}/5) = 1 - {(1 - k / 4):.2f}*{(1 - k / 5):.2f} = {pass_at_k_product(n, c, k):.3f}')

worst = 0.0
for n in range(1, 41):
    for c in range(n + 1):
        for k in range(1, n + 1):
            a, b2, b3 = pass_at_k_product(n, c, k), pass_at_k_comb(n, c, k), pass_at_k_numpy(n, c, k)
            worst = max(worst, abs(a - b2), abs(a - b3))
            if n <= 10:
                worst = max(worst, abs(a - pass_at_k_brute(n, c, k)[0]))
log(f'all n <= 40, every c and k: max |product - comb| or |product - numpy| = {worst:.1e} (brute force for n <= 10)')
R['agree_max_diff'] = worst

big = pass_at_k_comb(200, 3, 100)
log(f'Chen et al. use n=200: C(200,100) has {len(str(math.comb(200, 100)))} digits; '
    f'pass@100 with c=3: comb {big:.6f}, product {pass_at_k_product(200, 3, 100):.6f}')

log()
log('n = 16 samples per question, c correct: pass@k')
ks = [1, 2, 4, 8, 16]
log('   c | ' + ' '.join(f'pass@{k:<3}' for k in ks))
table = {}
for c in [0, 1, 2, 4, 8, 12, 16]:
    row = [pass_at_k_product(16, c, k) for k in ks]
    table[c] = row
    log(f'  {c:2d} | ' + ' '.join(f'{v:8.3f}' for v in row))
R['table_n16'] = {str(c): v for c, v in table.items()}
R['curve_n16'] = {str(c): [pass_at_k_product(16, c, k) for k in range(1, 17)] for c in [1, 2, 4, 8]}

# ================================================================== 3. the biased shortcut
log()
log('== 3. the shortcut 1 - (1 - c/n)^k is biased: simulation')
rng = np.random.default_rng(0)
q, n, k, trials = 0.1, 16, 8, 200_000          # true chance that one sample is right
truth = 1 - (1 - q) ** k
cs = rng.binomial(n, q, size=trials)           # how many of the 16 samples are right, in each trial
unb = np.array([pass_at_k_product(n, int(c), k) for c in range(n + 1)])[cs]
naive = 1 - (1 - cs / n) ** k
log(f'true success rate per sample q = {q}, n = {n}, k = {k}, {trials:,} simulated questions')
log(f'  true pass@{k} = 1 - (1-q)^{k} = {truth:.4f}')
log(f'  mean of unbiased estimator     = {unb.mean():.4f}')
log(f'  mean of 1 - (1 - c/n)^k        = {naive.mean():.4f}')
R['bias'] = dict(q=q, n=n, k=k, truth=truth, unbiased=float(unb.mean()), naive=float(naive.mean()))

# ================================================================== 4. cons@k
log()
log('== 4. cons@k: majority vote over k sampled answers')


def cons_at_k_own(answers, gold, ties='fraction'):
    """Method 1, our own. answers: k final answers (None = no answer). Count each answer, take the most frequent.
    ties='fraction': a tie between t answers gives 1/t credit if gold is among them; 'first': the earliest wins."""
    counts, first_seen = {}, {}
    for i, a in enumerate(answers):
        if a is None:
            continue                            # an answer that never finished does not vote
        counts[a] = counts.get(a, 0) + 1
        first_seen.setdefault(a, i)
    if not counts:
        return 0.0
    top = max(counts.values())
    winners = [a for a in counts if counts[a] == top]
    if ties == 'first':
        return float(min(winners, key=lambda a: first_seen[a]) == gold)
    return (gold in winners) / len(winners)


def cons_at_k_lib(answers, gold, ties='fraction'):
    """Method 2: collections.Counter (keeps first-seen order on ties) and numpy.unique for the fraction rule."""
    votes = [a for a in answers if a is not None]
    if not votes:
        return 0.0
    if ties == 'first':
        return float(Counter(votes).most_common(1)[0][0] == gold)
    vals, cnt = np.unique(np.array(votes, dtype=object), return_counts=True)
    winners = vals[cnt == cnt.max()]
    return float(gold in winners) / len(winners)


cases = [
    ('plurality right', ['12', '7', '12', '9', '12', '7', '3', '12'], '12'),
    ('plurality wrong', ['7', '12', '7', '7', '12', '9', '7', '3'], '12'),
    ('right but only 3 of 8', ['12', '5', '12', '7', '9', '12', '4', '6'], '12'),
    ('tie 2-2', ['12', '7', '7', '12'], '12'),
    ('tie 2-2, wrong first', ['7', '12', '7', '12'], '12'),
    ('unfinished answers', [None, None, '12', '7', None, '12', None, None], '12'),
]
for name, ans, gold in cases:
    p1 = sum(a == gold for a in ans) / len(ans)
    f1, f2 = cons_at_k_own(ans, gold), cons_at_k_lib(ans, gold)
    t1, t2 = cons_at_k_own(ans, gold, 'first'), cons_at_k_lib(ans, gold, 'first')
    assert f1 == f2 and t1 == t2
    log(f'  {name:22s} pass@1 {p1:.3f}  cons (tie=fraction) {f1:.2f}  cons (tie=first) {t1:.0f}   own == lib: True')
R['cons_cases'] = len(cases)

save('r1_part7_passk', R)
log()
log('saved results/r1_part7_passk.json')
