"""Part 5 (Limits, and What Came After): the numbers behind the limits of harness search.

1. Noise on 89 tasks: one TerminalBench-2 task is 1.12 points; the standard error of a pass rate, from scratch and
   with scipy; which runs-per-task count fits the percentages of the Appendix A.2 search log.
2. The winner's curse: pick the best of N equal candidates on a small search set and its score looks better than it
   is. Method 1 a pure-Python simulation, Method 2 the exact answer with scipy's binomial distribution.
3. A real run (Qwen2.5-0.5B-Instruct, SST-2): 12 prompt "harnesses" scored on a 60-example search set, a 60-example
   held-out set and a 300-example test set. Label scoring two ways (our logits vs the transformers loss), the search vs
   test gap of the chosen harness, and the rank correlation of search and test (from scratch and with scipy).
4. Self-Harness's acceptance rule (Zhang et al. 2026) from scratch and with numpy, applied to the same 12 harnesses.
5. Cost per step of three objectives from the Recursive Harness Self-Improvement paper: counting pairs by hand
   and with math.comb.
Run: ~/.venvs/attn/bin/python mh_part5.py   (about 2 minutes on an M-series Mac)"""
import math, random
import numpy as np
from scipy import stats
from common import Log, save, timer

log = Log('part5')
elapsed = timer()
out = {}

# ---------------------------------------------------------------------------------------------------------------
log('== 1. Noise on a small benchmark: TerminalBench-2 has 89 tasks')
N_TASKS = 89
log(f'one task = 100 / {N_TASKS} = {100 / N_TASKS:.2f} points')


def pass_rate_se(p, n):
    """Method 1: standard error of a pass rate p measured on n independent tasks, in points."""
    return 100 * math.sqrt(p * (1 - p) / n)


p0 = 0.644                                         # the Terminus-KIRA baseline in the Appendix A.2 search log
se1 = pass_rate_se(p0, N_TASKS)
se2 = 100 * stats.binom(N_TASKS, p0).std() / N_TASKS   # Method 2: the binomial distribution of scipy
log(f'standard error at 64.4% on 89 tasks: from scratch {se1:.2f} points, scipy {se2:.2f} points, '
    f'same: {abs(se1 - se2) < 1e-9}')
log(f'rough 95% range for one run: 64.4 +- {1.96 * se1:.1f} points')
for name, d in [('evo_marker_fix', -5.6), ('evo_single_confirm', -6.7), ('evo_strip_only', -1.1)]:
    log(f'  {name:20s} {d:+.1f} points = {d / (100 / N_TASKS):+.1f} tasks if each task ran once')

# Which number of runs per task r makes every percentage of the log a whole number of passes out of 89*r?
LOG_PCTS = [64.4, 58.9, 57.8, 63.3]
fits = [r for r in range(1, 11)
        if all(any(round(100 * k / (N_TASKS * r), 1) == pct for k in range(N_TASKS * r + 1)) for pct in LOG_PCTS)]
log(f'runs per task (1 to 10) that fit 64.4, 58.9, 57.8 and 63.3 exactly: {fits}')
out['noise'] = dict(one_task=100 / N_TASKS, se=se1, fits=fits)

# ---------------------------------------------------------------------------------------------------------------
log('')
log("== 2. The winner's curse: the best of N equal candidates on a small search set")
TRUE_ACC, N_CAND, N_SEARCH, TRIALS = 0.40, 40, 100, 2000


def best_of_n_sim(true_acc, n_cand, n_search, trials, seed=0):
    """Method 1: simulate. Every candidate has the same true accuracy; we keep the best search score."""
    rng = random.Random(seed)
    best = []
    for _ in range(trials):
        scores = [sum(rng.random() < true_acc for _ in range(n_search)) for _ in range(n_cand)]
        best.append(max(scores) / n_search)
    return 100 * sum(best) / trials


def best_of_n_exact(true_acc, n_cand, n_search):
    """Method 2: exact. P(max >= k) = 1 - F(k-1)^N, and E[max] = sum over k of P(max >= k)."""
    F = stats.binom(n_search, true_acc).cdf(np.arange(n_search))      # F(0) ... F(n-1)
    return 100 * np.sum(1 - F ** n_cand) / n_search


sim = best_of_n_sim(TRUE_ACC, N_CAND, N_SEARCH, TRIALS)
exact = best_of_n_exact(TRUE_ACC, N_CAND, N_SEARCH)
log(f'{N_CAND} candidates, all truly {100 * TRUE_ACC:.0f}% accurate, search set of {N_SEARCH} examples')
log(f'expected best search score: simulation {sim:.2f}%, exact {exact:.2f}%, agree within 0.3: {abs(sim - exact) < 0.3}')
log(f'the winner looks {exact - 100 * TRUE_ACC:.1f} points better than it is')
log('inflation of the best search score (points), exact:')
log('  candidates  ' + ''.join(f'n={n:<7d}' for n in (50, 88, 100, 250)))
table = {}
for nc in (1, 4, 10, 40, 109):
    row = [best_of_n_exact(TRUE_ACC, nc, n) - 100 * TRUE_ACC for n in (50, 88, 100, 250)]
    table[nc] = row
    log(f'  {nc:10d}  ' + ''.join(f'{v:<9.1f}' for v in row))
out['winners_curse'] = dict(sim=sim, exact=exact, table=table)

# ---------------------------------------------------------------------------------------------------------------
log('')
log('== 3. A real run: 12 prompt harnesses around Qwen2.5-0.5B-Instruct on SST-2')
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
log(f'model {MODEL}, device {dev}, float32')

data = load_dataset('nyu-mll/glue', 'sst2')
val = list(data['validation'])
random.Random(0).shuffle(val)
SEARCH, HOLDOUT, TEST = val[:60], val[60:120], val[120:420]
log(f'SST-2 validation split into search {len(SEARCH)}, held-out {len(HOLDOUT)}, test {len(TEST)} (disjoint)')
train = [r for r in data['train'] if len(r['sentence'].split()) <= 20]

LABELS = [' negative', ' positive']                  # label 0, label 1
LABEL_IDS = [tok.encode(w)[0] for w in LABELS]
assert all(len(tok.encode(w)) == 1 for w in LABELS)
TEMPLATES = {'A': 'Review: {s}\nSentiment:', 'B': 'Text: {s}\nIs this text positive or negative? Answer:'}


def shots(seed, k=4):
    rng = random.Random(seed)
    pos = rng.sample([r for r in train if r['label'] == 1], k // 2)
    neg = rng.sample([r for r in train if r['label'] == 0], k // 2)
    ex = pos + neg
    rng.shuffle(ex)
    return ex


def make_harness(template, seed):
    """A 'harness' here is a function from an input sentence to a prompt: a template plus 0 or 4 examples."""
    ex = [] if seed is None else shots(seed)
    head = ''.join(TEMPLATES[template].format(s=r['sentence'].strip()) + LABELS[r['label']] + '\n\n' for r in ex)
    return lambda s: head + TEMPLATES[template].format(s=s.strip())


HARNESSES = {}
for t in 'AB':
    HARNESSES[f'{t}-zero'] = make_harness(t, None)
    for seed in range(1, 6):
        HARNESSES[f'{t}-4shot-s{seed}'] = make_harness(t, seed)


@torch.no_grad()
def scores_method1(prompts, bs=16):
    """Method 1: one forward pass; the log-probability of each label's token at the last position."""
    res = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors='pt', padding=True).to(dev)
        logits = model(**enc).logits[:, -1, :]                         # (batch, vocab)
        logp = torch.log_softmax(logits.float(), dim=-1)[:, LABEL_IDS]  # (batch, 2)
        res.append(logp.cpu())
    return torch.cat(res)


@torch.no_grad()
def logprob_method2(prompt, label_word):
    """Method 2: the library's own loss. With labels on the label token only, loss = -log P(label | prompt)."""
    ids_p = tok(prompt, return_tensors='pt').input_ids
    ids_l = tok(label_word, return_tensors='pt').input_ids
    ids = torch.cat([ids_p, ids_l], dim=1).to(dev)
    labels = ids.clone()
    labels[:, :ids_p.shape[1]] = -100                                  # ignore the prompt positions
    return -model(input_ids=ids, labels=labels).loss.item()


h = HARNESSES['A-4shot-s1']
check_prompts = [h(r['sentence']) for r in SEARCH[:5]]
m1 = scores_method1(check_prompts, bs=5)
m2 = torch.tensor([[logprob_method2(p, w) for w in LABELS] for p in check_prompts])
diff = (m1 - m2).abs().max().item()
log(f'label scoring check on 5 prompts: max difference ours vs library loss = {diff:.2e}, agree: {diff < 1e-3}')


def accuracy(harness, rows):
    lp = scores_method1([harness(r['sentence']) for r in rows])
    pred = lp.argmax(dim=1).tolist()
    return [int(p == r['label']) for p, r in zip(pred, rows)]


results = {}
log(f'{"harness":14s} {"search":>7s} {"held-out":>9s} {"test":>6s}')
for name, h in HARNESSES.items():
    s, ho, te = accuracy(h, SEARCH), accuracy(h, HOLDOUT), accuracy(h, TEST)
    results[name] = dict(search=sum(s), holdout=sum(ho), test=sum(te),
                         search_acc=100 * sum(s) / len(s), holdout_acc=100 * sum(ho) / len(ho),
                         test_acc=100 * sum(te) / len(te))
    r = results[name]
    log(f'{name:14s} {r["search_acc"]:6.1f}% {r["holdout_acc"]:8.1f}% {r["test_acc"]:5.1f}%')

names = list(results)
chosen = max(names, key=lambda n: results[n]['search'])                 # Meta-Harness style: best on the search set
best_test = max(names, key=lambda n: results[n]['test'])
mean_test = sum(results[n]['test_acc'] for n in names) / len(names)
gap = results[chosen]['search_acc'] - results[chosen]['test_acc']
log(f'chosen by search score: {chosen}: search {results[chosen]["search_acc"]:.1f}%, '
    f'test {results[chosen]["test_acc"]:.1f}%, gap {gap:+.1f} points')
log(f'best on test (unknown during search): {best_test} at {results[best_test]["test_acc"]:.1f}%; '
    f'mean test of all 12: {mean_test:.1f}%')


def ranks(xs):
    """Ranks 1..n, ties get the average of their positions."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    """Method 1: Spearman rank correlation = Pearson correlation of the ranks."""
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return cov / math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))


xs = [results[n]['search_acc'] for n in names]
ys = [results[n]['test_acc'] for n in names]
rho1 = spearman(xs, ys)
rho2 = stats.spearmanr(xs, ys).statistic                              # Method 2: scipy
log(f'rank correlation of search and test over 12 harnesses: ours {rho1:.3f}, scipy {rho2:.3f}, '
    f'agree: {abs(rho1 - rho2) < 1e-9}')
se_search = pass_rate_se(results[chosen]['search_acc'] / 100, len(SEARCH))
log(f'standard error of one accuracy on 60 examples near {results[chosen]["search_acc"]:.0f}%: {se_search:.1f} points')
out['real'] = dict(results=results, chosen=chosen, best_test=best_test, mean_test=mean_test, gap=gap,
                   spearman=rho1, label_check=diff)

# ---------------------------------------------------------------------------------------------------------------
log('')
log("== 4. Self-Harness's acceptance rule: improve one split, hurt neither")


def accept(d_in, d_ho):
    """Method 1: the rule of Self-Harness, Section 3.4, for one candidate (pass-count changes)."""
    return d_in >= 0 and d_ho >= 0 and max(d_in, d_ho) > 0


def accept_np(d_in, d_ho):
    """Method 2: the same rule on whole arrays with numpy."""
    d_in, d_ho = np.asarray(d_in), np.asarray(d_ho)
    return (d_in >= 0) & (d_ho >= 0) & (np.maximum(d_in, d_ho) > 0)


grid = [(a, b) for a in range(-2, 3) for b in range(-2, 3)]
same = all(accept(a, b) == bool(v) for (a, b), v in zip(grid, accept_np([g[0] for g in grid], [g[1] for g in grid])))
log(f'rule checked on all 25 pairs of changes from -2 to +2: both methods agree: {same}')
log(f'examples: (+3, 0) -> {accept(3, 0)}, (+5, -1) -> {accept(5, -1)}, (0, 0) -> {accept(0, 0)}')

base = 'A-zero'
log(f'current harness: {base} (search {results[base]["search"]}/60, held-out {results[base]["holdout"]}/60)')
accepted = []
for n in names:
    if n == base:
        continue
    d_in = results[n]['search'] - results[base]['search']
    d_ho = results[n]['holdout'] - results[base]['holdout']
    ok = accept(d_in, d_ho)
    if ok:
        accepted.append((d_in + d_ho, d_ho, n))   # total gain first; on a tie, the larger held-out gain
    log(f'  {n:14s} d_in {d_in:+3d}  d_ho {d_ho:+3d}  -> {"accept" if ok else "reject"}')
log(f'accepted {len(accepted)} of {len(names) - 1}')
if accepted:
    pick = max(accepted)[2]
    log(f'largest accepted gain: {pick}: test {results[pick]["test_acc"]:.1f}% '
        f'(current harness {results[base]["test_acc"]:.1f}%)')
    out['self_harness'] = dict(accepted=[a[2] for a in accepted], pick=pick)

# ---------------------------------------------------------------------------------------------------------------
log('')
log('== 5. Cost per step: compare every pair, or only with the last harness')


def pairs_by_hand(m):
    """Method 1: count the pairwise comparisons in a population of m outputs, one by one."""
    c = 0
    for i in range(m):
        for j in range(i + 1, m):
            c += 1
    return c


for m in (2, 10, 40, 109):
    log(f'  population {m:3d}: {pairs_by_hand(m):5d} pairs by hand, {math.comb(m, 2):5d} with math.comb, '
        f'trajectory-local: 1 pair')
out['pairs'] = {m: math.comb(m, 2) for m in (2, 10, 40, 109)}

save('part5', out)
log('')
log(f'done in {elapsed():.0f} s')
