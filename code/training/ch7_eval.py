"""Chapter 7: how good is the reward model we trained, and what has it learned besides quality?
  1. held-out accuracy with a bootstrap interval, and by how far apart the two ratings are (as in Llama 2 Table 8)
  2. calibration: when it says 80%, is it right 80% of the time?
  3. reward distributions for chosen and rejected answers
  4. length: does a longer answer get a higher reward? (and do the labels themselves prefer length?)
  5. probes: append a polite closing line, repeat the answer, cut it in half (our RM and the gold RM)
  6. the other reward models: margin loss, two more seeds, an ensemble, a small RM; the gold RM on the same test set"""
import math, random
import numpy as np
import torch
from common import Log, save
from ch7_common import load_pairs, load_many, score_ids, score_texts, load_gold, score_gold, gap_bucket, GAP_NAMES, tok

log = Log('ch7_eval')
test = load_pairs('test_prefs', 600, seed=123)
gaps = np.array([p['sc'] - p['sr'] for p in test])
lc, lr = np.array([p['ntok_c'] for p in test]), np.array([p['ntok_r'] for p in test])
NAMES = ['main', 'lr1', 'margin', 'seed1', 'seed2', 'small']
m = load_many(NAMES)
S = {}
for n in NAMES:
    m.set_adapter(n)
    S[n] = (np.array(score_ids(m, [p['ids_c'] for p in test])), np.array(score_ids(m, [p['ids_r'] for p in test])))
rc, rr = S['main']
d = rc - rr
correct = d > 0
R = {'example': dict(rc=float(rc[544]), rr=float(rr[544]))}       # the pair shown in the chapter (ch7_data.py)


def boot(x, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


log(f'example pair #544: r(chosen) {rc[544]:+.3f}, r(rejected) {rr[544]:+.3f}')
log('== 1. accuracy on 600 held-out pairs ==')
lo, hi = boot(correct.astype(float))
log(f'main RM: accuracy {correct.mean():.3f} (95% bootstrap interval {lo:.3f} to {hi:.3f}); mean loss '
    f'{np.mean(np.log1p(np.exp(-d))):.4f}')
R['acc'] = dict(acc=float(correct.mean()), ci=[lo, hi])
log('by rating gap (chosen rating minus rejected rating):')
R['by_gap'] = {}
for b, nm in enumerate(GAP_NAMES):
    sel = np.array([gap_bucket(g) == b for g in gaps])
    log(f'  {nm:21s} n={sel.sum():3d}  ' + '  '.join(f'{n} {np.mean((S[n][0] - S[n][1])[sel] > 0):.3f}' for n in NAMES))
    R['by_gap'][nm] = dict(n=int(sel.sum()), **{n: float(np.mean((S[n][0] - S[n][1])[sel] > 0)) for n in NAMES})

log('\n== 2. calibration ==')
p = 1 / (1 + np.exp(-d))
conf = np.maximum(p, 1 - p)
edges = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
ece, R['calib'] = 0.0, []
for a, b in zip(edges[:-1], edges[1:]):
    sel = (conf >= a) & (conf < b)
    if sel.sum() == 0:
        continue
    acc_b, conf_b = correct[sel].mean(), conf[sel].mean()
    ece += sel.mean() * abs(acc_b - conf_b)
    R['calib'].append(dict(lo=a, hi=min(b, 1.0), n=int(sel.sum()), conf=float(conf_b), acc=float(acc_b)))
    log(f'  confidence {a:.1f} to {min(b, 1):.1f}: n={sel.sum():3d}  mean confidence {conf_b:.3f}  accuracy {acc_b:.3f}')
log(f'expected calibration error (ECE) {ece:.3f}')
R['ece'] = float(ece)

log('\n== 3. reward distributions ==')
log(f'chosen   mean {rc.mean():+.3f} sd {rc.std():.3f} | rejected mean {rr.mean():+.3f} sd {rr.std():.3f} | '
    f'gap mean {d.mean():+.3f} sd {d.std():.3f}')
R['dist'] = dict(rc=rc.tolist(), rr=rr.tolist())

log('\n== 4. length ==')
rank = lambda v: np.argsort(np.argsort(v))
pear = lambda a, b: float(np.corrcoef(a, b)[0, 1])
spear = lambda a, b: pear(rank(a), rank(b))
allr, alll = np.concatenate([rc, rr]), np.concatenate([lc, lr])
allg = np.concatenate([[p['sc'] for p in test], [p['sr'] for p in test]])
log(f'over all 1,200 test answers: corr(reward, length) Pearson {pear(allr, alll):.3f} Spearman {spear(allr, alll):.3f}')
log(f'                             corr(GPT-4 rating, length) Pearson {pear(allg, alll):.3f} Spearman {spear(allg, alll):.3f}')
log(f'within pairs: corr(reward gap, length gap) {spear(d, lc - lr):.3f}; corr(rating gap, length gap) {spear(gaps, lc - lr):.3f}')
longer = lc > lr
log(f'accuracy when the chosen answer is longer {correct[longer].mean():.3f} (n={longer.sum()}), '
    f'shorter {correct[lc < lr].mean():.3f} (n={(lc < lr).sum()})')
R['length'] = dict(pearson=pear(allr, alll), spearman=spear(allr, alll), rating_pearson=pear(allg, alll),
                   rating_spearman=spear(allg, alll), pair_spearman=spear(d, lc - lr), acc_longer=float(correct[longer].mean()),
                   acc_shorter=float(correct[lc < lr].mean()), n_longer=int(longer.sum()), r=allr.tolist(), len=alll.tolist())

log('\n== 5. probes on 150 chosen answers ==')
m.set_adapter('main')
probe = [p for p in test if p['words_c'] >= 20][:150]
P_, Y = [p['prompt'] for p in probe], [p['chosen'] for p in probe]
edits = {
    'add a polite closing line': lambda y: y + '\n\nI hope this helps! Let me know if you have any other questions.',
    'repeat the answer twice': lambda y: y + '\n\n' + y,
    'cut the answer in half': lambda y: ' '.join(y.split()[:len(y.split()) // 2]),
}
gt, gm = load_gold()
base_main = np.array(score_texts(m, P_, Y))
base_gold = np.array(score_gold(gt, gm, P_, Y))
R['probes'] = {}
for nm, f in edits.items():
    Y2 = [f(y) for y in Y]
    dm = np.array(score_texts(m, P_, Y2)) - base_main
    dg = np.array(score_gold(gt, gm, P_, Y2)) - base_gold
    log(f'  {nm:26s} our RM: mean change {dm.mean():+.3f} (sd of RM {base_main.std():.2f}), up in {np.mean(dm > 0):.2f} | '
        f'gold RM: mean change {dg.mean():+.3f} (sd {base_gold.std():.2f}), up in {np.mean(dg > 0):.2f}')
    R['probes'][nm] = dict(main=dm.tolist(), gold=dg.tolist(), main_sd=float(base_main.std()), gold_sd=float(base_gold.std()))

log('\n== 6. other reward models ==')
gc = np.array(score_gold(gt, gm, [p['prompt'] for p in test], [p['chosen'] for p in test]))
gr = np.array(score_gold(gt, gm, [p['prompt'] for p in test], [p['rejected'] for p in test]))
S['gold'] = (gc, gr)
R['by_gap_gold'] = {nm: float(np.mean((gc - gr)[np.array([gap_bucket(g) == b for g in gaps])] > 0)) for b, nm in enumerate(GAP_NAMES)}
log('  gold RM by rating gap: ' + ', '.join(f'{k} {v:.3f}' for k, v in R['by_gap_gold'].items()))
for n in NAMES + ['gold']:
    a = np.mean(S[n][0] > S[n][1])
    log(f'  {n:7s} accuracy {a:.3f}  reward sd {np.concatenate(S[n]).std():.3f}')
z = lambda n: (S[n][0] - S[n][1]) / np.concatenate(S[n]).std()      # gap in units of that model's reward sd
ens = (z('main') + z('seed1') + z('seed2')) / 3
agree = (np.sign(z('main')) == np.sign(z('seed1'))) & (np.sign(z('main')) == np.sign(z('seed2')))
log(f'  ensemble (mean of main, seed1, seed2 after scaling each to unit sd) accuracy {np.mean(ens > 0):.3f}')
log(f'  the three agree on {agree.mean():.2f} of pairs: accuracy there {np.mean(ens[agree] > 0):.3f}; '
    f'where they disagree {np.mean(ens[~agree] > 0):.3f}')
corr = {f'{a}-{b}': pear(np.concatenate(S[a]), np.concatenate(S[b])) for a, b in
        [('main', 'seed1'), ('main', 'seed2'), ('seed1', 'seed2'), ('main', 'margin'), ('main', 'gold'), ('small', 'gold')]}
log('  correlation of rewards over the 1,200 answers: ' + ', '.join(f'{k} {v:.2f}' for k, v in corr.items()))
big = {n: float(np.mean(np.abs(S[n][0] - S[n][1]) > 2)) for n in ['main', 'lr1', 'margin']}
log('  share of pairs with |reward gap| > 2: ' + ', '.join(f'{k} {v:.3f}' for k, v in big.items()))
R['others'] = {n: dict(acc=float(np.mean(S[n][0] > S[n][1])), sd=float(np.concatenate(S[n]).std()),
                       rc=S[n][0].tolist(), rr=S[n][1].tolist()) for n in NAMES + ['gold']}
R['ensemble'] = dict(acc=float(np.mean(ens > 0)), agree=float(agree.mean()), acc_agree=float(np.mean(ens[agree] > 0)),
                     acc_disagree=float(np.mean(ens[~agree] > 0)), corr=corr, big_gap=big)
save('ch7_eval', R)
