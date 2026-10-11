"""Chapter 8: read results/ch8_eval.json and print the comparisons used in the text (no model is run here):
proxy against gold per checkpoint with bootstrap intervals, the EOS-rule proxy, length effects, and side-by-side replies."""
import json
import numpy as np
from common import Log

log = Log('ch8_analysis')
E = json.load(open('results/ch8_eval.json'))
gm, gs = E['gold_norm']['mean'], E['gold_norm']['sd']
rng = np.random.default_rng(0)


def stats(r):
    p = np.array(r['proxy']); g = (np.array(r['gold']) - gm) / gs; f = np.array(r['finished'])
    return p, g, f


def boot(x, n=2000):
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


p0, g0, f0 = stats(E['start'])
log(f'{len(p0)} held-out replies per checkpoint (same prompts, same sampling seed)')
for run in ['beta005', 'nokl']:
    for s in sorted(E[run], key=int):
        r = E[run][s]
        p, g, f = stats(r)
        dp, dg = p - p0, g - g0                          # paired by prompt and sample index
        lo, hi = boot(dg)
        plo, phi = boot(dp)
        pe = np.where(f, p, -1.0)
        log(f'{run:8s} step {s:>2s}: proxy change {dp.mean():+.3f} [{plo:+.3f}, {phi:+.3f}]  gold change {dg.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  '
            f'proxy with EOS rule {pe.mean():+.3f} (start {np.where(f0, p0, -1.0).mean():+.3f})  finished {f.mean():.2f} (start {f0.mean():.2f})  '
            f'tokens {np.mean(r["length"]):.1f} (start {np.mean(E["start"]["length"]):.1f})  KL {np.mean(r["kl"]):.2f}')

log('\ncorrelation of score with reply length (tokens), finished replies only')
for name, r in [('start', E['start']), ('beta005 step 20', E['beta005']['20']), ('nokl step 20', E['nokl']['20'])]:
    p, g, f = stats(r)
    L = np.array(r['length'])
    log(f'  {name:16s} proxy {np.corrcoef(p[f], L[f])[0, 1]:+.3f}  gold {np.corrcoef(g[f], L[f])[0, 1]:+.3f}')

log('\nside by side (first reply to each prompt), start vs beta 0.05 step 20 vs no KL step 20')
S, B, N = E['start'], E['beta005']['20'], E['nokl']['20']
order = np.argsort(-(np.array(B['proxy']) - np.array(S['proxy'])))
shown = 0
for i in list(range(0, len(S['texts']), 2))[:6]:
    log(f'\nPROMPT: {S["prompts"][i][:200]!r}')
    for nm, r in [('start', S), ('beta 0.05', B), ('no KL', N)]:
        log(f'  [{nm:9s}] proxy {r["proxy"][i]:+.2f} gold {(r["gold"][i] - gm) / gs:+.2f} tokens {r["length"][i]:3d} '
            f'{"" if r["finished"][i] else "(cut off) "}{r["texts"][i][:400]!r}')
