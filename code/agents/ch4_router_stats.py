"""Chapter 4: the cascade's confidence check read as a verifier, from the SAVED results of ch4_router.py (no API calls).
The check is "two cheap samples agree". Accept = agree. Among the cheap model's first answers:
  r = P(accept | cheap answer right), f = P(accept | cheap answer wrong) = the false-accept rate,
  precision = P(cheap answer right | accepted) = p r / (p r + (1 - p) f), with p = P(cheap answer right)."""
import json
from common import Log

log = Log('ch4_router_stats')
R = json.load(open('results/ch4_router.json'))
rows = R['rows']
right = [r for r in rows if r['cheap_ok']]
wrong = [r for r in rows if not r['cheap_ok']]
p = len(right) / len(rows)
r_ = sum(r['agree'] for r in right) / len(right)
f = sum(r['agree'] for r in wrong) / len(wrong)
acc = [r for r in rows if r['agree']]
prec = sum(r['cheap_ok'] for r in acc) / len(acc)
log('== The cascade check (two cheap samples agree) as a verifier, from results/ch4_router.json ==')
log(f'p = P(cheap right) = {len(right)}/{len(rows)} = {p:.0%}')
log(f'r = P(agree | right) = {sum(r["agree"] for r in right)}/{len(right)} = {r_:.0%}   (right answers the check let through)')
log(f'f = P(agree | wrong) = {sum(r["agree"] for r in wrong)}/{len(wrong)} = {f:.0%}   (the false-accept rate: wrong answers the check let through)')
log(f'precision = P(right | agree) = {sum(r["cheap_ok"] for r in acc)}/{len(acc)} = {prec:.0%};  Bayes: p r / (p r + (1-p) f) = {p * r_ / (p * r_ + (1 - p) * f):.0%}')
for r in wrong:
    if r['agree']:
        log(f'  accepted but wrong: {r["q"][:60]!r}  both cheap samples said {r["cheap"]!r}, gold {r["gold"]}')
log('reading: agreement between two samples of the same model is a weak verifier: when the model is confidently wrong, it is wrong twice.')
