"""Chapter 5, Section 5.5: a second look at the saved hand-off run (results/ch5b_handoff.json). No API calls.
1. Which KIND of fact disappears: for each hop, does the exact gold figure appear anywhere in the text (price "11.40", days "9", minimum
   order "300" or "1,200")? A crude check that ignores which supplier a number is attached to, so it can only over-count; it complements
   the probe, which does attach facts to suppliers.
2. Did the editor obey its 250-word limit?
3. How noisy was the verifier: how many of its flagged "problems" say in their own words that there is no error?"""
import json, re
from common import Log, save

log = Log('ch5b_handoff_stats')
D = json.load(open('results/ch5b_handoff.json'))
F = json.load(open('data/ch5/b/supplier_facts.json'))
ROLES = ['researcher', 'analyst', 'risk reviewer', 'writer', 'editor']


def present(text, f):
    t = text.replace(',', '')
    price = f'{f["price_gbp"]:.2f}' in t
    days = re.search(rf'(?<![\d.]){f["uk_days"]}(?![\d.])\s*(?:-|\s)?(?:calendar\s+|door|day|d\b)', t, re.I) is not None \
        or re.search(rf'days"?\s*:\s*"?{f["uk_days"]}(?![\d.])', t) is not None        # the JSON field "uk_delivery_days": 9
    moq = re.search(rf'(?<![\d.]){f["moq"]}(?![\d.])', t) is not None
    return price, days, moq


out = {}
log('== Share of the 6 suppliers whose exact figure appears somewhere in the hop\'s text (mean of 3 runs) ==')
log(f'{"condition":<12}{"fact":<7}' + ''.join(f'{r[:10]:>12}' for r in ROLES))
for cond in ['free', 'free+ver', 'schema', 'schema+ver']:
    runs = [r for r in D['runs'] if r['cond'] == cond]
    out[cond] = {}
    for k, name in enumerate(['price', 'days', 'moq']):
        vals = []
        for h in range(5):
            vals.append(sum(present(r['hops'][h]['text'], f)[k] for r in runs for f in F) / (6 * len(runs)))
        out[cond][name] = vals
        log(f'{cond:<12}{name:<7}' + ''.join(f'{v:>12.0%}' for v in vals))
log('\n== Editor word counts (limit 250) ==')
for cond in ['free', 'free+ver', 'schema', 'schema+ver']:
    runs = [r for r in D['runs'] if r['cond'] == cond]
    w = [r['hops'][4]['words'] for r in runs]
    memo = []
    for r in runs:
        j = None
        try:
            j = json.loads(re.search(r'\{.*\}', r['hops'][4]['text'], re.S).group(0))
        except Exception:
            pass
        memo.append(len(str(j.get('work', '')).split()) if isinstance(j, dict) else None)
    out[cond]['editor_words'] = w; out[cond]['memo_words'] = memo
    log(f'  {cond:<11} whole hand-off {w} words' + (f'; memo field alone {memo} words' if cond.startswith('schema') else ''))
log('\n== Verifier flags ==')
NOERR = re.compile(r'no (?:factual )?(?:error|issue|problem)s?\b|not (?:a |an )?(?:factual )?error|that is correct|no other factual contradiction|so no issue|no discrepanc', re.I)
for cond in ['free+ver', 'schema+ver']:
    runs = [r for r in D['runs'] if r['cond'] == cond]
    flags = [p for r in runs for h in r['hops'] for p in (h['verifier_problems'] or [])]
    noerr = [p for p in flags if NOERR.search(str(p))]
    hops_flagged = sum(bool(h['verifier_problems']) for r in runs for h in r['hops'])
    out[cond]['flags'] = len(flags); out[cond]['flags_saying_no_error'] = len(noerr); out[cond]['hops_flagged'] = hops_flagged
    log(f'  {cond:<11} {len(flags)} problems flagged over {len(runs) * 5} hops ({hops_flagged} hops redone); '
        f'{len(noerr)} flags call the item correct or not an error')
    if noerr:
        log(f'    e.g. "{str(noerr[0])[:110].replace(chr(0x2014), "-")}..."')
save('ch5b_handoff_stats', out)

log('\n== Schema drift: hops whose supplier table no longer has plain numbers in the three numeric fields ==')
for cond in ['schema', 'schema+ver']:
    runs = [r for r in D['runs'] if r['cond'] == cond]
    drift, example = 0, None
    for r in runs:
        for h in r['hops']:
            try:
                j = json.loads(re.search(r'\{.*\}', h['text'], re.S).group(0))
                ok = all(isinstance(s.get(k), (int, float)) for s in j['suppliers'] for k in ['price_gbp_per_unit_uk', 'uk_delivery_days', 'moq_units'])
            except Exception:
                ok = False
            if not ok:
                drift += 1
                if example is None:
                    example = f'run {r["run"] + 1}, {h["role"]}'
    out[cond]['schema_drift_hops'] = drift
    log(f'  {cond:<11} {drift} of {len(runs) * 5} hops' + (f' (first: {example})' if example else ''))
save('ch5b_handoff_stats', out)
