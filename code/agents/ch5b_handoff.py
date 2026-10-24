"""Chapter 5, Section 5.5: the telephone game. How facts are lost or distorted as work passes along a chain of agents, and what a
structured hand-off schema and a verifier agent change. Real API (gpt-5-mini, reasoning_effort=minimal).

Task: Northwind's supplier question ("which three new kitchenware suppliers for 2027? compare price, delivery time to the UK and risks").
Source: six short supplier pages (data/ch5/b/suppliers/), 30 gold facts (data/ch5/b/supplier_facts.json): for each supplier the price
per unit delivered to the UK, the delivery days to the UK, the minimum order and two risks. The pages also hold distractors (delivery days
to the EU, prices in other currencies) that a careless reader can copy instead of the right figure.

Five agents in a chain. Only the first reads the pages; every later agent sees only the hand-off from the agent before it:
  researcher -> analyst (ranks on price and delivery) -> risk reviewer -> writer (memo for the director) -> editor (memo of at most 250 words)
Four conditions, three runs each:
  free      : each hand-off is prose, written for the next colleague.
  free+ver  : prose, plus a verifier agent after every hop that sees the source pages and lists errors and omissions; the hop is redone once.
  schema    : each hand-off is JSON: a supplier table (fixed fields) that later agents copy unchanged, plus a "work" field for their own work.
  schema+ver: the schema, plus the same verifier after every hop.
Measurement: after every hop a separate PROBE call reads only that hop's output and extracts, for each supplier, the four kinds of fact (null
if absent). Each numeric fact is graded kept / lost / distorted against the gold key; each risk kept or lost by keywords. The probe is
calibrated on the source pages themselves (hop 0). Usage: python ch5b_handoff.py"""
import json, os, re
from concurrent.futures import ThreadPoolExecutor
from common import Log, save, HERE
import ch4_llm as llm

log = Log('ch5b_handoff')
llm.set_log(log)
D = os.path.join(HERE, 'data', 'ch5', 'b')
FACTS = json.load(open(os.path.join(D, 'supplier_facts.json')))
PAGES = '\n\n'.join(open(os.path.join(D, 'suppliers', f'{f["id"]}.md')).read().strip() for f in FACTS)
RUNS = 3
TASK = ('Northwind Home, a UK online shop, asks: "Which three new kitchenware suppliers should we add for 2027? Compare price, '
        'delivery time to the UK, minimum order and risks." The buying director wants all six candidate suppliers compared and three recommended.')
ROLES = [
    ('researcher', 'You are the researcher. Read the six supplier pages below and prepare the facts the team needs.'),
    ('analyst', 'You are the analyst. Compare the six suppliers on price and delivery time to the UK, and rank them.'),
    ('risk reviewer', 'You are the risk reviewer. Assess each supplier\'s risks and say which are serious.'),
    ('writer', 'You are the writer. Draft the recommendation memo for the buying director: recommend three suppliers and compare all six.'),
    ('editor', 'You are the editor. Tighten the memo for the buying director to at most 250 words, keeping what the director needs.'),
]
FREE = ('Write your hand-off for the next colleague in prose. The next colleague will see only what you write, not the pages or earlier notes.')
SCHEMA = ('Write your hand-off as JSON only, with exactly two keys. "suppliers": a list of six objects with the keys id, name, price_gbp_per_unit_uk, '
          'uk_delivery_days, moq_units and risks (a list of short strings). "work": a string with your own contribution. '
          'The next colleague will see only this JSON. {copy}')
COPY_FIRST = 'Fill the supplier table from the pages exactly as stated.'
COPY_LATER = ('Copy the supplier table you received unchanged; change a field only to correct a clear error. Put your own work in "work"; '
              'for the editor, "work" is the memo of at most 250 words.')
VERIFY = ('You are a fact checker. Compare the hand-off below with the source pages. List every supplier fact (price per unit delivered to the UK, '
          'delivery days to the UK, minimum order, risks) that is wrong or missing in the hand-off. Reply as JSON: {{"problems": ["..."]}}, '
          'an empty list if there are none.\n\nSOURCE PAGES:\n{pages}\n\nHAND-OFF:\n{text}')
REDO = '\n\nA fact checker who compared your hand-off with the source pages reported these problems. Fix them and write the hand-off again:\n{problems}'
PROBE = ('Read the text below and extract, for each of these six suppliers: Kestrel Cookware (S1), Lotus Steelworks (S2), Harbin Enamel Co. (S3), '
         'Vardar Kitchen (S4), Brightline Housewares (S5), Oakmere Forge (S6), the facts the text states. Use only the text. Reply as JSON: '
         '{{"S1": {{"price_gbp_per_unit_uk": number or null, "uk_delivery_days": number or null, "moq_units": number or null, "risks": [strings]}}, ...}} '
         'Use null or an empty list when the text does not state it.\n\nTEXT:\n{text}')
RISK_KEYS = {'S1': [['kiln'], ['single factory', 'one factory', 'family']], 'S2': [['iso', 'certif'], ['late']],
             'S3': [['lead', 'migration', 'coating'], ['june']], 'S4': [['dispute', 'distributor', 'contract', 'legal'], ['ukca']],
             'S5': [['typhoon', 'flood'], ['owner']], 'S6': [['capacity', '4,000', '4000'], ['surcharge', 'steel']]}
DISTRACT = {'S1': {'days': 4, 'price': 13.30}, 'S2': {'days': 29, 'price': 830}, 'S3': {'days': 38, 'price': 7.70},
            'S4': {'days': 6, 'price': 11.60}, 'S5': {'days': 33, 'price': 10.45}, 'S6': {'days': 8, 'price': None}}


def num(x):
    try:
        return float(str(x).replace(',', '').replace('£', '').split()[0])
    except (ValueError, IndexError, TypeError):
        return None


def grade(probe):
    """Compare a probe's extraction with the gold key. Returns counts and a list of distortions."""
    c = {'kept': 0, 'lost': 0, 'distorted': 0, 'distractor': 0}
    bad = []
    for f in FACTS:
        p = (probe or {}).get(f['id']) or {}
        for key, gold, kind in [('price_gbp_per_unit_uk', f['price_gbp'], 'price'), ('uk_delivery_days', f['uk_days'], 'days'), ('moq_units', f['moq'], 'moq')]:
            v = num(p.get(key))
            if v is None:
                c['lost'] += 1
            elif abs(v - gold) < 0.006:
                c['kept'] += 1
            else:
                c['distorted'] += 1
                d = DISTRACT[f['id']].get(kind)
                if d is not None and abs(v - d) < 0.006:
                    c['distractor'] += 1
                bad.append(f'{f["id"]} {kind}: {v:g} (gold {gold:g})')
        risks = ' '.join(str(r) for r in (p.get('risks') or [])).lower()
        for keys in RISK_KEYS[f['id']]:
            if any(k in risks for k in keys):
                c['kept'] += 1
            else:
                c['lost'] += 1
    return c, bad


def probe(text, use):
    t, u = llm.ask(PROBE.format(text=text), max_tokens=2500)
    use.append(('probe', u))
    return llm.parse_json(t)


def chain(cond, run):
    use, hops = [], []
    prev = None
    for h, (role, job) in enumerate(ROLES):
        fmt = SCHEMA.format(copy=COPY_FIRST if h == 0 else COPY_LATER) if cond.startswith('schema') else FREE
        given = f'SUPPLIER PAGES:\n{PAGES}' if h == 0 else f'HAND-OFF YOU RECEIVED:\n{prev}'
        prompt = f'{TASK}\n\n{job}\n{fmt}\n\n{given}'
        text, u = llm.ask(prompt, max_tokens=4000)
        use.append((role, u))
        problems = None
        if cond.endswith('+ver'):
            vt, vu = llm.ask(VERIFY.format(pages=PAGES, text=text), max_tokens=2000)
            use.append(('verifier', vu))
            problems = (llm.parse_json(vt) or {}).get('problems') or []
            if problems:
                text, u = llm.ask(prompt + REDO.format(problems='\n'.join(f'- {p}' for p in problems)), max_tokens=4000)
                use.append((role + ' (redo)', u))
        pr = probe(text, use)
        c, bad = grade(pr)
        hops.append({'role': role, 'words': len(text.split()), 'counts': c, 'distortions': bad, 'verifier_problems': problems, 'text': text})
        prev = text
    return {'cond': cond, 'run': run, 'hops': hops, 'cost': sum(llm.cost(u) for _, u in use),
            'calls': len(use), 'tokens': sum(u['input'] + u['output'] for _, u in use),
            'cost_ex_probe': sum(llm.cost(u) for r, u in use if r != 'probe')}


before = llm.snapshot()
log(f'== The telephone game: 5 agents in a chain, 30 gold facts, {llm.MODEL}, reasoning_effort=minimal ==')
cal_use = []
cal, cal_bad = grade(probe(PAGES, cal_use))
log(f'probe calibration on the source pages themselves: kept {cal["kept"]}/30, lost {cal["lost"]}, distorted {cal["distorted"]} {cal_bad}')
CONDS = ['free', 'free+ver', 'schema', 'schema+ver']
jobs = [(c, r) for c in CONDS for r in range(RUNS)]
with ThreadPoolExecutor(max_workers=12) as ex:
    res = list(ex.map(lambda j: chain(*j), jobs))
for r in res:
    log(f'\n-- {r["cond"]}, run {r["run"] + 1}: {r["calls"]} calls, {r["tokens"]:,} tokens, ${r["cost"]:.4f} --')
    for h in r['hops']:
        c = h['counts']
        v = '' if h['verifier_problems'] is None else f'  verifier flagged {len(h["verifier_problems"])}'
        log(f'  {h["role"]:<14} {h["words"]:>4} words  kept {c["kept"]:>2}/30  lost {c["lost"]:>2}  distorted {c["distorted"]}{v}'
            + (f'  e.g. {h["distortions"][0]}' if h['distortions'] else ''))

summary = {}
log('\n== Facts kept after each hop (mean of 3 runs, out of 30) ==')
log(f'{"condition":<12}' + ''.join(f'{r[:10]:>12}' for r, _ in ROLES) + f'{"distorted@5":>13}{"cost/run":>10}')
for c in CONDS:
    rs = [r for r in res if r['cond'] == c]
    kept = [sum(r['hops'][h]['counts']['kept'] for r in rs) / RUNS for h in range(5)]
    dist = [sum(r['hops'][h]['counts']['distorted'] for r in rs) / RUNS for h in range(5)]
    lost = [sum(r['hops'][h]['counts']['lost'] for r in rs) / RUNS for h in range(5)]
    words = [sum(r['hops'][h]['words'] for r in rs) / RUNS for h in range(5)]
    cost = sum(r['cost_ex_probe'] for r in rs) / RUNS
    per_hop = (kept[-1] / 30) ** (1 / 5)
    summary[c] = {'kept': kept, 'distorted': dist, 'lost': lost, 'words': words, 'cost_per_run_ex_probe': cost, 'per_hop_retention': per_hop,
                  'final_kept_runs': [r['hops'][-1]['counts']['kept'] for r in rs],
                  'distractor_total': sum(h['counts']['distractor'] for r in rs for h in r['hops'])}
    log(f'{c:<12}' + ''.join(f'{k:>12.1f}' for k in kept) + f'{dist[-1]:>13.1f}{"$" + format(cost, ".4f"):>10}')
for c in CONDS:
    s = summary[c]
    log(f'  {c}: final hop kept {s["final_kept_runs"]} per run; mean words per hop {[round(w) for w in s["words"]]}; '
        f'implied per-hop retention {s["per_hop_retention"]:.3f}; distortions that copied a distractor: {s["distractor_total"]}')
spent = llm.diff(before, llm.snapshot())
log(f'\nAPI: {spent["calls"]} calls, {spent["input"]:,} input and {spent["output"]:,} output tokens, ${llm.cost(spent):.3f} (probes included). '
    f'Stub used: {llm.stub_used()}')
save('ch5b_handoff', {'summary': summary, 'calibration': cal, 'runs': res, 'usage': spent, 'cost': llm.cost(spent), 'model': llm.MODEL})
