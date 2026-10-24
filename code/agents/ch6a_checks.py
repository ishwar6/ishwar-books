"""Chapter 6 (first half): code-based pass/fail checks on the support agent's saved traces, with confidence intervals.
No model calls: every check is plain code over the trace (the final JSON, the tool calls and their arguments, the reply text).
For the v1 run it also compares the code verdict with the author's hand labels (data/ch6/a/labels_v1.json).
Usage: python ch6a_checks.py   ->  results/ch6a_checks.json and results/ch6a_checks_stdout.txt"""
import json, math, os, re
from common import Log, save

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'data', 'ch6', 'a')
R = os.path.join(HERE, 'results')
ORDERS = {o['order_id']: o for o in json.load(open(os.path.join(D, 'orders.json')))}
MSGS = {m['id']: m for m in json.load(open(os.path.join(D, 'messages.json')))}
ACTIONS = {'refund', 'escalate', 'decline', 'answer', 'ask_info'}
CLAIM = re.compile(r"(i(?:'|’)ve|i have|i)\s+(?:processed|issued|refunded|made)|refund (?:has been|was) (?:processed|issued)|refund id|refund reference", re.I)
PROMISE = re.compile(r'\b(will (?:arrive|be delivered|reach you)|arriv\w* (?:on|by) (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow)|'
                     r'delivered (?:by|on) (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow))', re.I)
log = Log('ch6a_checks')


def calls(tr, name):
    return [s for s in tr['steps'] if s.get('tool') == name]


NEGATED = re.compile(r"(can(?:'|’)?t|cannot|not able to|unable to|don(?:'|’)t|do not|won(?:'|’)t) (?:promise|guarantee|give|confirm)", re.I)


def promises_date(reply, revised):
    if not revised:
        return bool(PROMISE.search(reply))                                                               # draft: any match fails
    for sent in re.split(r'(?<=[.!?])\s+', reply):                                                      # revised: ignore a sentence that refuses to promise
        if PROMISE.search(sent) and not NEGATED.search(sent):
            return True
    return False


def check(tr, revised=False):
    """Return {check_name: True/False} for the checks that apply to this message (None = not applicable).
    draft checks: the first version. revised checks: after reading the v2 failures, quotes_order is dropped (no policy asks for it)
    and the date-promise check ignores a sentence that refuses to promise ("I can't promise the exact day ...")."""
    m, f = MSGS[tr['id']], tr['final'] or {}
    reply = f.get('reply') if isinstance(f.get('reply'), str) else ''
    refunds = tr['refunds']
    out = {}
    out['format'] = f.get('action') in ACTIONS and len(reply.strip()) > 0                                   # schema of the final answer
    out['action'] = f.get('action') == m['expected_action']                                                  # exact match on the action
    out['lookup'] = None if not m['order_id'] else any(str(s['args'].get('order_id', '')).upper() == m['order_id'] for s in calls(tr, 'lookup_order'))
    if m['expected_action'] == 'refund':                                                                     # tool-call assertion: one refund, right amount
        out['refund_call'] = len(refunds) == 1 and refunds[0]['order_id'] == m['order_id'] and abs(float(refunds[0]['amount_gbp'] or 0) - m['refund_gbp']) < 0.01
    else:
        out['refund_call'] = len(refunds) == 0                                                               # tool-call assertion: no refund at all
    out['no_false_claim'] = not (CLAIM.search(reply) and not refunds)                                        # says it refunded, but never called the tool
    out['no_date_promise'] = not promises_date(reply, revised)
    out['quotes_order'] = None if (revised or not m['order_id']) else m['order_id'] in reply or m['order_id'][4:] in reply
    return out


def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion k/n (better than p +/- 1.96 SE when n is small or p is near 0 or 1)."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def score(path, revised):
    data = json.load(open(path))
    rows = []
    for tr in data['traces']:
        c = check(tr, revised)
        rows.append({'id': tr['id'], 'kind': MSGS[tr['id']]['kind'], 'checks': c, 'pass': all(v for v in c.values() if v is not None)})
    return data, rows


RUNS = [('v1', 'ch6a_traces_v1.json'), ('v1 repeat', 'ch6a_traces_v1_r2.json'), ('v2', 'ch6a_traces_v2.json'), ('v2 repeat', 'ch6a_traces_v2_r2.json')]
names = ['format', 'action', 'lookup', 'refund_call', 'no_false_claim', 'no_date_promise', 'quotes_order']
log('Code-based checks on the support agent (52 messages). A message passes only if every check that applies to it passes.')
ALL = {}
for revised in (False, True):
    tag = 'revised' if revised else 'draft'
    summary, all_rows = [], {}
    log('')
    log(f'== {tag} checks ==')
    log(f'{"run":<10} ' + ' '.join(f'{n:>15}' for n in names) + f' {"PASS":>7} {"rate":>6}  95% Wilson CI     cost $')
    for label, fn in RUNS:
        p = os.path.join(R, fn)
        if not os.path.exists(p):
            log(f'{label:<10} (no traces file {fn})')
            continue
        data, rows = score(p, revised)
        all_rows[label] = rows
        n, k = len(rows), sum(r['pass'] for r in rows)
        lo, hi = wilson(k, n)
        per = {c: (sum(1 for r in rows if r['checks'][c]), sum(1 for r in rows if r['checks'][c] is not None)) for c in names}
        summary.append({'run': label, 'n': n, 'pass': k, 'rate': k / n, 'ci': [lo, hi], 'per_check': per, 'cost_usd': data['cost_usd'],
                        'calls': data['usage']['calls']})
        log(f'{label:<10} ' + ' '.join(f'{per[c][0]:>11}/{per[c][1]:<3}' for c in names) + f' {k:>4}/{n} {k / n:>6.1%}  [{lo:.1%}, {hi:.1%}]   {data["cost_usd"]:.4f}')
    ALL[tag] = (summary, all_rows)

# agreement of the code verdict with the hand labels, on v1 (the run that was read)
lab = {l['id']: l for l in json.load(open(os.path.join(D, 'labels_v1.json')))['labels']}
AGREE = {}
for tag, (summary, all_rows) in ALL.items():
    if 'v1' not in all_rows:
        continue
    tab = {'both pass': 0, 'both fail': 0, 'code pass, human fail': [], 'code fail, human pass': []}
    for r in all_rows['v1']:
        h = lab[r['id']]['pass']
        if r['pass'] and h: tab['both pass'] += 1
        elif not r['pass'] and not h: tab['both fail'] += 1
        elif r['pass']: tab['code pass, human fail'].append(r['id'])
        else: tab['code fail, human pass'].append(r['id'] + ':' + ','.join(c for c, v in r['checks'].items() if v is False))
    agree = (tab['both pass'] + tab['both fail']) / len(all_rows['v1'])
    AGREE[tag] = {'table': tab, 'agreement': agree}
    log('')
    log(f'v1, {tag} checks against the hand labels: both pass {tab["both pass"]}, both fail {tab["both fail"]}, agreement {agree:.1%}')
    log(f'  code pass but human fail ({len(tab["code pass, human fail"])}): {" ".join(tab["code pass, human fail"])}')
    log(f'  code fail but human pass ({len(tab["code fail, human pass"])}): {" ".join(tab["code fail, human pass"])}')


def diff_ci(a, b):
    """Difference of two proportions with a normal-approximation 95% interval, treating the two runs as independent samples."""
    pa, pb = a['rate'], b['rate']
    se = math.sqrt(pa * (1 - pa) / a['n'] + pb * (1 - pb) / b['n'])
    return pb - pa, pb - pa - 1.96 * se, pb - pa + 1.96 * se, se


summary, all_rows = ALL['revised']
S = {s['run']: s for s in summary}
log('')
log('Revised checks: changes between runs (independent-samples interval)')
diffs = {}
for a, b in [('v1', 'v1 repeat'), ('v2', 'v2 repeat'), ('v1', 'v2'), ('v1 repeat', 'v2 repeat')]:
    if a in S and b in S:
        d, lo, hi, se = diff_ci(S[a], S[b])
        diffs[f'{a} -> {b}'] = {'diff': d, 'ci': [lo, hi], 'se': se}
        log(f'{a:>9} -> {b:<9} change {d:+.1%}  (SE {se:.1%}, 95% CI [{lo:+.1%}, {hi:+.1%}])')
# paired view: the same 52 messages under v1 and v2; only the discordant messages carry information
P = {r['id']: r['pass'] for r in all_rows['v1']}
Q = {r['id']: r['pass'] for r in all_rows['v2']}
b_ = sum(1 for k in P if P[k] and not Q[k]); c_ = sum(1 for k in P if Q[k] and not P[k])
flips_v1 = sum(1 for r1, r2 in zip(all_rows['v1'], all_rows['v1 repeat']) if r1['pass'] != r2['pass'])
flips_v2 = sum(1 for r1, r2 in zip(all_rows['v2'], all_rows['v2 repeat']) if r1['pass'] != r2['pass'])
mcn = (abs(b_ - c_) - 1) ** 2 / (b_ + c_) if b_ + c_ else 0.0
log(f'paired v1 -> v2: {c_} messages fail on v1 and pass on v2, {b_} pass on v1 and fail on v2; McNemar chi-square (corrected) {mcn:.2f} '
    f'(3.84 needed for p < 0.05)')
log(f'messages whose verdict flipped between two runs of the SAME prompt: v1 {flips_v1} of 52, v2 {flips_v2} of 52')
log('Same prompt, same messages, two runs: the change between repeats is the noise floor. A prompt change must clear it.')
save('ch6a_checks', {'summary': {t: v[0] for t, v in ALL.items()}, 'diffs': diffs, 'agreement_v1': AGREE,
                     'paired_v1_v2': {'v1_only': b_, 'v2_only': c_, 'mcnemar': mcn}, 'flips': {'v1': flips_v1, 'v2': flips_v2},
                     'rows': {t: v[1] for t, v in ALL.items()}})
