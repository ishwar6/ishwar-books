"""Chapter 6 (first half): count the failure categories from error analysis of the v1 traces.
The labels (data/ch6/a/labels_v1.json) were written by reading every trace: a free-text note per trace (open coding), then a
small set of categories that group the notes (axial coding). This script only counts them and adds facts read from the traces.
Usage: python ch6a_errors.py   ->  results/ch6a_errors.json and results/ch6a_errors_stdout.txt"""
import json, os
from collections import Counter, defaultdict
from common import Log, save

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'data', 'ch6', 'a')
MSGS = {m['id']: m for m in json.load(open(os.path.join(D, 'messages.json')))}
LABELS = json.load(open(os.path.join(D, 'labels_v1.json')))['labels']
TRACES = {t['id']: t for t in json.load(open(os.path.join(HERE, 'results', 'ch6a_traces_v1.json')))['traces']}
log = Log('ch6a_errors')

n = len(LABELS)
fails = [l for l in LABELS if not l['pass']]
log(f'v1 traces read: {n}; judged acceptable to send: {n - len(fails)}; failed: {len(fails)}')
log('')
log('Open codes (one note per failed trace), first eight:')
for l in fails[:8]:
    log(f'  {l["id"]}  {l["note"]}')
log('')
cats = Counter(l['category'] for l in fails)
by_expected = defaultdict(Counter)
for l in fails:
    by_expected[l['category']][MSGS[l['id']]['expected_action']] += 1
log(f'{"axial category":<50} {"count":>5} {"share of fails":>15}   right action would have been')
for c, k in cats.most_common():
    log(f'{c:<50} {k:>5} {k / len(fails):>15.0%}   {dict(by_expected[c])}')
log(f'{"total":<50} {len(fails):>5}')
log('')
used_policy = sum(1 for t in TRACES.values() if any(s.get('tool') == 'search_policy' for s in t['steps']))
fail_ids = {l['id'] for l in fails}
used_policy_fail = sum(1 for i in fail_ids if any(s.get('tool') == 'search_policy' for s in TRACES[i]['steps']))
bogus = [t['id'] for t in TRACES.values() if any(s.get('tool') not in ('lookup_order', 'search_policy', 'request_refund') and 'tool' in s for s in t['steps'])]
log(f'traces that called search_policy at least once: {used_policy} of {n}; among the failed traces: {used_policy_fail} of {len(fails)}')
log(f'traces that called a tool that does not exist: {len(bogus)} {bogus}')
fail_by_kind = Counter(MSGS[i]['kind'] for i in fail_ids)
all_by_kind = Counter(m['kind'] for m in MSGS.values())
log('')
log('failures by message kind: ' + ', '.join(f'{k} {fail_by_kind.get(k, 0)}/{v}' for k, v in all_by_kind.items()))
save('ch6a_errors', {'n': n, 'fails': len(fails), 'categories': dict(cats), 'by_expected': {k: dict(v) for k, v in by_expected.items()},
                     'used_policy': used_policy, 'used_policy_fail': used_policy_fail, 'bogus_tool': bogus,
                     'fail_by_kind': dict(fail_by_kind), 'all_by_kind': dict(all_by_kind)})
