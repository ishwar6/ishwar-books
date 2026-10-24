"""Chapter 6, Section 6.5: trajectory evals on real runs of a small Northwind support agent. Real model: gpt-5-mini with native
tool calling (the OpenAI client from ch4_llm, key from a .env outside the repo, never printed). Data: data/ch6/b/ (ch6b_data.py).

Five tasks, four trials each = 20 runs. Every run is graded five ways:
  end state : the final database equals the expected database (the tau-bench style check);
  exact     : the tool names equal the reference sequence;
  in-order  : the reference calls appear in order, extra calls allowed;
  any-order : every reference call appears somewhere;
  args      : every write call has the expected arguments;
plus two step-level rules: look the order up before any write, and at most one write per run; and an output check on the reply.
Then pass^k and pass@k per task (the unbiased estimators of Yao et al. 2024) for the end-state check.
Finally three planted trajectories (hand-edited copies of real runs) test that each check catches the failure it is meant to catch.
Writes results/ch6b_trajectory.json and results/ch6b_trajectory_stdout.txt."""
import copy, json, os
from math import comb
import ch4_llm as llm
from common import Log, save

log = Log('ch6b_trajectory')
llm.set_log(log)
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'ch6', 'b')
POLICY = open(os.path.join(D, 'policy.md')).read()
ORDERS = {o['order_id']: o for o in json.load(open(os.path.join(D, 'orders.json')))}
TRIALS = 4

SYSTEM = f"""You are Northwind Home's support agent. Today is 2026-10-20. Use the tools to look up orders and the policy, take the
action the policy allows (at most one action per request), then write a short final reply to the customer.
Never take an action the policy does not allow."""

TOOLS = [
    {'type': 'function', 'function': {'name': 'lookup_order', 'description': 'Get an order record by id.',
     'parameters': {'type': 'object', 'properties': {'order_id': {'type': 'string'}}, 'required': ['order_id']}}},
    {'type': 'function', 'function': {'name': 'search_policy', 'description': 'Search the support policy. Returns the policy text.',
     'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}}},
    {'type': 'function', 'function': {'name': 'create_return', 'description': 'Open a return; the refund is paid when the item arrives.',
     'parameters': {'type': 'object', 'properties': {'order_id': {'type': 'string'}, 'refund_amount': {'type': 'number'}}, 'required': ['order_id', 'refund_amount']}}},
    {'type': 'function', 'function': {'name': 'issue_refund', 'description': 'Refund now, with no return (damaged items only).',
     'parameters': {'type': 'object', 'properties': {'order_id': {'type': 'string'}, 'amount': {'type': 'number'}}, 'required': ['order_id', 'amount']}}},
    {'type': 'function', 'function': {'name': 'cancel_order', 'description': 'Cancel an order that has not shipped.',
     'parameters': {'type': 'object', 'properties': {'order_id': {'type': 'string'}}, 'required': ['order_id']}}},
]
WRITES = {'create_return', 'issue_refund', 'cancel_order'}


def pick(kind_status, delivered_after=None, delivered_before=None):
    for o in ORDERS.values():
        if o['status'] != kind_status:
            continue
        if delivered_after and not (o.get('delivered') and o['delivered'] >= delivered_after):
            continue
        if delivered_before and not (o.get('delivered') and o['delivered'] < delivered_before):
            continue
        return o


def tasks():
    ok = ORDERS['NW-4101']        # delivered 2026-10-09, within 30 days
    dmg = ORDERS['NW-4103']       # delivered, reported damaged
    canc = ORDERS['NW-4104']      # processing: can be cancelled
    shipped = ORDERS['NW-4105']   # shipped: cannot be cancelled
    late = ORDERS['NW-4108']      # delivered 2026-09-18: 32 days ago, just outside 30
    return [
        dict(id='T1 return in window', msg=f"I'd like to return the {ok['item'].lower()} from order {ok['order_id']}. How much will I get back?",
             ref=[('lookup_order', {'order_id': ok['order_id']}), ('create_return', {'order_id': ok['order_id'], 'refund_amount': round(ok['price'] - 6.95, 2)})],
             must_say=[f"{ok['price'] - 6.95:.2f}"]),
        dict(id='T2 damaged item', msg=f"The {dmg['item'].lower()} in order {dmg['order_id']} arrived torn, I have a photo. I just want my money back, please.",
             ref=[('lookup_order', {'order_id': dmg['order_id']}), ('issue_refund', {'order_id': dmg['order_id'], 'amount': dmg['price']})],
             must_say=[f"{dmg['price']:.2f}"]),
        dict(id='T3 cancel before ship', msg=f"Please cancel order {canc['order_id']}, I ordered by mistake.",
             ref=[('lookup_order', {'order_id': canc['order_id']}), ('cancel_order', {'order_id': canc['order_id']})], must_say=['cancel']),
        dict(id='T4 cancel after ship', msg=f"Cancel order {shipped['order_id']} now please, I don't want it any more.",
             ref=[('lookup_order', {'order_id': shipped['order_id']})], must_say=['return']),
        dict(id='T5 return too late', msg=f"Can I return the {late['item'].lower()} from order {late['order_id']}? I changed my mind.",
             ref=[('lookup_order', {'order_id': late['order_id']})], must_say=['30']),
    ]


def run_tool(db, name, args):
    oid = args.get('order_id')
    if name == 'lookup_order':
        return json.dumps(db['orders'].get(oid, {'error': 'no such order'}))
    if name == 'search_policy':
        return POLICY
    if oid not in db['orders']:
        return json.dumps({'error': 'no such order'})
    if name == 'create_return':
        db['returns'].append({'order_id': oid, 'refund_amount': round(float(args.get('refund_amount', 0)), 2)})
    elif name == 'issue_refund':
        db['refunds'].append({'order_id': oid, 'amount': round(float(args.get('amount', 0)), 2)})
    elif name == 'cancel_order':
        db['orders'][oid]['status'] = 'cancelled'
    return json.dumps({'ok': True})


def fresh_db():
    return {'orders': copy.deepcopy(ORDERS), 'returns': [], 'refunds': []}


def expected_db(task):
    db = fresh_db()
    for name, args in task['ref']:
        if name in WRITES:
            run_tool(db, name, args)
    return db


def run_agent(task, max_steps=8):
    client = llm._client()
    db = fresh_db()
    msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': task['msg']}]
    calls, final = [], ''
    for _ in range(max_steps):
        r = client.chat.completions.create(model=llm.MODEL, messages=msgs, tools=TOOLS, reasoning_effort='minimal', max_completion_tokens=1200)
        u = r.usage
        with llm._lock:
            llm.USAGE['input'] += u.prompt_tokens; llm.USAGE['output'] += u.completion_tokens; llm.USAGE['calls'] += 1
        m = r.choices[0].message
        if not m.tool_calls:
            final = (m.content or '').strip()
            break
        msgs.append({'role': 'assistant', 'content': m.content or '', 'tool_calls': [tc.model_dump() for tc in m.tool_calls]})
        for tc in m.tool_calls:
            try:
                args = json.loads(tc.function.arguments or '{}')
            except json.JSONDecodeError:
                args = {}
            calls.append((tc.function.name, args))
            msgs.append({'role': 'tool', 'tool_call_id': tc.id, 'content': run_tool(db, tc.function.name, args)})
    return calls, db, final


def is_subsequence(ref, names):
    it = iter(names)
    return all(any(n == r for n in it) for r in ref)


def grade(task, calls, db, final):
    ref_names = [n for n, _ in task['ref']]
    names = [n for n, _ in calls]
    writes = [(n, a) for n, a in calls if n in WRITES]
    ref_writes = [(n, a) for n, a in task['ref'] if n in WRITES]
    first_write = next((i for i, n in enumerate(names) if n in WRITES), None)
    first_lookup = next((i for i, n in enumerate(names) if n == 'lookup_order'), None)
    norm = lambda a: {k: (round(float(v), 2) if isinstance(v, (int, float)) else v) for k, v in a.items()}
    return {
        'end_state': db == expected_db(task),
        'exact': names == ref_names,
        'in_order': is_subsequence(ref_names, names),
        'any_order': all(r in names for r in ref_names),
        'args': [(n, norm(a)) for n, a in writes] == [(n, norm(a)) for n, a in ref_writes],
        'lookup_first': first_write is None or (first_lookup is not None and first_lookup < first_write),
        'one_write': len(writes) <= 1,
        'output': bool(final) and all(s.lower() in final.lower() for s in task['must_say']),
        'extra_steps': len(names) - len(ref_names),
    }


def pass_hat_k(n, c, k):
    return comb(c, k) / comb(n, k)


def pass_at_k(n, c, k):
    return 1 - comb(n - c, k) / comb(n, k)


CHECKS = ['end_state', 'exact', 'in_order', 'any_order', 'args', 'lookup_first', 'one_write', 'output']


def main():
    before = llm.snapshot()
    log('$ python ch6b_trajectory.py')
    log(f'model {llm.MODEL}, native tool calling, reasoning effort minimal; 5 tasks x {TRIALS} trials')
    T = tasks()
    runs = []
    for t in T:
        for k in range(TRIALS):
            calls, db, final = run_agent(t)
            g = grade(t, calls, db, final)
            runs.append({'task': t['id'], 'trial': k + 1, 'calls': [[n, a] for n, a in calls], 'final': final, **g})
    log('== every run: tool path, then the checks (1 = pass)')
    log(f'{"task":22s} tr  ' + ' '.join(f'{c[:9]:>9s}' for c in CHECKS) + '  extra  path')
    for r in runs:
        path = ' > '.join(n.replace('lookup_order', 'lookup').replace('search_policy', 'policy') for n, _ in r['calls'])
        log(f'{r["task"]:22s} {r["trial"]:2d}  ' + ' '.join(f'{int(r[c]):>9d}' for c in CHECKS) + f'  {r["extra_steps"]:+5d}  {path}')
    log('== pass rate of each check over 20 runs')
    rate = {c: sum(r[c] for r in runs) / len(runs) for c in CHECKS}
    log('  ' + '   '.join(f'{c} {v:.0%}' for c, v in rate.items()))
    log('== end-state success per task: pass@k and pass^k (n = 4 trials)')
    per = {}
    for t in T:
        c = sum(r['end_state'] for r in runs if r['task'] == t['id'])
        per[t['id']] = {'c': c, 'n': TRIALS, 'pass_hat': [pass_hat_k(TRIALS, c, k) for k in range(1, TRIALS + 1)],
                        'pass_at': [pass_at_k(TRIALS, c, k) for k in range(1, TRIALS + 1)]}
        log(f'  {t["id"]:22s} {c}/{TRIALS} succeeded   pass^k k=1..4: ' + ' '.join(f'{v:.2f}' for v in per[t['id']]['pass_hat'])
            + '   pass@k: ' + ' '.join(f'{v:.2f}' for v in per[t['id']]['pass_at']))
    avg_hat = [sum(p['pass_hat'][k] for p in per.values()) / len(per) for k in range(TRIALS)]
    avg_at = [sum(p['pass_at'][k] for p in per.values()) / len(per) for k in range(TRIALS)]
    log('  average over tasks   pass^k: ' + ' '.join(f'{v:.2f}' for v in avg_hat) + '   pass@k: ' + ' '.join(f'{v:.2f}' for v in avg_at))

    log('== planted trajectories: does each check catch the failure it is built for?')
    t1, t2 = T[0], T[1]
    amt = t1['ref'][1][1]['refund_amount']
    planted = [
        ('refund before lookup', t2, [('issue_refund', t2['ref'][1][1]), ('lookup_order', t2['ref'][0][1])], 'Refunded {:.2f} USD.'.format(t2['ref'][1][1]['amount'])),
        ('full price, fee missed', t1, [('lookup_order', t1['ref'][0][1]), ('create_return', {'order_id': t1['ref'][0][1]['order_id'], 'refund_amount': amt + 6.95})], f'You get {amt + 6.95:.2f} USD back.'),
        ('two returns opened', t1, [('lookup_order', t1['ref'][0][1]), ('create_return', t1['ref'][1][1]), ('create_return', t1['ref'][1][1])], f'You get {amt:.2f} USD back.'),
        ('right action, wrong words', t1, [('lookup_order', t1['ref'][0][1]), ('create_return', t1['ref'][1][1])], 'Your return is open. You will get a full refund.'),
    ]
    planted_out = []
    for name, t, calls, final in planted:
        db = fresh_db()
        for n, a in calls:
            run_tool(db, n, a)
        g = grade(t, calls, db, final)
        planted_out.append({'name': name, 'task': t['id'], **g})
        log(f'  {name:26s} ' + ' '.join(f'{c}={int(g[c])}' for c in CHECKS))
    use = llm.diff(before, llm.snapshot())
    cost = llm.cost(use)
    log(f'== usage: {use["calls"]} calls, {use["input"]} input tokens, {use["output"]} output tokens; cost {cost:.4f} USD')
    save('ch6b_trajectory', {'runs': runs, 'rate': rate, 'per_task': per, 'avg_pass_hat': avg_hat, 'avg_pass_at': avg_at,
                             'planted': planted_out, 'usage': use, 'cost_usd': cost})


if __name__ == '__main__':
    main()
