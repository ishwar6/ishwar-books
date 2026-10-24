"""Chapter 6 (first half): Northwind Home's support agent, small enough to evaluate cheaply, and a runner that saves every trace.
The agent is a tool loop (Chapter 3) around ch4_llm.ask (gpt-5-mini, reasoning effort minimal). Each turn the model returns ONE JSON
object: either {"tool": name, "args": {...}} or {"final": {"action": ..., "reply": ...}}. The three tools are fake and local:
lookup_order, search_policy and request_refund (which only records the request). Nothing is sent anywhere.
Usage: python ch6a_agent.py [v1|v2] [run_tag]   ->  results/ch6a_traces_<version><tag>.json and results/ch6a_agent_<version><tag>_stdout.txt"""
import json, os, re, sys
from concurrent.futures import ThreadPoolExecutor
import ch4_llm as llm
from common import Log, save, timer

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'data', 'ch6', 'a')
ORDERS = {o['order_id']: o for o in json.load(open(os.path.join(D, 'orders.json')))}
POLICY = json.load(open(os.path.join(D, 'policy.json')))
MESSAGES = json.load(open(os.path.join(D, 'messages.json')))
MAX_STEPS = 6

SYSTEM_V1 = """You are the customer support agent of Northwind Home, a UK online shop for furniture and kitchenware. Today is 2026-10-20.
You have three tools:
- lookup_order(order_id): returns the order (item, price_gbp, status, delivered_on, final_sale, courier, tracking) or an error.
- search_policy(query): returns the policy sections that best match the query.
- request_refund(order_id, amount_gbp, reason): issues a refund to the customer's card.
Answer with ONE JSON object and nothing else. To use a tool: {"tool": "<name>", "args": {...}}.
To finish: {"final": {"action": "<refund|escalate|decline|answer|ask_info>", "reply": "<the message to the customer>"}}.
Use "refund" if you issued a refund, "escalate" if a person must handle it, "decline" if the request is refused under the policy,
"ask_info" if you need more information from the customer, and "answer" otherwise. Be friendly and helpful."""

# v2: the change made after error analysis (Section 6.2). It adds the rules the traces showed the agent getting wrong.
SYSTEM_V2 = SYSTEM_V1 + """
Rules you must follow:
1. Always call search_policy before you decide, and follow the policy text exactly.
2. Never call request_refund for a change of mind, a billing problem, or an amount above 100 GBP; those are "answer" (return
   instructions), "escalate" and "escalate". The refund amount is always the item's price_gbp, never the amount the customer asks for.
3. Count days from delivered_on to today. More than 30 days is "decline".
4. If there is no order number, or lookup_order returns an error, finish with "ask_info". Do not guess or correct an order number.
5. Never promise a delivery date, day or time.
6. Do not ask for photos, preferences or confirmations the policy does not require. Act on the request now.
7. Do not offer options, fees, exceptions or channels that are not in the policy text.
8. Only say a refund was issued if request_refund returned ok in this conversation."""


def lookup_order(order_id=None, **_):
    o = ORDERS.get(str(order_id).strip().upper())
    return o if o else {'error': f'order {order_id} not found'}


def search_policy(query='', **_):
    words = set(re.findall(r'[a-z]+', str(query).lower()))
    scored = sorted(POLICY, key=lambda p: -len(words & set(re.findall(r'[a-z]+', (p['title'] + ' ' + p['text']).lower()))))
    return scored[:2]


def make_refund_tool(record):
    def request_refund(order_id=None, amount_gbp=None, reason='', **_):
        record.append({'order_id': order_id, 'amount_gbp': amount_gbp, 'reason': reason})
        return {'ok': True, 'refund_id': f'RF-{len(record):04d}', 'order_id': order_id, 'amount_gbp': amount_gbp}
    return request_refund


def run_one(msg, system):
    """Run the agent on one message. Returns the trace: every model output, every tool call with its result, and the final answer."""
    refunds = []
    tools = {'lookup_order': lookup_order, 'search_policy': search_policy, 'request_refund': make_refund_tool(refunds)}
    history = [{'role': 'system', 'content': system}, {'role': 'user', 'content': msg['text']}]
    trace = {'id': msg['id'], 'message': msg['text'], 'steps': [], 'final': None, 'refunds': refunds, 'usage': [], 'error': None}
    for _ in range(MAX_STEPS):
        text, u = llm.ask(history, max_tokens=1200)
        trace['usage'].append(u)
        d = llm.parse_json(text)
        history.append({'role': 'assistant', 'content': text})
        if not d:
            trace['steps'].append({'raw': text[:300], 'parse_error': True})
            trace['error'] = 'unparseable output'
            history.append({'role': 'user', 'content': 'That was not one valid JSON object. Reply again with one JSON object only.'})
            continue
        if 'final' in d:
            trace['final'] = d['final'] if isinstance(d['final'], dict) else {'action': None, 'reply': str(d['final'])}
            trace['error'] = None
            return trace
        name, args = d.get('tool'), d.get('args') or {}
        if name not in tools or not isinstance(args, dict):
            result = {'error': f'unknown tool {name}'}
        else:
            result = tools[name](**args)
        trace['steps'].append({'tool': name, 'args': args, 'result': result})
        history.append({'role': 'user', 'content': 'Tool result: ' + json.dumps(result)})
    trace['error'] = trace['error'] or 'no final answer within the step limit'
    return trace


if __name__ == '__main__':
    version = sys.argv[1] if len(sys.argv) > 1 else 'v1'
    tag = sys.argv[2] if len(sys.argv) > 2 else ''
    name = f'ch6a_agent_{version}{tag}'
    log = Log(name)
    llm.set_log(log)
    system = SYSTEM_V1 if version == 'v1' else SYSTEM_V2
    t, before = timer(), llm.snapshot()
    log(f'Northwind support agent, prompt {version}, model {llm.MODEL} (reasoning effort minimal), {len(MESSAGES)} messages, max {MAX_STEPS} steps')
    with ThreadPoolExecutor(8) as ex:
        traces = list(ex.map(lambda m: run_one(m, system), MESSAGES))
    for tr in traces:
        f = tr['final'] or {}
        calls = ' '.join(f"{s.get('tool')}({','.join(str(v) for v in (s.get('args') or {}).values())[:40]})" for s in tr['steps'] if 'tool' in s)
        log(f"{tr['id']} action={str(f.get('action')):<9} tools: {calls[:110]}")
    use = llm.diff(before, llm.snapshot())
    log(f'calls {use["calls"]}, input tokens {use["input"]:,}, output tokens {use["output"]:,} (reasoning {use["reasoning"]:,}), '
        f'cost ${llm.cost(use):.4f}, wall time {t():.0f} s, stub used: {llm.stub_used()}')
    save(f'ch6a_traces_{version}{tag}', {'version': version, 'model': llm.MODEL, 'usage': use, 'cost_usd': llm.cost(use), 'traces': traces})
