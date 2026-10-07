"""Chapter 4, Section 4.2: a three-step prompt chain over twenty support tickets, with and without programmatic gates.
Step 1 extracts structured facts (issue type, order id, urgency); step 2 chooses a reply category and drafts the reply; step 3 checks
the reply against a policy list. In the GATED condition a schema check runs after every step and a failed check retries the step once
with the error in the prompt. In the UNGATED condition whatever the model returned flows to the next step unchanged. Success is judged
at the end by a programmatic check of the final reply (right category, order id present, at most 70 words, no banned promises), and the
first step at which something went wrong is recorded, so the chapter can show where failures happen along a chain and what gates buy.
Real API (gpt-5-mini, reasoning_effort=minimal); token usage and cost from the API's usage fields; a labelled stub if the API fails."""
import json, re
from common import Log, save
import ch4_llm as llm

log = Log('ch4_chain')
llm.set_log(log)

ISSUE_TYPES = ['billing', 'delivery', 'damaged_or_faulty', 'account', 'other']
CATEGORY_FOR = {'billing': 'billing_review', 'delivery': 'tracking_update', 'damaged_or_faulty': 'replacement_offer',
                'account': 'account_help', 'other': 'general_info'}
BANNED = ['24 hours', '48 hours', 'immediately', 'guarantee', 'today', 'right away', 'as soon as possible', 'shortly', 'asap']    # timing promises the policy forbids
ORDER_RE = re.compile(r'^ORD-\d{5}$')

# fifteen tickets, deliberately messy (order ids in five formats, some with none), with the gold facts the chain must recover
TICKETS = [
    ("Hi, order #48213 arrived with the screen cracked. I need a replacement before Friday, it's a gift.", 'damaged_or_faulty', 'ORD-48213'),
    ("I was charged twice for ORD-10988. Please fix this.", 'billing', 'ORD-10988'),
    ("Where is my parcel? Tracking hasn't moved in 6 days. Order 77120.", 'delivery', 'ORD-77120'),
    ("I can't log in, the password reset email never arrives. No order, just my account (j.okafor@example.com).", 'account', None),
    ("Do you ship to the Isle of Man?", 'other', None),
    ("ORD 55501: the box was soaked through and the contents are ruined.", 'damaged_or_faulty', 'ORD-55501'),
    ("The invoice for order no. 30377 shows VAT at 25% but I'm in the UK, it should be 20%.", 'billing', 'ORD-30377'),
    ("URGENT: order 90001 is a cake stand for a wedding tomorrow and the courier app just says 'delayed'.", 'delivery', 'ORD-90001'),
    ("My subscription renewed and took £12 from my card but I cancelled it last month. Account name m.rossi.", 'billing', None),
    ("Delivered to the wrong address (order 61234). Maybe the neighbour at number 14 has it?", 'delivery', 'ORD-61234'),
    ("The kettle (ORD-20045) stopped working after two weeks and there were sparks from the base.", 'damaged_or_faulty', 'ORD-20045'),
    ("Please delete my account and all my data. Thanks.", 'account', None),
    ("I'd like to change the delivery address for order 88800 before it ships.", 'delivery', 'ORD-88800'),
    ("Charged £49.99 but the site said £39.99 for order 12345.", 'billing', 'ORD-12345'),
    ("Can I get the manual for the X200 I bought last year? No order number to hand, sorry.", 'other', None),
    ("Mi pedido 33310 llego roto, la tapa esta partida.", 'damaged_or_faulty', 'ORD-33310'),
    ("I sent back order ORD-70707 three weeks ago and still no money back on my card.", 'billing', 'ORD-70707'),
    ("Two orders: 11111 arrived fine but 22222 has not turned up at all.", 'delivery', 'ORD-22222'),
    ("Your courier left my parcel (order 50505) in the rain and the printer inside will not turn on.", 'damaged_or_faulty', 'ORD-50505'),
    ("How do I change the email address on my account?", 'account', None),
]

STEP1 = """Extract facts from this customer support ticket. Reply with ONE JSON object and nothing else, with exactly these keys:
"issue_type": one of {types}
"order_id": the order number written in the canonical form ORD-NNNNN (five digits, add the ORD- prefix if the customer left it out), or null if there is none
"urgency": one of ["low", "medium", "high"]
"summary": one short sentence

Ticket: {ticket}"""

STEP1_LOOSE = """Extract the issue type, the order id and the urgency from this customer support ticket. Reply in JSON.

Ticket: {ticket}"""

STEP2 = """You draft replies for a customer support team. You are given the facts extracted from a ticket (as JSON).
Choose the reply category from this fixed mapping of issue_type to category:
billing -> billing_review; delivery -> tracking_update; damaged_or_faulty -> replacement_offer; account -> account_help; other -> general_info
Policy for the reply text:
P1 at most 70 words. P2 if there is an order id, quote it exactly as given (ORD-NNNNN). P3 never promise timing (no "within 24 hours", "immediately",
"today", "right away", "guarantee", "as soon as possible", "shortly", "asap", "24 hours", "48 hours"). P4 do not mention a refund unless the category is billing_review. P5 be polite and end by offering further help.
Reply with ONE JSON object and nothing else: {{"category": "<one of the five categories>", "reply": "<the reply text>"}}

Extracted facts: {facts}"""

STEP3 = """You are the policy checker for a customer support team. Check this draft reply against the policy and reply with ONE JSON object
and nothing else: {{"ok": true or false, "violations": [list of the policy ids violated, from "P1".."P5"]}}
Policy: P1 at most 70 words. P2 if the facts contain an order id, the reply must quote it exactly in the form ORD-NNNNN. P3 the reply must not
promise timing ("within 24 hours", "immediately", "today", "right away", "guarantee", "as soon as possible", "shortly", "asap", "24 hours", "48 hours"). P4 the reply must not mention a refund unless the category
is billing_review. P5 the reply must be polite and end by offering further help.

Facts: {facts}
Draft: {draft}"""


# ---- the programmatic gates: each returns a list of problems (empty = pass)
def gate1(d):
    if not isinstance(d, dict):
        return ['not a JSON object']
    p = []
    if d.get('issue_type') not in ISSUE_TYPES:
        p.append(f'issue_type must be one of {ISSUE_TYPES}, got {d.get("issue_type")!r}')
    oid = d.get('order_id')
    if oid is not None and not (isinstance(oid, str) and ORDER_RE.match(oid)):
        p.append(f'order_id must match ORD-NNNNN or be null, got {oid!r}')
    if d.get('urgency') not in ['low', 'medium', 'high']:
        p.append(f'urgency must be low, medium or high, got {d.get("urgency")!r}')
    return p


def policy(reply, order_id, category):
    """The mechanical part of the policy (P1 to P4), used both as the step-2 gate and as the end-of-chain truth."""
    p = []
    words = len(reply.split())
    if words > 70:
        p.append(f'P1: {words} words, limit 70')
    if order_id and not re.search(r'(?<![\w-])' + re.escape(order_id) + r'(?!\d)', reply):
        p.append(f'P2: the order id {order_id} is not quoted')
    low = reply.lower()
    hits = [b for b in BANNED if b in low]
    if hits:
        p.append(f'P3: timing promise {hits}')
    if 'refund' in low and category != 'billing_review':
        p.append('P4: mentions a refund outside billing_review')
    return p


def gate2(d, order_id):
    if not isinstance(d, dict) or not isinstance(d.get('reply'), str):
        return ['not a JSON object with a "reply" string']
    p = []
    if d.get('category') not in CATEGORY_FOR.values():
        p.append(f'category must be one of {sorted(CATEGORY_FOR.values())}, got {d.get("category")!r}')
    return p + policy(d['reply'], order_id, d.get('category'))


def gate3(d):
    if not isinstance(d, dict) or not isinstance(d.get('ok'), bool) or not isinstance(d.get('violations'), list):
        return ['not a JSON object with "ok" (boolean) and "violations" (list)']
    return []


def step(prompt, gate, gated, use):
    """One chain step. With gates: parse, check, retry once with the problems appended. Without: return whatever came back."""
    text, u = llm.ask(prompt, max_tokens=500)
    use.append(u)
    d = llm.parse_json(text)
    if not gated:
        return d, text, 0
    problems = gate(d)
    if not problems:
        return d, text, 0
    retry = prompt + '\n\nYour previous answer failed these checks:\n- ' + '\n- '.join(problems) + '\nReply again with a corrected JSON object only.'
    text, u = llm.ask(retry, max_tokens=500)
    use.append(u)
    d = llm.parse_json(text)
    return (d if not gate(d) else None), text, 1


def run_chain(ticket, gold_type, gold_oid, gated, step1=STEP1):
    use, retries = [], 0
    rec = {'ticket': ticket, 'gold_type': gold_type, 'gold_oid': gold_oid, 'first_fault': None, 'success': False, 'notes': []}
    # step 1: extract
    facts, raw1, r = step(step1.format(types=json.dumps(ISSUE_TYPES), ticket=ticket), gate1, gated, use); retries += r
    if facts is None and gated:
        rec['first_fault'] = 'step1'; rec['notes'].append('gate 1 stopped the chain'); rec['use'] = use; rec['retries'] = retries
        return rec
    ext_type = facts.get('issue_type') if isinstance(facts, dict) else None
    ext_oid = facts.get('order_id') if isinstance(facts, dict) else None
    if ext_type != gold_type or ext_oid != gold_oid:
        rec['first_fault'] = 'step1'; rec['notes'].append(f'extracted {ext_type!r}/{ext_oid!r}, gold {gold_type!r}/{gold_oid!r}')
    facts_text = json.dumps(facts) if isinstance(facts, dict) else raw1        # ungated: a broken step-1 output flows on as raw text
    # step 2: draft
    draft, raw2, r = step(STEP2.format(facts=facts_text), lambda d: gate2(d, ext_oid if isinstance(ext_oid, str) else None), gated, use); retries += r
    if draft is None and gated:
        rec['first_fault'] = rec['first_fault'] or 'step2'; rec['notes'].append('gate 2 stopped the chain'); rec['use'] = use; rec['retries'] = retries
        return rec
    cat = draft.get('category') if isinstance(draft, dict) else None
    reply = draft.get('reply') if isinstance(draft, dict) and isinstance(draft.get('reply'), str) else (raw2 or '')
    truth = policy(reply, gold_oid, cat)                                         # the end-of-chain truth, against the GOLD order id
    rec['policy_ok'] = not policy(reply, None, cat)                              # P1, P3, P4 only: the part gate 2 also checks (shared with the feedback)
    rec['gold_ok'] = cat == CATEGORY_FOR[gold_type] and not (gold_oid and policy(reply, gold_oid, cat) and any(t.startswith('P2') for t in policy(reply, gold_oid, cat)))
    # gold_ok uses only labels no gate ever sees: the right category, and the GOLD order id quoted
    if cat != CATEGORY_FOR[gold_type]:
        truth.append(f'category {cat!r}, expected {CATEGORY_FOR[gold_type]!r}')
    if truth and not rec['first_fault']:
        rec['first_fault'] = 'step2'
    if truth:
        rec['notes'] += truth
    # step 3: check
    verdict, raw3, r = step(STEP3.format(facts=facts_text, draft=json.dumps(draft) if isinstance(draft, dict) else raw2), gate3, gated, use); retries += r
    ok = verdict.get('ok') if isinstance(verdict, dict) else None
    mech_ok = not [t for t in truth if t.startswith('P')]                        # P1-P4 only: the checker cannot know the gold category
    rec['checker_ok'] = ok
    rec['checker_violations'] = verdict.get('violations') if isinstance(verdict, dict) else None
    rec['checker_agrees'] = (ok == mech_ok) if isinstance(ok, bool) else False
    if not rec['checker_agrees'] and not rec['first_fault']:
        rec['first_fault'] = 'step3'; rec['notes'].append(f'checker said ok={ok!r} but the mechanical check says ok={mech_ok}')
    rec['success'] = not truth
    rec['category'] = cat; rec['reply'] = reply; rec['use'] = use; rec['retries'] = retries
    return rec


def summarise(name, recs):
    tot = {'input': 0, 'output': 0, 'reasoning': 0}
    for r in recs:
        for u in r['use']:
            for k in tot:
                tot[k] += u[k]
    calls = sum(len(r['use']) for r in recs)
    faults = {s: sum(1 for r in recs if r['first_fault'] == s) for s in ['step1', 'step2', 'step3']}
    succ = sum(1 for r in recs if r['success'])
    gold_ok = sum(1 for r in recs if r.get('gold_ok'))
    pol_ok = sum(1 for r in recs if r.get('policy_ok'))
    c = llm.cost(tot)
    return {'condition': name, 'success': succ, 'gold_ok': gold_ok, 'policy_ok': pol_ok, 'n': len(recs), 'faults': faults, 'calls': calls, 'retries': sum(r['retries'] for r in recs),
            'tokens': tot, 'cost_usd': c, 'cost_per_success': (c / succ) if succ else None,
            'checker_agreement': sum(1 for r in recs if r.get('checker_agrees')) / max(1, sum(1 for r in recs if 'checker_agrees' in r))}


log(f'== A three-step prompt chain over {len(TICKETS)} tickets, with and without gates ({llm.MODEL}, reasoning_effort=minimal) ==')
log('steps: 1 extract facts (JSON) -> 2 choose category and draft the reply (JSON) -> 3 check the reply against the policy (JSON)')
log('gated: a schema and policy check after every step, one retry with the error shown. ungated: whatever came back flows on.')
log('loose: the same chain with a first-draft step-1 prompt that does not spell out the keys, the allowed values or the order-id format.')
log('')
results = {}
CONDITIONS = [('ungated', False, STEP1), ('gated', True, STEP1), ('loose-ungated', False, STEP1_LOOSE), ('loose-gated', True, STEP1_LOOSE)]
for name, gated, p1 in CONDITIONS:
    recs = [run_chain(t, g, o, gated, p1) for t, g, o in TICKETS]
    results[name] = recs
    log(f'-- {name} --')
    log(f'{"#":>2} {"gold type":<18} {"gold id":<9} {"ok":<3} {"first fault":<11} {"calls":>5} notes')
    for i, r in enumerate(recs):
        log(f'{i + 1:>2} {r["gold_type"]:<18} {str(r["gold_oid"]):<9} {"yes" if r["success"] else "NO ":<3} {str(r["first_fault"]):<11} {len(r["use"]):>5} '
            + ('; '.join(r['notes'])[:95]))
    log('')

summary = {k: summarise(k, v) for k, v in results.items()}
log('-- summary (tokens and cost from the API usage fields; price per million: '
    f'${llm.PRICE_IN} input, ${llm.PRICE_OUT} output) --')
log(f'{"condition":<13} {"success":>8} {"gold ok":>8} {"policy ok":>9} {"fault@1":>8} {"fault@2":>8} {"fault@3":>8} {"calls":>6} {"retries":>8} {"in tok":>8} {"out tok":>8} {"cost $":>8} {"$/success":>10} {"checker agrees":>15}')
for s in summary.values():
    log(f'{s["condition"]:<13} {s["success"]:>5}/{s["n"]:<2} {s["gold_ok"]:>5}/{s["n"]:<2} {s["policy_ok"]:>6}/{s["n"]:<2} {s["faults"]["step1"]:>8} {s["faults"]["step2"]:>8} {s["faults"]["step3"]:>8} {s["calls"]:>6} '
        f'{s["retries"]:>8} {s["tokens"]["input"]:>8,} {s["tokens"]["output"]:>8,} {s["cost_usd"]:>8.4f} '
        + (f'{s["cost_per_success"]:>10.4f}' if s['cost_per_success'] else f'{"n/a":>10}') + f' {s["checker_agreement"]:>14.0%}')
log('success = gold ok AND policy ok. "gold ok" (right category, the GOLD order id quoted) uses labels no gate sees; "policy ok"')
log('(word limit, no timing promises, no refund talk outside billing) is the part gate 2 also checks, so it is not independent of the feedback.')
from collections import Counter
flags = Counter(v for recs in results.values() for r in recs for v in (r.get('checker_violations') or []) if isinstance(v, str))
log(f'policy ids the step-3 model checker flagged, all four conditions together: {dict(sorted(flags.items()))}')
log('(the mechanical check covers P1 to P4; P5, politeness and offering help, is judgement only)')
U = llm.snapshot()
log('')
log(f'total spend: ${llm.cost(U):.4f} ({U["calls"]} calls, {U["input"]:,} input + {U["output"]:,} output tokens, {U["reasoning"]} hidden reasoning tokens)')
log('stub used: ' + ('YES, results are not real' if llm.stub_used() else 'no, every answer came from the API'))
save('ch4_chain', {'checker_flags': dict(flags), 'model': llm.MODEL, 'stub': llm.stub_used(), 'summary': summary, 'results': results, 'usage': U})
