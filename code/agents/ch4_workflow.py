"""Chapter 4, Section 4.7: one complete, runnable composed workflow for Northwind Home's support tickets, with the real API
(gpt-5-mini, reasoning_effort=minimal). Everything the chapter recommends, in about 250 lines:
  typed state (a dataclass) and typed step outputs (pydantic schemas, validated in code);
  a router that is code first and asks the model only when the rules do not decide;
  a gated chain: extract -> draft, each step schema-validated, retried once with the exact errors, then handed to a person;
  a policy check in code (word limit, banned phrases, order id quoted) after the draft;
  a human-in-the-loop interrupt: refunds over GBP 50 pause the run until someone approves;
  an idempotent side effect: replies go to an outbox keyed by an idempotency key, so a resumed run cannot send twice;
  budgets per ticket (model calls, tokens) that stop the run instead of letting it spin;
  a checkpoint after every node, and resume from the last checkpoint after a crash;
  a trace: one JSON line per node (trace id, node, outcome, tokens, milliseconds) in results/ch4_workflow_trace.jsonl.
The demo processes six tickets, injects one crash after the draft step of ticket T3, resumes it, and approves the paused refund."""
import hashlib, json, os, re, time, uuid
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional
from pydantic import BaseModel, Field, ValidationError
from common import Log, save, RESULTS
import ch4_llm as llm

log = Log('ch4_workflow')
llm.set_log(log)
CKPT = os.path.join(RESULTS, 'ch4_workflow_ckpt')
OUTBOX = os.path.join(RESULTS, 'ch4_workflow_outbox.json')
TRACE = os.path.join(RESULTS, 'ch4_workflow_trace.jsonl')
MAX_CALLS, MAX_TOKENS, REFUND_LIMIT = 6, 8000, 50.0
BANNED = ['24 hours', 'immediately', 'guarantee', 'today', 'right away', 'as soon as possible', 'shortly']


# ---------------------------------------------------------------- typed step outputs (the schemas the gates enforce)
class Facts(BaseModel):
    issue_type: Literal['billing', 'delivery', 'damaged_or_faulty', 'account', 'other']
    order_id: Optional[str] = Field(default=None, pattern=r'^ORD-\d{5}$')
    refund_amount_gbp: Optional[float] = Field(default=None, ge=0)


class Draft(BaseModel):
    reply: str = Field(min_length=20)


# ---------------------------------------------------------------- typed workflow state
@dataclass
class State:
    ticket_id: str
    text: str
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    node: str = 'route'                 # the next node to run; 'done' and 'needs_human' and 'awaiting_approval' are terminal or paused
    route: Optional[str] = None
    facts: Optional[dict] = None
    reply: Optional[str] = None
    approved: Optional[bool] = None
    calls: int = 0
    tokens: int = 0
    notes: list = field(default_factory=list)


class BudgetExceeded(Exception):
    pass


class InjectedCrash(Exception):
    pass


# ---------------------------------------------------------------- infrastructure: checkpoints, trace, model calls with budgets
def checkpoint(st):
    os.makedirs(CKPT, exist_ok=True)
    json.dump(asdict(st), open(os.path.join(CKPT, st.ticket_id + '.json'), 'w'), indent=1)


def load(ticket_id):
    p = os.path.join(CKPT, ticket_id + '.json')
    return State(**json.load(open(p))) if os.path.exists(p) else None


def span(st, node, outcome, t0, use=None):
    rec = {'trace_id': st.trace_id, 'ticket': st.ticket_id, 'node': node, 'outcome': outcome, 'ms': round((time.time() - t0) * 1000),
           'tokens': (use or {}).get('input', 0) + (use or {}).get('output', 0)}
    open(TRACE, 'a').write(json.dumps(rec) + '\n')
    log(f'  [{st.trace_id}] {st.ticket_id} {node:<9} {outcome:<34} {rec["tokens"]:>5} tok {rec["ms"]:>6} ms')


def call(st, prompt):
    if st.calls >= MAX_CALLS or st.tokens >= MAX_TOKENS:
        raise BudgetExceeded(f'budget: {st.calls} calls, {st.tokens} tokens')
    text, use = llm.ask(prompt, max_tokens=600)
    st.calls += 1
    st.tokens += use['input'] + use['output']
    return text, use


def validated(st, prompt, model_cls, extra_check=None):
    """A gated step: call, parse, validate against the schema (and an optional extra check), retry once with the exact errors."""
    total = {'input': 0, 'output': 0}
    for attempt in range(2):
        text, use = call(st, prompt)
        for k in total:
            total[k] += use[k]
        try:
            obj = model_cls.model_validate(llm.parse_json(text) or {})
            problems = extra_check(obj) if extra_check else []
        except ValidationError as e:
            obj, problems = None, [f'{".".join(str(x) for x in err["loc"])}: {err["msg"]}' for err in e.errors()]
        if not problems:
            return obj, total, attempt
        prompt = prompt + '\n\nYour previous answer failed these checks:\n- ' + '\n- '.join(problems) + '\nReply with a corrected JSON object only.'
    return None, total, 1


# ---------------------------------------------------------------- the nodes
def route(st):
    t = st.text.lower()
    if re.search(r'\b(refund|charged|invoice|broken|cracked|damaged|parcel|delivery|order)\b', t):     # code decides when it can
        return 'support', None
    text, use = call(st, 'Is this message a support request about an order or account (answer "support") or a general question '
                         f'about products, shipping zones or opening hours (answer "faq")? Reply with one word.\n\nMessage: {st.text}')
    return ('faq' if 'faq' in text.lower() else 'support'), use


def extract(st):
    prompt = ('Extract facts from this support ticket. Reply with ONE JSON object with keys: "issue_type" (one of billing, delivery, '
              'damaged_or_faulty, account, other), "order_id" (the order number as ORD-NNNNN, five digits, or null), '
              f'"refund_amount_gbp" (a number if the customer asks for money back and states an amount, else null).\n\nTicket: {st.text}')
    return validated(st, prompt, Facts)


def policy_problems(reply, order_id):
    p = []
    if len(reply.split()) > 80:
        p.append(f'the reply has {len(reply.split())} words; the limit is 80')
    hits = [b for b in BANNED if b in reply.lower()]
    if hits:
        p.append(f'the reply promises timing: {hits}; remove these phrases')
    if order_id and order_id not in reply:
        p.append(f'the reply must quote the order id {order_id}')
    return p


def draft(st):
    f = st.facts
    prompt = ('Write a short, polite reply (at most 80 words) to this customer, for Northwind Home. Do not promise any timing. '
              + (f'Quote the order id {f["order_id"]}. ' if f.get('order_id') else '')
              + ('Say that the refund request has been passed for approval. ' if f.get('refund_amount_gbp') else '')
              + 'Reply with ONE JSON object: ' + json.dumps({'reply': '...'}) + f'\n\nTicket: {st.text}\nFacts: {json.dumps(f)}')
    return validated(st, prompt, Draft, lambda d: policy_problems(d.reply, f.get('order_id')))


def send(st):
    """The side effect. Idempotent: the key is derived from the ticket and the reply, and the outbox ignores a key it has seen."""
    key = f'{st.ticket_id}:' + hashlib.sha256(st.reply.encode()).hexdigest()[:12]   # stable across processes, unlike hash()
    box = json.load(open(OUTBOX)) if os.path.exists(OUTBOX) else {}
    if key in box:
        return 'already sent (idempotency key seen)'
    box[key] = {'ticket': st.ticket_id, 'reply': st.reply, 'at': time.strftime('%H:%M:%S')}
    json.dump(box, open(OUTBOX, 'w'), indent=1)
    return 'sent'


# ---------------------------------------------------------------- the graph runner
def run(st, crash_after=None):
    """Run from st.node until the workflow finishes, pauses or fails; checkpoint after every node."""
    try:
        while st.node not in ('done', 'needs_human', 'awaiting_approval'):
            t0 = time.time()
            if st.node == 'route':
                st.route, use = route(st)
                span(st, 'route', f'-> {st.route}' + (' (model)' if use else ' (rule)'), t0, use)
                st.node = 'extract' if st.route == 'support' else 'faq'
            elif st.node == 'faq':
                st.notes.append('sent to the FAQ answerer (out of scope for this demo)')
                span(st, 'faq', 'handed to the FAQ workflow', t0)
                st.node = 'done'
            elif st.node == 'extract':
                obj, use, retried = extract(st)
                if obj is None:
                    span(st, 'extract', 'gate failed twice -> person', t0, use); st.node = 'needs_human'
                else:
                    st.facts = obj.model_dump()
                    span(st, 'extract', f'ok{" after retry" if retried else ""} {st.facts["issue_type"]} {st.facts["order_id"]}', t0, use)
                    st.node = 'draft'
            elif st.node == 'draft':
                obj, use, retried = draft(st)
                if obj is None:
                    span(st, 'draft', 'gate failed twice -> person', t0, use); st.node = 'needs_human'
                else:
                    st.reply = obj.reply
                    span(st, 'draft', f'ok{" after retry" if retried else ""} ({len(st.reply.split())} words)', t0, use)
                    st.node = 'review'
            elif st.node == 'review':
                amt = st.facts.get('refund_amount_gbp') or 0
                if amt > REFUND_LIMIT and st.approved is None:
                    span(st, 'review', f'refund GBP {amt:.2f} > {REFUND_LIMIT:.0f}: PAUSE', t0); st.node = 'awaiting_approval'
                elif st.approved is False:
                    span(st, 'review', 'refund rejected by a person', t0); st.node = 'needs_human'
                else:
                    span(st, 'review', 'policy ok' + (' (approved)' if st.approved else ''), t0); st.node = 'send'
            elif st.node == 'send':
                span(st, 'send', send(st), t0); st.node = 'done'
            checkpoint(st)
            if crash_after and st.node == crash_after[1] and st.ticket_id == crash_after[0]:
                raise InjectedCrash(f'process killed after checkpointing, before running {st.node}')
    except BudgetExceeded as e:
        span(st, st.node, f'STOP: {e}', time.time()); st.node = 'needs_human'; checkpoint(st)
    return st


TICKETS = [('T1', 'Order #48213 arrived with the screen cracked. Please send a replacement.'),
           ('T2', 'I was charged twice for ORD-10988, please refund the extra GBP 39.99.'),
           ('T3', 'Where is my parcel? Tracking has not moved in 6 days. Order 77120.'),
           ('T4', 'Do you ship to the Isle of Man?'),
           ('T5', 'The sofa from order 61234 has a torn seam. I want my money back, GBP 649.'),
           ('T6', 'Mi pedido 33310 llego roto.')]

if __name__ == '__main__':
    import shutil
    for p in [CKPT]:
        shutil.rmtree(p, ignore_errors=True)
    for p in [OUTBOX, TRACE]:
        if os.path.exists(p):
            os.remove(p)
    log(f'== A composed workflow for Northwind Home support ({llm.MODEL}, minimal effort): router -> gated chain -> policy -> approval -> send ==')
    log(f'budgets per ticket: {MAX_CALLS} model calls, {MAX_TOKENS} tokens; refunds over GBP {REFUND_LIMIT:.0f} need a person')
    log('')
    log('-- first run (a crash is injected in T3 after the draft is checkpointed) --')
    for tid, text in TICKETS:
        st = State(tid, text)
        try:
            run(st, crash_after=('T3', 'review'))
        except InjectedCrash as e:
            log(f'  !! {tid}: {e}')
    log('')
    log('-- resume: load every checkpoint that is not finished and continue from its saved node --')
    for tid, _ in TICKETS:
        st = load(tid)
        if st.node not in ('done', 'needs_human', 'awaiting_approval'):
            log(f'  {tid}: resuming at node "{st.node}" (calls so far {st.calls}; extract and draft are NOT re-run)')
            run(st)
    log('')
    log('-- a person approves the paused refund, and the run continues from the checkpoint --')
    for tid, _ in TICKETS:
        st = load(tid)
        if st.node == 'awaiting_approval':
            st.approved = True
            st.node = 'review'
            log(f'  {tid}: approved by a person')
            run(st)
    log('')
    log('-- replaying "send" for T1 to show idempotency --')
    st = load('T1'); t0 = time.time(); span(st, 'send', send(st), t0)
    log('')
    final = [load(t) for t, _ in TICKETS]
    log(f'{"ticket":<7} {"route":<8} {"final node":<12} {"calls":>5} {"tokens":>7}  facts')
    for st in final:
        f = st.facts or {}
        log(f'{st.ticket_id:<7} {str(st.route):<8} {st.node:<12} {st.calls:>5} {st.tokens:>7}  {f.get("issue_type", "-")} {f.get("order_id", "-")} '
            f'{("refund " + str(f.get("refund_amount_gbp"))) if f.get("refund_amount_gbp") else ""}')
    box = json.load(open(OUTBOX))
    spans = [json.loads(l) for l in open(TRACE)]
    U = llm.snapshot()
    log(f'outbox: {len(box)} replies sent for {sum(1 for s in final if s.node == "done" and s.route == "support")} finished support tickets; '
        f'trace: {len(spans)} spans in results/ch4_workflow_trace.jsonl')
    log(f'total spend: ${llm.cost(U):.4f} ({U["calls"]} calls); stub used: ' + ('YES' if llm.stub_used() else 'no'))
    save('ch4_workflow', {'states': [asdict(s) for s in final], 'spans': spans, 'outbox': box, 'usage': U, 'stub': llm.stub_used()})
