"""Chapter 5 (first half): the pieces shared by the two experiments. A real tool-calling agent loop (OpenAI chat completions with
tools, model gpt-5-mini), the Northwind research tools over the local fake document set in data/ch5/a/, a lead agent that can start
subagents in parallel, and the grader that compares an answer with the known answer in gold.json.
The API key is loaded by ch4_llm (python-dotenv, a .env file outside the repo) and is never printed. Token counts come from the
API's usage fields. Nothing here searches the live web: every tool reads local files, so runs are repeatable and cheap."""
import json, os, re, threading, time
from concurrent.futures import ThreadPoolExecutor
import ch4_llm as llm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data', 'ch5', 'a')
MODEL = llm.MODEL
PRICE_IN, PRICE_OUT = llm.PRICE_IN, llm.PRICE_OUT
GOLD = json.load(open(os.path.join(DATA, 'gold.json')))
INDEX = json.load(open(os.path.join(DATA, 'index.json')))
NEWS = json.load(open(os.path.join(DATA, 'news.json')))
SHIP = json.load(open(os.path.join(DATA, 'shipping.json')))
REQUIREMENTS = open(os.path.join(DATA, 'requirements.md')).read()

TASK = ('Which three new kitchenware suppliers should Northwind Home add for 2027? Compare price, delivery time to the UK, and risks. '
        'Apply Northwind\'s buying rules exactly. When you are done, reply with JSON only, in this shape: '
        '{"picks": [{"id": "S..", "name": "...", "landed_cost_gbp": 0.0, "delivery_days": 0, "risks": "..."}], '
        '"rejected": [{"id": "S..", "reason": "..."}]}')


# ---------------------------------------------------------------- the tools (plain Python over local files)
def read_requirements():
    return REQUIREMENTS


def list_suppliers():
    return '\n'.join(f'{s["id"]}  {s["name"]}, {s["city"]}, {s["country"]}' for s in INDEX)


def read_supplier(supplier_id):
    sid = supplier_id.strip().upper()
    path = os.path.join(DATA, 'suppliers', sid + '.md')
    return open(path).read() if os.path.exists(path) else f'No supplier page for {supplier_id!r}. Use ids such as S01.'


def search_news(query):
    words = [w for w in re.findall(r'[a-z0-9]+', query.lower()) if len(w) > 3]
    hits = [n for n in NEWS if any(w in (n['title'] + ' ' + n['text']).lower() for w in words)]
    if not hits:
        return 'No news items match.'
    return '\n\n'.join(f'[{n["id"]}, {n["date"]}] {n["title"]}\n{n["text"]}' for n in hits[:6])


def shipping_lookup(port):
    p = port.strip().lower()
    for name, v in SHIP.items():
        if name.lower() == p or name.lower() in p or p in name.lower():
            return f'{name}: transit {v["transit_days"]} days to the UK, freight {v["freight_gbp_per_basket"]:.2f} GBP per basket.'
    return f'Unknown port {port!r}. Known ports: {", ".join(SHIP)}.'


def schema(name, desc, props=None, required=None):
    return {'type': 'function', 'function': {'name': name, 'description': desc,
            'parameters': {'type': 'object', 'properties': props or {}, 'required': required or list((props or {}).keys())}}}


T_REQ = schema('read_requirements', 'Read Northwind Home\'s buying rules for new suppliers.')
T_LIST = schema('list_suppliers', 'List every candidate supplier: id, name, city, country.')
T_READ = schema('read_supplier', 'Read one supplier\'s full catalogue page (trade terms, price, lead time, MOQ, port).',
                {'supplier_id': {'type': 'string', 'description': 'for example S07'}})
T_NEWS = schema('search_news', 'Search the trade news feed (recalls, audits, company news) by keywords. Returns up to 6 items.',
                {'query': {'type': 'string'}})
T_SHIP = schema('shipping_lookup', 'Look up transit days to the UK and freight per basket (GBP) for a shipping port.',
                {'port': {'type': 'string'}})
TOOLS = {'read_requirements': read_requirements, 'list_suppliers': list_suppliers, 'read_supplier': read_supplier,
         'search_news': search_news, 'shipping_lookup': shipping_lookup}


# ---------------------------------------------------------------- usage accounting, shared by every agent of one run
class Meter:
    def __init__(self):
        self.lock = threading.Lock()
        self.by_agent = {}

    def add(self, agent, u, prompt_tokens):
        with self.lock:
            a = self.by_agent.setdefault(agent, {'calls': 0, 'input': 0, 'output': 0, 'reasoning': 0, 'peak_context': 0, 'tool_calls': 0})
            a['calls'] += 1
            a['input'] += u['input']
            a['output'] += u['output']
            a['reasoning'] += u['reasoning']
            a['peak_context'] = max(a['peak_context'], prompt_tokens)

    def tool(self, agent):
        with self.lock:
            self.by_agent.setdefault(agent, {'calls': 0, 'input': 0, 'output': 0, 'reasoning': 0, 'peak_context': 0, 'tool_calls': 0})['tool_calls'] += 1

    def total(self):
        t = {'calls': 0, 'input': 0, 'output': 0, 'reasoning': 0, 'tool_calls': 0}
        for a in self.by_agent.values():
            for k in t:
                t[k] += a[k]
        t['peak_context'] = max((a['peak_context'] for a in self.by_agent.values()), default=0)
        t['usd'] = t['input'] / 1e6 * PRICE_IN + t['output'] / 1e6 * PRICE_OUT
        return t


def chat(messages, tools=None, effort='low', max_tokens=6000, parallel=True):
    """One real API call. Returns (message, usage, prompt_tokens). Retries once on an API error."""
    client = llm._client()
    kw = dict(model=MODEL, messages=messages, max_completion_tokens=max_tokens, reasoning_effort=effort)
    if tools:
        kw['tools'] = tools
        if not parallel:
            kw['parallel_tool_calls'] = False                      # the model may request only one tool call per turn
    for attempt in range(3):
        try:
            r = client.chat.completions.create(**kw)
            u = r.usage
            d = u.completion_tokens_details
            use = {'input': u.prompt_tokens, 'output': u.completion_tokens, 'reasoning': (d.reasoning_tokens or 0) if d else 0}
            return r.choices[0].message, use, u.prompt_tokens
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(3 + 3 * attempt)


def run_agent(name, messages, tools, handlers, meter, max_calls=30, effort='low', trace=None, parallel=True):
    """The agent loop of Chapter 1: call the model; if it asks for tools, run them and append the results; stop when it answers
    in text or the call budget runs out. Several tool calls in one turn are all executed, in order."""
    messages = list(messages)
    for _ in range(max_calls):
        msg, use, ptok = chat(messages, tools=tools, effort=effort, parallel=parallel)
        meter.add(name, use, ptok)
        if trace is not None:
            trace.append({'agent': name, 'prompt_tokens': ptok, 'output': use['output'],
                          'tools': [c.function.name for c in (msg.tool_calls or [])],
                          'args': [(c.function.arguments or '')[:80] for c in (msg.tool_calls or [])]})
        if not msg.tool_calls:
            return (msg.content or '').strip(), messages + [{'role': 'assistant', 'content': msg.content or ''}]
        messages.append({'role': 'assistant', 'content': msg.content or '',
                         'tool_calls': [{'id': c.id, 'type': 'function', 'function': {'name': c.function.name, 'arguments': c.function.arguments}}
                                        for c in msg.tool_calls]})
        for c in msg.tool_calls:
            meter.tool(name)
            try:
                args = json.loads(c.function.arguments or '{}')
                out = handlers[c.function.name](**args)
            except Exception as e:                                  # a bad call is reported to the model, as in Chapter 2
                out = f'Tool error: {type(e).__name__}: {e}'
            messages.append({'role': 'tool', 'tool_call_id': c.id, 'content': str(out)})
    return '', messages                                             # budget exhausted: no answer


# ---------------------------------------------------------------- the three agent roles
SINGLE_SYSTEM = ('You are a research agent for Northwind Home, a UK homeware shop. Use your tools to answer the buyer\'s question. '
                 'Read the buying rules first. Check every candidate supplier: its page, the news feed and the shipping table. '
                 'Do the arithmetic carefully. When you have checked everything, give the final answer in the requested format.')

LEAD_SYSTEM = ('You are the lead research agent for Northwind Home, a UK homeware shop. You cannot read supplier pages, news or '
               'shipping data yourself: subagents do that with their own tools (read_supplier, search_news, shipping_lookup). '
               'Read the buying rules and the supplier list, then split the suppliers among 3 to 6 subagents and start them all at once '
               'with run_subagents. A subagent sees ONLY the brief you write, nothing else from this conversation. So every brief needs: '
               'the objective, the rules the subagent must apply (copy them, with all numbers), the exact output format, which tools to use, '
               'and clear limits (which suppliers, what not to do). When the results come back, check them, combine them and give the final '
               'answer in the requested format.')

SUB_SYSTEM = ('You are a research subagent working for a lead agent. Do exactly the task in the brief, using your tools '
              '(read_supplier, search_news, shipping_lookup). Report back in the format the brief asks for.')

T_SPAWN = schema('run_subagents', 'Start several research subagents in parallel. Each gets only its brief. Returns every subagent\'s final report.',
                 {'tasks': {'type': 'array', 'description': 'one item per subagent',
                            'items': {'type': 'object', 'properties': {'brief': {'type': 'string'}}, 'required': ['brief']}}})

SUB_TOOLS = [T_READ, T_NEWS, T_SHIP]
SUB_HANDLERS = {k: TOOLS[k] for k in ('read_supplier', 'search_news', 'shipping_lookup')}


def run_subagent(name, brief, meter, trace=None, max_calls=25, history=None):
    msgs = [{'role': 'system', 'content': SUB_SYSTEM}] + (history or []) + [{'role': 'user', 'content': brief}]
    t0 = time.time()
    out, _ = run_agent(name, msgs, SUB_TOOLS, SUB_HANDLERS, meter, max_calls=max_calls, trace=trace)
    return out, time.time() - t0


def run_single(meter, trace=None, max_calls=90, parallel=True):
    msgs = [{'role': 'system', 'content': SINGLE_SYSTEM}, {'role': 'user', 'content': TASK}]
    return run_agent('single', msgs, [T_REQ, T_LIST, T_READ, T_NEWS, T_SHIP], TOOLS, meter, max_calls=max_calls, trace=trace, parallel=parallel)[0]


def run_lead(meter, trace=None, briefs_out=None, max_subagents=6, max_calls=12, sub_seconds=None):
    """The lead agent. Its run_subagents tool fans the briefs out on threads; the lead waits for all of them (synchronous, as in
    Anthropic's description) and receives their reports as the tool result."""
    counter = {'n': 0}

    def run_subagents(tasks):
        tasks = tasks[:max_subagents]                                   # a cap in code, not only in the prompt
        names = []
        for t in tasks:
            counter['n'] += 1
            names.append(f'sub{counter["n"]}')
        if briefs_out is not None:
            briefs_out.extend(t["brief"] for t in tasks)
        with ThreadPoolExecutor(len(tasks)) as ex:
            res = list(ex.map(lambda nt: run_subagent(nt[0], nt[1]['brief'], meter, trace), zip(names, tasks)))
        if sub_seconds is not None:
            sub_seconds.extend(sec for _, sec in res)
        return '\n\n'.join(f'=== report from {n} ({s:.0f} s) ===\n{out or "(no answer: budget exhausted)"}' for n, (out, s) in zip(names, res))

    msgs = [{'role': 'system', 'content': LEAD_SYSTEM}, {'role': 'user', 'content': TASK}]
    handlers = {'read_requirements': read_requirements, 'list_suppliers': list_suppliers, 'run_subagents': run_subagents}
    return run_agent('lead', msgs, [T_REQ, T_LIST, T_SPAWN], handlers, meter, max_calls=max_calls, trace=trace)[0]


# ---------------------------------------------------------------- grading against the known answer
def parse(text):
    return llm.parse_json(text) or {}


def coverage(trace):
    """Which supplier pages were actually read (by any agent of the run), from the tool-call arguments in the trace."""
    ids = set()
    for t in trace:
        for name, a in zip(t['tools'], t.get('args', [])):
            if name == 'read_supplier':
                ids.update(re.findall(r'S\d\d', a.upper()))
    return sorted(ids)


def grade(answer):
    """Compare a final answer with gold.json: how many of the three gold picks it chose, picks that break a rule, and whether the
    numbers it reports for its picks are right (landed cost within 1%, delivery days exact)."""
    a = parse(answer)
    table = {r['id']: r for r in GOLD['table']}
    picks = [p for p in a.get('picks', []) if isinstance(p, dict)]
    ids = [str(p.get('id', '')).upper() for p in picks]
    right = len(set(ids) & set(GOLD['picks']))
    breaks = [i for i in ids if i in table and not table[i]['eligible']]
    num_ok = 0
    for p in picks:
        g = table.get(str(p.get('id', '')).upper())
        try:
            if g and abs(float(p.get('landed_cost_gbp')) - g['landed_gbp']) <= 0.01 * g['landed_gbp'] and int(p.get('delivery_days')) == g['delivery_days']:
                num_ok += 1
        except (TypeError, ValueError):
            pass
    rejected = {str(r.get('id', '')).upper() for r in a.get('rejected', []) if isinstance(r, dict)}
    traps = {'S03 open recall': 'S03' in rejected, 'S11 MOQ': 'S11' in rejected, 'S07 delivery': 'S07' in rejected,
             'S19 audit': 'S19' in rejected, 'S08 not wrongly rejected': 'S08' not in rejected}
    return {'parsed': bool(a), 'picks': ids, 'gold_in_picks': right, 'rule_breaking_picks': breaks, 'numbers_ok': num_ok, 'traps': traps}
