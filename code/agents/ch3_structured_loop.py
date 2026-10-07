"""Chapter 3: a complete, modern structured-tool loop, in one file, against the real OpenAI API (gpt-5-mini, reasoning_effort=minimal).
It is the same ReAct shape as ch3_react.py, rebuilt the way production loops are built today:
  - native tool calling: the model returns structured tool calls, not text for a regular expression to parse;
  - every tool has a JSON Schema, and the loop VALIDATES the arguments before running anything (a small validator below covers the
    keywords used here; in a real system use the `jsonschema` package, or the provider's strict mode plus your own semantic checks);
  - an explicit state object (RunState) holds everything the run knows: messages, step and token counts, calls seen, status;
  - two budgets: model steps and total tokens; hitting either ends the run with a named status, never silently;
  - error handling: invalid JSON, unknown tools, schema violations and tool exceptions come back to the model as tool results it can
    act on; API errors are retried with a back-off; a repeated identical call gets a "result unchanged, answer now" note (the guard);
  - a trace: one JSON line per event in results/ch3_structured_trace.jsonl, and a readable version on stdout.
The question is the running Northwind example. Afterwards the same loop answers the twelve questions of ch3_react.py three times each,
graded by the same typed grader. The key comes from a .env outside the repo and is never printed."""
import json, os, re, sys, time
from dataclasses import dataclass, field
from common import Log, save, RESULTS
from ch3_react import QUESTIONS, grade, wiki_v2, calc

MODEL = 'gpt-5-mini'
PRICE_IN, PRICE_OUT = 0.25, 2.00          # USD per million tokens
MAX_STEPS = 8                             # model calls per run
MAX_TOKENS = 20_000                       # input + output tokens per run
REPEATS = 3

# ---- 1. tools: a JSON Schema for the arguments, and the function that runs them


def calculate(expression):
    out = calc(expression)                    # the calculator of ch3_react.py returns 'Error: ...' as text; here an error is an error
    if out.startswith('Error'):
        raise ValueError(out[len('Error: '):])
    return out


TOOLS = {
    'lookup_entry': dict(
        description='Look up ONE entry in the company encyclopaedia by its name (a person, a place, a depot or a company). '
                    'Returns the entry text and the names of related entries you can look up next.',
        schema={'type': 'object', 'properties': {'name': {'type': 'string', 'minLength': 2, 'maxLength': 60}},
                'required': ['name'], 'additionalProperties': False},
        run=lambda a: wiki_v2(a['name'])),
    'calculate': dict(
        description='Evaluate an arithmetic expression with numbers and + - * / ( ) only, for example "2020 - 2017".',
        schema={'type': 'object', 'properties': {'expression': {'type': 'string', 'maxLength': 80, 'pattern': r'^[0-9.+\-*/() ]+$'}},
                'required': ['expression'], 'additionalProperties': False},
        run=lambda a: calculate(a['expression'])),
    'final_answer': dict(
        description='Give the final answer. `answer` is only the answer, as short as possible, with no working. '
                    '`evidence` lists the entry names the answer rests on.',
        schema={'type': 'object', 'properties': {'answer': {'type': 'string', 'minLength': 1, 'maxLength': 120},
                                                 'evidence': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1}},
                'required': ['answer', 'evidence'], 'additionalProperties': False},
        run=None),
}
API_TOOLS = [{'type': 'function', 'function': {'name': n, 'description': t['description'], 'parameters': t['schema']}} for n, t in TOOLS.items()]
SYSTEM = ('You answer questions about Northwind Freight and a few world facts using the tools. Look facts up; never guess a fact. '
          'Use calculate for every arithmetic step. When you have the answer, call final_answer.')


def validate(value, schema, path='args'):
    """A small JSON Schema validator for the keywords used above. Returns a list of error strings (empty = valid)."""
    errs, t = [], schema.get('type')
    types = {'object': dict, 'string': str, 'array': list, 'number': (int, float)}
    if t and not isinstance(value, types[t]):
        return [f'{path} must be of type {t}']
    if t == 'object':
        for k in schema.get('required', []):
            if k not in value:
                errs.append(f'{path}.{k} is required')
        for k, v in value.items():
            if k in schema.get('properties', {}):
                errs += validate(v, schema['properties'][k], f'{path}.{k}')
            elif schema.get('additionalProperties') is False:
                errs.append(f'{path}.{k} is not allowed')
    if t == 'string':
        if len(value) < schema.get('minLength', 0):
            errs.append(f'{path} is shorter than {schema["minLength"]} characters')
        if len(value) > schema.get('maxLength', 10 ** 9):
            errs.append(f'{path} is longer than {schema["maxLength"]} characters')
        if 'pattern' in schema and not re.search(schema['pattern'], value):
            errs.append(f'{path} does not match {schema["pattern"]}')
    if t == 'array':
        if len(value) < schema.get('minItems', 0):
            errs.append(f'{path} needs at least {schema["minItems"]} item(s)')
        for i, v in enumerate(value):
            errs += validate(v, schema.get('items', {}), f'{path}[{i}]')
    return errs


def execute(name, raw_args):
    """Run one tool call safely. Every failure becomes a result the model can read and act on; nothing raises."""
    try:
        args = json.loads(raw_args or '{}')
    except json.JSONDecodeError as e:
        return {'ok': False, 'error': f'arguments are not valid JSON ({e.msg}); send a JSON object'}, None
    if name not in TOOLS:
        return {'ok': False, 'error': f'unknown tool {name!r}; the tools are {", ".join(TOOLS)}'}, args
    errs = validate(args, TOOLS[name]['schema'])
    if errs:
        return {'ok': False, 'error': 'invalid arguments: ' + '; '.join(errs)}, args
    if TOOLS[name]['run'] is None:
        return {'ok': True, 'result': 'answer recorded'}, args
    try:
        return {'ok': True, 'result': TOOLS[name]['run'](args)}, args
    except Exception as e:                                         # a tool bug must not crash the loop
        return {'ok': False, 'error': f'the tool failed: {type(e).__name__}: {e}'}, args


# ---- 2. the explicit state of one run
@dataclass
class RunState:
    question: str
    messages: list = field(default_factory=list)
    steps: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    seen: dict = field(default_factory=dict)                       # (tool, canonical args) -> result: the repeated-call guard
    status: str = 'running'                                       # running | answered | step_budget | token_budget | api_error
    answer: str = None
    evidence: list = None
    events: list = field(default_factory=list)                    # the trace

    def event(self, kind, **kw):
        self.events.append({'step': self.steps, 'kind': kind, **kw})


_client = None
USAGE = {'input': 0, 'output': 0, 'calls': 0}


def call_model(state):
    """One model call with retries and back-off. Returns the message, or None after three failures."""
    global _client
    if _client is None:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv('/Users/admin/Desktop/Learning/rag_learning/.env')     # OPENAI_API_KEY, outside the repo, never printed
        _client = OpenAI()
    for attempt in range(3):
        try:
            r = _client.chat.completions.create(model=MODEL, messages=state.messages, tools=API_TOOLS, tool_choice='auto',
                                                max_completion_tokens=600, reasoning_effort='minimal')
            state.tokens_in += r.usage.prompt_tokens; state.tokens_out += r.usage.completion_tokens
            USAGE['input'] += r.usage.prompt_tokens; USAGE['output'] += r.usage.completion_tokens; USAGE['calls'] += 1
            return r.choices[0].message
        except Exception as e:
            state.event('api_error', attempt=attempt + 1, error=f'{type(e).__name__}: {str(e)[:80]}')
            time.sleep(2 ** attempt)
    return None


# ---- 3. the loop
def run(question):
    s = RunState(question, messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': question}])
    while s.status == 'running':
        if s.steps >= MAX_STEPS:
            s.status = 'step_budget'; break
        if s.tokens_in + s.tokens_out >= MAX_TOKENS:
            s.status = 'token_budget'; break
        msg = call_model(s)
        s.steps += 1
        if msg is None:
            s.status = 'api_error'; break
        calls = msg.tool_calls or []
        s.event('model', tool_calls=len(calls), text=(msg.content or '')[:100], tokens=s.tokens_in + s.tokens_out)
        s.messages.append({'role': 'assistant', 'content': msg.content,
                           **({'tool_calls': [{'id': c.id, 'type': 'function', 'function': {'name': c.function.name, 'arguments': c.function.arguments}}
                                               for c in calls]} if calls else {})})
        if not calls:                                              # plain text instead of a tool call: say what is expected
            s.messages.append({'role': 'user', 'content': 'Reply with a tool call: look something up, calculate, or call final_answer.'})
            continue
        for c in calls:
            s.tool_calls += 1
            result, args = execute(c.function.name, c.function.arguments)
            key = (c.function.name, json.dumps(args, sort_keys=True))
            if result['ok'] and c.function.name != 'final_answer':
                if key in s.seen:                                  # the guard: same call, same arguments
                    result = {'ok': True, 'result': s.seen[key], 'note': 'repeated call, result unchanged; answer with what you have'}
                s.seen[key] = result['result']
            if not result['ok']:
                s.tool_errors += 1
            if result['ok'] and c.function.name == 'final_answer':
                s.answer, s.evidence, s.status = args['answer'], args['evidence'], 'answered'
            s.event('tool', tool=c.function.name, args=args, ok=result['ok'], result=str(result.get('result', result.get('error')))[:200],
                    note=result.get('note'))
            s.messages.append({'role': 'tool', 'tool_call_id': c.id, 'content': json.dumps(result)})
    s.event('end', status=s.status, answer=s.answer)
    return s


def write_trace(s, path):
    with open(path, 'a') as f:
        for e in s.events:
            f.write(json.dumps({'question': s.question, **e}) + '\n')


ERROR_CASES = [('lookup_entry', '{"name": ""}'), ('lookup_entry', '{"name": "Porto depot", "year": 2017}'),
               ('calculate', '{"expression": "120 trucks - 41"}'), ('calculate', '{"expression": "1/0"}'),
               ('lookup_entry', '{name: Porto}'), ('delete_depot', '{"name": "Leeds depot"}')]
DEMO_Q = 'Which Northwind Freight depot has the most trucks, and how many more trucks does it have than the Leeds depot?'
DEMO_EXPECTED = dict(entities=['rotterdam'], numbers=[79])


def main():
    trace_path = os.path.join(RESULTS, 'ch3_structured_trace.jsonl')
    open(trace_path, 'w').close()
    s = run(DEMO_Q)
    write_trace(s, trace_path)
    rows = []
    for rep in range(REPEATS):
        for i, (qq, exp) in enumerate(QUESTIONS):
            r = run(qq)
            write_trace(r, trace_path)
            rows.append({'rep': rep, 'i': i, 'answer': r.answer, 'ok': grade(r.answer or '', exp), 'status': r.status, 'steps': r.steps,
                         'tool_calls': r.tool_calls, 'tool_errors': r.tool_errors, 'tokens': r.tokens_in + r.tokens_out})
    save('ch3_structured_loop', {'model': MODEL, 'max_steps': MAX_STEPS, 'max_tokens': MAX_TOKENS, 'repeats': REPEATS, 'demo': {
        'question': DEMO_Q, 'answer': s.answer, 'evidence': s.evidence, 'ok': grade(s.answer or '', DEMO_EXPECTED), 'status': s.status,
        'steps': s.steps, 'tool_calls': s.tool_calls, 'tool_errors': s.tool_errors, 'tokens': s.tokens_in + s.tokens_out, 'events': s.events},
        'rows': rows, 'usage': USAGE})
    report()


def report():
    """The readable log, rendered from the saved results (python ch3_structured_loop.py --report re-renders it, no API calls)."""
    R = json.load(open(os.path.join(RESULTS, 'ch3_structured_loop.json')))
    log = Log('ch3_structured_loop')
    log(f'== A structured-tool loop: native tool calls, schema-checked arguments, explicit state, budgets, a trace ({R["model"]}) ==')
    log('-- error paths, exercised directly: each becomes a tool result the model can read; nothing crashes the loop --')
    for name, raw in ERROR_CASES:
        res, _ = execute(name, raw)
        msg = res.get('error') or res.get('result')
        log(f'  {name:<12} {raw:<38} -> {msg if len(msg) <= 74 else msg[:71] + "..."}')
    d = R['demo']
    log('')
    log('-- the Northwind question --')
    log(f'   {d["question"]}')
    for e in d['events']:
        if e['kind'] == 'model':
            log(f'step {e["step"]}  model -> {e["tool_calls"]} tool call(s)   (tokens so far {e["tokens"]:,})')
        elif e['kind'] == 'tool':
            a = json.dumps(e['args'], ensure_ascii=False)
            a = a if len(a) <= 44 else a[:41] + '...'
            r = e['result'] if len(e['result']) <= 56 else e['result'][:53] + '...'
            log(f'        {e["tool"]:<12} {a:<44} {"ok " if e["ok"] else "ERR"} {r}' + ('  [guard]' if e.get('note') else ''))
        elif e['kind'] == 'api_error':
            log(f'        API error, attempt {e["attempt"]}: {e["error"]}')
    log(f'status: {d["status"]} after {d["steps"]} model calls, {d["tool_calls"]} tool calls ({d["tool_errors"]} errors), {d["tokens"]:,} tokens; '
        f'budgets {R["max_steps"]} calls, {R["max_tokens"]:,} tokens')
    log(f'final_answer: {d["answer"]!r}')
    log(f'   evidence {d["evidence"]}; typed grader: {"correct" if d["ok"] else "wrong"}')
    rows = R['rows']
    reps = R['repeats']
    n_ok = sum(r['ok'] for r in rows)
    statuses = {k: sum(1 for r in rows if r['status'] == k) for k in sorted({r['status'] for r in rows})}
    pub = sum(r['ok'] for r in rows if r['i'] < 6) / (6 * reps); prv = sum(r['ok'] for r in rows if r['i'] >= 6) / (6 * reps)
    log('')
    log(f'-- the same loop on the twelve questions of ch3_react.py, {reps} repeats, the same typed exact grader --')
    log(f'correct {n_ok}/{len(rows)} ({n_ok / len(rows):.0%}); public {pub:.0%}, private {prv:.0%}; end status {statuses}')
    log(f'mean model calls per run {sum(r["steps"] for r in rows) / len(rows):.1f}; mean tokens per run {sum(r["tokens"] for r in rows) / len(rows):,.0f}; '
        f'tool errors {sum(r["tool_errors"] for r in rows)}')
    for r in [r for r in rows if not r['ok']][:4]:
        log(f'  wrong: q{r["i"] + 1} repeat {r["rep"] + 1}: {str(r["answer"])[:70]!r}')
    u = R['usage']
    cost = u['input'] / 1e6 * PRICE_IN + u['output'] / 1e6 * PRICE_OUT
    log(f'total spend this run: ${cost:.4f} ({u["calls"]} calls, {u["input"]:,} input + {u["output"]:,} output tokens)')
    log('one JSON line per event: results/ch3_structured_trace.jsonl')
    R.update({'acc': n_ok / len(rows), 'n_ok': n_ok, 'n': len(rows), 'acc_public': pub, 'acc_private': prv, 'statuses': statuses, 'cost_usd': cost})
    save('ch3_structured_loop', R)


if __name__ == '__main__':
    report() if '--report' in sys.argv else main()
