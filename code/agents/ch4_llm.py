"""Chapter 4: the one model call shared by the four workflow experiments (ch4_chain.py, ch4_router.py, ch4_parallel.py,
ch4_evaluator.py). Real OpenAI API, model gpt-5-mini. The key is loaded with python-dotenv from a .env file OUTSIDE the repo and is
never printed. Every call records its token usage (input, output, hidden reasoning) so the chapter's cost numbers come from the
API's own usage fields. If the API fails twice in a row the helper switches to a clearly labelled STUB that returns an empty answer,
and every script prints whether the stub was used. Thread-safe, because ch4_parallel.py fans calls out over threads."""
import json, re, threading, time

MODEL = 'gpt-5-mini'
PRICE_IN, PRICE_OUT = 0.25, 2.00        # USD per million tokens, gpt-5-mini list price at the time of writing (same as Chapter 3)
USAGE = {'input': 0, 'output': 0, 'reasoning': 0, 'calls': 0}
STATE = {'client': None, 'stub': False, 'log': print}
_lock = threading.Lock()


def set_log(fn):
    STATE['log'] = fn


def _client():
    if STATE['client'] is None:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv('/Users/admin/Desktop/Learning/rag_learning/.env')     # OPENAI_API_KEY, outside the repo, never printed
        STATE['client'] = OpenAI()
    return STATE['client']


def ask(messages, effort='minimal', max_tokens=800):
    """One chat completion. messages: a string (one user turn) or a list of {'role','content'} dicts.
    Returns (text, usage) where usage = {'input','output','reasoning'} for this call. Falls back to a labelled stub after two failures."""
    if isinstance(messages, str):
        messages = [{'role': 'user', 'content': messages}]
    if STATE['stub']:
        return '', {'input': 0, 'output': 0, 'reasoning': 0}
    client = _client()
    for attempt in range(2):
        try:
            r = client.chat.completions.create(model=MODEL, messages=messages, max_completion_tokens=max_tokens, reasoning_effort=effort)
            u = r.usage
            d = u.completion_tokens_details
            use = {'input': u.prompt_tokens, 'output': u.completion_tokens, 'reasoning': (d.reasoning_tokens or 0) if d else 0}
            with _lock:
                for k, v in use.items():
                    USAGE[k] += v
                USAGE['calls'] += 1
            return (r.choices[0].message.content or '').strip(), use
        except Exception as e:                                        # network, rate limit, bad request: try once more, then stub
            STATE['log'](f'  API error ({type(e).__name__}), attempt {attempt + 1}: {str(e)[:90]}')
            time.sleep(2)
    STATE['stub'] = True
    STATE['log']('  *** API failed twice: switching to a STUB that answers nothing. These results are NOT real. ***')
    return '', {'input': 0, 'output': 0, 'reasoning': 0}


def cost(use):
    """Dollars for a usage dict (output includes the hidden reasoning tokens, which are billed as output)."""
    return use['input'] / 1e6 * PRICE_IN + use['output'] / 1e6 * PRICE_OUT


def parse_json(text):
    """Return the first JSON object in the text, or None. Tolerates code fences and leading prose (a lenient parser, as in production)."""
    if not text:
        return None
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def snapshot():
    with _lock:
        return dict(USAGE)


def diff(before, after):
    return {k: after[k] - before[k] for k in before}


def stub_used():
    return STATE['stub']
