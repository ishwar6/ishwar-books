"""Chapter 3: direct answer vs chain of thought vs ReAct on twelve multi-hop questions, with the REAL OpenAI API (gpt-5-mini,
reasoning_effort=minimal so that the visible text is the reasoning we study). Two tools: a tiny local "wiki" (a dict of short
entries, half of them about a fictional company the model cannot know) and a calculator. The key is loaded with python-dotenv
from a .env file outside the repo and never printed. Token usage comes from the API responses. If the API fails twice the
script falls back to a clearly labelled stub so the chapter can still be assembled; the log says which happened."""
import json, os, re, time
from common import Log, save

log = Log('ch3_react')
tlog = Log('ch3_react_trace')
PRICE_IN, PRICE_OUT = 0.25, 2.00          # USD per million tokens, gpt-5-mini list price at the time of writing (see the chapter)
MODEL = 'gpt-5-mini'

# ---- tool 1: a local lookup table. Real-world facts (the model may know them) and a fictional company (it cannot).
WIKI = {
    'eiffel tower': 'Wrought-iron tower in Paris, France. Completed in 1889. Height 330 m (1,083 ft). Designed by the company of Gustave Eiffel.',
    'sydney opera house': 'Performing arts centre in Sydney, Australia. Opened in 1973. Designed by Jorn Utzon.',
    'university of oxford': 'University in Oxford, England. Teaching existed in 1096; it is the oldest university in the English-speaking world.',
    'harvard university': 'University in Cambridge, Massachusetts, United States. Founded in 1636.',
    'mount everest': 'Highest mountain on Earth. Height 8,849 m. Located on the border of Nepal and China.',
    'mount kilimanjaro': 'Highest mountain in Africa. Height 5,895 m. Located in Tanzania.',
    'amazon river': 'River in South America. Length about 6,400 km. Discharges into the Atlantic Ocean.',
    'danube': 'River in Europe. Length 2,850 km. Flows through ten countries into the Black Sea.',
    'apollo 11': 'First crewed Moon landing. Landed 20 July 1969. Crew: Armstrong, Aldrin, Collins.',
    'sputnik 1': 'First artificial Earth satellite. Launched 4 October 1957 by the Soviet Union.',
    'tokyo': 'Capital of Japan. Population of the metropolis about 14.0 million (2023).',
    'london': 'Capital of the United Kingdom. Population about 8.9 million (2023).',
    # a fictional company: facts the model cannot have seen
    'northwind freight': 'Fictional logistics company founded in 2011 in Rotterdam. Chief executive: Mira Takacs. Depots: Rotterdam, Gdansk, Porto, Leeds.',
    'mira takacs': 'Fictional person. Chief executive of Northwind Freight since 2019. Previously chief operating officer of Harbour Rail.',
    'harbour rail': 'Fictional rail operator founded in 2004 in Antwerp. Fleet: 48 locomotives. Acquired by Northwind Freight in 2021.',
    'rotterdam depot': 'Northwind Freight depot opened 2011. Vehicles: 120 trucks. Manager: Joris Vos.',
    'gdansk depot': 'Northwind Freight depot opened 2014. Vehicles: 85 trucks. Manager: Ola Nowak.',
    'porto depot': 'Northwind Freight depot opened 2017. Vehicles: 64 trucks. Manager: Ines Carvalho.',
    'leeds depot': 'Northwind Freight depot opened 2020. Vehicles: 41 trucks. Manager: Sam Hartley.',
    'joris vos': 'Fictional person. Manager of the Rotterdam depot of Northwind Freight since 2016. Born in Utrecht.',
    'ola nowak': 'Fictional person. Manager of the Gdansk depot of Northwind Freight since 2018.',
}


def wiki(q):
    """Look up one entry by name (case-insensitive, partial match). Returns the entry or a not-found message with hints."""
    k = q.strip().lower().strip('"\'')
    if k in WIKI:
        return WIKI[k]
    hits = [n for n in WIKI if k in n or n in k]
    if len(hits) == 1:
        return WIKI[hits[0]]
    return 'Not found. ' + (f'Did you mean: {", ".join(hits)}?' if hits else 'Try a shorter name, for example the name of a place, person or thing.')


def calc(expr):
    """Evaluate an arithmetic expression (digits and + - * / ( ) . only)."""
    if not re.fullmatch(r'[0-9.+\-*/() ]+', expr):
        return 'Error: only numbers and + - * / ( ) are allowed.'
    try:
        return str(round(eval(expr), 4))      # safe: the regex above admits nothing but arithmetic
    except Exception as e:
        return f'Error: {e}'


RELATED = {'northwind freight': ['Rotterdam depot', 'Gdansk depot', 'Porto depot', 'Leeds depot', 'Mira Takacs', 'Harbour Rail'],
           'rotterdam depot': ['Joris Vos'], 'gdansk depot': ['Ola Nowak'], 'mira takacs': ['Harbour Rail', 'Northwind Freight'],
           'harbour rail': ['Northwind Freight']}          # the "links" of a wiki page: names the model can look up next


def wiki_v2(q):
    """Same table, better observations: an entry comes with the names of related entries (like links on a wiki page);
    several matches are returned instead of an error; a miss explains how entries are named."""
    k = q.strip().lower().strip('"\'')
    hits = [k] if k in WIKI else [n for n in WIKI if k in n or n in k]
    if len(hits) == 1:
        n = hits[0]
        return WIKI[n] + (f' Related entries: {", ".join(RELATED[n])}.' if n in RELATED else '')
    if hits:
        return 'Several entries match; here they are. ' + ' || '.join(f'{n}: {WIKI[n]}' for n in hits[:3])
    return 'Not found. Entries are named by ONE thing: a person ("Mira Takacs"), a place ("Tokyo"), a depot ("Porto depot") or a company. Try one name.'


TOOLS = {'wiki': wiki, 'calc': calc}
TOOLS_V2 = {'wiki': wiki_v2, 'calc': calc}

# ---- twelve questions: (question, accepted answers). Half need the fictional entries; most need two hops and some arithmetic.
QUESTIONS = [
    ('How many years passed between the completion of the Eiffel Tower and the opening of the Sydney Opera House?', ['84']),
    ('How many metres taller is Mount Everest than Mount Kilimanjaro?', ['2954', '2,954']),
    ('How many years after the launch of Sputnik 1 did Apollo 11 land on the Moon?', ['12', 'twelve']),
    ('How many centuries, rounded down, separate the first teaching at the University of Oxford from the founding of Harvard University?', ['5', 'five']),
    ('How many kilometres longer is the Amazon River than the Danube?', ['3550', '3,550']),
    ('By how many million people does the population of the Tokyo metropolis exceed that of London (one decimal place)?', ['5.1']),
    ('Who is the chief executive of Northwind Freight, and in which city was Northwind Freight founded?', ['mira takacs rotterdam', 'takacs rotterdam']),
    ('How many trucks do the Rotterdam and Gdansk depots of Northwind Freight have together?', ['205']),
    ('Which company did the chief executive of Northwind Freight work for before, and in which year was that company founded?', ['harbour rail 2004']),
    ('How many years after the Porto depot of Northwind Freight opened did the Leeds depot open?', ['3', 'three']),
    ('In which city was the manager of the Rotterdam depot of Northwind Freight born?', ['utrecht']),
    ('How many locomotives did Harbour Rail have, and how many years after its founding was it acquired by Northwind Freight?', ['48 17']),
]

# ---- the three prompts
DIRECT = 'Answer the question with only the final answer, no explanation.\nQuestion: {q}\nAnswer:'
COT = ('Answer the question. First think step by step in a few short sentences, then write the final answer on its own line '
       'starting with "Final answer:".\nQuestion: {q}')
REACT = """Answer the question by interleaving Thought, Action and Observation steps.
You have two tools:
  wiki[name]   looks up one entry about a place, person, company or thing in a small local encyclopaedia (one name per call).
  calc[expr]   evaluates an arithmetic expression such as calc[1973-1889].
Write exactly one Thought line and one Action line, then STOP and wait for the Observation (do not write it yourself).
When you know the answer, write Action: finish[answer] with only the answer, as short as possible.
Example:
Question: How many years after the Eiffel Tower was completed did Apollo 11 land on the Moon?
Thought: I need the completion year of the Eiffel Tower and the landing year of Apollo 11.
Action: wiki[Eiffel Tower]
Observation: Wrought-iron tower in Paris, France. Completed in 1889. Height 330 m ...
Thought: Completed in 1889. Now the Apollo 11 landing year.
Action: wiki[Apollo 11]
Observation: First crewed Moon landing. Landed 20 July 1969 ...
Thought: 1969 minus 1889.
Action: calc[1969-1889]
Observation: 80
Thought: The answer is 80 years.
Action: finish[80]

Question: {q}
"""

# ---- the model, through the API, with usage recorded and a stub fallback
USAGE = {'input': 0, 'output': 0, 'reasoning': 0, 'calls': 0}
STUB = False
client = None


def model(prompt, max_tokens=300):
    """One completion. Returns the text. Falls back to a labelled stub after two failures (and says so in the log)."""
    global STUB, client
    if STUB:
        return 'Action: finish[stub: no API]'
    if client is None:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv('/Users/admin/Desktop/Learning/rag_learning/.env')     # OPENAI_API_KEY, outside the repo, never printed
        client = OpenAI()
    for attempt in range(2):
        try:
            r = client.chat.completions.create(model=MODEL, messages=[{'role': 'user', 'content': prompt}],
                                               max_completion_tokens=max_tokens, reasoning_effort='minimal')
            u = r.usage
            USAGE['input'] += u.prompt_tokens; USAGE['output'] += u.completion_tokens; USAGE['calls'] += 1
            USAGE['reasoning'] += (u.completion_tokens_details.reasoning_tokens or 0) if u.completion_tokens_details else 0
            return (r.choices[0].message.content or '').strip()
        except Exception as e:
            log(f'  API error ({type(e).__name__}), attempt {attempt + 1}: {str(e)[:80]}')
            time.sleep(2)
    STUB = True
    log('  *** API failed twice: switching to a STUB that answers nothing. These results are NOT real. ***')
    return 'Action: finish[stub: no API]'


def norm(s):
    s = s.lower().replace(',', '').replace('.0 ', ' ')
    s = re.sub(r'\b(years?|metres?|meters?|m|km|kilometres?|million|people|trucks|locomotives|and|in|the|was|is|by|about|approximately)\b', ' ', s)
    s = re.sub(r'[^a-z0-9. ]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip().rstrip('.')


def correct(answer, accepted):
    a = norm(answer)
    return any(all(tok in a.split() or tok in a for tok in norm(x).split()) for x in accepted)


# ---- the three conditions
def run_direct(q):
    a = model(DIRECT.format(q=q), 60)
    return a, [a]


def run_cot(q):
    out = model(COT.format(q=q), 400)
    m = re.search(r'final answer:\s*(.*)', out, re.I | re.S)
    return (m.group(1).strip() if m else out.strip().split('\n')[-1]), [out]


def run_react_plus(q):
    return run_react(q, tools=TOOLS_V2, guard=True)


def run_react(q, max_steps=7, tools=TOOLS, guard=False):
    prompt = REACT.format(q=q)
    trace, seen = [], {}
    for step in range(max_steps):
        out = model(prompt, 200)
        # keep only the first Thought/Action pair; drop anything the model wrote after its Action line (e.g. an invented Observation)
        m = re.search(r'(.*?Action:\s*(\w+)\[(.*?)\])', out, re.S)
        if not m:
            trace.append(f'(no action parsed) {out[:80]}')
            prompt += out + '\nObservation: Your reply had no Action line. Write one Thought and one Action.\n'
            continue
        chunk, tool, arg = m.group(1).strip(), m.group(2), m.group(3)
        trace.append(chunk)
        if tool == 'finish':
            return arg.strip(), trace
        obs = tools[tool](arg) if tool in tools else f'Unknown tool {tool}. Use wiki[...] or calc[...].'
        if guard and (tool, arg) in seen:                      # the loop, not the model, notices a repeated action and forces the finish
            obs = (f'(repeated action, result unchanged: {seen[(tool, arg)]}) You have been round this loop once. Using ONLY the observations above, '
                   f'write a final Thought and then Action: finish[answer] now.')
        else:
            seen[(tool, arg)] = obs
        trace.append(f'Observation: {obs}')
        prompt += chunk + f'\nObservation: {obs}\n'
    return '(step budget exhausted)', trace


log(f'== Direct vs chain of thought vs ReAct: {len(QUESTIONS)} questions, model {MODEL} (reasoning_effort=minimal) ==')
results = {'direct': [], 'cot': [], 'react': [], 'react+': []}
per_cond_usage = {}
for name, fn in [('direct', run_direct), ('cot', run_cot), ('react', run_react), ('react+', run_react_plus)]:
    before = dict(USAGE)
    for i, (q, acc) in enumerate(QUESTIONS):
        ans, trace = fn(q)
        ok = correct(ans, acc)
        results[name].append({'q': q, 'answer': ans, 'ok': ok, 'trace': trace, 'steps': sum(1 for t in trace if t.startswith('Observation')) + 1})
    per_cond_usage[name] = {k: USAGE[k] - before[k] for k in USAGE}

# ---- traces (separate log): a repetition loop of plain ReAct, then a full ReAct+ trace on a private two-hop question
def show(tl, r, limit=None):
    tl(f'Question: {r["q"]}')
    lines = [l for t in r['trace'] for l in t.split('\n')]
    for line in lines[:limit] if limit else lines:
        tl('  ' + line[:118])
    if limit and len(lines) > limit:
        tl(f'  ... {len(lines) - limit} more lines, same action repeated until the step budget ran out')
    tl(f'=> answer: {r["answer"]}  ({"correct" if r["ok"] else "wrong"})')


tlog(f'== ReAct traces, {MODEL} (reasoning_effort=minimal) ==')
tlog('')
tlog('-- plain ReAct, question 3: the model has the answer after step 2 and repeats the action instead of finishing --')
show(tlog, results['react'][2], limit=8)
tlog('')
tlog('-- ReAct with a loop guard and a wiki that lists related entries, question 10 (two private look-ups and a subtraction) --')
show(tlog, results['react+'][9])
tlog('')
tlog('-- the same question, direct and chain of thought --')
tlog(f'  direct: {results["direct"][9]["answer"][:100]}  ({"correct" if results["direct"][9]["ok"] else "wrong"})')
cot_txt = results['cot'][9]['trace'][0].replace('\n', ' ')
tlog(f'  cot:    {cot_txt[:200]}  ({"correct" if results["cot"][9]["ok"] else "wrong"})')

# ---- per-question table
log('')
log('-- per question (public = facts the model may know; private = a fictional company it cannot know) --')
log(f'{"#":>2} {"kind":<8} {"direct":<7} {"cot":<7} {"react":<7} {"react+":<7} {"steps":>5} {"steps+":>6}  question')
for i, (q, _) in enumerate(QUESTIONS):
    kind = 'public' if i < 6 else 'private'
    cells = ' '.join(f'{"ok" if results[c][i]["ok"] else "wrong":<7}' for c in ['direct', 'cot', 'react', 'react+'])
    log(f'{i + 1:>2} {kind:<8} {cells} {results["react"][i]["steps"]:>5} {results["react+"][i]["steps"]:>6}  {q[:48]}')

# ---- the summary table: accuracy, tokens, cost (from the API usage fields)
log('')
log('-- summary (tokens and cost from the API usage fields; price per million: '
    f'${PRICE_IN} in, ${PRICE_OUT} out; react+ = a loop guard that forces the finish after a repeated action + a wiki that lists related entries) --')
log(f'{"condition":<9} {"acc all":>8} {"public":>7} {"private":>8} {"calls":>6} {"in tok":>8} {"out tok":>8} {"cost $":>8} {"$/correct":>10}')
summary = {}
for name in ['direct', 'cot', 'react', 'react+']:
    r = results[name]
    acc = sum(x['ok'] for x in r) / len(r)
    pub = sum(x['ok'] for x in r[:6]) / 6
    prv = sum(x['ok'] for x in r[6:]) / 6
    u = per_cond_usage[name]
    cost = u['input'] / 1e6 * PRICE_IN + u['output'] / 1e6 * PRICE_OUT
    n_ok = sum(x['ok'] for x in r)
    summary[name] = {'acc': acc, 'acc_public': pub, 'acc_private': prv, 'calls': u['calls'], 'input_tokens': u['input'],
                     'output_tokens': u['output'], 'reasoning_tokens': u['reasoning'], 'cost_usd': cost,
                     'cost_per_correct': (cost / n_ok if n_ok else None)}
    log(f'{name:<9} {acc:>8.0%} {pub:>7.0%} {prv:>8.0%} {u["calls"]:>6} {u["input"]:>8,} {u["output"]:>8,} {cost:>8.4f} '
        f'{(cost / n_ok if n_ok else float("nan")):>10.4f}')
log(f'total spend this run: ${USAGE["input"] / 1e6 * PRICE_IN + USAGE["output"] / 1e6 * PRICE_OUT:.4f} '
    f'({USAGE["calls"]} calls, {USAGE["input"]:,} input + {USAGE["output"]:,} output tokens, {USAGE["reasoning"]} hidden reasoning tokens)')
log('stub used: ' + ('YES, results are not real' if STUB else 'no, every answer came from the API'))
save('ch3_react', {'model': MODEL, 'price_in': PRICE_IN, 'price_out': PRICE_OUT, 'stub': STUB, 'summary': summary,
                   'results': results, 'usage': USAGE, 'questions': QUESTIONS})
