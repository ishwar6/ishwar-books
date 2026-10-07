"""Chapter 3: six system configurations on twelve two-hop questions, with the REAL OpenAI API (gpt-5-mini, reasoning_effort=minimal so
that the visible text is the reasoning we study). This is a comparison of SYSTEM CONFIGURATIONS, not of loop shapes alone: the
conditions differ in tool access, prompt and demonstration, completion budget, observation quality and the repeated-action guard.
  direct        one call, answer only (no tools)
  cot           one call, "think step by step" (no tools)
  react         the ReAct text loop with the original wiki tool, no guard
  react+guard   the same loop and tool, plus a repeated-action guard
  react+tool    the same loop, no guard, with the improved wiki tool (related entries, several matches, helpful misses)
  react+both    the improved tool and the guard (the "ReAct+" of earlier drafts)
Each condition runs REPEATS times over the twelve questions (the model samples at its default temperature), so every rate is over 36 runs.
Grading is TYPED and exact: numbers are parsed and compared as numbers, entities are normalised explicitly, and an answer must contain
exactly the expected numbers (no more, no fewer). A self-test at the start shows the cases the earlier substring grader got wrong.
The key is loaded with python-dotenv from a .env file outside the repo and never printed. Token usage comes from the API responses.
If the API fails twice the script falls back to a clearly labelled stub; the log says which happened."""
import json, os, re, sys, time, threading, unicodedata
from concurrent.futures import ThreadPoolExecutor
from common import Log, save

PRICE_IN, PRICE_OUT = 0.25, 2.00          # USD per million tokens, gpt-5-mini list price at the time of writing
MODEL = 'gpt-5-mini'
REPEATS = 3

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

# ---- twelve questions with TYPED expected answers: the numbers the answer must contain (exactly), and the entities it must name.
QUESTIONS = [
    ('How many years passed between the completion of the Eiffel Tower and the opening of the Sydney Opera House?', dict(numbers=[84])),
    ('How many metres taller is Mount Everest than Mount Kilimanjaro?', dict(numbers=[2954])),
    ('How many years after the launch of Sputnik 1 did Apollo 11 land on the Moon?', dict(numbers=[12])),
    ('How many centuries, rounded down, separate the first teaching at the University of Oxford from the founding of Harvard University?', dict(numbers=[5])),
    ('How many kilometres longer is the Amazon River than the Danube?', dict(numbers=[3550])),
    ('By how many million people does the population of the Tokyo metropolis exceed that of London (one decimal place)?', dict(numbers=[5.1])),
    ('Who is the chief executive of Northwind Freight, and in which city was Northwind Freight founded?', dict(entities=['mira takacs', 'rotterdam'])),
    ('How many trucks do the Rotterdam and Gdansk depots of Northwind Freight have together?', dict(numbers=[205])),
    ('Which company did the chief executive of Northwind Freight work for before, and in which year was that company founded?', dict(entities=['harbour rail'], numbers=[2004])),
    ('How many years after the Porto depot of Northwind Freight opened did the Leeds depot open?', dict(numbers=[3])),
    ('In which city was the manager of the Rotterdam depot of Northwind Freight born?', dict(entities=['utrecht'])),
    ('How many locomotives did Harbour Rail have, and how many years after its founding was it acquired by Northwind Freight?', dict(numbers=[48, 17])),
]

# ---- the typed grader
WORDS = {w: i for i, w in enumerate('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen '
                                    'sixteen seventeen eighteen nineteen twenty'.split())}
WORDS.update({'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90})


def normalise(s):
    """Explicit entity normalisation: strip accents (NFKD), lower-case, turn every non-alphanumeric run into one space."""
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()
    return ' '.join(re.sub(r'[^a-z0-9.]+', ' ', s).replace('. ', ' ').split()).strip('.')


def numbers_in(s):
    """Every number in the answer, as floats: 2,954 -> 2954; 5.1 -> 5.1; 'three' -> 3. Years, counts and decimals alike."""
    s = unicodedata.normalize('NFKD', s).lower()
    out = [float(m.replace(',', '')) for m in re.findall(r'(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.,])\d+(?:\.\d+)?', s)]
    out += [float(WORDS[w]) for w in re.findall(r'[a-z]+', s) if w in WORDS]
    return out


def grade(answer, expected):
    """Correct only if the answer contains EXACTLY the expected numbers (as a multiset, compared numerically) and names every
    expected entity as a whole phrase after normalisation. 'It is not 84; the answer is 74.' contains two numbers: wrong."""
    got = sorted(numbers_in(answer))
    want = sorted(float(x) for x in expected.get('numbers', []))
    if len(got) != len(want) or any(abs(g - w) > 1e-9 for g, w in zip(got, want)):
        return False
    text = f' {normalise(answer)} '
    return all(f' {normalise(e)} ' in text for e in expected.get('entities', []))


def grade_substring_old(answer, accepted):
    """The grader of the first draft, kept ONLY for the self-test: a token-substring match, which accepts '184' for 84."""
    norm = lambda s: re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9. ]+', ' ', s.lower().replace(',', ''))).strip().rstrip('.')
    a = norm(answer)
    return any(all(tok in a.split() or tok in a for tok in norm(x).split()) for x in accepted)


SELF_TEST = [  # (answer, typed expectation, the first draft's accepted strings, intended verdict)
    ('184', dict(numbers=[84]), ['84'], False), ('13', dict(numbers=[3]), ['3', 'three'], False),
    ('It is not 84; the answer is 74.', dict(numbers=[84]), ['84'], False), ('84 years', dict(numbers=[84]), ['84'], True),
    ('2,954 m', dict(numbers=[2954]), ['2954', '2,954'], True), ('three', dict(numbers=[3]), ['3', 'three'], True),
    ('5.1 million', dict(numbers=[5.1]), ['5.1'], True), ('Mira Takacs; Rotterdam', dict(entities=['mira takacs', 'rotterdam']), ['mira takacs rotterdam'], True),
    ('Mira Takacs', dict(entities=['mira takacs', 'rotterdam']), ['mira takacs rotterdam'], False),
    ('48 locomotives; acquired 17 years later', dict(numbers=[48, 17]), ['48 17'], True)]


def self_test(log):
    log('-- grader self-test: the typed exact grader against the first draft\'s substring grader --')
    log(f'{"answer":<42} {"expected":<24} {"old":<6} {"typed":<6} as intended?')
    bad = 0
    show = lambda v: 'right' if v else 'wrong'
    for ans, exp, old_acc, want in SELF_TEST:
        old, new = grade_substring_old(ans, old_acc), grade(ans, exp)
        bad += new != want
        e = ' + '.join([*exp.get('entities', []), *[f'{x:g}' for x in exp.get('numbers', [])]])
        log(f'{ans!r:<42} {e:<24} {show(old):<6} {show(new):<6} {"yes" if new == want else "NO"}')
    log(f'self-test: {len(SELF_TEST) - bad}/{len(SELF_TEST)} verdicts as intended')
    assert bad == 0, 'grader self-test failed'

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

# ---- the model, through the API, with usage recorded (thread-safe) and a stub fallback
USAGE = {'input': 0, 'output': 0, 'reasoning': 0, 'calls': 0}
LOCK = threading.Lock()
STUB = False
_client = None


def client():
    global _client
    if _client is None:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv('/Users/admin/Desktop/Learning/rag_learning/.env')     # OPENAI_API_KEY, outside the repo, never printed
        _client = OpenAI()
    return _client


def model(prompt, max_tokens, usage):
    """One completion. Adds its tokens to the global USAGE and to the caller's usage dict. Stub after two failures."""
    global STUB
    if STUB:
        return 'Action: finish[stub: no API]'
    for attempt in range(2):
        try:
            r = client().chat.completions.create(model=MODEL, messages=[{'role': 'user', 'content': prompt}],
                                                 max_completion_tokens=max_tokens, reasoning_effort='minimal')
            u = r.usage
            rt = (u.completion_tokens_details.reasoning_tokens or 0) if u.completion_tokens_details else 0
            with LOCK:
                for d in (USAGE, usage):
                    d['input'] += u.prompt_tokens; d['output'] += u.completion_tokens; d['calls'] += 1; d['reasoning'] += rt
            return (r.choices[0].message.content or '').strip()
        except Exception as e:
            print(f'  API error ({type(e).__name__}), attempt {attempt + 1}: {str(e)[:80]}')
            time.sleep(2)
    STUB = True
    print('  *** API failed twice: switching to a STUB that answers nothing. These results are NOT real. ***')
    return 'Action: finish[stub: no API]'


# ---- the conditions
def run_direct(q, usage):
    a = model(DIRECT.format(q=q), 60, usage)
    return a, [a], {}


def run_cot(q, usage):
    out = model(COT.format(q=q), 400, usage)
    m = re.search(r'final answer:\s*(.*)', out, re.I | re.S)
    return (m.group(1).strip() if m else out.strip().split('\n')[-1]), [out], {}


def run_react(q, usage, tools=None, guard=False, max_steps=7):
    """The paper's text loop: one Thought and one Action per call, parsed with a regular expression; the loop runs the tool and
    appends the Observation. With guard=True the loop notices an action it has already run and tells the model to finish."""
    tools = tools or TOOLS
    prompt = REACT.format(q=q)
    trace, seen, repeats = [], {}, 0
    for step in range(max_steps):                                  # the step budget
        out = model(prompt, 200, usage)
        m = re.search(r'(.*?Action:\s*(\w+)\[(.*?)\])', out, re.S)  # keep the first Thought/Action pair only
        if not m:
            trace.append(f'(no action parsed) {out[:80]}')
            prompt += out + '\nObservation: Your reply had no Action line. Write one Thought and one Action.\n'
            continue
        chunk, tool, arg = m.group(1).strip(), m.group(2), m.group(3)
        trace.append(chunk)
        if tool == 'finish':
            return arg.strip(), trace, {'steps': step + 1, 'repeats': repeats, 'budget_hit': False}
        obs = tools[tool](arg) if tool in tools else f'Unknown tool {tool}. Use wiki[...] or calc[...].'
        if (tool, arg) in seen:
            repeats += 1
            if guard:                                              # the loop, not the model, notices the repeat
                obs = (f'(repeated action, result unchanged: {seen[(tool, arg)]}) You have been round this loop once. Using ONLY the '
                       'observations above, write a final Thought and then Action: finish[answer] now.')
        else:
            seen[(tool, arg)] = obs
        trace.append(f'Observation: {obs}')
        prompt += chunk + f'\nObservation: {obs}\n'
    return '(step budget exhausted)', trace, {'steps': max_steps, 'repeats': repeats, 'budget_hit': True}


CONDITIONS = [
    ('direct', 'direct answer', run_direct),
    ('cot', 'chain of thought', run_cot),
    ('react', 'ReAct, original tool, no guard', lambda q, u: run_react(q, u)),
    ('react+guard', 'ReAct + guard only', lambda q, u: run_react(q, u, guard=True)),
    ('react+tool', 'ReAct + improved tool only', lambda q, u: run_react(q, u, tools=TOOLS_V2)),
    ('react+both', 'ReAct + guard + improved tool', lambda q, u: run_react(q, u, tools=TOOLS_V2, guard=True)),
]


def main():
    log = Log('ch3_react')
    log(f'== Six system configurations, {len(QUESTIONS)} two-hop questions x {REPEATS} repeats, {MODEL} (reasoning_effort=minimal) ==')
    self_test(log)
    results, per_usage = {}, {}
    for key, _, fn in CONDITIONS:
        usage = {'input': 0, 'output': 0, 'reasoning': 0, 'calls': 0}
        jobs = [(rep, i) for rep in range(REPEATS) for i in range(len(QUESTIONS))]

        def one(job):
            rep, i = job
            q, exp = QUESTIONS[i]
            ans, trace, info = fn(q, usage)
            return {'rep': rep, 'i': i, 'answer': ans, 'ok': grade(ans, exp), 'trace': trace, **info}
        with ThreadPoolExecutor(8) as ex:
            results[key] = sorted(ex.map(one, jobs), key=lambda r: (r['rep'], r['i']))
        per_usage[key] = usage
        print(f'  {key}: done, {sum(r["ok"] for r in results[key])}/{len(jobs)} correct')

    # ---- per-question table: correct runs out of REPEATS
    log('')
    log(f'-- correct runs out of {REPEATS}, per question (1-6 public facts the model may know; 7-12 a fictional company it cannot know) --')
    log(f'{"#":>2} {"kind":<8} ' + ' '.join(f'{k:<11}' for k, _, _ in CONDITIONS) + ' question')
    for i, (q, _) in enumerate(QUESTIONS):
        cells = ' '.join(f'{sum(r["ok"] for r in results[k] if r["i"] == i)}/{REPEATS}{"":<8}' for k, _, _ in CONDITIONS)
        log(f'{i + 1:>2} {"public" if i < 6 else "private":<8} {cells} {q[:34]}')

    # ---- summary
    log('')
    log(f'-- summary over {REPEATS * len(QUESTIONS)} runs per condition (tokens and cost from the API usage fields; ${PRICE_IN} in, ${PRICE_OUT} out per M) --')
    log(f'{"condition":<12} {"acc":>5} {"range":>9} {"public":>7} {"private":>8} {"calls":>6} {"in tok":>8} {"out tok":>8} {"cost $":>7} '
        f'{"$/correct":>9} {"budget":>6} {"rep.":>5}')
    summary = {}
    for key, label, _ in CONDITIONS:
        r = results[key]
        n_ok = sum(x['ok'] for x in r)
        per_rep = [sum(x['ok'] for x in r if x['rep'] == k) / len(QUESTIONS) for k in range(REPEATS)]
        u = per_usage[key]
        cost = u['input'] / 1e6 * PRICE_IN + u['output'] / 1e6 * PRICE_OUT
        pub = sum(x['ok'] for x in r if x['i'] < 6) / (6 * REPEATS)
        prv = sum(x['ok'] for x in r if x['i'] >= 6) / (6 * REPEATS)
        budget = sum(1 for x in r if x.get('budget_hit'))
        reps = sum(1 for x in r if x.get('repeats'))
        summary[key] = {'label': label, 'acc': n_ok / len(r), 'n_ok': n_ok, 'n': len(r), 'acc_per_repeat': per_rep, 'acc_public': pub,
                        'acc_private': prv, 'calls': u['calls'], 'input_tokens': u['input'], 'output_tokens': u['output'],
                        'reasoning_tokens': u['reasoning'], 'cost_usd': cost, 'cost_per_correct': (cost / n_ok if n_ok else None),
                        'budget_hit': budget, 'runs_with_repeat': reps}
        rng = f'{min(per_rep):.0%}-{max(per_rep):.0%}'
        log(f'{key:<12} {n_ok / len(r):>5.0%} {rng:>9} {pub:>7.0%} {prv:>8.0%} {u["calls"]:>6} {u["input"]:>8,} {u["output"]:>8,} {cost:>7.4f} '
            f'{(cost / n_ok if n_ok else float("nan")):>9.5f} {budget:>6} {reps:>5}')
    log('budget = runs that hit the 7-step budget; rep. = runs in which the model repeated an action it had already run')

    # ---- answers that the strict grader marked wrong although they contain the expected numbers among others
    strict = []
    for key, _, _ in CONDITIONS:
        for x in results[key]:
            exp = QUESTIONS[x['i']][1]
            nums = numbers_in(x['answer'])
            if not x['ok'] and exp.get('numbers') and all(any(abs(n - w) < 1e-9 for n in nums) for w in exp['numbers']):
                strict.append({'condition': key, 'q': x['i'] + 1, 'answer': x['answer'][:80]})
    log(f'answers marked wrong only because they also contain other numbers: {len(strict)}' +
        (f', e.g. {strict[0]["condition"]} q{strict[0]["q"]}: {strict[0]["answer"][:52]!r}' if strict else ''))
    total = USAGE['input'] / 1e6 * PRICE_IN + USAGE['output'] / 1e6 * PRICE_OUT
    log(f'total spend this run: ${total:.4f} ({USAGE["calls"]} calls, {USAGE["input"]:,} input + {USAGE["output"]:,} output tokens, '
        f'{USAGE["reasoning"]} hidden reasoning tokens)')
    log('stub used: ' + ('YES, results are not real' if STUB else 'no, every answer came from the API'))

    save('ch3_react', {'model': MODEL, 'price_in': PRICE_IN, 'price_out': PRICE_OUT, 'stub': STUB, 'repeats': REPEATS,
                       'conditions': [[k, l] for k, l, _ in CONDITIONS], 'summary': summary, 'results': results, 'usage': USAGE,
                       'strict_only_wrong': strict, 'questions': QUESTIONS, 'grader_self_test': [list(t) for t in SELF_TEST]})
    write_traces()


def write_traces():
    """The trace log, rendered from the saved results (python ch3_react.py --traces re-renders it without new API calls)."""
    R = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'ch3_react.json')))
    res = R['results']
    tlog = Log('ch3_react_trace')

    def show(r, limit=None):
        tlog(f'Question: {QUESTIONS[r["i"]][0]}')
        lines = [l for t in r['trace'] for l in t.split('\n')]
        for line in lines[:limit] if limit else lines:
            tlog('  ' + (line if len(line) <= 126 else line[:122] + ' ...'))
        if limit and len(lines) > limit:
            tlog(f'  ... {len(lines) - limit} more lines: it starts the same look-ups again until the step budget runs out')
        tlog(f'=> answer: {r["answer"]}  ({"correct" if r["ok"] else "wrong"})')
    tlog(f'== ReAct traces, {R["model"]} (reasoning_effort=minimal) ==')
    stuck = next((x for x in res['react'] if x.get('budget_hit') and x.get('repeats')), None)
    if stuck:
        tlog('')
        tlog(f'-- plain ReAct, question {stuck["i"] + 1} (repeat {stuck["rep"] + 1}): the answer is in the context; the model starts again --')
        show(stuck, limit=12)
    good = next((x for x in res['react+both'] if x['i'] == 9 and x['ok']), None)
    if good:
        tlog('')
        tlog(f'-- ReAct + guard + improved tool, question 10 (repeat {good["rep"] + 1}): two private look-ups and a subtraction --')
        show(good)
    tlog('')
    tlog('-- the same question, direct and chain of thought (repeat 1) --')
    d = next(x for x in res['direct'] if x['i'] == 9 and x['rep'] == 0)
    c = next(x for x in res['cot'] if x['i'] == 9 and x['rep'] == 0)
    tlog(f'  direct: {d["answer"][:100]}  ({"correct" if d["ok"] else "wrong"})')
    cot = c['trace'][0].replace(chr(10), ' ')
    tlog(f'  cot:    {cot[:110]} ...')
    tlog(f'          final answer line: {c["answer"][:80]}  ({"correct" if c["ok"] else "wrong"})')


if __name__ == '__main__':
    write_traces() if '--traces' in sys.argv else main()
