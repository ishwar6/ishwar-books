"""Chapter 3: reflection with an EXTERNAL check against reflection by self-critique alone, on ten small coding tasks with hidden
unit tests, using the real OpenAI API (gpt-5-mini, reasoning_effort=minimal). Each task is attempted five times (pass@1 is a rate over
50 first attempts). Every FAILED first attempt is continued in two ways from the same starting point:
  self-only   the model is asked to review and improve its own code, WITHOUT seeing the test output (the setting of Huang et al.);
  reflexion   the failing assertions are fed back as text and the model tries again (the Reflexion shape, with tests as the critic).
Up to three attempts in total per branch. Tests run in a subprocess with a timeout. The key comes from a .env outside the repo and is never printed.
If the API fails twice the script switches to a labelled stub and the log says so."""
import json, os, re, subprocess, sys, tempfile, time
from common import Log, save

log = Log('ch3_reflexion')
PRICE_IN, PRICE_OUT = 0.25, 2.00          # USD per million tokens, gpt-5-mini list price at the time of writing
MODEL = 'gpt-5-mini'
MAX_ATTEMPTS = 3

# ---- ten tasks: a spec the model sees, and hidden tests. Every rule the tests check is stated in the spec.
TASKS = [
    dict(name='eval_expr', spec="""Write eval_expr(s: str) -> float. Evaluate an arithmetic expression with + - * / ^, unary minus, parentheses and
non-negative integer or decimal numbers (e.g. "2.5"). Precedence from lowest to highest: + -, then * /, then unary minus, then ^ (right-associative).
So "-2^2" is -4.0, "2^3^2" is 512.0 and "2*-3" is -6.0. Spaces between tokens are allowed. Division by zero raises ZeroDivisionError.
Anything malformed (empty string, missing operand, two numbers in a row, unbalanced parentheses, unknown characters) raises ValueError. Return a float.""",
         tests=['assert eval_expr("1+2*3") == 7.0', 'assert eval_expr("2^3^2") == 512.0', 'assert eval_expr("-2^2") == -4.0', 'assert eval_expr("2*-3") == -6.0',
                'assert eval_expr("(1+2)*3") == 9.0', 'assert eval_expr("10/4") == 2.5', 'assert eval_expr(" 7 - 2 - 1 ") == 4.0', 'assert eval_expr("2.5*2") == 5.0',
                'try:\n    eval_expr("1/0"); raise AssertionError("expected ZeroDivisionError")\nexcept ZeroDivisionError:\n    pass',
                'for bad in ["", "1+", "2 3", "(1+2", "1 $ 2", "*1"]:\n    _raises(eval_expr, bad)']),
    dict(name='parse_csv_line', spec="""Write parse_csv_line(s: str) -> list[str] for ONE line of CSV in the RFC 4180 style. Fields are separated by commas.
A field may be wrapped in double quotes; inside a quoted field a comma is literal and a double quote is written as two double quotes ("").
Quotes may only wrap a whole field: a quote appearing inside an unquoted field, or any character after the closing quote other than a comma, raises ValueError.
An unterminated quoted field raises ValueError. Spaces are part of the field (do not strip). The empty string is one empty field: [""]. A trailing comma means a final empty field.""",
         tests=['assert parse_csv_line("a,b,c") == ["a", "b", "c"]', 'assert parse_csv_line(\'a,"b,c",d\') == ["a", "b,c", "d"]', 'assert parse_csv_line(\'"a""b"\') == [\'a"b\']',
                'assert parse_csv_line("") == [""]', 'assert parse_csv_line("a,,b") == ["a", "", "b"]', 'assert parse_csv_line("a, b") == ["a", " b"]', 'assert parse_csv_line(",") == ["", ""]',
                'assert parse_csv_line("a,") == ["a", ""]', 'assert parse_csv_line(\'""\') == [""]',
                'for bad in [\'"a"b\', \'"abc\', \'a"b\', \'"a",b"\']:\n    _raises(parse_csv_line, bad)']),
    dict(name='TTLCache', spec="""Write class TTLCache(capacity: int, ttl: float, clock) where clock is a zero-argument function returning the current time in seconds.
Methods: set(key, value), get(key) -> value or None, and __len__. An entry expires when clock() >= the time it was set plus ttl (expiry is checked lazily, on every call).
Expired entries are removed before anything else happens in set, get and __len__, so they never count toward capacity or len().
When set is called and the cache is full (after removing expired entries), evict the least recently used entry; both get and set count as a use.
Setting an existing key updates its value, makes it most recently used and restarts its ttl. capacity < 1 raises ValueError.""",
         tests=['t = [0.0]\nclock = lambda: t[0]\nc = TTLCache(2, 10, clock)\nc.set("a", 1); c.set("b", 2)\nassert c.get("a") == 1\nc.set("c", 3)\nassert c.get("b") is None and c.get("a") == 1 and c.get("c") == 3',
                't = [0.0]\nclock = lambda: t[0]\nc = TTLCache(2, 10, clock)\nc.set("a", 1)\nt[0] = 10.0\nassert c.get("a") is None and len(c) == 0',
                't = [0.0]\nclock = lambda: t[0]\nc = TTLCache(1, 10, clock)\nc.set("a", 1)\nt[0] = 10.0\nc.set("b", 2)\nassert len(c) == 1 and c.get("b") == 2',
                't = [0.0]\nclock = lambda: t[0]\nc = TTLCache(2, 10, clock)\nc.set("a", 1); c.set("b", 2)\nt[0] = 5.0\nc.set("a", 9)\nt[0] = 12.0\nassert c.get("a") == 9 and c.get("b") is None',
                't = [0.0]\nclock = lambda: t[0]\nc = TTLCache(2, 10, clock)\nc.set("a", 1)\nt[0] = 9.9\nassert c.get("a") == 1\nt[0] = 10.0\nassert len(c) == 0',
                't = [0.0]\nclock = lambda: t[0]\nc = TTLCache(2, 10, clock)\nc.set("a", 1); c.set("b", 2); c.set("a", 3); c.set("c", 4)\nassert c.get("b") is None and c.get("a") == 3',
                '_raises(TTLCache, 0, 10, lambda: 0.0)']),
    dict(name='cron_matches', spec="""Write cron_matches(expr: str, minute: int, hour: int, dom: int, month: int, dow: int) -> bool for a five-field cron expression
"minute hour day-of-month month day-of-week". Each field is "*", a number, a range "a-b", a list "1,2,5", or any of these with a step "/n" (e.g. "*/15", "1-10/3").
Ranges: minute 0-59, hour 0-23, day-of-month 1-31, month 1-12, day-of-week 0-7 where both 0 and 7 mean Sunday. A value outside its range, a descending range,
a step below 1 or a field with anything else raises ValueError, as does an expression without exactly five fields.
Standard cron rule: if BOTH day-of-month and day-of-week are restricted (neither is "*"), the date matches when EITHER of them matches; otherwise both must match as usual.""",
         tests=['assert cron_matches("*/15 * * * *", 30, 3, 1, 1, 0) is True', 'assert cron_matches("*/15 * * * *", 10, 3, 1, 1, 0) is False',
                'assert cron_matches("0 9-17 * * 1-5", 0, 12, 10, 6, 3) is True', 'assert cron_matches("0 9-17 * * 1-5", 0, 12, 10, 6, 0) is False',
                'assert cron_matches("0 0 * * 7", 0, 0, 10, 6, 0) is True', 'assert cron_matches("0 0 1 * 0", 0, 0, 15, 6, 0) is True', 'assert cron_matches("0 0 1 * 0", 0, 0, 1, 6, 3) is True',
                'assert cron_matches("0 0 1 * 0", 0, 0, 15, 6, 3) is False', 'assert cron_matches("1-5/2 * * * *", 3, 0, 1, 1, 0) is True', 'assert cron_matches("1-5/2 * * * *", 2, 0, 1, 1, 0) is False',
                'assert cron_matches("0 0 * 1,6 *", 0, 0, 1, 6, 0) is True',
                'for bad in ["60 * * * *", "* * * 13 *", "* * * * 8", "5-3 * * * *", "*/0 * * * *", "* * * *", "a * * * *"]:\n    _raises(cron_matches, bad, 0, 0, 1, 1, 0)']),
    dict(name='slugify', spec="""Write slugify(s: str, max_len: int | None = None) -> str. Steps: transliterate accented letters to plain ASCII letters (NFKD decomposition,
then drop every non-ASCII character), lower-case, replace every run of characters other than a-z and 0-9 with a single hyphen, strip hyphens from both ends.
If max_len is given and the result is longer, cut it at the last hyphen at or before position max_len so that no word is cut in the middle; if there is no such
hyphen (the first word alone is too long), cut the first word hard at max_len. Strip hyphens from the ends again after cutting. max_len < 1 raises ValueError.
If the result is empty, return "n-a".""",
         tests=['assert slugify("Hello, World!") == "hello-world"', 'assert slugify("  --Creme Brulee--  ") == "creme-brulee"', 'assert slugify("Cr\\u00e8me Br\\u00fbl\\u00e9e") == "creme-brulee"',
                'assert slugify("a---b") == "a-b"', 'assert slugify("!!!") == "n-a"', 'assert slugify("hello wonderful world", max_len=10) == "hello"',
                'assert slugify("hello wonderful world", max_len=15) == "hello-wonderful"', 'assert slugify("supercalifragilistic", max_len=5) == "super"',
                'assert slugify("ab  cd") == "ab-cd"', 'assert slugify("Ab1-2c") == "ab1-2c"', '_raises(slugify, "x", 0)']),
    dict(name='osa_distance', spec="""Write osa_distance(a: str, b: str) -> int, the optimal string alignment distance: the minimum number of insertions, deletions,
substitutions and transpositions of two ADJACENT characters needed to turn a into b, where each operation costs 1 and, unlike the full Damerau-Levenshtein distance,
no substring may be edited more than once (so osa_distance("ca", "abc") is 3, not 2). Empty strings are allowed.""",
         tests=['assert osa_distance("kitten", "sitting") == 3', 'assert osa_distance("ab", "ba") == 1', 'assert osa_distance("", "abc") == 3', 'assert osa_distance("abc", "") == 3',
                'assert osa_distance("abcd", "acbd") == 1', 'assert osa_distance("ca", "abc") == 3', 'assert osa_distance("same", "same") == 0', 'assert osa_distance("a", "b") == 1',
                'assert osa_distance("abcdef", "abcfed") == 2']),
    dict(name='number_to_words', spec="""Write number_to_words(n: int) -> str for 0 <= n < 1,000,000 in British English. Rules: 0 is "zero"; 21 is "twenty-one" (hyphen);
"and" goes between hundreds and the rest ("one hundred and one", "one hundred and fifteen") and between thousands and a remainder below one hundred
("one thousand and one", "two thousand and five"); when both a hundreds part and a thousands part are present, separate them with a comma
("nine hundred and ninety-nine thousand, nine hundred and ninety-nine"); no "and" inside the thousands group except its own hundreds ("one hundred and one thousand").
Exact multiples: "one thousand", "thirty thousand", "one hundred". Negative numbers, n >= 1,000,000 or a non-int raise ValueError.""",
         tests=['assert number_to_words(0) == "zero"', 'assert number_to_words(21) == "twenty-one"', 'assert number_to_words(100) == "one hundred"', 'assert number_to_words(101) == "one hundred and one"',
                'assert number_to_words(115) == "one hundred and fifteen"', 'assert number_to_words(1000) == "one thousand"', 'assert number_to_words(1001) == "one thousand and one"',
                'assert number_to_words(2005) == "two thousand and five"', 'assert number_to_words(30000) == "thirty thousand"', 'assert number_to_words(1100) == "one thousand, one hundred"',
                'assert number_to_words(999999) == "nine hundred and ninety-nine thousand, nine hundred and ninety-nine"', 'assert number_to_words(101000) == "one hundred and one thousand"',
                'assert number_to_words(40) == "forty"', 'for bad in [-1, 1000000, 2.5]:\n    _raises(number_to_words, bad)']),
    dict(name='topological_sort', spec="""Write topological_sort(deps: dict[str, list[str]]) -> list[str]. deps maps a node to the nodes it depends on (which must come before it).
Nodes that appear only inside a dependency list are nodes too. Return an order in which every node comes after all its dependencies; when several nodes are
available at the same time, always pick the alphabetically smallest first (Kahn's algorithm with a sorted frontier). A cycle, including a node that depends on itself,
raises ValueError whose message contains the word "cycle". An empty dict returns [].""",
         tests=['assert topological_sort({"b": ["a"]}) == ["a", "b"]', 'assert topological_sort({"a": [], "b": [], "c": ["a", "b"]}) == ["a", "b", "c"]',
                'assert topological_sort({"c": ["b"], "b": ["a"]}) == ["a", "b", "c"]', 'assert topological_sort({"x": ["y"]}) == ["y", "x"]',
                'assert topological_sort({"b": [], "a": ["b"]}) == ["b", "a"]', 'assert topological_sort({}) == []',
                'assert topological_sort({"d": ["b", "c"], "c": ["a"], "b": ["a"], "e": []}) == ["a", "b", "c", "d", "e"]',
                'for bad in [{"a": ["b"], "b": ["a"]}, {"a": ["a"]}]:\n    try:\n        topological_sort(bad); raise AssertionError("expected ValueError")\n    except ValueError as e:\n        assert "cycle" in str(e)']),
    dict(name='roman_to_int', spec="""Write roman_to_int(s: str) -> int for Roman numerals from 1 to 3999 using I V X L C D M.
Rules: the input must be in canonical form, otherwise raise ValueError. Canonical means: no more than three of I, X, C or M in a row;
V, L and D never repeated; subtractive pairs only IV, IX, XL, XC, CD, CM; "IIII", "VV", "IC", "IM", "XCX", "IXI" and lower-case input are invalid; the empty string is invalid.""",
         tests=['assert roman_to_int("III") == 3', 'assert roman_to_int("IV") == 4', 'assert roman_to_int("MCMXCIV") == 1994', 'assert roman_to_int("MMMCMXCIX") == 3999',
                'assert roman_to_int("XL") == 40', 'assert roman_to_int("LVIII") == 58',
                'for bad in ["IIII", "VV", "IC", "IM", "iv", "", "MMMM", "XCX", "IXI", "VX", "LL"]:\n    _raises(roman_to_int, bad)']),
    dict(name='parse_version', spec="""Write parse_version(v: str) -> tuple. Parse semantic versions such as "1.2.3", "1.2.3-beta.1", "1.2.3+build.5" and "1.2.3-rc.1+build.5".
Return (major, minor, patch, prerelease, build) where major/minor/patch are ints, prerelease is a tuple of dot-separated identifiers (purely numeric identifiers converted to int,
others kept as str; empty tuple if absent) and build is the raw build string (empty string if absent). A leading "v" is allowed and ignored.
Raise ValueError for: fewer or more than three numeric parts, a numeric part with a leading zero such as "01" (but "0" is fine), a negative number, an empty identifier
(e.g. "1.2.3-" or "1.2.3-beta..1"), or any other malformed input.""",
         tests=['assert parse_version("1.2.3") == (1, 2, 3, (), "")', 'assert parse_version("v1.2.3") == (1, 2, 3, (), "")', 'assert parse_version("1.2.3-beta.1") == (1, 2, 3, ("beta", 1), "")',
                'assert parse_version("1.2.3+build.5") == (1, 2, 3, (), "build.5")', 'assert parse_version("1.2.3-rc.1+build.5") == (1, 2, 3, ("rc", 1), "build.5")',
                'assert parse_version("0.0.0") == (0, 0, 0, (), "")', 'assert parse_version("10.20.30-alpha") == (10, 20, 30, ("alpha",), "")',
                'for bad in ["1.2", "1.2.3.4", "01.2.3", "1.2.3-", "a.b.c", "", "1.2.-3", "1.2.3-beta..1", "1.2.3+"]:\n    _raises(parse_version, bad)']),
]
SAMPLES = 5     # first attempts per task (sampling at the default temperature), so pass@1 is a rate over 50 attempts, not 10 coin flips

# ---- the API, with usage recorded and a labelled stub fallback
USAGE = {'input': 0, 'output': 0, 'calls': 0}
STUB = False
client = None


def model(messages, max_tokens=2000):
    global STUB, client
    if STUB:
        return 'def _stub():\n    pass'
    if client is None:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv('/Users/admin/Desktop/Learning/rag_learning/.env')
        client = OpenAI()
    for attempt in range(2):
        try:
            r = client.chat.completions.create(model=MODEL, messages=messages, max_completion_tokens=max_tokens, reasoning_effort='minimal')
            USAGE['input'] += r.usage.prompt_tokens; USAGE['output'] += r.usage.completion_tokens; USAGE['calls'] += 1
            return (r.choices[0].message.content or '').strip()
        except Exception as e:
            log(f'  API error ({type(e).__name__}), attempt {attempt + 1}: {str(e)[:80]}')
            time.sleep(2)
    STUB = True
    log('  *** API failed twice: switching to a STUB. These results are NOT real. ***')
    return 'def _stub():\n    pass'


def extract_code(text):
    m = re.findall(r'```(?:python)?\n(.*?)```', text, re.S)
    return '\n\n'.join(m) if m else text


def run_tests(code, tests):
    """Run the hidden tests in a fresh interpreter with a timeout. Returns (passed, failure_text)."""
    helper = ('def _raises(f, *a):\n    try:\n        f(*a)\n    except ValueError:\n        return\n'
              '    raise AssertionError(f"expected ValueError for {a!r}, got a result")\n\n')
    body = helper + code + '\n\n' + '\n'.join(f'# test {i + 1}\n' + t for i, t in enumerate(tests)) + '\nprint("ALL TESTS PASSED")\n'
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(body); path = f.name
    try:
        p = subprocess.run([sys.executable, '-I', path], capture_output=True, text=True, timeout=10)
        out = (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        out = 'TimeoutError: the tests did not finish within 10 seconds'
    finally:
        os.unlink(path)
    if 'ALL TESTS PASSED' in out:
        return True, ''
    # which test failed? the traceback names the line in the temporary file; map it back to a "# test i" block
    src_lines = body.split('\n')
    nums = [int(m) for m in re.findall(r'line (\d+)', out)]
    block = None
    for n in reversed(nums):
        for k in range(min(n, len(src_lines)) - 1, -1, -1):
            if src_lines[k].startswith('# test '):
                block = int(src_lines[k].split()[2]); break
        if block:
            break
    err = next((l for l in reversed(out.strip().split('\n')) if l.strip()), out[-200:])      # the last line: the error type and message
    if block:
        return False, f'failing test:\n{tests[block - 1]}\n-> {err[:300]}'
    return False, err[-400:]


SYSTEM = 'You write correct Python 3 functions. Reply with one ```python code block containing only the function(s) and any imports. No tests, no prose.'


def branch(msgs, reply, fail, task, arm):
    """Continue from a failed first attempt in one of two ways. Returns the attempt number that passed, or None."""
    msgs = msgs + [{'role': 'assistant', 'content': reply}]
    for n in range(2, MAX_ATTEMPTS + 1):
        if arm == 'self-only':      # Huang et al.: the model is asked to review itself, with no external signal
            msgs.append({'role': 'user', 'content': 'Your function may contain a bug. Carefully review it against the specification, '
                                                    'find any mistake, and reply with the corrected code block.'})
        else:                       # reflexion: the failing assertions go back as text
            msgs.append({'role': 'user', 'content': 'Your code failed these hidden tests:\n' + fail +
                                                    '\n\nFirst write one or two sentences on what went wrong, then reply with the corrected code block.'})
        reply = model(msgs)
        ok, fail = run_tests(extract_code(reply), task['tests'])
        msgs.append({'role': 'assistant', 'content': reply})
        if ok:
            return n
    return None


log(f'== Reflection with an external check vs self-critique alone: {len(TASKS)} tasks x {SAMPLES} samples, {MODEL}, up to {MAX_ATTEMPTS} attempts ==')
rows = []
first_fail_text = {}
for t in TASKS:
    for k in range(SAMPLES):
        msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': t['spec']}]
        reply = model(msgs)
        ok, fail = run_tests(extract_code(reply), t['tests'])
        row = {'task': t['name'], 'sample': k, 'pass1': ok, 'self_only': 1 if ok else None, 'reflexion': 1 if ok else None, 'first_failure': fail}
        if not ok:
            first_fail_text.setdefault(t['name'], fail)
            u0 = dict(USAGE)
            row['self_only'] = branch(msgs, reply, fail, t, 'self-only')
            row['reflexion'] = branch(msgs, reply, fail, t, 'reflexion')
        rows.append(row)

log('')
log(f'-- per task: how many of the {SAMPLES} samples passed on attempt 1, after self-critique only, after tests fed back (<= {MAX_ATTEMPTS} attempts) --')
log(f'{"task":<17} {"pass@1":>7} {"self-only":>10} {"reflexion":>10}   first failing test fed back (reflexion arm)')
for t in TASKS:
    rs = [r for r in rows if r['task'] == t['name']]
    p1 = sum(r['pass1'] for r in rs); so = sum(1 for r in rs if r['self_only']); rf = sum(1 for r in rs if r['reflexion'])
    fb = first_fail_text.get(t['name'], '').replace('failing test:\n', '').replace('\n', ' | ')[:66]
    log(f'{t["name"]:<17} {p1:>5}/{SAMPLES} {so:>8}/{SAMPLES} {rf:>8}/{SAMPLES}   {fb}')

n = len(rows)
p1 = sum(r['pass1'] for r in rows); so = sum(1 for r in rows if r['self_only']); rf = sum(1 for r in rows if r['reflexion'])
fails = [r for r in rows if not r['pass1']]
so_fix = sum(1 for r in fails if r['self_only']); rf_fix = sum(1 for r in fails if r['reflexion'])
log('')
log('-- summary --')
log(f'first attempts: {n}; passed on attempt 1: {p1} ({p1 / n:.0%})')
log(f'of the {len(fails)} failures: self-critique alone fixed {so_fix}, tests fed back fixed {rf_fix} (up to {MAX_ATTEMPTS - 1} more attempts each)')
log(f'pass rate within {MAX_ATTEMPTS} attempts: self-only {so}/{n} ({so / n:.0%}), reflexion {rf}/{n} ({rf / n:.0%})')
cost = USAGE['input'] / 1e6 * PRICE_IN + USAGE['output'] / 1e6 * PRICE_OUT
log(f'total spend this run: ${cost:.4f} ({USAGE["calls"]} calls, {USAGE["input"]:,} input + {USAGE["output"]:,} output tokens)')
log('stub used: ' + ('YES, results are not real' if STUB else 'no, every answer came from the API'))
summary = {'n': n, 'pass1': p1, 'self_only': so, 'reflexion': rf, 'failures': len(fails), 'self_only_fixed': so_fix, 'reflexion_fixed': rf_fix,
           'cost_usd': cost, 'calls': USAGE['calls']}
save('ch3_reflexion', {'model': MODEL, 'stub': STUB, 'summary': summary, 'rows': rows, 'usage': USAGE, 'max_attempts': MAX_ATTEMPTS, 'samples': SAMPLES,
                       'tasks': [t['name'] for t in TASKS]})
