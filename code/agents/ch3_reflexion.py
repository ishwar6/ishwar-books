"""Chapter 3: repair strategies for small coding tasks, with the real OpenAI API (gpt-5-mini, reasoning_effort=minimal).
Ten tasks, five first attempts each (50). Every task has TWO test suites:
  feedback suite   the project's own tests: a repair loop may run them, show a failure to the model and stop when they pass;
  held-out suite   different inputs for the same rules in the spec; never shown to the model and never used to stop anything.
                   Every number in the summary is measured on the held-out suite.
Both suites are first validated against reference solutions written for this script (all must pass, or the run stops).
EVERY first attempt, right or wrong, then goes through two repair strategies, so wrong-to-right AND right-to-wrong changes are counted:
  self-review      two rounds of "review your function against the specification; fix it or return it unchanged", no test output;
  feedback tests   run the feedback suite; if it fails, show the first failing test and ask for a fix; up to two more attempts;
                   stop as soon as the feedback suite passes (the loop's own check, not the held-out one).
The test runner starts a separate `python -I` process with a 10-second timeout. That is a DEMONSTRATION runner, not a sandbox: the
generated code could still read and write files, use the network or exhaust memory. Run untrusted code in a real sandbox (a container
or VM with no network and resource limits). The key comes from a .env outside the repo and is never printed."""
import json, os, re, subprocess, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
from common import Log, save

PRICE_IN, PRICE_OUT = 0.25, 2.00          # USD per million tokens, gpt-5-mini list price at the time of writing
MODEL = 'gpt-5-mini'
SAMPLES = 5        # first attempts per task: rates are over 50 attempts
SELF_ROUNDS = 2    # self-review rounds, always run
FIX_ATTEMPTS = 2   # extra attempts allowed to the feedback-test loop

# ---- ten tasks: a spec the model sees and a FEEDBACK suite (the project's own tests: the loop may run them and show failures).
# Every rule the tests check is stated in the spec. The untouched HELD-OUT suites and the reference solutions follow below.
TASKS = [
    dict(name='eval_expr', spec="""Write eval_expr(s: str) -> float. Evaluate an arithmetic expression with + - * / ^, unary minus, parentheses and
non-negative integer or decimal numbers (e.g. "2.5"). Precedence from lowest to highest: + -, then * /, then unary minus, then ^ (right-associative).
So "-2^2" is -4.0, "2^3^2" is 512.0 and "2*-3" is -6.0. Spaces between tokens are allowed. Division by zero raises ZeroDivisionError.
Anything malformed (empty string, missing operand, two numbers in a row, unbalanced parentheses, unknown characters) raises ValueError. Return a float.""",
         feedback=['assert eval_expr("1+2*3") == 7.0', 'assert eval_expr("2^3^2") == 512.0', 'assert eval_expr("-2^2") == -4.0', 'assert eval_expr("2*-3") == -6.0',
                'assert eval_expr("(1+2)*3") == 9.0', 'assert eval_expr("10/4") == 2.5', 'assert eval_expr(" 7 - 2 - 1 ") == 4.0', 'assert eval_expr("2.5*2") == 5.0',
                'try:\n    eval_expr("1/0"); raise AssertionError("expected ZeroDivisionError")\nexcept ZeroDivisionError:\n    pass',
                'for bad in ["", "1+", "2 3", "(1+2", "1 $ 2", "*1"]:\n    _raises(eval_expr, bad)']),
    dict(name='parse_csv_line', spec="""Write parse_csv_line(s: str) -> list[str] for ONE line of CSV in the RFC 4180 style. Fields are separated by commas.
A field may be wrapped in double quotes; inside a quoted field a comma is literal and a double quote is written as two double quotes ("").
Quotes may only wrap a whole field: a quote appearing inside an unquoted field, or any character after the closing quote other than a comma, raises ValueError.
An unterminated quoted field raises ValueError. Spaces are part of the field (do not strip). The empty string is one empty field: [""]. A trailing comma means a final empty field.""",
         feedback=['assert parse_csv_line("a,b,c") == ["a", "b", "c"]', 'assert parse_csv_line(\'a,"b,c",d\') == ["a", "b,c", "d"]', 'assert parse_csv_line(\'"a""b"\') == [\'a"b\']',
                'assert parse_csv_line("") == [""]', 'assert parse_csv_line("a,,b") == ["a", "", "b"]', 'assert parse_csv_line("a, b") == ["a", " b"]', 'assert parse_csv_line(",") == ["", ""]',
                'assert parse_csv_line("a,") == ["a", ""]', 'assert parse_csv_line(\'""\') == [""]',
                'for bad in [\'"a"b\', \'"abc\', \'a"b\', \'"a",b"\']:\n    _raises(parse_csv_line, bad)']),
    dict(name='TTLCache', spec="""Write class TTLCache(capacity: int, ttl: float, clock) where clock is a zero-argument function returning the current time in seconds.
Methods: set(key, value), get(key) -> value or None, and __len__. An entry expires when clock() >= the time it was set plus ttl (expiry is checked lazily, on every call).
Expired entries are removed before anything else happens in set, get and __len__, so they never count toward capacity or len().
When set is called and the cache is full (after removing expired entries), evict the least recently used entry; both get and set count as a use.
Setting an existing key updates its value, makes it most recently used and restarts its ttl. capacity < 1 raises ValueError.""",
         feedback=['t = [0.0]\nclock = lambda: t[0]\nc = TTLCache(2, 10, clock)\nc.set("a", 1); c.set("b", 2)\nassert c.get("a") == 1\nc.set("c", 3)\nassert c.get("b") is None and c.get("a") == 1 and c.get("c") == 3',
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
         feedback=['assert cron_matches("*/15 * * * *", 30, 3, 1, 1, 0) is True', 'assert cron_matches("*/15 * * * *", 10, 3, 1, 1, 0) is False',
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
         feedback=['assert slugify("Hello, World!") == "hello-world"', 'assert slugify("  --Creme Brulee--  ") == "creme-brulee"', 'assert slugify("Cr\\u00e8me Br\\u00fbl\\u00e9e") == "creme-brulee"',
                'assert slugify("a---b") == "a-b"', 'assert slugify("!!!") == "n-a"', 'assert slugify("hello wonderful world", max_len=10) == "hello"',
                'assert slugify("hello wonderful world", max_len=15) == "hello-wonderful"', 'assert slugify("supercalifragilistic", max_len=5) == "super"',
                'assert slugify("ab  cd") == "ab-cd"', 'assert slugify("Ab1-2c") == "ab1-2c"', '_raises(slugify, "x", 0)']),
    dict(name='osa_distance', spec="""Write osa_distance(a: str, b: str) -> int, the optimal string alignment distance: the minimum number of insertions, deletions,
substitutions and transpositions of two ADJACENT characters needed to turn a into b, where each operation costs 1 and, unlike the full Damerau-Levenshtein distance,
no substring may be edited more than once (so osa_distance("ca", "abc") is 3, not 2). Empty strings are allowed.""",
         feedback=['assert osa_distance("kitten", "sitting") == 3', 'assert osa_distance("ab", "ba") == 1', 'assert osa_distance("", "abc") == 3', 'assert osa_distance("abc", "") == 3',
                'assert osa_distance("abcd", "acbd") == 1', 'assert osa_distance("ca", "abc") == 3', 'assert osa_distance("same", "same") == 0', 'assert osa_distance("a", "b") == 1',
                'assert osa_distance("abcdef", "abcfed") == 2']),
    dict(name='number_to_words', spec="""Write number_to_words(n: int) -> str for 0 <= n < 1,000,000 in British English. Rules: 0 is "zero"; 21 is "twenty-one" (hyphen);
"and" goes between hundreds and the rest ("one hundred and one", "one hundred and fifteen") and between thousands and a remainder below one hundred
("one thousand and one", "two thousand and five"); when both a hundreds part and a thousands part are present, separate them with a comma
("nine hundred and ninety-nine thousand, nine hundred and ninety-nine"); no "and" inside the thousands group except its own hundreds ("one hundred and one thousand").
Exact multiples: "one thousand", "thirty thousand", "one hundred". Negative numbers, n >= 1,000,000 or a non-int raise ValueError.""",
         feedback=['assert number_to_words(0) == "zero"', 'assert number_to_words(21) == "twenty-one"', 'assert number_to_words(100) == "one hundred"', 'assert number_to_words(101) == "one hundred and one"',
                'assert number_to_words(115) == "one hundred and fifteen"', 'assert number_to_words(1000) == "one thousand"', 'assert number_to_words(1001) == "one thousand and one"',
                'assert number_to_words(2005) == "two thousand and five"', 'assert number_to_words(30000) == "thirty thousand"', 'assert number_to_words(1100) == "one thousand, one hundred"',
                'assert number_to_words(999999) == "nine hundred and ninety-nine thousand, nine hundred and ninety-nine"', 'assert number_to_words(101000) == "one hundred and one thousand"',
                'assert number_to_words(40) == "forty"', 'for bad in [-1, 1000000, 2.5]:\n    _raises(number_to_words, bad)']),
    dict(name='topological_sort', spec="""Write topological_sort(deps: dict[str, list[str]]) -> list[str]. deps maps a node to the nodes it depends on (which must come before it).
Nodes that appear only inside a dependency list are nodes too. Return an order in which every node comes after all its dependencies; when several nodes are
available at the same time, always pick the alphabetically smallest first (Kahn's algorithm with a sorted frontier). A cycle, including a node that depends on itself,
raises ValueError whose message contains the word "cycle". An empty dict returns [].""",
         feedback=['assert topological_sort({"b": ["a"]}) == ["a", "b"]', 'assert topological_sort({"a": [], "b": [], "c": ["a", "b"]}) == ["a", "b", "c"]',
                'assert topological_sort({"c": ["b"], "b": ["a"]}) == ["a", "b", "c"]', 'assert topological_sort({"x": ["y"]}) == ["y", "x"]',
                'assert topological_sort({"b": [], "a": ["b"]}) == ["b", "a"]', 'assert topological_sort({}) == []',
                'assert topological_sort({"d": ["b", "c"], "c": ["a"], "b": ["a"], "e": []}) == ["a", "b", "c", "d", "e"]',
                'for bad in [{"a": ["b"], "b": ["a"]}, {"a": ["a"]}]:\n    try:\n        topological_sort(bad); raise AssertionError("expected ValueError")\n    except ValueError as e:\n        assert "cycle" in str(e)']),
    dict(name='roman_to_int', spec="""Write roman_to_int(s: str) -> int for Roman numerals from 1 to 3999 using I V X L C D M.
Rules: the input must be in canonical form, otherwise raise ValueError. Canonical means: no more than three of I, X, C or M in a row;
V, L and D never repeated; subtractive pairs only IV, IX, XL, XC, CD, CM; "IIII", "VV", "IC", "IM", "XCX", "IXI" and lower-case input are invalid; the empty string is invalid.""",
         feedback=['assert roman_to_int("III") == 3', 'assert roman_to_int("IV") == 4', 'assert roman_to_int("MCMXCIV") == 1994', 'assert roman_to_int("MMMCMXCIX") == 3999',
                'assert roman_to_int("XL") == 40', 'assert roman_to_int("LVIII") == 58',
                'for bad in ["IIII", "VV", "IC", "IM", "iv", "", "MMMM", "XCX", "IXI", "VX", "LL"]:\n    _raises(roman_to_int, bad)']),
    dict(name='parse_version', spec="""Write parse_version(v: str) -> tuple. Parse semantic versions such as "1.2.3", "1.2.3-beta.1", "1.2.3+build.5" and "1.2.3-rc.1+build.5".
Return (major, minor, patch, prerelease, build) where major/minor/patch are ints, prerelease is a tuple of dot-separated identifiers (purely numeric identifiers converted to int,
others kept as str; empty tuple if absent) and build is the raw build string (empty string if absent). A leading "v" is allowed and ignored.
Raise ValueError for: fewer or more than three numeric parts, a numeric part with a leading zero such as "01" (but "0" is fine), a negative number, an empty identifier
(e.g. "1.2.3-" or "1.2.3-beta..1"), or any other malformed input.""",
         feedback=['assert parse_version("1.2.3") == (1, 2, 3, (), "")', 'assert parse_version("v1.2.3") == (1, 2, 3, (), "")', 'assert parse_version("1.2.3-beta.1") == (1, 2, 3, ("beta", 1), "")',
                'assert parse_version("1.2.3+build.5") == (1, 2, 3, (), "build.5")', 'assert parse_version("1.2.3-rc.1+build.5") == (1, 2, 3, ("rc", 1), "build.5")',
                'assert parse_version("0.0.0") == (0, 0, 0, (), "")', 'assert parse_version("10.20.30-alpha") == (10, 20, 30, ("alpha",), "")',
                'for bad in ["1.2", "1.2.3.4", "01.2.3", "1.2.3-", "a.b.c", "", "1.2.-3", "1.2.3-beta..1", "1.2.3+"]:\n    _raises(parse_version, bad)']),
]

# ---- reference solutions (used only to check that every test is right) and the held-out suites
REFERENCE = {
'eval_expr': r'''
import re
def eval_expr(s):
    toks = re.findall(r'\d+(?:\.\d+)?|[-+*/^()]|\S', s)
    if not toks or any(not re.fullmatch(r'\d+(?:\.\d+)?|[-+*/^()]', t) for t in toks):
        raise ValueError('bad input')
    pos = [0]
    peek = lambda: toks[pos[0]] if pos[0] < len(toks) else None
    def take():
        t = peek()
        if t is None:
            raise ValueError('unexpected end')
        pos[0] += 1
        return t
    def expr():
        v = term()
        while peek() in ('+', '-'):
            op = take(); r = term(); v = v + r if op == '+' else v - r
        return v
    def term():
        v = unary()
        while peek() in ('*', '/'):
            op = take(); r = unary()
            if op == '*':
                v = v * r
            else:
                if r == 0:
                    raise ZeroDivisionError('division by zero')
                v = v / r
        return v
    def unary():
        if peek() == '-':
            take(); return -unary()
        return power()
    def power():
        b = primary()
        if peek() == '^':
            take(); return b ** unary()
        return b
    def primary():
        t = take()
        if t == '(':
            v = expr()
            if take() != ')':
                raise ValueError('expected )')
            return v
        if re.fullmatch(r'\d+(?:\.\d+)?', t):
            return float(t)
        raise ValueError('unexpected token')
    v = expr()
    if pos[0] != len(toks):
        raise ValueError('trailing tokens')
    return float(v)
''',
'parse_csv_line': r'''
def parse_csv_line(s):
    out, i, n = [], 0, len(s)
    while True:
        if i < n and s[i] == '"':
            i += 1; buf = []
            while True:
                if i >= n:
                    raise ValueError('unterminated')
                if s[i] == '"':
                    if i + 1 < n and s[i + 1] == '"':
                        buf.append('"'); i += 2; continue
                    i += 1; break
                buf.append(s[i]); i += 1
            out.append(''.join(buf))
            if i < n and s[i] != ',':
                raise ValueError('text after closing quote')
        else:
            j = s.find(',', i)
            j = n if j < 0 else j
            f = s[i:j]
            if '"' in f:
                raise ValueError('quote in unquoted field')
            out.append(f); i = j
        if i >= n:
            return out
        i += 1
        if i == n:
            out.append(''); return out
''',
'TTLCache': r'''
from collections import OrderedDict
class TTLCache:
    def __init__(self, capacity, ttl, clock):
        if capacity < 1:
            raise ValueError('capacity')
        self.cap, self.ttl, self.clock, self.d = capacity, ttl, clock, OrderedDict()
    def _purge(self):
        now = self.clock()
        for k in [k for k, (v, t) in self.d.items() if now >= t + self.ttl]:
            del self.d[k]
    def set(self, key, value):
        self._purge()
        if key in self.d:
            del self.d[key]
        elif len(self.d) >= self.cap:
            self.d.popitem(last=False)
        self.d[key] = (value, self.clock())
    def get(self, key):
        self._purge()
        if key not in self.d:
            return None
        self.d.move_to_end(key)
        return self.d[key][0]
    def __len__(self):
        self._purge()
        return len(self.d)
''',
'cron_matches': r'''
import re
def cron_matches(expr, minute, hour, dom, month, dow):
    f = expr.split()
    if len(f) != 5:
        raise ValueError('five fields')
    R = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 7)]
    def vals(field, lo, hi):
        out = set()
        for part in field.split(','):
            m = re.fullmatch(r'(\*|\d+|\d+-\d+)(?:/(\d+))?', part)
            if not m:
                raise ValueError('bad field')
            base, step = m.group(1), int(m.group(2)) if m.group(2) else 1
            if step < 1:
                raise ValueError('step')
            if base == '*':
                a, b = lo, hi
            elif '-' in base:
                a, b = map(int, base.split('-'))
            else:
                a = b = int(base)
            if a < lo or b > hi or a > b:
                raise ValueError('range')
            out.update(range(a, b + 1, step))
        return out
    sets = [vals(x, *r) for x, r in zip(f, R)]
    if 7 in sets[4]:
        sets[4].add(0)
    if 0 in sets[4]:
        sets[4].add(7)
    if minute not in sets[0] or hour not in sets[1] or month not in sets[3]:
        return False
    d_ok, w_ok = dom in sets[2], dow in sets[4]
    if f[2] != '*' and f[4] != '*':
        return d_ok or w_ok
    return d_ok and w_ok
''',
'slugify': r'''
import re, unicodedata
def slugify(s, max_len=None):
    if max_len is not None and max_len < 1:
        raise ValueError('max_len')
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r'[^a-z0-9]+', '-', s).strip('-')
    if max_len is not None and len(s) > max_len:
        cut = s.rfind('-', 0, max_len + 1)
        s = s[:cut] if cut > 0 else s[:max_len]
        s = s.strip('-')
    return s or 'n-a'
''',
'osa_distance': r'''
def osa_distance(a, b):
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[len(a)][len(b)]
''',
'number_to_words': r'''
def number_to_words(n):
    if type(n) is not int or n < 0 or n >= 1000000:
        raise ValueError('range')
    ones = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
    tens = 'x x twenty thirty forty fifty sixty seventy eighty ninety'.split()
    def two(k):
        return ones[k] if k < 20 else tens[k // 10] + ('-' + ones[k % 10] if k % 10 else '')
    def three(k):
        h, r = divmod(k, 100)
        if h and r:
            return ones[h] + ' hundred and ' + two(r)
        return ones[h] + ' hundred' if h else two(r)
    if n == 0:
        return 'zero'
    th, rest = divmod(n, 1000)
    parts = three(th) + ' thousand' if th else ''
    if rest:
        if not th:
            return three(rest)
        return parts + (' and ' + two(rest) if rest < 100 else ', ' + three(rest))
    return parts
''',
'topological_sort': r'''
import heapq
def topological_sort(deps):
    nodes = set(deps) | {d for v in deps.values() for d in v}
    indeg = {n: 0 for n in nodes}
    users = {n: [] for n in nodes}
    for n, ds in deps.items():
        for d in ds:
            indeg[n] += 1; users[d].append(n)
    h = [n for n in nodes if indeg[n] == 0]; heapq.heapify(h)
    out = []
    while h:
        n = heapq.heappop(h); out.append(n)
        for u in users[n]:
            indeg[u] -= 1
            if indeg[u] == 0:
                heapq.heappush(h, u)
    if len(out) != len(nodes):
        raise ValueError('cycle detected')
    return out
''',
'roman_to_int': r'''
import re
def roman_to_int(s):
    if not isinstance(s, str) or not s or not re.fullmatch(r'M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})', s):
        raise ValueError('not canonical')
    v = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    for i, c in enumerate(s):
        total += -v[c] if i + 1 < len(s) and v[s[i + 1]] > v[c] else v[c]
    return total
''',
'parse_version': r'''
import re
def parse_version(v):
    m = re.fullmatch(r'v?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?', v)
    if not m:
        raise ValueError('malformed')
    pre = tuple(int(x) if x.isdigit() else x for x in m.group(4).split('.')) if m.group(4) else ()
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), pre, m.group(5) or ''
''',
}

HELDOUT = {
'eval_expr': ['assert eval_expr("2^2^3") == 256.0', 'assert eval_expr("-(2+3)*2") == -10.0', 'assert eval_expr("((4))") == 4.0',
              'assert eval_expr("8/2/2") == 2.0', 'assert eval_expr("3-2^2") == -1.0', 'assert eval_expr("1.5+1.5") == 3.0', 'assert eval_expr("2 ^ 3") == 8.0', 'assert eval_expr("3*-2^2") == -12.0',
              'try:\n    eval_expr("5/(2-2)"); raise AssertionError("expected ZeroDivisionError")\nexcept ZeroDivisionError:\n    pass',
              'for bad in ["()", "1+*2", "1 2", "(1))", "3 # 4"]:\n    _raises(eval_expr, bad)'],
'parse_csv_line': ['assert parse_csv_line(\'a,"",b\') == ["a", "", "b"]', 'assert parse_csv_line(\'"x,y",z\') == ["x,y", "z"]',
                   'assert parse_csv_line(\'"a""""b"\') == [\'a""b\']', 'assert parse_csv_line(\'"","x"\') == ["", "x"]', 'assert parse_csv_line(" a ,b") == [" a ", "b"]',
                   'assert parse_csv_line("a,b,") == ["a", "b", ""]', 'assert parse_csv_line(",,") == ["", "", ""]', 'for bad in [\'"a" ,b\', \'ab"\', \'"a""\']:\n    _raises(parse_csv_line, bad)'],
'TTLCache': ['t = [0.0]\nclock = lambda: t[0]\nc = TTLCache(3, 5, clock)\nc.set("a", 1)\nt[0] = 3.0\nc.set("b", 2)\nt[0] = 5.0\nassert len(c) == 1 and c.get("b") == 2 and c.get("a") is None',
             't = [0.0]\nclock = lambda: t[0]\nc = TTLCache(2, 10, clock)\nc.set("a", 1)\nt[0] = 8.0\nc.set("a", 5)\nt[0] = 12.0\nassert c.get("a") == 5',
             't = [0.0]\nclock = lambda: t[0]\nc = TTLCache(2, 10, clock)\nc.set("a", 1); c.set("b", 2)\nassert c.get("b") == 2\nc.set("c", 3)\nassert c.get("a") is None and c.get("b") == 2 and c.get("c") == 3',
             't = [0.0]\nclock = lambda: t[0]\nc = TTLCache(2, 10, clock)\nc.set("a", 1)\nt[0] = 5.0\nc.set("b", 2)\nt[0] = 10.0\nc.set("c", 3)\nassert c.get("b") == 2 and c.get("c") == 3 and len(c) == 2',
             't = [0.0]\nclock = lambda: t[0]\nc = TTLCache(2, 10, clock)\nassert c.get("missing") is None and len(c) == 0',
             't = [0.0]\nclock = lambda: t[0]\nc = TTLCache(2, 10, clock)\nc.set("a", 1)\nt[0] = 6.0\nassert c.get("a") == 1\nt[0] = 10.0\nassert c.get("a") is None',
             '_raises(TTLCache, -1, 10, lambda: 0.0)'],
'cron_matches': ['assert cron_matches("30 2 * * *", 30, 2, 5, 5, 2) is True', 'assert cron_matches("30 2 * * *", 31, 2, 5, 5, 2) is False',
                 'assert cron_matches("0 12 * 12 *", 0, 12, 1, 12, 1) is True', 'assert cron_matches("0 12 * 12 *", 0, 12, 1, 11, 1) is False',
                 'assert cron_matches("*/20 */6 * * *", 40, 18, 3, 3, 3) is True', 'assert cron_matches("*/20 */6 * * *", 40, 19, 3, 3, 3) is False',
                 'assert cron_matches("0 0 15 * 1-5", 0, 0, 15, 6, 6) is True', 'assert cron_matches("0 0 15 * 1-5", 0, 0, 14, 6, 6) is False',
                 'assert cron_matches("0 0 15 * 1-5", 0, 0, 14, 6, 3) is True', 'assert cron_matches("5,10-12 * * * *", 11, 0, 1, 1, 1) is True',
                 'assert cron_matches("5,10-12 * * * *", 9, 0, 1, 1, 1) is False', 'assert cron_matches("* * 10-20/5 * *", 0, 0, 15, 1, 1) is True',
                 'assert cron_matches("* * 10-20/5 * *", 0, 0, 16, 1, 1) is False', 'assert cron_matches("0 0 * * 5-7", 0, 0, 1, 1, 0) is True',
                 'for bad in ["* 24 * * *", "* * 0 * *", "1-60 * * * *", "* * * * * *", "*/x * * * *", "-1 * * * *"]:\n    _raises(cron_matches, bad, 0, 0, 1, 1, 0)'],
'slugify': ['assert slugify("\\u00c7a va tr\\u00e8s bien") == "ca-va-tres-bien"', 'assert slugify("foo_bar baz") == "foo-bar-baz"',
            'assert slugify("Hello World", max_len=11) == "hello-world"', 'assert slugify("one two three", max_len=8) == "one-two"',
            'assert slugify("abcdefgh ij", max_len=4) == "abcd"', 'assert slugify("--a--") == "a"', 'assert slugify("") == "n-a"',
            'assert slugify("\\u65e5\\u672c") == "n-a"', 'assert slugify("a b", max_len=1) == "a"', 'assert slugify("ab cd ef", max_len=5) == "ab-cd"', '_raises(slugify, "x", -1)'],
'osa_distance': ['assert osa_distance("abc", "acb") == 1', 'assert osa_distance("abc", "bca") == 2', 'assert osa_distance("", "") == 0',
                 'assert osa_distance("abcd", "badc") == 2', 'assert osa_distance("flaw", "lawn") == 2', 'assert osa_distance("sunday", "saturday") == 3',
                 'assert osa_distance("ab", "a") == 1'],
'number_to_words': ['assert number_to_words(7) == "seven"', 'assert number_to_words(19) == "nineteen"', 'assert number_to_words(99) == "ninety-nine"',
                    'assert number_to_words(110) == "one hundred and ten"', 'assert number_to_words(999) == "nine hundred and ninety-nine"',
                    'assert number_to_words(1010) == "one thousand and ten"', 'assert number_to_words(12345) == "twelve thousand, three hundred and forty-five"',
                    'assert number_to_words(100001) == "one hundred thousand and one"', 'assert number_to_words(20000) == "twenty thousand"',
                    'assert number_to_words(1200) == "one thousand, two hundred"', 'assert number_to_words(21021) == "twenty-one thousand and twenty-one"',
                    'for bad in [-5, 1000001, 3.0]:\n    _raises(number_to_words, bad)'],
'topological_sort': ['assert topological_sort({"a": ["c"], "b": ["c"]}) == ["c", "a", "b"]', 'assert topological_sort({"z": [], "y": [], "x": []}) == ["x", "y", "z"]',
                     'assert topological_sort({"b": ["a"], "c": ["a"], "d": ["b"]}) == ["a", "b", "c", "d"]',
                     'assert topological_sort({"d": ["a"], "b": ["c"], "c": []}) == ["a", "c", "b", "d"]',
                     'try:\n    topological_sort({"a": ["b"], "b": ["c"], "c": ["a"]}); raise AssertionError("expected ValueError")\nexcept ValueError as e:\n    assert "cycle" in str(e)'],
'roman_to_int': ['assert roman_to_int("XIV") == 14', 'assert roman_to_int("CDXLIV") == 444', 'assert roman_to_int("MMXXIV") == 2024', 'assert roman_to_int("IX") == 9',
                 'assert roman_to_int("XC") == 90', 'assert roman_to_int("MCM") == 1900',
                 'for bad in ["XXXX", "DD", "IL", "VL", "CMC", "XM", "IIV", "XIIII", "MMMMI", "xiv"]:\n    _raises(roman_to_int, bad)'],
'parse_version': ['assert parse_version("2.0.0-rc.1") == (2, 0, 0, ("rc", 1), "")', 'assert parse_version("1.0.0-0.3.7") == (1, 0, 0, (0, 3, 7), "")',
                  'assert parse_version("1.0.0-x.7.z.92") == (1, 0, 0, ("x", 7, "z", 92), "")', 'assert parse_version("1.2.3+20130313144700") == (1, 2, 3, (), "20130313144700")',
                  'assert parse_version("v0.1.0") == (0, 1, 0, (), "")',
                  'for bad in ["1.02.3", "v", "1.2.3-beta+", "-1.2.3", "1..3"]:\n    _raises(parse_version, bad)'],
}

for t in TASKS:
    t['heldout'] = HELDOUT[t['name']]

# ---- the API, with usage recorded and a labelled stub fallback
USAGE = {'input': 0, 'output': 0, 'calls': 0}
LOCK = threading.Lock()
STUB = False
_client = None


def model(messages, max_tokens=2500):
    global STUB, _client
    if STUB:
        return 'def _stub():\n    pass'
    if _client is None:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv('/Users/admin/Desktop/Learning/rag_learning/.env')
        _client = OpenAI()
    for attempt in range(3):
        try:
            r = _client.chat.completions.create(model=MODEL, messages=messages, max_completion_tokens=max_tokens, reasoning_effort='minimal')
            with LOCK:
                USAGE['input'] += r.usage.prompt_tokens; USAGE['output'] += r.usage.completion_tokens; USAGE['calls'] += 1
            return (r.choices[0].message.content or '').strip()
        except Exception as e:
            print(f'  API error ({type(e).__name__}), attempt {attempt + 1}: {str(e)[:80]}')
            time.sleep(3)
    STUB = True
    print('  *** API failed: switching to a STUB. These results are NOT real. ***')
    return 'def _stub():\n    pass'


def extract_code(text):
    m = re.findall(r'```(?:python)?\n(.*?)```', text, re.S)
    return '\n\n'.join(m) if m else text


def run_tests(code, tests):
    """Run tests against code in a fresh `python -I` process with a timeout. Returns (passed, failure_text).
    A demonstration runner, NOT a sandbox: see the docstring at the top of this file."""
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
    src_lines = body.split('\n')                       # map the traceback's line number back to a "# test i" block
    block = None
    for n in reversed([int(m) for m in re.findall(r'line (\d+)', out)]):
        for k in range(min(n, len(src_lines)) - 1, -1, -1):
            if src_lines[k].startswith('# test '):
                block = int(src_lines[k].split()[2]); break
        if block:
            break
    err = next((l for l in reversed(out.strip().split('\n')) if l.strip()), out[-200:])
    return False, (f'failing test:\n{tests[block - 1]}\n-> {err[:300]}' if block else err[-400:])


def validate_suites(log):
    """Every feedback and held-out test must pass on the reference solution, or the experiment would grade against a wrong test."""
    bad = []
    for t in TASKS:
        for suite in ('feedback', 'heldout'):
            ok, fail = run_tests(REFERENCE[t['name']], t[suite])
            if not ok:
                bad.append(f'{t["name"]}/{suite}: {fail[:120]}')
    n_f, n_h = sum(len(t['feedback']) for t in TASKS), sum(len(t['heldout']) for t in TASKS)
    log(f'test suites checked against reference solutions: {len(TASKS) * 2 - len(bad)}/{len(TASKS) * 2} pass '
        f'({n_f} feedback test blocks, {n_h} held-out test blocks)')
    for b in bad:
        log('  BAD SUITE ' + b)
    assert not bad, 'a test suite fails on its reference solution'


SYSTEM = 'You write correct Python 3 functions. Reply with one ```python code block containing only the function(s) and any imports. No tests, no prose.'
SELF_PROMPT = ('Review your function carefully against the specification. If you find a mistake, reply with the corrected code block. '
               'If you find no mistake, reply with the same code block unchanged.')
FIX_PROMPT = ('Your code failed this test from the project\'s test suite:\n{fail}\n\n'
              'First write one or two sentences on what went wrong, then reply with the corrected code block.')
same = lambda a, b: re.sub(r'\s+', '', a) == re.sub(r'\s+', '', b)


def self_review(msgs, reply):
    """Always SELF_ROUNDS rounds; no test is run or shown. Returns the final code and whether any round changed it."""
    msgs = msgs + [{'role': 'assistant', 'content': reply}]
    code0 = code = extract_code(reply)
    for _ in range(SELF_ROUNDS):
        msgs.append({'role': 'user', 'content': SELF_PROMPT})
        reply = model(msgs)
        msgs.append({'role': 'assistant', 'content': reply})
        code = extract_code(reply)
    return code, not same(code, code0)


def feedback_loop(msgs, reply, task):
    """Run the feedback suite; while it fails and attempts remain, show the first failing test and ask for a fix."""
    msgs = msgs + [{'role': 'assistant', 'content': reply}]
    code = extract_code(reply)
    ok, fail = run_tests(code, task['feedback'])
    extra = 0
    while not ok and extra < FIX_ATTEMPTS:
        msgs.append({'role': 'user', 'content': FIX_PROMPT.format(fail=fail)})
        reply = model(msgs)
        msgs.append({'role': 'assistant', 'content': reply})
        code = extract_code(reply)
        ok, fail = run_tests(code, task['feedback'])
        extra += 1
    return code, ok, extra


def one_attempt(job):
    t, k = job
    msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': t['spec']}]
    reply = model(msgs)
    code = extract_code(reply)
    f0, ffail = run_tests(code, t['feedback'])
    h0, _ = run_tests(code, t['heldout'])
    s_code, s_changed = self_review(msgs, reply)
    s_h, _ = run_tests(s_code, t['heldout'])
    fb_code, fb_ok, fb_extra = feedback_loop(msgs, reply, t)
    fb_h, _ = run_tests(fb_code, t['heldout'])
    return {'task': t['name'], 'sample': k, 'first_feedback': f0, 'first_heldout': h0, 'first_failure': ffail,
            'code_first': code, 'code_self': s_code, 'code_fb': fb_code, 'heldout_failure_first': run_tests(code, t['heldout'])[1],
            'self_heldout': s_h, 'self_changed': s_changed, 'fb_heldout': fb_h, 'fb_feedback_pass': fb_ok, 'fb_extra_attempts': fb_extra}


def main():
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(one_attempt, [(t, k) for t in TASKS for k in range(SAMPLES)]))
    save('ch3_reflexion', {'model': MODEL, 'stub': STUB, 'rows': rows, 'usage': USAGE, 'samples': SAMPLES,
                           'self_rounds': SELF_ROUNDS, 'fix_attempts': FIX_ATTEMPTS, 'tasks': [t['name'] for t in TASKS]})
    report()


def report():
    """The log and the summary, computed from the saved rows (python ch3_reflexion.py --report redoes it without API calls)."""
    R = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'ch3_reflexion.json')))
    rows, usage = R['rows'], R['usage']
    log = Log('ch3_reflexion')
    log(f'== Repair strategies on {len(TASKS)} coding tasks x {SAMPLES} first attempts, {R["model"]}; every number below is on the HELD-OUT tests ==')
    log('runner: separate `python -I` process + 10 s timeout = a demonstration, not a sandbox')
    validate_suites(log)
    log('')
    log(f'-- per task, out of {SAMPLES}: held-out passes for the first attempt, after self-review, after the feedback-test loop --')
    log(f'{"task":<17} {"feedback@1":>10} {"held-out@1":>10} {"self-review":>11} {"fb-tests":>9} {"self changed":>12}')
    for t in TASKS:
        rs = [r for r in rows if r['task'] == t['name']]
        c = lambda key: sum(1 for r in rs if r[key])
        log(f'{t["name"]:<17} {c("first_feedback"):>8}/{SAMPLES} {c("first_heldout"):>8}/{SAMPLES} {c("self_heldout"):>9}/{SAMPLES} '
            f'{c("fb_heldout"):>7}/{SAMPLES} {c("self_changed"):>10}/{SAMPLES}')
    n = len(rows)
    cnt = lambda f: sum(1 for r in rows if f(r))
    first = cnt(lambda r: r['first_heldout'])
    s_final, f_final = cnt(lambda r: r['self_heldout']), cnt(lambda r: r['fb_heldout'])
    s_fix, s_break = cnt(lambda r: not r['first_heldout'] and r['self_heldout']), cnt(lambda r: r['first_heldout'] and not r['self_heldout'])
    f_fix, f_break = cnt(lambda r: not r['first_heldout'] and r['fb_heldout']), cnt(lambda r: r['first_heldout'] and not r['fb_heldout'])
    f_acc = cnt(lambda r: r['first_feedback'])
    f_acc_wrong = cnt(lambda r: r['first_feedback'] and not r['first_heldout'])
    f_rej_right = cnt(lambda r: not r['first_feedback'] and r['first_heldout'])
    s_changed = cnt(lambda r: r['self_changed'])
    s_changed_right = cnt(lambda r: r['first_heldout'] and r['self_changed'])
    fb_touched = cnt(lambda r: r['fb_extra_attempts'] > 0)
    log('')
    log('-- summary (held-out tests) --')
    log(f'first attempts: {first}/{n} pass the held-out tests ({first / n:.0%})')
    log(f'self-review, {SELF_ROUNDS} rounds on all {n}: {s_final}/{n} pass ({s_final / n:.0%}); wrong->right {s_fix}, right->wrong {s_break}')
    log(f'   the code changed in {s_changed} of {n} attempts, {s_changed_right} of them already right')
    log(f'feedback-test loop on all {n}: {f_final}/{n} pass ({f_final / n:.0%}); wrong->right {f_fix}, right->wrong {f_break}')
    log(f'   {fb_touched} attempts failed the feedback suite and entered the loop; the rest were left alone')
    log(f'feedback suite as a check on first attempts: accepted {f_acc}, of which {f_acc_wrong} fail held-out (false accept);')
    log(f'   rejected {n - f_acc}, of which {f_rej_right} pass held-out (false reject)')
    cost = usage['input'] / 1e6 * PRICE_IN + usage['output'] / 1e6 * PRICE_OUT
    log(f'total spend this run: ${cost:.4f} ({usage["calls"]} calls, {usage["input"]:,} input + {usage["output"]:,} output tokens)')
    log('stub used: ' + ('YES, results are not real' if R['stub'] else 'no, every answer came from the API'))
    R['summary'] = {'n': n, 'first_heldout': first, 'self_heldout': s_final, 'fb_heldout': f_final, 'self_fixed': s_fix, 'self_broke': s_break,
                    'fb_fixed': f_fix, 'fb_broke': f_break, 'self_changed': s_changed, 'self_changed_right': s_changed_right,
                    'fb_entered': fb_touched, 'feedback_accept': f_acc, 'feedback_false_accept': f_acc_wrong, 'feedback_reject_right': f_rej_right,
                    'cost_usd': cost, 'calls': usage['calls']}
    save('ch3_reflexion', R)


if __name__ == '__main__':
    if '--check-tests' in sys.argv:
        validate_suites(print)
    elif '--report' in sys.argv:
        report()
    else:
        main()
