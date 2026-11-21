"""The rule-based rewards of DeepSeek-R1-Zero (§2.2.2), two ways each. Used by r1_part3.py and r1_part3_run.py.
Accuracy (maths): our extract_boxed + normalise + compare, and the math-verify library.
Accuracy (code): a trusted test harness in a subprocess with a time limit, and stdin/stdout cases like a contest judge.
Format: one regular expression, and a small tag parser. Plus the naive and strict rewards of the hacking demo."""
import os, re, secrets, subprocess, sys, tempfile
from fractions import Fraction
from math_verify import parse, verify   # Method 2 of the maths reward (math-verify 0.9.0)


def extract_boxed(text):
    """Return what is inside the LAST \\boxed{...} of text, or None. Counts braces, so nested braces work:
    '\\boxed{\\frac{\\sqrt{3}}{2}}' -> '\\frac{\\sqrt{3}}{2}'."""
    start = text.rfind('\\boxed')                  # the last box is the final answer
    if start < 0:
        return None
    i = text.find('{', start)                      # the opening brace of the box
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == '{':
            depth += 1                             # one level deeper
        elif text[j] == '}':
            depth -= 1                             # one level up
            if depth == 0:                         # back to the level of the box: this brace closes it
                return text[i + 1:j]
    return None                                    # the box never closes: no answer


def naive_boxed(text):
    """The one-line regex many people write first. It stops at the FIRST closing brace."""
    m = re.findall(r'\\boxed\{(.*?)\}', text)
    return m[-1] if m else None


UNITS = r'(dollars?|cents?|cm|centimeters?|meters?|km|kg|grams?|inches|inch|feet|foot|miles?|hours?|minutes?|seconds?|' \
        r'days?|years?|degrees?|units?|square units|sq units|people|students|eggs|pages)'


def normalise(s):
    """Turn an answer string into one standard spelling. Each line removes one harmless difference."""
    s = s.strip()
    s = re.sub(r'\\text\{\s*([^{}]*?)\s*\}', r' \1', s)        # \text{ cm} -> ' cm' (keep the words, drop the macro)
    s = s.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')   # display and text fractions are fractions
    s = s.replace('\\left', '').replace('\\right', '')         # bracket sizes do not change the value
    s = s.replace('\\$', '').replace('$', '')                  # dollar signs
    s = s.replace('\\%', '').replace('%', '')                  # per cent signs (see the table: this is a choice)
    s = s.replace('\\!', '').replace('\\,', '').replace('{,}', ',')   # LaTeX spacing
    s = re.sub(r'^\s*[a-zA-Z]\s*=\s*', '', s)                  # 'x = 3' -> '3'
    s = re.sub(r'\s*\b' + UNITS + r'\s*\.?$', '', s)           # a trailing unit word: '18 dollars' -> '18'
    s = re.sub(r'(\d),(\d\d\d)\b', r'\1\2', s)                 # thousands separator: 1,000 -> 1000
    s = re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r'(\1)/(\2)', s)   # \frac{a}{b} -> (a)/(b)
    s = re.sub(r'\\frac(\d)(\d)', r'(\1)/(\2)', s)             # \frac12 -> (1)/(2)
    s = s.replace(' ', '').rstrip('.')
    return s


NUM = re.compile(r'^\(?(-?\d+(?:\.\d+)?)\)?(?:/\(?(-?\d+(?:\.\d+)?)\)?)?$')


def to_number(s):
    """'(3)/(4)' -> Fraction(3, 4); '0.5' -> Fraction(1, 2); anything else -> None. Exact, no rounding."""
    sign = -1 if s.startswith('-(') else 1         # '-(1)/(2)' -> minus sign outside the brackets
    m = NUM.match(s[1:] if sign < 0 else s)
    if not m:
        return None
    a = Fraction(m.group(1))
    return sign * (a / Fraction(m.group(2)) if m.group(2) else a)


def same_answer(pred, gold):
    """Method 1 compare: equal as exact numbers if both are numbers, else equal as normalised strings."""
    p, g = normalise(pred), normalise(gold)
    pn, gn = to_number(p), to_number(g)
    if pn is not None and gn is not None:
        return pn == gn
    return p == g


def accuracy_reward_ours(output, gold):
    """Method 1: 1.0 if the last boxed answer equals the gold answer, else 0.0."""
    pred = extract_boxed(output)
    return float(pred is not None and same_answer(pred, gold))


def accuracy_reward_lib(output, gold):
    """Method 2: math-verify parses both sides into SymPy objects and compares them as maths."""
    g = parse('$' + gold + '$')          # the gold answer, read as LaTeX
    p = parse(output)                    # the model output: math-verify finds the boxed answer itself
    return float(bool(g) and bool(p) and verify(g, p))


def run_python(program, stdin='', timeout=2.0):
    """Run a Python program in a fresh process. Returns (exit code, stdout, stderr) or None on a timeout."""
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(program)
    try:
        r = subprocess.run([sys.executable, '-I', f.name], input=stdin, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return None                       # an endless loop gets no reward and does not hang the trainer
    finally:
        os.unlink(f.name)


def code_reward_marker(code, tests, timeout=2.0):
    """A WRONG grader, kept only to show the hole: it appends the asserts and trusts a printed 'ALL PASS'.
    A submission that prints 'ALL PASS' itself and exits before the asserts gets reward 1."""
    program = code + '\n\n' + '\n'.join(f'assert {c} == {w!r}' for c, w in tests) + '\nprint("ALL PASS")\n'
    r = run_python(program, timeout=timeout)
    return float(r is not None and r[0] == 0 and 'ALL PASS' in r[1])


HARNESS = """
import sys, importlib.util
nonce = sys.stdin.readline().strip()           # the secret is read BEFORE the submission runs
tests = TESTS
spec = importlib.util.spec_from_file_location('submission', PATH)
sub = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(sub)               # run the submission as a module, inside a try
except BaseException as e:                     # BaseException also catches sys.exit() and SystemExit
    print(f'{type(e).__name__}: {e}', file=sys.stderr); sys.exit(1)
for call, want in tests:
    try:
        got = eval(call, dict(vars(sub)))      # call the submitted function
    except BaseException as e:
        print(f'{type(e).__name__} in {call}', file=sys.stderr); sys.exit(1)
    if type(got) is not type(want) or got != want:   # same type, so an object that claims to equal anything fails
        print(f'wrong: {call} == {want!r}', file=sys.stderr); sys.exit(1)
print(nonce)                                   # reached only after every test passed
"""


def code_reward_asserts(code, tests, timeout=2.0):
    """Method 1: a trusted harness imports the submission and checks each test (call, expected value) itself.
    It prints a fresh random nonce only after every test passes, so a submission that prints a success marker
    or exits early gets nothing. (Not a full sandbox: real systems also lock the process down.)"""
    nonce = secrets.token_hex(16)                                    # new for every call; the submission never sees it
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(code)                                                # the submission, in its own file
    harness = HARNESS.replace('TESTS', repr(list(tests))).replace('PATH', repr(f.name))
    try:
        r = run_python(harness, stdin=nonce + '\n', timeout=timeout)
    finally:
        os.unlink(f.name)
    if r is None:
        return 0.0, 'timeout'
    code_, out, err = r
    if code_ == 0 and out.strip().splitlines()[-1:] == [nonce]:
        return 1.0, 'all tests pass'
    last = (err.strip().splitlines() or ['no nonce printed'])[-1]
    last = re.sub(r' \(tmp\w+\.py, line \d+\)', '', last)           # drop the temporary file name
    return 0.0, last[:60]


def code_reward_judge(program, cases, timeout=2.0):
    """Method 2: the way a contest judge works. Feed each input on stdin, compare stdout with the expected output."""
    passed = 0
    for inp, want in cases:
        r = run_python(program, stdin=inp, timeout=timeout)
        if r is not None and r[0] == 0 and r[1].strip() == want.strip():
            passed += 1
    return float(passed == len(cases)), f'{passed}/{len(cases)} cases'


FORMAT_RE = re.compile(r'^\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*$', re.DOTALL)


def format_reward_regex(output):
    """Method 1: one regular expression for the whole output."""
    return float(FORMAT_RE.match(output) is not None)


TAG = re.compile(r'</?(think|answer)>')


def format_reward_parser(output):
    """Method 2: read the tags in order and check every rule by name."""
    tags = [(m.group(0), m.start(), m.end()) for m in TAG.finditer(output)]
    if [t[0] for t in tags] != ['<think>', '</think>', '<answer>', '</answer>']:
        return 0.0, 'tags: ' + ' '.join(t[0] for t in tags) if tags else 'no tags'
    (_, _, a), (_, b, c), (_, d, e), (_, f, g) = tags
    if output[:tags[0][1]].strip():
        return 0.0, 'text before <think>'
    if output[g:].strip():
        return 0.0, 'text after </answer>'
    if output[c:d].strip():
        return 0.0, 'text between the blocks'
    if not output[a:b].strip():
        return 0.0, 'empty think'
    if not output[e:f].strip():
        return 0.0, 'empty answer'
    return 1.0, 'ok'


def naive_reward(output, gold):
    """'The gold answer appears somewhere in the output.' Sounds fair; it is not."""
    return float(re.search(r'(?<![\d.])' + re.escape(gold) + r'(?![\d.])', output) is not None)


def answer_block(output):
    m = FORMAT_RE.match(output)
    return m.group(2) if m else None


def strict_reward(output, gold):
    """The strict format parser first, then exactly one boxed answer inside <answer>, then the accuracy check."""
    if format_reward_parser(output)[0] != 1.0:                       # the parser, not the loose regex
        return 0.0
    ans = answer_block(output)
    if ans is None or ans.count('\\boxed') != 1:
        return 0.0
    return accuracy_reward_ours(ans, gold)
